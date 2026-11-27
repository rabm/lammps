#!/usr/bin/env python3
"""A constraint's two centroid virial contributions must follow the same atom."""
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def trajectory(directory, executable, options, launcher, sort_frequency):
    deck = f"""units lj
atom_style molecular
processors * 1 1
atom_modify map array sort {sort_frequency} 0.4
read_data bodies.data
pair_style zero 1.0
pair_coeff * *
comm_modify cutoff 2.5
neighbor 0.25 bin
neigh_modify every 1 delay 0 check no
group bodies molecule 1 2
group mobile subtract all bodies
fix integrator bodies rigid/small molecule
fix free mobile nve
compute stress all centroid/stress/atom NULL fix
dump out all custom 1 rows id proc c_stress[*]
dump_modify out sort id format float %.17g
timestep 0.01
run 16
"""
    (directory / "in.rigid").write_text(deck)
    result = subprocess.run([*launcher, executable, *options, "-in", "in.rigid"],
                            cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    if result.returncode:
        raise RuntimeError(result.stdout)
    frames = []
    for block in (directory / "rows").read_text().split("ITEM: TIMESTEP\n")[1:]:
        _, rows = block.split("ITEM: ATOMS ", 1)
        frames.append([tuple(map(float, line.split())) for line in rows.splitlines()[1:]])
    return frames


with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    # Two freely rotating triangles and an unconstrained atom start in the left
    # subdomain. The right rank starts empty and receives both kinds of atoms.
    (directory / "bodies.data").write_text("""Rigid centroid migration

7 atoms
1 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1

Atoms # molecular

1 1 1 4.94 4.0 4.0
2 1 1 4.40 4.3 4.0
3 1 1 4.75 4.1 4.5
4 2 1 4.16 6.0 6.6
5 2 1 4.72 6.2 6.3
6 2 1 4.51 6.4 6.7
7 0 1 4.97 2.0 2.0

Velocities

1 3.00 0.24 0
2 2.70 -0.30 0
3 2.90 0.05 0
4 -2.50 -0.37 0
5 -2.70 0.19 0
6 -2.90 -0.02 0
7 3.90 0 0
""")
    executable = str(Path(sys.argv[1]).resolve())
    launcher = sys.argv[2:]
    options = shlex.split(os.environ.get("LAMMPS_ACCELERATOR_ARGS", ""))
    if launcher[:1] == ["--kokkos"]:
        if not options:
            options = ["-k", "on", "t", "1", "-sf", "kk"]
        launcher = launcher[1:]
    control = trajectory(directory, executable, options, [], 0)
    tested = trajectory(directory, executable, options, launcher, 0 if launcher else 1)
    assert len(control) == len(tested) == 17
    assert max(abs(x) for frame in control for row in frame for x in row[2:]) > 0.01
    if launcher:
        assert all(row[1] == 0 for row in tested[0])
        assert any(row[1] == 1 and row[0] != 7 for row in tested[-1])
        assert tested[-1][-1][0:2] == (7, 1)
    for step, (reference, actual) in enumerate(zip(control, tested)):
        assert len(reference) == len(actual) == 7
        for expected, observed in zip(reference, actual):
            assert expected[0] == observed[0]
            for a, b in zip(expected[2:], observed[2:]):
                assert math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-8), (step, expected[0], a, b)
