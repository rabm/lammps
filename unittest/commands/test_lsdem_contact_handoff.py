#!/usr/bin/env python3
"""A sliding node must retain its history when its nearest partner changes images."""
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from lsdem_spheres import write_shapes


def forces(directory, executable, flags, center, arms, launcher=()):
    # Rotating the large sphere puts two equally high surface nodes on either
    # side of the sliding path. At the box edge the second is a periodic image.
    deck = f"""units lj
atom_style ls/dem
atom_modify map yes sort 0 0.0
newton off
{"processors 2 1 1" if launcher else ""}
region box block -5 5 -6 6 -6 6
create_box 1 box
mass 1 1
molecule small sphere1.mol
molecule large sphere2.mol
create_atoms 0 single {center-0.025} -2.90 0 mol small 123 rotate 0 0 0 1 units box
create_atoms 0 single {center} 0 0 mol large 1234 rotate 45 0 0 1 units box
group slider molecule 1
velocity slider set 1 0 0 units box
comm_modify vel yes cutoff 6.5
neighbor 0.25 bin
neigh_modify delay 0 every 1 check yes exclude molecule/intra all
fix bodies all rigid/small/ls/dem molecule
pair_style ls/dem 1.8 order exact
timestep 0.001
pair_coeff * * 20 10 0.5 0 0 0 1.8 {arms}
run 45
write_dump all custom forces id fx fy fz modify sort id format float %.17g
"""
    result = subprocess.run([*launcher, executable, *flags], input=deck, cwd=directory, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout)
    if flags and "pair ls/dem/kk," not in result.stdout:
        raise AssertionError("The Kokkos pair style must be active")
    rows = (directory / "forces").read_text().split("ITEM: ATOMS id fx fy fz\n")[1]
    return {int(a): tuple(map(float, f)) for a, *f in map(str.split, rows.splitlines())}


with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    write_shapes(directory, (1, 2))
    exe = str(Path(sys.argv[1]).resolve())
    # Optional arguments are an MPI launcher followed by -- and LAMMPS arguments.
    launcher = sys.argv[2:]
    flags = []
    if "--" in launcher:
        split = launcher.index("--")
        launcher, flags = launcher[:split], launcher[split+1:]
    for arms in ("", "5 0.05 3 0.02"):
        expected = forces(directory, exe, [], 0.0, arms)
        actual = forces(directory, exe, flags, 4.99, arms, launcher)
        assert expected.keys() == actual.keys() == set(range(1, 13))
        assert max(abs(fx) for fx, _, _ in expected.values()) > 0.1
        for tag in expected:
            for got, ref in zip(actual[tag], expected[tag]):
                assert math.isclose(got, ref, rel_tol=1.e-6, abs_tol=1.e-7), (tag, got, ref)
