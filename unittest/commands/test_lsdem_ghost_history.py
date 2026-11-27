#!/usr/bin/env python3
"""Compare forces and contact partners after ghost slots change during sliding."""
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from lsdem_spheres import write_shapes


def state(directory, executable, rigid, flags):
    small = rigid == "rigid/small/ls/dem"
    run = subprocess.run([executable, *flags, "-var", "small", str(int(small)),
                          "-var", "save_bodies", str(int(small and not flags)), "-in", "in.slide"], cwd=directory,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if run.returncode:
        raise RuntimeError(run.stdout)
    if flags and "pair ls/dem/kk," not in run.stdout:
        raise AssertionError("The Kokkos pair style must be active")
    rows = (directory / "forces").read_text().split("ITEM: ATOMS id i_ls_dem_touch_id fx fy fz\n")[1]
    return {int(tag): (int(partner), tuple(map(float, force)))
            for tag, partner, *force in map(str.split, rows.splitlines())}


with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    write_shapes(directory, (1,))
    # Nodes of the first sphere leave the periodic send list while the other two
    # slide across the x boundary. Their contact persists as ghost slots shift.
    (directory / "in.slide").write_text("""units lj
atom_style ls/dem
atom_modify map yes sort 0 0.0
newton off
comm_modify vel yes cutoff 3.5
region box block -5 5 -5 5 -5 5
create_box 1 box
mass 1 1
molecule sphere sphere1.mol
create_atoms 0 single -1.53 3 0 mol sphere 123 rotate 0 0 0 1 units box
create_atoms 0 single -4.08 0 0 mol sphere 1234 rotate 0 0 0 1 units box
create_atoms 0 single 4.08 0 0 mol sphere 2345 rotate 0 0 0 1 units box
group moving molecule 2
velocity all set 3 0 0 units box
velocity moving set NULL 1 0 units box
neighbor 0.01 bin
neigh_modify every 1 delay 0 check no exclude molecule/intra all
if "${small} == 1" then "fix bodies all rigid/small/ls/dem molecule" &
  else "fix bodies all rigid/ls/dem molecule infile initial.restart.rigid"
pair_style ls/dem 0.75 order exact
pair_coeff * * 100 25 0.5 0 0 0 0.75
timestep 0.001
run 0
if "${save_bodies} == 1" then "write_restart initial.restart"
run 11
write_dump all custom forces id i_ls_dem_touch_id fx fy fz modify sort id format float %.17g
""")
    executable = str(Path(sys.argv[1]).resolve())
    for rigid in ("rigid/small/ls/dem", "rigid/ls/dem"):
        reference = state(directory, executable, rigid, [])
        actual = state(directory, executable, rigid, ["-k", "on", *sys.argv[2:], "-sf", "kk",
                                                      "-pk", "kokkos", "neigh", "half", "newton", "off"])
        assert reference.keys() == actual.keys() == set(range(1, 19))
        partners = {partner for partner, _ in reference.values()}
        assert -1 in partners and len(partners) > 1
        assert max(abs(force[1]) for _, force in reference.values()) > 1.0
        for tag, (partner, force) in reference.items():
            assert actual[tag][0] == partner, (rigid, tag, actual[tag][0], partner)
            for got, expected in zip(actual[tag][1], force):
                assert math.isclose(got, expected, rel_tol=1.e-6, abs_tol=1.e-7), (rigid, tag, got, expected)
