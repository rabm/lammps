#!/usr/bin/env python3
"""Check two contacts that use the same level-set grid at different scales."""
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from lsdem_spheres import write_shapes


def forces(directory, executable, method, flags):
    deck = f"""units lj
atom_style ls/dem
atom_modify map yes sort 0 0.0
comm_modify vel yes cutoff 6.0
newton off
region box block -20 20 -10 10 -10 10
create_box 1 box
mass 1 1.0
molecule small sphere1.mol
molecule large sphere2.mol
create_atoms 0 single -7.0 0 0 mol small 1234 rotate 0 0 0 1 units box
create_atoms 0 single -5.1 0 0 mol small 1234 rotate 0 0 0 1 units box
create_atoms 0 single  3.0 0 0 mol large 1234 rotate 0 0 0 1 units box
create_atoms 0 single  6.8 0 0 mol large 1234 rotate 0 0 0 1 units box
neighbor 0.10 bin
neigh_modify exclude molecule/intra all
fix bodies all rigid/small/ls/dem molecule
pair_style ls/dem 0.75 method {method}
pair_coeff * * 100 25 0.5 0 0 0 0.75
run 0
write_dump all custom forces id mol fx fy fz modify sort id format float %.17g
"""
    proc = subprocess.run([executable, *flags], cwd=directory, input=deck, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(proc.stdout)
    if flags and "pair ls/dem/kk," not in proc.stdout:
        raise AssertionError("Kokkos pair style did not run")
    rows = (directory / "forces").read_text().split("ITEM: ATOMS id mol fx fy fz\n")[1]
    return {tuple(map(int, row.split()[:2])): tuple(map(float, row.split()[2:]))
            for row in rows.splitlines()}


with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    write_shapes(directory, (1, 2))
    exe = str(Path(sys.argv[1]).resolve())
    expected = forces(directory, exe, "transparent", [])
    assert len(expected) == 24
    for grain in range(1, 5):
        assert any(abs(force[0]) > 1.e-6 for (_, mol), force in expected.items() if mol == grain)
    actual = forces(directory, exe, "fast", sys.argv[2:])
    assert actual.keys() == expected.keys()
    for atom in expected:
        for got, ref in zip(actual[atom], expected[atom]):
            assert math.isclose(got, ref, rel_tol=1.e-6, abs_tol=1.e-7), (atom, got, ref)
