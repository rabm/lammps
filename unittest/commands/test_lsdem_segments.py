#!/usr/bin/env python3
"""Compare CPU and Kokkos contact forces when all neighbor entries are local."""
import itertools
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def write_sphere(directory):
    # Six surface nodes at the tips of the coordinate axes.
    nodes = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    coords = "".join(f"{i} {x} {y} {z}\n" for i, (x, y, z) in enumerate(nodes, 1))
    types = "".join(f"{i} 1\n" for i in range(1, 7))
    (directory / "sphere.mol").write_text(
        "Sphere\n6 atoms\n1.0 mass\ngrid 0 1.0 lsdem\n\nCoords\n\n"
        + coords + "\nTypes\n\n" + types
    )
    with (directory / "grid").open("w") as grid:
        grid.write("# Unit sphere\n7 7 7\n0.5\n-1.5 -1.5 -1.5\n")
        for z, y, x in itertools.product(range(-3, 4), repeat=3):
            print(0.5 * math.sqrt(x*x + y*y + z*z) - 1.0, file=grid)


def forces(directory, command):
    run = subprocess.run(command + ["-in", "in.contact"], cwd=directory,
                         stdout=subprocess.PIPE, text=True)
    if run.returncode:
        raise RuntimeError(run.stdout)
    if "-sf" in command and "pair ls/dem/kk," not in run.stdout:
        raise AssertionError("The Kokkos pair style must be active")
    if not any(line.split()[:2] == ["Nghost:", "0"] for line in run.stdout.splitlines()):
        raise AssertionError("The regression needs a neighbor list without ghost entries")
    atoms = (directory / "forces").read_text().split("ITEM: ATOMS id fx fy fz\n")[1]
    return [float(value) for row in atoms.splitlines() for value in row.split()[1:]]


with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    write_sphere(directory)
    (directory / "in.contact").write_text("""units lj
atom_style ls/dem
atom_modify map yes
comm_modify vel yes cutoff 3.5
newton off
region box block -10 10 -10 10 -10 10
create_box 1 box
mass 1 1
molecule sphere sphere.mol
create_atoms 0 single -0.92 0 0 mol sphere 1234 rotate 0 0 0 1 units box
create_atoms 0 single  0.92 0 0 mol sphere 2345 rotate 0 0 0 1 units box
neighbor 0.1 bin
fix bodies all rigid/small/ls/dem molecule
pair_style ls/dem 0.75 order exact
pair_coeff * * 100 25 0.5 0 0 0 0.75
run 1
write_dump all custom forces id fx fy fz modify sort id format float %.17g
""")
    executable = str(Path(sys.argv[1]).resolve())
    cpu = forces(directory, [executable])
    kokkos = forces(directory, [executable, "-k", "on", *sys.argv[2:], "-sf", "kk"])
    assert len(cpu) == len(kokkos) == 36 and max(map(abs, cpu)) > 0
    assert all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7) for a, b in zip(cpu, kokkos))
