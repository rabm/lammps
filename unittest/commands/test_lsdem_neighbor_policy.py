#!/usr/bin/env python3
"""LS-DEM cutoff selection and efficiency warnings must leave shared neighbor settings intact."""
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from lsdem_spheres import write_shapes


def simulate(directory, pair, coeff, neighbors="", *, expect_error=False):
    commands = f"""units lj
atom_style ls/dem
atom_modify map yes
comm_modify vel yes cutoff 20
newton off
region box block -25 25 -25 25 -25 25
create_box 1 box
mass 1 1
molecule sphere sphere1.mol
create_atoms 0 single 0 0 0 mol sphere 1234 rotate 0 0 0 1 units box
{"fix bodies all rigid/small/ls/dem molecule" if "ls/dem" in pair else ""}
{neighbors}
pair_style {pair}
{coeff}
run 0
run 0
write_dump all custom forces id fx fy fz modify sort id format float %.17g
"""
    result = subprocess.run([str(Path(sys.argv[1]).resolve()), "-log", "none"], input=commands,
                            text=True, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if (result.returncode != 0) != expect_error:
        raise RuntimeError(result.stdout)
    return result.stdout


def list_distance(log):
    return float(re.search(r"master list distance cutoff = (\S+)", log)[1])


def forces(directory):
    return (directory / "forces").read_text().split("ITEM: ATOMS id fx fy fz\n")[1]


with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    write_shapes(d, (1,))
    coefficients = "pair_coeff * * 100 25 0.5 0 0 0"
    no_exclusion = "may be inefficient without intramolecular exclusion"
    wide_cut = "may be inefficient for this node spacing"
    excluded = "neigh_modify exclude molecule/intra all"
    # Nearest-node spacing is sqrt(2); the LJ unit skin is 0.3.
    auto = simulate(d, "ls/dem auto", coefficients + " 12")
    assert math.isclose(list_distance(auto), 2*math.sqrt(2) + 0.3, rel_tol=1e-7)
    assert auto.count(no_exclusion) == 1 and wide_cut not in auto
    explicit = simulate(d, "ls/dem", coefficients + " 12", "neighbor 0.25 bin\n" + excluded)
    assert list_distance(explicit) == 12.25 and no_exclusion not in explicit
    alias = simulate(d, "ls/dem 2", coefficients, "group grains type 1\nneigh_modify exclude molecule/intra grains")
    assert no_exclusion not in alias
    assert explicit.count(wide_cut) == 1 and "cutoff near 2.8284271" in explicit
    assert list_distance(simulate(d, "ls/dem", coefficients + " auto")) == list_distance(auto)
    assert list_distance(simulate(d, "ls/dem 2", coefficients)) == 2.3
    assert "numeric cutoff or explicit auto" in simulate(d, "ls/dem", coefficients, expect_error=True)
    # LJ acts on same-molecule nodes; its forces must survive the overlay.
    # Compare directly against the independent LJ pair style.
    hybrid = "hybrid/overlay ls/dem {} lj/cut 4"
    hybrid_coeff = "pair_coeff * * ls/dem 0 0 0 0 0 0 {}\npair_coeff * * lj/cut 1 1"
    reference = simulate(d, "lj/cut 4", "pair_coeff * * 1 1")
    expected = forces(d)
    assert any(float(f) != 0.0 for row in expected.splitlines() for f in row.split()[1:])
    overlay = simulate(d, hybrid.format("auto"), hybrid_coeff.format("12"))
    assert list_distance(reference) == list_distance(overlay) == 4.3
    assert forces(d) == expected and overlay.count(no_exclusion) == 1
