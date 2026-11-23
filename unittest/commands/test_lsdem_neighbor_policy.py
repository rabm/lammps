#!/usr/bin/env python3
"""LS-DEM cutoff selection and efficiency warnings must leave shared neighbor settings intact."""
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from lsdem_spheres import write_shapes


def simulate(directory, pair, coeff, neighbors="", *, expect_error=False, two_types=False):
    if sys.argv[2:] and pair.startswith("hybrid/overlay"):
        # Hybrid lists report the input sub-style name even when a suffix is active.
        pair = pair.replace(" ls/dem ", " ls/dem/kk ")
        coeff = coeff.replace(" ls/dem ", " ls/dem/kk ")
    commands = f"""units lj
atom_style ls/dem
atom_modify map yes
comm_modify vel yes cutoff 20
newton off
region box block -25 25 -25 25 -25 25
create_box {2 if two_types else 1} box
mass * 1
molecule sphere sphere1.mol
create_atoms 0 single 0 0 0 mol sphere 1234 rotate 0 0 0 1 units box
{"create_atoms 1 single 3 0 0 mol sphere 5678 rotate 0 0 0 1 units box" if two_types else ""}
{"fix bodies all rigid/small/ls/dem molecule" if "ls/dem" in pair else ""}
{neighbors}
pair_style {pair}
{coeff}
run 0
run 0
write_dump all custom forces id fx fy fz modify sort id format float %.17g
"""
    result = subprocess.run([str(Path(sys.argv[1]).resolve()), *sys.argv[2:], "-log", "none"], input=commands,
                            text=True, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if (result.returncode != 0) != expect_error:
        raise RuntimeError(result.stdout)
    if sys.argv[2:] and not expect_error and "ls/dem" in pair:
        assert "pair ls/dem/kk," in result.stdout
    return result.stdout


def list_distance(log):
    return float(re.search(r"master list distance cutoff = (\S+)", log)[1])


def forces(directory):
    rows = (directory / "forces").read_text().split("ITEM: ATOMS id fx fy fz\n")[1]
    return {int(row.split()[0]): tuple(map(float, row.split()[1:])) for row in rows.splitlines()}


def check_forces(actual, expected):
    assert actual.keys() == expected.keys()
    for atom, reference in expected.items():
        for got, ref in zip(actual[atom], reference):
            assert math.isclose(got, ref, rel_tol=1e-6, abs_tol=1e-7), (atom, got, ref)


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
    assert any(f != 0.0 for row in expected.values() for f in row)
    overlay = simulate(d, hybrid.format("auto"), hybrid_coeff.format("12"))
    assert list_distance(reference) == list_distance(overlay) == 4.3
    check_forces(forces(d), expected)
    assert overlay.count(no_exclusion) == 1
    # Only one type pair is assigned to LS-DEM. The other type still feels LJ forces.
    simulate(d, "lj/cut 4", "pair_coeff * * 1 1", two_types=True)
    expected = forces(d)
    simulate(d, hybrid.format("auto"), hybrid_coeff.format("12").replace("* * ls/dem", "1 1 ls/dem"),
             two_types=True)
    check_forces(forces(d), expected)
