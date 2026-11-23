#!/usr/bin/env python3
"""Small spherical level sets for LS-DEM command tests."""
import itertools
import math


def write_shapes(directory, scales):
    with (directory / "grid").open("w") as grid:
        grid.write("# Unit sphere\n13 13 13\n0.25\n-1.5 -1.5 -1.5\n")
        for z, y, x in itertools.product(range(-6, 7), repeat=3):
            print(0.25 * math.sqrt(x*x + y*y + z*z) - 1.0, file=grid)
    for scale in scales:
        points = [(scale, 0, 0), (-scale, 0, 0), (0, scale, 0),
                  (0, -scale, 0), (0, 0, scale), (0, 0, -scale)]
        with (directory / f"sphere{scale}.mol").open("w") as mol:
            print(f"Sphere\n6 atoms\n1.0 mass\ngrid 0 {scale} lsdem\n\nCoords\n", file=mol)
            for i, (x, y, z) in enumerate(points, 1):
                print(i, x, y, z, file=mol)
            print("\nTypes\n", file=mol)
            for i in range(1, 7):
                print(i, 1, file=mol)
