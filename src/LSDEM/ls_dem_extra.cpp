// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "ls_dem_extra.h"

#include "math_const.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;

namespace LSDEMExtra {

double smeared_heaviside_step(double x)
{
  // A function that smoothly transition from 0 to 1 when x goes from -1 to 1.
  // For x < -1, the function should be 0. For x > 1, the function should be 1.
  // See Kawamoto et al. (2016).
  if (x <= -1) { // Outside and far away from boundary
    return 0.0;
  } else if (x >= 1) { // Inside and far away from boundary
    return 1.0;
  } else { // Close to boundary
    return 0.5 * (1.0 + x + sin(MY_PI * x) / MY_PI);
  }
}

/* ---------------------------------------------------------------------- */

double compute_volume(int dimension, int *grid_size, double stride, double *grid_values, double epsilon)
{
  // Volume integration without centre of mass (re)computation and level-set offset epsilon

  // This is the reference distance values that determines the smearing with of
  // the Heaviside step function. Current expression is the half-diagional of the
  // grid cell divided by a smearing constant.
  double smearCoeff = 1.5;
  double ls_ref = 1.0;
  if (smearCoeff != 0)
    ls_ref = sqrt(0.75) * stride / smearCoeff;

  // Initialise volume and centre of mass
  double volume = 0.0;
  // Cell volume, temporary grid points, integration volume.
  double volume_cell = stride * stride;
  if (dimension == 3) volume_cell *= stride;

  // Integration
  double dV, ls_val;
  for (int ind_x = 0; ind_x < grid_size[0]; ind_x++) {
    for (int ind_y = 0; ind_y < grid_size[1]; ind_y++) {
      for (int ind_z = 0; ind_z < grid_size[2]; ind_z++) {
        ls_val = grid_values[ind_x + ind_y * grid_size[0] + ind_z * grid_size[0] * grid_size[1]] + epsilon;
        dV = smeared_heaviside_step(-ls_val / ls_ref) * volume_cell;
        if (dV > 0.0) {
          volume += dV;
        }
      }
    }
  }
  return volume;
}

/* --------------------------------------------------------------------------------------
   Improved surface area calculation w.r.t. Duriez and Galusinski (2025) Comp. Phys. Comm.
--------------------------------------------------------------------------------------- */

double compute_surface_area(int dimension, int *grid_size, double stride, double *grid_values)
{
  // Computation of the surface area as the volume derivative over a thin shell of one grid stride.
  double epsilon, vol_in, vol_out, area;
  // Value of epsilon below gives the most accurate results. Why? Level set does not have more information
  // than is in the grid, and larger values increase error on the finite-difference approximation.
  epsilon = 0.5*stride;
  vol_in = compute_volume(dimension, grid_size, stride, grid_values, epsilon);
  vol_out = compute_volume(dimension, grid_size, stride, grid_values, -epsilon);
  // Finite central difference
  area = (vol_out - vol_in) / (2.0 * epsilon);

  return area;
}

}