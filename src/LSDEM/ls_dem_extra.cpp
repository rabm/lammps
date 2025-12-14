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
using namespace LSDEMExtra;

static constexpr double EPSILON_INERTIA = 1.0e-3; // 0.1%

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

/* ----------------------------------------------------------------------
  Compute CoM, moment of inertia, and volume of a grid
------------------------------------------------------------------------- */

double LSDEMExtra::compute_grid_properties(int *grid_size, double stride, double *grid_values, double *com_temp, double inertia_temp[3][3], int dimension)
{
  // Volume integration

  // This is the reference distance values that determines the smearing with of
  // the Heaviside step function. Current expression is the half-diagional of the
  // grid cell divided by a smearing constant.
  double smearCoeff = 1.5;
  double ls_ref = 1.0;
  if (smearCoeff != 0)
    ls_ref = sqrt(0.75) * stride / smearCoeff;

  // Cell volume, temporary grid points, integration volume.
  double volume_cell = stride * stride;
  if (dimension == 3) volume_cell *= stride;

  // Integration
  double dV, ls_val;
  double volume = 0.0;
  for (int a = 0; a < 3; a++) com_temp[a] = 0.0;
  for (int ind_x = 0; ind_x < grid_size[0]; ind_x++) {
    for (int ind_y = 0; ind_y < grid_size[1]; ind_y++) {
      for (int ind_z = 0; ind_z < grid_size[2]; ind_z++) {
        ls_val = grid_values[ind_x + ind_y * grid_size[0] + ind_z * grid_size[0] * grid_size[1]];
        dV = smeared_heaviside_step(-ls_val / ls_ref) * volume_cell;
        if (dV > 0.0) {
          volume += dV;
          com_temp[0] += ind_x * stride * dV;
          com_temp[1] += ind_y * stride * dV;
          com_temp[2] += ind_z * stride * dV;
        }
      }
    }
  }
  com_temp[0] /= volume;
  com_temp[1] /= volume;
  com_temp[2] /= volume;

  // Computing the inertia tensor (a second loop is unavoidable)
  double delx, dely, delz;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      inertia_temp[a][b] = 0.0;
    }
  }
  for (int ind_x = 0; ind_x < grid_size[0]; ind_x++) {
    for (int ind_y = 0; ind_y < grid_size[1]; ind_y++) {
      for (int ind_z = 0; ind_z < grid_size[2]; ind_z++) {
        ls_val = grid_values[ind_x + ind_y * grid_size[0] + ind_z * grid_size[0] * grid_size[1]];
        dV = smeared_heaviside_step(-ls_val / ls_ref) * volume_cell;
        if (dV > 0.0) {
          delx = ind_x * stride - com_temp[0];
          dely = ind_y * stride - com_temp[1];
          delz = ind_z * stride - com_temp[2];
          inertia_temp[0][0] += (dely * dely + delz * delz) * dV;
          inertia_temp[1][1] += (delx * delx + delz * delz) * dV;
          inertia_temp[2][2] += (delx * delx + dely * dely) * dV;
          inertia_temp[0][1] -= delx * dely * dV;
          inertia_temp[0][2] -= delx * delz * dV;
          inertia_temp[1][2] -= dely * delz * dV;
        }
      }
    }
  }
  inertia_temp[1][0] = inertia_temp[0][1];
  inertia_temp[2][0] = inertia_temp[0][2];
  inertia_temp[2][1] = inertia_temp[1][2];
  // Check to see if level set has a non-inertial reference frame
  double I_diag_norm = sqrt(inertia_temp[0][0] * inertia_temp[0][0] + inertia_temp[1][1] * inertia_temp[1][1] + inertia_temp[2][2] * inertia_temp[2][2]);
  double I_off_diag_norm = sqrt(2.0 * (inertia_temp[0][1] * inertia_temp[0][1] + inertia_temp[0][2] * inertia_temp[0][2] + inertia_temp[1][2] * inertia_temp[1][2]));
  if (I_off_diag_norm / I_diag_norm > EPSILON_INERTIA)
    volume = -1;

  return volume;
}
