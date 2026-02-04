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
#include "math_extra.h"
#include "rigid_ls_dem_const.h"

#include <cmath>
#include <vector>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace RigidLSDEMConst;
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
  epsilon = 0.5 * stride;
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

  // Preallocate storage for smeared Heaviside values so we don't recompute.
  const int nx = grid_size[0];
  const int ny = grid_size[1];
  const int nz = grid_size[2];
  const int n_cells = nx * ny * nz;
  std::vector<double> heaviside_vals(n_cells);

  // Volume integration
  double h, ls_val;
  int idx;
  double volume = 0.0; // In voxel units
  for (int a = 0; a < 3; a++) com_temp[a] = 0.0;
  for (int ind_x = 0; ind_x < nx; ind_x++) {
    for (int ind_y = 0; ind_y < ny; ind_y++) {
      for (int ind_z = 0; ind_z < nz; ind_z++) {
        idx = ind_x + ind_y * nx + ind_z * nx * ny;
        ls_val = grid_values[idx];
        h = smeared_heaviside_step(-ls_val / ls_ref);
        heaviside_vals[idx] = h;
        if (h > 0.0) {
          volume += h;
          com_temp[0] += ind_x * h;
          com_temp[1] += ind_y * h;
          com_temp[2] += ind_z * h;
        }
      }
    }
  }
  com_temp[0] /= volume; // Still all in voxel units
  com_temp[1] /= volume;
  com_temp[2] /= volume;

  // Computing the inertia tensor (a second loop is unavoidable)
  double delx, dely, delz, delxx, delyy, delzz;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      inertia_temp[a][b] = 0.0;
    }
  }
  for (int ind_x = 0; ind_x < nx; ind_x++) {
    for (int ind_y = 0; ind_y < ny; ind_y++) {
      for (int ind_z = 0; ind_z < nz; ind_z++) {
        idx = ind_x + ind_y * nx + ind_z * nx * ny;
        h = heaviside_vals[idx];
        if (h > 0.0) {
          delx = ind_x - com_temp[0];
          dely = ind_y - com_temp[1];
          delz = ind_z - com_temp[2];
          delxx = delx*delx;
          delyy = dely*dely;
          delzz = delz*delz;
          inertia_temp[0][0] += (delyy + delzz) * h;
          inertia_temp[1][1] += (delxx + delzz) * h;
          inertia_temp[2][2] += (delxx + delyy) * h;
          inertia_temp[0][1] -= delx * dely * h;
          inertia_temp[0][2] -= delx * delz * h;
          inertia_temp[1][2] -= dely * delz * h;
        }
      }
    }
  }

  // Cell volume
  double volume_cell = stride * stride;
  if (dimension == 3) volume_cell *= stride;

  // Back to real units
  volume *= volume_cell;
  com_temp[0] *= stride;
  com_temp[1] *= stride;
  com_temp[2] *= stride;
  double Iscale = volume_cell*stride*stride; // Works in both 2D and 3D
  inertia_temp[0][0] *= Iscale;
  inertia_temp[1][1] *= Iscale;
  inertia_temp[2][2] *= Iscale;
  inertia_temp[0][1] *= Iscale;
  inertia_temp[0][2] *= Iscale;
  inertia_temp[1][2] *= Iscale;

  // Populate other half of inertia tensor
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

/* ----------------------------------------------------------------------
  Perform trilinear interpolation to get level-set value and normal
-------------------------------------------------------------------------*/

double LSDEMExtra::interpolate_LS(int dimension, double *mygrid, int ncol, int nrow, int nslice,
                                 double x_red, double y_red, double z_red, double normal[3], double stride)
{
  double dist, nx, ny, nz(0.0);

  // Calculate index from relative coordinate, being careful with integer division.
  int ind_x = int(x_red);
  int ind_y = int(y_red);
  int ind_z = int(z_red); // Should always be zero in 2D.

  // Checking whether x_local lies within the grid. Avoids edge cases where finite precision
  // leads to e.g. a x=-0.1 coordinate to fall outside of a grid that starts at x=-0.1.
  if ((ind_x < 0 || ind_x >= (ncol - 1)) || (ind_y < 0 || ind_y >= (nrow - 1)) ||
      ((dimension == 3) && (ind_z < 0 || ind_z >= (nslice - 1))))
    return BIG; // To avoid having to perfectly match the neighbour listing cutoff with the grid size.

  //  Interpolate
  int my_index = ind_x + ind_y * ncol + ind_z * ncol * nrow;

  // Level-set value of lower corner
  double ls000 = mygrid[my_index];

  // Short circuit the level-set interpolation if we know we're so far from the surface we won't use the
  // value anyway. NB: Need adjustment for bonding. Voxel diagonal is at most sqrt(3)*stride = 1.7*stride
  // so 2.0 is safe.
  if (ls000 > 2.0*stride){
    return ls000;
  }

  // Rest of the level-set values on the grid points in the lower z plane (ind_z)
  double ls100 = mygrid[my_index + 1];
  double ls010 = mygrid[my_index + ncol];
  double ls110 = mygrid[my_index + 1 + ncol]; // move this and // Interpolate upwards, add short circuit!!

  // The normalised coordinates within the current grid cell.
  // May be safer to cap them with math::max(math::min(x_red, 1.0), 0.0)
  x_red = x_red - static_cast<double>(ind_x);
  y_red = y_red - static_cast<double>(ind_y);
  z_red = z_red - static_cast<double>(ind_z); // Should always be zero in 2D.

  // Bi-linear interpolation in the lower z plane (ind_z)
  double lsxy0 = ls000 + y_red * (ls010 - ls000) +
                 x_red * (ls100 - ls000 + y_red * (ls110 - ls100 - ls010 + ls000));
  dist = lsxy0;

  // Computing normal as the gradient of trilinear interpolation
  // Chain rule: d(dist)/d(x_local) = d(dist)/d(x_red) * (1/stride)
  // Vector eventually normalized to enforce unit normal, so 1/stride factor omitted
  nx = ls100 - ls000 + y_red * (ls110 - ls100 - ls010 + ls000);
  ny = ls010 - ls000 + x_red * (ls110 - ls100 - ls010 + ls000);

  if (dimension == 3) { // 3D
    // Level-set values on the grid points in the upper z plane (ind_z+1)
    double ls001 = mygrid[my_index + ncol * nrow];
    double ls101 = mygrid[my_index + 1 + ncol * nrow];
    double ls011 = mygrid[my_index + ncol + ncol * nrow];
    double ls111 = mygrid[my_index + 1 + ncol + ncol * nrow];

    // Bi-linear interpolation in the upper z plane (ind_z+1)
    double lsxy1 = ls001 + y_red * (ls011 - ls001) +
                   x_red * (ls101 - ls001 + y_red * (ls111 - ls101 - ls011 + ls001));

    // Affecting tri-linear interpolation by linear interpolation of the two bi-linear interpolations.
    dist = z_red * (lsxy1 - lsxy0) + lsxy0;
    nx *= 1 - z_red;
    nx += z_red * (ls101 - ls001 + y_red * (ls111 - ls101 - ls011 + ls001));
    ny *= 1 - z_red;
    ny += z_red * (ls011 - ls001 + x_red * (ls111 - ls101 - ls011 + ls001));
    nz = lsxy1 - lsxy0;
  }

  // Normal normally doesn't need scaling, but we scaled grid_min and grid_stride
  // but not the level-set values, hence it is necessary. However, we'll normalise later anyway.

  // Assign normal
  normal[0] = nx;
  normal[1] = ny;
  normal[2] = nz;
  MathExtra::norm3(normal);

  return dist;
}
