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

#include "error.h"
#include "math_const.h"
#include "math_extra.h"
#include "rigid_ls_dem_const.h"

#include <cmath>
#include <unordered_map>
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

/* ----------------------------------------------------------------------
  Compute CoM, moment of inertia, and volume of a grid
------------------------------------------------------------------------- */

double compute_grid_properties(int *grid_size, double stride, double *grid_values, double *com, double *inertia, int dimension)
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
  for (int a = 0; a < 3; a++) com[a] = 0.0;
  for (int ind_x = 0; ind_x < nx; ind_x++) {
    for (int ind_y = 0; ind_y < ny; ind_y++) {
      for (int ind_z = 0; ind_z < nz; ind_z++) {
        idx = ind_x + ind_y * nx + ind_z * nx * ny;
        ls_val = grid_values[idx];
        h = smeared_heaviside_step(-ls_val / ls_ref);
        heaviside_vals[idx] = h;
        if (h > 0.0) {
          volume += h;
          com[0] += ind_x * h;
          com[1] += ind_y * h;
          com[2] += ind_z * h;
        }
      }
    }
  }
  com[0] /= volume; // Still all in voxel units
  com[1] /= volume;
  com[2] /= volume;

  // Computing the inertia tensor (a second loop is unavoidable)
  double delx, dely, delz, delxx, delyy, delzz;
  for (int a = 0; a < 6; a++)
    inertia[a] = 0.0;
  for (int ind_x = 0; ind_x < nx; ind_x++) {
    for (int ind_y = 0; ind_y < ny; ind_y++) {
      for (int ind_z = 0; ind_z < nz; ind_z++) {
        idx = ind_x + ind_y * nx + ind_z * nx * ny;
        h = heaviside_vals[idx];
        if (h > 0.0) {
          delx = ind_x - com[0];
          dely = ind_y - com[1];
          delz = ind_z - com[2];
          delxx = delx * delx;
          delyy = dely * dely;
          delzz = delz * delz;
          inertia[0] += (delyy + delzz) * h;
          inertia[1] += (delxx + delzz) * h;
          inertia[2] += (delxx + delyy) * h;
          inertia[3] -= dely * delz * h;
          inertia[4] -= delx * delz * h;
          inertia[5] -= delx * dely * h;
        }
      }
    }
  }

  // Cell volume
  double volume_cell = stride * stride;
  if (dimension == 3) volume_cell *= stride;

  // Back to real units
  volume *= volume_cell;
  com[0] *= stride;
  com[1] *= stride;
  com[2] *= stride;
  double Iscale = volume_cell * stride * stride; // Works in both 2D and 3D
  inertia[0] *= Iscale;
  inertia[1] *= Iscale;
  inertia[2] *= Iscale;
  inertia[3] *= Iscale;
  inertia[4] *= Iscale;
  inertia[5] *= Iscale;

  // Check to see if level set has a non-inertial reference frame
  double I_diag_norm = sqrt(inertia[0] * inertia[0] + inertia[1] * inertia[1] + inertia[2] * inertia[2]);
  double I_off_diag_norm = sqrt(2.0 * (inertia[5] * inertia[5] + inertia[4] * inertia[4] + inertia[3] * inertia[3]));
  if (I_off_diag_norm / I_diag_norm > EPSILON_INERTIA)
    volume = -1;

  return volume;
}

/* ----------------------------------------------------------------------
  Perform trilinear interpolation to get level-set value and normal
-------------------------------------------------------------------------*/

double interpolate_LS_array(int dimension, int mybin, double *mygrid, int *ngrid, double *x_red, int *ix, double nvec[3], double stride)
{
  double dist;

  // Checking whether x_local lies within the grid. Avoids edge cases where finite precision
  // leads to e.g. a x=-0.1 coordinate to fall outside of a grid that starts at x=-0.1.
  if ((ix[0] < 0 || ix[0] >= (ngrid[0] - 1)) || (ix[1] < 0 || ix[1] >= (ngrid[1] - 1)) ||
      ((dimension == 3) && (ix[2] < 0 || ix[2] >= (ngrid[2] - 1))))
    return BIG; // To avoid having to perfectly match the neighbour listing cutoff with the grid size.

  // Level-set value of lower corner
  double ls000 = mygrid[mybin];

  // Short circuit the level-set interpolation if we know we're so far from the surface we won't use the
  // value anyway. NB: Need adjustment for bonding. Voxel diagonal is at most sqrt(3)*stride = 1.7*stride
  // so 2.0 is safe.
  if (ls000 > 2.0 * stride) {
    return ls000;
  }

  // Rest of the level-set values on the grid points in the lower z plane (ind_z)
  double ls100 = mygrid[mybin + 1];
  double ls010 = mygrid[mybin + ngrid[0]];
  double ls110 = mygrid[mybin + 1 + ngrid[0]]; // move this and // Interpolate upwards, add short circuit!!

  // The normalised coordinates within the current grid cell.
  // May be safer to cap them with math::max(math::min(x_red, 1.0), 0.0)
  double x_red_local[3];
  x_red_local[0] = x_red[0] - static_cast<double>(ix[0]);
  x_red_local[1] = x_red[1] - static_cast<double>(ix[1]);
  x_red_local[2] = x_red[2] - static_cast<double>(ix[2]); // Should always be zero in 2D.

  // Bi-linear interpolation in the lower z plane (ind_z)
  double lsxy0 = ls000 + x_red_local[1] * (ls010 - ls000) +
                 x_red_local[0] * (ls100 - ls000 + x_red_local[1] * (ls110 - ls100 - ls010 + ls000));
  dist = lsxy0;

  // Computing normal as the gradient of trilinear interpolation
  // Chain rule: d(dist)/d(x_local) = d(dist)/d(x_red) * (1/stride)
  // Vector eventually normalized to enforce unit normal, so 1/stride factor omitted
  nvec[0] = ls100 - ls000 + x_red_local[1] * (ls110 - ls100 - ls010 + ls000);
  nvec[1] = ls010 - ls000 + x_red_local[0] * (ls110 - ls100 - ls010 + ls000);
  nvec[2] = 0.0;

  if (dimension == 3) { // 3D
    // Level-set values on the grid points in the upper z plane (ind_z+1)
    double ls001 = mygrid[mybin + ngrid[0] * ngrid[1]];
    double ls101 = mygrid[mybin + 1 + ngrid[0] * ngrid[1]];
    double ls011 = mygrid[mybin + ngrid[0] + ngrid[0] * ngrid[1]];
    double ls111 = mygrid[mybin + 1 + ngrid[0] + ngrid[0] * ngrid[1]];

    // Bi-linear interpolation in the upper z plane (ind_z+1)
    double lsxy1 = ls001 + x_red_local[1] * (ls011 - ls001) +
                   x_red_local[0] * (ls101 - ls001 + x_red_local[1] * (ls111 - ls101 - ls011 + ls001));

    // Affecting tri-linear interpolation by linear interpolation of the two bi-linear interpolations.
    dist = x_red_local[2] * (lsxy1 - lsxy0) + lsxy0;
    nvec[0] *= 1 - x_red_local[2];
    nvec[0] += x_red_local[2] * (ls101 - ls001 + x_red_local[1] * (ls111 - ls101 - ls011 + ls001));
    nvec[1] *= 1 - x_red_local[2];
    nvec[1] += x_red_local[2] * (ls011 - ls001 + x_red_local[0] * (ls111 - ls101 - ls011 + ls001));
    nvec[2] = lsxy1 - lsxy0;
  }

  // Normal normally doesn't need scaling, but we scaled grid_min and grid_stride
  // but not the level-set values, hence it is necessary. However, we'll normalise later anyway.

  MathExtra::norm3(nvec);

  return dist;
}

/* ----------------------------------------------------------------------
  Search two unordered maps (owned + buffer) for LS value
-------------------------------------------------------------------------*/

double get_ws_ls_value(int mybin, std::unordered_map<int, double> *mytable, std::unordered_map<int, double> *mybuffer)
{
  if (mytable->find(mybin) != mytable->end())
    return mytable->at(mybin);
  if (mybuffer->find(mybin) != mybuffer->end())
    return mybuffer->at(mybin);
  return BIG;
}

/* ----------------------------------------------------------------------
  Perform trilinear interpolation to get level-set value and normal
-------------------------------------------------------------------------*/

double interpolate_LS_watershed(int dimension, int mybin, std::unordered_map<int, double> *mytable, std::unordered_map<int, double> *mybuffer,
                                int ngrid[3], double x_red[3], int ix[3], double nvec[3], double stride)
{
  double dist;

  // Checking whether x_local lies within the grid. Avoids edge cases where finite precision
  // leads to e.g. a x=-0.1 coordinate to fall outside of a grid that starts at x=-0.1.
  if ((ix[0] < 0 || ix[0] >= (ngrid[0] - 1)) || (ix[1] < 0 || ix[1] >= (ngrid[1] - 1)) ||
      ((dimension == 3) && (ix[2] < 0 || ix[2] >= (ngrid[2] - 1))))
    return BIG; // To avoid having to perfectly match the neighbour listing cutoff with the grid size.

  // Level-set value of lower corner
  if (mytable->find(mybin) == mytable->end())
    return BIG;

  double ls000 = mytable->at(mybin);

  // Short circuit the level-set interpolation if we know we're so far from the surface we won't use the
  // value anyway. NB: Need adjustment for bonding. Voxel diagonal is at most sqrt(3)*stride = 1.7*stride
  // so 2.0 is safe.
  if (ls000 > 2.0 * stride)
    return ls000;

  double ls100 = get_ws_ls_value(mybin + 1, mytable, mybuffer);
  double ls010 = get_ws_ls_value(mybin + ngrid[0], mytable, mybuffer);
  double ls110 = get_ws_ls_value(mybin + 1 + ngrid[0], mytable, mybuffer);

  // Rest of the level-set values on the grid points in the lower z plane (ind_z)
  if (ls100 == BIG || ls010 == BIG || ls110 == BIG)
    return BIG;

  // The normalised coordinates within the current grid cell.
  // May be safer to cap them with math::max(math::min(x_red, 1.0), 0.0)
  double x_red_local[3];
  x_red_local[0] = x_red[0] - static_cast<double>(ix[0]);
  x_red_local[1] = x_red[1] - static_cast<double>(ix[1]);
  x_red_local[2] = x_red[2] - static_cast<double>(ix[2]); // Should always be zero in 2D.

  // Bi-linear interpolation in the lower z plane (ind_z)
  double lsxy0 = ls000 + x_red_local[1] * (ls010 - ls000) +
                 x_red_local[0] * (ls100 - ls000 + x_red_local[1] * (ls110 - ls100 - ls010 + ls000));
  dist = lsxy0;

  // Computing normal as the gradient of trilinear interpolation
  // Chain rule: d(dist)/d(x_local) = d(dist)/d(x_red) * (1/stride)
  // Vector eventually normalized to enforce unit normal, so 1/stride factor omitted
  nvec[0] = ls100 - ls000 + x_red_local[1] * (ls110 - ls100 - ls010 + ls000);
  nvec[1] = ls010 - ls000 + x_red_local[0] * (ls110 - ls100 - ls010 + ls000);
  nvec[2] = 0.0;
  if (dimension == 3) { // 3D

    double ls001 = get_ws_ls_value(mybin + ngrid[0] * ngrid[1], mytable, mybuffer);
    double ls101 = get_ws_ls_value(mybin + 1 + ngrid[0] * ngrid[1], mytable, mybuffer);
    double ls011 = get_ws_ls_value(mybin + ngrid[0] + ngrid[0] * ngrid[1], mytable, mybuffer);
    double ls111 = get_ws_ls_value(mybin + 1 + ngrid[0] + ngrid[0] * ngrid[1], mytable, mybuffer);
    // Level-set values on the grid points in the upper z plane (ind_z+1)
    if (ls001 == BIG || ls101 == BIG || ls011 == BIG || ls111 == BIG)
      return BIG;

    // Bi-linear interpolation in the upper z plane (ind_z+1)
    double lsxy1 = ls001 + x_red_local[1] * (ls011 - ls001) +
                   x_red_local[0] * (ls101 - ls001 + x_red_local[1] * (ls111 - ls101 - ls011 + ls001));

    // Affecting tri-linear interpolation by linear interpolation of the two bi-linear interpolations.
    dist = x_red_local[2] * (lsxy1 - lsxy0) + lsxy0;
    nvec[0] *= 1 - x_red_local[2];
    nvec[0] += x_red_local[2] * (ls101 - ls001 + x_red_local[1] * (ls111 - ls101 - ls011 + ls001));
    nvec[1] *= 1 - x_red_local[2];
    nvec[1] += x_red_local[2] * (ls011 - ls001 + x_red_local[0] * (ls111 - ls101 - ls011 + ls001));
    nvec[2] = lsxy1 - lsxy0;
  }


  // Normal normally doesn't need scaling, but we scaled grid_min and grid_stride
  // but not the level-set values, hence it is necessary. However, we'll normalise later anyway.

  MathExtra::norm3(nvec);

  return dist;
}

/* ----------------------------------------------------------------------
  Store local grid values and calculate grid minima
-------------------------------------------------------------------------*/

int store_distributed(int i, int dimension, int *nx, int *subgrid_size, double stride, double scale, double rcell,
                      double *dx, double *gmin, double *qatom, double *global_grid_values, double *gmin_local, double *grid_values)
{
  int need_padding = 0;

  // Rotate to LS frame (for now, just the atomic quaternion)
  double quat_conj[4], dx_local[3];
  MathExtra::qconjugate(qatom, quat_conj);
  MathExtra::quatrotvec(quat_conj, dx, dx_local);

  // Location of atom/node relative to entire grain grid minimum.
  MathExtra::sub3(dx_local, gmin, dx_local);

  // Index of atom/node in entire grain grid.
  int ix_node[3];
  ix_node[0] = int(dx_local[0] / stride);
  ix_node[1] = int(dx_local[1] / stride);
  ix_node[2] = int(dx_local[2] / stride);

  // Index of local grid minimum in entire grain grid. If any goes below zero, error below catches it.
  int index_grid_min_local[3];
  index_grid_min_local[0] = ix_node[0] - rcell;
  index_grid_min_local[1] = ix_node[1] - rcell;
  index_grid_min_local[2] = (dimension == 3) ? ix_node[2] - rcell : 0;

  // Location of local grid minimum relative to CoM
  gmin_local[0] = index_grid_min_local[0] * stride + gmin[0];
  gmin_local[1] = index_grid_min_local[1] * stride + gmin[1];
  gmin_local[2] = index_grid_min_local[2] * stride + gmin[2];

  int ix_global, iy_global, iz_global, index_local, index_global;
  for (int iz_local = 0; iz_local < subgrid_size[2]; iz_local++) {
    for (int iy_local = 0; iy_local < subgrid_size[1]; iy_local++) {
      for (int ix_local = 0; ix_local < subgrid_size[0]; ix_local++) {
        index_local = ix_local + iy_local * subgrid_size[0] + iz_local * subgrid_size[0] * subgrid_size[1];

        // Shift local cell to global cell
        ix_global = ix_local + index_grid_min_local[0];
        iy_global = iy_local + index_grid_min_local[1];
        iz_global = iz_local + index_grid_min_local[2];

        // Explicit bounds check per dimension (safer and clearer)
        if (ix_global < 0 || ix_global >= nx[0] ||
            iy_global < 0 || iy_global >= nx[1] ||
            iz_global < 0 || iz_global >= nx[2]) {
          need_padding = 1;
          grid_values[index_local] = BIG;
        } else {
          // True (scaled) level-set stored for DISTRIBUTED approach where unique local grid is saved on node
          index_global = ix_global + iy_global * nx[0] + iz_global * nx[0] * nx[1];
          if (index_global < 0 || index_global >= nx[0] * nx[1] * nx[2])
            return -1;

          grid_values[index_local] = global_grid_values[index_global] * scale;
        }
      }
    }
  }

  return need_padding;
}

}