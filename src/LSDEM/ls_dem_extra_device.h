/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Device-callable LS-DEM helpers for the GPU/Kokkos port (Milestone 4a).
   interpolate_LS_array lived in ls_dem_extra.cpp; it is moved here as a
   KOKKOS_INLINE_FUNCTION so the device force kernel (K2, M4b) can call it,
   and ls_dem_extra.cpp now includes this header to keep the CPU symbol
   (the CPU result is unchanged -- ls_dem_norm3 matches MathExtra::norm3).

   The header is included by NON-Kokkos CPU files too (the rigid fixes),
   so KOKKOS_INLINE_FUNCTION must degrade to `inline` when Kokkos is absent.
------------------------------------------------------------------------- */

#ifndef LMP_LS_DEM_EXTRA_DEVICE_H
#define LMP_LS_DEM_EXTRA_DEVICE_H

#include "rigid_ls_dem_const.h"   // BIG
#include <cmath>

// degrade KOKKOS_INLINE_FUNCTION to plain inline in non-Kokkos builds
#ifndef KOKKOS_INLINE_FUNCTION
#define LSDEM_KK_DEVICE_FALLBACK
#define KOKKOS_INLINE_FUNCTION inline
#endif

namespace LSDEMExtra {

using LAMMPS_NS::RigidLSDEMConst::BIG;

// ---- device-callable replica of MathExtra::norm3 (MathExtra is host-only) ----
// Matches MathExtra::norm3 arithmetic exactly so the CPU interpolate path stays bitwise.
KOKKOS_INLINE_FUNCTION
void ls_dem_norm3(double *v)
{
  double scale = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  v[0] *= scale;
  v[1] *= scale;
  v[2] *= scale;
}

// ---- device-callable replicas of the remaining MathExtra contact-path ops ----
// Used by the K2 force kernel (M4b). MathExtra is host-only, so these mirror its
// formulas verbatim. They are device-only (the /kk gate is within-tol, not bitwise),
// but matching the arithmetic exactly keeps /kk as close to the CPU path as possible.

KOKKOS_INLINE_FUNCTION
double ls_dem_dot3(const double *v1, const double *v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

KOKKOS_INLINE_FUNCTION
double ls_dem_len3(const double *v)
{
  return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

KOKKOS_INLINE_FUNCTION
double ls_dem_lensq3(const double *v)
{
  return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

KOKKOS_INLINE_FUNCTION
void ls_dem_cross3(const double *v1, const double *v2, double *ans)
{
  ans[0] = v1[1] * v2[2] - v1[2] * v2[1];
  ans[1] = v1[2] * v2[0] - v1[0] * v2[2];
  ans[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

KOKKOS_INLINE_FUNCTION
void ls_dem_sub3(const double *v1, const double *v2, double *ans)
{
  ans[0] = v1[0] - v2[0];
  ans[1] = v1[1] - v2[1];
  ans[2] = v1[2] - v2[2];
}

KOKKOS_INLINE_FUNCTION
void ls_dem_add3(const double *v1, const double *v2, double *ans)
{
  ans[0] = v1[0] + v2[0];
  ans[1] = v1[1] + v2[1];
  ans[2] = v1[2] + v2[2];
}

KOKKOS_INLINE_FUNCTION
void ls_dem_negate3(double *v)
{
  v[0] = -v[0];
  v[1] = -v[1];
  v[2] = -v[2];
}

// conjugate of a quaternion (assumes unit length): qc = conj(q)
KOKKOS_INLINE_FUNCTION
void ls_dem_qconjugate(const double *q, double *qc)
{
  qc[0] = q[0];
  qc[1] = -q[1];
  qc[2] = -q[2];
  qc[3] = -q[3];
}

// quaternion rotation of a vector: c = a * b * conj(a). b is read fully into temp
// before c is written, so in-place use (b == c, as in get_ls_value) is safe.
KOKKOS_INLINE_FUNCTION
void ls_dem_quatrotvec(const double *a, const double *b, double *c)
{
  double temp[4];

  // temp = a*b
  temp[0] = -a[1] * b[0] - a[2] * b[1] - a[3] * b[2];
  temp[1] = a[0] * b[0] + a[2] * b[2] - a[3] * b[1];
  temp[2] = a[0] * b[1] + a[3] * b[0] - a[1] * b[2];
  temp[3] = a[0] * b[2] + a[1] * b[1] - a[2] * b[0];

  // c = temp*conj(a)
  c[0] = -a[1] * temp[0] + a[0] * temp[1] - a[3] * temp[2] + a[2] * temp[3];
  c[1] = -a[2] * temp[0] + a[3] * temp[1] + a[0] * temp[2] - a[1] * temp[3];
  c[2] = -a[3] * temp[0] - a[2] * temp[1] + a[1] * temp[2] + a[0] * temp[3];
}

// ---- device-callable minimum image (orthogonal box) ----
// Replaces domain->minimum_image on the contact path. All LS-DEM cases are orthogonal;
// triclinic is not yet supported on device (a future item; no triclinic LS-DEM test exists).
// Caller passes the box periods + periodicity (px/py/pz = 1 if periodic in that dim).
KOKKOS_INLINE_FUNCTION
void ls_dem_minimum_image_ortho(double &dx, double &dy, double &dz,
                                double xprd, double yprd, double zprd,
                                int px, int py, int pz)
{
  if (px) {
    while (dx >  0.5 * xprd) dx -= xprd;
    while (dx < -0.5 * xprd) dx += xprd;
  }
  if (py) {
    while (dy >  0.5 * yprd) dy -= yprd;
    while (dy < -0.5 * yprd) dy += yprd;
  }
  if (pz) {
    while (dz >  0.5 * zprd) dz -= zprd;
    while (dz < -0.5 * zprd) dz += zprd;
  }
}

/* ----------------------------------------------------------------------
  Trilinear interpolation of the level-set value + normal (moved verbatim
  from ls_dem_extra.cpp; only MathExtra::norm3 -> ls_dem_norm3).
-------------------------------------------------------------------------*/

KOKKOS_INLINE_FUNCTION
double interpolate_LS_array(int dimension, int mybin, double *mygrid, int *ngrid,
                            double *x_red, int *ix, double nvec[3], double stride)
{
  double dist;

  if ((ix[0] < 0 || ix[0] >= (ngrid[0] - 1)) || (ix[1] < 0 || ix[1] >= (ngrid[1] - 1)) ||
      ((dimension == 3) && (ix[2] < 0 || ix[2] >= (ngrid[2] - 1))))
    return BIG;

  double ls000 = mygrid[mybin];

  if (ls000 > 2.0 * stride) {
    return ls000;
  }

  double ls100 = mygrid[mybin + 1];
  double ls010 = mygrid[mybin + ngrid[0]];
  double ls110 = mygrid[mybin + 1 + ngrid[0]];

  double x_red_local[3];
  x_red_local[0] = x_red[0] - static_cast<double>(ix[0]);
  x_red_local[1] = x_red[1] - static_cast<double>(ix[1]);
  x_red_local[2] = x_red[2] - static_cast<double>(ix[2]);

  double lsxy0 = ls000 + x_red_local[1] * (ls010 - ls000) +
                 x_red_local[0] * (ls100 - ls000 + x_red_local[1] * (ls110 - ls100 - ls010 + ls000));
  dist = lsxy0;

  nvec[0] = ls100 - ls000 + x_red_local[1] * (ls110 - ls100 - ls010 + ls000);
  nvec[1] = ls010 - ls000 + x_red_local[0] * (ls110 - ls100 - ls010 + ls000);
  nvec[2] = 0.0;

  if (dimension == 3) {
    double ls001 = mygrid[mybin + ngrid[0] * ngrid[1]];
    double ls101 = mygrid[mybin + 1 + ngrid[0] * ngrid[1]];
    double ls011 = mygrid[mybin + ngrid[0] + ngrid[0] * ngrid[1]];
    double ls111 = mygrid[mybin + 1 + ngrid[0] + ngrid[0] * ngrid[1]];

    double lsxy1 = ls001 + x_red_local[1] * (ls011 - ls001) +
                   x_red_local[0] * (ls101 - ls001 + x_red_local[1] * (ls111 - ls101 - ls011 + ls001));

    dist = x_red_local[2] * (lsxy1 - lsxy0) + lsxy0;
    nvec[0] *= 1 - x_red_local[2];
    nvec[0] += x_red_local[2] * (ls101 - ls001 + x_red_local[1] * (ls111 - ls101 - ls011 + ls001));
    nvec[1] *= 1 - x_red_local[2];
    nvec[1] += x_red_local[2] * (ls011 - ls001 + x_red_local[0] * (ls111 - ls101 - ls011 + ls001));
    nvec[2] = lsxy1 - lsxy0;
  }

  ls_dem_norm3(nvec);

  return dist;
}

}    // namespace LSDEMExtra

#ifdef LSDEM_KK_DEVICE_FALLBACK
#undef KOKKOS_INLINE_FUNCTION
#undef LSDEM_KK_DEVICE_FALLBACK
#endif

#endif
