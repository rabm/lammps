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

/* ----------------------------------------------------------------------
   PRECISION / MERGE-CONFLICT NOTE (do not remove).
   The ls_dem_* vector/quaternion ops below intentionally DUPLICATE
   MathExtraKokkos::{add3,sub3,len3,lensq3,dot3,cross3,negate3,norm3,qconjugate}
   (upstream src/KOKKOS/math_extra_kokkos.h, branch kokkos-rigid-small) but are
   hardcoded `double`. That is deliberate: KK_FLOAT is `float` in single- and
   mixed-precision Kokkos builds (double only under LMP_KOKKOS_DOUBLE_DOUBLE),
   and the float contamination reaches INSIDE the upstream integration helpers'
   temporaries (richardson / quat_to_mat / mq_to_omega), not just the generic-op
   signatures. The LS-DEM /kk correctness gate is within-tol DOUBLE, so do NOT
   replace these with MathExtraKokkos, and do NOT #include math_extra_kokkos.h on
   the contact / integration path, without re-validating precision. Keeping a
   separate header + namespace (LSDEMExtra) also avoids a file/namespace collision
   when upstream's rigid/small/kk port merges into develop.
------------------------------------------------------------------------- */

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

/* ----------------------------------------------------------------------
   Rigid-body integration math for the M7 (increment 2b) device kernels.
   DOUBLE-precision replicas transcribed VERBATIM from math_extra.{h,cpp}
   (NOT MathExtraKokkos, which is KK_FLOAT in single/mixed builds and would
   both break the correctness gate and collide with the upstream file/namespace).
   Operation order is preserved exactly (matters for the within-tol gate).
------------------------------------------------------------------------- */

// quaternion normalize (math_extra.h qnormalize)
KOKKOS_INLINE_FUNCTION
void ls_dem_qnormalize(double *q)
{
  double norm = 1.0 / sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  q[0] *= norm; q[1] *= norm; q[2] *= norm; q[3] *= norm;
}

// vector-quaternion multiply c = a*b, a = (0,a) (math_extra.h vecquat)
KOKKOS_INLINE_FUNCTION
void ls_dem_vecquat(const double *a, const double *b, double *c)
{
  c[0] = -a[0]*b[1] - a[1]*b[2] - a[2]*b[3];
  c[1] =  b[0]*a[0] + a[1]*b[3] - a[2]*b[2];
  c[2] =  b[0]*a[1] + a[2]*b[1] - a[0]*b[3];
  c[3] =  b[0]*a[2] + a[0]*b[2] - a[1]*b[1];
}

// quaternion-quaternion multiply c = a*b (math_extra.h quatquat)
KOKKOS_INLINE_FUNCTION
void ls_dem_quatquat(const double *a, const double *b, double *c)
{
  c[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
  c[1] = a[0]*b[1] + b[0]*a[1] + a[2]*b[3] - a[3]*b[2];
  c[2] = a[0]*b[2] + b[0]*a[2] + a[3]*b[1] - a[1]*b[3];
  c[3] = a[0]*b[3] + b[0]*a[3] + a[1]*b[2] - a[2]*b[1];
}

// matrix times vector (math_extra.h matvec)
KOKKOS_INLINE_FUNCTION
void ls_dem_matvec(const double m[3][3], const double *v, double *ans)
{
  ans[0] = m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2];
  ans[1] = m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2];
  ans[2] = m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2];
}

// transposed matrix times vector (math_extra.h transpose_matvec)
KOKKOS_INLINE_FUNCTION
void ls_dem_transpose_matvec(const double m[3][3], const double *v, double *ans)
{
  ans[0] = m[0][0]*v[0] + m[1][0]*v[1] + m[2][0]*v[2];
  ans[1] = m[0][1]*v[0] + m[1][1]*v[1] + m[2][1]*v[2];
  ans[2] = m[0][2]*v[0] + m[1][2]*v[1] + m[2][2]*v[2];
}

// rotation matrix from quaternion (math_extra.cpp quat_to_mat)
KOKKOS_INLINE_FUNCTION
void ls_dem_quat_to_mat(const double *quat, double mat[3][3])
{
  double w2 = quat[0]*quat[0];
  double i2 = quat[1]*quat[1];
  double j2 = quat[2]*quat[2];
  double k2 = quat[3]*quat[3];
  double twoij = 2.0*quat[1]*quat[2];
  double twoik = 2.0*quat[1]*quat[3];
  double twojk = 2.0*quat[2]*quat[3];
  double twoiw = 2.0*quat[1]*quat[0];
  double twojw = 2.0*quat[2]*quat[0];
  double twokw = 2.0*quat[3]*quat[0];
  mat[0][0] = w2+i2-j2-k2; mat[0][1] = twoij-twokw; mat[0][2] = twojw+twoik;
  mat[1][0] = twoij+twokw; mat[1][1] = w2-i2+j2-k2; mat[1][2] = twojk-twoiw;
  mat[2][0] = twoik-twojw; mat[2][1] = twojk+twoiw; mat[2][2] = w2-i2-j2+k2;
}

// omega from angular momentum, space frame, via principal axes (math_extra.cpp angmom_to_omega)
KOKKOS_INLINE_FUNCTION
void ls_dem_angmom_to_omega(const double *m, const double *ex, const double *ey,
                            const double *ez, const double *idiag, double *w)
{
  double wbody[3];
  if (idiag[0] == 0.0) wbody[0] = 0.0;
  else wbody[0] = (m[0]*ex[0] + m[1]*ex[1] + m[2]*ex[2]) / idiag[0];
  if (idiag[1] == 0.0) wbody[1] = 0.0;
  else wbody[1] = (m[0]*ey[0] + m[1]*ey[1] + m[2]*ey[2]) / idiag[1];
  if (idiag[2] == 0.0) wbody[2] = 0.0;
  else wbody[2] = (m[0]*ez[0] + m[1]*ez[1] + m[2]*ez[2]) / idiag[2];
  w[0] = wbody[0]*ex[0] + wbody[1]*ey[0] + wbody[2]*ez[0];
  w[1] = wbody[0]*ex[1] + wbody[1]*ey[1] + wbody[2]*ez[1];
  w[2] = wbody[0]*ex[2] + wbody[1]*ey[2] + wbody[2]*ez[2];
}

// omega from angular momentum, via the quaternion (math_extra.cpp mq_to_omega)
KOKKOS_INLINE_FUNCTION
void ls_dem_mq_to_omega(const double *m, const double *q, const double *moments, double *w)
{
  double wbody[3];
  double rot[3][3];
  ls_dem_quat_to_mat(q, rot);
  ls_dem_transpose_matvec(rot, m, wbody);
  if (moments[0] == 0.0) wbody[0] = 0.0; else wbody[0] /= moments[0];
  if (moments[1] == 0.0) wbody[1] = 0.0; else wbody[1] /= moments[1];
  if (moments[2] == 0.0) wbody[2] = 0.0; else wbody[2] /= moments[2];
  ls_dem_matvec(rot, wbody, w);
}

// space-frame ex,ey,ez from quaternion (math_extra.cpp q_to_exyz)
KOKKOS_INLINE_FUNCTION
void ls_dem_q_to_exyz(const double *q, double *ex, double *ey, double *ez)
{
  ex[0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
  ex[1] = 2.0 * (q[1]*q[2] + q[0]*q[3]);
  ex[2] = 2.0 * (q[1]*q[3] - q[0]*q[2]);
  ey[0] = 2.0 * (q[1]*q[2] - q[0]*q[3]);
  ey[1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
  ey[2] = 2.0 * (q[2]*q[3] + q[0]*q[1]);
  ez[0] = 2.0 * (q[1]*q[3] + q[0]*q[2]);
  ez[1] = 2.0 * (q[2]*q[3] - q[0]*q[1]);
  ez[2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
}

// Richardson iteration: update quaternion q from angular velocity w + angular
// momentum m, return omega at the 1/2 step (math_extra.cpp richardson).
// Operation order MUST match the host for the within-tol gate.
KOKKOS_INLINE_FUNCTION
void ls_dem_richardson(double *q, double *m, double *w, const double *moments, double dtq)
{
  double wq[4];
  ls_dem_vecquat(w, q, wq);

  double qfull[4];
  qfull[0] = q[0] + dtq*wq[0]; qfull[1] = q[1] + dtq*wq[1];
  qfull[2] = q[2] + dtq*wq[2]; qfull[3] = q[3] + dtq*wq[3];
  ls_dem_qnormalize(qfull);

  double qhalf[4];
  qhalf[0] = q[0] + 0.5*dtq*wq[0]; qhalf[1] = q[1] + 0.5*dtq*wq[1];
  qhalf[2] = q[2] + 0.5*dtq*wq[2]; qhalf[3] = q[3] + 0.5*dtq*wq[3];
  ls_dem_qnormalize(qhalf);

  ls_dem_mq_to_omega(m, qhalf, moments, w);
  ls_dem_vecquat(w, qhalf, wq);

  qhalf[0] += 0.5*dtq*wq[0]; qhalf[1] += 0.5*dtq*wq[1];
  qhalf[2] += 0.5*dtq*wq[2]; qhalf[3] += 0.5*dtq*wq[3];
  ls_dem_qnormalize(qhalf);

  q[0] = 2.0*qhalf[0] - qfull[0]; q[1] = 2.0*qhalf[1] - qfull[1];
  q[2] = 2.0*qhalf[2] - qfull[2]; q[3] = 2.0*qhalf[3] - qfull[3];
  ls_dem_qnormalize(q);
}

}    // namespace LSDEMExtra

#ifdef LSDEM_KK_DEVICE_FALLBACK
#undef KOKKOS_INLINE_FUNCTION
#undef LSDEM_KK_DEVICE_FALLBACK
#endif

#endif
