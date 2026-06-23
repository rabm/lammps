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

#ifdef FIX_CLASS
// clang-format off
FixStyle(rigid/small/ls/dem/kk,FixRigidSmallLSDEMKokkos<LMPDeviceType>);
FixStyle(rigid/small/ls/dem/kk/device,FixRigidSmallLSDEMKokkos<LMPDeviceType>);
FixStyle(rigid/small/ls/dem/kk/host,FixRigidSmallLSDEMKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_RIGID_SMALL_LS_DEM_KOKKOS_H
#define LMP_FIX_RIGID_SMALL_LS_DEM_KOKKOS_H

#include "fix_rigid_small_ls_dem.h"
#include "kokkos_base.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

// GPU/Kokkos variant of fix rigid/small/ls/dem (M7 milestone).
//
// INCREMENT 1 ported ONLY the per-body force/torque reduction
// (compute_forces_and_torques) to the device.
//
// INCREMENT 2a (this file) is the RESIDENCY LIFECYCLE scaffold: it makes the
// fix kokkosable and converts the five FixRigidSmall per-atom arrays
// (bodyown/bodytag/atom2body/xcmimage/displace) into AtomKokkos-style DualViews
// so the device reduction reads them on the device while the existing host
// comm/exchange/sort keep writing the host side. The body integration,
// set_xv/set_v and the grain-field scatter STILL run on the host base
// (wrapped here with atomKK->sync(Host,...)/modified(Host,...) so the legacy
// host path sees current data); only increment 2b replaces those with device
// kernels (the actual Modify-cost win). 2a therefore stays WITHIN-TOL identical
// to increment 1 -- its sole job is to prove the kokkosable + DualView
// lifecycle runs on a real GPU without the "Concurrent modification of host and
// device views" abort.
//
// DualView discipline (the abort fix): the per-atom arrays are DualViews that
// are READ-ONLY on the device (only the inc-1 reduction reads d_atom2body); the
// host base writes their host side through the aliased raw pointers, so every
// such write is reconciled with modify_host() before the next sync_device()
// (in grow_arrays / sync_peratom_to_device) and never collides with a
// device-modify flag. LSDEM's own bodyownLS stays a PLAIN host array (no device
// kernel indexes it -- the kernels use atom2body), sidestepping that landmine.
// Per-body data (body[]/bodyLS[]) stays host-resident in 2a; device residency
// (a plain Kokkos::View<Body*>) arrives with the 2b kernels.
template<class DeviceType>
class FixRigidSmallLSDEMKokkos : public FixRigidSmallLSDEM, public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixRigidSmallLSDEMKokkos(class LAMMPS *, int, char **);
  ~FixRigidSmallLSDEMKokkos() override;

  void setup(int) override;
  void setup_pre_neighbor() override;
  void pre_neighbor() override;
  void pre_force(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void grow_arrays(int) override;

  // public because it launches an extended __host__ __device__ lambda: nvcc
  // forbids the enclosing function of such a lambda from being protected/private
  // (it overrides the protected base method, which C++ permits widening).
  void compute_forces_and_torques() override;

 protected:
  class CommKokkos *commKK;

  // The five FixRigidSmall per-atom arrays, aliased as DualViews so the host
  // base writes the host side while the device reduction reads the device side.
  // READ-ONLY on the device.
  DAT::tdual_int_1d       k_bodyown;
  DAT::tdual_tagint_1d    k_bodytag;
  DAT::tdual_int_1d       k_atom2body;
  DAT::tdual_imageint_1d  k_xcmimage;
  DAT::tdual_double_2d_lr k_displace;

  typename AT::t_int_1d       d_bodyown;
  typename AT::t_tagint_1d    d_bodytag;
  typename AT::t_int_1d       d_atom2body;
  typename AT::t_imageint_1d  d_xcmimage;
  typename AT::t_double_2d_lr d_displace;

  // per-molecule force/torque accumulator [6*(maxmol+1)] on the device + its
  // host mirror (the source for the molecule-id MPI_Allreduce) -- from inc-1.
  Kokkos::View<double *, DeviceType> d_ft;
  typename Kokkos::View<double *, DeviceType>::HostMirror h_ft;

  // push the per-atom DualViews (written on the host) to the device. Run on the
  // reneighbor cadence (setup / setup_pre_neighbor / pre_neighbor); body
  // assignment of local atoms is stable between neighbour-list builds. Subsumes
  // inc-1's standalone upload_atom2body().
  void sync_peratom_to_device();
};

}    // namespace LAMMPS_NS

#endif
#endif
