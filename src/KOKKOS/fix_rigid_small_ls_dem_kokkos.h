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
#include "kokkos_type.h"

namespace LAMMPS_NS {

// GPU/Kokkos variant of fix rigid/small/ls/dem (M7 milestone, INCREMENT 1).
//
// Increment 1 ports ONLY the per-body force/torque reduction
// (compute_forces_and_torques) to the device: it replaces the O(natoms) host
// summation loop with a device kernel that atomic_adds each node's f/torque
// into a per-molecule device buffer, then keeps the EXISTING host-staged
// molecule-id MPI_Allreduce (which guarantees a body's total reaches its owner
// across ranks -- a device reverse_comm would drop boundary-straddling-grain
// force, the bug the Allreduce fixed). The LS-DEM atoms already carry the exact
// per-node contact torque from the pair, so there is NO (x-xcm)x f lever-arm
// term (porting the upstream FixRigidSmallKokkos lever-arm block would
// double-count and be physically wrong).
//
// This increment is deliberately NON-kokkosable: the Kokkos run loop still
// syncs atom data to the host around the (inherited, host) initial_integrate /
// final_integrate / set_xv / set_v / grain scatter, so those remain correct and
// unchanged. Only the reduction LOOP moves to the device. Removing the host
// f/torque sync (keeping f device-resident) + the set_xv/set_v + grain scatter
// kernels are increment 2 (device residency).
//
// The atomic_add reduction reorders a body's per-node sum, so /kk results are
// WITHIN-TOL (not bitwise) -- consistent with the K2 force kernel. CPU verify.sh
// is unaffected (it never runs -sf kk).
template<class DeviceType>
class FixRigidSmallLSDEMKokkos : public FixRigidSmallLSDEM {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixRigidSmallLSDEMKokkos(class LAMMPS *, int, char **);

  // public because it launches an extended __host__ __device__ lambda: nvcc
  // forbids the enclosing function of such a lambda from being protected/private
  // (it overrides the protected base method, which C++ permits widening).
  void compute_forces_and_torques() override;

 protected:
  // device atom2body, refreshed on the reneighbor cadence (body assignment of
  // local atoms is stable between neighbour-list builds)
  typename AT::t_int_1d d_atom2body;
  typename AT::t_int_1d::HostMirror h_atom2body;
  bigint a2b_lastbuild;

  // per-molecule force/torque accumulator [6*(maxmol+1)] on the device + its
  // host mirror (the source for the molecule-id MPI_Allreduce)
  Kokkos::View<double *, DeviceType> d_ft;
  typename Kokkos::View<double *, DeviceType>::HostMirror h_ft;

  void upload_atom2body();
};

}    // namespace LAMMPS_NS

#endif
#endif
