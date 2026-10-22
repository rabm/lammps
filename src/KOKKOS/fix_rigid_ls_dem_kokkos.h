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
FixStyle(rigid/ls/dem/kk,FixRigidLSDEMKokkos<LMPDeviceType>);
FixStyle(rigid/ls/dem/kk/device,FixRigidLSDEMKokkos<LMPDeviceType>);
FixStyle(rigid/ls/dem/kk/host,FixRigidLSDEMKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_RIGID_LS_DEM_KOKKOS_H
#define LMP_FIX_RIGID_LS_DEM_KOKKOS_H

#include "fix_rigid_ls_dem.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

// fix rigid/ls/dem with the per-atom work on the device
//
// The bodies are global and the same on all ranks, so their few numbers are integrated on
//   the host as in fix rigid/ls/dem and copied to the device at each step.  The per-atom
//   parts run on the device: the positions and velocities of the nodes, the copy of the
//   body position, orientation and angular velocity to the nodes, the sum of the node forces
//   and torques per body, and the virial.
// Everything at setup and reneighboring stays on the host, as in fix rigid/ls/dem.
// The sum per body runs over the nodes in atom index order in chunks.  With "pair_style ls/dem
//   order exact" it uses one chunk, as on the CPU, so the results do not depend on the device.

template <class DeviceType> class FixRigidLSDEMKokkos : public FixRigidLSDEM {
 public:
  typedef ArrayTypes<DeviceType> AT;

  FixRigidLSDEMKokkos(class LAMMPS *, int, char **);

  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void pre_neighbor() override;

  // with device lambdas, public for nvcc

  void upload_atoms();
  void upload_bodies();
  void set_xv_kokkos(int);
  void copy_bodies_to_nodes();
  void sum_forces_and_torques();

 protected:
  // set while the host owns the data during setup
  bool host_phase;

  // number of chunks the nodes of a body are summed in
  int nchunk;

  // per-atom data of fix rigid, copied from the host at each reneighbor
  DAT::tdual_int_1d k_body;
  DAT::tdual_double_2d_lr k_displace;
  DAT::tdual_imageint_1d k_xcmimage;

  // owned atoms of each body in atom index order
  DAT::tdual_int_1d k_bstart, k_batom;

  // xcm, vcm, omega, ex_space, ey_space, ez_space, quat and quatd2g of each body
  DAT::tdual_double_2d_lr k_bstate;

  // sum of the node forces and torques of each body, and its parts per chunk
  DAT::tdual_double_2d_lr k_bsum;
  typename AT::t_double_1d d_bpart;

  void set_xv() override;
  void set_v() override;
  void compute_forces_and_torques() override;
};

}    // namespace LAMMPS_NS

#endif
#endif
