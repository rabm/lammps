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

#include "fix_rigid_small_kokkos.h"
#include "fix_rigid_small_ls_dem.h"

namespace LAMMPS_NS {

// fix rigid/small/ls/dem on top of fix rigid/small/kk
//
// The per-step LS-DEM parts run on the device: the copy of the body position, orientation
//   and angular velocity to the nodes, the forward communication of these to the ghost nodes,
//   and the sum of the node forces and torques per body.  The sum runs over the nodes of a body
//   in atom index order, as on the CPU, so the result does not depend on thread scheduling.
// Everything at setup and reneighboring stays on the host, as in fix rigid/small/ls/dem:
//   the exchange and sort of atoms and the communication of the level-set data.

template <class DeviceType>
class FixRigidSmallLSDEMKokkos : public FixRigidSmallLSDEMT<FixRigidSmallKokkos<DeviceType>> {
 public:
  typedef FixRigidSmallKokkos<DeviceType> KKBase;
  typedef FixRigidSmallLSDEMT<KKBase> Parent;
  typedef ArrayTypes<DeviceType> AT;

  FixRigidSmallLSDEMKokkos(class LAMMPS *, int, char **);

  void init() override;
  void setup_pre_neighbor() override;
  void setup(int) override;
  void setup_pre_force(int) override;
  void initial_integrate(int) override;
  void pre_neighbor() override;
  void pre_force(int) override;
  void post_force(int) override;
  void final_integrate() override;
  void reset_atom2body() override;
  int *get_atom2body_array() override;

  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d &, int,
                               int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d &) override;

  // with device lambdas, public for nvcc

  void upload_ls();
  void copy_bodies_to_nodes();
  void reset_atom2body_ghost_kokkos();
  void compute_forces_and_torques_ls();

 protected:
  // set while the host owns the data during setup
  bool host_phase;

  // quatd2g of the owned bodies
  DAT::tdual_double_2d_lr k_quatd2g;

  // owned atoms of each owned and ghost body in atom index order
  typename AT::t_int_1d d_bstart, d_bcount, d_batom;

  void pre_force_host(int);
};

}    // namespace LAMMPS_NS

#endif
#endif
