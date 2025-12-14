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
FixStyle(rigid/small/ls/dem,FixRigidSmallLSDEM);
// clang-format on
#else

#ifndef LMP_FIX_RIGID_SMALL_LS_DEM_H
#define LMP_FIX_RIGID_SMALL_LS_DEM_H

#include "fix_rigid_small.h"

namespace LAMMPS_NS {

class FixRigidSmallLSDEM : public FixRigidSmall {
 public:
  FixRigidSmallLSDEM(class LAMMPS *, int, char **);
  ~FixRigidSmallLSDEM() override;
  int setmask() override;
  void post_constructor() override;
  void init() override;
  void setup(int) override;
  void setup_pre_force(int) override;
  void pre_force(int) override;

  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  void set_arrays(int) override;

  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  void setup_pre_neighbor() override;
  void pre_neighbor() override;

  double memory_usage() override;

 protected:
  int stored_flag, distributed_flag;
  int comm_flag2;
  char *id_fix, *id_fix2;
  int index_ls_dem_com;
  int index_ls_dem_quat;
  int index_ls_dem_omega;
  int index_ls_dem_n;
  int index_ls_dem_fs;
  int index_ls_dem_touch_id;
  int index_ls_dem_fn1;
  int index_ls_dem_fs1;
  int index_ls_dem_size;

  double **global_grids;
  double *grid_scale;
  double maxcut, warncut;
  int dim, rcell;
  int subgrid_size[3];     // number of distributed subgrid points in each dimension

  struct BodyLS {
    int ilocal;            // index of owning atom, duplicate from Body
    int style;             // distributed vs. global memory
    int grid_index;        // index of body's global memory, -1 if distributed
    double grid_stride;    // the LS grid stride, assumed equal in all direction
    double grid_vol;       // volume of LS grid
    double node_area;      // area associated with each grid node
    double grid_min[3];    // minimum xyz coordinates of LS grid
  };

  BodyLS *bodyLS;          // list of rigid bodies, owned and ghost
  int nlocal_bodyLS;       // # of owned rigid bodies
  int nghost_bodyLS;       // # of ghost rigid bodies
  int nmax_bodyLS;         // max # of bodies that body can hold
  int bodysizeLS;          // sizeof(BodyLS) in doubles

  // per-atom quantities
  // only defined for owned atoms, except bodyown for own+ghost

  int *bodyownLS;           // mirror of bodyown

  // local methods

  void compute_forces_and_torques() override;
  void grow_body_ls();

};

}    // namespace LAMMPS_NS

#endif
#endif
