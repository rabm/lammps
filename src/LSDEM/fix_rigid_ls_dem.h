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
FixStyle(rigid/ls/dem,FixRigidLSDEM);
// clang-format on
#else

#ifndef LMP_FIX_RIGID_LS_DEM_H
#define LMP_FIX_RIGID_LS_DEM_H

#include "fix_rigid.h"

namespace LAMMPS_NS {

class FixRigidLSDEM : public FixRigid {
 public:
  FixRigidLSDEM(class LAMMPS *, int, char **);
  ~FixRigidLSDEM() override;
  int setmask() override;
  void post_constructor() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  void init() override;
  void setup(int) override;
  void setup_pre_force(int) override;
  void initial_integrate(int) override;
  void pre_force(int) override;
  void write_restart_file(const char *) override;

  double memory_usage() override;

  inline double *get_vol_array() { return grid_vol; };
  inline double *get_area_array() { return node_area; };
  inline int *get_body_array() { return body; };
  inline int get_nbody() { return nbody; };

  double get_ls_value(int, int, double*);

 protected:
  int stored_flag, distributed_flag;
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

  int index_grid_values;
  int index_grid_min;

  int **grid_size;              // size of each grid
  int subgrid_size[3];          // number of distributed subgrid points in each dimension
  int *grid_style;              // distributed vs. global memory
  int *grid_index;              // index of body's global memory, -1 otherwise
  double **grid_min;            // minimum xyz coordinates of LS grid
  double *grid_stride;          // the LS grid stride, assumed equal in all directions
  double *grid_vol;
  double *node_area;
  int *grid_nnodes;             // the number of nodes in the grid

  double **global_grids;
  double *grid_scale;
  double maxcut, warncut;
  int dim, rcell;

  void compute_forces_and_torques() override;
  void read_gridfile_names(char **);
  void read_gridfile(int, int, std::string, int **, double *);
  double compute_grid_properties(int *, double, double *, double *, double[3][3], std::string);
  inline double smeared_heaviside_step(double);
  inline double compute_volume(int *, double, double *, double);
  double compute_surface_area(int *, double, double *);
};

}    // namespace LAMMPS_NS

#endif
#endif
