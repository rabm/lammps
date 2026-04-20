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
  void init() override;
  void initial_integrate(int) override;
  void set_arrays(int) override;
  void write_restart_file(const char *) override;

  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_border(int, int *, double *) override;
  int unpack_border(int, int, double *) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

  double memory_usage() override;

  inline double *get_vol_array() { return grid_vol; };
  inline double *get_area_array() { return node_area; };
  inline int *get_body_array() { return body; };
  inline int get_nbody() { return nbody; };

  double get_ls_value(int, int, double*);

 protected:
  int stored_flag, global_flag, distributed_flag, watershed_flag;
  char *id_fix;
  int index_ls_dem_touch_id;

  int n_dist_grid;
  double **dist_grid_values;
  double **dist_grid_min;
  double min_stride;

  int **grid_size;              // size of each grid
  int subgrid_size[3];          // number of distributed subgrid points in each dimension
  int *grid_style;              // distributed vs. global memory
  int *grid_index;              // index of body's global memory, -1 distributed
  double **grid_min;            // minimum xyz coordinates of LS grid
  double *grid_stride;          // the LS grid stride, assumed equal in all directions
  double *grid_vol;
  double *node_area;

  char **gridfiles;
  double **quat_custom;         // temporary storage of infile quat
  double **quatd2g;             // quaternion that rotates from diagonal to grid frame for each rigid body

  double **global_grids;
  double *grid_scale;
  double maxcut;
  int rcell;

  void compute_forces_and_torques() override;
  void compute_grain_properties(int, double*, std::string);
  void read_gridfile(int, int, std::string, int **, double *);
  int read_infile(char **);
};

}    // namespace LAMMPS_NS

#endif
#endif
