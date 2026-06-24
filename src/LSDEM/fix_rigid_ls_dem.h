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

#include <unordered_map>

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
  inline int get_storage_model() { return storage_flag; };

  // read-only accessors used by the Kokkos pair for the device GLOBAL get_ls_value (large fix).
  // grid metadata is PER-BODY here (unlike the small fix's per-grid d_grid_min); global_grids is
  // indexed by grid_index[body]; num_global_grids = # of distinct GLOBAL grids.
  inline int get_num_global_grids() { return num_global_grids; };
  inline double** get_global_grids_array() { return global_grids; };
  inline int* get_grid_index_array() { return grid_index; };
  inline int** get_grid_size_array() { return grid_size; };
  inline double** get_grid_min_array() { return grid_min; };
  inline double* get_grid_scale_array() { return grid_scale; };
  inline double* get_grid_stride_array() { return grid_stride; };
  inline int* get_grid_style_array() { return grid_style; };
  inline int get_distributed_flag() { return distributed_flag; };

  double get_ls_value(int, int, int, double*, double*);
  int get_bin(int, int, double*);
  int check_watershed_bin(int, int);

 protected:
  int ls_read_flag, global_flag, distributed_flag, storage_flag;
  char *id_fix;
  int index_ls_dem_touch_id;

  int n_dist_grid = 0;          // sized in init() before grow_arrays(); 0 until then
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
  int num_global_grids = 0;     // # of distinct GLOBAL grids (= index_global_grid from setup)
  double *grid_scale;
  double maxcut;
  int rcell;

  std::vector<std::vector <std::unordered_map<int, double>>> global_ws_tables;
  std::vector<std::vector <std::unordered_map<int, double>>> global_ws_buffers;
  std::vector<std::unordered_map<int, double>> dist_ws_tables;
  std::vector<std::unordered_map<int, double>> dist_ws_buffers;
  int *node_type;
  int nmax_node_type;

  void compute_forces_and_torques() override;
  void compute_grain_properties(int, double*, std::string);
  void read_gridfile(int, int, std::string, int **, double *);
  int read_infile(char **);
  double get_ls_value_array(int, int, double*);
  double get_ls_value_watershed(int, int, int, double*, double*);
};

}    // namespace LAMMPS_NS

#endif
#endif
