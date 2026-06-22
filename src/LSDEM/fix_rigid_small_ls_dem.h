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

#include <map>

namespace LAMMPS_NS {

class FixRigidSmallLSDEM : public FixRigidSmall {
  friend class ComputeRigidLSDEMLocal;

 public:
  FixRigidSmallLSDEM(class LAMMPS *, int, char **);
  ~FixRigidSmallLSDEM() override;
  int setmask() override;
  void post_constructor() override;
  void init() override;
  void setup_pre_force(int) override;
  void initial_integrate(int) override;
  void pre_force(int) override;

  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  void set_arrays(int) override;
  void write_restart_file(const char *) override;
  void set_molecule(int, tagint, int, double *, double *, double *) override;

  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  void setup_pre_neighbor() override;
  void pre_neighbor() override;

  double memory_usage() override;

  struct BodyLS {
    // These are only needed by local bodies
    int ilocal;            // index of owning atom, duplicate from Body

    // These are needed by local + ghost bodies (the pair reads them for ghost
    // partner/representative bodies, so they must be communicated and must not
    // depend on a separate post-comm mutation step).
    int file_id;           // integer id of file
    int natoms;            // # of surface nodes in the body (global node count)
    double grid_vol;       // volume of LS grid
    double node_area;      // TOTAL surface area of the body; per-node area is
                           // node_area / natoms, computed at point of use
    int style;             // style of memory, GLOBAL or distributed
    int grid_index;        // index of body's global memory
    double grid_scale;     // scale factor for grid values, only needed for GLOBAL
    double grid_stride;    // the LS grid stride, assumed equal in all direction
    double quatd2g[4];     // quaternion that rotates from diagonal to grid frame for each rigid body
  };

  inline int get_maxmol() { return maxmol; };
  inline BodyLS* get_bodyLS_array() { return bodyLS; };
  inline int* get_atom2body_array() { return atom2body; };

  // read-only accessors used by the Kokkos pair (pair_ls_dem_kokkos) for the
  // device uploads of the level-set grid + body data (GPU port, Milestone 4a).
  inline int get_nbodyLS() { return nlocal_bodyLS + nghost_bodyLS; };
  inline int get_num_global_grids() { return num_global_grids; };
  inline double** get_global_grids_array() { return global_grids; };
  inline int** get_global_grids_size_array() { return global_grids_size; };
  inline double** get_global_grids_min_array() { return global_grids_min; };

  double get_ls_value(int, int, double*);

 protected:
  int ls_read_flag, global_flag, distributed_flag;
  int read_quat;
  int commflag_ls;
  char *id_fix, *id_fix2;
  int index_ls_dem_touch_id;

  double **quat_custom;

  double maxcut, warncut;
  int dim, rcell;
  int subgrid_size[3];     // number of distributed subgrid points in each dimension

  BodyLS *bodyLS;          // list of rigid bodies, owned and ghost
  int nlocal_bodyLS;       // # of owned rigid bodies
  int nghost_bodyLS;       // # of ghost rigid bodies
  int nmax_bodyLS;         // max # of bodies that body can hold
  int bodysizeLS;          // sizeof(BodyLS) in doubles

  // per-atom quantities
  // only defined for owned atoms, except bodyown for own+ghost

  int *bodyownLS;           // mirror of bodyown

  // pointers for per-atom distributed quantities

  int index_grid_values;
  int index_grid_min;

  // arrays for global quantities

  double **global_grids;
  int **global_grids_size;
  double **global_grids_min;
  int num_global_grids = 0;   // # of GLOBAL grids (= index_global_grid from process_levelsets)

  // LS grid file data

  struct LSData {
    int id;                     // integer ID for the grid
    int grid_index;             // index of grid in global storage
    int style;                  // memory style
    double stride;              // stride used in LS
    double grid_min[3];         // minimum LS grid point
    int grid_size[3];           // size of LS grid
    std::vector<tagint> bodies; // bodies using this data in the infile
    std::vector<double> scales; // scales for each body in the infile
  };

  std::map<std::string, LSData> gridfile_data;
  std::map<int, std::string> id_to_gridfile;
  std::map<std::string, int> gridfile_to_id;

  // local methods

  void process_levelsets();
  void compute_forces_and_torques() override;
  void compute_grain_properties(int, int*, double*, double*);
  void read_infile();
  void read_gridfile(int, std::string, double *);
  void grow_body_ls();
  void reset_atom2body_ghost();
};

}    // namespace LAMMPS_NS

#endif
#endif
