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
#include "ls_dem_extra.h"

#include <map>
#include <unordered_map>

namespace LAMMPS_NS {

// what pair ls/dem and fix rigid/ls/dem use of fix rigid/small/ls/dem and its accelerated variants

class FixRigidSmallLSDEMAccess {
 public:
  struct BodyLS {
    // These are only needed by local bodies
    int ilocal;            // index of owning atom, duplicate from Body
    int file_id;           // integer id of file
    double grid_vol;       // volume of LS grid
    double node_area;      // area associated with each grid node

    // These are needed by local + ghost bodies
    int style;             // style of memory, GLOBAL or DISTRIBUTED
    int grid_index;        // index of body's global memory
    int grid_size[3];      // size of total grain grid
    double grid_min[3];    // minimum values of total grain grid
    double grid_scale;     // scale factor for grid values, only needed for GLOBAL
    double grid_stride;    // the LS grid stride, assumed equal in all direction
    double quatd2g[4];     // quaternion that rotates from diagonal to grid frame for each rigid body
    tagint tagmin;              // smallest tag of atom in the body
  };

  virtual ~FixRigidSmallLSDEMAccess() noexcept(false) {}
  virtual int get_maxmol() = 0;
  virtual BodyLS *get_bodyLS_array() = 0;
  virtual int *get_atom2body_array() = 0;
  virtual int get_storage_model() = 0;
  virtual void get_grid_info(int, LSDEMExtra::GridInfo &) = 0;
};

// the LS-DEM part of fix rigid/small/ls/dem, on top of fix rigid/small (Base) or one of
//   its accelerated variants

template <class Base> class FixRigidSmallLSDEMT : public Base, public FixRigidSmallLSDEMAccess {
  friend class ComputeRigidLSDEMLocal;

 protected:
  typedef typename Base::Body Body;
  using Base::atom;
  using Base::comm;
  using Base::displace;
  using Base::domain;
  using Base::eflags;
  using Base::error;
  using Base::force;
  using Base::lmp;
  using Base::memory;
  using Base::modify;
  using Base::update;
  using Base::world;
  using Base::atom2body;
  using Base::body;
  using Base::bodyown;
  using Base::bodytag;
  using Base::commflag;
  using Base::extended;
  using Base::grav_group_bit;
  using Base::grow_body;
  using Base::gvec;
  using Base::id_gravity;
  using Base::inpfile;
  using Base::itensor_custom;
  using Base::langflag;
  using Base::mass_custom;
  using Base::maxmol;
  using Base::n_extra_attributes;
  using Base::nbody;
  using Base::nghost_body;
  using Base::nlocal_body;
  using Base::nmax_body;
  using Base::onemols;
  using Base::reinitflag;
  using Base::reset_atom2body;
  using Base::setupflag;
  using Base::xcm_custom;
  using Base::xcmimage;

 public:
  using Base::comm_border;
  using Base::comm_forward;
  using Base::groupbit;
  using Base::id;
  using Base::maxexchange;
  using Base::restart_file;
  using Base::style;

  FixRigidSmallLSDEMT(class LAMMPS *, int, char **);
  ~FixRigidSmallLSDEMT() override;
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

  int pack_border(int, int *, double *) override;
  int unpack_border(int, int, double *) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  void setup_pre_neighbor() override;
  void pre_neighbor() override;

  double memory_usage() override;


  int get_maxmol() override { return maxmol; }
  BodyLS *get_bodyLS_array() override { return bodyLS; }
  int *get_atom2body_array() override { return atom2body; }
  int get_storage_model() override { return storage_mode; }

  double get_ls_value(int, int, int, double*, double*);
  int get_bin(int, int, double*);
  int check_watershed_bin(int, int);
  void get_grid_info(int, LSDEMExtra::GridInfo &) override;

 protected:
  enum { PARENT, FULL_BODY_LS, INITIAL_LS, PROCESS_GS, PREFORCE_LS, SUBGRID_LS, WATERSHED_LS };

  int ls_read_flag, global_flag, distributed_flag, storage_mode;
  int read_quat;
  int commflag_ls;
  int commflag_rev_ls;
  int nmax_distributed;
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

  int *bodyownLS;           // mirror of bodyown
  int *node_index;          // node index in grain used in watershedding
  int *atom_grid_style;     // copy of LSData style for communication (hard to access BodyLS)

  // variables for global quantities

  int max_grid_size_flat;
  int n_global_grids;
  int index_grid_values;
  int index_grid_min;

  double **global_grids;

  // storage for global watershed

  std::vector<std::vector <std::unordered_map<int, double>>> global_ws_tables;
  std::vector<std::vector <std::unordered_map<int, double>>> global_ws_buffers;

  // storage for distributed watershed

  std::vector<std::unordered_map<int, double>> dist_ws_tables;
  std::vector<std::unordered_map<int, double>> dist_ws_buffers;

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
  int pack_watershed(int, int *, double *);
  int unpack_watershed(int, int, double *);
  void calculate_xcom();
  void copy_mol_data_to_bodies();
  void copy_in_data_to_bodies();
  void compute_forces_and_torques() override;
  void compute_grain_properties(int, int*, double*, double*);
  void read_infile();
  void read_gridfile(int, std::string, double *);
  void grow_body_ls();
  void reset_atom2body_ghost();
  double get_ls_value_array(int, int, double*);
  double get_ls_value_watershed(int, int, int, double*, double*);
};

class FixRigidSmallLSDEM final : public FixRigidSmallLSDEMT<FixRigidSmall> {
 public:
  FixRigidSmallLSDEM(class LAMMPS *lmp, int narg, char **arg) :
      FixRigidSmallLSDEMT<FixRigidSmall>(lmp, narg, arg)
  {
  }
};

}    // namespace LAMMPS_NS

#endif
#endif
