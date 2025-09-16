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

class FixRigidLSDEM : public FixRigid { // TODO: delete all functions that this class will not redefine
 public:
  FixRigidLSDEM(class LAMMPS *, int, char **);
  ~FixRigidLSDEM() override;
  int setmask() override;
  void post_constructor() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  void init() override;
  void setup_pre_force(int) override;
  void initial_integrate(int) override;
  void pre_force(int) override;
  void write_restart_file(const char *) override;

  double memory_usage() override;

  inline int *get_body_array() { return body; };
  inline int get_nbody() { return nbody; };

  double get_ls_value(int, int, double*);

 protected:
  int stored_flag, distributed_flag;
  char *id_fix, *id_fix2;
  int index_ls_dem_vol;
  int index_ls_dem_com;
  int index_ls_dem_quat;
  int index_ls_dem_size;

  int index_ls_values;
  int index_ls_local_gridmin;

  int ngrid_local[3];    // number of local grid points in each dimension
  int *grid_style;       // distributed vs. global memory
  double **grid_min;     // minimum xyz coordinates of LS grid
  double *grid_stride;   // the LS grid stride, assumed equal in all directions
  double *grid_scale;
  double maxcut;
  int dim, rcell;

  void read_gridfile_names(char **);
  void read_gridfile(int, int, std::string, int **, double *);
  double process_ls_grid(int *, double, double *, double *, std::string);
  inline double smeared_heaviside_step(double);
};

}    // namespace LAMMPS_NS

#endif
#endif
