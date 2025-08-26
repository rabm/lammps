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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(ls/dem,PairLSDEM);
// clang-format on
#else

#ifndef LMP_PAIR_LS_DEM_H
#define LMP_PAIR_LS_DEM_H

#include "pair.h"

#include "fix_rigid_ls_dem.h"

namespace LAMMPS_NS {

class PairLSDEM : public Pair {
 public:
  PairLSDEM(class LAMMPS *);
  ~PairLSDEM() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void settings(int, char **) override;
  void init_style() override;
  void setup() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;

  double maxcut;

 protected:
  double **kn, **kt, **mu, **cut, **gamma;

  int index_ls_grid;
  int index_ls_dem_com;
  int index_ls_dem_quat;
  int index_ls_dem_vol;

  class FixRigidLSDEM *fix_rigid;

  void allocate();
  double smearedHeavisideStep(double);
};

}    // namespace LAMMPS_NS

#endif
#endif
