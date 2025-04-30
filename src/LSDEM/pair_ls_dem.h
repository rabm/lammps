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

namespace LAMMPS_NS {

class PairLSDEM : public Pair {
 public:
  PairLSDEM(class LAMMPS *);
  ~PairLSDEM() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  void setup() override;

 protected:
  double **k, **cut, **gamma;

  int index_ls_dem_grid;
  int index_ls_dem_gridx;
  int index_ls_dem_gridy;
  int index_ls_dem_gridz;
  int index_ls_dem_com;
  int index_ls_dem_quat;
  int index_ls_dem_vol;
  int ngrid, nrow, ncol, nslice;
  double grid_min[3];
  double spac;

  void allocate();
  double get_ls_value(int, int, double *);
  double smearedHeavisideStep(double);
};

}    // namespace LAMMPS_NS

#endif
#endif
