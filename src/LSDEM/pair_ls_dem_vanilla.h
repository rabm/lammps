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
PairStyle(ls/dem/vanilla,PairLSDEMVanilla);
// clang-format on
#else

#ifndef LMP_PAIR_LS_DEM_VANILLA_H
#define LMP_PAIR_LS_DEM_VANILLA_H

#include "pair.h"
#include "pair_ls_dem.h"

#include "fix_rigid_ls_dem.h"
#include "fix_rigid_small_ls_dem.h"

namespace LAMMPS_NS {

class PairLSDEMVanilla : public PairLSDEM {
 public:
  PairLSDEMVanilla(class LAMMPS *);
  ~PairLSDEMVanilla() override;
  void compute(int, int) override;

 protected:

  int *mybody_large, *mybody_small;
  double *node_area;
  FixRigidSmallLSDEM::BodyLS *bodyLS;


  void process_contact(int, int, int, double *);
};

}    // namespace LAMMPS_NS

#endif
#endif
