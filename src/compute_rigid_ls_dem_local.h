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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(rigid/ls/dem/local,ComputeRigidLSDEMLocal);
// clang-format on
#else

#ifndef LMP_COMPUTE_RIGID_LS_DEM_LOCAL_H
#define LMP_COMPUTE_RIGID_LS_DEM_LOCAL_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeRigidLSDEMLocal : public Compute {
 public:
  ComputeRigidLSDEMLocal(class LAMMPS *, int, char **);
  ~ComputeRigidLSDEMLocal() override;
  void init() override;
  void compute_local() override;
  double memory_usage() override;

 private:
  int nvalues;
  int ncount;
  int *rstyle;

  char *idrigid;
  class FixRigidSmallLSDEM *fixrigid;

  int nmax;
  double *vlocal;
  double **alocal;

  int compute_rigid(int);
  void reallocate(int);
};

}    // namespace LAMMPS_NS

#endif
#endif
