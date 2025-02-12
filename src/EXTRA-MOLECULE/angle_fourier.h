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

#ifdef ANGLE_CLASS
// clang-format off
AngleStyle(fourier,AngleFourier);
// clang-format on
#else

#ifndef ANGLE_FOURIER_H
#define ANGLE_FOURIER_H

#include "angle.h"

namespace LAMMPS_NS {

class AngleFourier : public Angle {
 public:
  AngleFourier(class LAMMPS *);
  ~AngleFourier() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  double equilibrium_angle(int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  double single(int, int, int, int) override;
  void born_matrix(int type, int i1, int i2, int i3, double &du, double &du2) override;
  void *extract(const char *, int &) override;

 protected:
  double *k, *C0, *C1, *C2;

  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
