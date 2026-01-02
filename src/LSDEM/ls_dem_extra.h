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

#ifndef LMP_LS_DEM_EXTRA_H
#define LMP_LS_DEM_EXTRA_H

namespace LSDEMExtra {

  double smeared_heaviside_step(double);
  double compute_volume(int, int *, double, double *, double);
  double compute_surface_area(int, int *, double, double *);
  double compute_grid_properties(int *, double, double *, double *, double[3][3], int);
  double interpolate_LS(int, double *, int, int, int, double, double, double, double[3], double);
}

#endif