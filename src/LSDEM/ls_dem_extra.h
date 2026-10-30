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

#include <unordered_map>

namespace LSDEMExtra {

  // a grid cell has points up to its diagonal (sqrt(dimension) strides) from its lower corner and a
  //   level set changes by at most the distance, so the watershed keeps cells with a lower corner
  //   up to sqrt(dimension) strides outside the surface
  inline double ws_outer(int dimension)
  {
    return (dimension == 2) ? 1.4142135623730951 : 1.7320508075688772;
  }

  double smeared_heaviside_step(double);
  double compute_volume(int, int *, double, double *, double);
  double compute_surface_area(int, int *, double, double *);
  double compute_grid_properties(int *, double, double *, double *, double *, int);
  double interpolate_LS_array(int, int, double *, int *, double *, int *, double[3], double);
  double interpolate_LS_watershed(int, int, std::unordered_map<int, double> *, std::unordered_map<int, double> *, int[3], double[3], int *, double[3], double);
  double get_ws_ls_value(int, std::unordered_map<int, double> *, std::unordered_map<int, double> *);
  int store_dist_array(int, int, int *, int *, double, double, double, double *, double *, double *, double *, double *, double *);
}

#endif
