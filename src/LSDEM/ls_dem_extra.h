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

  // level-set grid of the grain of a node, as read by get_ls_value()
  //   watershed storage: the grid of the grain and the table and buffer of the node, no values

  struct GridInfo {
    int global;               // 1 for a grid shared by grains, its values are scaled by scale
    int ngrid[3];             // number of grid points in each dimension
    double grid_min[3];       // lower corner of the grid in the grain frame
    double stride;            // grid spacing
    double scale;             // scale factor for the values of a shared grid
    const double *values;     // grid values
    int nvalues;              // number of grid values
    const std::unordered_map<int, double> *ws_table;     // watershed bins owned by the node
    const std::unordered_map<int, double> *ws_buffer;    // watershed bins next to them
  };

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
