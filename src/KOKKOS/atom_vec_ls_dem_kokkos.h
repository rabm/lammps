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

/* ----------------------------------------------------------------------
   LS-DEM Kokkos atom style (GPU port, Milestone 1).
   Mirrors atom_vec_sphere_kokkos: AtomVecKokkos device plumbing + the CPU
   AtomVecLSDEM field definitions. New, additive file (not a core edit).
------------------------------------------------------------------------- */

#ifdef ATOM_CLASS
// clang-format off
AtomStyle(ls/dem/kk,AtomVecLSDEMKokkos);
AtomStyle(ls/dem/kk/device,AtomVecLSDEMKokkos);
AtomStyle(ls/dem/kk/host,AtomVecLSDEMKokkos);
// clang-format on
#else

// clang-format off
#ifndef LMP_ATOM_VEC_LS_DEM_KOKKOS_H
#define LMP_ATOM_VEC_LS_DEM_KOKKOS_H

#include "atom_vec_kokkos.h"
#include "atom_vec_ls_dem.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

class AtomVecLSDEMKokkos : public AtomVecKokkos, public AtomVecLSDEM {
 public:
  AtomVecLSDEMKokkos(class LAMMPS *);
  void init() override;

  void grow(int) override;
  void grow_pointers() override;
  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter) override;
  void sync(ExecutionSpace space, uint64_t mask) override;
  void modified(ExecutionSpace space, uint64_t mask) override;
  void sync_pinned(ExecutionSpace space, uint64_t mask, int async_flag = 0) override;

 private:
  // Device/host mirrors for the LS-DEM custom fields that the AtomVecKokkos base
  // does not already provide. molecule/omega/torque reuse the base d_*/h_* members.
  DAT::t_kkfloat_1d_3 d_xcom;
  HAT::t_kkfloat_1d_3 h_xcom;
  DAT::t_kkfloat_1d_4 d_quat;
  HAT::t_kkfloat_1d_4 h_quat;
  DAT::t_int_1d d_grid_index;
  HAT::t_int_1d h_grid_index;
};

}    // namespace LAMMPS_NS

#endif
#endif
