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
   LS-DEM Kokkos pair style (GPU port, Milestone 2: host bridge).
   Derives from PairLSDEM. For now compute() runs the proven CPU
   PairLSDEM::compute on host-synced atom data (no device kernel yet);
   the two device kernels replace it in M3/M4.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(ls/dem/kk,PairLSDEMKokkos<LMPDeviceType>);
PairStyle(ls/dem/kk/device,PairLSDEMKokkos<LMPDeviceType>);
PairStyle(ls/dem/kk/host,PairLSDEMKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_LS_DEM_KOKKOS_H
#define LMP_PAIR_LS_DEM_KOKKOS_H

#include "pair_ls_dem.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairLSDEMKokkos : public PairLSDEM {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  PairLSDEMKokkos(class LAMMPS *);
  void compute(int, int) override;
  void init_style() override;
};

}    // namespace LAMMPS_NS

#endif
#endif
