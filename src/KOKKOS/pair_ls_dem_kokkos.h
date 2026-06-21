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

 protected:
  // M3 Kernel 1 (device winner-reduction). At neighbour rebuild the host rep_segs CSR is
  // flattened + uploaded here; the per-step reduction runs on device and emits one contact
  // per segment (seg-indexed, i==-1 = no contact), which the host collects in segment order
  // (== the CPU emit order, so the result stays bitwise). The force pass stays on host (M4).
  void upload_segments_to_device();

  Kokkos::View<int*, DeviceType> d_seg_rep, d_cand_offset, d_cand;
  typename Kokkos::View<int*, DeviceType>::HostMirror h_seg_rep, h_cand_offset, h_cand;
  Kokkos::View<int*, DeviceType> d_contacts_i, d_contacts_j, d_contacts_calc;
  typename Kokkos::View<int*, DeviceType>::HostMirror h_contacts_i, h_contacts_j, h_contacts_calc;

  int nseg = 0, seg_cap = 0, cand_cap = 0;
};

}    // namespace LAMMPS_NS

#endif
#endif
