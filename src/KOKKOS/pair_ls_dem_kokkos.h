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

  // ---- M4a(2): device uploads of the read-only LS data the K2 force kernel (M4b)
  //      will consume. In M4a these are uploaded ONCE and round-trip self-checked;
  //      the force pass still runs on host, so this milestone is behaviour-neutral.
  //      M4b moves the body/atom2body/binfo uploads onto the per-step path (bodyLS
  //      is forward-comm'd every step) and lets K2 read them on device.
  void upload_grids_to_device();    // all GLOBAL grids packed flat + per-grid offsets (setup-once)
  void upload_bodies_to_device();   // bodyLS SoA + atom2body + cached binfo (per-step in M4b)
  void upload_coeffs_to_device();   // per-type contact coefficients (init/coeff-time)
  void selfcheck_uploads();         // deep-copy-back assertion vs the host source arrays

  // packed GLOBAL grids: d_grid_values is the concatenation of global_grids[gi][0..n),
  // d_grid_offset[gi] its start, count(gi) = grid_size[gi] product.
  Kokkos::View<double*, DeviceType> d_grid_values, d_grid_min;
  Kokkos::View<int*, DeviceType>    d_grid_offset, d_grid_size;
  typename Kokkos::View<double*, DeviceType>::HostMirror h_grid_values, h_grid_min;
  typename Kokkos::View<int*, DeviceType>::HostMirror    h_grid_offset, h_grid_size;
  int num_grids = 0;

  // bodyLS SoA (indexed by body = atom2body[atom]); quatd2g flattened [nbody*4]
  Kokkos::View<int*, DeviceType>    d_body_style, d_body_grid_index;
  Kokkos::View<double*, DeviceType> d_body_grid_scale, d_body_grid_stride, d_body_quatd2g;
  typename Kokkos::View<int*, DeviceType>::HostMirror    h_body_style, h_body_grid_index;
  typename Kokkos::View<double*, DeviceType>::HostMirror h_body_grid_scale, h_body_grid_stride, h_body_quatd2g;
  int num_bodies = 0;

  // per-atom (local+ghost): atom2body + cached body info (binfo)
  Kokkos::View<int*, DeviceType>    d_atom2body, d_binfo_bID, d_binfo_bidx, d_binfo_grp, d_binfo_off;
  Kokkos::View<double*, DeviceType> d_binfo_vol, d_binfo_area;
  typename Kokkos::View<int*, DeviceType>::HostMirror    h_atom2body, h_binfo_bID, h_binfo_bidx, h_binfo_grp, h_binfo_off;
  typename Kokkos::View<double*, DeviceType>::HostMirror h_binfo_vol, h_binfo_area;
  int num_atoms_uploaded = 0;

  // per-type coeff tables, flattened row-major over [itype][jtype], stride (ntypes+1)
  Kokkos::View<double*, DeviceType> d_kn, d_kt, d_mu, d_knp, d_etan, d_etat,
                                    d_etan1, d_decayn1, d_etat1, d_decayt1;
  typename Kokkos::View<double*, DeviceType>::HostMirror h_kn, h_kt, h_mu, h_knp, h_etan, h_etat,
                                    h_etan1, h_decayn1, h_etat1, h_decayt1;
  int coeff_stride = 0;

  bool uploads_done = false;        // M4a: upload + self-check once (K2 not wired yet)
};

}    // namespace LAMMPS_NS

#endif
#endif
