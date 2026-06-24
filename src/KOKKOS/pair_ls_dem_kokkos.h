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
  // Kernel 1 (device winner-reduction). At neighbour rebuild the host rep_segs CSR is
  // flattened + uploaded here as a 3-level CSR: per-rep segment ranges
  // (d_atom_seg_offset[ntotal+1]) -> per-segment candidate ranges (d_cand_offset) ->
  // candidate atom indices (d_cand). K1 runs ONE THREAD PER REP NODE and emits at most
  // ONE contact per rep (d_contacts_* indexed by rep atom, i==-1 = none) = the closest
  // candidate across ALL of that rep's segments. This is the one-contact-per-rep guard
  // (decision 2026-06-22): a rep touching >1 body keeps only its closest contact, so K2's
  // per-rep history RMW hits distinct slots and needs no atomics. Identical to the old
  // per-segment winner for single-segment reps; differs only for (rare) multi-body reps.
  void upload_segments_to_device();

  Kokkos::View<int*, DeviceType> d_atom_seg_offset, d_cand_offset, d_cand;
  typename Kokkos::View<int*, DeviceType>::HostMirror h_atom_seg_offset, h_cand_offset, h_cand;
  Kokkos::View<int*, DeviceType> d_contacts_i, d_contacts_j, d_contacts_calc;   // indexed by rep atom
  typename Kokkos::View<int*, DeviceType>::HostMirror h_contacts_i, h_contacts_j, h_contacts_calc;

  int nseg = 0, seg_cap = 0, cand_cap = 0, slot_cap = 0;

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
  int num_bodies = 0, body_cap = 0;

  // per-atom (local+ghost): atom2body + cached body info (binfo)
  Kokkos::View<int*, DeviceType>    d_atom2body, d_binfo_bID, d_binfo_bidx, d_binfo_grp, d_binfo_off;
  Kokkos::View<double*, DeviceType> d_binfo_vol, d_binfo_area;
  typename Kokkos::View<int*, DeviceType>::HostMirror    h_atom2body, h_binfo_bID, h_binfo_bidx, h_binfo_grp, h_binfo_off;
  typename Kokkos::View<double*, DeviceType>::HostMirror h_binfo_vol, h_binfo_area;
  int num_atoms_uploaded = 0, peratom_cap = 0;

  // per-type coeff tables, flattened row-major over [itype][jtype], stride (ntypes+1)
  Kokkos::View<double*, DeviceType> d_kn, d_kt, d_mu, d_knp, d_etan, d_etat,
                                    d_etan1, d_decayn1, d_etat1, d_decayt1;
  typename Kokkos::View<double*, DeviceType>::HostMirror h_kn, h_kt, h_mu, h_knp, h_etan, h_etat,
                                    h_etan1, h_decayn1, h_etat1, h_decayt1;
  int coeff_stride = 0;

  bool constants_uploaded = false;  // grids + coeffs uploaded once (constant for the run)
  bool selfcheck_done = false;      // upload deep-copy-back self-check run once (step 0)

  // M4b history host-sync BRIDGE (full device residency of the 5 shear-history arrays is M5/M6).
  // Before K2 the host history arrays are read into device buffers; after K2 they are written
  // back. n/fs/touch_id are host-only (no DualView); fn1/fs1 are k_dvector DualViews but reading
  // the host pointer here is correct (the M3/M4a bitwise gate read them the same way). No
  // reverse_comm: a node's history is authoritative on its owner rank (calc==1); ghost writes
  // (calc==0) are discarded and overwritten by the fix's `ghost yes` forward-comm next step.
  void sync_history_to_device();
  void sync_history_from_device();

  // K2 launcher templated on EVFLAG: EVFLAG==1 -> parallel_reduce accumulating the global virial
  // into `ev` (device virial); EVFLAG==0 -> parallel_for (no energy/virial). Defined in the .cpp.
  template<int EVFLAG> void launch_K2(int ntotal, int nlocal, EV_FLOAT &ev);

  Kokkos::View<double*, DeviceType> d_hist_n, d_hist_fs;     // flat ntotal*3
  Kokkos::View<int*, DeviceType>    d_hist_touch;            // ntotal
  Kokkos::View<double*, DeviceType> d_hist_fn1, d_hist_fs1;  // ntotal
  typename Kokkos::View<double*, DeviceType>::HostMirror h_hist_n, h_hist_fs, h_hist_fn1, h_hist_fs1;
  typename Kokkos::View<int*, DeviceType>::HostMirror    h_hist_touch;
  int hist_cap = 0;
  int hist_lastbuild = -1;   // reneighbor stamp: host->device history sync is reneighbor-cadence (M5)

  // DISTRIBUTED per-atom subgrid: uploaded per reneighbor from the host property/atom darray
  // (its ghost rows are kept current by the CPU border comm -> single-rank correct). A fully
  // device-resident store (device exchange/border) is a later increment.
  void upload_distributed_to_device();
  Kokkos::View<double*, DeviceType> d_subgrid, d_dgrid_min;
  typename Kokkos::View<double*, DeviceType>::HostMirror h_subgrid, h_dgrid_min;
  int dist_N = 0, subgrid_dim[3] = {0,0,0};
  int idx_grid_values = -1, idx_grid_min = -1;
};

}    // namespace LAMMPS_NS

#endif
#endif
