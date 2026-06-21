// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "pair_ls_dem_kokkos.h"

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "kokkos.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   K1 device functor: one thread per flattened CSR segment. Reproduces the CPU
   per-step winner reduction + emit (pair_ls_dem.cpp:281-296) exactly: first-min
   rsq argmin over the segment's candidates within maxcutsq (sweep-order tie-break),
   then the Newton-OFF rep/partner-local emit rule. Emits seg-indexed (i==-1 = none)
   so the host can collect contacts in segment order == CPU emit order (=> bitwise).
   Standalone struct (only Views + scalars) so no functor-copy/copymode concerns.
------------------------------------------------------------------------- */

namespace LAMMPS_NS {

template<class DeviceType>
struct PairLSDEMK1 {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  Kokkos::View<int*, DeviceType> d_seg_rep, d_cand_offset, d_cand;
  Kokkos::View<int*, DeviceType> d_ci, d_cj, d_ccalc;
  int nlocal;
  double maxcutsq;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int seg) const
  {
    const int r = d_seg_rep(seg);
    const bool r_local = (r < nlocal);
    const double xr0 = x(r,0), xr1 = x(r,1), xr2 = x(r,2);

    int widx = -1;
    double minrsq = 0.0;
    const int cs = d_cand_offset(seg);
    const int ce = d_cand_offset(seg + 1);
    for (int cc = cs; cc < ce; cc++) {
      const int jc = d_cand(cc);
      const double dx = xr0 - x(jc,0);
      const double dy = xr1 - x(jc,1);
      const double dz = xr2 - x(jc,2);
      const double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq > maxcutsq) continue;
      if (widx < 0 || rsq < minrsq) { minrsq = rsq; widx = jc; }   // first-min, sweep order
    }

    // Newton-OFF emit rule, verbatim from pair_ls_dem.cpp:294-296:
    //   widx<0 -> none; r local -> {r,widx,1}; else widx<nlocal -> {widx,r,0}; else both-ghost -> none.
    int ei = -1, ej = -1, ecalc = 0;
    if (widx >= 0) {
      if (r_local)            { ei = r;    ej = widx; ecalc = 1; }
      else if (widx < nlocal) { ei = widx; ej = r;    ecalc = 0; }
    }
    d_ci(seg) = ei;
    d_cj(seg) = ej;
    d_ccalc(seg) = ecalc;
  }
};

}    // namespace LAMMPS_NS

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLSDEMKokkos<DeviceType>::PairLSDEMKokkos(LAMMPS *lmp) : PairLSDEM(lmp)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  // fields the CPU contact model reads / writes (XCOM/QUAT feed get_ls_value via the rigid fix)
  datamask_read = X_MASK | V_MASK | XCOM_MASK | QUAT_MASK | OMEGA_MASK | TYPE_MASK | MASK_MASK |
                  F_MASK | TORQUE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | TORQUE_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::init_style()
{
  PairLSDEM::init_style();

  // M3: the per-step winner reduction runs on device (Kernel 1); the contact force pass
  // (process_contact) still runs on host, consuming the device-built contact list. So we keep the
  // legacy half + REQ_GHOST neighbor list PairLSDEM::init_style requested (build_rep_segments walks
  // it on host). A device neighbor list arrives with the force kernel (M4).
  if (lmp->kokkos->neighflag == FULL)
    error->all(FLERR, "Cannot use a full neighbor list with pair style ls/dem/kk");

  if (comm->me == 0)
    utils::logmesg(lmp, "PairLSDEMKokkos (M3): device winner-reduction kernel + host force pass\n");
}

/* ----------------------------------------------------------------------
   flatten the host rep_segs CSR to flat device arrays + upload (neighbour rebuild only)
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::upload_segments_to_device()
{
  const int ntotal = atom->nlocal + atom->nghost;

  int ns = 0, nc = 0;
  for (int r = 0; r < ntotal; r++)
    for (const auto &s : rep_segs[r]) { ns++; nc += (int) s.cand.size(); }
  nseg = ns;

  // (re)allocate device views + host mirrors when grown (margin avoids per-rebuild realloc churn).
  // d_cand_offset needs nseg+1 entries; size the seg-length arrays to seg_cap >= nseg+1.
  if (ns + 1 > seg_cap) {
    seg_cap = (ns + 1) + (ns + 1) / 4 + 16;
    Kokkos::realloc(d_seg_rep, seg_cap);
    Kokkos::realloc(d_cand_offset, seg_cap);
    Kokkos::realloc(d_contacts_i, seg_cap);
    Kokkos::realloc(d_contacts_j, seg_cap);
    Kokkos::realloc(d_contacts_calc, seg_cap);
    h_seg_rep       = Kokkos::create_mirror_view(d_seg_rep);
    h_cand_offset   = Kokkos::create_mirror_view(d_cand_offset);
    h_contacts_i    = Kokkos::create_mirror_view(d_contacts_i);
    h_contacts_j    = Kokkos::create_mirror_view(d_contacts_j);
    h_contacts_calc = Kokkos::create_mirror_view(d_contacts_calc);
  }
  if (nc > cand_cap) {
    cand_cap = nc + nc / 4 + 16;
    Kokkos::realloc(d_cand, cand_cap);
    h_cand = Kokkos::create_mirror_view(d_cand);
  }

  // fill host mirrors in (rep, then segment) order == the CPU emit order (pair_ls_dem.cpp:278)
  int seg = 0, c = 0;
  h_cand_offset(0) = 0;
  for (int r = 0; r < ntotal; r++)
    for (const auto &s : rep_segs[r]) {
      h_seg_rep(seg) = r;
      for (int jc : s.cand) h_cand(c++) = jc;        // sweep order preserved (pair_ls_dem.cpp:177)
      h_cand_offset(seg + 1) = c;
      seg++;
    }

  Kokkos::deep_copy(d_seg_rep, h_seg_rep);
  Kokkos::deep_copy(d_cand_offset, h_cand_offset);
  Kokkos::deep_copy(d_cand, h_cand);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::compute(int eflag, int vflag)
{
  // Watershed path (not functional / never hit by the GLOBAL test cases): defer to the exact CPU
  // compute, which has the watershed force loop. watershed_flag is set in PairLSDEM::setup().
  if (watershed_flag != 0) {
    atomKK->sync(Host, datamask_read);
    PairLSDEM::compute(eflag, vflag);
    atomKK->modified(Host, datamask_modify);
    return;
  }

  if (eflag || vflag) ev_setup(eflag, vflag);
  else evflag = vflag_fdotr = 0;

  const int ntotal = atom->nlocal + atom->nghost;
  const int nlocal = atom->nlocal;
  const double maxcutsq = maxcut * maxcut;            // cf. pair_ls_dem.cpp:231

  // K1 reads x on device; the host process_contact force pass reads everything on host.
  atomKK->sync(execution_space, X_MASK);
  atomKK->sync(Host, datamask_read);

  // rebuild branch (host), gated exactly like pair_ls_dem.cpp:253-263
  if (neighbor->lastcall != binfo_lastbuild || (int) binfo.size() < ntotal) {
    cache_body_info(ntotal);                          // pair_ls_dem.cpp:96
    binfo_lastbuild = neighbor->lastcall;
  }
  if (neighbor->lastcall != segs_lastbuild || (int) rep_segs.size() < ntotal) {
    build_rep_segments();                             // pair_ls_dem.cpp:141 (host neigh-list walk)
    upload_segments_to_device();                      // flatten + deep_copy
    segs_lastbuild = neighbor->lastcall;
  }

  // K1: device winner reduction + per-segment emit (replaces pair_ls_dem.cpp:278-298)
  contacts.clear();
  if (nseg > 0) {
    PairLSDEMK1<DeviceType> f;
    f.x = atomKK->k_x.view<DeviceType>();
    f.d_seg_rep = d_seg_rep;  f.d_cand_offset = d_cand_offset;  f.d_cand = d_cand;
    f.d_ci = d_contacts_i;    f.d_cj = d_contacts_j;            f.d_ccalc = d_contacts_calc;
    f.nlocal = nlocal;        f.maxcutsq = maxcutsq;

    Kokkos::parallel_for("PairLSDEM::K1",
        Kokkos::RangePolicy<DeviceType>(0, nseg), f);
    Kokkos::fence();

    Kokkos::deep_copy(h_contacts_i, d_contacts_i);
    Kokkos::deep_copy(h_contacts_j, d_contacts_j);
    Kokkos::deep_copy(h_contacts_calc, d_contacts_calc);

    // collect in segment order (== CPU emit order, pair_ls_dem.cpp:278-298) -> bitwise
    contacts.reserve(nseg);
    for (int seg = 0; seg < nseg; seg++)
      if (h_contacts_i(seg) >= 0)
        contacts.push_back({h_contacts_i(seg), h_contacts_j(seg), h_contacts_calc(seg)});
  }

  // force pass (host, unchanged from pair_ls_dem.cpp:432-433)
  double xl0[3] = {0.0, 0.0, 0.0};
  for (Contact &ct : contacts)
    process_contact(ct.i, ct.j, ct.calc, -1, xl0);

  if (vflag_fdotr) virial_fdotr_compute();            // pair_ls_dem.cpp:436

  atomKK->modified(Host, datamask_modify);            // force/torque written on host
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairLSDEMKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairLSDEMKokkos<LMPHostType>;
#endif
}
