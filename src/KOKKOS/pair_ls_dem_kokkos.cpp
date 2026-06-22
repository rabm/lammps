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

/* ----------------------------------------------------------------------
   M4a(2) device uploads of the read-only LS data the K2 force kernel (M4b)
   will consume: (a) the GLOBAL level-set grids, (b) the per-body BodyLS SoA +
   atom2body + cached per-atom binfo, (c) the per-type contact coefficients.
   Sources are the rigid/small fix's host arrays (via read-only accessors) and
   this pair's own coeff tables / binfo. In M4a these are uploaded once and
   round-trip self-checked; the force pass still runs on host (behaviour-neutral).
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::upload_coeffs_to_device()
{
  if (!allocated) return;
  const int np1 = atom->ntypes + 1;
  coeff_stride = np1;
  const int n = np1 * np1;

  Kokkos::realloc(d_kn, n);      Kokkos::realloc(d_kt, n);      Kokkos::realloc(d_mu, n);
  Kokkos::realloc(d_knp, n);     Kokkos::realloc(d_etan, n);    Kokkos::realloc(d_etat, n);
  Kokkos::realloc(d_etan1, n);   Kokkos::realloc(d_decayn1, n); Kokkos::realloc(d_etat1, n);
  Kokkos::realloc(d_decayt1, n);
  h_kn = Kokkos::create_mirror_view(d_kn);         h_kt = Kokkos::create_mirror_view(d_kt);
  h_mu = Kokkos::create_mirror_view(d_mu);         h_knp = Kokkos::create_mirror_view(d_knp);
  h_etan = Kokkos::create_mirror_view(d_etan);     h_etat = Kokkos::create_mirror_view(d_etat);
  h_etan1 = Kokkos::create_mirror_view(d_etan1);   h_decayn1 = Kokkos::create_mirror_view(d_decayn1);
  h_etat1 = Kokkos::create_mirror_view(d_etat1);   h_decayt1 = Kokkos::create_mirror_view(d_decayt1);

  // valid type indices are 1..ntypes; idx = i*np1 + j (row 0 / col 0 unused)
  for (int i = 1; i < np1; i++)
    for (int j = 1; j < np1; j++) {
      const int idx = i * np1 + j;
      h_kn(idx) = kn[i][j];           h_kt(idx) = kt[i][j];           h_mu(idx) = mu[i][j];
      h_knp(idx) = knp[i][j];         h_etan(idx) = etan[i][j];       h_etat(idx) = etat[i][j];
      h_etan1(idx) = etan1[i][j];     h_decayn1(idx) = decayn1[i][j]; h_etat1(idx) = etat1[i][j];
      h_decayt1(idx) = decayt1[i][j];
    }

  Kokkos::deep_copy(d_kn, h_kn);         Kokkos::deep_copy(d_kt, h_kt);
  Kokkos::deep_copy(d_mu, h_mu);         Kokkos::deep_copy(d_knp, h_knp);
  Kokkos::deep_copy(d_etan, h_etan);     Kokkos::deep_copy(d_etat, h_etat);
  Kokkos::deep_copy(d_etan1, h_etan1);   Kokkos::deep_copy(d_decayn1, h_decayn1);
  Kokkos::deep_copy(d_etat1, h_etat1);   Kokkos::deep_copy(d_decayt1, h_decayt1);
}

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::upload_grids_to_device()
{
  num_grids = fix_rigid_small ? fix_rigid_small->get_num_global_grids() : 0;
  if (num_grids <= 0) return;    // DISTRIBUTED storage path is deferred (K2 v1 = GLOBAL)

  double **gg   = fix_rigid_small->get_global_grids_array();
  int    **gsz  = fix_rigid_small->get_global_grids_size_array();
  double **gmin = fix_rigid_small->get_global_grids_min_array();

  Kokkos::realloc(d_grid_offset, num_grids + 1);
  Kokkos::realloc(d_grid_size, num_grids * 3);
  Kokkos::realloc(d_grid_min, num_grids * 3);
  h_grid_offset = Kokkos::create_mirror_view(d_grid_offset);
  h_grid_size   = Kokkos::create_mirror_view(d_grid_size);
  h_grid_min    = Kokkos::create_mirror_view(d_grid_min);

  int total = 0;
  h_grid_offset(0) = 0;
  for (int gi = 0; gi < num_grids; gi++) {
    const int ncount = gsz[gi][0] * gsz[gi][1] * gsz[gi][2];   // == ntotal_global[gi]
    total += ncount;
    h_grid_offset(gi + 1) = total;
    for (int a = 0; a < 3; a++) {
      h_grid_size(gi * 3 + a) = gsz[gi][a];
      h_grid_min(gi * 3 + a)  = gmin[gi][a];
    }
  }

  Kokkos::realloc(d_grid_values, total);
  h_grid_values = Kokkos::create_mirror_view(d_grid_values);
  for (int gi = 0; gi < num_grids; gi++) {
    const int base   = h_grid_offset(gi);
    const int ncount = h_grid_offset(gi + 1) - base;
    for (int k = 0; k < ncount; k++) h_grid_values(base + k) = gg[gi][k];
  }

  Kokkos::deep_copy(d_grid_offset, h_grid_offset);
  Kokkos::deep_copy(d_grid_size, h_grid_size);
  Kokkos::deep_copy(d_grid_min, h_grid_min);
  Kokkos::deep_copy(d_grid_values, h_grid_values);
}

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::upload_bodies_to_device()
{
  if (!fix_rigid_small) return;
  num_bodies = fix_rigid_small->get_nbodyLS();
  num_atoms_uploaded = atom->nlocal + atom->nghost;
  FixRigidSmallLSDEM::BodyLS *body = fix_rigid_small->get_bodyLS_array();
  int *a2b = fix_rigid_small->get_atom2body_array();

  if (num_bodies > 0) {
    Kokkos::realloc(d_body_style, num_bodies);       Kokkos::realloc(d_body_grid_index, num_bodies);
    Kokkos::realloc(d_body_grid_scale, num_bodies);  Kokkos::realloc(d_body_grid_stride, num_bodies);
    Kokkos::realloc(d_body_quatd2g, num_bodies * 4);
    h_body_style       = Kokkos::create_mirror_view(d_body_style);
    h_body_grid_index  = Kokkos::create_mirror_view(d_body_grid_index);
    h_body_grid_scale  = Kokkos::create_mirror_view(d_body_grid_scale);
    h_body_grid_stride = Kokkos::create_mirror_view(d_body_grid_stride);
    h_body_quatd2g     = Kokkos::create_mirror_view(d_body_quatd2g);
    for (int b = 0; b < num_bodies; b++) {
      h_body_style(b)       = body[b].style;
      h_body_grid_index(b)  = body[b].grid_index;
      h_body_grid_scale(b)  = body[b].grid_scale;
      h_body_grid_stride(b) = body[b].grid_stride;
      for (int a = 0; a < 4; a++) h_body_quatd2g(b * 4 + a) = body[b].quatd2g[a];
    }
    Kokkos::deep_copy(d_body_style, h_body_style);
    Kokkos::deep_copy(d_body_grid_index, h_body_grid_index);
    Kokkos::deep_copy(d_body_grid_scale, h_body_grid_scale);
    Kokkos::deep_copy(d_body_grid_stride, h_body_grid_stride);
    Kokkos::deep_copy(d_body_quatd2g, h_body_quatd2g);
  }

  // per-atom atom2body + cached binfo (filled by cache_body_info earlier this compute)
  const int na = num_atoms_uploaded;
  Kokkos::realloc(d_atom2body, na);
  Kokkos::realloc(d_binfo_vol, na);  Kokkos::realloc(d_binfo_area, na);
  Kokkos::realloc(d_binfo_bID, na);  Kokkos::realloc(d_binfo_bidx, na);
  Kokkos::realloc(d_binfo_grp, na);  Kokkos::realloc(d_binfo_off, na);
  h_atom2body = Kokkos::create_mirror_view(d_atom2body);
  h_binfo_vol = Kokkos::create_mirror_view(d_binfo_vol);   h_binfo_area = Kokkos::create_mirror_view(d_binfo_area);
  h_binfo_bID = Kokkos::create_mirror_view(d_binfo_bID);   h_binfo_bidx = Kokkos::create_mirror_view(d_binfo_bidx);
  h_binfo_grp = Kokkos::create_mirror_view(d_binfo_grp);   h_binfo_off  = Kokkos::create_mirror_view(d_binfo_off);
  for (int a = 0; a < na; a++) {
    h_atom2body(a) = a2b[a];
    const BodyInfo &bi = binfo[a];
    h_binfo_vol(a)  = bi.vol;          h_binfo_area(a) = bi.area;
    h_binfo_bID(a)  = bi.bID;          h_binfo_bidx(a) = bi.bidx;
    h_binfo_grp(a)  = (int) bi.grp;    h_binfo_off(a)  = (int) bi.off;
  }
  Kokkos::deep_copy(d_atom2body, h_atom2body);
  Kokkos::deep_copy(d_binfo_vol, h_binfo_vol);   Kokkos::deep_copy(d_binfo_area, h_binfo_area);
  Kokkos::deep_copy(d_binfo_bID, h_binfo_bID);   Kokkos::deep_copy(d_binfo_bidx, h_binfo_bidx);
  Kokkos::deep_copy(d_binfo_grp, h_binfo_grp);   Kokkos::deep_copy(d_binfo_off, h_binfo_off);
}

/* ----------------------------------------------------------------------
   Deep-copy every uploaded View back to host and assert byte-equality with the
   ORIGINAL host source arrays (not the upload mirrors), so a fill bug or a
   size/layout bug is caught. doubles compare exact (verbatim copy). One-shot.
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::selfcheck_uploads()
{
  long mism = 0;

  // ---- coeffs ----
  if (allocated) {
    const int np1 = coeff_stride;
    auto c_kn = Kokkos::create_mirror_view(d_kn);           Kokkos::deep_copy(c_kn, d_kn);
    auto c_kt = Kokkos::create_mirror_view(d_kt);           Kokkos::deep_copy(c_kt, d_kt);
    auto c_mu = Kokkos::create_mirror_view(d_mu);           Kokkos::deep_copy(c_mu, d_mu);
    auto c_knp = Kokkos::create_mirror_view(d_knp);         Kokkos::deep_copy(c_knp, d_knp);
    auto c_etan = Kokkos::create_mirror_view(d_etan);       Kokkos::deep_copy(c_etan, d_etan);
    auto c_etat = Kokkos::create_mirror_view(d_etat);       Kokkos::deep_copy(c_etat, d_etat);
    auto c_etan1 = Kokkos::create_mirror_view(d_etan1);     Kokkos::deep_copy(c_etan1, d_etan1);
    auto c_decayn1 = Kokkos::create_mirror_view(d_decayn1); Kokkos::deep_copy(c_decayn1, d_decayn1);
    auto c_etat1 = Kokkos::create_mirror_view(d_etat1);     Kokkos::deep_copy(c_etat1, d_etat1);
    auto c_decayt1 = Kokkos::create_mirror_view(d_decayt1); Kokkos::deep_copy(c_decayt1, d_decayt1);
    for (int i = 1; i < np1; i++)
      for (int j = 1; j < np1; j++) {
        const int idx = i * np1 + j;
        mism += (c_kn(idx) != kn[i][j]) + (c_kt(idx) != kt[i][j]) + (c_mu(idx) != mu[i][j]) +
                (c_knp(idx) != knp[i][j]) + (c_etan(idx) != etan[i][j]) + (c_etat(idx) != etat[i][j]) +
                (c_etan1(idx) != etan1[i][j]) + (c_decayn1(idx) != decayn1[i][j]) +
                (c_etat1(idx) != etat1[i][j]) + (c_decayt1(idx) != decayt1[i][j]);
      }
  }

  // ---- grids ----
  if (num_grids > 0) {
    int    **gsz  = fix_rigid_small->get_global_grids_size_array();
    double **gmin = fix_rigid_small->get_global_grids_min_array();
    double **gg   = fix_rigid_small->get_global_grids_array();
    auto c_off  = Kokkos::create_mirror_view(d_grid_offset); Kokkos::deep_copy(c_off, d_grid_offset);
    auto c_size = Kokkos::create_mirror_view(d_grid_size);   Kokkos::deep_copy(c_size, d_grid_size);
    auto c_min  = Kokkos::create_mirror_view(d_grid_min);    Kokkos::deep_copy(c_min, d_grid_min);
    auto c_val  = Kokkos::create_mirror_view(d_grid_values); Kokkos::deep_copy(c_val, d_grid_values);
    int off = 0;
    if (c_off(0) != 0) mism++;
    for (int gi = 0; gi < num_grids; gi++) {
      const int ncount = gsz[gi][0] * gsz[gi][1] * gsz[gi][2];
      for (int a = 0; a < 3; a++) {
        mism += (c_size(gi * 3 + a) != gsz[gi][a]) + (c_min(gi * 3 + a) != gmin[gi][a]);
      }
      mism += (c_off(gi) != off);
      for (int k = 0; k < ncount; k++)
        if (c_val(off + k) != gg[gi][k]) mism++;
      off += ncount;
    }
    if (c_off(num_grids) != off) mism++;
  }

  // ---- bodies + per-atom ----
  if (fix_rigid_small) {
    FixRigidSmallLSDEM::BodyLS *body = fix_rigid_small->get_bodyLS_array();
    int *a2b = fix_rigid_small->get_atom2body_array();
    if (num_bodies > 0) {
      auto c_st = Kokkos::create_mirror_view(d_body_style);        Kokkos::deep_copy(c_st, d_body_style);
      auto c_gi = Kokkos::create_mirror_view(d_body_grid_index);   Kokkos::deep_copy(c_gi, d_body_grid_index);
      auto c_gs = Kokkos::create_mirror_view(d_body_grid_scale);   Kokkos::deep_copy(c_gs, d_body_grid_scale);
      auto c_gt = Kokkos::create_mirror_view(d_body_grid_stride);  Kokkos::deep_copy(c_gt, d_body_grid_stride);
      auto c_q  = Kokkos::create_mirror_view(d_body_quatd2g);      Kokkos::deep_copy(c_q, d_body_quatd2g);
      for (int b = 0; b < num_bodies; b++) {
        mism += (c_st(b) != body[b].style) + (c_gi(b) != body[b].grid_index) +
                (c_gs(b) != body[b].grid_scale) + (c_gt(b) != body[b].grid_stride);
        for (int a = 0; a < 4; a++) if (c_q(b * 4 + a) != body[b].quatd2g[a]) mism++;
      }
    }
    const int na = num_atoms_uploaded;
    auto c_a2b = Kokkos::create_mirror_view(d_atom2body);  Kokkos::deep_copy(c_a2b, d_atom2body);
    auto c_vol = Kokkos::create_mirror_view(d_binfo_vol);  Kokkos::deep_copy(c_vol, d_binfo_vol);
    auto c_area = Kokkos::create_mirror_view(d_binfo_area); Kokkos::deep_copy(c_area, d_binfo_area);
    auto c_bID = Kokkos::create_mirror_view(d_binfo_bID);  Kokkos::deep_copy(c_bID, d_binfo_bID);
    auto c_bidx = Kokkos::create_mirror_view(d_binfo_bidx); Kokkos::deep_copy(c_bidx, d_binfo_bidx);
    auto c_grp = Kokkos::create_mirror_view(d_binfo_grp);  Kokkos::deep_copy(c_grp, d_binfo_grp);
    auto c_off = Kokkos::create_mirror_view(d_binfo_off);  Kokkos::deep_copy(c_off, d_binfo_off);
    for (int a = 0; a < na; a++) {
      const BodyInfo &bi = binfo[a];
      mism += (c_a2b(a) != a2b[a]) + (c_vol(a) != bi.vol) + (c_area(a) != bi.area) +
              (c_bID(a) != bi.bID) + (c_bidx(a) != bi.bidx) +
              (c_grp(a) != (int) bi.grp) + (c_off(a) != (int) bi.off);
    }
  }

  if (mism != 0)
    error->one(FLERR, "PairLSDEMKokkos (M4a): device upload self-check FAILED ({} mismatches)", mism);
  if (comm->me == 0)
    utils::logmesg(lmp, "PairLSDEMKokkos (M4a): device upload self-check PASSED "
                        "({} grids, {} bodies, {} atoms, {} types)\n",
                        num_grids, num_bodies, num_atoms_uploaded, atom->ntypes);
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

  // M4a(2): one-shot device upload of the read-only LS data (grids/bodies/coeffs)
  // + a deep-copy-back self-check. K2 (M4b) consumes these on device; for now the
  // force pass still runs on host, so this block is behaviour-neutral (verify.sh
  // stays bitwise). binfo was filled by cache_body_info in the rebuild branch above.
  if (!uploads_done && fix_rigid_small) {
    upload_coeffs_to_device();
    upload_grids_to_device();
    upload_bodies_to_device();
    selfcheck_uploads();
    uploads_done = true;
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
