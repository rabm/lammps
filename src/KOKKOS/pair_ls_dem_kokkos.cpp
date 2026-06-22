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

#include "ls_dem_extra_device.h"   // LSDEMExtra:: device math + get_ls_value helpers + interpolate_LS_array (K2)

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "kokkos.h"
#include "neighbor.h"
#include "update.h"

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
  Kokkos::View<int*, DeviceType> d_atom_seg_offset, d_cand_offset, d_cand;
  Kokkos::View<int*, DeviceType> d_ci, d_cj, d_ccalc;
  int nlocal;
  double maxcutsq;

  // One thread per REP NODE r (RangePolicy over all local+ghost atoms). Reduces the
  // single closest candidate across ALL of r's segments (one segment == one partner body),
  // i.e. the one-contact-per-rep guard: a rep node straddling >1 body keeps only its closest
  // (~most-penetrating) contact. For a single-segment rep this is identical to the old
  // per-segment winner; it differs only for (rare) multi-body reps. Emits seg/sweep-order
  // first-min so the host collects contacts in rep-atom order (deterministic).
  KOKKOS_INLINE_FUNCTION
  void operator()(const int r) const
  {
    const int s0 = d_atom_seg_offset(r);
    const int s1 = d_atom_seg_offset(r + 1);
    if (s0 == s1) { d_ci(r) = -1; d_cj(r) = -1; d_ccalc(r) = 0; return; }   // r is not a rep

    const bool r_local = (r < nlocal);
    const double xr0 = x(r,0), xr1 = x(r,1), xr2 = x(r,2);

    int widx = -1;
    double minrsq = 0.0;
    for (int seg = s0; seg < s1; seg++) {           // over this rep's segments (partner bodies)
      const int cs = d_cand_offset(seg);
      const int ce = d_cand_offset(seg + 1);
      for (int cc = cs; cc < ce; cc++) {
        const int jc = d_cand(cc);
        const double dx = xr0 - x(jc,0);
        const double dy = xr1 - x(jc,1);
        const double dz = xr2 - x(jc,2);
        const double rsq = dx*dx + dy*dy + dz*dz;
        if (rsq > maxcutsq) continue;
        if (widx < 0 || rsq < minrsq) { minrsq = rsq; widx = jc; }   // first-min, segment-then-sweep
      }
    }

    // Newton-OFF emit rule, verbatim from pair_ls_dem.cpp:294-296:
    //   widx<0 -> none; r local -> {r,widx,1}; else widx<nlocal -> {widx,r,0}; else both-ghost -> none.
    int ei = -1, ej = -1, ecalc = 0;
    if (widx >= 0) {
      if (r_local)            { ei = r;    ej = widx; ecalc = 1; }
      else if (widx < nlocal) { ei = widx; ej = r;    ecalc = 0; }
    }
    d_ci(r) = ei;
    d_cj(r) = ej;
    d_ccalc(r) = ecalc;
  }
};

/* ----------------------------------------------------------------------
   K2 device force functor: one thread per rep-indexed contact slot. Replicates
   PairLSDEM::process_contact (pair_ls_dem.cpp:448-890) + the GLOBAL branch of
   FixRigidSmallLSDEM::get_ls_value (fix_rigid_small_ls_dem.cpp:1602-1674) on the
   device, reading the M4a uploads (grids/bodies/coeffs/binfo) + the bridged history
   buffers. f/torque are written via Kokkos::atomic_add (a local node can be the
   force target of multiple contacts). History is per-rep and exclusive (one contact
   per rep), so its RMW needs no atomics. Virial is DEFERRED on device (M4b v1): the
   compute() driver runs the exact CPU path on virial/energy steps (evflag != 0).
   Within-tol vs CPU (atomic_add reorders the per-body force sum); NOT bitwise.
------------------------------------------------------------------------- */

template<class DeviceType>
struct PairLSDEMK2 {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_randomread x, v, xcom, omega;
  typename AT::t_kkfloat_1d_4 quat;
  typename AT::t_kkacc_1d_3 f, torque;                 // atomic_add targets
  typename AT::t_int_1d_randomread type;

  Kokkos::View<int*, DeviceType> d_ci, d_cj, d_ccalc;  // rep-indexed contact slots (K1)

  // M4a uploads (read-only)
  Kokkos::View<double*, DeviceType> d_grid_values, d_grid_min;
  Kokkos::View<int*, DeviceType>    d_grid_offset, d_grid_size;
  Kokkos::View<int*, DeviceType>    d_body_grid_index;
  Kokkos::View<double*, DeviceType> d_body_grid_scale, d_body_grid_stride;
  Kokkos::View<int*, DeviceType>    d_atom2body, d_binfo_bID, d_binfo_bidx;
  Kokkos::View<double*, DeviceType> d_binfo_area;
  Kokkos::View<double*, DeviceType> d_kn,d_kt,d_mu,d_knp,d_etan,d_etat,d_etan1,d_decayn1,d_etat1,d_decayt1;
  int coeff_stride;

  // history bridge buffers (per-atom, host-synced around K2)
  Kokkos::View<double*, DeviceType> d_hist_n, d_hist_fs;    // flat ntotal*3
  Kokkos::View<int*, DeviceType>    d_hist_touch;           // ntotal
  Kokkos::View<double*, DeviceType> d_hist_fn1, d_hist_fs1; // ntotal

  int nlocal, dim;
  double dt, xprd, yprd, zprd;
  int px, py, pz;

  // device replica of FixRigidSmallLSDEM::get_ls_value, GLOBAL branch (fix:1602-1674)
  KOKKOS_INLINE_FUNCTION
  double get_ls_value(const int i, const int j, double *normal) const
  {
    const int jbody = d_atom2body(j);
    const double jstride = d_body_grid_stride(jbody);
    const double strideinv = 1.0 / jstride;

    double delx = x(i,0) - xcom(j,0);
    double dely = x(i,1) - xcom(j,1);
    double delz = x(i,2) - xcom(j,2);
    LSDEMExtra::ls_dem_minimum_image_ortho(delx, dely, delz, xprd, yprd, zprd, px, py, pz);

    double dxv[3] = {delx, dely, delz};
    double qj[4]  = {quat(j,0), quat(j,1), quat(j,2), quat(j,3)};
    double qconj[4]; LSDEMExtra::ls_dem_qconjugate(qj, qconj);
    double x_local[3]; LSDEMExtra::ls_dem_quatrotvec(qconj, dxv, x_local);

    const int gi = d_body_grid_index(jbody);            // GLOBAL storage only
    const double gscale = d_body_grid_scale(jbody);
    const int base = d_grid_offset(gi);
    int ngrid[3] = { d_grid_size(gi*3+0), d_grid_size(gi*3+1), d_grid_size(gi*3+2) };
    x_local[0] -= d_grid_min(gi*3+0) * gscale;
    x_local[1] -= d_grid_min(gi*3+1) * gscale;
    x_local[2] -= d_grid_min(gi*3+2) * gscale;

    double x_red[3] = { x_local[0]*strideinv, x_local[1]*strideinv, x_local[2]*strideinv };
    int ix[3] = { (int) x_red[0], (int) x_red[1], (int) x_red[2] };
    int mybin = ix[0] + ix[1]*ngrid[0] + ix[2]*ngrid[0]*ngrid[1];

    double dist = LSDEMExtra::interpolate_LS_array(dim, mybin, &d_grid_values(base),
                                                   ngrid, x_red, ix, normal, jstride);
    dist *= gscale;                                     // GLOBAL post-multiply (fix:1667)
    if (dist < 0.0) LSDEMExtra::ls_dem_quatrotvec(qj, normal, normal);   // back-rotate (fix:1671)
    return dist;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int r) const
  {
    const int i = d_ci(r);
    if (i < 0) return;                                  // no contact for this rep
    const int j = d_cj(r);
    const int calc = d_ccalc(r);
    const double EPS = 1e-12;                            // PairLSDEM::EPSILON

    const int itype = type(i), jtype = type(j);
    const int ibody = d_binfo_bidx(i), jbody = d_binfo_bidx(j);
    if (ibody < 0 || jbody < 0) return;                 // CPU errors; device skips (shouldn't happen)
    const int ibodyID = d_binfo_bID(i), jbodyID = d_binfo_bID(j);
    const double areai = d_binfo_area(i), areaj = d_binfo_area(j);

    const int rep = calc ? i : j;
    const int par = calc ? j : i;

    double normal[3];
    double u = - get_ls_value(rep, par, normal);

    const int ni = rep;
    const double hsign = calc ? 1.0 : -1.0;
    const double narea = calc ? areai : areaj;
    const int partner_body = calc ? jbodyID : ibodyID;

    // history (per-rep, exclusive under the one-contact-per-rep guard)
    double fs_ni[3] = { d_hist_fs(ni*3+0), d_hist_fs(ni*3+1), d_hist_fs(ni*3+2) };
    double n_ni[3]  = { d_hist_n(ni*3+0),  d_hist_n(ni*3+1),  d_hist_n(ni*3+2) };
    int touch       = d_hist_touch(ni);

    if (u <= 0.0) {                                     // no contact (:524-537)
      if (touch == partner_body) {
        d_hist_touch(ni) = -1;
        d_hist_fs(ni*3+0)=0.0; d_hist_fs(ni*3+1)=0.0; d_hist_fs(ni*3+2)=0.0;
        d_hist_n(ni*3+0)=0.0;  d_hist_n(ni*3+1)=0.0;  d_hist_n(ni*3+2)=0.0;
      }
      return;
    }

    const double xitmp=x(i,0), yitmp=x(i,1), zitmp=x(i,2);
    const double xjtmp=x(j,0), yjtmp=x(j,1), zjtmp=x(j,2);
    const double vxi=v(i,0), vyi=v(i,1), vzi=v(i,2);
    const double vxj=v(j,0), vyj=v(j,1), vzj=v(j,2);
    const double icomx=xcom(i,0), icomy=xcom(i,1), icomz=xcom(i,2);
    const double jcomx=xcom(j,0), jcomy=xcom(j,1), jcomz=xcom(j,2);
    const double iomx=omega(i,0), iomy=omega(i,1), iomz=omega(i,2);
    const double jomx=omega(j,0), jomy=omega(j,1), jomz=omega(j,2);

    const double normsign = calc ? -1.0 : 1.0;          // (:560)
    normal[0]*=normsign; normal[1]*=normsign; normal[2]*=normsign;
    const double npos0 = calc ? xitmp : xjtmp;
    const double npos1 = calc ? yitmp : yjtmp;
    const double npos2 = calc ? zitmp : zjtmp;
    double contact_point[3];
    contact_point[0]=npos0 - 0.5*u*hsign*normal[0];
    contact_point[1]=npos1 - 0.5*u*hsign*normal[1];
    contact_point[2]=npos2 - 0.5*u*hsign*normal[2];

    const int cidx = itype*coeff_stride + jtype;        // coeff flatten i*np1+j
    const double knij=d_kn(cidx), ktij=d_kt(cidx), muij=d_mu(cidx), knpij=d_knp(cidx);
    const double etanij=d_etan(cidx), etatij=d_etat(cidx);
    const double etan1ij=d_etan1(cidx), decayn1ij=d_decayn1(cidx);
    const double etat1ij=d_etat1(cidx), decayt1ij=d_decayt1(cidx);

    // ---- normal stress ----
    double fn_mag = (fabs(knpij) < EPS) ? (knij*u) : (knij*pow(u, knpij));
    double v_rel[3] = { vxi-vxj, vyi-vyj, vzi-vzj };
    double v_rel_n_mag = LSDEMExtra::ls_dem_dot3(v_rel, normal);
    if (etanij > 0.0) fn_mag += etanij*v_rel_n_mag;
    if (etan1ij > 0.0) {                                // Maxwell arm 1 (fn1 RMW :623)
      double fn1v = decayn1ij*d_hist_fn1(ni) + etan1ij*(1.0-decayn1ij)*v_rel_n_mag;
      d_hist_fn1(ni) = fn1v;
      fn_mag += fn1v;
    }
    fn_mag = (fn_mag > 0.0) ? fn_mag : 0.0;
    double fpair[3] = { -fn_mag*normal[0], -fn_mag*normal[1], -fn_mag*normal[2] };

    // ---- tangent stress ----
    if (touch == -1) touch = partner_body;              // (:649-651; mismatch-warn skipped on device)

    double fs_tmp[3] = { hsign*fs_ni[0], hsign*fs_ni[1], hsign*fs_ni[2] };
    double normal_old[3] = { hsign*n_ni[0], hsign*n_ni[1], hsign*n_ni[2] };

    if (LSDEMExtra::ls_dem_lensq3(normal_old) > 0.0) {  // unnormalised Rodrigues (:672-710)
      double k[3]; LSDEMExtra::ls_dem_cross3(normal_old, normal, k);
      double spin_norm = 0.5*dt*((iomx+jomx)*normal_old[0] + (iomy+jomy)*normal_old[1]
                                 + (iomz+jomz)*normal_old[2]);
      double t1[3]; LSDEMExtra::ls_dem_cross3(normal_old, k, t1);
      double bch_coef = -0.5*fabs(spin_norm);
      k[0]+=spin_norm*normal_old[0]+bch_coef*t1[0];
      k[1]+=spin_norm*normal_old[1]+bch_coef*t1[1];
      k[2]+=spin_norm*normal_old[2]+bch_coef*t1[2];
      double sinsq = LSDEMExtra::ls_dem_lensq3(k);
      if (sinsq > EPS*EPS) {
        double costheta = sqrt(((1.0-sinsq)>0.0)?(1.0-sinsq):0.0);
        double kxfs[3]; LSDEMExtra::ls_dem_cross3(k, fs_tmp, kxfs);
        double term2 = LSDEMExtra::ls_dem_dot3(k, fs_tmp)/(1.0+costheta);
        fs_tmp[0]=fs_tmp[0]*costheta+kxfs[0]+k[0]*term2;
        fs_tmp[1]=fs_tmp[1]*costheta+kxfs[1]+k[1]*term2;
        fs_tmp[2]=fs_tmp[2]*costheta+kxfs[2]+k[2]*term2;
      }
    }

    double v_rel_t[3] = { v_rel[0]-v_rel_n_mag*normal[0], v_rel[1]-v_rel_n_mag*normal[1],
                          v_rel[2]-v_rel_n_mag*normal[2] };
    double v_rel_t_mag = LSDEMExtra::ls_dem_len3(v_rel_t);
    double tangent[3], inv;
    if (v_rel_t_mag > EPS) {
      inv = 1.0/v_rel_t_mag;
      tangent[0]=v_rel_t[0]*inv; tangent[1]=v_rel_t[1]*inv; tangent[2]=v_rel_t[2]*inv;
    } else {
      double nrm = LSDEMExtra::ls_dem_len3(fs_tmp);
      inv = (nrm != 0.0) ? 1.0/nrm : 0.0;
      tangent[0]=fs_tmp[0]*inv; tangent[1]=fs_tmp[1]*inv; tangent[2]=fs_tmp[2]*inv;
    }

    double shear_incr = ktij*v_rel_t_mag*dt;
    fs_tmp[0]-=shear_incr*tangent[0]; fs_tmp[1]-=shear_incr*tangent[1]; fs_tmp[2]-=shear_incr*tangent[2];
    double fs_mag_trial = LSDEMExtra::ls_dem_len3(fs_tmp);
    double fs_max = muij*fn_mag;
    double fs_mag = (fs_max < fs_mag_trial) ? fs_max : fs_mag_trial;
    if (fs_mag_trial > EPS) {
      fs_tmp[0]=fs_mag*(fs_tmp[0]/fs_mag_trial);
      fs_tmp[1]=fs_mag*(fs_tmp[1]/fs_mag_trial);
      fs_tmp[2]=fs_mag*(fs_tmp[2]/fs_mag_trial);
    }

    // store history (elastic shear after first clamp + current normal; :768-773)
    d_hist_fs(ni*3+0)=hsign*fs_tmp[0]; d_hist_fs(ni*3+1)=hsign*fs_tmp[1]; d_hist_fs(ni*3+2)=hsign*fs_tmp[2];
    d_hist_n(ni*3+0)=hsign*normal[0];  d_hist_n(ni*3+1)=hsign*normal[1];  d_hist_n(ni*3+2)=hsign*normal[2];
    d_hist_touch(ni)=touch;

    double fs_mag_add = 0.0;
    if (etatij > 0.0) fs_mag_add += etatij*v_rel_t_mag;
    if (etan1ij > 0.0) {                                // Maxwell arm 1 tangential (fs1 RMW :799)
      double nrm = LSDEMExtra::ls_dem_len3(fs_tmp);
      double tinv = (nrm != 0.0) ? 1.0/nrm : 0.0;
      double tangent_old[3] = { fs_tmp[0]*tinv, fs_tmp[1]*tinv, fs_tmp[2]*tinv };
      double fs1v = d_hist_fs1(ni)
                    + decayt1ij*d_hist_fs1(ni)*LSDEMExtra::ls_dem_dot3(tangent_old, tangent)
                    + etat1ij*(1.0-decayt1ij)*v_rel_t_mag;
      d_hist_fs1(ni) = fs1v;
      fs_mag_add += fs1v;
    }
    fs_tmp[0]-=fs_mag_add*tangent[0]; fs_tmp[1]-=fs_mag_add*tangent[1]; fs_tmp[2]-=fs_mag_add*tangent[2];
    fs_mag_trial = LSDEMExtra::ls_dem_len3(fs_tmp);
    fs_mag = (fs_max < fs_mag_trial) ? fs_max : fs_mag_trial;
    if (fs_mag > 0.0 && fs_mag_trial != 0.0) {
      fs_tmp[0]=fs_mag*(fs_tmp[0]/fs_mag_trial);
      fs_tmp[1]=fs_mag*(fs_tmp[1]/fs_mag_trial);
      fs_tmp[2]=fs_mag*(fs_tmp[2]/fs_mag_trial);
      fpair[0]+=fs_tmp[0]; fpair[1]+=fs_tmp[1]; fpair[2]+=fs_tmp[2];
    }

    // ---- total force/torque ----
    fpair[0]*=narea; fpair[1]*=narea; fpair[2]*=narea;  // stress -> force (:835-837)

    Kokkos::atomic_add(&f(i,0), fpair[0]);
    Kokkos::atomic_add(&f(i,1), fpair[1]);
    Kokkos::atomic_add(&f(i,2), fpair[2]);
    double lever[3] = { contact_point[0]-icomx, contact_point[1]-icomy, contact_point[2]-icomz };
    LSDEMExtra::ls_dem_minimum_image_ortho(lever[0],lever[1],lever[2], xprd,yprd,zprd, px,py,pz);
    double tq[3]; LSDEMExtra::ls_dem_cross3(lever, fpair, tq);
    Kokkos::atomic_add(&torque(i,0), tq[0]);
    Kokkos::atomic_add(&torque(i,1), tq[1]);
    Kokkos::atomic_add(&torque(i,2), tq[2]);

    if (j < nlocal) {                                   // Newton-OFF mirror (:871-887)
      LSDEMExtra::ls_dem_negate3(fpair);
      Kokkos::atomic_add(&f(j,0), fpair[0]);
      Kokkos::atomic_add(&f(j,1), fpair[1]);
      Kokkos::atomic_add(&f(j,2), fpair[2]);
      double leverj[3] = { contact_point[0]-jcomx, contact_point[1]-jcomy, contact_point[2]-jcomz };
      LSDEMExtra::ls_dem_minimum_image_ortho(leverj[0],leverj[1],leverj[2], xprd,yprd,zprd, px,py,pz);
      double tqj[3]; LSDEMExtra::ls_dem_cross3(leverj, fpair, tqj);
      Kokkos::atomic_add(&torque(j,0), tqj[0]);
      Kokkos::atomic_add(&torque(j,1), tqj[1]);
      Kokkos::atomic_add(&torque(j,2), tqj[2]);
    }
    // virial (:889): DEFERRED on device — handled by the host fallback on evflag steps.
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
  PairLSDEM::init_style();   // adds the REQ_GHOST (half + ghost) request; finds the rigid fixes

  // No FULL-list rejection needed: our request is REQ_GHOST (half) and we never call
  // enable_full(), so the list stays a HALF list with ghosts even when the GPU default
  // neighflag is FULL (the global kokkos neighflag is only a per-pair hint that the neighbor
  // system does NOT auto-apply to our request). build_rep_segments walks it on the host (the
  // legacy firstneigh stays populated as in the M3/M4 path); the atom data goes to the device
  // for K1/K2. => the user does NOT need `-pk kokkos neigh half` on GPU. (Do NOT force
  // kokkos_host/device on the request -- that changes the list layout and segfaults
  // build_rep_segments by leaving firstneigh unpopulated.)

  if (comm->me == 0)
    utils::logmesg(lmp, "PairLSDEMKokkos: device K1+K2 kernels over a host-walked half neighbour list\n");
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
  // per-segment candidate-offset CSR: needs nseg+1 entries.
  if (ns + 1 > seg_cap) {
    seg_cap = (ns + 1) + (ns + 1) / 4 + 16;
    Kokkos::realloc(d_cand_offset, seg_cap);
    h_cand_offset = Kokkos::create_mirror_view(d_cand_offset);
  }
  if (nc > cand_cap) {
    cand_cap = nc + nc / 4 + 16;
    Kokkos::realloc(d_cand, cand_cap);
    h_cand = Kokkos::create_mirror_view(d_cand);
  }
  // per-rep (atom-indexed) arrays: d_atom_seg_offset[ntotal+1] + one contact slot per rep atom.
  if (ntotal + 1 > slot_cap) {
    slot_cap = (ntotal + 1) + (ntotal + 1) / 4 + 16;
    Kokkos::realloc(d_atom_seg_offset, slot_cap);
    Kokkos::realloc(d_contacts_i, slot_cap);
    Kokkos::realloc(d_contacts_j, slot_cap);
    Kokkos::realloc(d_contacts_calc, slot_cap);
    h_atom_seg_offset = Kokkos::create_mirror_view(d_atom_seg_offset);
    h_contacts_i      = Kokkos::create_mirror_view(d_contacts_i);
    h_contacts_j      = Kokkos::create_mirror_view(d_contacts_j);
    h_contacts_calc   = Kokkos::create_mirror_view(d_contacts_calc);
  }

  // Fill in (rep, then segment, then candidate) order == the CPU emit order
  // (pair_ls_dem.cpp:278). Segments for rep r are contiguous, so d_atom_seg_offset is a
  // simple prefix over reps: rep r owns segments [h_atom_seg_offset(r), h_atom_seg_offset(r+1)).
  int seg = 0, c = 0;
  h_cand_offset(0) = 0;
  for (int r = 0; r < ntotal; r++) {
    h_atom_seg_offset(r) = seg;
    for (const auto &s : rep_segs[r]) {
      for (int jc : s.cand) h_cand(c++) = jc;        // sweep order preserved (pair_ls_dem.cpp:177)
      h_cand_offset(seg + 1) = c;
      seg++;
    }
  }
  h_atom_seg_offset(ntotal) = seg;                   // == nseg

  Kokkos::deep_copy(d_atom_seg_offset, h_atom_seg_offset);
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

/* ----------------------------------------------------------------------
   M4b history bridge: copy the 5 host shear-history arrays into device buffers
   before K2 (and back after). n/fs (darray) + touch_id (ivector) are host-only;
   fn1/fs1 (dvector) are read via the host pointer (the M3/M4a bitwise gate did the
   same). Buffers are sized to ntotal (local+ghost) and grown with margin.
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::sync_history_to_device()
{
  const int ntotal = atom->nlocal + atom->nghost;
  if (ntotal > hist_cap) {
    hist_cap = ntotal + ntotal / 4 + 16;
    Kokkos::realloc(d_hist_n, hist_cap * 3);   Kokkos::realloc(d_hist_fs, hist_cap * 3);
    Kokkos::realloc(d_hist_touch, hist_cap);
    Kokkos::realloc(d_hist_fn1, hist_cap);     Kokkos::realloc(d_hist_fs1, hist_cap);
    h_hist_n     = Kokkos::create_mirror_view(d_hist_n);
    h_hist_fs    = Kokkos::create_mirror_view(d_hist_fs);
    h_hist_touch = Kokkos::create_mirror_view(d_hist_touch);
    h_hist_fn1   = Kokkos::create_mirror_view(d_hist_fn1);
    h_hist_fs1   = Kokkos::create_mirror_view(d_hist_fs1);
  }
  double **n  = atom->darray[index_ls_dem_n];
  double **fs = atom->darray[index_ls_dem_fs];
  int *touch  = atom->ivector[index_ls_dem_touch_id];
  double *fn1 = atom->dvector[index_ls_dem_fn1];
  double *fs1 = atom->dvector[index_ls_dem_fs1];
  for (int a = 0; a < ntotal; a++) {
    h_hist_n(a*3+0)=n[a][0]; h_hist_n(a*3+1)=n[a][1]; h_hist_n(a*3+2)=n[a][2];
    h_hist_fs(a*3+0)=fs[a][0]; h_hist_fs(a*3+1)=fs[a][1]; h_hist_fs(a*3+2)=fs[a][2];
    h_hist_touch(a)=touch[a]; h_hist_fn1(a)=fn1[a]; h_hist_fs1(a)=fs1[a];
  }
  Kokkos::deep_copy(d_hist_n, h_hist_n);     Kokkos::deep_copy(d_hist_fs, h_hist_fs);
  Kokkos::deep_copy(d_hist_touch, h_hist_touch);
  Kokkos::deep_copy(d_hist_fn1, h_hist_fn1); Kokkos::deep_copy(d_hist_fs1, h_hist_fs1);
}

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::sync_history_from_device()
{
  const int ntotal = atom->nlocal + atom->nghost;
  Kokkos::deep_copy(h_hist_n, d_hist_n);     Kokkos::deep_copy(h_hist_fs, d_hist_fs);
  Kokkos::deep_copy(h_hist_touch, d_hist_touch);
  Kokkos::deep_copy(h_hist_fn1, d_hist_fn1); Kokkos::deep_copy(h_hist_fs1, d_hist_fs1);
  double **n  = atom->darray[index_ls_dem_n];
  double **fs = atom->darray[index_ls_dem_fs];
  int *touch  = atom->ivector[index_ls_dem_touch_id];
  double *fn1 = atom->dvector[index_ls_dem_fn1];
  double *fs1 = atom->dvector[index_ls_dem_fs1];
  for (int a = 0; a < ntotal; a++) {
    n[a][0]=h_hist_n(a*3+0); n[a][1]=h_hist_n(a*3+1); n[a][2]=h_hist_n(a*3+2);
    fs[a][0]=h_hist_fs(a*3+0); fs[a][1]=h_hist_fs(a*3+1); fs[a][2]=h_hist_fs(a*3+2);
    touch[a]=h_hist_touch(a); fn1[a]=h_hist_fn1(a); fs1[a]=h_hist_fs1(a);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::compute(int eflag, int vflag)
{
  // K2 v1 handles the SMALL fix (fix rigid/small/ls/dem) + GLOBAL storage only, no watershed.
  // Defer to the exact CPU compute (host) in every other case: watershed, a LARGE fix
  // (fix rigid/ls/dem) present, the small fix absent, or DISTRIBUTED storage (num_global_grids==0).
  // The large-body get_ls_value + DISTRIBUTED + watershed paths are future milestones; without
  // this guard K2 would read unallocated upload Views (e.g. the sphere-plate cases use the large
  // fix) and segfault. watershed_flag is set in PairLSDEM::setup().
  const bool device_ok = (watershed_flag == 0) && fix_rigid_small && !fix_rigid &&
                         (fix_rigid_small->get_num_global_grids() > 0);
  if (!device_ok) {
    atomKK->sync(Host, datamask_read);
    PairLSDEM::compute(eflag, vflag);
    atomKK->modified(Host, datamask_modify);
    return;
  }

  if (eflag || vflag) ev_setup(eflag, vflag);
  else evflag = vflag_fdotr = 0;

  // M4b v1: device VIRIAL is not yet ported. K2 computes forces/torques on device; the
  // per-contact virial (ev_tally_xyz, pair_ls_dem.cpp:889) is deferred, so pressure/stress
  // under /kk is not valid this milestone (the force/trajectory gate is unaffected). TODO M5+.

  const int ntotal = atom->nlocal + atom->nghost;
  const int nlocal = atom->nlocal;
  const double maxcutsq = maxcut * maxcut;            // cf. pair_ls_dem.cpp:231

  // K1 + K2 read these atom fields on device; K2 atomic-adds f/torque on device.
  atomKK->sync(execution_space, datamask_read);

  // rebuild branch (host): cache binfo + rep-segment CSR, gated on neighbour rebuild
  // (pair_ls_dem.cpp:253-263). build_rep_segments is position-independent; cache_body_info
  // reads host mask/molecule -- both valid after the device sync (DualView keeps host live).
  if (neighbor->lastcall != binfo_lastbuild || (int) binfo.size() < ntotal) {
    cache_body_info(ntotal);                          // pair_ls_dem.cpp:96
    binfo_lastbuild = neighbor->lastcall;
  }
  if (neighbor->lastcall != segs_lastbuild || (int) rep_segs.size() < ntotal) {
    build_rep_segments();                             // pair_ls_dem.cpp:141 (host neigh-list walk)
    upload_segments_to_device();                      // flatten the rep CSR + deep_copy
    segs_lastbuild = neighbor->lastcall;
  }

  // device uploads: grids+coeffs ONCE (constant for the run); bodyLS/atom2body/binfo EVERY
  // step -- bodyLS is forward-comm'd each step and atom2body/binfo change on exchange, so the
  // per-step re-upload fixes the M4a stale-after-exchange gap now that K2 reads them on device.
  if (!constants_uploaded && fix_rigid_small) {
    upload_coeffs_to_device();
    upload_grids_to_device();
    constants_uploaded = true;
  }
  if (fix_rigid_small) upload_bodies_to_device();
  if (!selfcheck_done && fix_rigid_small) { selfcheck_uploads(); selfcheck_done = true; }

  if (nseg > 0) {
    // K1: device rep-indexed winner reduction (one contact per rep; replaces :278-298)
    PairLSDEMK1<DeviceType> f1;
    f1.x = atomKK->k_x.view<DeviceType>();
    f1.d_atom_seg_offset = d_atom_seg_offset; f1.d_cand_offset = d_cand_offset; f1.d_cand = d_cand;
    f1.d_ci = d_contacts_i; f1.d_cj = d_contacts_j; f1.d_ccalc = d_contacts_calc;
    f1.nlocal = nlocal; f1.maxcutsq = maxcutsq;
    Kokkos::parallel_for("PairLSDEM::K1", Kokkos::RangePolicy<DeviceType>(0, ntotal), f1);
    Kokkos::fence();

    // shear-history host -> device (bridge)
    sync_history_to_device();

    // K2: device contact-force kernel (replaces the M3 device->host copy + host force pass)
    PairLSDEMK2<DeviceType> f2;
    f2.x = atomKK->k_x.view<DeviceType>();
    f2.v = atomKK->k_v.view<DeviceType>();
    f2.xcom = atomKK->k_xcom.view<DeviceType>();
    f2.omega = atomKK->k_omega.view<DeviceType>();
    f2.quat = atomKK->k_quat.view<DeviceType>();
    f2.f = atomKK->k_f.view<DeviceType>();
    f2.torque = atomKK->k_torque.view<DeviceType>();
    f2.type = atomKK->k_type.view<DeviceType>();
    f2.d_ci = d_contacts_i; f2.d_cj = d_contacts_j; f2.d_ccalc = d_contacts_calc;
    f2.d_grid_values = d_grid_values; f2.d_grid_min = d_grid_min;
    f2.d_grid_offset = d_grid_offset; f2.d_grid_size = d_grid_size;
    f2.d_body_grid_index = d_body_grid_index;
    f2.d_body_grid_scale = d_body_grid_scale; f2.d_body_grid_stride = d_body_grid_stride;
    f2.d_atom2body = d_atom2body; f2.d_binfo_bID = d_binfo_bID;
    f2.d_binfo_bidx = d_binfo_bidx; f2.d_binfo_area = d_binfo_area;
    f2.d_kn=d_kn; f2.d_kt=d_kt; f2.d_mu=d_mu; f2.d_knp=d_knp;
    f2.d_etan=d_etan; f2.d_etat=d_etat; f2.d_etan1=d_etan1; f2.d_decayn1=d_decayn1;
    f2.d_etat1=d_etat1; f2.d_decayt1=d_decayt1; f2.coeff_stride=coeff_stride;
    f2.d_hist_n=d_hist_n; f2.d_hist_fs=d_hist_fs; f2.d_hist_touch=d_hist_touch;
    f2.d_hist_fn1=d_hist_fn1; f2.d_hist_fs1=d_hist_fs1;
    f2.nlocal=nlocal; f2.dim=domain->dimension; f2.dt=update->dt;
    f2.xprd=domain->xprd; f2.yprd=domain->yprd; f2.zprd=domain->zprd;
    f2.px=domain->xperiodic; f2.py=domain->yperiodic; f2.pz=domain->zperiodic;
    Kokkos::parallel_for("PairLSDEM::K2", Kokkos::RangePolicy<DeviceType>(0, ntotal), f2);
    Kokkos::fence();

    // shear-history device -> host (bridge); no reverse_comm (owner-authoritative writes)
    sync_history_from_device();
  }

  atomKK->modified(execution_space, F_MASK | TORQUE_MASK);   // K2 wrote f/torque on device
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairLSDEMKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairLSDEMKokkos<LMPHostType>;
#endif
}
