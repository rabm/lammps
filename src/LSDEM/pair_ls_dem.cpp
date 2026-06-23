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

#include "pair_ls_dem.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_rigid_ls_dem.h"
#include "fix_rigid_small_ls_dem.h"
#include "force.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "utils.h"
#include "update.h"
#include <cmath>
#include <cstring>
#include <unordered_map>

static constexpr double EPSILON = 1e-12;
// "pair_style ls/dem auto": node-node cutoff = this factor x the (worst-case)
// surface node spacing. The cutoff must exceed the node spacing so a partner
// node stays in range at contact; the factor adds margin for staggered surfaces
// and modest penetration while keeping the cutoff small (few false positives).
static constexpr double LS_DEM_AUTO_CUT_FACTOR = 2.0;

using namespace LAMMPS_NS;
using namespace MathConst;

//TODO:
//  add checks for key overflow
//  add checks to rigid/small/ls/dem nbodies < 2bil (or generalize)
//  should watershed be a 2nd toggle and not a memory style? Such that global also creates (but does not communicates) the data?
//    primarily, is it faster than doing the 2nd nlist loop?
//  create a 2nd set of page files and process nlist whenever built to skip far atoms

/* ---------------------------------------------------------------------- */

PairLSDEM::PairLSDEM(LAMMPS *_lmp) : Pair(_lmp),
  cut(nullptr), decayn1(nullptr), decayt1(nullptr), etan(nullptr),
  etan1(nullptr), etat(nullptr),  etat1(nullptr),kn(nullptr),
  knp(nullptr),kt(nullptr), mu(nullptr), fix_rigid(nullptr),
  fix_rigid_small(nullptr)
{
  writedata = 1;
  single_enable = 0;
  cutoff_auto = 0;
  auto_cut_factor = LS_DEM_AUTO_CUT_FACTOR;
  normal_filter = 0;
  binfo_lastbuild = -1;
  segs_lastbuild = -1;
}

/* ---------------------------------------------------------------------- */

PairLSDEM::~PairLSDEM()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(kn);
    memory->destroy(kt);
    memory->destroy(mu);
    memory->destroy(etan);
    memory->destroy(etat);
    memory->destroy(knp);
    memory->destroy(cut);
    memory->destroy(decayn1);
    memory->destroy(etan1);
    memory->destroy(decayt1);
    memory->destroy(etat1);
  }
}

/* ----------------------------------------------------------------------
   Cache each (local+ghost) atom's body info once per step, so the contact-pass
   loops read it instead of redoing the group test + double-indirection
   body/volume/area lookups for every neighbour pair. Bitwise-identical values.
------------------------------------------------------------------------- */

void PairLSDEM::cache_body_info(int ntotal)
{
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int *mybody_large = nullptr, *mybody_small = nullptr;
  double *grain_vol = nullptr, *node_area = nullptr;
  FixRigidSmallLSDEM::BodyLS *bodyLS = nullptr;
  if (fix_rigid) {
    mybody_large = fix_rigid->get_body_array();
    grain_vol = fix_rigid->get_vol_array();
    node_area = fix_rigid->get_area_array();
  }
  if (fix_rigid_small) {
    mybody_small = fix_rigid_small->get_atom2body_array();
    bodyLS = fix_rigid_small->get_bodyLS_array();
  }

  binfo.resize(ntotal);
  for (int a = 0; a < ntotal; a++) {
    BodyInfo &b = binfo[a];
    if (fix_rigid && (mask[a] & groupbit_large)) {
      int bi = mybody_large[a];
      b.grp = 2; b.bidx = bi; b.bID = bi; b.off = 0;
      b.vol  = (bi >= 0) ? grain_vol[bi] : 0.0;
      b.area = (bi >= 0) ? node_area[bi] : 0.0;
    } else if (fix_rigid_small && (mask[a] & groupbit_small)) {
      int bi = mybody_small[a];
      b.grp = 1; b.bidx = bi; b.bID = (int) molecule[a]; b.off = 1;
      b.vol  = (bi >= 0) ? bodyLS[bi].grid_vol : 0.0;
      b.area = (bi >= 0) ? bodyLS[bi].node_area / bodyLS[bi].natoms : 0.0;
    } else {
      b.grp = 0; b.bidx = -1; b.bID = -1; b.off = 0; b.vol = 0.0; b.area = 0.0;
    }
  }
}

/* ----------------------------------------------------------------------
   Build the per-representative-node CANDIDATE SEGMENTS from the neighbour list.
   Called only when the list is rebuilt. No positions/rsq here: the rep choice and
   partner body are position-independent, so the per-step pass just re-reduces rsq
   over each segment. Candidates are appended in the SAME order the old single sweep
   visited them -> the per-step first-min winner is bitwise-identical. (watershed_flag==0
   only; the watershed path uses get_bin, not these segments.)
------------------------------------------------------------------------- */

void PairLSDEM::build_rep_segments()
{
  const int inum = list->inum, allnum = inum + list->gnum;
  int *ilist = list->ilist, *numneigh = list->numneigh, **firstneigh = list->firstneigh;
  const int ntotal = atom->nlocal + atom->nghost;
  double *special_lj = force->special_lj;
  tagint *tag = atom->tag;
  // item-2c normal-aware filter inputs (only read when normal_filter is on).
  // x and the per-node grain COM xcom are valid for own+ghost here: build runs
  // in compute() (and the /kk rebuild branch) after the fix's pre_force has
  // forward-communicated/synced the grain fields to the host.
  double **x = atom->x;
  double **xcom = atom->xcom;

  if ((int) rep_segs.size() < ntotal) rep_segs.resize(ntotal);
  for (int a = 0; a < ntotal; a++) rep_segs[a].clear();

  for (int ii = 0; ii < allnum; ii++) {
    int i = ilist[ii];
    const BodyInfo &bi = binfo[i];
    if (bi.grp == 0)
      error->one(FLERR, "Atom {} does not belong to a fix rigid ls/dem group", tag[i]);
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      if (special_lj[sbmask(j)] == 0.0) continue;
      j &= NEIGHMASK;
      const BodyInfo &bj = binfo[j];
      if (bj.grp == 0)
        error->one(FLERR, "Atom {} does not belong to a fix rigid ls/dem group", tag[j]);
      // Representative = smaller grain (volume, then body-id tie-break). Same test as
      // the force loop. Record the candidate on the partner body of the rep node.
      int rep, pbody, poff, cand;
      if (bi.vol < bj.vol || (bi.vol == bj.vol && bi.bID < bj.bID)) {
        rep = i; pbody = bj.bID; poff = bj.off; cand = j;
      } else {
        rep = j; pbody = bi.bID; poff = bi.off; cand = i;
      }
      // item-2c: drop the pair if the rep node faces away from the partner grain,
      // or the partner node faces away from the rep grain. Hemisphere test against
      // the grain COMs (a cheap surface-normal proxy). For CONVEX grains the
      // closest-node winner is always front-facing, so this only ever removes
      // non-winners -> bitwise; for non-convex grains it is approximate (the
      // chosen contact node may shift). Opt-in (default off), so all existing
      // benchmarks are unchanged.
      if (normal_filter) {
        const double tx = xcom[cand][0] - xcom[rep][0];   // rep grain -> cand grain
        const double ty = xcom[cand][1] - xcom[rep][1];
        const double tz = xcom[cand][2] - xcom[rep][2];
        const double rox = x[rep][0] - xcom[rep][0];      // rep node outward
        const double roy = x[rep][1] - xcom[rep][1];
        const double roz = x[rep][2] - xcom[rep][2];
        if (rox*tx + roy*ty + roz*tz <= 0.0) continue;    // rep node back-facing
        // NOTE: only the REP node is filtered. Filtering the partner (cand) node
        // by the same hemisphere test is NOT safe -- cand is the partner node used
        // to LOCATE the LS-lookup region (closest partner node to rep), and for a
        // penetrating/edge contact that closest node is not guaranteed to be on
        // the partner's COM-facing hemisphere; filtering it shifts the chosen
        // region and changes the force even for convex grains.
      }
      std::vector<Seg> &segs = rep_segs[rep];
      bool found = false;
      for (Seg &s : segs)
        if (s.pbody == pbody && s.poff == poff) { s.cand.push_back(cand); found = true; break; }
      if (!found) { segs.push_back(Seg{pbody, (char) poff, {}}); segs.back().cand.push_back(cand); }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairLSDEM::compute(int eflag, int vflag)
{
  int i, j, ii, jj, allnum, inum, jnum, itype, jtype, ibody, jbody, ibodyID, jbodyID;
  tagint itag, jtag;
  double xitmp, yitmp, zitmp, xjtmp, yjtmp, zjtmp, delx, dely, delz, dr, evdwl;
  double r, rsq, rinv, factor_lj, u, ivol, jvol, icomx, icomy, icomz, jcomx, jcomy, jcomz;
  int *ilist, *jlist, *numneigh, **firstneigh, calc_force_of_i_on_j, calc_force_of_j_on_i;
  double vxitmp, vyitmp, vzitmp, vxjtmp, vyjtmp, vzjtmp, delvx, delvy, delvz, dot, smooth;
  double iomegax, iomegay, iomegaz, jomegax, jomegay, jomegaz, spin_norm;
  double normal[3], fn_mag, fpair[3], fpair_mag, contact_point[3], lever[3], torque_pair[3];
  double fs_tmp[3], fs_mag, fs_max, fs_mag_trial, k[3], sintheta, costheta, term1[3], term2;
  double tangent[3], shear_incr, v_rel[3], v_rel_t[3], v_rel_n_mag, v_rel_t_mag, v_rel_t_mag_inv;
  double normal_old[3], tangent_old[3], fs_mag_add, areai, areaj;

  // Currently require:
  //   Newton pair off.
  //   Exclude intramolecule interactions
  // In future, remove restrictions

  evdwl = 0.0;
  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  // Node quantities
  double **x = atom->x;
  double **v = atom->v; // This is the half-step, giving O(dt^2) accuracy for damping instead of O(dt).
  double **f = atom->f;
  double **torque = atom->torque;
  double **n = atom->darray[index_ls_dem_n]; // Contact normal (to be update from previous time step)
  double **fs = atom->darray[index_ls_dem_fs]; // Shear component of f (to be update from previous time step)
  int *touch_id = atom->ivector[index_ls_dem_touch_id]; // Grain in contact with this node (to be update from previous time step)
  double *fn1 = atom->dvector[index_ls_dem_fn1]; // Maxwell element force history (magnitude only, normal)
  double *fs1 = atom->dvector[index_ls_dem_fs1]; // Maxwell element force history (magnitude only, shear)

  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double dt = update->dt;
  // Compare squared distances against the squared cutoff so the closest-node
  // search avoids a sqrt for every neighbour pair (ordering is preserved).
  double maxcutsq = maxcut * maxcut;

  // Grain quantities
  double **grain_com = atom->xcom;
  double **grain_omega = atom->omega;

  // Per-atom body info (grain id, volume, node area, etc.) is cached once per step
  // in cache_body_info() below and read from `binfo` in both contact-pass loops.
  int offset_i, offset_j;

  inum = list->inum;
  allnum = inum + list->gnum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // Clear (keeping the allocated capacity) and reserve. The key count scales with
  // the number of interacting body pairs; this is a heuristic upper estimate and
  // only affects performance, never the stored result.
  // Cache per-atom body info; it only changes when the neighbour list is rebuilt
  // (body assignments + ghost set are stable between builds), so refill only then.
  int ntotal = atom->nlocal + atom->nghost;
  if (neighbor->lastcall != binfo_lastbuild || (int) binfo.size() < ntotal) {
    cache_body_info(ntotal);
    binfo_lastbuild = neighbor->lastcall;
  }

  // Rebuild the candidate segments only when the neighbour list was (re)built.
  if (watershed_flag == 0 &&
      (neighbor->lastcall != segs_lastbuild || (int) rep_segs.size() < ntotal)) {
    build_rep_segments();
    segs_lastbuild = neighbor->lastcall;
  }

  // Per-step closest-node reduction over the precomputed candidate segments, FUSED with
  // the flat-contact emit: for each representative node pick the min-rsq partner within the
  // cutoff (first on ties, in stored neighbour-sweep order -> bitwise-identical to the old
  // single sweep) and emit the contact directly. ONE pass over rep_segs -- no separate
  // rep_buckets build + emit pass (that double O(ntotal) traversal regressed fcc2). The emit
  // keeps the LOCAL node as i so process_contact's unconditional f[i]+= / conditional mirror
  // reproduces the newton-off "force to local nodes only" rule: rep local -> {i=r,j=widx,
  // calc=1}; rep ghost + partner local -> {i=widx,j=r,calc=0}; both ghost -> skip.
  // (rep_buckets is now vestigial: only the unreachable non-watershed branch of the
  // watershed-gated loop 2 still references it via rep_winner(); kept sized for safety.)
  if ((int) rep_buckets.size() < ntotal) rep_buckets.resize(ntotal);
  if (watershed_flag == 0) {
    contacts.clear();
    for (int r = 0; r < ntotal; r++) {
      std::vector<Seg> &segs = rep_segs[r];
      if (segs.empty()) continue;
      double xr0 = x[r][0], xr1 = x[r][1], xr2 = x[r][2];
      const bool r_local = (r < nlocal);
      for (Seg &s : segs) {
        int widx = -1;
        double minrsq = 0.0;
        for (int jc : s.cand) {
          delx = xr0 - x[jc][0];
          dely = xr1 - x[jc][1];
          delz = xr2 - x[jc][2];
          rsq = delx * delx + dely * dely + delz * delz;
          if (rsq > maxcutsq) continue;
          if (widx < 0 || rsq < minrsq) { minrsq = rsq; widx = jc; }
        }
        if (widx < 0) continue;
        if (r_local)            contacts.push_back({r, widx, 1});
        else if (widx < nlocal) contacts.push_back({widx, r, 0});
      }
    }
  }

  // NOTE: calc_force_of_j_on_i and calc_force_of_i_on_j now cause branching.
  // The contact model might do the same. May it be worth it to pre-sort the pairs
  // such that it is always i_on_j and the contact models are sorted?
  // Currently, compiler vectorisation is scrambled, which might be particularly
  // bad for any future Kokkos GPU port.

  int tmp_bin;
  double x_local[3];
  saved_bins.resize(ntotal);
  for (auto& my_map : saved_bins)
    my_map.clear();

  if (watershed_flag) {
  // Only loop over local atoms to calculate forces
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xitmp = x[i][0];
    yitmp = x[i][1];
    zitmp = x[i][2];
    vxitmp = v[i][0];
    vyitmp = v[i][1];
    vzitmp = v[i][2];
    itype = type[i];
    { const BodyInfo &bc = binfo[i];   // cached per step (see cache_body_info)
      if (bc.grp == 0)
        error->one(FLERR, "Atom {} does not belong to a fix rigid ls/dem group", tag[i]);
      ibody = bc.bidx; ibodyID = bc.bID; ivol = bc.vol; areai = bc.area; offset_i = bc.off; }
    icomx = grain_com[i][0];
    icomy = grain_com[i][1];
    icomz = grain_com[i][2];
    iomegax = grain_omega[i][0];
    iomegay = grain_omega[i][1];
    iomegaz = grain_omega[i][2];

    itag = tag[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];

      if (factor_lj == 0) continue;

      j &= NEIGHMASK;
      { const BodyInfo &bc = binfo[j];   // cached per step (see cache_body_info)
        if (bc.grp == 0)
          error->one(FLERR, "Atom {} does not belong to a fix rigid ls/dem group", tag[j]);
        jbody = bc.bidx; jbodyID = bc.bID; jvol = bc.vol; areaj = bc.area; offset_j = bc.off; }
      jtag = tag[j];
      jtype = type[j];

      // Figure out whether to use the nodes of grain i or j.
      // We use the nodes on the smaller grain since this will
      // be more accurate. If the volumes are tied, use the grain ID.
      // Only calculate force between closest pair of node and grid.

      calc_force_of_j_on_i = 0;
      calc_force_of_i_on_j = 0;

      if (watershed_flag) {
        // Use the nodes of the smallest grain.
        if (ivol < jvol || (ivol == jvol && ibodyID < jbodyID)) {
          calc_force_of_i_on_j = 1;
        } else {
          calc_force_of_j_on_i = 1;
        }

        if (calc_force_of_i_on_j) {
          if (saved_bins[i].find(jbodyID) == saved_bins[i].end()) {
            if (fix_rigid && (mask[j] & groupbit_large))
              tmp_bin = fix_rigid->get_bin(i, j, x_local);
            //else if (fix_rigid_small && (mask[j] & groupbit_small))
              // TBD tmp_bin_i = static_cast<int>(fix_rigid_small);
            else
              error->one(FLERR, "Atom {} does not belong to a fix rigid ls/dem group", tag[j]);
            saved_bins[i][jbodyID] = std::make_tuple(tmp_bin, x_local[0], x_local[1], x_local[2]);
          } else {
            tmp_bin = std::get<0>(saved_bins[i][jbodyID]);
            x_local[0] = std::get<1>(saved_bins[i][jbodyID]);
            x_local[1] = std::get<2>(saved_bins[i][jbodyID]);
            x_local[2] = std::get<3>(saved_bins[i][jbodyID]);
          }
          if (fix_rigid && (mask[j] & groupbit_large))
            if (fix_rigid->check_watershed_bin(tmp_bin, j) == 0) continue;
        } else {
          if (saved_bins[j].find(ibodyID) == saved_bins[j].end()) {
            if (fix_rigid && (mask[i] & groupbit_large))
              tmp_bin = fix_rigid->get_bin(j, i, x_local);
            //else if (fix_rigid_small && (mask[i] & groupbit_small))
              // TBD tmp_bin_j = static_cast<int>(fix_rigid_small);
            else
              error->one(FLERR, "Atom {} does not belong to a fix rigid ls/dem group", tag[i]);
            saved_bins[j][ibodyID] = std::make_tuple(tmp_bin, x_local[0], x_local[1], x_local[2]);
          } else {
            tmp_bin = std::get<0>(saved_bins[j][ibodyID]);
            x_local[0] = std::get<1>(saved_bins[j][ibodyID]);
            x_local[1] = std::get<2>(saved_bins[j][ibodyID]);
            x_local[2] = std::get<3>(saved_bins[j][ibodyID]);
          }
          if (fix_rigid && (mask[i] & groupbit_large))
            if (fix_rigid->check_watershed_bin(tmp_bin, i) == 0) continue;
        }
      } else {

        // Use the nodes of the smallest grain.
        if (ivol < jvol || (ivol == jvol && ibodyID < jbodyID)) {
          // Grain i is smaller, use nodes of i and level set of j.
          if (rep_winner(rep_buckets[i], jbodyID, offset_j) == (int) jtag)
            calc_force_of_i_on_j = 1;
        } else {
          // Grain j is smaller, use nodes of j and level set of i.
          if (rep_winner(rep_buckets[j], ibodyID, offset_i) == (int) itag)
            calc_force_of_j_on_i = 1;
        }

        tmp_bin = -1;
        x_local[0] = 0.0;
        x_local[1] = 0.0;
        x_local[2] = 0.0;
      }

      // If no forces are calculated
      if (calc_force_of_i_on_j + calc_force_of_j_on_i == 0) continue;

      process_contact(i, j, calc_force_of_i_on_j, tmp_bin, x_local);
    }
  }
  } else {
    // Non-watershed: drive contact pass from pre-built flat contact list.
    // No second firstneigh re-walk; emit loop above built (i,j,calc) keyed to local nodes.
    double xl0[3] = {0.0, 0.0, 0.0};
    for (Contact &ct : contacts) process_contact(ct.i, ct.j, ct.calc, -1, xl0);
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   Evaluate the LS-DEM contact model for ONE arbitration-winning node-grain pair
   and apply the force/torque to grain i (and, newton off, to grain j when local).
   Lifted verbatim from the in-line contact pass so a single driver can call it
   once per winner; behaviour is bitwise-identical. calc_force_of_i_on_j picks
   which grain owns the representative node; tmp_bin/x_local carry the watershed
   bin (tmp_bin = -1 on the non-watershed path).
------------------------------------------------------------------------- */

void PairLSDEM::process_contact(int i, int j, int calc_force_of_i_on_j,
                                int tmp_bin, double *x_local)
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **torque = atom->torque;
  double **n = atom->darray[index_ls_dem_n];
  double **fs = atom->darray[index_ls_dem_fs];
  int *touch_id = atom->ivector[index_ls_dem_touch_id];
  double *fn1 = atom->dvector[index_ls_dem_fn1];
  double *fs1 = atom->dvector[index_ls_dem_fs1];
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  double **grain_com = atom->xcom;
  double **grain_omega = atom->omega;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  double dt = update->dt;
  double evdwl = 0.0;

  double xitmp, yitmp, zitmp, xjtmp, yjtmp, zjtmp, u;
  double vxitmp, vyitmp, vzitmp, vxjtmp, vyjtmp, vzjtmp;
  double icomx, icomy, icomz, jcomx, jcomy, jcomz;
  double iomegax, iomegay, iomegaz, jomegax, jomegay, jomegaz;
  double normal[3], fn_mag, fpair[3], contact_point[3], lever[3], torque_pair[3];
  double fs_tmp[3], fs_mag, fs_max, fs_mag_trial, k[3], sintheta, costheta, term1[3], term2;
  double tangent[3], shear_incr, v_rel[3], v_rel_t[3], v_rel_n_mag, v_rel_t_mag, v_rel_t_mag_inv;
  double normal_old[3], tangent_old[3], fs_mag_add, spin_norm, areai, areaj;
  int itype, jtype, ibody, jbody, ibodyID, jbodyID;

  itype = type[i];
  xitmp = x[i][0]; yitmp = x[i][1]; zitmp = x[i][2];
  vxitmp = v[i][0]; vyitmp = v[i][1]; vzitmp = v[i][2];
  icomx = grain_com[i][0]; icomy = grain_com[i][1]; icomz = grain_com[i][2];
  iomegax = grain_omega[i][0]; iomegay = grain_omega[i][1]; iomegaz = grain_omega[i][2];
  { const BodyInfo &bc = binfo[i]; ibody = bc.bidx; ibodyID = bc.bID; areai = bc.area; }
  jtype = type[j];
  { const BodyInfo &bc = binfo[j]; jbody = bc.bidx; jbodyID = bc.bID; areaj = bc.area; }

      // ghosts may not find bodytag if comm distance too small
      if (ibody < 0)
        error->one(FLERR, "Atom {} cannot find atom that owns body, consider increasing the communication cutoff", tag[i]);
      if (jbody < 0)
        error->one(FLERR, "Atom {} cannot find atom that owns body, consider increasing the communication cutoff", tag[j]);

      // Evaluate the level set. The representative node (rep, on the smaller grain) is
      // tested against the partner grain's (par) level set. Resolving rep/par by a
      // branchless select unifies the two calc_force_of_i_on_j call-sites into ONE call,
      // so a future GPU contact kernel does not diverge here (bitwise on CPU: the same
      // call with the same arguments). The level set is negative inside the particle, so
      // the sign is swapped to get the overlap distance. Force/direction go i -> j.
      const int rep = calc_force_of_i_on_j ? i : j; // node on the smaller grain
      const int par = calc_force_of_i_on_j ? j : i; // partner grain providing the level set
      if (fix_rigid && (mask[par] & groupbit_large))
        u = - fix_rigid->get_ls_value(rep, par, tmp_bin, normal, x_local);
      else if (fix_rigid_small && (mask[par] & groupbit_small))
        u = - fix_rigid_small->get_ls_value(rep, par, normal);
      else
        error->one(FLERR, "Atom {} does not belong to a fix rigid ls/dem group", tag[rep]);

      // Resolve the direction-dependent quantities once, so the rest of the contact
      // path is free of calc_force_of_i_on_j branches. We always use the node on the
      // chosen grain (ni), its stored history (fs_ni, n_ni, fn1[ni], fs1[ni]) and node
      // area (narea), and the body it is touching (partner_body). The stored shear and
      // normal vectors are kept in the node's own i->j frame, so when we use node j we
      // flip their sign (hsign = -1) on both read and write.
      const int ni = rep;   // representative node (== calc_force_of_i_on_j ? i : j)
      const double hsign = calc_force_of_i_on_j ? 1.0 : -1.0;
      const double narea = calc_force_of_i_on_j ? areai : areaj;
      const int partner_body = calc_force_of_i_on_j ? jbodyID : ibodyID;
      double *fs_ni = fs[ni];
      double *n_ni = n[ni];

      // No adhesion, cohesion, or ranged forces.
      if (u <= 0) {
        // Reset shear force if no contact
        // If i and j are not a shear-interacting pair, it will skip the reset
        if (touch_id[ni] == partner_body) {
          touch_id[ni] = -1;
          fs_ni[0] = 0.0;
          fs_ni[1] = 0.0;
          fs_ni[2] = 0.0;
          n_ni[0] = 0.0;
          n_ni[1] = 0.0;
          n_ni[2] = 0.0;
        }
        // Skip force calculation
        return;
      } else { // Physical contact!!
        // These per-neighbour quantities are only needed once contact is
        // confirmed, so they are loaded here instead of for every neighbour pair.
        xjtmp = x[j][0];
        yjtmp = x[j][1];
        zjtmp = x[j][2];
        vxjtmp = v[j][0];
        vyjtmp = v[j][1];
        vzjtmp = v[j][2];
        jcomx = grain_com[j][0];
        jcomy = grain_com[j][1];
        jcomz = grain_com[j][2];
        jomegax = grain_omega[j][0];
        jomegay = grain_omega[j][1];
        jomegaz = grain_omega[j][2];

        // Compute contact point, correct normal if needed.
        // The normal returned for node i points away from j, so it is negated to point
        // i->j; for node j it already points i->j. After this, normal points i->j in
        // both cases, and the contact point uses the chosen node's position.
        // Branchless: negate for the i-branch (normal points away from j -> flip to i->j),
        // leave for the j-branch (already i->j). Multiply by +-1.0 is exact (bitwise).
        const double normsign = calc_force_of_i_on_j ? -1.0 : 1.0;
        normal[0] *= normsign; normal[1] *= normsign; normal[2] *= normsign;
        const double npos0 = calc_force_of_i_on_j ? xitmp : xjtmp;
        const double npos1 = calc_force_of_i_on_j ? yitmp : yjtmp;
        const double npos2 = calc_force_of_i_on_j ? zitmp : zjtmp;
        // The contact point is the representative node displaced HALF the overlap toward
        // the partner's surface. With normal in the i->j frame that direction is
        // rep->partner for the i-branch (hsign=+1) but partner->rep for the j-branch
        // (hsign=-1), so the offset must carry hsign. Without it, a contact reached via
        // the j-branch (which only happens under MPI domain decomposition, when local
        // reindexing makes the representative the neighbour rather than the loop atom)
        // placed the contact point on the WRONG side of the node -> a different lever ->
        // a different body torque than the serial i-branch result, breaking serial==MPI
        // for the shear/friction torque (the normal torque ~cancels for near-head-on hits,
        // which is why this only showed up with friction on).
        contact_point[0] = npos0 - 0.5 * u * hsign * normal[0];
        contact_point[1] = npos1 - 0.5 * u * hsign * normal[1];
        contact_point[2] = npos2 - 0.5 * u * hsign * normal[2];
      }

      // Hoist the pairwise coefficients (constant for this i-j type pair) into
      // locals. They are read several times below, so this avoids repeated
      // two-level array indexing in the contact path.
      const double knij = kn[itype][jtype];
      const double ktij = kt[itype][jtype];
      const double muij = mu[itype][jtype];
      const double knpij = knp[itype][jtype];
      const double etanij = etan[itype][jtype];
      const double etatij = etat[itype][jtype];
      const double etan1ij = etan1[itype][jtype];
      const double decayn1ij = decayn1[itype][jtype];
      const double etat1ij = etat1[itype][jtype];
      const double decayt1ij = decayt1[itype][jtype];

      ///////////////////
      // Normal stress //
      ///////////////////

      // Elastic spring
      // With positive penetration distance u
      if (fabs(knpij) < EPSILON) {
        fn_mag = knij * u;
      } else {
        fn_mag = knij * pow(u, knpij);
      }

      // Relative velocity at the grain surface at the half step t + 0.5*dt.
      // Note: The velocity at the node due to an angular velocity of the grain around its
      // centre of mass is already included, so the difference of linear velocities suffices.
      v_rel[0] = vxitmp - vxjtmp;
      v_rel[1] = vyitmp - vyjtmp;
      v_rel[2] = vzitmp - vzjtmp;

      // Relative velocity in normal direction with sign (positive for approach)
      v_rel_n_mag = MathExtra::dot3(v_rel, normal);

      // Viscous damping or dashpot (parallel)
      if (etanij > 0.0) {
        fn_mag += etanij * v_rel_n_mag;
      }

      // Maxwell arm (1st, parallel, only repulsive)
      if (etan1ij > 0.0) { // preprocessing guarantees that decayn1 > 0 if etan1 > 0
        fn1[ni] = decayn1ij * fn1[ni] + etan1ij * (1.0 - decayn1ij) * v_rel_n_mag;
        fn_mag += fn1[ni];
        // Maxwell arm (2nd)
        //fn2_mag[i] = decayn2[itype][jtype] * fh2_mag[i] + etan2[itype][jtype] * (1-decayn2[itype][jtype]) * MAX(v_rel_n_mag, 0.0);
        //fn_mag += fn2_mag[i]
      }

      // Only repulsive, do not allow tensile force without explicit bonding / cohesion
      fn_mag = MAX(fn_mag, 0.0);

      // The pair force vector should point j->i because of repulsion.
      // With normal n (i->j), we have: F_{j on i} = f(ls_value) = - fn_mag * n.
      fpair[0] = - fn_mag * normal[0];
      fpair[1] = - fn_mag * normal[1];
      fpair[2] = - fn_mag * normal[2];

      ////////////////////
      // Tangent stress //
      ////////////////////

      // Tangent force only exists if mu > 0 and kt > 0
      //if ( (kt[itype][jtype] > 0) && (mu[itype][jtype] > 0) ){
      // DvdH: Grains without friction are silly, so I took out this check.

      // Check if the pair is valid for shear history calculation
      // Initialise if no contact
      if (touch_id[ni] == -1) {
        touch_id[ni] = partner_body;
      }
      if (touch_id[ni] != partner_body) {
        if (comm->me == 0) {
          tagint node_tag = calc_force_of_i_on_j ? tag[i] : tag[j];
          int node_body = calc_force_of_i_on_j ? ibodyID : jbodyID;
          tagint partner_tag = calc_force_of_i_on_j ? tag[j] : tag[i];
          error->warning(FLERR, "Shear history of node {} on grain {} penetrating node {} on {} cannot be computed at step {}",
            node_tag, node_body, partner_tag, partner_body, update->ntimestep);
        }
      }

      // Get old elastic shear stress and old node normal.
      // The stored vectors are in the node's i->j frame; flip sign when using node j.
      fs_tmp[0] = hsign * fs_ni[0];
      fs_tmp[1] = hsign * fs_ni[1];
      fs_tmp[2] = hsign * fs_ni[2];
      normal_old[0] = hsign * n_ni[0];
      normal_old[1] = hsign * n_ni[1];
      normal_old[2] = hsign * n_ni[2];

      // Adjust fs_tmp to account for rotation of the contact normal and plane.
      if( MathExtra::lensq3(normal_old) > 0.0 ) { // len3 > 0, but without the sqrt
        // Account for tilt. This is an exact correction over rotation of the normal
        // from the previous to the current time step.
        MathExtra::cross3(normal_old, normal, k); // n_old x n_new

        // Account for spin. This is an approximation using the half-step angular
        // velocities. We rotate about the OLD normal so that, together with the
        // tilt, the additive reconstruction matches the spin-then-tilt rotation.
        // spin_norm = dt*(omega_avg . n_old) = dt*|omega_sp| (signed).
        spin_norm = 0.5*dt*(
          (iomegax + jomegax)*normal_old[0] +
          (iomegay + jomegay)*normal_old[1] +
          (iomegaz + jomegaz)*normal_old[2]); // 0.5*dt*(omegai+omegaj) \dot n_old

        // First Baker-Campbell-Hausdorff corrective term for the non-commutativity
        // of the spin and tilt rotations: -0.5*|omega_sp|*dt*(n_old x (n_old x n_new)).
        // The non-commutativity error dominates and only manifests itself in the tilt.
        MathExtra::cross3(normal_old, k, term1); // n_old x (n_old x n_new)
        double bch_coef = -0.5*fabs(spin_norm);
        k[0] += spin_norm*normal_old[0] + bch_coef*term1[0];
        k[1] += spin_norm*normal_old[1] + bch_coef*term1[1];
        k[2] += spin_norm*normal_old[2] + bch_coef*term1[2];

        // Applying the rotation. UNNORMALISED Rodrigues (Chareyre): k already carries
        // axis*sin(theta), so the cross term needs no sin(theta) factor and the axis needs
        // no normalisation -- saves a sqrt (sin) + a reciprocal + the 3 normalise mults.
        // Using (1-cos)/sin^2 == 1/(1+cos) also avoids small-angle cancellation:
        //   fs_rot = fs*cos + (k x fs) + k*(k.fs)/(1+cos),   cos = sqrt(1 - |k|^2).
        // Algebraically identical to the normalised form (differs only at FP rounding).
        double sinsq = MathExtra::lensq3(k); // sin^2(theta) = |k|^2
        if (sinsq > EPSILON * EPSILON) { // Don't apply rotation if magnitude is tiny
          costheta = sqrt(MAX(1.0 - sinsq, 0.0));
          MathExtra::cross3(k, fs_tmp, term1);                   // k x fs (already scaled by sin)
          term2 = MathExtra::dot3(k, fs_tmp) / (1.0 + costheta); // (k.fs)/(1+cos), 1+cos >= 1
          fs_tmp[0] = fs_tmp[0] * costheta + term1[0] + k[0] * term2;
          fs_tmp[1] = fs_tmp[1] * costheta + term1[1] + k[1] * term2;
          fs_tmp[2] = fs_tmp[2] * costheta + term1[2] + k[2] * term2;
        }
      }

      // Relative velocity in tangential direction
      v_rel_t[0] = v_rel[0] - v_rel_n_mag * normal[0];
      v_rel_t[1] = v_rel[1] - v_rel_n_mag * normal[1];
      v_rel_t[2] = v_rel[2] - v_rel_n_mag * normal[2];
      v_rel_t_mag = MathExtra::len3(v_rel_t);
      // We note that this way of calculating the shear velocity is an approximation.
      // The error lies mainly in the fact that the velocity difference was computed
      // between the surface nodes, meaning that there is a small arm length that has
      // not been accounted for. As a consequence, this breaks objectivity since v_rel_t
      // will increase with the rigid body rotation omega_b. The extra arm length does
      // not unjustifiably increase v_rel_t in the absence of rigid body motion, since
      // the surface can indeed be said to move at omega x R.

      // Tangent normal
      if (v_rel_t_mag > EPSILON) {
        // v_rel_t_mag > EPSILON (1e-12) > 0 here, so the reciprocal is always well-defined.
        v_rel_t_mag_inv = 1.0 / v_rel_t_mag;
        tangent[0] = v_rel_t[0] * v_rel_t_mag_inv;
        tangent[1] = v_rel_t[1] * v_rel_t_mag_inv;
        tangent[2] = v_rel_t[2] * v_rel_t_mag_inv;
      } else {
        // Backup: Use old shear force direction as tangent.
        double norm = MathExtra::len3(fs_tmp);
        if (norm != 0) {
          v_rel_t_mag_inv = 1.0 / norm;
        } else {
          v_rel_t_mag_inv = 0.0;
        }
        tangent[0] = fs_tmp[0] * v_rel_t_mag_inv;
        tangent[1] = fs_tmp[1] * v_rel_t_mag_inv;
        tangent[2] = fs_tmp[2] * v_rel_t_mag_inv;
      }

      // Elastic spring shear stress increment
      shear_incr = ktij * v_rel_t_mag * dt;

      // New elastic force
      fs_tmp[0] -= shear_incr * tangent[0];
      fs_tmp[1] -= shear_incr * tangent[1];
      fs_tmp[2] -= shear_incr * tangent[2];
      fs_mag_trial = MathExtra::len3(fs_tmp);

      // Coulomb limit
      fs_max = muij * fn_mag;

      // Perfectly plastic Coulomb friction criterion
      fs_mag = std::min(fs_max, fs_mag_trial);

      // Final shear or tangential stress
      if (fs_mag_trial > EPSILON){
        fs_tmp[0] = fs_mag * (fs_tmp[0] / fs_mag_trial);
        fs_tmp[1] = fs_mag * (fs_tmp[1] / fs_mag_trial);
        fs_tmp[2] = fs_mag * (fs_tmp[2] / fs_mag_trial);
      }

      // Update saved elastic shear stress and normal (back into the node's i->j frame)
      fs_ni[0] = hsign * fs_tmp[0];
      fs_ni[1] = hsign * fs_tmp[1];
      fs_ni[2] = hsign * fs_tmp[2];
      n_ni[0] = hsign * normal[0];
      n_ni[1] = hsign * normal[1];
      n_ni[2] = hsign * normal[2];

      // Placeholder for viscous and viscoelastic stress components
      fs_mag_add = 0.0;

      // Viscous damping or dashpot (parallel, allowed in any direction)
      if(etatij > 0) {
        fs_mag_add += etatij * v_rel_t_mag;
      }

      // Maxwell arm (1st, parallel, only repulsive)
      if (etan1ij > 0.0) { // preprocessing guarantees that decayt1 > 0 if etat1 > 0
        // Get the old tangent vector
        double norm = MathExtra::len3(fs_tmp);
        if (norm != 0) {
          v_rel_t_mag_inv = 1.0 / norm;
        } else {
          v_rel_t_mag_inv = 0.0;
        }
        tangent_old[0] = fs_tmp[0] * v_rel_t_mag_inv;
        tangent_old[1] = fs_tmp[1] * v_rel_t_mag_inv;
        tangent_old[2] = fs_tmp[2] * v_rel_t_mag_inv;
        // The dot(t_old,t) part accounts for the in-plane rotation of the tangent force.
        // When the shear direction reverses, it correctly preserves the direction of the old force.
        // However, when rotating towards the orthogonal direction, we inevitably lose some of the force.
        // We could track the full vector, but it would cost more memory (and accessing time)
        fs1[ni] += decayt1ij * fs1[ni] * MathExtra::dot3(tangent_old, tangent)
          + etat1ij * (1 - decayt1ij) * v_rel_t_mag; // v_rel_t_mag is always positive
        fs_mag_add += fs1[ni];
        // Maxwell arm (2nd)
        // fs2_mag[i] = exps2*fs2_mag[i] + etat2[itype][jtype]*(1-exps2)*v_rel_t_mag;
        // fs_mag -= fs2_mag[i]
      }

      // Add potential viscous and viscoelastic components to shear stress (repulsive again)
      fs_tmp[0] -= fs_mag_add * tangent[0];
      fs_tmp[1] -= fs_mag_add * tangent[1];
      fs_tmp[2] -= fs_mag_add * tangent[2];
      // Update trial shear stress magnitude
      fs_mag_trial = MathExtra::len3(fs_tmp);

      // Re-apply plastic Coulomb friction criterion
      fs_mag = std::min(fs_max, fs_mag_trial);

      if (fs_mag > 0 && fs_mag_trial != 0) {
        // Final shear or tangential stress
        fs_tmp[0] = fs_mag * (fs_tmp[0] / fs_mag_trial);
        fs_tmp[1] = fs_mag * (fs_tmp[1] / fs_mag_trial);
        fs_tmp[2] = fs_mag * (fs_tmp[2] / fs_mag_trial);
        // Add shear stress to the pair force stress
        fpair[0] += fs_tmp[0];
        fpair[1] += fs_tmp[1];
        fpair[2] += fs_tmp[2];
      }

      //} // Check if kt > 0 and mu > 0

      //////////////////////////////
      // Total forces and torques //
      //////////////////////////////

      // Multiply by node area to make the force independent of discretisation (fpair was a stress)
      fpair[0] *= narea;
      fpair[1] *= narea;
      fpair[2] *= narea;

      // Force on grain i
      f[i][0] += fpair[0];
      f[i][1] += fpair[1];
      f[i][2] += fpair[2];

#ifndef NDEBUG
      if (std::isnan(fpair[0]) || std::isnan(fpair[1]) || std::isnan(fpair[2]))
        error->one(FLERR, "Bad force calculated between atoms {} {} on bodies {} {} with overlap {} and normal {} {} {}\n", tag[i], tag[j], ibody, jbody, u, normal[0], normal[1], normal[2]);
#endif

      // Lever arm on grain i
      lever[0] = contact_point[0] - icomx;
      lever[1] = contact_point[1] - icomy;
      lever[2] = contact_point[2] - icomz;
      // Account for PBCs
      domain->minimum_image(FLERR, lever);

      // Compute torque
      MathExtra::cross3(lever, fpair, torque_pair);

      // Apply torques on grain i
      torque[i][0] += torque_pair[0];
      torque[i][1] += torque_pair[1];
      torque[i][2] += torque_pair[2];

#ifndef NDEBUG
      if (std::isnan(torque_pair[0]) || std::isnan(torque_pair[1]) || std::isnan(torque_pair[2]))
        error->one(FLERR, "Bad torque calculated between atoms {} {} on bodies {} {} with overlap {} and normal {} {} {}\n", tag[i], tag[j], ibody, jbody, u, normal[0], normal[1], normal[2]);
#endif


      // Mirror forces and torques on grain j
      if (newton_pair || j < nlocal) { // Need to check this again if we end up enabling newton_pair
        MathExtra::negate3(fpair); // Sign swap
        f[j][0] += fpair[0];
        f[j][1] += fpair[1];
        f[j][2] += fpair[2];

        lever[0] = contact_point[0] - jcomx;
        lever[1] = contact_point[1] - jcomy;
        lever[2] = contact_point[2] - jcomz;
        domain->minimum_image(FLERR, lever);

        MathExtra::cross3(lever, fpair, torque_pair);

        torque[j][0] += torque_pair[0];
        torque[j][1] += torque_pair[1];
        torque[j][2] += torque_pair[2];
      }

      if (evflag) ev_tally_xyz(i, j, nlocal, force->newton_pair, evdwl, 0.0, fpair[0], fpair[1], fpair[2], normal[0] * u, normal[1] * u, normal[2] * u);
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLSDEM::allocate()
{
  allocated = 1;
  const int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  for (int i = 1; i < np1; i++)
    for (int j = i; j < np1; j++) setflag[i][j] = 0;

  memory->create(cutsq, np1, np1, "pair:cutsq");

  memory->create(kn, np1, np1, "pair:kn");
  memory->create(kt, np1, np1, "pair:kt");
  memory->create(mu, np1, np1, "pair:mu");
  memory->create(etan, np1, np1, "pair:etan");
  memory->create(etat, np1, np1, "pair:etat");
  memory->create(knp, np1, np1, "pair:knp");
  memory->create(cut, np1, np1, "pair:cut");
  memory->create(decayn1, np1, np1, "pair:decayn1");
  memory->create(etan1, np1, np1, "pair:etan1");
  memory->create(decayt1, np1, np1, "pair:decayt1");
  memory->create(etat1, np1, np1, "pair:etat1");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLSDEM::settings(int narg, char ** arg)
{
  if (narg < 1)
    error->all(FLERR, "Illegal pair_style command");

  normal_filter = 0;
  int iarg;

  // "auto" estimates the node-node cutoff from the grain node spacing in
  // init_style(); otherwise the cutoff is the given number.
  if (strcmp(arg[0], "auto") == 0) {
    cutoff_auto = 1;
    maxcut = -1.0;    // sentinel, resolved in init_style()
    iarg = 1;
    // optional node-spacing multiplier directly after "auto" (default
    // LS_DEM_AUTO_CUT_FACTOR). A smaller factor tightens the auto cutoff (fewer
    // neighbours, faster) but leaves less margin for staggered surfaces +
    // penetration; it must stay above 1 (the worst-case node spacing) or
    // contacts will be missed. A trailing keyword (e.g. nfilter) is NOT a factor.
    if (iarg < narg && strcmp(arg[iarg], "nfilter") != 0) {
      auto_cut_factor = utils::numeric(FLERR, arg[iarg], false, lmp);
      if (auto_cut_factor <= 0.0)
        error->all(FLERR, "pair ls/dem auto factor {} must be positive", auto_cut_factor);
      iarg++;
    }
  } else {
    cutoff_auto = 0;
    maxcut = utils::numeric(FLERR, arg[0], false, lmp);
    iarg = 1;
  }

  // optional trailing keywords
  while (iarg < narg) {
    if (strcmp(arg[iarg], "nfilter") == 0) {
      // item-2c: enable the normal-aware candidate filter in build_rep_segments.
      // Opt-in + shape-gated: it drops node pairs whose rep/partner node faces
      // away from the other grain's COM. EXACT (bitwise) for convex grains -- the
      // closest-node winner is always front-facing -- and approximate (may shift
      // the chosen contact node) for non-convex grains. Default OFF.
      normal_filter = 1;
      iarg++;
    } else {
      error->all(FLERR, "Illegal pair_style ls/dem keyword: {}", arg[iarg]);
    }
  }

  if (force->newton_pair)
    error->all(FLERR, "Temporarily do not support newton pair on with LS/DEM");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLSDEM::coeff(int narg, char **arg)
{
  if (narg != 9 && narg != 11 && narg != 13)
    error->all(FLERR, "Incorrect number of args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double kn_0 = utils::numeric(FLERR, arg[2], false, lmp);
  double kt_0 = utils::numeric(FLERR, arg[3], false, lmp);
  double mu_0 = utils::numeric(FLERR, arg[4], false, lmp);
  double etan_0 = utils::numeric(FLERR, arg[5], false, lmp);
  double etat_0 = utils::numeric(FLERR, arg[6], false, lmp);
  double knp_0 = utils::numeric(FLERR, arg[7], false, lmp);
  double cut_one = utils::numeric(FLERR, arg[8], false, lmp);

  double kn_1, kt_1, etan_1, etat_1;
  if (narg > 9) {
    kn_1 = utils::numeric(FLERR, arg[9], false, lmp);
    etan_1 = utils::numeric(FLERR, arg[10], false, lmp);
  }

  if (narg > 11) {
    kt_1 = utils::numeric(FLERR, arg[11], false, lmp);
    etat_1 = utils::numeric(FLERR, arg[12], false, lmp);
  }

  if (kn_0 < 0.0) error->all(FLERR, "The normal stiffness {} must be postitive", kn_0);
  if (kt_0 < 0.0) error->all(FLERR, "The tangential stiffness {} must be postitive", kt_0);
  if (mu_0 < 0.0) error->all(FLERR, "The friction coefficient {} must be postitive", mu_0);
  if (etan_0 < 0.0) error->all(FLERR, "The normal damping {} must be postitive", etan_0);
  if (etat_0 < 0.0) error->all(FLERR, "The tangential damping {} must be postitive", etat_0);

  // Values of knp_0 can be both positive and negative.
  if (narg > 9) {
    if (kn_1 < 0.0) error->all(FLERR, "The extra maxwell normal stiffness {} must be postitive", kn_1);
    if (etan_1 < 0.0) error->all(FLERR, "The extra maxwell normal damping {} must be postitive", etan_1);
    // If active, neither k or eta in a Maxwell arm are allowed to be zero. Check if both zero or both positive.
    if ((kn_1 == 0.0) || (etan_1 == 0.0)) {
      error->all(FLERR, "Maxwell arm requires normal stiffness k and damping eta to both be positive");
    }
  }

  if (narg > 11) {
    if (kt_1 < 0.0) error->all(FLERR, "The extra maxwell tangential stiffness {} must be postitive", kt_1);
    if (etat_1 < 0.0) error->all(FLERR, "The extra maxwell tangential damping {} must be postitive", etat_1);
    if ((kt_1 == 0.0) || (etat_1 == 0.0)) {
      error->all(FLERR, "Maxwell arm requires tangential stiffness k and damping eta to both be positive");
    }
  }

  // Determine contact model type, this pre-computed flag helps evaluate branching
  // conditions more economically
  // int mode = 0; // Default mode: purely elastic
  // if ((etan_0 > 0.0) || (etat_0 > 0.0)){
  //   mode = 1; // Elastic + parallel viscous damping
  // }
  // if ((kn_1 > 0.0) || (etan_1 > 0.0) || (kt_1 > 0.0) || (etat_1 > 0.0)){
  //   mode = 2; // Generalised Maxwell solid
  // }
  // Neither k or eta in a Maxwell arm are allowed to be zero.

  // Need a check for dt staying constant. If it changes, decayn1 and decayt1 need to be recomputed.

  double dt = update->dt;
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      kn[i][j] = kn_0;
      kt[i][j] = kt_0;
      mu[i][j] = mu_0;
      etan[i][j] = etan_0;
      etat[i][j] = etat_0;
      knp[i][j] = knp_0;
      cut[i][j] = cut_one;

      if (narg > 9) {
        decayn1[i][j] = exp(-dt * kn_1 / etan_1);
        etan1[i][j] = etan_1;
      } else {
        decayn1[i][j] = 0.0;
        etan1[i][j] = 0.0;
      }
      if (narg > 11) {
        decayt1[i][j] = exp(-dt * kt_1 / etat_1);
        etat1[i][j] = etat_1;
      } else {
        decayt1[i][j] = 0.0;
        etat1[i][j] = 0.0;
      }

      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLSDEM::init_style()
{
  if (comm->ghost_velocity == 0)
    error->all(FLERR, "Pair ls/dem requires ghost atoms store velocity");

  if (!atom->xcom_flag || !atom->omega_flag || !atom->quat_flag  || !atom->grid_index_flag)
    error->all(FLERR, "Pair ls/dem requires atom style ls/dem");

  // Resolve an "auto" node-node cutoff from the LS-DEM molecule templates. This
  // runs before Pair::init() calls init_one() -> cutsq, so the estimate drives
  // the neighbour-list cutoff (and the fix reads pair->maxcut afterwards).
  if (cutoff_auto) {
    double char_spacing = -1.0;
    for (int im = 0; im < atom->nmolecule; im++) {
      Molecule *onemol = atom->molecules[im];
      if (!onemol->lsdemflag) continue;
      int nn = onemol->natoms;
      double **mx = onemol->x;
      // worst-case (largest) nearest-neighbour gap among this template's nodes
      double tmpl_max_nn = 0.0;
      for (int a = 0; a < nn; a++) {
        double best = -1.0;
        for (int b = 0; b < nn; b++) {
          if (b == a) continue;
          double dx = mx[a][0] - mx[b][0];
          double dy = mx[a][1] - mx[b][1];
          double dz = mx[a][2] - mx[b][2];
          double r2 = dx * dx + dy * dy + dz * dz;
          if (best < 0.0 || r2 < best) best = r2;
        }
        if (best > 0.0) tmpl_max_nn = MAX(tmpl_max_nn, sqrt(best));
      }
      char_spacing = MAX(char_spacing, tmpl_max_nn);
    }
    if (char_spacing <= 0.0)
      error->all(FLERR, "pair ls/dem cutoff 'auto' requires LS-DEM molecule templates "
                 "(create_atoms ... mol); set the cutoff explicitly instead");

    double auto_cut = auto_cut_factor * char_spacing;
    maxcut = auto_cut;
    if (allocated)
      for (int i = 1; i <= atom->ntypes; i++)
        for (int j = i; j <= atom->ntypes; j++)
          if (setflag[i][j]) cut[i][j] = auto_cut;

    if (comm->me == 0)
      utils::logmesg(lmp, "pair ls/dem: auto node-node cutoff = {:.4g} "
                     "({:g} x worst-case node spacing {:.4g})\n",
                     auto_cut, auto_cut_factor, char_spacing);
  }

  neighbor->add_request(this, NeighConst::REQ_GHOST);
}

/* ---------------------------------------------------------------------- */

void PairLSDEM::setup()
{
  int n = atom->ntypes;
  double maxcut2 = -1;
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      maxcut2 = MAX(maxcut2, cut[i][j]); // Can we compute a sensible value for this somehow?

  if (maxcut < maxcut2)
    error->all(FLERR, "Maximum cutoff {} less than cutoff defined in pair coefficients {}", maxcut, maxcut2);

  auto fixlist1 = modify->get_fix_by_style("rigid/ls/dem");
  auto fixlist2 = modify->get_fix_by_style("rigid/small/ls/dem");

  if (fixlist1.size() > 1)
    error->all(FLERR, "Must have no more than one instance of fix rigid/ls/dem");
  if (fixlist2.size() > 1)
    error->all(FLERR, "Must have no more than one instance of fix rigid/small/ls/dem");

  if (fixlist1.size() == 0 && fixlist2.size() == 0)
    error->all(FLERR, "Pair ls/dem requires at least one instance of fix rigid/ls/dem or rigid/small/ls/dem");


  int igroup_large, igroup_small, ws_large, ws_small;
  groupbit_large = -1;
  ws_large = -1;
  if (fixlist1.size() == 1) {
    fix_rigid = dynamic_cast<FixRigidLSDEM *>(fixlist1.front());
    groupbit_large = fix_rigid->groupbit;
    igroup_large = fix_rigid->igroup;
    ws_large = fix_rigid->get_storage_model();
  }

  groupbit_small = -1;
  ws_small = -1;
  if (fixlist2.size() == 1) {
    fix_rigid_small = dynamic_cast<FixRigidSmallLSDEM *>(fixlist2.front());
    groupbit_small = fix_rigid_small->groupbit;
    igroup_small = fix_rigid_small->igroup;
    // fix rigid/small/ls/dem has no watershed storage model: it always uses the
    // ARRAY (closest-node) arbitration path. Report ARRAY (0) explicitly so that
    // a small-only run sets watershed_flag = MAX(-1, 0) = 0 below. Leaving it at
    // -1 both skips the closest-node loop (gated on watershed_flag == 0) and,
    // since -1 is truthy, wrongly enters the watershed branch in the force loop
    // (which is large-fix only) -> error/zero forces in a small-only run.
    ws_small = 0;    // ARRAY
  }

  if (ws_large != -1 && ws_small != -1)
    if (ws_large != ws_small)
      error->all(FLERR, "Must use same storage option for rigid/ls/dem and rigid/small/ls/dem");
  watershed_flag = MAX(ws_large, ws_small);

  if (groupbit_small == -1 && igroup_large != 0)
    error->all(FLERR, "If only using fix rigid/ls/dem, it must use group all");

  if (groupbit_large == -1 && igroup_small != 0)
    error->all(FLERR, "If only using fix rigid/small/ls/dem, it must use group all");

  if (groupbit_small != -1 && groupbit_large != -1) {
    int missing_atoms = 0;
    int overlap_atoms = 0;
    int *mask = atom->mask;
    for (size_t i = 0; i < atom->nlocal; i++) {
      if ((mask[i] & groupbit_small) && (mask[i] & groupbit_large)) {
        overlap_atoms++;
      } else if (!(mask[i] & groupbit_small) && !(mask[i] & groupbit_large)) {
        missing_atoms++;
      }
    }

    int total_atoms;
    MPI_Allreduce(&missing_atoms, &total_atoms, 1, MPI_INT, MPI_SUM, world);
    if (total_atoms > 0)
      error->all(FLERR, "All atoms must be in group of either fix rigid/ls/dem or fix rigid/small/ls/dem");

    MPI_Allreduce(&overlap_atoms, &total_atoms, 1, MPI_INT, MPI_SUM, world);
    if (total_atoms > 0)
      error->all(FLERR, "No atoms may be in both groups fix rigid/ls/dem or fix rigid/small/ls/dem");
  }

  int tmp1, tmp2;
  index_ls_dem_n = atom->find_custom("ls_dem_n", tmp1, tmp2);
  index_ls_dem_fs = atom->find_custom("ls_dem_fs", tmp1, tmp2);
  index_ls_dem_touch_id = atom->find_custom("ls_dem_touch_id", tmp1, tmp2);
  index_ls_dem_fn1 = atom->find_custom("ls_dem_fn1", tmp1, tmp2);
  index_ls_dem_fs1 = atom->find_custom("ls_dem_fs1", tmp1, tmp2);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLSDEM::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
    kn[i][j] = mix_energy(kn[i][i], kn[j][j], cut[i][i], cut[j][j]);
    kt[i][j] = mix_energy(kt[i][i], kt[j][j], cut[i][i], cut[j][j]);
    mu[i][j] = 0.5 * (mu[i][i] + mu[j][j]); // Arithmetic mean mixing rule
    etan[i][j] = mix_energy(etan[i][i], etan[j][j], cut[i][i], cut[j][j]);
    etat[i][j] = mix_energy(etat[i][i], etat[j][j], cut[i][i], cut[j][j]);
    knp[i][j] = mix_energy(knp[i][i], knp[j][j], cut[i][i], cut[j][j]);
    decayn1[i][j] = mix_energy(decayn1[i][i], decayn1[j][j], cut[i][i], cut[j][j]);
    etan1[i][j] = mix_energy(etan1[i][i], etan1[j][j], cut[i][i], cut[j][j]);
    decayt1[i][j] = mix_energy(decayt1[i][i], decayt1[j][j], cut[i][i], cut[j][j]);
    etat1[i][j] = mix_energy(etat1[i][i], etat1[j][j], cut[i][i], cut[j][j]);
  }

  // DvdH: For most contact models mixing will not be simple.
  // I would probably discourage the use of this, or give a warning.

  // Enforces symmetry
  cut[j][i] = cut[i][j];
  kn[j][i] = kn[i][j];
  kt[j][i] = kt[i][j];
  mu[j][i] = mu[i][j];
  knp[j][i] = knp[i][j];
  etan[j][i] = etan[i][j];
  etat[j][i] = etat[i][j];
  decayn1[j][i] = decayn1[i][j];
  etan1[j][i] = etan1[i][j];
  decayt1[j][i] = decayt1[i][j];
  etat1[j][i] = etat1[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLSDEM::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&kn[i][j], sizeof(double), 1, fp);
        fwrite(&kt[i][j], sizeof(double), 1, fp);
        fwrite(&mu[i][j], sizeof(double), 1, fp);
        fwrite(&etan[i][j], sizeof(double), 1, fp);
        fwrite(&etat[i][j], sizeof(double), 1, fp);
        fwrite(&knp[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
        fwrite(&decayn1[i][j], sizeof(double), 1, fp);
        fwrite(&etan1[i][j], sizeof(double), 1, fp);
        fwrite(&decayt1[i][j], sizeof(double), 1, fp);
        fwrite(&etat1[i][j], sizeof(double), 1, fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLSDEM::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &kn[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &kt[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &mu[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &etan[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &etat[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &knp[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &decayn1[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &etan1[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &decayt1[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &etat1[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&kn[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&kt[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&mu[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&etan[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&etat[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&knp[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&decayn1[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&etan1[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&decayt1[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&etat1[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLSDEM::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp, "%d %g %g %g %g %g %g %g %g %g %g %g\n", i, kn[i][i], kt[i][i], mu[i][i], etan[i][i], etat[i][i], knp[i][i], cut[i][i],
      decayn1[i][i], etan1[i][i], decayt1[i][i], etat1[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLSDEM::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g %g %g %g %g %g %g %g\n", i, j, kn[i][j], kt[i][j], mu[i][j], etan[i][j], etat[i][j], knp[i][j], cut[i][j],
        decayn1[i][j], etan1[i][j], decayt1[i][j], etat1[i][j]);
}
