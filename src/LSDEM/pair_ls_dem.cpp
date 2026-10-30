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
#include "output.h"
#include "update.h"
#include "utils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static constexpr double EPSILON = 1e-12;
static constexpr double BIG = 1.0e20;
// default for "pair_style ls/dem auto": node-node cutoff = factor x largest node spacing
static constexpr double AUTO_CUT_FACTOR = 2.0;

using namespace LAMMPS_NS;
using namespace MathConst;

/* ----------------------------------------------------------------------
   largest distance from a node to its nearest neighbor node in a set,
   found by sorting along x and pruning the scan once dx exceeds the best gap
------------------------------------------------------------------------- */

static double max_nearest_node_gap(std::vector<std::array<double, 3>> &p)
{
  const int n = p.size();
  if (n < 2) return 0.0;
  std::sort(p.begin(), p.end());

  double maxgapsq = 0.0;
  for (int a = 0; a < n; a++) {
    double best = -1.0;
    for (int b = a + 1; b < n; b++) {
      const double dx = p[b][0] - p[a][0];
      if ((best >= 0.0) && (dx * dx >= best)) break;
      const double dy = p[b][1] - p[a][1];
      const double dz = p[b][2] - p[a][2];
      const double rsq = dx * dx + dy * dy + dz * dz;
      if ((best < 0.0) || (rsq < best)) best = rsq;
    }
    for (int b = a - 1; b >= 0; b--) {
      const double dx = p[a][0] - p[b][0];
      if ((best >= 0.0) && (dx * dx >= best)) break;
      const double dy = p[a][1] - p[b][1];
      const double dz = p[a][2] - p[b][2];
      const double rsq = dx * dx + dy * dy + dz * dz;
      if ((best < 0.0) || (rsq < best)) best = rsq;
    }
    if (best > 0.0) maxgapsq = MAX(maxgapsq, best);
  }
  return sqrt(maxgapsq);
}

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
  fix_rigid_small(nullptr), fix_rigid_small_cpu(nullptr)
{
  writedata = 1;
  single_enable = 0;
  cutoff_auto = 0;
  auto_cut_factor = AUTO_CUT_FACTOR;
  exact_order = 0;
  overlap_det = overlap_limit = maxoverlap = 0.0;
  overlap_warned = 0;
  binfo_lastbuild = -1;
  segs_lastbuild = -1;
}

/* ---------------------------------------------------------------------- */

PairLSDEM::~PairLSDEM()
{
  if (copymode) return;

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
   store the body of each local and ghost atom with its volume and node area,
   so the contact loops do not repeat the group tests and lookups for every pair
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
      const int ib = mybody_large[a];
      b.grp = 2;
      b.bidx = ib;
      b.bID = ib;
      b.off = 0;
      b.vol = (ib >= 0) ? grain_vol[ib] : 0.0;
      b.area = (ib >= 0) ? node_area[ib] : 0.0;
    } else if (fix_rigid_small && (mask[a] & groupbit_small)) {
      const int ib = mybody_small[a];
      b.grp = 1;
      b.bidx = ib;
      b.bID = (int) molecule[a];
      b.off = 1;
      b.vol = (ib >= 0) ? bodyLS[ib].grid_vol : 0.0;
      b.area = (ib >= 0) ? bodyLS[ib].node_area : 0.0;
    } else {
      b.grp = 0;
      b.bidx = -1;
      b.bID = -1;
      b.off = 0;
      b.vol = 0.0;
      b.area = 0.0;
    }
  }
}

/* ----------------------------------------------------------------------
   for each node, list the neighbor nodes on each other grain in the order of the
   neighbor list, under the node of the grain whose nodes are used,
   grains of equal volume are listed both ways since their choice depends on position
------------------------------------------------------------------------- */

void PairLSDEM::build_rep_segments()
{
  const int allnum = list->inum + list->gnum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  const int ntotal = atom->nlocal + atom->nghost;
  double *special_lj = force->special_lj;
  tagint *tag = atom->tag;

  const int inum = list->inum;
  if ((int) rep_segs.size() < ntotal) rep_segs.resize(ntotal);
  for (int a = 0; a < ntotal; a++) rep_segs[a].clear();

  // a node and another grain without a pair from the neighbor list of a local atom
  //   have no contacts, so only the ones found in those lists are kept,
  //   most pairs of ghost atoms are not

  auto segkey = [](int r, int bid, int off) {
    return ((uint64_t) (uint32_t) r << 32) | ((uint64_t) (uint32_t) bid << 1) | (uint64_t) off;
  };

  // level-set bound of the grain of each atom, the largest one of its nodes here
  //   (see LSDEMExtra::grid_bound()), so it is the same for all candidates of a segment

  std::vector<double> bound(ntotal);
  {
    std::unordered_map<int, double> grain_bound;
    for (int a = 0; a < ntotal; a++) {
      bound[a] = atom_grid_bound(a);
      auto it = grain_bound.emplace(binfo[a].bID, bound[a]).first;
      it->second = MAX(it->second, bound[a]);
    }
    for (int a = 0; a < ntotal; a++) bound[a] = grain_bound[binfo[a].bID];
  }

  // a node farther from the center of the other grain than its bound plus the skin cannot reach
  //   it before the next reneighboring, since nodes and centers move less than half the skin,
  //   so that segment would be skipped at each step (see compute()) and is not built,
  //   unless the node still has shear history with that grain

  double **x = atom->x;
  double **grain_com = atom->xcom;
  int *touch_id = atom->ivector[index_ls_dem_touch_id];
  const double skin = neighbor->skin;
  auto near = [&](int r, int c) {
    const double bnd = bound[c];
    if ((bnd >= BIG) || (touch_id[r] == binfo[c].bID)) return true;
    if (bnd < 0.0) return false;
    double del[3] = {x[r][0] - grain_com[c][0], x[r][1] - grain_com[c][1],
                     x[r][2] - grain_com[c][2]};
    domain->minimum_image(FLERR, del);
    return del[0] * del[0] + del[1] * del[1] + del[2] * del[2] <= (bnd + skin) * (bnd + skin);
  };

  std::unordered_set<uint64_t> live;
  for (int ii = 0; ii < allnum; ii++) {
    const int i = ilist[ii];
    const BodyInfo &bi = binfo[i];
    if (bi.grp == 0)
      error->one(FLERR, "Atom {} does not belong to a fix rigid ls/dem group", tag[i]);
    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      if (special_lj[sbmask(j)] == 0) continue;
      j &= NEIGHMASK;
      const BodyInfo &bj = binfo[j];
      if (bj.grp == 0)
        error->one(FLERR, "Atom {} does not belong to a fix rigid ls/dem group", tag[j]);
      if (ii >= inum) continue;
      if ((bi.vol <= bj.vol) && near(i, j)) live.insert(segkey(i, bj.bID, bj.off));
      if ((bi.vol >= bj.vol) && near(j, i)) live.insert(segkey(j, bi.bID, bi.off));
    }
  }

  // only a node with another image (periodic images with the same tag) can be in a
  //   segment twice, so only those are compared with the other candidates

  if ((int) has_image.size() < ntotal) has_image.resize(ntotal);
  for (int a = 0; a < ntotal; a++)
    has_image[a] = (atom->sametag[a] >= 0) || (atom->map(tag[a]) != a);

  // pairs in the neighbor lists of local atoms are numbered in the order of the lists,
  //   which is the order the contacts are evaluated in, pairs of ghost atoms get -1

  int ipair = 0;
  for (int ii = 0; ii < allnum; ii++) {
    const int i = ilist[ii];
    const BodyInfo &bi = binfo[i];
    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      if (special_lj[sbmask(j)] == 0) continue;
      j &= NEIGHMASK;
      const BodyInfo &bj = binfo[j];
      const int sij = (bi.vol <= bj.vol) && near(i, j);
      const int sji = (bi.vol >= bj.vol) && near(j, i);
      if (!sij && !sji) continue;
      const int key = (ii < inum) ? ipair++ : -1;
      const int tie = (bi.vol == bj.vol);
      if (sij && live.count(segkey(i, bj.bID, bj.off)))
        add_rep_candidate(i, bj.bID, bj.off, j, tie, key, 1);
      if (sji && live.count(segkey(j, bi.bID, bi.off)))
        add_rep_candidate(j, bi.bID, bi.off, i, tie, key, 0);
    }
  }

  // bound of the grain of the candidates of each segment

  for (int r = 0; r < ntotal; r++)
    for (Seg &seg : rep_segs[r]) {
      const double bnd = bound[seg.cand[0].idx];
      if (bnd < 0.0)
        seg.bsq = -1.0;
      else
        seg.bsq = (bnd >= BIG) ? BIG : bnd * bnd;
    }
}

/* ---------------------------------------------------------------------- */

void PairLSDEM::add_rep_candidate(int r, int pbody, int poff, int c, int tie, int key,
                                  int rep_owns_pair)
{
  std::vector<Seg> &segs = rep_segs[r];
  const Cand cand = {c, key, rep_owns_pair};
  for (Seg &seg : segs)
    if ((seg.pbody == pbody) && (seg.poff == poff)) {
      // a node may appear more than once with the same tag (periodic images)
      if (has_image[c])
        for (const Cand &other : seg.cand)
          if (atom->tag[other.idx] == atom->tag[c]) seg.dup = 1;
      // with the exact order, the candidate with a key and the lowest atom index is kept first
      seg.cand.push_back(cand);
      if (exact_order && (key >= 0) && ((seg.cand[0].key < 0) || (c < seg.cand[0].idx)))
        std::swap(seg.cand.front(), seg.cand.back());
      return;
    }
  segs.push_back({pbody, (char) poff, (char) tie, 0, BIG, {cand}});
}

/* ----------------------------------------------------------------------
   distance from the center of the grain of atom a beyond which its level set is positive,
   BIG if the grid of the atom is not known
------------------------------------------------------------------------- */

double PairLSDEM::atom_grid_bound(int a)
{
  const BodyInfo &b = binfo[a];
  if ((b.bidx < 0) || (b.grp == 0)) return BIG;
  LSDEMExtra::GridInfo g;
  if (b.grp == 2)
    fix_rigid->get_grid_info(a, g);
  else
    fix_rigid_small->get_grid_info(a, g);

  // shared grids and watershed tables are found by their values

  const void *shared = nullptr;
  if (g.global) shared = g.ws_table ? (const void *) g.ws_table : (const void *) g.values;
  if (shared) {
    auto it = gbound_cache.find(shared);
    if (it == gbound_cache.end())
      it = gbound_cache.emplace(shared, LSDEMExtra::grid_bound(g)).first;
    return it->second;
  }
  return LSDEMExtra::grid_bound(g);
}

/* ---------------------------------------------------------------------- */

void PairLSDEM::compute(int eflag, int vflag)
{
  double xitmp, yitmp, zitmp, delx, dely, delz, rsq;

  // Currently require:
  //   Newton pair off.
  //   Exclude intramolecule interactions
  // In future, remove restrictions

  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **x = atom->x;
  tagint *tag = atom->tag;
  // Compare squared distances against the squared cutoff so the closest-node
  // search avoids a sqrt for every neighbour pair (ordering is preserved).
  double maxcutsq = maxcut * maxcut;
  const int exact = exact_order;

  // body info of local and ghost atoms only changes when the neighbor list is rebuilt

  int ntotal = atom->nlocal + atom->nghost;
  if ((neighbor->lastcall != binfo_lastbuild) || ((int) binfo.size() < ntotal)) {
    cache_body_info(ntotal);
    binfo_lastbuild = neighbor->lastcall;
  }

  // candidate nodes on each other grain only change when the neighbor list is rebuilt

  if ((neighbor->lastcall != segs_lastbuild) || ((int) rep_segs.size() < ntotal)) {
    build_rep_segments();
    segs_lastbuild = neighbor->lastcall;
  }

  // Array: find the closest node on each other grain for each node of the grain whose nodes
  //   are used, of equally close nodes the first one wins.
  //   Comparing squared distances (rsq) gives the same closest node as comparing r.
  //   Each pair of the closest node and a node with its tag (periodic images) in the
  //   neighbor list of a local atom is a contact.
  // Watershed: find the bin of the node in the grid of each other grain once, from the
  //   first candidate in the neighbor list of a local atom, each candidate owning that bin
  //   is a contact.
  // The contacts are then evaluated in neighbor list order.
  // With the exact order, the lowest atom index takes the place of the first candidate and
  //   contacts are evaluated in the order of the atom indices of their pair. None of this then
  //   depends on the order of the neighbor list, so pair ls/dem/kk with a neighbor list built
  //   on the device gives the same results bit for bit. It costs some time, so it is optional.

  double **grain_com = atom->xcom;
  int *touch_id = atom->ivector[index_ls_dem_touch_id];

  contacts.clear();
  wsbins.clear();
  for (int irep = 0; irep < ntotal; irep++) {
    std::vector<Seg> &segs = rep_segs[irep];
    if (segs.empty()) continue;
    xitmp = x[irep][0];
    yitmp = x[irep][1];
    zitmp = x[irep][2];
    for (Seg &seg : segs) {
      // grains of equal volume choose whose nodes are used by their current positions,
      //   from the first candidate, with the exact order the one with a key and the lowest
      //   index, a segment without a key has no contacts
      if (exact && (seg.cand[0].key < 0)) continue;
      const int ncand = seg.cand.size();
      const int c0 = seg.cand[0].idx;
      if (seg.tie && !use_nodes_of_i(irep, c0, binfo[irep].vol, binfo[c0].vol)) continue;

      // a node beyond the bound of the grain of the candidates has no contact with it, unless
      //   it touched that grain before and its shear history has to be reset
      if (seg.bsq < BIG) {
        double del[3] = {xitmp - grain_com[c0][0], yitmp - grain_com[c0][1],
                         zitmp - grain_com[c0][2]};
        domain->minimum_image(FLERR, del);
        if ((del[0] * del[0] + del[1] * del[1] + del[2] * del[2] > seg.bsq) &&
            (touch_id[irep] != binfo[c0].bID))
          continue;
      }

      if (watershed_flag) {
        int first = 0;
        while ((first < ncand) && (seg.cand[first].key < 0)) first++;
        if (first == ncand) continue;
        // the grid and watershed of the other grain are found through its body
        if (binfo[seg.cand[first].idx].bidx < 0)
          error->one(FLERR,
                     "Atom {} cannot find atom that owns body, consider increasing the "
                     "communication cutoff",
                     tag[seg.cand[first].idx]);
        WsBin wb;
        if (seg.poff)
          wb.bin = fix_rigid_small_cpu->get_bin(irep, seg.cand[first].idx, wb.xl);
        else
          wb.bin = fix_rigid->get_bin(irep, seg.cand[first].idx, wb.xl);
        const int ws = wsbins.size();
        wsbins.push_back(wb);
        for (int m = first; m < ncand; m++) {
          const Cand &cand = seg.cand[m];
          if (cand.key < 0) continue;
          const int owns = seg.poff ? fix_rigid_small_cpu->check_watershed_bin(wb.bin, cand.idx)
                                    : fix_rigid->check_watershed_bin(wb.bin, cand.idx);
          if (owns) add_contact(irep, cand, ws);
        }
        continue;
      }

      // separate loops, so the check of the exact order costs no time in the other one
      int win = -1;
      double minrsq = 0.0;
      if (exact) {
        for (int m = 0; m < ncand; m++) {
          const int c = seg.cand[m].idx;
          delx = xitmp - x[c][0];
          dely = yitmp - x[c][1];
          delz = zitmp - x[c][2];
          rsq = delx * delx + dely * dely + delz * delz;
          if (rsq > maxcutsq) continue;
          if ((win < 0) || (rsq < minrsq) || ((rsq == minrsq) && (c < seg.cand[win].idx))) {
            minrsq = rsq;
            win = m;
          }
        }
      } else {
        for (int m = 0; m < ncand; m++) {
          const int c = seg.cand[m].idx;
          delx = xitmp - x[c][0];
          dely = yitmp - x[c][1];
          delz = zitmp - x[c][2];
          rsq = delx * delx + dely * dely + delz * delz;
          if (rsq > maxcutsq) continue;
          if ((win < 0) || (rsq < minrsq)) {
            minrsq = rsq;
            win = m;
          }
        }
      }
      if (win < 0) continue;
      if (seg.dup) {
        const tagint wtag = tag[seg.cand[win].idx];
        for (const Cand &cand : seg.cand)
          if (tag[cand.idx] == wtag) add_contact(irep, cand, -1);
      } else {
        add_contact(irep, seg.cand[win], -1);
      }
    }
  }
  if (exact_order)
    std::sort(contacts.begin(), contacts.end(), [](const Contact &a, const Contact &b) {
      return (a.i < b.i) || ((a.i == b.i) && (a.j < b.j));
    });
  else
    std::sort(contacts.begin(), contacts.end(),
              [](const Contact &a, const Contact &b) { return a.key < b.key; });

  double x_local[3];
  for (const Contact &ct : contacts) {
    if (ct.ws < 0) {
      x_local[0] = x_local[1] = x_local[2] = 0.0;
      process_contact(ct.i, ct.j, ct.calc, -1, x_local);
    } else {
      const WsBin &wb = wsbins[ct.ws];
      x_local[0] = wb.xl[0];
      x_local[1] = wb.xl[1];
      x_local[2] = wb.xl[2];
      process_contact(ct.i, ct.j, ct.calc, wb.bin, x_local);
    }
  }

  check_overlap();
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   contact model for one node of grain i (calc_force_of_i_on_j = 1) or of grain j (0)
   touching the level set of the other grain, applies the force and torque to
   atom i and, if it is a local atom, to atom j
   tmp_bin and x_local are the watershed bin and grid position (tmp_bin = -1 for array)
------------------------------------------------------------------------- */

void PairLSDEM::process_contact(int i, int j, int calc_force_of_i_on_j, int tmp_bin,
                                double *x_local)
{
  int itype, jtype, ibody, jbody, ibodyID, jbodyID;
  double xitmp, yitmp, zitmp, xjtmp, yjtmp, zjtmp, u;
  double vxitmp, vyitmp, vzitmp, vxjtmp, vyjtmp, vzjtmp;
  double icomx, icomy, icomz, jcomx, jcomy, jcomz;
  double iomegax, iomegay, iomegaz, jomegax, jomegay, jomegaz, spin_norm;
  double normal[3], fn_mag, fpair[3], contact_point[3], lever[3], torque_pair[3];
  double fs_tmp[3], fs_mag, fs_max, fs_mag_trial, k[3], costheta, term1[3], term2;
  double tangent[3], shear_incr, v_rel[3], v_rel_t[3], v_rel_n_mag, v_rel_t_mag, v_rel_t_mag_inv;
  double normal_old[3], tangent_old[3], fs_mag_add, areai, areaj;

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
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  double dt = update->dt;
  double **grain_com = atom->xcom;
  double **grain_omega = atom->omega;

  itype = type[i];
  jtype = type[j];
  xitmp = x[i][0];
  yitmp = x[i][1];
  zitmp = x[i][2];
  vxitmp = v[i][0];
  vyitmp = v[i][1];
  vzitmp = v[i][2];
  icomx = grain_com[i][0];
  icomy = grain_com[i][1];
  icomz = grain_com[i][2];
  iomegax = grain_omega[i][0];
  iomegay = grain_omega[i][1];
  iomegaz = grain_omega[i][2];
  ibody = binfo[i].bidx;
  ibodyID = binfo[i].bID;
  areai = binfo[i].area;
  jbody = binfo[j].bidx;
  jbodyID = binfo[j].bID;
  areaj = binfo[j].area;

  // ghosts may not find bodytag if comm distance too small
  if (ibody < 0)
    error->one(
        FLERR,
        "Atom {} cannot find atom that owns body, consider increasing the communication cutoff",
        tag[i]);
  if (jbody < 0)
    error->one(
        FLERR,
        "Atom {} cannot find atom that owns body, consider increasing the communication cutoff",
        tag[j]);

  // Evaluate the level set of the other grain (par) at the node used (ni), one call for
  // both directions. Force magnitude and direction go i -> j by definition.
  // Level set is by definition negative inside the particle,
  // so swap the sign to get the overlap distance.
  const int ni = calc_force_of_i_on_j ? i : j;
  const int par = calc_force_of_i_on_j ? j : i;
  if (binfo[par].grp == 2)
    u = -fix_rigid->get_ls_value(ni, par, tmp_bin, normal, x_local);
  else if (binfo[par].grp == 1)
    u = -fix_rigid_small_cpu->get_ls_value(ni, par, tmp_bin, normal, x_local);
  else
    error->one(FLERR, "Atom {} does not belong to a fix rigid ls/dem group", tag[par]);

  // Resolve the direction-dependent quantities once, so the rest of the contact
  // path is free of calc_force_of_i_on_j branches. We always use the node on the
  // chosen grain (ni), its stored history (fs_ni, n_ni, fn1[ni], fs1[ni]) and node
  // area (narea), and the body it is touching (partner_body). The stored shear and
  // normal vectors are kept in the node's own i->j frame, so when we use node j we
  // flip their sign (hsign = -1) on both read and write.
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
  } else {    // Physical contact!!
    maxoverlap = MAX(maxoverlap, u);
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
    const double nsign = calc_force_of_i_on_j ? -1.0 : 1.0;
    normal[0] *= nsign;
    normal[1] *= nsign;
    normal[2] *= nsign;
    const double npos0 = calc_force_of_i_on_j ? xitmp : xjtmp;
    const double npos1 = calc_force_of_i_on_j ? yitmp : yjtmp;
    const double npos2 = calc_force_of_i_on_j ? zitmp : zjtmp;
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
  if (etanij > 0.0) { fn_mag += etanij * v_rel_n_mag; }

  // Maxwell arm (1st, parallel, only repulsive)
  if (etan1ij > 0.0) {    // preprocessing guarantees that decayn1 > 0 if etan1 > 0
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
  fpair[0] = -fn_mag * normal[0];
  fpair[1] = -fn_mag * normal[1];
  fpair[2] = -fn_mag * normal[2];

  ////////////////////
  // Tangent stress //
  ////////////////////

  // Tangent force only exists if mu > 0 and kt > 0
  //if ( (kt[itype][jtype] > 0) && (mu[itype][jtype] > 0) ){
  // DvdH: Grains without friction are silly, so I took out this check.

  // Check if the pair is valid for shear history calculation
  // Initialise if no contact
  if (touch_id[ni] == -1) { touch_id[ni] = partner_body; }
  if (touch_id[ni] != partner_body) {
    if (comm->me == 0) {
      tagint node_tag = calc_force_of_i_on_j ? tag[i] : tag[j];
      int node_body = calc_force_of_i_on_j ? ibodyID : jbodyID;
      tagint partner_tag = calc_force_of_i_on_j ? tag[j] : tag[i];
      error->warning(FLERR,
                     "Shear history of node {} on grain {} penetrating node {} on {} cannot be "
                     "computed at step {}",
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
  if (MathExtra::lensq3(normal_old) > 0.0) {    // len3 > 0, but without the sqrt
    // Account for tilt. This is an exact correction over rotation of the normal
    // from the previous to the current time step.
    MathExtra::cross3(normal_old, normal, k);    // n_old x n_new

    // Account for spin. This is an approximation using the half-step angular
    // velocities. We rotate about the OLD normal so that, together with the
    // tilt, the additive reconstruction matches the spin-then-tilt rotation.
    // spin_norm = dt*(omega_avg . n_old) = dt*|omega_sp| (signed).
    spin_norm = 0.5 * dt *
        ((iomegax + jomegax) * normal_old[0] + (iomegay + jomegay) * normal_old[1] +
         (iomegaz + jomegaz) * normal_old[2]);    // 0.5*dt*(omegai+omegaj) \dot n_old

    // First Baker-Campbell-Hausdorff corrective term for the non-commutativity
    // of the spin and tilt rotations: -0.5*|omega_sp|*dt*(n_old x (n_old x n_new)).
    // The non-commutativity error dominates and only manifests itself in the tilt.
    MathExtra::cross3(normal_old, k, term1);    // n_old x (n_old x n_new)
    double bch_coef = -0.5 * fabs(spin_norm);
    k[0] += spin_norm * normal_old[0] + bch_coef * term1[0];
    k[1] += spin_norm * normal_old[1] + bch_coef * term1[1];
    k[2] += spin_norm * normal_old[2] + bch_coef * term1[2];

    // Applying the rotation with the unnormalised Rodrigues formula:
    //   k already carries axis * sin(theta), so neither the axis nor the cross term needs normalising
    //   and (1 - cos) / sin^2 = 1 / (1 + cos) avoids small-angle cancellation
    //   fs_rot = fs cos + (k x fs) + k (k . fs) / (1 + cos), with cos = sqrt(1 - |k|^2)
    double sinsq = MathExtra::lensq3(k);    // sin^2(theta) = |k|^2
    if (sinsq > EPSILON * EPSILON) {        // Don't apply rotation if magnitude is tiny
      costheta = sqrt(MAX(1.0 - sinsq, 0.0));
      MathExtra::cross3(k, fs_tmp, term1);
      term2 = MathExtra::dot3(k, fs_tmp) / (1.0 + costheta);
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
  if (fs_mag_trial > EPSILON) {
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
  if (etatij > 0) { fs_mag_add += etatij * v_rel_t_mag; }

  // Maxwell arm (1st, parallel, only repulsive)
  if (etan1ij > 0.0) {    // preprocessing guarantees that decayt1 > 0 if etat1 > 0
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
    fs1[ni] += decayt1ij * fs1[ni] * MathExtra::dot3(tangent_old, tangent) +
        etat1ij * (1 - decayt1ij) * v_rel_t_mag;    // v_rel_t_mag is always positive
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
    error->one(FLERR,
               "Bad force calculated between atoms {} {} on bodies {} {} with overlap {} and "
               "normal {} {} {}\n",
               tag[i], tag[j], ibody, jbody, u, normal[0], normal[1], normal[2]);
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
    error->one(FLERR,
               "Bad torque calculated between atoms {} {} on bodies {} {} with overlap {} and "
               "normal {} {} {}\n",
               tag[i], tag[j], ibody, jbody, u, normal[0], normal[1], normal[2]);
#endif

  // Virial of the contact split at the contact point: node i gets (x_i - contact point) x F
  //   and node j gets (contact point - x_j) x F, with F the force on i, each on the rank that
  //   owns it. The sum is (x_i - x_j) x F, as the constraint virial of the rigid fixes uses the
  //   atom positions, and per grain the centroid stress sums to the contact point stress.
  if (evflag) {
    const double deli[3] = {x[i][0] - contact_point[0], x[i][1] - contact_point[1],
                            x[i][2] - contact_point[2]};
    const double delj[3] = {contact_point[0] - x[j][0], contact_point[1] - x[j][1],
                            contact_point[2] - x[j][2]};
    const double vi[6] = {deli[0] * fpair[0], deli[1] * fpair[1], deli[2] * fpair[2],
                          deli[0] * fpair[1], deli[0] * fpair[2], deli[1] * fpair[2]};
    const double vj[6] = {delj[0] * fpair[0], delj[1] * fpair[1], delj[2] * fpair[2],
                          delj[0] * fpair[1], delj[0] * fpair[2], delj[1] * fpair[2]};
    if (vflag_global) {
      for (int n = 0; n < 6; n++) virial[n] += vi[n];
      if (j < nlocal)
        for (int n = 0; n < 6; n++) virial[n] += vj[n];
    }
    if (vflag_atom) {
      for (int n = 0; n < 6; n++) vatom[i][n] += vi[n];
      if (j < nlocal)
        for (int n = 0; n < 6; n++) vatom[j][n] += vj[n];
    }
  }

  // Mirror forces and torques on grain j
  if (newton_pair || j < nlocal) {    // Need to check this again if we end up enabling newton_pair
    MathExtra::negate3(fpair);        // Sign swap
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
}

/* ----------------------------------------------------------------------
   return true if the nodes of atom i's grain should probe the level set of atom j's grain
   use the smaller grain, if volumes are tied use the direction between the grain centers
     since molecule IDs/tags can be assigned in a decomposition-dependent order
------------------------------------------------------------------------- */

bool PairLSDEM::use_nodes_of_i(int i, int j, double ivol, double jvol) const
{
  if (ivol != jvol) return ivol < jvol;

  double **xcom = atom->xcom;
  double del[3];
  MathExtra::sub3(xcom[j], xcom[i], del);
  domain->minimum_image(FLERR, del);

  static constexpr double EPS_TIE = 1.0e-12;
  if (del[0] > EPS_TIE) return true;
  if (del[0] < -EPS_TIE) return false;
  if (del[1] > EPS_TIE) return true;
  if (del[1] < -EPS_TIE) return false;
  return del[2] > 0.0;
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
  if (narg < 1) utils::missing_cmd_args(FLERR, "pair_style ls/dem", error);

  // "auto" estimates the node-node cutoff from the node spacing of the grains in init_style()
  //   with an optional factor, otherwise the cutoff is the given number
  int iarg = 1;
  if (strcmp(arg[0], "auto") == 0) {
    cutoff_auto = 1;
    maxcut = -1.0;
    auto_cut_factor = AUTO_CUT_FACTOR;
    if ((narg > 1) && (strcmp(arg[1], "order") != 0)) {
      auto_cut_factor = utils::numeric(FLERR, arg[1], false, lmp);
      if (auto_cut_factor <= 0.0)
        error->all(FLERR, "The pair ls/dem auto cutoff factor {} must be positive",
                   auto_cut_factor);
      if ((auto_cut_factor <= 1.0) && (comm->me == 0))
        error->warning(FLERR,
                       "A pair ls/dem auto cutoff factor of {} does not exceed the node "
                       "spacing, so contacts may be missed",
                       auto_cut_factor);
      iarg = 2;
    }
  } else {
    cutoff_auto = 0;
    maxcut = utils::numeric(FLERR, arg[0], false, lmp);
  }

  // "order exact" makes the results independent of the order of the neighbor list
  exact_order = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "order") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style ls/dem order", error);
      if (strcmp(arg[iarg + 1], "exact") == 0)
        exact_order = 1;
      else if (strcmp(arg[iarg + 1], "list") == 0)
        exact_order = 0;
      else
        error->all(FLERR, "Unknown pair_style ls/dem order {}", arg[iarg + 1]);
      iarg += 2;
    } else
      error->all(FLERR, "Unknown pair_style ls/dem keyword {}", arg[iarg]);
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

  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, "Pair ls/dem requires an atom map, see atom_modify");

  // resolve an "auto" cutoff before Pair::init() calls init_one(), so it sets the neighbor cutoff

  const double spacing = auto_node_spacing();
  if (cutoff_auto) {
    if (spacing <= 0.0)
      error->all(FLERR,
                 "Pair ls/dem cutoff auto found no grains to estimate the node spacing "
                 "from, set the cutoff explicitly instead");

    maxcut = auto_cut_factor * spacing;
    if (allocated)
      for (int i = 1; i <= atom->ntypes; i++)
        for (int j = i; j <= atom->ntypes; j++)
          if (setflag[i][j]) cut[i][j] = maxcut;

    if (comm->me == 0)
      utils::logmesg(lmp,
                     "pair ls/dem: auto node-node cutoff = {:.8g} ({:g} x largest node "
                     "spacing {:.8g})\n",
                     maxcut, auto_cut_factor, spacing);
  }

  // get fix data


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
  igroup_large = -1;
  ws_large = -1;
  if (fixlist1.size() == 1) {
    fix_rigid = dynamic_cast<FixRigidLSDEM *>(fixlist1.front());
    groupbit_large = fix_rigid->groupbit;
    igroup_large = fix_rigid->igroup;
    ws_large = fix_rigid->get_storage_model();
  }

  groupbit_small = -1;
  igroup_small = -1;
  ws_small = -1;
  if (fixlist2.size() == 1) {
    fix_rigid_small = dynamic_cast<FixRigidSmallLSDEMAccess *>(fixlist2.front());
    fix_rigid_small_cpu = dynamic_cast<FixRigidSmallLSDEM *>(fixlist2.front());
    if (!fix_rigid_small_cpu && !kokkosable)
      error->all(FLERR, "Fix {} requires pair ls/dem/kk", fixlist2.front()->style);
    groupbit_small = fixlist2.front()->groupbit;
    igroup_small = fixlist2.front()->igroup;
    ws_small = fix_rigid_small->get_storage_model();
  }

  // node pairs within one grain never interact, so leave them out of the neighbor list,
  //   which saves building and looping over them; this needs every grain to be one molecule

  bool by_molecule = true;
  if (fix_rigid && !fix_rigid->grains_by_molecule()) by_molecule = false;
  if (fix_rigid_small && !fix_rigid_small->grains_by_molecule()) by_molecule = false;
  if (by_molecule && atom->molecule_flag) {
    bool excluded = false;
    for (int m = 0; m < neighbor->nex_mol; m++)
      if (neighbor->ex_mol_intra[m] && (neighbor->ex_mol_group[m] == 0)) excluded = true;
    if (!excluded) {
      neighbor->modify_params("exclude molecule/intra all");
      if (comm->me == 0)
        utils::logmesg(lmp,
                       "Pair ls/dem: node pairs within one grain are left out of the "
                       "neighbor list\n");
    }
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

  // a node at depth u inside a grain only has a contact while a node of that grain is within
  //   the cutoff, on a flat surface its nearest node is up to h = spacing / sqrt(2) to the side,
  //   so contacts are found while u < sqrt(cutoff^2 - h^2); deeper overlaps lose contacts and
  //   grains can pass through each other; the warning comes a quarter node spacing before that

  const double hside = spacing / MY_SQRT2;
  overlap_det = overlap_limit = 0.0;
  if ((spacing > 0.0) && (maxcut > hside)) {
    overlap_det = sqrt(maxcut * maxcut - hside * hside);
    overlap_limit = overlap_det - 0.25 * spacing;
  }
  maxoverlap = 0.0;
  overlap_warned = 0;

  if (watershed_flag == 0)
    neighbor->add_request(this, NeighConst::REQ_GHOST);
  else
    neighbor->add_request(this);
}

/* ----------------------------------------------------------------------
   warn once per run if a contact is deeper than the pair cutoff can follow,
   checked on thermo output steps and the last step of a run
------------------------------------------------------------------------- */

void PairLSDEM::check_overlap()
{
  if (overlap_warned || (overlap_limit <= 0.0)) return;
  if ((update->ntimestep != output->next_thermo) && (update->ntimestep != update->laststep))
    return;

  double maxall = 0.0;
  MPI_Allreduce(&maxoverlap, &maxall, 1, MPI_DOUBLE, MPI_MAX, world);
  if (maxall <= overlap_limit) return;
  overlap_warned = 1;
  if (comm->me == 0)
    error->warning(FLERR,
                   "Pair ls/dem overlap {:.6g} by step {} is close to the deepest overlap {:.6g} "
                   "that the cutoff {:.6g} can follow, deeper overlaps lose contacts and grains "
                   "may pass through each other: use stiffer contacts, a smaller load or "
                   "timestep, or a larger cutoff",
                   maxall, update->ntimestep, overlap_det, maxcut);
}

/* ----------------------------------------------------------------------
   largest nearest-neighbor node spacing over all grains
   fix rigid/small/ls/dem grains use their molecule templates,
   fix rigid/ls/dem grains (no templates) gather their atoms, so the
   estimate does not depend on the number of MPI ranks
------------------------------------------------------------------------- */

double PairLSDEM::auto_node_spacing()
{
  double spacing = 0.0;
  std::vector<std::array<double, 3>> p;

  for (int im = 0; im < atom->nmolecule; im++) {
    Molecule *onemol = atom->molecules[im];
    if (!onemol->lsdemflag) continue;
    p.resize(onemol->natoms);
    for (int a = 0; a < onemol->natoms; a++)
      for (int d = 0; d < 3; d++) p[a][d] = onemol->x[a][d];
    spacing = MAX(spacing, max_nearest_node_gap(p));
  }

  auto fixlist = modify->get_fix_by_style("^rigid/ls/dem");
  if (fixlist.empty()) return spacing;
  if (!atom->molecule_flag)
    error->all(FLERR, "Pair ls/dem cutoff auto with fix rigid/ls/dem requires molecule IDs");

  // gather the unwrapped coordinates and molecule IDs of all atoms in the fix group

  const int groupbit_fix = fixlist.front()->groupbit;
  const int nlocal = atom->nlocal;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  imageint *image = atom->image;
  double **x = atom->x;

  std::vector<double> sendbuf;
  double unwrap[3];
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit_fix)) continue;
    domain->unmap(x[i], image[i], unwrap);
    sendbuf.push_back((double) molecule[i]);
    sendbuf.insert(sendbuf.end(), unwrap, unwrap + 3);
  }

  const int nprocs = comm->nprocs;
  int nsend = sendbuf.size();
  std::vector<int> recvcounts(nprocs), displs(nprocs);
  MPI_Allgather(&nsend, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);
  bigint ntotal = 0;
  for (int iproc = 0; iproc < nprocs; iproc++) {
    displs[iproc] = ntotal;
    ntotal += recvcounts[iproc];
  }
  if (ntotal > MAXSMALLINT)
    error->all(FLERR, "Too many atoms in fix rigid/ls/dem for pair ls/dem cutoff auto");
  if (sendbuf.empty()) sendbuf.push_back(0.0);
  std::vector<double> recvbuf(MAX(ntotal, 1));
  MPI_Allgatherv(sendbuf.data(), nsend, MPI_DOUBLE, recvbuf.data(), recvcounts.data(),
                 displs.data(), MPI_DOUBLE, world);

  std::map<tagint, std::vector<std::array<double, 3>>> bodies;
  for (bigint m = 0; m < ntotal; m += 4)
    bodies[(tagint) recvbuf[m]].push_back({recvbuf[m + 1], recvbuf[m + 2], recvbuf[m + 3]});

  // share the per-body work across ranks, the maximum is the same on any number of ranks

  double spacing_local = 0.0, spacing_large = 0.0;
  int ibody = 0;
  for (auto &body : bodies) {
    if (ibody++ % nprocs != comm->me) continue;
    spacing_local = MAX(spacing_local, max_nearest_node_gap(body.second));
  }
  MPI_Allreduce(&spacing_local, &spacing_large, 1, MPI_DOUBLE, MPI_MAX, world);

  return MAX(spacing, spacing_large);
}

/* ---------------------------------------------------------------------- */

void PairLSDEM::setup()
{
  // the ghost atoms are recreated in each setup, so refill the body info at the next compute()
  binfo_lastbuild = -1;
  segs_lastbuild = -1;
  gbound_cache.clear();

  int n = atom->ntypes;
  double maxcut2 = -1;
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      maxcut2 = MAX(maxcut2, cut[i][j]); // Can we compute a sensible value for this somehow?

  if (maxcut < maxcut2)
    error->all(FLERR, "Maximum cutoff {} less than cutoff defined in pair coefficients {}", maxcut, maxcut2);

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
