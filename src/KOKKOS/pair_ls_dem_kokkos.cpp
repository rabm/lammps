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

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_rigid_ls_dem.h"
#include "fix_rigid_small_ls_dem.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <algorithm>
#include <unordered_set>

using namespace LAMMPS_NS;

static constexpr double EPSILON = 1e-12;
static constexpr double EPS_TIE = 1.0e-12;
static constexpr double BIG = 1.0e20;

/* ----------------------------------------------------------------------
   copy a host vector into the front of a device view, growing it if needed
------------------------------------------------------------------------- */

template <class ViewType, class T>
static void upload(ViewType &d, const std::vector<T> &h, const char *name)
{
  typedef typename ViewType::value_type V;
  const size_t n = h.size();
  if (d.extent(0) < MAX(n, (size_t) 1))
    d = ViewType(Kokkos::view_alloc(std::string(name), Kokkos::WithoutInitializing), MAX(n, (size_t) 1));
  if (n == 0) return;
  auto h_view = Kokkos::create_mirror_view(Kokkos::subview(d, std::make_pair((size_t) 0, n)));
  for (size_t m = 0; m < n; m++) h_view(m) = static_cast<V>(h[m]);
  Kokkos::deep_copy(Kokkos::subview(d, std::make_pair((size_t) 0, n)), h_view);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLSDEMKokkos<DeviceType>::PairLSDEMKokkos(LAMMPS *lmp) : PairLSDEM(lmp)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | V_MASK | F_MASK | TORQUE_MASK | OMEGA_MASK | TYPE_MASK | TAG_MASK |
    XCOM_MASK | QUAT_MASK;
  datamask_modify = F_MASK | TORQUE_MASK;

  nseg = ncand = nrep = nkey = 0;
  kern.flag = Kokkos::View<int *, DeviceType>("pair:flag", 3);
  kern.flag_tag = Kokkos::View<tagint *, DeviceType>("pair:flag_tag", 5);
  kern.ncontact = Kokkos::View<int, DeviceType>("pair:ncontact");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLSDEMKokkos<DeviceType>::~PairLSDEMKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_eatom, eatom);
  memoryKK->destroy_kokkos(k_vatom, vatom);
  eatom = nullptr;
  vatom = nullptr;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::allocate()
{
  PairLSDEM::allocate();

  const int np1 = atom->ntypes + 1;
  k_params = typename Kernels::tdual_params("pair:params", np1, np1);
  kern.params = k_params.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::init_style()
{
  PairLSDEM::init_style();

  // the half neighbor list with ghost atoms has no KOKKOS variant, so it is built on the host
  //   in the same order as for pair ls/dem, the segments and contact keys follow that order

  auto request = neighbor->find_request(this);
  request->set_kokkos_host(0);
  request->set_kokkos_device(0);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::setup()
{
  PairLSDEM::setup();

  if (watershed_flag)
    error->all(FLERR, "Pair ls/dem/kk does not yet support the watershed storage of level sets");

  const int n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++) {
      auto &p = k_params.view_host()(i, j);
      p.kn = kn[i][j];
      p.kt = kt[i][j];
      p.mu = mu[i][j];
      p.etan = etan[i][j];
      p.etat = etat[i][j];
      p.knp = knp[i][j];
      p.etan1 = etan1[i][j];
      p.decayn1 = decayn1[i][j];
      p.etat1 = etat1[i][j];
      p.decayt1 = decayt1[i][j];
    }
  k_params.modify_host();
  k_params.template sync<DeviceType>();

  // the grids may be recreated in each setup, so copy them again at the next reneighboring

  ggrid_offset.clear();
  h_ggrid.clear();
}

/* ----------------------------------------------------------------------
   segments of PairLSDEM::build_rep_segments() built from the neighbor list into flat
   arrays on the device, with the level-set grids of the candidate nodes,
   the candidates of each node and the contacts of each atom are ordered by key
   the order of the segments of a node does not matter, each is evaluated on its own
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::build_segments()
{
  const int nlocal = atom->nlocal;
  const int ntotal = nlocal + atom->nghost;
  const int inum = list->inum;
  const int allnum = list->inum + list->gnum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  double *special_lj = force->special_lj;
  tagint *tag = atom->tag;

  // candidates of each node in the order of the neighbor list, as in build_rep_segments()

  struct Rec {
    int bid;
    int cand;
    int key;
    char off;
    char owns;
    char tie;
  };

  // a segment without a pair from the neighbor list of a local atom has no contacts,
  //   so only the segments found in those lists are kept, most pairs of ghost atoms are not

  auto segkey = [](int r, int bid, int off) {
    return ((uint64_t) (uint32_t) r << 32) | ((uint64_t) (uint32_t) bid << 1) | (uint64_t) off;
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
      if (bi.vol <= bj.vol) live.insert(segkey(i, bj.bID, bj.off));
      if (bi.vol >= bj.vol) live.insert(segkey(j, bi.bID, bi.off));
    }
  }

  // pass 1 counts and pass 2 stores the candidates of each node of the kept segments

  std::vector<bigint> rbeg(ntotal + 1, 0);
  std::vector<Rec> rec;
  for (int pass = 1; pass <= 2; pass++) {
    std::vector<bigint> fill;
    if (pass == 2) {
      for (int a = 0; a < ntotal; a++) rbeg[a + 1] += rbeg[a];
      rec.resize(rbeg[ntotal]);
      fill.assign(rbeg.begin(), rbeg.end() - 1);
    }
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
        const int key = (ii < inum) ? ipair++ : -1;
        const int tie = (bi.vol == bj.vol);
        if ((bi.vol <= bj.vol) && live.count(segkey(i, bj.bID, bj.off))) {
          if (pass == 1) rbeg[i + 1]++;
          else rec[fill[i]++] = {bj.bID, j, key, bj.off, 1, (char) tie};
        }
        if ((bi.vol >= bj.vol) && live.count(segkey(j, bi.bID, bi.off))) {
          if (pass == 1) rbeg[j + 1]++;
          else rec[fill[j]++] = {bi.bID, i, key, bi.off, 0, (char) tie};
        }
      }
    }
  }
  std::unordered_set<uint64_t>().swap(live);

  // group the candidates of each node by the grain they belong to, keeping their order

  std::vector<int> h_seg_rep, h_seg_flag, h_seg_cbeg(1, 0);
  std::vector<int> h_cand_idx, h_cand_key, h_cand_owns;
  std::vector<int> h_rep_atom, h_rep_kbeg(1, 0), h_rep_kslot;
  std::vector<char> partner(ntotal, 0);
  std::vector<std::pair<int, int>> keyslot;
  std::vector<tagint> tags;
  int maxkey = -1;
  bigint maxcontact = 0;
  h_cand_idx.reserve(rec.size());
  h_cand_key.reserve(rec.size());
  h_cand_owns.reserve(rec.size());

  for (int r = 0; r < ntotal; r++) {
    if (rbeg[r] == rbeg[r + 1]) continue;
    auto first = rec.begin() + rbeg[r];
    auto last = rec.begin() + rbeg[r + 1];
    std::stable_sort(first, last, [](const Rec &x, const Rec &y) {
      return (x.bid < y.bid) || ((x.bid == y.bid) && (x.off < y.off));
    });
    keyslot.clear();
    for (auto sb = first; sb != last;) {
      auto se = sb;
      while ((se != last) && (se->bid == sb->bid) && (se->off == sb->off)) ++se;

      // a node may appear more than once with the same tag (periodic images)
      tags.clear();
      int nkeyed = 0;
      for (auto c = sb; c != se; ++c) {
        tags.push_back(tag[c->cand]);
        nkeyed += (c->key >= 0);
      }
      std::sort(tags.begin(), tags.end());
      const int dup = std::adjacent_find(tags.begin(), tags.end()) != tags.end();

      // a segment has one contact, or one per image of the closest node
      maxcontact += dup ? nkeyed : MIN(nkeyed, 1);
      h_seg_rep.push_back(r);
      h_seg_flag.push_back((sb->tie ? 1 : 0) | (dup ? 2 : 0));
      for (auto c = sb; c != se; ++c) {
        const int slot = h_cand_idx.size();
        h_cand_idx.push_back(c->cand);
        h_cand_key.push_back(c->key);
        h_cand_owns.push_back(c->owns);
        partner[c->cand] = 1;
        if (c->key >= 0) {
          keyslot.emplace_back(c->key, slot);
          maxkey = MAX(maxkey, c->key);
        }
      }
      h_seg_cbeg.push_back(h_cand_idx.size());
      sb = se;
    }
    std::sort(keyslot.begin(), keyslot.end());
    h_rep_atom.push_back(r);
    for (const auto &ks : keyslot) h_rep_kslot.push_back(ks.second);
    h_rep_kbeg.push_back(h_rep_kslot.size());
  }
  std::vector<Rec>().swap(rec);
  std::vector<bigint>().swap(rbeg);

  nseg = h_seg_rep.size();
  ncand = h_cand_idx.size();
  nrep = h_rep_atom.size();
  nkey = maxkey + 1;
  if ((nkey > MAXSMALLINT / 2) || (maxcontact > MAXSMALLINT / 6))
    error->one(FLERR, "Too many node pairs for pair ls/dem/kk");

  // the atoms of each contact: i holds the pair in its neighbor list and is local

  std::vector<int> key_i(nkey, -1), key_j(nkey, -1);
  for (int q = 0; q < nrep; q++) {
    const int r = h_rep_atom[q];
    for (int k = h_rep_kbeg[q]; k < h_rep_kbeg[q + 1]; k++) {
      const int slot = h_rep_kslot[k];
      const int key = h_cand_key[slot];
      key_i[key] = h_cand_owns[slot] ? r : h_cand_idx[slot];
      key_j[key] = h_cand_owns[slot] ? h_cand_idx[slot] : r;
    }
  }

  // contacts of each local atom, filled in key order so each list is ordered by key

  std::vector<int> h_gat_beg(nlocal + 1, 0), h_gat_ent;
  for (int key = 0; key < nkey; key++) {
    if (key_i[key] < 0) continue;
    h_gat_beg[key_i[key] + 1]++;
    if (key_j[key] < nlocal) h_gat_beg[key_j[key] + 1]++;
  }
  for (int a = 0; a < nlocal; a++) h_gat_beg[a + 1] += h_gat_beg[a];
  h_gat_ent.resize(h_gat_beg[nlocal]);
  {
    std::vector<int> fill(h_gat_beg.begin(), h_gat_beg.end() - 1);
    for (int key = 0; key < nkey; key++) {
      if (key_i[key] < 0) continue;
      h_gat_ent[fill[key_i[key]]++] = 2 * key;
      if (key_j[key] < nlocal) h_gat_ent[fill[key_j[key]]++] = 2 * key + 1;
    }
  }
  h_key_jlocal.assign(nkey, 0);
  for (int key = 0; key < nkey; key++) h_key_jlocal[key] = (key_j[key] >= 0) && (key_j[key] < nlocal);

  // body of each atom and the level-set grid of the atoms that are candidates

  std::vector<double> h_vol(ntotal), h_area(ntotal), h_gmin(3 * ntotal, 0.0);
  std::vector<double> h_gstride(ntotal, 1.0), h_gscale(ntotal, 1.0), h_dgrid;
  std::vector<int> h_bid(ntotal), h_bidx(ntotal), h_gstyle(ntotal, -1), h_gngrid(3 * ntotal, 0);
  std::vector<bigint> h_goff(ntotal, 0);
  int newglobal = 0;
  LSDEMExtra::GridInfo g;

  for (int a = 0; a < ntotal; a++) {
    const BodyInfo &b = binfo[a];
    h_vol[a] = b.vol;
    h_area[a] = b.area;
    h_bid[a] = b.bID;
    h_bidx[a] = b.bidx;
    if (!partner[a] || (b.bidx < 0) || (b.grp == 0)) continue;

    if (b.grp == 2)
      fix_rigid->get_grid_info(a, g);
    else
      fix_rigid_small->get_grid_info(a, g);

    h_gstyle[a] = g.global;
    h_gstride[a] = g.stride;
    h_gscale[a] = g.scale;
    for (int k = 0; k < 3; k++) {
      h_gngrid[3 * a + k] = g.ngrid[k];
      h_gmin[3 * a + k] = g.grid_min[k];
    }
    if (g.global) {
      auto it = ggrid_offset.find(g.values);
      if (it == ggrid_offset.end()) {
        it = ggrid_offset.emplace(g.values, (bigint) h_ggrid.size()).first;
        h_ggrid.insert(h_ggrid.end(), g.values, g.values + g.nvalues);
        newglobal = 1;
      }
      h_goff[a] = it->second;
    } else {
      h_goff[a] = h_dgrid.size();
      h_dgrid.insert(h_dgrid.end(), g.values, g.values + g.nvalues);
    }
  }

  upload(kern.seg_rep, h_seg_rep, "pair:seg_rep");
  upload(kern.seg_flag, h_seg_flag, "pair:seg_flag");
  upload(kern.seg_cbeg, h_seg_cbeg, "pair:seg_cbeg");
  upload(kern.cand_idx, h_cand_idx, "pair:cand_idx");
  upload(kern.cand_key, h_cand_key, "pair:cand_key");
  upload(kern.cand_owns, h_cand_owns, "pair:cand_owns");
  upload(kern.rep_atom, h_rep_atom, "pair:rep_atom");
  upload(kern.rep_kbeg, h_rep_kbeg, "pair:rep_kbeg");
  upload(kern.rep_kslot, h_rep_kslot, "pair:rep_kslot");
  upload(kern.gat_beg, h_gat_beg, "pair:gat_beg");
  upload(kern.gat_ent, h_gat_ent, "pair:gat_ent");
  upload(kern.vol, h_vol, "pair:vol");
  upload(kern.area, h_area, "pair:area");
  upload(kern.bid, h_bid, "pair:bid");
  upload(kern.bidx, h_bidx, "pair:bidx");
  upload(kern.gstyle, h_gstyle, "pair:gstyle");
  upload(kern.gngrid, h_gngrid, "pair:gngrid");
  upload(kern.gmin, h_gmin, "pair:gmin");
  upload(kern.gstride, h_gstride, "pair:gstride");
  upload(kern.gscale, h_gscale, "pair:gscale");
  upload(kern.goff, h_goff, "pair:goff");
  upload(kern.dgrid, h_dgrid, "pair:dgrid");
  if (newglobal || (kern.ggrid.extent(0) == 0)) upload(kern.ggrid, h_ggrid, "pair:ggrid");

  if ((int) kern.cand_on.extent(0) < MAX(ncand, 1))
    kern.cand_on = Kokkos::View<int *, DeviceType>("pair:cand_on", MAX(ncand, 1));
  if ((int) kern.kon.extent(0) < MAX(nkey, 1))
    kern.kon = Kokkos::View<int *, DeviceType>("pair:kon", MAX(nkey, 1));
  if ((int) kern.kf.extent(0) < 3 * MAX(maxcontact, 1)) {
    kern.kf = Kokkos::View<KK_FLOAT *, DeviceType>("pair:kf", 3 * MAX(maxcontact, 1));
    kern.kti = Kokkos::View<KK_FLOAT *, DeviceType>("pair:kti", 3 * MAX(maxcontact, 1));
    kern.ktj = Kokkos::View<KK_FLOAT *, DeviceType>("pair:ktj", 3 * MAX(maxcontact, 1));
    kern.kv = Kokkos::View<KK_FLOAT *, DeviceType>("pair:kv", 6 * MAX(maxcontact, 1));
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  ev_init(eflag_in, vflag_in, 0);
  if (cvflag_atom)
    error->all(FLERR, "Pair ls/dem/kk does not support the centroid stress per atom");

  // the pair energy is zero, the per-atom energy stays the zeros it was created with

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pair:vatom");
    kern.vatom = k_vatom.template view<DeviceType>();
  }

  const int nlocal = atom->nlocal;
  const int ntotal = nlocal + atom->nghost;

  // body info and candidate nodes only change when the neighbor list is rebuilt

  if ((neighbor->lastcall != segs_lastbuild) || ((int) binfo.size() < ntotal)) {
    atomKK->sync(Host, X_MASK | TAG_MASK | MASK_MASK | MOLECULE_MASK | DARRAY_MASK);
    cache_body_info(ntotal);
    binfo_lastbuild = neighbor->lastcall;
    build_segments();
    segs_lastbuild = neighbor->lastcall;
  }

  atomKK->sync(execution_space, datamask_read);
  atomKK->modified(execution_space, datamask_modify);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  kern.x = x;
  kern.f = f;
  kern.v = atomKK->k_v.view<DeviceType>();
  kern.omega = atomKK->k_omega.view<DeviceType>();
  kern.xcom = atomKK->k_xcom.view<DeviceType>();
  kern.quat = atomKK->k_quat.view<DeviceType>();
  kern.torque = atomKK->k_torque.view<DeviceType>();
  kern.type = atomKK->k_type.view<DeviceType>();
  kern.tag = atomKK->k_tag.view<DeviceType>();

  // shear history of the nodes, only these custom per-atom arrays are synced,
  //   not the distributed level sets that are also stored as custom arrays

  auto &k_hist_n = atomKK->k_darray.view_host()[index_ls_dem_n].k_view;
  auto &k_hist_fs = atomKK->k_darray.view_host()[index_ls_dem_fs].k_view;
  atomKK->sync(execution_space, IVECTOR_MASK | DVECTOR_MASK);
  k_hist_n.template sync<DeviceType>();
  k_hist_fs.template sync<DeviceType>();
  kern.d_ivector = atomKK->k_ivector.template view<DeviceType>();
  kern.d_dvector = atomKK->k_dvector.template view<DeviceType>();
  kern.d_hist_n = k_hist_n.template view<DeviceType>();
  kern.d_hist_fs = k_hist_fs.template view<DeviceType>();
  kern.touch_row = index_ls_dem_touch_id;
  kern.fn1_row = index_ls_dem_fn1;
  kern.fs1_row = index_ls_dem_fs1;

  kern.nlocal = nlocal;
  kern.dimension = domain->dimension;
  kern.maxcutsq = maxcut * maxcut;
  kern.dt = update->dt;
  kern.vflag_either = vflag_either;
  kern.vflag_atom = vflag_atom;
  kern.triclinic = domain->triclinic;
  kern.xperiodic = domain->xperiodic;
  kern.yperiodic = domain->yperiodic;
  kern.zperiodic = domain->zperiodic;
  kern.xprd = domain->xprd;
  kern.yprd = domain->yprd;
  kern.zprd = domain->zprd;
  kern.xprd_half = domain->xprd_half;
  kern.yprd_half = domain->yprd_half;
  kern.zprd_half = domain->zprd_half;
  kern.xy = domain->xy;
  kern.xz = domain->xz;
  kern.yz = domain->yz;

  Kokkos::deep_copy(kern.flag, 0);
  if (ncand) Kokkos::deep_copy(Kokkos::subview(kern.cand_on, std::make_pair(0, ncand)), 0);
  if (nkey) Kokkos::deep_copy(Kokkos::subview(kern.kon, std::make_pair(0, nkey)), -1);
  Kokkos::deep_copy(kern.ncontact, 0);

  // closest nodes, then the contacts of each node in key order (they share its shear history),
  //   then the sums of the forces and torques on each atom in key order, as in pair ls/dem

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairLSDEMSelect>(0, nseg), kern);
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairLSDEMContact>(0, nrep), kern);
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairLSDEMGather>(0, nlocal), kern);

  atomKK->modified(execution_space, IVECTOR_MASK | DVECTOR_MASK);
  k_hist_n.template modify<DeviceType>();
  k_hist_fs.template modify<DeviceType>();

  // errors and warnings found on the device

  auto h_flag = Kokkos::create_mirror_view_and_copy(LMPHostType::memory_space(), kern.flag);
  if (h_flag(0) || h_flag(1)) {
    auto h_flag_tag = Kokkos::create_mirror_view_and_copy(LMPHostType::memory_space(), kern.flag_tag);
    if (h_flag(0))
      error->one(FLERR,
                 "Atom {} cannot find atom that owns body, consider increasing the communication cutoff",
                 h_flag_tag(0));
    if (comm->me == 0)
      error->warning(FLERR,
                     "Shear history of node {} on grain {} penetrating node {} on {} cannot be "
                     "computed at step {} ({} such contacts in this step)",
                     h_flag_tag(1), h_flag_tag(2), h_flag_tag(3), h_flag_tag(4),
                     update->ntimestep, h_flag(1));
  }

  // global virial summed on the host in key order, as ev_tally_xyz() does for pair ls/dem

  if (vflag_global && nkey) {
    auto h_kon = Kokkos::create_mirror_view_and_copy(LMPHostType::memory_space(), kern.kon);
    auto h_kv = Kokkos::create_mirror_view_and_copy(LMPHostType::memory_space(), kern.kv);
    for (int key = 0; key < nkey; key++) {
      const int c = h_kon(key);
      if (c < 0) continue;
      for (int n = 0; n < 6; n++) virial[n] += 0.5 * h_kv(6 * c + n);
      if (h_key_jlocal[key])
        for (int n = 0; n < 6; n++) virial[n] += 0.5 * h_kv(6 * c + n);
    }
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.sync_host();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);
}

/* ----------------------------------------------------------------------
   minimum image convention as in Domain::minimum_image()
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairLSDEMKernelsKokkos<DeviceType>::minimum_image(KK_FLOAT &dx, KK_FLOAT &dy, KK_FLOAT &dz) const
{
  if (triclinic == 0) {
    if (xperiodic) {
      while (Kokkos::fabs(dx) > xprd_half) {
        if (dx < 0.0) dx += xprd;
        else dx -= xprd;
      }
    }
    if (yperiodic) {
      while (Kokkos::fabs(dy) > yprd_half) {
        if (dy < 0.0) dy += yprd;
        else dy -= yprd;
      }
    }
    if (zperiodic) {
      while (Kokkos::fabs(dz) > zprd_half) {
        if (dz < 0.0) dz += zprd;
        else dz -= zprd;
      }
    }
  } else {
    if (zperiodic) {
      while (Kokkos::fabs(dz) > zprd_half) {
        if (dz < 0.0) {
          dz += zprd;
          dy += yz;
          dx += xz;
        } else {
          dz -= zprd;
          dy -= yz;
          dx -= xz;
        }
      }
    }
    if (yperiodic) {
      while (Kokkos::fabs(dy) > yprd_half) {
        if (dy < 0.0) {
          dy += yprd;
          dx += xy;
        } else {
          dy -= yprd;
          dx -= xy;
        }
      }
    }
    if (xperiodic) {
      while (Kokkos::fabs(dx) > xprd_half) {
        if (dx < 0.0) dx += xprd;
        else dx -= xprd;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   PairLSDEM::use_nodes_of_i() for grains i and j of equal volume
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
bool PairLSDEMKernelsKokkos<DeviceType>::use_nodes_of_i(int i, int j) const
{
  if (vol(i) != vol(j)) return vol(i) < vol(j);

  KK_FLOAT del0 = xcom(j, 0) - xcom(i, 0);
  KK_FLOAT del1 = xcom(j, 1) - xcom(i, 1);
  KK_FLOAT del2 = xcom(j, 2) - xcom(i, 2);
  minimum_image(del0, del1, del2);

  if (del0 > EPS_TIE) return true;
  if (del0 < -EPS_TIE) return false;
  if (del1 > EPS_TIE) return true;
  if (del1 < -EPS_TIE) return false;
  return del2 > 0.0;
}

/* ----------------------------------------------------------------------
   level set of the grain of atom j at node i and its normal, as get_ls_value_array()
   of fix rigid/ls/dem and rigid/small/ls/dem with interpolate_LS_array()
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairLSDEMKernelsKokkos<DeviceType>::ls_value(int i, int j, KK_FLOAT *nvec) const
{
  KK_FLOAT dx[3], xl[3], qc[4], temp[4];

  dx[0] = x(i, 0) - xcom(j, 0);
  dx[1] = x(i, 1) - xcom(j, 1);
  dx[2] = x(i, 2) - xcom(j, 2);
  minimum_image(dx[0], dx[1], dx[2]);

  // rotate into the grid frame of grain j with the conjugate of its quaternion

  qc[0] = quat(j, 0);
  qc[1] = -quat(j, 1);
  qc[2] = -quat(j, 2);
  qc[3] = -quat(j, 3);
  temp[0] = -qc[1] * dx[0] - qc[2] * dx[1] - qc[3] * dx[2];
  temp[1] = qc[0] * dx[0] + qc[2] * dx[2] - qc[3] * dx[1];
  temp[2] = qc[0] * dx[1] + qc[3] * dx[0] - qc[1] * dx[2];
  temp[3] = qc[0] * dx[2] + qc[1] * dx[1] - qc[2] * dx[0];
  xl[0] = -qc[1] * temp[0] + qc[0] * temp[1] - qc[3] * temp[2] + qc[2] * temp[3];
  xl[1] = -qc[2] * temp[0] + qc[3] * temp[1] + qc[0] * temp[2] - qc[1] * temp[3];
  xl[2] = -qc[3] * temp[0] - qc[2] * temp[1] + qc[1] * temp[2] + qc[0] * temp[3];

  // position in units of grid cells from the lower corner of the grid

  xl[0] = xl[0] - gmin(3 * j);
  xl[1] = xl[1] - gmin(3 * j + 1);
  xl[2] = xl[2] - gmin(3 * j + 2);
  const KK_FLOAT stride = gstride(j);
  const KK_FLOAT sinv = 1.0 / stride;
  xl[0] *= sinv;
  xl[1] *= sinv;
  xl[2] *= sinv;

  const int ng0 = gngrid(3 * j);
  const int ng1 = gngrid(3 * j + 1);
  const int ng2 = gngrid(3 * j + 2);
  const int mybin = int(xl[0]) + int(xl[1]) * ng0 + int(xl[2]) * ng0 * ng1;
  const int ix0 = int(xl[0]);
  const int ix1 = int(xl[1]);
  const int ix2 = int(xl[2]);

  const int global = gstyle(j);
  const KK_FLOAT *mygrid = global ? (ggrid.data() + goff(j)) : (dgrid.data() + goff(j));
  KK_FLOAT dist;

  if ((ix0 < 0 || ix0 >= (ng0 - 1)) || (ix1 < 0 || ix1 >= (ng1 - 1)) ||
      ((dimension == 3) && (ix2 < 0 || ix2 >= (ng2 - 1)))) {
    dist = BIG;
  } else {
    const KK_FLOAT ls000 = mygrid[mybin];
    if (ls000 > 2.0 * stride) {
      dist = ls000;
    } else {
      const KK_FLOAT ls100 = mygrid[mybin + 1];
      const KK_FLOAT ls010 = mygrid[mybin + ng0];
      const KK_FLOAT ls110 = mygrid[mybin + 1 + ng0];

      KK_FLOAT xr[3];
      xr[0] = xl[0] - static_cast<KK_FLOAT>(ix0);
      xr[1] = xl[1] - static_cast<KK_FLOAT>(ix1);
      xr[2] = xl[2] - static_cast<KK_FLOAT>(ix2);

      const KK_FLOAT lsxy0 = ls000 + xr[1] * (ls010 - ls000) +
        xr[0] * (ls100 - ls000 + xr[1] * (ls110 - ls100 - ls010 + ls000));
      dist = lsxy0;

      nvec[0] = ls100 - ls000 + xr[1] * (ls110 - ls100 - ls010 + ls000);
      nvec[1] = ls010 - ls000 + xr[0] * (ls110 - ls100 - ls010 + ls000);
      nvec[2] = 0.0;

      if (dimension == 3) {
        const int up = ng0 * ng1;
        const KK_FLOAT ls001 = mygrid[mybin + up];
        const KK_FLOAT ls101 = mygrid[mybin + 1 + up];
        const KK_FLOAT ls011 = mygrid[mybin + ng0 + up];
        const KK_FLOAT ls111 = mygrid[mybin + 1 + ng0 + up];

        const KK_FLOAT lsxy1 = ls001 + xr[1] * (ls011 - ls001) +
          xr[0] * (ls101 - ls001 + xr[1] * (ls111 - ls101 - ls011 + ls001));

        dist = xr[2] * (lsxy1 - lsxy0) + lsxy0;
        nvec[0] *= 1 - xr[2];
        nvec[0] += xr[2] * (ls101 - ls001 + xr[1] * (ls111 - ls101 - ls011 + ls001));
        nvec[1] *= 1 - xr[2];
        nvec[1] += xr[2] * (ls011 - ls001 + xr[0] * (ls111 - ls101 - ls011 + ls001));
        nvec[2] = lsxy1 - lsxy0;
      }

      const KK_FLOAT val = nvec[0] * nvec[0] + nvec[1] * nvec[1] + nvec[2] * nvec[2];
      if (val > 0.0) {
        const KK_FLOAT scale = 1.0 / Kokkos::sqrt(val);
        nvec[0] *= scale;
        nvec[1] *= scale;
        nvec[2] *= scale;
      }
    }
  }

  // values of a shared grid are not scaled

  if (global) dist *= gscale(j);

  // rotate the normal back to the global frame, it is only used for a contact (dist < 0)

  if (dist < 0.0) {
    const KK_FLOAT a0 = quat(j, 0), a1 = quat(j, 1), a2 = quat(j, 2), a3 = quat(j, 3);
    const KK_FLOAT b0 = nvec[0], b1 = nvec[1], b2 = nvec[2];
    temp[0] = -a1 * b0 - a2 * b1 - a3 * b2;
    temp[1] = a0 * b0 + a2 * b2 - a3 * b1;
    temp[2] = a0 * b1 + a3 * b0 - a1 * b2;
    temp[3] = a0 * b2 + a1 * b1 - a2 * b0;
    nvec[0] = -a1 * temp[0] + a0 * temp[1] - a3 * temp[2] + a2 * temp[3];
    nvec[1] = -a2 * temp[0] + a3 * temp[1] + a0 * temp[2] - a1 * temp[3];
    nvec[2] = -a3 * temp[0] - a2 * temp[1] + a1 * temp[2] + a0 * temp[3];
  }

  return dist;
}

/* ----------------------------------------------------------------------
   closest node of each segment, as in PairLSDEM::compute()
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairLSDEMKernelsKokkos<DeviceType>::operator()(TagPairLSDEMSelect, const int s) const
{
  const int r = seg_rep(s);
  const int b = seg_cbeg(s);
  const int e = seg_cbeg(s + 1);
  const int sflag = seg_flag(s);

  // grains of equal volume choose whose nodes are used by their current positions

  if ((sflag & 1) && !use_nodes_of_i(r, cand_idx(b))) return;

  const KK_FLOAT xitmp = x(r, 0);
  const KK_FLOAT yitmp = x(r, 1);
  const KK_FLOAT zitmp = x(r, 2);
  int win = -1;
  KK_FLOAT minrsq = 0.0;
  for (int m = b; m < e; m++) {
    const int c = cand_idx(m);
    const KK_FLOAT delx = xitmp - x(c, 0);
    const KK_FLOAT dely = yitmp - x(c, 1);
    const KK_FLOAT delz = zitmp - x(c, 2);
    const KK_FLOAT rsq = delx * delx + dely * dely + delz * delz;
    if (rsq > maxcutsq) continue;
    if ((win < 0) || (rsq < minrsq)) {
      minrsq = rsq;
      win = m;
    }
  }
  if (win < 0) return;

  if (sflag & 2) {
    const tagint wtag = tag(cand_idx(win));
    for (int m = b; m < e; m++)
      if ((tag(cand_idx(m)) == wtag) && (cand_key(m) >= 0)) cand_on(m) = 1;
  } else if (cand_key(win) >= 0) {
    cand_on(win) = 1;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairLSDEMKernelsKokkos<DeviceType>::operator()(TagPairLSDEMContact, const int q) const
{
  const int r = rep_atom(q);
  for (int k = rep_kbeg(q); k < rep_kbeg(q + 1); k++) {
    const int slot = rep_kslot(k);
    if (!cand_on(slot)) continue;
    const int c = cand_idx(slot);
    if (cand_owns(slot))
      contact(r, c, 1, cand_key(slot));
    else
      contact(c, r, 0, cand_key(slot));
  }
}

/* ----------------------------------------------------------------------
   contact model of PairLSDEM::process_contact() with array storage,
   stores the force and torques of the contact by key instead of applying them
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairLSDEMKernelsKokkos<DeviceType>::contact(int i, int j, int calc_force_of_i_on_j,
                                                  int key) const
{
  KK_FLOAT normal[3], fpair[3], contact_point[3], lever[3], torque_pair[3];
  KK_FLOAT fs_tmp[3], k[3], term1[3], tangent[3], v_rel[3], v_rel_t[3];
  KK_FLOAT normal_old[3], tangent_old[3];

  // ghosts may not find bodytag if comm distance too small

  if ((bidx(i) < 0) || (bidx(j) < 0)) {
    if (Kokkos::atomic_fetch_add(&flag(0), 1) == 0) flag_tag(0) = (bidx(i) < 0) ? tag(i) : tag(j);
    return;
  }

  const int ni = calc_force_of_i_on_j ? i : j;
  const int par = calc_force_of_i_on_j ? j : i;
  normal[0] = normal[1] = normal[2] = 0.0;
  const KK_FLOAT u = -ls_value(ni, par, normal);

  const KK_FLOAT hsign = calc_force_of_i_on_j ? 1.0 : -1.0;
  const KK_FLOAT narea = calc_force_of_i_on_j ? area(i) : area(j);
  const int partner_body = calc_force_of_i_on_j ? bid(j) : bid(i);

  if (u <= 0) {
    if (d_ivector(touch_row, ni) == partner_body) {
      d_ivector(touch_row, ni) = -1;
      d_hist_fs(ni, 0) = 0.0;
      d_hist_fs(ni, 1) = 0.0;
      d_hist_fs(ni, 2) = 0.0;
      d_hist_n(ni, 0) = 0.0;
      d_hist_n(ni, 1) = 0.0;
      d_hist_n(ni, 2) = 0.0;
    }
    return;
  }

  const KK_FLOAT nsign = calc_force_of_i_on_j ? -1.0 : 1.0;
  normal[0] *= nsign;
  normal[1] *= nsign;
  normal[2] *= nsign;
  const KK_FLOAT npos0 = calc_force_of_i_on_j ? x(i, 0) : x(j, 0);
  const KK_FLOAT npos1 = calc_force_of_i_on_j ? x(i, 1) : x(j, 1);
  const KK_FLOAT npos2 = calc_force_of_i_on_j ? x(i, 2) : x(j, 2);
  contact_point[0] = npos0 - 0.5 * u * hsign * normal[0];
  contact_point[1] = npos1 - 0.5 * u * hsign * normal[1];
  contact_point[2] = npos2 - 0.5 * u * hsign * normal[2];

  const params_lsdem &p = params(type(i), type(j));

  // normal stress

  KK_FLOAT fn_mag;
  if (Kokkos::fabs(p.knp) < EPSILON)
    fn_mag = p.kn * u;
  else
    fn_mag = p.kn * Kokkos::pow(u, p.knp);

  v_rel[0] = v(i, 0) - v(j, 0);
  v_rel[1] = v(i, 1) - v(j, 1);
  v_rel[2] = v(i, 2) - v(j, 2);
  const KK_FLOAT v_rel_n_mag = v_rel[0] * normal[0] + v_rel[1] * normal[1] + v_rel[2] * normal[2];

  if (p.etan > 0.0) fn_mag += p.etan * v_rel_n_mag;

  if (p.etan1 > 0.0) {
    d_dvector(fn1_row, ni) = p.decayn1 * d_dvector(fn1_row, ni) +
      p.etan1 * (1.0 - p.decayn1) * v_rel_n_mag;
    fn_mag += d_dvector(fn1_row, ni);
  }

  fn_mag = (fn_mag > 0.0) ? fn_mag : 0.0;

  fpair[0] = -fn_mag * normal[0];
  fpair[1] = -fn_mag * normal[1];
  fpair[2] = -fn_mag * normal[2];

  // tangential stress

  if (d_ivector(touch_row, ni) == -1) d_ivector(touch_row, ni) = partner_body;
  if (d_ivector(touch_row, ni) != partner_body) {
    Kokkos::atomic_add(&flag(1), 1);
    if (Kokkos::atomic_fetch_add(&flag(2), 1) == 0) {
      flag_tag(1) = calc_force_of_i_on_j ? tag(i) : tag(j);
      flag_tag(2) = calc_force_of_i_on_j ? bid(i) : bid(j);
      flag_tag(3) = calc_force_of_i_on_j ? tag(j) : tag(i);
      flag_tag(4) = partner_body;
    }
  }

  fs_tmp[0] = hsign * d_hist_fs(ni, 0);
  fs_tmp[1] = hsign * d_hist_fs(ni, 1);
  fs_tmp[2] = hsign * d_hist_fs(ni, 2);
  normal_old[0] = hsign * d_hist_n(ni, 0);
  normal_old[1] = hsign * d_hist_n(ni, 1);
  normal_old[2] = hsign * d_hist_n(ni, 2);

  // rotation of the contact plane by the tilt and spin of the normal

  if (normal_old[0] * normal_old[0] + normal_old[1] * normal_old[1] +
      normal_old[2] * normal_old[2] > 0.0) {
    k[0] = normal_old[1] * normal[2] - normal_old[2] * normal[1];
    k[1] = normal_old[2] * normal[0] - normal_old[0] * normal[2];
    k[2] = normal_old[0] * normal[1] - normal_old[1] * normal[0];

    const KK_FLOAT spin_norm = 0.5 * dt *
      ((omega(i, 0) + omega(j, 0)) * normal_old[0] + (omega(i, 1) + omega(j, 1)) * normal_old[1] +
       (omega(i, 2) + omega(j, 2)) * normal_old[2]);

    term1[0] = normal_old[1] * k[2] - normal_old[2] * k[1];
    term1[1] = normal_old[2] * k[0] - normal_old[0] * k[2];
    term1[2] = normal_old[0] * k[1] - normal_old[1] * k[0];
    const KK_FLOAT bch_coef = -0.5 * Kokkos::fabs(spin_norm);
    k[0] += spin_norm * normal_old[0] + bch_coef * term1[0];
    k[1] += spin_norm * normal_old[1] + bch_coef * term1[1];
    k[2] += spin_norm * normal_old[2] + bch_coef * term1[2];

    const KK_FLOAT sinsq = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
    if (sinsq > EPSILON * EPSILON) {
      const KK_FLOAT cmax = 1.0 - sinsq;
      const KK_FLOAT costheta = Kokkos::sqrt((cmax > 0.0) ? cmax : 0.0);
      term1[0] = k[1] * fs_tmp[2] - k[2] * fs_tmp[1];
      term1[1] = k[2] * fs_tmp[0] - k[0] * fs_tmp[2];
      term1[2] = k[0] * fs_tmp[1] - k[1] * fs_tmp[0];
      const KK_FLOAT term2 = (k[0] * fs_tmp[0] + k[1] * fs_tmp[1] + k[2] * fs_tmp[2]) /
        (1.0 + costheta);
      fs_tmp[0] = fs_tmp[0] * costheta + term1[0] + k[0] * term2;
      fs_tmp[1] = fs_tmp[1] * costheta + term1[1] + k[1] * term2;
      fs_tmp[2] = fs_tmp[2] * costheta + term1[2] + k[2] * term2;
    }
  }

  v_rel_t[0] = v_rel[0] - v_rel_n_mag * normal[0];
  v_rel_t[1] = v_rel[1] - v_rel_n_mag * normal[1];
  v_rel_t[2] = v_rel[2] - v_rel_n_mag * normal[2];
  const KK_FLOAT v_rel_t_mag =
    Kokkos::sqrt(v_rel_t[0] * v_rel_t[0] + v_rel_t[1] * v_rel_t[1] + v_rel_t[2] * v_rel_t[2]);

  KK_FLOAT v_rel_t_mag_inv;
  if (v_rel_t_mag > EPSILON) {
    v_rel_t_mag_inv = 1.0 / v_rel_t_mag;
    tangent[0] = v_rel_t[0] * v_rel_t_mag_inv;
    tangent[1] = v_rel_t[1] * v_rel_t_mag_inv;
    tangent[2] = v_rel_t[2] * v_rel_t_mag_inv;
  } else {
    const KK_FLOAT norm =
      Kokkos::sqrt(fs_tmp[0] * fs_tmp[0] + fs_tmp[1] * fs_tmp[1] + fs_tmp[2] * fs_tmp[2]);
    if (norm != 0)
      v_rel_t_mag_inv = 1.0 / norm;
    else
      v_rel_t_mag_inv = 0.0;
    tangent[0] = fs_tmp[0] * v_rel_t_mag_inv;
    tangent[1] = fs_tmp[1] * v_rel_t_mag_inv;
    tangent[2] = fs_tmp[2] * v_rel_t_mag_inv;
  }

  const KK_FLOAT shear_incr = p.kt * v_rel_t_mag * dt;

  fs_tmp[0] -= shear_incr * tangent[0];
  fs_tmp[1] -= shear_incr * tangent[1];
  fs_tmp[2] -= shear_incr * tangent[2];
  KK_FLOAT fs_mag_trial =
    Kokkos::sqrt(fs_tmp[0] * fs_tmp[0] + fs_tmp[1] * fs_tmp[1] + fs_tmp[2] * fs_tmp[2]);

  // Coulomb limit, std::min(fs_max, fs_mag_trial)

  const KK_FLOAT fs_max = p.mu * fn_mag;
  KK_FLOAT fs_mag = (fs_mag_trial < fs_max) ? fs_mag_trial : fs_max;

  if (fs_mag_trial > EPSILON) {
    fs_tmp[0] = fs_mag * (fs_tmp[0] / fs_mag_trial);
    fs_tmp[1] = fs_mag * (fs_tmp[1] / fs_mag_trial);
    fs_tmp[2] = fs_mag * (fs_tmp[2] / fs_mag_trial);
  }

  d_hist_fs(ni, 0) = hsign * fs_tmp[0];
  d_hist_fs(ni, 1) = hsign * fs_tmp[1];
  d_hist_fs(ni, 2) = hsign * fs_tmp[2];
  d_hist_n(ni, 0) = hsign * normal[0];
  d_hist_n(ni, 1) = hsign * normal[1];
  d_hist_n(ni, 2) = hsign * normal[2];

  KK_FLOAT fs_mag_add = 0.0;
  if (p.etat > 0) fs_mag_add += p.etat * v_rel_t_mag;

  if (p.etan1 > 0.0) {
    const KK_FLOAT norm =
      Kokkos::sqrt(fs_tmp[0] * fs_tmp[0] + fs_tmp[1] * fs_tmp[1] + fs_tmp[2] * fs_tmp[2]);
    if (norm != 0)
      v_rel_t_mag_inv = 1.0 / norm;
    else
      v_rel_t_mag_inv = 0.0;
    tangent_old[0] = fs_tmp[0] * v_rel_t_mag_inv;
    tangent_old[1] = fs_tmp[1] * v_rel_t_mag_inv;
    tangent_old[2] = fs_tmp[2] * v_rel_t_mag_inv;
    d_dvector(fs1_row, ni) += p.decayt1 * d_dvector(fs1_row, ni) *
      (tangent_old[0] * tangent[0] + tangent_old[1] * tangent[1] + tangent_old[2] * tangent[2]) +
      p.etat1 * (1 - p.decayt1) * v_rel_t_mag;
    fs_mag_add += d_dvector(fs1_row, ni);
  }

  fs_tmp[0] -= fs_mag_add * tangent[0];
  fs_tmp[1] -= fs_mag_add * tangent[1];
  fs_tmp[2] -= fs_mag_add * tangent[2];
  fs_mag_trial =
    Kokkos::sqrt(fs_tmp[0] * fs_tmp[0] + fs_tmp[1] * fs_tmp[1] + fs_tmp[2] * fs_tmp[2]);

  fs_mag = (fs_mag_trial < fs_max) ? fs_mag_trial : fs_max;

  if (fs_mag > 0 && fs_mag_trial != 0) {
    fs_tmp[0] = fs_mag * (fs_tmp[0] / fs_mag_trial);
    fs_tmp[1] = fs_mag * (fs_tmp[1] / fs_mag_trial);
    fs_tmp[2] = fs_mag * (fs_tmp[2] / fs_mag_trial);
    fpair[0] += fs_tmp[0];
    fpair[1] += fs_tmp[1];
    fpair[2] += fs_tmp[2];
  }

  // force and torques, the force is a stress until multiplied by the node area

  fpair[0] *= narea;
  fpair[1] *= narea;
  fpair[2] *= narea;
  const int c = Kokkos::atomic_fetch_add(&ncontact(), 1);
  kf(3 * c) = fpair[0];
  kf(3 * c + 1) = fpair[1];
  kf(3 * c + 2) = fpair[2];

  lever[0] = contact_point[0] - xcom(i, 0);
  lever[1] = contact_point[1] - xcom(i, 1);
  lever[2] = contact_point[2] - xcom(i, 2);
  minimum_image(lever[0], lever[1], lever[2]);
  kti(3 * c) = lever[1] * fpair[2] - lever[2] * fpair[1];
  kti(3 * c + 1) = lever[2] * fpair[0] - lever[0] * fpair[2];
  kti(3 * c + 2) = lever[0] * fpair[1] - lever[1] * fpair[0];

  if (j < nlocal) {
    fpair[0] = -fpair[0];
    fpair[1] = -fpair[1];
    fpair[2] = -fpair[2];
    lever[0] = contact_point[0] - xcom(j, 0);
    lever[1] = contact_point[1] - xcom(j, 1);
    lever[2] = contact_point[2] - xcom(j, 2);
    minimum_image(lever[0], lever[1], lever[2]);
    torque_pair[0] = lever[1] * fpair[2] - lever[2] * fpair[1];
    torque_pair[1] = lever[2] * fpair[0] - lever[0] * fpair[2];
    torque_pair[2] = lever[0] * fpair[1] - lever[1] * fpair[0];
    ktj(3 * c) = torque_pair[0];
    ktj(3 * c + 1) = torque_pair[1];
    ktj(3 * c + 2) = torque_pair[2];
  }

  // virial of ev_tally_xyz() with the force on j if j is local, else on i

  if (vflag_either) {
    const KK_FLOAT delx = normal[0] * u;
    const KK_FLOAT dely = normal[1] * u;
    const KK_FLOAT delz = normal[2] * u;
    kv(6 * c) = delx * fpair[0];
    kv(6 * c + 1) = dely * fpair[1];
    kv(6 * c + 2) = delz * fpair[2];
    kv(6 * c + 3) = delx * fpair[1];
    kv(6 * c + 4) = delx * fpair[2];
    kv(6 * c + 5) = dely * fpair[2];
  }

  kon(key) = c;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairLSDEMKernelsKokkos<DeviceType>::operator()(TagPairLSDEMGather, const int a) const
{
  for (int m = gat_beg(a); m < gat_beg(a + 1); m++) {
    const int ent = gat_ent(m);
    const int key = ent >> 1;
    const int c = kon(key);
    if (c < 0) continue;
    if (ent & 1) {
      f(a, 0) += -kf(3 * c);
      f(a, 1) += -kf(3 * c + 1);
      f(a, 2) += -kf(3 * c + 2);
      torque(a, 0) += ktj(3 * c);
      torque(a, 1) += ktj(3 * c + 1);
      torque(a, 2) += ktj(3 * c + 2);
    } else {
      f(a, 0) += kf(3 * c);
      f(a, 1) += kf(3 * c + 1);
      f(a, 2) += kf(3 * c + 2);
      torque(a, 0) += kti(3 * c);
      torque(a, 1) += kti(3 * c + 1);
      torque(a, 2) += kti(3 * c + 2);
    }
    if (vflag_atom)
      for (int n = 0; n < 6; n++) vatom(a, n) += 0.5 * kv(6 * c + n);
  }
}

namespace LAMMPS_NS {
template class PairLSDEMKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairLSDEMKokkos<LMPHostType>;
#endif
}
