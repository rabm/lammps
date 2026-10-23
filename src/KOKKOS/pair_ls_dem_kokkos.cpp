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
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "output.h"
#include "update.h"

#include <Kokkos_NestedSort.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <algorithm>
#include <array>

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
  kern.maxu = Kokkos::View<KK_FLOAT, DeviceType>("pair:maxu");
  kern.stamp = 0;
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
  Kokkos::deep_copy(kern.maxu, 0.0);

  // half neighbor list with ghost atoms built by KOKKOS, the contact keys do not depend
  //   on its order, see build_segments()

  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> &&
                           !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::setup()
{
  PairLSDEM::setup();

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
  h_gwkey.clear();
}

/* ----------------------------------------------------------------------
   segments of PairLSDEM::build_rep_segments() built on the device from the device
   neighbor list into flat arrays, with the level-set grids of the candidate nodes,
   the contacts of each node and of each atom are ordered by the atoms of their pair
   as pair ls/dem evaluates them, so nothing depends on the order of the neighbor list
   the order of the segments of a node does not matter, each is evaluated on its own
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::build_segments()
{
  typedef Kokkos::RangePolicy<DeviceType> Range;
  typedef Kokkos::TeamPolicy<DeviceType> Team;
  typedef typename Team::member_type TeamMember;
  typedef Kokkos::View<int *, DeviceType> IView;
  const int nlocal = atom->nlocal;
  const int ntotal = nlocal + atom->nghost;
  const int inum = list->inum;
  const int allnum = list->inum + list->gnum;

  for (int a = 0; a < ntotal; a++)
    if (binfo[a].grp == 0)
      error->one(FLERR, "Atom {} does not belong to a fix rigid ls/dem group", atom->tag[a]);

  // body of each atom

  {
    std::vector<double> h_vol(ntotal), h_area(ntotal);
    std::vector<int> h_bid(ntotal), h_bidx(ntotal), h_boff(ntotal);
    for (int a = 0; a < ntotal; a++) {
      const BodyInfo &b = binfo[a];
      h_vol[a] = b.vol;
      h_area[a] = b.area;
      h_bid[a] = b.bID;
      h_bidx[a] = b.bidx;
      h_boff[a] = b.off;
    }
    upload(kern.vol, h_vol, "pair:vol");
    upload(kern.area, h_area, "pair:area");
    upload(kern.bid, h_bid, "pair:bid");
    upload(kern.bidx, h_bidx, "pair:bidx");
    upload(d_boff, h_boff, "pair:boff");
  }

  atomKK->sync(execution_space, TAG_MASK);
  auto d_tag = atomKK->k_tag.view<DeviceType>();
  auto d_vol = kern.vol;
  auto d_bid = kern.bid;
  auto d_off = d_boff;

  auto k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  auto d_ilist = k_list->d_ilist;
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  Kokkos::Array<double, 4> sp;
  for (int k = 0; k < 4; k++) sp[k] = force->special_lj[k];

  auto segkey = [] KOKKOS_FUNCTION (int r, int bid, int off) {
    return ((uint64_t) (uint32_t) r << 32) | ((uint64_t) (uint32_t) bid << 1) | (uint64_t) off;
  };

  // pairs of each atom without excluded pairs, those of local atoms are numbered in
  //   the order of the list (their key), which is only used to find their contact

  IView pbeg(Kokkos::view_alloc("pair:pbeg", Kokkos::WithoutInitializing), allnum + 1);
  bigint npair = 0;
  Kokkos::parallel_scan(Range(0, allnum), KOKKOS_LAMBDA(const int ii, bigint &upd, const bool fin) {
    const int i = d_ilist(ii);
    int n = 0;
    for (int jj = 0; jj < d_numneigh(i); jj++) n += (sp[(d_neighbors(i, jj) >> SBBITS) & 3] != 0.0);
    if (fin) pbeg(ii) = upd;
    upd += n;
  }, npair);
  if (2 * npair > MAXSMALLINT) error->one(FLERR, "Too many node pairs for pair ls/dem/kk");
  int nlocpair = 0;
  if (inum) Kokkos::deep_copy(nlocpair, Kokkos::subview(pbeg, inum));
  nkey = nlocpair;

  // a segment without a pair from the neighbor list of a local atom has no contacts,
  //   so only the segments found in those lists are kept, most pairs of ghost atoms are not:
  //   a pair of a local atom is in a kept segment, a pair of a ghost atom only if the node of
  //   that segment has a kept segment (most do not) and that one is found

  typedef Kokkos::UnorderedMap<uint64_t, void, DeviceType> LiveMap;
  LiveMap live(MAX(16 * inum, 1024));
  IView kept("pair:kept", ntotal);
  int failed = 1;
  while (failed) {
    failed = 0;
    Kokkos::parallel_reduce(Range(0, inum), KOKKOS_LAMBDA(const int ii, int &fail) {
      const int i = d_ilist(ii);
      for (int jj = 0; jj < d_numneigh(i); jj++) {
        int j = d_neighbors(i, jj);
        if (sp[(j >> SBBITS) & 3] == 0.0) continue;
        j &= NEIGHMASK;
        if (d_vol(i) <= d_vol(j)) {
          fail += live.insert(segkey(i, d_bid(j), d_off(j))).failed();
          kept(i) = 1;
        }
        if (d_vol(i) >= d_vol(j)) {
          fail += live.insert(segkey(j, d_bid(i), d_off(i))).failed();
          kept(j) = 1;
        }
      }
    }, failed);
    if (failed) live.rehash(2 * live.capacity());
  }
  auto is_kept = [=] KOKKOS_FUNCTION (int ii, int r, int c) {
    return (ii < inum) || (kept(r) && live.exists(segkey(r, d_bid(c), d_off(c))));
  };

  // candidates of each node of the kept segments, as add_rep_candidate(), filled in any order
  //   and then sorted by grain, offset and atom index, so each segment is a contiguous range

  IView rbeg("pair:rbeg", ntotal + 1);
  Kokkos::parallel_for(Range(0, allnum), KOKKOS_LAMBDA(const int ii) {
    const int i = d_ilist(ii);
    for (int jj = 0; jj < d_numneigh(i); jj++) {
      int j = d_neighbors(i, jj);
      if (sp[(j >> SBBITS) & 3] == 0.0) continue;
      j &= NEIGHMASK;
      if ((d_vol(i) <= d_vol(j)) && is_kept(ii, i, j)) Kokkos::atomic_inc(&rbeg(i));
      if ((d_vol(i) >= d_vol(j)) && is_kept(ii, j, i)) Kokkos::atomic_inc(&rbeg(j));
    }
  });
  bigint nrec = 0;
  Kokkos::parallel_scan(Range(0, ntotal + 1),
                        KOKKOS_LAMBDA(const int a, bigint &upd, const bool fin) {
    const int n = (a < ntotal) ? rbeg(a) : 0;
    if (fin) rbeg(a) = upd;
    upd += n;
  }, nrec);
  if (nrec > MAXSMALLINT) error->one(FLERR, "Too many node pairs for pair ls/dem/kk");
  ncand = nrec;

  // a view that is too small is freed before the larger one is allocated, so both never
  //   take device memory at the same time

  auto grow = [](IView &d, int n, const char *name) {
    if ((int) d.extent(0) < MAX(n, 1)) {
      d = IView();
      d = IView(Kokkos::view_alloc(std::string(name), Kokkos::WithoutInitializing), MAX(n, 1));
    }
  };
  grow(kern.cand_idx, ncand, "pair:cand_idx");
  grow(kern.cand_pk, ncand, "pair:cand_pk");
  if (watershed_flag) grow(kern.cand_seg, ncand, "pair:cand_seg");
  auto cand_idx = kern.cand_idx;
  auto cand_pk = kern.cand_pk;
  auto cand_seg = kern.cand_seg;
  const int ncand_ = ncand;

  IView rfill("pair:rfill", ntotal);
  Kokkos::parallel_for(Range(0, allnum), KOKKOS_LAMBDA(const int ii) {
    const int i = d_ilist(ii);
    int n = 0;
    for (int jj = 0; jj < d_numneigh(i); jj++) {
      int j = d_neighbors(i, jj);
      if (sp[(j >> SBBITS) & 3] == 0.0) continue;
      j &= NEIGHMASK;
      const int key = (ii < inum) ? pbeg(ii) + n : -1;
      n++;
      if ((d_vol(i) <= d_vol(j)) && is_kept(ii, i, j)) {
        const int m = rbeg(i) + Kokkos::atomic_fetch_add(&rfill(i), 1);
        cand_idx(m) = j;
        cand_pk(m) = 2 * key + 1;
      }
      if ((d_vol(i) >= d_vol(j)) && is_kept(ii, j, i)) {
        const int m = rbeg(j) + Kokkos::atomic_fetch_add(&rfill(j), 1);
        cand_idx(m) = i;
        cand_pk(m) = 2 * key;
      }
    }
  });
  rfill = IView();

  // candidates of each node sorted by grain, offset, tag and atom index (unique),
  //   so the images of a node with the same tag are next to each other;
  //   grain, offset and the low 32 bits of the tag of each atom are packed in one number,
  //   so most comparisons read one value of each candidate

  Kokkos::View<uint64_t *, DeviceType> sortkey(
      Kokkos::view_alloc("pair:sortkey", Kokkos::WithoutInitializing), MAX(ntotal, 1));
  Kokkos::parallel_for(Range(0, ntotal), KOKKOS_LAMBDA(const int a) {
    sortkey(a) = ((uint64_t) (2 * (uint32_t) d_bid(a) + d_off(a)) << 32) |
      (uint64_t) (uint32_t) d_tag(a);
  });
  // one team sorts the candidates of a node, their key and owner move with them
  Kokkos::parallel_for(Team(ntotal, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &t) {
    const int r = t.league_rank();
    const int b = rbeg(r);
    const int e = rbeg(r + 1);
    if (e - b < 2) return;
    auto less = [=](const int cx, const int cy) {
      if (sortkey(cx) != sortkey(cy)) return sortkey(cx) < sortkey(cy);
      if (d_tag(cx) != d_tag(cy)) return d_tag(cx) < d_tag(cy);
      return cx < cy;
    };
    Kokkos::Experimental::sort_by_key_team(t, Kokkos::subview(cand_idx, Kokkos::make_pair(b, e)),
                                           Kokkos::subview(cand_pk, Kokkos::make_pair(b, e)), less);
  });

  // segments: the candidates of a node on one grain

  IView rseg("pair:rseg", ntotal + 1);
  Kokkos::parallel_scan(Range(0, ntotal + 1), KOKKOS_LAMBDA(const int r, int &upd, const bool fin) {
    int n = 0;
    if (r < ntotal)
      for (int m = rbeg(r); m < rbeg(r + 1); m++)
        n += (m == rbeg(r)) || (d_bid(cand_idx(m)) != d_bid(cand_idx(m - 1))) ||
          (d_off(cand_idx(m)) != d_off(cand_idx(m - 1)));
    if (fin) rseg(r) = upd;
    upd += n;
  }, nseg);
  grow(kern.seg_rep, nseg, "pair:seg_rep");
  grow(kern.seg_flag, nseg, "pair:seg_flag");
  grow(kern.seg_cbeg, nseg + 1, "pair:seg_cbeg");
  grow(kern.seg_kmin, nseg, "pair:seg_kmin");
  auto seg_kmin = kern.seg_kmin;
  auto seg_rep = kern.seg_rep;
  auto seg_flag = kern.seg_flag;
  auto seg_cbeg = kern.seg_cbeg;
  const int nseg_ = nseg;
  const int watershed_ = watershed_flag;
  Kokkos::parallel_for(Range(0, ntotal), KOKKOS_LAMBDA(const int r) {
    int s = rseg(r) - 1;
    for (int m = rbeg(r); m < rbeg(r + 1); m++) {
      if ((m == rbeg(r)) || (d_bid(cand_idx(m)) != d_bid(cand_idx(m - 1))) ||
          (d_off(cand_idx(m)) != d_off(cand_idx(m - 1)))) {
        s++;
        seg_cbeg(s) = m;
        seg_rep(s) = r;
        seg_flag(s) = (d_vol(r) == d_vol(cand_idx(m))) ? 1 : 0;
      }
      if (watershed_) cand_seg(m) = s;
    }
    if (r == 0) seg_cbeg(nseg_) = ncand_;
  });

  // a node may appear more than once with the same tag (periodic images),
  //   a segment has one contact, or one per image of the closest node

  if ((int) d_partner.extent(0) < MAX(ntotal, 1))
    d_partner = IView("pair:partner", MAX(ntotal, 1));
  else
    Kokkos::deep_copy(d_partner, 0);
  auto partner = d_partner;
  bigint maxcontact = 0;
  Kokkos::parallel_reduce(Range(0, nseg), KOKKOS_LAMBDA(const int s, bigint &sum) {
    const int b = seg_cbeg(s);
    const int e = seg_cbeg(s + 1);
    int dup = 0, nkeyed = 0, kmin = -1;
    for (int m = b; m < e; m++) {
      nkeyed += (cand_pk(m) >= 0);
      if ((cand_pk(m) >= 0) && ((kmin < 0) || (cand_idx(m) < cand_idx(kmin)))) kmin = m;
      Kokkos::atomic_max(&partner(cand_idx(m)), 1);
      if ((m > b) && (d_tag(cand_idx(m)) == d_tag(cand_idx(m - 1)))) dup = 1;
    }
    if (dup) seg_flag(s) |= 2;
    seg_kmin(s) = kmin;
    sum += dup ? nkeyed : MIN(nkeyed, 1);
  }, maxcontact);
  if (maxcontact > MAXSMALLINT / 6) error->one(FLERR, "Too many node pairs for pair ls/dem/kk");

  // nodes with segments and their candidates in the order of the atoms of their pair

  IView rrep(Kokkos::view_alloc("pair:rrep", Kokkos::WithoutInitializing), ntotal);
  Kokkos::parallel_scan(Range(0, ntotal), KOKKOS_LAMBDA(const int r, int &upd, const bool fin) {
    const int has = (rbeg(r + 1) > rbeg(r));
    if (fin) rrep(r) = upd;
    upd += has;
  }, nrep);
  grow(kern.rep_atom, nrep, "pair:rep_atom");
  grow(kern.rep_kbeg, nrep + 1, "pair:rep_kbeg");
  auto rep_atom = kern.rep_atom;
  auto rep_kbeg = kern.rep_kbeg;
  IView rep_cbeg(Kokkos::view_alloc("pair:rep_cbeg", Kokkos::WithoutInitializing), nrep + 1);
  const int nrep_ = nrep;
  Kokkos::parallel_for(Range(0, ntotal), KOKKOS_LAMBDA(const int r) {
    if (rbeg(r + 1) > rbeg(r)) {
      rep_atom(rrep(r)) = r;
      rep_cbeg(rrep(r)) = rbeg(r);
    }
    if (r == 0) rep_cbeg(nrep_) = ncand_;
  });
  grow(kern.seg_q, nseg, "pair:seg_q");
  auto seg_q = kern.seg_q;
  Kokkos::parallel_for(Range(0, nseg), KOKKOS_LAMBDA(const int s) { seg_q(s) = rrep(seg_rep(s)); });
  rrep = IView();
  rseg = IView();
  int nkslot = 0;
  Kokkos::parallel_scan(Range(0, nrep), KOKKOS_LAMBDA(const int q, int &upd, const bool fin) {
    int n = 0;
    for (int m = rep_cbeg(q); m < rep_cbeg(q + 1); m++) n += (cand_pk(m) >= 0);
    if (fin) rep_kbeg(q) = upd;
    upd += n;
    if (fin && (q == nrep_ - 1)) rep_kbeg(nrep_) = upd;
  }, nkslot);
  grow(kern.rep_kslot, nkslot, "pair:rep_kslot");
  auto rep_kslot = kern.rep_kslot;

  // the atoms of each contact: i holds the pair in its neighbor list and is local,
  //   as i * 2^32 + j so ordering by it orders by i, then j (none if not a candidate)

  typedef Kokkos::View<uint64_t *, DeviceType> UView;
  auto pack_ij = [] KOKKOS_FUNCTION (int i, int j) {
    return ((uint64_t) (uint32_t) i << 32) | (uint64_t) (uint32_t) j;
  };
  const uint64_t NOPAIR = ~((uint64_t) 0);
  UView key_ij(Kokkos::view_alloc("pair:key_ij", Kokkos::WithoutInitializing), MAX(nkey, 1));
  Kokkos::deep_copy(key_ij, NOPAIR);
  Kokkos::parallel_for(Range(0, nrep), KOKKOS_LAMBDA(const int q) {
    const int r = rep_atom(q);
    const int kb = rep_kbeg(q);
    int n = 0;
    for (int m = rep_cbeg(q); m < rep_cbeg(q + 1); m++) {
      if (cand_pk(m) < 0) continue;
      const int c = cand_idx(m);
      rep_kslot(kb + n++) = m;
      key_ij(cand_pk(m) >> 1) = (cand_pk(m) & 1) ? pack_ij(r, c) : pack_ij(c, r);
    }
  });
  Kokkos::parallel_for(Team(nrep, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &t) {
    const int q = t.league_rank();
    const int kb = rep_kbeg(q);
    const int ke = rep_kbeg(q + 1);
    if (ke - kb < 2) return;
    auto less = [=](const int x, const int y) {
      return key_ij(cand_pk(x) >> 1) < key_ij(cand_pk(y) >> 1);
    };
    Kokkos::Experimental::sort_team(t, Kokkos::subview(rep_kslot, Kokkos::make_pair(kb, ke)), less);
  });

  // contacts of each local atom in the order of the atoms of their pair: 2 * key, + 1 if j

  grow(kern.gat_beg, nlocal + 1, "pair:gat_beg");
  auto gat_beg = kern.gat_beg;
  IView gcnt("pair:gcnt", nlocal + 1);
  Kokkos::parallel_for(Range(0, nkey), KOKKOS_LAMBDA(const int key) {
    const uint64_t kij = key_ij(key);
    if (kij == NOPAIR) return;
    const int i = kij >> 32, j = kij & 0xffffffffu;
    Kokkos::atomic_inc(&gcnt(i));
    if (j < nlocal) Kokkos::atomic_inc(&gcnt(j));
  });
  int ngat = 0;
  Kokkos::parallel_scan(Range(0, nlocal + 1), KOKKOS_LAMBDA(const int a, int &upd, const bool fin) {
    if (fin) gat_beg(a) = upd;
    upd += (a < nlocal) ? gcnt(a) : 0;
  }, ngat);
  grow(kern.gat_ent, ngat, "pair:gat_ent");
  auto gat_ent = kern.gat_ent;
  Kokkos::deep_copy(gcnt, 0);
  Kokkos::parallel_for(Range(0, nkey), KOKKOS_LAMBDA(const int key) {
    const uint64_t kij = key_ij(key);
    if (kij == NOPAIR) return;
    const int i = kij >> 32, j = kij & 0xffffffffu;
    gat_ent(gat_beg(i) + Kokkos::atomic_fetch_add(&gcnt(i), 1)) = 2 * key;
    if (j < nlocal) gat_ent(gat_beg(j) + Kokkos::atomic_fetch_add(&gcnt(j), 1)) = 2 * key + 1;
  });
  Kokkos::parallel_for(Team(nlocal, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &t) {
    const int a = t.league_rank();
    const int b = gat_beg(a);
    const int e = gat_beg(a + 1);
    if (e - b < 2) return;
    auto less = [=](const int x, const int y) { return key_ij(x >> 1) < key_ij(y >> 1); };
    Kokkos::Experimental::sort_team(t, Kokkos::subview(gat_ent, Kokkos::make_pair(b, e)), less);
  });

  auto h_partner = Kokkos::create_mirror_view_and_copy(LMPHostType::memory_space(), d_partner);

  // level-set grid of the atoms that are candidates

  std::vector<double> h_gmin(3 * ntotal, 0.0);
  std::vector<double> h_gstride(ntotal, 1.0), h_gscale(ntotal, 1.0), h_dgrid;
  std::vector<int> h_gstyle(ntotal, -1), h_gngrid(3 * ntotal, 0);
  std::vector<int> h_gnws(ntotal, 0), h_dwkey;
  std::vector<bigint> h_goff(ntotal, 0);
  std::vector<double> h_gbound(ntotal, BIG);
  int newglobal = 0;
  LSDEMExtra::GridInfo g;

  // watershed bins of a node ordered by bin, a bin of its table is not also taken from its buffer

  std::vector<std::pair<int, double>> wsbin;
  auto add_ws = [&](std::vector<int> &keys, std::vector<double> &values) {
    wsbin.clear();
    for (const auto &e : *g.ws_table) wsbin.emplace_back(e.first, e.second);
    const int nown = wsbin.size();
    for (const auto &e : *g.ws_buffer)
      if (g.ws_table->count(e.first) == 0) wsbin.emplace_back(e.first, e.second);
    for (int m = 0; m < (int) wsbin.size(); m++) {
      if ((wsbin[m].first < 0) || (wsbin[m].first > MAXSMALLINT / 2))
        error->one(FLERR, "Watershed bin {} out of range for pair ls/dem/kk", wsbin[m].first);
      wsbin[m].first = 2 * wsbin[m].first + (m < nown);
    }
    std::sort(wsbin.begin(), wsbin.end());
    for (const auto &e : wsbin) {
      keys.push_back(e.first);
      values.push_back(e.second);
    }
    return (int) wsbin.size();
  };

  for (int a = 0; a < ntotal; a++) {
    const BodyInfo &b = binfo[a];
    if (!h_partner(a) || (b.bidx < 0) || (b.grp == 0)) continue;

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
    if (watershed_flag) {
      if (!g.ws_table)
        error->one(FLERR, "Pair ls/dem/kk requires watershed storage in all fix rigid ls/dem");
      if (g.global) {
        auto it = ggrid_offset.find(g.ws_table);
        if (it == ggrid_offset.end()) {
          const bigint off = h_ggrid.size();
          const int n = add_ws(h_gwkey, h_ggrid);
          it = ggrid_offset.emplace(g.ws_table, std::make_pair(off, n)).first;
          newglobal = 1;
        }
        h_goff[a] = it->second.first;
        h_gnws[a] = it->second.second;
        auto ib = gbound_cache.find(g.ws_table);
        if (ib == gbound_cache.end())
          ib = gbound_cache.emplace(g.ws_table, LSDEMExtra::grid_bound(g)).first;
        h_gbound[a] = ib->second;
      } else {
        h_goff[a] = h_dgrid.size();
        h_gnws[a] = add_ws(h_dwkey, h_dgrid);
        h_gbound[a] = LSDEMExtra::grid_bound(g);
      }
      continue;
    }
    if (g.global) {
      auto it = ggrid_offset.find(g.values);
      if (it == ggrid_offset.end()) {
        it = ggrid_offset.emplace(g.values, std::make_pair((bigint) h_ggrid.size(), 0)).first;
        h_ggrid.insert(h_ggrid.end(), g.values, g.values + g.nvalues);
        newglobal = 1;
      }
      h_goff[a] = it->second.first;
      auto ib = gbound_cache.find(g.values);
      if (ib == gbound_cache.end())
        ib = gbound_cache.emplace(g.values, LSDEMExtra::grid_bound(g)).first;
      h_gbound[a] = ib->second;
    } else {
      h_goff[a] = h_dgrid.size();
      h_dgrid.insert(h_dgrid.end(), g.values, g.values + g.nvalues);
      h_gbound[a] = LSDEMExtra::grid_bound(g);
    }
  }

  upload(kern.gstyle, h_gstyle, "pair:gstyle");
  upload(kern.gngrid, h_gngrid, "pair:gngrid");
  upload(kern.gmin, h_gmin, "pair:gmin");
  upload(kern.gstride, h_gstride, "pair:gstride");
  upload(kern.gscale, h_gscale, "pair:gscale");
  upload(kern.goff, h_goff, "pair:goff");
  upload(kern.gbound, h_gbound, "pair:gbound");
  upload(kern.dgrid, h_dgrid, "pair:dgrid");
  if (newglobal || (kern.ggrid.extent(0) == 0)) upload(kern.ggrid, h_ggrid, "pair:ggrid");
  if (watershed_flag) {
    upload(kern.gnws, h_gnws, "pair:gnws");
    upload(kern.dwkey, h_dwkey, "pair:dwkey");
    if (newglobal || (kern.gwkey.extent(0) == 0)) upload(kern.gwkey, h_gwkey, "pair:gwkey");
    if ((int) kern.seg_bin.extent(0) < MAX(nseg, 1)) {
      kern.seg_bin = Kokkos::View<int *, DeviceType>("pair:seg_bin", MAX(nseg, 1));
      kern.seg_xl = Kokkos::View<KK_FLOAT *, DeviceType>("pair:seg_xl", 3 * MAX(nseg, 1));
    }
  }

  // largest bound of the candidates of each segment, they are nodes of one grain

  if ((int) kern.seg_bsq.extent(0) < MAX(nseg, 1))
    kern.seg_bsq = Kokkos::View<KK_FLOAT *, DeviceType>("pair:seg_bsq", MAX(nseg, 1));
  {
    auto gbound = kern.gbound;
    auto seg_bsq = kern.seg_bsq;
    auto seg_cbeg = kern.seg_cbeg;
    auto cand_idx = kern.cand_idx;
    Kokkos::parallel_for(Range(0, nseg), KOKKOS_LAMBDA(const int s) {
      KK_FLOAT bmax = -1.0;
      for (int m = seg_cbeg(s); m < seg_cbeg(s + 1); m++)
        bmax = (gbound(cand_idx(m)) > bmax) ? gbound(cand_idx(m)) : bmax;
      if (bmax < 0.0)
        seg_bsq(s) = -1.0;
      else
        seg_bsq(s) = (bmax >= BIG) ? BIG : bmax * bmax;
    });
  }

  if ((int) kern.cand_on.extent(0) < MAX(ncand, 1))
    kern.cand_on = Kokkos::View<int *, DeviceType>("pair:cand_on", MAX(ncand, 1));
  if ((int) kern.kon.extent(0) < MAX(nkey, 1)) {
    kern.kon = Kokkos::View<int *, DeviceType>("pair:kon", MAX(nkey, 1));
    kern.kstamp = Kokkos::View<int *, DeviceType>("pair:kstamp", MAX(nkey, 1));
  }
  if ((int) kern.rep_on.extent(0) < MAX(nrep, 1))
    kern.rep_on = Kokkos::View<int *, DeviceType>("pair:rep_on", MAX(nrep, 1));
  if ((int) kern.atom_on.extent(0) < MAX(nlocal, 1))
    kern.atom_on = Kokkos::View<int *, DeviceType>("pair:atom_on", MAX(nlocal, 1));
  clear_marks();
  if ((int) kern.kf.extent(0) < 3 * MAX(maxcontact, 1)) {
    kern.kf = Kokkos::View<KK_FLOAT *, DeviceType>("pair:kf", 3 * MAX(maxcontact, 1));
    kern.kti = Kokkos::View<KK_FLOAT *, DeviceType>("pair:kti", 3 * MAX(maxcontact, 1));
    kern.ktj = Kokkos::View<KK_FLOAT *, DeviceType>("pair:ktj", 3 * MAX(maxcontact, 1));
    kern.kv = Kokkos::View<KK_FLOAT *, DeviceType>("pair:kv", 6 * MAX(maxcontact, 1));
    kern.kij = Kokkos::View<int *, DeviceType>("pair:kij", 2 * MAX(maxcontact, 1));
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::clear_marks()
{
  Kokkos::deep_copy(kern.cand_on, 0);
  Kokkos::deep_copy(kern.kstamp, 0);
  Kokkos::deep_copy(kern.rep_on, 0);
  Kokkos::deep_copy(kern.atom_on, 0);
  kern.stamp = 0;
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
  kern.watershed = watershed_flag;
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
  if (kern.stamp == MAXSMALLINT) clear_marks();
  kern.stamp++;
  Kokkos::deep_copy(kern.ncontact, 0);

  // closest nodes, then the contacts of each node in the order of their pairs (they share its
  //   shear history), then the sums of the forces and torques on each atom in that order,
  //   as in pair ls/dem

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

  // global virial summed on the host in the order of the contacts of pair ls/dem,
  //   as ev_tally_xyz() does

  if (vflag_global) {
    int nc = 0;
    Kokkos::deep_copy(nc, kern.ncontact);
    if (nc) {
      auto h_kij = Kokkos::create_mirror_view_and_copy(
          LMPHostType::memory_space(), Kokkos::subview(kern.kij, std::make_pair(0, 2 * nc)));
      auto h_kv = Kokkos::create_mirror_view_and_copy(
          LMPHostType::memory_space(), Kokkos::subview(kern.kv, std::make_pair(0, 6 * nc)));
      std::vector<int> order(nc);
      for (int c = 0; c < nc; c++) order[c] = c;
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        return (h_kij(2 * a) < h_kij(2 * b)) ||
          ((h_kij(2 * a) == h_kij(2 * b)) && (h_kij(2 * a + 1) < h_kij(2 * b + 1)));
      });
      for (const int c : order) {
        for (int n = 0; n < 6; n++) virial[n] += 0.5 * h_kv(6 * c + n);
        if (h_kij(2 * c + 1) < nlocal)
          for (int n = 0; n < 6; n++) virial[n] += 0.5 * h_kv(6 * c + n);
      }
    }
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.sync_host();
  }

  // largest overlap, copied from the device only on the steps it is checked

  if (!overlap_warned && (overlap_limit > 0.0) &&
      ((update->ntimestep == output->next_thermo) || (update->ntimestep == update->laststep))) {
    KK_FLOAT h_maxu = 0.0;
    Kokkos::deep_copy(h_maxu, kern.maxu);
    maxoverlap = h_maxu;
  }
  check_overlap();

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
   position of node i in the level-set grid of the grain of atom j in units of grid cells,
   as get_bin() of fix rigid/ls/dem and rigid/small/ls/dem
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairLSDEMKernelsKokkos<DeviceType>::grid_position(int i, int j, KK_FLOAT *xl) const
{
  KK_FLOAT dx[3], qc[4], temp[4];

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
}

/* ----------------------------------------------------------------------
   level set of the grain of atom j at node i and its normal, as get_ls_value_array()
   of fix rigid/ls/dem and rigid/small/ls/dem with interpolate_LS_array()
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairLSDEMKernelsKokkos<DeviceType>::ls_value(int i, int j, KK_FLOAT *nvec) const
{
  KK_FLOAT xl[3];
  grid_position(i, j, xl);
  const KK_FLOAT stride = gstride(j);

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

  return ls_global_frame(j, dist, nvec);
}

/* ----------------------------------------------------------------------
   level set of a grid cell corner in the watershed bins of node j, ordered by bin,
   found if the node owns the bin or, if owned = 0, if the bin is next to them,
   as get_ws_ls_value() and the table lookup of interpolate_LS_watershed()
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int PairLSDEMKernelsKokkos<DeviceType>::ws_find(int j, int bin, int owned, KK_FLOAT &val) const
{
  const int global = gstyle(j);
  const int *keys = global ? (gwkey.data() + goff(j)) : (dwkey.data() + goff(j));
  int lo = 0;
  int hi = gnws(j);
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (keys[mid] < 2 * bin) lo = mid + 1;
    else hi = mid;
  }
  if ((lo == gnws(j)) || ((keys[lo] >> 1) != bin)) return 0;
  if (owned && !(keys[lo] & 1)) return 0;
  val = global ? ggrid(goff(j) + lo) : dgrid(goff(j) + lo);
  return 1;
}

/* ----------------------------------------------------------------------
   level set of the grain of atom j at the node of segment s and its normal,
   as get_ls_value_watershed() of fix rigid/ls/dem and rigid/small/ls/dem
   with interpolate_LS_watershed(), the bin of the node was found by the select kernel
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairLSDEMKernelsKokkos<DeviceType>::ls_value_watershed(int j, int s, KK_FLOAT *nvec) const
{
  KK_FLOAT xl[3];
  xl[0] = seg_xl(3 * s);
  xl[1] = seg_xl(3 * s + 1);
  xl[2] = seg_xl(3 * s + 2);
  const int mybin = seg_bin(s);
  const KK_FLOAT stride = gstride(j);

  const int ng0 = gngrid(3 * j);
  const int ng1 = gngrid(3 * j + 1);
  const int ng2 = gngrid(3 * j + 2);
  const int ix0 = int(xl[0]);
  const int ix1 = int(xl[1]);
  const int ix2 = int(xl[2]);
  KK_FLOAT dist = BIG;
  KK_FLOAT ls000, ls100, ls010, ls110, ls001, ls101, ls011, ls111;

  if ((ix0 < 0 || ix0 >= (ng0 - 1)) || (ix1 < 0 || ix1 >= (ng1 - 1)) ||
      ((dimension == 3) && (ix2 < 0 || ix2 >= (ng2 - 1)))) {
    dist = BIG;
  } else if (!ws_find(j, mybin, 1, ls000)) {
    dist = BIG;
  } else if (ls000 > 2.0 * stride) {
    dist = ls000;
  } else if (ws_find(j, mybin + 1, 0, ls100) && ws_find(j, mybin + ng0, 0, ls010) &&
             ws_find(j, mybin + 1 + ng0, 0, ls110)) {
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

    int found = 1;
    if (dimension == 3) {
      const int up = ng0 * ng1;
      found = ws_find(j, mybin + up, 0, ls001) && ws_find(j, mybin + 1 + up, 0, ls101) &&
        ws_find(j, mybin + ng0 + up, 0, ls011) && ws_find(j, mybin + 1 + ng0 + up, 0, ls111);
      if (found) {
        const KK_FLOAT lsxy1 = ls001 + xr[1] * (ls011 - ls001) +
          xr[0] * (ls101 - ls001 + xr[1] * (ls111 - ls101 - ls011 + ls001));

        dist = xr[2] * (lsxy1 - lsxy0) + lsxy0;
        nvec[0] *= 1 - xr[2];
        nvec[0] += xr[2] * (ls101 - ls001 + xr[1] * (ls111 - ls101 - ls011 + ls001));
        nvec[1] *= 1 - xr[2];
        nvec[1] += xr[2] * (ls011 - ls001 + xr[0] * (ls111 - ls101 - ls011 + ls001));
        nvec[2] = lsxy1 - lsxy0;
      } else {
        dist = BIG;
      }
    }

    if (found) {
      const KK_FLOAT val = nvec[0] * nvec[0] + nvec[1] * nvec[1] + nvec[2] * nvec[2];
      if (val > 0.0) {
        const KK_FLOAT scale = 1.0 / Kokkos::sqrt(val);
        nvec[0] *= scale;
        nvec[1] *= scale;
        nvec[2] *= scale;
      }
    }
  }

  return ls_global_frame(j, dist, nvec);
}

/* ----------------------------------------------------------------------
   level set of a shared grid scaled to grain j and the normal rotated back to the global frame
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairLSDEMKernelsKokkos<DeviceType>::ls_global_frame(int j, KK_FLOAT dist,
                                                              KK_FLOAT *nvec) const
{
  // values of a shared grid are not scaled

  if (gstyle(j)) dist *= gscale(j);

  // rotate the normal back to the global frame, it is only used for a contact (dist < 0)

  if (dist < 0.0) {
    KK_FLOAT temp[4];
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

  // grains of equal volume choose whose nodes are used by their current positions,
  //   from the candidate with a key and the lowest index, a segment without a key has no contacts

  const int first = seg_kmin(s);
  if (first < 0) return;
  if ((sflag & 1) && !use_nodes_of_i(r, cand_idx(first))) return;

  // a node beyond the bound of the grain of the candidates has no contact with it, unless
  //   it touched that grain before and its shear history has to be reset

  if (seg_bsq(s) < BIG) {
    const int c0 = cand_idx(first);
    KK_FLOAT dx = x(r, 0) - xcom(c0, 0);
    KK_FLOAT dy = x(r, 1) - xcom(c0, 1);
    KK_FLOAT dz = x(r, 2) - xcom(c0, 2);
    minimum_image(dx, dy, dz);
    if ((dx * dx + dy * dy + dz * dz > seg_bsq(s)) && (d_ivector(touch_row, r) != bid(c0)))
      return;
  }

  // watershed: the bin of the node in the grid of the other grain is found once, from that
  //   candidate, each candidate owning it is a contact

  if (watershed) {
    const int c = cand_idx(first);
    if (bidx(c) < 0) {
      if (Kokkos::atomic_fetch_add(&flag(0), 1) == 0) flag_tag(0) = tag(c);
      return;
    }
    KK_FLOAT xl[3];
    grid_position(r, c, xl);
    const int ng0 = gngrid(3 * c);
    const int ng1 = gngrid(3 * c + 1);
    const int bin = int(xl[0]) + int(xl[1]) * ng0 + int(xl[2]) * ng0 * ng1;
    seg_bin(s) = bin;
    seg_xl(3 * s) = xl[0];
    seg_xl(3 * s + 1) = xl[1];
    seg_xl(3 * s + 2) = xl[2];
    KK_FLOAT val;
    for (int m = b; m < e; m++)
      if ((cand_pk(m) >= 0) && ws_find(cand_idx(m), bin, 1, val)) {
        cand_on(m) = stamp;
        rep_on(seg_q(s)) = stamp;
      }
    return;
  }

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
    if ((win < 0) || (rsq < minrsq) || ((rsq == minrsq) && (c < cand_idx(win)))) {
      minrsq = rsq;
      win = m;
    }
  }
  if (win < 0) return;

  if (sflag & 2) {
    const tagint wtag = tag(cand_idx(win));
    for (int m = b; m < e; m++)
      if ((tag(cand_idx(m)) == wtag) && (cand_pk(m) >= 0)) {
        cand_on(m) = stamp;
        rep_on(seg_q(s)) = stamp;
      }
  } else if (cand_pk(win) >= 0) {
    cand_on(win) = stamp;
    rep_on(seg_q(s)) = stamp;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairLSDEMKernelsKokkos<DeviceType>::operator()(TagPairLSDEMContact, const int q) const
{
  if (rep_on(q) != stamp) return;
  const int r = rep_atom(q);
  for (int k = rep_kbeg(q); k < rep_kbeg(q + 1); k++) {
    const int slot = rep_kslot(k);
    if (cand_on(slot) != stamp) continue;
    const int c = cand_idx(slot);
    const int pk = cand_pk(slot);
    if (pk & 1)
      contact(r, c, 1, pk >> 1, slot);
    else
      contact(c, r, 0, pk >> 1, slot);
  }
}

/* ----------------------------------------------------------------------
   contact model of PairLSDEM::process_contact() for the candidate in slot,
   stores the force and torques of the contact by key instead of applying them
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairLSDEMKernelsKokkos<DeviceType>::contact(int i, int j, int calc_force_of_i_on_j,
                                                  int key, int slot) const
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
  const KK_FLOAT u =
    watershed ? -ls_value_watershed(par, cand_seg(slot), normal) : -ls_value(ni, par, normal);

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
  Kokkos::atomic_max(&maxu(), u);

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
  kij(2 * c) = i;
  kij(2 * c + 1) = j;
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
  kstamp(key) = stamp;
  atom_on(i) = stamp;
  if (j < nlocal) atom_on(j) = stamp;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairLSDEMKernelsKokkos<DeviceType>::operator()(TagPairLSDEMGather, const int a) const
{
  if (atom_on(a) != stamp) return;
  for (int m = gat_beg(a); m < gat_beg(a + 1); m++) {
    const int ent = gat_ent(m);
    const int key = ent >> 1;
    if (kstamp(key) != stamp) continue;
    const int c = kon(key);
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
