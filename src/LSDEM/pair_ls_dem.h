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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(ls/dem,PairLSDEM);
// clang-format on
#else

#ifndef LMP_PAIR_LS_DEM_H
#define LMP_PAIR_LS_DEM_H

#include "pair.h"

#include "fix_rigid_ls_dem.h"
#include "fix_rigid_small_ls_dem.h"

#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace LAMMPS_NS {

class PairLSDEM : public Pair {
 public:
  PairLSDEM(class LAMMPS *);
  ~PairLSDEM() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void settings(int, char **) override;
  void init_style() override;
  void setup() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;

  double maxcut;
  int cutoff_auto;   // 1 if "pair_style ls/dem auto": cutoff estimated from node spacing

 protected:
  double **kn, **kt, **mu, **etan, **etat, **knp, **cut, **decayn1, **etan1, **decayt1, **etat1;

  int groupbit_small, groupbit_large;
  int watershed_flag;

  int index_ls_dem_vol;
  int index_ls_dem_n;
  int index_ls_dem_fs;
  int index_ls_dem_touch_id;
  int index_ls_dem_fn1;
  int index_ls_dem_fs1;

  class FixRigidLSDEM *fix_rigid;
  class FixRigidSmallLSDEM *fix_rigid_small;

  // Per-atom body info, cached once per compute() over local+ghost atoms so the
  // two contact-pass loops do not recompute the group test + double-indirection
  // body/volume/area lookups for every neighbour pair (each pair touched them ~4x).
  // grp: 0 = not in an ls/dem group, 1 = small fix, 2 = large fix.
  struct BodyInfo { double vol; double area; int bID; int bidx; char grp; char off; };
  std::vector<BodyInfo> binfo;
  bigint binfo_lastbuild;   // neighbor->lastcall when binfo was last filled (-1 = never)
  void cache_body_info(int ntotal);

  std::vector<std::unordered_map<int, std::tuple<int, double, double, double>>> saved_bins;
  // Closest-partner arbitration, bucketed BY REPRESENTATIVE NODE (atom index)
  // instead of a single global std::unordered_map<long,...>. For each smaller-grain
  // (representative) node we keep a short list of its distinct partner bodies and,
  // per partner body, the closest partner node + rsq. Indexing by atom index and
  // scanning a tiny per-node list is far more cache-friendly than hashing a sparse
  // 64-bit key for every neighbour pair (the old map's build/find dominated the
  // contact pass at ~100 ns/pair). Same sweep order + update rule -> bitwise-identical
  // winner. Cleared (capacity kept) each step.
  struct RepEntry { int pbody; char poff; int ctag; double rsq; };
  std::vector<std::vector<RepEntry>> rep_buckets;
  // Tag of the winning partner node for (body,offset), or a sentinel that no real tag matches.
  inline int rep_winner(const std::vector<RepEntry> &b, int pbody, int poff) const {
    for (const auto &e : b) if (e.pbody == pbody && e.poff == poff) return e.ctag;
    return -1;
  }

  // CSR-style CANDIDATE SEGMENTS, rebuilt only when the neighbour list is rebuilt.
  // For each representative node we keep, per partner body, the list of candidate
  // partner ATOM INDICES (in neighbour-sweep order). The rep choice + partner body
  // are constant between rebuilds (grain volumes + molecule ids don't change), so
  // the per-step pass only re-reduces rsq over each segment to pick the winner ->
  // it no longer redoes the group test / rep decision / bucket find for every pair.
  // The candidate ORDER matches the old sweep, so the first-min winner is identical.
  struct Seg { int pbody; char poff; std::vector<int> cand; };
  std::vector<std::vector<Seg>> rep_segs;   // indexed by representative atom index
  bigint segs_lastbuild;                    // neighbor->lastcall when rep_segs was built
  void build_rep_segments();

  // Evaluate the contact model for ONE arbitration-winning node-grain pair and
  // apply the force/torque. Lifted out of the contact pass so a single driver can
  // call it once per winner (the step-(b) rep-local shape / GPU contact kernel).
  void process_contact(int i, int j, int calc_force_of_i_on_j, int tmp_bin, double *x_local);

  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
