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
  int cutoff_auto;           // 1 if the cutoff is estimated from the node spacing
  double auto_cut_factor;    // multiple of the node spacing used for the auto cutoff

 protected:
  double auto_node_spacing();

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
  std::vector<std::unordered_map<int, std::tuple<int, double, double, double>>> saved_bins;

  // body of each local and ghost atom, refilled when the neighbor list is rebuilt
  //   grp: 0 = not in an ls/dem group, 1 = rigid/small/ls/dem, 2 = rigid/ls/dem

  struct BodyInfo {
    double vol;
    double area;
    int bID;
    int bidx;
    char grp;
    char off;
  };
  std::vector<BodyInfo> binfo;
  bigint binfo_lastbuild;
  void cache_body_info(int);

  // closest node on each other grain, bucketed by the node of the grain whose nodes are used

  struct RepEntry {
    int pbody;
    char poff;
    int ctag;
    double rsq;
  };
  std::vector<std::vector<RepEntry>> rep_buckets;

  // tag of the closest node on grain (pbody, poff), or -1 which matches no atom
  inline int rep_winner(const std::vector<RepEntry> &b, int pbody, int poff) const
  {
    for (const auto &e : b)
      if ((e.pbody == pbody) && (e.poff == poff)) return e.ctag;
    return -1;
  }

  // neighbor nodes of a node on another grain (pbody, poff), rebuilt with the neighbor list
  //   tie: the grains have equal volume, so whose nodes are used is decided each step

  struct Seg {
    int pbody;
    char poff;
    char tie;
    std::vector<int> cand;
  };
  std::vector<std::vector<Seg>> rep_segs;
  bigint segs_lastbuild;
  void build_rep_segments();
  void add_rep_candidate(int, int, int, int, int);

  void allocate();
  bool use_nodes_of_i(int, int, double, double) const;
};

}    // namespace LAMMPS_NS

#endif
#endif
