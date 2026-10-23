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

#include <unordered_map>

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
  int exact_order;           // 1 if results do not depend on the order of the neighbor list
  double auto_cut_factor;    // multiple of the node spacing used for the auto cutoff

 protected:
  double auto_node_spacing();

  // deepest overlap whose contacts the cutoff can still find (see init_style()), the overlap
  //   that is reported, the deepest overlap found so far on this processor, and whether it was
  double overlap_det, overlap_limit, maxoverlap;
  int overlap_warned;
  void check_overlap();

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
  class FixRigidSmallLSDEMAccess *fix_rigid_small;
  class FixRigidSmallLSDEM *fix_rigid_small_cpu;    // null for accelerated variants

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

  // neighbor nodes of a node on another grain (pbody, poff), rebuilt with the neighbor list
  //   key: number of the pair in the neighbor lists of local atoms, -1 for ghost atoms
  //   rep_owns_pair: the pair is in the neighbor list of the node (1) or of the candidate (0)
  //   tie: the grains have equal volume, so whose nodes are used is decided each step
  //   dup: a candidate tag appears more than once (periodic images)

  struct Cand {
    int idx;
    int key;
    int rep_owns_pair;
  };
  // bsq: square of the largest distance from the center of the grain of the candidates
  //   at which their level set may be negative (-1: nowhere, BIG: not known)
  struct Seg {
    int pbody;
    char poff;
    char tie;
    char dup;
    double bsq;
    std::vector<Cand> cand;
  };
  std::vector<std::vector<Seg>> rep_segs;
  std::vector<char> has_image;
  bigint segs_lastbuild;
  void build_rep_segments();

  // bounds of the level sets of the atoms, shared grids by their values
  std::unordered_map<const void *, double> gbound_cache;
  double atom_grid_bound(int);
  void add_rep_candidate(int, int, int, int, int, int, int);

  // watershed bin of a node in the grid of another grain and its grid position

  struct WsBin {
    int bin;
    double xl[3];
  };
  std::vector<WsBin> wsbins;

  // contacts of the closest nodes (array) or of the nodes owning the bin (watershed),
  //   evaluated by process_contact() in the order of i, then j
  //   i is the atom whose neighbor list holds the pair, calc = 1 if i is the node used
  //   ws: index in wsbins, -1 for array

  struct Contact {
    int i;
    int j;
    int calc;
    int key;
    int ws;
  };
  std::vector<Contact> contacts;
  inline void add_contact(int irep, const Cand &cand, int ws)
  {
    if (cand.key < 0) return;
    if (cand.rep_owns_pair)
      contacts.push_back({irep, cand.idx, 1, cand.key, ws});
    else
      contacts.push_back({cand.idx, irep, 0, cand.key, ws});
  }

  void process_contact(int, int, int, int, double *);

  virtual void allocate();
  bool use_nodes_of_i(int, int, double, double) const;
};

}    // namespace LAMMPS_NS

#endif
#endif
