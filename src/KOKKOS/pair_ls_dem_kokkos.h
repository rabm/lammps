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
PairStyle(ls/dem/kk,PairLSDEMKokkos<LMPDeviceType>);
PairStyle(ls/dem/kk/device,PairLSDEMKokkos<LMPDeviceType>);
PairStyle(ls/dem/kk/host,PairLSDEMKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_LS_DEM_KOKKOS_H
#define LMP_PAIR_LS_DEM_KOKKOS_H

#include "pair_ls_dem.h"
#include "pair_kokkos.h"
#include "kokkos_type.h"

#include <unordered_map>

namespace LAMMPS_NS {

struct TagPairLSDEMSelect {};
struct TagPairLSDEMContact {};
struct TagPairLSDEMGather {};

/* ----------------------------------------------------------------------
   device data and kernels of pair ls/dem/kk, kept apart from the pair style
   so a kernel launch copies only views and scalars, not the host containers
------------------------------------------------------------------------- */

template <class DeviceType>
struct PairLSDEMKernelsKokkos {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  struct params_lsdem {
    KOKKOS_INLINE_FUNCTION
    params_lsdem() {kn=kt=mu=etan=etat=knp=etan1=decayn1=etat1=decayt1=0;}
    KOKKOS_INLINE_FUNCTION
    params_lsdem(int /*i*/) {kn=kt=mu=etan=etat=knp=etan1=decayn1=etat1=decayt1=0;}
    KK_FLOAT kn, kt, mu, etan, etat, knp, etan1, decayn1, etat1, decayt1;
  };
  typedef Kokkos::DualView<params_lsdem**, Kokkos::LayoutRight, DeviceType> tdual_params;

  typename AT::t_kkfloat_1d_3_lr x;
  typename AT::t_kkfloat_1d_3 v;
  typename AT::t_kkfloat_1d_3 omega;
  typename AT::t_kkfloat_1d_3 xcom;
  typename AT::t_kkfloat_1d_4 quat;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_kkacc_1d_3 torque;
  typename AT::t_int_1d type;
  typename AT::t_tagint_1d tag;

  // shear history of the nodes

  typename AT::t_int_2d_lr d_ivector;
  typename AT::t_kkfloat_2d d_dvector;
  typename AT::t_double_2d_lr d_hist_n, d_hist_fs;
  int touch_row, fn1_row, fs1_row;

  typename tdual_params::t_dev params;

  // body and level-set grid of each local and ghost atom, refilled at reneighboring
  //   gstyle: -1 = no grid, 0 = grid of a grain (distributed), 1 = shared grid (global)
  //   gngrid, gmin: 3 values per atom
  //   goff: offset of the grid values in dgrid (distributed) or ggrid (global)
  //   watershed storage: the bins of the node from goff on, gnws of them, ordered by bin,
  //     gwkey (global) or dwkey (distributed) is 2 * bin, + 1 if the node owns the bin

  Kokkos::View<KK_FLOAT *, DeviceType> vol, area;
  Kokkos::View<int *, DeviceType> bid, bidx;
  Kokkos::View<int *, DeviceType> gstyle, gngrid;
  Kokkos::View<KK_FLOAT *, DeviceType> gmin, gstride, gscale;
  Kokkos::View<bigint *, DeviceType> goff;
  Kokkos::View<KK_FLOAT *, DeviceType> dgrid, ggrid;
  Kokkos::View<int *, DeviceType> gnws, dwkey, gwkey;

  // segments of PairLSDEM::rep_segs and their candidates, flattened
  //   seg_flag: 1 = tie, 2 = dup

  //   seg_kmin: slot of the candidate with a key and the lowest atom index
  Kokkos::View<int *, DeviceType> seg_rep, seg_flag, seg_cbeg, seg_kmin;
  //   cand_pk: 2 * key, + 1 if the node holds the pair in its list (key -1: pair of a ghost)
  Kokkos::View<int *, DeviceType> cand_idx, cand_pk, cand_on;

  // watershed storage: segment of each candidate, bin and grid position of the node of
  //   each segment in the grid of the other grain

  Kokkos::View<int *, DeviceType> cand_seg, seg_bin;
  Kokkos::View<KK_FLOAT *, DeviceType> seg_xl;

  // gbound: distance from the center of the grain of each candidate beyond which its level set
  //   is positive, seg_bsq: square of the largest one of the candidates of each segment
  //   (-1: no contact anywhere, BIG: not known)

  Kokkos::View<KK_FLOAT *, DeviceType> gbound, seg_bsq;

  // nodes with segments and their candidates in the order of the atoms of their pair

  Kokkos::View<int *, DeviceType> rep_atom, rep_kbeg, rep_kslot;

  // marks of the current step instead of flags that are cleared at each step, so the work of a
  //   step depends on its contacts, not on all candidates:
  //   cand_on of a candidate that is a contact, rep_on of a node with one (seg_q: node of each
  //   segment), atom_on of an atom with a force, kstamp of a key with a contact in kon
  //   stamp: mark of the current step, 0 is no mark
  Kokkos::View<int *, DeviceType> seg_q, rep_on, atom_on, kstamp;
  int stamp;

  // force on atom i, torques on atoms i and j and virial of each contact that applied a force,
  //   in the order the contacts were found, kon: index of the contact of each key marked in kstamp

  //   kij: atoms i and j of each contact, the virial is summed on the host in their order

  Kokkos::View<KK_FLOAT *, DeviceType> kf, kti, ktj, kv;
  Kokkos::View<int *, DeviceType> kij;
  Kokkos::View<int *, DeviceType> kon;
  Kokkos::View<int, DeviceType> ncontact;
  typename AT::t_kkacc_1d_6 vatom;

  // largest overlap of the contacts in this run, see PairLSDEM::check_overlap()

  Kokkos::View<KK_FLOAT, DeviceType> maxu;

  // contacts of each local atom in the order of the atoms of their pair:
  //   2 * key, + 1 if the atom is j

  Kokkos::View<int *, DeviceType> gat_beg, gat_ent;

  // errors and warnings found on the device
  //   flag[0]: atom without body, flag[1]: shear history warnings, flag[2]: first warning stored
  //   flag_tag[0]: atom without body, flag_tag[1-4]: node, its body, partner node, partner body

  Kokkos::View<int *, DeviceType> flag;
  Kokkos::View<tagint *, DeviceType> flag_tag;

  int nlocal, dimension, vflag_either, vflag_atom, watershed;
  KK_FLOAT maxcutsq, dt;
  int triclinic, xperiodic, yperiodic, zperiodic;
  KK_FLOAT xprd, yprd, zprd, xprd_half, yprd_half, zprd_half, xy, xz, yz;

  // closest node of each segment, marks the candidates that are contacts

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairLSDEMSelect, const int) const;

  // contacts of one node in the order of the atoms of their pair, which updates its shear history

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairLSDEMContact, const int) const;

  // forces and torques on one local atom summed in the order of the atoms of their pairs

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairLSDEMGather, const int) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void minimum_image(KK_FLOAT &, KK_FLOAT &, KK_FLOAT &) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool use_nodes_of_i(int, int) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void grid_position(int, int, KK_FLOAT *) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  KK_FLOAT ls_value(int, int, KK_FLOAT *) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  int ws_find(int, int, int, KK_FLOAT &) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  KK_FLOAT ls_global_frame(int, KK_FLOAT, KK_FLOAT *) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  KK_FLOAT ls_value_watershed(int, int, KK_FLOAT *) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void contact(int, int, int, int, int) const;
};

/* ---------------------------------------------------------------------- */

template <class DeviceType>
class PairLSDEMKokkos : public PairLSDEM {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef PairLSDEMKernelsKokkos<DeviceType> Kernels;

  PairLSDEMKokkos(class LAMMPS *);
  ~PairLSDEMKokkos() override;
  void compute(int, int) override;
  void init_style() override;
  void setup() override;

  // public so its device lambdas can be compiled for CUDA
  void build_segments();
  void clear_marks();

 protected:
  typename AT::t_kkfloat_1d_3_lr x;
  typename AT::t_kkacc_1d_3 f;

  Kernels kern;
  typename Kernels::tdual_params k_params;

  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;


  int nseg, ncand, nrep, nkey;

  // global grids or watershed tables already on the device by their host address,
  //   with their offset and number of watershed bins

  std::unordered_map<const void *, std::pair<bigint, int>> ggrid_offset;
  std::vector<double> h_ggrid;
  std::vector<int> h_gwkey;

  // work arrays of build_segments(): grain offset and candidate flag of each atom

  Kokkos::View<int *, DeviceType> d_boff, d_partner;

  void allocate() override;

  friend void pair_virial_fdotr_compute<PairLSDEMKokkos>(PairLSDEMKokkos *);
};

}

#endif
#endif
