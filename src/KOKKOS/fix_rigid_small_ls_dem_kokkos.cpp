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

#include "fix_rigid_small_ls_dem_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm_kokkos.h"
#include "error.h"
#include "math_extra_kokkos.h"
#include "rigid_const.h"

#include <Kokkos_NestedSort.hpp>

using namespace LAMMPS_NS;
using namespace RigidConst;

static constexpr uint64_t NODE_MASK = XCOM_MASK | QUAT_MASK | OMEGA_MASK;

/* ----------------------------------------------------------------------
   send the communication of this fix through the host packers while it lives
------------------------------------------------------------------------- */

namespace {
struct HostComm {
  Fix *fix;
  int forward, reverse;
  HostComm(Fix *f) : fix(f), forward(f->forward_comm_device), reverse(f->reverse_comm_device)
  {
    fix->forward_comm_device = fix->reverse_comm_device = 0;
  }
  ~HostComm()
  {
    fix->forward_comm_device = forward;
    fix->reverse_comm_device = reverse;
  }
};
}    // namespace

/* ---------------------------------------------------------------------- */

template <class DeviceType>
FixRigidSmallLSDEMKokkos<DeviceType>::FixRigidSmallLSDEMKokkos(LAMMPS *lmp, int narg, char **arg) :
    Parent(lmp, narg, arg), host_phase(false)
{
  // the level-set data moves with the atoms on the host

  this->exchange_comm_device = 0;
  this->sort_device = 0;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::init()
{
  Parent::init();
  this->exchange_comm_device = 0;
  this->sort_device = 0;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::setup_pre_neighbor()
{
  host_phase = true;
  {
    HostComm hc(this);
    Parent::setup_pre_neighbor();
  }
  host_phase = false;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::setup(int vflag)
{
  KKBase::setup(vflag);
  upload_ls();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::setup_pre_force(int vflag)
{
  pre_force_host(vflag);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::initial_integrate(int vflag)
{
  // quatd2g does not change, so the ghost bodies keep the copy they got at reneighboring

  KKBase::initial_integrate(vflag);
  copy_bodies_to_nodes();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::pre_neighbor()
{
  {
    HostComm hc(this);
    Parent::pre_neighbor();
  }
  if (!host_phase && this->setupflag) upload_ls();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::pre_force(int vflag)
{
  if (host_phase || !this->setupflag) {
    pre_force_host(vflag);
    return;
  }

  this->atomKK->sync(this->execution_space, NODE_MASK);
  this->k_bodytag.template sync<DeviceType>();
  this->commflag_ls = Parent::PREFORCE_LS;
  this->commKK->template forward_comm_device<DeviceType>(this, 11);
  this->commflag_ls = Parent::PARENT;
  this->k_bodytag.template modify<DeviceType>();
  this->atomKK->modified(this->execution_space, NODE_MASK);

  reset_atom2body_ghost_kokkos();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::pre_force_host(int vflag)
{
  this->atomKK->sync(Host, NODE_MASK | MASK_MASK);
  this->k_bodytag.sync_host();
  this->k_bodyown.sync_host();
  this->k_atom2body.sync_host();
  {
    HostComm hc(this);
    Parent::pre_force(vflag);
  }
  this->atomKK->modified(Host, NODE_MASK);
  this->k_bodytag.modify_host();
  this->k_atom2body.modify_host();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::post_force(int /*vflag*/)
{
  if (this->earlyflag) compute_forces_and_torques_ls();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::final_integrate()
{
  if (!this->earlyflag) compute_forces_and_torques_ls();

  // the forces and torques are summed already, so the base class must not sum them again

  const int earlyflag = this->earlyflag;
  this->earlyflag = 1;
  KKBase::final_integrate();
  this->earlyflag = earlyflag;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::reset_atom2body()
{
  if (host_phase)
    FixRigidSmall::reset_atom2body();
  else
    KKBase::reset_atom2body();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
int *FixRigidSmallLSDEMKokkos<DeviceType>::get_atom2body_array()
{
  this->k_atom2body.sync_host();
  return this->atom2body;
}

/* ----------------------------------------------------------------------
   copy quatd2g of the owned and ghost bodies to the device and list the owned atoms
     of each body in atom index order
------------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::upload_ls()
{
  const int nbody = this->nlocal_body + this->nghost_body;
  if ((this->nlocal_bodyLS != this->nlocal_body) || (this->nghost_bodyLS != this->nghost_body))
    this->error->one(FLERR, "Fix {} has {} level-set bodies but {} rigid bodies", this->style,
                     this->nlocal_bodyLS + this->nghost_bodyLS, nbody);

  if (k_quatd2g.extent_int(0) < nbody)
    k_quatd2g = DAT::tdual_double_2d_lr("rigid/small/ls/dem/kk:quatd2g", this->nmax_body, 4);
  auto h_quatd2g = k_quatd2g.view_host();
  for (int ibody = 0; ibody < nbody; ibody++)
    for (int k = 0; k < 4; k++) h_quatd2g(ibody, k) = this->bodyLS[ibody].quatd2g[k];
  k_quatd2g.modify_host();
  k_quatd2g.template sync<DeviceType>();

  const int nlocal = this->atom->nlocal;
  if (d_bstart.extent_int(0) < nbody + 1) {
    d_bstart = typename AT::t_int_1d("rigid/small/ls/dem/kk:bstart", this->nmax_body + 1);
    d_bcount = typename AT::t_int_1d("rigid/small/ls/dem/kk:bcount", this->nmax_body);
  }
  if (d_batom.extent_int(0) < MAX(nlocal, 1))
    d_batom = typename AT::t_int_1d("rigid/small/ls/dem/kk:batom", this->atom->nmax);

  this->atomKK->sync(this->execution_space, MASK_MASK);
  this->k_atom2body.template sync<DeviceType>();
  auto d_mask = this->atomKK->k_mask.template view<DeviceType>();
  auto d_atom2body = this->k_atom2body.template view<DeviceType>();
  auto bstart = d_bstart;
  auto bcount = d_bcount;
  auto batom = d_batom;
  const int groupbit = this->groupbit;
  typedef Kokkos::RangePolicy<DeviceType> Range;

  Kokkos::deep_copy(bcount, 0);
  Kokkos::parallel_for(Range(0, nlocal), KOKKOS_LAMBDA(const int i) {
    if ((d_mask(i) & groupbit) && (d_atom2body(i) >= 0))
      Kokkos::atomic_inc(&bcount(d_atom2body(i)));
  });
  Kokkos::parallel_scan(Range(0, nbody + 1), KOKKOS_LAMBDA(const int ibody, int &upd, const bool fin) {
    if (fin) bstart(ibody) = upd;
    if (ibody < nbody) upd += bcount(ibody);
  });
  Kokkos::deep_copy(bcount, 0);
  Kokkos::parallel_for(Range(0, nlocal), KOKKOS_LAMBDA(const int i) {
    if ((d_mask(i) & groupbit) && (d_atom2body(i) >= 0)) {
      const int ibody = d_atom2body(i);
      batom(bstart(ibody) + Kokkos::atomic_fetch_add(&bcount(ibody), 1)) = i;
    }
  });
  typedef Kokkos::TeamPolicy<DeviceType> Team;
  typedef typename Team::member_type TeamMember;
  Kokkos::parallel_for(Team(nbody, Kokkos::AUTO), KOKKOS_LAMBDA(const TeamMember &t) {
    const int ibody = t.league_rank();
    const int b = bstart(ibody);
    const int e = bstart(ibody + 1);
    if (e - b > 1)
      Kokkos::Experimental::sort_team(t, Kokkos::subview(batom, Kokkos::make_pair(b, e)));
  });
}

/* ----------------------------------------------------------------------
   body position, orientation in the level-set grid frame and angular velocity on the owned nodes
------------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::copy_bodies_to_nodes()
{
  auto atomKK = this->atomKK;
  atomKK->sync(this->execution_space, NODE_MASK | MASK_MASK);
  this->k_atom2body.template sync<DeviceType>();
  k_quatd2g.template sync<DeviceType>();

  auto d_mask = atomKK->k_mask.template view<DeviceType>();
  auto d_xcom = atomKK->k_xcom.template view<DeviceType>();
  auto d_quat = atomKK->k_quat.template view<DeviceType>();
  auto d_omega = atomKK->k_omega.template view<DeviceType>();
  auto d_atom2body = this->k_atom2body.template view<DeviceType>();
  auto d_quatd2g = k_quatd2g.template view<DeviceType>();
  auto d_body = this->d_body;
  const int groupbit = this->groupbit;

  int nbad = 0;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0, this->atom->nlocal),
                          KOKKOS_LAMBDA(const int i, int &bad) {
    if (!(d_mask(i) & groupbit)) return;
    const int ibody = d_atom2body(i);
    if (ibody < 0) {
      bad++;
      return;
    }
    auto &b = d_body(ibody);
    double q[4], d2g[4], g[4];
    for (int k = 0; k < 4; k++) {
      q[k] = b.quat[k];
      d2g[k] = d_quatd2g(ibody, k);
    }
    MathExtraKokkos::quatquat(q, d2g, g);
    for (int k = 0; k < 3; k++) {
      d_xcom(i, k) = b.xcm[k];
      d_omega(i, k) = b.omega[k];
    }
    for (int k = 0; k < 4; k++) d_quat(i, k) = g[k];
  }, nbad);
  atomKK->modified(this->execution_space, NODE_MASK);

  if (nbad) this->error->one(FLERR, "Fix {} has {} atoms without a body", this->style, nbad);
}

/* ----------------------------------------------------------------------
   body of the ghost atoms whose body owner is a ghost or owned atom, else -1
------------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::reset_atom2body_ghost_kokkos()
{
  auto atomKK = this->atomKK;
  atomKK->sync(this->execution_space, MASK_MASK);
  this->k_bodyown.template sync<DeviceType>();
  this->k_atom2body.template sync<DeviceType>();

  auto d_mask = atomKK->k_mask.template view<DeviceType>();
  auto d_bodytag = this->k_bodytag.template view<DeviceType>();
  auto d_bodyown = this->k_bodyown.template view<DeviceType>();
  auto d_atom2body = this->k_atom2body.template view<DeviceType>();
  const int groupbit = this->groupbit;

  const int map_style = this->atom->map_style;
  decltype(atomKK->k_map_array) k_map_array;
  decltype(atomKK->k_map_hash) k_map_hash;
  if (map_style == Atom::MAP_ARRAY) {
    k_map_array = atomKK->k_map_array;
    k_map_array.template sync<DeviceType>();
  } else if (map_style == Atom::MAP_HASH) {
    k_map_hash = atomKK->k_map_hash;
    k_map_hash.template sync<DeviceType>();
  }

  const int nlocal = this->atom->nlocal;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(nlocal, nlocal + this->atom->nghost),
                       KOKKOS_LAMBDA(const int i) {
    if (!(d_mask(i) & groupbit)) return;
    d_atom2body(i) = -1;
    if (d_bodytag(i)) {
      const int iowner =
          AtomKokkos::map_kokkos<DeviceType>(d_bodytag(i), map_style, k_map_array, k_map_hash);
      if (iowner != -1) d_atom2body(i) = d_bodyown(iowner);
    }
  });
  this->k_atom2body.template modify<DeviceType>();
}

/* ----------------------------------------------------------------------
   force and torque on each body from its owned nodes, as in
     FixRigidSmallLSDEMT::compute_forces_and_torques(): the torque on a body is the sum
     of the node torques set by pair ls/dem, without the torque of the node forces
------------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::compute_forces_and_torques_ls()
{
  auto atomKK = this->atomKK;
  atomKK->sync(this->execution_space, F_MASK | TORQUE_MASK | MASK_MASK | TYPE_MASK | RMASS_MASK);
  atomKK->k_mass.template sync<DeviceType>();

  auto d_f = atomKK->k_f.template view<DeviceType>();
  auto d_torque = atomKK->k_torque.template view<DeviceType>();
  auto d_mask = atomKK->k_mask.template view<DeviceType>();
  auto d_type = atomKK->k_type.template view<DeviceType>();
  auto d_rmass = atomKK->k_rmass.template view<DeviceType>();
  auto d_mass = atomKK->k_mass.template view<DeviceType>();
  const int rmass_flag = (this->atom->rmass != nullptr);
  auto d_body = this->d_body;
  auto bstart = d_bstart;
  auto batom = d_batom;

  const int gravity = (this->id_gravity != nullptr);
  const int grav_bit = this->grav_group_bit;
  const double gx = gravity ? this->gvec[0] : 0.0;
  const double gy = gravity ? this->gvec[1] : 0.0;
  const double gz = gravity ? this->gvec[2] : 0.0;
  typedef Kokkos::RangePolicy<DeviceType> Range;

  Kokkos::parallel_for(Range(0, this->nlocal_body + this->nghost_body), KOKKOS_LAMBDA(const int ibody) {
    double fcm[3] = {0.0, 0.0, 0.0};
    double tcm[3] = {0.0, 0.0, 0.0};
    const int beg = bstart(ibody), end = bstart(ibody + 1);
    for (int k = beg; k < end; k++) {
      const int i = batom(k);
      for (int d = 0; d < 3; d++) {
        fcm[d] += static_cast<double>(d_f(i, d));
        tcm[d] += static_cast<double>(d_torque(i, d));
      }
    }

    // subtract gravity forces from any atoms

    if (gravity) {
      for (int k = beg; k < end; k++) {
        const int i = batom(k);
        if (!(d_mask(i) & grav_bit)) continue;
        const double massone = rmass_flag ? static_cast<double>(d_rmass(i))
                                          : static_cast<double>(d_mass(d_type(i)));
        fcm[0] -= gx * massone;
        fcm[1] -= gy * massone;
        fcm[2] -= gz * massone;
      }
    }

    auto &b = d_body(ibody);
    for (int d = 0; d < 3; d++) {
      b.fcm[d] = fcm[d];
      b.torque[d] = tcm[d];
    }
  });

  // reverse communicate fcm, torque of all bodies

  this->commflag = FORCE_TORQUE;
  this->commKK->template reverse_comm_device<DeviceType>(this, 6);

  // add gravity force to COM of each body

  if (gravity) {
    Kokkos::parallel_for(Range(0, this->nlocal_body), KOKKOS_LAMBDA(const int ibody) {
      auto &b = d_body(ibody);
      if (!(d_mask(b.ilocal) & grav_bit)) return;
      b.fcm[0] += gx * b.mass;
      b.fcm[1] += gy * b.mass;
      b.fcm[2] += gz * b.mass;
    });
  }
}

/* ----------------------------------------------------------------------
   PREFORCE_LS: body tag, position, angular velocity and orientation of the nodes
------------------------------------------------------------------------- */

template <class DeviceType>
int FixRigidSmallLSDEMKokkos<DeviceType>::pack_forward_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                                                  DAT::tdual_double_1d &k_buf,
                                                                  int pbc_flag, int *pbc)
{
  if (this->commflag_ls != Parent::PREFORCE_LS)
    return KKBase::pack_forward_comm_kokkos(n, k_sendlist, k_buf, pbc_flag, pbc);

  auto d_sendlist = k_sendlist.template view<DeviceType>();
  auto d_buf = k_buf.template view<DeviceType>();
  auto d_bodytag = this->k_bodytag.template view<DeviceType>();
  auto d_xcom = this->atomKK->k_xcom.template view<DeviceType>();
  auto d_quat = this->atomKK->k_quat.template view<DeviceType>();
  auto d_omega = this->atomKK->k_omega.template view<DeviceType>();

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int ii) {
    const int j = d_sendlist(ii);
    int m = 11 * ii;
    d_buf(m++) = d_ubuf(d_bodytag(j)).d;
    for (int k = 0; k < 3; k++) d_buf(m++) = d_xcom(j, k);
    for (int k = 0; k < 3; k++) d_buf(m++) = d_omega(j, k);
    for (int k = 0; k < 4; k++) d_buf(m++) = d_quat(j, k);
  });
  return 11 * n;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first,
                                                                     DAT::tdual_double_1d &k_buf)
{
  if (this->commflag_ls != Parent::PREFORCE_LS) {
    KKBase::unpack_forward_comm_kokkos(n, first, k_buf);
    return;
  }

  auto d_buf = k_buf.template view<DeviceType>();
  auto d_bodytag = this->k_bodytag.template view<DeviceType>();
  auto d_xcom = this->atomKK->k_xcom.template view<DeviceType>();
  auto d_quat = this->atomKK->k_quat.template view<DeviceType>();
  auto d_omega = this->atomKK->k_omega.template view<DeviceType>();

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int ii) {
    const int i = first + ii;
    int m = 11 * ii;
    d_bodytag(i) = (tagint) d_ubuf(d_buf(m++)).i;
    for (int k = 0; k < 3; k++) d_xcom(i, k) = d_buf(m++);
    for (int k = 0; k < 3; k++) d_omega(i, k) = d_buf(m++);
    for (int k = 0; k < 4; k++) d_quat(i, k) = d_buf(m++);
  });
}

namespace LAMMPS_NS {
template class FixRigidSmallLSDEMKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixRigidSmallLSDEMKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS
