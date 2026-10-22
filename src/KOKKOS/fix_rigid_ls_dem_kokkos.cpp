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

#include "fix_rigid_ls_dem_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_extra_kokkos.h"
#include "pair_ls_dem.h"

#include <vector>

using namespace LAMMPS_NS;

static constexpr uint64_t NODE_MASK = XCOM_MASK | QUAT_MASK | OMEGA_MASK;

// columns of the body state copied to the device at each step

enum {
  BXCM = 0, BVCM = 3, BOMEGA = 6, BEX = 9, BEY = 12, BEZ = 15, BQUAT = 18, BD2G = 22, NBSTATE = 26
};

// number of chunks the nodes of a body are summed in, without the exact order

static constexpr int NCHUNK = 64;

/* ---------------------------------------------------------------------- */

template <class DeviceType>
FixRigidLSDEMKokkos<DeviceType>::FixRigidLSDEMKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixRigidLSDEM(lmp, narg, arg), host_phase(false), nchunk(NCHUNK)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  // the syncs are done here, around the parts that run on the host

  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  // the per-atom arrays of fix rigid move with the atoms on the host

  exchange_comm_device = 0;
  sort_device = 0;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidLSDEMKokkos<DeviceType>::init()
{
  FixRigidLSDEM::init();
  exchange_comm_device = 0;
  sort_device = 0;

  // the nodes of a body are summed in one chunk, in atom index order as on the CPU,
  //   only with the exact order of pair ls/dem

  auto pair = dynamic_cast<PairLSDEM *>(force->pair);
  nchunk = (pair && pair->exact_order) ? 1 : NCHUNK;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidLSDEMKokkos<DeviceType>::setup(int vflag)
{
  if (extended)
    error->all(FLERR, "Fix {} does not support finite-size particles", style);

  atomKK->sync(Host, X_MASK | V_MASK | F_MASK | TORQUE_MASK | MASK_MASK | TYPE_MASK | IMAGE_MASK |
               RMASS_MASK | NODE_MASK);
  host_phase = true;
  FixRigidLSDEM::setup(vflag);
  host_phase = false;
  atomKK->modified(Host, V_MASK);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidLSDEMKokkos<DeviceType>::initial_integrate(int vflag)
{
  // the bodies on the host, the nodes on the device in set_xv()

  FixRigid::initial_integrate(vflag);
  copy_bodies_to_nodes();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidLSDEMKokkos<DeviceType>::pre_neighbor()
{
  atomKK->sync(Host, MASK_MASK | IMAGE_MASK);
  FixRigid::pre_neighbor();
  upload_atoms();
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidLSDEMKokkos<DeviceType>::set_xv()
{
  if (host_phase)
    FixRigid::set_xv();
  else
    set_xv_kokkos(1);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidLSDEMKokkos<DeviceType>::set_v()
{
  if (host_phase)
    FixRigid::set_v();
  else
    set_xv_kokkos(0);
}

/* ----------------------------------------------------------------------
   copy the per-atom data of fix rigid to the device and list the owned atoms
     of each body in atom index order
------------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidLSDEMKokkos<DeviceType>::upload_atoms()
{
  const int nlocal = atom->nlocal;
  const int nmax = MAX(atom->nmax, 1);
  int *mask = atom->mask;

  if (k_body.extent_int(0) < nmax) {
    k_body = DAT::tdual_int_1d("rigid/ls/dem/kk:body", nmax);
    k_displace = DAT::tdual_double_2d_lr("rigid/ls/dem/kk:displace", nmax, 3);
    k_xcmimage = DAT::tdual_imageint_1d("rigid/ls/dem/kk:xcmimage", nmax);
    k_batom = DAT::tdual_int_1d("rigid/ls/dem/kk:batom", nmax);
  }
  if (k_bstart.extent_int(0) < nbody + 1)
    k_bstart = DAT::tdual_int_1d("rigid/ls/dem/kk:bstart", nbody + 1);

  auto h_body = k_body.view_host();
  auto h_displace = k_displace.view_host();
  auto h_xcmimage = k_xcmimage.view_host();
  auto h_bstart = k_bstart.view_host();
  auto h_batom = k_batom.view_host();

  for (int ibody = 0; ibody <= nbody; ibody++) h_bstart(ibody) = 0;
  for (int i = 0; i < nlocal; i++) {
    h_body(i) = body[i];
    h_xcmimage(i) = xcmimage[i];
    for (int k = 0; k < 3; k++) h_displace(i, k) = displace[i][k];
    if ((body[i] >= 0) && (mask[i] & groupbit)) h_bstart(body[i] + 1)++;
  }
  for (int ibody = 0; ibody < nbody; ibody++) h_bstart(ibody + 1) += h_bstart(ibody);
  std::vector<int> fill(nbody, 0);
  for (int i = 0; i < nlocal; i++)
    if ((body[i] >= 0) && (mask[i] & groupbit)) h_batom(h_bstart(body[i]) + fill[body[i]]++) = i;

  k_body.modify_host();
  k_displace.modify_host();
  k_xcmimage.modify_host();
  k_bstart.modify_host();
  k_batom.modify_host();
  k_body.template sync<DeviceType>();
  k_displace.template sync<DeviceType>();
  k_xcmimage.template sync<DeviceType>();
  k_bstart.template sync<DeviceType>();
  k_batom.template sync<DeviceType>();
}

/* ----------------------------------------------------------------------
   copy the state of the bodies to the device
------------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidLSDEMKokkos<DeviceType>::upload_bodies()
{
  if ((k_bstate.extent_int(0) < nbody) || (k_bstate.extent_int(0) == 0))
    k_bstate = DAT::tdual_double_2d_lr("rigid/ls/dem/kk:bstate", MAX(nbody, 1), NBSTATE);

  auto h = k_bstate.view_host();
  for (int ibody = 0; ibody < nbody; ibody++) {
    for (int k = 0; k < 3; k++) {
      h(ibody, BXCM + k) = xcm[ibody][k];
      h(ibody, BVCM + k) = vcm[ibody][k];
      h(ibody, BOMEGA + k) = omega[ibody][k];
      h(ibody, BEX + k) = ex_space[ibody][k];
      h(ibody, BEY + k) = ey_space[ibody][k];
      h(ibody, BEZ + k) = ez_space[ibody][k];
    }
    for (int k = 0; k < 4; k++) {
      h(ibody, BQUAT + k) = quat[ibody][k];
      h(ibody, BD2G + k) = quatd2g[ibody][k];
    }
  }
  k_bstate.modify_host();
  k_bstate.template sync<DeviceType>();
}

/* ----------------------------------------------------------------------
   set the positions (setxflag = 1) and velocities of the atoms in the bodies,
     as FixRigid::set_xv() and FixRigid::set_v() do for point particles
------------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidLSDEMKokkos<DeviceType>::set_xv_kokkos(int setxflag)
{
  if (cvflag_atom)
    error->all(FLERR, "Fix {} does not support the per-atom centroid stress", style);

  upload_bodies();

  const int rmass_flag = (atom->rmass != nullptr);
  atomKK->sync(execution_space,
               X_MASK | V_MASK | F_MASK | TYPE_MASK | (rmass_flag ? RMASS_MASK : 0));
  if (!rmass_flag) atomKK->k_mass.template sync<DeviceType>();

  auto d_x = atomKK->k_x.template view<DeviceType>();
  auto d_v = atomKK->k_v.template view<DeviceType>();
  auto d_f = atomKK->k_f.template view<DeviceType>();
  auto d_type = atomKK->k_type.template view<DeviceType>();
  auto d_rmass = atomKK->k_rmass.template view<DeviceType>();
  auto d_mass = atomKK->k_mass.template view<DeviceType>();
  auto d_body = k_body.template view<DeviceType>();
  auto d_displace = k_displace.template view<DeviceType>();
  auto d_xcmimage = k_xcmimage.template view<DeviceType>();
  auto d_bstate = k_bstate.template view<DeviceType>();

  const int nlocal = atom->nlocal;
  const int tri = triclinic;
  const int dim2 = (domain->dimension == 2);
  const double xprd = domain->xprd;
  const double yprd = domain->yprd;
  const double zprd = domain->zprd;
  const double xy = domain->xy;
  const double xz = domain->xz;
  const double yz = domain->yz;
  const double dtf = this->dtf;
  const int ev = evflag;
  const int vglobal = vflag_global;
  const int vatomflag = vflag_atom;

  Kokkos::View<double *[6], DeviceType> d_vr;
  if (ev && vatomflag)
    d_vr = Kokkos::View<double *[6], DeviceType>("rigid/ls/dem/kk:vr", MAX(nlocal, 1));

  EV_FLOAT evsum;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0, nlocal),
                          KOKKOS_LAMBDA(const int i, EV_FLOAT &evl) {
    const int ibody = d_body(i);
    if (ibody < 0) return;

    const int xbox = (d_xcmimage(i) & IMGMASK) - IMGMAX;
    const int ybox = (d_xcmimage(i) >> IMGBITS & IMGMASK) - IMGMAX;
    const int zbox = (d_xcmimage(i) >> IMG2BITS) - IMGMAX;

    // save old positions and velocities for virial

    double x0 = 0.0, x1 = 0.0, x2 = 0.0, v0 = 0.0, v1 = 0.0, v2 = 0.0;
    if (ev) {
      if (setxflag) {
        if (tri == 0) {
          x0 = static_cast<double>(d_x(i, 0)) + xbox * xprd;
          x1 = static_cast<double>(d_x(i, 1)) + ybox * yprd;
          x2 = static_cast<double>(d_x(i, 2)) + zbox * zprd;
        } else {
          x0 = static_cast<double>(d_x(i, 0)) + xbox * xprd + ybox * xy + zbox * xz;
          x1 = static_cast<double>(d_x(i, 1)) + ybox * yprd + zbox * yz;
          x2 = static_cast<double>(d_x(i, 2)) + zbox * zprd;
        }
      }
      v0 = static_cast<double>(d_v(i, 0));
      v1 = static_cast<double>(d_v(i, 1));
      v2 = static_cast<double>(d_v(i, 2));
    }

    // displacement from the center of mass, based on the body orientation

    double delta[3];
    for (int k = 0; k < 3; k++)
      delta[k] = d_bstate(ibody, BEX + k) * d_displace(i, 0) +
          d_bstate(ibody, BEY + k) * d_displace(i, 1) + d_bstate(ibody, BEZ + k) * d_displace(i, 2);

    // v = vcm + omega around center of mass

    const double w0 = d_bstate(ibody, BOMEGA + 0);
    const double w1 = d_bstate(ibody, BOMEGA + 1);
    const double w2 = d_bstate(ibody, BOMEGA + 2);
    double vnew[3];
    vnew[0] = w1 * delta[2] - w2 * delta[1] + d_bstate(ibody, BVCM + 0);
    vnew[1] = w2 * delta[0] - w0 * delta[2] + d_bstate(ibody, BVCM + 1);
    vnew[2] = w0 * delta[1] - w1 * delta[0] + d_bstate(ibody, BVCM + 2);
    if (dim2) vnew[2] = 0.0;
    for (int k = 0; k < 3; k++) d_v(i, k) = static_cast<KK_FLOAT>(vnew[k]);
    if (dim2) delta[2] = 0.0;

    // add center of mass to displacement, mapped back into the periodic box

    if (setxflag) {
      const double xc0 = d_bstate(ibody, BXCM + 0);
      const double xc1 = d_bstate(ibody, BXCM + 1);
      const double xc2 = d_bstate(ibody, BXCM + 2);
      if (tri == 0) {
        d_x(i, 0) = static_cast<KK_FLOAT>(delta[0] + (xc0 - xbox * xprd));
        d_x(i, 1) = static_cast<KK_FLOAT>(delta[1] + (xc1 - ybox * yprd));
        d_x(i, 2) = static_cast<KK_FLOAT>(delta[2] + (xc2 - zbox * zprd));
      } else {
        d_x(i, 0) = static_cast<KK_FLOAT>(delta[0] + (xc0 - xbox * xprd - ybox * xy - zbox * xz));
        d_x(i, 1) = static_cast<KK_FLOAT>(delta[1] + (xc1 - ybox * yprd - zbox * yz));
        d_x(i, 2) = static_cast<KK_FLOAT>(delta[2] + (xc2 - zbox * zprd));
      }
    } else if (ev) {
      if (tri == 0) {
        x0 = static_cast<double>(d_x(i, 0)) + xbox * xprd;
        x1 = static_cast<double>(d_x(i, 1)) + ybox * yprd;
        x2 = static_cast<double>(d_x(i, 2)) + zbox * zprd;
      } else {
        x0 = static_cast<double>(d_x(i, 0)) + xbox * xprd + ybox * xy + zbox * xz;
        x1 = static_cast<double>(d_x(i, 1)) + ybox * yprd + zbox * yz;
        x2 = static_cast<double>(d_x(i, 2)) + zbox * zprd;
      }
    }

    // virial = unwrapped coords dotted into body constraint force, half of it here

    if (ev) {
      const double massone =
          rmass_flag ? static_cast<double>(d_rmass(i)) : static_cast<double>(d_mass(d_type(i)));
      const double fc0 =
          massone * (static_cast<double>(d_v(i, 0)) - v0) / dtf - static_cast<double>(d_f(i, 0));
      const double fc1 =
          massone * (static_cast<double>(d_v(i, 1)) - v1) / dtf - static_cast<double>(d_f(i, 1));
      const double fc2 =
          massone * (static_cast<double>(d_v(i, 2)) - v2) / dtf - static_cast<double>(d_f(i, 2));
      double vr[6];
      vr[0] = 0.5 * x0 * fc0;
      vr[1] = 0.5 * x1 * fc1;
      vr[2] = 0.5 * x2 * fc2;
      vr[3] = 0.5 * x0 * fc1;
      vr[4] = 0.5 * x0 * fc2;
      vr[5] = 0.5 * x1 * fc2;
      if (vglobal)
        for (int k = 0; k < 6; k++) evl.v[k] += vr[k];
      if (vatomflag)
        for (int k = 0; k < 6; k++) d_vr(i, k) = vr[k];
    }
  }, evsum);

  atomKK->modified(execution_space, setxflag ? (X_MASK | V_MASK) : V_MASK);

  if (ev) {
    if (vglobal)
      for (int k = 0; k < 6; k++) virial[k] += static_cast<double>(evsum.v[k]);
    if (vatomflag) {
      auto h_vr = Kokkos::create_mirror_view_and_copy(LMPHostType::memory_space(), d_vr);
      for (int i = 0; i < nlocal; i++) {
        if (body[i] < 0) continue;
        for (int k = 0; k < 6; k++) vatom[i][k] += h_vr(i, k);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   body position, orientation in the level-set grid frame and angular velocity on the nodes,
     as in FixRigidLSDEM::initial_integrate()
------------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidLSDEMKokkos<DeviceType>::copy_bodies_to_nodes()
{
  atomKK->sync(execution_space, NODE_MASK | MASK_MASK);

  auto d_mask = atomKK->k_mask.template view<DeviceType>();
  auto d_xcom = atomKK->k_xcom.template view<DeviceType>();
  auto d_quat = atomKK->k_quat.template view<DeviceType>();
  auto d_omega = atomKK->k_omega.template view<DeviceType>();
  auto d_body = k_body.template view<DeviceType>();
  auto d_bstate = k_bstate.template view<DeviceType>();
  const int groupbit = this->groupbit;

  int nbad = 0;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0, atom->nlocal),
                          KOKKOS_LAMBDA(const int i, int &bad) {
    if (!(d_mask(i) & groupbit)) return;
    const int ibody = d_body(i);
    if (ibody < 0) {
      bad++;
      return;
    }
    double q[4], d2g[4], g[4];
    for (int k = 0; k < 4; k++) {
      q[k] = d_bstate(ibody, BQUAT + k);
      d2g[k] = d_bstate(ibody, BD2G + k);
    }
    MathExtraKokkos::quatquat(q, d2g, g);
    for (int k = 0; k < 3; k++) {
      d_xcom(i, k) = static_cast<KK_FLOAT>(d_bstate(ibody, BXCM + k));
      d_omega(i, k) = static_cast<KK_FLOAT>(d_bstate(ibody, BOMEGA + k));
    }
    for (int k = 0; k < 4; k++) d_quat(i, k) = static_cast<KK_FLOAT>(g[k]);
  }, nbad);
  atomKK->modified(execution_space, NODE_MASK);

  if (nbad) error->one(FLERR, "Fix {} has {} atoms without a body", style, nbad);
}

/* ----------------------------------------------------------------------
   sum of the forces and torques of the owned nodes of each body
     each body is summed in nchunk chunks of its nodes in atom index order
------------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidLSDEMKokkos<DeviceType>::sum_forces_and_torques()
{
  atomKK->sync(execution_space, F_MASK | TORQUE_MASK);

  if ((k_bsum.extent_int(0) < nbody) || (k_bsum.extent_int(0) == 0))
    k_bsum = DAT::tdual_double_2d_lr("rigid/ls/dem/kk:bsum", MAX(nbody, 1), 6);
  if (d_bpart.extent_int(0) < 6 * nchunk * MAX(nbody, 1))
    d_bpart = typename AT::t_double_1d("rigid/ls/dem/kk:bpart", 6 * nchunk * MAX(nbody, 1));

  auto d_f = atomKK->k_f.template view<DeviceType>();
  auto d_torque = atomKK->k_torque.template view<DeviceType>();
  auto d_bstart = k_bstart.template view<DeviceType>();
  auto d_batom = k_batom.template view<DeviceType>();
  auto bpart = d_bpart;
  auto d_bsum = k_bsum.template view<DeviceType>();
  const int nc = nchunk;
  typedef Kokkos::RangePolicy<DeviceType> Range;

  Kokkos::parallel_for(Range(0, nbody * nc), KOKKOS_LAMBDA(const int n) {
    const int ibody = n / nc;
    double s[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int k = d_bstart(ibody) + n % nc; k < d_bstart(ibody + 1); k += nc) {
      const int i = d_batom(k);
      for (int d = 0; d < 3; d++) {
        s[d] += static_cast<double>(d_f(i, d));
        s[3 + d] += static_cast<double>(d_torque(i, d));
      }
    }
    for (int d = 0; d < 6; d++) bpart(6 * n + d) = s[d];
  });
  Kokkos::parallel_for(Range(0, nbody), KOKKOS_LAMBDA(const int ibody) {
    double s[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int c = 0; c < nc; c++)
      for (int d = 0; d < 6; d++) s[d] += bpart(6 * (ibody * nc + c) + d);
    for (int d = 0; d < 6; d++) d_bsum(ibody, d) = s[d];
  });
  k_bsum.template modify<DeviceType>();
  k_bsum.sync_host();
}

/* ----------------------------------------------------------------------
   as FixRigidLSDEM::compute_forces_and_torques(), with the sum over the nodes on the device
------------------------------------------------------------------------- */

template <class DeviceType>
void FixRigidLSDEMKokkos<DeviceType>::compute_forces_and_torques()
{
  if (host_phase) {
    FixRigidLSDEM::compute_forces_and_torques();
    return;
  }

  sum_forces_and_torques();
  auto h_bsum = k_bsum.view_host();
  for (int ibody = 0; ibody < nbody; ibody++)
    for (int k = 0; k < 6; k++) sum[ibody][k] = h_bsum(ibody, k);

  MPI_Allreduce(sum[0], all[0], 6 * nbody, MPI_DOUBLE, MPI_SUM, world);

  for (int ibody = 0; ibody < nbody; ibody++) {
    fcm[ibody][0] = all[ibody][0];
    fcm[ibody][1] = all[ibody][1];
    fcm[ibody][2] = all[ibody][2];
    torque[ibody][0] = all[ibody][3];
    torque[ibody][1] = all[ibody][4];
    torque[ibody][2] = all[ibody][5];
  }

  // add gravity force to COM of each body

  if (id_gravity) {
    for (int ibody = 0; ibody < nbody; ibody++) {
      if (apply_grav[ibody]) {
        fcm[ibody][0] += gvec[0] * masstotal[ibody];
        fcm[ibody][1] += gvec[1] * masstotal[ibody];
        fcm[ibody][2] += gvec[2] * masstotal[ibody];
      }
    }
  }
}

namespace LAMMPS_NS {
template class FixRigidLSDEMKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixRigidLSDEMKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS
