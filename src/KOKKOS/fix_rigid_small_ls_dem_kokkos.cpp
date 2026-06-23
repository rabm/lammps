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
#include "domain.h"
#include "error.h"
#include "ls_dem_extra_device.h"
#include "memory_kokkos.h"
#include "rigid_const.h"

#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace RigidConst;

/* ----------------------------------------------------------------------
   Increment 2a -- residency lifecycle scaffold. See the header for the design
   and the DualView discipline that fixes the "Concurrent modification" abort.
------------------------------------------------------------------------- */

template<class DeviceType>
FixRigidSmallLSDEMKokkos<DeviceType>::FixRigidSmallLSDEMKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixRigidSmallLSDEM(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  commKK = (CommKokkos *) comm;
  body_resident_device = false;
  // execution_space = Device (2b): the per-step compute moves to device kernels so
  // x/v stay device-resident (no host round-trips = the Modify-cost win). The
  // framework (ModifyKokkos) then wraps each lifecycle method with
  // sync(Device, datamask_read) / modified(Device, datamask_modify). Methods that
  // still run HOST code (setup/setup_pre_neighbor/pre_neighbor/pre_force +, until
  // the kernels land, initial/final_integrate) therefore MUST end with a trailing
  // atomKK->sync(execution_space, datamask_modify) AFTER their modified(Host,...)
  // -- this pushes host->device and clears the host-dirty flag so the framework's
  // appended modified(Device, datamask_modify) does NOT collide ("Concurrent
  // modification ... atom:x"). (2a kept this Host to dodge that collision while it
  // had no device kernels; the trailing-push reconcile is the 2b way.)
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  // What the (host) base methods read / write each step. Used by the wrappers
  // below and by the framework to sync this fix against the device pair. The
  // grain custom-field masks (XCOM/QUAT/OMEGA/GRID_INDEX) are >32-bit, so they
  // are OR'd directly into the 64-bit datamask members (a narrower local would
  // truncate them).
  datamask_read = X_MASK | V_MASK | F_MASK | TAG_MASK | TYPE_MASK | MASK_MASK |
                  IMAGE_MASK | RMASS_MASK | TORQUE_MASK | MOLECULE_MASK |
                  XCOM_MASK | QUAT_MASK | OMEGA_MASK | GRID_INDEX_MASK;
  datamask_modify = X_MASK | V_MASK | XCOM_MASK | QUAT_MASK | OMEGA_MASK;

  // Convert the five FixRigidSmall per-atom arrays to DualViews, preserving the
  // live bodytag/bodyown the base ctor populated (atom2body/xcmimage/displace
  // are recomputed in setup_bodies_static, so no data to preserve). LSDEM's
  // bodyownLS stays a plain host array (grown again in grow_arrays); no device
  // kernel indexes it.
  tagint *bodytag_tmp = bodytag;
  int *bodyown_tmp = bodyown;
  bodytag = nullptr;
  bodyown = nullptr;
  memory->destroy(atom2body);  atom2body = nullptr;
  memory->destroy(xcmimage);   xcmimage = nullptr;
  memory->destroy(displace);   displace = nullptr;

  grow_arrays(atom->nmax);

  for (int i = 0; i < atom->nlocal; i++) {
    bodytag[i] = bodytag_tmp[i];
    bodyown[i] = bodyown_tmp[i];
  }
  k_bodytag.modify_host();
  k_bodyown.modify_host();
  memory->destroy(bodytag_tmp);
  memory->destroy(bodyown_tmp);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixRigidSmallLSDEMKokkos<DeviceType>::~FixRigidSmallLSDEMKokkos()
{
  if (copymode) return;

  // null the base-class pointers so ~FixRigidSmall does not double-free the
  // now-DualView-owned allocations. bodyownLS stays a plain array (freed by the
  // base chain exactly as in the host fix).
  memoryKK->destroy_kokkos(k_bodyown,   bodyown);     bodyown = nullptr;
  memoryKK->destroy_kokkos(k_bodytag,   bodytag);     bodytag = nullptr;
  memoryKK->destroy_kokkos(k_atom2body, atom2body);   atom2body = nullptr;
  memoryKK->destroy_kokkos(k_xcmimage,  xcmimage);    xcmimage = nullptr;
  memoryKK->destroy_kokkos(k_displace,  displace);    displace = nullptr;
}

/* ----------------------------------------------------------------------
   need PRE_EXCHANGE to pull the device-resident body data back to the host
   before atoms (and their bodies) migrate during the (host) exchange.
------------------------------------------------------------------------- */

template<class DeviceType>
int FixRigidSmallLSDEMKokkos<DeviceType>::setmask()
{
  return FixRigidSmallLSDEM::setmask() | PRE_EXCHANGE;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::pre_exchange()
{
  // bring d_body back to host body[] so the host exchange/sort migrate correct
  // body data. Touches NO atom DualView -> raises no atom flag (no collision).
  if (body_resident_device) { copy_body_to_host(); body_resident_device = false; }
}

/* ----------------------------------------------------------------------
   per-body device residency: plain Kokkos::View<Body*> + a cached host mirror
   (mirrors upstream FixRigidSmallKokkos copy_body_to_{device,host}). d_quatd2g
   carries the BodyLS quatd2g for the grain scatter, sized over the full body
   index range (what atom2body indexes), not the LS body count.
------------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::copy_body_to_device()
{
  const int n = nlocal_body + nghost_body;
  if ((int) d_body.extent(0) < n) {
    Kokkos::resize(d_body, n);
    h_body = Kokkos::create_mirror_view(d_body);
  }
  for (int i = 0; i < n; i++) {
    const Body &s = body[i];
    DBody &d = h_body(i);
    d.mass = s.mass;
    for (int k = 0; k < 3; k++) {
      d.xcm[k]=s.xcm[k]; d.xgc[k]=s.xgc[k]; d.vcm[k]=s.vcm[k]; d.fcm[k]=s.fcm[k];
      d.torque[k]=s.torque[k]; d.inertia[k]=s.inertia[k];
      d.ex_space[k]=s.ex_space[k]; d.ey_space[k]=s.ey_space[k]; d.ez_space[k]=s.ez_space[k];
      d.xgc_body[k]=s.xgc_body[k]; d.angmom[k]=s.angmom[k]; d.omega[k]=s.omega[k];
    }
    for (int k = 0; k < 4; k++) d.quat[k]=s.quat[k];
  }
  Kokkos::deep_copy(d_body, h_body);
}

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::copy_body_to_host()
{
  const int n = nlocal_body + nghost_body;
  Kokkos::deep_copy(h_body, d_body);
  for (int i = 0; i < n; i++) {
    const DBody &s = h_body(i);
    Body &d = body[i];
    d.mass = s.mass;
    for (int k = 0; k < 3; k++) {
      d.xcm[k]=s.xcm[k]; d.xgc[k]=s.xgc[k]; d.vcm[k]=s.vcm[k]; d.fcm[k]=s.fcm[k];
      d.torque[k]=s.torque[k]; d.inertia[k]=s.inertia[k];
      d.ex_space[k]=s.ex_space[k]; d.ey_space[k]=s.ey_space[k]; d.ez_space[k]=s.ez_space[k];
      d.xgc_body[k]=s.xgc_body[k]; d.angmom[k]=s.angmom[k]; d.omega[k]=s.omega[k];
    }
    for (int k = 0; k < 4; k++) d.quat[k]=s.quat[k];
  }
}

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::copy_bodyLS_to_device()
{
  // size + loop over the BODY index range (atom2body indexes d_quatd2g[ib]).
  // The LSDEM base co-indexes bodyLS[ibody] with body[ibody] (it does
  // bodyLS[atom2body[i]] in the host scatter), so this is the same domain; the
  // assert guards the FULL_BODY vs FULL_BODY_LS ghost counts staying in lockstep.
  if (nlocal_body != nlocal_bodyLS || nghost_body != nghost_bodyLS)
    error->one(FLERR, "rigid/small/ls/dem/kk: body vs bodyLS count mismatch "
               "({}+{} vs {}+{})", nlocal_body, nghost_body, nlocal_bodyLS, nghost_bodyLS);
  const int n = nlocal_body + nghost_body;
  if ((int) d_quatd2g.extent(0) < n) {
    Kokkos::resize(d_quatd2g, n);
    h_quatd2g = Kokkos::create_mirror_view(d_quatd2g);
  }
  for (int ib = 0; ib < n; ib++)
    for (int k = 0; k < 4; k++) h_quatd2g(ib, k) = bodyLS[ib].quatd2g[k];
  Kokkos::deep_copy(d_quatd2g, h_quatd2g);
}

/* ----------------------------------------------------------------------
   allocate the per-atom rigid-body arrays as DualViews (mirrors the upstream
   FixRigidSmallKokkos::grow_arrays). The host base writes these arrays through
   the aliased raw pointers without flagging the DualView, so mark host modified
   and sync to device BEFORE grow_kokkos would rebuild the host mirror from
   stale device data and drop the host-written ownership.

   The ORDER is load-bearing on a real GPU. These arrays are DEVICE-READ-ONLY
   (only the host base writes them, through the aliased raw pointers; the device
   kernels only read), so HOST is always the authoritative side. Mark host
   modified BEFORE grow_kokkos() and sync to device only AFTER. That way
   DualView::resize() sees host strictly newer and takes the HOST branch -- it
   resizes the host view in place and rebuilds the DEVICE mirror FROM host,
   preserving the host-written ownership AND leaving the modified flag on the
   host side (never the device side). The earlier "modify_host(); sync_device()
   BEFORE grow_kokkos" equalised the flags, so resize() took the DEVICE branch,
   rebuilt the host mirror from the stale device, and silently ZEROED bodyown /
   bodytag / atom2body (while the plain bodyownLS survived) -> a later
   xcm_custom[atom2body[i]==-1] segfault in process_levelsets. Pushing host->
   device only AFTER the resize keeps the flags from ever reaching the
   device-dirty state that caused the "Concurrent modification" abort.
   (All invisible on OpenMP-host where Device==Host.)
------------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::grow_arrays(int nmax)
{
  k_bodyown.modify_host();
  k_bodytag.modify_host();
  k_atom2body.modify_host();
  k_xcmimage.modify_host();
  k_displace.modify_host();

  memoryKK->grow_kokkos(k_bodyown,   bodyown,   nmax,    "rigid/small/ls/dem/kk:bodyown");
  memoryKK->grow_kokkos(k_bodytag,   bodytag,   nmax,    "rigid/small/ls/dem/kk:bodytag");
  memoryKK->grow_kokkos(k_atom2body, atom2body, nmax,    "rigid/small/ls/dem/kk:atom2body");
  memoryKK->grow_kokkos(k_xcmimage,  xcmimage,  nmax,    "rigid/small/ls/dem/kk:xcmimage");
  memoryKK->grow_kokkos(k_displace,  displace,  nmax, 3, "rigid/small/ls/dem/kk:displace");

  // refresh the device copy FROM the resized host, then re-grab the device views
  k_bodyown.sync_device();   d_bodyown   = k_bodyown.view<DeviceType>();
  k_bodytag.sync_device();   d_bodytag   = k_bodytag.view<DeviceType>();
  k_atom2body.sync_device(); d_atom2body = k_atom2body.view<DeviceType>();
  k_xcmimage.sync_device();  d_xcmimage  = k_xcmimage.view<DeviceType>();
  k_displace.sync_device();  d_displace  = k_displace.view<DeviceType>();

  // LSDEM extra per-atom array: plain host allocation (no device kernel indexes
  // it -- the device reduction/kernels use atom2body).
  memory->grow(bodyownLS, nmax, "rigid/small/ls/dem/kk:bodyownLS");

  // extended-particle arrays (host-only) + per-atom virial, exactly as the base
  if (extended) {
    memory->grow(eflags, nmax, "rigid/small:eflags");
    if (orientflag) memory->grow(orient, nmax, orientflag, "rigid/small:orient");
    if (dorientflag) memory->grow(dorient, nmax, 3, "rigid/small:dorient");
  }
  if (nmax > maxvatom) {
    maxvatom = atom->nmax;
    memory->grow(vatom, maxvatom, 6, "fix:vatom");
  }
}

/* ----------------------------------------------------------------------
   push the per-atom DualViews (written on the host) to the device. The host
   base just wrote them via raw pointers, so flag host-modified, push, and
   re-grab the device views. Subsumes inc-1's upload_atom2body(); runs on the
   reneighbor cadence (setup / setup_pre_neighbor / pre_neighbor).
------------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::sync_peratom_to_device()
{
  // Host is authoritative (these arrays are device-read-only); the host base
  // just rewrote them via raw pointers (reset_atom2body, the body comm), so
  // mark host modified and push to the device read-cache. grow_arrays already
  // left the flags synced, so modify_host() here never reaches the device-dirty
  // state -> no "Concurrent modification" abort.
  k_bodyown.modify_host();   k_bodyown.sync_device();   d_bodyown   = k_bodyown.view<DeviceType>();
  k_bodytag.modify_host();   k_bodytag.sync_device();   d_bodytag   = k_bodytag.view<DeviceType>();
  k_atom2body.modify_host(); k_atom2body.sync_device(); d_atom2body = k_atom2body.view<DeviceType>();
  k_xcmimage.modify_host();  k_xcmimage.sync_device();  d_xcmimage  = k_xcmimage.view<DeviceType>();
  k_displace.modify_host();  k_displace.sync_device();  d_displace  = k_displace.view<DeviceType>();
}

/* ----------------------------------------------------------------------
   lifecycle hooks (2b STEP 1+2 -- framework pivot to execution_space=Device,
   compute STILL host-fallback). These host-fallback methods read/write atom
   data on the HOST, so each: sync(Host, datamask_read) in, run the host base,
   modified(Host, datamask_modify), then a TRAILING sync(execution_space,
   datamask_modify) that pushes host->device and clears the host-dirty flag --
   so the framework's wrapper modified(Device, datamask_modify) (appended by
   ModifyKokkos because execution_space=Device) finds the view device-current
   and does NOT abort with "Concurrent modification". The 2b device kernels
   (STEP 4-6) replace the host bodies of initial/final_integrate and drop their
   host wrappers (the kernels mark modified(Device) directly).
------------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::setup(int vflag)
{
  atomKK->sync(Host, datamask_read);
  FixRigidSmallLSDEM::setup(vflag);
  atomKK->modified(Host, datamask_modify);
  atomKK->sync(execution_space, datamask_modify);
  sync_peratom_to_device();
  copy_body_to_device(); copy_bodyLS_to_device(); body_resident_device = true;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::setup_pre_neighbor()
{
  atomKK->sync(Host, datamask_read);
  FixRigidSmallLSDEM::setup_pre_neighbor();
  atomKK->modified(Host, datamask_modify);
  atomKK->sync(execution_space, datamask_modify);
  sync_peratom_to_device();
  copy_body_to_device(); copy_bodyLS_to_device(); body_resident_device = true;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::pre_neighbor()
{
  atomKK->sync(Host, datamask_read);
  FixRigidSmallLSDEM::pre_neighbor();
  atomKK->modified(Host, datamask_modify);
  atomKK->sync(execution_space, datamask_modify);
  sync_peratom_to_device();
  copy_body_to_device(); copy_bodyLS_to_device(); body_resident_device = true;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::pre_force(int vflag)
{
  // PREFORCE_LS ghost forward-comm packs the grain custom fields on the host;
  // sync them (and reset_atom2body_ghost's inputs) to the host first.
  atomKK->sync(Host, datamask_read);
  FixRigidSmallLSDEM::pre_force(vflag);
  atomKK->modified(Host, datamask_modify);
  atomKK->sync(execution_space, datamask_modify);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::initial_integrate(int vflag)
{
  // device body integration against the resident d_body (Richardson, double
  // precision LSDEMExtra). Mirrors FixRigidSmall::initial_integrate body update.
  auto l_body = d_body;
  const double l_dtf = dtf, l_dtv = dtv, l_dtq = dtq;
  Kokkos::parallel_for("lsdem/kk initial_integrate",
    Kokkos::RangePolicy<DeviceType>(0, nlocal_body),
    KOKKOS_LAMBDA(const int ibody) {
      DBody &b = l_body(ibody);
      const double dtfm = l_dtf / b.mass;
      b.vcm[0] += dtfm * b.fcm[0];
      b.vcm[1] += dtfm * b.fcm[1];
      b.vcm[2] += dtfm * b.fcm[2];
      b.xcm[0] += l_dtv * b.vcm[0];
      b.xcm[1] += l_dtv * b.vcm[1];
      b.xcm[2] += l_dtv * b.vcm[2];
      b.angmom[0] += l_dtf * b.torque[0];
      b.angmom[1] += l_dtf * b.torque[1];
      b.angmom[2] += l_dtf * b.torque[2];
      LSDEMExtra::ls_dem_angmom_to_omega(b.angmom, b.ex_space, b.ey_space, b.ez_space, b.inertia, b.omega);
      LSDEMExtra::ls_dem_richardson(b.quat, b.angmom, b.omega, b.inertia, l_dtq);
      LSDEMExtra::ls_dem_q_to_exyz(b.quat, b.ex_space, b.ey_space, b.ez_space);
    });

  v_init(vflag);   // host virial setup (no atom DualView)

  // host-staged body forward-comm: base INITIAL (29 doubles) THEN LSDEM
  // INITIAL_LS quatd2g (4), exactly as FixRigidSmall + FixRigidSmallLSDEM do on
  // the host. comm->forward_comm(Fix*) routes to the host pack (forward_comm_device
  // unset). Then upload body[] + bodyLS.quatd2g back to the device.
  copy_body_to_host();
  commflag = INITIAL;       comm->forward_comm(this, 29);
  commflag_ls = INITIAL_LS; comm->forward_comm(this, 4);
  commflag_ls = PARENT;
  copy_body_to_device();
  copy_bodyLS_to_device();

  set_xv_kokkos(1);                // positions + velocities on device
  scatter_grain_fields_kokkos();   // grain xcom/quat/omega for the pair K2
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::final_integrate()
{
  if (!earlyflag) compute_forces_and_torques();   // device reduction -> d_body

  // device vcm/angmom/omega update; 2d enforce2d folded in BEFORE the kick.
  auto l_body = d_body;
  const double l_dtf = dtf;
  const int dim2 = (domain->dimension == 2);
  Kokkos::parallel_for("lsdem/kk final_integrate",
    Kokkos::RangePolicy<DeviceType>(0, nlocal_body),
    KOKKOS_LAMBDA(const int ibody) {
      DBody &b = l_body(ibody);
      if (dim2) {
        b.xcm[2] = 0.0; b.vcm[2] = 0.0; b.fcm[2] = 0.0; b.xgc[2] = 0.0;
        b.torque[0] = 0.0; b.torque[1] = 0.0;
        b.angmom[0] = 0.0; b.angmom[1] = 0.0;
        b.omega[0] = 0.0; b.omega[1] = 0.0;
      }
      const double dtfm = l_dtf / b.mass;
      b.vcm[0] += dtfm * b.fcm[0];
      b.vcm[1] += dtfm * b.fcm[1];
      b.vcm[2] += dtfm * b.fcm[2];
      b.angmom[0] += l_dtf * b.torque[0];
      b.angmom[1] += l_dtf * b.torque[1];
      b.angmom[2] += l_dtf * b.torque[2];
      LSDEMExtra::ls_dem_angmom_to_omega(b.angmom, b.ex_space, b.ey_space, b.ez_space, b.inertia, b.omega);
    });

  // host-staged FINAL body forward-comm (vcm/omega/conjqm to ghosts).
  copy_body_to_host();
  commflag = FINAL; comm->forward_comm(this, 10);
  copy_body_to_device();

  set_xv_kokkos(0);                // velocities only
}

/* ----------------------------------------------------------------------
   set space-frame coords + velocity of each atom in each rigid body on the
   device. setx=1 also updates positions. Ported from upstream
   FixRigidSmallKokkos::set_xv_kokkos; LSDEM does not override set_xv/set_v so
   this reproduces the inherited base routine. Per-atom virial -> host fallback.
------------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::set_xv_kokkos(int setx)
{
  if (evflag && (vflag_atom || cvflag_atom)) {
    atomKK->sync(Host, X_MASK | V_MASK | F_MASK | TYPE_MASK | RMASS_MASK);
    if (setx) FixRigidSmall::set_xv();
    else FixRigidSmall::set_v();
    atomKK->modified(Host, X_MASK | V_MASK);
    atomKK->sync(execution_space, X_MASK | V_MASK);   // reconcile: clear host-dirty
    return;
  }

  const double xprd = domain->xprd;
  const double yprd = domain->yprd;
  const double zprd = domain->zprd;
  const double xy = domain->xy;
  const double xz = domain->xz;
  const double yz = domain->yz;
  const int triclinic = domain->triclinic;
  const int dim = domain->dimension;
  const int l_evflag = evflag;
  const double l_dtf = dtf;
  const int rmass_flag = (atom->rmass != nullptr);

  atomKK->sync(execution_space, X_MASK | V_MASK | F_MASK | TYPE_MASK | RMASS_MASK);
  atomKK->k_mass.template sync<DeviceType>();
  auto l_x = atomKK->k_x.view<DeviceType>();
  auto l_v = atomKK->k_v.view<DeviceType>();
  auto l_f = atomKK->k_f.view<DeviceType>();
  auto l_mass = atomKK->k_mass.view<DeviceType>();
  auto l_rmass = atomKK->k_rmass.view<DeviceType>();
  auto l_type = atomKK->k_type.view<DeviceType>();

  auto l_body = d_body;
  auto l_atom2body = d_atom2body;
  auto l_displace = d_displace;
  auto l_xcmimage = d_xcmimage;
  const int nlocal = atom->nlocal;

  EV_FLOAT ev;
  Kokkos::parallel_reduce("lsdem/kk set_xv",
    Kokkos::RangePolicy<DeviceType>(0, nlocal),
    KOKKOS_LAMBDA(const int i, EV_FLOAT &ev) {
      if (l_atom2body(i) < 0) return;
      const DBody &b = l_body(l_atom2body(i));

      const int xbox = (l_xcmimage(i) & IMGMASK) - IMGMAX;
      const int ybox = (l_xcmimage(i) >> IMGBITS & IMGMASK) - IMGMAX;
      const int zbox = (l_xcmimage(i) >> IMG2BITS) - IMGMAX;

      const double v0 = l_v(i,0), v1 = l_v(i,1), v2 = l_v(i,2);
      const double ox0 = l_x(i,0), ox1 = l_x(i,1), ox2 = l_x(i,2);

      double delta[3];
      delta[0] = b.ex_space[0]*l_displace(i,0) + b.ey_space[0]*l_displace(i,1) + b.ez_space[0]*l_displace(i,2);
      delta[1] = b.ex_space[1]*l_displace(i,0) + b.ey_space[1]*l_displace(i,1) + b.ez_space[1]*l_displace(i,2);
      delta[2] = b.ex_space[2]*l_displace(i,0) + b.ey_space[2]*l_displace(i,1) + b.ez_space[2]*l_displace(i,2);

      double nv0 = b.omega[1]*delta[2] - b.omega[2]*delta[1] + b.vcm[0];
      double nv1 = b.omega[2]*delta[0] - b.omega[0]*delta[2] + b.vcm[1];
      double nv2 = b.omega[0]*delta[1] - b.omega[1]*delta[0] + b.vcm[2];

      if (dim == 2) { nv2 = 0.0; delta[2] = 0.0; }

      l_v(i,0) = nv0; l_v(i,1) = nv1; l_v(i,2) = nv2;

      if (setx) {
        if (triclinic == 0) {
          l_x(i,0) = delta[0] + b.xcm[0] - xbox*xprd;
          l_x(i,1) = delta[1] + b.xcm[1] - ybox*yprd;
          l_x(i,2) = delta[2] + b.xcm[2] - zbox*zprd;
        } else {
          l_x(i,0) = delta[0] + b.xcm[0] - xbox*xprd - ybox*xy - zbox*xz;
          l_x(i,1) = delta[1] + b.xcm[1] - ybox*yprd - zbox*yz;
          l_x(i,2) = delta[2] + b.xcm[2] - zbox*zprd;
        }
      }

      if (l_evflag) {
        const double massone = rmass_flag ? l_rmass(i) : l_mass(l_type(i));
        const double fc0 = massone*(nv0 - v0)/l_dtf - l_f(i,0);
        const double fc1 = massone*(nv1 - v1)/l_dtf - l_f(i,1);
        const double fc2 = massone*(nv2 - v2)/l_dtf - l_f(i,2);
        double X0, X1, X2;
        if (triclinic == 0) {
          X0 = ox0 + xbox*xprd; X1 = ox1 + ybox*yprd; X2 = ox2 + zbox*zprd;
        } else {
          X0 = ox0 + xbox*xprd + ybox*xy + zbox*xz;
          X1 = ox1 + ybox*yprd + zbox*yz;
          X2 = ox2 + zbox*zprd;
        }
        ev.v[0] += 0.5*X0*fc0;
        ev.v[1] += 0.5*X1*fc1;
        ev.v[2] += 0.5*X2*fc2;
        ev.v[3] += 0.5*X0*fc1;
        ev.v[4] += 0.5*X0*fc2;
        ev.v[5] += 0.5*X1*fc2;
      }
    }, ev);

  if (evflag && vflag_global)
    for (int k = 0; k < 6; k++) virial[k] += ev.v[k];

  if (setx) {
    const int nbody = nlocal_body + nghost_body;
    auto l_body2 = d_body;
    Kokkos::parallel_for("lsdem/kk set_xv xgc",
      Kokkos::RangePolicy<DeviceType>(0, nbody),
      KOKKOS_LAMBDA(const int ibody) {
        DBody &b = l_body2(ibody);
        b.xgc[0] = b.ex_space[0]*b.xgc_body[0] + b.ey_space[0]*b.xgc_body[1] + b.ez_space[0]*b.xgc_body[2] + b.xcm[0];
        b.xgc[1] = b.ex_space[1]*b.xgc_body[0] + b.ey_space[1]*b.xgc_body[1] + b.ez_space[1]*b.xgc_body[2] + b.xcm[1];
        b.xgc[2] = b.ex_space[2]*b.xgc_body[0] + b.ey_space[2]*b.xgc_body[1] + b.ez_space[2]*b.xgc_body[2] + b.xcm[2];
      });
  }

  if (setx) atomKK->modified(execution_space, X_MASK | V_MASK);
  else atomKK->modified(execution_space, V_MASK);
}

/* ----------------------------------------------------------------------
   scatter the body COM / orientation / angular velocity to each node's grain
   custom fields (atom->xcom/quat/omega) that the pair K2 reads. Device port of
   the FixRigidSmallLSDEM::initial_integrate scatter loop. quat is rotated to the
   grid frame: grain_quat = quatquat(body.quat, bodyLS.quatd2g). Runs ONLY in
   initial_integrate (once per step, after set_xv).
------------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::scatter_grain_fields_kokkos()
{
  const int nlocal = atom->nlocal;
  atomKK->sync(execution_space, MASK_MASK);
  auto l_mask  = atomKK->k_mask.template view<DeviceType>();
  auto l_xcom  = atomKK->k_xcom.template view<DeviceType>();
  auto l_quat  = atomKK->k_quat.template view<DeviceType>();
  auto l_omega = atomKK->k_omega.template view<DeviceType>();
  auto l_body = d_body;
  auto l_atom2body = d_atom2body;
  auto l_quatd2g = d_quatd2g;
  const int l_groupbit = groupbit;

  Kokkos::parallel_for("lsdem/kk grain scatter",
    Kokkos::RangePolicy<DeviceType>(0, nlocal),
    KOKKOS_LAMBDA(const int i) {
      if (!(l_mask(i) & l_groupbit)) return;
      const int ib = l_atom2body(i);
      if (ib < 0) return;                       // host error->one becomes a device-safe skip
      const DBody &b = l_body(ib);
      l_xcom(i,0) = b.xcm[0]; l_xcom(i,1) = b.xcm[1]; l_xcom(i,2) = b.xcm[2];
      const double qd[4] = {l_quatd2g(ib,0), l_quatd2g(ib,1), l_quatd2g(ib,2), l_quatd2g(ib,3)};
      double q[4];
      LSDEMExtra::ls_dem_quatquat(b.quat, qd, q);
      l_quat(i,0) = q[0]; l_quat(i,1) = q[1]; l_quat(i,2) = q[2]; l_quat(i,3) = q[3];
      l_omega(i,0) = b.omega[0]; l_omega(i,1) = b.omega[1]; l_omega(i,2) = b.omega[2];
    });

  atomKK->modified(execution_space, XCOM_MASK | QUAT_MASK | OMEGA_MASK);
}

/* ----------------------------------------------------------------------
   Per-body force/torque reduction on the device (increment 1, unchanged except
   that it now reads the DualView-backed d_atom2body refreshed by
   sync_peratom_to_device -- no standalone upload). Sums each node's f AND
   torque by molecule into a per-molecule buffer (NO lever arm -- the pair
   supplies the exact per-node torque about the body COM), then keeps the host
   molecule-id MPI_Allreduce so a body's total reaches its owner across ranks.
   Gravity falls back to the exact host path.
------------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::compute_forces_and_torques()
{
  if (id_gravity) {
    // gravity host-fallback: the host reduction reads f/torque + type/rmass on
    // the host, so sync them down; then push the updated body[] back to d_body
    // (else the device integration kernel reads stale body f/torque).
    atomKK->sync(Host, F_MASK | TORQUE_MASK | MASK_MASK | MOLECULE_MASK | TYPE_MASK | RMASS_MASK);
    FixRigidSmallLSDEM::compute_forces_and_torques();
    copy_body_to_device();
    return;
  }

  const int nlocal = atom->nlocal;
  const int nm = maxmol + 1;

  if ((int) d_ft.extent(0) < 6 * nm) {
    d_ft = Kokkos::View<double *, DeviceType>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "lsdem/kk:ft"), 6 * nm);
    h_ft = Kokkos::create_mirror_view(d_ft);
  }
  Kokkos::deep_copy(d_ft, 0.0);

  atomKK->sync(execution_space, F_MASK | TORQUE_MASK | MASK_MASK | MOLECULE_MASK);
  auto l_f = atomKK->k_f.template view<DeviceType>();
  auto l_torque = atomKK->k_torque.template view<DeviceType>();
  auto l_mask = atomKK->k_mask.template view<DeviceType>();
  auto l_molecule = atomKK->k_molecule.template view<DeviceType>();
  auto l_atom2body = d_atom2body;
  auto l_ft = d_ft;
  const int l_groupbit = groupbit;

  Kokkos::parallel_for("lsdem/kk reduce f/torque by molecule",
    Kokkos::RangePolicy<DeviceType>(0, nlocal),
    KOKKOS_LAMBDA(const int i) {
      if (!(l_mask(i) & l_groupbit)) return;
      if (l_atom2body(i) < 0) return;
      const int m = 6 * (int) l_molecule(i);
      Kokkos::atomic_add(&l_ft(m + 0), l_f(i, 0));
      Kokkos::atomic_add(&l_ft(m + 1), l_f(i, 1));
      Kokkos::atomic_add(&l_ft(m + 2), l_f(i, 2));
      Kokkos::atomic_add(&l_ft(m + 3), l_torque(i, 0));
      Kokkos::atomic_add(&l_ft(m + 4), l_torque(i, 1));
      Kokkos::atomic_add(&l_ft(m + 5), l_torque(i, 2));
    });

  Kokkos::deep_copy(h_ft, d_ft);

  // host-staged molecule-id Allreduce: a body's total reaches its owner even
  // when its contact nodes are on a different rank than the owner.
  std::vector<double> ftall(6 * nm, 0.0);
  MPI_Allreduce(h_ft.data(), ftall.data(), 6 * nm, MPI_DOUBLE, MPI_SUM, world);

  tagint *molecule = atom->molecule;
  for (int ibody = 0; ibody < nlocal_bodyLS; ibody++) {
    double *g = &ftall[6 * (int) molecule[body[ibody].ilocal]];
    double *fcm = body[ibody].fcm;     fcm[0] = g[0];  fcm[1] = g[1];  fcm[2] = g[2];
    double *tcm = body[ibody].torque;  tcm[0] = g[3];  tcm[1] = g[4];  tcm[2] = g[5];
  }

  // push the freshly-reduced owned-body f/torque to the device for the final
  // integration kernel. (Only owned slots are written above; ghost f/torque are
  // never read by a device kernel -- set_xv reads ghost xcm/vcm/omega, not f.)
  copy_body_to_device();
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixRigidSmallLSDEMKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixRigidSmallLSDEMKokkos<LMPHostType>;
#endif
}
