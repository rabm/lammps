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
#include "memory_kokkos.h"

#include <vector>

using namespace LAMMPS_NS;

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
  // execution_space = Host (NOT Device): in 2a all compute is host-fallback, so
  // ModifyKokkos must wrap our lifecycle methods with sync(Host,...)/
  // modified(Host,...) -- matching the host data our base code reads/writes and
  // syncing the property/atom customs to host for free. Setting it to Device
  // makes ModifyKokkos do modified(Device, datamask_modify) AFTER our method,
  // which collides with the host-modified flag our own modified(Host,...) set
  // -> "Concurrent modification ... atom:x" abort. The device per-body reduction
  // kernel syncs via a LOCAL ExecutionSpaceFromDevice space, independent of this
  // member, so it is unaffected. (2b moves compute to the device and flips this.)
  execution_space = Host;

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
   lifecycle hooks. All per-body integration / set_xv / set_v / grain scatter
   still run in the (host) base in 2a; sync the atom data to/from the host
   around each call (datamask_read in, datamask_modify out -- mirroring the
   upstream FixRigidSmallKokkos host-fallback pattern: precise masks, NOT
   ALL_MASK, which over-claims and collides with the framework's own device
   modifies), then push the per-atom DualViews back to the device for the
   reduction kernel.
------------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::setup(int vflag)
{
  atomKK->sync(Host, datamask_read);
  FixRigidSmallLSDEM::setup(vflag);
  atomKK->modified(Host, datamask_modify);
  sync_peratom_to_device();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::setup_pre_neighbor()
{
  atomKK->sync(Host, datamask_read);
  FixRigidSmallLSDEM::setup_pre_neighbor();
  atomKK->modified(Host, datamask_modify);
  sync_peratom_to_device();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::pre_neighbor()
{
  atomKK->sync(Host, datamask_read);
  FixRigidSmallLSDEM::pre_neighbor();
  atomKK->modified(Host, datamask_modify);
  sync_peratom_to_device();
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
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::initial_integrate(int vflag)
{
  atomKK->sync(Host, datamask_read);
  FixRigidSmallLSDEM::initial_integrate(vflag);
  atomKK->modified(Host, datamask_modify);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::final_integrate()
{
  // FixRigidSmallLSDEM::final_integrate calls compute_forces_and_torques()
  // (our device reduction override) and then the host velocity kick; the
  // reduction does its own device sync of f/torque.
  atomKK->sync(Host, datamask_read);
  FixRigidSmallLSDEM::final_integrate();
  atomKK->modified(Host, datamask_modify);
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
  if (id_gravity) { FixRigidSmallLSDEM::compute_forces_and_torques(); return; }

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
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixRigidSmallLSDEMKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixRigidSmallLSDEMKokkos<LMPHostType>;
#endif
}
