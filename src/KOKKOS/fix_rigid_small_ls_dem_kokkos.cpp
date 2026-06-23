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
#include "domain.h"
#include "memory.h"
#include "neighbor.h"

#include <vector>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixRigidSmallLSDEMKokkos<DeviceType>::FixRigidSmallLSDEMKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixRigidSmallLSDEM(lmp, narg, arg)
{
  // Increment 1 is intentionally INVISIBLE to the framework as a device fix:
  // we do NOT set kokkosable / execution_space / atomKK / datamask. Setting
  // execution_space = Device (its value on a CUDA build) misclassifies this fix
  // as device-resident, so the run loop skips host-syncing its per-atom arrays
  // around the inherited (host) legacy sort/exchange/integration -> corrupted
  // body assignment -> "Lost atoms" (seen on GPU; harmless on the OpenMP-host
  // build where Device==Host). Left untouched, the framework treats this EXACTLY
  // like the host fix rigid/small/ls/dem (full host sync, legacy host
  // sort/exchange/comm) -- correct + unchanged. We run ONE device kernel (the
  // per-body reduction) via a LOCAL AtomKokkos cast in compute_forces_and_torques.
  a2b_lastbuild = -1;
}

/* ----------------------------------------------------------------------
   refresh the device copy of atom2body. Body assignment of LOCAL atoms is
   stable between neighbour-list builds, so this runs on the reneighbor cadence.
------------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::upload_atom2body()
{
  const int n = atom->nlocal;
  if ((int) d_atom2body.extent(0) < n) {
    d_atom2body = typename AT::t_int_1d(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "lsdem/kk:atom2body"), n);
    h_atom2body = Kokkos::create_mirror_view(d_atom2body);
  }
  for (int i = 0; i < n; i++) h_atom2body(i) = atom2body[i];
  Kokkos::deep_copy(d_atom2body, h_atom2body);
}

/* ----------------------------------------------------------------------
   Per-body force/torque reduction on the device.
   Replaces the O(natoms) host summation loop of
   FixRigidSmallLSDEM::compute_forces_and_torques() with a device kernel that
   atomic_adds each node's f/torque into a per-molecule buffer; the rest (the
   molecule-id MPI_Allreduce + assignment to bodies) is kept on the host,
   bitwise-identical to the base. NO lever arm (the pair already supplies the
   exact per-node torque about the body COM). Gravity falls back to the exact
   host path (rare; needs per-atom mass on the device).
------------------------------------------------------------------------- */

template<class DeviceType>
void FixRigidSmallLSDEMKokkos<DeviceType>::compute_forces_and_torques()
{
  if (id_gravity) { FixRigidSmallLSDEM::compute_forces_and_torques(); return; }

  if (neighbor->lastcall != a2b_lastbuild) {
    upload_atom2body();
    a2b_lastbuild = neighbor->lastcall;
  }

  const int nlocal = atom->nlocal;
  const int nm = maxmol + 1;

  if ((int) d_ft.extent(0) < 6 * nm) {
    d_ft = Kokkos::View<double *, DeviceType>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "lsdem/kk:ft"), 6 * nm);
    h_ft = Kokkos::create_mirror_view(d_ft);
  }
  Kokkos::deep_copy(d_ft, 0.0);

  // local cast so this fix carries no Kokkos-fix state (see the constructor).
  // The framework has just synced everything to the host for this host-treated
  // fix; sync the few arrays we read back to the device for the kernel.
  AtomKokkos *akk = (AtomKokkos *) atom;
  akk->sync(ExecutionSpaceFromDevice<DeviceType>::space,
            F_MASK | TORQUE_MASK | MASK_MASK | MOLECULE_MASK);
  auto l_f = akk->k_f.template view<DeviceType>();
  auto l_torque = akk->k_torque.template view<DeviceType>();
  auto l_mask = akk->k_mask.template view<DeviceType>();
  auto l_molecule = akk->k_molecule.template view<DeviceType>();
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
