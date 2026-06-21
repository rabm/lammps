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
#include "error.h"
#include "kokkos.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLSDEMKokkos<DeviceType>::PairLSDEMKokkos(LAMMPS *lmp) : PairLSDEM(lmp)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  // fields the CPU contact model reads / writes (XCOM/QUAT feed get_ls_value via the rigid fix)
  datamask_read = X_MASK | V_MASK | XCOM_MASK | QUAT_MASK | OMEGA_MASK | TYPE_MASK | MASK_MASK |
                  F_MASK | TORQUE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | TORQUE_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::init_style()
{
  PairLSDEM::init_style();

  // Milestone 2 host bridge: the device kernels (M3/M4) are not written yet, so compute() runs
  // the proven CPU PairLSDEM::compute on host-synced atom data, reusing the legacy (non-Kokkos)
  // half + REQ_GHOST neighbor list that PairLSDEM::init_style already requested. On the OpenMP
  // backend host and device share memory, so this reproduces the CPU pair bitwise. A device
  // neighbor list + on-device residency arrive with the kernels.
  if (lmp->kokkos->neighflag == FULL)
    error->all(FLERR, "Cannot use a full neighbor list with pair style ls/dem/kk");

  if (comm->me == 0)
    utils::logmesg(lmp, "PairLSDEMKokkos (M2 host bridge): CPU compute on Kokkos-synced atom data\n");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLSDEMKokkos<DeviceType>::compute(int eflag, int vflag)
{
  atomKK->sync(Host, datamask_read);
  PairLSDEM::compute(eflag, vflag);
  atomKK->modified(Host, datamask_modify);
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairLSDEMKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairLSDEMKokkos<LMPHostType>;
#endif
}
