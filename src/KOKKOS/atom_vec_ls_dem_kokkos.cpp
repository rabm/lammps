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

#include "atom_vec_ls_dem_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm_kokkos.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "modify.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecLSDEMKokkos::AtomVecLSDEMKokkos(LAMMPS *lmp) : AtomVec(lmp),
AtomVecKokkos(lmp), AtomVecLSDEM(lmp)
{
}

/* ----------------------------------------------------------------------
   process field strings, initialize data structs for all other methods
------------------------------------------------------------------------- */

void AtomVecLSDEMKokkos::init()
{
  AtomVecLSDEM::init();

  set_atom_masks();
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by a chunk
   n > 0 allocates arrays to size n
------------------------------------------------------------------------- */

void AtomVecLSDEMKokkos::grow(int n)
{
  auto DELTA = LMP_KOKKOS_AV_DELTA;
  int step = MAX(DELTA,nmax*0.01);
  if (n == 0) nmax += step;
  else nmax = n;
  atom->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT)
    error->one(FLERR,"Per-processor system is too big");

  atomKK->sync(Device,ALL_MASK);
  atomKK->modified(Device,ALL_MASK);

  memoryKK->grow_kokkos(atomKK->k_tag,atomKK->tag,nmax,"atom:tag");
  memoryKK->grow_kokkos(atomKK->k_type,atomKK->type,nmax,"atom:type");
  memoryKK->grow_kokkos(atomKK->k_mask,atomKK->mask,nmax,"atom:mask");
  memoryKK->grow_kokkos(atomKK->k_image,atomKK->image,nmax,"atom:image");

  memoryKK->grow_kokkos(atomKK->k_x,atomKK->x,nmax,"atom:x");
  memoryKK->grow_kokkos(atomKK->k_v,atomKK->v,nmax,"atom:v");
  memoryKK->grow_kokkos(atomKK->k_f,atomKK->f,nmax,"atom:f");

  memoryKK->grow_kokkos(atomKK->k_molecule,atomKK->molecule,nmax,"atom:molecule");
  memoryKK->grow_kokkos(atomKK->k_xcom,atomKK->xcom,nmax,"atom:xcom");
  memoryKK->grow_kokkos(atomKK->k_quat,atomKK->quat,nmax,"atom:quat");
  memoryKK->grow_kokkos(atomKK->k_omega,atomKK->omega,nmax,"atom:omega");
  memoryKK->grow_kokkos(atomKK->k_torque,atomKK->torque,nmax,"atom:torque");
  memoryKK->grow_kokkos(atomKK->k_grid_index,atomKK->grid_index,nmax,"atom:grid_index");

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);

  grow_pointers();
  atomKK->sync(Host,ALL_MASK);
}

/* ----------------------------------------------------------------------
   reset local array ptrs
------------------------------------------------------------------------- */

void AtomVecLSDEMKokkos::grow_pointers()
{
  tag = atomKK->tag;
  d_tag = atomKK->k_tag.view_device();
  h_tag = atomKK->k_tag.view_host();
  type = atomKK->type;
  d_type = atomKK->k_type.view_device();
  h_type = atomKK->k_type.view_host();
  mask = atomKK->mask;
  d_mask = atomKK->k_mask.view_device();
  h_mask = atomKK->k_mask.view_host();
  image = atomKK->image;
  d_image = atomKK->k_image.view_device();
  h_image = atomKK->k_image.view_host();

  x = atomKK->x;
  d_x = atomKK->k_x.view_device();
  h_x = atomKK->k_x.view_hostkk();
  v = atomKK->v;
  d_v = atomKK->k_v.view_device();
  h_v = atomKK->k_v.view_hostkk();
  f = atomKK->f;
  d_f = atomKK->k_f.view_device();
  h_f = atomKK->k_f.view_hostkk();

  // molecule and torque have no AtomVecLSDEM raw-pointer member (the host field
  // mechanism uses atom->molecule / atom->torque directly), so only refresh the
  // device/host views, which live in the AtomVecKokkos base.
  d_molecule = atomKK->k_molecule.view_device();
  h_molecule = atomKK->k_molecule.view_host();
  d_torque = atomKK->k_torque.view_device();
  h_torque = atomKK->k_torque.view_hostkk();

  xcom = atomKK->xcom;
  d_xcom = atomKK->k_xcom.view_device();
  h_xcom = atomKK->k_xcom.view_hostkk();
  quat = atomKK->quat;
  d_quat = atomKK->k_quat.view_device();
  h_quat = atomKK->k_quat.view_hostkk();
  omega = atomKK->omega;
  d_omega = atomKK->k_omega.view_device();
  h_omega = atomKK->k_omega.view_hostkk();
  grid_index = atomKK->grid_index;
  d_grid_index = atomKK->k_grid_index.view_device();
  h_grid_index = atomKK->k_grid_index.view_host();
}

/* ----------------------------------------------------------------------
   sort atom arrays on device
------------------------------------------------------------------------- */

void AtomVecLSDEMKokkos::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  atomKK->sync(Device, ALL_MASK & ~F_MASK & ~TORQUE_MASK);

  Sorter.sort(LMPDeviceType(), d_tag);
  Sorter.sort(LMPDeviceType(), d_type);
  Sorter.sort(LMPDeviceType(), d_mask);
  Sorter.sort(LMPDeviceType(), d_image);
  Sorter.sort(LMPDeviceType(), d_x);
  Sorter.sort(LMPDeviceType(), d_v);
  Sorter.sort(LMPDeviceType(), d_molecule);
  Sorter.sort(LMPDeviceType(), d_xcom);
  Sorter.sort(LMPDeviceType(), d_quat);
  Sorter.sort(LMPDeviceType(), d_omega);
  Sorter.sort(LMPDeviceType(), d_grid_index);

  atomKK->modified(Device, ALL_MASK & ~F_MASK & ~TORQUE_MASK);
}

/* ---------------------------------------------------------------------- */

void AtomVecLSDEMKokkos::sync(ExecutionSpace space, uint64_t mask)
{
  if (space == Device) {
    if (mask & X_MASK) atomKK->k_x.sync_device();
    if (mask & V_MASK) atomKK->k_v.sync_device();
    if (mask & F_MASK) atomKK->k_f.sync_device();
    if (mask & TAG_MASK) atomKK->k_tag.sync_device();
    if (mask & TYPE_MASK) atomKK->k_type.sync_device();
    if (mask & MASK_MASK) atomKK->k_mask.sync_device();
    if (mask & IMAGE_MASK) atomKK->k_image.sync_device();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.sync_device();
    if (mask & OMEGA_MASK) atomKK->k_omega.sync_device();
    if (mask & TORQUE_MASK) atomKK->k_torque.sync_device();
    if (mask & XCOM_MASK) atomKK->k_xcom.sync_device();
    if (mask & QUAT_MASK) atomKK->k_quat.sync_device();
    if (mask & GRID_INDEX_MASK) atomKK->k_grid_index.sync_device();
  } else if (space == Host) {
    if (mask & X_MASK) atomKK->k_x.sync_host();
    if (mask & V_MASK) atomKK->k_v.sync_host();
    if (mask & F_MASK) atomKK->k_f.sync_host();
    if (mask & TAG_MASK) atomKK->k_tag.sync_host();
    if (mask & TYPE_MASK) atomKK->k_type.sync_host();
    if (mask & MASK_MASK) atomKK->k_mask.sync_host();
    if (mask & IMAGE_MASK) atomKK->k_image.sync_host();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.sync_host();
    if (mask & OMEGA_MASK) atomKK->k_omega.sync_host();
    if (mask & TORQUE_MASK) atomKK->k_torque.sync_host();
    if (mask & XCOM_MASK) atomKK->k_xcom.sync_host();
    if (mask & QUAT_MASK) atomKK->k_quat.sync_host();
    if (mask & GRID_INDEX_MASK) atomKK->k_grid_index.sync_host();
  } else if (space == HostKK) {
    if (mask & X_MASK) atomKK->k_x.sync_hostkk();
    if (mask & V_MASK) atomKK->k_v.sync_hostkk();
    if (mask & F_MASK) atomKK->k_f.sync_hostkk();
    if (mask & TAG_MASK) atomKK->k_tag.sync_host();
    if (mask & TYPE_MASK) atomKK->k_type.sync_host();
    if (mask & MASK_MASK) atomKK->k_mask.sync_host();
    if (mask & IMAGE_MASK) atomKK->k_image.sync_host();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.sync_host();
    if (mask & OMEGA_MASK) atomKK->k_omega.sync_hostkk();
    if (mask & TORQUE_MASK) atomKK->k_torque.sync_hostkk();
    if (mask & XCOM_MASK) atomKK->k_xcom.sync_hostkk();
    if (mask & QUAT_MASK) atomKK->k_quat.sync_hostkk();
    if (mask & GRID_INDEX_MASK) atomKK->k_grid_index.sync_host();
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecLSDEMKokkos::modified(ExecutionSpace space, uint64_t mask)
{
  if (space == Device) {
    if (mask & X_MASK) atomKK->k_x.modify_device();
    if (mask & V_MASK) atomKK->k_v.modify_device();
    if (mask & F_MASK) atomKK->k_f.modify_device();
    if (mask & TAG_MASK) atomKK->k_tag.modify_device();
    if (mask & TYPE_MASK) atomKK->k_type.modify_device();
    if (mask & MASK_MASK) atomKK->k_mask.modify_device();
    if (mask & IMAGE_MASK) atomKK->k_image.modify_device();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.modify_device();
    if (mask & OMEGA_MASK) atomKK->k_omega.modify_device();
    if (mask & TORQUE_MASK) atomKK->k_torque.modify_device();
    if (mask & XCOM_MASK) atomKK->k_xcom.modify_device();
    if (mask & QUAT_MASK) atomKK->k_quat.modify_device();
    if (mask & GRID_INDEX_MASK) atomKK->k_grid_index.modify_device();
  } else if (space == Host) {
    if (mask & X_MASK) atomKK->k_x.modify_host();
    if (mask & V_MASK) atomKK->k_v.modify_host();
    if (mask & F_MASK) atomKK->k_f.modify_host();
    if (mask & TAG_MASK) atomKK->k_tag.modify_host();
    if (mask & TYPE_MASK) atomKK->k_type.modify_host();
    if (mask & MASK_MASK) atomKK->k_mask.modify_host();
    if (mask & IMAGE_MASK) atomKK->k_image.modify_host();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.modify_host();
    if (mask & OMEGA_MASK) atomKK->k_omega.modify_host();
    if (mask & TORQUE_MASK) atomKK->k_torque.modify_host();
    if (mask & XCOM_MASK) atomKK->k_xcom.modify_host();
    if (mask & QUAT_MASK) atomKK->k_quat.modify_host();
    if (mask & GRID_INDEX_MASK) atomKK->k_grid_index.modify_host();
  } else if (space == HostKK) {
    if (mask & X_MASK) atomKK->k_x.modify_hostkk();
    if (mask & V_MASK) atomKK->k_v.modify_hostkk();
    if (mask & F_MASK) atomKK->k_f.modify_hostkk();
    if (mask & TAG_MASK) atomKK->k_tag.modify_host();
    if (mask & TYPE_MASK) atomKK->k_type.modify_host();
    if (mask & MASK_MASK) atomKK->k_mask.modify_host();
    if (mask & IMAGE_MASK) atomKK->k_image.modify_host();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.modify_host();
    if (mask & OMEGA_MASK) atomKK->k_omega.modify_hostkk();
    if (mask & TORQUE_MASK) atomKK->k_torque.modify_hostkk();
    if (mask & XCOM_MASK) atomKK->k_xcom.modify_hostkk();
    if (mask & QUAT_MASK) atomKK->k_quat.modify_hostkk();
    if (mask & GRID_INDEX_MASK) atomKK->k_grid_index.modify_host();
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecLSDEMKokkos::sync_pinned(ExecutionSpace space, uint64_t mask, int async_flag)
{
  if (space == Device) {
    if ((mask & X_MASK) && atomKK->k_x.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3_lr>(atomKK->k_x,space,async_flag);
    if ((mask & V_MASK) && atomKK->k_v.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3>(atomKK->k_v,space,async_flag);
    if ((mask & F_MASK) && atomKK->k_f.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkacc_1d_3>(atomKK->k_f,space,async_flag);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync_device())
      perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_tag,space,async_flag);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync_device())
      perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_type,space,async_flag);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync_device())
      perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_mask,space,async_flag);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync_device())
      perform_pinned_copy<DAT::tdual_imageint_1d>(atomKK->k_image,space,async_flag);
    if ((mask & MOLECULE_MASK) && atomKK->k_molecule.need_sync_device())
      perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_molecule,space,async_flag);
    if ((mask & OMEGA_MASK) && atomKK->k_omega.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3>(atomKK->k_omega,space,async_flag);
    if ((mask & TORQUE_MASK) && atomKK->k_torque.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkacc_1d_3>(atomKK->k_torque,space,async_flag);
    if ((mask & XCOM_MASK) && atomKK->k_xcom.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3>(atomKK->k_xcom,space,async_flag);
    if ((mask & QUAT_MASK) && atomKK->k_quat.need_sync_device())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_4>(atomKK->k_quat,space,async_flag);
    if ((mask & GRID_INDEX_MASK) && atomKK->k_grid_index.need_sync_device())
      perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_grid_index,space,async_flag);
  } else {
    if ((mask & X_MASK) && atomKK->k_x.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3_lr>(atomKK->k_x,space,async_flag);
    if ((mask & V_MASK) && atomKK->k_v.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3>(atomKK->k_v,space,async_flag);
    if ((mask & F_MASK) && atomKK->k_f.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkacc_1d_3>(atomKK->k_f,space,async_flag);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync_host())
      perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_tag,space,async_flag);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync_host())
      perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_type,space,async_flag);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync_host())
      perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_mask,space,async_flag);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync_host())
      perform_pinned_copy<DAT::tdual_imageint_1d>(atomKK->k_image,space,async_flag);
    if ((mask & MOLECULE_MASK) && atomKK->k_molecule.need_sync_host())
      perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_molecule,space,async_flag);
    if ((mask & OMEGA_MASK) && atomKK->k_omega.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3>(atomKK->k_omega,space,async_flag);
    if ((mask & TORQUE_MASK) && atomKK->k_torque.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkacc_1d_3>(atomKK->k_torque,space,async_flag);
    if ((mask & XCOM_MASK) && atomKK->k_xcom.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_3>(atomKK->k_xcom,space,async_flag);
    if ((mask & QUAT_MASK) && atomKK->k_quat.need_sync_host())
      perform_pinned_copy_transform<DAT::ttransform_kkfloat_1d_4>(atomKK->k_quat,space,async_flag);
    if ((mask & GRID_INDEX_MASK) && atomKK->k_grid_index.need_sync_host())
      perform_pinned_copy<DAT::tdual_int_1d>(atomKK->k_grid_index,space,async_flag);
  }
}
