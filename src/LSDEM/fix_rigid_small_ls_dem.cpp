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

#include "fix_rigid_small_ls_dem.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "hashlittle.h"
#include "input.h"
#include "ls_dem_extra.h"
#include "math_const.h"
#include "math_eigen.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "neighbor.h"
#include "respa.h"
#include "rigid_const.h"
#include "tokenizer.h"
#include "update.h"
#include "variable.h"

#include <cmath>
#include <cstring>
#include <map>
#include <utility>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
using namespace RigidConst;

enum {GLOBAL, DISTRIBUTED};

static constexpr double EPSILON_VOL_DIFF = 1.0e-6; // 0.0001%
static constexpr double EPSILON_INERTIA = 1.0e-3; // 0.1%
static constexpr int MAX_ITERATIONS = 100; // For surface area integration
static constexpr int RECOMMENDED_MAX_NGRID = 1000; // For local node grid, 10x10x10

/* ---------------------------------------------------------------------- */

FixRigidSmallLSDEM::FixRigidSmallLSDEM(LAMMPS *lmp, int narg, char **arg) :
  FixRigidSmall(lmp, narg, arg), bodyLS(nullptr), bodyownLS(nullptr)
{
  maxcut = -1;
  stored_flag = 0;
  distributed_flag = 0;

  n_extra_attributes = 3;

  if (!inpfile)
    error->all(FLERR, "Must specify infile with level set for fix rigid/small/ls/dem");

  nmax_bodyLS = 0;
  while (nmax_bodyLS < nlocal_body) nmax_bodyLS += DELTA_BODY;
  bodyLS = (BodyLS *) memory->smalloc(nmax_bodyLS * sizeof(BodyLS), "rigid/small/ls/dem:bodyls");

  // set bodyown for owned atoms

  tagint *tag = atom->tag;
  nlocal_bodyLS = nghost_bodyLS = 0;
  for (int i = 0; i < atom->nlocal; i++)
    if (bodytag[i] == tag[i]) {
      bodyLS[nlocal_bodyLS].ilocal = i;
      bodyownLS[i] = nlocal_bodyLS++;
    } else bodyownLS[i] = -1;

  // bodysizeLS = sizeof(BodyLS) in doubles

  bodysizeLS = sizeof(BodyLS) / sizeof(double);
  if (bodysizeLS * sizeof(double) != sizeof(BodyLS)) bodysizeLS++;

  // increase max comm size needed for LSDEM

  comm_forward += bodysizeLS;

  if (!atom->omega_flag)
    error->all(FLERR, "Fix rigid/ls/dem requires atom attribute omega");
}

/* ---------------------------------------------------------------------- */

FixRigidSmallLSDEM::~FixRigidSmallLSDEM()
{
  memory->sfree(bodyLS);
}

/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::setup_pre_neighbor()
{
  FixRigidSmall::setup_pre_neighbor();
  nghost_bodyLS = 0;
}

/* ----------------------------------------------------------------------
   compute initial fcm and torque on bodies, also initial virial
   reset all particle velocities to be consistent with vcm and omega

     TODO: this is a lot of code duplication. A cleaner way to do that
           could be to write little helper functions for computing torques from forces
           and not call it for LSDEM
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::setup(int vflag)
{
  int i, n, ibody;

  // error if maxextent > comm->cutghost
  // NOTE: could just warn if an override flag set
  // NOTE: this could fail for comm multi mode if user sets a wrong cutoff
  //       for atom types in rigid bodies - need a more careful test
  // must check here, not in init, b/c neigh/comm values set after fix init

  double cutghost = MAX(neighbor->cutneighmax, comm->cutghostuser);
  if (maxextent > cutghost)
    error->all(FLERR, "Rigid body extent {} > ghost atom cutoff - use comm_modify cutoff", maxextent);

  //check(1);

  // sum fcm, torque across all rigid bodies

  compute_forces_and_torques();

  // enforce 2d body forces and torques

  if (domain->dimension == 2) enforce2d();

  // virial setup before call to set_v

  v_init(vflag);

  // compute and forward communicate vcm and omega of all bodies

  for (ibody = 0; ibody < nlocal_body; ibody++) {
    Body *b = &body[ibody];
    MathExtra::angmom_to_omega(b->angmom, b->ex_space, b->ey_space,
                               b->ez_space, b->inertia, b->omega);
  }

  commflag = FINAL;
  comm->forward_comm(this, 10);

  // set velocity/rotation of atoms in rigid bodues

  set_v();

  // guesstimate virial as 2x the set_v contribution

  int nlocal = atom->nlocal;
  if (vflag_global)
    for (n = 0; n < 6; n++) virial[n] *= 2.0;
  if (vflag_atom) {
    for (i = 0; i < nlocal; i++)
      for (n = 0; n < 6; n++)
        vatom[i][n] *= 2.0;
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::pre_neighbor()
{
  FixRigidSmall::pre_neighbor();
  nghost_bodyLS = 0;
}


/* ----------------------------------------------------------------------
   Calculation of the forces and torques for LS-DEM grains

   Forces apply at the contact point between a surface atom and a level-set.
   There is no LAMMPS structure for it so forces are applied on nearest atoms
   Torques computed from forces applied at the atom position would be off.
   To avoid this miscalculation:
     1. exact torques are applied on (extended) atoms in pair_ls_dem
     2. torques are not computed from forces on atoms (unlike Fix Rigid)
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::compute_forces_and_torques()
{
  int i, ibody;

  //check(3);

  // sum over atoms to get force and torque on rigid body

  double **x = atom->x;
  double **f = atom->f;
  int nlocal = atom->nlocal;
  double *fcm,*tcm;

  for (ibody = 0; ibody < nlocal_body + nghost_body; ibody++) {
    fcm = body[ibody].fcm;
    fcm[0] = fcm[1] = fcm[2] = 0.0;
    tcm = body[ibody].torque;
    tcm[0] = tcm[1] = tcm[2] = 0.0;
  }

  for (i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    fcm = b->fcm;
    fcm[0] += f[i][0];
    fcm[1] += f[i][1];
    fcm[2] += f[i][2];
  }

  // extended particles add their torque to torque of body

  if (extended) {
    double **torque = atom->torque;

    for (i = 0; i < nlocal; i++) {
      if (atom2body[i] < 0) continue;

      if (eflags[i] & TORQUE) {
        tcm = body[atom2body[i]].torque;
        tcm[0] += torque[i][0];
        tcm[1] += torque[i][1];
        tcm[2] += torque[i][2];
      }
    }
  }

  // reverse communicate fcm, torque of all bodies

  commflag = FORCE_TORQUE;
  comm->reverse_comm(this,6);

  // add gravity force to COM of each body

  if (id_gravity) {
    double mass;
    for (ibody = 0; ibody < nlocal_body; ibody++) {
      mass = body[ibody].mass;
      fcm = body[ibody].fcm;
      fcm[0] += gvec[0] * mass;
      fcm[1] += gvec[1] * mass;
      fcm[2] += gvec[2] * mass;
    }
  }
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::grow_arrays(int nmax)
{
  FixRigidSmall::grow_arrays(nmax);
  memory->grow(bodyownLS, nmax, "rigid/small/ls/dem:bodyownLS");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::copy_arrays(int i, int j, int delflag)
{
  FixRigidSmall::copy_arrays(i, j, delflag);

  // if deleting atom J via delflag and J owns a body, then delete it

  if (delflag && bodyownLS[j] >= 0) {
    bodyownLS[bodyLS[nlocal_bodyLS - 1].ilocal] = bodyownLS[j];
    memcpy(&bodyLS[bodyownLS[j]], &bodyLS[nlocal_bodyLS - 1], sizeof(BodyLS));
    nlocal_bodyLS--;
  }

  // if atom I owns a body, reset I's body.ilocal to loc J
  // do NOT do this if self-copy (I=J) since I's body is already deleted

  if (bodyownLS[i] >= 0 && i != j) bodyLS[bodyownLS[i]].ilocal = j;
  bodyownLS[j] = bodyownLS[i];
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::set_arrays(int i)
{
  FixRigidSmall::set_arrays(i);
  bodyownLS[i] = -1;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixRigidSmallLSDEM::pack_exchange(int i, double *buf)
{
  int m = FixRigidSmall::pack_exchange(i, buf);buf[0] = ubuf(bodytag[i]).d;

  // atom not in a rigid body

  if (!bodytag[i]) return m;

  // atom does not own its rigid body

  if (bodyownLS[i] < 0) {
    buf[m++] = 0;
    return m;
  }

  // body info for atom that owns a rigid body

  buf[m++] = 1;
  memcpy(&buf[m], &bodyLS[bodyownLS[i]], sizeof(BodyLS));
  m += bodysizeLS;
  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixRigidSmallLSDEM::unpack_exchange(int nlocal, double *buf)
{
  int m = FixRigidSmall::unpack_exchange(nlocal, buf);

  // atom not in a rigid body

  if (!bodytag[nlocal]) {
    bodyownLS[nlocal] = -1;
    return m;
  }

  // atom does not own its rigid body

  bodyownLS[nlocal] = static_cast<int> (buf[m++]);
  if (bodyownLS[nlocal] == 0) {
    bodyownLS[nlocal] = -1;
    return m;
  }

  // body info for atom that owns a rigid body

  if (nlocal_bodyLS == nmax_body) grow_body_ls();
  memcpy(&bodyLS[nlocal_bodyLS], &buf[m], sizeof(BodyLS));
  m += bodysizeLS;
  bodyLS[nlocal_bodyLS].ilocal = nlocal;
  bodyownLS[nlocal] = nlocal_bodyLS++;
  return m;
}

/* ----------------------------------------------------------------------
   only pack body info if own or ghost atom owns the body
   for FULL_BODY, send 0/1 flag with every atom
------------------------------------------------------------------------- */

int FixRigidSmallLSDEM::pack_forward_comm(int n, int *list, double *buf,
                                     int /*pbc_flag*/, int * /*pbc*/)
{
  int m = FixRigidSmall::pack_forward_comm(n, list, buf, 0, nullptr);

  if (commflag == FULL_BODY) {
    int i, j;
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyownLS[j] < 0) buf[m++] = 0;
      else {
        buf[m++] = 1;
        memcpy(&buf[m], &bodyLS[bodyownLS[j]], sizeof(BodyLS));
        m += bodysizeLS;
      }
    }
  }

  return m;
}

/* ----------------------------------------------------------------------
   only ghost atoms are looped over
   for FULL_BODY, store a new ghost body if this atom owns it
   for other commflag values, only unpack body info if atom owns it
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::unpack_forward_comm(int n, int first, double *buf)
{
  FixRigidSmall::unpack_forward_comm(n, first, buf);

  if (commflag == FULL_BODY) {
    int i, j, last;
    last = first + n;
    int m = 0;
    for (i = first; i < last; i++)
      if (bodyown[i] != 0) m += bodysize;

    for (i = first; i < last; i++) {
      bodyownLS[i] = static_cast<int> (buf[m++]);
      if (bodyownLS[i] == 0) bodyownLS[i] = -1;
      else {
        j = nlocal_bodyLS + nghost_bodyLS;
        if (j == nmax_bodyLS) grow_body_ls();
        memcpy(&bodyLS[j],&buf[m],sizeof(BodyLS));
        m += bodysizeLS;
        bodyLS[j].ilocal = i;
        bodyownLS[i] = j;
        nghost_bodyLS++;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   only ghost atoms are looped over
   only pack body info if atom owns it
------------------------------------------------------------------------- */

int FixRigidSmallLSDEM::pack_reverse_comm(int n, int first, double *buf)
{
  int m = FixRigidSmall::pack_reverse_comm(n, first, buf);
  return m;
}

/* ----------------------------------------------------------------------
   only unpack body info if own or ghost atom owns the body
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::unpack_reverse_comm(int n, int *list, double *buf)
{
  FixRigidSmall::unpack_reverse_comm(n, list, buf);
}


/* ----------------------------------------------------------------------
   grow bodyLS data structure
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::grow_body_ls()
{
  nmax_bodyLS += DELTA_BODY;
  bodyLS = (BodyLS *) memory->srealloc(bodyLS, nmax_bodyLS * sizeof(BodyLS),
                                   "rigid/small/ls/dem:bodyLS");
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixRigidSmallLSDEM::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = FixRigidSmall::memory_usage();
  bytes += (double)nmax * sizeof(int);
  bytes += (double)nmax_body * sizeof(BodyLS);
  return bytes;
}
