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

#include "atom_vec_ls_dem.h"

#include "atom.h"
#include "error.h"
#include "fix.h"
#include "math_const.h"
#include "modify.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

AtomVecLSDEM::AtomVecLSDEM(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = Atom::ATOMIC;
  mass_type = PER_TYPE;

  atom->molecule_flag = 1;
  atom->quat_flag = atom->xcom_flag = atom->omega_flag = 1;
  atom->torque_flag = atom->grid_index_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = {"molecule", "xcom", "quat", "grid_index", "omega", "torque"};
  fields_copy = {"molecule", "xcom", "quat", "grid_index", "omega"};
  fields_comm_vel = {"xcom", "quat", "omega"};
  fields_reverse = {"torque"};
  fields_border = {"molecule"};
  fields_border_vel = {"molecule", "xcom", "quat", "omega"};
  fields_exchange = {"molecule", "grid_index"};
  fields_restart = {"molecule", "grid_index"};
  fields_create = {"molecule", "xcom", "quat", "omega", "grid_index"};
  fields_data_atom = {"id", "molecule", "type", "x"};
  fields_data_vel = {"id", "v"};

  setup_fields();
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecLSDEM::grow_pointers()
{
  xcom = atom->xcom;
  omega = atom->omega;
  quat = atom->quat;
  grid_index = atom->grid_index;
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecLSDEM::data_atom_post(int ilocal)
{
  omega[ilocal][0] = 0.0;
  omega[ilocal][1] = 0.0;
  omega[ilocal][2] = 0.0;

  xcom[ilocal][0] = 0.0;
  xcom[ilocal][1] = 0.0;
  xcom[ilocal][2] = 0.0;

  quat[ilocal][0] = 1.0;
  quat[ilocal][1] = 0.0;
  quat[ilocal][2] = 0.0;
  quat[ilocal][3] = 0.0;

  grid_index[ilocal] = -1;
}

/* ----------------------------------------------------------------------
   initialize non-zero atom quantities
------------------------------------------------------------------------- */

void AtomVecLSDEM::create_atom_post(int ilocal)
{
  omega[ilocal][0] = 0.0;
  omega[ilocal][1] = 0.0;
  omega[ilocal][2] = 0.0;

  xcom[ilocal][0] = 0.0;
  xcom[ilocal][1] = 0.0;
  xcom[ilocal][2] = 0.0;

  quat[ilocal][0] = 1.0;
  quat[ilocal][1] = 0.0;
  quat[ilocal][2] = 0.0;
  quat[ilocal][3] = 0.0;

  grid_index[ilocal] = -1;
}
