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
#include "input.h"
#include "ls_dem_extra.h"
#include "math_const.h"
#include "math_eigen.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "neighbor.h"
#include "pair_ls_dem.h"
#include "respa.h"
#include "rigid_const.h"
#include "tokenizer.h"
#include "update.h"
#include "variable.h"

#include <cmath>
#include <cfloat>
#include <cstring>
#include <map>
#include <unordered_map>
#include <utility>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
using namespace RigidConst;
using namespace LSDEMExtra;

enum {GLOBAL, DISTRIBUTED};
enum {ARRAY, WATERSHED};
enum {PARENT, FULL_BODY_LS, INITIAL_LS, PREFORCE_LS};

static constexpr double EPSILON_VOL_DIFF = 1.0e-6; // 0.0001%
static constexpr int MAX_ITERATIONS = 100; // For surface area integration
static constexpr int RECOMMENDED_MAX_NGRID = 1000; // For local node grid, 10x10x10

// Todo: should fix we have different instance of fix property/atom
//   for distributed memory when there are different cutoffs between
//   two types of grains?
// Todo: do we want to manually handle communication of distributed data to skip global atoms?
//    and if so, skip that stored_distributed call?
// Todo: can create_atoms be used twice with two run commands?

/* ---------------------------------------------------------------------- */

FixRigidSmallLSDEM::FixRigidSmallLSDEM(LAMMPS *lmp, int narg, char **arg) :
  FixRigidSmall(lmp, narg, arg), bodyLS(nullptr), bodyownLS(nullptr), global_grids(nullptr), global_grids_min(nullptr),
  global_grids_size(nullptr), id_fix(nullptr), id_fix2(nullptr), quat_custom(nullptr)
{
  maxcut = -1;
  ls_read_flag = 0;
  global_flag = 0;
  distributed_flag = 0;
  storage_mode = ARRAY;
  nmax_distributed = 0;

  n_extra_attributes = 3;

  nmax_bodyLS = nmax_body;
  bodyLS = (BodyLS *) memory->smalloc(nmax_bodyLS * sizeof(BodyLS), "rigid/small/ls/dem:bodyls");
  memory->grow(bodyownLS, atom->nmax, "rigid/small/ls/dem:bodyownLS");
  for (int i = 0; i < nmax_bodyLS; i++)
    bodyLS[i].style = -1;

  // always write restart file
  restart_file = 1;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "ls/storage") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, fmt::format("fix {} ls/storage", style), error);
      if (strcmp(arg[iarg + 1], "array") == 0)
        storage_mode = ARRAY;
      else if (strcmp(arg[iarg + 1], "watershed") == 0)
        storage_mode = WATERSHED;
      else
        error->all(FLERR, "Illegal fix {} command option ls/storage {}", style, arg[iarg + 1]);
      iarg += 2;
    } else {
      iarg ++;
    }
  }

  // set bodyownLS for owned atoms

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

  commflag_ls = PARENT;
  comm_forward += 1 + bodysizeLS;

  if (storage_mode == WATERSHED) {
    atom->add_callback(Atom::BORDER);
    memory->create(node_index, nbody, "rigid/ls/dem:node_index");
  }

  if (langflag)
    error->all(FLERR, "Langevin thermostat not supported with fix rigid/small/ls/dem");

  // currently only call FixRigidSmall::setup_bodies_static() once
  //   need to think how to support this...
  reinitflag = 0;
}

/* ---------------------------------------------------------------------- */

FixRigidSmallLSDEM::~FixRigidSmallLSDEM()
{
  // delete extra property/atom fixes

  if (id_fix && modify->nfix) modify->delete_fix(id_fix);
  delete[] id_fix;
  if (id_fix2 && modify->nfix) modify->delete_fix(id_fix2);
  delete[] id_fix2;

  memory->sfree(bodyLS);
  memory->destroy(global_grids);
  memory->destroy(global_grids_min);
  memory->destroy(global_grids_size);

  memory->destroy(itensor_custom);
  memory->destroy(xcm_custom);
  memory->destroy(mass_custom);
  memory->destroy(quat_custom);
}

/* ---------------------------------------------------------------------- */

int FixRigidSmallLSDEM::setmask()
{
  int mask = FixRigidSmall::setmask();
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::post_constructor()
{
  int tmp1, tmp2;
  index_ls_dem_touch_id = atom->find_custom("ls_dem_touch_id", tmp1, tmp2);

  // May be defined by another fix rigid/ls/dem
  if (index_ls_dem_touch_id == -1) {
    // Store positional information of grain on all atoms
    id_fix = utils::strdup(id + std::string("_FIX_PROP_ATOM"));
    modify->add_fix(fmt::format(
      "{} all property/atom d2_ls_dem_n 3 d2_ls_dem_fs 3 i_ls_dem_touch_id d_ls_dem_fn1 d_ls_dem_fs1 ghost yes writedata no",
       id_fix));
    index_ls_dem_touch_id = atom->find_custom("ls_dem_touch_id", tmp1, tmp2);
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::init()
{
  FixRigidSmall::init();

  if (!atom->xcom_flag || !atom->omega_flag || !atom->quat_flag  || !atom->grid_index_flag)
    error->all(FLERR, "Pair ls/dem requires atom style ls/dem");

  // Pair cutoff sets size of LS around nodes for distributed case
  if (!utils::strmatch(force->pair_style, "^ls/dem"))
    error->all(FLERR, "Must use pair ls/dem with fix rigid/small/ls/dem");
  auto pair = dynamic_cast<PairLSDEM *>(force->pair);
  maxcut = pair->maxcut;

  if (ls_read_flag) return;
  // set in setup_pre_neighbor()

  // allocate storage for LS-derived quantities (used in FixRigidSmall::setup_pre_neighbor() except for quat)

  memory->create(itensor_custom, nlocal_body, 6, "rigid/small:itensor_custom");
  memory->create(xcm_custom, nlocal_body, 3, "rigid/small:xcm_custom");
  memory->create(mass_custom, nlocal_body, "rigid/small:mass_custom");
  memory->create(quat_custom, nlocal_body, 4, "rigid/small:quat_custom");

  // Create complete list of LS gridfile names and data
  //   run in init b/c need to calculate size for fix property/atom
  //   to set comm limits for distributed grid

  read_quat = 0;
  if (inpfile) {
    for (int i = 0; i < nlocal_body; i++)
      mass_custom[i] = -1;

    read_infile();

    for (int i = 0; i < nlocal_body; i++)
      if (mass_custom[i] == -1)
        error->all(FLERR, "Must define all bodies in infile for ls/dem");
  } else {
    Molecule *onemol;
    int *grid_index = atom->grid_index;
    std::string gridfile;
    LSData mydata;
    int set_size;
    for (int i = 0; i < atom->nmolecule; i++) {
      onemol = atom->molecules[i];
      if (!onemol->lsdemflag) continue;

      gridfile = onemol->grid_file;
      if (gridfile_data.find(gridfile) == gridfile_data.end()) {
        set_size = gridfile_data.size();
        gridfile_data[gridfile] = mydata;
        if (onemol->grid_style != DISTRIBUTED && onemol->grid_style != GLOBAL)
          error->all(FLERR, "Invalid_rigid memory model {}", onemol->grid_style);
        gridfile_data[gridfile].style = onemol->grid_style;
        gridfile_data[gridfile].id = set_size;
        id_to_gridfile[set_size] = gridfile;
        gridfile_to_id[gridfile] = set_size;
      } else {
        if (gridfile_data[gridfile].style != onemol->grid_style)
          error->all(FLERR, "Grid file {} has two different memory styles across molecules", gridfile);
      }

      if (onemol->grid_style == GLOBAL)
        global_flag = 1;
      else if (onemol->grid_style == DISTRIBUTED)
        distributed_flag = 1;
    }
  }

  // Read extra data from gridfile (stride, min, size)
  std::string gridfile;
  for (const auto& pair : gridfile_data) {
    gridfile = pair.first;
    read_gridfile(0, gridfile, nullptr);
  }

  LSData mydata;
  double scale;
  double min_stride = DBL_MAX;
  for (const auto& pair : gridfile_data) {
    gridfile = pair.first;
    mydata = pair.second;
    for (const double& scale : mydata.scales)
      min_stride = MIN(min_stride, mydata.stride * scale);
  }

  // All local grids sized on finest grid (fix property/atom requiresfixed-size containers)
  rcell = maxcut / min_stride + 2; // +1 for interpolation +1 for safety

  if (distributed_flag && storage_mode == ARRAY) {
    // todo try remove +1 and cast to int
    for (int a = 0; a < 3; a++) subgrid_size[a] = 2 * rcell + 1;
    if (domain->dimension == 2) subgrid_size[2] = 1;

    int tmp1, tmp2;
    index_grid_values = atom->find_custom("grid_values", tmp1, tmp2);
    index_grid_min = atom->find_custom("grid_min", tmp1, tmp2);

    if (index_grid_values == -1) {
      id_fix2 = utils::strdup(id + std::string("_FIX_PROP_ATOM_2"));
      int ntotal = subgrid_size[0] * subgrid_size[1] * subgrid_size[2];
      if (ntotal > RECOMMENDED_MAX_NGRID)
        error->warning(FLERR, "A large per-atom subgrid of size {}x{}x{} is being allocated for distributed level sets with a cutoff of {} and a min stride of {}", subgrid_size[0], subgrid_size[1], subgrid_size[2], maxcut, min_stride);
      modify->add_fix(fmt::format("{} all property/atom d2_grid_values {} d2_grid_min {} writedata no ghost yes", id_fix2, ntotal, 3));

      index_grid_values = atom->find_custom("grid_values", tmp1, tmp2);
      index_grid_min = atom->find_custom("grid_min", tmp1, tmp2);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::setup_pre_neighbor()
{
  if (!ls_read_flag) {
    // acquire ghost bodies via forward comm
    // set atom2body for ghost atoms via forward comm
    // set atom2body for other owned atoms via reset_atom2body()

    nghost_body = 0;
    commflag = FULL_BODY;
    comm->forward_comm(this);
    reset_atom2body();

    process_levelsets();
  }

  FixRigidSmall::setup_pre_neighbor();

  if (!ls_read_flag) {
    // extra calculations using values from parent

    int iatom;
    double quat_conj[4];
    for (int ibody = 0; ibody < nlocal_body; ibody++) {
      bodyLS[ibody].node_area /= body[ibody].natoms;

      // calculate relative rotation from inerital frame to LS grid
      //   assume all atoms in body have equivalent initial quaterions
      //   (user could incorrectly use diplace_atoms on subset)
      int iatom = body[ibody].ilocal;

      if (read_quat) {
        MathExtra::qconjugate(body[ibody].quat, quat_conj);
        MathExtra::quatquat(quat_conj, quat_custom[ibody], bodyLS[ibody].quatd2g);
      } else {
        MathExtra::qconjugate(body[ibody].quat, quat_conj);
        MathExtra::quatquat(quat_conj, atom->quat[iatom], bodyLS[ibody].quatd2g);
      }

      if (!inpfile) {
        // scale velocities by # of nodes unlike in rigid b/c nodal count is arbitrary
        body[ibody].vcm[0] /= body[ibody].natoms;
        body[ibody].vcm[1] /= body[ibody].natoms;
        body[ibody].vcm[2] /= body[ibody].natoms;
        body[ibody].angmom[0] /= body[ibody].natoms;
        body[ibody].angmom[1] /= body[ibody].natoms;
        body[ibody].angmom[2] /= body[ibody].natoms;
      }
    }

    commflag_ls = FULL_BODY_LS;
    comm->forward_comm(this, 1 + bodysizeLS);
    commflag_ls = PARENT;

    commflag = FULL_BODY;
    comm->forward_comm(this);

    memory->destroy(itensor_custom);
    memory->destroy(xcm_custom);
    memory->destroy(mass_custom);
    memory->destroy(quat_custom);

    ls_read_flag = 1;
  }

  // Sanity check for agreement with parent
  for (int i = 0; i < atom->nlocal + atom->nghost; i++) {
    if (bodyown[i] != bodyownLS[i]) {
      error->one(FLERR, "Different bodies detected, {} vs {}, for atom {}", bodyown[i], bodyownLS[i], atom->tag[i]);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::process_levelsets()
{
  // Note, FixRigidSmall has not yet populated body structure, use custom fields

  int ibody, i, a;
  int dimension = domain->dimension;

  int *touch_id = atom->ivector[index_ls_dem_touch_id];
  int *grid_index = atom->grid_index;

  tagint *molecule = atom->molecule;
  double **quat_atom = atom->quat;
  double **xcom = atom->xcom;
  double **x = atom->x;
  LSData mydata;

  // -------------------------------- //
  // Calc memory info for level sets  //
  // -------------------------------- //

  // Find max grid size and location+size of all globally stored grids
  int *ntotal_global;
  memory->create(ntotal_global, nbody, "rigid/ls/dem:ntotal_global");

  std::string gridfile;
  int index_global_grid, grid_size_flat, max_grid_size_flat;
  index_global_grid = max_grid_size_flat = 0;
  for (const auto& pair : gridfile_data) {
    gridfile = pair.first;

    grid_size_flat = gridfile_data[gridfile].grid_size[0];
    grid_size_flat *= gridfile_data[gridfile].grid_size[1];
    grid_size_flat *= gridfile_data[gridfile].grid_size[2];
    max_grid_size_flat = MAX(max_grid_size_flat, grid_size_flat);

    if (gridfile_data[gridfile].style == GLOBAL) {
      gridfile_data[gridfile].grid_index = index_global_grid;
      ntotal_global[index_global_grid] = grid_size_flat;
      index_global_grid += 1;
    } else {
      gridfile_data[gridfile].grid_index = -1;
    }
  }

  // ------------------------------- //
  // Copy data to body + bodyLS      //
  // ------------------------------- //

  int *mask = atom->mask;
  for (i = 0; i < atom->nlocal; i++)
    if (mask[i] & groupbit)
      touch_id[i] = -1; // set to zero for preexisting atoms (rest set in set_array)

  // Copy body geometry + infile data into body structure
  if (inpfile) {
    tagint myid;
    for (int i = 0; i < atom->nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      if (bodyown[i] < 0) continue;
      myid = molecule[i];
      ibody = atom2body[i];

      // check all bodies in infile to find match, O(nlocal_body x nbody)
      //   only done once and presumably nbody small if using infile
      for (const auto& pair : gridfile_data) {
        gridfile = pair.first;
        mydata = pair.second;
        for (int j = 0; j < mydata.bodies.size(); j++) {
          if (mydata.bodies[j] == myid) {
            bodyLS[ibody].file_id = mydata.id;
            bodyLS[ibody].grid_scale = mydata.scales[j];
          }
        }
      }
    }
  } else {
    Molecule *onemol;
    for (i = 0; i < atom->nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      if (bodyownLS[i] == -1) continue;
      onemol = atom->molecules[grid_index[i]];
      ibody = atom2body[i];
      bodyLS[ibody].file_id = gridfile_to_id[onemol->grid_file];
      bodyLS[ibody].style = onemol->grid_style;
      bodyLS[ibody].grid_scale = onemol->grid_scale;
      mass_custom[ibody] = onemol->masstotal;
      xcm_custom[ibody][0] = xcom[i][0];
      xcm_custom[ibody][1] = xcom[i][1];
      xcm_custom[ibody][2] = xcom[i][2];
    }
  }

  // Copy gridfile extra data to bodyLS structure
  for (int i = 0; i < atom->nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (bodyown[i] < 0) continue;
    ibody = atom2body[i];
    mydata = gridfile_data[id_to_gridfile[bodyLS[ibody].file_id]];
    bodyLS[ibody].grid_index = mydata.grid_index;
    bodyLS[ibody].style = mydata.style;
    bodyLS[ibody].grid_stride = mydata.stride;
  }

  // Send to ghosts
  nghost_bodyLS = 0;
  commflag_ls = FULL_BODY_LS;
  comm->forward_comm(this, 1 + bodysizeLS);
  commflag_ls = PARENT;

  // ------------------------------- //
  // Read + process LS grid data     //
  // ------------------------------- //

  double *temp_grid_values;
  memory->create(temp_grid_values, max_grid_size_flat, "rigid/small/lsdem:temp_grid_values");

  double **grid_values, **grid_min_local;
  int nlocal = atom->nlocal;
  if (distributed_flag && storage_mode == ARRAY) {
    grid_values = atom->darray[index_grid_values];
    grid_min_local = atom->darray[index_grid_min];
  }

  if (global_flag) {
    if (index_global_grid == 0)
      error->all(FLERR, "No global grids defined but global_flag set");

    if (storage_mode == WATERSHED) {
      global_ws_tables.resize(index_global_grid);
      global_ws_buffers.resize(index_global_grid);
    } else {
      memory->create_ragged(global_grids, index_global_grid, ntotal_global, "rigid/small/ls/dem:global_grids");
      memory->create(global_grids_min, index_global_grid, 3, "rigid/small/ls/dem:global_grids_min");
      memory->create(global_grids_size, index_global_grid, 3, "rigid/small/ls/dem:global_grids_size");
    }
  }

  if (distributed_flag) {
    if (storage_mode == WATERSHED) {
      dist_ws_tables.resize(nlocal);
      dist_ws_buffers.resize(nlocal);
    }
  }

  // --------------------------------- //
  // If watershed, calculate node type //
  // --------------------------------- //

  std::vector <std::set <int>> node_bins;
  std::vector <std::set <int>> node_buffer_bins;
  int *global_node_bins = nullptr;
  if (storage_mode == WATERSHED) {

    int max_node_index = -1;
    for (ibody = 0; ibody < nlocal_bodyLS + nghost_bodyLS; ibody++) {

      tagint min_id = -1;
      tagint *tag = atom->tag;
      for (i = 0; i < nlocal; i++) {
        if (atom2body[i] != ibody) continue;

        if (min_id == -1 || tag[i] < min_id)
          min_id = tag[i];
      }

      for (i = 0; i < nlocal; i++) {
        if (atom2body[i] != ibody) continue;
        node_index[i] = tag[i] - min_id;
        max_node_index = MAX(max_node_index, node_index[i]);
      }
    }

    MPI_Allreduce(&max_node_index, &max_node_index, 1, MPI_INT, MPI_MAX, world);

    node_bins.resize(max_node_index + 1);
    node_buffer_bins.resize(max_node_index + 1);
  }

  // ------------------------------ //
  // Read and store level sets      //
  // ------------------------------ //

  double dx[3], gmin[3];
  int index_local, index_global, error_code, error_code_global, nx[3];
  for (const auto& pair : gridfile_data) {
    gridfile = pair.first;

    read_gridfile(1, gridfile, temp_grid_values);

    // Compute grain properties (volume, area, inertia...) for each body using this grid
    for (ibody = 0; ibody < nlocal_bodyLS; ibody++) {
      if (pair.second.id != bodyLS[ibody].file_id)
        continue;
      compute_grain_properties(ibody, gridfile_data[gridfile].grid_size, gridfile_data[gridfile].grid_min, temp_grid_values);
    }

    if (gridfile_data[gridfile].style == GLOBAL)
      index_global_grid = gridfile_data[gridfile].grid_index;

    if (storage_mode == ARRAY) {
      if (gridfile_data[gridfile].style == GLOBAL) {
        index_global_grid = gridfile_data[gridfile].grid_index;
        for (int n = 0; n < ntotal_global[index_global_grid]; n++)
          // Unscaled values (and min) stored globally to avoid duplicating memory
          global_grids[index_global_grid][n] = temp_grid_values[n];

        for (a = 0; a < 3; a++) {
          global_grids_min[index_global_grid][a] = gridfile_data[gridfile].grid_min[a];
          global_grids_size[index_global_grid][a] = gridfile_data[gridfile].grid_size[a];
        }
      } else {

        for (i = 0; i < atom->nlocal; i++) {
          if (!(mask[i] & groupbit)) continue;
          ibody = atom2body[i];

          // Ideally use list of all atoms in body... not sure this exists
          if (pair.second.id != bodyLS[ibody].file_id)
            continue;

          nx[0] = gridfile_data[gridfile].grid_size[0];
          nx[1] = gridfile_data[gridfile].grid_size[1];
          nx[2] = gridfile_data[gridfile].grid_size[2];

          MathExtra::scale3(bodyLS[ibody].grid_scale, gridfile_data[gridfile].grid_min, gmin);

          // Location of atom/node relative to CoM (not yet defined in body structure) + remap periodically
          MathExtra::sub3(x[i], xcm_custom[ibody], dx);
          domain->minimum_image(FLERR, dx[0], dx[1], dx[2]);

          // Calculate local grid values and minima
          error_code = store_distributed(i, dimension, nx, subgrid_size, bodyLS[ibody].grid_stride, bodyLS[ibody].grid_scale,
                                         rcell, dx, gmin, quat_atom[i], temp_grid_values, grid_min_local[i], grid_values[i]);

          if (error_code == -1)
            error->one(FLERR, "Unexpected out of bounds error in distributed level set creation, atom {}", atom->tag[i]);

          MPI_Allreduce(&error_code, &error_code_global, 1, MPI_INT, MPI_MAX, world);
          if (error_code_global && comm->me == 0)
            error->warning(FLERR, "Level set of body {} does not include a large enough buffer for the distributed grid cutoff on "
              "atom {}\nLocal grid padded with BIG values\nWarning will not print for other nodes in this body.", ibody, atom->tag[i]);
        }
      }
    } else {
      // Calculate watershed and temporarily store peratom data

      // First, accumulate list of all bins in the grain

      // calculate the # bins & nodes in this gridfile
      int nbin, nx, ny, nz;
      int max_node_index = -1;
      for (ibody = 0; ibody < nlocal_bodyLS; ibody++) {
        if (pair.second.id != bodyLS[ibody].file_id)
          continue;

        gridfile = bodyLS[ibody].grid_index;

        nx = gridfile_data[gridfile].grid_size[0];
        ny = gridfile_data[gridfile].grid_size[1];
        nz = gridfile_data[gridfile].grid_size[2];
        nbin = nx * ny * nz;

        max_node_index = MAX(max_node_index, node_index[i]);
      }
      max_node_index += 1; // zero indexed

      if (nbin > max_grid_size_flat) {
        max_grid_size_flat = nbin;
        memory->grow(global_node_bins, max_grid_size_flat, "rigid/ls/dem:global_node_bins");
      }


      // Next find the bins which contain nodes
      int currentbin, ix[3];
      double x_local[3], quat_conj[4];
      for (i = 0; i < nlocal; i++) {
        ibody = atom2body[i];
        if (ibody != pair.second.id)
          continue;

        nx = gridfile_data[gridfile].grid_size[0];
        ny = gridfile_data[gridfile].grid_size[1];
        nz = gridfile_data[gridfile].grid_size[2];
        MathExtra::sub3(x[i], body[ibody].xcm, dx);

        MathExtra::qconjugate(quat_atom[i], quat_conj);
        MathExtra::quatrotvec(quat_conj, dx, x_local);
        MathExtra::sub3(x_local, gridfile_data[gridfile].grid_min, x_local);
        MathExtra::scale3(1.0 / bodyLS[ibody].grid_stride, x_local);

        ix[0] = int(x_local[0]);
        ix[1] = int(x_local[1]);
        ix[2] = int(x_local[2]);

        currentbin = ix[0] + ix[1] * nx + ix[2] * nx * ny;

        global_node_bins[node_index[i]] = currentbin;
      }
      MPI_Allreduce(global_node_bins, global_node_bins, nbin, MPI_INT, MPI_MAX, world);

      // Now, run watershed operation
      int j, ns, newbin, current_walker;
      int max_bins_per_node = -1;
      int ntotal = nlocal + atom->nghost;
      std::vector <int> bin_owners(nbin);
      std::vector <std::pair <int, int>> walkers;
      std::vector <std::pair <int, int>> next_walkers;

      for (auto& s : node_bins) s.clear();
      for (auto& s : node_buffer_bins) s.clear();

      walkers.clear();
      next_walkers.clear();
      for (i = 0; i < nbin; i++)
        bin_owners[i] = -1;

      for (i = 0; i < max_node_index; i++) {
        currentbin = global_node_bins[i];

        // Skip if not a bin which contains a node
        if (currentbin < 0) continue;

        bin_owners[currentbin] = i;
        node_bins[i].insert(currentbin);
        walkers.emplace_back(std::make_pair(i, currentbin));
      }

      while (!walkers.empty()) {
        // find unvisited sites and add new walkers
        //   if a walker borders a visited site, only add owner to create a buffer
        int cycle = 0;
        for (auto walker : walkers) {
          i = walker.first;
          currentbin = walker.second;

          cycle += 1;

          for (int a = 0; a < dimension; a++) {
            if (a == 0) ns = 1;
            else if (a == 1) ns = nx;
            else ns = nx * ny;
            for (int sign = -1; sign <= 1; sign += 2) {

              newbin = currentbin + sign * ns;

              if (newbin < 0 || newbin >= nbin) continue;
              if (temp_grid_values[newbin] > bodyLS[ibody].grid_stride) continue;
              if (temp_grid_values[newbin] < -rcell) continue;

              // if unvisited, add walker
              if (bin_owners[newbin] == -1)
                next_walkers.emplace_back(std::make_pair(i, newbin));
            }
          }
        }

        // add ownership from all of these walkers
        //   break ties depending on which atom owns fewer bins
        for (auto walker : walkers) {
          i = walker.first;
          currentbin = walker.second;

          if (bin_owners[currentbin] == -1) {
            bin_owners[currentbin] = i;
             if (i < nlocal)
              node_bins[node_index[i]].insert(currentbin);
          } else {
            j = bin_owners[currentbin];
            if (node_bins[node_index[i]].size() < node_bins[node_index[j]].size()) {
              bin_owners[currentbin] = i;
              if (i < nlocal)
                node_bins[node_index[i]].insert(currentbin);
              node_bins[node_index[j]].erase(currentbin);
            }
          }
        }

        // replace walkers
        std::swap(walkers, next_walkers);
        next_walkers.clear();
      }

      // Create buffer (1st and 2nd neighbors of all bins)
      for (i = 0; i < max_node_index; i++) {
        for (const auto& currentbin : node_bins[i]) {
          for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
              for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                if (dimension == 2 && dz != 0) continue;
                newbin = currentbin + dx + dy * nx + dz * nx * ny;
                if (newbin < 0 || newbin >= nbin)
                  error->one(FLERR, "Bad bin index in watershed buffer creation");
                if (bin_owners[newbin] != i)
                  node_buffer_bins[i].insert(newbin);
              }
            }
          }
        }
      }

      // Relocate bins to global per-grain/node-type storage
      if (global_flag) {
        global_ws_tables[index_global].resize(max_node_index);
        global_ws_buffers[index_global].resize(max_node_index);

        for (i = 0; i < max_node_index; i++) {
          for (auto bin : node_bins[i])
            global_ws_tables[index_global][i].insert(std::make_pair(bin, temp_grid_values[bin]));
          for (auto bin : node_buffer_bins[i])
            global_ws_buffers[index_global][i].insert(std::make_pair(bin, temp_grid_values[bin]));
        }
      } else {
        int nt, size_sum;
        for (i = 0; i < nlocal; i++) {
          ibody = atom2body[i];
          if (ibody != pair.second.id)
            continue;

          nt = node_index[i];
          size_sum = (int) (node_bins[nt].size() + node_buffer_bins[nt].size());
          max_bins_per_node = MAX(max_bins_per_node, size_sum);

          for (auto bin : node_bins[nt])
            dist_ws_tables[i].insert(std::make_pair(bin, temp_grid_values[bin]));
          for (auto bin : node_buffer_bins[nt])
            dist_ws_buffers[i].insert(std::make_pair(bin, temp_grid_values[bin]));
        }
      }
    }
  }

  memory->destroy(temp_grid_values);
  memory->destroy(ntotal_global);


  // ------------------------------ //
  // Set communication variables    //
  // ------------------------------ //

  if (storage_mode == WATERSHED) {
    comm_border = 1;
    if (distributed_flag) {
      int max_nbins = -1;
      for (i = 0; i < nlocal; i++)
        max_nbins = MAX(max_nbins, int(dist_ws_tables[i].size()) + int(dist_ws_buffers[i].size()));
      MPI_Allreduce(&max_nbins, &nmax_distributed, 1, MPI_INT, MPI_MAX, world);

      maxexchange = 2 + 2 * nmax_distributed;  // +2 for # of owned & buffer bins
      comm_border += 2 + 2 * nmax_distributed; // +2 x # bins for bin, value pairs
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::setup_pre_force(int vflag)
{
  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::pre_force(int vflag)
{
  commflag_ls = PREFORCE_LS;
  comm->forward_comm(this, 11);
  commflag_ls = PARENT;

  reset_atom2body_ghost();
}

/* ----------------------------------------------------------------------
   reset atom2body for all ghost atoms possible, namely those  which
   the atom that owns the grid is also a ghost/owned
   otherwise leave atom2body at -1
   will error in pair if this atom needed
   comm cutoff must be at least 1/2 of longest body, can this be relaxed?
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::reset_atom2body_ghost()
{
  int iowner;

  // iowner = index of atom that owns the body that atom I is in

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int *mask = atom->mask;
  for (int i = nlocal; i < nlocal + nghost; i++) {
    if (!(atom->mask[i] & groupbit)) continue;
    atom2body[i] = -1;
    if (bodytag[i]) {
      iowner = atom->map(bodytag[i]);
      if (iowner != -1)
        atom2body[i] = bodyown[iowner];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::initial_integrate(int vflag)
{
  FixRigidSmall::initial_integrate(vflag);

  // communicate quatd2g
  commflag_ls = INITIAL_LS;
  comm->forward_comm(this, 4);
  commflag_ls = PARENT;

  int *mask = atom->mask;
  double **grain_com = atom->xcom;
  double **grain_quat = atom->quat;
  double **grain_omega = atom->omega;

  int ibody;
  for (int i = 0; i < atom->nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    ibody = atom2body[i];

    if (ibody < 0)
      error->one(FLERR, "Bad body index for atom {}", atom->tag[i]);
    Body *b = &body[ibody];

    grain_com[i][0] = b->xcm[0];
    grain_com[i][1] = b->xcm[1];
    grain_com[i][2] = b->xcm[2];

    // calculate rotation from current orientation to LS grid
    MathExtra::quatquat(b->quat, bodyLS[ibody].quatd2g, grain_quat[i]);

    grain_omega[i][0] = b->omega[0];
    grain_omega[i][1] = b->omega[1];
    grain_omega[i][2] = b->omega[2];
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::pre_neighbor()
{
  FixRigidSmall::pre_neighbor();

  nghost_bodyLS = 0;
  commflag_ls = FULL_BODY_LS;
  comm->forward_comm(this, 1 + bodysizeLS);
  commflag_ls = PARENT;
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

  // sum over atoms to get force and torque on rigid body

  int *mask = atom->mask;
  double **x = atom->x;
  double **f = atom->f;
  double **torque = atom->torque;
  int nlocal = atom->nlocal;
  double *fcm,*tcm;

  for (ibody = 0; ibody < nlocal_bodyLS + nghost_bodyLS; ibody++) {
    fcm = body[ibody].fcm;
    fcm[0] = fcm[1] = fcm[2] = 0.0;
    tcm = body[ibody].torque;
    tcm[0] = tcm[1] = tcm[2] = 0.0;
  }

  for (i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    fcm = b->fcm;
    fcm[0] += f[i][0];
    fcm[1] += f[i][1];
    fcm[2] += f[i][2];

    tcm = b->torque;
    tcm[0] += torque[i][0];
    tcm[1] += torque[i][1];
    tcm[2] += torque[i][2];
  }

  // subtract gravity forces from any atoms

  if (id_gravity) {
    int *type = atom->type;
    int *mask = atom->mask;
    double *rmass = atom->rmass;
    double *mass = atom->mass;
    double massone;
    for (i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      if (!(mask[i] & grav_group_bit)) continue;
      if (atom2body[i] < 0) continue;
      Body *b = &body[atom2body[i]];

      fcm = b->fcm;

      if (rmass)
        massone = rmass[i];
      else
        massone = mass[type[i]];

      fcm[0] -= gvec[0] * massone;
      fcm[1] -= gvec[1] * massone;
      fcm[2] -= gvec[2] * massone;
    }
  }

  // reverse communicate fcm, torque of all bodies

  commflag = FORCE_TORQUE;
  comm->reverse_comm(this, 6);

  // add gravity force to COM of each body

  if (id_gravity) {
    double mass;
    int *mask = atom->mask;
    for (ibody = 0; ibody < nlocal_bodyLS; ibody++) {
      i = body[ibody].ilocal;
      if (!(mask[i] & grav_group_bit)) continue;

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

  if (storage_mode == WATERSHED) {
    memory->grow(node_index, nmax, "rigid/small/ls/dem:node_index");
    if (distributed_flag) {
      dist_ws_tables.resize(nmax);
      dist_ws_buffers.resize(nmax);
    }
  }
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

  if (!ls_read_flag) return; // if called before processing LS (e.g. when sorting in setup)

  if (storage_mode == WATERSHED) node_index[j] = node_index[i];

  if (distributed_flag) {
    if (bodyLS[atom2body[i]].style != DISTRIBUTED)
      return;

    if (storage_mode == WATERSHED) {
      dist_ws_tables[j].clear();
      dist_ws_buffers[j].clear();

      for (auto entry : dist_ws_tables[i])
        dist_ws_tables[j][entry.first] = entry.second;
      for (auto entry : dist_ws_buffers[i])
        dist_ws_buffers[j][entry.first] = entry.second;

      dist_ws_tables[i].clear();
      dist_ws_buffers[i].clear();
    }
  }
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::set_arrays(int i)
{
  FixRigidSmall::set_arrays(i);
  bodyownLS[i] = -1;

  atom->ivector[index_ls_dem_touch_id][i] = -1;
  if (storage_mode == WATERSHED)
    node_index[i] = -1;
}

/* ----------------------------------------------------------------------
   write out restart info for mass, COM, inertia tensor to file
   identical format to inpfile option, so info can be read in when restarting
   each proc contributes info for rigid bodies it owns
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::write_restart_file(const char *file)
{
  FILE *fp;

  // do not write file if bodies have not yet been initialized

  if (!setupflag) return;

  // proc 0 opens file and writes header

  if (comm->me == 0) {
    auto outfile = std::string(file) + ".rigid";
    fp = fopen(outfile.c_str(),"w");
    if (fp == nullptr)
      error->one(FLERR, "Cannot open fix {} restart file {}: {}",
                 style, outfile, utils::getsyserror());

    utils::print(fp,"# fix rigid mass, COM, inertia tensor info for "
               "{} bodies on timestep {}\n\n",nbody,update->ntimestep);
    utils::print(fp,"{}\n",nbody);
  }

  // communication buffer for all my rigid body info
  // max_size = largest buffer needed by any proc
  // ncol = # of values per line in output file

  int ncol = ATTRIBUTE_PERBODY + n_extra_attributes + 4;
  int sendrow = nlocal_body;
  int maxrow;
  MPI_Allreduce(&sendrow,&maxrow,1,MPI_INT,MPI_MAX,world);

  double **buf;
  if (comm->me == 0) memory->create(buf,MAX(1,maxrow),ncol,"rigid/small:buf");
  else memory->create(buf,MAX(1,sendrow),ncol,"rigid/small/ls/dem:buf");

  // pack my rigid body info into buf
  // compute I tensor against xyz axes from diagonalized I and current quat
  // Ispace = P Idiag P_transpose
  // P is stored column-wise in exyz_space

  double p[3][3],pdiag[3][3],ispace[3][3];

  for (int i = 0; i < nlocal_body; i++) {
    MathExtra::col2mat(body[i].ex_space,body[i].ey_space,body[i].ez_space,p);
    MathExtra::times3_diag(p,body[i].inertia,pdiag);
    MathExtra::times3_transpose(pdiag,p,ispace);

    buf[i][0] = atom->molecule[body[i].ilocal];
    buf[i][1] = body[i].mass;
    buf[i][2] = body[i].xcm[0];
    buf[i][3] = body[i].xcm[1];
    buf[i][4] = body[i].xcm[2];
    buf[i][5] = ispace[0][0];
    buf[i][6] = ispace[1][1];
    buf[i][7] = ispace[2][2];
    buf[i][8] = ispace[0][1];
    buf[i][9] = ispace[0][2];
    buf[i][10] = ispace[1][2];
    buf[i][11] = body[i].vcm[0];
    buf[i][12] = body[i].vcm[1];
    buf[i][13] = body[i].vcm[2];
    buf[i][14] = body[i].angmom[0];
    buf[i][15] = body[i].angmom[1];
    buf[i][16] = body[i].angmom[2];
    buf[i][17] = (body[i].image & IMGMASK) - IMGMAX;
    buf[i][18] = (body[i].image >> IMGBITS & IMGMASK) - IMGMAX;
    buf[i][19] = (body[i].image >> IMG2BITS) - IMGMAX;

    // LSDEM quantities
    buf[i][20] = (double) bodyLS[i].style;
    buf[i][21] = bodyLS[i].grid_scale;
    buf[i][22] = (double) bodyLS[i].file_id;
    buf[i][23] = body[i].quat[0];
    buf[i][24] = body[i].quat[1];
    buf[i][25] = body[i].quat[2];
    buf[i][26] = body[i].quat[3];
  }

  // write one chunk of rigid body info per proc to file
  // proc 0 pings each proc, receives its chunk, writes to file
  // all other procs wait for ping, send their chunk to proc 0

  int tmp,recvrow;

  if (comm->me == 0) {
    MPI_Status status;
    MPI_Request request;
    for (int iproc = 0; iproc < comm->nprocs; iproc++) {
      if (iproc) {
        MPI_Irecv(&buf[0][0],maxrow*ncol,MPI_DOUBLE,iproc,0,world,&request);
        MPI_Send(&tmp,0,MPI_INT,iproc,0,world);
        MPI_Wait(&request,&status);
        MPI_Get_count(&status,MPI_DOUBLE,&recvrow);
        recvrow /= ncol;
      } else recvrow = sendrow;

      for (int i = 0; i < recvrow; i++)
        fprintf(fp,"%d %-1.16e %-1.16e %-1.16e %-1.16e "
                "%-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e "
                "%-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d "
                "%d %-1.16e %s %-1.16e %-1.16e %-1.16e %-1.16e\n",
                static_cast<int> (buf[i][0]),buf[i][1],
                buf[i][2],buf[i][3],buf[i][4],
                buf[i][5],buf[i][6],buf[i][7],
                buf[i][8],buf[i][9],buf[i][10],
                buf[i][11],buf[i][12],buf[i][13],
                buf[i][14],buf[i][15],buf[i][16],
                static_cast<int> (buf[i][17]),
                static_cast<int> (buf[i][18]),
                static_cast<int> (buf[i][19]),
                static_cast<int> (buf[i][20]), // Start of LS
                buf[i][21],
                id_to_gridfile[static_cast<int> (buf[i][22])].c_str(),
                buf[i][23],buf[i][24],buf[i][25],buf[i][26]);
    }

  } else {
    MPI_Recv(&tmp,0,MPI_INT,0,0,world,MPI_STATUS_IGNORE);
    MPI_Rsend(&buf[0][0],sendrow*ncol,MPI_DOUBLE,0,0,world);
  }

  // clean up and close file

  memory->destroy(buf);
  if (comm->me == 0) fclose(fp);
}

/* ----------------------------------------------------------------------
   initialize a molecule inserted by another fix, e.g. deposit or pour
   called when molecule is created
   nlocalprev = # of atoms on this proc before molecule inserted
   tagprev = atom ID previous to new atoms in the molecule
   xgeom = geometric center of new molecule
   vcm = COM velocity of new molecule
   quat = rotation of new molecule (around geometric center)
          relative to template in Molecule class
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::set_molecule(int nlocalprev, tagint tagprev, int imol,
                                 double *xgeom, double *vcm, double *quat)
{
  //TODO: modify to update LSDEM properties
  //   note that all global grids should be stored since molecules defined in advance
  //   need mechanism to store grids before distribution... or just require global grids for pouring
  //   need to update maxmol for pair ls/dem

  error->one(FLERR, "Molecule insertion not yet supported for fix rigid/small/ls/dem");

  int m;
  double ctr2com[3],ctr2com_rotate[3];
  double rotmat[3][3];

  // increment total # of rigid bodies

  nbody++;

  // loop over atoms I added for the new body

  int nlocal = atom->nlocal;
  if (nlocalprev == nlocal) return;

  tagint *tag = atom->tag;

  for (int i = nlocalprev; i < nlocal; i++) {
    bodytag[i] = tagprev + onemols[imol]->comatom;
    if (tag[i]-tagprev == onemols[imol]->comatom) bodyown[i] = nlocal_bodyLS;

    m = tag[i] - tagprev-1;
    displace[i][0] = onemols[imol]->dxbody[m][0];
    displace[i][1] = onemols[imol]->dxbody[m][1];
    displace[i][2] = onemols[imol]->dxbody[m][2];

    if (extended) {
      eflags[i] = 0;
      if (onemols[imol]->radiusflag) {
        eflags[i] |= SPHERE;
        eflags[i] |= OMEGA;
        eflags[i] |= TORQUE;
      }
    }

    if (bodyown[i] >= 0) {
      if (nlocal_bodyLS == nmax_body) grow_body();
      Body *b = &body[nlocal_bodyLS];
      b->mass = onemols[imol]->masstotal;
      b->natoms = onemols[imol]->natoms;
      b->xgc[0] = xgeom[0];
      b->xgc[1] = xgeom[1];
      b->xgc[2] = xgeom[2];

      // new COM = Q (onemols[imol]->xcm - onemols[imol]->center) + xgeom
      // Q = rotation matrix associated with quat

      MathExtra::quat_to_mat(quat,rotmat);
      MathExtra::sub3(onemols[imol]->com,onemols[imol]->center,ctr2com);
      MathExtra::matvec(rotmat,ctr2com,ctr2com_rotate);
      MathExtra::add3(ctr2com_rotate,xgeom,b->xcm);

      b->vcm[0] = vcm[0];
      b->vcm[1] = vcm[1];
      b->vcm[2] = vcm[2];
      b->inertia[0] = onemols[imol]->inertia[0];
      b->inertia[1] = onemols[imol]->inertia[1];
      b->inertia[2] = onemols[imol]->inertia[2];

      // final quat is product of insertion quat and original quat
      // true even if insertion rotation was not around COM

      MathExtra::quatquat(quat,onemols[imol]->quat,b->quat);
      MathExtra::q_to_exyz(b->quat,b->ex_space,b->ey_space,b->ez_space);

      MathExtra::transpose_matvec(b->ex_space,b->ey_space,b->ez_space,
                                  ctr2com_rotate,b->xgc_body);
      b->xgc_body[0] *= -1;
      b->xgc_body[1] *= -1;
      b->xgc_body[2] *= -1;

      b->angmom[0] = b->angmom[1] = b->angmom[2] = 0.0;
      b->omega[0] = b->omega[1] = b->omega[2] = 0.0;
      b->conjqm[0] = b->conjqm[1] = b->conjqm[2] = b->conjqm[3] = 0.0;

      b->image = ((imageint) IMGMAX << IMG2BITS) |
        ((imageint) IMGMAX << IMGBITS) | IMGMAX;
      b->ilocal = i;
      nlocal_bodyLS++;
    }
  }
}

/* ----------------------------------------------------------------------
   pack values for border communication at re-neighboring
------------------------------------------------------------------------- */

int FixRigidSmallLSDEM::pack_border(int n, int *list, double *buf)
{
  int i, j, k;
  int m = 0;

  if (storage_mode != WATERSHED) return 0;

  for (i = 0; i < n; i++) {
    j = list[i];

    if (distributed_flag) {

      if (bodyLS[bodyownLS[j]].style != DISTRIBUTED) {
        buf[m++] = 0;
        continue;
      }
      buf[m++] = 1;

      buf[m++] = (double) (dist_ws_tables[j].size());
      for (auto bin_pair : dist_ws_tables[j]) {
        buf[m++] = ubuf(bin_pair.first).d;
        buf[m++] = bin_pair.second;
      }
      buf[m++] = (double) (dist_ws_buffers[j].size());
      for (auto buffer_pair : dist_ws_buffers[j]) {
        buf[m++] = ubuf(buffer_pair.first).d;
        buf[m++] = buffer_pair.second;
      }
    }

    buf[m++] = ubuf(node_index[j]).d;
  }
  return m;
}

/* ----------------------------------------------------------------------
   unpack values for border communication at re-neighboring
------------------------------------------------------------------------- */

int FixRigidSmallLSDEM::unpack_border(int n, int first, double *buf)
{
  int i, last, flag;
  std::size_t k;

  if (storage_mode != WATERSHED) return 0;

  int m = 0;
  last = first + n;
  for (i = first; i < last; i++) {

    if (distributed_flag) {
      flag = buf[m++];
      if (flag == 0) continue; // no grid info for this atom

      std::size_t n_table = (std::size_t) buf[m++];
      dist_ws_tables[i].clear();
      for (k = 0; k < n_table; k++) {
        int bin = ubuf(buf[m++]).i;
        double value = buf[m++];
        dist_ws_tables[i].insert(std::make_pair(bin, value));
      }
      std::size_t n_buffer = (std::size_t) buf[m++];
      dist_ws_buffers[i].clear();
      for (k = 0; k < n_buffer; k++) {
        int bin = ubuf(buf[m++]).i;
        double value = buf[m++];
        dist_ws_buffers[i].insert(std::make_pair(bin, value));
      }
    }

    node_index[i] = (int) ubuf(buf[m++]).i;
  }

  return m;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixRigidSmallLSDEM::pack_exchange(int i, double *buf)
{
  int m = FixRigidSmall::pack_exchange(i, buf);

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

  // handle distributed watershed here to adjust size

  if (distributed_flag && storage_mode == WATERSHED) {
    if (bodyLS[bodyownLS[i]].style != DISTRIBUTED) {
      buf[m++] = 0;
      return m;
    }

    buf[m++] = 1;

    buf[m++] = (double) (dist_ws_tables[i].size());
    for (auto bin_pair : dist_ws_tables[i]) {
      buf[m++] = ubuf(bin_pair.first).d;
      buf[m++] = bin_pair.second;
    }
    buf[m++] = (double) (dist_ws_buffers[i].size());
    for (auto buffer_pair : dist_ws_buffers[i]) {
      buf[m++] = ubuf(buffer_pair.first).d;
      buf[m++] = buffer_pair.second;
    }
  }

  if (storage_mode == WATERSHED)
    buf[m++] = ubuf(node_index[i]).d;

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


  if (distributed_flag && storage_mode == WATERSHED) {
    int flag = buf[m++];
    if (flag == 0)
      return m;

    std::size_t n_table = (std::size_t) buf[m++];
    dist_ws_tables[nlocal].clear();
    for (std::size_t k = 0; k < n_table; k++) {
      int bin = ubuf(buf[m++]).i;
      double value = buf[m++];
      dist_ws_tables[nlocal].insert(std::make_pair(bin, value));
    }
    std::size_t n_buffer = (std::size_t) buf[m++];
    dist_ws_buffers[nlocal].clear();
    for (std::size_t k = 0; k < n_buffer; k++) {
      int bin = ubuf(buf[m++]).i;
      double value = buf[m++];
      dist_ws_buffers[nlocal].insert(std::make_pair(bin, value));
    }
  }

  if (storage_mode == WATERSHED)
    node_index[nlocal] = (int) ubuf(buf[m++]).i;

  return m;
}

/* ----------------------------------------------------------------------
   only pack body info if own or ghost atom owns the body
   for FULL_BODY, send 0/1 flag with every atom
------------------------------------------------------------------------- */

int FixRigidSmallLSDEM::pack_forward_comm(int n, int *list, double *buf,
                                     int /*pbc_flag*/, int * /*pbc*/)
{
  int m, i, j;
  if (commflag_ls == INITIAL_LS) {
    double *quatd2g;
    m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyownLS[j] < 0) continue;
      quatd2g = bodyLS[bodyownLS[j]].quatd2g;
      buf[m++] = quatd2g[0];
      buf[m++] = quatd2g[1];
      buf[m++] = quatd2g[2];
      buf[m++] = quatd2g[3];
    }
  } else if (commflag_ls == PREFORCE_LS) {
    double **grain_com = atom->xcom;
    double **grain_quat = atom->quat;
    double **grain_omega = atom->omega;
    m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(bodytag[j]).d;
      buf[m++] = grain_com[j][0];
      buf[m++] = grain_com[j][1];
      buf[m++] = grain_com[j][2];
      buf[m++] = grain_omega[j][0];
      buf[m++] = grain_omega[j][1];
      buf[m++] = grain_omega[j][2];
      buf[m++] = grain_quat[j][0];
      buf[m++] = grain_quat[j][1];
      buf[m++] = grain_quat[j][2];
      buf[m++] = grain_quat[j][3];
    }
  } else if (commflag_ls == FULL_BODY_LS) {
    m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyownLS[j] < 0) buf[m++] = 0;
      else  {
        buf[m++] = 1;
        memcpy(&buf[m], &bodyLS[bodyownLS[j]], sizeof(BodyLS));
        m += bodysizeLS;
      }
    }
  } else {
    m = FixRigidSmall::pack_forward_comm(n, list, buf, 0, nullptr);
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
  int m, i, j, last;
  last = first + n;
  m = 0;

  if (commflag_ls == INITIAL_LS) {
    double *quatd2g;
    for (i = first; i < last; i++) {
      if (bodyownLS[i] < 0) continue;
      quatd2g = bodyLS[bodyownLS[i]].quatd2g;
      quatd2g[0] = buf[m++];
      quatd2g[1] = buf[m++];
      quatd2g[2] = buf[m++];
      quatd2g[3] = buf[m++];
    }
  } else if (commflag_ls == PREFORCE_LS) {
    double **grain_com = atom->xcom;
    double **grain_quat = atom->quat;
    double **grain_omega = atom->omega;
    for (i = first; i < last; i++) {
      bodytag[i] = (tagint) ubuf(buf[m++]).i;
      grain_com[i][0] = buf[m++];
      grain_com[i][1] = buf[m++];
      grain_com[i][2] = buf[m++];
      grain_omega[i][0] = buf[m++];
      grain_omega[i][1] = buf[m++];
      grain_omega[i][2] = buf[m++];
      grain_quat[i][0] = buf[m++];
      grain_quat[i][1] = buf[m++];
      grain_quat[i][2] = buf[m++];
      grain_quat[i][3] = buf[m++];
    }
  } else if (commflag_ls == FULL_BODY_LS) {
    for (i = first; i < last; i++) {
      bodyownLS[i] = static_cast<int> (buf[m++]);

      if (bodyownLS[i] == 0) bodyownLS[i] = -1;
      else {
        j = nlocal_bodyLS + nghost_bodyLS;
        if (j == nmax_bodyLS) grow_body_ls();
        memcpy(&bodyLS[j], &buf[m], sizeof(BodyLS));
        m += bodysizeLS;
        bodyLS[j].ilocal = i;
        bodyownLS[i] = j;
        nghost_bodyLS++;
      }
    }
  } else {
    FixRigidSmall::unpack_forward_comm(n, first, buf);
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

  for (int i = nmax_bodyLS - DELTA_BODY; i < nmax_bodyLS; i++)
    bodyLS[i].style = -1;
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

/* ----------------------------------------------------------------------
   one-time prereading of file names for LS grid
     collect unique gridfile names and min scale factors
     set distributed flag
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::read_infile()
{
  tagint id;
  int nchunk, eofflag, nlines;
  FILE *fp;
  char *eof, *start, *next, *buf;
  char line[MAXLINE] = {'\0'};
  std::string gridfile;

  // create local hash with key/value pairs
  // key = mol ID of bodies my atoms own
  // value = index into local body array

  int nlocal = atom->nlocal;

  std::unordered_map<tagint,int> hash;
  for (int i = 0; i < nlocal; i++)
    if (bodyown[i] >= 0) hash[atom->molecule[i]] = bodyown[i];

  if (comm->me == 0) {
    fp = fopen(inpfile,"r");
    if (fp == nullptr)
      error->one(FLERR, "Cannot open fix rigid/small/ls/dem infile {}: {}", inpfile, utils::getsyserror());
    while (true) {
      eof = fgets(line, MAXLINE, fp);
      if (eof == nullptr) error->one(FLERR, "Unexpected end of fix rigid/small/ls/dem infile");
      start = &line[strspn(line, " \t\n\v\f\r")];
      if (*start != '\0' && *start != '#') break;
    }
    nlines = utils::inumeric(FLERR, utils::trim(line), true, lmp);
    if (nlines == 0) fclose(fp);
  }
  MPI_Bcast(&nlines, 1, MPI_INT, 0, world);

  if (nlines == 0) return;
  else if (nlines < 0) error->all(FLERR, "Fix rigid infile has incorrect format");

  auto buffer = new char[CHUNK * MAXLINE];
  int nread = 0;
  int me = comm->me;
  int mem_style, set_size, skip, m;
  double scale;
  LSData mydata;
  while (nread < nlines) {
    nchunk = MIN(nlines - nread, CHUNK);
    eofflag = utils::read_lines_from_file(fp, nchunk, MAXLINE, buffer, me, world);
    if (eofflag) error->all(FLERR, "Unexpected end of fix rigid/small/ls/dem infile");

    buf = buffer;
    next = strchr(buf, '\n');
    *next = '\0';
    int nwords = utils::count_words(utils::trim_comment(buf));
    *next = '\n';

    if (nwords == (ATTRIBUTE_PERBODY + n_extra_attributes)) {
      read_quat = 0;
    } else if (nwords == (ATTRIBUTE_PERBODY + n_extra_attributes + 4)) {
      read_quat = 1;
    } else {
      error->all(FLERR, "Incorrect rigid body format in fix rigid/small/ls/dem file");
    }

    for (int i = 0; i < nchunk; i++) {
      next = strchr(buf,'\n');
      *next = '\0';

      try {
        ValueTokenizer values(buf);
        id = values.next_tagint();

        if (id <= 0 || id > maxmol)
          error->all(FLERR,"Invalid rigid body molecule ID {} in fix {} file", id, style);

        skip = 0;
        if (hash.find(id) == hash.end()) {
          skip = 1;
        } else {
          m = hash[id];
          skip = 0;
        }

        if (!skip) {
          mass_custom[m] = values.next_double();
          xcm_custom[m][0] = values.next_double();
          xcm_custom[m][1] = values.next_double();
          xcm_custom[m][2] = values.next_double();
        } else {
          values.skip(4);
        }

        values.skip(15);
        mem_style = values.next_int();

        if (mem_style != DISTRIBUTED && mem_style != GLOBAL)
          throw TokenizerException("invalid_rigid memory model ", std::to_string(mem_style));

        if (mem_style == GLOBAL)
          global_flag = 1;
        if (mem_style == DISTRIBUTED)
          distributed_flag = 1;

        scale = values.next_double();
        if (scale <= 0)
          error->one(FLERR, "Invalid scaling factor {}", scale);

        gridfile = values.next_string();
        if (gridfile_data.find(gridfile) == gridfile_data.end()) {
          set_size = gridfile_data.size();
          gridfile_data[gridfile] = mydata;
          gridfile_data[gridfile].style = mem_style;
          gridfile_data[gridfile].id = set_size;
          id_to_gridfile[set_size] = gridfile;
          gridfile_to_id[gridfile] = set_size;
        } else {
          if (gridfile_data[gridfile].style != mem_style)
            error->all(FLERR, "Grid file {} has two different memory styles", gridfile);
        }

        gridfile_data[gridfile].bodies.push_back(id);
        gridfile_data[gridfile].scales.push_back(scale);

        if (read_quat) {
          if (!skip) {
            quat_custom[m][0] = values.next_double();
            quat_custom[m][1] = values.next_double();
            quat_custom[m][2] = values.next_double();
            quat_custom[m][3] = values.next_double();
          } else {
            values.skip(4);
          }
        }
      } catch (TokenizerException &e) {
        error->all(FLERR, "Invalid fix rigid/small/ls/dem infile: {}", e.what());
      }
      buf = next + 1;
    }
    nread += nchunk;
  }

  if (comm->me == 0) fclose(fp);
  delete[] buffer;
}

/* ----------------------------------------------------------------------
   read per rigid body level-set grid values from user-provided file
   files gridfiles to read from stored previously by readfile() function
   first line = grid_sizex grid_sizey grid_sizez
   followed by grid_sizex * grid_sizey * grid_sizez lines of level set values at the grid points
   which = 0, read only metadata of the level-set grid
   which = 1, read the values of the level-set grid
   for context, see function in FixRigidLSDEM
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::read_gridfile(int which, std::string filename, double *grid_values)
{
  int dim = domain->dimension;
  int grid_shape_buf[dim];
  double grid_size_buf[dim + 1];
  int nchunk, eofflag;
  FILE *fp;
  char *eof, *start, *next, *buf;
  char line[MAXLINE] = {'\0'};

  int nlines = 1;
  const char* gridfile = filename.c_str();
  if (comm->me == 0) {
    fp = fopen(gridfile, "r");
    if (fp == nullptr) {
      error->one(FLERR, "Cannot open fix rigid/small/ls/dem gridfile {}: {}", gridfile, utils::getsyserror());
    } while (true) {
      eof = fgets(line, MAXLINE, fp);
      if (eof == nullptr) error->one(FLERR,"Unexpected end of fix rigid/small/ls/dem gridfile");
      start = &line[strspn(line, " \t\n\v\f\r")];
      if (*start != '\0' && *start != '#') break;
    }
    auto grid_shape = utils::split_words(line);
    if (grid_shape.size() != dim)
      error->one(FLERR, "Fix rigid/small/ls/dem gridfile {} has {} dimensions but simulation is {}D",
                          gridfile, grid_shape.size(), dim);
    for (int idim = 0; idim < dim; idim++)
      grid_shape_buf[idim] = utils::inumeric(FLERR, grid_shape[idim], false, lmp);

    eof = fgets(line, MAXLINE, fp);
    if (eof == nullptr) error->one(FLERR, "Unexpected end of fix rigid/small/ls/dem gridfile");
    grid_size_buf[0] = utils::numeric(FLERR, utils::trim(line), false, lmp);
    if (grid_size_buf[0] <= 0.0)
      error->one(FLERR, "Grid stride for rigid/small/ls/dem gridfile {} must be positive", gridfile);

    eof = fgets(line, MAXLINE, fp);
    if (eof == nullptr) error->one(FLERR, "Unexpected end of fix rigid/small/ls/dem gridfile");
    auto grid_corner = utils::split_words(line);
    if (grid_corner.size() != dim)
      error->one(FLERR, "Fix rigid/small/ls/dem gridfile {} specifies {} grid corner cooridnates but simulation is {}D",
                          gridfile, grid_corner.size(), dim);
    for (int idim = 0; idim < dim; idim++)
      grid_size_buf[idim + 1] = utils::numeric(FLERR, grid_corner[idim], false, lmp);
    if (which == 0)
      utils::logmesg(lmp, "Reading ls/dem grid data from file {}\n", gridfile);
  }
  MPI_Bcast(grid_shape_buf, dim, MPI_INT, 0, world);
  MPI_Bcast(grid_size_buf, dim + 1, MPI_DOUBLE, 0, world);

  for (int idim = 0; idim < dim; idim++)
    nlines *= grid_shape_buf[idim];

  if (nlines == 0) return;
  else if (nlines < 0) error->all(FLERR, "Fix rigid/small/ls/dem gridfile has incorrect format");

  if (which == 0) {
    gridfile_data[gridfile].stride = grid_size_buf[0];
    for (int idim = 0; idim < dim; idim++) {
      gridfile_data[gridfile].grid_min[idim] = grid_size_buf[idim + 1];
      gridfile_data[gridfile].grid_size[idim] = (int) grid_shape_buf[idim];
    }

    if (dim == 2) {
      gridfile_data[gridfile].grid_min[2] = 0.0;
      gridfile_data[gridfile].grid_size[2] = 1;
    }

  } else {
    auto buffer = new char[CHUNK * MAXLINE];
    int nread = 0;
    int me = comm->me;
    while (nread < nlines) {
      nchunk = MIN(nlines-nread, CHUNK);
      eofflag = utils::read_lines_from_file(fp, nchunk, MAXLINE, buffer, me, world);
      if (eofflag) error->all(FLERR, "Unexpected end of fix rigid/small/ls/dem gridfile");

      buf = buffer;
      next = strchr(buf, '\n');
      *next = '\0';
      int nwords = utils::count_words(utils::trim_comment(buf));
      *next = '\n';

      if (nwords != 1)
        error->all(FLERR, "LSDEM gridfile format requires one entry per line");

      for (int i = 0; i < nchunk; i++) {
        next = strchr(buf, '\n');
        *next = '\0';

        try {
          ValueTokenizer values(buf);
          grid_values[nread + i] = values.next_double();
        } catch (TokenizerException &e) {
          error->all(FLERR, "Invalid fix rigid/small/ls/dem gridfile: {}", e.what());
        }
        buf = next + 1;
      }
      nread += nchunk;
    }
    delete[] buffer;
  }
  if (comm->me == 0) fclose(fp);
}

/* ----------------------------------------------------------------------
   Using a full level set, calculate properites of grain
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::compute_grain_properties(int ibody, int *grid_size, double *grid_min, double *grid_values)
{
  int dimension = domain->dimension;
  double com_temp[3];

  bodyLS[ibody].grid_vol = compute_grid_properties(grid_size, bodyLS[ibody].grid_stride, grid_values, com_temp, itensor_custom[ibody], dimension);

  if (bodyLS[ibody].grid_vol < 0)
    error->all(FLERR, "Non-inertial reference frame for level set in {}", id_to_gridfile[bodyLS[ibody].file_id]);

  // Check CoM misalignment, would apply forces/rotations at incorrect positions
  double sum[3];
  MathExtra::add3(grid_min, com_temp, sum);
  if (MathExtra::len3(sum) > (0.5 * bodyLS[ibody].grid_stride))
    error->all(FLERR, "Centre of mass computed from LS grid does not agree with provided value, grid min at {} {} {} and CoM computed at {} {} {}",
    grid_min[0], grid_min[1], grid_min[2], com_temp[0], com_temp[1], com_temp[2]);

  // Surface area calculation
  //   default epsilon (diff between inner and outer) = 2x grid stride.

  double area = compute_surface_area(dimension, grid_size, bodyLS[ibody].grid_stride, grid_values);
  if (!((area > 0.0) && std::isfinite(area)))
    error->all(FLERR, "Invalid surface area calculated {}", area);
  bodyLS[ibody].node_area = area;

  // Scale relevant quantities by grain size

  double scale = bodyLS[ibody].grid_scale;
  double scale2 = scale * scale;
  double scale3 = scale * scale2;
  double density = mass_custom[ibody] / bodyLS[ibody].grid_vol;
  bodyLS[ibody].grid_stride *= scale;
  if (dimension == 3) {
    bodyLS[ibody].node_area *= scale2;
    bodyLS[ibody].grid_vol *= scale3;
    for (int a = 0; a < 6; a++)
      itensor_custom[ibody][a] *= density * scale2 * scale3;
  } else {
    bodyLS[ibody].node_area *= scale;
    bodyLS[ibody].grid_vol *= scale2;
    for (int a = 0; a < 6; a++)
      itensor_custom[ibody][a] *= density * scale2 * scale2;
  }
}

/* ----------------------------------------------------------------------
   Find the value of node (atom) i in j's LS grid
   see FixRigidLSDEM for context and explanation
------------------------------------------------------------------------- */

double FixRigidSmallLSDEM::get_ls_value(int i, int j, int currentbin, double *normal, double *x_local)
{
  if (storage_mode == WATERSHED)
    return get_ls_value_watershed(i, j, currentbin, normal, x_local);
  else
    return get_ls_value_array(i, j, normal);
}

/* ---------------------------------------------------------------------- */

int FixRigidSmallLSDEM::get_bin(int i, int j, double *x_local)
{
  double **x = atom->x;
  double **grain_com = atom->xcom;
  double **grain_quat = atom->quat;
  int jbody = atom2body[j];

  // Location of the node (atom) of i relative to the centre of mass (CoM) of j
  double delx = x[i][0] - grain_com[j][0];
  double dely = x[i][1] - grain_com[j][1];
  double delz = x[i][2] - grain_com[j][2];

  // Account for PBCs
  domain->minimum_image(FLERR, delx, dely, delz);

  // Apply quaternion rotation to move into local reference frame of grain j grid.
  // Here, grain_quat is local->global. Therefore, grain_quat_conj is global -> local.
  double dx[3] = {delx, dely, delz};
  double grain_quat_conj[4];

  MathExtra::qconjugate(grain_quat[j], grain_quat_conj);
  MathExtra::quatrotvec(grain_quat_conj, dx, x_local);

  int ngrid[3];
  if (storage_mode == ARRAY && bodyLS[jbody].style == DISTRIBUTED) {
    // Translate local coordinates relative to lower corner of the node's grid
    double **local_grid_min = atom->darray[index_grid_min];
    MathExtra::sub3(x_local, local_grid_min[j], x_local);
    ngrid[0] = subgrid_size[0];
    ngrid[1] = subgrid_size[1];
    ngrid[2] = subgrid_size[2];
  } else {
    // Translate local coordinates relative to lower corner of the grain's grid
    int gi = bodyLS[jbody].grid_index;
    double grid_min_scaled[3];
    MathExtra::scale3(bodyLS[jbody].grid_scale, global_grids_min[gi], grid_min_scaled);
    MathExtra::sub3(x_local, grid_min_scaled, x_local);
    ngrid[0] = global_grids_size[gi][0];
    ngrid[1] = global_grids_size[gi][1];
    ngrid[2] = global_grids_size[gi][2];
  }

  // Normalise the coordinates to be in units of the number of grid cells.
  MathExtra::scale3(1.0 / bodyLS[jbody].grid_stride, x_local);

  int currentbin = int(x_local[0]) + int(x_local[1]) * ngrid[0] + int(x_local[2]) * ngrid[0] * ngrid[1];

  return currentbin;
}

/* ---------------------------------------------------------------------- */

int FixRigidSmallLSDEM::check_watershed_bin(int currentbin, int j)
{
  int jbody = atom2body[j];
  if (bodyLS[jbody].style == DISTRIBUTED ) {
    if (dist_ws_tables[j].find(currentbin) != dist_ws_tables[j].end())
      return 1;
  } else {
    int gj = bodyLS[jbody].grid_index;
    int ntj = node_index[j];
    if (global_ws_tables[gj][ntj].find(currentbin) != global_ws_tables[gj][ntj].end())
      return 1;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

double FixRigidSmallLSDEM::get_ls_value_watershed(int i, int j, int currentbin, double *normal, double *x_local)
{
  int ngrid[3], ix[3];
  int jbody = atom2body[j];

  int id = bodyLS[jbody].file_id;

  ngrid[0] = gridfile_data[id_to_gridfile[id]].grid_size[0];
  ngrid[1] = gridfile_data[id_to_gridfile[id]].grid_size[1];
  ngrid[2] = gridfile_data[id_to_gridfile[id]].grid_size[2];
  ix[0] = int(x_local[0]);
  ix[1] = int(x_local[1]);
  ix[2] = int(x_local[2]);

  int dim = domain->dimension;
  std::unordered_map<int, double> *mytable;
  std::unordered_map<int, double> *mybuffer;

  if (bodyLS[jbody].style == DISTRIBUTED) {
    mytable = &dist_ws_tables[j];
    mybuffer = &dist_ws_buffers[j];
  } else {
    int gj = bodyLS[jbody].grid_index;
    int ntj = node_index[j];
    mytable = &global_ws_tables[gj][ntj];
    mybuffer = &global_ws_buffers[gj][ntj];
  }

  double dist = interpolate_LS_watershed(dim, currentbin, mytable, mybuffer, ngrid, x_local, ix, normal, bodyLS[jbody].grid_stride);

  // Grain-stored grid values are shared and un-scaled, so apply scaling
  if (bodyLS[jbody].style == GLOBAL) dist *= bodyLS[jbody].grid_scale;

  // Rotate normal back to global coordinates
  double **grain_quat = atom->quat;
  MathExtra::quatrotvec(grain_quat[j], normal, normal);

  return dist;
}

/* ---------------------------------------------------------------------- */

double FixRigidSmallLSDEM::get_ls_value_array(int i, int j, double *normal)
{
  int ngrid[3], ix[3];
  double x_local[3];
  int jbody = atom2body[j];
  int currentbin = get_bin(i, j, x_local);

if (bodyLS[jbody].style == DISTRIBUTED) {
    ngrid[0] = subgrid_size[0];
    ngrid[1] = subgrid_size[1];
    ngrid[2] = subgrid_size[2];
  } else {
    int gj = bodyLS[jbody].grid_index;
    ngrid[0] = global_grids_size[gj][0];
    ngrid[1] = global_grids_size[gj][1];
    ngrid[2] = global_grids_size[gj][2];
  }

  ix[0] = int(x_local[0]);
  ix[1] = int(x_local[1]);
  ix[2] = int(x_local[2]);

  int ncol, nrow, nslice;
  double *mygrid;
  if (bodyLS[jbody].style == DISTRIBUTED) {
    mygrid = atom->darray[index_grid_values][j];
  } else {
    mygrid = global_grids[bodyLS[jbody].grid_index];
  }

  int dim = domain->dimension;
  double jstride = bodyLS[jbody].grid_stride;
  double dist = interpolate_LS_array(dim, currentbin, mygrid, ngrid, x_local, ix, normal, jstride);

  if (bodyLS[jbody].style == GLOBAL) dist *= bodyLS[jbody].grid_scale;
  double **grain_quat = atom->quat;
  MathExtra::quatrotvec(grain_quat[j], normal, normal);

  return dist;
}
