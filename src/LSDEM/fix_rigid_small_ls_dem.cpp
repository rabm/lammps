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
enum {REGULAR, PREFORCE};

static constexpr double EPSILON_VOL_DIFF = 1.0e-6; // 0.0001%
static constexpr int MAX_ITERATIONS = 100; // For surface area integration
static constexpr int RECOMMENDED_MAX_NGRID = 1000; // For local node grid, 10x10x10

/* ---------------------------------------------------------------------- */

FixRigidSmallLSDEM::FixRigidSmallLSDEM(LAMMPS *lmp, int narg, char **arg) :
  FixRigidSmall(lmp, narg, arg), bodyLS(nullptr), bodyownLS(nullptr), global_grids(nullptr), id_fix(nullptr), id_fix2(nullptr)
{
  maxcut = -1;
  stored_flag = 0;
  distributed_flag = 0;

  n_extra_attributes = 3;

  nmax_bodyLS = 0;
  while (nmax_bodyLS < nlocal_body) nmax_bodyLS += DELTA_BODY;
  bodyLS = (BodyLS *) memory->smalloc(nmax_bodyLS * sizeof(BodyLS), "rigid/small/ls/dem:bodyls");
  memory->grow(bodyownLS, atom->nmax, "rigid/small/ls/dem:bodyownLS");
  atom->add_callback(Atom::GROW);

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

  comm_forward += 1 + bodysizeLS;
  comm_flag2 = REGULAR;

  if (langflag)
    error->all(FLERR, "Langevin thermostat not supported with fix rigid/small/ls/dem");
}

/* ---------------------------------------------------------------------- */

FixRigidSmallLSDEM::~FixRigidSmallLSDEM()
{
  // delete extra property/atom fixes

  if (id_fix && modify->nfix) modify->delete_fix(id_fix);
  delete[] id_fix;
  if (id_fix2 && modify->nfix) modify->delete_fix(id_fix2);
  delete[] id_fix2;

  // unregister callbacks to this fix from Atom class

  if (modify->get_fix_by_id(id)) atom->delete_callback(id,Atom::GROW);

  memory->sfree(bodyLS);

  // delete global memory data

  memory->destroy(global_grids);
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
  // Store positional information of grain on all atoms
  id_fix = utils::strdup(id + std::string("_FIX_PROP_ATOM"));
  modify->add_fix(fmt::format(
    "{} all property/atom d2_ls_dem_n 3 d2_ls_dem_fs 3 i_ls_dem_touch_id d_ls_dem_fn1 d_ls_dem_fs1 ghost yes writedata no",
     id_fix));
  int tmp1, tmp2;
  index_ls_dem_touch_id = atom->find_custom("ls_dem_touch_id", tmp1, tmp2);
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

  // if distributed, need to calculate size for fix property/atom local grids
  //   needs to be defined in init to set comm limits

  std::map<std::string, double> gridfile_map;
  std::string gridfile;

  if (inpfile) {
    preread_gridfile_names(gridfile_map);
  } else {
    Molecule *onemol;
    int *grid_index = atom->grid_index;
    for (int i = 0; i < atom->nlocal; i++) {
      onemol = atom->molecules[grid_index[i]];
      if (!onemol->grid_file.empty()) {
        gridfile = onemol->grid_file;
        if (gridfile_map.find(gridfile) == gridfile_map.end())
          gridfile_map[gridfile] = onemol->grid_scale;
        else
          gridfile_map[gridfile] = MAX(gridfile_map[gridfile], onemol->grid_scale);

        if (onemol->grid_style)
          distributed_flag = 1;
      }
    }
  }

  double stride, scale;
  double min_stride = DBL_MAX;
  for (const auto& pair : gridfile_map) {
    gridfile = pair.first;
    scale = pair.second;
    stride = preread_gridfile(gridfile);
    min_stride = MIN(min_stride, stride * scale);
  }

  // All local grids sized on finest grid (fix property/atom requires fixed-size containers)
  rcell = maxcut / min_stride + 2; // +1 for interpolation +1 for safety

  if (distributed_flag) {
    // todo try remove +1 and cast to int
    for (int a = 0; a < 3; a++) subgrid_size[a] = 2 * rcell + 1;
    if (domain->dimension == 2) subgrid_size[2] = 1;

    id_fix2 = utils::strdup(id + std::string("_FIX_PROP_ATOM_2"));
    int ntotal = subgrid_size[0] * subgrid_size[1] * subgrid_size[2];
    if (ntotal > RECOMMENDED_MAX_NGRID)
      error->warning(FLERR, "A large per-atom subgrid of size {}x{}x{} is being allocated for distributed level sets with a cutoff of {} and a min stride of {}", subgrid_size[0], subgrid_size[1], subgrid_size[2], maxcut, min_stride);
    modify->add_fix(fmt::format("{} all property/atom d2_grid_values {} d2_grid_min {} writedata no ghost yes", id_fix2, ntotal, 3));

    int tmp1, tmp2;
    index_grid_values = atom->find_custom("grid_values", tmp1, tmp2);
    index_grid_min = atom->find_custom("grid_min", tmp1, tmp2);
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::setup_pre_neighbor()
{
  FixRigidSmall::setup_pre_neighbor();

  // compared to rigid/ls/dem, need to wait until atom2body made

  int ibody, i, a;
  int dimension = domain->dimension;

  int index_global = 0;
  int *touch_id = atom->ivector[index_ls_dem_touch_id];
  int *grid_index = atom->grid_index;
  if (!stored_flag) {
    stored_flag = 1;

    for (i = 0; i < atom->nlocal; i++)
      touch_id[i] = -1; // set to zero for preexisting atoms (rest set in set_array)

    int *ntotal_global;
    char **gridfiles;
    memory->create(ntotal_global, nbody, "rigid/ls/dem:ntotal_global");
    memory->create(gridfiles, nbody, MAXLINE, "rigid/ls/dem:gridfiles");

    if (inpfile) {
      read_gridfile_names(gridfiles);
    } else {
      Molecule *onemol;
      for (i = 0; i < atom->nlocal; i++) {
        if (bodyownLS[i] == -1) continue;
        onemol = atom->molecules[grid_index[i]];
        if (onemol->grid_file.empty())
          error->all(FLERR, "Molecule {} is missing a level set grid file", onemol->id);
        ibody = atom2body[i];
        strcpy(gridfiles[ibody], onemol->grid_file.c_str());
        bodyLS[ibody].grid_style = onemol->grid_style;
        bodyLS[ibody].grid_scale = onemol->grid_scale;
        body[ibody].mass = onemol->masstotal;
      }
    }

    // Read grid dimensions for all bodies
    std::map <std::string, std::set<int>> file_map;
    std::string filename;
    int grid_size_flat, max_grid_size_flat(0);
    for (ibody = 0; ibody < nbody; ibody++) {
      filename.assign(gridfiles[ibody]); // Retrieve file name
      read_gridfile(ibody, 0, filename, nullptr); // Get only grid sizes (which 0)
      file_map[filename].insert(ibody);

      // Calculate and save grid properties
      grid_size_flat = bodyLS[ibody].grid_size[0] * bodyLS[ibody].grid_size[1] * bodyLS[ibody].grid_size[2];
      max_grid_size_flat = MAX(max_grid_size_flat, grid_size_flat);

      // Store global info
      bodyLS[ibody].grid_index = -1;
      if (bodyLS[ibody].grid_style == GLOBAL) {
        // Copy from prior entry if it exists
        if (file_map.find(filename) != file_map.end())
          for (const auto& jbody : file_map[filename])
            if (bodyLS[jbody].grid_index != -1)
              bodyLS[ibody].grid_index = bodyLS[jbody].grid_index;

        // If no global instances, add new index
        if (bodyLS[ibody].grid_index == -1) {
          bodyLS[ibody].grid_index = index_global;
          ntotal_global[index_global] = grid_size_flat;
          index_global += 1;
        }
      }
    }

    // ------------------------------ //
    // Allocate memory for level sets //
    // ------------------------------ //

    for (ibody = 0; ibody < nbody; ibody++)
      bodyLS[ibody].grid_nnodes = 0;
    for (i = 0; i < atom->nlocal; i++) {
      ibody = atom2body[i];
      bodyLS[ibody].grid_nnodes += 1;
    }

    if (index_global) {
      memory->create_ragged(global_grids, index_global, ntotal_global, "rigid/small/ls/dem:global_grids");
    }

    // ------------------------------ //
    // Read and store level sets      //
    // ------------------------------ //

    double *temp_grid_values;
    memory->create(temp_grid_values, max_grid_size_flat, "rigid/small/ls/dem:temp_grid_values");

    double **grid_values, **grid_min_local;
    if (distributed_flag) {
      grid_values = atom->darray[index_grid_values];
      grid_min_local = atom->darray[index_grid_min];
    }

    double **x = atom->x;

    int need_distributed, need_global, need_padding, nx, ny, nz, ix_node, iy_node, iz_node;
    int ix_global, iy_global, iz_global, index_global, index_local, index_grid_min_local[3];
    double temp[3], com_temp[3], inertia_temp[3][3], evectors[3][3];
    double delx, dely, delz, area, density, scale, scale2, scale3;
    for (const auto& pair : file_map) { // Loop over all <filename, [bodyIDs]>
      filename = pair.first;
      read_gridfile(-1, 1, filename, temp_grid_values);

      // Compute grain properties per unique grid
      for (ibody = 0; ibody < nbody; ibody++) {
        if (pair.second.find(ibody) == pair.second.end())
          continue;

        // Compute properties from the level-set grid
        bodyLS[ibody].grid_vol = compute_grid_properties(bodyLS[ibody].grid_size, bodyLS[ibody].grid_stride, temp_grid_values, com_temp, inertia_temp, dimension);
        if (bodyLS[ibody].grid_vol < 0)
          error->all(FLERR, "Non-inertial reference frame detected for level set in {}, integration of rotational motion will be wrong", filename);

        // Comparing if CoM in level-set grid is indeed aligned with CoM
        // Misalignment would cause forces/rotations to be applied to the wrong point in space
        MathExtra::add3(bodyLS[ibody].grid_min, com_temp, temp);
        if (MathExtra::len3(temp) > (0.5 * bodyLS[ibody].grid_stride))
          error->all(FLERR, "Centre of mass computed from the LS grid does not agree with that provided in the input grid file! Grid min given at {} {} {} and CoM computed at {} {} {}.",
            bodyLS[ibody].grid_min[0], bodyLS[ibody].grid_min[1], bodyLS[ibody].grid_min[2], com_temp[0], com_temp[1], com_temp[2]);

        // Overwrite inertia, could modify logic (compare or warn) if desired

        // Calculate eigen system of inertia tensor
        int ierror = MathEigen::jacobi3(inertia_temp, body[ibody].inertia, evectors, 1);
        if (ierror) error->all(FLERR, "Insufficient Jacobi rotations for LS grid");

        // Set grain orientation based on eigenvectors of inertia tensor
        for (a = 0; a < 3; a++) {
          body[ibody].ex_space[a] = evectors[a][0];
          body[ibody].ey_space[a] = evectors[a][1];
          body[ibody].ez_space[a] = evectors[a][2];
        }

        // Surface area calculation with default epsilon (diff between inner and outer) of two times grid stride.
        area = compute_surface_area(dimension, bodyLS[ibody].grid_size, bodyLS[ibody].grid_stride, temp_grid_values);
        // Test for physical realism
        if (!((area > 0.0) && std::isfinite(area)))
          error->all(FLERR, "Surface area calculation returns nonsense, giving {}", area);
        bodyLS[ibody].node_area = area;

        // Normalise by number of nodes
        bodyLS[ibody].node_area /= bodyLS[ibody].grid_nnodes;

        // Scale all relevant quantities by given scaling of grain size
        scale = bodyLS[ibody].grid_scale;
        scale2 = scale * scale;
        scale3 = scale * scale2;
        density = body[ibody].mass / bodyLS[ibody].grid_vol;
        bodyLS[ibody].grid_stride *= scale;
        MathExtra::scale3(scale, bodyLS[ibody].grid_min);
        bodyLS[ibody].node_area *= scale2;
        bodyLS[ibody].grid_vol *= scale3;
        MathExtra::scale3(density * scale2*scale3, body[ibody].inertia);
      }

      // Start handling memory approach
      need_distributed = 0; // Save relevant grid snippet at node, regardless of duplicity
      need_global = 0;  // Save the entire grid as a shared memory stucture between grains with the same grid
      for (const auto& jbody : file_map[filename]) {
        if (bodyLS[jbody].grid_style == DISTRIBUTED) {
          need_distributed = 1;
        } else if (bodyLS[jbody].grid_style == GLOBAL) {
          need_global = 1;
          index_global = bodyLS[jbody].grid_index;
        }
      }

      if (need_global) {
        for (int n = 0; n < ntotal_global[index_global]; n++) {
          // Unscaled grid values of grains stored globally to avoid duplicating memory
          global_grids[index_global][n] = temp_grid_values[n];
        }
      }

      if (need_distributed) {
        for (i = 0; i < atom->nlocal; i++) {
          ibody = atom2body[i];
          Body *b = &body[ibody];

          need_padding = 0;
          if (pair.second.find(ibody) == pair.second.end())
            continue; // Ideally would have list of all atoms in a rigid body... not sure if exists...

          nx = bodyLS[ibody].grid_size[0];
          ny = bodyLS[ibody].grid_size[1];
          nz = bodyLS[ibody].grid_size[2];

          // Location of atom/node relative to CoM
          delx = x[i][0] - b->xcm[0];
          dely = x[i][1] - b->xcm[1];
          delz = x[i][2] - b->xcm[2];

          // Account for PBCs
          domain->minimum_image(FLERR, delx, dely, delz);

          // Location of atom/node relative to entire grain grid minimum.
          delx -= bodyLS[ibody].grid_min[0];
          dely -= bodyLS[ibody].grid_min[1];
          delz -= bodyLS[ibody].grid_min[2];

          // Index of atom/node in entire grain grid.
          double stride = bodyLS[ibody].grid_stride;
          ix_node = int(delx / stride);
          iy_node = int(dely / stride);
          iz_node = int(delz / stride);

          // Index of local grid minimum in entire grain grid. If any goes below zero, error below catches it.
          index_grid_min_local[0] = ix_node - rcell;
          index_grid_min_local[1] = iy_node - rcell;
          index_grid_min_local[2] = (dimension == 3) ? iz_node - rcell : 0;

          // Location of local grid minimum relative to CoM
          grid_min_local[i][0] = index_grid_min_local[0] * stride + bodyLS[ibody].grid_min[0];
          grid_min_local[i][1] = index_grid_min_local[1] * stride + bodyLS[ibody].grid_min[1];
          grid_min_local[i][2] = index_grid_min_local[2] * stride + bodyLS[ibody].grid_min[2];

          for (int iz_local = 0; iz_local < subgrid_size[2]; iz_local++) {
            for (int iy_local = 0; iy_local < subgrid_size[1]; iy_local++) {
              for (int ix_local = 0; ix_local < subgrid_size[0]; ix_local++) {
                index_local = ix_local + iy_local * subgrid_size[0] + iz_local * subgrid_size[0] * subgrid_size[1];

                // Shift local cell to global cell
                ix_global = ix_local + index_grid_min_local[0];
                iy_global = iy_local + index_grid_min_local[1];
                iz_global = iz_local + index_grid_min_local[2];

                // Explicit bounds check per dimension (safer and clearer)
                if (ix_global < 0 || ix_global >= nx ||
                    iy_global < 0 || iy_global >= ny ||
                    iz_global < 0 || iz_global >= nz) {
                  need_padding = 1;
                  grid_values[i][index_local] = BIG;
                } else {
                  // True (scaled) level-set stored for DISTRIBUTED approach where unique local grid is saved on node
                  index_global = ix_global + iy_global * nx + iz_global * nx * ny;
                  grid_values[i][index_local] = temp_grid_values[index_global] * bodyLS[ibody].grid_scale;
                }
              }
            }
          }

          if (need_padding)
            error->warning(FLERR, "Level set of body {} does not include a large enough buffer for the distributed grid cutoff on atom {}. Local grid padded with BIG values", ibody, i);
        }
      }
    }

    memory->destroy(gridfiles);
    memory->destroy(temp_grid_values);
    memory->destroy(ntotal_global);
  }

  if (distributed_flag) {
    int tmp1, tmp2;
    index_grid_values = atom->find_custom("grid_values", tmp1, tmp2);
    index_grid_min = atom->find_custom("grid_min", tmp1, tmp2);
  }

  nghost_bodyLS = 0;
}

/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::setup_pre_force(int vflag)
{
  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::pre_force(int vflag)
{
  comm_flag2 = PREFORCE;
  comm->forward_comm(this, 1);
  comm_flag2 = REGULAR;
}


/* ---------------------------------------------------------------------- */

void FixRigidSmallLSDEM::initial_integrate(int vflag)
{
  FixRigidSmall::initial_integrate(vflag);

  double **grain_com = atom->xcom;
  double **grain_quat = atom->quat;
  double **grain_omega = atom->omega;

  int ibody;
  for (int i = 0; i < atom->nlocal; i++) {
    ibody = atom2body[i];
    Body *b = &body[ibody];

    grain_com[i][0] = b->xcm[0];
    grain_com[i][1] = b->xcm[1];
    grain_com[i][2] = b->xcm[2];

    grain_quat[i][0] = b->quat[0];
    grain_quat[i][1] = b->quat[1];
    grain_quat[i][2] = b->quat[2];
    grain_quat[i][3] = b->quat[3];

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

  double **x = atom->x;
  double **f = atom->f;
  double **torque = atom->torque;
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

    tcm = b->torque;
    tcm[0] += torque[i][0];
    tcm[1] += torque[i][1];
    tcm[2] += torque[i][2];
  }

  // reverse communicate fcm, torque of all bodies

  commflag = FORCE_TORQUE;
  comm->reverse_comm(this, 6);

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
  atom->ivector[index_ls_dem_touch_id][i] = -1;
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
  //      need to trigger reading level-set grid file...

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
    if (tag[i]-tagprev == onemols[imol]->comatom) bodyown[i] = nlocal_body;

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
      if (nlocal_body == nmax_body) grow_body();
      Body *b = &body[nlocal_body];
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
      nlocal_body++;
    }
  }
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
  int m, i, j;
  if (comm_flag2 != PREFORCE) {
    m = FixRigidSmall::pack_forward_comm(n, list, buf, 0, nullptr);

    if (commflag == FULL_BODY) {
      // Communicate all of bodyownLS first so it can be used to calculate how much parent class communictes
      for (i = 0; i < n; i++) {
        j = list[i];
        if (bodyownLS[j] < 0) buf[m++] = 0;
        else buf[m++] = 1;
      }
      for (i = 0; i < n; i++) {
        j = list[i];
        if (bodyownLS[j] >= 0) {
          memcpy(&buf[m], &bodyLS[bodyownLS[j]], sizeof(BodyLS));
          m += bodysizeLS;
        }
      }
    }
  } else {
    m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(atom2body[j]).d;
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
  int m, i, j, last;
  last = first + n;
  m = 0;

  if (comm_flag2 != PREFORCE) {
    FixRigidSmall::unpack_forward_comm(n, first, buf);

    if (commflag == FULL_BODY) {
      // calculate amount of data sent by parent
      for (i = first; i < last; i++) {
        bodyownLS[i] = static_cast<int> (buf[m++]);
        if (bodyownLS[i] != 0) m += bodysize;
      }

      for (i = first; i < last; i++) {
        if (bodyownLS[i] == 0) {
          bodyownLS[i] = -1;
        } else {
          j = nlocal_bodyLS + nghost_bodyLS;
          if (j == nmax_bodyLS) grow_body_ls();
          memcpy(&bodyLS[j], &buf[m], sizeof(BodyLS));
          m += bodysizeLS;
          bodyLS[j].ilocal = i;
          bodyownLS[i] = j;
          nghost_bodyLS++;
        }
      }
    }
  } else {
    for (i = first; i < last; i++)
      atom2body[i] = ubuf(buf[m++]).i;
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

/* ----------------------------------------------------------------------
   one-time prereading of file names for LS grid
     collect unique gridfile names and min scale factors
     set distributed flag
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::preread_gridfile_names(std::map<std::string, double> &gridfile_map)
{
  tagint id;
  int nchunk, eofflag, nlines;
  FILE *fp;
  char *eof, *start, *next, *buf;
  char line[MAXLINE] = {'\0'};
  std::string gridfile;

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
  int style;
  double scale;
  while (nread < nlines) {
    nchunk = MIN(nlines - nread, CHUNK);
    eofflag = utils::read_lines_from_file(fp, nchunk, MAXLINE, buffer, me, world);
    if (eofflag) error->all(FLERR, "Unexpected end of fix rigid/small/ls/dem infile");

    buf = buffer;
    next = strchr(buf, '\n');
    *next = '\0';
    int nwords = utils::count_words(utils::trim_comment(buf));
    *next = '\n';

    if (nwords != (ATTRIBUTE_PERBODY + n_extra_attributes))
      error->all(FLERR, "Incorrect rigid body format in fix rigid/small/ls/dem file");

    for (int i = 0; i < nchunk; i++) {
      next = strchr(buf,'\n');
      *next = '\0';

      try {
        ValueTokenizer values(buf);
        id = values.next_tagint();
        values.skip(19);
        style = values.next_int();
        if (style == 1)
          distributed_flag = 1;

        scale = values.next_double();
        gridfile = values.next_string();
        if (gridfile_map.find(gridfile) == gridfile_map.end())
          gridfile_map[gridfile] = scale;
        else
          gridfile_map[gridfile] = MIN(gridfile_map[gridfile], scale);
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
   preread per rigid body level-set grid values from user-provided file
     grab stride to calculate maximum distributed local grid size
------------------------------------------------------------------------- */

double FixRigidSmallLSDEM::preread_gridfile(std::string filename)
{
  int dim = domain->dimension;
  double grid_size_buf[dim + 1];
  FILE *fp;
  char *eof, *start, *buf;
  char line[MAXLINE] = {'\0'};

  int nlines = 1;
  const char* gridfile = filename.c_str();
  if (comm->me == 0) {
    fp = fopen(gridfile, "r");
    if (fp == nullptr)
      error->one(FLERR, "Cannot open fix rigid/small/ls/dem gridfile {}: {}", gridfile, utils::getsyserror());
    while (true) {
      eof = fgets(line, MAXLINE, fp);
      if (eof == nullptr) error->one(FLERR,"Unexpected end of fix rigid/small/ls/dem gridfile");
      start = &line[strspn(line, " \t\n\v\f\r")];
      if (*start != '\0' && *start != '#') break;
    }

    eof = fgets(line, MAXLINE, fp);
    if (eof == nullptr) error->one(FLERR, "Unexpected end of fix rigid/small/ls/dem gridfile");
    grid_size_buf[0] = utils::numeric(FLERR, utils::trim(line), false, lmp);
    if (grid_size_buf[0] <= 0.0)
      error->one(FLERR, "Grid stride for rigid/small/ls/dem gridfile {} must be positive", gridfile);
  }
  MPI_Bcast(grid_size_buf, dim + 1, MPI_DOUBLE, 0, world);

  double grid_stride = grid_size_buf[0];
  if (comm->me == 0) fclose(fp);
  return grid_stride;
}

/* ----------------------------------------------------------------------
   one-time reading of file names for LS grid
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::read_gridfile_names(char **gridfiles)
{
  tagint id;
  int nchunk, eofflag, nlines;
  FILE *fp;
  char *eof, *start, *next, *buf;
  char line[MAXLINE] = {'\0'};

  int nlocal = atom->nlocal;

  std::unordered_map<tagint,int> hash;
  for (int i = 0; i < nlocal; i++)
    if (bodyown[i] >= 0)
      hash[atom->molecule[i]] = bodyown[i];

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
  while (nread < nlines) {
    nchunk = MIN(nlines - nread, CHUNK);
    eofflag = utils::read_lines_from_file(fp, nchunk, MAXLINE, buffer, me, world);
    if (eofflag) error->all(FLERR, "Unexpected end of fix rigid/small/ls/dem infile");

    buf = buffer;
    next = strchr(buf, '\n');
    *next = '\0';
    int nwords = utils::count_words(utils::trim_comment(buf));
    *next = '\n';

    if (nwords != (ATTRIBUTE_PERBODY + n_extra_attributes))
      error->all(FLERR, "Incorrect rigid body format in fix rigid/small/ls/dem file");

    for (int i = 0; i < nchunk; i++) {
      next = strchr(buf,'\n');
      *next = '\0';

      try {
        ValueTokenizer values(buf);
        id = values.next_tagint();

        if (id <= 0 || id > maxmol)
          error->all(FLERR,"Invalid rigid body molecude ID {} in fix {} file", id, style);

        if (hash.find(id) == hash.end()) {
          buf = next + 1;
          continue;
        }
        int m = hash[id];

        values.skip(19);
        bodyLS[m].grid_style = values.next_int();
        if (bodyLS[m].grid_style != 0 && bodyLS[m].grid_style != 1)
          throw TokenizerException("invalid_rigid memory model ", std::to_string(bodyLS[m].grid_style));

        bodyLS[m].grid_scale = values.next_double();
        strcpy(gridfiles[m], values.next_string().data());
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
   which = 0, read only the size of the level-set grid
   which = 1, read the values of the level-set grid
   for context, see function in FixRigidLSDEM
------------------------------------------------------------------------- */

void FixRigidSmallLSDEM::read_gridfile(int ibody, int which, std::string filename, double *grid_values)
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
    if (fp == nullptr)
      error->one(FLERR, "Cannot open fix rigid/small/ls/dem gridfile {}: {}", gridfile, utils::getsyserror());
    while (true) {
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
      utils::logmesg(lmp, "Reading ls/dem grid data for body {} from file {}\n", ibody, gridfile);
  }
  MPI_Bcast(grid_shape_buf, dim, MPI_INT, 0, world);
  MPI_Bcast(grid_size_buf, dim + 1, MPI_DOUBLE, 0, world);

  for (int idim = 0; idim < dim; idim++)
    nlines *= grid_shape_buf[idim];

  if (nlines == 0) return;
  else if (nlines < 0) error->all(FLERR, "Fix rigid/small/ls/dem gridfile has incorrect format");

  if (which == 0) {
    bodyLS[ibody].grid_stride = grid_size_buf[0];
    for (int idim = 0; idim < dim; idim++) {
      bodyLS[ibody].grid_min[idim] = grid_size_buf[idim + 1];
      bodyLS[ibody].grid_size[idim] = (int) grid_shape_buf[idim];
    }

    if (dim == 2) {
      bodyLS[ibody].grid_min[2] = 0.0;
      bodyLS[ibody].grid_size[2] = 1;
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
   Find the value of node (atom) i in j's LS grid
   see FixRigidLSDEM for context and explanation
------------------------------------------------------------------------- */

double FixRigidSmallLSDEM::get_ls_value(int i, int j, double *normal)
{
  double **x = atom->x;
  double **grain_com = atom->xcom;
  double **grain_quat = atom->quat;

  int jbody = atom2body[j];
  double jstride = bodyLS[jbody].grid_stride;
  double strideinv = 1.0 / jstride;

  double delx = x[i][0] - grain_com[j][0];
  double dely = x[i][1] - grain_com[j][1];
  double delz = x[i][2] - grain_com[j][2];
  domain->minimum_image(FLERR, delx, dely, delz);

  double x_local[3];
  double dx[3] = {delx, dely, delz};
  double grain_quat_conj[4];
  MathExtra::qconjugate(grain_quat[j], grain_quat_conj);
  MathExtra::quatrotvec(grain_quat_conj, dx, x_local);

  int ncol, nrow, nslice;
  double *mygrid;
  if (bodyLS[jbody].grid_style == DISTRIBUTED) {
    mygrid = atom->darray[index_grid_values][j];
    double **local_grid_min = atom->darray[index_grid_min];
    x_local[0] -= local_grid_min[j][0];
    x_local[1] -= local_grid_min[j][1];
    x_local[2] -= local_grid_min[j][2];

    ncol = subgrid_size[0];
    nrow = subgrid_size[1];
    nslice = subgrid_size[2];
  } else {
    mygrid = global_grids[bodyLS[jbody].grid_index];
    x_local[0] -= bodyLS[jbody].grid_min[0];
    x_local[1] -= bodyLS[jbody].grid_min[1];
    x_local[2] -= bodyLS[jbody].grid_min[2];

    ncol = bodyLS[jbody].grid_size[0];
    nrow = bodyLS[jbody].grid_size[1];
    nslice = bodyLS[jbody].grid_size[2];
  }

  double x_red = x_local[0] * strideinv;
  double y_red = x_local[1] * strideinv;
  double z_red = x_local[2] * strideinv;

  int dim = domain->dimension;
  double dist = interpolate_LS(dim, mygrid, ncol, nrow, nslice, x_red, y_red, z_red, normal, jstride);

  if (bodyLS[jbody].grid_style == GLOBAL) dist *= bodyLS[jbody].grid_scale;
  MathExtra::quatrotvec(grain_quat[j], normal, normal);

  return dist;
}
