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

#include "fix_rigid_ls_dem.h"

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
#include "pair.h"
#include "pair_ls_dem.h"
#include "rigid_const.h"
#include "tokenizer.h"

#include <cmath>
#include <cfloat> // DBL_MAX
#include <cstring>
#include <map>
#include <set>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
using namespace RigidConst;
using namespace LSDEMExtra;

enum {GLOBAL, DISTRIBUTED};

static constexpr double EPSILON_VOL_DIFF = 1.0e-6; // 0.0001%
static constexpr double EPSILON_INERTIA = 1.0e-3; // 0.1%
static constexpr int MAX_ITERATIONS = 100; // For surface area integration
static constexpr int RECOMMENDED_MAX_NGRID = 1000; // For local node grid, 10x10x10

/* ---------------------------------------------------------------------- */

FixRigidLSDEM::FixRigidLSDEM(LAMMPS *lmp, int narg, char **arg) :
    FixRigid(lmp, narg, arg), id_fix(nullptr), id_fix2(nullptr), global_grids(nullptr),
    grid_style(nullptr), grid_min(nullptr), grid_stride(nullptr), grid_scale(nullptr), grid_index(nullptr), grid_size(nullptr), grid_vol(nullptr), node_area(nullptr), grid_nnodes(nullptr)
{
  comm_forward = 8;
  maxcut = warncut = -1;
  stored_flag = 0;
  distributed_flag = 0;

  n_extra_attributes = 3;

  if (!inpfile)
    error->all(FLERR, "Must specify infile with level set for fix rigid/ls/dem");

  memory->create(grid_style, nbody, "rigid/ls/dem:grid_style");
  memory->create(grid_min, nbody, 3, "rigid/ls/dem:grid_min");
  memory->create(grid_stride, nbody, "rigid/ls/dem:grid_stride");
  memory->create(grid_scale, nbody, "rigid/ls/dem:grid_scale");
  memory->create(grid_index, nbody, "rigid/ls/dem:grid_index");
  memory->create(grid_size, nbody, 3, "rigid/ls/dem:grid_size");
  memory->create(grid_vol, nbody, "rigid/ls/dem:grid_vol");
  memory->create(node_area, nbody, "rigid/ls/dem:node_area");
  memory->create(grid_nnodes, nbody, "rigid/ls/dem:grid_nnodes");


  if (!atom->omega_flag)
    error->all(FLERR, "Fix rigid/ls/dem requires atom attribute omega");
}

/* ---------------------------------------------------------------------- */

FixRigidLSDEM::~FixRigidLSDEM()
{
  // delete extra property/atom fixes

  if (id_fix && modify->nfix) modify->delete_fix(id_fix);
  delete[] id_fix;
  if (id_fix2 && modify->nfix) modify->delete_fix(id_fix2);
  delete[] id_fix2;

  // delete nbody-length arrays

  memory->destroy(grid_style);
  memory->destroy(grid_min);
  memory->destroy(grid_stride);
  memory->destroy(grid_scale);
  memory->destroy(grid_index);
  memory->destroy(grid_size);
  memory->destroy(grid_vol);
  memory->destroy(node_area);
  memory->destroy(grid_nnodes);

  memory->destroy(global_grids);
}

/* ---------------------------------------------------------------------- */

int FixRigidLSDEM::setmask()
{
  int mask = FixRigid::setmask();
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRigidLSDEM::post_constructor()
{
  // Store positional information of grain on all atoms
  id_fix = utils::strdup(id + std::string("_FIX_PROP_ATOM"));
  modify->add_fix(fmt::format(
    "{} all property/atom d2_ls_dem_com 3 d2_ls_dem_quat 4 d2_ls_dem_omega 3 d2_ls_dem_n 3 d2_ls_dem_fs 3 i_ls_dem_touch_id d_ls_dem_fn1 d_ls_dem_fs1 ghost yes writedata no",
     id_fix));
  int tmp1, tmp2;
  index_ls_dem_com = atom->find_custom("ls_dem_com", tmp1, tmp2);
  index_ls_dem_quat = atom->find_custom("ls_dem_quat", tmp1, tmp2);
  index_ls_dem_omega = atom->find_custom("ls_dem_omega", tmp1, tmp2);
  index_ls_dem_n = atom->find_custom("ls_dem_n", tmp1, tmp2);
  index_ls_dem_fs = atom->find_custom("ls_dem_fs", tmp1, tmp2);
  index_ls_dem_touch_id = atom->find_custom("ls_dem_touch_id", tmp1, tmp2);
  index_ls_dem_fn1 = atom->find_custom("ls_dem_fn1", tmp1, tmp2);
  index_ls_dem_fs1 = atom->find_custom("ls_dem_fs1", tmp1, tmp2);
}

/* ---------------------------------------------------------------------- */

void FixRigidLSDEM::init()
{
  FixRigid::init();

  // For updating center of mass
  double **grain_com = atom->darray[index_ls_dem_com];
  double **grain_quat = atom->darray[index_ls_dem_quat];
  double **grain_omega = atom->darray[index_ls_dem_omega];
  int *touch_id = atom->ivector[index_ls_dem_touch_id];
  int ibody, i, a;
  int dimension = domain->dimension;

  for (i = 0; i < atom->nlocal; i++) {
    ibody = body[i];
    if (ibody == -1)
      error->all(FLERR, "Cannot mix LS DEM and regular DEM grains");
    grain_com[i][0] = xcm[ibody][0];
    grain_com[i][1] = xcm[ibody][1];
    grain_com[i][2] = xcm[ibody][2];

    grain_quat[i][0] = quat[ibody][0];
    grain_quat[i][1] = quat[ibody][1];
    grain_quat[i][2] = quat[ibody][2];
    grain_quat[i][3] = quat[ibody][3];

    grain_omega[i][0] = omega[ibody][0];
    grain_omega[i][1] = omega[ibody][1];
    grain_omega[i][2] = omega[ibody][2];

    touch_id[i] = -1;
  }

  // Copy maximum cutoff from pair style
  // This will determine radius of level set around nodes in the distributed case.
  // TBD: Some (automatic?) optimisation.
  if (!utils::strmatch(force->pair_style,"^ls/dem"))
    error->all(FLERR, "Must use pair ls/dem with fix rigid/ls/dem");
  auto pair = dynamic_cast<PairLSDEM *>(force->pair);
  maxcut = pair->maxcut;

  int index_global = 0;
  int distributed_flag = 0;
  if (!stored_flag) {
    stored_flag = 1;

    int *ntotal_global;
    char **gridfiles;
    memory->create(ntotal_global, nbody, "rigid/ls/dem:ntotal_global");
    memory->create(gridfiles, nbody, MAXLINE, "rigid/ls/dem:gridfiles");
    read_gridfile_names(gridfiles);

    // Read grid dimensions for all bodies
    std::map <std::string, std::set<int>> file_map;
    std::string filename;
    int grid_size_flat, max_grid_size_flat(0);
    double min_stride = DBL_MAX;
    for (ibody = 0; ibody < nbody; ibody++) {
      filename.assign(gridfiles[ibody]); // Retrieve file name
      read_gridfile(ibody, 0, filename, grid_size, nullptr); // Get only grid sizes (tag 0)
      file_map[filename].insert(ibody);

      // Calculate and save grid properties
      grid_size_flat = grid_size[ibody][0] * grid_size[ibody][1] * grid_size[ibody][2];
      max_grid_size_flat = MAX(max_grid_size_flat, grid_size_flat);
      min_stride = MIN(min_stride, grid_stride[ibody] * grid_scale[ibody]);

      // Store global info
      grid_index[ibody] = -1;
      if (grid_style[ibody] == GLOBAL) {
        // Copy from prior entry if it exists
        if (file_map.find(filename) != file_map.end())
          for (const auto& jbody : file_map[filename])
            if (grid_index[jbody] != -1)
              grid_index[ibody] = grid_index[jbody];

        // If no global instances, add new index
        if (grid_index[ibody] == -1) {
          grid_index[ibody] = index_global;
          ntotal_global[index_global] = grid_size_flat;
          index_global += 1;
        }
      } else {
        distributed_flag = 1;
      }
    }

    // ------------------------------ //
    // Allocate memory for level sets //
    // ------------------------------ //

    int ntotal;
    // All local grids sized on finest grid (fix property/atom requires fixed-size containers)
    rcell = maxcut / min_stride + 2; // +1 for interpolation +1 for safety
    warncut = maxcut - min_stride; // Currently unused

    for (ibody = 0; ibody < nbody; ibody++)
      grid_nnodes[ibody] = 0;
    for (i = 0; i < atom->nlocal; i++)
      grid_nnodes[body[i]] += 1;

    if (distributed_flag) {
      for (a = 0; a < 3; a++) subgrid_size[a] = 2 * rcell + 1;  // +1 for middle cell (needed?) DvdH: I think +1 is not needed, but result should be cast to int?
      if (dimension == 2) subgrid_size[2] = 1;
      id_fix2 = utils::strdup(id + std::string("_FIX_PROP_ATOM_2"));
      ntotal = subgrid_size[0] * subgrid_size[1] * subgrid_size[2];
      if (ntotal > RECOMMENDED_MAX_NGRID)
        error->warning(FLERR, "A large per-atom subgrid of size {}x{}x{} is being allocated for distributed level sets with a cutoff of {} and a min stride of {}", subgrid_size[0], subgrid_size[1], subgrid_size[2], maxcut, min_stride);
      modify->add_fix(fmt::format("{} all property/atom d2_grid_values {} d2_grid_min {} writedata no ghost yes", id_fix2, ntotal, 3));

      int tmp1, tmp2;
      index_grid_values = atom->find_custom("grid_values", tmp1, tmp2);
      index_grid_min = atom->find_custom("grid_min", tmp1, tmp2);
    }

    if (index_global) {
      memory->create_ragged(global_grids, index_global, ntotal_global, "rigid/ls/dem:global_grids");
    }

    // ------------------------------ //
    // Read and store level sets      //
    // ------------------------------ //

    double *temp_grid_values = nullptr;
    memory->create(temp_grid_values, max_grid_size_flat, "rigid/ls/dem:temp_grid_values");

    double **grid_values, **grid_min_local;
    if (distributed_flag) {
      grid_values = atom->darray[index_grid_values];
      grid_min_local = atom->darray[index_grid_min];
    }

    // TODO: DOES THIS ONLY WORK WHEN GRAINS ARE AXIS-ALIGNED ?
    //       I.E. WE MUST TELL THE USERS NOT TO ROTATE ANYTHING BEFORE RIGID IS DONE ?
    // DvdH: The LS grid should ALWAYS be axis-aligned. Something else should be giving an orientation to which we rotate just after loading.

    double *ls_val;
    double delx, dely, delz, area;
    double **x = atom->x;
    int need_distributed, need_global, need_padding;
    double com_temp[3], density, inertia_temp[3][3], evectors[3][3], scale, scale2, scale3;
    int nx, ny, nz, ix_node, iy_node, iz_node, xmincell, ymincell, zmincell, index;
    int ix_global, iy_global, iz_global, index_global, index_local, index_grid_min_local[3];
    for (const auto& pair : file_map) { // Loop over all <filename, [bodyIDs]>
      filename = pair.first;
      // DvdH: Why is ibody here -1?
      read_gridfile(-1, 1, filename, nullptr, temp_grid_values);

      // Compute grain properties per unique grid
      for (ibody = 0; ibody < nbody; ibody++) {
        // TODO JBC: this is the quick and easy way of doing this.
        //           All procs share the same nbody so we can instead replace with range-based loop: for(int ibody : pair.second) {
        //           But range-based loops require a declaration, which clashes with LAMMPS style of declaring ibody at the start (which I don't like, but think was easier as of now than changing all the indices of this function)
        if (pair.second.find(ibody) == pair.second.end())
          continue;

        // Compute properties from the level-set grid
        grid_vol[ibody] = compute_grid_properties(grid_size[ibody], grid_stride[ibody], temp_grid_values, com_temp, inertia_temp, filename);

        // Comparing if CoM in level-set grid is indeed aligned with CoM provided in the input file.
        // A misalignment would mean that the forces and rotations are applied to the wrong point in
        // space, leading to integration issues.
        if ( sqrt( (grid_min[ibody][0]+com_temp[0])*(grid_min[ibody][0]+com_temp[0])+
                  (grid_min[ibody][1]+com_temp[1])*(grid_min[ibody][1]+com_temp[1])+
                  (grid_min[ibody][2]+com_temp[2])*(grid_min[ibody][2]+com_temp[2])) > (0.5 * grid_stride[ibody])
        ) {
          error->all(FLERR, "Centre of mass computed from the LS grid does not agree with that provided in the input grid file! Grid min given at {} {} {} and CoM computed at {} {} {}.",
            grid_min[ibody][0],grid_min[ibody][1],grid_min[ibody][2],com_temp[0],com_temp[1],com_temp[2]);
        }

        // Overwrite inertia, could modify logic (compare or warn) if desired

        // Calculate eigen system of inertia tensor
        int ierror = MathEigen::jacobi3(inertia_temp, inertia[ibody], evectors, 1);
        if (ierror) error->all(FLERR, "Insufficient Jacobi rotations for LS grid");

        // Set grain orientation based on eigenvectors of inertia tensor
        for (a = 0; a < 3; a++) {
          ex_space[ibody][a] = evectors[a][0];
          ey_space[ibody][a] = evectors[a][1];
          ez_space[ibody][a] = evectors[a][2];
        }

        // Surface area calculation with default epsilon (diff between inner and outer) of two times grid stride.
        area = compute_surface_area(dimension, grid_size[ibody], grid_stride[ibody], temp_grid_values);
        // Test for physical realism
        if (!((area > 0.0) && std::isfinite(area)))
          error->all(FLERR, "Surface area calculation returns nonsense, giving {}", area);
        node_area[ibody] = area;

        // Normalise by number of nodes
        node_area[ibody] /= grid_nnodes[ibody];

        // Scale all relevant quantities by given scaling of grain size
        scale = grid_scale[ibody];
        scale2 = scale*scale;
        scale3 = scale*scale2;
        density = masstotal[ibody] / grid_vol[ibody];
        grid_stride[ibody] *= scale;
        MathExtra::scale3(scale, grid_min[ibody]);
        node_area[ibody] *= scale2;
        grid_vol[ibody] *= scale3;
        MathExtra::scale3(density*scale2*scale3, inertia[ibody]);
      }

      // Start handling memory approach
      need_distributed = 0; // Save relevant grid snippet at node, regardless of duplicity
      need_global = 0;  // Save the entire grid as a shared memory stucture between grains with the same grid
      for (const auto& jbody : file_map[filename]) {
        if (grid_style[jbody] == DISTRIBUTED) {
          need_distributed = 1;
        } else if (grid_style[jbody] == GLOBAL) {
          need_global = 1;
          index_global = grid_index[jbody];
        }
      }

      if (need_global) {
        for (int n = 0; n < ntotal_global[index_global]; n++)
          // Unscaled grid values of grains stored globally to avoid duplicating memory
          global_grids[index_global][n] = temp_grid_values[n];
      }

      if (need_distributed) {
        for (i = 0; i < atom->nlocal; i++) {
          ibody = body[i];

          need_padding = 0;
          if (pair.second.find(ibody) == pair.second.end())
            continue; // Ideally would have list of all atoms in a rigid body... not sure if exists...

          nx = grid_size[ibody][0];
          ny = grid_size[ibody][1];
          nz = grid_size[ibody][2];
          ntotal = nx * ny * nz;

          // Location of atom/node relative to CoM
          delx = x[i][0] - grain_com[i][0];
          dely = x[i][1] - grain_com[i][1];
          delz = x[i][2] - grain_com[i][2];

          // Account for PBCs
          domain->minimum_image(FLERR, delx, dely, delz);

          // Location of atom/node relative to entire grain grid minimum.
          delx -= grid_min[ibody][0];
          dely -= grid_min[ibody][1];
          delz -= grid_min[ibody][2];

          // Index of atom/node in entire grain grid.
          double stride = grid_stride[ibody];
          ix_node = int(delx / stride);
          iy_node = int(dely / stride);
          iz_node = int(delz / stride);

          // Index of local grid minimum in entire grain grid. If any goes below zero, error below catches it.
          index_grid_min_local[0] = ix_node - rcell;
          index_grid_min_local[1] = iy_node - rcell;
          index_grid_min_local[2] = (dimension == 3) ? iz_node - rcell : 0;

          // Location of local grid minimum relative to CoM
          grid_min_local[i][0] = index_grid_min_local[0] * stride + grid_min[ibody][0];
          grid_min_local[i][1] = index_grid_min_local[1] * stride + grid_min[ibody][1];
          grid_min_local[i][2] = index_grid_min_local[2] * stride + grid_min[ibody][2];

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
                  grid_values[i][index_local] = temp_grid_values[index_global] * grid_scale[ibody];
                }
              }
            }
          }

          // JBC: This might be deleted with watershed. If we keep it, might consider moving it so it doesn't print too many warnings
          if (need_padding)
            error->warning(FLERR, "Level set of body {} does not include a large enough buffer for the distributed grid cutoff on atom {}. Local grid padded with BIG values", ibody, i);
        }
      }

    }

    memory->destroy(gridfiles);
    memory->destroy(temp_grid_values);
    memory->destroy(ntotal_global);

  } else {
    if (distributed_flag) {
      int tmp1, tmp2;
      index_grid_values = atom->find_custom("grid_values", tmp1, tmp2);
      index_grid_min = atom->find_custom("grid_min", tmp1, tmp2);
    }
  }
}

/* ----------------------------------------------------------------------
   compute initial fcm and torque on bodies, also initial virial
   reset all particle velocities to be consistent with vcm and omega

   Forces apply at the contact point between a surface atom and a level-set.
   There is no LAMMPS structure for it so forces are applied on nearest atoms
   Torques computed from forces applied at the atom position would be off.
   To avoid this miscalculation:
     1. exact torques are applied on (extended) atoms in pair_ls_dem
     2. torques are not computed from forces on atoms (unlike Fix Rigid)

     TODO: this is a lot of code duplication. A cleaner way to do that
           could be to write little helper functions for computing torques from forces
           and not call it for LSDEM
------------------------------------------------------------------------- */

void FixRigidLSDEM::setup(int vflag)
{
  compute_forces_and_torques();

  // enforce 2d body forces and torques

  if (domain->dimension == 2) enforce2d();

  // zero langextra in case Langevin thermostat not used
  // no point to calling post_force() here since langextra
  // is only added to fcm/torque in final_integrate()

  int ibody, i, n;
  for (ibody = 0; ibody < nbody; ibody++)
    for (i = 0; i < 6; i++) langextra[ibody][i] = 0.0;

  // virial setup before call to set_v

  v_init(vflag);

  // set velocities from angmom & omega

  for (ibody = 0; ibody < nbody; ibody++)
    MathExtra::angmom_to_omega(angmom[ibody], ex_space[ibody], ey_space[ibody],
                               ez_space[ibody], inertia[ibody], omega[ibody]);

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

  if (id_no_grav) {
    for (ibody = 0; ibody < nbody; ibody++)
      apply_grav[ibody] = 1;
    int *mask = atom->mask;
    for (i = 0; i < nlocal; i++) {
      ibody = body[i];
      if (ibody < 0) continue;
      if (mask[i] & no_grav_group_bit)
        apply_grav[ibody] = 0;
    }

    MPI_Allreduce(MPI_IN_PLACE, apply_grav, nbody, MPI_INT, MPI_MIN, world);
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidLSDEM::setup_pre_force(int vflag)
{
  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixRigidLSDEM::pre_force(int vflag)
{
  comm->forward_comm(this);
}

/* ---------------------------------------------------------------------- */

void FixRigidLSDEM::initial_integrate(int vflag)
{
  FixRigid::initial_integrate(vflag);

  double **grain_com = atom->darray[index_ls_dem_com];
  double **grain_quat = atom->darray[index_ls_dem_quat];
  double **grain_omega = atom->darray[index_ls_dem_omega];

  int ibody;
  for (int i = 0; i < atom->nlocal; i++) {
    ibody = body[i];
    grain_com[i][0] = xcm[ibody][0];
    grain_com[i][1] = xcm[ibody][1];
    grain_com[i][2] = xcm[ibody][2];

    grain_quat[i][0] = quat[ibody][0];
    grain_quat[i][1] = quat[ibody][1];
    grain_quat[i][2] = quat[ibody][2];
    grain_quat[i][3] = quat[ibody][3];

    grain_omega[i][0] = omega[ibody][0];
    grain_omega[i][1] = omega[ibody][1];
    grain_omega[i][2] = omega[ibody][2];
  }
}

/* ---------------------------------------------------------------------- */

int FixRigidLSDEM::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i, j, m;
  double **grain_com = atom->darray[index_ls_dem_com];
  double **grain_quat = atom->darray[index_ls_dem_quat];
  double **grain_omega = atom->darray[index_ls_dem_omega];

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(body[j]).d;

    buf[m++] = grain_com[j][0];
    buf[m++] = grain_com[j][1];
    buf[m++] = grain_com[j][2];

    buf[m++] = grain_quat[j][0];
    buf[m++] = grain_quat[j][1];
    buf[m++] = grain_quat[j][2];
    buf[m++] = grain_quat[j][3];

    buf[m++] = grain_omega[j][0];
    buf[m++] = grain_omega[j][1];
    buf[m++] = grain_omega[j][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixRigidLSDEM::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;
  double **grain_com = atom->darray[index_ls_dem_com];
  double **grain_quat = atom->darray[index_ls_dem_quat];
  double **grain_omega = atom->darray[index_ls_dem_omega];

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    body[i] = (int) ubuf(buf[m++]).i;

    grain_com[i][0] = buf[m++];
    grain_com[i][1] = buf[m++];
    grain_com[i][2] = buf[m++];

    grain_quat[i][0] = buf[m++];
    grain_quat[i][1] = buf[m++];
    grain_quat[i][2] = buf[m++];
    grain_quat[i][3] = buf[m++];

    grain_omega[i][0] = buf[m++];
    grain_omega[i][1] = buf[m++];
    grain_omega[i][2] = buf[m++];
  }
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

void FixRigidLSDEM::compute_forces_and_torques()
{
  int i,ibody;

  // sum over atoms to get force and torque on rigid body

  double **f = atom->f;
  int nlocal = atom->nlocal;

  for (ibody = 0; ibody < nbody; ibody++)
    for (i = 0; i < 6; i++) sum[ibody][i] = 0.0;

  for (i = 0; i < nlocal; i++) {
    if (body[i] < 0) continue;
    ibody = body[i];

    sum[ibody][0] += f[i][0];
    sum[ibody][1] += f[i][1];
    sum[ibody][2] += f[i][2];
  }

  // extended particles add their torque to torque of body

  if (extended) {
    double **torque_one = atom->torque;

    for (i = 0; i < nlocal; i++) {
      if (body[i] < 0) continue;
      ibody = body[i];

      if (eflags[i] & TORQUE) {
        sum[ibody][3] += torque_one[i][0];
        sum[ibody][4] += torque_one[i][1];
        sum[ibody][5] += torque_one[i][2];
      }
    }
  }

  MPI_Allreduce(sum[0], all[0], 6 * nbody, MPI_DOUBLE, MPI_SUM, world);

  // No Langevin thermostat forces included

  for (ibody = 0; ibody < nbody; ibody++) {
    fcm[ibody][0] = all[ibody][0];
    fcm[ibody][1] = all[ibody][1];
    fcm[ibody][2] = all[ibody][2];
    torque[ibody][0] = all[ibody][3];
    torque[ibody][1] = all[ibody][4];
    torque[ibody][2] = all[ibody][5];
  }

  // add gravity force to COM of each body

  if (id_gravity) {
    for (ibody = 0; ibody < nbody; ibody++) {
      if (apply_grav[ibody]) {
        fcm[ibody][0] += gvec[0] * masstotal[ibody];
        fcm[ibody][1] += gvec[1] * masstotal[ibody];
        fcm[ibody][2] += gvec[2] * masstotal[ibody];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   one-time reading of file names for LS grid
------------------------------------------------------------------------- */

void FixRigidLSDEM::read_gridfile_names(char **gridfiles)
{
  int nchunk, id, eofflag, nlines;
  FILE *fp;
  char *eof, *start, *next, *buf;
  char line[MAXLINE] = {'\0'};

  // open file and read and parse first non-empty, non-comment line containing the number of bodies
  if (comm->me == 0) {
    fp = fopen(inpfile,"r");
    if (fp == nullptr)
      error->one(FLERR, "Cannot open fix rigid/ls/dem infile {}: {}", inpfile, utils::getsyserror());
    while (true) {
      eof = fgets(line, MAXLINE, fp);
      if (eof == nullptr) error->one(FLERR, "Unexpected end of fix rigid/ls/dem infile");
      start = &line[strspn(line, " \t\n\v\f\r")];
      if (*start != '\0' && *start != '#') break;
    }
    nlines = utils::inumeric(FLERR, utils::trim(line), true, lmp);
    if (nlines == 0) fclose(fp);
  }
  MPI_Bcast(&nlines, 1, MPI_INT, 0, world);

  // empty file with 0 lines is needed to trigger initial restart file
  // generation when no infile was previously used.

  if (nlines == 0) return;
  else if (nlines < 0) error->all(FLERR, "Fix rigid infile has incorrect format");

  auto buffer = new char[CHUNK * MAXLINE];
  int nread = 0;
  int me = comm->me;
  while (nread < nlines) {
    nchunk = MIN(nlines - nread, CHUNK);
    eofflag = utils::read_lines_from_file(fp, nchunk, MAXLINE, buffer, me, world);
    if (eofflag) error->all(FLERR, "Unexpected end of fix rigid/ls/dem infile");

    buf = buffer;
    next = strchr(buf, '\n');
    *next = '\0';
    int nwords = utils::count_words(utils::trim_comment(buf));
    *next = '\n';

    if (nwords != (ATTRIBUTE_PERBODY + n_extra_attributes))
      error->all(FLERR, "Incorrect rigid body format in fix rigid/ls/dem file");

    // loop over lines of rigid body attributes
    // tokenize the line into values
    // id = rigid body ID
    // use ID as-is for SINGLE, as mol-ID for MOLECULE, as-is for GROUP

    for (int i = 0; i < nchunk; i++) {
      next = strchr(buf,'\n');
      *next = '\0';

      try {
        ValueTokenizer values(buf);
        id = values.next_int();
        if (rstyle == MOLECULE) {
          if (id <= 0 || id > maxmol)
            throw TokenizerException("invalid rigid molecule ID ", std::to_string(id));
          id = mol2body[id];
        } else id--;

        if (id < 0 || id >= nbody)
          throw TokenizerException("invalid_rigid body ID ", std::to_string(id + 1));

        values.skip(19);
        grid_style[id] = values.next_int();
        if (grid_style[id] != 0 && grid_style[id] != 1)
          throw TokenizerException("invalid_rigid memory model ", std::to_string(grid_style[id]));

        grid_scale[id] = values.next_double();

        strcpy(gridfiles[id], values.next_string().data());
      } catch (TokenizerException &e) {
        error->all(FLERR, "Invalid fix rigid/ls/dem infile: {}", e.what());
      }
      buf = next + 1;
    }
    nread += nchunk;
  }

  if (comm->me == 0) fclose(fp);
  delete[] buffer;
}

/* ----------------------------------------------------------------------
   write out restart info for mass, COM, inertia tensor, image flags to file
   identical format to inpfile option, so info can be read in when restarting
   only proc 0 writes list of global bodies to file
------------------------------------------------------------------------- */

void FixRigidLSDEM::write_restart_file(const char *file)
{
  if (comm->me) return;

  FixRigid::write_restart_file(file); // Todo, save LS DEM data
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixRigidLSDEM::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = FixRigid::memory_usage();
  // todo
  return bytes;
}

/* ----------------------------------------------------------------------
   read per rigid body level-set grid values from user-provided file
   files gridfiles to read from stored previously by readfile() function
   first line = grid_sizex grid_sizey grid_sizez
   followed by grid_sizex * grid_sizey * grid_sizez lines of level set values at the grid points
   which = 0, read only the size of the level-set grid
   which = 1, read the values of the level-set grid
------------------------------------------------------------------------- */

void FixRigidLSDEM::read_gridfile(int ibody, int which, std::string filename, int **grid_size, double *grid_values)
{
  int grid_shape_buf[dim];
  double grid_size_buf[dim + 1];
  int nchunk, eofflag;
  FILE *fp;
  char *eof, *start, *next, *buf;
  char line[MAXLINE] = {'\0'};

  // open file and read and parse first non-empty, non-comment line containing the 2 or 3 grid dimensions
  // Broadcast to other procs
  // TODO: there must be a better way to read the first 2,3 lines
  int nlines = 1;
  const char* gridfile = filename.c_str();
  if (comm->me == 0) {
    fp = fopen(gridfile, "r");
    if (fp == nullptr)
      error->one(FLERR, "Cannot open fix rigid/ls/dem gridfile {}: {}", gridfile, utils::getsyserror());
    while (true) {
      eof = fgets(line, MAXLINE, fp);
      if (eof == nullptr) error->one(FLERR,"Unexpected end of fix rigid/ls/dem gridfile");
      start = &line[strspn(line, " \t\n\v\f\r")];
      if (*start != '\0' && *start != '#') break;
    }
    auto grid_shape = utils::split_words(line);
    if (grid_shape.size() != dim)
      error->one(FLERR, "Fix rigid/ls/dem gridfile {} has {} dimensions but simulation is {}D",
                          gridfile, grid_shape.size(), dim);
    for (int idim = 0; idim < dim; idim++)
      grid_shape_buf[idim] = utils::inumeric(FLERR, grid_shape[idim], false, lmp);

    eof = fgets(line, MAXLINE, fp);
    if (eof == nullptr) error->one(FLERR, "Unexpected end of fix rigid/ls/dem gridfile");
    grid_size_buf[0] = utils::numeric(FLERR, utils::trim(line), false, lmp);
    if (grid_size_buf[0] <= 0.0)
      error->one(FLERR, "Grid stride for rigid/ls/dem gridfile {} must be positive", gridfile);

    eof = fgets(line, MAXLINE, fp);
    if (eof == nullptr) error->one(FLERR, "Unexpected end of fix rigid/ls/dem gridfile");
    auto grid_corner = utils::split_words(line);
    if (grid_corner.size() != dim)
      error->one(FLERR, "Fix rigid/ls/dem gridfile {} specifies {} grid corner cooridnates but simulation is {}D",
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

  // TODO: I left the 2 lines below from original rigid::readline() notsure if needed
  // empty file with 0 lines is needed to trigger initial restart file
  // generation when no infile was previously used.
  if (nlines == 0) return;
  else if (nlines < 0) error->all(FLERR, "Fix rigid/ls/dem gridfile has incorrect format");

  if (which == 0) {
    // All these quantities are stored per body (grain) because different scaling of the
    // grain size might be applied later. They are needed at the grain level anyway for
    // most memory distribution methods.
    grid_stride[ibody] = grid_size_buf[0];
    for (int idim = 0; idim < dim; idim++) {
      // The grid_size_buf is [stride, xmin, ymin, zmin].
      grid_min[ibody][idim] = grid_size_buf[idim + 1];
      // The grid_shape_buf is [nx, ny, nz]
      grid_size[ibody][idim] = (int) grid_shape_buf[idim];
    }

    if (dim == 2) {
      grid_min[ibody][2] = 0.0;
      grid_size[ibody][2] = 1;
    }

  } else { // change to elif check
    auto buffer = new char[CHUNK * MAXLINE];
    int nread = 0;
    int me = comm->me;
    while (nread < nlines) {
      nchunk = MIN(nlines-nread, CHUNK);
      eofflag = utils::read_lines_from_file(fp, nchunk, MAXLINE, buffer, me, world);
      if (eofflag) error->all(FLERR, "Unexpected end of fix rigid/ls/dem gridfile");

      buf = buffer;
      next = strchr(buf, '\n');
      *next = '\0';
      int nwords = utils::count_words(utils::trim_comment(buf));
      *next = '\n';

      // TODO: there must be a better way than tokenizing single value
      // Kept as is for now to re-use existing rigid::readfile() code
      // Maybe in the future we want to have multiple value per line,
      // In which case it will be useful to have that architecture
      if (nwords != 1)
        error->all(FLERR, "LSDEM gridfile format requires one entry per line");

      // loop over lines of level set grid and tokenize level set values
      for (int i = 0; i < nchunk; i++) {
        next = strchr(buf, '\n');
        *next = '\0';

        try {
          // Level-set values are read into the temporary grid_values array
          ValueTokenizer values(buf);
          grid_values[nread + i] = values.next_double();
        } catch (TokenizerException &e) {
          error->all(FLERR, "Invalid fix rigid/ls/dem gridfile: {}", e.what());
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
  Compute CoM, moment of inertia, and volume of a grid
------------------------------------------------------------------------- */

double FixRigidLSDEM::compute_grid_properties(int *grid_size, double stride, double *grid_values, double *com_temp, double inertia_temp[3][3], std::string filename)
{
  // Volume integration

  // This is the reference distance values that determines the smearing with of
  // the Heaviside step function. Current expression is the half-diagional of the
  // grid cell divided by a smearing constant.
  double smearCoeff = 1.5;
  double ls_ref = 1.0;
  if (smearCoeff != 0)
    ls_ref = sqrt(0.75) * stride / smearCoeff;

  // Cell volume, temporary grid points, integration volume.
  double volume_cell = stride * stride;
  if (domain->dimension == 3) volume_cell *= stride;

  // Integration
  double dV, ls_val;
  double volume = 0.0;
  for (int a = 0; a < 3; a++) com_temp[a] = 0.0;
  for (int ind_x = 0; ind_x < grid_size[0]; ind_x++) {
    for (int ind_y = 0; ind_y < grid_size[1]; ind_y++) {
      for (int ind_z = 0; ind_z < grid_size[2]; ind_z++) {
        ls_val = grid_values[ind_x + ind_y * grid_size[0] + ind_z * grid_size[0] * grid_size[1]];
        dV = smeared_heaviside_step(-ls_val / ls_ref) * volume_cell;
        if (dV > 0.0) {
          volume += dV;
          com_temp[0] += ind_x * stride * dV;
          com_temp[1] += ind_y * stride * dV;
          com_temp[2] += ind_z * stride * dV;
        }
      }
    }
  }
  com_temp[0] /= volume;
  com_temp[1] /= volume;
  com_temp[2] /= volume;

  // Computing the inertia tensor (a second loop is unavoidable)
  double delx, dely, delz;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      inertia_temp[a][b] = 0.0;
    }
  }
  for (int ind_x = 0; ind_x < grid_size[0]; ind_x++) {
    for (int ind_y = 0; ind_y < grid_size[1]; ind_y++) {
      for (int ind_z = 0; ind_z < grid_size[2]; ind_z++) {
        ls_val = grid_values[ind_x + ind_y * grid_size[0] + ind_z * grid_size[0] * grid_size[1]];
        dV = smeared_heaviside_step(-ls_val / ls_ref) * volume_cell;
        if (dV > 0.0) {
          delx = ind_x * stride - com_temp[0];
          dely = ind_y * stride - com_temp[1];
          delz = ind_z * stride - com_temp[2];
          inertia_temp[0][0] += (dely * dely + delz * delz) * dV;
          inertia_temp[1][1] += (delx * delx + delz * delz) * dV;
          inertia_temp[2][2] += (delx * delx + dely * dely) * dV;
          inertia_temp[0][1] -= delx * dely * dV;
          inertia_temp[0][2] -= delx * delz * dV;
          inertia_temp[1][2] -= dely * delz * dV;
        }
      }
    }
  }
  inertia_temp[1][0] = inertia_temp[0][1];
  inertia_temp[2][0] = inertia_temp[0][2];
  inertia_temp[2][1] = inertia_temp[1][2];
  // Check to see if level set has a non-inertial reference frame
  double I_diag_norm = sqrt(inertia_temp[0][0] * inertia_temp[0][0] + inertia_temp[1][1] * inertia_temp[1][1] + inertia_temp[2][2] * inertia_temp[2][2]);
  double I_off_diag_norm = sqrt(2.0 * (inertia_temp[0][1] * inertia_temp[0][1] + inertia_temp[0][2] * inertia_temp[0][2] + inertia_temp[1][2] * inertia_temp[1][2]));
  if (I_off_diag_norm / I_diag_norm > EPSILON_INERTIA)
    error->all(FLERR, "Non-inertial reference frame detected for level set in {}. Intergration of rotational motion will be wrong.", filename);

  return volume;
}

/* ----------------------------------------------------------------------
   Find the value of node (atom) i in j's LS grid.
------------------------------------------------------------------------- */

double FixRigidLSDEM::get_ls_value(int i, int j, double *normal)
{
  double **x = atom->x;
  double **grain_com = atom->darray[index_ls_dem_com];
  double **grain_quat = atom->darray[index_ls_dem_quat];

  int jbody = body[j];
  double dist, nx, ny, nz(0.0);
  double strideinv = 1.0 / grid_stride[jbody];

  // Calculate position of node i in node j's grid using:
  //   x[i][0-2] = location of i
  //   x[j][0-2] = location of j
  //   grain_com[j][0-2] = CoM of j's grain
  //   grain_quat[j][0-3] = quat of j's grain

  // Location of the node (atom) of i relative to the centre of mass (CoM) of j
  double delx = x[i][0] - grain_com[j][0];
  double dely = x[i][1] - grain_com[j][1];
  double delz = x[i][2] - grain_com[j][2];

  // Account for PBCs
  domain->minimum_image(FLERR, delx, dely, delz);

  // Apply quaternion rotation to move into local reference frame of grain j grid.
  // Here, grain_quat is local->global. Therefore, grain_quat_conj is global -> local.
  double x_local[3];
  double dx[3] = {delx, dely, delz};
  double grain_quat_conj[4];
  MathExtra::qconjugate(grain_quat[j], grain_quat_conj);
  MathExtra::quatrotvec(grain_quat_conj, dx, x_local);
  // See comments above functions in math_extra.h/cpp for details

  int ncol, nrow, nslice;
  double *mygrid;
  if (grid_style[jbody] == DISTRIBUTED) {
    mygrid = atom->darray[index_grid_values][j];
    // Translate local coordinates such that they are relative
    // to the lower corner of the node's level set grid.
    double **local_grid_min = atom->darray[index_grid_min];
    x_local[0] -= local_grid_min[j][0];
    x_local[1] -= local_grid_min[j][1];
    x_local[2] -= local_grid_min[j][2];

    ncol = subgrid_size[0];
    nrow = subgrid_size[1];
    nslice = subgrid_size[2];
  } else {
    mygrid = global_grids[grid_index[jbody]];
    // Translate local coordinates such that they are relative
    // to the lower corner of the grain's level set grid.
    x_local[0] -= grid_min[jbody][0];
    x_local[1] -= grid_min[jbody][1];
    x_local[2] -= grid_min[jbody][2];

    ncol = grid_size[jbody][0];
    nrow = grid_size[jbody][1];
    nslice = grid_size[jbody][2];
  }

  // Normalise the coordinates to be in units of the number of grid cells.
  double x_red = x_local[0] * strideinv;
  double y_red = x_local[1] * strideinv;
  double z_red = x_local[2] * strideinv;

  // Calculate index from relative coordinate, being careful with integer division.
  int ind_x = int(x_red);
  int ind_y = int(y_red);
  int ind_z = int(z_red); // Should always be zero in 2D.

  // Checking whether x_local lies within the grid. Avoids edge cases where finite precision
  // leads to e.g. a x=-0.1 coordinate to fall outside of a grid that starts at x=-0.1.
  if ((ind_x < 0 || ind_x >= (nrow - 1)) || (ind_y < 0 || ind_y >= (ncol - 1)) ||
      ((domain->dimension == 3) && (ind_z < 0 || ind_z >= (nslice - 1))))
    //error->one(FLERR, "Contacting node {} is outside of node {}'s LS grid", atom->tag[i], atom->tag[j]);
    return BIG; // To avoid having to perfectly match the neighbour listing cutoff with the grid size.

  // The normalised coordinates within the current grid cell.
  // May be safer to cap them with math::max(math::min(x_red, 1.0), 0.0)
  x_red = x_red - static_cast<double>(ind_x);
  y_red = y_red - static_cast<double>(ind_y);
  z_red = z_red - static_cast<double>(ind_z); // Should always be zero in 2D.

  //  Interpolate
  int my_index = ind_x + ind_y * ncol + ind_z * ncol * nrow;

  // Level-set values on the grid points in the lower z plane (ind_z)
  double ls000 = mygrid[my_index];
  double ls100 = mygrid[my_index + 1];
  double ls010 = mygrid[my_index + ncol];
  double ls110 = mygrid[my_index + 1 + ncol];

  // Bi-linear interpolation in the lower z plane (ind_z)
  double lsxy0 = ls000 + y_red * (ls010 - ls000) +
                 x_red * (ls100 - ls000 + y_red * (ls110 - ls100 - ls010 + ls000));
  dist = lsxy0;

  // Computing normal as the gradient of trilinear interpolation
  // Chain rule: d(dist)/d(x_local) = d(dist)/d(x_red) * (1/stride)
  // Vector eventually normalized to enforce unit normal, so 1/stride factor omitted
  nx = ls100 - ls000 + y_red * (ls110 - ls100 - ls010 + ls000);
  ny = ls010 - ls000 + x_red * (ls110 - ls100 - ls010 + ls000);

  if (domain->dimension == 3) { // 3D
    // Level-set values on the grid points in the upper z plane (ind_z+1)
    double ls001 = mygrid[my_index + ncol * nrow];
    double ls101 = mygrid[my_index + 1 + ncol * nrow];
    double ls011 = mygrid[my_index + ncol + ncol * nrow];
    double ls111 = mygrid[my_index + 1 + ncol + ncol * nrow];

    // Bi-linear interpolation in the upper z plane (ind_z+1)
    double lsxy1 = ls001 + y_red * (ls011 - ls001) +
                   x_red * (ls101 - ls001 + y_red * (ls111 - ls101 - ls011 + ls001));

    // Affecting tri-linear interpolation by linear interpolation of the two bi-linear interpolations.
    dist = z_red * (lsxy1 - lsxy0) + lsxy0;
    nx *= 1 - z_red;
    nx += z_red * (ls101 - ls001 + y_red * (ls111 - ls101 - ls011 + ls001));
    ny *= 1 - z_red;
    ny += z_red * (ls011 - ls001 + x_red * (ls111 - ls101 - ls011 + ls001));
    nz = lsxy1 - lsxy0;
  }

  // Grain-stored grid values are shared and un-scaled, so apply scaling
  if (grid_style[jbody] == GLOBAL) dist *= grid_scale[jbody];

  // Normal normally doesn't need scaling, but we scaled grid_min and grid_stride
  // but not the level-set values, hence it is necessary. However, we'll normalise later anyway.

  // Get magnitude of discrete gradient for normalisation
  double mag = 1.0/sqrt(nx*nx+ny*ny+nz*nz);

  // Assign normal
  normal[0] = nx*mag;
  normal[1] = ny*mag;
  normal[2] = nz*mag;

  // Rotate normal back to global coordinates
  MathExtra::quatrotvec(grain_quat[j], normal, normal);

  //if (-dist > warncut) maybe warn that you are about to penetrate too far



  return dist;
}
