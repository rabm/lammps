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
#include <cstring>
#include <map>
#include <set>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
using namespace RigidConst;

enum {GLOBAL, DISTRIBUTED};

static constexpr double EPSILON_ITERATION = 1.0e-15;
static constexpr double EPSILON_INERTIA = 5.0e-4; // 0.5%
static constexpr int MAX_ITERATIONS = 100; // For surface area integration
static constexpr int RECOMMENDED_MAX_NGRID = 1000; // For local node grid

inline double FixRigidLSDEM::smeared_heaviside_step(double x)
{
  // A function that smoothly transition from 0 to 1 when x goes from -1 to 1.
  // For x < -1, the function should be 0. For x > 1, the function should be 1.
  // See Kawamoto et al. (2016).
  if (x <= -1){ // Outside and far away from boundary
    return 0.0;
  }else if (x >= 1){ // Inside and far away from boundary
    return 1.0;
  }else{ // Close to boundary
    return 0.5 * (1.0 + x + sin(MY_PI * x) / MY_PI);
  }
}

inline double FixRigidLSDEM::compute_volume(int *grid_size, double stride, double *grid_values, double epsilon)
{
  // Volume integration without centre of mass (re)computation and level-set offset epsilon

  // This is the reference distance values that determines the smearing with of
  // the Heaviside step function. Current expression is the half-diagional of the
  // grid cell divided by a smearing constant.
  double smearCoeff = 1.5;
  double ls_ref = 1.0;
  if (smearCoeff != 0)
    ls_ref = sqrt(0.75) * stride / smearCoeff;

  // Initialise volume and centre of mass
  double volume = 0.0;
  // Cell volume, temporary grid points, integration volume.
  double volume_cell = stride * stride;
  if (domain->dimension == 3) volume_cell *= stride;

  // Integration
  double dV, ls_val;
  for (int ind_x = 0; ind_x < grid_size[0]; ind_x++) {
    for (int ind_y = 0; ind_y < grid_size[1]; ind_y++) {
      for (int ind_z = 0; ind_z < grid_size[2]; ind_z++) {
        ls_val = grid_values[ind_x + ind_y * grid_size[0] + ind_z * grid_size[0] * grid_size[1]] - epsilon;
        dV = smeared_heaviside_step( -ls_val / ls_ref ) * volume_cell;
        if (dV > 0.0) {
          volume += dV;
        }
      }
    }
  }
  return volume;
}

//TODO: Should we have a flag (or child classes) for different memory distribution strategies?
//      a) all procs store grids, b) sub grids for each atom, c) hash table for each atom
//      then benchmark across different limits? Few large grains, lots of small grains, jamming vs. flow...

/* ---------------------------------------------------------------------- */

FixRigidLSDEM::FixRigidLSDEM(LAMMPS *lmp, int narg, char **arg) :
    FixRigid(lmp, narg, arg), id_fix(nullptr), id_fix2(nullptr), global_grids(nullptr),
    grid_style(nullptr), grid_min(nullptr), grid_stride(nullptr), grid_scale(nullptr), grid_index(nullptr), grid_size(nullptr), grid_vol(nullptr), grid_area(nullptr), grid_nnodes(nullptr)
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
  memory->create(grid_area, nbody, "rigid/ls/dem:grid_area");
  memory->create(grid_nnodes, nbody, "rigid/ls/dem:grid_nnodes");
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
  memory->destroy(grid_scale);
  memory->destroy(grid_index);
  memory->destroy(grid_size);
  memory->destroy(grid_vol);
  memory->destroy(grid_area);
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

  // Copy maximum cutoff
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
    dim = domain->dimension;
    int max_grid_size[3] = {0, 0, 0};
    double max_stride = 0;
    for (ibody = 0; ibody < nbody; ibody++) {
      filename.assign(gridfiles[ibody]);
      read_gridfile(ibody, 0, filename, grid_size, nullptr);
      file_map[filename].insert(ibody);

      // Calculate and save grid properties
      if (dim == 2) grid_size[ibody][2] = 1;
      for (a = 0; a < 3; a++)
        if (grid_size[ibody][a] > max_grid_size[a])
          max_grid_size[a] = grid_size[ibody][a];
      max_stride = MAX(max_stride, grid_stride[ibody]);

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
          ntotal_global[index_global] = grid_size[ibody][0] * grid_size[ibody][1] * grid_size[ibody][2];
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
    rcell = maxcut / max_stride + 2; // +1 for interpolation +1 for safety
    warncut = maxcut - max_stride;

    for (ibody = 0; ibody < nbody; ibody++)
      grid_nnodes[ibody] = 0;
    for (i = 0; i < atom->nlocal; i++)
      grid_nnodes[body[i]] += 1;

    if (distributed_flag) {
      for (a = 0; a < 3; a++) subgrid_size[a] = 2 * rcell + 1;  // +1 for middle cell (needed?)
      if (dim == 2) subgrid_size[2] = 1;
      id_fix2 = utils::strdup(id + std::string("_FIX_PROP_ATOM_2"));
      ntotal = subgrid_size[0] * subgrid_size[1] * subgrid_size[2];
      if (ntotal > RECOMMENDED_MAX_NGRID)
        error->warning(FLERR, "A large per-atom subgrid of size {}x{}x{} is being allocated for distributed level sets with a cutoff of {} and a max stride of {}", subgrid_size[0], subgrid_size[1], subgrid_size[2], maxcut, max_stride);
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

    ntotal = max_grid_size[0] * max_grid_size[1] * max_grid_size[2];
    double *temp_grid_values = nullptr;
    memory->create(temp_grid_values, ntotal, "rigid/ls/dem:temp_grid_values");

    double **grid_values, **grid_min_local;
    if (distributed_flag) {
      grid_values = atom->darray[index_grid_values];
      grid_min_local = atom->darray[index_grid_min];
    }

    // TODO: DOES THIS ONLY WORK WHEN GRAINS ARE AXIS-ALIGNED ?
    //       I.E. WE MUST TELL THE USERS NOT TO ROTATE ANYTHING BEFORE RIGID IS DONE ?

    double *ls_val;
    double delx, dely, delz;
    double **x = atom->x;
    int need_distributed, need_global;
    int nx, ny, nz, ix_node, iy_node, iz_node, xmincell, ymincell, zmincell, index;
    int ix_global, iy_global, iz_global, index_global, index_local, index_grid_min_local[3];
    for (const auto& pair : file_map) {
      filename = pair.first;
      read_gridfile(-1, 1, filename, nullptr, temp_grid_values);

      need_distributed = 0;
      need_global = 0;
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
          global_grids[index_global][n] = temp_grid_values[n];
      }

      if (need_distributed) {
        for (i = 0; i < atom->nlocal; i++) {
          ibody = body[i];

          if (pair.second.find(ibody) == pair.second.end())
            continue; // Ideally would have list of all atoms in a rigid body... not sure if   exists...

          nx = grid_size[ibody][0];
          ny = grid_size[ibody][1];
          nz = (dim == 3) ? grid_size[ibody][2] : 1;
          ntotal = nx * ny * nz;

          // location of atom/node relative to CoM
          delx = x[i][0] - grain_com[i][0];
          dely = x[i][1] - grain_com[i][1];
          delz = x[i][2] - grain_com[i][2];

          // Account for PBCs
          domain->minimum_image(delx, dely, delz);

          // location of atom/node relative to global grid minimum
          delx -= grid_min[ibody][0];
          dely -= grid_min[ibody][1];
          if (domain->dimension == 3) delz -= grid_min[ibody][2];

          // index of atom/node in global grid
          double stride = grid_stride[ibody];
          ix_node = int(delx / stride);
          iy_node = int(dely / stride);
          iz_node = (dim == 3) ? int(delz / stride) : 0;

          // index of local grid minimum in global grid.
          // JBC: Can this be negative if not enough padding of the LS grid relative to grain     surface? i.e. ix < rcell ?
          index_grid_min_local[0] = ix_node - rcell;
          index_grid_min_local[1] = iy_node - rcell;
          index_grid_min_local[2] = (dim == 3) ? iz_node - rcell : 0;

          // location of local grid minimum relative to CoM
          grid_min_local[i][0] = index_grid_min_local[0] * stride + grid_min[ibody][0];
          grid_min_local[i][1] = index_grid_min_local[1] * stride + grid_min[ibody][1];
          grid_min_local[i][2] = (dim == 3) ? index_grid_min_local[2] * stride + grid_min[ibody]    [2] : 0.0;

          for (int iz_local = 0; iz_local < subgrid_size[2]; iz_local++) {
            for (int iy_local = 0; iy_local < subgrid_size[1]; iy_local++) {
              for (int ix_local = 0; ix_local < subgrid_size[0]; ix_local++) {
                // Shift local cell to global cell
                ix_global = ix_local + index_grid_min_local[0];
                iy_global = iy_local + index_grid_min_local[1];
                iz_global = (dim == 3) ? iz_local + index_grid_min_local[2] : 0;

                // Explicit bounds check per dimension (safer and clearer)
                if (ix_global < 0 || ix_global >= nx ||
                    iy_global < 0 || iy_global >= ny ||
                    iz_global < 0 || iz_global >= nz)
                  error->all(FLERR, "Level set does not include a large enough buffer for the   cutoff");

                index_global = ix_global + iy_global * nx + iz_global * nx * ny;
                index_local = ix_local + iy_local * subgrid_size[0] + iz_local * subgrid_size[0] * subgrid_size[1];

                // Final sanity check (defensive)
                if (index_global < 0 || index_global >= ntotal)
                  error->all(FLERR, "Level set does not include a large enough buffer for the   cutoff");
                grid_values[i][index_local] = temp_grid_values[index_global] * grid_scale[ibody];
              }
            }
          }
        }
      }

      double com_temp[3], inertia_temp[6], inertia_matrix[3][3], evectors[3][3];
      for (ibody = 0; ibody < nbody; ibody++) {
        grid_vol[ibody] = compute_grid_properties(grid_size[ibody], grid_stride[ibody], temp_grid_values, com_temp, inertia_temp, filename);

        // Comparing if CoM in level-set grid is indeed aligned with CoM provided in the input file.
        // A misalignment would mean that the forces and rotations are applied to the wrong point in
        // space, leading to integration issues.
        if( sqrt( (grid_min[ibody][0]+com_temp[0])*(grid_min[ibody][0]+com_temp[0])+
                  (grid_min[ibody][1]+com_temp[1])*(grid_min[ibody][1]+com_temp[1])+
                  (grid_min[ibody][2]+com_temp[2])*(grid_min[ibody][2]+com_temp[2])) > (0.5 * grid_stride[ibody])
        ){
          error->all(FLERR, "Centre of mass computed from the LS grid does not agree with that provided in the input file!");
        }

        // Overwrite inertia, could modify logic (compare or warn) if desired
        for (a = 0; a < 3; a++)
          inertia_matrix[a][a] = inertia_temp[a];

        inertia_matrix[0][1] = inertia_matrix[1][0] = inertia_temp[3];
        inertia_matrix[0][2] = inertia_matrix[2][0] = inertia_temp[4];
        inertia_matrix[1][2] = inertia_matrix[2][1] = inertia_temp[5];

        // Calculate eigen system of inertia tensor
        int ierror = MathEigen::jacobi3(inertia_matrix, inertia[ibody], evectors, 1);
        if (ierror) error->all(FLERR, "Insufficient Jacobi rotations for LS grid");

        for (a = 0; a < 3; a++) {
          ex_space[ibody][a] = evectors[a][0];
          ey_space[ibody][a] = evectors[a][1];
          ez_space[ibody][a] = evectors[a][2];
        }

        // Surface area calculation with default epsilon (diff between inner and outer) of two times grid stride.
        grid_area[ibody] = compute_surface_area(grid_size[ibody], grid_stride[ibody], temp_grid_values);

        // Normalise by number of nodes
        grid_area[ibody] /= grid_nnodes[ibody];
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
  int i,n,ibody;

  // fcm = force on center-of-mass of each rigid body

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

  MPI_Allreduce(sum[0],all[0],6*nbody,MPI_DOUBLE,MPI_SUM,world);

  for (ibody = 0; ibody < nbody; ibody++) {
    fcm[ibody][0] = all[ibody][0];
    fcm[ibody][1] = all[ibody][1];
    fcm[ibody][2] = all[ibody][2];
  }

  // torque = torque on each rigid body

  double **x = atom->x;

  for (ibody = 0; ibody < nbody; ibody++)
    for (i = 0; i < 6; i++) sum[ibody][i] = 0.0;

  // extended particles add their torque to torque of body

  if (extended) {
    double **torque_one = atom->torque;

    for (i = 0; i < nlocal; i++) {
      if (body[i] < 0) continue;
      ibody = body[i];
      if (eflags[i] & TORQUE) {
        sum[ibody][0] += torque_one[i][0];
        sum[ibody][1] += torque_one[i][1];
        sum[ibody][2] += torque_one[i][2];
      }
    }
  }

  MPI_Allreduce(sum[0],all[0],6*nbody,MPI_DOUBLE,MPI_SUM,world);

  for (ibody = 0; ibody < nbody; ibody++) {
    torque[ibody][0] = all[ibody][0];
    torque[ibody][1] = all[ibody][1];
    torque[ibody][2] = all[ibody][2];
  }

  // enforce 2d body forces and torques

  if (domain->dimension == 2) enforce2d();

  // zero langextra in case Langevin thermostat not used
  // no point to calling post_force() here since langextra
  // is only added to fcm/torque in final_integrate()

  for (ibody = 0; ibody < nbody; ibody++)
    for (i = 0; i < 6; i++) langextra[ibody][i] = 0.0;

  // virial setup before call to set_v

  v_init(vflag);

  // set velocities from angmom & omega

  for (ibody = 0; ibody < nbody; ibody++)
    MathExtra::angmom_to_omega(angmom[ibody],ex_space[ibody],ey_space[ibody],
                               ez_space[ibody],inertia[ibody],omega[ibody]);

  set_v();

  // guesstimate virial as 2x the set_v contribution

  if (vflag_global)
    for (n = 0; n < 6; n++) virial[n] *= 2.0;
  if (vflag_atom) {
    for (i = 0; i < nlocal; i++)
      for (n = 0; n < 6; n++)
        vatom[i][n] *= 2.0;
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

  if (extended) { // TODO: check and error out if particle not extended? Or no check at all (checked somewhere else, e.g., in init() ?
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

  MPI_Allreduce(sum[0],all[0],6*nbody,MPI_DOUBLE,MPI_SUM,world);

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
      fcm[ibody][0] += gvec[0]*masstotal[ibody];
      fcm[ibody][1] += gvec[1]*masstotal[ibody];
      fcm[ibody][2] += gvec[2]*masstotal[ibody];
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
  // DvdH: The names grid_scale_buf and grid_size_buf are a bit confusing as I would expected them swapped
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
    grid_stride[ibody] = grid_size_buf[0] * grid_scale[ibody];
    for (int idim = 0; idim < dim; idim++) {
      // The grid_size_buf is [stride, xmin, ymin, zmin]. The grid_scale is the scaling factor.
      grid_min[ibody][idim] = grid_size_buf[idim + 1] * grid_scale[ibody];
      // The grid_shape_buf is [nx, ny, nz]
      grid_size[ibody][idim] = (int) grid_shape_buf[idim];
    }

    if (dim == 2) {
      grid_min[ibody][2] = 0.0;
      grid_size[ibody][2] = 1;
    }

  } else {
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

double FixRigidLSDEM::compute_grid_properties(int *grid_size, double stride, double *grid_values, double *com_temp, double *inertia_temp, std::string filename)
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
        dV = smeared_heaviside_step( -ls_val / ls_ref ) * volume_cell;
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
  for (int a = 0; a < 6; a++) inertia_temp[a] = 0.0;
  for (int ind_x = 0; ind_x < grid_size[0]; ind_x++) {
    for (int ind_y = 0; ind_y < grid_size[1]; ind_y++) {
      for (int ind_z = 0; ind_z < grid_size[2]; ind_z++) {
        ls_val = grid_values[ind_x + ind_y * grid_size[0] + ind_z * grid_size[0] * grid_size[1]];
        dV = smeared_heaviside_step( -ls_val / ls_ref ) * volume_cell;
        if (dV > 0.0) {
          delx = ind_x * stride - com_temp[0];
          dely = ind_y * stride - com_temp[1];
          delz = ind_z * stride - com_temp[2];
          inertia_temp[0] += (dely * dely + delz * delz) * dV;
          inertia_temp[1] += (delx * delx + delz * delz) * dV;
          inertia_temp[2] += (delx * delx + dely * dely) * dV;
          inertia_temp[3] -= delx * dely * dV;
          inertia_temp[4] -= delx * delz * dV;
          inertia_temp[5] -= dely * delz * dV;
        }
      }
    }
  }

  // Check to see if level set has a non-inertial reference frame
  double I_diag_norm = sqrt(inertia_temp[0] * inertia_temp[0] + inertia_temp[1] * inertia_temp[1] + inertia_temp[2] * inertia_temp[2]);
  double I_off_diag_norm = sqrt(2.0 * (inertia_temp[3] * inertia_temp[3] + inertia_temp[4] * inertia_temp[4] + inertia_temp[5] * inertia_temp[5]));
  if (I_off_diag_norm / I_diag_norm > EPSILON_INERTIA)
    error->all(FLERR, "Non-inertial reference frame detected for level set in {}. Intergration of rotational motion will be wrong.", filename);

  return volume;
}

/* ----------------------------------------------------------------------
   Calculate surface area using method from ...
------------------------------------------------------------------------- */

double FixRigidLSDEM::compute_surface_area(int *grid_size, double stride, double *grid_values)
{
  // Compute the surface area as the limit of the difference in volume
	unsigned int iter, iter_max;
	double epsilon, vol_in, vol_out, area, area_old, diff;

  // First computation of area
  epsilon = stride;
  vol_in = compute_volume(grid_size, stride, grid_values, epsilon);
  vol_out = compute_volume(grid_size, stride, grid_values, -epsilon);
  if ( fabs(vol_in - vol_out) < EPSILON_ITERATION )
      error->warning(FLERR, "Inside and outside volumes are the same for surface area calculation iteration {}", iter);
  area_old = (vol_out - vol_in) / (2.0 * epsilon);
  area = area_old;

  // Iterations to improve area estimate
  iter = 0;
  iter_max = MAX_ITERATIONS;
	while (iter < iter_max) {
		epsilon *= 0.5; // Dilation measure
    vol_in = compute_volume(grid_size, stride, grid_values, -epsilon);
    vol_out = compute_volume(grid_size, stride, grid_values, epsilon);
		area = (vol_out - vol_in) / (2.0 * epsilon);
		diff = fabs( (area - area_old) / area_old );
    // Test for convergence
		if (diff < EPSILON_ITERATION) // TODO: Declare hard-coded tolerance value based on global variable?
			break;
		area_old = area;
		iter++;
	}

  // Test for convergence
	if (iter == iter_max)
    error->warning(FLERR, "Surface area calculation did not converge in {} iterations", iter);
	return area;
}

/* ----------------------------------------------------------------------
   Find the value of node (atom) i in j's LS grid.
------------------------------------------------------------------------- */

double FixRigidLSDEM::get_ls_value(int i, int j, double *normal)
{
  double **x = atom->x;
  double **grain_com = atom->darray[index_ls_dem_com];
  double **grain_quat = atom->darray[index_ls_dem_quat];

  int ibody = body[i];
  int jbody = body[j];
  double dist, nx(0.0), ny(0.0), nz(0.0);
  double stride = grid_stride[ibody];

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
  domain->minimum_image(delx, dely, delz);

  // Apply quaternion rotation to move into local reference frame of grain j grid.
  // Here, grain_quat is local->global. Therefore, grain_quat_conj is global -> local.
  double x_local[3];
  double dx[3] = {delx, dely, delz};
  double grain_quat_conj[4];
  MathExtra::qconjugate(grain_quat[j], grain_quat_conj);
  MathExtra::quatrotvec(grain_quat_conj, dx, x_local);
  // See comments above functions in math_extra.h/cpp for details

  int ncol, nrow, nslice;
  if (grid_style[ibody == DISTRIBUTED]) {
    // Translate local coordinates such that they are relative
    // to the lower corner of the level set grid.
    double **local_grid_min = atom->darray[index_grid_min];
    x_local[0] -= local_grid_min[j][0];
    x_local[1] -= local_grid_min[j][1];
    x_local[2] -= local_grid_min[j][2];
    // This local_grid_min should be the minimum of the node's grid in the distributed case,
    // while of the global grid in the global / non-distributed case.

    ncol = subgrid_size[0]; // Danny: this should be for the node's grid as well
    nrow = subgrid_size[1];
    nslice = subgrid_size[2];
  } else {
    x_local[0] -= grid_min[jbody][0];
    x_local[1] -= grid_min[jbody][1];
    x_local[2] -= grid_min[jbody][2];

    ncol = grid_size[ibody][0];
    nrow = grid_size[ibody][1];
    nslice = grid_size[ibody][2];
  }

  // Normalise the coordinates to be in units of the number of grid cells.
  double x_red = x_local[0] / stride;
  double y_red = x_local[1] / stride;
  double z_red = x_local[2] / stride;

  // Calculate index from relative coordinate, being careful with integer division.
  int ind_x = int(x_red);
  int ind_y = int(y_red);
  int ind_z = int(z_red); // Should always be zero in 2D.

  // JBC: There is some padding for detection / normal caculation that I don't understand clearly
  // Danny: Does the below clarify? Or is there something else that is missing?
  // Checking whether x_local lies within the grid. Avoids edge cases where finite precision
  // leads to e.g. a x=-0.1 coordinate to fall outside of a grid that starts at x=-0.1.
  if ((ind_x < 0 || ind_x >= (nrow - 1)) || (ind_y < 0 || ind_y >= (ncol - 1)) ||
      ((domain->dimension == 3) && (ind_z < 0 || ind_z >= (nslice - 1))))
    //error->one(FLERR, "Contacting node {} is outside of node {}'s LS grid", atom->tag[i], atom->tag[j]);
    return BIG;

  // The normalised coordinates within the current grid cell.
  // May be safer to cap them with math::max(math::min(x_red, 1.0), 0.0)
  x_red = x_red - static_cast<double>(ind_x);
  y_red = y_red - static_cast<double>(ind_y);
  z_red = z_red - static_cast<double>(ind_z); // Should always be zero in 2D.

  //  Interpolate
  int my_index = ind_x + ind_y * ncol + ind_z * ncol * nrow;
  if (grid_style[ibody] == DISTRIBUTED) {
    double **node_local_grid = atom->darray[index_grid_values];

    // Level-set values on the grid points in the lower z plane (ind_z)
    double ls000 = node_local_grid[j][my_index];
    double ls100 = node_local_grid[j][my_index + 1];
    double ls010 = node_local_grid[j][my_index + ncol];
    double ls110 = node_local_grid[j][my_index + 1 + ncol];

    // Bi-linear interpolation in the lower z plane (ind_z)
    double lsxy0 = ls000 + y_red * (ls010 - ls000) +
                   x_red * (ls100 - ls000 + y_red * (ls110 - ls100 - ls010 + ls000));

    if (domain->dimension == 3) { // 3D
      // Level-set values on the grid points in the upper z plane (ind_z+1)
      double ls001 = node_local_grid[j][my_index + ncol * nrow];
      double ls101 = node_local_grid[j][my_index + 1 + ncol * nrow];
      double ls011 = node_local_grid[j][my_index + ncol + ncol * nrow];
      double ls111 = node_local_grid[j][my_index + 1 + ncol + ncol * nrow];

      // Bi-linear interpolation in the upper z plane (ind_z+1)
      double lsxy1 = ls001 + y_red * (ls011 - ls001) +
                     x_red * (ls101 - ls001 + y_red * (ls111 - ls101 - ls011 + ls001));

      // Affecting tri-linear interpolation by linear interpolation of the two bi-linear interpolations.
      dist = z_red * (lsxy1 - lsxy0) + lsxy0;

      // Computing normal as the gradient of trilinear interpolation
      for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
          for (int c = 0; c < 2; c++) {
            double lsVal = node_local_grid[j][(ind_x + a) + (ind_y + b) * ncol + (ind_z + c) * ncol * nrow];
            nx += lsVal * (2 * a - 1) * ((1 - b) * (1 - y_red) + b * y_red) * ((1 - c) * (1 - z_red) + c * z_red);
            ny += lsVal * (2 * b - 1) * ((1 - a) * (1 - x_red) + a * x_red) * ((1 - c) * (1 - z_red) + c * z_red);
            nz += lsVal * (2 * c - 1) * ((1 - a) * (1 - x_red) + a * x_red) * ((1 - b) * (1 - y_red) + b * y_red);
          }
        }
      }
    } else { // 2D
      // Bi-linear interpolation
      dist = lsxy0;
      // Computing normal as the gradient of bilinear interpolation
      for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
          double lsVal = node_local_grid[j][(ind_x + a) + (ind_y + b) * ncol];
          nx += lsVal * (2 * a - 1) * ((1 - b) * (1 - y_red) + b * y_red);
          ny += lsVal * (2 * b - 1) * ((1 - a) * (1 - x_red) + a * x_red);
        }
      }
      nz = 0.0;
    }
  } else {
    int my_index = grid_index[jbody];
    double *my_grid = global_grids[my_index];

    // Level-set values on the grid points in the lower z plane (ind_z)
    double ls000 = my_grid[ind_x     + ind_y       * ncol + ind_z * ncol * nrow]; // todo: maybe update indices like above
    double ls100 = my_grid[ind_x + 1 + ind_y       * ncol + ind_z * ncol * nrow];
    double ls010 = my_grid[ind_x     + (ind_y + 1) * ncol + ind_z * ncol * nrow];
    double ls110 = my_grid[ind_x + 1 + (ind_y + 1) * ncol + ind_z * ncol * nrow];

    // Bi-linear interpolation in the lower z plane (ind_z)
    double lsxy0 = ls000 + y_red * (ls010 - ls000) +
                   x_red * (ls100 - ls000 + y_red * (ls110 - ls100 - ls010 + ls000));

    if (domain->dimension == 3) { // 3D
      // Level-set values on the grid points in the upper z plane (ind_z+1)
      double ls001 = my_grid[ind_x     + ind_y       * ncol + (ind_z + 1) * ncol * nrow];
      double ls101 = my_grid[ind_x + 1 + ind_y       * ncol + (ind_z + 1) * ncol * nrow];
      double ls011 = my_grid[ind_x     + (ind_y + 1) * ncol + (ind_z + 1) * ncol * nrow];
      double ls111 = my_grid[ind_x + 1 + (ind_y + 1) * ncol + (ind_z + 1) * ncol * nrow];

      // Bi-linear interpolation in the upper z plane (ind_z+1)
      double lsxy1 = ls001 + y_red * (ls011 - ls001) +
                     x_red * (ls101 - ls001 + y_red * (ls111 - ls101 - ls011 + ls001));

      // Affecting tri-linear interpolation by linear interpolation of the two bi-linear interpolations.
      dist = z_red * (lsxy1 - lsxy0) + lsxy0;

      // Computing normal as the gradient of trilinear interpolation
      for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
          for (int c = 0; c < 2; c++) {
            double lsVal = my_grid[(ind_x + a) + (ind_y + b) * ncol + (ind_z + c) * ncol * nrow];
            nx += lsVal * (2 * a - 1) * ((1 - b) * (1 - y_red) + b * y_red) * ((1 - c) * (1 - z_red) + c * z_red);
            ny += lsVal * (2 * b - 1) * ((1 - a) * (1 - x_red) + a * x_red) * ((1 - c) * (1 - z_red) + c * z_red);
            nz += lsVal * (2 * c - 1) * ((1 - a) * (1 - x_red) + a * x_red) * ((1 - b) * (1 - y_red) + b * y_red);
          }
        }
      }
    } else { // 2D
      // Bi-linear interpolation
      dist = lsxy0;
      // Computing normal as the gradient of bilinear interpolation
      for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
          double lsVal = my_grid[(ind_x + a) + (ind_y + b) * ncol];
          nx += lsVal * (2 * a - 1) * ((1 - b) * (1 - y_red) + b * y_red);
          ny += lsVal * (2 * b - 1) * ((1 - a) * (1 - x_red) + a * x_red);
        }
      }
      nz = 0.0;
    }
  }

  // Assign normal
  normal[0] = nx;
  normal[1] = ny;
  normal[2] = nz;

  // Rotate normal back to global coordinates
  MathExtra::quatrotvec(grain_quat[j], normal, normal);

  //if (-dist > warncut) maybe warn that you are about to penetrate too far

  return dist;
}
