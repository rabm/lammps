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

#include "update.h"

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
static constexpr int MAX_ITERATIONS = 100; // For surface area integration
static constexpr int RECOMMENDED_MAX_NGRID = 1000; // For local node grid, 10x10x10

/* ---------------------------------------------------------------------- */

FixRigidLSDEM::FixRigidLSDEM(LAMMPS *lmp, int narg, char **arg) :
    FixRigid(lmp, narg, arg), id_fix(nullptr), id_fix2(nullptr), global_grids(nullptr),
    grid_style(nullptr), grid_min(nullptr), grid_stride(nullptr), grid_scale(nullptr),
    grid_index(nullptr), grid_size(nullptr), grid_vol(nullptr), node_area(nullptr),
    grid_nnodes(nullptr), quatd2g(nullptr)
{
  comm_forward = 1;
  maxcut = -1;
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
  memory->create(quatd2g, nbody, 4, "rigid/ls/dem:quatd2g");

  if (langflag)
    error->all(FLERR, "Langevin thermostat not supported with fix rigid/ls/dem");
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
  memory->destroy(quatd2g);

  // delete global memory data

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
    "{} all property/atom d2_ls_dem_n 3 d2_ls_dem_fs 3 i_ls_dem_touch_id d_ls_dem_fn1 d_ls_dem_fs1 ghost yes writedata no",
     id_fix));
  int tmp1, tmp2;
  index_ls_dem_touch_id = atom->find_custom("ls_dem_touch_id", tmp1, tmp2);
}

/* ---------------------------------------------------------------------- */

void FixRigidLSDEM::init()
{
  FixRigid::init();

  if (!atom->xcom_flag || !atom->omega_flag || !atom->quat_flag  || !atom->grid_index_flag)
    error->all(FLERR, "Pair ls/dem requires atom style ls/dem");

  int iatom, ibody, i, a;
  int dimension = domain->dimension;

  // Pair cutoff sets size of LS around nodes for distributed case
  if (!utils::strmatch(force->pair_style,"^ls/dem"))
    error->all(FLERR, "Must use pair ls/dem with fix rigid/ls/dem");
  auto pair = dynamic_cast<PairLSDEM *>(force->pair);
  maxcut = pair->maxcut;

  int index_global = 0;
  int *touch_id = atom->ivector[index_ls_dem_touch_id];
  if (!stored_flag) {
    stored_flag = 1;

    for (i = 0; i < atom->nlocal; i++)
      touch_id[i] = -1; // set to zero for preexisting atoms (rest set in set_array)

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
      read_gridfile(ibody, 0, filename, grid_size, nullptr); // Get only grid sizes (which 0)
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

    for (ibody = 0; ibody < nbody; ibody++)
      grid_nnodes[ibody] = 0;
    for (i = 0; i < atom->nlocal; i++)
      grid_nnodes[body[i]] += 1;

    if (distributed_flag) {
      for (a = 0; a < 3; a++) subgrid_size[a] = 2 * rcell + 1; // try remove +1 and cast to int
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

    double *temp_grid_values;
    memory->create(temp_grid_values, max_grid_size_flat, "rigid/ls/dem:temp_grid_values");

    double **grid_values, **grid_min_local;
    if (distributed_flag) {
      grid_values = atom->darray[index_grid_values];
      grid_min_local = atom->darray[index_grid_min];
    }

    double **x = atom->x;
    double **quat_atom = atom->quat;

    int need_distributed, need_global, need_padding, nx, ny, nz, ix_node, iy_node, iz_node;
    int ix_global, iy_global, iz_global, index_global, index_local, index_grid_min_local[3];
    double temp[3], com_temp[3], quat_conj[4], inertia_temp[3][3], evectors[3][3], cross[3];
    double delx, dely, delz, area, density, scale, scale2, scale3;
    for (const auto& pair : file_map) { // Loop over all <filename, [bodyIDs]>
      filename = pair.first;
      read_gridfile(-1, 1, filename, nullptr, temp_grid_values);

      // Compute grain properties per unique grid
      for (ibody = 0; ibody < nbody; ibody++) {
        if (pair.second.find(ibody) == pair.second.end())
          continue;

        // Compute properties from the level-set grid
        grid_vol[ibody] = compute_grid_properties(grid_size[ibody], grid_stride[ibody], temp_grid_values, com_temp, inertia_temp, dimension);
        if (grid_vol[ibody] < 0)
          error->all(FLERR, "Non-inertial reference frame detected for level set in {}, integration of rotational motion will be wrong", filename);

        // Comparing if CoM in level-set grid is indeed aligned with CoM
        // Misalignment would cause forces/rotations to be applied to the wrong point in space
        MathExtra::add3(grid_min[ibody], com_temp, temp);
        if (MathExtra::len3(temp) > (0.5 * grid_stride[ibody])) {
          error->all(FLERR, "Centre of mass computed from the LS grid does not agree with that provided in the input grid file! Grid min given at {} {} {} and CoM computed at {} {} {}.",
            grid_min[ibody][0], grid_min[ibody][1], grid_min[ibody][2], com_temp[0], com_temp[1], com_temp[2]);
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

        // copy of calculations from FixRigid::setup_bodies_static()
        // for 2d, ensure that evector along z axis is last
        // necessary so that quaternion is a simple rotation around +z axis
        //   or a 180 degree rotation for a -z axis
        // otherwise richardson() method for a body with a tiny evalue (near-linear)
        //  may not preserve the correct z-aligned quat and associated evectors
        //  over time due to round-off accumulation

        if (domain->dimension == 2) {
          if (fabs(ez_space[ibody][0]) > EPSILON || fabs(ez_space[ibody][1]) > EPSILON) {
            std::swap(inertia[ibody][1],inertia[ibody][2]);
            std::swap(ey_space[ibody][0],ez_space[ibody][0]);
            std::swap(ey_space[ibody][1],ez_space[ibody][1]);
            std::swap(ey_space[ibody][2],ez_space[ibody][2]);
          }
        }

        // if any principal moment < scaled EPSILON, set to 0.0

        double max;
        max = MAX(inertia[ibody][0],inertia[ibody][1]);
        max = MAX(max,inertia[ibody][2]);

        if (inertia[ibody][0] < EPSILON*max) inertia[ibody][0] = 0.0;
        if (inertia[ibody][1] < EPSILON*max) inertia[ibody][1] = 0.0;
        if (inertia[ibody][2] < EPSILON*max) inertia[ibody][2] = 0.0;

        // enforce 3 evectors as a right-handed coordinate system
        // flip 3rd vector if needed

        MathExtra::cross3(ex_space[ibody],ey_space[ibody],cross);
        if (MathExtra::dot3(cross,ez_space[ibody]) < 0.0)
          MathExtra::negate3(ez_space[ibody]);

        // create initial quaternion relative to inertial frame

        MathExtra::exyz_to_q(ex_space[ibody],ey_space[ibody],ez_space[ibody],
                         quat[ibody]);

        // additionally, calculate relative rotation from inerital frame to LS grid
        //   assume any additional rotations on grains (e.g. by displace_atoms)
        //   were performed correctly s.t. all atoms have equivalent initial quaterions
        // Note: do not do something similar for CoM b/c there is no way to save
        //   atom coordinates before being shifted in read_data. Also, this can be
        //   achieved easily by just setting the shift in the infile.
        for (iatom = 0; iatom < atom->nlocal; iatom++)
          if (body[iatom] == ibody) break;

        MathExtra::qconjugate(quat[ibody], quat_conj);
        MathExtra::quatquat(quat_conj, quat_atom[iatom], quatd2g[ibody]);

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
        scale2 = scale * scale;
        scale3 = scale * scale2;
        density = masstotal[ibody] / grid_vol[ibody];
        grid_stride[ibody] *= scale;
        MathExtra::scale3(scale, grid_min[ibody]);
        if (dimension == 3) {
          node_area[ibody] *= scale2;
          grid_vol[ibody] *= scale3;
          MathExtra::scale3(density * scale2 * scale3, inertia[ibody]);
        } else {
          node_area[ibody] *= scale;
          grid_vol[ibody] *= scale2;
          MathExtra::scale3(density * scale2 * scale2, inertia[ibody]);
        }
      }

      // Start handling memory of LS grid

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

          // Location of atom/node relative to CoM
          double dx[3], dx_local[3];
          dx[0] = x[i][0] - xcm[ibody][0];
          dx[1] = x[i][1] - xcm[ibody][1];
          dx[2] = x[i][2] - xcm[ibody][2];

          // Account for PBCs
          domain->minimum_image(FLERR, delx, dely, delz);

          // Rotate to LS frame (for now, just the atomic quaternion)
          double quat_conj[4];
          MathExtra::qconjugate(quat_atom[i], quat_conj);
          MathExtra::quatrotvec(quat_conj, dx, dx_local);

          // Location of atom/node relative to entire grain grid minimum.
          dx_local[0] -= grid_min[ibody][0];
          dx_local[1] -= grid_min[ibody][1];
          dx_local[2] -= grid_min[ibody][2];

          // Index of atom/node in entire grain grid.
          double stride = grid_stride[ibody];
          ix_node = int(dx_local[0] / stride);
          iy_node = int(dx_local[1] / stride);
          iz_node = int(dx_local[2] / stride);

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
                  if (index_global < 0 || index_global >= nx * ny * nz)
                    error->one(FLERR, "Unexpected out of bounds error in distributed level set creation, indices {} {} {}", ix_global, iy_global, iz_global);
                  grid_values[i][index_local] = temp_grid_values[index_global] * grid_scale[ibody];
                }
              }
            }
          }

          if (need_padding)
            error->warning(FLERR, "Level set of body {} does not include a large enough buffer for the distributed grid cutoff on atom {}. Local grid padded with BIG values", ibody, atom->tag[i]);
        }
      }
    }

    memory->destroy(gridfiles);
    memory->destroy(temp_grid_values);
    memory->destroy(ntotal_global);

    // Redefine displace - initial atom coords in basis of principal axes - with new inertia/exspace values
    //   copy of calculations from FixRigid::setup_bodies_static()

    int *periodicity = domain->periodicity;
    double xprd = domain->xprd;
    double yprd = domain->yprd;
    double zprd = domain->zprd;
    double xy = domain->xy;
    double xz = domain->xz;
    double yz = domain->yz;
    double delta[3];
    int xbox,ybox,zbox;
    double xunwrap,yunwrap,zunwrap;

    for (i = 0; i < atom->nlocal; i++) {
      if (body[i] < 0)  continue;

      ibody = body[i];

      xbox = (xcmimage[i] & IMGMASK) - IMGMAX;
      ybox = (xcmimage[i] >> IMGBITS & IMGMASK) - IMGMAX;
      zbox = (xcmimage[i] >> IMG2BITS) - IMGMAX;

      if (triclinic == 0) {
        xunwrap = x[i][0] + xbox*xprd;
        yunwrap = x[i][1] + ybox*yprd;
        zunwrap = x[i][2] + zbox*zprd;
      } else {
        xunwrap = x[i][0] + xbox*xprd + ybox*xy + zbox*xz;
        yunwrap = x[i][1] + ybox*yprd + zbox*yz;
        zunwrap = x[i][2] + zbox*zprd;
      }

      delta[0] = xunwrap - xcm[ibody][0];
      delta[1] = yunwrap - xcm[ibody][1];
      delta[2] = zunwrap - xcm[ibody][2];
      MathExtra::transpose_matvec(ex_space[ibody],ey_space[ibody],
                                  ez_space[ibody],delta,displace[i]);
    }
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

  double **grain_com = atom->xcom;
  double **grain_quat = atom->quat;
  double **grain_omega = atom->omega;

  int ibody;
  for (int i = 0; i < atom->nlocal; i++) {
    ibody = body[i];

    grain_com[i][0] = xcm[ibody][0];
    grain_com[i][1] = xcm[ibody][1];
    grain_com[i][2] = xcm[ibody][2];

    // calculate rotation from current orientation to LS grid
    MathExtra::quatquat(quat[ibody], quatd2g[ibody], grain_quat[i]);

    grain_omega[i][0] = omega[ibody][0];
    grain_omega[i][1] = omega[ibody][1];
    grain_omega[i][2] = omega[ibody][2];
  }
}

/* ---------------------------------------------------------------------- */

int FixRigidLSDEM::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(body[j]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixRigidLSDEM::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++)
    body[i] = (int) ubuf(buf[m++]).i;
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
  int i, ibody;

  // sum over atoms to get force and torque on rigid body

  double **f = atom->f;
  double **torque_one = atom->torque;
  int nlocal = atom->nlocal;

  for (ibody = 0; ibody < nbody; ibody++)
    for (i = 0; i < 6; i++) sum[ibody][i] = 0.0;

  // all particles add forces/torques to body

  for (i = 0; i < nlocal; i++) {
    if (body[i] < 0) continue;
    ibody = body[i];

    sum[ibody][0] += f[i][0];
    sum[ibody][1] += f[i][1];
    sum[ibody][2] += f[i][2];
    sum[ibody][3] += torque_one[i][0];
    sum[ibody][4] += torque_one[i][1];
    sum[ibody][5] += torque_one[i][2];
  }

  MPI_Allreduce(sum[0], all[0], 6 * nbody, MPI_DOUBLE, MPI_SUM, world);

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
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixRigidLSDEM::set_arrays(int i)
{
  FixRigid::set_arrays(i);
  atom->ivector[index_ls_dem_touch_id][i] = -1;
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
   one-time reading of file names for LS grid
------------------------------------------------------------------------- */

void FixRigidLSDEM::read_gridfile_names(char **gridfiles)
{
  tagint id;
  int nchunk, eofflag, nlines;
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
        id = values.next_tagint();
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
   read per rigid body level-set grid values from user-provided file
   files gridfiles to read from stored previously by readfile() function
   first line = grid_sizex grid_sizey grid_sizez
   followed by grid_sizex * grid_sizey * grid_sizez lines of level set values at the grid points
   which = 0, read only the size of the level-set grid
   which = 1, read the values of the level-set grid
------------------------------------------------------------------------- */

void FixRigidLSDEM::read_gridfile(int ibody, int which, std::string filename, int **grid_size, double *grid_values)
{
  int dim = domain->dimension;
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
   Find the value of node (atom) i in j's LS grid.
------------------------------------------------------------------------- */

double FixRigidLSDEM::get_ls_value(int i, int j, double *normal)
{
  double **x = atom->x;
  double **grain_com = atom->xcom;
  double **grain_quat = atom->quat;

  int jbody = body[j];
  double jstride = grid_stride[jbody];
  double strideinv = 1.0 / jstride;

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

  int dim = domain->dimension;
  double dist = interpolate_LS(dim, mygrid, ncol, nrow, nslice, x_red, y_red, z_red, normal, jstride);

  // Grain-stored grid values are shared and un-scaled, so apply scaling
  if (grid_style[jbody] == GLOBAL) dist *= grid_scale[jbody];

  // Rotate normal back to global coordinates
  MathExtra::quatrotvec(grain_quat[j], normal, normal);

  return dist;
}
