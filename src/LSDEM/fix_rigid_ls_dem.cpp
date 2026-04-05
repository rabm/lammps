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
    quatd2g(nullptr), gridfiles(nullptr), quat_custom(nullptr)
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
  memory->create(quatd2g, nbody, 4, "rigid/ls/dem:quatd2g");
  memory->create(gridfiles, nbody, MAXLINE, "rigid/ls/dem:gridfiles");

  if (langflag)
    error->all(FLERR, "Langevin thermostat not supported with fix rigid/ls/dem");

  // only call FixRigidSmall::setup_bodies_static() once
  reinitflag = 0;
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
  memory->destroy(quatd2g);
  memory->destroy(gridfiles);

  memory->destroy(itensor_custom);
  memory->destroy(quat_custom);

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

void FixRigidLSDEM::init()
{
  if (!atom->xcom_flag || !atom->omega_flag || !atom->quat_flag  || !atom->grid_index_flag)
    error->all(FLERR, "Pair ls/dem requires atom style ls/dem");

  // Pair cutoff sets size of LS around nodes for distributed case
  if (!utils::strmatch(force->pair_style,"^ls/dem"))
    error->all(FLERR, "Must use pair ls/dem with fix rigid/ls/dem");
  auto pair = dynamic_cast<PairLSDEM *>(force->pair);
  maxcut = pair->maxcut;

  if (stored_flag) return;
  stored_flag = 1;

  // allocate storage for LS-derived quantities (used in FixRigid::init())

  memory->create(itensor_custom, nbody, 6, "rigid:itensor_custom");
  memory->create(quat_custom, nbody, 4, "rigid:quat_custom");

  int iatom, ibody, i, a;
  int dimension = domain->dimension;
  int index_global_grid = 0;
  int *touch_id = atom->ivector[index_ls_dem_touch_id];
  int *mask = atom->mask;

  for (i = 0; i < atom->nlocal; i++)
    if (mask[i] & groupbit)
      touch_id[i] = -1; // set to zero for preexisting atoms (rest set in set_array)

  int *ntotal_global;
  memory->create(ntotal_global, nbody, "rigid/ls/dem:ntotal_global");
  int read_quat = read_infile(gridfiles);

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
        grid_index[ibody] = index_global_grid;
        ntotal_global[index_global_grid] = grid_size_flat;
        index_global_grid += 1;
      }
    } else {
      distributed_flag = 1;
    }
  }

  // ------------------------------ //
  // Allocate memory for level sets //
  // ------------------------------ //

  int ntotal;
  // All local grids sized on finest grid (fix property/atom requiresfixed-size containers)
  rcell = maxcut / min_stride + 2; // +1 for interpolation +1 for safety

  if (distributed_flag) {
    for (a = 0; a < 3; a++) subgrid_size[a] = 2 * rcell + 1; // try remove +1 and cast to int
    if (dimension == 2) subgrid_size[2] = 1;

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

  if (index_global_grid) {
    memory->create_ragged(global_grids, index_global_grid, ntotal_global, "rigid/ls/dem:global_grids");
  }

  // ------------------------------ //
  // Read and store level sets      //
  // ------------------------------ //

  double *temp_grid_values;
  memory->create(temp_grid_values, max_grid_size_flat, "rigid/lsdem:temp_grid_values");

  double **grid_values, **grid_min_local;
  if (distributed_flag) {
    grid_values = atom->darray[index_grid_values];
    grid_min_local = atom->darray[index_grid_min];
  }

  double **x = atom->x;
  double **quat_atom = atom->quat;

  double dx[3];
  int need_distributed, need_global, error_code, error_code_global, index_global;
  for (const auto& pair : file_map) { // Loop over <filename, [bodyIDs]>
    filename = pair.first;
    read_gridfile(-1, 1, filename, nullptr, temp_grid_values);

    // Compute grain properties (volume, area, inertia...) for each body using this grid
    for (ibody = 0; ibody < nbody; ibody++) {
      if (pair.second.find(ibody) == pair.second.end())
        continue;
      compute_grain_properties(ibody, temp_grid_values, filename);
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
        if (!(mask[i] & groupbit)) continue;
        ibody = body[i];

        if (pair.second.find(ibody) == pair.second.end())
          continue; // Ideally would have list of all atoms in a rigid body... not sure if exists...

        // Location of atom/node relative to CoM + remap periodically
        MathExtra::sub3(x[i], xcm[ibody], dx);
        domain->minimum_image(FLERR, dx[0], dx[1], dx[2]);

        // Calculate local grid values and minima
        error_code = store_distributed(i, dimension, grid_size[ibody], subgrid_size, grid_stride[ibody], grid_scale[ibody],
                                      rcell, dx, grid_min[ibody], quat_atom[i], temp_grid_values, grid_min_local[i], grid_values[i]);

        if (error_code == -1)
          error->one(FLERR, "Unexpected out of bounds error in distributed level set creation, atom {}", atom->tag[i]);

        MPI_Allreduce(&error_code, &error_code_global, 1, MPI_INT, MPI_MAX, world);
        if (error_code_global && comm->me == 0)
          error->warning(FLERR, "Level set of body {} does not include a large enough buffer for the distributed grid cutoff on "
            "atom {}\nyLocal grid padded with BIG values\nWarning will not print for other nodes in this body.", ibody, atom->tag[i]);
      }
    }
  }

  memory->destroy(temp_grid_values);
  memory->destroy(ntotal_global);

  FixRigid::init();


  // extra calculations using values from parent

  double quat_conj[4];
  for (int ibody = 0; ibody < nbody; ibody++) {

    // calculate relative rotation from inerital frame to LS grid
    //   assume all atoms in body have equivalent initial quaterions
    //   (user could incorrectly use diplace_atoms on subset)
    for (iatom = 0; iatom < atom->nlocal; iatom++) {
      if (!(mask[iatom] & groupbit)) continue;
      if (body[iatom] == ibody) break;
    }

    MathExtra::qconjugate(quat[ibody], quat_conj);
    if (read_quat) {
      MathExtra::qconjugate(quat[ibody], quat_conj);
      MathExtra::quatquat(quat_conj, quat_custom[ibody], quatd2g[ibody]);
    } else {
      MathExtra::qconjugate(quat[ibody], quatd2g[ibody]);
    }
  }

  memory->destroy(itensor_custom);
  memory->destroy(quat_custom);
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

  int *mask = atom->mask;
  double **grain_com = atom->xcom;
  double **grain_quat = atom->quat;
  double **grain_omega = atom->omega;

  int ibody;
  for (int i = 0; i < atom->nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
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

  int *mask = atom->mask;
  double **f = atom->f;
  double **torque_one = atom->torque;
  int nlocal = atom->nlocal;

  for (ibody = 0; ibody < nbody; ibody++)
    for (i = 0; i < 6; i++) sum[ibody][i] = 0.0;

  // all particles add forces/torques to body

  for (i = 0; i < nlocal; i++) {
    if (body[i] < 0) continue;
    if (!(mask[i] & groupbit)) continue;
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


  auto outfile = std::string(file) + ".rigid";
  FILE *fp = fopen(outfile.c_str(),"w");
  if (fp == nullptr)
    error->one(FLERR,"Cannot open fix rigid restart file {}: {}",outfile,utils::getsyserror());

  utils::print(fp,"# fix rigid mass, COM, inertia tensor info for {} bodies on timestep {}\n\n",nbody,update->ntimestep);
  utils::print(fp,"{}\n",nbody);

  // compute I tensor against xyz axes from diagonalized I and current quat
  // Ispace = P Idiag P_transpose
  // P is stored column-wise in exyz_space

  int xbox,ybox,zbox;
  double p[3][3],pdiag[3][3],ispace[3][3];

  int id;
  for (int i = 0; i < nbody; i++) {
    if (rstyle == SINGLE || rstyle == GROUP) id = i+1;
    else id = body2mol[i];

    MathExtra::col2mat(ex_space[i],ey_space[i],ez_space[i],p);
    MathExtra::times3_diag(p,inertia[i],pdiag);
    MathExtra::times3_transpose(pdiag,p,ispace);

    xbox = (imagebody[i] & IMGMASK) - IMGMAX;
    ybox = (imagebody[i] >> IMGBITS & IMGMASK) - IMGMAX;
    zbox = (imagebody[i] >> IMG2BITS) - IMGMAX;

    fprintf(fp,"%d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e "
            "%-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d "
            "%d %-1.16e %s %-1.16e %-1.16e %-1.16e %-1.16e\n",
            id,masstotal[i],xcm[i][0],xcm[i][1],xcm[i][2],ispace[0][0],ispace[1][1],ispace[2][2],
            ispace[0][1],ispace[0][2],ispace[1][2],vcm[i][0],vcm[i][1],vcm[i][2],
            angmom[i][0],angmom[i][1],angmom[i][2],xbox,ybox,zbox,
            grid_style[i], grid_scale[i], gridfiles[i],
            quat[i][0], quat[i][1], quat[i][2], quat[i][3]);
  }

  fclose(fp);
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

int FixRigidLSDEM::read_infile(char **gridfiles)
{
  tagint id;
  int nchunk, eofflag, nlines, read_quat;
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

  if (nlines == 0) return 0;
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

    read_quat = 0;
    if (nwords == (ATTRIBUTE_PERBODY + n_extra_attributes)) {
      read_quat = 0;
    } else if (nwords == (ATTRIBUTE_PERBODY + n_extra_attributes + 4)) {
      read_quat = 1;
    } else {
      error->all(FLERR, "Incorrect rigid body format in fix rigid/ls/dem file");
    }

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

        // need early to calculate LS properties
        masstotal[id] = values.next_double();
        xcm[id][0] = values.next_double();
        xcm[id][1] = values.next_double();
        xcm[id][2] = values.next_double();

        values.skip(15);

        grid_style[id] = values.next_int();
        if (grid_style[id] != DISTRIBUTED && grid_style[id] != GLOBAL)
          throw TokenizerException("invalid_rigid memory model ", std::to_string(grid_style[id]));

        grid_scale[id] = values.next_double();
        if (grid_scale[id] <= 0)
          error->one(FLERR, "Invalid scaling factor {}", grid_scale[id]);

        strcpy(gridfiles[id], values.next_string().data());
        if (read_quat) {
          quat_custom[id][0] = values.next_double();
          quat_custom[id][1] = values.next_double();
          quat_custom[id][2] = values.next_double();
          quat_custom[id][3] = values.next_double();
        }
      } catch (TokenizerException &e) {
        error->all(FLERR, "Invalid fix rigid/ls/dem infile: {}", e.what());
      }
      buf = next + 1;
    }
    nread += nchunk;
  }

  if (comm->me == 0) fclose(fp);
  delete[] buffer;

  return read_quat;
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
   Using a full level set, calculate properites of grain
------------------------------------------------------------------------- */

void FixRigidLSDEM::compute_grain_properties(int ibody, double *grid_values, std::string filename)
{
  int dimension = domain->dimension;
  double com_temp[3];

  grid_vol[ibody] = compute_grid_properties(grid_size[ibody], grid_stride[ibody], grid_values, com_temp, itensor_custom[ibody], dimension);

  if (grid_vol[ibody] < 0)
    error->all(FLERR, "Non-inertial reference frame for level set in {}", filename);

  // Check CoM misalignment, would apply forces/rotations at incorrect positions
  double sum[3];
  MathExtra::add3(grid_min[ibody], com_temp, sum);
  if (MathExtra::len3(sum) > (0.5 * grid_stride[ibody])) {
    error->all(FLERR, "Centre of mass computed from LS grid does not agree with provided value, grid min at {} {} {} and CoM computed at {} {} {}",
      grid_min[ibody][0], grid_min[ibody][1], grid_min[ibody][2], com_temp[0], com_temp[1], com_temp[2]);
  }

  // Surface area calculation
  //   default epsilon (diff between inner and outer) = 2x grid stride.

  double area = compute_surface_area(dimension, grid_size[ibody], grid_stride[ibody], grid_values);
  // Test for physical realism
  if (!((area > 0.0) && std::isfinite(area)))
    error->all(FLERR, "Invalid surface area calculated {}", area);
  node_area[ibody] = area;

  // Normalise by number of nodes
  node_area[ibody] /= nrigid[ibody];

  // Scale relevant quantities by grain size

  double scale = grid_scale[ibody];
  double scale2 = scale * scale;
  double scale3 = scale * scale2;
  double density = masstotal[ibody] / grid_vol[ibody];
  grid_stride[ibody] *= scale;
  MathExtra::scale3(scale, grid_min[ibody]);
  if (dimension == 3) {
    node_area[ibody] *= scale2;
    grid_vol[ibody] *= scale3;
    for (int a = 0; a < 6; a++)
      itensor_custom[ibody][a] *= density * scale2 * scale3;
  } else {
    node_area[ibody] *= scale;
    grid_vol[ibody] *= scale2;
    for (int a = 0; a < 6; a++)
      itensor_custom[ibody][a] *= density * scale2 * scale2;
  }
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
