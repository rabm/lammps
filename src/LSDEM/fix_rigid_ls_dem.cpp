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

static constexpr double EPSILON_INERTIA = 1e-7;

inline double FixRigidLSDEM::smeared_heaviside_step(double x)
{
  // A function that smoothly transition from 0 to 1 when x goes from -1 to 1.
  // For x < -1, the function should be 0. For x > 1, the function should be 1.
  // This is not implemented here, and up to the user. See Kawamoto et al. (2016).
  return 0.5 * (1.0 + x + sin(MY_PI * x) / MY_PI);
}

//inline double FixRigidLSDEM::compute_volume()


Real LevelSet::volumeInsideThreshold(Real epsilon) const
{
	if (smearCoeff <= 0)
		LOG_WARN("Using volumeInsideThreshold (for surface measurement, probably) with a negative smearCoeff = " << smearCoeff << " is not expected");
	Real vol(0.);                                                // to-be-returned volume which is inside phi = epsilon
	Real phiRef(0.5 * sqrt(3.0) * lsGrid->spacing / smearCoeff); // the reference length for smoothing the Heaviside
	Real volCell(pow(lsGrid->spacing, 3));                       // lsGrid voxel volume
	for (int xIndex = 0; xIndex < lsGrid->nGP[0]; xIndex++) {
		for (int yIndex = 0; yIndex < lsGrid->nGP[1]; yIndex++) {
			for (int zIndex = 0; zIndex < lsGrid->nGP[2]; zIndex++) {
				vol += smearedHeaviside((epsilon - distField[xIndex][yIndex][zIndex]) / phiRef) * volCell;
			}
		}
	}
	return vol;
}

// Volume integration

  // This is the reference distance values that determines the smearing with of
  // the Heaviside step function. Current expression is the half-diagional of the
  // grid cell divided by a smearing constant.
  double smearCoeff = 1.5;
  double ls_ref = 0.0;
  if (smearCoeff != 0)
    ls_ref = sqrt(0.75) * stride / smearCoeff;

  // Initialise volume and centre of mass
  double volume = 0.0, x_com = 0.0, y_com = 0.0, z_com = 0.0;
  // Cell volume, temporary grid points, integration volume.
  double volume_cell = stride * stride;
  if (domain->dimension == 3) volume_cell *= stride;

  // Integration
  double dV, ls_val;
  for (int ind_x = 0; ind_x < grid_size[0]; ind_x++) {
    for (int ind_y = 0; ind_y < grid_size[1]; ind_y++) {
      for (int ind_z = 0; ind_z < grid_size[2]; ind_z++) {
        ls_val = grid_values[ind_x + ind_y * grid_size[0] + ind_z * grid_size[0] * grid_size[1]];
        if (abs(ls_val) < ls_ref) {
          // Close to boundary if abs(ls_val) < ls_ref, apply smearing.
          dV = smeared_heaviside_step(-ls_val / ls_ref) * volume_cell;
        } else if (ls_val < 0) {
          // Inside and far away from boundary
          dV = volume_cell;
        } else if (ls_val > 0) {
          // Outside and far away from boundary
          dV = 0.0;
        }
        if (dV > 0.0) {
          volume += dV;
          x_com += ind_x * stride * dV;
          y_com += ind_y * stride * dV;
          z_com += ind_z * stride * dV;
        }
      }
    }
  }
  x_com /= volume;
  y_com /= volume;
  z_com /= volume;

Real LevelSet::getSurface_epsilon(Real epsilon) const // to avoid code duplication in getSurface
{
	Real volExcess(volumeInsideThreshold(epsilon)), volDefault(volumeInsideThreshold(-epsilon));
	if (volExcess == volDefault) // may happen in case of failed iterative search in getSurface and an epsilon really fading to 0
		LOG_WARN(
		        "Measuring twice the same volume when using epsilon = " << epsilon << " for a grid spacing of " << lsGrid->spacing
		                                                                << ", we will obtain a zero surface");
	return (volExcess - volDefault) / (2 * epsilon);
}
Real LevelSet::getSurface(Real epsilon) const
{
	unsigned int cptr(0), cptrMax(100);
	Real         surfOld(getSurface_epsilon(epsilon));
	Real         surfNew(-1), relChange(-1);
	while (cptr < cptrMax) {
		epsilon   = epsilon / 2;
		surfNew   = (getSurface_epsilon(epsilon));
		relChange = math::abs(surfOld - surfNew) / surfOld;
		LOG_INFO(
		        "During the iteration nbr " << cptr + 1 << " (with new epsilon = " << epsilon << "), it is computed " << surfNew
		                                    << " for new surface value, to compare with " << surfOld << " for old surface value, ie a " << relChange
		                                    << " relative change");
		if (relChange < 1.e-7) // we converged to a limit
			break;
		else { // we go for another round
			surfOld = surfNew;
			cptr++;
		}
	}
	if (cptr == cptrMax) LOG_ERROR("We reached " << cptrMax << " iterations wo converging to a limit surface value");
	return surfNew;
}








//TODO: Should we have a flag (or child classes) for different memory distribution strategies?
//      a) all procs store grids, b) sub grids for each atom, c) hash table for each atom
//      then benchmark across different limits? Few large grains, lots of small grains, jamming vs. flow...

/* ---------------------------------------------------------------------- */

FixRigidLSDEM::FixRigidLSDEM(LAMMPS *lmp, int narg, char **arg) :
    FixRigid(lmp, narg, arg), id_fix(nullptr), id_fix2(nullptr), global_grids(nullptr),
    grid_style(nullptr), grid_min(nullptr), grid_stride(nullptr), grid_scale(nullptr), grid_index(nullptr), grid_size(nullptr)
{
  comm_forward = 8;
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
    "{} all property/atom d2_ls_dem_com 3 d2_ls_dem_quat 4 d_ls_dem_vol d2_ls_dem_n 3 d2_ls_dem_fs 3 i_ls_dem_touch_id d2_ls_dem_fn1 d2_ls_dem_fs1 ghost yes writedata no",
     id_fix));
  int tmp1, tmp2;
  index_ls_dem_com = atom->find_custom("ls_dem_com", tmp1, tmp2);
  index_ls_dem_quat = atom->find_custom("ls_dem_quat", tmp1, tmp2);
  index_ls_dem_vol = atom->find_custom("ls_dem_vol", tmp1, tmp2);
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

  // Update center of mass
  double **grain_com = atom->darray[index_ls_dem_com];
  double **quat_lsdem = atom->darray[index_ls_dem_quat];
  int *touch_id = atom->ivector[index_ls_dem_touch_id];
  int ibody;

  for (int i = 0; i < atom->nlocal; i++) {
    ibody = body[i];
    grain_com[i][0] = xcm[ibody][0];
    grain_com[i][1] = xcm[ibody][1];
    grain_com[i][2] = xcm[ibody][2];

    quat_lsdem[i][0] = quat[ibody][0];
    quat_lsdem[i][1] = quat[ibody][1];
    quat_lsdem[i][2] = quat[ibody][2];
    quat_lsdem[i][3] = quat[ibody][3];

    touch_id[i] = -1;
  }

  // Copy maximum cutoff
  if (!utils::strmatch(force->pair_style,"^ls/dem"))
    error->all(FLERR, "Must use pair ls/dem with fix rigid/ls/dem");
  auto pair = dynamic_cast<PairLSDEM *>(force->pair);
  maxcut = pair->maxcut;

  int index_global = 0;
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
    for (int ibody = 0; ibody < nbody; ibody++) {
      filename.assign(gridfiles[ibody]);
      read_gridfile(ibody, 0, filename, grid_size, nullptr);
      file_map[filename].insert(ibody);

      // Calculate and save grid properties
      if (dim == 2) grid_size[ibody][2] = 1;
      for (int a = 0; a < 3; a++)
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
      }
    }

    // ------------------------------ //
    // Allocate memory for level sets //
    // ------------------------------ //

    rcell = maxcut / max_stride + 2; // +1 for interpolation +1 for safety

    for (int a = 0; a < 3; a++) subgrid_size[a] = 2 * rcell + 1;  // +1 for middle cell (needed?)
    if (dim == 2) subgrid_size[2] = 1;
    id_fix2 = utils::strdup(id + std::string("_FIX_PROP_ATOM_2"));
    int ntotal = subgrid_size[0] * subgrid_size[1] * subgrid_size[2];
    modify->add_fix(fmt::format("{} all property/atom d2_grid_values {} d2_grid_min {} writedata no ghost yes", id_fix2, ntotal, 3));

    int tmp1, tmp2;
    index_grid_values = atom->find_custom("grid_values", tmp1, tmp2);
    index_grid_min = atom->find_custom("grid_min", tmp1, tmp2);

    memory->create_ragged(global_grids, index_global, ntotal_global, "rigid/ls/dem:global_grids");

    // ------------------------------ //
    // Read and store level sets      //
    // ------------------------------ //

    ntotal = max_grid_size[0] * max_grid_size[1] * max_grid_size[2];
    double *temp_grid_values;
    memory->create(temp_grid_values, ntotal, "rigid/ls/dem:temp_grid_values");

    double **grid_values = atom->darray[index_grid_values];
    double **grid_min_local = atom->darray[index_grid_min];
    double *ls_dem_vol = atom->dvector[index_ls_dem_vol];

    // TODO: DOES THIS ONLY WORK WHEN GRAINS ARE AXIS-ALIGNED ?
    //       I.E. WE MUST TELL THE USERS NOT TO ROTATE ANYTHING BEFORE RIGID IS DONE ?

    double *ls_val;
    double delx, dely, delz, inertia_ls[6];
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
        for (int i = 0; i < atom->nlocal; i++) {
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
                index_local = ix_local + iy_local * subgrid_size[0] + iz_local *   subgrid_size[0] * subgrid_size[1];

                // Final sanity check (defensive)
                if (index_global < 0 || index_global >= ntotal)
                  error->all(FLERR, "Level set does not include a large enough buffer for the   cutoff");
                grid_values[i][index_local] = temp_grid_values[index_global] * grid_scale[ibody];
              }
            }
          }
          // compare/replace inertia with inertia_ls
        }
      }

      for (int i = 0; i < atom->nlocal; i++) {
        ibody = body[i];
        ls_dem_vol[i] = process_ls_grid(grid_size[ibody], grid_stride[ibody], temp_grid_values,   inertia_ls, filename);
        // calculate surface area as well
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

  double **x_lsdem = atom->darray[index_ls_dem_com];
  double **quat_lsdem = atom->darray[index_ls_dem_quat];

  int ibody;
  for (int i = 0; i < atom->nlocal; i++) {
    ibody = body[i];
    x_lsdem[i][0] = xcm[ibody][0];
    x_lsdem[i][1] = xcm[ibody][1];
    x_lsdem[i][2] = xcm[ibody][2];

    quat_lsdem[i][0] = quat[ibody][0];
    quat_lsdem[i][1] = quat[ibody][1];
    quat_lsdem[i][2] = quat[ibody][2];
    quat_lsdem[i][3] = quat[ibody][3];
  }
}

/* ---------------------------------------------------------------------- */

int FixRigidLSDEM::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i, j, m;
  double **x_lsdem = atom->darray[index_ls_dem_com];
  double **quat_lsdem = atom->darray[index_ls_dem_quat];

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(body[j]).d;

    buf[m++] = x_lsdem[j][0];
    buf[m++] = x_lsdem[j][1];
    buf[m++] = x_lsdem[j][2];

    buf[m++] = quat_lsdem[j][0];
    buf[m++] = quat_lsdem[j][1];
    buf[m++] = quat_lsdem[j][2];
    buf[m++] = quat_lsdem[j][3];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixRigidLSDEM::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;
  double **x_lsdem = atom->darray[index_ls_dem_com];
  double **quat_lsdem = atom->darray[index_ls_dem_quat];

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    body[i] = (int) ubuf(buf[m++]).i;

    x_lsdem[i][0] = buf[m++];
    x_lsdem[i][1] = buf[m++];
    x_lsdem[i][2] = buf[m++];

    quat_lsdem[i][0] = buf[m++];
    quat_lsdem[i][1] = buf[m++];
    quat_lsdem[i][2] = buf[m++];
    quat_lsdem[i][3] = buf[m++];
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
      grid_min[ibody][idim] = grid_size_buf[idim + 1] * grid_scale[ibody];
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
  Process a grid file
------------------------------------------------------------------------- */

double FixRigidLSDEM::process_ls_grid(int *grid_size, double stride, double *grid_values, double *inertia_ls, std::string filename)
{
  // Volume integration

  // This is the reference distance values that determines the smearing with of
  // the Heaviside step function. Current expression is the half-diagional of the
  // grid cell divided by a smearing constant.
  double smearCoeff = 1.5;
  double ls_ref = 0.0;
  if (smearCoeff != 0)
    ls_ref = sqrt(0.75) * stride / smearCoeff;

  // Initialise volume and centre of mass
  double volume = 0.0, x_com = 0.0, y_com = 0.0, z_com = 0.0;
  // Cell volume, temporary grid points, integration volume.
  double volume_cell = stride * stride;
  if (domain->dimension == 3) volume_cell *= stride;

  // Integration
  double dV, ls_val;
  for (int ind_x = 0; ind_x < grid_size[0]; ind_x++) {
    for (int ind_y = 0; ind_y < grid_size[1]; ind_y++) {
      for (int ind_z = 0; ind_z < grid_size[2]; ind_z++) {
        ls_val = grid_values[ind_x + ind_y * grid_size[0] + ind_z * grid_size[0] * grid_size[1]];
        if (abs(ls_val) < ls_ref) {
          // Close to boundary if abs(ls_val) < ls_ref, apply smearing.
          dV = smeared_heaviside_step(-ls_val / ls_ref) * volume_cell;
        } else if (ls_val < 0) {
          // Inside and far away from boundary
          dV = volume_cell;
        } else if (ls_val > 0) {
          // Outside and far away from boundary
          dV = 0.0;
        }
        if (dV > 0.0) {
          volume += dV;
          x_com += ind_x * stride * dV;
          y_com += ind_y * stride * dV;
          z_com += ind_z * stride * dV;
        }
      }
    }
  }
  x_com /= volume;
  y_com /= volume;
  z_com /= volume;

  // Computing the inertia tensor (a double loop is unavoidable)
  double delx, dely, delz;
  for (int a = 0; a < 6; a++) inertia_ls[a] = 0.0;
  for (int ind_x = 0; ind_x < grid_size[0]; ind_x++) {
    for (int ind_y = 0; ind_y < grid_size[1]; ind_y++) {
      for (int ind_z = 0; ind_z < grid_size[2]; ind_z++) {
        ls_val = grid_values[ind_x + ind_y * grid_size[0] + ind_z * grid_size[0] * grid_size[1]];
        if (abs(ls_val) < ls_ref) {
          // Close to boundary if abs(ls_val) < ls_ref, apply smearing.
          dV = smeared_heaviside_step(-ls_val / ls_ref) * volume_cell;
        } else if (ls_val < 0) {
          // Inside and far away from boundary
          dV = volume_cell;
        } else if (ls_val > 0) {
          // Outside and far away from boundary
          dV = 0.0;
        }
        if (dV > 0.0) {
          delx = ind_x * stride - x_com;
          dely = ind_y * stride - y_com;
          delz = ind_z * stride - z_com;
          inertia_ls[0] += (dely * dely + delz * delz) * dV;
          inertia_ls[1] += (delx * delx + delz * delz) * dV;
          inertia_ls[2] += (delx * delx + dely * dely) * dV;
          inertia_ls[3] -= delx * dely * dV;
          inertia_ls[4] -= delx * delz * dV;
          inertia_ls[5] -= dely * delz * dV;
        }
      }
    }
  }

  // Check to see if level set has a non-inertial reference frame
  double I_diag_norm = sqrt(inertia_ls[0] * inertia_ls[0] + inertia_ls[1] * inertia_ls[1] + inertia_ls[2] * inertia_ls[2]);
  double I_off_diag_norm = sqrt(2.0 * (inertia_ls[3] * inertia_ls[3] + inertia_ls[4] * inertia_ls[4] + inertia_ls[5] * inertia_ls[5]));
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
  if ( (ind_x < 0) || (ind_y < 0) || ((domain->dimension == 3) && (ind_z < 0)) ) {
    // Point is outside the LS grid of grain j. Cannot compute distance or normal.
    error->one(FLERR, "Contacting node {} is outside of node {}'s LS grid", atom->tag[i], atom->tag[j]);
  } else if ( (ind_x >= nrow - 1) || (ind_y >= ncol - 1) || ((domain->dimension == 3) && (ind_z >= nslice - 1))) {
    // Point is outside the LS grid of grain j. Cannot compute distance or normal.
    error->one(FLERR, "Contacting node {} is outside of node {}'s LS grid", atom->tag[i], atom->tag[j]);
  }

  // The normalised coordinates within the current grid cell.
  // May be safer to cap them with math::max(math::min(x_red, 1.0), 0.0)
  x_red = x_red - static_cast<double>(ind_x);
  y_red = y_red - static_cast<double>(ind_y);
  z_red = z_red - static_cast<double>(ind_z); // Should always be zero in 2D.

  //  Interpolate
  if (grid_style[ibody] == DISTRIBUTED) {
    double **node_local_grid = atom->darray[index_grid_values];

    // Level-set values on the grid points in the lower z plane (ind_z)
    double ls000 = node_local_grid[j][ind_x     + ind_y       * ncol + ind_z * ncol * nrow];
    double ls100 = node_local_grid[j][ind_x + 1 + ind_y       * ncol + ind_z * ncol * nrow];
    double ls010 = node_local_grid[j][ind_x     + (ind_y + 1) * ncol + ind_z * ncol * nrow];
    double ls110 = node_local_grid[j][ind_x + 1 + (ind_y + 1) * ncol + ind_z * ncol * nrow];

    // Bi-linear interpolation in the lower z plane (ind_z)
    double lsxy0 = ls000 + y_red * (ls010 - ls000) +
                   x_red * (ls100 - ls000 + y_red * (ls110 - ls100 - ls010 + ls000));

    if (domain->dimension == 3) { // 3D
      // Level-set values on the grid points in the upper z plane (ind_z+1)
      double ls001 = node_local_grid[j][ind_x     + ind_y       * ncol + (ind_z + 1) * ncol * nrow];
      double ls101 = node_local_grid[j][ind_x + 1 + ind_y       * ncol + (ind_z + 1) * ncol * nrow];
      double ls011 = node_local_grid[j][ind_x     + (ind_y + 1) * ncol + (ind_z + 1) * ncol * nrow];
      double ls111 = node_local_grid[j][ind_x + 1 + (ind_y + 1) * ncol + (ind_z + 1) * ncol * nrow];

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
    double ls000 = my_grid[ind_x     + ind_y       * ncol + ind_z * ncol * nrow];
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

  return dist;
}
