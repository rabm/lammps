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

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
using namespace RigidConst;

//TODO: Should we have a flag (or child classes) for different memory distribution strategies?
//      a) all procs store grids, b) sub grids for each atom, c) hash table for each atom
//      then benchmark across different limits? Few large grains, lots of small grains, jamming vs. flow...

/* ---------------------------------------------------------------------- */

FixRigidLSDEM::FixRigidLSDEM(LAMMPS *lmp, int narg, char **arg) :
    FixRigid(lmp, narg, arg), id_fix(nullptr), id_fix2(nullptr),
    ngrid(nullptr), grid_ls_val(nullptr), grid_min(nullptr), grid_stride(nullptr)
{
  comm_forward = 8;
  maxcut = -1;

  n_extra_attributes = 2;

  if (!inpfile)
    error->all(FLERR, "Must specify infile with level set for fix rigid/ls/dem");

  memory->create(ngrid, nbody, domain->dimension, "rigid/ls/dem:ngrid");
  memory->create(grid_min, nbody, domain->dimension, "rigid/ls/dem:grid_min");
  memory->create(grid_stride, nbody, "rigid/ls/dem:grid_stride");
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

  memory->destroy(ngrid);
  memory->destroy(grid_ls_val);
  memory->destroy(grid_min);
  memory->destroy(grid_stride);
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
  modify->add_fix(fmt::format("{} all property/atom d2_ls_dem_quat 4 d2_ls_dem_com 3 d_ls_dem_vol ghost yes writedata no", id_fix));
  int tmp1, tmp2;
  index_ls_dem_com = atom->find_custom("ls_dem_com", tmp1, tmp2);
  index_ls_dem_quat = atom->find_custom("ls_dem_quat", tmp1, tmp2);
  index_ls_dem_vol = atom->find_custom("ls_dem_vol", tmp1, tmp2);
}

/* ---------------------------------------------------------------------- */

void FixRigidLSDEM::init()
{
  FixRigid::init(); // Calls setup_bodies_static()

  // Copy maximum
  if (!utils::strmatch(force->pair_style,"^ls/dem"))
    error->all(FLERR, "Must use pair ls/dem with fix rigid/ls/dem");
  auto pair = dynamic_cast<PairLSDEM *>(force->pair);
  maxcut = pair->maxcut;

  // Initialize peratom arrays once
  if (id_fix2) return;

  // Create per-atom properties necessary for current implementation of LS-DEM
  // TODO: THIS IS TEMPORARY FOR A SINGLE TYPE OF GRAINS AS ALL ATOMS STORE THE SAME SIZE
  // TODO: CREATE TEMP GROUPS TO PUT ATOMS OF SAME GRAIN TOGETHER AND CREATE FIX PROPERTY/ATOM OF DIFFERENT SIZE
  // TODO: MUST BE SOME PARALLEL COMPLICATION, LOOK AT THE GROUP COMMAND CODE TO SEE HOW IT'S DONE
  id_fix2 = utils::strdup(id + std::string("_FIX_PROP_ATOM_2"));
  spac = 0.5; // TODO make a user specified input. TODO: manage different spacing for different rigid bodies
  rcell = maxcut / spac + 2; // +1 for interpolation +1 for safety
  // JBC: Can size of rcell, or ngrid_local always be the smallest for interpolation, i.e. 3 ?
  //      and if atom is outside of local grid of its neighbor, then we just pass? Or is that check expensive? and that's why we make sure it's always inside cutoff?
  for (int a = 0; a < 3; a++) ngrid_local[a] = 2 * rcell + 1;  // +1 for middle cell (is this needed?)
  if (domain->dimension == 2) ngrid_local[2] = 1;
  if (!modify->get_fix_by_id(id_fix2)) {
    int n = ngrid_local[0] * ngrid_local[1] * ngrid_local[2];

    modify->add_fix(fmt::format("{} all property/atom d2_ls_grid {} d2_ls_local_gridmin {} writedata no ghost yes",
                                id_fix2, n, 3));
  }

  int tmp1, tmp2;
  index_ls_grid = atom->find_custom("ls_grid", tmp1, tmp2);
  index_ls_local_gridmin = atom->find_custom("ls_local_gridmin", tmp1, tmp2);

  // Update center of mass
  double **grain_com = atom->darray[index_ls_dem_com];
  double **quat_lsdem = atom->darray[index_ls_dem_quat];
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
  }

  // Populate local arrays
  // TODO: DOES THIS ONLY WORK WHEN GRAINS ARE AXIS-ALIGNED ?
  //       I.E. WE MUST TELL THE USERS NOT TO ROTATE ANYTHING BEFORE RIGID IS DONE ?
  double **grid = atom->darray[index_ls_grid];
  double **grid_min_local = atom->darray[index_ls_local_gridmin];
  double *ls_dem_vol = atom->dvector[index_ls_dem_vol];
  double *ls_val = grid_ls_val[0]; // TODO: Using [0] only works for identical grains
  int ncol = ngrid[0][0]; // TODO: rename nx, ny, nz
  int nrow = ngrid[0][1];
  int nslice = (domain->dimension == 3) ? ngrid[0][2] : 1;
  int ngrid_global = ncol * nrow * nslice;

  double delx, dely, delz;
  double **x = atom->x;
  int ix_node, iy_node, iz_node, xmincell, ymincell, zmincell, index;
  int ix_global, iy_global, iz_global;
  int index_global, index_local, index_grid_min_local[3];
  for (int i = 0; i < atom->nlocal; i++) {
    ibody = body[i];

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
    // TODO: explicit conversion ? int(delx / spac) ?
    ix_node = delx / spac;
    iy_node = dely / spac;
    iz_node = delz / spac; // delz should always be zero in 2D.
                           // If negative due to roundoff, must integer cast to zero: pick int() vs floor() wisely

    // index of local grid minimum in global grid.
    // JBC: Can this be negative if not enough padding of the LS grid relative to grain surface? i.e. ix < rcell ?
    index_grid_min_local[0] = ix_node - rcell;
    index_grid_min_local[1] = iy_node - rcell;
    index_grid_min_local[2] = iz_node - rcell * (domain->dimension == 3); // local cell has zero z-dimension in 2D. Could also be (ngrid_local[1]-1)/2 to work for all dims without check, but probably uglier

    // location of local grid minimum relative to CoM
    grid_min_local[i][0] = index_grid_min_local[0] * spac + grid_min[ibody][0];
    grid_min_local[i][1] = index_grid_min_local[1] * spac + grid_min[ibody][1];
    grid_min_local[i][2] = index_grid_min_local[2] * spac; // should always be zero in 2D
    if (domain->dimension == 3) grid_min_local[i][2] += grid_min[ibody][2];

    for (int iz_local = 0; iz_local < ngrid_local[2]; iz_local++) {
      for (int iy_local = 0; iy_local < ngrid_local[1]; iy_local++) {
        for (int ix_local = 0; ix_local < ngrid_local[0]; ix_local++) {
          // shift local cell to global cell
          // JBC, is `ix_node` above already the global index ?
          ix_global = ix_local + index_grid_min_local[0];
          iy_global = iy_local + index_grid_min_local[1];
          iz_global = iz_local + index_grid_min_local[2]; // should always be zero in 2D

          index_global = ix_global + iy_global * ncol + iz_global * ncol * nrow;
          index_local = ix_local + iy_local * ngrid_local[0] + iz_local * ngrid_local[0] * ngrid_local[1];

          if (index_global > ngrid_global || index_global < 0)
            error->all(FLERR, "Level set does not include a large enough buffer for the cutoff");
          grid[i][index_local] = ls_val[index_global];
        }
      }
    }
    // TODO: pass volume through I/O, or compute some heuristic based on counting negative LS grid cells ?
    ls_dem_vol[i] = MY_PI * pow(5.0, 2); //vol
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
   one-time initialization of static rigid body attributes
   runs default parent method, then grabs & processes file names for LS grid
------------------------------------------------------------------------- */

void FixRigidLSDEM::setup_bodies_static()
{
  FixRigid::setup_bodies_static();

  int *inbody;
  memory->create(inbody,nbody,"rigid:inbody");
  for (int ibody = 0; ibody < nbody; ibody++) inbody[ibody] = 0;

  // Read the level-set information from gridfiles
  char **ls_grid_files;
  double *scale;
  memory->create(scale, nbody, "rigid/ls/dem:scale");
  memory->create(ls_grid_files, nbody, MAXLINE, "rigid/ls/dem:ls_grid_files");

  // Read scaling factors and gridfiles names
  readfile_lsdem(scale, inbody, ls_grid_files);

  // Read grid dimensions for all bodies
  read_gridfile(0, ls_grid_files, scale);
  int *ngrid_flat;
  memory->create(ngrid_flat, nbody, "rigid/ls/dem:ngrid_flat");
  for (int ibody = 0; ibody < nbody; ibody++) {
    ngrid_flat[ibody] = 1;
    for (int idim = 0; idim < domain->dimension; idim++)
      ngrid_flat[ibody] *= ngrid[ibody][idim];
  }

  // Create grid_ls_val from dimensions read into ngrid by read_gridfile()
  // This cannot be done before reading gridfiles, e.g., in the constructor where we create ngrid
  memory->create_ragged(grid_ls_val, nbody, ngrid_flat, "rigid/ls/dem:grid_ls_val");

  // Read and scale level-set values for all bodies (requires grid_ls_val to be sized correctly)
  read_gridfile(1, ls_grid_files, scale);

  memory->destroy(ls_grid_files);
  memory->destroy(scale);
  memory->destroy(ngrid_flat);
  memory->destroy(inbody);
}

/* ----------------------------------------------------------------------
   read per rigid body info from user-provided file to read LSDEM scaling and gridfile
   flag inbody = 0 for bodies whose info is read from file
   nlines = # of lines of rigid body info
   one line = rigid-ID mass xcm ycm zcm ixx iyy izz ixy ixz iyz
              vxcm vycm vzcm lx ly lz ix iy iz ls-scale ls-filename
------------------------------------------------------------------------- */

void FixRigidLSDEM::readfile_lsdem(double *scale, int *inbody, char** gridfiles)
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

        inbody[id] = 1;

        values.skip(19);
        scale[id] = values.next_double();
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
   files ls_grid_files to read from stored previously by readfile() function
   first line = ngridx ngridy ngridz
   followed by ngridx * ngridy * ngridz lines of level set values at the grid points
   which = 0, read only the size of the level-set grid
   which = 1, read the values of the level-set grid
   TODO: User responsible for knowing what the LS values in their file are
   scaled to, and pick the correct scaling factor
   TODO: this assumes all rigid bodies are LSDEM grains. Otherwise, we should pass and read *inbody.
        Not sure if there is a use for this: why would fix rigid lsdem have rigid bodies not be LSDEM ?
        Refactor readfile() accordingly if this is the route we take
------------------------------------------------------------------------- */

void FixRigidLSDEM::read_gridfile(int which, char **ls_grid_files, double* scale)
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
  for (int ibody = 0; ibody < nbody; ibody++) {
    int nlines = 1;
    char* gridfile = ls_grid_files[ibody];
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

    // TODO: I left the 2 lines below from original rigid::readline() not sure if needed
    // empty file with 0 lines is needed to trigger initial restart file
    // generation when no infile was previously used.
    if (nlines == 0) return;
    else if (nlines < 0) error->all(FLERR, "Fix rigid/ls/dem gridfile has incorrect format");

    if (which == 0) {
      grid_stride[ibody] = grid_size_buf[0] * scale[ibody];
      for (int idim = 0; idim < dim; idim++) {
        grid_min[ibody][idim] = grid_size_buf[idim + 1] * scale[ibody];
        ngrid[ibody][idim] = grid_shape_buf[idim];
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
            grid_ls_val[ibody][nread + i] = values.next_double() * scale[ibody];
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
}

/* ----------------------------------------------------------------------
   Find the value of node (atom) i in j's LS grid.
------------------------------------------------------------------------- */

double FixRigidLSDEM::get_ls_value(int i, int j, double *normal)
{
  double **x = atom->x;
  double **grain_com = atom->darray[index_ls_dem_com];
  double **grain_quat = atom->darray[index_ls_dem_quat];
  double **grain_grid = atom->darray[index_ls_grid]; // Danny: This is per atom/node, so I would call it node_grid.
  double **local_grid_min = atom->darray[index_ls_local_gridmin];

  int ncol = ngrid_local[0]; // Danny: this should be for the node's grid as well
  int nrow = ngrid_local[1];
  int nslice = ngrid_local[2];

  // Calculate position of node i in node j's grid using:
  //   x[i][0-2] = location of i
  //   x[j][0-2] = location of j
  //   grain_com[j][0-2] = CoM of j's grain
  //   grain_quat[j][0-3] = quat of j's grain

  //
  //  GET NODE I IN LOCAL COORDINATES OF J GRAIN
  //

  // Location of the node (atom) of i relative to the centre of mass (CoM) of j.
  double delx = x[i][0] - grain_com[j][0];
  double dely = x[i][1] - grain_com[j][1];
  double delz = x[i][2] - grain_com[j][2];

  // Account for PBCs.
  domain->minimum_image(delx, dely, delz);

  // Apply quaternion rotation to move into local reference frame of grain j grid.
  // Here, grain_quat is local->global. Therefore, grain_quat_conj is global -> local.
  double x_local[3];
  double dx[3] = {delx, dely, delz};
  double grain_quat_conj[4];
  MathExtra::qconjugate(grain_quat[j], grain_quat_conj);
  MathExtra::quatrotvec(grain_quat_conj, dx, x_local);
  // See comments above functions in math_extra.h/cpp for details

  //
  //  COMPUTE THE LS GRID INDICES
  //

  // Translate local coordinates such that they are relative
  // to the lower corner of the level set grid.
  x_local[0] -= local_grid_min[j][0];
  x_local[1] -= local_grid_min[j][1];
  x_local[2] -= local_grid_min[j][2];
  // Danny: THIS local_grid_min NEEDS TO BE THE MINIMUM OF THE NODE'S GRID, NOT THE FULL GRID!

  // Normalise the coordinates to be in units of the number of grid cells.
  double x_red = x_local[0] / spac;
  double y_red = x_local[1] / spac;
  double z_red = x_local[2] / spac;

  // Calculate index from relative coordinate, being careful with integer division.
  int ind_x = int(x_red); // Here, int() does the same as floor() + conversion.
  int ind_y = int(y_red);
  int ind_z = int(z_red); // Should always be zero in 2D.
  // JBC: There is some padding for detection / normal caculation that I don't understand clearly
  // Danny: Does the below clarify? Or is there something else that is missing?

  // Checking whether x_local lies within the grid. Avoids edge cases where finite precision
  // leads to e.g. a x=-0.1 coordinate to fall outside of a grid that starts at x=-0.1.
  if ( (ind_x < 0) || (ind_y < 0) || ((domain->dimension == 3) && (ind_z < 0)) ) {
    // Point is outside the LS grid of grain j. Cannot compute distance or normal.
    error->one(FLERR, "Contacting node {} is outside of node {}'s LS grid", atom->tag[i], atom->tag[j]);
  } else if ( (ind_x >= nrow - 1) || (ind_y >= ncol - 1) || ((domain->dimension == 3) && (ind_z >= nslice - 1)) ) {
    // Point is outside the LS grid of grain j. Cannot compute distance or normal.
    error->one(FLERR, "Contacting node {} is outside of node {}'s LS grid", atom->tag[i], atom->tag[j]);
  }

  //
  //  DO THE INTERPOLATION
  //
  double dist, nx(0.0), ny(0.0), nz(0.0);

  // The normalised coordinates within the current grid cell.
  // May be safer to cap them with math::max(math::min(x_red, 1.0), 0.0)
  x_red = x_red - static_cast<double>(ind_x);
  y_red = y_red - static_cast<double>(ind_y);
  z_red = z_red - static_cast<double>(ind_z); // Should always be zero in 2D.

  // Level-set values on the grid points in the lower z plane (ind_z)
  double ls000 = grain_grid[j][ind_x   + ind_y     * ncol + ind_z * ncol * nrow];
  double ls100 = grain_grid[j][ind_x+1 + ind_y     * ncol + ind_z * ncol * nrow];
  double ls010 = grain_grid[j][ind_x   + (ind_y+1) * ncol + ind_z * ncol * nrow];
  double ls110 = grain_grid[j][ind_x+1 + (ind_y+1) * ncol + ind_z * ncol * nrow];

  // Bi-linear interpolation in the lower z plane (ind_z)
  double lsxy0 = ls000 + y_red * (ls010 - ls000) +
                         x_red * (ls100 - ls000 +
                                  y_red * (ls110 - ls100 - ls010 + ls000));

  if (domain->dimension == 3) { // 3D
    // Level-set values on the grid points in the upper z plane (ind_z+1)
    double ls001 = grain_grid[j][ind_x   + ind_y     * ncol + (ind_z+1) * ncol * nrow];
    double ls101 = grain_grid[j][ind_x+1 + ind_y     * ncol + (ind_z+1) * ncol * nrow];
    double ls011 = grain_grid[j][ind_x   + (ind_y+1) * ncol + (ind_z+1) * ncol * nrow];
    double ls111 = grain_grid[j][ind_x+1 + (ind_y+1) * ncol + (ind_z+1) * ncol * nrow];

    // Bi-linear interpolation in the upper z plane (ind_z+1)
    double lsxy1 = ls001 + y_red * (ls011 - ls001) +
                           x_red * (ls101 - ls001 +
                                    y_red * (ls111 - ls101 - ls011 + ls001));

    // Affecting tri-linear interpolation by linear interpolation of the two bi-linear interpolations.
    dist = z_red * (lsxy1 - lsxy0) + lsxy0;

	  // Computing normal as the gradient of trilinear interpolation
    // TODO: maybe hardcode without loops ? Danny: Compiler should optimise this automatically, don't think it changes anything?
    for (int a = 0; a < 2; a++) {
      for (int b = 0; b < 2; b++) {
        for (int c = 0; c < 2; c++) {
          double lsVal = grain_grid[j][(ind_x + a) + (ind_y + b) * ncol + (ind_z + c) * ncol * nrow];
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
        double lsVal = grain_grid[j][(ind_x + a) + (ind_y + b) * ncol];
        nx += lsVal * (2 * a - 1) * ((1 - b) * (1 - y_red) + b * y_red);
        ny += lsVal * (2 * b - 1) * ((1 - a) * (1 - x_red) + a * x_red);
      }
    }
    nz = 0.0;
  }

  // Assign normal
  normal[0] = nx;
  normal[1] = ny;
  normal[2] = nz;

  // Rotate normal back to global coordinates
  MathExtra::quatrotvec(grain_quat[j], normal, normal);

  return dist;
}
