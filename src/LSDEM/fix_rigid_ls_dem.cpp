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
#include "atom_vec_ellipsoid.h"
#include "atom_vec_line.h"
#include "atom_vec_tri.h"
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
#include "random_mars.h"
#include "respa.h"
#include "rigid_ls_dem_const.h"
#include "tokenizer.h"
#include "update.h"
#include "variable.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
using namespace RigidLSDEMConst;

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
  triclinic = domain->triclinic;

  // atom style pointers to particles that store extra info

  avec_ellipsoid = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  avec_line = dynamic_cast<AtomVecLine *>(atom->style_match("line"));
  avec_tri = dynamic_cast<AtomVecTri *>(atom->style_match("tri"));

  // warn if more than one rigid fix
  // if earlyflag, warn if any post-force fixes come after a rigid fix

  int count = 0;
  for (auto &ifix : modify->get_fix_list())
    if (ifix->rigid_flag) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one fix rigid");

  if (earlyflag) {
    bool rflag = false;
    for (auto &ifix : modify->get_fix_list()) {
      if (ifix->rigid_flag) rflag = true;
      if ((comm->me == 0) && rflag && (ifix->setmask() & POST_FORCE) && !ifix->rigid_flag)
        error->warning(FLERR, "Fix {} with ID {} alters forces after fix rigid",
                       ifix->style, ifix->id);
    }
  }

  // warn if body properties are read from inpfile
  //   and the gravity keyword is not set and a gravity fix exists
  // this could mean body particles are overlapped
  //   and gravity is not applied correctly

  if (inpfile && !id_gravity) {
    if (modify->get_fix_by_style("^gravity").size() > 0)
      if (comm->me == 0)
        error->warning(FLERR,"Gravity may not be correctly applied to rigid "
                       "bodies if they consist of overlapped particles");
  }

  //  error if a fix changing the box comes before rigid fix

  bool boxflag = false;
  for (auto &ifix : modify->get_fix_list()) {
    if (boxflag && utils::strmatch(ifix->style,"^rigid"))
        error->all(FLERR,"Rigid fixes must come before any box changing fix");
    if (ifix->box_change) boxflag = true;
  }

  // add gravity forces based on gravity vector from fix

  if (id_gravity) {
    auto ifix = modify->get_fix_by_id(id_gravity);
    if (!ifix) error->all(FLERR,"Fix rigid cannot find fix gravity ID {}", id_gravity);
    if (!utils::strmatch(ifix->style,"^gravity"))
      error->all(FLERR,"Fix rigid gravity fix ID {} is not a gravity fix style", id_gravity);
    int tmp;
    gvec = (double *) ifix->extract("gvec", tmp);
  }

  // timestep info

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dtq = 0.5 * update->dt;

  if (utils::strmatch(update->integrate_style,"^respa"))
    step_respa = (dynamic_cast<Respa *>(update->integrate))->step;

  // setup rigid bodies, using current atom info. if reinitflag is not set,
  // do the initialization only once, b/c properties may not be re-computable
  // especially if overlapping particles.
  //   do not do dynamic init if read body properties from inpfile.
  // this is b/c the inpfile defines the static and dynamic properties and may
  // not be computable if contain overlapping particles.
  //   setup_bodies_static() reads inpfile itself

  if (reinitflag || !setupflag) {
    setup_bodies_static();
    if (!inpfile) setup_bodies_dynamic();
    setupflag = 1;
  }

  // temperature scale factor

  double ndof = 0.0;
  for (int ibody = 0; ibody < nbody; ibody++) {
    ndof += fflag[ibody][0] + fflag[ibody][1] + fflag[ibody][2];
    ndof += tflag[ibody][0] + tflag[ibody][1] + tflag[ibody][2];
  }
  ndof -= nlinear;
  if (ndof > 0.0) tfactor = force->mvv2e / (ndof * force->boltz);
  else tfactor = 0.0;

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
   sets extended flags, masstotal, center-of-mass
   sets Cartesian and diagonalized inertia tensor
   sets body image flags
   may read some properties from inpfile
------------------------------------------------------------------------- */
// TODO: there is so much stuff we don't need in here for LS-DEM
// I started removing it in the first dump commit but we should do that again
// For now, I kept everything as is, and added a new `which`=2 mode in readfile()
// to read the level-set data and call that readfile(2, ...)
void FixRigidLSDEM::setup_bodies_static()
{
  int i,ibody;

  // extended = 1 if any particle in a rigid body is finite size or has a dipole moment

  extended = orientflag = dorientflag = 0;

  AtomVecEllipsoid::Bonus *ebonus;
  if (avec_ellipsoid) ebonus = avec_ellipsoid->bonus;
  AtomVecLine::Bonus *lbonus;
  if (avec_line) lbonus = avec_line->bonus;
  AtomVecTri::Bonus *tbonus;
  if (avec_tri) tbonus = avec_tri->bonus;
  double **mu = atom->mu;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *ellipsoid = atom->ellipsoid;
  int *line = atom->line;
  int *tri = atom->tri;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  if (atom->radius_flag || atom->ellipsoid_flag || atom->line_flag ||
      atom->tri_flag || atom->mu_flag) {
    int flag = 0;
    for (i = 0; i < nlocal; i++) {
      if (body[i] < 0) continue;
      if (radius && radius[i] > 0.0) flag = 1;
      if (ellipsoid && ellipsoid[i] >= 0) flag = 1;
      if (line && line[i] >= 0) flag = 1;
      if (tri && tri[i] >= 0) flag = 1;
      if (mu && mu[i][3] > 0.0) flag = 1;
    }

    MPI_Allreduce(&flag,&extended,1,MPI_INT,MPI_MAX,world);
  }

  // grow extended arrays and set extended flags for each particle
  // orientflag = 4 if any particle stores ellipsoid or tri orientation
  // orientflag = 1 if any particle stores line orientation
  // dorientflag = 1 if any particle stores dipole orientation

  if (extended) {
    if (atom->ellipsoid_flag) orientflag = 4;
    if (atom->line_flag) orientflag = 1;
    if (atom->tri_flag) orientflag = 4;
    if (atom->mu_flag) dorientflag = 1;
    grow_arrays(atom->nmax);

    for (i = 0; i < nlocal; i++) {
      eflags[i] = 0;
      if (body[i] < 0) continue;

      // set to POINT or SPHERE or ELLIPSOID or LINE

      if (radius && radius[i] > 0.0) {
        eflags[i] |= SPHERE;
        eflags[i] |= OMEGA;
        eflags[i] |= TORQUE;
      } else if (ellipsoid && ellipsoid[i] >= 0) {
        eflags[i] |= ELLIPSOID;
        eflags[i] |= ANGMOM;
        eflags[i] |= TORQUE;
      } else if (line && line[i] >= 0) {
        eflags[i] |= LINE;
        eflags[i] |= OMEGA;
        eflags[i] |= TORQUE;
      } else if (tri && tri[i] >= 0) {
        eflags[i] |= TRIANGLE;
        eflags[i] |= ANGMOM;
        eflags[i] |= TORQUE;
      } else eflags[i] |= POINT;

      // set DIPOLE if atom->mu and mu[3] > 0.0

      if (atom->mu_flag && mu[i][3] > 0.0)
        eflags[i] |= DIPOLE;
    }
  }

  // set body xcmimage flags = true image flags

  imageint *image = atom->image;
  for (i = 0; i < nlocal; i++)
    if (body[i] >= 0) xcmimage[i] = image[i];
    else xcmimage[i] = 0;

  // compute masstotal & center-of-mass of each rigid body
  // error if image flag is not 0 in a non-periodic dim

  double **x = atom->x;

  int *periodicity = domain->periodicity;
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double xy = domain->xy;
  double xz = domain->xz;
  double yz = domain->yz;

  for (ibody = 0; ibody < nbody; ibody++)
    for (i = 0; i < 6; i++) sum[ibody][i] = 0.0;
  int xbox,ybox,zbox;
  double massone,xunwrap,yunwrap,zunwrap;

  for (i = 0; i < nlocal; i++) {
    if (body[i] < 0) continue;
    ibody = body[i];

    xbox = (xcmimage[i] & IMGMASK) - IMGMAX;
    ybox = (xcmimage[i] >> IMGBITS & IMGMASK) - IMGMAX;
    zbox = (xcmimage[i] >> IMG2BITS) - IMGMAX;
    if (rmass) massone = rmass[i];
    else massone = mass[type[i]];

    if ((xbox && !periodicity[0]) || (ybox && !periodicity[1]) ||
        (zbox && !periodicity[2]))
      error->one(FLERR,"Fix rigid atom has non-zero image flag in a non-periodic dimension");

    if (triclinic == 0) {
      xunwrap = x[i][0] + xbox*xprd;
      yunwrap = x[i][1] + ybox*yprd;
      zunwrap = x[i][2] + zbox*zprd;
    } else {
      xunwrap = x[i][0] + xbox*xprd + ybox*xy + zbox*xz;
      yunwrap = x[i][1] + ybox*yprd + zbox*yz;
      zunwrap = x[i][2] + zbox*zprd;
    }

    sum[ibody][0] += xunwrap * massone;
    sum[ibody][1] += yunwrap * massone;
    sum[ibody][2] += zunwrap * massone;
    sum[ibody][3] += massone;
  }

  MPI_Allreduce(sum[0],all[0],6*nbody,MPI_DOUBLE,MPI_SUM,world);

  for (ibody = 0; ibody < nbody; ibody++) {
    masstotal[ibody] = all[ibody][3];
    xcm[ibody][0] = all[ibody][0]/masstotal[ibody];
    xcm[ibody][1] = all[ibody][1]/masstotal[ibody];
    xcm[ibody][2] = all[ibody][2]/masstotal[ibody];
  }

  // set vcm, angmom = 0.0 in case inpfile is used
  // and doesn't overwrite all body's values
  // since setup_bodies_dynamic() will not be called

  for (ibody = 0; ibody < nbody; ibody++) {
    vcm[ibody][0] = vcm[ibody][1] = vcm[ibody][2] = 0.0;
    angmom[ibody][0] = angmom[ibody][1] = angmom[ibody][2] = 0.0;
  }

  // set rigid body image flags to default values

  for (ibody = 0; ibody < nbody; ibody++)
    imagebody[ibody] = ((imageint) IMGMAX << IMG2BITS) |
      ((imageint) IMGMAX << IMGBITS) | IMGMAX;

  // overwrite masstotal, center-of-mass, image flags with file values
  // inbody[i] = 0/1 if Ith rigid body is initialized by file

  int *inbody;
  if (inpfile) {
    // must call it here so it doesn't override read in data but
    // initialize bodies whose dynamic settings not set in inpfile

    setup_bodies_dynamic();

    memory->create(inbody,nbody,"rigid:inbody");
    for (ibody = 0; ibody < nbody; ibody++) inbody[ibody] = 0;
    readfile(0,masstotal,xcm,vcm,angmom,imagebody,inbody,nullptr);
  }

  // remap the xcm of each body back into simulation box
  //   and reset body and atom xcmimage flags via pre_neighbor()

  pre_neighbor();

  // compute 6 moments of inertia of each body in Cartesian reference frame
  // dx,dy,dz = coords relative to center-of-mass
  // symmetric 3x3 inertia tensor stored in Voigt notation as 6-vector

  double dx,dy,dz;

  for (ibody = 0; ibody < nbody; ibody++)
    for (i = 0; i < 6; i++) sum[ibody][i] = 0.0;

  for (i = 0; i < nlocal; i++) {
    if (body[i] < 0) continue;
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

    dx = xunwrap - xcm[ibody][0];
    dy = yunwrap - xcm[ibody][1];
    dz = zunwrap - xcm[ibody][2];

    if (rmass) massone = rmass[i];
    else massone = mass[type[i]];

    sum[ibody][0] += massone * (dy*dy + dz*dz);
    sum[ibody][1] += massone * (dx*dx + dz*dz);
    sum[ibody][2] += massone * (dx*dx + dy*dy);
    sum[ibody][3] -= massone * dy*dz;
    sum[ibody][4] -= massone * dx*dz;
    sum[ibody][5] -= massone * dx*dy;
  }

  // extended particles may contribute extra terms to moments of inertia

  if (extended) {
    double ivec[6];
    double *shape,*quatatom,*inertiaatom;
    double length,theta;

    for (i = 0; i < nlocal; i++) {
      if (body[i] < 0) continue;
      ibody = body[i];
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];

      if (eflags[i] & SPHERE) {
        sum[ibody][0] += SINERTIA*massone * radius[i]*radius[i];
        sum[ibody][1] += SINERTIA*massone * radius[i]*radius[i];
        sum[ibody][2] += SINERTIA*massone * radius[i]*radius[i];
      } else if (eflags[i] & ELLIPSOID) {
        shape = ebonus[ellipsoid[i]].shape;
        quatatom = ebonus[ellipsoid[i]].quat;
        MathExtra::inertia_ellipsoid(shape,quatatom,massone,ivec);
        sum[ibody][0] += ivec[0];
        sum[ibody][1] += ivec[1];
        sum[ibody][2] += ivec[2];
        sum[ibody][3] += ivec[3];
        sum[ibody][4] += ivec[4];
        sum[ibody][5] += ivec[5];
      } else if (eflags[i] & LINE) {
        length = lbonus[line[i]].length;
        theta = lbonus[line[i]].theta;
        MathExtra::inertia_line(length,theta,massone,ivec);
        sum[ibody][0] += ivec[0];
        sum[ibody][1] += ivec[1];
        sum[ibody][2] += ivec[2];
        sum[ibody][3] += ivec[3];
        sum[ibody][4] += ivec[4];
        sum[ibody][5] += ivec[5];
      } else if (eflags[i] & TRIANGLE) {
        inertiaatom = tbonus[tri[i]].inertia;
        quatatom = tbonus[tri[i]].quat;
        MathExtra::inertia_triangle(inertiaatom,quatatom,massone,ivec);
        sum[ibody][0] += ivec[0];
        sum[ibody][1] += ivec[1];
        sum[ibody][2] += ivec[2];
        sum[ibody][3] += ivec[3];
        sum[ibody][4] += ivec[4];
        sum[ibody][5] += ivec[5];
      }
    }
  }

  MPI_Allreduce(sum[0],all[0],6*nbody,MPI_DOUBLE,MPI_SUM,world);

  // overwrite Cartesian inertia tensor with file values

  if (inpfile) readfile(1,nullptr,all,nullptr,nullptr,nullptr,inbody,nullptr);

  // diagonalize inertia tensor for each body via Jacobi rotations
  // inertia = 3 eigenvalues = principal moments of inertia
  //   request that jacobi3() return them in ascending order,
  ///  so that in 2d last evector is z-axis
  // evectors and exzy_space = 3 evectors = principal axes of rigid body

  int ierror;
  double cross[3];
  double tensor[3][3],evectors[3][3];

  for (ibody = 0; ibody < nbody; ibody++) {
    tensor[0][0] = all[ibody][0];
    tensor[1][1] = all[ibody][1];
    tensor[2][2] = all[ibody][2];
    tensor[1][2] = tensor[2][1] = all[ibody][3];
    tensor[0][2] = tensor[2][0] = all[ibody][4];
    tensor[0][1] = tensor[1][0] = all[ibody][5];

    ierror = MathEigen::jacobi3(tensor,inertia[ibody],evectors,1);
    if (ierror) error->all(FLERR,
                           "Insufficient Jacobi rotations for rigid body");

    ex_space[ibody][0] = evectors[0][0];
    ex_space[ibody][1] = evectors[1][0];
    ex_space[ibody][2] = evectors[2][0];
    ey_space[ibody][0] = evectors[0][1];
    ey_space[ibody][1] = evectors[1][1];
    ey_space[ibody][2] = evectors[2][1];
    ez_space[ibody][0] = evectors[0][2];
    ez_space[ibody][1] = evectors[1][2];
    ez_space[ibody][2] = evectors[2][2];

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

    // create initial quaternion

    MathExtra::exyz_to_q(ex_space[ibody],ey_space[ibody],ez_space[ibody],
                         quat[ibody]);
  }

  // displace = initial atom coords in basis of principal axes
  // set displace = 0.0 for atoms not in any rigid body
  // for extended particles, set their orientation wrt to rigid body

  double qc[4],delta[3];
  double *quatatom;
  double theta_body;

  for (i = 0; i < nlocal; i++) {
    if (body[i] < 0) {
      displace[i][0] = displace[i][1] = displace[i][2] = 0.0;
      continue;
    }

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

    if (extended) {
      if (eflags[i] & ELLIPSOID) {
        quatatom = ebonus[ellipsoid[i]].quat;
        MathExtra::qconjugate(quat[ibody],qc);
        MathExtra::quatquat(qc,quatatom,orient[i]);
        MathExtra::qnormalize(orient[i]);
      } else if (eflags[i] & LINE) {
        if (quat[ibody][3] >= 0.0) theta_body = 2.0*acos(quat[ibody][0]);
        else theta_body = -2.0*acos(quat[ibody][0]);
        orient[i][0] = lbonus[line[i]].theta - theta_body;
        while (orient[i][0] <= -MY_PI) orient[i][0] += MY_2PI;
        while (orient[i][0] > MY_PI) orient[i][0] -= MY_2PI;
        if (orientflag == 4) orient[i][1] = orient[i][2] = orient[i][3] = 0.0;
      } else if (eflags[i] & TRIANGLE) {
        quatatom = tbonus[tri[i]].quat;
        MathExtra::qconjugate(quat[ibody],qc);
        MathExtra::quatquat(qc,quatatom,orient[i]);
        MathExtra::qnormalize(orient[i]);
      } else if (orientflag == 4) {
        orient[i][0] = orient[i][1] = orient[i][2] = orient[i][3] = 0.0;
      } else if (orientflag == 1)
        orient[i][0] = 0.0;

      if (eflags[i] & DIPOLE) {
        MathExtra::transpose_matvec(ex_space[ibody],ey_space[ibody],
                                    ez_space[ibody],mu[i],dorient[i]);
        MathExtra::snormalize3(mu[i][3],dorient[i],dorient[i]);
      } else if (dorientflag)
        dorient[i][0] = dorient[i][1] = dorient[i][2] = 0.0;
    }
  }

  // test for valid principal moments & axes
  // recompute moments of inertia around new axes
  // 3 diagonal moments should equal principal moments
  // 3 off-diagonal moments should be 0.0
  // extended particles may contribute extra terms to moments of inertia

  for (ibody = 0; ibody < nbody; ibody++)
    for (i = 0; i < 6; i++) sum[ibody][i] = 0.0;

  for (i = 0; i < nlocal; i++) {
    if (body[i] < 0) continue;
    ibody = body[i];
    if (rmass) massone = rmass[i];
    else massone = mass[type[i]];

    sum[ibody][0] += massone *
      (displace[i][1]*displace[i][1] + displace[i][2]*displace[i][2]);
    sum[ibody][1] += massone *
      (displace[i][0]*displace[i][0] + displace[i][2]*displace[i][2]);
    sum[ibody][2] += massone *
      (displace[i][0]*displace[i][0] + displace[i][1]*displace[i][1]);
    sum[ibody][3] -= massone * displace[i][1]*displace[i][2];
    sum[ibody][4] -= massone * displace[i][0]*displace[i][2];
    sum[ibody][5] -= massone * displace[i][0]*displace[i][1];
  }

  if (extended) {
    double ivec[6];
    double *shape,*inertiaatom;
    double length;

    for (i = 0; i < nlocal; i++) {
      if (body[i] < 0) continue;
      ibody = body[i];
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];

      if (eflags[i] & SPHERE) {
        sum[ibody][0] += SINERTIA*massone * radius[i]*radius[i];
        sum[ibody][1] += SINERTIA*massone * radius[i]*radius[i];
        sum[ibody][2] += SINERTIA*massone * radius[i]*radius[i];
      } else if (eflags[i] & ELLIPSOID) {
        shape = ebonus[ellipsoid[i]].shape;
        MathExtra::inertia_ellipsoid(shape,orient[i],massone,ivec);
        sum[ibody][0] += ivec[0];
        sum[ibody][1] += ivec[1];
        sum[ibody][2] += ivec[2];
        sum[ibody][3] += ivec[3];
        sum[ibody][4] += ivec[4];
        sum[ibody][5] += ivec[5];
      } else if (eflags[i] & LINE) {
        length = lbonus[line[i]].length;
        MathExtra::inertia_line(length,orient[i][0],massone,ivec);
        sum[ibody][0] += ivec[0];
        sum[ibody][1] += ivec[1];
        sum[ibody][2] += ivec[2];
        sum[ibody][3] += ivec[3];
        sum[ibody][4] += ivec[4];
        sum[ibody][5] += ivec[5];
      } else if (eflags[i] & TRIANGLE) {
        inertiaatom = tbonus[tri[i]].inertia;
        MathExtra::inertia_triangle(inertiaatom,orient[i],massone,ivec);
        sum[ibody][0] += ivec[0];
        sum[ibody][1] += ivec[1];
        sum[ibody][2] += ivec[2];
        sum[ibody][3] += ivec[3];
        sum[ibody][4] += ivec[4];
        sum[ibody][5] += ivec[5];
      }
    }
  }

  MPI_Allreduce(sum[0],all[0],6*nbody,MPI_DOUBLE,MPI_SUM,world);

  // error check that re-computed moments of inertia match diagonalized ones
  // do not do test for bodies with params read from inpfile

  double norm;
  for (ibody = 0; ibody < nbody; ibody++) {
    if (inpfile && inbody[ibody]) continue;
    if (inertia[ibody][0] == 0.0) {
      if (fabs(all[ibody][0]) > TOLERANCE)
        error->all(FLERR,"Fix rigid: Bad principal moments");
    } else {
      if (fabs((all[ibody][0]-inertia[ibody][0])/inertia[ibody][0]) >
          TOLERANCE) error->all(FLERR,"Fix rigid: Bad principal moments");
    }
    if (inertia[ibody][1] == 0.0) {
      if (fabs(all[ibody][1]) > TOLERANCE)
        error->all(FLERR,"Fix rigid: Bad principal moments");
    } else {
      if (fabs((all[ibody][1]-inertia[ibody][1])/inertia[ibody][1]) >
          TOLERANCE) error->all(FLERR,"Fix rigid: Bad principal moments");
    }
    if (inertia[ibody][2] == 0.0) {
      if (fabs(all[ibody][2]) > TOLERANCE)
        error->all(FLERR,"Fix rigid: Bad principal moments");
    } else {
      if (fabs((all[ibody][2]-inertia[ibody][2])/inertia[ibody][2]) >
          TOLERANCE) error->all(FLERR,"Fix rigid: Bad principal moments");
    }
    norm = (inertia[ibody][0] + inertia[ibody][1] + inertia[ibody][2]) / 3.0;
    if (fabs(all[ibody][3]/norm) > TOLERANCE ||
        fabs(all[ibody][4]/norm) > TOLERANCE ||
        fabs(all[ibody][5]/norm) > TOLERANCE)
      error->all(FLERR,"Fix rigid: Bad principal moments");
  }

  // Read the level-set information from gridfiles
  if (inpfile) {
    char **ls_grid_files;
    double *scale;
    memory->create(scale,nbody,"rigid/ls/dem:scale"); // TODO: I copied this pattern from inbody, not sure why we cannot do double scale[nbody] ?
    memory->create(ls_grid_files,nbody,MAXLINE,"rigid/ls/dem:ls_grid_files");

    // Read scaling factors and gridfiles names
    readfile(2,scale,nullptr,nullptr,nullptr,nullptr,inbody,ls_grid_files);

    // Read grid dimensions for all bodies
    read_gridfile(0,ls_grid_files,scale);
    int *ngrid_flat;
    memory->create(ngrid_flat,nbody,"rigid/ls/dem:ngrid_flat");
    for (int ibody = 0; ibody < nbody ; ibody++) {
      ngrid_flat[ibody] = 1;
      for (int idim = 0 ; idim < domain->dimension ; idim++)
        ngrid_flat[ibody] *= ngrid[ibody][idim];
    }

    // Create grid_ls_val from dimensions read into ngrid by read_gridfile()
    // This cannot be done before reading gridfiles, e.g., in the constructor where we create ngrid
    memory->create_ragged(grid_ls_val, nbody, ngrid_flat, "rigid/ls/dem:grid_ls_val");

    // Read and scale level-set values for all bodies (requires grid_ls_val to be sized correctly)
    read_gridfile(1,ls_grid_files,scale);

    memory->destroy(ls_grid_files);
    memory->destroy(scale);
    memory->destroy(ngrid_flat);
  }

  if (inpfile) memory->destroy(inbody);
}

/* ----------------------------------------------------------------------
   read per rigid body info from user-provided file
   which = 0 to read everything except 6 moments of inertia
   which = 1 to read 6 moments of inertia
   which = 2 to read LSDEM scaling and gridfile
   flag inbody = 0 for bodies whose info is read from file
   nlines = # of lines of rigid body info
   one line = rigid-ID mass xcm ycm zcm ixx iyy izz ixy ixz iyz
              vxcm vycm vzcm lx ly lz ix iy iz
------------------------------------------------------------------------- */

void FixRigidLSDEM::readfile(int which, double *vec, double **array1, double **array2, double **array3,
                        imageint *ivec, int *inbody, char** gridfiles)
{
  int nchunk,id,eofflag,xbox,ybox,zbox;
  int nlines;
  FILE *fp;
  char *eof,*start,*next,*buf;
  char line[MAXLINE] = {'\0'};

  // open file and read and parse first non-empty, non-comment line containing the number of bodies
  if (comm->me == 0) {
    fp = fopen(inpfile,"r");
    if (fp == nullptr)
      error->one(FLERR,"Cannot open fix rigid infile {}: {}", inpfile, utils::getsyserror());
    while (true) {
      eof = fgets(line,MAXLINE,fp);
      if (eof == nullptr) error->one(FLERR,"Unexpected end of fix rigid infile");
      start = &line[strspn(line," \t\n\v\f\r")];
      if (*start != '\0' && *start != '#') break;
    }
    nlines = utils::inumeric(FLERR, utils::trim(line), true, lmp);
    if (which == 0)
      utils::logmesg(lmp, "Reading rigid body data for {} bodies from file {}\n", nlines, inpfile);
    if (nlines == 0) fclose(fp);
  }
  MPI_Bcast(&nlines,1,MPI_INT,0,world);

  // empty file with 0 lines is needed to trigger initial restart file
  // generation when no infile was previously used.

  if (nlines == 0) return;
  else if (nlines < 0) error->all(FLERR,"Fix rigid infile has incorrect format");

  auto buffer = new char[CHUNK*MAXLINE];
  int nread = 0;
  int me = comm->me;
  while (nread < nlines) {
    nchunk = MIN(nlines-nread,CHUNK);
    eofflag = utils::read_lines_from_file(fp,nchunk,MAXLINE,buffer,me,world);
    if (eofflag) error->all(FLERR,"Unexpected end of fix rigid infile");

    buf = buffer;
    next = strchr(buf,'\n');
    *next = '\0';
    int nwords = utils::count_words(utils::trim_comment(buf));
    *next = '\n';

    if (nwords != ATTRIBUTE_PERBODY)
      error->all(FLERR,"Incorrect rigid body format in fix rigid file");

    // loop over lines of rigid body attributes
    // tokenize the line into values
    // id = rigid body ID
    // use ID as-is for SINGLE, as mol-ID for MOLECULE, as-is for GROUP
    // for which = 0, store all but inertia in vecs and arrays
    // for which = 1, store inertia tensor array, invert 3,4,5 values to Voigt

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
          throw TokenizerException("invalid_rigid body ID ", std::to_string(id+1));

        inbody[id] = 1;

        if (which == 0) {
          vec[id] = values.next_double();
          array1[id][0] = values.next_double();
          array1[id][1] = values.next_double();
          array1[id][2] = values.next_double();
          values.skip(6);
          array2[id][0] = values.next_double();
          array2[id][1] = values.next_double();
          array2[id][2] = values.next_double();
          array3[id][0] = values.next_double();
          array3[id][1] = values.next_double();
          array3[id][2] = values.next_double();
          xbox = values.next_int();
          ybox = values.next_int();
          zbox = values.next_int();
          ivec[id] = ((imageint) (xbox + IMGMAX) & IMGMASK) |
            (((imageint) (ybox + IMGMAX) & IMGMASK) << IMGBITS) |
            (((imageint) (zbox + IMGMAX) & IMGMASK) << IMG2BITS);
        } else if (which == 1) {
          values.skip(4);
          array1[id][0] = values.next_double();
          array1[id][1] = values.next_double();
          array1[id][2] = values.next_double();
          array1[id][5] = values.next_double();
          array1[id][4] = values.next_double();
          array1[id][3] = values.next_double();
        } else if (which == 2) {
          values.skip(19);
          vec[id] = values.next_double();
          strcpy(gridfiles[id], values.next_string().data()); // TODO: I'm not up to date on C-string vs std::string in LAMMPS. Possible important refactor here with std::vector<string> instead of char**
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
   TODO: left my editor with 4-space tab, LAMMPS style is 2-space tab
   TODO: this assumes all rigid bodies are LSDEM grains. Otherwise, we should pass and read *inbody.
        Not sure if there is a use for this: why would fix rigid lsdem have tigid bodies not be LSDEM ?
        Refactor readfile() accordingly if this is the route we take
------------------------------------------------------------------------- */

void FixRigidLSDEM::read_gridfile(int which, char **ls_grid_files, double* scale)
{
  int dim = domain->dimension;
  int grid_shape_buf[dim];
  double grid_size_buf[dim + 1];
  int nchunk,eofflag;
  FILE *fp;
  char *eof,*start,*next,*buf;
  char line[MAXLINE] = {'\0'};

  // open file and read and parse first non-empty, non-comment line containing the 2 or 3 grid dimensions
  // Broadcast to other procs
  // TODO: there must be a better way to read the first 2,3 lines
  for (int ibody = 0 ; ibody < nbody ; ibody++) {
    int nlines = 1;
    char* gridfile = ls_grid_files[ibody];
    if (comm->me == 0) {
      fp = fopen(gridfile,"r");
      if (fp == nullptr)
        error->one(FLERR,"Cannot open fix rigid/ls/dem gridfile {}: {}", gridfile, utils::getsyserror());
      while (true) {
        eof = fgets(line,MAXLINE,fp);
        if (eof == nullptr) error->one(FLERR,"Unexpected end of fix rigid/ls/dem gridfile");
        start = &line[strspn(line," \t\n\v\f\r")];
        if (*start != '\0' && *start != '#') break;
      }
      auto grid_shape = utils::split_words(line);
      if (grid_shape.size() != dim)
        error->one(FLERR,"Fix rigid/ls/dem gridfile {} has {} dimensions but simulation is {}D",
                            gridfile, grid_shape.size(), dim);
      for (int idim = 0 ; idim < dim ; idim++)
        grid_shape_buf[idim] = utils::inumeric(FLERR, grid_shape[idim], false, lmp);

      eof = fgets(line,MAXLINE,fp);
      if (eof == nullptr) error->one(FLERR,"Unexpected end of fix rigid/ls/demgridfile");
      grid_size_buf[0] = utils::numeric(FLERR, utils::trim(line), false, lmp);
      if (grid_size_buf[0] <= 0.0)
        error->one(FLERR,"Grid stride for rigid/ls/dem gridfile {} must be positive",gridfile);

      eof = fgets(line,MAXLINE,fp);
      if (eof == nullptr) error->one(FLERR,"Unexpected end of fix rigid/ls/demgridfile");
      auto grid_corner = utils::split_words(line);
      if (grid_corner.size() != dim)
        error->one(FLERR,"Fix rigid/ls/dem gridfile {} specifies {} grid corner cooridnates but simulation is {}D",
                            gridfile, grid_corner.size(), dim);
      for (int idim = 0 ; idim < dim ; idim++)
        grid_size_buf[idim + 1] = utils::numeric(FLERR, grid_corner[idim], false, lmp);
      utils::logmesg(lmp, "Reading ls/dem grid data for body {} from file {}\n", ibody, gridfile);
    }
    MPI_Bcast(grid_shape_buf, dim, MPI_INT, 0, world);
    MPI_Bcast(grid_size_buf, dim + 1, MPI_DOUBLE, 0, world);

    for (int idim = 0 ; idim < dim ; idim++)
      nlines *= grid_shape_buf[idim];

    // TODO: I left the 2 lines below from original rigid::readline() not sure if needed
    // empty file with 0 lines is needed to trigger initial restart file
    // generation when no infile was previously used.
    if (nlines == 0) return;
    else if (nlines < 0) error->all(FLERR,"Fix rigid/ls/dem gridfile has incorrect format");

    if (which == 0) {
      grid_stride[ibody] = grid_size_buf[0] * scale[ibody];
      for (int idim = 0 ; idim < dim ; idim++) {
        grid_min[ibody][idim] = grid_size_buf[idim + 1] * scale[ibody];
        ngrid[ibody][idim] = grid_shape_buf[idim];
      }
    } else {
      auto buffer = new char[CHUNK*MAXLINE];
      int nread = 0;
      int me = comm->me;
      while (nread < nlines) {
        nchunk = MIN(nlines-nread,CHUNK);
        eofflag = utils::read_lines_from_file(fp,nchunk,MAXLINE,buffer,me,world);
        if (eofflag) error->all(FLERR,"Unexpected end of fix rigid/ls/dem gridfile");

        buf = buffer;
        next = strchr(buf,'\n');
        *next = '\0';
        int nwords = utils::count_words(utils::trim_comment(buf));
        *next = '\n';

        // TODO: there must be a better way than tokenizing single value
        // Kept as is for now to re-use existing rigid::readfile() code
        // Maybe in the future we want to have multiple value per line,
        // In which case it will be useful to have that architecture
        if (nwords != 1)
          error->all(FLERR,"LSDEM gridfile format requires one entry per line");

        // loop over lines of level set grid and tokenize level set values
        for (int i = 0; i < nchunk; i++) {
          next = strchr(buf,'\n');
          *next = '\0';

          try {
            ValueTokenizer values(buf);
            grid_ls_val[ibody][nread+i] = values.next_double() * scale[ibody];
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

// End of file