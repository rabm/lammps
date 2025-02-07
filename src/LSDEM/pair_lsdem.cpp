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

#include "pair_lsdem.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix_rigid.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>
#include <unordered_map>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLSDEM::PairLSDEM(LAMMPS *_lmp) : Pair(_lmp), k(nullptr), cut(nullptr), gamma(nullptr)
{
  writedata = 1;
  id_fix = nullptr;
}

/* ---------------------------------------------------------------------- */

PairLSDEM::~PairLSDEM()
{
  if (id_fix && modify->nfix) modify->delete_fix(id_fix);
  delete[] id_fix;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(k);
    memory->destroy(cut);
    memory->destroy(gamma);
  }
}

/* ---------------------------------------------------------------------- */

void PairLSDEM::compute(int eflag, int vflag)
{
  int i, j, ii, jj, key, inum, jnum, itype, jtype, ibody, jbody;
  tagint itag, jtag;
  double xtmp, ytmp, ztmp, delx, dely, delz, dr, evdwl, fpair;
  double r, rsq, rinv, factor_lj, ls_value;
  int *ilist, *jlist, *numneigh, **firstneigh, calc_force_of_i_on_j, calc_force_of_j_on_i;
  double vxtmp, vytmp, vztmp, delvx, delvy, delvz, dot, smooth;

  evdwl = 0.0;
  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  double *special_lj = force->special_lj;

  std::unordered_map<int, std::pair<int, double>> min_distances;

  auto fixlist = modify->get_fix_by_style("rigid");
  if (fixlist.size() != 1)
    error->all(FLERR, "Must have one instance of fix rigid for pair lsdem");
  auto fixrigid = dynamic_cast<FixRigid *>(fixlist.front());
  int *body = fixrigid->get_body_array();
  int nbody = fixrigid->get_nbody();

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop to find closest neighbors

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    ibody = body[i];
    itag = tag[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];

      if (factor_lj == 0) continue;

      j &= NEIGHMASK;

      jbody = body[j];
      jtag = tag[j];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      r = sqrt(rsq);

      key = nbody * itag + ibody;
      // If first interation between atom i and j's grain, create entry
      if (min_distances.find(key) == min_distances.end()) {
        min_distances[key] = std::make_pair(jtag, r);
      } else {
        // overwrite if ij are closer
        if (r < min_distances[key].second)
          min_distances[key] = std::make_pair(jtag, r);
      }

      // do the same for atom j
      if (newton_pair || j < nlocal) {
        key = nbody * jtag + jbody;
        if (min_distances.find(key) == min_distances.end()) {
          min_distances[key] = std::make_pair(itag, r);
        } else {
          // overwrite if ij are closer
          if (r < min_distances[key].second)
            min_distances[key] = std::make_pair(itag, r);
        }
      }
    }
  }

  // if (newton_pair) need to somehow reverse comm distances...

  // loop to calculate forces

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    itype = type[i];
    ibody = body[i];
    itag = tag[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];

      if (factor_lj == 0) continue;

      j &= NEIGHMASK;
      jbody = body[j];
      jtag = tag[j];

      // if node j is closest on its grain to i
      calc_force_of_j_on_i = 0;
      key = nbody * jtag + jbody;
      if (min_distances.find(key) != min_distances.end())
        calc_force_of_j_on_i = 1;

      // if node i is closest on its grain to j & j is owned
      calc_force_of_i_on_j = 0;
      if (newton_pair || j < nlocal) {
        key = nbody * itag + ibody;
        if (min_distances.find(key) != min_distances.end())
          calc_force_of_i_on_j = 1;
      }

      // Neither is closest
      if (calc_force_of_i_on_j + calc_force_of_j_on_i == 1) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      jtype = type[j];

      // calculate other joint properties...

      if (calc_force_of_i_on_j) {
        ls_value = get_ls_value(delx, dely, delz, i, j);
        // calculate force...
        fpair = 0.0;
        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
      }

      if (calc_force_of_i_on_j) {
        ls_value = get_ls_value(-delx, -dely, -delz, j, i);
        // calculate force...
        fpair = 0.0;
        f[j][0] -= delx * fpair;
        f[j][1] -= dely * fpair;
        f[j][2] -= delz * fpair;
      }

      // virial contribution TBD
      // if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLSDEM::allocate()
{
  allocated = 1;
  const int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  for (int i = 1; i < np1; i++)
    for (int j = i; j < np1; j++) setflag[i][j] = 0;

  memory->create(cutsq, np1, np1, "pair:cutsq");

  memory->create(k, np1, np1, "pair:k");
  memory->create(cut, np1, np1, "pair:cut");
  memory->create(gamma, np1, np1, "pair:gamma");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLSDEM::settings(int narg, char ** arg)
{
  int iarg = 0;
  while (iarg < narg) {
    error->all(FLERR, "Illegal pair_style command {}", arg[iarg]);
  }

  if (id_fix && modify->nfix) modify->delete_fix(id_fix);

  if (!id_fix)
    id_fix = utils::strdup(std::string("PAIR_LS_DEM") + std::to_string(instance_me));

  if (force->newton)
    error->all(FLERR, "Temporarily do not support newton on with LS DEM");

  nrow = 21;
  ncol = 21;
  double l_grid = 1.0;
  double x_com = 10.5;
  double y_com = 10.5;
  double r = 10.0;

  ngrid = nrow * ncol;

  modify->add_fix(fmt::format("{} all property/atom d2_ls_dem_grid {} writedata no ghost yes", id_fix, ngrid));
  int tmp1, tmp2;
  index_ls_dem_grid = atom->find_custom("ls_dem_grid", tmp1, tmp2);
  index_ls_dem_com = atom->find_custom("ls_dem_com", tmp1, tmp2);
  index_ls_dem_quat = atom->find_custom("ls_dem_quat", tmp1, tmp2);

  double **ls_dem_grid = atom->darray[index_ls_dem_grid];

  double delx, dely;
  for (int i = 0; i < atom->nlocal; i++) {
    for (int a = 0; a < ncol; a++) {
      for (int b = 0; b < nrow; b++) {
        // stored value is at lower left corner of grid
        delx = a * l_grid - x_com;
        dely = b * l_grid - y_com;
        ls_dem_grid[i][b * ncol + a] = sqrt(delx * delx + dely * dely) - r;

        //printf("%.3g ", ls_dem_grid[i][b * ncol + a]);
      }
      //printf("\n");
    }
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLSDEM::coeff(int narg, char **arg)
{
  if (narg != 5)
    error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double k_one = utils::numeric(FLERR, arg[2], false, lmp);
  double cut_one = utils::numeric(FLERR, arg[3], false, lmp);
  double gamma_one = utils::numeric(FLERR, arg[4], false, lmp);

  if (cut_one <= 0.0) error->all(FLERR, "Incorrect args for pair coefficients");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      k[i][j] = k_one;
      cut[i][j] = cut_one;
      gamma[i][j] = gamma_one;

      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLSDEM::init_style()
{
  if (comm->ghost_velocity == 0)
    error->all(FLERR, "Pair bpm/spring requires ghost atoms store velocity");

  neighbor->add_request(this);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLSDEM::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
    k[i][j] = mix_energy(k[i][i], k[j][j], cut[i][i], cut[j][j]);
    gamma[i][j] = mix_energy(gamma[i][i], gamma[j][j], cut[i][i], cut[j][j]);
  }

  cut[j][i] = cut[i][j];
  k[j][i] = k[i][j];
  gamma[j][i] = gamma[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLSDEM::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&k[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
        fwrite(&gamma[i][j], sizeof(double), 1, fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLSDEM::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &k[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &gamma[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&k[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&gamma[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}


/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLSDEM::write_restart_settings(FILE *fp)
{
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLSDEM::read_restart_settings(FILE *fp)
{
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLSDEM::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp, "%d %g %g %g\n", i, k[i][i], cut[i][i], gamma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLSDEM::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g\n", i, j, k[i][j], cut[i][j], gamma[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairLSDEM::single(int i, int j, int itype, int jtype, double rsq, double /*factor_coul*/,
                             double factor_lj, double &fforce)
{
  double fpair, r, rinv, dr;
  double delx, dely, delz, delvx, delvy, delvz, dot, smooth;

  if (rsq > cutsq[itype][jtype]) return 0.0;

  double **x = atom->x;
  double **v = atom->v;

  r = sqrt(rsq);
  rinv = 1.0 / r;

  dr = r - cut[itype][jtype];
  fpair = -k[itype][jtype] * dr;

  smooth = rsq / cutsq[itype][jtype];
  smooth *= smooth;
  smooth = 1.0 - smooth;
  delx = x[i][0] - x[j][0];
  dely = x[i][1] - x[j][1];
  delz = x[i][2] - x[j][2];
  delvx = v[i][0] - v[j][0];
  delvy = v[i][1] - v[j][1];
  delvz = v[i][2] - v[j][2];
  dot = delx * delvx + dely * delvy + delz * delvz;
  fpair -= gamma[itype][jtype] * dot * rinv * smooth;

  fpair *= factor_lj;
  fforce = fpair;

  return 0.0;
}

/* ----------------------------------------------------------------------
   Find the value of node (atom) i in j's LS grid, (dx, dy, dz) = xi - xj,
   where xi and xj are the centre coordinates of i and j, respectively.
------------------------------------------------------------------------- */

double PairLSDEM::get_ls_value(double dx, double dy, double dz, int i, int j)
{
  // dx, dy, and dz will not be used. Maybe remove?
  double **x = atom->x;
  double **grain_com = atom->darray[index_ls_dem_com];
  double **grain_quat = atom->darray[index_ls_dem_quat];
  double **grain_grid = atom->darray[index_ls_dem_grid];

  int nrow_offset = 0; // Offsets for local subgrid, to implement later
  int ncol_offset = 0; //   currently subgrid = grid, so offsets are zero
  int nslice_offset = 0;

  // Calculate position of i in j's grid using:
  //   x[i][0-2] = location of i
  //   x[j][0-2] = location of j
  //   grain_com[j][0-2] = CoM of j's grain
  //   grain_quat[j][0-3] = quat of j's grain

  //
  //  GET NODE I IN LOCAL COORDINATES OF J GRAIN
  //

  // Relative coordinate of node i w.r.t. centre of mass grain j
  // Danny: I will asumme x is in global coordinates.
  delx = x[i][0]-grain_com[j][0];
  dely = x[i][1]-grain_com[j][1];
  delz = 0; // x[i][2]-grain_com[j][2];

  // Extract quaternion components
  // Danny: How is grain_quat defined? Is it the rotation local -> global or global -> local?
  // I will assume it is local -> global 
  // Remove minus signs if grain_quat is global -> local
  double q_w = grain_quat[j][0];
  double q_x = -grain_quat[j][1];
  double q_y = -grain_quat[j][2];
  double q_z = -grain_quat[j][3];

  // Apply quaternion rotation to move into local reference frame of grain j grid
  // x' = (q*x)*q^-1
  double x_local = (1 - 2 * (q_y * q_y + q_z * q_z)) * delx + 2 * (q_x * q_y - q_w * q_z) * dely + 2 * (q_x * q_z + q_w * q_y) * delz;
  double y_local = 2 * (q_x * q_y + q_w * q_z) * delx + (1 - 2 * (q_x * q_x + q_z * q_z)) * dely + 2 * (q_y * q_z - q_w * q_x) * delz;
  double z_local = 2 * (q_x * q_z - q_w * q_y) * delx + 2 * (q_y * q_z + q_w * q_x) * dely + (1 - 2 * (q_x * q_x + q_y * q_y)) * delz;

  //
  //  COMPUTE THE LS GRID INDICES
  //

  // Calculate index from coordinate, need to be more careful with integer division
  // Danny: We need to get grid_min, the lowest corner (in -1,-1,-1 direction) of the grid
  //        and spac, the grid spacing. (If we want to keep this in normalised coords, we
  //        will have to normalise )
  int ind_x = int( (x_local - grid_min[0]) / spac ); // Here, int() do the same as floor() + conversion
  int ind_y = int( (y_local - grid_min[1]) / spac );
  // int ind_z = int( (z_local - grid_min[2]) / spac );

  // We might need an extra check. If x_local is very close to grid_min, it may pass and give
  // errors later.

  if ( (ind_x < 0) || (ind_y < 0) ) { // || (indz < 0)
    // Point is outside the LS grid of grain j. Cannot compute distance or normal.
    // Some error message?
    return (NaN);
  }else if ( (ind_x > nrow-1) || (ind_y > ncol-1)  ) {  // || (ind_z > nslice-1)
    // Point is outside the LS grid of grain j. Cannot compute distance or normal.
    // Some error message?
    return (NaN);
  }

  // Apply offsets, there is probably a more proper way
  ind_x = ind_x - nrow_offset;
  ind_y = ind_y - ncol_offset;
  // ind_z = ind_z - nslice_offset;

  //
  //  DO THE BILINEAR INTERPOLATION
  //

  // Coordinates of the grid lower grid point of the cell we are in
  double x0 = grain_grid_coord[j][ind_x + ind_y * ncol][0]; // + ind_z + nslice
  double y0 = grain_grid_coord[j][ind_x + ind_y * ncol][1];
  //double z0 = grain_grid_coord[j][ind_x + ind_y * ncol + ind_z * nslice][2];

  // Level-set values on the grid points
  double ls000 = grain_grid[j][ind_x   + ind_y     * ncol]; // + ind_z * nslice
  double ls100 = grain_grid[j][ind_x+1 + ind_y     * ncol];
  double ls010 = grain_grid[j][ind_x   + (ind_y+1) * ncol];
  double ls110 = grain_grid[j][ind_x+1 + (ind_y+1) * ncol];

  // The reduced coordinates
  // May be safe to cap them with math::max(math::min(x_red, 1.0), 0.0)
  double x_red = (x_local - x0) / spac;
  double y_red = (y_local - y0) / spac;
  double z_red = (z_local - z0) / spac;

  // The bilinear interpolation
  double term = y_red * (ls110 - ls100 - ls010 + ls000) + ls100 - ls000;
  double dist = x_red * term + y_red * (ls010 - ls000) + ls000;

  /*
    FOR 3D

    double ls001 = grain_grid[j][ind_x   + ind_y     * ncol + (ind_z+1) * nslice];
    double ls101 = grain_grid[j][ind_x+1 + ind_y     * ncol + (ind_z+1) * nslice];
    double ls011 = grain_grid[j][ind_x   + (ind_y+1) * ncol + (ind_z+1) * nslice];
    double ls111 = grain_grid[j][ind_x+1 + (ind_y+1) * ncol + (ind_z+1) * nslice];

    // Exactly the same but for other z plane / face of the grid cell
    double term = y_red * (ls111 - ls101 - ls011 + ls001) + ls101 - ls001;
    double dist_xy1 = x_red / spac * term + y_red * (ls011 - ls001) + ls001;

	  dist = z_red * (dist_xy1 - dist_xy0) + dist_xy0;
  */

  // Normal
  double nx = 0;
  double ny = 0;
  double nz = 0;

	// Computing normal as the gradient of trilinear interpolation
	for (int a = 0; a < 2; a++) {
		for (int b = 0; b < 2; b++) {
			for (int c = 0; c < 2; c++) {
				double lsVal = grain_grid[j][(ind_x + a) + (ind_x + b)*ncol + (ind_z + c)*nslice];
				nx += lsVal * (2 * a - 1) * ((1 - b) * (1 - y_red) + b * y_red) * ((1 - c) * (1 - z_red) + c * z_red);
				ny += lsVal * (2 * b - 1) * ((1 - a) * (1 - x_red) + a * x_red) * ((1 - c) * (1 - z_red) + c * z_red);
				nz += lsVal * (2 * c - 1) * ((1 - a) * (1 - x_red) + a * x_red) * ((1 - b) * (1 - y_red) + b * y_red);
			}
		}
	}
	// May be safer to normalise the normal
  // Not sure how you usually make arrays / vectors in LAMMPS.
  // double** normal;
  // normal = new double*[3]
  // normal[0] = nx; normal[1] = ny; normal[2] = nz;

  return dist //, normal
}



