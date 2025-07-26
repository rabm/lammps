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

#include "pair_ls_dem.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_rigid_ls_dem.h"
#include "force.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>
#include <unordered_map>

static constexpr double EPSILON = 1e-10;

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairLSDEM::PairLSDEM(LAMMPS *_lmp) : Pair(_lmp), k(nullptr), cut(nullptr), gamma(nullptr), fix_rigid(nullptr)
{
  writedata = 1;
  single_enable = 0;
}

/* ---------------------------------------------------------------------- */

PairLSDEM::~PairLSDEM()
{
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
  int i, j, ii, jj, key, allnum, inum, jnum, itype, jtype, ibody, jbody;
  tagint itag, jtag;
  double xtmp, ytmp, ztmp, delx, dely, delz, dr, evdwl;
  double r, rsq, rinv, factor_lj, u, ivol, jvol;
  int *ilist, *jlist, *numneigh, **firstneigh, calc_force_of_i_on_j, calc_force_of_j_on_i;
  double vxtmp, vytmp, vztmp, delvx, delvy, delvz, dot, smooth;
  double normal[3], fpair_mag, fpair[3], contact_point[3], lever[3], torque_pair[3];

  // Currently require:
  //   Newton pair off.
  //   Exclude intramolecule interactions
  // In future, remove restrictions

  evdwl = 0.0;
  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **torque = atom->torque;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  double **grain_com = atom->darray[index_ls_dem_com]; // Need CoM for torques
  double *grain_vol = atom->dvector[index_ls_dem_vol];
  std::unordered_map<int, std::pair<int, double>> min_distances;

  int *body = fix_rigid->get_body_array();
  int nbody = fix_rigid->get_nbody();

  inum = list->inum;
  allnum = inum + list->gnum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // Loop over local+ghost atoms to find closest neighbors
  for (ii = 0; ii < allnum; ii++) {
    // Loop through local nodes
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    ibody = body[i];
    itag = tag[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      // Loop through neighbouring nodes
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];

      if (factor_lj == 0) continue;

      j &= NEIGHMASK; // Danny: What does this do?

      jbody = body[j];
      jtag = tag[j];

      // Separation distance between the two nodes
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      r = sqrt(rsq);

      if (r > maxcut) continue;

      // What does this code do exactly? Does this make both neighbour lists
      // min_distance(i) = j and min_distance(j) = i?

      key = nbody * itag + jbody;
      // If first interation between i and j's grain, create entry
      if (min_distances.find(key) == min_distances.end()) {
        min_distances[key] = std::make_pair(jtag, r);
      } else {
        // Overwrite if i and j are closer
        if (r < min_distances[key].second)
          min_distances[key] = std::make_pair(jtag, r);
      }

      // Do the same for node j
      key = nbody * jtag + ibody;
      if (min_distances.find(key) == min_distances.end()) {
        min_distances[key] = std::make_pair(itag, r);
      } else {
        // Overwrite if i and j are closer
        if (r < min_distances[key].second)
          min_distances[key] = std::make_pair(itag, r);
      }
    }
  }

  // only loop over local atoms to calculate forces
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
    ivol = grain_vol[i];
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
      jvol = grain_vol[j];
      jtype = type[j];

      // Figure out whether to use the nodes of grain i or j.
      // We use the nodes on the smaller grain since this will
      // be more accurate. If the volumes are tied, use the grain ID.
      // Only calculate force between closest pair of node and grid.
      // Danny: Can we reduce the above nieghbour search making use of this knowledge?

      calc_force_of_j_on_i = 0;
      calc_force_of_i_on_j = 0;

      // Danny: I'm a bit confused here, it looks as if we use j nodes if i is smaller, which is the opposite of what we want.
      if (ivol < jvol || (ivol == jvol && ibody < jbody)) {
        // if node j is closest on its grain to i
        key = nbody * itag + jbody;
        if (min_distances.find(key) != min_distances.end())
          if (jtag == min_distances[key].first)
            calc_force_of_j_on_i = 1;
      } else {
        // if node i is closest on its grain to j & j is owned
        key = nbody * jtag + ibody;
        if (min_distances.find(key) != min_distances.end())
          if (itag == min_distances[key].first)
            calc_force_of_i_on_j = 1;
      }

      // If no forces are calculated
      if (calc_force_of_i_on_j + calc_force_of_j_on_i == 0) continue;

      // Evaluate the level set, and assign the interaction direction based on the
      // node-grain combination. Force magnitude and direction go i -> j by definition.
      if (calc_force_of_i_on_j) {
        // Level set is by definition negative inside the particle, 
        // so swap the sign to get the overlap distance.
        u = fix_rigid->get_ls_value(i, j, normal); // Minus
        // The normal is also swapped and points away from j, correct signs.
        MathExtra::negate3(normal);
      } else {
        u = fix_rigid->get_ls_value(j, i, normal); // Minus
        // The normal points towards j, no correction needed.
      }

      // Apply forces and torques

      // No adhesion, cohesion, or ranged forces.
      if (u > 0) continue; // Swap

      // With penetration distance u and normal n (i->j),
      // we have: F_{j on i} = f(ls_value) = - k_n * u * n.
      fpair_mag = k[itype][jtype] * u; // Minus

      // The pair force vector
      fpair[0] = fpair_mag * normal[0];
      fpair[1] = fpair_mag * normal[1];
      fpair[2] = fpair_mag * normal[2];

      // Contact point
      contact_point[0] = xtmp - 0.5 * u * normal[0];
      contact_point[1] = ytmp - 0.5 * u * normal[1];
      contact_point[2] = ztmp - 0.5 * u * normal[2];

      // Force on grain i
      f[i][0] += fpair[0];
      f[i][1] += fpair[1];
      f[i][2] += fpair[2];

      // Lever arm on grain i
      lever[0] = contact_point[0] - grain_com[i][0];
      lever[1] = contact_point[1] - grain_com[i][1];
      lever[2] = contact_point[2] - grain_com[i][2];
      // Account for PBCs
      domain->minimum_image(lever);

      // Compute torque
      MathExtra::cross3(lever, fpair, torque_pair);

      // Apply torques on grain i
      torque[i][0] += torque_pair[0];
      torque[i][1] += torque_pair[1];
      torque[i][2] += torque_pair[2];

      // Mirror forces and torques on grain j
      if (newton_pair || j < nlocal) {
        MathExtra::negate3(fpair); // Sign swap
        f[j][0] += fpair[0];
        f[j][1] += fpair[1];
        f[j][2] += fpair[2];

        lever[0] = contact_point[0] - grain_com[j][0];
        lever[1] = contact_point[1] - grain_com[j][1];
        lever[2] = contact_point[2] - grain_com[j][2];
        domain->minimum_image(lever);

        MathExtra::cross3(lever, fpair, torque_pair);

        torque[j][0] += torque_pair[0];
        torque[j][1] += torque_pair[1];
        torque[j][2] += torque_pair[2];
      }

      // virial contribution: need to check
      if (evflag) ev_tally(i, j, nlocal, 0, evdwl, 0.0, fpair_mag, normal[0], normal[1], normal[2]);
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
  if (narg != 1)
    error->all(FLERR, "Illegal pair_style command");

  maxcut = utils::numeric(FLERR, arg[0], false, lmp);

  if (force->newton_pair)
    error->all(FLERR, "Temporarily do not support newton pair on with LS/DEM");

//   nrow = 20;
//   ncol = 20;
//   nslice = 1;
//   double l_grid = 0.5;
//   double x_com = 5.25;
//   double y_com = 5.25;
//   double r = 2.5;

//   ngrid = nrow * ncol;
//   spac = l_grid;

  /*

  // Volume integration
  double **grain_grid = atom->darray[index_ls_grid];
  double **grain_grid_x = atom->darray[index_ls_gridx];
  double **grain_grid_y = atom->darray[index_ls_gridy];
  double **grain_grid_z = atom->darray[index_ls_gridz];

  // This is the reference distance values that determines the smearing with of
  // the Heaviside step function. Current expression is the half-diagional of the
  // grid cell divided by a smearing constant.
  double smearCoeff = 1.5;
  double ls_ref = 0.0;
  if (smearCoeff != 0){
    ls_ref = sqrt(0.75) * spac / smearCoeff;
  }
  // Initialise volume and centre of mass
  double volume = 0.0, x_com = 0.0, y_com = 0.0, z_com = 0.0;
  // Cell volume, temporary grid points, integration volume.
  double volume_cell = spac*spac*spac;
  double x_grid, y_grid, z_grid, dV = 0.0;

  // Integration
	for (int ind_x = 0; ind_x < nrow; xIndex++){
		for (int ind_y = 0; ind_y < ncol; yIndex++){
			for (int ind_z = 0; ind_z < nslice; zIndex++){
        ls_val = grain_grid[ind_x + ind_y * nrow + ind_z * nrow * ncol];
				if (abs(ls_val) < ls_ref){
          // Close to boundary if abs(ls_val) < ls_ref, apply smearing.
					dV = smearedHeavisideStep(-ls_val/ls_ref)* volume_cell;
        }else if (ls_val < 0){
          // Inside and far away from boundary
					dV = volume_cell;
        }else if (ls_val > 0){
          // Outside and far away from boundary
					dV = 0.0;
        }
				if (dV > 0.) {
          volume += dV;
          x_grid = grain_grid_x[ind_x + ind_y * nrow + ind_z * nrow * ncol];
          y_grid = grain_grid_y[ind_x + ind_y * nrow + ind_z * nrow * ncol];
          z_grid = grain_grid_z[ind_x + ind_y * nrow + ind_z * nrow * ncol];
					x_com += x_grid * dV;
					y_com += y_grid * dV;
					z_com += z_grid * dV;
				}
			}
		}
	}
  x_com /= volume;
  y_com /= volume;
  z_com /= volume;

  // Computing the inertia tensor (a double loop is unavoidable).
  double Ixx = 0.0, Iyy = 0.0, Izz = 0.0, Ixy = 0.0, Ixz = 0.0, Iyz = 0.0;
	for (int ind_x = 0; ind_x < nrow; xIndex++){
		for (int ind_y = 0; ind_y < ncol; yIndex++){
			for (int ind_z = 0; ind_z < nslice; zIndex++){
        ls_val = grain_grid[ind_x + ind_y * nrow + ind_z * nrow * ncol];
				if (abs(ls_val) < ls_ref){
          // Close to boundary if abs(ls_val) < ls_ref, apply smearing.
					dV = smearedHeavisideStep(-ls_val/ls_ref)* volume_cell;
        }else if (ls_val < 0){
          // Inside and far away from boundary
					dV = volume_cell;
        }else if (ls_val > 0){
          // Outside and far away from boundary
					dV = 0.0;
        }
				if (dV > 0.) {
          x_grid = grain_grid_x[ind_x + ind_y * nrow + ind_z * nrow * ncol];
          y_grid = grain_grid_y[ind_x + ind_y * nrow + ind_z * nrow * ncol];
          z_grid = grain_grid_z[ind_x + ind_y * nrow + ind_z * nrow * ncol];
          Ixx += (pow(y_grid - y_com, 2) + pow(z_grid - z_com, 2)) * dV;
					Iyy += (pow(x_grid - x_com, 2) + pow(z_grid - z_com, 2)) * dV;
					Izz += (pow(x_grid - x_com, 2) + pow(y_grid - y_com, 2)) * dV;
					Ixy -= (x_grid - x_com) * (y_grid - y_com) * dV;
					Ixz -= (x_grid - x_com) * (z_grid - z_com) * dV;
					Iyz -= (y_grid - y_com) * (z_grid - z_com) * dV;
        }
			}
		}
	}

  // Check to see if level set has a non-inertial reference frame
  double I_diag_norm = sqrt(Ixx*Ixx + Iyy*Iyy + Izz*Izz);
  double I_off_diag_norm = sqrt(2*Ixy*Ixy + 2*Ixz*Ixz + 2*Iyz*Iyz);
  if (I_off_diag_norm / I_diag_norm > 0.01){
    // Throw some kind of error. Level set is not given in a non-inertial frame.
    // Intergration of rotational motion will be wrong.
  }
  // ASSIGN INERTIA

  // This is the part where we load or initialise surface nodes :)

  */


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
    error->all(FLERR, "Pair LS/DEM requires ghost atoms store velocity");

  neighbor->add_request(this, NeighConst::REQ_GHOST);
}

/* ---------------------------------------------------------------------- */

void PairLSDEM::setup()
{
  int n = atom->ntypes;
  double maxcut2 = -1;
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      maxcut2 = MAX(maxcut2, cut[i][j]);

  if (maxcut < maxcut2)
    error->all(FLERR, "Maximum cutoff {} less than cutoff defined in pair coefficients {}", maxcut, maxcut2);

  // TODO: THIS IS TEMPORARY FOR A SINGLE TYPE OF GRAINS AS ALL ATOMS STORE THE SAME SIZE
  // TODO: CREATE TEMP GROUPS TO PUT ATOMS OF SAME GRAIN TOGETHER AND CREATE FIX PROPERTY/ATOM OF DIFFERENT SIZE
  // TODO: MUST BE SOME PARALLEL COMPLICATION, LOOK AT THE GROUP COMMAND CODE TO SEE HOW IT'S DONE

  auto fixlist = modify->get_fix_by_style("rigid/ls/dem");
  if (fixlist.size() != 1)
  error->all(FLERR, "Must have one, and only one, instance of fix rigid/ls/dem for pair LS-DEM.");
  fix_rigid = dynamic_cast<FixRigidLSDEM *>(fixlist.front());


  int tmp1, tmp2;
  index_ls_dem_com = atom->find_custom("ls_dem_com", tmp1, tmp2);
  index_ls_dem_quat = atom->find_custom("ls_dem_quat", tmp1, tmp2);
  index_ls_dem_vol = atom->find_custom("ls_dem_vol", tmp1, tmp2);
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

/* ----------------------------------------------------------------------
   Smeared Heaviside step function
------------------------------------------------------------------------- */

double PairLSDEM::smearedHeavisideStep(double x)
{
  // A function that smoothly transition from 0 to 1 when x goes from -1 to 1.
  // For x < -1, the function should be 0. For x > 1, the function should be 1.
  // This is not implemented here, and up to the user to take care of ouside
  // this function. See Kawamoto et al. (2016).
  return 0.5 * (1.0 + x + sin(MY_PI * x) / MY_PI);
}

// End of file