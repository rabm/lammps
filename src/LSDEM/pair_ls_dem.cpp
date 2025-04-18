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
#include "fix_rigid.h"
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

PairLSDEM::PairLSDEM(LAMMPS *_lmp) : Pair(_lmp), k(nullptr), cut(nullptr), gamma(nullptr)
{
  writedata = 1;
  id_fix = nullptr;
  single_enable = 0;
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

  auto fixlist = modify->get_fix_by_style("rigid");
  if (fixlist.size() != 1)
    error->all(FLERR, "Must have one instance of fix rigid for pair LS-DEM.");
  auto fixrigid = dynamic_cast<FixRigid *>(fixlist.front());
  int *body = fixrigid->get_body_array();
  int nbody = fixrigid->get_nbody();

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // Loop to find closest neighbors
  for (ii = 0; ii < inum; ii++) {
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

      j &= NEIGHMASK;

      jbody = body[j];
      jtag = tag[j];

      // Separation distance between the two nodes
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      r = sqrt(rsq);

      // Need an additional check such that only nodes of the smallest grain i
      // are used in combination with the level set of grain j.
      // If grain volumes are equal, always take the nodes of the grian with the lowest
      // particle id number.
      // Joel: added grain_vol[i] vs grain_vol[j], currently both are hard coded (and equal)

      // What does this code do exactly? Does this make both neighbour lists 
      // min_distance(i) = j and min_distance(j) = i? 

      key = nbody * itag + jbody;
      // If first interation between i and j's grain, create entry
      if (min_distances.find(key) == min_distances.end()) {
        min_distances[key] = std::make_pair(jtag, r);
      } else {
        // Overwrite if i and j are closer
        if (r < min_distances[key].second) {
          min_distances[key] = std::make_pair(jtag, r);
        }
      }

      // Do the same for node j
      key = nbody * jtag + ibody;
      if (min_distances.find(key) == min_distances.end()) {
        min_distances[key] = std::make_pair(itag, r);
      } else {
        // Overwrite if i and j are closer
        if (r < min_distances[key].second) {
          min_distances[key] = std::make_pair(itag, r);
        }
      }
    }
  }

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

      // Figure out whether either force is calculated
      //   Only calculate force of smaller grain on larger grain
      //     in ties, go by grain ID
      //   Only calculate force between closest set of nodes
      // Danny: There is a similar note above, do we resolve this there or here?
      //        If only one of the two is used, can we reduce the nieghbour search above?

      calc_force_of_j_on_i = 0;
      calc_force_of_i_on_j = 0;

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

      // Evaluate the level set, and assign the interaction direction based on
      // node-grain combination. Force magnitude and direction go i -> j by definition.
      if (calc_force_of_i_on_j) {
        // The normal points away from j, correct signs
        u = get_ls_value(i, j, normal);
        MathExtra::negate3(normal);
      } else {
        u = get_ls_value(j, i, normal);
      }

      // Apply forces and torques

      // No adhesion, cohesion, or ranged forces
      // Danny: the result of get_ls_value should be negative because level-sets are typically
      // defined to have a negative value inside the grain. Should adjust line 237 and 240 to
      // u = -get_ls_value ... once this convention has been applied.
      if (u < 0) continue;

      // With penetration distance u and normal n (i->j),
      // we have: F_{j on i} = f(ls_value) = - k_n * u * n.
      fpair_mag = k[itype][jtype] * u;

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

      // Compute torque
      MathExtra::cross3(lever, fpair, torque_pair);

      // Apply torques on grain i
      torque[i][0] += torque_pair[0];
      torque[i][1] += torque_pair[1];
      torque[i][2] += torque_pair[2];

      // Mirror forces and torques on grain j
      // Danny: Shouldn't these need a sign swap?
      if (newton_pair || j < nlocal) {
        MathExtra::negate3(fpair);
        f[j][0] += fpair[0];
        f[j][1] += fpair[1];
        f[j][2] += fpair[2];

        lever[0] = contact_point[0] - grain_com[j][0];
        lever[1] = contact_point[1] - grain_com[j][1];
        lever[2] = contact_point[2] - grain_com[j][2];

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
  int iarg = 0;
  while (iarg < narg) {
    error->all(FLERR, "Illegal pair_style command {}", arg[iarg]);
  }

  if (id_fix && modify->nfix) modify->delete_fix(id_fix);

  if (!id_fix)
    id_fix = utils::strdup(std::string("PAIR_LS_DEM") + std::to_string(instance_me));

  if (force->newton_pair)
    error->all(FLERR, "Temporarily do not support newton pair on with LS/DEM");

  nrow = 20;
  ncol = 20;
  nslice = 1;
  double l_grid = 0.5;
  double x_com = 5.25;
  double y_com = 5.25;
  double r = 2.5;

  ngrid = nrow * ncol;
  spac = l_grid;

  /*
  
  // Volume integration
  double **grain_grid = atom->darray[index_ls_dem_grid];
  double **grain_grid_x = atom->darray[index_ls_dem_gridx];
  double **grain_grid_y = atom->darray[index_ls_dem_gridy];
  double **grain_grid_z = atom->darray[index_ls_dem_gridz];

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

  modify->add_fix(fmt::format("{} all property/atom d2_ls_dem_grid {} d2_ls_dem_gridx {} d2_ls_dem_gridy {} d2_ls_dem_gridz {} writedata no ghost yes",
    id_fix, ngrid, ngrid, ngrid, ngrid));
  int tmp1, tmp2;
  index_ls_dem_grid = atom->find_custom("ls_dem_grid", tmp1, tmp2);
  index_ls_dem_gridx = atom->find_custom("ls_dem_gridx", tmp1, tmp2);
  index_ls_dem_gridy = atom->find_custom("ls_dem_gridy", tmp1, tmp2);
  index_ls_dem_gridz = atom->find_custom("ls_dem_gridz", tmp1, tmp2);

  index_ls_dem_com = atom->find_custom("ls_dem_com", tmp1, tmp2);
  index_ls_dem_quat = atom->find_custom("ls_dem_quat", tmp1, tmp2);
  index_ls_dem_vol = atom->find_custom("ls_dem_vol", tmp1, tmp2);

  double **ls_dem_grid = atom->darray[index_ls_dem_grid];
  double **ls_dem_gridx = atom->darray[index_ls_dem_gridx];
  double **ls_dem_gridy = atom->darray[index_ls_dem_gridy];
  double **ls_dem_gridz = atom->darray[index_ls_dem_gridz];
  double *ls_dem_vol = atom->dvector[index_ls_dem_vol];

  double delx, dely;
  for (int i = 0; i < atom->nlocal; i++) {
    for (int a = 0; a < ncol; a++) {
      for (int b = 0; b < nrow; b++) {
        // stored value is at lower left corner of grid
        delx = a * l_grid - x_com;
        dely = b * l_grid - y_com;
        ls_dem_grid[i][b * ncol + a] = r - sqrt(delx * delx + dely * dely);
        ls_dem_gridx[i][b * ncol + a] = delx;
        ls_dem_gridy[i][b * ncol + a] = dely;
        ls_dem_gridz[i][b * ncol + a] = 0;

        //if (i == 0) printf("%.3g ", ls_dem_grid[i][b * ncol + a]);
        if (a == 0 && b == 0) {
          grid_min[0] = delx;  // Later convert to peratom values
          grid_min[1] = dely;
          grid_min[2] = 0;
        }
      }
      //if (i == 0) printf("\n");
    }
    ls_dem_vol[i] = MY_PI * pow(5.0, 2);
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
    error->all(FLERR, "Pair LS/DEM requires ghost atoms store velocity");

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
   Find the value of node (atom) i in j's LS grid.
------------------------------------------------------------------------- */

double PairLSDEM::get_ls_value(int i, int j, double *normal)
{
  double **x = atom->x;
  double **grain_com = atom->darray[index_ls_dem_com];
  double **grain_quat = atom->darray[index_ls_dem_quat];
  double **grain_grid = atom->darray[index_ls_dem_grid];
  double **grain_grid_x = atom->darray[index_ls_dem_gridx];
  double **grain_grid_y = atom->darray[index_ls_dem_gridy];
  double **grain_grid_z = atom->darray[index_ls_dem_gridz];

  int nrow_offset = 0; // Offsets for local subgrid, to implement later
  int ncol_offset = 0; //   currently subgrid = grid, so offsets are zero
  int nslice_offset = 0;

  // Calculate position of i in j's grid using:
  //   x[i][0-2] = location of i
  //   x[j][0-2] = location of j
  //   grain_com[j][0-2] = CoM of j's grain
  //   grain_quat[j][0-3] = quat of j's grain

  // Danny: How does LAMMPS take care of the periodic shift?
  //        YADE uses an offset coordinate shift2 that is added to x[j] or grain_com[j]

  //
  //  GET NODE I IN LOCAL COORDINATES OF J GRAIN
  //

  // Relative coordinate of node i w.r.t. centre of mass grain j
  // Danny: I will asumme x is in global coordinates.
  double delx = x[i][0]-grain_com[j][0];
  double dely = x[i][1]-grain_com[j][1];
  double delz = 0; // x[i][2]-grain_com[j][2];

  // Account for PBCs
  domain->minimum_image(delx, dely, delz);

  // Extract quaternion components
  // Danny: How is grain_quat defined? Is it the rotation local -> global or global -> local?
  // I will assume it is local -> global
  // Remove minus signs if grain_quat is global -> local
  double q_w = grain_quat[j][0];
  double q_x = -grain_quat[j][1];
  double q_y = -grain_quat[j][2];
  double q_z = -grain_quat[j][3];

  // Apply quaternion rotation to move into local reference frame of grain j grid
  // x' = (q*x)*q^-1 // Joel: this code might have a typo so I replaced it with the method below
  //double x_local = (1 - 2 * (q_y * q_y + q_z * q_z)) * delx + 2 * (q_x * q_y - q_w * q_z) * dely + 2 * (q_x * q_z + q_w * q_y) * delz;
  //double y_local = 2 * (q_x * q_y + q_w * q_z) * delx + (1 - 2 * (q_x * q_x + q_z * q_z)) * dely + 2 * (q_y * q_z - q_w * q_x) * delz;
  //double z_local = 2 * (q_x * q_z - q_w * q_y) * delx + 2 * (q_y * q_z + q_w * q_x) * dely + (1 - 2 * (q_x * q_x + q_y * q_y)) * delz;

  // sample code with math_extra, feel free to add MathExtra to namespace if helpful
  double x_local[3];
  double dx[3] = {delx, dely, delz};
  MathExtra::quatrotvec(grain_quat[j], dx, x_local); // I think it's global -> local, but if you need to take a conjugate there's a function qconjugate()
  // see comments above functions in math_extra.h/cpp for details

  //
  //  COMPUTE THE LS GRID INDICES
  //

  // Calculate index from coordinate, need to be more careful with integer division
  // Danny: We need to get grid_min, the lowest corner (in -1,-1,-1 direction) of the grid
  //        and spac, the grid spacing. (If we want to keep this in normalised coords, we
  //        will have to normalise )
  int ind_x = int( (x_local[0] - grid_min[0]) / spac ); // Here, int() does the same as floor() + conversion
  int ind_y = int( (x_local[1] - grid_min[1]) / spac );
  int ind_z = 0; //int( (x_local[2] - grid_min[2]) / spac );

  // We might need an extra check. If x_local is very close to grid_min, it may pass and give
  // errors later.
  // Joel: I added a macro EPSILON which might be useful for biasing rounding

  if ( (ind_x < 0) || (ind_y < 0) ) { // || (indz < 0)
    // Point is outside the LS grid of grain j. Cannot compute distance or normal.
    error->one(FLERR, "Contacting node {} is outside of node {}'s LS grid", atom->tag[i], atom->tag[j]);
  } else if ( (ind_x > nrow - 1) || (ind_y > ncol - 1)  ) {  // || (ind_z > nslice-1)
    // Point is outside the LS grid of grain j. Cannot compute distance or normal.
    error->one(FLERR, "Contacting node {} is outside of node {}'s LS grid", atom->tag[i], atom->tag[j]);
  }

  // Apply offsets, there is probably a more proper way
  ind_x = ind_x - nrow_offset;
  ind_y = ind_y - ncol_offset;
  ind_z = ind_z - nslice_offset;

  //
  //  DO THE BILINEAR INTERPOLATION
  //

  // Coordinates of the grid lower grid point of the cell we are in
  double x0 = grain_grid_x[j][ind_x + ind_y * ncol]; // + ind_z + nslice
  double y0 = grain_grid_y[j][ind_x + ind_y * ncol];
  double z0 = 0; //grain_grid_z[j][ind_x + ind_y * ncol + ind_z * nslice][2];

  // Level-set values on the grid points
  double ls000 = grain_grid[j][ind_x   + ind_y     * ncol]; // + ind_z * nslice
  double ls100 = grain_grid[j][ind_x+1 + ind_y     * ncol];
  double ls010 = grain_grid[j][ind_x   + (ind_y+1) * ncol];
  double ls110 = grain_grid[j][ind_x+1 + (ind_y+1) * ncol];

  // The reduced coordinates
  // May be safe to cap them with math::max(math::min(x_red, 1.0), 0.0)
  double x_red = (x_local[0] - x0) / spac;
  double y_red = (x_local[1] - y0) / spac;
  double z_red = (x_local[2] - z0) / spac;

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
			//for (int c = 0; c < 2; c++) { // Joel: I temporarily commented out the z stuff so it's easier to debug
			double lsVal = grain_grid[j][(ind_x + a) + (ind_y + b) * ncol]; // + (ind_z + c)*nslice];
			nx += lsVal * (2 * a - 1) * ((1 - b) * (1 - y_red) + b * y_red); // * ((1 - c) * (1 - z_red) + c * z_red);
			ny += lsVal * (2 * b - 1) * ((1 - a) * (1 - x_red) + a * x_red); // * ((1 - c) * (1 - z_red) + c * z_red);
			//nz += lsVal * (2 * c - 1) * ((1 - a) * (1 - x_red) + a * x_red) * ((1 - b) * (1 - y_red) + b * y_red);
			//}
		}
	}

  // Assign normal
  normal[0] = nx;
  normal[1] = ny;
  normal[2] = nz;

  // Rotate normal back to global coordinates
  double quatconj[4];
  MathExtra::qconjugate(grain_quat[j], quatconj);
  MathExtra::quatrotvec(quatconj, normal, normal);

  return dist;
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