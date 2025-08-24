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

    memory->destroy(kn);
    memory->destroy(kt);
    memory->destroy(mu);
    memory->destroy(cut);
    memory->destroy(gamma);
  }
}

/* ---------------------------------------------------------------------- */

void PairLSDEM::compute(int eflag, int vflag)
{
  int i, j, ii, jj, key, allnum, inum, jnum, itype, jtype, ibody, jbody;
  tagint itag, jtag;
  double xitmp, yitmp, zitmp, xjtmp, yjtmp, zjtmp, delx, dely, delz, dr, evdwl;
  double r, rsq, rinv, factor_lj, u, ivol, jvol;
  int *ilist, *jlist, *numneigh, **firstneigh, calc_force_of_i_on_j, calc_force_of_j_on_i;
  double vxitmp, vyitmp, vzitmp, vxjtmp, vyjtmp, vzjtmp, delvx, delvy, delvz, dot, smooth;
  double normal[3], fpair_mag, fpair[3], contact_point[3], lever[3], torque_pair[3];
  // double normal_old[3];

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
  // double **n_old = atom->n_old;
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

  // MIGHT BE ABLE TO DELETE THIS WITH OPTIMISATIONS
  // Loop over local+ghost atoms to find closest neighbors
  for (ii = 0; ii < allnum; ii++) {
    // Loop through local nodes
    i = ilist[ii];
    xitmp = x[i][0];
    yitmp = x[i][1];
    zitmp = x[i][2];
    ibody = body[i];
    itag = tag[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      // Loop through neighbouring nodes
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];

      if (factor_lj == 0) continue;

      // Make the neighbour mask an integer again (discarding history flags etc.)
      j &= NEIGHMASK;

      jbody = body[j];
      jtag = tag[j];

      // Separation distance between the two nodes
      delx = xitmp - x[j][0];
      dely = yitmp - x[j][1];
      delz = zitmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      r = sqrt(rsq);

      if (r > maxcut) continue;

      // Create a dictionary for each grain that holds the
      // tag of the closest interacting node on the other grain.
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
    xitmp = x[i][0];
    yitmp = x[i][1];
    zitmp = x[i][2];
    vxitmp = v[i][0];
    vyitmp = v[i][1];
    vzitmp = v[i][2];
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
      xjtmp = x[j][0];
      yjtmp = x[j][1];
      zjtmp = x[j][2];
      vxjtmp = v[j][0];
      vyjtmp = v[j][1];
      vzjtmp = v[j][2];
      jbody = body[j];
      jtag = tag[j];
      jvol = grain_vol[j];
      jtype = type[j];

      // Figure out whether to use the nodes of grain i or j.
      // We use the nodes on the smaller grain since this will
      // be more accurate. If the volumes are tied, use the grain ID.
      // Only calculate force between closest pair of node and grid.

      calc_force_of_j_on_i = 0;
      calc_force_of_i_on_j = 0;

      // Use the nodes of the smallest grain.
      if (ivol < jvol || (ivol == jvol && ibody < jbody)) {
        // Grain i is smaller, use nodes of i and level set of j.
        key = nbody * jtag + ibody;
        if (min_distances.find(key) != min_distances.end())
          if (itag == min_distances[key].first)
            calc_force_of_i_on_j = 1;
      } else {
        // Grain j is smaller, use nodes of j and level set of i.
        key = nbody * itag + jbody;
        if (min_distances.find(key) != min_distances.end())
          if (jtag == min_distances[key].first)
            calc_force_of_j_on_i = 1;
      }

      // If no forces are calculated
      if (calc_force_of_i_on_j + calc_force_of_j_on_i == 0) continue;

      // Evaluate the level set, and assign the interaction direction based on the
      // node-grain combination. Force magnitude and direction go i -> j by definition.
      if (calc_force_of_i_on_j) { // Use node of i.
        // Level set is by definition negative inside the particle, 
        // so swap the sign to get the overlap distance.
        u = - fix_rigid->get_ls_value(i, j, normal);
        // The normal is also swapped and points away from j, correct signs. Already in global coordinates.
        MathExtra::negate3(normal);

        // Contact point
        contact_point[0] = xitmp - 0.5 * u * normal[0];
        contact_point[1] = yitmp - 0.5 * u * normal[1];
        contact_point[2] = zitmp - 0.5 * u * normal[2];

        // Tangent force! 
        // Get old node normal
        // normal_old[0] = n_old[i][0]
        // normal_old[1] = n_old[i][1]
        // normal_old[2] = n_old[i][2]

        // if( norm(normal_old) > 0){
        // Applying Rodrigues' rotation formula to get the new shear displacement
        // Rotation axis
        // k = MathExtra::cross3(normal_old, normal);
        // sint = abs(k);
        // cost = sqrt(1- sint*sint);
        // k = k / (sint + eps);
        // nodeFs = nodeFs * cost + MathExtra::cross3(k,nodeFs) * sint + k * MathExta::dot3(k,nodeFs) * (1-cost);
        // }

        // n_old = normal_old;
        // v = vi + omegai x (xi - xi_grain) - vj - omegaj x (xj - xj_grain);  
        // ds = (v - MathExtra::dot3(v,n)*n)*dt;

        // Standard elastic-perfectly-plastic Coulomb friction model.
        // nodeFs -= ds * kt;
        // nodeFsMag = |nodeFs|;
        // nodeContactGrain[i] = j; // ADD SANITY CHECK FOR CONTACT WITH 2 GRAINS?
        // FsMag = min( mu*|Fn|, nodeFsMag)
        
        // if(FsMag > 0){
        // Fs = FsMag * nodeFs/nodeFsMag;
        // F += Fs;

        //}

      } else { // Use node of j.
        u = - fix_rigid->get_ls_value(j, i, normal);
        // The normal points towards j, no correction needed. Already in global coordinates.

        // Contact point
        contact_point[0] = xjtmp - 0.5 * u * normal[0];
        contact_point[1] = yjtmp - 0.5 * u * normal[1];
        contact_point[2] = zjtmp - 0.5 * u * normal[2];
      }

      // Apply forces and torques

      // Reset shear force if no contact
      // if (u < 0) {
      //   nodeFs[i] = 0; // or j
      //   n_old[i][0] = 0;
      //   n_old[i][1] = 0;
      //   n_old[i][2] = 0;
      //   nodeContactGrain[i] = inf;
      //   continue;
      // }

      // No adhesion, cohesion, or ranged forces.
      if (u < 0) continue;

      // With penetration distance u and normal n (i->j),
      // we have: F_{j on i} = f(ls_value) = - k_n * u * n.
      fpair_mag = - kn[itype][jtype] * u;

      // The pair force vector
      fpair[0] = fpair_mag * normal[0];
      fpair[1] = fpair_mag * normal[1];
      fpair[2] = fpair_mag * normal[2];

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
      if (newton_pair || j < nlocal) { // Need to check this again if we end up enabling newton_pair
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

  memory->create(kn, np1, np1, "pair:kn");
  memory->create(kt, np1, np1, "pair:kt");
  memory->create(mu, np1, np1, "pair:mu");
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
  if (narg != 7)
    error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double kn_one = utils::numeric(FLERR, arg[2], false, lmp);
  double kt_one = utils::numeric(FLERR, arg[3], false, lmp);
  double mu_one = utils::numeric(FLERR, arg[4], false, lmp);
  double cut_one = utils::numeric(FLERR, arg[5], false, lmp);
  double gamma_one = utils::numeric(FLERR, arg[6], false, lmp);

  if (cut_one <= 0.0) error->all(FLERR, "Incorrect args for pair coefficients");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      kn[i][j] = kn_one;
      kt[i][j] = kt_one;
      mu[i][j] = mu_one;
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
    kn[i][j] = mix_energy(kn[i][i], kn[j][j], cut[i][i], cut[j][j]);
    kt[i][j] = mix_energy(kt[i][i], kt[j][j], cut[i][i], cut[j][j]);
    mu[i][j] = 0.5*(mu[i][i] + mu[j][j]); // Arithmetic mean mixing rule
    gamma[i][j] = mix_energy(gamma[i][i], gamma[j][j], cut[i][i], cut[j][j]);
  }

  // Enforces symmetry
  cut[j][i] = cut[i][j];
  kn[j][i] = kn[i][j];
  kt[j][i] = kt[i][j];
  mu[j][i] = mu[i][j];
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
        fwrite(&kn[i][j], sizeof(double), 1, fp);
        fwrite(&kt[i][j], sizeof(double), 1, fp);
        fwrite(&mu[i][j], sizeof(double), 1, fp);
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
          utils::sfread(FLERR, &kn[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &kt[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &mu[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &gamma[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&kn[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&kt[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&mu[i][j], 1, MPI_DOUBLE, 0, world);
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
    fprintf(fp, "%d %g %g %g %g %g\n", i, kn[i][i], kt[i][i], mu[i][i], cut[i][i], gamma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLSDEM::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %g %g %g %g %g\n", i, kn[i][i], kt[i][i], mu[i][i], cut[i][i], gamma[i][i]);
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