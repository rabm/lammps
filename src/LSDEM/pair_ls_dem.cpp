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
#include "utils.h"
#include "update.h"
#include <cmath>
#include <unordered_map>

static constexpr double EPSILON = 1e-12;

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairLSDEM::PairLSDEM(LAMMPS *_lmp) : Pair(_lmp), kn(nullptr), kt(nullptr), mu(nullptr), etan(nullptr), etat(nullptr), cut(nullptr),
 decayn1(nullptr), etan1(nullptr), decayt1(nullptr), etat1(nullptr), fix_rigid(nullptr) // gamma(nullptr),
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
    memory->destroy(etan);
    memory->destroy(etat);
    memory->destroy(knp);
    memory->destroy(cut);
    memory->destroy(decayn1);
    memory->destroy(etan1);
    memory->destroy(decayt1);
    memory->destroy(etat1);
    //memory->destroy(gamma);
    // other consts
  }
}

/* ---------------------------------------------------------------------- */

void PairLSDEM::compute(int eflag, int vflag)
{
  int i, j, ii, jj, key, allnum, inum, jnum, itype, jtype, ibody, jbody;
  tagint itag, jtag;
  double xitmp, yitmp, zitmp, xjtmp, yjtmp, zjtmp, delx, dely, delz, dr, evdwl;
  double r, rsq, rinv, factor_lj, u, ivol, jvol, icomx, icomy, icomz, jcomx, jcomy, jcomz;
  int *ilist, *jlist, *numneigh, **firstneigh, calc_force_of_i_on_j, calc_force_of_j_on_i;
  double vxitmp, vyitmp, vzitmp, vxjtmp, vyjtmp, vzjtmp, delvx, delvy, delvz, dot, smooth;
  double normal[3], fn_mag, fpair[3], fpair_mag, contact_point[3], lever[3], torque_pair[3];
  double fs_tmp[3], fs_mag, fs_mag_trial, k[3], sintheta, costheta, term1[3], term2;
  double tangent[3], shear_incr, v_rel[3], v_rel_t[3], v_rel_n_mag, v_rel_t_mag, v_rel_t_mag_inv;
  double normal_old[3], tangent_old[3], areai, areaj;

  // Currently require:
  //   Newton pair off.
  //   Exclude intramolecule interactions
  // In future, remove restrictions

  evdwl = 0.0;
  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  // Node quantities
  double **x = atom->x;
  double **v = atom->v; // This is the half-step, giving O(dt^2) accuracy for damping instead of O(dt).
  double **f = atom->f;
  double **torque = atom->torque;
  double **n = atom->darray[index_ls_dem_n]; // Contact normal (to be update from previous time step)
  double **fs = atom->darray[index_ls_dem_fs]; // Shear component of f (to be update from previous time step)
  int *touch_id = atom->ivector[index_ls_dem_touch_id]; // Grain in contact with this node (to be update from previous time step)
  double *fn1 = atom->dvector[index_ls_dem_fn1]; // Maxwell element force history (magnitude only, normal)
  double *fs1 = atom->dvector[index_ls_dem_fs1]; // Maxwell element force history (magnitude only, shear)

  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double dt = update->dt;

  // Grain quantities
  double **grain_com = atom->darray[index_ls_dem_com]; // Need CoM for torques
  std::unordered_map<int, std::pair<int, double>> min_distances;

  int *body = fix_rigid->get_body_array();
  int nbody = fix_rigid->get_nbody();
  double *grain_vol = fix_rigid->get_vol_array();
  double *node_area = fix_rigid->get_area_array(); // Area per node used to normalise forces

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

  // NOTE: calc_force_of_j_on_i and calc_force_of_i_on_j now cuase branching.
  // The contact model might do the same. May it be worth it to pre-sort the pairs
  // such that it is always i_on_j and the contact models are sorted?
  // Currently, compiler vectorisation is scrambled, which might be particularly
  // bad for any future Kokkos GPU port.

  // Only loop over local atoms to calculate forces
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
    icomx = grain_com[i][0];
    icomy = grain_com[i][1];
    icomz = grain_com[i][2];

    ivol = grain_vol[ibody];
    areai = node_area[ibody];

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
      jcomx = grain_com[j][0];
      jcomy = grain_com[j][1];
      jcomz = grain_com[j][2];
      jtype = type[j];

      jvol = grain_vol[jbody];
      areaj = node_area[jbody];

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
      } else { // Use node of j.
        u = - fix_rigid->get_ls_value(j, i, normal);
        // The normal points towards j, no correction needed. Already in global coordinates.

        // Contact point
        contact_point[0] = xjtmp - 0.5 * u * normal[0];
        contact_point[1] = yjtmp - 0.5 * u * normal[1];
        contact_point[2] = zjtmp - 0.5 * u * normal[2];
      }

      // No adhesion, cohesion, or ranged forces.
      if (u <= 0) {
        // Reset shear force if no contact
        // If i and j are not a shear-interacting pair, it will skip the reset
        if (calc_force_of_i_on_j) {
          if (touch_id[i] == jbody) {
            touch_id[i] = -1;
            fs[i][0] = 0.0;
            fs[i][1] = 0.0;
            fs[i][2] = 0.0;
            n[i][0] = 0.0;
            n[i][1] = 0.0;
            n[i][2] = 0.0;
          }
        } else { // calc_force_of_j_on_i already guaranteed to be true (see line ~233)
          if (touch_id[j] == ibody) {
            touch_id[j] = -1;
            fs[j][0] = 0.0;
            fs[j][1] = 0.0;
            fs[j][2] = 0.0;
            n[j][0] = 0.0;
            n[j][1] = 0.0;
            n[j][2] = 0.0;
          }
        }
        continue;
      }

      //////////////////
      // Normal force //
      //////////////////

      // Elastic spring
      // With positive penetration distance u
      if (fabs(knp[itype][jtype]) < EPSILON) {
        fn_mag = kn[itype][jtype] * u;
      } else {
        fn_mag = kn[itype][jtype] * pow(u, knp[itype][jtype]);
      }

      // Relative velocity at the grain surface at the half step t + 0.5*dt.
      // Note: The velocity at the node due to an angular velocity of the grain around its
      // centre of mass is already included, so the difference of linear velocities suffices.
      v_rel[0] = vxitmp - vxjtmp;
      v_rel[1] = vyitmp - vyjtmp;
      v_rel[2] = vzitmp - vzjtmp;

      // Relative velocity in normal direction with sign
      v_rel_n_mag = MathExtra::dot3(v_rel, normal);

      // Viscous damping or dashpot (parallel, only repulsive, i.e. no attractive force if v_rel_n_mag < 0)
      if (etan[itype][jtype] > 0.0) {
        fn_mag += etan[itype][jtype] * MAX(v_rel_n_mag, 0.0);
      }

      // Maxwell arm (1st, parallel, only repulsive)
      if (etan1[itype][jtype] > 0.0) { // preprocessing guarantees that decayn1 > 0 if etan1 > 0
        if (calc_force_of_i_on_j) { // Node of i.
          fn1[i] = decayn1[itype][jtype] * fn1[i] + etan1[itype][jtype] * (1.0 - decayn1[itype][jtype]) * MAX(v_rel_n_mag, 0.0);
          fn_mag += fn1[i];
        } else { // Node of j.
          fn1[j] = decayn1[itype][jtype] * fn1[j] + etan1[itype][jtype] * (1.0 - decayn1[itype][jtype]) * MAX(v_rel_n_mag, 0.0);
          fn_mag += fn1[j];
        }
        // Maxwell arm (2nd)
        //fn2_mag[i] = decayn2[itype][jtype] * fh2_mag[i] + etan2[itype][jtype] * (1-decayn2[itype][jtype]) * MAX(v_rel_n_mag, 0.0);
        //fn_mag -= fn2_mag[i]
      }

      // Multiply by node area to make the force independent of discretisation (fn_mag was a stress)
      if (calc_force_of_i_on_j) { // Node of i.
        fn_mag *= areai;
      } else { // Node of j.
        fn_mag *= areaj;
      }

      // The pair force vector should point j->i because of repulsion.
      // With normal n (i->j), we have: F_{j on i} = f(ls_value) = - fn_mag * n.
      fpair[0] = - fn_mag * normal[0];
      fpair[1] = - fn_mag * normal[1];
      fpair[2] = - fn_mag * normal[2];

      ///////////////////
      // Tangent force //
      ///////////////////

      // Tangent force only exists if mu > 0 and kt > 0
      //if ( (kt[itype][jtype] > 0) && (mu[itype][jtype] > 0) ){
      // DvdH: Grains without friction are silly, so I took out this check.

      // Check if the pair is valid for shear history calculation
      // Initialise if no contact
      if (calc_force_of_i_on_j) {
        if (touch_id[i] == -1) {
          touch_id[i] = jbody;
        }
        if (touch_id[i] != jbody) {
          if (comm->me == 0) {
            error->warning(FLERR, "Shear history of node on grain {} penetrating {} cannot be computed at step {}",
              ibody, jbody, update->ntimestep);
          }
        }
      } else {
        if (touch_id[j] == -1) {
          touch_id[j] = ibody;
        }
        if (touch_id[j] != ibody) {
          if (comm->me == 0) {
            error->warning(FLERR, "Shear history of node on grain {} penetrating {} cannot be computed at step {}",
              jbody, ibody, update->ntimestep);
          }
        }
      }

      // Get old shear force and old node normal
      if (calc_force_of_i_on_j) { // Use node of i.
        fs_tmp[0] = fs[i][0];
        fs_tmp[1] = fs[i][1];
        fs_tmp[2] = fs[i][2];
        normal_old[0] = n[i][0];
        normal_old[1] = n[i][1];
        normal_old[2] = n[i][2];
      } else { // Use node of j.
        // Swap sign due to change of j->i to i->j reference frame.
        fs_tmp[0] = -fs[j][0];
        fs_tmp[1] = -fs[j][1];
        fs_tmp[2] = -fs[j][2];
        normal_old[0] = -n[j][0];
        normal_old[1] = -n[j][1];
        normal_old[2] = -n[j][2];
      }

      // Adjust fs_tmp to account for rotation of the contact plane.
      if( MathExtra::len3(normal_old) > 0 ) {
        MathExtra::cross3(normal_old, normal, k); // Rotation vector
        sintheta = MathExtra::len3(k); // Rotation magnitude
        if(sintheta > EPSILON){ // Don't apply rotation if magnitude is tiny
          costheta = sqrt(1 - sintheta * sintheta);
          k[0] = k[0] / sintheta; // Rotation axis
          k[1] = k[1] / sintheta;
          k[2] = k[2] / sintheta;
          // Applying Rodrigues' rotation formula to get the rotated shear displacement
          MathExtra::cross3(k, fs_tmp, term1);
          term2 = MathExtra::dot3(k, fs_tmp) * (1.0 - costheta);
          fs_tmp[0] = fs_tmp[0] * costheta + term1[0] * sintheta + k[0] * term2;
          fs_tmp[1] = fs_tmp[1] * costheta + term1[1] * sintheta + k[1] * term2;
          fs_tmp[2] = fs_tmp[2] * costheta + term1[2] * sintheta + k[2] * term2;
        }
      }

      // Relative velocity in tangential direction
      v_rel_t[0] = v_rel[0] - v_rel_n_mag * normal[0];
      v_rel_t[1] = v_rel[1] - v_rel_n_mag * normal[1];
      v_rel_t[2] = v_rel[2] - v_rel_n_mag * normal[2];
      v_rel_t_mag = MathExtra::len3(v_rel_t);

      // Tangent normal
      if (v_rel_t_mag > EPSILON) {
        v_rel_t_mag_inv = 1.0 / v_rel_t_mag;
        tangent[0] = v_rel_t[0] * v_rel_t_mag_inv;
        tangent[1] = v_rel_t[1] * v_rel_t_mag_inv;
        tangent[2] = v_rel_t[2] * v_rel_t_mag_inv;
      } else {
        // Backup: Use old shear force direction as tangent.
        v_rel_t_mag_inv = 1.0/MathExtra::len3(fs_tmp);
        tangent[0] = fs_tmp[0] * v_rel_t_mag_inv;
        tangent[1] = fs_tmp[1] * v_rel_t_mag_inv;
        tangent[2] = fs_tmp[2] * v_rel_t_mag_inv;
      }

      // Magnitude increment of the shear displacement
      shear_incr = v_rel_t_mag*dt;

      // Elastic spring
      fs_mag_trial = kt[itype][jtype] * shear_incr;

      // Viscous damping or dashpot (parallel, only resistive, no tensile force if v_rel_t_mag < 0)
      if(etat[itype][jtype] > 0) {
        fs_mag_trial += etat[itype][jtype] * MAX(v_rel_t_mag, 0.0);
      }

      // Maxwell arm (1st, parallel, only repulsive)
      if (etan1[itype][jtype] > 0.0) { // preprocessing guarantees that decayt1 > 0 if etat1 > 0
        // Get the old tangent vector
        v_rel_t_mag_inv = 1.0 / MathExtra::len3(fs_tmp);
        tangent_old[0] = fs_tmp[0] * v_rel_t_mag_inv;
        tangent_old[1] = fs_tmp[1] * v_rel_t_mag_inv;
        tangent_old[2] = fs_tmp[2] * v_rel_t_mag_inv;
        // The dot(t_old,t) part accounts for the in-plane rotation of the tangent force.
        // When the shear direction reverses, it correctly preserves the direction of the old force.
        // However, when rotating towards the orthogonal direction, we inevitably lose some of the force.
        // We could track the full vector, but it would cost more memory (and accessing time)
        if (calc_force_of_i_on_j) { // Node of i.
          fs1[i] += decayt1[itype][jtype] * fs1[i] * MathExtra::dot3(tangent_old, tangent)
            + etat1[itype][jtype] * (1 - decayt1[itype][jtype]) * v_rel_t_mag; // v_rel_t_mag is always positive
          fs_mag_trial += fs1[i];
        } else { // Node of j.
          fs1[j] += decayt1[itype][jtype] * fs1[j] * MathExtra::dot3(tangent_old, tangent)
            + etat1[itype][jtype] * (1 - decayt1[itype][jtype]) * v_rel_t_mag; // v_rel_t_mag is always positive
          fs_mag_trial += fs1[j];
        }
        // Maxwell arm (2nd)
        // fs2_mag[i] = exps2*fs2_mag[i] + etat2[itype][jtype]*(1-exps2)*v_rel_t_mag;
        // fs_mag -= fs2_mag[i]
      }

      // Multiply by node area to make the force independent of discretisation (fs_mag_trial was a stress)
      if (calc_force_of_i_on_j) { // Node of i.
        fs_mag_trial *= areai;
      } else { // Node of j.
        fs_mag_trial *= areaj;
      }

      // Total shear or tangent force update (repulsive again)
      fs_tmp[0] -= fs_mag_trial * tangent[0];
      fs_tmp[1] -= fs_mag_trial * tangent[1];
      fs_tmp[2] -= fs_mag_trial * tangent[2];
      // Update trial shear force magnitude
      fs_mag_trial = MathExtra::len3(fs_tmp);

      // Perfectly plastic Coulomb friction criterion
      fs_mag = std::min(mu[itype][jtype] * fn_mag, fs_mag_trial);

      if (fs_mag > 0) {
        // Final shear or tangential force
        fs_tmp[0] = fs_mag * (fs_tmp[0] / fs_mag_trial);
        fs_tmp[1] = fs_mag * (fs_tmp[1] / fs_mag_trial);
        fs_tmp[2] = fs_mag * (fs_tmp[2] / fs_mag_trial);
        // Add shear force to the pair force vector
        fpair[0] += fs_tmp[0];
        fpair[1] += fs_tmp[1];
        fpair[2] += fs_tmp[2];
      }

      // Update saved shear force and normal
      if (calc_force_of_i_on_j) { // Node of i.
        fs[i][0] = fs_tmp[0];
        fs[i][1] = fs_tmp[1];
        fs[i][2] = fs_tmp[2];
        n[i][0] = normal[0];
        n[i][1] = normal[1];
        n[i][2] = normal[2];
      } else { // Node of j.
        // Swap sign due to change of i->j to j->i reference frame.
        fs[j][0] = -fs_tmp[0];
        fs[j][1] = -fs_tmp[1];
        fs[j][2] = -fs_tmp[2];
        n[j][0] = -normal[0];
        n[j][1] = -normal[1];
        n[j][2] = -normal[2];
      }

      //} // Check if kt > 0 and mu > 0

      //////////////////////////////
      // Total forces and torques //
      //////////////////////////////

      // Force on grain i
      f[i][0] += fpair[0];
      f[i][1] += fpair[1];
      f[i][2] += fpair[2];

      // Lever arm on grain i
      lever[0] = contact_point[0] - icomx;
      lever[1] = contact_point[1] - icomy;
      lever[2] = contact_point[2] - icomz;
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

        lever[0] = contact_point[0] - jcomx;
        lever[1] = contact_point[1] - jcomy;
        lever[2] = contact_point[2] - jcomz;
        domain->minimum_image(lever);

        MathExtra::cross3(lever, fpair, torque_pair);

        torque[j][0] += torque_pair[0];
        torque[j][1] += torque_pair[1];
        torque[j][2] += torque_pair[2];
      }

      // Virial contribution: need to check
      fpair_mag = MathExtra::len3(fpair);
      if (evflag) ev_tally(i, j, nlocal, 0, evdwl, 0.0, fpair_mag, fpair[0]/fpair_mag, fpair[1]/fpair_mag, fpair[2]/fpair_mag);
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
  memory->create(etan, np1, np1, "pair:etan");
  memory->create(etat, np1, np1, "pair:etat");
  memory->create(knp, np1, np1, "pair:knp");
  memory->create(cut, np1, np1, "pair:cut");
  memory->create(decayn1, np1, np1, "pair:decayn1");
  memory->create(etan1, np1, np1, "pair:etan1");
  memory->create(decayt1, np1, np1, "pair:decayt1");
  memory->create(etat1, np1, np1, "pair:etat1");
  //memory->create(gamma, np1, np1, "pair:gamma");
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

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLSDEM::coeff(int narg, char **arg)
{
  if (narg < 7)
    error->all(FLERR, "Incorrect number of args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double kn_0 = utils::numeric(FLERR, arg[2], false, lmp);
  double kt_0 = utils::numeric(FLERR, arg[3], false, lmp);
  double mu_0 = utils::numeric(FLERR, arg[4], false, lmp);
  double etan_0 = utils::numeric(FLERR, arg[5], false, lmp);
  double etat_0 = utils::numeric(FLERR, arg[6], false, lmp);
  double knp_0 = utils::numeric(FLERR, arg[7], false, lmp);
  double cut_one = utils::numeric(FLERR, arg[8], false, lmp);
  double kn_1 = utils::numeric(FLERR, arg[9], false, lmp);
  double etan_1 = utils::numeric(FLERR, arg[10], false, lmp);
  double kt_1 = utils::numeric(FLERR, arg[11], false, lmp);
  double etat_1 = utils::numeric(FLERR, arg[12], false, lmp);
  //double gamma_one = utils::numeric(FLERR, arg[6], false, lmp); // Doesn't do anything IIRC

  if (kn_0 < 0.0) error->all(FLERR, "Incorrect negative args for pair coefficients.");
  if (kt_0 < 0.0) error->all(FLERR, "Incorrect negative args for pair coefficients.");
  if (mu_0 < 0.0) error->all(FLERR, "Incorrect negative args for pair coefficients.");
  if (etan_0 < 0.0) error->all(FLERR, "Incorrect negative args for pair coefficients.");
  if (etat_0 < 0.0) error->all(FLERR, "Incorrect negative args for pair coefficients.");
  // Values of knp_0 can be both positive and negative.
  if (narg >= 7){
    if (kn_1 < 0.0) error->all(FLERR, "Incorrect negative args for pair coefficients.");
    if (etan_1 < 0.0) error->all(FLERR, "Incorrect negative args for pair coefficients.");
    // If active, neither k or eta in a Maxwell arm are allowed to be zero. Check if both zero or both positive.
    if ( (kn_1 == 0.0) ^ (etan_1 == 0.0) ) {
      error->all(FLERR, "Incorrect args for pair coefficients. Maxwell arm requires k and eta to both be zero or both be positive.");
    }
  }
  if (narg >= 9){
    if (kt_1 < 0.0) error->all(FLERR, "Incorrect negative args for pair coefficients.");
    if (etat_1 < 0.0) error->all(FLERR, "Incorrect negative args for pair coefficients.");
    if ( (kt_1 == 0.0) ^ (etat_1 == 0.0) ) {
      error->all(FLERR, "Incorrect args for pair coefficients. Maxwell arm requires k and eta to both be zero or both be positive.");
    }
  }

  // Determine contact model type, this pre-computed flag helps evaluate branching
  // conditions more economically
  // int mode = 0; // Default mode: purely elastic
  // if ((etan_0 > 0.0) || (etat_0 > 0.0)){
  //   mode = 1; // Elastic + parallel viscous damping
  // }
  // if ((kn_1 > 0.0) || (etan_1 > 0.0) || (kt_1 > 0.0) || (etat_1 > 0.0)){
  //   mode = 2; // Generalised Maxwell solid
  // }
  // Neither k or eta in a Maxwell arm are allowed to be zero.

  // Need a check for dt staying constant. If it changes, decayn1 and decayt1 need to be recomputed.

  double dt = update->dt;
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      kn[i][j] = kn_0;
      kt[i][j] = kt_0;
      mu[i][j] = mu_0;
      etan[i][j] = etan_0;
      etat[i][j] = etat_0;
      knp[i][j] = knp_0;
      cut[i][j] = cut_one;
      if (narg >= 7){
        decayn1[i][j] = exp(-dt/etan_1*kn_1);
        etan1[i][j] = etan_1;
      }
      if (narg >= 9){
        decayt1[i][j] = exp(-dt/etat_1*kt_1);
        etat1[i][j] = etat_1;
      }

      // gamma[i][j] = gamma_one;
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
      maxcut2 = MAX(maxcut2, cut[i][j]); // Can we compute a sensible value for this somehow?

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
  index_ls_dem_n = atom->find_custom("ls_dem_n", tmp1, tmp2);
  index_ls_dem_fs = atom->find_custom("ls_dem_fs", tmp1, tmp2);
  index_ls_dem_touch_id = atom->find_custom("ls_dem_touch_id", tmp1, tmp2);
  index_ls_dem_fn1 = atom->find_custom("ls_dem_fn1", tmp1, tmp2);
  index_ls_dem_fs1 = atom->find_custom("ls_dem_fs1", tmp1, tmp2);
  index_ls_dem_node_area = atom->find_custom("ls_dem_node_area", tmp1, tmp2);
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
    etan[i][j] = mix_energy(etan[i][i], etan[j][j], cut[i][i], cut[j][j]);
    etat[i][j] = mix_energy(etat[i][i], etat[j][j], cut[i][i], cut[j][j]);
    knp[i][j] = mix_energy(knp[i][i], knp[j][j], cut[i][i], cut[j][j]);
    decayn1[i][j] = mix_energy(decayn1[i][i], decayn1[j][j], cut[i][i], cut[j][j]);
    etan1[i][j] = mix_energy(etan1[i][i], etan1[j][j], cut[i][i], cut[j][j]);
    decayt1[i][j] = mix_energy(decayt1[i][i], decayt1[j][j], cut[i][i], cut[j][j]);
    etat1[i][j] = mix_energy(etat1[i][i], etat1[j][j], cut[i][i], cut[j][j]);
    //gamma[i][j] = mix_energy(gamma[i][i], gamma[j][j], cut[i][i], cut[j][j]);
  }

  // DvdH: For most contact models mixing will not be simple.
  // I would probably discourage the use of this, or give a warning.

  // Enforces symmetry
  cut[j][i] = cut[i][j];
  kn[j][i] = kn[i][j];
  kt[j][i] = kt[i][j];
  mu[j][i] = mu[i][j];
  knp[j][i] = knp[i][j];
  etan[i][j] = etan[j][i];
  etat[i][j] = etat[j][i];
  decayn1[i][j] = decayn1[j][i];
  etan1[i][j] = etan1[j][i];
  decayt1[i][j] = decayt1[j][i];
  etat1[i][j] = etat1[j][i];
  //gamma[j][i] = gamma[i][j];

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
        fwrite(&etan[i][j], sizeof(double), 1, fp);
        fwrite(&etat[i][j], sizeof(double), 1, fp);
        fwrite(&knp[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
        fwrite(&decayn1[i][j], sizeof(double), 1, fp);
        fwrite(&etan1[i][j], sizeof(double), 1, fp);
        fwrite(&decayt1[i][j], sizeof(double), 1, fp);
        fwrite(&etat1[i][j], sizeof(double), 1, fp);
        //fwrite(&gamma[i][j], sizeof(double), 1, fp);
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
          utils::sfread(FLERR, &etan[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &etat[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &knp[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &decayn1[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &etan1[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &decayt1[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &etat1[i][j], sizeof(double), 1, fp, nullptr, error);
          //utils::sfread(FLERR, &gamma[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&kn[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&kt[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&mu[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&etan[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&etat[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&knp[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&decayn1[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&etan1[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&decayt1[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&etat1[i][j], 1, MPI_DOUBLE, 0, world);
        //MPI_Bcast(&gamma[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLSDEM::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp, "%d %g %g %g %g %g %g %g %g %g %g %g\n", i, kn[i][i], kt[i][i], mu[i][i], etan[i][i], etat[i][i], knp[i][i], cut[i][i],
      decayn1[i][i], etan1[i][i], decayt1[i][i], etat1[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLSDEM::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g %g %g %g %g %g %g %g\n", i, j, kn[i][j], kt[i][j], mu[i][j], etan[i][j], etat[i][j], knp[i][j], cut[i][j],
        decayn1[i][j], etan1[i][j], decayt1[i][j], etat1[i][j]);
}
