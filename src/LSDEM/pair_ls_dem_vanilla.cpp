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

#include "pair_ls_dem_vanilla.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_rigid_ls_dem.h"
#include "fix_rigid_small_ls_dem.h"
#include "force.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static constexpr double EPSILON = 1e-12;

using namespace LAMMPS_NS;
using namespace MathConst;

//TODO:
//  add checks for key overflow
//  add checks to rigid/small/ls/dem nbodies < 2bil (or generalize)
//  should watershed be a 2nd toggle and not a memory style? Such that global also creates (but does not communicates) the data?
//    primarily, is it faster than doing the 2nd nlist loop?
//  create a 2nd set of page files and process nlist whenever built to skip far atoms

/* ---------------------------------------------------------------------- */

PairLSDEMVanilla::PairLSDEMVanilla(LAMMPS *_lmp) : PairLSDEM(_lmp)
{
}

/* ---------------------------------------------------------------------- */

PairLSDEMVanilla::~PairLSDEMVanilla()
{
}

/* ---------------------------------------------------------------------- */

void PairLSDEMVanilla::compute(int eflag, int vflag)
{
  int i, j, i2, j2, ii, jj, inum, jnum, ibody, jbody, ibodyID, jbodyID;
  int maxbodyID_i, maxbodyID_j, offset_i, offset_j, tmp_bin;
  int *ilist, *jlist, *numneigh, **firstneigh;
  double rsq, delx, dely, delz, xitmp, yitmp, zitmp, ivol, jvol, x_local[3];
  tagint itag, jtag;
  long key;

  // Currently require:
  //   Newton pair off.
  //   Exclude intramolecule interactions
  // In future, remove restrictions

  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **x = atom->x;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  tagint *molecule = atom->molecule;

  double *special_lj = force->special_lj;
  int nlocal = atom->nlocal;
  double maxcutsq = maxcut * maxcut;


  // body info of local and ghost atoms only changes when the neighbor list is rebuilt

  int ntotal = atom->nlocal + atom->nghost;
  if ((neighbor->lastcall != binfo_lastbuild) || ((int) binfo.size() < ntotal)) {
    cache_body_info(ntotal);
    binfo_lastbuild = neighbor->lastcall;
    checked_bins.assign(ntotal, {});
  }

  int maxbodyID_large = 0;
  int maxbodyID_small = 0;
  double *grain_vol;
  if (fix_rigid) { // How is nbody updated during fix pour?
    mybody_large = fix_rigid->get_body_array();
    maxbodyID_large = fix_rigid->get_nbody();
    grain_vol = fix_rigid->get_vol_array();
    node_area = fix_rigid->get_area_array();
  }

  if (fix_rigid_small) {
    mybody_small = fix_rigid_small->get_atom2body_array();
    maxbodyID_small = fix_rigid_small->get_maxmol();
    bodyLS = fix_rigid_small->get_bodyLS_array();
  }

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  long ibodykey, jbodykey;
  long maxbodyID = MAX(maxbodyID_small, maxbodyID_large);

  // Loop over local + ghost atoms to find closest neighbors
  if (watershed_flag == 0) {

    int allnum = inum + list->gnum;
    std::unordered_map<long, std::pair<double, std::pair<int, int>>> min_distances;
    min_distances.reserve(static_cast<size_t>(4) * allnum);

    for (ii = 0; ii < allnum; ii++) {
      // Loop through local nodes
      i = ilist[ii];

      if (binfo[i].bidx < 0) continue;

      ivol = binfo[i].vol;
      ibodyID = binfo[i].bID;
      offset_i = binfo[i].off;

      itag = tag[i];
      xitmp = x[i][0];
      yitmp = x[i][1];
      zitmp = x[i][2];
      //ibodykey = 2 * ibodyID + offset_i;

      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        if (special_lj[sbmask(j)] == 0) continue;
        j &= NEIGHMASK;

        if (binfo[j].bidx < 0) continue;

        delx = xitmp - x[j][0];
        dely = yitmp - x[j][1];
        delz = zitmp - x[j][2];
        rsq = delx * delx + dely * dely + delz * delz;
        if (rsq > maxcutsq) continue;

        jvol = binfo[j].vol;
        jbodyID = binfo[j].bID;
        offset_j = binfo[j].off;

        jtag = tag[j];

        if (use_nodes_of_i(i, j, ivol, jvol)) {
          // node i probes body j
          key = 2 * (maxbodyID * itag + jbodyID) + offset_j;
          if (min_distances.find(key) == min_distances.end() || rsq < min_distances[key].first)
            min_distances[key] = std::make_pair(rsq, std::make_pair(i, j));
        } else {
          // node j probes body i
          key = 2 * (maxbodyID * jtag + ibodyID) + offset_i;
          if (min_distances.find(key) == min_distances.end() || rsq < min_distances[key].first)
            min_distances[key] = std::make_pair(rsq, std::make_pair(j, i));
        }
      }
    }

    // Loop through list of min_distances
    MathExtra::zero3(x_local);
    for (const auto &entry : min_distances) {
      i = entry.second.second.first;
      j = entry.second.second.second;

      if (i >= nlocal && j >= nlocal) continue;

      process_contact(i, j, -1, x_local);
    }
  } else {
    std::unordered_set<long> found_watershed;
    std::vector<std::unordered_map<long, std::tuple<int, double, double, double>>> saved_bins;
    int j2large;
    int ntotal = atom->nlocal + atom->nghost;
    saved_bins.resize(ntotal);
    long jbodykey2;

    for (ii = 0; ii < inum; ii++) {
      // Loop through local nodes
      i = ilist[ii];

      if (binfo[i].bidx < 0) continue;

      ivol = binfo[i].vol;
      ibodyID = binfo[i].bID;
      offset_i = binfo[i].off;

      itag = tag[i];
      xitmp = x[i][0];
      yitmp = x[i][1];
      zitmp = x[i][2];
      ibodykey = 2 * ibodyID + offset_i;

      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        if (special_lj[sbmask(j)] == 0) continue;
        j &= NEIGHMASK;

        if (binfo[j].bidx < 0) continue;

        delx = xitmp - x[j][0];
        dely = yitmp - x[j][1];
        delz = zitmp - x[j][2];
        rsq = delx * delx + dely * dely + delz * delz;
        if (rsq > maxcutsq) continue;

        jvol = binfo[j].vol;
        jbodyID = binfo[j].bID;
        offset_j = binfo[j].off;

        jbodykey = 2 * jbodyID + offset_j;

        if (use_nodes_of_i(i, j, ivol, jvol)) {
          i2 = i;
          j2 = j;
          jbodykey2 = jbodykey;
          key = 2 * (maxbodyID * itag + jbodyID) + offset_j;
        } else {
          i2 = j;
          j2 = i;
          jbodykey2 = ibodykey;
          key = 2 * (maxbodyID * tag[j] + ibodyID) + offset_i;
        }

        // Skip if we've already found the watershed for atom i2 on j2's body
        if (found_watershed.find(key) != found_watershed.end()) continue;

        j2large = binfo[j2].grp == 2;

        // Check if I already calculated bin for atom i2 on j2's body
        if (saved_bins[i2].find(jbodykey2) == saved_bins[i2].end()) {
          if (j2large) tmp_bin = fix_rigid->get_bin(i2, j2, x_local);
          else tmp_bin = fix_rigid_small->get_bin(i2, j2, x_local);

          saved_bins[i2][jbodykey2] = std::make_tuple(tmp_bin, x_local[0], x_local[1], x_local[2]);
        } else {
          tmp_bin = std::get<0>(saved_bins[i2][jbodykey2]);
          x_local[0] = std::get<1>(saved_bins[i2][jbodykey2]);
          x_local[1] = std::get<2>(saved_bins[i2][jbodykey2]);
          x_local[2] = std::get<3>(saved_bins[i2][jbodykey2]);
        }

        // Check if j2 owns this watershed bin
        if (checked_bins[j2].find(tmp_bin) == checked_bins[j2].end()) {
          if (j2large) {
            if (fix_rigid->check_watershed_bin(tmp_bin, j2) ==   0) continue;
          } else {
            if (fix_rigid_small->check_watershed_bin(tmp_bin, j2) == 0) continue;
          }
          checked_bins[j2].insert(tmp_bin);
        }

        found_watershed.insert(key);
        process_contact(i2, j2, tmp_bin, x_local);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   contact model for node i on node j's body
   touching the level set of the other grain, applies the force and torque to
   atom i and, if it is a local atom, to atom j
   tmp_bin and x_local are the watershed bin and grid position (tmp_bin = -1 for array)
------------------------------------------------------------------------- */

void PairLSDEMVanilla::process_contact(int i, int j, int tmp_bin, double *x_local)
{
  int itype, jtype, ibody, jbody, ibodyID, jbodyID;
  double xitmp, yitmp, zitmp, xjtmp, yjtmp, zjtmp, u;
  double vxitmp, vyitmp, vzitmp, vxjtmp, vyjtmp, vzjtmp;
  double icomx, icomy, icomz, jcomx, jcomy, jcomz;
  double iomegax, iomegay, iomegaz, jomegax, jomegay, jomegaz, spin_norm;
  double normal[3], fn_mag, fpair[3], contact_point[3], lever[3], torque_pair[3];
  double fs_tmp[3], fs_mag, fs_max, fs_mag_trial, k[3], costheta, term1[3], term2;
  double tangent[3], shear_incr, v_rel[3], v_rel_t[3], v_rel_n_mag, v_rel_t_mag, v_rel_t_mag_inv;
  double normal_old[3], tangent_old[3], fs_mag_add, areai, areaj;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **torque = atom->torque;
  double **n = atom->darray[index_ls_dem_n];
  double **fs = atom->darray[index_ls_dem_fs];
  int *touch_id = atom->ivector[index_ls_dem_touch_id];
  double *fn1 = atom->dvector[index_ls_dem_fn1];
  double *fs1 = atom->dvector[index_ls_dem_fs1];
  tagint *tag = atom->tag;
  tagint *molecule = atom->molecule;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  double dt = update->dt;
  double **grain_com = atom->xcom;
  double **grain_omega = atom->omega;

  itype = type[i];
  jtype = type[j];
  xitmp = x[i][0];
  yitmp = x[i][1];
  zitmp = x[i][2];
  vxitmp = v[i][0];
  vyitmp = v[i][1];
  vzitmp = v[i][2];
  icomx = grain_com[i][0];
  icomy = grain_com[i][1];
  icomz = grain_com[i][2];
  iomegax = grain_omega[i][0];
  iomegay = grain_omega[i][1];
  iomegaz = grain_omega[i][2];

  if (fix_rigid && (mask[i] & groupbit_large)) {
    ibody = mybody_large[i];
    ibodyID = ibody;
    areai = node_area[ibody];
  } else if (fix_rigid_small && (mask[i] & groupbit_small)) {
    ibody = mybody_small[i];
    ibodyID = (int) molecule[i];
    areai = bodyLS[ibody].node_area;
  }

  int large_fix;
  if (fix_rigid && (mask[j] & groupbit_large)) {
    jbody = mybody_large[j];
    jbodyID = jbody;
    areaj = node_area[jbody];
    large_fix = 1;
  } else if (fix_rigid_small && (mask[j] & groupbit_small)) {
    jbody = mybody_small[j];
    jbodyID = (int) molecule[j];
    areaj = bodyLS[jbody].node_area;
    large_fix = 0;
  }

  // ghosts may not find bodytag if comm distance too small
  if (ibody < 0)
    error->one(FLERR, "Atom {} cannot find atom that owns body, consider increasing the comm cutoff", tag[i]);
  if (jbody < 0)
    error->one(FLERR, "Atom {} cannot find atom that owns body, consider increasing the comm cutoff", tag[j]);

  // Evaluate the level set of the other grain (j) at the node used (i), one call for
  // both directions. Force magnitude and direction go i -> j by definition.
  // Level set is by definition negative inside the particle,
  // so swap the sign to get the overlap distance.
  if (large_fix)
    u = -fix_rigid->get_ls_value(i, j, tmp_bin, normal, x_local);
  else
    u = -fix_rigid_small->get_ls_value(i, j, tmp_bin, normal, x_local);

  double *fs_i = fs[i];
  double *n_i = n[i];

  // No adhesion, cohesion, or ranged forces.
  if (u <= 0) {
    // Reset shear force if no contact
    // If i and j are not a shear-interacting pair, it will skip the reset
    if (touch_id[i] == jbodyID) {
      touch_id[i] = -1;
      fs_i[0] = 0.0;
      fs_i[1] = 0.0;
      fs_i[2] = 0.0;
      n_i[0] = 0.0;
      n_i[1] = 0.0;
      n_i[2] = 0.0;
    }
    // Skip force calculation
    return;
  } else {    // Physical contact!!
    // These per-neighbour quantities are only needed once contact is
    // confirmed, so they are loaded here instead of for every neighbour pair.
    xjtmp = x[j][0];
    yjtmp = x[j][1];
    zjtmp = x[j][2];
    vxjtmp = v[j][0];
    vyjtmp = v[j][1];
    vzjtmp = v[j][2];
    jcomx = grain_com[j][0];
    jcomy = grain_com[j][1];
    jcomz = grain_com[j][2];
    jomegax = grain_omega[j][0];
    jomegay = grain_omega[j][1];
    jomegaz = grain_omega[j][2];

    // Compute contact point, negate normal
    // The normal returned for node i points away from j, so it is negated to point i->j
    MathExtra::negate3(normal);
    contact_point[0] = xitmp - 0.5 * u * normal[0];
    contact_point[1] = yitmp - 0.5 * u * normal[1];
    contact_point[2] = zitmp - 0.5 * u * normal[2];
  }

  const double knij = kn[itype][jtype];
  const double ktij = kt[itype][jtype];
  const double muij = mu[itype][jtype];
  const double knpij = knp[itype][jtype];
  const double etanij = etan[itype][jtype];
  const double etatij = etat[itype][jtype];
  const double etan1ij = etan1[itype][jtype];
  const double decayn1ij = decayn1[itype][jtype];
  const double etat1ij = etat1[itype][jtype];
  const double decayt1ij = decayt1[itype][jtype];

  ///////////////////
  // Normal stress //
  ///////////////////

  // Elastic spring
  // With positive penetration distance u
  if (fabs(knpij) < EPSILON) {
    fn_mag = knij * u;
  } else {
    fn_mag = knij * pow(u, knpij);
  }

  // Relative velocity at the grain surface at the half step t + 0.5*dt.
  // Note: The velocity at the node due to an angular velocity of the grain around its
  // centre of mass is already included, so the difference of linear velocities suffices.
  v_rel[0] = vxitmp - vxjtmp;
  v_rel[1] = vyitmp - vyjtmp;
  v_rel[2] = vzitmp - vzjtmp;

  // Relative velocity in normal direction with sign (positive for approach)
  v_rel_n_mag = MathExtra::dot3(v_rel, normal);

  // Viscous damping or dashpot (parallel)
  if (etanij > 0.0) { fn_mag += etanij * v_rel_n_mag; }

  // Maxwell arm (1st, parallel, only repulsive)
  if (etan1ij > 0.0) {    // preprocessing guarantees that decayn1 > 0 if etan1 > 0
    fn1[i] = decayn1ij * fn1[i] + etan1ij * (1.0 - decayn1ij) * v_rel_n_mag;
    fn_mag += fn1[i];
    // Maxwell arm (2nd)
    //fn2_mag[i] = decayn2[itype][jtype] * fh2_mag[i] + etan2[itype][jtype] * (1-decayn2[itype][jtype]) * MAX(v_rel_n_mag, 0.0);
    //fn_mag += fn2_mag[i]
  }

  // Only repulsive, do not allow tensile force without explicit bonding / cohesion
  fn_mag = MAX(fn_mag, 0.0);

  // The pair force vector should point j->i because of repulsion.
  // With normal n (i->j), we have: F_{j on i} = f(ls_value) = - fn_mag * n.
  fpair[0] = -fn_mag * normal[0];
  fpair[1] = -fn_mag * normal[1];
  fpair[2] = -fn_mag * normal[2];

  ////////////////////
  // Tangent stress //
  ////////////////////

  // Tangent force only exists if mu > 0 and kt > 0
  //if ( (kt[itype][jtype] > 0) && (mu[itype][jtype] > 0) ){
  // DvdH: Grains without friction are silly, so I took out this check.

  // Check if the pair is valid for shear history calculation
  // Initialise if no contact
  if (touch_id[i] == -1) { touch_id[i] = jbodyID; }
  if (touch_id[i] != jbodyID) {
    if (comm->me == 0) {
      error->warning(FLERR,
                     "Shear history of node {} on grain {} penetrating node {} on {} cannot be "
                     "computed at step {}",
                     tag[i], ibodyID, tag[j], jbodyID, update->ntimestep);
    }
  }

  // Get old elastic shear stress and old node normal.
  MathExtra::copy3(fs_i, fs_tmp);
  MathExtra::copy3(n_i, normal_old);

  // Adjust fs_tmp to account for rotation of the contact normal and plane.
  if (MathExtra::lensq3(normal_old) > 0.0) {    // len3 > 0, but without the sqrt
    // Account for tilt. This is an exact correction over rotation of the normal
    // from the previous to the current time step.
    MathExtra::cross3(normal_old, normal, k);    // n_old x n_new

    // Account for spin. This is an approximation using the half-step angular
    // velocities. We rotate about the OLD normal so that, together with the
    // tilt, the additive reconstruction matches the spin-then-tilt rotation.
    // spin_norm = dt*(omega_avg . n_old) = dt*|omega_sp| (signed).
    spin_norm = 0.5 * dt *
        ((iomegax + jomegax) * normal_old[0] + (iomegay + jomegay) * normal_old[1] +
         (iomegaz + jomegaz) * normal_old[2]);    // 0.5*dt*(omegai+omegaj) \dot n_old

    // First Baker-Campbell-Hausdorff corrective term for the non-commutativity
    // of the spin and tilt rotations: -0.5*|omega_sp|*dt*(n_old x (n_old x n_new)).
    // The non-commutativity error dominates and only manifests itself in the tilt.
    MathExtra::cross3(normal_old, k, term1);    // n_old x (n_old x n_new)
    double bch_coef = -0.5 * fabs(spin_norm);
    k[0] += spin_norm * normal_old[0] + bch_coef * term1[0];
    k[1] += spin_norm * normal_old[1] + bch_coef * term1[1];
    k[2] += spin_norm * normal_old[2] + bch_coef * term1[2];

    // Applying the rotation with the unnormalised Rodrigues formula:
    //   k already carries axis * sin(theta), so neither the axis nor the cross term needs normalising
    //   and (1 - cos) / sin^2 = 1 / (1 + cos) avoids small-angle cancellation
    //   fs_rot = fs cos + (k x fs) + k (k . fs) / (1 + cos), with cos = sqrt(1 - |k|^2)
    double sinsq = MathExtra::lensq3(k);    // sin^2(theta) = |k|^2
    if (sinsq > EPSILON * EPSILON) {        // Don't apply rotation if magnitude is tiny
      costheta = sqrt(MAX(1.0 - sinsq, 0.0));
      MathExtra::cross3(k, fs_tmp, term1);
      term2 = MathExtra::dot3(k, fs_tmp) / (1.0 + costheta);
      fs_tmp[0] = fs_tmp[0] * costheta + term1[0] + k[0] * term2;
      fs_tmp[1] = fs_tmp[1] * costheta + term1[1] + k[1] * term2;
      fs_tmp[2] = fs_tmp[2] * costheta + term1[2] + k[2] * term2;
    }
  }

  // Relative velocity in tangential direction
  v_rel_t[0] = v_rel[0] - v_rel_n_mag * normal[0];
  v_rel_t[1] = v_rel[1] - v_rel_n_mag * normal[1];
  v_rel_t[2] = v_rel[2] - v_rel_n_mag * normal[2];
  v_rel_t_mag = MathExtra::len3(v_rel_t);
  // We note that this way of calculating the shear velocity is an approximation.
  // The error lies mainly in the fact that the velocity difference was computed
  // between the surface nodes, meaning that there is a small arm length that has
  // not been accounted for. As a consequence, this breaks objectivity since v_rel_t
  // will increase with the rigid body rotation omega_b. The extra arm length does
  // not unjustifiably increase v_rel_t in the absence of rigid body motion, since
  // the surface can indeed be said to move at omega x R.

  // Tangent normal
  if (v_rel_t_mag > EPSILON) {
    v_rel_t_mag_inv = 1.0 / v_rel_t_mag;
    tangent[0] = v_rel_t[0] * v_rel_t_mag_inv;
    tangent[1] = v_rel_t[1] * v_rel_t_mag_inv;
    tangent[2] = v_rel_t[2] * v_rel_t_mag_inv;
  } else {
    // Backup: Use old shear force direction as tangent.
    double norm = MathExtra::len3(fs_tmp);
    if (norm != 0) {
      v_rel_t_mag_inv = 1.0 / norm;
    } else {
      v_rel_t_mag_inv = 0.0;
    }
    tangent[0] = fs_tmp[0] * v_rel_t_mag_inv;
    tangent[1] = fs_tmp[1] * v_rel_t_mag_inv;
    tangent[2] = fs_tmp[2] * v_rel_t_mag_inv;
  }

  // Elastic spring shear stress increment
  shear_incr = ktij * v_rel_t_mag * dt;

  // New elastic force
  fs_tmp[0] -= shear_incr * tangent[0];
  fs_tmp[1] -= shear_incr * tangent[1];
  fs_tmp[2] -= shear_incr * tangent[2];
  fs_mag_trial = MathExtra::len3(fs_tmp);

  // Coulomb limit
  fs_max = muij * fn_mag;

  // Perfectly plastic Coulomb friction criterion
  fs_mag = std::min(fs_max, fs_mag_trial);

  // Final shear or tangential stress
  if (fs_mag_trial > EPSILON) {
    fs_tmp[0] = fs_mag * (fs_tmp[0] / fs_mag_trial);
    fs_tmp[1] = fs_mag * (fs_tmp[1] / fs_mag_trial);
    fs_tmp[2] = fs_mag * (fs_tmp[2] / fs_mag_trial);
  }

  // Update saved elastic shear stress and normal (back into the node's i->j frame)
  MathExtra::copy3(fs_tmp, fs_i);
  MathExtra::copy3(normal, n_i);

  // Placeholder for viscous and viscoelastic stress components
  fs_mag_add = 0.0;

  // Viscous damping or dashpot (parallel, allowed in any direction)
  if (etatij > 0) { fs_mag_add += etatij * v_rel_t_mag; }

  // Maxwell arm (1st, parallel, only repulsive)
  if (etan1ij > 0.0) {    // preprocessing guarantees that decayt1 > 0 if etat1 > 0
    // Get the old tangent vector
    double norm = MathExtra::len3(fs_tmp);
    if (norm != 0) {
      v_rel_t_mag_inv = 1.0 / norm;
    } else {
      v_rel_t_mag_inv = 0.0;
    }
    tangent_old[0] = fs_tmp[0] * v_rel_t_mag_inv;
    tangent_old[1] = fs_tmp[1] * v_rel_t_mag_inv;
    tangent_old[2] = fs_tmp[2] * v_rel_t_mag_inv;
    // The dot(t_old,t) part accounts for the in-plane rotation of the tangent force.
    // When the shear direction reverses, it correctly preserves the direction of the old force.
    // However, when rotating towards the orthogonal direction, we inevitably lose some of the force.
    // We could track the full vector, but it would cost more memory (and accessing time)
    fs1[i] += decayt1ij * fs1[i] * MathExtra::dot3(tangent_old, tangent) +
        etat1ij * (1 - decayt1ij) * v_rel_t_mag;    // v_rel_t_mag is always positive
    fs_mag_add += fs1[i];
    // Maxwell arm (2nd)
    // fs2_mag[i] = exps2*fs2_mag[i] + etat2[itype][jtype]*(1-exps2)*v_rel_t_mag;
    // fs_mag -= fs2_mag[i]
  }

  // Add potential viscous and viscoelastic components to shear stress (repulsive again)
  fs_tmp[0] -= fs_mag_add * tangent[0];
  fs_tmp[1] -= fs_mag_add * tangent[1];
  fs_tmp[2] -= fs_mag_add * tangent[2];
  // Update trial shear stress magnitude
  fs_mag_trial = MathExtra::len3(fs_tmp);

  // Re-apply plastic Coulomb friction criterion
  fs_mag = std::min(fs_max, fs_mag_trial);

  if (fs_mag > 0 && fs_mag_trial != 0) {
    // Final shear or tangential stress
    fs_tmp[0] = fs_mag * (fs_tmp[0] / fs_mag_trial);
    fs_tmp[1] = fs_mag * (fs_tmp[1] / fs_mag_trial);
    fs_tmp[2] = fs_mag * (fs_tmp[2] / fs_mag_trial);
    // Add shear stress to the pair force stress
    fpair[0] += fs_tmp[0];
    fpair[1] += fs_tmp[1];
    fpair[2] += fs_tmp[2];
  }

  //} // Check if kt > 0 and mu > 0

  //////////////////////////////
  // Total forces and torques //
  //////////////////////////////

  // Multiply by node area to make the force independent of discretisation (fpair was a stress)
  fpair[0] *= areai;
  fpair[1] *= areai;
  fpair[2] *= areai;

  // Force on grain i
  f[i][0] += fpair[0];
  f[i][1] += fpair[1];
  f[i][2] += fpair[2];

#ifndef NDEBUG
  if (std::isnan(fpair[0]) || std::isnan(fpair[1]) || std::isnan(fpair[2]))
    error->one(FLERR,
               "Bad force calculated between atoms {} {} on bodies {} {} with overlap {} and "
               "normal {} {} {}\n",
               tag[i], tag[j], ibody, jbody, u, normal[0], normal[1], normal[2]);
#endif

  // Lever arm on grain i
  lever[0] = contact_point[0] - icomx;
  lever[1] = contact_point[1] - icomy;
  lever[2] = contact_point[2] - icomz;
  // Account for PBCs
  domain->minimum_image(FLERR, lever);

  // Compute torque
  MathExtra::cross3(lever, fpair, torque_pair);

  // Apply torques on grain i
  torque[i][0] += torque_pair[0];
  torque[i][1] += torque_pair[1];
  torque[i][2] += torque_pair[2];

#ifndef NDEBUG
  if (std::isnan(torque_pair[0]) || std::isnan(torque_pair[1]) || std::isnan(torque_pair[2]))
    error->one(FLERR,
               "Bad torque calculated between atoms {} {} on bodies {} {} with overlap {} and "
               "normal {} {} {}\n",
               tag[i], tag[j], ibody, jbody, u, normal[0], normal[1], normal[2]);
#endif

  // Virial of the contact split at the contact point: node i gets (x_i - contact point) x F
  //   and node j gets (contact point - x_j) x F, with F the force on i, each on the rank that
  //   owns it. The sum is (x_i - x_j) x F, as the constraint virial of the rigid fixes uses the
  //   atom positions, and per grain the centroid stress sums to the contact point stress.
  if (evflag) {
    const double deli[3] = {x[i][0] - contact_point[0], x[i][1] - contact_point[1],
                            x[i][2] - contact_point[2]};
    const double delj[3] = {contact_point[0] - x[j][0], contact_point[1] - x[j][1],
                            contact_point[2] - x[j][2]};
    const double vi[6] = {deli[0] * fpair[0], deli[1] * fpair[1], deli[2] * fpair[2],
                          deli[0] * fpair[1], deli[0] * fpair[2], deli[1] * fpair[2]};
    const double vj[6] = {delj[0] * fpair[0], delj[1] * fpair[1], delj[2] * fpair[2],
                          delj[0] * fpair[1], delj[0] * fpair[2], delj[1] * fpair[2]};
    if (vflag_global) {
      for (int n = 0; n < 6; n++) virial[n] += vi[n];
      if (j < nlocal)
        for (int n = 0; n < 6; n++) virial[n] += vj[n];
    }
    if (vflag_atom) {
      for (int n = 0; n < 6; n++) vatom[i][n] += vi[n];
      if (j < nlocal)
        for (int n = 0; n < 6; n++) vatom[j][n] += vj[n];
    }
  }

  // Mirror forces and torques on grain j
  if (newton_pair || j < nlocal) {    // Need to check this again if we end up enabling newton_pair
    MathExtra::negate3(fpair);        // Sign swap
    f[j][0] += fpair[0];
    f[j][1] += fpair[1];
    f[j][2] += fpair[2];

    lever[0] = contact_point[0] - jcomx;
    lever[1] = contact_point[1] - jcomy;
    lever[2] = contact_point[2] - jcomz;
    domain->minimum_image(FLERR, lever);

    MathExtra::cross3(lever, fpair, torque_pair);

    torque[j][0] += torque_pair[0];
    torque[j][1] += torque_pair[1];
    torque[j][2] += torque_pair[2];
  }
}
