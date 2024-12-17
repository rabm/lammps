/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(rigid/lsdem,FixRigidLsdem);
// clang-format on
#else

#ifndef LMP_FIX_RIGID_LSDEM_H
#define LMP_FIX_RIGID_LSDEM_H

#include "fix.h"

namespace LAMMPS_NS {

class FixRigidLsdem : public Fix {
  friend class ComputeRigidLocal;

 public:
  FixRigidLsdem(class LAMMPS *, int, char **);
  ~FixRigidLsdem() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void post_force(int) override;
  void final_integrate() override;
  void initial_integrate_respa(int, int, int) override;
  void final_integrate_respa(int, int) override;
  void write_restart_file(const char *) override;

  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  void set_arrays(int) override;

  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  void setup_pre_neighbor() override;
  void pre_neighbor() override;
  bigint dof(int) override;
  void deform(int) override;
  void reset_dt() override;
  void zero_momentum() override;
  void zero_rotation() override;
  int modify_param(int, char **) override;
  void *extract(const char *, int &) override;
  double extract_ke();
  double extract_erotational();
  double compute_scalar() override;
  double memory_usage() override;

 protected:
  double dtv, dtf, dtq;
  double *step_respa;
  int triclinic;

  char *inpfile;       // file to read rigid body attributes from
  int setupflag;       // 1 if body properties are setup, else 0
  int earlyflag;       // 1 if forces/torques are computed at post_force()
  int commflag;        // various modes of forward/reverse comm
  int nbody;           // total # of rigid bodies
  int nlinear;         // total # of linear rigid bodies
  tagint maxmol;       // max mol-ID
  double maxextent;    // furthest distance from body owner to body atom

  struct Body {
    int natoms;            // total number of atoms in body
    int ilocal;            // index of owning atom
    double mass;           // total mass of body
    double xcm[3];         // COM position
    double xgc[3];         // geometric center position
    double vcm[3];         // COM velocity
    double fcm[3];         // force on COM
    double torque[3];      // torque around COM
    double quat[4];        // quaternion for orientation of body
    double inertia[3];     // 3 principal components of inertia
    double ex_space[3];    // principal axes in space coords
    double ey_space[3];
    double ez_space[3];
    double xgc_body[3];    // geometric center relative to xcm in body coords
    double angmom[3];      // space-frame angular momentum of body
    double omega[3];       // space-frame omega of body
    double conjqm[4];      // conjugate quaternion momentum
    int remapflag[4];      // PBC remap flags
    int nls[3];            // Number of level-set grid points in each principal direction
    int xls[3];            // Position of the 000 corner of level-set grid in principal directions, relative to the COM
    int sls[3];            // Stride of level-set grid in each principal direction
    int lls[3];            // Length of level-set grid in each principal direction
    float *** fls;         // Values of the level-set at grid points in principal coordinates
    imageint image;        // image flags of xcm
    imageint dummy;        // dummy entry for better alignment
  };

  Body *body;         // list of rigid bodies, owned and ghost
  int nlocal_body;    // # of owned rigid bodies
  int nghost_body;    // # of ghost rigid bodies
  int nmax_body;      // max # of bodies that body can hold
  int bodysize;       // sizeof(Body) in doubles

  // per-atom quantities
  // only defined for owned atoms, except bodyown for own+ghost

  int *bodyown;          // index of body if atom owns a body, -1 if not
  tagint *bodytag;       // ID of body this atom is in, 0 if none
                         // ID = tag of atom that owns body
  int *atom2body;        // index of owned/ghost body this atom is in, -1 if not
                         // can point to original or any image of the body
  imageint *xcmimage;    // internal image flags for atoms in rigid bodies
                         // set relative to in-box xcm of each body
  double **displace;     // displacement of each atom in body coords

  // temporary per-body storage

  int **counts;        // counts of atom types in bodies
  double **itensor;    // 6 space-frame components of inertia tensor

  // mass per body, accessed by granular pair styles

  double *mass_body;
  int nmax_mass;

  char *id_gravity;    // ID of fix gravity command to add gravity forces
  double *gvec;        // ptr to gravity vector inside the fix

  // class data used by ring communication callbacks

  double rsqfar;

  struct InRvous {
    int me, ilocal;
    tagint atomID, bodyID;
    double x[3];
  };

  struct OutRvous {
    int ilocal;
    tagint atomID;
  };

  // local methods

  void image_shift();
  void set_xv();
  void set_v();
  void create_bodies(tagint *);
  void setup_bodies_static();
  void setup_bodies_dynamic();
  virtual void compute_forces_and_torques();
  void enforce2d();
  void readfile(int, double **);
  void grow_body();
  void reset_atom2body();

  // callback function for rendezvous communication

  static int rendezvous_body(int, char *, int &, int *&, char *&, void *);

  // debug

  //void check(int);
};

}    // namespace LAMMPS_NS

#endif
#endif
