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

#ifndef LMP_LS_DEM_SORT_KOKKOS_H
#define LMP_LS_DEM_SORT_KOKKOS_H

#include <Kokkos_Core.hpp>

namespace LAMMPS_NS {
namespace LSDEMKokkos {

  /* ----------------------------------------------------------------------
     heap sort of n items in one thread, less(x, y) compares and swap(x, y) exchanges items x and y
  ------------------------------------------------------------------------- */

  template <class Less, class Swap>
  KOKKOS_INLINE_FUNCTION void heap_sort(const int n, const Less &less, const Swap &swap)
  {
    auto sift = [&](int root, const int end) {
      while (2 * root + 1 < end) {
        int child = 2 * root + 1;
        if ((child + 1 < end) && less(child, child + 1)) child++;
        if (!less(root, child)) return;
        swap(root, child);
        root = child;
      }
    };
    for (int start = n / 2 - 1; start >= 0; start--) sift(start, n);
    for (int end = n - 1; end > 0; end--) {
      swap(0, end);
      sift(0, end);
    }
  }

}    // namespace LSDEMKokkos
}    // namespace LAMMPS_NS

#endif
