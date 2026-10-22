.. index:: fix rigid/ls/dem
.. index:: fix rigid/small/ls/dem
.. index:: fix rigid/small/ls/dem/kk

fix rigid/ls/dem command
========================

fix rigid/small/ls/dem command
==============================

Accelerator Variants: *rigid/small/ls/dem/kk*

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID rigid/ls/dem molecule infile filename ls/storage mode
   fix ID group-ID rigid/small/ls/dem molecule infile filename ls/storage mode

* ID, group-ID are documented in the :doc:`fix <fix>` command
* molecule = group the surface nodes into bodies by molecule ID
* filename = rigid body file containing body properties and level-set grid information
* mode = *array* (default) or *watershed*

Description
"""""""""""

These fixes integrate rigid grains represented by surface nodes with
:doc:`pair_style ls/dem <pair_ls_dem>` and atom style *ls/dem*.  The
*rigid/ls/dem* style stores the rigid body data on every processor.
The *rigid/small/ls/dem* style assigns each body to the processor that
owns its central atom.  See :doc:`fix rigid <fix_rigid>` for the body
integration options and :doc:`pair_style ls/dem <pair_ls_dem>` for the
required ghost region.

The *ls/storage* keyword selects the grid representation.  The body
file specifies the level-set grid for each grain and whether that grid
is global or distributed.

.. include:: accel_styles.rst
