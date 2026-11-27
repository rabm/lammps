.. index:: compute rigid/ls/dem/local

compute rigid/ls/dem/local command
==================================

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID rigid/ls/dem/local rigidID attribute ...

* ID, group-ID are documented in the :doc:`compute <compute>` command
* rigidID = ID of a :doc:`fix rigid/small/ls/dem <fix_rigid_ls_dem>`
* attribute = any attribute of :doc:`compute rigid/local <compute_rigid_local>`,
  or *grid*, *net/quatw*, *net/quati*, *net/quatj*, *net/quatk*

Description
"""""""""""

Return one row per rigid body whose owning atom belongs to the group.
As with :doc:`compute rigid/local <compute_rigid_local>`, one attribute
produces a local vector and multiple attributes produce a local array.
The *grid* attribute gives the body's level-set grid index.  The four
*net/quat* components give the product of the body quaternion and the
fixed relative quaternion for its level-set grid.

Examples
""""""""

.. code-block:: LAMMPS

   compute grains all rigid/ls/dem/local bodies mol grid x y z
