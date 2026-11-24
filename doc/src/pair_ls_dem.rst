.. index:: pair_style ls/dem

pair_style ls/dem command
=========================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style ls/dem [cutoff | auto [factor]] [method fast|transparent] [order list|exact]
   pair_coeff I J kn kt mu etan etat knp [cutoff | auto] [kn1 etan1 [kt1 etat1]]

Description
"""""""""""

This style computes contact forces between rigid grains represented by surface
nodes and level sets. It requires the LSDEM and RIGID packages.

.. versionadded:: TBD

Supply a numeric cutoff or explicitly request ``auto`` on the pair style or
its coefficients. A numeric style cutoff is the maximum LS-DEM search
distance. Coefficients that omit a cutoff inherit that style setting.

``pair_style ls/dem auto`` selects twice the largest estimated surface-node
spacing for every type pair assigned to LS-DEM. Use ``auto 3`` to select
three times the spacing. In this mode numeric cutoffs on ``pair_coeff``
are placeholders retained for compatibility. With a bare ``pair_style ls/dem``,
specify a cutoff or ``auto`` on each assigned coefficient. The :doc:`neighbor`
command and the units determine the shared neighbor skin.

A cutoff at least twice the standard automatic recommendation triggers a
one-time efficiency warning with the estimated distance. A molecule-based
rigid fix can also trigger a one-time warning recommending
``neigh_modify exclude molecule/intra group-ID``. This optional exclusion
affects other neighbor lists using that group. Enable it only when those
interactions can be omitted for all styles.

In the default fast method, each query node keeps a separate list of nearby
surface nodes for each partner grain. A neighbor rebuild can discard the
whole list when that grain's level set cannot reach the query node before
the next rebuild and there is no shear history to clear. At each step the
remaining lists supply the closest candidate or the owner of the watershed
bin. The transparent method looks up contacts directly for comparison.

For molecule-defined grains with ``fix rigid/small/ls/dem``, the ghost
communication cutoff must cover the largest owner-to-node distance plus
the LS-DEM pair cutoff. Use :doc:`comm_modify cutoff <comm_modify>` when
the neighbor cutoff is shorter. For a small number of large grains,
``fix rigid/ls/dem`` stores body information on every rank and avoids
this owner-distance requirement at the cost of global reductions.
