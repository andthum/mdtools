# This file is part of MDTools.
# Copyright (C) 2021, The MDTools Development Team and all contributors
# listed in the file AUTHORS.rst
#
# MDTools is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# MDTools is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with MDTools.  If not, see <http://www.gnu.org/licenses/>.


"""
Functions to calculate structural properties.

This module can be called from :mod:`mdtools` via the shortcut ``strc``::

    import mdtools as mdt
    mdt.strc  # insetad of mdt.structure

"""


# Standard libraries
import warnings
from datetime import datetime, timedelta
# Third party libraries
import psutil
import numpy as np
from scipy import sparse
import MDAnalysis.lib.distances as mdadist
# Local application/library specific imports
import mdtools as mdt


def wcenter_pos(
    pos,
    weights=None,
    wrap_pos=False,
    wrap_result=False,
    box=None,
    mda_backend=None,
):
    r"""
    Calculate the weighted center of an position array.

    Parameters
    ----------
    pos : array_like
        The position array for which to calculate the weighted center.
        Must be either of shape ``(3,)``, ``(n, 3)`` or ``(k, n, 3)``,
        where ``n`` is the number of particles and ``k`` is the number
        of frames.  If `pos` has shape ``(k, n, 3)``, the center for
        each frame is computed.  If `pos` has shape ``(3,)``, i.e. `pos`
        contains the position of a single particle, the weighted center
        is identical to `pos`.
    weights : None or array_like, optional
        Array of shape ``(n,)`` containing the weight of each particle
        contained in `pos`.  If `weights` is ``None``, all particles are
        assumed to have a weight equal to one.
    wrap_pos : bool, optional
        If ``True``, wrap all positions in `pos` back to the primary
        unit cell using :func:`mdtools.box.wrap_pos` **before**
        calculating the weigted center.  Note that this likely splits
        molecules across periodic boundaries, which is undesired when
        calculating their centers.  If ``True``, `box` must be provided.
    wrap_result : bool, optional
        If ``True``, wrap the calculated center(s) into the primary unit
        cell using :func:`mdtools.box.wrap_pos`.  If ``True``, `box`
        must be provided.
    box : None or array_like, optional
        See :func:`mdtools.box.wrap_pos`.  Note that the shape of `pos`
        is interpreted as ``(k, 3)``, if both `pos` and `box` are
        2-dimensional.
    mda_backend : {None, 'serial', 'OpenMP'}, optional
        See :func:`mdtools.box.wrap_pos`.

    Returns
    -------
    center : numpy.ndarray
        The weighted center of all particle positions in `pos` per
        frame.

        .. list-table:: Shapes
            :align: left
            :header-rows: 1

            *   - `pos`
                - `box`
                - `center`
            *   - ``(3,)``
                - ``(3,)``
                - ``(3,)``
            *   - ``(n, 3)``
                - ``(3,)``
                - ``(3,)``
            *   - ``(k, 3)``
                - ``(k, 3)``
                - ``(k, 3)``
            *   - ``(k, n, 3)``
                - ``(3,)``
                - ``(k, 3)``
            *   - ``(k, n, 3)``
                - ``(k, 3)``
                - ``(k, 3)``

        ``n`` particles are lumped into one center for each individual
        frame ``k``.

    See Also
    --------
    :func:`numpy.average` : Compute the weighted average
    :meth:`MDAnalysis.core.groups.AtomGroup.center` :
        Weighted center of (compounds of) the group
    :meth:`MDAnalysis.core.groups.AtomGroup.center_of_geometry` :
        Center of geometry of (compounds of) the group
    :meth:`MDAnalysis.core.groups.AtomGroup.center_of_mass` :
        Center of mass of (compounds of) the group
    :func:`mdtools.structure.center` :
        Different types of centers of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.wcenter` :
        Weighted center of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.coc` :
        Center of charge of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.cog` :
        Center of geometry of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.com` :
        Center of mass of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`

    Notes
    -----
    The weighted center is calculated accroding to

    .. math::

        \mathbf{r}_{center} = \frac{1}{\sum_i^N w_i} \
        \sum_i^N w_i \mathbf{r}_i

    where :math:`r_i` is the position vector of the :math:`i`-th
    particle, :math:`w_i` is its weight and :math:`N` is the number of
    particles.

    Examples
    --------
    Shape of `pos` is ``(3,)``:

    >>> pos = np.array([0, 2, 5])
    >>> mdt.strc.wcenter_pos(pos)
    array([0, 2, 5])
    >>> mdt.strc.wcenter_pos(pos, weights=[3])
    array([0, 2, 5])
    >>> box = np.array([2, 2, 2, 90, 90, 90])
    >>> mdt.strc.wcenter_pos(pos, wrap_pos=True, box=box)
    array([0., 0., 1.], dtype=float32)
    >>> mdt.strc.wcenter_pos(pos, wrap_result=True, box=box)
    array([0., 0., 1.], dtype=float32)
    >>> mdt.strc.wcenter_pos(pos, wrap_result=True)
    Traceback (most recent call last):
    ...
    ValueError: ...

    Shape of `pos` is ``(n, 3)``:

    >>> pos = np.array([[0, 2, 5],
    ...                 [1, 0, 1]])
    >>> mdt.strc.wcenter_pos(pos)
    array([0.5, 1. , 3. ])
    >>> mdt.strc.wcenter_pos(pos, weights=[3, 1])
    array([0.25, 1.5 , 4.  ])
    >>> mdt.strc.wcenter_pos(pos, wrap_pos=True, box=box)
    array([0.5, 0. , 1. ], dtype=float32)
    >>> mdt.strc.wcenter_pos(pos, wrap_result=True, box=box)
    array([0.5, 1. , 1. ], dtype=float32)

    Shape of `pos` is ``(k, 3)``:

    >>> box = np.array([[2, 2, 2, 90, 90, 90],
    ...                 [2, 1, 4, 90, 90, 90]])
    >>> mdt.strc.wcenter_pos(pos, box=box)
    array([[0, 2, 5],
           [1, 0, 1]])
    >>> mdt.strc.wcenter_pos(pos, wrap_pos=True, box=box)
    array([[0., 0., 1.],
           [1., 0., 1.]], dtype=float32)
    >>> mdt.strc.wcenter_pos(pos, wrap_result=True, box=box)
    array([[0., 0., 1.],
           [1., 0., 1.]], dtype=float32)

    Shape of `pos` is ``(k, n, 3)``:

    >>> pos = np.array([[[0, 2, 5],
    ...                  [1, 0, 1]],
    ...
    ...                 [[1, 0, 1],
    ...                  [0, 2, 5]]])
    >>> mdt.strc.wcenter_pos(pos)
    array([[0.5, 1. , 3. ],
           [0.5, 1. , 3. ]])
    >>> mdt.strc.wcenter_pos(pos, weights=[3, 1])
    array([[0.25, 1.5 , 4.  ],
           [0.75, 0.5 , 2.  ]])
    >>> box = np.array([2, 2, 2, 90, 90, 90])
    >>> mdt.strc.wcenter_pos(pos, wrap_pos=True, box=box)
    array([[0.5, 0. , 1. ],
           [0.5, 0. , 1. ]], dtype=float32)
    >>> mdt.strc.wcenter_pos(pos, wrap_result=True, box=box)
    array([[0.5, 1. , 1. ],
           [0.5, 1. , 1. ]], dtype=float32)
    >>> box = np.array([[2, 2, 2, 90, 90, 90],
    ...                 [2, 1, 4, 90, 90, 90]])
    >>> mdt.strc.wcenter_pos(pos, wrap_pos=True, box=box)
    array([[0.5, 0. , 1. ],
           [0.5, 0. , 1. ]], dtype=float32)
    >>> mdt.strc.wcenter_pos(pos, wrap_result=True, box=box)
    array([[0.5, 1. , 1. ],
           [0.5, 0. , 3. ]], dtype=float32)
    """
    pos = mdt.check.pos_array(pos)
    if box is None and (wrap_pos or wrap_result):
        raise ValueError(
            "'box' must be provided if 'wrap_pos' and/or 'wrap_result' is set"
        )

    if wrap_pos:
        pos = mdt.box.wrap_pos(pos=pos, box=box, mda_backend=mda_backend)
    # The dimension of position arrays can be either 1, 2 or 3 (see
    # `mdtools.check.pos_array`).  If `ndim` is
    #     * 1 (single particle), the center is simply the particle
    #       position.
    #     * 2 and ``box.ndim`` is 2 (multiple frames, single particles),
    #       the center is simply the particle position in each frame.
    #     * 2 and ``box.ndim`` is 1 (single frame, multiple particles),
    #       the center is the average over all particle positions
    #       (``axis=0``).
    #     * 3 (multiple frames), the center is the average over all
    #       particle positions in each frame (``axis=1``).
    if pos.ndim == 1:
        center = pos
    elif box is not None and box.ndim == 2 and pos.ndim == 2:
        if pos.shape[0] != box.shape[0]:
            raise ValueError(
                "If 'box' has 2 dimensions, pos.shape[0] ({}) must"
                " match box.shape[0] ({})".format(pos.shape[0], box.shape[0])
            )
        center = pos
    elif pos.ndim in (2, 3):
        center = np.average(pos, axis=pos.ndim - 2, weights=weights)
    else:
        # This else clause should never be entered, because this error
        # should already be raised by `mdt.check.pos_array(pos)`.
        raise ValueError(
            "'pos' has shape {} but must have shape (3,) or (n, 3) or"
            " (k, n, 3).  This should not have happened".format(pos.shape)
        )
    if wrap_result:
        center = mdt.box.wrap_pos(pos=center, box=box, mda_backend=mda_backend)
    return center


def wcenter(
    ag, weights=None, pbc=False, cmp='group', make_whole=False, debug=False
):
    """
    Calculate the weighted center of (compounds of) an MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` for
        which to calculate the weighted center.
    weights : array_like or None, optional
        Weights to be used, given as a 1d array containing the weights
        for each :class:`~MDAnalysis.core.groups.Atom` in `ag`.  Weights
        are mapped to :class:`Atoms <MDAnalysis.core.groups.Atom>` by
        the order they appear in `ag`.  Setting `weights` to ``None`` is
        equivalent to passing identical weights for all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` in `ag` and
        therefore this is equivalent to calculating the center of
        geometry.  If the weights of a compound sum up to zero, the
        coordinates of that compound's weighted center will be ``nan``.
    pbc : bool, optional
        If ``True`` and `cmp` is ``'group'``, move all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` in `ag` to the
        primary unit cell **before** calculating the weighted center.
        If ``True`` and `cmp` is not ``'group'``, the weighted center of
        each compound will be calculated without moving any
        :class:`Atoms <MDAnalysis.core.groups.Atom>` to keep the
        compounds intact (if they were intact before).  Instead, the
        resulting position vectors will be moved to the primary unit
        cell **after** calculating the weighted center.
    cmp : {'group', 'segments', 'residues', 'molecules', 'fragments', \
        'atoms'}, optional
        The compounds of `ag` for which to calculate the weighted
        center.  If ``'group'``, the weighted center of all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` in `ag` will be
        returned as a single position vector.  If ``'atoms'``, simply
        the positions of all atoms in `ag` will be returned (2d array).
        Else, the weighted center of each
        :class:`~MDAnalysis.core.groups.Segment`,
        :class:`~MDAnalysis.core.groups.Residue`, molecule, or
        :attr:`fragment <MDAnalysis.core.groups.AtomGroup.fragments>`
        contained in `ag` will be returned as an array of position
        vectors, i.e. a 2d array.  Refer to the MDAnalysis' user guide
        for an |explanation_of_these_terms|.  Note that in any case,
        also if `cmp` is e.g. ``'residues'``, only the
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to `ag`
        are taken into account, even if the compound might comprise
        additional :class:`Atoms <MDAnalysis.core.groups.Atom>` that are
        not contained in `ag`.
    make_whole : bool, optional
        If ``True``, compounds whose bonds are split across periodic
        boundaries are made whole before calculating their center.  Note
        that all :class:`Atoms <MDAnalysis.core.groups.Atom>` in `ag`
        are wrapped back into the primary unit cell before making
        compounds whole to ensure that the algorithm is working
        properly.  This means that making compounds whole in an
        unwrapped trajectory will lead to a wrapped trajectory with
        whole compounds (some
        :class:`Atoms <MDAnalysis.core.groups.Atom>` may lie outside the
        primary unit cell, but each compound's center will lie inside
        the primary unit cell).  Note that `make_whole` has no effect if
        `cmp` is set to ``'atoms'``.
    debug : bool, optional
        If ``True``, run in debug mode.

    Returns
    -------
    centers : numpy.ndarray
        Position of the weighted center of each compound in `ag`.  If
        `cmp` was set to ``'group'``, the output will be a single
        position vector of shape ``(3,)``.  Else, the output will be a
        2d array of shape ``(n, 3)`` where ``n`` is the number of
        compounds in `ag`.

    Raises
    ------
    RuntimeWarning :
        If `debug` is ``True`` and the weighted center of any compound
        is ``nan``.

    See Also
    --------
    :meth:`MDAnalysis.core.groups.AtomGroup.center` :
        Weighted center of (compounds of) the group
    :meth:`MDAnalysis.core.groups.AtomGroup.center_of_geometry` :
        Center of geometry of (compounds of) the group
    :meth:`MDAnalysis.core.groups.AtomGroup.center_of_mass` :
        Center of mass of (compounds of) the group
    :func:`mdtools.structure.center` :
        Different types of centers of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.wcenter_pos` :
        Calculate the weighted center of a position array
    :func:`mdtools.structure.coc` :
        Center of charge of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.cog` :
        Center of geometry of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.com` :
        Center of mass of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`

    Notes
    -----
    This function uses the
    :meth:`~MDAnalysis.core.groups.AtomGroup.center` method of the input
    :class:`~MDAnalysis.core.groups.AtomGroup` to calculate the weighted
    center.

    .. important::

        If the weights of a compound sum up to zero, the coordinates of
        that compound's weighted center will be ``nan``!  If `debug` is
        set to ``True``, a warning will be raised if any compound's
        weighted center is ``nan``.

    If `make_whole` is ``True``, all
    :class:`Atoms <MDAnalysis.core.groups.Atom>` in `ag` are wrapped
    back into the primary unit cell using :func:`mdtools.box.wrap`
    before calling
    :meth:`~MDAnalysis.core.groups.AtomGroup.center` with the option
    `unwrap` set to ``True``.  This is done to ensure that the unwrap
    algorithm (better called "make whole" algorithm) of
    :meth:`~MDAnalysis.core.groups.AtomGroup.center` is working
    properly.  This means that making compounds whole in an unwrapped
    trajectory will lead to a wrapped trajectory with whole compounds
    (some :class:`Atoms <MDAnalysis.core.groups.Atom>` may lie outside
    the primary unit cell, but each compound's center will lie inside
    the primary unit cell).  `make_whole` has no effect if `cmp` is set
    to ``'atoms'``.

    .. todo::

        Check if it is really necessary to wrap all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` back into the
        primary unit cell before calling
        :meth:`~MDAnalysis.core.groups.AtomGroup.center` with `unwrap`
        set to ``True``.  The currently done back-wrapping is a serious
        problem, because it implies an inplace change of the
        :class:`~MDAnalysis.core.groups.Atom` coordinates.

    """
    if cmp == 'atoms':
        if pbc:
            centers = mdt.box.wrap(
                ag=ag,
                compound=cmp,
                center='cog',  # Does not rely on masses
                inplace=False,
                debug=debug,
            )
        else:
            centers = ag.positions
        debug_info = "Some of your atom positions might be NaN"
    else:
        if make_whole:
            mdt.box.wrap(
                ag=ag,
                compound='atoms',
                center='cog',  # Does not rely on masses
                inplace=True,
                debug=debug,
            )
        centers = ag.center(
            weights=weights, pbc=pbc, compound=cmp, unwrap=make_whole,
        )
        debug_info = "Probably, the weights of this compound sum up to zero"
    if debug and np.any(np.isnan(centers)):
        warnings.warn(
            "At least one compound's weighted center is NaN.  " + debug_info,
            RuntimeWarning
        )
    return centers


def center(
    ag, center='cog', pbc=False, cmp='group', make_whole=False, debug=False
):
    """
    Calculate different types of centers of (compounds of) an MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` for
        which to calculate the center.
    center : {'cog', 'com', 'coc'}, optional
        Which type of center to calculate.

            * ``'cog'``: Center of geometry
            * ``'com'``: Center of mass
            * ``'coc'``: Center of charge

    pbc : bool, optional
        If ``True`` and `cmp` is ``'group'``, move all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` in `ag` to the
        primary unit cell **before** calculating the center.  If
        ``True`` and `cmp` is not ``'group'``, the center of each
        compound will be calculated without moving any
        :class:`Atoms <MDAnalysis.core.groups.Atom>` to keep the
        compounds intact.  Instead, the resulting position vectors will
        be moved to the primary unit cell **after** calculating the
        center.  Note that this option does not make compounds whole
        that are split across periodic boundaries.  To fix broken
        compounds before calculating their center use the option
        `make_whole`.
    cmp : {'group', 'segments', 'residues', 'molecules', 'fragments', \
        'atoms'}, optional
        The compounds of `ag` for which to calculate the center.  If
        ``'group'``, the center of all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` in `ag` will be
        returned as a single position vector.  If ``'atoms'``, simply
        the positions of all atoms in `ag` will be returned (2d array).
        Else, the center of each
        :class:`~MDAnalysis.core.groups.Segment`,
        :class:`~MDAnalysis.core.groups.Residue`, molecule, or
        :attr:`fragment <MDAnalysis.core.groups.AtomGroup.fragments>`
        contained in `ag` will be returned as an array of position
        vectors, i.e. a 2d array.  Refer to the MDAnalysis' user guide
        for an |explanation_of_these_terms|.  Note that in any case,
        also if `cmp` is e.g. ``'residues'``, only the
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to `ag`
        are taken into account, even if the compound might comprise
        additional :class:`Atoms <MDAnalysis.core.groups.Atom>` that are
        not contained in `ag`.
    make_whole : bool, optional
        If ``True``, compounds whose bonds are split across periodic
        boundaries are made whole before calculating their center.  Note
        that all :class:`Atoms <MDAnalysis.core.groups.Atom>` in `ag`
        are wrapped back into the primary unit cell before making
        compounds whole to ensure that the algorithm is working
        properly.  This means that making compounds whole in an
        unwrapped trajectory will lead to a wrapped trajectory with
        whole compounds (some
        :class:`Atoms <MDAnalysis.core.groups.Atom>` may lie outside the
        primary unit cell, but each compound's center will lie inside
        the primary unit cell).  Note that `make_whole` has no effect if
        `cmp` is set to ``'atoms'``.
    debug : bool, optional
        If ``True``, run in debug mode.

    Returns
    -------
    centers : numpy.ndarray
        Position of the center of each compound in `ag`.  If `cmp` was
        set to ``'group'``, the output will be a single position vector
        of shape ``(3,)``.  Else, the output will be a 2d array of shape
        ``(n, 3)`` where ``n`` is the number of compounds in `ag`.

    Raises
    ------
    RuntimeWarning :
        If `debug` is ``True`` and the center of any compound is
        ``nan``.

    See Also
    --------
    :meth:`MDAnalysis.core.groups.AtomGroup.center` :
        Weighted center of (compounds of) the group
    :meth:`MDAnalysis.core.groups.AtomGroup.center_of_geometry` :
        Center of geometry of (compounds of) the group
    :meth:`MDAnalysis.core.groups.AtomGroup.center_of_mass` :
        Center of mass of (compounds of) the group
    :func:`mdtools.structure.wcenter` :
        Weighted center of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.wcenter_pos` :
        Calculate the weighted center of a position array
    :func:`mdtools.structure.coc` :
        Center of charge of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.cog` :
        Center of geometry of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.com` :
        Center of mass of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`

    Notes
    -----
    This function calls either :func:`mdtools.structure.coc`,
    :func:`mdtools.structure.cog` or :func:`mdtools.structure.com`
    depending on the value of `center`.  These functions in turn call
    the corresponding method of the input
    :class:`~MDAnalysis.core.groups.AtomGroup`.

    .. important::

        If the weights of a compound (i.e. the charges in the case of
        ``center='coc'`` or the masses in the case of ``center='com'``)
        sum up to zero, the coordinates of that compound's center will
        be ``nan``!  If `debug` is set to ``True``, a warning will be
        raised if any compound's center is ``nan``.

    If `make_whole` is ``True``, all
    :class:`Atoms <MDAnalysis.core.groups.Atom>` in `ag` are wrapped
    back into the primary unit cell using :func:`mdtools.box.wrap`
    before making compounds whole.  This is done to ensure that the unwrap
    algorithm (better called "make whole" algorithm) of MDAnalysis is
    working properly.  This means that making compounds whole in an
    unwrapped trajectory will lead to a wrapped trajectory with whole
    compounds (some :class:`Atoms <MDAnalysis.core.groups.Atom>` may lie
    outside the primary unit cell, but each compound's center will lie
    inside the primary unit cell).  `make_whole` has no effect if `cmp`
    is set to ``'atoms'``.

    .. todo::

        Check if it is really necessary to wrap all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` back into the
        primary unit cell before making compounds whole.  The currently
        done back-wrapping is a serious problem, because it implies an
        inplace-change of the :class:`~MDAnalysis.core.groups.Atom`
        coordinates.

    .. note:: For developers

        This function is meant to be used in MDTools scripts that offer
        the user to choose a specific compound and center for which to
        perform the analysis (see the \\--cmp and \\--center options of
        the :mod:`scripts.script_template`).  To get the requested
        positions, you can simply do
        ``pos = mdt.strc.center(ag, center=args.CENTER, cmp=args.CMP)``.

    """  # noqa D301
    if center == 'cog':
        return mdt.strc.cog(
            ag=ag, pbc=pbc, cmp=cmp, make_whole=make_whole, debug=debug
        )
    elif center == 'com':
        return mdt.strc.com(
            ag=ag, pbc=pbc, compound=cmp, make_whole=make_whole, debug=debug
        )
    elif center == 'coc':
        return mdt.strc.coc(
            ag=ag, pbc=pbc, cmp=cmp, make_whole=make_whole, debug=debug
        )
    else:
        raise ValueError(
            "'center' must be either 'cog', 'com' or 'coc', but you gave"
            " {}".format(center)
        )


def coc(ag, pbc=False, cmp='group', make_whole=False, debug=False):
    """
    Calculate the center of charge of (compounds of) an MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` for
        which to calculate the center of charge.
    pbc : bool, optional
        See :func:`mdtools.structure.center`.
    cmp : {'group', 'segments', 'residues', 'molecules', 'fragments', \
        'atoms'}, optional
        See :func:`mdtools.structure.center`.
    make_whole : bool, optional
        See :func:`mdtools.structure.center`.
    debug : bool, optional
        If ``True``, run in debug mode.

    Returns
    -------
    center : numpy.ndarray
        Center of charge position of each compound in `ag`.  If `cmp`
        was set to ``'group'``, the output will be a single position
        vector of shape ``(3,)``.  Else, the output will be a 2d array
        of shape ``(n, 3)`` where ``n`` is the number of compounds in
        `ag`.

    Raises
    ------
    RuntimeWarning :
        If `debug` is ``True`` and the center of charge of any compound
        is ``nan``.

    See Also
    --------
    :meth:`MDAnalysis.core.groups.AtomGroup.center` :
        Weighted center of (compounds of) the group
    :meth:`MDAnalysis.core.groups.AtomGroup.center_of_geometry` :
        Center of geometry of (compounds of) the group
    :meth:`MDAnalysis.core.groups.AtomGroup.center_of_mass` :
        Center of mass of (compounds of) the group
    :func:`mdtools.structure.center` :
        Different types of centers of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.wcenter` :
        Weighted center of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.wcenter_pos` :
        Calculate the weighted center of a position array
    :func:`mdtools.structure.cog` :
        Center of geometry of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.com` :
        Center of mass of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`

    Notes
    -----
    This function uses :func:`mdtools.structure.wcenter` which in turn
    relies on the :meth:`~MDAnalysis.core.groups.AtomGroup.center`
    method of the input :class:`~MDAnalysis.core.groups.AtomGroup` to
    calculate the center of charge.

    .. important::

        If the charges of a compound sum up to zero, the coordinates of
        that compound's center of charge will be ``nan``!  If `debug` is
        set to ``True``, a warning will be raised if any compound's
        center of charge is ``nan``.

    """
    return mdt.strc.wcenter(
        ag=ag,
        weights=ag.charges,
        pbc=pbc,
        cmp=cmp,
        make_whole=make_whole,
        debug=debug,
    )


def cog(ag, pbc=False, cmp='group', make_whole=False, debug=False):
    """
    Calculate the center of geometry (a.k.a centroid) of (compounds of)
    an MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` for
        which to calculate the center of geometry.
    pbc : bool, optional
        See :func:`mdtools.structure.center`.
    cmp : {'group', 'segments', 'residues', 'molecules', 'fragments', \
       'atoms'}, optional
        See :func:`mdtools.structure.center`.
    make_whole : bool, optional
        See :func:`mdtools.structure.center`.
    debug : bool, optional
        If ``True``, run in debug mode.

    Returns
    -------
    center : numpy.ndarray
        Center of geometry position of each compound in `ag`.  If `cmp`
        was set to ``'group'``, the output will be a single position
        vector of shape ``(3,)``.  Else, the output will be a 2d array
        of shape ``(n, 3)`` where ``n`` is the number of compounds in
        `ag`.

    Raises
    ------
    RuntimeWarning :
        If `debug` is ``True`` and the center of geometry of any
        compound is ``nan``.

    See Also
    --------
    :meth:`MDAnalysis.core.groups.AtomGroup.center` :
        Weighted center of (compounds of) the group
    :meth:`MDAnalysis.core.groups.AtomGroup.center_of_geometry` :
        Center of geometry of (compounds of) the group
    :meth:`MDAnalysis.core.groups.AtomGroup.center_of_mass` :
        Center of mass of (compounds of) the group
    :func:`mdtools.structure.center` :
        Different types of centers of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.wcenter` :
        Weighted center of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.wcenter_pos` :
        Calculate the weighted center of a position array
    :func:`mdtools.structure.coc` :
        Center of charge of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.com` :
        Center of mass of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`

    Notes
    -----
    This function uses the
    :meth:`~MDAnalysis.core.groups.AtomGroup.center_of_geometry` method
    of the input :class:`~MDAnalysis.core.groups.AtomGroup` to calculate
    the center of geometry.

    .. todo::

        Check if it is really necessary to wrap all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` back into the
        primary unit cell before calling
        :meth:`~MDAnalysis.core.groups.AtomGroup.center_of_geometry`
        with `unwrap` set to ``True``.  The currently done back-wrapping
        is a serious problem, because it implies an inplace change of
        the :class:`~MDAnalysis.core.groups.Atom` coordinates.

    """
    if cmp == 'atoms':
        if pbc:
            centers = mdt.box.wrap(
                ag=ag,
                compound=cmp,
                center='cog',  # Does not rely on masses
                inplace=False,
                debug=debug,
            )
        else:
            centers = ag.positions
    else:
        if make_whole:
            mdt.box.wrap(
                ag=ag,
                compound='atoms',
                center='cog',  # Does not rely on masses
                inplace=True,
                debug=debug,
            )
        centers = ag.center_of_geometry(
            pbc=pbc, compound=cmp, unwrap=make_whole
        )
    if debug and np.any(np.isnan(centers)):
        warnings.warn(
            "At least one compound's center of geometry is NaN.  Some of your"
            " atom positions might be NaN",
            RuntimeWarning
        )
    return centers


def com(ag, pbc=False, compound='group', make_whole=False, debug=False):
    """
    Calculate the center of mass of (compounds of) an MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` for
        which to calculate the center of mass.
    pbc : bool, optional
        See :func:`mdtools.structure.center`.
    compound : {'group', 'segments', 'residues', 'molecules',\
        'fragments', 'atoms'}, optional
        See :func:`mdtools.structure.center`.
    make_whole : bool, optional
        See :func:`mdtools.structure.center`.
    debug : bool, optional
        If ``True``, run in debug mode.

    Returns
    -------
    center : numpy.ndarray
        Center of mass position of each compound in `ag`.  If
        `compound` was set to ``'group'``, the output will be a single
        position vector of shape ``(3,)``.  Else, the output will be a
        2d array of shape ``(n, 3)`` where ``n`` is the number of
        compounds in `ag`.

    Raises
    ------
    RuntimeWarning :
        If `debug` is ``True`` and the center of mass of any compound is
        ``nan``.

    See Also
    --------
    :meth:`MDAnalysis.core.groups.AtomGroup.center` :
        Weighted center of (compounds of) the group
    :meth:`MDAnalysis.core.groups.AtomGroup.center_of_geometry` :
        Center of geometry of (compounds of) the group
    :meth:`MDAnalysis.core.groups.AtomGroup.center_of_mass` :
        Center of mass of (compounds of) the group
    :func:`mdtools.structure.center` :
        Different types of centers of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.wcenter` :
        Weighted center of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.wcenter_pos` :
        Calculate the weighted center of a position array
    :func:`mdtools.structure.coc` :
        Center of charge of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.cog` :
        Center of geometry of (compounds of) an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`

    Notes
    -----
    This function uses the
    :meth:`~MDAnalysis.core.groups.AtomGroup.center_of_mass` method of
    the input :class:`~MDAnalysis.core.groups.AtomGroup` to calculate
    the center of mass.

    .. important::

        `MDAnalysis always guesses atom masses`_!  If MDAnalysis cannot
        guess the mass from the atom type, it will asign this atom a
        zero mass.  If the mass of a compound sum up to zero, the
        coordinates of that compound's center of mass will be ``nan``!
        If `debug` is set to ``True``, a warning will be raised if any
        compound's weighted center is ``nan``.  Also in the case if
        ``debug is false``, a ValueError will be raised, if the mass of
        any atom in `ag` is less than or equal to zero.

    .. todo::

        Check if it is really necessary to wrap all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` back into the
        primary unit cell before calling
        :meth:`~MDAnalysis.core.groups.AtomGroup.center_of_mass` with
        `unwrap` set to ``True``.  The currently done back-wrapping is a
        serious problem, because it implies an inplace change of the
        :class:`~MDAnalysis.core.groups.Atom` coordinates.

    .. _MDAnalysis always guesses atom masses:
        https://userguide.mdanalysis.org/formats/guessing.html
    """
    if compound == 'atoms':
        if pbc:
            centers = mdt.box.wrap(
                ag=ag,
                compound=compound,
                center='cog',  # Does not rely on masses
                inplace=False,
                debug=debug,
            )
        else:
            centers = ag.positions
    else:
        mdt.check.masses_new(ag=ag, verbose=debug)
        if make_whole:
            mdt.box.wrap(
                ag=ag,
                compound='atoms',
                center='cog',  # Does not rely on masses
                inplace=True,
                debug=debug,
            )
        centers = ag.center_of_mass(
            pbc=pbc, compound=compound, unwrap=make_whole
        )
    if debug and np.any(np.isnan(centers)):
        # Masses cannot sum up to zero, because `mdt.check.masses_new`
        # raises a ValueError if the mass of any atom is less than or
        # equal to zero.
        warnings.warn(
            "At least one compound's center of mass is NaN.  Some of your atom"
            " positions might be NaN",
            RuntimeWarning
        )
    return centers


def discrete_pos_trj(
        sel, trj=None, topfile=None, trjfile=None, begin=0, end=-1,
        every=1, compound='atoms', direction='z', bin_start=0,
        bin_stop=None, bin_num=10, bins=None, tol=1e-6,
        return_bins=False, return_lbox=False, return_dt=False, dtype=int,
        verbose=True, debug=False, **sel_kwargs):
    """
    Create a discrete posotion trajectory.

    Discretize the positions of compounds of a MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup` in a given spatial
    direction.

    .. todo::

        Allow to choose between center of mass and center of geometry.

    Parameters
    ----------
    sel : MDAnalysis.core.groups.AtomGroup or str
        'Selection group': Either a MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup` (if `trj` is not
        ``None``) or a selection string for creating a MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup` (if `trj` is
        ``None``).  See MDAnalysis' |selection_syntax| for possible
        choices of selection strings.
    trj : MDAnalysis.coordinates.base.ReaderBase or MDAnalysis.coordinates.base.FrameIteratorBase, optional
        |mda_trj| to read.  If ``None``, a new MDAnalysis
        :class:`~MDAnalysis.core.universe.Universe` and trajectory are
        created from `topfile` and `trjfile`.
    topfile : str, optional
        Topology file.  See |supported_topology_formats| of MDAnalysis.
        Ignored if `trj` is not ``None``.
    trjfile : str
        Trajectory file.  See |supported_coordinate_formats| of
        MDAnalysis.  Ignored if `trj` is not ``None``.
    begin : int, optional
        First frame to read from a newly created trajectory.  Frame
        numbering starts at zero.  Ignored if `trj` is not ``None``.  If
        you want to use only specific frames from an already existing
        trajectory, slice the existing trajectory accordingly and parse
        it as :class:`MDAnalysis.coordinates.base.FrameIteratorSliced`
        object to the `trj` argument.
    end : int, optional
        Last frame to read from a newly created trajectory.  This is
        exclusive, i.e. the last frame read is actually ``end - 1``.  A
        value of ``-1`` means to read the very last frame.  Ignored if
        `trj` is not ``None``.
    every : int, optional
        Read every n-th frame from the newly created trajectory.
        Ignored if `trj` is not ``None``.
    compound : {'atoms', 'group', 'segments', 'residues', 'fragments'}, optional
        The compounds of the selection group whose center of mass
        positions should be discretized.  If ``'group'``, the center of
        mass of all :class:`Atoms <MDAnalysis.core.groups.Atom>` in the
        selection group will be used.  Else, the center of mass
        positions of each :class:`~MDAnalysis.core.groups.Segment`,
        :class:`~MDAnalysis.core.groups.Residue`,
        :attr:`fragment <MDAnalysis.core.groups.AtomGroup.fragments>` or
        :class:`~MDAnalysis.core.groups.Atom` contained in the selection
        group will be discretized.  Refer to the MDAnalysis' user guide
        for an |explanation_of_these_terms|.  Compounds are made whole
        before calculating their centers of mass.  The centers of mass
        are wrapped back into the primary unit cell before discretizing
        their positions.  Note that in any case, even if `compound` is
        e.g. ``'residues'``, only the
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
        selection group are taken into account, even if the compound
        might comprise additional
        :class:`Atoms <MDAnalysis.core.groups.Atom>` that are not
        contained in the selection group.
    direction : {'z', 'y', 'x'}, optional
        The spatial direction along which the discretization is done.
    bin_start : scalar, optional
        Point (in Angstrom) on the chosen spatial direction to start
        binning.  Note that binning naturally starts at zero (origin of
        the simulation box).  If parsing a start value greater than
        zero, the first bin interval will be [0, `bin_start`).  In this
        way you can determine the width of the first bin independently
        from the other bins.  Note that `bin_start` must lie within the
        simulation box obtained from the first frame read and it must be
        smaller than `bin_stop`.
    bin_stop : scalar, optional
        Point (in Angstrom) on the chosen spatial direction to stop
        binning.  Note that binning naturally ends at ``lbox + tol``,
        where ``lbox`` is the length of the simulation box in the given
        spatial direction and ``tol`` is a small tolerance to account
        for the right-open bin interval.  If parsing a value less than
        ``lbox``, the last bin interval will be [`bin_stop`,
        ``lbox + tol``).  In this way you can determine the width of the
        last bin independently from the other bins.  Note that `bin_stop`
        must lie within the simulation box obtained from the first frame
        read and it must be greater than `bin_start`.  If ``None``,
        `bin_stop` is set to ``lbox + tol``.
    bin_num : int, optional
        Number of equidistant bins (not bin edges!) to use for
        discretizing the given spatial direction between `bin_start` and
        `bin_stop`.  Note that two additional bins, [0, `bin_start`) and
        [`bin_stop`, ``lbox + tol``), are created if `bin_start` is not
        zero and `bin_stop` is not ``lbox``.
    bins : array_like, optional
        Array of custom bin edges.  Bins do not need to be equidistant.
        All bin edges must lie within the simulation box as obtained
        from the first frame read.  If `bins` is given, it takes
        precedence over all other bin arguments.
    tol : scalar, optional
        The tolerance value added to ``lbox`` to account for the
        right-open bin interval of the last bin.
    return_bins : bool, optional
        If ``True``, return the bin endges used for the discretization.
    return_lbox : bool, optional
        If ``True``, return the average box length in the given spatial
        direction.
    return_dt : bool, optional
        If ``True`` return the
        :attr:`time step <MDAnalysis.coordinates.base.Timestep.dt>` of
        the created discrete trajectory in ps.  **Attention**: If `trj`
        is not ``None``, the time step is simply taken from the first
        frame of the input trajectory.  If `trj` is ``None`` the
        returned time step is the time step of the first frame of the
        newly created trajectory times `every`.
    dtype : dtype, optional
        Data type of the returned discrete trajectory.
    verbose : bool, optional
        If ``True``, print progress information to standard output.
    debug : bool, optional
        If ``True``, run in :ref:`debug mode <debug-mode-label>`.
    sel_kwargs : dict, optional
        Additional keyword arguments to parse to
        :meth:`MDAnalysis.core.universe.Universe.select_atoms` besides
        the selection string given by `sel`.  If you parse keywords to
        create an :class:`~MDAnalysis.core.groups.UpdatingAtomGroup`,
        the number of compounds in this
        :class:`~MDAnalysis.core.groups.UpdatingAtomGroup` must stay
        constant.  Ignored if `trj` is not ``None``.

    Returns
    -------
    dtrj : numpy.ndarray
        Array of shape ``(n, f)`` containing for each selection compound
        for each frame the index of the position bin in which the
        selection compound currently resides.  ``n`` is the number of
        selection compounds, ``f`` is the number of frames.
    bins : numpy.ndarray
        The bin edges used for the discretization.  Only returned if
        `return_bins` is ``True``.
    lbox_av : scalar
        Average box length in the given spatial direction.  Only
        returned if `return_lbox` is ``True``.
    dt : scalar
        :attr:`Time step <MDAnalysis.coordinates.base.Timestep.dt>` of
        the discrete position trajectory in ps.  Only returned if
        `return_dt` is ``True``.

    See Also
    --------
    :mod:`discrete_pos` :
        Script to create a discrete position trajectory
    :func:`numpy.digitize` :
        Return the indices of the bins to which each value in the input
        array belongs

    Notes
    -----
    The simulation box must be orthogonal.

    Compounds are asigned to bins according to their center of mass
    position.  Compounds are made whole before calculating their centers
    of mass.  The centers of mass are wrapped back into the primary unit
    cell before discretizing their positions.

    The discretization of the compounds' positions is done in relative
    box coordinates.  The final output is scaled by the average box
    length in the given spatial direction.  Doing so accounts for
    possible fluctuations of the simulation box (e.g. due to pressure
    scaling).  Note that MDAnalysis always sets the origin of the
    simulation box to the origin of the cartesian coordinate system.

    All bin intervals are left-closed and right-open, i.e. [a, b) ->
    a <= x < b.  The first bin edge is always zero.  The last bin edge
    is always the (average) box length in the chosen spatial direction
    (i.e. 1 in relative box coordinates) plus a small tolerance to
    account for the right-open bin interval.  Thus, the number of bin
    edges is ``len(bins)``, the number of bins is ``len(bins) - 1`` and
    ``bins[1:] - np.diff(bins) / 2`` yields the bin centers.

    The bin indices in the returned discretized trajectory start at
    zero.  This is different from the output of :func:`numpy.digitize`,
    where the index of the first bin is one.
    """
    if verbose:
        timer_tot = datetime.now()
        proc = psutil.Process()
        proc.cpu_percent()  # Initiate monitoring of CPU usage
        print("Running mdtools.structure.discrete_pos_trj()...")
    if trj is None and (topfile is None or trjfile is None):
        raise ValueError("Either 'trj' or 'topfile' and 'trjfile' must"
                         " be given")
    if direction not in ('x', 'y', 'z'):
        raise ValueError("'direction' must be either 'x', 'y' or 'z',"
                         " but you gave {}".format(direction))
    dim = {'x': 0, 'y': 1, 'z': 2}
    ixd = dim[direction]

    if trj is None:
        if verbose:
            print()
        u = mdt.select.universe(top=topfile,
                                trj=trjfile,
                                verbose=verbose)
        if verbose:
            print()
        sel = mdt.select.atoms(ag=u,
                               sel=sel,
                               verbose=verbose,
                               **sel_kwargs)
        if verbose:
            print()
        N_FRAMES_TOT = u.trajectory.n_frames
        BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
            start=begin,
            stop=end,
            step=every,
            n_frames_tot=u.trajectory.n_frames,
            verbose=verbose
        )
        trj = u.trajectory[BEGIN:END:EVERY]
    else:
        N_FRAMES_TOT = len(trj)
        BEGIN, END, EVERY, N_FRAMES = (0, len(trj), 1, len(trj))
        if isinstance(sel, str):
            raise ValueError("'sel' is a string, but if 'trj' is given,"
                             " 'sel' must be a"
                             " MDAnalysis.core.groups.AtomGroup instance")
    if debug:
        if verbose:
            print()
        try:
            mdt.check.time_step(trj=trj, verbose=verbose)
        except ValueError as error:
            warnings.warn("During checking time step equality, an"
                          " exception was raised: {}".format(error),
                          RuntimeWarning)
    time_step = trj[BEGIN].dt * EVERY

    if verbose:
        print()
        print("Creating/checking bins...")
        timer = datetime.now()
    lbox = trj[0].dimensions[ixd]
    if lbox <= 0:
        raise ValueError("Invalid simulation box: The box length ({}) in"
                         " the given spatial direction ({}) is less than"
                         " or equal to zero".format(lbox, direction))
    if bins is None:
        if bin_stop is None:
            STOP = lbox
        else:
            STOP = bin_stop
        START, STOP, STEP, NUM = mdt.check.bins(start=bin_start / lbox,
                                                stop=STOP / lbox,
                                                num=bin_num,
                                                amin=0,
                                                amax=1,
                                                verbose=verbose)
        bins = np.linspace(START, STOP, NUM + 1)
    else:
        bins = np.unique(bins) / lbox
    mdt.check.bin_edges(bins=bins,
                        amin=0,
                        amax=1,
                        tol=tol,
                        verbose=verbose)
    if verbose:
        print("Elapsed time:         {}".format(datetime.now() - timer))
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss / 2**20))

    # Prepare discrete trajectory:
    if compound == 'group':
        N_CMPS = 1
    elif compound == 'segments':
        N_CMPS = sel.n_segments
    elif compound == 'residues':
        N_CMPS = sel.n_residues
    elif compound == 'fragments':
        N_CMPS = sel.n_fragments
    elif compound == 'atoms':
        N_CMPS = sel.n_atoms
    else:
        raise ValueError("'compound' must be either 'group', 'segments',"
                         " 'residues', 'fragments' or 'atoms', but you"
                         " gave {}".format(compound))
    if compound != 'atoms':
        if verbose:
            print()
        mdt.check.masses_new(ag=sel, verbose=verbose)
    dtrj = np.zeros((N_FRAMES, N_CMPS), dtype=dtype)

    # Read trajectory:
    if verbose:
        print()
        print("Reading trajectory...")
        print("Total number of frames: {:>8d}".format(N_FRAMES_TOT))
        print("Frames to read:         {:>8d}".format(N_FRAMES))
        print("First frame to read:    {:>8d}".format(BEGIN))
        print("Last frame to read:     {:>8d}".format(END - 1))
        print("Read every n-th frame:  {:>8d}".format(EVERY))
        print("Time first frame:       {:>12.3f} (ps)"
              .format(trj[BEGIN].time))
        print("Time last frame:        {:>12.3f} (ps)"
              .format(trj[END - 1].time))
        print("Time step first frame:  {:>12.3f} (ps)"
              .format(trj[BEGIN].dt))
        print("Time step last frame:   {:>12.3f} (ps)"
              .format(trj[END - 1].dt))
        timer = datetime.now()
        trj = mdt.rti.ProgressBar(trj)
    lbox_av = 0  # Average box length in the given direction
    for i, ts in enumerate(trj):
        mdt.check.box(box=ts.dimensions,
                      with_angles=True,
                      orthorhombic=True,
                      dim=1)
        lbox_av += ts.dimensions[ixd]
        if compound == 'atoms':
            pos = mdt.box.wrap(ag=sel, debug=debug)
        else:
            pos = mdt.strc.com(ag=sel,
                               pbc=True,
                               compound=compound,
                               make_whole=True,
                               debug=debug)
        pos = pos[:, ixd]
        pos /= ts.dimensions[ixd]
        if debug:
            mdt.check.array(pos, shape=(N_CMPS,), amin=0, amax=1)
        dtrj[i] = np.digitize(pos, bins=bins)
        if verbose:
            progress_bar_mem = proc.memory_info().rss / 2**20
            trj.set_postfix_str("{:>7.2f}MiB".format(progress_bar_mem),
                                refresh=False)
    del pos
    # Discrete trajectories are returned in a format consistent with
    # PyEMMA, i.e. the first dimension contains the compounds and the
    # second dimension the frames.
    dtrj = np.asarray(dtrj.T, order='C')
    dtrj -= 1  # Compounds in first bin get index 0
    lbox_av /= N_FRAMES
    bins *= lbox_av  # Convert relative box coordinates to real space
    if verbose:
        trj.close()
        print("Elapsed time:         {}".format(datetime.now() - timer))
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss / 2**20))

    # Internal consistency check
    if np.any(dtrj < 0) or np.any(dtrj >= len(bins) - 1):
        raise ValueError("At least one compound position lies outside"
                         " the primary unit cell. This should not have"
                         " happened")

    if verbose:
        print()
        print("mdtools.structure.discrete_pos_trj() done")
        print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
        print("CPU time:             {}"
              .format(timedelta(seconds=sum(proc.cpu_times()[:4]))))
        print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss / 2**20))

    if not np.any([return_bins, return_lbox, return_dt]):
        return dtrj
    else:
        output = np.array([bins, lbox_av, time_step], dtype=object)
        output = output[[return_bins, return_lbox, return_dt]].tolist()
        return tuple([dtrj, ] + output)


def natms_per_cmp(
        ag, compound, return_array=False, return_cmp_ix=False,
        check_contiguos=False):
    """
    Get the number of :class:`Atoms <MDAnalysis.core.groups.Atom>` of
    each compound in an MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` for
        which to get the number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>` per compound.
    compound : {'group', 'segments', 'residues', 'fragments', 'atoms'}
        The compounds of `ag` for which to get the number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  If ``'atoms'``,
        the output will simply be ``1`` or an array of ones (depending
        on the value of `return_array`).  Else, the number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>` in the entire
        group or of each
        :class:`~MDAnalysis.core.groups.Segment`,
        :class:`~MDAnalysis.core.groups.Residue` or
        :attr:`fragment <MDAnalysis.core.groups.AtomGroup.fragments>` in
        `ag` is returned.  Refer to the MDAnalysis' user guide for an
        |explanation_of_these_terms|.  Note that in any case, even if
        `compound` is e.g. ``'residues'``, only the
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to `ag`
        are taken into account, even if the compound might comprise
        additional :class:`Atoms <MDAnalysis.core.groups.Atom>` that are
        not contained in `ag`.
    return_array : bool, optional
        If ``True``, `natms_per_cmp` (the output of this function) will
        always be an array with a length equal to the number of
        compounds in `ag`.  If ``False``, `natms_per_cmp` will be an
        integer if all compounds in `ag` have the same number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  However, if this
        is not the case, `natms_per_cmp` will be an array even if
        `return_array` is ``False``.
    return_cmp_ix : bool, optional
        If ``True``, additionally return the unique indices of the
        compounds as assigned by MDAnalysis.  If `compound` is e.g.
        ``'residues'``, this is ``np.unique(ag.resindices)``.
    check_contiguos : bool, optional
        If ``True`` (default), check if
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
        same compound form a contiguous set in the input
        :class:`~MDAnalysis.core.groups.AtomGroup`.  This is e.g.
        important when constructing compound contact matrices from
        :class:`~MDAnalysis.core.groups.Atom` contact matrices with
        :func:`mdtools.structure.cmp_contact_matrix`.

    Returns
    -------
    natms_per_cmp : int or numpy.ndarray
        Number of :class:`Atoms <MDAnalysis.core.groups.Atom>` of
        each compound contained in `ag`.  If `return_array` is ``False``
        and all compounds in `ag` have the same number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>`, a single integer
        is returned.  If `return_array` is ``True`` or if not all
        compounds in `ag` have the same number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>`, an array is
        returned containing the number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>` of each single
        compound in `ag`.
    cmp_ix : array_like
        Unique compound indices as assigned by MDAnalysis.  Only
        returned if `return_cmp_ix` is ``True``.

    See Also
    --------
    :func:`mdtools.structure.cmp_contact_matrix` :
        Convert an :class:`~MDAnalysis.core.groups.Atom` contact matrix
        to a compound contact matrix
    :func:`mdtools.structure.cmp_contact_count_matrix` :
        Take an :class:`~MDAnalysis.core.groups.Atom` contact matrix and
        sum the contacts of all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
        same compound
    :func:`mdtools.structure.contact_hists` :
        Bin the number of contacts between reference and selection
        compounds into histograms
    """
    if compound == 'atoms':
        if return_array or ag.n_atoms == 0:
            natms_per_cmp = np.ones(ag.n_atoms, dtype=int)
        else:
            natms_per_cmp = 1
        if return_cmp_ix:
            return natms_per_cmp, ag.indices
        else:
            return natms_per_cmp
    elif compound == 'group':
        if return_array:
            natms_per_cmp = np.array([ag.n_atoms], dtype=int)
        else:
            natms_per_cmp = ag.n_atoms
        if return_cmp_ix:
            return natms_per_cmp, np.array([0], dtype=int)
        else:
            return natms_per_cmp
    elif compound == 'segments':
        cmp_ix = ag.segindices
    elif compound == 'residues':
        cmp_ix = ag.resindices
    elif compound == 'fragments':
        cmp_ix = ag.fragindices
    else:
        raise ValueError("compound must be either 'group', 'segments',"
                         " 'residues', 'fragments' or 'atoms', but you"
                         " gave '{}'".format(compound))
    if check_contiguos and not np.array_equal(cmp_ix, np.sort(cmp_ix)):
        raise ValueError("Atoms belonging to the same compound must"
                         " form a contiguous set")
    cmp_ix, natms_per_cmp = np.unique(cmp_ix, return_counts=True)
    if (not return_array and
        len(natms_per_cmp) > 0 and
            np.all(natms_per_cmp == natms_per_cmp[0])):
        natms_per_cmp = natms_per_cmp[0]
    if return_cmp_ix:
        return natms_per_cmp, cmp_ix
    else:
        return natms_per_cmp


def cmp_contact_count_matrix(
        cm, natms_per_refcmp=1, natms_per_selcmp=1, dtype=int):
    """
    Take an :class:`~MDAnalysis.core.groups.Atom` contact matrix and sum
    the contacts of all :class:`Atoms <MDAnalysis.core.groups.Atom>`
    belonging to the same compound.

    A compound is usually a chemically meaningfull subgroup of an
    :class:`~MDAnalysis.core.groups.AtomGroup`.  This can e.g. be a
    :class:`~MDAnalysis.core.groups.Segment`,
    :class:`~MDAnalysis.core.groups.Residue`,
    :attr:`fragment <MDAnalysis.core.groups.AtomGroup.fragments>` or
    a single :class:`~MDAnalysis.core.groups.Atom`.
    Refer to the MDAnalysis' user guide for an
    |explanation_of_these_terms|.  Note that in any case, only
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
    original :class:`~MDAnalysis.core.groups.AtomGroup` are taken into
    account, even if the compound might comprise additional
    :class:`Atoms <MDAnalysis.core.groups.Atom>` that are not contained
    in the original :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    cm : array_like
        (Boolean) contact matrix of shape ``(m, n)`` as e.g. generated
        by :func:`mdtools.structure.contact_matrix`, where ``m`` is the
        number of reference :class:`Atoms <MDAnalysis.core.groups.Atom>`
        and ``n`` is the number of selection
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.
    natms_per_refcmp : int or array_like, optional
        Number of :class:`Atoms <MDAnalysis.core.groups.Atom>` per
        reference compound.  Can be a single integer or an array of
        integers.  If `natms_per_refcmp` is a single integer, all
        reference compounds are assumed to contain the same number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  In this case,
        `natms_per_refcmp` must be an integer divisor of ``cm.shape[0]``.
        If `natms_per_refcmp` is an array of integers, it must contain
        the number of reference
        :class:`Atoms <MDAnalysis.core.groups.Atom>` for each single
        reference compound.  In this case,
        ``numpy.sum(natms_per_refcmp)`` must be equal to ``cm.shape[0]``.
    natms_per_selcmp : int or array_like, optional
        Same for selection compounds (`natms_per_selcmp` is checked
        against ``cm.shape[1]``).
    dtype : dtype, optional
        Data type of the output array.

    Returns
    -------
    cmp_ccm : numpy.ndarray
        A new (smaller) contact **count** matrix indicating how many
        contacts exist between the
        :class:`Atoms <MDAnalysis.core.groups.Atom>` of a given
        reference and selection compound.  If both `natms_per_refcmp`
        and `natms_per_selcmp` are ``1``, the input contact matrix is
        returned instead with its data type converted to `dtype`.

    See Also
    --------
    :func:`mdtools.structure.contact_matrix` :
        Construct a boolean contact matrix for two MDAnalysis
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`
    :func:`mdtools.structure.natms_per_cmp`:
        Get the number of :class:`Atoms <MDAnalysis.core.groups.Atom>`
        of each compound in an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.cmp_contact_matrix` :
        Convert an :class:`~MDAnalysis.core.groups.Atom` contact matrix
        to a compound contact matrix
    :func:`mdtools.structure.contact_hists` :
        Bin the number of contacts between reference and selection
        compounds into histograms.
    :meth:`numpy.ufunc.reduceat` :
        Perform a (local) :meth:`~numpy.ufunc.reduce` with specified
        slices over a single axis

    Notes
    -----
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the same
    compound must form a contiguous set in the input contact matrix,
    otherwise the result will be wrong.

    If both `natms_per_refcmp` and `natms_per_selcmp` are ``1``,
    ``cmp_contact_count_matrix(cm, dtype=dtype)`` is equivalent to
    ``np.asarray(cm, dtype=dtype)``.

    Examples
    --------
    >>> cm = np.array([[ True,  True, False, False],
    ...                [ True, False, False,  True],
    ...                [False, False,  True,  True]])
    >>> mdt.strc.cmp_contact_count_matrix(cm, natms_per_refcmp=3)
    array([[2, 1, 1, 2]])
    >>> mdt.strc.cmp_contact_count_matrix(cm, natms_per_selcmp=2)
    array([[2, 0],
           [1, 1],
           [0, 2]])
    >>> mdt.strc.cmp_contact_count_matrix(cm,
    ...                                   natms_per_refcmp=[1, 2],
    ...                                   natms_per_selcmp=[2, 2])
    array([[2, 0],
           [1, 3]])
    >>> mdt.strc.cmp_contact_count_matrix(cm,
    ...                                   natms_per_refcmp=[1, 2],
    ...                                   natms_per_selcmp=[2, 2],
    ...                                   dtype=np.uint32)
    array([[2, 0],
           [1, 3]], dtype=uint32)

    If both `natms_per_refcmp` and `natms_per_selcmp` are ``1``,
    ``cmp_contact_count_matrix(cm, dtype=dtype)`` is equivalent to
    ``np.asarray(cm, dtype=dtype)``:

    >>> mdt.strc.cmp_contact_count_matrix(cm, dtype=bool) is cm
    True
    >>> mdt.strc.cmp_contact_count_matrix(cm, dtype=int) is cm
    False

    Edge cases:

    >>> cm = np.array([], dtype=bool).reshape(0, 4)
    >>> mdt.strc.cmp_contact_count_matrix(cm, natms_per_refcmp=[])
    array([], shape=(0, 4), dtype=int64)
    >>> mdt.strc.cmp_contact_count_matrix(cm, natms_per_refcmp=1)
    array([], shape=(0, 4), dtype=int64)

    >>> cm = np.array([], dtype=bool).reshape(6, 0)
    >>> mdt.strc.cmp_contact_count_matrix(cm,
    ...                                   natms_per_refcmp=3,
    ...                                   natms_per_selcmp=[],
    ...                                   dtype=np.uint32)
    array([], shape=(2, 0), dtype=uint32)
    """
    cm = np.asarray(cm, dtype=dtype)
    if cm.ndim != 2:
        raise ValueError("'cm' must have 2 dimensions, not {}"
                         .format(cm.ndim))
    natms_per_cmp = [natms_per_refcmp, natms_per_selcmp]
    arg_names = ("natms_per_refcmp", "natms_per_selcmp")
    for i, napc in enumerate(natms_per_cmp):
        if np.all(np.equal(napc, 1)):
            continue
        if np.any(np.less(napc, 1)):
            raise ValueError("(All elements of) '{}' must be greater"
                             " than zero".format(arg_names[i]))
        if np.ndim(napc) == 0:
            if cm.shape[i] % napc != 0:
                raise ValueError(
                    "'{}' ({}) is not an integer divisor of"
                    " 'cm.shape[{}]' ({})"
                    .format(arg_names[i], napc, i, cm.shape[i])
                )
            napc = np.full(cm.shape[i] // napc, napc, dtype=np.uint32)
        elif np.ndim(napc) == 1:
            if np.sum(napc) != cm.shape[i]:
                raise ValueError(
                    "The sum of '{}' ({}) is not equal to 'cm.shape[{}]'"
                    " ({})"
                    .format(arg_names[i], np.sum(napc), i, cm.shape[i])
                )
        else:
            raise ValueError("'{}' must be an integer or a 1d array"
                             .format(arg_names[i]))
        slices = np.cumsum(napc[:-1], dtype=np.uint32)
        slices = np.insert(slices, 0, 0)
        # if i == 0:
        #   Sum all rows containing atoms belonging to the same
        #   compound. => Sum contacts of all atoms belonging to the
        #   same reference compound.
        # elif i == 1:
        #   Sum all columns containing atoms belonging to the same
        #   compound. => Sum contacts of all atoms belonging to the
        #   same selection compound.
        cm = np.add.reduceat(cm, slices, axis=i, dtype=dtype)
    return cm


def cmp_contact_matrix(
        cm, natms_per_refcmp=1, natms_per_selcmp=1, min_contacts=1):
    """
    Convert an :class:`~MDAnalysis.core.groups.Atom` contact matrix to a
    compound contact matrix.

    First sum the contacts of all
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the same
    compound.  Then decide based on the number of contacts between the
    :class:`Atoms <MDAnalysis.core.groups.Atom>` of two given compounds
    whether these two compounds are in contact or not.  Two compounds
    are considered to be in contact if the summed number of contacts
    between their :class:`Atoms <MDAnalysis.core.groups.Atom>` is equal
    to or higher than `min_contacts`.

    A compound is usually a chemically meaningfull subgroup of an
    :class:`~MDAnalysis.core.groups.AtomGroup`.  This can e.g. be a
    :class:`~MDAnalysis.core.groups.Segment`,
    :class:`~MDAnalysis.core.groups.Residue`,
    :attr:`fragment <MDAnalysis.core.groups.AtomGroup.fragments>` or
    a single :class:`~MDAnalysis.core.groups.Atom`.
    Refer to the MDAnalysis' user guide for an
    |explanation_of_these_terms|.  Note that in any case, only
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
    original :class:`~MDAnalysis.core.groups.AtomGroup` are taken into
    account, even if the compound might comprise additional
    :class:`Atoms <MDAnalysis.core.groups.Atom>` that are not contained
    in the original :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    cm : array_like
        (Boolean) contact matrix of shape ``(m, n)`` as e.g. generated
        by :func:`mdtools.structure.contact_matrix`, where ``m`` is the number of
        reference :class:`Atoms <MDAnalysis.core.groups.Atom>` and ``n``
        is the number of selection
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.
    natms_per_refcmp : int or array_like, optional
        Number of :class:`Atoms <MDAnalysis.core.groups.Atom>` per
        reference compound.  Can be a single integer or an array of
        integers.  If `natms_per_refcmp` is a single integer, all
        reference compounds are assumed to contain the same number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  In this case,
        `natms_per_refcmp` must be an integer divisor of ``cm.shape[0]``.
        If `natms_per_refcmp` is an array of integers, it must contain
        the number of reference
        :class:`Atoms <MDAnalysis.core.groups.Atom>` for each single
        reference compound.  In this case,
        ``numpy.sum(natms_per_refcmp)`` must be equal to ``cm.shape[0]``.
    natms_per_selcmp : int or array_like, optional
        Same for selection compounds (`natms_per_selcmp` is checked
        against ``cm.shape[1]``).
    min_contacts : int, optional
        Two compounds are considered to be in contact if the summed
        number of contacts between their
        :class:`Atoms <MDAnalysis.core.groups.Atom>` is equal to or
        higher than `min_contacts`.  Must be greater than zero.
        `min_contacts` is ignored if both `natms_per_refcmp` and
        `natms_per_selcmp` are ``1``.

    Returns
    -------
    cmp_cm : numpy.ndarray
        A new (smaller) boolean contact matrix, whose elements evaluate
        to ``True`` if a reference and selection compound have at least
        `min_contacts` contacts.  If both `natms_per_refcmp` and
        `natms_per_selcmp` are ``1``, the input contact matrix is
        returned instead with its data type converted to bool (if it was
        not already bool before).

    See Also
    --------
    :func:`mdtools.structure.contact_matrix` :
        Construct a boolean contact matrix for two MDAnalysis
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`
    :func:`mdtools.structure.natms_per_cmp`:
        Get the number of :class:`Atoms <MDAnalysis.core.groups.Atom>`
        of each compound in an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.cmp_contact_count_matrix` :
        Take an :class:`~MDAnalysis.core.groups.Atom` contact matrix and
        sum the contacts of all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
        same compound

    Notes
    -----
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the same
    compound must form a contiguous set in the input contact matrix,
    otherwise the result will be wrong.

    If both `natms_per_refcmp` and `natms_per_selcmp` are ``1``,
    ``cmp_contact_matrix(cm)`` is equivalent to
    ``np.asarray(cm, dtype=bool)``.

    Examples
    --------
    >>> cm = np.array([[ True,  True, False, False],
    ...                [ True, False, False,  True],
    ...                [False, False,  True,  True]])
    >>> mdt.strc.cmp_contact_matrix(cm, natms_per_refcmp=3)
    array([[ True,  True,  True,  True]])
    >>> mdt.strc.cmp_contact_matrix(cm, natms_per_selcmp=2)
    array([[ True, False],
           [ True,  True],
           [False,  True]])
    >>> mdt.strc.cmp_contact_matrix(cm,
    ...                             natms_per_selcmp=2,
    ...                             min_contacts=2)
    array([[ True, False],
           [False, False],
           [False,  True]])
    >>> mdt.strc.cmp_contact_matrix(cm,
    ...                             natms_per_refcmp=[1, 2],
    ...                             natms_per_selcmp=2)
    array([[ True, False],
           [ True,  True]])

    If both `natms_per_refcmp` and `natms_per_selcmp` are ``1``,
    ``cmp_contact_matrix(cm)`` is equivalent to
    ``np.asarray(cm, dtype=bool)``.

    >>> mdt.strc.cmp_contact_matrix(cm) is cm
    True
    >>> mdt.strc.cmp_contact_matrix(cm, min_contacts=2) is cm
    True

    Edge cases:

    >>> cm = np.array([], dtype=bool).reshape(0, 4)
    >>> mdt.strc.cmp_contact_matrix(cm, natms_per_refcmp=[])
    array([], shape=(0, 4), dtype=bool)
    >>> mdt.strc.cmp_contact_matrix(cm, natms_per_refcmp=1)
    array([], shape=(0, 4), dtype=bool)

    >>> cm = np.array([], dtype=bool).reshape(6, 0)
    >>> mdt.strc.cmp_contact_matrix(cm,
    ...                             natms_per_refcmp=3,
    ...                             natms_per_selcmp=[],
    ...                             min_contacts=2)
    array([], shape=(2, 0), dtype=bool)
    """
    cm = np.asarray(cm, dtype=bool)
    if (np.any(np.not_equal(natms_per_refcmp, 1)) or
            np.any(np.not_equal(natms_per_selcmp, 1))):
        if (np.size(natms_per_refcmp) > 0 and
                np.size(natms_per_selcmp) > 0):
            # np.max([]) raises an exception
            max_contacts = np.prod([np.max(natms_per_refcmp),
                                    np.max(natms_per_selcmp)])
        elif np.size(natms_per_refcmp) > 0:
            max_contacts = np.max(natms_per_refcmp)
        elif np.size(natms_per_selcmp) > 0:
            max_contacts = np.max(natms_per_selcmp)
        else:
            max_contacts = min_contacts
        if min_contacts > max_contacts:
            raise ValueError("min_contacts ({}) is greater than the"
                             " maximally possible number of contacts"
                             " ({})".format(min_contacts, max_contacts))
        elif min_contacts < 1:
            raise ValueError("'min_contacts' ({}) must be greater than"
                             " zero.".format(min_contacts))
        cm = cmp_contact_count_matrix(cm=cm,
                                      natms_per_refcmp=natms_per_refcmp,
                                      natms_per_selcmp=natms_per_selcmp,
                                      dtype=np.uint32)
        cm = (cm >= min_contacts)
    return cm


def cm_fill_missing_cmp_ix(cm, refix=None, selix=None):
    """
    Insert elements in a contact matrix at missing *intermediate*
    compound indices.

    The inserted matrix elements will evaluate to ``False``.

    Parameters
    ----------
    cm : array_like
        (Boolean) contact matrix of shape ``(m, n)`` as e.g. generated
        by :func:`mdtools.structure.contact_matrix`, where ``m`` is the
        number of reference :class:`Atoms <MDAnalysis.core.groups.Atom>`
        or compounds and ``n`` is the number of selection
        :class:`Atoms <MDAnalysis.core.groups.Atom>` or compounds.
    refix, selix : array_like of ints
        Array of compound indices corresponding to the
        reference/selection compounds contained in `cm`.  If the indices
        contain gaps, these gaps will be filled by this function.
        Indices must not be negative, dublicate indices will be removed.

    Returns
    -------
    cm : numpy.ndarray
        The input contact matrix with added rows for missing reference
        compound indices in `refix` and added columns for missing
        selection compound indices in `selix`.  If both `refix` and
        `selix` do not have any gaps (i.e. the indices are contiguous),
        the input matrix is returned instead as :class:`numpy.ndarray`.

    See Also
    --------
    :func:`mdtools.structure.contact_matrix` :
        Construct a boolean contact matrix for two MDAnalysis
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`
    :func:`mdtools.structure.cmp_contact_matrix` :
        Convert an :class:`~MDAnalysis.core.groups.Atom` contact matrix
        to a compound contact matrix

    Examples
    --------
    >>> cm = np.array([[ True,  True, False],
    ...                [False,  True,  True]])
    >>> mdt.strc.cm_fill_missing_cmp_ix(cm, refix=[2, 4])  # 3 missing
    array([[ True,  True, False],
           [False, False, False],
           [False,  True,  True]])
    >>> mdt.strc.cm_fill_missing_cmp_ix(cm, selix=[0, 1, 4])  # 2, 3 missing
    array([[ True,  True, False, False, False],
           [False,  True, False, False,  True]])

    If both `refix` and `selix` do not have any gaps, the input contact
    matrix is returned as :class:`numpy.ndarray`.  If it already was a
    :class:`numpy.ndarray` before, no copy is made.

    >>> result = mdt.strc.cm_fill_missing_cmp_ix(cm,
    ...                                          refix=[2, 3],
    ...                                          selix=[1, 2, 3])
    >>> result is cm
    True
    """
    cm = np.asarray(cm)
    if cm.ndim != 2:
        raise ValueError("'cm' must have 2 dimensions, not {}"
                         .format(cm.ndim))
    agix = (refix, selix)
    for i, ix in enumerate(agix):
        if ix is None:
            continue
        ix = np.unique(ix)  # Unique and sorted indices
        if ix[0] < 0:
            raise ValueError("Indices must not be negative")
        ix -= ix[0]
        if ix[-1] + 1 != len(ix):
            shape = list(cm.shape)
            shape.pop(i)  # Remove current axis from shape => Number of compounds
            insertion = np.zeros(shape, dtype=cm.dtype)
            missing_ix = np.setdiff1d(np.arange(ix[-1] + 1),
                                      ix,
                                      assume_unique=True)
            for m_ix in missing_ix:
                cm = np.insert(cm, m_ix, insertion, axis=i)
        # Internal consistency check
        if cm.shape[i] != ix[-1] + 1:
            raise ValueError("'cm.shape[{}]' is {} but must be {}. This"
                             " should not have happened"
                             .format(i, cm.shape[i], ix[-1] + 1))
    return cm


def contact_matrix(
        ref, sel, cutoff, compound='atoms', min_contacts=1,
        fill_missing_cmp_ix=False, box=None, result=None,
        mdabackend='serial', debug=False):
    """
    Construct a contact matrix for two MDAnalysis
    :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`.

    Construct a boolean matrix whose elements indicate whether a contact
    exists between a given reference and selection compound or not.  The
    matrix element ``cm[i][j]`` will be ``True``, if compound ``j`` of
    the selection :class:`~MDAnalysis.core.groups.AtomGroup` has at
    least `min_contacts` contacts to compound ``i`` of the reference
    :class:`~MDAnalysis.core.groups.AtomGroup`.  A "contact" is defined
    as two :class:`Atoms <MDAnalysis.core.groups.Atom>` from different
    :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>` whose
    distance is less than or equal to a given cutoff distance.

    Parameters
    ----------
    ref, sel : MDAnalysis.core.groups.AtomGroup
        The reference/selection
        :class:`~MDAnalysis.core.groups.AtomGroup`.  Must be unique,
        i.e. must not contain dublicate
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.
    cutoff : scalar
        A reference and selection :class:`~MDAnalysis.core.groups.Atom`
        are considered to be in contact, if their distance is less than
        or equal to this cutoff.  Must be greater than zero.
    compound : {'atoms', 'group', 'segments', 'residues', 'fragments'}, optional
        The compounds of `ref` and `sel` for which to calculate the
        contact matrix.  Must be either a single string which is applied
        to both, `ref` and `sel`, or a 1-dimensional array of two
        strings, the first for `ref` and and the second for `sel`.  If
        `compound` is ``'atoms'``, the resulting contact matrix
        represents contacts between the individual
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  Else, the contact
        matrix represents contacts between
        entire :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`,
        :class:`Segments <MDAnalysis.core.groups.Segment>`,
        :class:`Residues <MDAnalysis.core.groups.Residue>`,
        or :attr:`~MDAnalysis.core.groups.AtomGroup.fragments`.  Refer
        to the MDAnalysis' user guide for an
        |explanation_of_these_terms|.  Note that in any case, even if
        e.g. `compound` is ``'residues'``, only the
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to `ref`
        and `sel` are taken into account, even if the compound might
        comprise additional :class:`Atoms <MDAnalysis.core.groups.Atom>`
        that are not contained in these
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`.
    min_contacts : int, optional
        Two compounds are considered to be in contact if the summed
        number of contacts between their
        :class:`Atoms <MDAnalysis.core.groups.Atom>` is equal to or
        higher than `min_contacts`.  Must be greater than zero.
        `min_contacts` is ignored if `compound` is ``'atoms'``, because
        :class:`Atoms <MDAnalysis.core.groups.Atom>` can only have one
        or no contact with another :class:`~MDAnalysis.core.groups.Atom`.
    fill_missing_cmp_ix : bool, optional
        If ``True``, also create matrix elements for missing
        *intermediate* compound indices.  These matrix elements will
        evaluate to ``False``.  Only relevant, if the compounds of the
        reference and/or selection group do not form a contiguous set of
        indices.  See :func:`mdtools.structure.cm_fill_missing_cmp_ix`
        for more details.
    box : array_like, option
        The unit cell dimensions of the system, which can be orthogonal
        or triclinic and must be provided in the same format as returned
        by :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.  If provided, the minimum
        image convention will be taken into account.
    result : numpy.ndarray, optional
        Preallocated result array for
        :func:`MDAnalysis.lib.distances.distance_array` which must have
        the shape ``(ref.n_atoms, sel.n_atoms)`` and dtype
        ``numpy.float64``.  Avoids creating the array which saves time
        when the function is called repeatedly.
    mdabackend : {'serial', 'OpenMP'}, optional
        Keyword selecting the type of acceleration for
        :func:`MDAnalysis.lib.distances.distance_array`.  See there for
        further information.
    debug : bool, optional
        If ``True``, run in :ref:`debug mode <debug-mode-label>`.

        .. deprecated:: 0.0.0.dev0
            This argument is without use and will be removed in a future
            release.

    Returns
    -------
    cm : numpy.ndarray
        Contact matrix as :class:`numpy.ndarray` of dtype ``bool``.
        Matrix elements evaluating to ``True`` indicate a contact
        between the respective compounds of the reference and selection
        :class:`~MDAnalysis.core.groups.AtomGroup`.

    See Also
    --------
    :func:`mdtools.structure.contact_matrices` :
        Construct a contact matrix for each frame in a trajectory
    :func:`mdtools.structure.natms_per_cmp`:
        Get the number of :class:`Atoms <MDAnalysis.core.groups.Atom>`
        of each compound in an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.cmp_contact_matrix` :
        Convert an :class:`~MDAnalysis.core.groups.Atom` contact matrix
        to a compound contact matrix
    :func:`mdtools.structure.cmp_contact_count_matrix` :
        Take an :class:`~MDAnalysis.core.groups.Atom` contact matrix and
        sum the contacts of all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
        same compound
    :func:`mdtools.structure.cm_fill_missing_cmp_ix` :
        Insert elements in a contact matrix at missing *intermediate*
        compound indices
    :func:`mdtools.structure.contact_hists` :
        Bin the number of contacts between reference and selection
        compounds into histograms
    :func:`MDAnalysis.lib.distances.distance_array` :
        Calculate all possible distances between a reference set of
        coordinates and another configuration
    :mod:`scipy.sparse` :
        SciPy 2-D sparse matrix package

    Notes
    -----
    When holding multiply contact matrices in memory at once, you might
    want to convert them to :mod:`SciPy sparse matrices <scipy.sparse>`
    to save memory.
    """
    ags = (ref, sel)
    ag_names = ("reference", "selection")
    n_ags = len(ags)
    if cutoff <= 0:
        raise ValueError("'cutoff' ({}) must be greater than zero."
                         .format(cutoff))
    for i, ag in enumerate(ags):
        if ag.unique != ag:
            raise ValueError("The {} group must not contain dublicate"
                             " atoms".format(ag_names[i]))

    for i, ag in enumerate(ags):
        if ag.n_atoms == 0:
            cm = np.array([], dtype=bool)
            return cm.reshape([a.n_atoms for a in ags])
    dists = mdadist.distance_array(reference=ref.positions,
                                   configuration=sel.positions,
                                   box=box,
                                   result=result,
                                   backend=mdabackend)
    cm = (dists <= cutoff)
    if np.ndim(compound) == 0:
        compound = (compound,) * n_ags
    elif np.ndim(compound) == 1 and len(compound) == 1:
        compound = (compound[0],) * n_ags
    elif np.ndim(compound) > 1:
        raise ValueError("'compound' must either be a string or a 1d"
                         " array containing {} strings".format(n_ags))
    if len(compound) != n_ags:
        raise ValueError("'compound' must either be a string or a 1d"
                         " array containing {} strings".format(n_ags))
    cmp_ix = [None, ] * n_ags
    napc = [None, ] * n_ags
    for i, ag in enumerate(ags):
        napc[i], cmp_ix[i] = natms_per_cmp(ag=ag,
                                           compound=compound[i],
                                           return_cmp_ix=True,
                                           check_contiguos=True)
    if np.any([cmp != 'atoms' for cmp in compound]):
        cm = cmp_contact_matrix(cm=cm,
                                natms_per_refcmp=napc[0],
                                natms_per_selcmp=napc[1],
                                min_contacts=min_contacts)
    # Internal consistency check
    if cm.shape != tuple(len(cix) for cix in cmp_ix):
        raise ValueError("cm has shape {}, but must have shape {}. This"
                         " should not have happened"
                         .format(cm.shape,
                                 tuple(len(cix) for cix in cmp_ix)))
    if fill_missing_cmp_ix:
        cm = cm_fill_missing_cmp_ix(cm=cm,
                                    refix=cmp_ix[0],
                                    selix=cmp_ix[1])

    return cm


def contact_matrices(
        ref, sel, cutoff, trj=None, topfile=None, trjfile=None, begin=0,
        end=-1, every=1, updating_ref=False, updating_sel=False,
        compound='atoms', min_contacts=1, fill_missing_cmp_ix=False,
        mdabackend='serial', verbose=True):
    """
    Construct a contact matrix for two MDAnalysis
    :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>` for each
    frame in a trajectory.

    Parameters
    ----------
    ref, sel : MDAnalysis.core.groups.AtomGroup or str
        'Reference/Selection group': Either a MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup` (if `trj` is not
        ``None``) or a selection strings for creating a MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup` (if `trj` is
        ``None``).  See MDAnalysis' |selection_syntax| for possible
        choices of selection strings.
    cutoff : scalar
        A reference and selection :class:`~MDAnalysis.core.groups.Atom`
        are considered to be in contact, if their distance is less than
        or equal to this cutoff.  Must be greater than zero.
    trj : MDAnalysis.coordinates.base.ReaderBase or MDAnalysis.coordinates.base.FrameIteratorBase, optional
        |mda_trj| to read.  If ``None``, a new MDAnalysis
        :class:`~MDAnalysis.core.universe.Universe` and trajectory are
        created from `topfile` and `trjfile`.
    topfile : str, optional
        Topology file.  See |supported_topology_formats| of MDAnalysis.
        Ignored if `trj` is not ``None``.
    trjfile : str, optional
        Trajectory file.  See |supported_coordinate_formats| of
        MDAnalysis.  Ignored if `trj` is not ``None``.
    begin : int, optional
        First frame to read from a newly created trajectory.  Frame
        numbering starts at zero.  Ignored if `trj` is not ``None``.  If
        you want to use only specific frames from an already existing
        trajectory, slice the existing trajectory accordingly and parse
        it as :class:`MDAnalysis.coordinates.base.FrameIteratorSliced`
        object to the `trj` argument.
    end : int, optional
        Last frame to read from a newly created trajectory.  This is
        exclusive, i.e. the last frame read is actually ``end - 1``.  A
        value of ``-1`` means to read the very last frame.  Ignored if
        `trj` is not ``None``.
    every : int, optional
        Read every n-th frame from the newly created trajectory.
        Ignored if `trj` is not ``None``.
    updating_ref, updating_sel : bool, optional
        Use and :class:`~MDAnalysis.core.groups.UpdatingAtomGroup` for
        a newly created reference/selection group.  Selection
        expressions of
        :class:`UpdatingAtomGroups <MDAnalysis.core.groups.UpdatingAtomGroup>`
        are re-evaluated every time step.  E.g. useful for
        position-based selections like 'type Li and prop z <= 2.0'.
        Note that the contact matrices for different frames might have
        different shapes when using
        :class:`UpdatingAtomGroups <MDAnalysis.core.groups.UpdatingAtomGroup>`.
        Ignored if `trj` is not ``None``.
    compound : {'atoms', 'group', 'segments', 'residues', 'fragments'}, optional
        The compounds of `ref` and `sel` for which to calculate the
        contact matrix.  Must be either a single string which is applied
        to both, `ref` and `sel`, or a 1-dimensional array of two
        strings, the first for `ref` and and the second for `sel`.  If
        `compound` is ``'atoms'``, the resulting contact matrix
        represents contacts between the individual
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  Else, the contact
        matrix represents contacts between
        entire :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`,
        :class:`Segments <MDAnalysis.core.groups.Segment>`,
        :class:`Residues <MDAnalysis.core.groups.Residue>`,
        or :attr:`~MDAnalysis.core.groups.AtomGroup.fragments`.  Refer
        to the MDAnalysis' user guide for an
        |explanation_of_these_terms|.  Note that in any case, even if
        e.g. `compound` is ``'residues'``, only the
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to `ref`
        and `sel` are taken into account, even if the compound might
        comprise additional :class:`Atoms <MDAnalysis.core.groups.Atom>`
        that are not contained in these
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`.
    min_contacts : int, optional
        Two compounds are considered to be in contact if the summed
        number of contacts between their
        :class:`Atoms <MDAnalysis.core.groups.Atom>` is equal to or
        higher than `min_contacts`.  Must be greater than zero.
        `min_contacts` is ignored if `compound` is ``'atoms'``, because
        :class:`Atoms <MDAnalysis.core.groups.Atom>` can only have one
        or no contact with another :class:`~MDAnalysis.core.groups.Atom`.
    fill_missing_cmp_ix : bool, optional
        If ``True``, also create matrix elements for missing
        *intermediate* compound indices.  These matrix elements will
        evaluate to ``False``.  Only relevant, if the compounds of the
        reference and/or selection group do not form a contiguous set of
        indices.  See :func:`mdtools.structure.cm_fill_missing_cmp_ix`
        for more details.
    mdabackend : {'serial', 'OpenMP'}, optional
        Keyword selecting the type of acceleration for
        :func:`MDAnalysis.lib.distances.distance_array`.  See there for
        further information.
    verbose : bool, optional
        If ``True``, print progress information to standard output.

    Returns
    -------
    cms : list
        List of contact matrices, one for each frame in the trajectory.
        The contact matrices are stored as
        :class:`scipy.sparse.csr_matrix`.  Each contact matrix has as
        many rows as reference compounds and as many columns as
        selection compounds.  See
        :func:`mdtools.structure.contact_matrix` for more details.

    See Also
    --------
    :func:`mdtools.structure.contact_matrix` :
        Construct a boolean contact matrix for two MDAnalysis
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`
    :mod:`scipy.sparse` :
        SciPy 2-D sparse matrix package
    """
    if verbose:
        timer_tot = datetime.now()
        proc = psutil.Process()
        proc.cpu_percent()  # Initiate monitoring of CPU usage
        print("Running mdtools.structure.contact_matrices()...")
    if trj is None and (topfile is None or trjfile is None):
        raise ValueError("Either 'trj' or 'topfile' and 'trjfile' must"
                         " be given")
    if np.ndim(compound) == 0:
        compound = (compound,) * 2
    elif np.ndim(compound) == 1 and len(compound) == 1:
        compound = (compound[0],) * 2
    elif np.ndim(compound) > 1:
        raise ValueError("'compound' must either be a string or a 1d"
                         " array containing 2 strings")
    if len(compound) != 2:
        raise ValueError("'compound' must either be a string or a 1d"
                         " array containing 2 strings")

    if trj is None:
        if verbose:
            print()
        u = mdt.select.universe(top=topfile,
                                trj=trjfile,
                                verbose=verbose)
        if verbose:
            print()
            print("Creating selections...")
            timer = datetime.now()
            ref_str = ref
            sel_str = sel
        ref = u.select_atoms(ref, updating=updating_ref)
        sel = u.select_atoms(sel, updating=updating_sel)
        if ref.n_atoms == 0 and not updating_ref:
            raise ValueError("The reference group contains no atoms")
        if sel.n_atoms == 0 and not updating_sel:
            raise ValueError("The selection group contains no atoms")
        if verbose:
            print("Reference group: '{}'".format(ref_str))
            print(mdt.rti.ag_info_str(ag=ref, indent=2))
            print("Selection group: '{}'".format(sel_str))
            print(mdt.rti.ag_info_str(ag=sel, indent=2))
            print("Elapsed time:         {}".format(datetime.now() - timer))
            print("Current memory usage: {:.2f} MiB"
                  .format(proc.memory_info().rss / 2**20))
            print()
        N_FRAMES_TOT = u.trajectory.n_frames
        BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
            start=begin,
            stop=end,
            step=every,
            n_frames_tot=u.trajectory.n_frames,
            verbose=verbose
        )
        trj = u.trajectory[BEGIN:END:EVERY]
    else:
        N_FRAMES_TOT = len(trj)
        BEGIN, END, EVERY, N_FRAMES = (0, len(trj), 1, len(trj))
        if isinstance(sel, str):
            raise ValueError("'sel' is a string, but if 'trj' is given,"
                             " 'sel' must be a"
                             " MDAnalysis.core.groups.AtomGroup instance")

    cms = [None, ] * N_FRAMES
    if not updating_ref:
        natms_per_refcmp, refcmp_ix = mdt.strc.natms_per_cmp(
            ag=ref,
            compound=compound[0],
            return_cmp_ix=True,
            check_contiguos=True
        )
    if not updating_sel:
        natms_per_selcmp, selcmp_ix = mdt.strc.natms_per_cmp(
            ag=sel,
            compound=compound[1],
            return_cmp_ix=True,
            check_contiguos=True
        )
    if not updating_ref and not updating_sel:
        dist_array_tmp = np.full((ref.n_atoms, sel.n_atoms),
                                 np.nan,
                                 dtype=np.float64)
    else:
        dist_array_tmp = None

    if verbose:
        print()
        print("Reading trajectory...")
        print("Total number of frames: {:>8d}".format(N_FRAMES_TOT))
        print("Frames to read:         {:>8d}".format(N_FRAMES))
        print("First frame to read:    {:>8d}".format(BEGIN))
        print("Last frame to read:     {:>8d}".format(END - 1))
        print("Read every n-th frame:  {:>8d}".format(EVERY))
        print("Time first frame:       {:>12.3f} (ps)"
              .format(trj[BEGIN].time))
        print("Time last frame:        {:>12.3f} (ps)"
              .format(trj[END - 1].time))
        print("Time step first frame:  {:>12.3f} (ps)"
              .format(trj[BEGIN].dt))
        print("Time step last frame:   {:>12.3f} (ps)"
              .format(trj[END - 1].dt))
        timer = datetime.now()
        trj = mdt.rti.ProgressBar(trj)
    for i, ts in enumerate(trj):
        cm = contact_matrix(ref=ref,
                            sel=sel,
                            cutoff=cutoff,
                            box=ts.dimensions,
                            result=dist_array_tmp,
                            mdabackend=mdabackend)
        if updating_ref and compound[0] != 'atoms':
            natms_per_refcmp, refcmp_ix = natms_per_cmp(
                ag=ref,
                compound=compound[0],
                return_cmp_ix=True,
                check_contiguos=True
            )
        if updating_sel and compound[1] != 'atoms':
            natms_per_selcmp, selcmp_ix = natms_per_cmp(
                ag=sel,
                compound=compound[1],
                return_cmp_ix=True,
                check_contiguos=True
            )
        if compound[0] != 'atoms' or compound[1] != 'atoms':
            cm = cmp_contact_matrix(cm=cm,
                                    natms_per_refcmp=natms_per_refcmp,
                                    natms_per_selcmp=natms_per_selcmp,
                                    min_contacts=min_contacts)
        if fill_missing_cmp_ix:
            cm = cm_fill_missing_cmp_ix(cm=cm,
                                        refix=refcmp_ix,
                                        selix=selcmp_ix)
        cms[i] = sparse.csr_matrix(cm, dtype=bool)
        if verbose:
            progress_bar_mem = proc.memory_info().rss / 2**20
            trj.set_postfix_str("{:>7.2f}MiB".format(progress_bar_mem),
                                refresh=False)
    if verbose:
        trj.close()
        print("Elapsed time:         {}".format(datetime.now() - timer))
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss / 2**20))
        print()
        print("mdtools.structure.contact_matrices() done")
        print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
        print("CPU time:             {}"
              .format(timedelta(seconds=sum(proc.cpu_times()[:4]))))
        print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss / 2**20))

    return cms


def cms_n_common_contacts(cms):
    """
    Get the number of contancts common in all contact matrices.

    Parameters
    ----------
    cms : tuple or list
        List of contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrices`.  The contact
        matrices can be given either as
        :class:`NumPy arrays <numpy.ndarray>` or as
        :mod:`SciPy sparse matrices <scipy.sparse>`.  A mix of
        :class:`NumPy arrays <numpy.ndarray>` and
        :mod:`SciPy sparse matrices <scipy.sparse>` is not possible.

    Returns
    -------
    n_bound_in_all_cms : int
        Number of non-zero elements of the product of all arrays in
        `cms` (i.e. the number of non-zero elements that are common in
        all arrays in `cms`).

    See Also
    --------
    :func:`mdtools.structure.contact_matrix` :
        Construct a boolean contact matrix for two MDAnalysis
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`
    :func:`mdtools.structure.contact_matrices` :
        Construct a contact matrix for each frame in a trajectory
    :func:`mdtools.structure.cms_n_contacts` :
        Get the number of contancts per contact matrix and the number of
        contacts common in all contact matrices

    Examples
    --------
    :class:`NumPy arrays <numpy.ndarray>` as input:

    >>> cm0 = np.array([[ True,  True, False],
    ...                 [False,  True,  True]])
    >>> cm1 = np.array([[ True, False, False],
    ...                 [False, False,  True]])
    >>> cm2 = np.array([[ True,  True, False],
    ...                 [False, False,  True]])
    >>> mdt.strc.cms_n_common_contacts([cm0,])
    4
    >>> cms = [cm0, cm1, cm2]
    >>> mdt.strc.cms_n_common_contacts(cms)
    2

    :mod:`SciPy sparse matrices <scipy.sparse>` as input:

    >>> from scipy import sparse
    >>> cms = [sparse.csr_matrix(cm) for cm in cms]
    >>> mdt.strc.cms_n_common_contacts([cms[0],])
    4
    >>> mdt.strc.cms_n_common_contacts(cms)
    2
    """
    mdt.check.list_of_cms(cms)
    if sparse.issparse(cms[0]):
        bound_in_all_cms = mdt.sph.multiple_multiply(*cms)
        n_bound_in_all_cms = bound_in_all_cms.count_nonzero()
    else:
        if len(cms) == 1:
            bound_in_all_cms = cms[0]
        else:
            bound_in_all_cms = np.logical_and.reduce(cms)
        n_bound_in_all_cms = np.count_nonzero(bound_in_all_cms)
    return n_bound_in_all_cms


def cms_n_contacts(cms, dtype=int):
    """
    Get the number of contancts per contact matrix and the number of
    contacts common in all contact matrices.

    Parameters
    ----------
    cms : tuple or list
        List of contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrices`.  The contact
        matrices can be given either as
        :class:`NumPy arrays <numpy.ndarray>` or as
        :mod:`SciPy sparse matrices <scipy.sparse>`.  A mix of
        :class:`NumPy arrays <numpy.ndarray>` and
        :mod:`SciPy sparse matrices <scipy.sparse>` is not possible.
    dtype : dtype, optional
        Data type of the output array.

    Returns
    -------
    n_contacts : numpy.ndarray
        Array of shape ``(len(cms)+1,)`` containing the number of
        non-zero elements of each array in `cms`.  The last element of
        `n_contacts` is the number of non-zero elements of the product
        of all arrays in `cms` (i.e. the number of non-zero elements
        that are common in all arrays in `cms`).

    See Also
    --------
    :func:`mdtools.structure.contact_matrix` :
        Construct a boolean contact matrix for two MDAnalysis
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`
    :func:`mdtools.structure.contact_matrices` :
        Construct a contact matrix for each frame in a trajectory
    :func:`mdtools.structure.cms_n_common_contacts` :
        Get the number of contancts common in all contact matrices

    Examples
    --------
    :class:`NumPy arrays <numpy.ndarray>` as input:

    >>> cm0 = np.array([[ True,  True, False],
    ...                 [False,  True,  True]])
    >>> cm1 = np.array([[ True, False, False],
    ...                 [False, False,  True]])
    >>> cm2 = np.array([[ True,  True, False],
    ...                 [False, False,  True]])
    >>> mdt.strc.cms_n_contacts([cm0,])
    array([4, 4])
    >>> cms = [cm0, cm1, cm2]
    >>> mdt.strc.cms_n_contacts(cms)
    array([4, 2, 3, 2])
    >>> n_contacts = np.array([np.count_nonzero(cm) for cm in cms])
    >>> n_contacts
    array([4, 2, 3])
    >>> np.array_equal(n_contacts, mdt.strc.cms_n_contacts(cms)[:-1])
    True

    :mod:`SciPy sparse matrices <scipy.sparse>` as input:

    >>> from scipy import sparse
    >>> cms = [sparse.csr_matrix(cm) for cm in cms]
    >>> mdt.strc.cms_n_contacts([cms[0],])
    array([4, 4])
    >>> mdt.strc.cms_n_contacts(cms)
    array([4, 2, 3, 2])
    """
    mdt.check.list_of_cms(cms)
    n_contacts = np.zeros(len(cms) + 1, dtype=dtype)
    if sparse.issparse(cms[0]):
        for i, cm in enumerate(cms):
            n_contacts[i] = cm.count_nonzero()
        bound_in_all_cms = mdt.sph.multiple_multiply(*cms)
        n_contacts[-1] = bound_in_all_cms.count_nonzero()
    else:
        for i, cm in enumerate(cms):
            n_contacts[i] = np.count_nonzero(cm)
        if len(cms) == 1:
            bound_in_all_cms = cms[0]
        else:
            bound_in_all_cms = np.logical_and.reduce(cms, axis=0)
        n_contacts[-1] = np.count_nonzero(bound_in_all_cms)
    return n_contacts


def cm_selix_stats(cm, unbound_nan=False):
    """
    Get statistics about the indices of selection compounds bound to
    reference compounds for each reference compound.

    Parameters
    ----------
    cm : array_like or :mod:`SciPy sparse matrix <scipy.sparse>`
        (Boolean) contact matrix of shape ``(m, n)`` as e.g. generated
        by :func:`mdtools.structure.contact_matrix`, where ``m`` is the
        number of reference compounds and ``n`` is the number of
        selection compounds.
    unbound_nan : bool, optional
        If ``True``, set the output values for reference compounds that
        are not in contact with any selection compound to ``numpy.nan``.
        Otherwise the output values for unbound reference compounds will
        be zero, which can be misinterpreted, because if a reference
        compound is only bound by the selection compound with index
        zero, all output values will be zero, too.

    Returns
    -------
    selix_stats : numpy.ndarray
        Array of shape ``(m, 5)``.  The five columns contain for each
        reference compound

            1. the number
            2. the mean of the indices
            3. the variance of the indices
            4. the minimum of the indices
            5. the maximum of the indices

        of selection compounds that are in contact with the given
        reference compound.

    Examples
    --------
    :class:`NumPy array <numpy.ndarray>` as input:

    >>> cm = np.eye(5, 4, -1, dtype=bool) + np.eye(5, 4, -2, dtype=bool)
    >>> cm
    array([[False, False, False, False],
           [ True, False, False, False],
           [ True,  True, False, False],
           [False,  True,  True, False],
           [False, False,  True,  True]])
    >>> # Contact matrix containing the indices of the selection
    >>> # compounds that are bound to reference compounds:
    >>> cm * np.arange(cm.shape[1])
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 2, 0],
           [0, 0, 2, 3]])
    >>> mdt.strc.cm_selix_stats(cm)
    array([[0.  , 0.  , 0.  , 0.  , 0.  ],
           [1.  , 0.  , 0.  , 0.  , 0.  ],
           [2.  , 0.5 , 0.25, 0.  , 1.  ],
           [2.  , 1.5 , 0.25, 1.  , 2.  ],
           [2.  , 2.5 , 0.25, 2.  , 3.  ]])
    >>> # [n_sel, mean, var , min , max ]
    >>> mdt.strc.cm_selix_stats(cm, unbound_nan=True)
    array([[0.  ,  nan,  nan,  nan,  nan],
           [1.  , 0.  , 0.  , 0.  , 0.  ],
           [2.  , 0.5 , 0.25, 0.  , 1.  ],
           [2.  , 1.5 , 0.25, 1.  , 2.  ],
           [2.  , 2.5 , 0.25, 2.  , 3.  ]])

    :mod:`SciPy sparse matrices <scipy.sparse>` as input:

    >>> from scipy import sparse
    >>> cm = sparse.csr_matrix(cm)
    >>> mdt.strc.cm_selix_stats(cm)
    array([[0.  , 0.  , 0.  , 0.  , 0.  ],
           [1.  , 0.  , 0.  , 0.  , 0.  ],
           [2.  , 0.5 , 0.25, 0.  , 1.  ],
           [2.  , 1.5 , 0.25, 1.  , 2.  ],
           [2.  , 2.5 , 0.25, 2.  , 3.  ]])
    >>> mdt.strc.cm_selix_stats(cm, unbound_nan=True)
    array([[0.  ,  nan,  nan,  nan,  nan],
           [1.  , 0.  , 0.  , 0.  , 0.  ],
           [2.  , 0.5 , 0.25, 0.  , 1.  ],
           [2.  , 1.5 , 0.25, 1.  , 2.  ],
           [2.  , 2.5 , 0.25, 2.  , 3.  ]])
    """
    if sparse.issparse(cm):
        cm = cm.astype(bool, copy=True)
        if cm.ndim != 2:
            raise ValueError("'cm' has {} dimension(s) but must have 2"
                             .format(cm.ndim))
        cm.eliminate_zeros()
        # Number of bound selection compounds per reference compound
        n_sel_bound = np.squeeze(np.asarray(cm.getnnz(axis=1)))
        bound = n_sel_bound.astype(bool)
        selix_min = np.squeeze(np.asarray(cm.argmax(axis=1)))
        # cm -> cm_selix (cm now contains for each reference compound
        # the indices of the bound selection compounds)
        cm = cm.multiply(np.arange(cm.shape[1], dtype=np.uint32))
        selix_max = np.squeeze(cm.max(axis=1).toarray())
        # Mean of non-zero elements:
        selix_mean = np.squeeze(np.asarray(cm.sum(axis=1,
                                                  dtype=np.float64)))
        np.divide(selix_mean, n_sel_bound, where=bound, out=selix_mean)
        # Variance of non-zero elements:
        selix_var = cm.multiply(cm).sum(axis=1, dtype=np.float64)
        selix_var = np.squeeze(np.asarray(selix_var))
        np.divide(selix_var, n_sel_bound, where=bound, out=selix_var)
        selix_var -= selix_mean**2
    else:
        cm = np.asarray(cm, dtype=bool)
        if cm.ndim != 2:
            raise ValueError("'cm' has {} dimension(s) but must have 2"
                             .format(cm.ndim))
        n_sel_bound = np.count_nonzero(cm, axis=1)
        bound = n_sel_bound.astype(bool)
        selix_min = cm.argmax(axis=1)
        # cm -> cm_selix
        cm = cm * np.arange(cm.shape[1], dtype=np.uint32)
        selix_max = cm.max(axis=1)
        selix_mean = cm.sum(axis=1, dtype=np.float64)
        np.divide(selix_mean, n_sel_bound, where=bound, out=selix_mean)
        cm *= cm
        selix_var = cm.sum(axis=1, dtype=np.float64)
        np.divide(selix_var, n_sel_bound, where=bound, out=selix_var)
        selix_var -= selix_mean**2
    if unbound_nan:
        selix_stats = np.column_stack([selix_mean,
                                       selix_var,
                                       selix_min,
                                       selix_max])
        selix_stats[~bound] = np.nan
        selix_stats = np.insert(selix_stats, 0, n_sel_bound, axis=1)
    else:
        selix_stats = np.column_stack([n_sel_bound,
                                       selix_mean,
                                       selix_var,
                                       selix_min,
                                       selix_max])
    return np.asarray(selix_stats, order='C')


def contact_hist_refcmp_diff_selcmp(
        cm, natms_per_refcmp=1, natms_per_selcmp=1, minlength=0,
        dtype=int):
    """
    Bin the number of contacts that reference compounds establish to
    **different** selection compounds into a histogram.

    A compound is usually a chemically meaningfull subgroup of an
    :class:`~MDAnalysis.core.groups.AtomGroup`.  This can e.g. be a
    :class:`~MDAnalysis.core.groups.Segment`,
    :class:`~MDAnalysis.core.groups.Residue`,
    :attr:`fragment <MDAnalysis.core.groups.AtomGroup.fragments>` or
    a single :class:`~MDAnalysis.core.groups.Atom`.
    Refer to the MDAnalysis' user guide for an
    |explanation_of_these_terms|.  Note that in any case, only
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
    original :class:`~MDAnalysis.core.groups.AtomGroup` are taken into
    account, even if the compound might comprise additional
    :class:`Atoms <MDAnalysis.core.groups.Atom>` that are not contained
    in the original :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    cm : array_like
        (Boolean) contact matrix of shape ``(m, n)`` as e.g. generated
        by :func:`mdtools.structure.contact_matrix`, where ``m`` is the
        number of reference :class:`Atoms <MDAnalysis.core.groups.Atom>`
        and ``n`` is the number of selection
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  Alternatively,
        `cm` can already be a compound contact matrix as e.g. generated
        by :func:`mdtools.structure.cmp_contact_matrix`.  In this case,
        you probably want to set `natms_per_refcmp` and
        `natms_per_selcmp` to ``1``, to keep `cm` unaltered.  It is also
        possible to parse a compound contact **count** matrix as e.g.
        generated by :func:`mdtools.structure.cmp_contact_count_matrix`.
    natms_per_refcmp : int or array_like, optional
        Number of :class:`Atoms <MDAnalysis.core.groups.Atom>` per
        reference compound.  Can be a single integer or an array of
        integers.  If `natms_per_refcmp` is a single integer, all
        reference compounds are assumed to contain the same number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  In this case,
        `natms_per_refcmp` must be an integer divisor of ``cm.shape[0]``.
        If `natms_per_refcmp` is an array of integers, it must contain
        the number of reference
        :class:`Atoms <MDAnalysis.core.groups.Atom>` for each single
        reference compound.  In this case,
        ``numpy.sum(natms_per_refcmp)`` must be equal to ``cm.shape[0]``.
    natms_per_selcmp : int or array_like, optional
        Same for selection compounds (`natms_per_selcmp` is checked
        against ``cm.shape[1]``).
    minlength : int, optional
        A minimum number of bins for the output array.  The output array
        will have at least this number of elements, though it will be
        longer if necessary.  See :func:`numpy.bincount` for furhter
        information.
    dtype : dtype, optional
        Data type of the output array.

    Returns
    -------
    hist_refcmp_diff_selcmp : numpy.ndarray
        Histogram of the number of contacts that reference compounds
        establish to **different** selection compounds.  Multiple
        contacts with the same selection compound are not taken into
        account.

    See Also
    --------
    :func:`mdtools.structure.contact_matrix` :
        Construct a boolean contact matrix for two MDAnalysis
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`
    :func:`mdtools.structure.natms_per_cmp`:
        Get the number of :class:`Atoms <MDAnalysis.core.groups.Atom>`
        of each compound in an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.contact_hists` :
        Bin the number of contacts between reference and selection
        compounds into histograms.
    :func:`mdtools.structure.contact_hist_refcmp_same_selcmp` :
        Bin the number of contacts that reference compounds establish to
        the **same** selection compound into a histogram.
    :func:`mdtools.structure.contact_hist_refcmp_selcmp_tot` :
        Bin the **total** number of contacts that reference compounds
        establish to selection compounds into a histogram.
    :func:`mdtools.structure.contact_hist_refcmp_selcmp_pair` :
        Bin the number of "bonds" (:class:`~MDAnalysis.core.groups.Atom`-
        :class:`~MDAnalysis.core.groups.Atom` contacts) between pairs of
        reference and selection compounds
    :func:`numpy.count_nonzero` :
        Count the number of non-zero values in an array
    :func:`numpy.bincount` :
        Count the number of occurrences of each value in an array of
        non-negative ints

    Notes
    -----
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the same
    compound must form a contiguous set in the input contact matrix,
    otherwise the result will be wrong.

    About the output array:

    `hist_refcmp_diff_selcmp`
        The first element is the number of reference compounds having no
        contact with any selection compound, the second element is the
        number of reference compounds having contact with exactly one
        selection compound, the third element is the number of reference
        compounds having contact with exactly two **different**
        selection compounds, and so on.

    If both `natms_per_refcmp` and `natms_per_selcmp` are ``1`` and `cm`
    is a true boolean contact matrix,
    :func:`mdtools.structure.contact_hist_refcmp_diff_selcmp` and
    :func:`mdtools.structure.contact_hist_refcmp_selcmp_tot` return the
    same result.

    Examples
    --------
    >>> cm = np.tril(np.ones((5,4), dtype=bool), -1)
    >>> cm[3, 0] = 0
    >>> cm
    array([[False, False, False, False],
           [ True, False, False, False],
           [ True,  True, False, False],
           [False,  True,  True, False],
           [ True,  True,  True,  True]])
    >>> mdt.strc.contact_hist_refcmp_diff_selcmp(cm)
    array([1, 1, 2, 0, 1])
    >>> mdt.strc.contact_hist_refcmp_diff_selcmp(cm=cm, minlength=7)
    array([1, 1, 2, 0, 1, 0, 0])
    >>> mdt.strc.contact_hist_refcmp_diff_selcmp(cm=cm, minlength=3)
    array([1, 1, 2, 0, 1])
    >>> mdt.strc.contact_hist_refcmp_diff_selcmp(cm=cm, dtype=np.uint32)
    array([1, 1, 2, 0, 1], dtype=uint32)

    >>> mdt.strc.cmp_contact_matrix(cm=cm, natms_per_refcmp=[2, 2, 1])
    array([[ True, False, False, False],
           [ True,  True,  True, False],
           [ True,  True,  True,  True]])
    >>> mdt.strc.contact_hist_refcmp_diff_selcmp(
    ...     cm=cm,
    ...     natms_per_refcmp=[2, 2, 1]
    ... )
    array([0, 1, 0, 1, 1])

    >>> mdt.strc.cmp_contact_matrix(cm=cm, natms_per_selcmp=2)
    array([[False, False],
           [ True, False],
           [ True, False],
           [ True,  True],
           [ True,  True]])
    >>> mdt.strc.contact_hist_refcmp_diff_selcmp(cm=cm,
    ...                                          natms_per_selcmp=2)
    array([1, 2, 2])

    >>> mdt.strc.cmp_contact_matrix(cm=cm,
    ...                             natms_per_refcmp=[2, 2, 1],
    ...                             natms_per_selcmp=2)
    array([[ True, False],
           [ True,  True],
           [ True,  True]])
    >>> mdt.strc.contact_hist_refcmp_diff_selcmp(
    ...     cm=cm,
    ...     natms_per_refcmp=[2, 2, 1],
    ...     natms_per_selcmp=2
    ... )
    array([0, 1, 2])

    Edge cases:

    >>> cm = np.array([], dtype=bool).reshape(0, 4)
    >>> mdt.strc.contact_hist_refcmp_diff_selcmp(cm, natms_per_refcmp=[])
    array([], dtype=int64)
    >>> mdt.strc.contact_hist_refcmp_diff_selcmp(cm, natms_per_refcmp=1)
    array([], dtype=int64)
    >>> mdt.strc.contact_hist_refcmp_diff_selcmp(cm,
    ...                                          natms_per_refcmp=[],
    ...                                          minlength=2,
    ...                                          dtype=np.uint32)
    array([0, 0], dtype=uint32)

    >>> cm = np.array([], dtype=bool).reshape(6, 0)
    >>> mdt.strc.contact_hist_refcmp_diff_selcmp(cm,
    ...                                          natms_per_refcmp=3,
    ...                                          natms_per_selcmp=[])
    array([2])
    """
    cm = cmp_contact_matrix(cm=cm,
                            natms_per_refcmp=natms_per_refcmp,
                            natms_per_selcmp=natms_per_selcmp)
    contacts = np.count_nonzero(cm, axis=1)
    hist_refcmp_diff_selcmp = np.bincount(contacts, minlength=minlength)
    return np.asarray(hist_refcmp_diff_selcmp, dtype=dtype)


def contact_hist_refcmp_same_selcmp(
        cm, natms_per_refcmp=1, natms_per_selcmp=1, minlength=0,
        dtype=int):
    """
    Bin the number of contacts that reference compounds establish to
    the **same** selection compound into a histogram.

    A compound is usually a chemically meaningfull subgroup of an
    :class:`~MDAnalysis.core.groups.AtomGroup`.  This can e.g. be a
    :class:`~MDAnalysis.core.groups.Segment`,
    :class:`~MDAnalysis.core.groups.Residue`,
    :attr:`fragment <MDAnalysis.core.groups.AtomGroup.fragments>` or
    a single :class:`~MDAnalysis.core.groups.Atom`.
    Refer to the MDAnalysis' user guide for an
    |explanation_of_these_terms|.  Note that in any case, only
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
    original :class:`~MDAnalysis.core.groups.AtomGroup` are taken into
    account, even if the compound might comprise additional
    :class:`Atoms <MDAnalysis.core.groups.Atom>` that are not contained
    in the original :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    cm : array_like
        (Boolean) contact matrix of shape ``(m, n)`` as e.g. generated
        by :func:`mdtools.structure.contact_matrix`, where ``m`` is the
        number of reference :class:`Atoms <MDAnalysis.core.groups.Atom>`
        and ``n`` is the number of selection
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  Alternatively,
        `cm` can already be a compound contact **count** matrix as e.g.
        generated by :func:`mdtools.structure.cmp_contact_count_matrix`.
        In this case, you probably want to set `natms_per_refcmp` and
        `natms_per_selcmp` to ``1``, to keep `cm` unaltered.
    natms_per_refcmp : int or array_like, optional
        Number of :class:`Atoms <MDAnalysis.core.groups.Atom>` per
        reference compound.  Can be a single integer or an array of
        integers.  If `natms_per_refcmp` is a single integer, all
        reference compounds are assumed to contain the same number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  In this case,
        `natms_per_refcmp` must be an integer divisor of ``cm.shape[0]``.
        If `natms_per_refcmp` is an array of integers, it must contain
        the number of reference
        :class:`Atoms <MDAnalysis.core.groups.Atom>` for each single
        reference compound.  In this case,
        ``numpy.sum(natms_per_refcmp)`` must be equal to ``cm.shape[0]``.
    natms_per_selcmp : int or array_like, optional
        Same for selection compounds (`natms_per_selcmp` is checked
        against ``cm.shape[1]``).
    minlength : int, optional
        A minimum number of bins for the output array.  The output array
        will have at least this number of elements, though it will be
        longer if necessary.
    dtype : dtype, optional
        Data type of the output array.

    Returns
    -------
    hist_refcmp_same_selcmp : numpy.ndarray
        Histogram of the number of contacts that reference compounds
        establish to the **same** selection compound.  Different
        selection compounds that are connected to the same reference
        compound via the same number of "bonds"
        (:class:`~MDAnalysis.core.groups.Atom`-
        :class:`~MDAnalysis.core.groups.Atom` contacts) are not taken
        into account.

    See Also
    --------
    :func:`mdtools.structure.contact_matrix` :
        Construct a boolean contact matrix for two MDAnalysis
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`
    :func:`mdtools.structure.natms_per_cmp`:
        Get the number of :class:`Atoms <MDAnalysis.core.groups.Atom>`
        of each compound in an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.contact_hists` :
        Bin the number of contacts between reference and selection
        compounds into histograms.
    :func:`mdtools.structure.contact_hist_refcmp_diff_selcmp` :
        Bin the number of contacts that reference compounds establish to
        **different** selection compounds into a histogram.
    :func:`mdtools.structure.contact_hist_refcmp_selcmp_tot` :
        Bin the **total** number of contacts that reference compounds
        establish to selection compounds into a histogram.
    :func:`mdtools.structure.contact_hist_refcmp_selcmp_pair` :
        Bin the number of "bonds" (:class:`~MDAnalysis.core.groups.Atom`-
        :class:`~MDAnalysis.core.groups.Atom` contacts) between pairs of
        reference and selection compounds

    Notes
    -----
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the same
    compound must form a contiguous set in the input contact matrix,
    otherwise the result will be wrong.

    About the output array:

    `hist_refcmp_same_selcmp`
        The first element is the number of reference compounds having no
        contact with any selection compound, the second element is the
        number of reference compounds having contact with **at least**
        one selection compound via exactly one "bond", the third element
        is the number of reference compounds having contact with
        **at least** one selection compound via exactly two "bonds", and
        so on.

        Important: Different selection compounds that are connected to
        the same reference compound via the same number of "bonds" are
        not taken into account.  For instance, if a reference compound
        is connected to two different selection compounds via one "bond",
        respectively, only the first selection compound is counted.
        However, if the reference compound is connected to the first
        selection compound via one "bond" and to the second selection
        compound via two "bonds", both selection compounds are counted.

        The sum of all histogram elements might therefore exceed the
        number of reference compounds, because a single reference
        compound can be connected to different selection compounds with
        different numbers of "bonds".  However, each histogram element
        on its own cannot exceed the number of reference compounds,
        because different selection compounds that are connected to the
        same reference compound via the same number of "bonds" are not
        taken into account.

        Hence it is e.g. possible to say that 100 % of the reference
        compounds are coordinated monodentately by selection compounds
        while at the same time 50 % of the reference compounds are
        additionally coordinated bidentately.

        This behavior is complementary to the histogram returned by
        :func:`mdtools.structure.contact_hist_refcmp_selcmp_pair`.

    If both `natms_per_refcmp` and `natms_per_selcmp` are ``1`` and `cm`
    is a true boolean contact matrix, `hist_refcmp_same_selcmp` is
    equal to ``[x, cm.shape[0]-x]``, where ``x`` is the number of
    reference compounds having no contact with any selection compound.

    Examples
    --------
    >>> cm = np.tril(np.ones((5,4), dtype=bool), -1)
    >>> cm[3, 0] = 0
    >>> cm
    array([[False, False, False, False],
           [ True, False, False, False],
           [ True,  True, False, False],
           [False,  True,  True, False],
           [ True,  True,  True,  True]])
    >>> mdt.strc.contact_hist_refcmp_same_selcmp(cm)
    array([1, 4])
    >>> mdt.strc.contact_hist_refcmp_same_selcmp(cm=cm, minlength=4)
    array([1, 4, 0, 0])
    >>> mdt.strc.contact_hist_refcmp_same_selcmp(cm=cm, minlength=1)
    array([1, 4])
    >>> mdt.strc.contact_hist_refcmp_same_selcmp(cm=cm, dtype=np.uint32)
    array([1, 4], dtype=uint32)
    >>> hist = mdt.strc.contact_hist_refcmp_same_selcmp(cm)
    >>> hist[1] == cm.shape[0] - hist[0]
    True

    >>> mdt.strc.cmp_contact_count_matrix(cm=cm,
    ...                                   natms_per_refcmp=[2, 2, 1])
    array([[1, 0, 0, 0],
           [1, 2, 1, 0],
           [1, 1, 1, 1]])
    >>> mdt.strc.contact_hist_refcmp_same_selcmp(
    ...     cm=cm,
    ...     natms_per_refcmp=[2, 2, 1]
    ... )
    array([0, 3, 1])

    >>> mdt.strc.cmp_contact_count_matrix(cm=cm, natms_per_selcmp=2)
    array([[0, 0],
           [1, 0],
           [2, 0],
           [1, 1],
           [2, 2]])
    >>> mdt.strc.contact_hist_refcmp_same_selcmp(cm=cm,
    ...                                          natms_per_selcmp=2)
    array([1, 2, 2])

    >>> mdt.strc.cmp_contact_count_matrix(cm=cm,
    ...                                   natms_per_refcmp=[2, 2, 1],
    ...                                   natms_per_selcmp=2)
    array([[1, 0],
           [3, 1],
           [2, 2]])
    >>> mdt.strc.contact_hist_refcmp_same_selcmp(
    ...     cm=cm,
    ...     natms_per_refcmp=[2, 2, 1],
    ...     natms_per_selcmp=2
    ... )
    array([0, 2, 1, 1])

    Edge cases:

    >>> cm = np.array([], dtype=bool).reshape(0, 4)
    >>> mdt.strc.contact_hist_refcmp_same_selcmp(cm, natms_per_refcmp=[])
    array([], dtype=int64)
    >>> mdt.strc.contact_hist_refcmp_same_selcmp(cm, natms_per_refcmp=1)
    array([], dtype=int64)
    >>> mdt.strc.contact_hist_refcmp_same_selcmp(cm,
    ...                                          natms_per_refcmp=[],
    ...                                          minlength=2,
    ...                                          dtype=np.uint32)
    array([0, 0], dtype=uint32)

    >>> cm = np.array([], dtype=bool).reshape(6, 0)
    >>> mdt.strc.contact_hist_refcmp_same_selcmp(cm,
    ...                                          natms_per_refcmp=3,
    ...                                          natms_per_selcmp=[])
    array([2])
    """
    cm = np.asarray(cm)
    cm_dtype = cm.dtype
    cm = cmp_contact_count_matrix(cm=cm,
                                  natms_per_refcmp=natms_per_refcmp,
                                  natms_per_selcmp=natms_per_selcmp,
                                  dtype=np.uint32)
    if np.any(np.equal(cm.shape, 0)):
        if cm.shape[0] == 0:
            hist = np.zeros(minlength, dtype=dtype)
        else:  # cm.shape[1] == 0 and cm.shape[0] != 0
            hist = np.array([cm.shape[0]], dtype=dtype)
            hist = mdt.nph.extend(hist, minlength)
        return hist
    occurring_contacts = np.unique(cm)  # Unique and sorted contacts
    if occurring_contacts[0] == 0:
        zero_contacts_exist = True
        hist_length = max(occurring_contacts[-1] + 1, minlength)
        occurring_contacts = occurring_contacts[1:]
    elif occurring_contacts[0] > 0:
        zero_contacts_exist = False
        hist_length = max(occurring_contacts[-1] + 1, minlength)
    elif occurring_contacts[0] < 0:
        raise ValueError("'cm' must not contain negative elements")
    hist = np.zeros(hist_length, dtype=dtype)
    pair_has_n_contacts = np.zeros_like(cm, dtype=bool)
    any_pair_has_n_contacts = np.zeros(cm.shape[0], dtype=bool)
    for n in occurring_contacts:
        # refcmp-selcmp pair has n contacts
        np.equal(cm, n, out=pair_has_n_contacts)
        # refcmp/selcmp has n contacts with *any* selcmp/refcmp
        np.any(pair_has_n_contacts, axis=1, out=any_pair_has_n_contacts)
        # Number of refcmp/selcmp having n contacts with any selcmp/refcmp
        hist[n] = np.count_nonzero(any_pair_has_n_contacts)
    if zero_contacts_exist:
        hist[0] = cm.shape[0]
        np.greater(cm, 0, out=pair_has_n_contacts)
        np.any(pair_has_n_contacts, axis=1, out=any_pair_has_n_contacts)
        hist[0] -= np.count_nonzero(any_pair_has_n_contacts)
    # Internal consistency check:
    if (np.issubdtype(cm_dtype, bool) and
        np.all(np.equal(natms_per_refcmp, 1)) and
            np.all(np.equal(natms_per_selcmp, 1))):
        # refcmp == refatm and selcmp == selatm
        np.equal(cm, 0, out=pair_has_n_contacts)
        # any_pair_has_n_contacts should now be called
        # all_pairs_have_n_contacts
        np.all(pair_has_n_contacts, axis=1, out=any_pair_has_n_contacts)
        n_unbound_ref = np.count_nonzero(any_pair_has_n_contacts)
        hist_test = np.array([n_unbound_ref, cm.shape[0] - n_unbound_ref])
        hist_test = mdt.nph.extend(hist_test, len(hist))
        if not np.array_equal(hist, hist_test):
            raise ValueError("refcmp == refatm and selcmp == selatm, but"
                             "'hist_refcmp_same_selcmp' !="
                             " [x, cm.shape[0]-x]")
    return hist


def contact_hist_refcmp_selcmp_tot(
        cm, natms_per_refcmp=1, natms_per_selcmp=1, minlength=0,
        dtype=int):
    """
    Bin the **total** number of contacts that reference compounds
    establish to selection compounds into a histogram.

    A compound is usually a chemically meaningfull subgroup of an
    :class:`~MDAnalysis.core.groups.AtomGroup`.  This can e.g. be a
    :class:`~MDAnalysis.core.groups.Segment`,
    :class:`~MDAnalysis.core.groups.Residue`,
    :attr:`fragment <MDAnalysis.core.groups.AtomGroup.fragments>` or
    a single :class:`~MDAnalysis.core.groups.Atom`.
    Refer to the MDAnalysis' user guide for an
    |explanation_of_these_terms|.  Note that in any case, only
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
    original :class:`~MDAnalysis.core.groups.AtomGroup` are taken into
    account, even if the compound might comprise additional
    :class:`Atoms <MDAnalysis.core.groups.Atom>` that are not contained
    in the original :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    cm : array_like
        (Boolean) contact matrix of shape ``(m, n)`` as e.g. generated
        by :func:`mdtools.structure.contact_matrix`, where ``m`` is the
        number of reference :class:`Atoms <MDAnalysis.core.groups.Atom>`
        and ``n`` is the number of selection
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  Alternatively,
        `cm` can already be a compound contact **count** matrix as e.g.
        generated by :func:`mdtools.structure.cmp_contact_count_matrix`.
        In this case, you probably want to set `natms_per_refcmp` and
        `natms_per_selcmp` to ``1``, to keep `cm` unaltered.
    natms_per_refcmp : int or array_like, optional
        Number of :class:`Atoms <MDAnalysis.core.groups.Atom>` per
        reference compound.  Can be a single integer or an array of
        integers.  If `natms_per_refcmp` is a single integer, all
        reference compounds are assumed to contain the same number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  In this case,
        `natms_per_refcmp` must be an integer divisor of ``cm.shape[0]``.
        If `natms_per_refcmp` is an array of integers, it must contain
        the number of reference
        :class:`Atoms <MDAnalysis.core.groups.Atom>` for each single
        reference compound.  In this case,
        ``numpy.sum(natms_per_refcmp)`` must be equal to ``cm.shape[0]``.
    natms_per_selcmp : int or array_like, optional
        Same for selection compounds (`natms_per_selcmp` is checked
        against ``cm.shape[1]``).
    minlength : int, optional
        A minimum number of bins for the output array.  The output array
        will have at least this number of elements, though it will be
        longer if necessary.  See :func:`numpy.bincount` for furhter
        information.
    dtype : dtype, optional
        Data type of the output array.

    Returns
    -------
    hist_refcmp_selcmp_tot : numpy.ndarray
        Histogram of the **total** number of contacts that reference
        compounds establish to selection compounds.  All contacts are
        taken into account.

    See Also
    --------
    :func:`mdtools.structure.contact_matrix` :
        Construct a boolean contact matrix for two MDAnalysis
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`
    :func:`mdtools.structure.natms_per_cmp`:
        Get the number of :class:`Atoms <MDAnalysis.core.groups.Atom>`
        of each compound in an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.contact_hists` :
        Bin the number of contacts between reference and selection
        compounds into histograms.
    :func:`mdtools.structure.contact_hist_refcmp_diff_selcmp` :
        Bin the number of contacts that reference compounds establish to
        **different** selection compounds into a histogram.
    :func:`mdtools.structure.contact_hist_refcmp_same_selcmp` :
        Bin the number of contacts that reference compounds establish to
        the **same** selection compound into a histogram.
    :func:`mdtools.structure.contact_hist_refcmp_selcmp_pair` :
        Bin the number of "bonds" (:class:`~MDAnalysis.core.groups.Atom`-
        :class:`~MDAnalysis.core.groups.Atom` contacts) between pairs of
        reference and selection compounds
    :func:`numpy.bincount` :
        Count the number of occurrences of each value in an array of
        non-negative ints

    Notes
    -----
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the same
    compound must form a contiguous set in the input contact matrix,
    otherwise the result will be wrong.

    About the output array:

    `hist_refcmp_selcmp_tot`
        The first element is the number of reference compounds having no
        contact with any selection compound, the second element is the
        number of reference compounds having exactly one contact with
        selection compounds, the third element is the number of
        reference compounds having exactly two contacts with selection
        compounds, and so on.

    If both `natms_per_refcmp` and `natms_per_selcmp` are ``1`` and `cm`
    is a true boolean contact matrix,
    :func:`mdtools.structure.contact_hist_refcmp_diff_selcmp` and
    :func:`mdtools.structure.contact_hist_refcmp_selcmp_tot` return the
    same result.

    The total number of contacts is given by::

        np.sum(hist_refcmp_selcmp_tot * np.arange(len(hist_refcmp_selcmp_tot)))

    Note that this is equal to the total number of refatm-selatm pairs,
    when every reference and selection compound contains only one
    :class:`~MDAnalysis.core.groups.Atom`, because a given
    :class:`~MDAnalysis.core.groups.Atom` can either have one or no
    contact with another :class:`~MDAnalysis.core.groups.Atom`, but two
    :class:`Atoms <MDAnalysis.core.groups.Atom>` cannot have multiple
    contacts with each other.  See also
    :func:`mdtools.structure.contact_hist_refcmp_selcmp_pair`.

    Examples
    --------
    >>> cm = np.tril(np.ones((5,4), dtype=bool), -1)
    >>> cm[3, 0] = 0
    >>> cm
    array([[False, False, False, False],
           [ True, False, False, False],
           [ True,  True, False, False],
           [False,  True,  True, False],
           [ True,  True,  True,  True]])
    >>> mdt.strc.contact_hist_refcmp_selcmp_tot(cm)
    array([1, 1, 2, 0, 1])
    >>> mdt.strc.contact_hist_refcmp_selcmp_tot(cm=cm, minlength=7)
    array([1, 1, 2, 0, 1, 0, 0])
    >>> mdt.strc.contact_hist_refcmp_selcmp_tot(cm=cm, minlength=3)
    array([1, 1, 2, 0, 1])
    >>> mdt.strc.contact_hist_refcmp_selcmp_tot(cm=cm, dtype=np.uint32)
    array([1, 1, 2, 0, 1], dtype=uint32)

    >>> mdt.strc.cmp_contact_count_matrix(cm=cm,
    ...                                   natms_per_refcmp=[2, 2, 1])
    array([[1, 0, 0, 0],
           [1, 2, 1, 0],
           [1, 1, 1, 1]])
    >>> mdt.strc.contact_hist_refcmp_selcmp_tot(
    ...     cm=cm,
    ...     natms_per_refcmp=[2, 2, 1]
    ... )
    array([0, 1, 0, 0, 2])

    >>> mdt.strc.cmp_contact_count_matrix(cm=cm, natms_per_selcmp=2)
    array([[0, 0],
           [1, 0],
           [2, 0],
           [1, 1],
           [2, 2]])
    >>> mdt.strc.contact_hist_refcmp_selcmp_tot(cm=cm,
    ...                                         natms_per_selcmp=2)
    array([1, 1, 2, 0, 1])

    >>> mdt.strc.cmp_contact_count_matrix(cm=cm,
    ...                                   natms_per_refcmp=[2, 2, 1],
    ...                                   natms_per_selcmp=2)
    array([[1, 0],
           [3, 1],
           [2, 2]])
    >>> mdt.strc.contact_hist_refcmp_selcmp_tot(
    ...     cm=cm,
    ...     natms_per_refcmp=[2, 2, 1],
    ...     natms_per_selcmp=2
    ... )
    array([0, 1, 0, 0, 2])

    Edge cases:

    >>> cm = np.array([], dtype=bool).reshape(0, 4)
    >>> mdt.strc.contact_hist_refcmp_selcmp_tot(cm, natms_per_refcmp=[])
    array([], dtype=int64)
    >>> mdt.strc.contact_hist_refcmp_selcmp_tot(cm, natms_per_refcmp=1)
    array([], dtype=int64)
    >>> mdt.strc.contact_hist_refcmp_selcmp_tot(cm,
    ...                                         natms_per_refcmp=[],
    ...                                         minlength=2,
    ...                                         dtype=np.uint32)
    array([0, 0], dtype=uint32)

    >>> cm = np.array([], dtype=bool).reshape(6, 0)
    >>> mdt.strc.contact_hist_refcmp_selcmp_tot(cm,
    ...                                         natms_per_refcmp=3,
    ...                                         natms_per_selcmp=[])
    array([2])
    """
    cm = cmp_contact_count_matrix(cm=cm,
                                  natms_per_refcmp=natms_per_refcmp,
                                  natms_per_selcmp=natms_per_selcmp,
                                  dtype=np.uint32)
    contacts = np.sum(cm, axis=1, dtype=int)  # bincount cannot handle uint
    hist_refcmp_selcmp_tot = np.bincount(contacts, minlength=minlength)
    return np.asarray(hist_refcmp_selcmp_tot, dtype=dtype)


def contact_hist_refcmp_selcmp_pair(
        cm, natms_per_refcmp=1, natms_per_selcmp=1, minlength=0,
        dtype=int):
    """
    Bin the number of "bonds" (:class:`~MDAnalysis.core.groups.Atom`-
    :class:`~MDAnalysis.core.groups.Atom` contacts) between pairs of
    reference and selection compounds.

    A compound is usually a chemically meaningfull subgroup of an
    :class:`~MDAnalysis.core.groups.AtomGroup`.  This can e.g. be a
    :class:`~MDAnalysis.core.groups.Segment`,
    :class:`~MDAnalysis.core.groups.Residue`,
    :attr:`fragment <MDAnalysis.core.groups.AtomGroup.fragments>` or
    a single :class:`~MDAnalysis.core.groups.Atom`.
    Refer to the MDAnalysis' user guide for an
    |explanation_of_these_terms|.  Note that in any case, only
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
    original :class:`~MDAnalysis.core.groups.AtomGroup` are taken into
    account, even if the compound might comprise additional
    :class:`Atoms <MDAnalysis.core.groups.Atom>` that are not contained
    in the original :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    cm : array_like
        (Boolean) contact matrix of shape ``(m, n)`` as e.g. generated
        by :func:`mdtools.structure.contact_matrix`, where ``m`` is the
        number of reference :class:`Atoms <MDAnalysis.core.groups.Atom>`
        and ``n`` is the number of selection
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  Alternatively,
        `cm` can already be a compound contact **count** matrix as e.g.
        generated by :func:`mdtools.structure.cmp_contact_count_matrix`.
        In this case, you probably want to set `natms_per_refcmp` and
        `natms_per_selcmp` to ``1``, to keep `cm` unaltered.
    natms_per_refcmp : int or array_like, optional
        Number of :class:`Atoms <MDAnalysis.core.groups.Atom>` per
        reference compound.  Can be a single integer or an array of
        integers.  If `natms_per_refcmp` is a single integer, all
        reference compounds are assumed to contain the same number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  In this case,
        `natms_per_refcmp` must be an integer divisor of ``cm.shape[0]``.
        If `natms_per_refcmp` is an array of integers, it must contain
        the number of reference
        :class:`Atoms <MDAnalysis.core.groups.Atom>` for each single
        reference compound.  In this case,
        ``numpy.sum(natms_per_refcmp)`` must be equal to ``cm.shape[0]``.
    natms_per_selcmp : int or array_like, optional
        Same for selection compounds (`natms_per_selcmp` is checked
        against ``cm.shape[1]``).
    minlength : int, optional
        A minimum number of bins for the output array.  The output array
        will have at least this number of elements, though it will be
        longer if necessary.  See :func:`numpy.bincount` for furhter
        information.
    dtype : dtype, optional
        Data type of the output array.

    Returns
    -------
    hist_refcmp_selcmp_pair : numpy.ndarray
        Histogram of the number of "bonds"
        (:class:`~MDAnalysis.core.groups.Atom`-
        :class:`~MDAnalysis.core.groups.Atom` contacts) between pairs of
        reference and selection compounds (refcmp-selcmp pairs).  A
        refcmp-selcmp pair is defined as a reference and selection
        compound that are connected with each other via at least one
        "bond".

    See Also
    --------
    :func:`mdtools.structure.contact_matrix` :
        Construct a boolean contact matrix for two MDAnalysis
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`
    :func:`mdtools.structure.contact_hists` :
        Bin the number of contacts between reference and selection
        compounds into histograms.
    :func:`mdtools.structure.contact_hist_refcmp_diff_selcmp` :
        Bin the number of contacts that reference compounds establish to
        **different** selection compounds into a histogram.
    :func:`mdtools.structure.contact_hist_refcmp_same_selcmp` :
        Bin the number of contacts that reference compounds establish to
        the **same** selection compound into a histogram.
    :func:`mdtools.structure.contact_hist_refcmp_selcmp_tot` :
        Bin the **total** number of contacts that reference compounds
        establish to selection compounds into a histogram.
    :func:`numpy.bincount` :
        Count the number of occurrences of each value in an array of
        non-negative ints

    Notes
    -----
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the same
    compound must form a contiguous set in the input contact matrix,
    otherwise the result will be wrong.

    About the output array:

    `hist_refcmp_selcmp_pair`
        The first element is meaningless (a refcmp-selcmp pair with zero
        "bonds" is not a pair) and therefore set to zero.  The second
        element is the number of refcmp-selcmp pairs connected via
        exactly one "bond", the third element is the number of
        refcmp-selcmp pairs connected via exactly two "bonds", and so on.

        The sum of all histogram elements might exceed the number of
        reference compounds, because a single reference compound can be
        connected to different selection compounds via different numbers
        of "bonds".  Even each histogram element on its own might exceed
        the number of reference compounds, because a single reference
        compound can be connected to different selection compounds via
        the same number of "bonds".

        Hence, this histogram should be normalized by the number of
        refcmp-selcmp pairs and not by the number of reference compounds.
        Then it is e.g. possible to say that 100 % of the refcmp-selcmp
        connections are monodentate while at the same time 50 % of the
        refcmp-selcmp connections are bidentate.

        This behavior is complementary to the histogram returned by
        :func:`mdtools.structure.contact_hist_refcmp_same_selcmp`

    If both `natms_per_refcmp` and `natms_per_selcmp` are ``1`` and `cm`
    is a true boolean contact matrix, `hist_refcmp_selcmp_pair` is
    equal to ``[0, y]``, where ``y`` is the number of refatm-selatm
    pairs.

    Examples
    --------
    >>> cm = np.tril(np.ones((5,4), dtype=bool), -1)
    >>> cm[3, 0] = 0
    >>> cm
    array([[False, False, False, False],
           [ True, False, False, False],
           [ True,  True, False, False],
           [False,  True,  True, False],
           [ True,  True,  True,  True]])
    >>> mdt.strc.contact_hist_refcmp_selcmp_pair(cm)
    array([0, 9])
    >>> mdt.strc.contact_hist_refcmp_selcmp_pair(cm=cm, minlength=4)
    array([0, 9, 0, 0])
    >>> mdt.strc.contact_hist_refcmp_selcmp_pair(cm=cm, minlength=1)
    array([0, 9])
    >>> mdt.strc.contact_hist_refcmp_selcmp_pair(cm=cm, dtype=np.uint32)
    array([0, 9], dtype=uint32)
    >>> hist = mdt.strc.contact_hist_refcmp_selcmp_pair(cm)
    >>> hist[1] == np.count_nonzero(cm)
    True

    >>> mdt.strc.cmp_contact_count_matrix(cm=cm,
    ...                                   natms_per_refcmp=[2, 2, 1])
    array([[1, 0, 0, 0],
           [1, 2, 1, 0],
           [1, 1, 1, 1]])
    >>> mdt.strc.contact_hist_refcmp_selcmp_pair(
    ...     cm=cm,
    ...     natms_per_refcmp=[2, 2, 1]
    ... )
    array([0, 7, 1])

    >>> mdt.strc.cmp_contact_count_matrix(cm=cm, natms_per_selcmp=2)
    array([[0, 0],
           [1, 0],
           [2, 0],
           [1, 1],
           [2, 2]])
    >>> mdt.strc.contact_hist_refcmp_selcmp_pair(cm=cm,
    ...                                          natms_per_selcmp=2)
    array([0, 3, 3])

    >>> mdt.strc.cmp_contact_count_matrix(cm=cm,
    ...                                   natms_per_refcmp=[2, 2, 1],
    ...                                   natms_per_selcmp=2)
    array([[1, 0],
           [3, 1],
           [2, 2]])
    >>> mdt.strc.contact_hist_refcmp_selcmp_pair(
    ...     cm=cm,
    ...     natms_per_refcmp=[2, 2, 1],
    ...     natms_per_selcmp=2
    ... )
    array([0, 2, 2, 1])

    Edge cases:

    >>> cm = np.array([], dtype=bool).reshape(0, 4)
    >>> mdt.strc.contact_hist_refcmp_selcmp_pair(cm, natms_per_refcmp=[])
    array([], dtype=int64)
    >>> mdt.strc.contact_hist_refcmp_selcmp_pair(cm, natms_per_refcmp=1)
    array([], dtype=int64)
    >>> mdt.strc.contact_hist_refcmp_selcmp_pair(cm,
    ...                                          natms_per_refcmp=[],
    ...                                          minlength=2,
    ...                                          dtype=np.uint32)
    array([0, 0], dtype=uint32)

    >>> cm = np.array([], dtype=bool).reshape(6, 0)
    >>> mdt.strc.contact_hist_refcmp_selcmp_pair(cm,
    ...                                          natms_per_refcmp=3,
    ...                                          natms_per_selcmp=[])
    array([], dtype=int64)
    """
    cm = np.asarray(cm)
    cm_dtype = cm.dtype
    cm = cmp_contact_count_matrix(cm=cm,
                                  natms_per_refcmp=natms_per_refcmp,
                                  natms_per_selcmp=natms_per_selcmp,
                                  dtype=np.uint32)
    if np.any(np.equal(cm.shape, 0)):
        hist_refcmp_selcmp_pair = np.zeros(minlength, dtype=dtype)
    else:
        hist_refcmp_selcmp_pair = np.bincount(cm.flat,
                                              minlength=minlength)
        hist_refcmp_selcmp_pair[0] = 0
    # Internal consistency check:
    if (np.issubdtype(cm_dtype, bool) and
        not np.any(np.equal(cm.shape, 0)) and
        np.all(np.equal(natms_per_refcmp, 1)) and
            np.all(np.equal(natms_per_selcmp, 1))):
        # refcmp == refatm and selcmp == selatm
        hist_test = np.array([0, np.count_nonzero(cm)])
        hist_test = mdt.nph.extend(hist_test,
                                   len(hist_refcmp_selcmp_pair))
        if not np.array_equal(hist_refcmp_selcmp_pair, hist_test):
            raise ValueError("refcmp == refatm and selcmp == selatm, but"
                             "'hist_refcmp_selcmp_pair' != [0, y]")
    return np.asarray(hist_refcmp_selcmp_pair, dtype=dtype)


def contact_hists(
        cm, natms_per_refcmp=1, natms_per_selcmp=1, minlength=0,
        dtype=int):
    """
    Bin the number of contacts between reference and selection compounds
    into histograms.

    A compound is usually a chemically meaningfull subgroup of an
    :class:`~MDAnalysis.core.groups.AtomGroup`.  This can e.g. be a
    :class:`~MDAnalysis.core.groups.Segment`,
    :class:`~MDAnalysis.core.groups.Residue`,
    :attr:`fragment <MDAnalysis.core.groups.AtomGroup.fragments>` or
    a single :class:`~MDAnalysis.core.groups.Atom`.
    Refer to the MDAnalysis' user guide for an
    |explanation_of_these_terms|.  Note that in any case, only
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
    original :class:`~MDAnalysis.core.groups.AtomGroup` are taken into
    account, even if the compound might comprise additional
    :class:`Atoms <MDAnalysis.core.groups.Atom>` that are not contained
    in the original :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    cm : array_like
        (Boolean) contact matrix of shape ``(m, n)`` as e.g. generated
        by :func:`mdtools.structure.contact_matrix`, where ``m`` is the
        number of reference :class:`Atoms <MDAnalysis.core.groups.Atom>`
        and ``n`` is the number of selection
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  Alternatively,
        `cm` can already be a compound contact **count** matrix as e.g.
        generated by :func:`mdtools.structure.cmp_contact_count_matrix`.
        In this case, you probably want to set `natms_per_refcmp` and
        `natms_per_selcmp` to ``1``, to keep `cm` unaltered.
    natms_per_refcmp : int or array_like, optional
        Number of :class:`Atoms <MDAnalysis.core.groups.Atom>` per
        reference compound.  Can be a single integer or an array of
        integers.  If `natms_per_refcmp` is a single integer, all
        reference compounds are assumed to contain the same number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  In this case,
        `natms_per_refcmp` must be an integer divisor of ``cm.shape[0]``.
        If `natms_per_refcmp` is an array of integers, it must contain
        the number of reference
        :class:`Atoms <MDAnalysis.core.groups.Atom>` for each single
        reference compound.  In this case,
        ``numpy.sum(natms_per_refcmp)`` must be equal to ``cm.shape[0]``.
    natms_per_selcmp : int or array_like, optional
        Same for selection compounds (`natms_per_selcmp` is checked
        against ``cm.shape[1]``).
    minlength : int, optional
        A minimum number of bins for the output arrays.  The output
        arrays will have at least this number of elements, though they
        will be longer if necessary.  All output arrays will always have
        the same length.
    dtype : dtype, optional
        Data type of the output array.

    Returns
    -------
    hist_refcmp_diff_selcmp : numpy.ndarray
        Histogram of the number of contacts that reference compounds
        establish to **different** selection compounds.  Multiple
        contacts with the same selection compound are not taken into
        account.
    hist_refcmp_same_selcmp : numpy.ndarray
        Histogram of the number of contacts that reference compounds
        establish to the **same** selection compound.  Different
        selection compounds that are connected to the same reference
        compound via the same number of "bonds"
        (:class:`~MDAnalysis.core.groups.Atom`-
        :class:`~MDAnalysis.core.groups.Atom` contacts) are not taken
        into account.
    hist_refcmp_selcmp_tot : numpy.ndarray
        Histogram of the **total** number of contacts that reference
        compounds establish to selection compounds.  All contacts are
        taken into account.
    hist_refcmp_selcmp_pair : numpy.ndarray
        Histogram of the number of "bonds"
        (:class:`~MDAnalysis.core.groups.Atom`-
        :class:`~MDAnalysis.core.groups.Atom` contacts) between pairs of
        reference and selection compounds (refcmp-selcmp pairs).  A
        refcmp-selcmp pair is defined as a reference and selection
        compound that are connected with each other via at least one
        "bond".

    See Also
    --------
    :mod:`contact_hist` :
        MDTools script to calculate contact histograms.
    :func:`mdtools.structure.contact_matrix` :
        Construct a boolean contact matrix for two MDAnalysis
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`
    :func:`mdtools.structure.natms_per_cmp`:
        Get the number of :class:`Atoms <MDAnalysis.core.groups.Atom>`
        of each compound in an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup`
    :func:`mdtools.structure.contact_hist_refcmp_diff_selcmp` :
        Bin the number of contacts that reference compounds establish to
        **different** selection compounds into a histogram.
    :func:`mdtools.structure.contact_hist_refcmp_same_selcmp` :
        Bin the number of contacts that reference compounds establish to
        the **same** selection compound into a histogram.
    :func:`mdtools.structure.contact_hist_refcmp_selcmp_tot` :
        Bin the **total** number of contacts that reference compounds
        establish to selection compounds into a histogram.
    :func:`mdtools.structure.contact_hist_refcmp_selcmp_pair` :
        Bin the number of "bonds" (:class:`~MDAnalysis.core.groups.Atom`-
        :class:`~MDAnalysis.core.groups.Atom` contacts) between pairs of
        reference and selection compounds

    Notes
    -----
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the same
    compound must form a contiguous set in the input contact matrix,
    otherwise the result will be wrong.

    This function gathers the output of

        * :func:`mdtools.structure.contact_hist_refcmp_diff_selcmp`
        * :func:`mdtools.structure.contact_hist_refcmp_same_selcmp`
        * :func:`mdtools.structure.contact_hist_refcmp_selcmp_tot`
        * :func:`mdtools.structure.contact_hist_refcmp_selcmp_pair`

    See there for further details about the returned histograms.

    If both `natms_per_refcmp` and `natms_per_selcmp` are ``1`` and `cm`
    is a true boolean contact matrix, `hist_refcmp_same_selcmp` and
    `hist_refcmp_selcmp_pair` are rather meaningless and
    `hist_refcmp_diff_selcmp` and `hist_refcmp_selcmp_tot` are equal.
    Thus, in this case it might be better to call
    :func:`mdtools.structure.contact_hist_refcmp_diff_selcmp` or
    :func:`mdtools.structure.contact_hist_refcmp_selcmp_tot` directly.

    Examples
    --------
    >>> cm = np.tril(np.ones((5,4), dtype=bool), -1)
    >>> cm[3, 0] = 0
    >>> cm
    array([[False, False, False, False],
           [ True, False, False, False],
           [ True,  True, False, False],
           [False,  True,  True, False],
           [ True,  True,  True,  True]])
    >>> hists = mdt.strc.contact_hists(cm, minlength=7, dtype=np.uint32)
    >>> hists[0]
    array([1, 1, 2, 0, 1, 0, 0], dtype=uint32)
    >>> hists[1]
    array([1, 4, 0, 0, 0, 0, 0], dtype=uint32)
    >>> hists[2]
    array([1, 1, 2, 0, 1, 0, 0], dtype=uint32)
    >>> hists[3]
    array([0, 9, 0, 0, 0, 0, 0], dtype=uint32)
    >>> np.array_equal(hists[0], hists[2])
    True
    >>> hists[1][1] == cm.shape[0] - hists[1][0]
    True
    >>> hists[3][1] == np.count_nonzero(cm)
    True

    >>> mdt.strc.cmp_contact_count_matrix(cm=cm,
    ...                                   natms_per_refcmp=[2, 2, 1],
    ...                                   natms_per_selcmp=2)
    array([[1, 0],
           [3, 1],
           [2, 2]])
    >>> hists = mdt.strc.contact_hists(cm=cm,
    ...                                natms_per_refcmp=[2, 2, 1],
    ...                                natms_per_selcmp=2)
    >>> hists[0]
    array([0, 1, 2, 0, 0])
    >>> hists[1]
    array([0, 2, 1, 1, 0])
    >>> hists[2]
    array([0, 1, 0, 0, 2])
    >>> hists[3]
    array([0, 2, 2, 1, 0])

    Edge cases:

    >>> cm = np.array([], dtype=bool).reshape(0, 4)
    >>> mdt.strc.contact_hists(cm, natms_per_refcmp=[])
    (array([], dtype=int64), array([], dtype=int64), array([], dtype=int64), array([], dtype=int64))
    >>> mdt.strc.contact_hists(cm, natms_per_refcmp=1)
    (array([], dtype=int64), array([], dtype=int64), array([], dtype=int64), array([], dtype=int64))

    >>> cm = np.array([], dtype=bool).reshape(6, 0)
    >>> mdt.strc.contact_hists(cm,
    ...                        natms_per_refcmp=3,
    ...                        natms_per_selcmp=[])
    (array([2]), array([2]), array([2]), array([0]))
    """
    cm = np.asarray(cm)
    cm_dtype = cm.dtype
    cm = cmp_contact_count_matrix(cm=cm,
                                  natms_per_refcmp=natms_per_refcmp,
                                  natms_per_selcmp=natms_per_selcmp,
                                  dtype=np.uint32)
    hist_refcmp_diff_selcmp = contact_hist_refcmp_diff_selcmp(
        cm=cm,
        minlength=minlength,
        dtype=dtype
    )
    hist_refcmp_same_selcmp = contact_hist_refcmp_same_selcmp(
        cm=cm,
        minlength=minlength,
        dtype=dtype
    )
    hist_refcmp_selcmp_tot = contact_hist_refcmp_selcmp_tot(
        cm=cm,
        minlength=minlength,
        dtype=dtype
    )
    hist_refcmp_selcmp_pair = contact_hist_refcmp_selcmp_pair(
        cm=cm,
        minlength=minlength,
        dtype=dtype
    )

    length = max(len(hist_refcmp_diff_selcmp),
                 len(hist_refcmp_same_selcmp),
                 len(hist_refcmp_selcmp_tot),
                 len(hist_refcmp_selcmp_pair),
                 minlength)
    hist_refcmp_diff_selcmp = mdt.nph.extend(hist_refcmp_diff_selcmp,
                                             length)
    hist_refcmp_same_selcmp = mdt.nph.extend(hist_refcmp_same_selcmp,
                                             length)
    hist_refcmp_selcmp_tot = mdt.nph.extend(hist_refcmp_selcmp_tot,
                                            length)
    hist_refcmp_selcmp_pair = mdt.nph.extend(hist_refcmp_selcmp_pair,
                                             length)

    # Internal consistency check:
    if (np.issubdtype(cm_dtype, bool) and
        np.all(np.equal(natms_per_refcmp, 1)) and
            np.all(np.equal(natms_per_selcmp, 1))):
        # refcmp == refatm and selcmp == selatm
        if not np.array_equal(hist_refcmp_diff_selcmp,
                              hist_refcmp_selcmp_tot):
            raise ValueError("refcmp == refatm and selcmp == selatm, but"
                             "'hist_refcmp_diff_selcmp' !="
                             " 'hist_refcmp_selcmp_tot'")
        if not np.any(np.equal(cm.shape, 0)):
            hist_test = np.array([hist_refcmp_diff_selcmp[0],
                                  cm.shape[0] - hist_refcmp_diff_selcmp[0]])
            hist_test = mdt.nph.extend(hist_test, length)
            if not np.array_equal(hist_refcmp_same_selcmp, hist_test):
                raise ValueError("refcmp == refatm and selcmp == selatm,"
                                 " but 'hist_refcmp_same_selcmp' !="
                                 " [x, cm.shape[0]-x]")
            hist_test = np.array([0, np.count_nonzero(cm.astype(bool))])
            hist_test = mdt.nph.extend(hist_test, length)
            if not np.array_equal(hist_refcmp_selcmp_pair, hist_test):
                raise ValueError("refcmp == refatm and selcmp == selatm,"
                                 " but 'hist_refcmp_selcmp_pair' !="
                                 " [0, y]")

    return (hist_refcmp_diff_selcmp,
            hist_refcmp_same_selcmp,
            hist_refcmp_selcmp_tot,
            hist_refcmp_selcmp_pair)


def rmsd(
    refpos,
    selpos,
    weights=None,
    center=False,
    inplace=False,
    xyz=False,
    box=None,
):
    r"""
    Calculate the Root Mean Square Deviation (RMSD) between two sets of
    positions.

    .. todo::

        Implement a "superposition" functionality like in
        :func:`MDAnalysis.analysis.rms.rmsd`.

    Parameters
    ----------
    refpos, selpos : array_like
        Reference and candidate set of positions.  Position arrays must
        be of shape ``(3,)``, ``(n, 3)`` or ``(k, n, 3)`` where ``n`` is
        the number of particles and ``k`` is the number of frames.
        `refpos` and `selpos` must contain the same number of particles
        ``n``, but can have different numbers of frames ``k``.  If
        `refpos` and `selpos` do not have the same shape, they must be
        broadcastable to a common shape (which becomes the shape of the
        output).  The user is responsible to provide inputs that result
        in a physically meaningful broadcasting!
    weights : None or array_like, optional
        Array of shape ``(n,)`` containing the weight of each particle
        contained in the position arrays.  If `weights` is ``None``, all
        particles are assumed to have a weight equal to one.
    center : bool, optional
        If ``True``, shift the reference and candidate particles by
        their weighted center, respectively, before calculating the
        RMSD.
    inplace : bool, optional
        If ``True``, subtract the weighted center from the reference and
        candidate positions in place (i.e. the input arrays will be
        changed).  Note that `refpos` and `selpos` must have an
        appropriate dtype in this case.
    xyz : bool, optional
        If ``True``, return the x, y and z component of the RMSD
        separately instead of summing all components.  Note however,
        that in this case the square root is not taken, because you
        cannot extract the square root of each summand individually.
    box : None or array_like, optional
        The unit cell dimensions of the system, which must be orthogonal
        and provided in the same format as returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.  `box` can also be an
        array of boxes of shape ``(k, 6)``, where ``k`` must match the
        number of frames in `refpos`.  If given, the minimum image
        convention is taken into account when calculating the distance
        between the reference and candidate positions.  Works currently
        only for orthogonal boxes.

    Raises
    ------
    ValueError :
        If `weights` is not of shape ``(n,)`` or if it sums up to zero.

    See Also
    --------
    :class:`MDAnalysis.analysis.rms.RMSD` :
        Class to perform RMSD analysis on a trajectory
    :func:`MDAnalysis.analysis.rms.rmsd` :
        Calculate the RMSD between two coordinate sets

    Notes
    -----
    The root mean square deviation is calculated by

    .. math::

        RMSD = \sqrt{\frac{1}{N} \frac{1}{W} \sum_i^N w_i \
        \left( \mathbf{r}_i - \mathbf{r}_i^{ref} \right)^2}

    where :math:`N` is the number of particles,
    :math:`w_i` is the weight of the :math:`i`-th particle,
    :math:`W = \sum_i^N w_i` is the sum of particle weights,
    :math:`\mathbf{r}_i` are the candidate positions and
    :math:`\mathbf{r}_i^{ref}` are the reference positions.

    If `xyz` is ``True``, each component of the RMSD is returned
    individually without taking the square root.  For instance, the
    :math:`x`-component is

    .. math::

        \langle \Delta x^2 \rangle = \frac{1}{N} \frac{1}{W} \
        \sum_i^N w_i \left( x_i - x_i^{ref} \right)^2.

    Examples
    --------
    Shape of `refpos` and `selpos` is ``(3,)``:

    >>> refpos = np.array([ 3,  4,  5])
    >>> selpos = np.array([ 1,  7, -1])
    >>> selpos - refpos
    array([-2,  3, -6])
    >>> mdt.strc.rmsd(refpos, selpos)
    7.0
    >>> mdt.strc.rmsd(refpos, selpos, xyz=True)
    array([ 4.,  9., 36.])
    >>> box = np.array([5, 3, 3, 90, 90, 90])
    >>> mdt.strc.rmsd(refpos, selpos, xyz=True, box=box)
    array([4., 0., 0.])
    >>> mdt.strc.rmsd(refpos, selpos, box=box)
    2.0
    >>> mdt.strc.rmsd(refpos, selpos, weights=[3])
    7.0
    >>> mdt.strc.rmsd(refpos, selpos, weights=[3], center=True)
    0.0
    >>> mdt.strc.rmsd(refpos, selpos, center=True, inplace=True)
    0.0
    >>> refpos
    array([0, 0, 0])
    >>> selpos
    array([0, 0, 0])

    Shape of `refpos` and `selpos` is ``(n, 3)``:

    >>> refpos = np.array([[ 3.,  4.,  5.],
    ...                    [ 1., -1.,  1.]])
    >>> selpos = np.array([[ 1.,  7., -1.],
    ...                    [-1.,  0., -1.]])
    >>> selpos - refpos
    array([[-2.,  3., -6.],
           [-2.,  1., -2.]])
    >>> rmsd = mdt.strc.rmsd(refpos, selpos)
    >>> rmsd_expected = np.sqrt(0.5 * (49 + 9))
    >>> np.isclose(rmsd, rmsd_expected, rtol=0)
    True
    >>> mdt.strc.rmsd(refpos, selpos, xyz=True)
    array([ 4.,  5., 20.])
    >>> mdt.strc.rmsd(refpos, selpos, xyz=True, box=box)
    array([4. , 0.5, 0.5])
    >>> weights = np.array([0.575, 0.425])
    >>> rmsd = mdt.strc.rmsd(refpos, selpos, weights=weights)
    >>> np.isclose(rmsd, 4, rtol=0)
    True
    >>> mdt.strc.rmsd(refpos, selpos, weights=weights, center=True)
    1.5632498200863483
    >>> mdt.strc.rmsd(
    ...     refpos, selpos, xyz=True, center=True, inplace=True
    ... )
    array([0., 1., 4.])
    >>> refpos
    array([[ 1. ,  2.5,  2. ],
           [-1. , -2.5, -2. ]])
    >>> selpos
    array([[ 1. ,  3.5,  0. ],
           [-1. , -3.5,  0. ]])

    Shape of `refpos` and `selpos` is ``(k, n, 3)``:

    >>> refpos = np.array([[[ 3.,  4.,  5.],
    ...                     [ 1., -1.,  1.]],
    ...
    ...                    [[ 5.,  0.,  1.],
    ...                     [ 1.,  2., -3.]]])
    >>> selpos = np.array([[[ 1.,  7., -1.],
    ...                     [-1.,  0., -1.]],
    ...
    ...                    [[ 3.,  4.,  5.],
    ...                     [ 1., -1.,  1.]]])
    >>> selpos - refpos
    array([[[-2.,  3., -6.],
            [-2.,  1., -2.]],
    <BLANKLINE>
           [[-2.,  4.,  4.],
            [ 0., -3.,  4.]]])
    >>> rmsd = mdt.strc.rmsd(refpos, selpos)
    >>> rmsd_expected = np.sqrt(0.5 * np.array([49 + 9, 36 + 25]))
    >>> np.allclose(rmsd, rmsd_expected, rtol=0)
    True
    >>> mdt.strc.rmsd(refpos, selpos, xyz=True)
    array([[ 4. ,  5. , 20. ],
           [ 2. , 12.5, 16. ]])
    >>> mdt.strc.rmsd(refpos, selpos, xyz=True, box=box)
    array([[4. , 0.5, 0.5],
           [2. , 0.5, 1. ]])
    >>> box = np.array([[5, 3, 3, 90, 90, 90],
    ...                 [3, 5, 5, 90, 90, 90]])
    >>> mdt.strc.rmsd(refpos, selpos, xyz=True, box=box)
    array([[4. , 0.5, 0.5],
           [0.5, 2.5, 1. ]])
    >>> rmsd = mdt.strc.rmsd(refpos, selpos, weights=weights)
    >>> rmsd_expected = np.sqrt([16.    , 15.6625])
    >>> np.allclose(rmsd, rmsd_expected, rtol=0)
    True
    >>> mdt.strc.rmsd(refpos, selpos, weights=weights, center=True)
    array([1.56324982, 2.54478634])
    >>> mdt.strc.rmsd(
    ...    refpos, selpos, xyz=True, center=True, inplace=True
    ... )
    array([[ 0.  ,  1.  ,  4.  ],
           [ 1.  , 12.25,  0.  ]])
    >>> refpos
    array([[[ 1. ,  2.5,  2. ],
            [-1. , -2.5, -2. ]],
    <BLANKLINE>
           [[ 2. , -1. ,  2. ],
            [-2. ,  1. , -2. ]]])
    >>> selpos
    array([[[ 1. ,  3.5,  0. ],
            [-1. , -3.5,  0. ]],
    <BLANKLINE>
           [[ 1. ,  2.5,  2. ],
            [-1. , -2.5, -2. ]]])

    Shape of `refpos` is ``(n, 3)`` and shape of `selpos` is
    ``(k, n, 3)``:

    >>> refpos = np.array([[ 3.,  4.,  5.],
    ...                    [ 1., -1.,  1.]])
    >>> selpos = np.array([[[ 1.,  7., -1.],
    ...                     [-1.,  0., -1.]],
    ...
    ...                    [[ 3.,  4.,  5.],
    ...                     [ 1., -1.,  1.]]])
    >>> selpos - refpos
    array([[[-2.,  3., -6.],
            [-2.,  1., -2.]],
    <BLANKLINE>
           [[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]])
    >>> rmsd = mdt.strc.rmsd(refpos, selpos)
    >>> rmsd_expected = np.sqrt(0.5 * np.array([49 + 9, 0]))
    >>> np.allclose(rmsd, rmsd_expected, rtol=0)
    True
    >>> mdt.strc.rmsd(refpos, selpos, xyz=True)
    array([[ 4.,  5., 20.],
           [ 0.,  0.,  0.]])
    >>> box = np.array([5, 3, 3, 90, 90, 90])
    >>> mdt.strc.rmsd(refpos, selpos, xyz=True, box=box)
    array([[4. , 0.5, 0.5],
           [0. , 0. , 0. ]])
    >>> rmsd = mdt.strc.rmsd(refpos, selpos, weights=weights)
    >>> rmsd_expected = np.array([4., 0.])
    >>> np.allclose(rmsd, rmsd_expected, rtol=0)
    True
    >>> mdt.strc.rmsd(refpos, selpos, weights=weights, center=True)
    array([1.56324982, 0.        ])
    >>> mdt.strc.rmsd(
    ...    refpos, selpos, xyz=True, center=True, inplace=True
    ... )
    array([[0., 1., 4.],
           [0., 0., 0.]])
    >>> refpos
    array([[ 1. ,  2.5,  2. ],
           [-1. , -2.5, -2. ]])
    >>> selpos
    array([[[ 1. ,  3.5,  0. ],
            [-1. , -3.5,  0. ]],
    <BLANKLINE>
           [[ 1. ,  2.5,  2. ],
            [-1. , -2.5, -2. ]]])
    """
    refpos = mdt.check.pos_array(refpos)
    selpos = mdt.check.pos_array(selpos)
    if refpos.ndim == 1:
        n_particles = 1
    elif refpos.ndim in (2, 3):
        n_particles = refpos.shape[refpos.ndim - 2]
    else:
        # This else clause should never be entered, because this error
        # should already be raised by `mdt.check.pos_array(refpos)`.
        raise ValueError(
            "The shape of 'refpos' must be either (3,) or (n, 3) or (k, n, 3)"
            " but is {}.  This should not have happened".format(refpos.shape)
        )
    if (
        (selpos.ndim == 1 and n_particles != 1)
        or (selpos.ndim == 2 and selpos.shape[0] != n_particles)
        or (selpos.ndim == 3 and selpos.shape[1] != n_particles)
    ):
        raise ValueError(
            "'selpos' does not contain the same number of particles as"
            " 'refpos'.  The shape of 'selpos' is {}.  The shape of 'refpos'"
            " is {}".format(selpos.shape, refpos.shape)
        )

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n_particles,):
            raise ValueError(
                "'weights' must have shape {} but has shape"
                " {}".format((n_particles,), weights.shape)
            )
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            raise ValueError("'weights' must not sum up to zero")
        weights /= weights_sum

    if box is not None:
        box = mdt.check.box(box)
        if box.ndim == 2:
            if refpos.ndim != 3:
                raise ValueError(
                    "box dimensions given for {} frames, but reference"
                    " positions given for only 1 frame".format(box.shape[0])
                )
            if box.shape[0] != refpos.shape[0]:
                raise ValueError(
                    "If 'box' has 2 dimensions, box.shape[0] ({}) must match"
                    " refpos.shape[0]"
                    " ({})".format(box.shape[0], refpos.shape[0])
                )

    if center:
        refcenter = mdt.strc.wcenter_pos(pos=refpos, weights=weights, box=box)
        if refpos.ndim == 3:
            refcenter = np.expand_dims(refcenter, axis=1)
        selcenter = mdt.strc.wcenter_pos(pos=selpos, weights=weights, box=box)
        if selpos.ndim == 3:
            selcenter = np.expand_dims(selcenter, axis=1)
        if inplace:
            refpos -= refcenter
            selpos -= selcenter
        else:
            refpos = refpos - refcenter
            selpos = selpos - selcenter

    rmsd = mdt.box.vdist(selpos, refpos, box=box)
    rmsd **= 2
    # The dimension of position arrays and thus of `rmsd` can be either
    # 1, 2 or 3 (see `mdtools.check.pos_array`).  If `component` is
    # ``True`` and `ndim` is
    #     * 1 (single particle), the RMSD is simply the distance between
    #       the reference and candidate particle.
    #     * 2 (multiple particles), the RMSD is the sum over all
    #       reference-candidate distances (``axis=0``).
    #     * 3 (multiple frames), the RMSD is the sum over all
    #       reference-candidate distances in each frame (``axis=1``).
    # If `component` is ``False``, the x, y and z component must be
    # summed up.
    if weights is not None and rmsd.ndim > 1:
        rmsd *= np.expand_dims(weights, axis=1)
    if rmsd.ndim == 1:
        if n_particles != 1:
            raise ValueError(
                "The shape of 'rmsd' ({}) does not match the number of"
                " particles ({}).  You might want to check the shape of"
                " 'refpos' ({}) and 'selpos' ({}) and which shape they"
                " broadcast to".format(
                    rmsd.shape, n_particles, refpos.shape, selpos.shape
                )
            )
        if not xyz:
            rmsd = np.sum(rmsd)
    elif rmsd.ndim == 2:
        if rmsd.shape[rmsd.ndim - 2] != n_particles:
            raise ValueError(
                "The shape of 'rmsd' ({}) does not match the number of"
                " particles ({}).  You might want to check the shape of"
                " 'refpos' ({}) and 'selpos' ({}) and which shape they"
                " broadcast to".format(
                    rmsd.shape, n_particles, refpos.shape, selpos.shape
                )
            )
        if xyz:
            rmsd = np.sum(rmsd, axis=rmsd.ndim - 2)
        else:
            rmsd = np.sum(rmsd)
    elif rmsd.ndim == 3:
        if rmsd.shape[rmsd.ndim - 2] != n_particles:
            raise ValueError(
                "The shape of 'rmsd' ({}) does not match the number of"
                " particles ({}).  You might want to check the shape of"
                " 'refpos' ({}) and 'selpos' ({}) and which shape they"
                " broadcast to".format(
                    rmsd.shape, n_particles, refpos.shape, selpos.shape
                )
            )
        rmsd = np.sum(rmsd, axis=rmsd.ndim - 2)
        if not xyz:
            rmsd = np.sum(rmsd, axis=1)
    else:
        raise ValueError(
            "The shape of 'rmsd' must be either (3,) or (n, 3) or (k, n, 3)"
            " but is {}.  This should not have happened.  You might want to"
            " check the shape of 'refpos' ({}) and 'selpos' ({}) and which"
            " shape they broadcast to".format(
                rmsd.shape, refpos.shape, selpos.shape
            )
        )
    rmsd /= n_particles
    if not xyz:
        rmsd = np.sqrt(rmsd)
    return rmsd
