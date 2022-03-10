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


"""Functions related to simulation boxes and boundary conditions."""


# Standard libraries
from datetime import datetime

# Third party libraries
import psutil
import numpy as np
import MDAnalysis as mda
import MDAnalysis.lib.distances as mdadist
import MDAnalysis.lib.mdamath as mdamath

# Local application/library specific imports
import mdtools as mdt


def box_volume(box, debug=False):
    """
    Calculate the volume of a (triclinic) box.

    The box can be triclinic or orthogonal.

    Parameters
    ----------
    box : array_like
        Box for which to calculate the volume.  It must provided in the
        same format as returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.
    debug : bool, optional
        If ``True``, check the input arguments.

        .. deprecated:: 0.0.0.dev0
            This argument is without functionality and will be removed
            in a future release.

    Returns
    -------
    volume : scalar
        The volume of the box.

    Raises
    ------
    ValueError
        If the volume of the box is equal to or less than zero.

    See Also
    --------
    :func:`volume` :
        Calculate the volume of (multiple) orthogonal box(es)
    :func:`~MDAnalysis.lib.mdamath.box_volume` :
        Calculate the volume of a single (triclinic) box

    Notes
    -----
    This function is just a wrapper around
    :func:`~MDAnalysis.lib.mdamath.box_volume` that raises an exception
    if the calculated volume is equal to or less than zero.
    """
    volume = mdamath.box_volume(box)
    if volume <= 0:
        mdt.check.box(box=box, with_angles=True, dim=1)
        raise ValueError(
            "The box volume ({}) is equal to or less than zero".format(volume)
        )
    return volume


def volume(box, debug=False, **kwargs):
    """
    Calculate the volume of orthogonal boxes.

    Parameters
    ----------
    box : array_like
        The box(es) for which to calculate the volume.  Must be
        orthogonal and provided as ``[lx, ly, lz]`` (shape ``(3,)``) or
        as an array of ``[lx, ly, lz]`` (shape ``(k, 3)``, e.g. one box
        for each trajectory frame).  The box(es) can also contain the
        angles like in the format returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.  In this case, it is
        checked if the box(es) is (are) orthogonal before calculating
        the volume(s).
    debug : bool, optional
        If ``True``, check the input arguments.

        .. deprecated:: 0.0.0.dev0
            This argument is without functionality and will be removed
            in a future release.

    **kwargs : dict, optional
        Additional keyword arguments to parse to :func:`numpy.prod`.
        See there for possible choices.  Note that the keyword `axis`
        will be ignored.

    Returns
    -------
    volume : scalar or numpy.ndarray
        The volume of the box.  If `box` was provided as an array of
        boxes, `volume` will be an array containing the volume for each
        box.

    See Also
    --------
    :func:`box_volume` :
       Calculate the volume of a single (triclinic) box
    :func:`~MDAnalysis.lib.mdamath.box_volume` :
        Calculate the volume of a single (triclinic) box
    :func:`numpy.prod` :
        Calculate the product of array elements over a given axis

    Notes
    -----
    This function calculates the box volume by multiplying all box
    lengths using :func:`numpy.prod`.
    """
    mdt.check.box(box=box, orthorhombic=True)
    kwargs.pop("axis", None)
    return np.prod(box, axis=-1, **kwargs)


def diagonal(box, dtype=None, debug=False):
    """
    Calculate the length of the diagonal of an orthogonal box.

    Parameters
    ----------
    box : array_like
        The box(es) for which to calculate the diagonal length.  Must be
        orthogonal and provided as ``[lx, ly, lz]`` (shape ``(3,)``) or
        as an array of ``[lx, ly, lz]`` (shape ``(k, 3)``, e.g. one box
        for each trajectory frame).  The box(es) can also contain the
        angles like in the format returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.  In this case, it is
        checked if the box(es) is (are) orthogonal before calculating
        the diagonal length(s).
    dtype : type, optional
        The data type of the returned array.  If `dtype` is not given
        (default), infer the data type from `box`.
    debug : bool, optional
        If ``True``, check the input arguments. Default: ``False``

        .. deprecated:: 0.0.0.dev0
            This argument is without functionality and will be removed
            in a future release.

    Returns
    -------
    diagonal : scalar or numpy.ndarray
        The length of the diagonal of the box.  If `box` was provided as
        an array of boxes, `diagonal` will be an array containing the
        length of the diagonal for each box.
    """
    mdt.check.box(box=box, orthorhombic=True)
    np.sum(np.square(box, dtype=dtype), axis=-1)
    return np.sqrt()


def wrap_pos(pos, box, mda_backend=None):
    """
    Shift all particle positions into the primary unit cell.

    Parameters
    ----------
    pos : array_like
        The position array which to move into the primary unit cell.
        Must be either of shape ``(3,)``, ``(n, 3)`` or ``(k, n, 3)``,
        where ``n`` is the number of particles and ``k`` is the number
        of frames.
    box : array_like, optional
        The unit cell dimensions of the system, which can be orthogonal
        or triclinic and must be provided in the same format as returned
        by :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.  `box` can also be an
        array of boxes of shape ``(k, 6)`` (one box for each frame).  In
        this case a 2-dimensional position array is interpreted to
        contain one position per frame instead of ``n`` positions for
        one frame.
    mda_backend : {None, 'serial', 'OpenMP'}, optional
        The backend to parse to
        :func:`MDAnalysis.lib.distances.apply_PBC`.  If ``None``, it
        will be set to ``'OpenMP'`` if more than one CPU is available
        (as determined by :func:`mdtools.run_time_info.get_num_CPUs`) or
        to ``'serial'`` if only one CPU is available.

    Returns
    -------
    pos_wrapped : numpy.ndarray
        Array of the same shape as `pos` containing the positions that
        all lie within the primary unit cell as defined by `box`.

    See Also
    --------
    :func:`MDAnalysis.lib.distances.apply_PBC` :
        Moves coordinates into the primary unit cell
    :func:`mdtools.box.wrap` :
        Shift compounds of an MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup` back into the primary
        unit cell

    Notes
    -----
    This function uses :func:`MDAnalysis.lib.distances.apply_PBC` to
    wrap the input positions into the primary unit cell.  But in
    contrast to :func:`MDAnalysis.lib.distances.apply_PBC`, this
    function also accepts position arrays of shape ``(k, n, 3)`` and box
    arrays of shape ``(k, 6)``.

    Examples
    --------
    >>> pos = np.array([0, 3, 6])
    >>> mdt.box.wrap_pos(pos, [4, 5, 6, 90, 90, 90])
    array([0., 3., 0.], dtype=float32)

    >>> pos = np.array([[0, 3, 6],
    ...                 [1, 5, 7]])
    >>> mdt.box.wrap_pos(pos, [4, 5, 6, 90, 90, 90])
    array([[0., 3., 0.],
           [1., 0., 1.]], dtype=float32)
    >>> box = np.array([[4, 5, 6, 90, 90, 90],
    ...                 [6, 5, 4, 90, 90, 90]])
    >>> mdt.box.wrap_pos(pos, box)
    array([[0., 3., 0.],
           [1., 0., 3.]], dtype=float32)

    >>> pos = np.array([[[0, 3, 6],
    ...                  [1, 5, 7]],
    ...
    ...                 [[2, 6, 10],
    ...                  [3, 7, 11]]])
    >>> mdt.box.wrap_pos(pos, [4, 5, 6, 90, 90, 90])
    array([[[0., 3., 0.],
            [1., 0., 1.]],
    <BLANKLINE>
           [[2., 1., 4.],
            [3., 2., 5.]]], dtype=float32)
    >>> mdt.box.wrap_pos(pos, box)
    array([[[0., 3., 0.],
            [1., 0., 1.]],
    <BLANKLINE>
           [[2., 1., 2.],
            [3., 2., 3.]]], dtype=float32)
    """
    pos = mdt.check.pos_array(pos)
    box = mdt.check.box(box)
    if mda_backend is None:
        if mdt.rti.get_num_CPUs() > 1:
            mda_backend = "OpenMP"
        else:
            mda_backend = "serial"

    if box.ndim == 1:
        if pos.ndim in (1, 2):
            pos_wrapped = mdadist.apply_PBC(
                coords=pos, box=box, backend=mda_backend
            )
        elif pos.ndim == 3:
            # `MDAnalysis.lib.distances.apply_PBC` returns array of
            # dtype numpy.float32
            pos_wrapped = np.full_like(pos, np.nan, dtype=np.float32)
            for i, pos_frame in enumerate(pos):
                pos_wrapped[i] = mdadist.apply_PBC(
                    coords=pos_frame, box=box, backend=mda_backend
                )
        else:
            # This else clause should never be entered, because this
            # error should already be raised by
            # `mdt.check.pos_array(pos)`.
            raise ValueError(
                "'pos' has shape {} but must have shape (3,) or (n, 3) or"
                " (k, n, 3).  This should not have happened".format(pos.shape)
            )
    elif box.ndim == 2:
        if pos.ndim not in (2, 3):
            raise ValueError(
                "If 'box' has shape (k, 6), 'pos' must have shape (k, n, 3) or"
                " (k, 3), but has shape {}".format(pos.shape)
            )
        if pos.shape[0] != box.shape[0]:
            raise ValueError(
                "If 'box' has 2 dimensions, pos.shape[0] ({}) must"
                " match box.shape[0] ({})".format(pos.shape[0], box.shape[0])
            )
        # `MDAnalysis.lib.distances.apply_PBC` returns array of dtype
        # numpy.float32
        pos_wrapped = np.full_like(pos, np.nan, dtype=np.float32)
        for i, pos_frame in enumerate(pos):
            pos_wrapped[i] = mdadist.apply_PBC(
                coords=pos_frame, box=box[i], backend=mda_backend
            )
    else:
        # This else clause should never be entered, because this error
        # should already be raised by `mdt.check.box(box)`
        raise ValueError(
            "'box' has shape {} but must have shape (3,) or (k, 3).  This"
            " should not have happened".format(box.shape)
        )
    return pos_wrapped


def wrap(
    ag, compound="atoms", center="cog", box=None, inplace=True, debug=False
):
    """
    Shift compounds of an MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup` back into the primary
    unit cell.

    .. todo::
        Include a `make_whole` argument that makes compounds whole after
        the wrapping, even if this implies that some
        :class:`Atoms <MDAnalysis.core.groups.Atom>` might lie outside
        the primary unit cell again.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup instance
        The MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` whose
        compounds shall be wrapped back into the primary unit cell.
    compound : {'atoms', 'group', 'segments', 'residues', 'molecules', \
        'fragments'}, optional
        Which type of compound to keep together during wrapping.
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
        same compound are shifted by the same translation vector such
        that the `center` of the compound lies within the box.  This
        might however mean that not all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` of the compound
        will be inside the unit cell after wrapping.  Default is
        ``'atoms'``, which means that each
        :class:`~MDAnalysis.core.groups.Atom` of `ag` is shifted by its
        individual translation vector so that finally all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` of `ag` will be
        inside the unit cell.  Refer to the MDAnalysis' user guide
        for an |explanation_of_these_terms|.  Note that in any case,
        even if `compound` is e.g. ``'residues'``, only the
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to `ag`
        are taken into account, even if the compound might comprise
        additional :class:`Atoms <MDAnalysis.core.groups.Atom>` that are
        not contained in `ag`.  Also note that broken compounds are
        note made whole by this function.
    center : {'cog', 'com'}, optional
        How to define the center of a compound.  Parse ``'cog'`` for
        center of geometry or ``'com'`` for center of mass.  If compound
        is ``'atoms'``, this parameter is meaningless and therefore
        ignored.
    box : array_like, optional
        The unit cell dimensions of the system, which can be orthogonal
        or triclinic and must be provided in the same format as returned
        by :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.  If ``None``, the
        :attr.`~MDAnalysis.coordinates.base.Timestep.dimensions` of the
        current :class.`~MDAnalysis.coordinates.base.Timestep` will be
        used.
    inplace : bool, optional
        If ``True``, coordinates are modified in place.
    debug : bool, optional
        If ``True``, run in debug mode.

    Returns
    -------
    wrapped_pos : numpy.ndarray
        Array of wrapped :class:`~MDAnalysis.core.groups.Atom`
        coordinates of dtype `numpy.float32` and shape
        ``(ag.n_atoms, 3)``.

    See Also
    --------
    :meth:`MDAnalysis.core.groups.AtomGroup.wrap` :
        Shift compounds of the
        :class:`~MDAnalysis.core.groups.AtomGroup` back into the primary
        unit cell
    :meth:`MDAnalysis.core.groups.AtomGroup.pack_into_box`
        Shift all :class:`Atoms <MDAnalysis.core.groups.Atom>` in the
        :class:`~MDAnalysis.core.groups.AtomGroup` into the primary unit
        cell
    :func:`MDAnalysis.lib.distances.apply_PBC` :
        Shift coordinates stored as :class:`numpy.ndarray` into the
        primary unit cell
    :func:`make_whole` :
        Make compounds of a MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup` whole

    Notes
    -----
    This function is just a wrapper around
    the :meth:`~MDAnalysis.core.groups.AtomGroup.wrap` method of the
    input :class:`~MDAnalysis.core.groups.AtomGroup` that additionally
    checks the masses of all
    :class:`Atoms <MDAnalysis.core.groups.Atom>` in `ag` when `center`
    is set to ``'com'`` and `compound` is not ``'atoms'``.  See
    :func:`mdtools.check.masses_new` for further information.

    When calling this function with `compounds` set to something
    different than ``'atoms'``, make sure that the respective compounds
    are whole.  Broken compounds will not be made whole by this
    function.  See :func:`make_whole` for a possibility to make
    compounds whole that are split across periodic boundaries.

    Because :mod:`MDAnalysis` will pull
    :attr:`~MDAnalysis.core.universe.Universe.trajectory` data directly
    from the file it is reading from, changes to
    :class:`~MDAnalysis.core.groups.Atom` coordinates and box
    :attr.`~MDAnalysis.coordinates.base.Timestep.dimensions` will not
    persist once the :attr:`~MDAnalysis.coordinates.base.Timestep.frame`
    is changed (even if `inplace` is ``True``).  The only way to make
    these changes permanent is to load the
    :attr:`~MDAnalysis.core.universe.Universe.trajectory` into memory,
    or to write a new
    :attr:`~MDAnalysis.core.universe.Universe.trajectory` to file for
    every :attr:`~MDAnalysis.coordinates.base.Timestep.frame`.  See the
    MDAnalysis user guide about trajectories_ (second last
    paragraph) and `in-memory trajectories`_.

    .. _trajectories:
        https://userguide.mdanalysis.org/stable/trajectories/trajectories.html
    .. _in-memory trajectories:
        https://userguide.mdanalysis.org/stable/reading_and_writing.html#in-memory-trajectories
    """
    if ag.n_atoms == 1:
        # => 'com' and 'cog' are the same, but because the mass might be
        # zero, it is safer to use 'cog'
        center = "cog"
    elif center == "com" and compound != "atoms":  # and ag.n_atoms > 1
        mdt.check.masses_new(ag=ag, verbose=debug)
    return ag.wrap(compound=compound, center=center, box=box, inplace=inplace)


def make_whole(
    ag, compound="fragments", reference="cog", inplace=True, debug=False
):
    """
    Make compounds of a MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup` whole that are split
    across periodic boundaries.

    Note that all :class:`Atoms <MDAnalysis.core.groups.Atom>` of the
    input :class:`~MDAnalysis.core.groups.AtomGroup` are wrapped back
    into the primary unit cell before making compounds whole.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup instance
        The MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` whose
        compounds shall be made whole.
    compound : {'group', 'segments', 'residues', 'molecules', \
        'fragments'}, optional
        Which type of component to make whole.  Note that in any case,
        all :class:`Atoms <MDAnalysis.core.groups.Atom>` within each
        compound must be interconnected by bonds, i.e. compounds must
        correspond to (parts of) molecules.  This is always fulfilled
        for :attr:`~MDAnalysis.core.groups.AtomGroup.fragments`, because
        a MDAnalysis
        :attr:`fragment <MDAnalysis.core.groups.AtomGroup.fragments>` is
        what is typically considered a molecule, i.e. an
        :class:`~MDAnalysis.core.groups.AtomGroup` where any
        :class:`~MDAnalysis.core.groups.Atom` is reachable from any
        other :class:`~MDAnalysis.core.groups.Atom` in the
        :class:`~MDAnalysis.core.groups.AtomGroup` by traversing bonds,
        and none of its :class:`Atoms <MDAnalysis.core.groups.Atom>` is
        bonded to any :class:`~MDAnalysis.core.groups.Atom` outside the
        :class:`~MDAnalysis.core.groups.AtomGroup`.  A 'molecule' in
        MDAnalysis refers to a GROMACS-specific concept.  See the
        `MDAnalysis user guide`_ for further information.  Note that in
        any case, even if `compound` is e.g. ``'residues'``, only the
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to `ag`
        are taken into account, even if the compound might comprise
        additional :class:`Atoms <MDAnalysis.core.groups.Atom>` that are
        not contained in `ag`..  If you want entire molecules to be made
        whole, make sure that all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` of these molecules
        are part of `ag`.

        .. _MDAnalysis user guide:
            https://userguide.mdanalysis.org/stable/groups_of_atoms.html

    reference : {'cog', 'com', None}, optional
        If 'cog' (center of geometry) or 'com' (center of mass), the
        compounds that were made whole will be shifted such that their
        individual reference point lies within the primary unit cell.
        If ``None``, no such shift is performed.
    inplace : bool, optional
        If ``True``, coordinates are modified in place.
    debug : bool, optional
        If ``True``, check the input arguments. Default: ``False``

    Returns
    -------
    pos : numpy.ndarray
        Array of shape ``(ag.n_atoms, 3)`` containing the coordinates of
        all :class:`Atoms <MDAnalysis.core.groups.Atom>` in `ag` after
        the specifed compouds of `ag` were made whole.

    See Also
    --------
    :meth:`~MDAnalysis.core.groups.AtomGroup.unwrap` :
        Move :class:`Atoms <MDAnalysis.core.groups.Atom>` in the
        :class:`~MDAnalysis.core.groups.AtomGroup` such that bonds
        within the group's compounds aren't split across periodic
        boundaries
    :func:`~MDAnalysis.lib.mdamath.make_whole` :
        Move all :class:`Atoms <MDAnalysis.core.groups.Atom>` in a
        MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` such that
        bonds don't split over periodic images
    :func:`wrap` :
        Shift compounds of a MDAnalysis
        :class:`~MDAnalysis.core.groups.AtomGroup` back into the primary
        unit cell`

    Notes
    -----
    This function is just a wrapper around the
    :meth:`~MDAnalysis.core.groups.AtomGroup.unwrap` method of the
    input :class:`~MDAnalysis.core.groups.AtomGroup` that additionally
    checks the masses of all
    :class:`Atoms <MDAnalysis.core.groups.Atom>` in `ag` when
    `reference` is set to ``'com'``.  See
    :func:`mdtools.check.masses_new` for further information.

    Before making any compound whole, all
    :class:`Atoms <MDAnalysis.core.groups.Atom>` in `ag` are wrapped
    back into the primary unit cell.  This is done to make sure that the
    unwrap algorithm (better "make whole" algorithm) of
    :meth:`~MDAnalysis.core.groups.AtomGroup.unwrap` is working
    properly.  This means that making compounds whole in an unwrapped
    :attr:`~MDAnalysis.core.universe.Universe.trajectory` while keeping
    the :attr:`~MDAnalysis.core.universe.Universe.trajectory` unwrapped
    is not possible with this function.

    .. todo::

        Check if it is really necessary to wrap all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` back into the
        primary unit before calling
        :meth:`~MDAnalysis.core.groups.AtomGroup.unwrap`.  This is a
        serious problem, because it implies an inplace change of the
        :class:`~MDAnalysis.core.groups.Atom` coordinates.

    Because :mod:`MDAnalysis` will pull
    :attr:`~MDAnalysis.core.universe.Universe.trajectory` data directly
    from the file it is reading from, changes to
    :class:`~MDAnalysis.core.groups.Atom` coordinates and box
    :attr.`~MDAnalysis.coordinates.base.Timestep.dimensions` will not
    persist once the :attr:`~MDAnalysis.coordinates.base.Timestep.frame`
    is changed (even if `inplace` is ``True``).  The only way to make
    these changes permanent is to load the
    :attr:`~MDAnalysis.core.universe.Universe.trajectory` into memory,
    or to write a new
    :attr:`~MDAnalysis.core.universe.Universe.trajectory` to file for
    every :attr:`~MDAnalysis.coordinates.base.Timestep.frame`.  See the
    MDAnalysis user guide about trajectories_ (second last
    paragraph) and `in-memory trajectories`_.

    .. _trajectories:
        https://userguide.mdanalysis.org/stable/trajectories/trajectories.html
    .. _in-memory trajectories:
        https://userguide.mdanalysis.org/stable/reading_and_writing.html#in-memory-trajectories
    """
    if ag.n_atoms == 1:
        # => 'com' and 'cog' are the same, but because the mass might be
        # zero, it is safer to use 'cog'
        reference = "cog"
    elif reference == "com":  # and ag.n_atoms > 1
        mdt.check.masses_new(ag=ag, verbose=debug)
    wrap(
        ag=ag,
        compound="atoms",
        center="cog",
        box=None,
        inplace=True,
        debug=debug,
    )
    return ag.unwrap(
        compound=compound, reference=reference, inplace=inplace, debug=debug
    )


def vdist(pos1, pos2, box=None, out=None):
    """
    Calculate the distance vectors between two position arrays.

    Parameters
    ----------
    pos1, pos2 : array_like
        The two position arrays between which to calculate the distance
        vectors.  Position arrays must be of shape ``(3,)``, ``(n, 3)``
        or ``(k, n, 3)`` where ``n`` is the number of particles and
        ``k`` is the number of frames.  If `pos1` and `pos2` do not have
        the same shape, they must be broadcastable to a common shape
        (which becomes the shape of the output).  The user is
        responsible to provide inputs that result in a physically
        meaningful broadcasting!
    box : None or array_like, optional
        The unit cell dimensions of the system, which must be orthogonal
        and provided in the same format as returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.  If supplied, the minimum
        image convention is taken into account.  Works currently only
        for orthogonal boxes.  `box` can also be an array of boxes of
        shape ``(k, 6)`` (one box for each frame).  In this case a
        2-dimensional position array is interpreted to contain one
        position per frame instead of ``n`` positions for one frame.
    out : None or numpy.ndarray, optional
        Preallocated array into which the result is stored.  If
        provided, it must have a shape that the inputs broadcast to.  If
        not provided, a freshly-allocated array is returned.

    Returns
    -------
    dist_vecs : numpy.ndarray
        Distance array containing the result of ``pos1 - pos2``.

    See Also
    --------
    :func:`MDAnalysis.lib.distances.distance_array` :
        Calculate all possible distances between a reference set and
        another configuration
    :func:`MDAnalysis.lib.distances.self_distance_array` :
        Calculate all possible distances within a configuration
    :func:`MDAnalysis.lib.distances.calc_bonds` :
        Calculate the bond lengths between pairs of atom positions from
        the two coordinate arrays
    :func:`MDAnalysis.analysis.distances.dist` :
        Calculate the distance between atoms in two MDAnalysis
        :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`

    Notes
    -----
    If your are only interested in the distances itself (i.e. the norm
    of the distance vectors), consider using
    :func:`MDAnalysis.analysis.distances.dist` or
    :func:`MDAnalysis.lib.distances.calc_bonds` instead.

    If `box` is provided, the minimum image convention is taken into
    account using algorithm C4 from Deiters [#]_
    (``dist_vecs -= numpy.floor(dist_vecs / box + 0.5) * box``).  This
    algorithm also works if the particle distances are an arbitrary
    multiple of the box length.

    References
    ----------
    .. [#] U. K. Deiters, `"Efficient Coding of the Minimum Image
            Convention" <https://doi.org/10.1524/zpch.2013.0311>`_,
            Zeitschrift für Physikalische Chemie, 2013, 227, 345-352.

    Examples
    --------
    Shape of `pos1` and `pos2` is ``(3,)``:

    >>> pos1 = np.array([0, 2, 4])
    >>> pos2 = np.array([5, 3, 1])
    >>> mdt.box.vdist(pos1, pos2)
    array([-5., -1.,  3.])
    >>> box = np.array([3, 2, 2, 90, 90, 90])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([ 1., -1., -1.])

    Shape of `pos1` and `pos2` is ``(n, 3)``:

    >>> pos1 = np.array([[0, 2, 4],
    ...                  [5, 3, 1]])
    >>> pos2 = np.array([[5, 3, 1],
    ...                  [0, 2, 4]])
    >>> mdt.box.vdist(pos1, pos2)
    array([[-5., -1.,  3.],
           [ 5.,  1., -3.]])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[ 1., -1., -1.],
           [-1., -1., -1.]])
    >>> box = np.array([[3, 2, 2, 90, 90, 90],
    ...                 [2, 3, 4, 90, 90, 90]])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[ 1., -1., -1.],
           [-1.,  1.,  1.]])

    Shape of `pos1` and `pos2` is ``(k, n, 3)``:

    >>> pos1 = np.array([[[0, 2, 4],
    ...                   [5, 3, 1]],
    ...
    ...                  [[4, 0, 2],
    ...                   [5, 3, 1]]])
    >>> pos2 = np.array([[[5, 3, 1],
    ...                   [0, 2, 4]],
    ...
    ...                  [[5, 3, 1],
    ...                   [4, 0, 2]]])
    >>> mdt.box.vdist(pos1, pos2)
    array([[[-5., -1.,  3.],
            [ 5.,  1., -3.]],
    <BLANKLINE>
           [[-1., -3.,  1.],
            [ 1.,  3., -1.]]])
    >>> box = np.array([3, 2, 2, 90, 90, 90])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [ 1., -1., -1.]]])
    >>> box = np.array([[3, 2, 2, 90, 90, 90],
    ...                 [2, 3, 4, 90, 90, 90]])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[-1.,  0.,  1.],
            [-1.,  0., -1.]]])

    Shape of `pos1` is ``(3,)`` and shape of `pos2` is ``(n, 3)``:

    >>> pos1 = np.array([0, 2, 4])
    >>> pos2 = np.array([[5, 3, 1],
    ...                  [0, 2, 4]])
    >>> mdt.box.vdist(pos1, pos2)
    array([[-5., -1.,  3.],
           [ 0.,  0.,  0.]])
    >>> box = np.array([3, 2, 2, 90, 90, 90])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[ 1., -1., -1.],
           [ 0.,  0.,  0.]])
    >>> box = np.array([[3, 2, 2, 90, 90, 90],
    ...                 [2, 3, 4, 90, 90, 90]])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[ 1., -1., -1.],
           [ 0.,  0.,  0.]])

    Shape of `pos1` is ``(3,)`` and shape of `pos2` is ``(k, n, 3)``:

    >>> pos1 = np.array([0, 2, 4])
    >>> pos2 = np.array([[[5, 3, 1],
    ...                   [0, 2, 4]],
    ...
    ...                  [[5, 3, 1],
    ...                   [4, 0, 2]]])
    >>> mdt.box.vdist(pos1, pos2)
    array([[[-5., -1.,  3.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[-5., -1.,  3.],
            [-4.,  2.,  2.]]])
    >>> box = np.array([3, 2, 2, 90, 90, 90])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[ 1., -1., -1.],
            [-1.,  0.,  0.]]])
    >>> box = np.array([[3, 2, 2, 90, 90, 90],
    ...                 [2, 3, 4, 90, 90, 90]])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [ 0., -1., -2.]]])

    Shape of `pos1` is ``(n, 3)`` and shape of `pos2` is ``(k, n, 3)``:

    >>> pos1 = np.array([[0, 2, 4],
    ...                  [5, 3, 1]])
    >>> pos2 = np.array([[[5, 3, 1],
    ...                   [0, 2, 4]],
    ...
    ...                  [[5, 3, 1],
    ...                   [4, 0, 2]]])
    >>> mdt.box.vdist(pos1, pos2)
    array([[[-5., -1.,  3.],
            [ 5.,  1., -3.]],
    <BLANKLINE>
           [[-5., -1.,  3.],
            [ 1.,  3., -1.]]])
    >>> box = np.array([3, 2, 2, 90, 90, 90])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[ 1., -1., -1.],
            [ 1., -1., -1.]]])
    >>> box = np.array([[3, 2, 2, 90, 90, 90],
    ...                 [2, 3, 4, 90, 90, 90]])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [-1.,  0., -1.]]])
    """
    pos1 = mdt.check.pos_array(pos1)
    pos2 = mdt.check.pos_array(pos2)
    dist_vecs = np.subtract(pos1, pos2, out=out, dtype=np.float64)
    if box is not None:
        box = mdt.check.box(box, with_angles=True, orthorhombic=True)
        if box.ndim == 1:
            box = box[:3]
        elif box.ndim == 2:
            box = box[:, :3]
            if dist_vecs.ndim == 3:
                box = np.expand_dims(box, axis=1)
            elif dist_vecs.ndim != 2:
                raise ValueError(
                    "If 'box' has shape (k, 6), 'dist_vecs' must have shape"
                    " (k, n, 3) or (k, 3), but has shape"
                    " {}".format(dist_vecs.shape)
                )
        else:
            # This else clause should never be entered, because this
            # error should already be raised by `mdt.check.box(box)`.
            raise ValueError(
                "'box' must have shape (6,) or (k, 6), but has shape"
                " {}".format(box.shape)
            )
        dist_vecs -= np.floor(dist_vecs / box + 0.5) * box
    return dist_vecs


def unwrap(
    atm_grp,
    coord_unwrapped_prev,
    displacement=None,
    inplace=False,
    debug=False,
):
    """
    Unwrap the atoms of a MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup` out of the primary unit
    cell, i.e. calculate their coordinates in real space.

    This function uses the algorithm proposed by von Bülow et al. in
    J. Chem. Phys., 2020, 153, 021101. Basically it calculates the atom
    displacements from frame to frame and adds these displacements to
    the previous atom positions to build the unwraped trajectory.

    The main difference to :func:`unwrap_trj` is that :func:`unwrap_trj`
    unwraps the complete trajectory while this function only unwraps a
    single frame based on the unwrapped coordinates of the previous
    frame.

    Note
    ----
    If you want to change the
    :attr:`~MDAnalysis.core.groups.AtomGroup.positions` attribute of the
    :class:`~MDAnalysis.core.groups.AtomGroup` to the outcome of this
    function, keep in mind that changing atom positions is not reflected
    in any file; reading any frame from the
    :attr:`~MDAnalysis.core.universe.Universe.trajectory` will replace
    the change with that from the file except if the
    :attr:`~MDAnalysis.core.universe.Universe.trajectory` is held in
    memory, e.g., when the
    :meth:`~MDAnalysis.core.universe.Universe.transfer_to_memory`
    method was used.

    Parameters
    ----------
    atm_grp : MDAnalysis.core.groups.AtomGroup
        The :class:`~MDAnalysis.core.groups.AtomGroup` whose atoms
        should be unwrapped into real space.
    coord_unwrapped_prev : array_like
        Array of the same shape as ``atm_grp.positions.shape``
        containing the unwrapped coordinates of `atm_grp` from the
        previous frame. When starting the unwrapping,
        `coord_unwrapped_prev` should be the wrapped coordinates of
        `atm_grp` from the very first frame and `atm_grp` should be
        parsed from the second frame so that ``atm_grp.positions``
        contains the atom coordinates of `atm_grp` from the second
        frame.  For all further frames, `coord_unwrapped_prev` should be
        the result of this function and `atm_grp` should be parsed from
        next frame.
    displacement : numpy.ndarray, optional
        Preallocated temporary array of the same shape as
        ``atm_grp.positions.shape`` and dtype ``numpy.float64``. If
        provided, this array is used to store the displacement vectors
        from the previous (`coord_unwrapped_prev`) to the current
        (``atm_grp.positions``) frame. Avoids creating the array which
        saves time when the function is called repeatedly and
        additionally the displacement vectors can be used outside this
        function for further analyses. But bear in mind that this array
        is overwritten each time this function is called.
    inplace : bool, optional
        If ``True``, change `coord_unwrapped_prev` inplace to the
        unwrapped coordinates of the current frame
    debug : bool, optional
        If ``True``, check the input arguments. Default: ``False``

    Returns
    -------
    coord_unwrapped : numpy.ndarray
        The unwrapped coordinates of `atm_grp` at the current frame,
        nominally ``coord_unwrapped_prev + displacement``.
    """
    if debug:
        mdt.check.pos_array(
            coord_unwrapped_prev, shape=atm_grp.positions.shape
        )
        if displacement is not None:
            mdt.check.pos_array(
                displacement, shape=atm_grp.positions.shape, dtype=np.float64
            )

    displacement = mdt.box.vdist(
        pos1=atm_grp.positions,
        pos2=coord_unwrapped_prev,
        box=atm_grp.dimensions,
        out=displacement,
    )
    if inplace:
        coord_unwrapped_prev += displacement
        return coord_unwrapped_prev
    else:
        return coord_unwrapped_prev + displacement


def unwrap_trj(
    topfile,
    trjfile,
    universe,
    atm_grp,
    end=-1,
    make_whole=False,
    keep_whole=False,
    compound="fragments",
    reference="com",
    verbose=False,
    debug=False,
):
    """
    Unwrap the atoms of a MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup` out of the primary unit
    cell, i.e. calculate their coordinates in real space. The
    coordinates are changed inplace.

    This function uses the algorithm proposed by von Bülow et al. in
    J. Chem. Phys., 2020, 153, 021101. Basically it calculates the atom
    displacements from frame to frame and adds these displacements to
    the previous atom positions to build the unwraped trajectory.

    The main difference to :func:`unwrap` is that :func:`unwrap` only
    unwraps a single frame while this function unwraps the complete
    trajectory.

    Parameters
    ----------
    topfile, trjfile : str
        Because any changes to the
        :attr:`~MDAnalysis.core.groups.AtomGroup.positions` attribute of
        a :class:`~MDAnalysis.core.groups.AtomGroup` are overwritten
        each time the frame of the
        :attr:`~MDAnalysis.core.universe.Universe.trajectory` is changed
        (except if the
        :attr:`~MDAnalysis.core.universe.Universe.trajectory` is held in
        memory), the changes are written to a new trajectory. The name
        of the new (unwrapped) trajectory is given by `trjfile` and the
        name of the corresponding topology is given by `topfile`. Only
        the atoms of `atm_grp` are written to the new trajectory. When
        the unwrapping precedure is done, you can create a new universe
        from this unwrapped trajectory and topology. See the
        `MDAnalysis user guide`_ for supported file formats.

        .. _MDAnalysis user guide:
            https://userguide.mdanalysis.org/1.0.0/formats/index.html

    universe : MDAnalysis.core.universe.Universe
        The universe holding the trajectory and
        :class:`~MDAnalysis.core.groups.AtomGroup` to unwrap. You cannot
        pass a :class:`MDAnalysis.coordinates.base.FrameIteratorSliced`
        here, since unwrapping always has to start with the first frame
        and is much safer if every single frame is considered, because
        the algorithm only works if particles do not move more than half
        the box length in one step. If you want to unwrap the trajectory
        only until a certain point, use the `end` argument.
    atm_grp : MDAnalysis.core.groups.AtomGroup
        The :class:`~MDAnalysis.core.groups.AtomGroup` whose atoms
        should be unwrapped into real space. The
        :class:`~MDAnalysis.core.groups.AtomGroup` must not be an
        :class:`~MDAnalysis.core.groups.UpdatingAtomGroup`.
    end : int, optional
        Last frame to unwrap (exclusive, i.e. the last frame to unwrap
        is actually ``end-1``). A negative value means unwrap the
        complete trajectory.
    make_whole : bool, optional
        If ``True``, make the compounds of `atm_grp` that are split
        across the simulation box edges whole again for each individual
        frame before unwrapping.
    keep_whole : bool, optional
        If the molecules in the `universe` are already whole for each
        frame, it is sufficient to start from a structure with (at least
        a part of) the whole molecules in the primary unit cell and then
        propagate these whole molecules in real space instead of making
        the molecules whole for each individual frame. Note that
        `make_whole` takes precedence over `keep_whole`.
    compound : str, optional
        Which type of component to make whole. Must be either
        ``'group'``, ``'segments'``, ``'residues'``, ``'molecules'`` or
        ``'fragments'``. See :func:`make_whole` for more details. Is
        meaningless if `make_whole` and `keep_whole` are ``False``.
    reference : str, optional
        If 'com' (center of mass) or 'cog' (center of geometry), the
        compounds that were made whole will be shifted so that their
        individual reference point lies within the primary unit cell. If
        None, no such shift is performed. This only affects the starting
        structure from which to begin the unwrapping, since the
        displacements, which are calculated according to the minimum
        image convention, (should) stay the same.  Is meaningless if
        `make_whole` and `keep_whole` are ``False``.
    verbose : bool, optional
        If ``True``, print progress information to standard output.
    debug : bool, optional
        If ``True``, check the input arguments.
    """
    if debug:
        if isinstance(atm_grp, mda.core.groups.UpdatingAtomGroup):
            raise TypeError("atm_grp must not be an UpdatingAtomGroup")
        if (
            make_whole
            and compound != "group"
            and compound != "segments"
            and compound != "residues"
            and compound != "molecules"
            and compound != "fragments"
        ):
            raise ValueError(
                "compound must be either 'group',"
                " 'segments', 'residues', 'molecules' or"
                " 'fragments', but you gave '{}'".format(compound)
            )
        if (
            make_whole
            and reference != "com"
            and reference != "cog"
            and reference is not None
        ):
            raise ValueError(
                "reference must be either 'com', 'cog' or None, but you gave"
                " {}".format(reference)
            )
        if make_whole and reference == "com":
            mdt.check.masses(ag=atm_grp, flash_test=False)
        if make_whole and len(atm_grp.bonds) <= 0:
            raise ValueError("The AtomGroup contains no bonds")

    if end < 0:
        end = universe.trajectory.n_frames
    _, end, _, _ = mdt.check.frame_slicing(
        start=0, stop=end, step=1, n_frames_tot=universe.trajectory.n_frames
    )
    displacement = np.zeros(atm_grp.positions.shape, dtype=np.float64)

    ts = universe.trajectory[0]
    if verbose:
        timer = datetime.now()
        proc = psutil.Process()
        last_frame = universe.trajectory[end - 1].frame
        ts = universe.trajectory[0]
        print("  Frame   {:12d}".format(ts.frame))
        print(
            "    Step: {:>12}    Time: {:>12} (ps)".format(
                ts.data["step"], ts.data["time"]
            ),
        )
        print(
            "    Elapsed time:             {}".format(datetime.now() - timer),
        )
        print(
            "    Current memory usage: {:18.2f} MiB".format(
                proc.memory_info().rss / 2**20
            ),
        )
        timer = datetime.now()
    if debug:
        mdt.check.box(box=ts.dimensions, with_angles=True, dim=1)
    if make_whole or keep_whole:
        coord_unwrapped = mdt.box.make_whole(
            ag=atm_grp,
            compound=compound,
            reference=reference,
            inplace=True,
            debug=debug,
        )
    else:
        coord_unwrapped = mdt.box.wrap(ag=atm_grp, inplace=True, debug=debug)

    mdt.fh.backup(topfile)
    atm_grp.write(topfile)
    mdt.fh.backup(trjfile)
    with mda.Writer(trjfile, atm_grp.n_atoms) as w:
        w.write(atm_grp)
        for ts in universe.trajectory[1:end]:
            if verbose and (
                ts.frame % 10 ** (len(str(ts.frame)) - 1) == 0
                or ts.frame == last_frame
            ):
                print("  Frame   {:12d}".format(ts.frame))
                print(
                    "    Step: {:>12}    Time: {:>12} (ps)".format(
                        ts.data["step"], ts.data["time"]
                    ),
                )
                print(
                    "    Elapsed time:             {}".format(
                        datetime.now() - timer
                    ),
                )
                print(
                    "    Current memory usage: {:18.2f} MiB".format(
                        proc.memory_info().rss / 2**20
                    ),
                )
                timer = datetime.now()
            if debug:
                mdt.check.box(box=ts.dimensions, with_angles=True, dim=1)
            if make_whole:
                mdt.box.make_whole(
                    ag=atm_grp,
                    compound=compound,
                    reference=reference,
                    inplace=True,
                    debug=debug,
                )
            mdt.box.unwrap(
                atm_grp=atm_grp,
                coord_unwrapped_prev=coord_unwrapped,
                displacement=displacement,
                inplace=True,
                debug=debug,
            )
            atm_grp.positions = coord_unwrapped
            w.write(atm_grp)

    if verbose:
        print()
        print("  Created {}".format(topfile))
        print("  Created {}".format(trjfile))
