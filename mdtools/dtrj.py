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
Functions dealing with discrete trajectories.

Discrete trajectories must be stored in arrays.  Arrays that serve as
discrete trajectory must meet the requirements listed in
:func:`mdtools.check.dtrj`.  If a function in this module deviates from
these requirements, it will be cleary stated.
"""


# Third party libraries
import numpy as np

# First party libraries
import mdtools as mdt


def locate_trans(dtrj, axis=-1, pin="end", wrap=False, tfft=False, tlft=False):
    """
    Locate the frames of state transitions inside a discrete trajectory.

    Get a boolean array of the same shape as `dtrj` that indicates at
    which positions in `dtrj` state transitions occur.

    Parameters
    ----------
    dtrj : array_like
        Array containing the discrete trajectory.
    axis : int
        The axis along which to search for state transitions.  For
        ordinary discrete trajectories with shape ``(n, f)`` or
        ``(f,)``, where ``n`` is the number of compounds and ``f`` is
        the number of frames, set `axis` to ``-1`` or to ``1``.  If you
        parse a transposed discrete trajectory of shape ``(f, n)``, set
        `axis` to ``0``.
    pin : {"end", "start", "both"}
        Whether to return the start (last frame where a given compound
        is in the previous state) or end (first frame where a given
        compound is in the next state) of the state transitions.  If set
        to ``"both"``, two output arrays will be returned, one for
        ``"start"`` and one for ``"end"``.
    wrap : bool, optional
        If ``True``, `dtrj` is assumed to be continued after the last
        frame by `dtrj` itself, like when using periodic boundary
        conditions.  Consequently, if the first and last state in the
        trajectory do not match, this is interpreted as state
        transition.  The start of this transition is the last frame of
        the trajectory and the end is the first frame.
    tfft : bool, optional
        Treat First Frame as Transition.  If ``True``, treat the first
        frame as the end of a state transition.  Has no effect if `pin`
        is set to ``"start"``.  Must not be used together with `wrap`.
    tlft : bool, optional
        Treat Last Frame as Transition.  If ``True``, treat the last
        frame as the start of a state transition.  Has no effect if
        `pin` is set to ``"end"``.  Must not be used together with
        `wrap`.

    Returns
    -------
    trans_start : numpy.ndarray
        Boolean array of the same shape as `dtrj`.  Elements that
        evaluate to ``True`` indicate frames in `dtrj` where a given
        compound is the last time in a given state before it will be in
        another state in the next frame.  Can be used to index `dtrj`
        and get the states right before a state transition.  Is not
        returned if `pin` is ``"end"``.
    trans_end : numpy.ndarray
        Boolean array of the same shape as `dtrj`.  Elements that
        evaluate to ``True`` indicate frames in `dtrj` where a given
        compound is the first time in a new state after it was in
        another state one frame before.  Can be used to index `dtrj` and
        get the states right after a state transition.  Is not returned
        if `pin` is ``"start"``.

    See Also
    --------
    :func:`mdtools.numpy_helper_functions.locate_item_change`:
        Locate the positions of item changes in arbitrary arrays

    Notes
    -----
    This function only checks if the input array is a discrete
    trajectory using :func:`mdtools.check.dtrj` and then calls
    :func:`mdtools.numpy_helper_functions.locate_item_change`.

    To get the indices (i.e. frames) of the state transitions for each
    compound, apply :func:`numpy.nonzero` to the output arrays of this
    function.

    Examples
    --------
    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2]])
    >>> start, end = mdt.dtrj.locate_trans(dtrj, pin="both")
    >>> start
    array([[ True, False,  True, False, False, False],
           [False,  True, False, False,  True, False],
           [False, False,  True,  True, False, False],
           [ True, False, False,  True, False, False]])
    >>> end
    array([[False,  True, False,  True, False, False],
           [False, False,  True, False, False,  True],
           [False, False, False,  True,  True, False],
           [False,  True, False, False,  True, False]])
    """
    dtrj = mdt.check.dtrj(dtrj)
    return mdt.nph.locate_item_change(
        a=dtrj, axis=axis, pin=pin, wrap=wrap, tfic=tfft, tlic=tlft
    )


def trans_ix(dtrj, axis=-1, pin="end", wrap=False, tfft=False, tlft=False):
    """
    Get the frames indices of state transitions in a discrete
    trajectory.

    Parameters
    ----------
    dtrj : array_like
        The discrete trajectory for which to get all frame indices where
        its elements change.
    axis : int
        See :func:`mdtools.dtrj.locate_trans`.
    pin : {"end", "start", "both"}
        See :func:`mdtools.dtrj.locate_trans`.
    wrap : bool, optional
        See :func:`mdtools.dtrj.locate_trans`.
    tfft : bool, optional
        See :func:`mdtools.dtrj.locate_trans`.
    tlft : bool, optional
        See :func:`mdtools.dtrj.locate_trans`.

    Returns
    -------
    ix_start : tuple of numpy.ndarray
        Tuple of arrays, one for each dimension of `dtrj`, containing
        the frame indices where a state transition will occur.  Can be
        used to index `dtrj` and get the states right before the state
        transition.  Is not returned if `pin` is ``"end"``.
    ix_end : tuple of numpy.ndarray
        Tuple of arrays, one for each dimension of `dtrj`, containing
        the frame indices where a state transition has occurred.  Can be
        used to index `dtrj` and get the states right after the state
        transition.  Is not returned if `pin` is ``"start"``.

    See Also
    --------
    :func:`mdtools.numpy_helper_functions.item_change_ix` :
        Get the indices of item changes in arbitrary arrays
    :func:`mdtools.dtrj.locate_trans` :
        Locate the frames of state transitions inside a discrete
        trajectory.

    Notes
    -----
    This function only applies :func:`numpy.nonzero` to the output of
    :func:`mdtools.dtrj.locate_trans`.

    Examples
    --------
    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2]])
    >>> start, end = mdt.dtrj.trans_ix(dtrj, pin="both")
    >>> start
    (array([0, 0, 1, 1, 2, 2, 3, 3]), array([0, 2, 1, 4, 2, 3, 0, 3]))
    >>> end
    (array([0, 0, 1, 1, 2, 2, 3, 3]), array([1, 3, 2, 5, 3, 4, 1, 4]))
    """
    trans = mdt.dtrj.locate_trans(
        dtrj=dtrj, axis=axis, pin=pin, wrap=wrap, tfft=tfft, tlft=tlft
    )
    if pin == "both":
        return tuple(np.nonzero(tr) for tr in trans)
    else:
        return np.nonzero(trans)


def trans_per_state(dtrj, axis=-1):
    """
    Count the number of transitions leading into or out of a given state.

    Parameters
    ----------
    dtrj : array_like
        Array containing the discrete trajectory.  All elements of the
        array must be integers or floats whose fractional part is zero,
        because they are interpreted as the indices of the states in
        which a given compound is at a given frame.  However, unlike
        ordinary discrete trajectory in MDTools, `dtrj` can have an
        arbitrary shape.
    axis : int
        The axis along which to search for state transitions.  For
        ordinary discrete trajectories with shape ``(n, f)`` or ``(f,)``,
        where ``n`` is the number of compounds and ``f`` is the number
        of frames, set `axis` to ``-1``.  If you parse a transposed
        discrete trajectory to `dtrj`, set `axis` to ``0``.

    Returns
    -------
    hist_start : numpy.ndarray
        Histogram counting how many state transitions **started** from a
        given state, i.e. how many transitons led out of a given state.
    hist_end : numpy.ndarray
        Histogram counting how many state transitions **ended** it a
        given state, i.e. how many transitions led into a given state.

    Notes
    -----
    The histograms contain the counts for all states ranging from the
    minimum to the maximum state in `dtrj` in whole numbers.  This means,
    the states corresponding to the counts in `hist_start` and `hist_end`
    are given by ``numpy.arange(np.min(dtrj), np.max(dtrj)+1)``.

    Examples
    --------
    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2]])
    >>> hist_start, hist_end = mdt.dtrj.trans_per_state(dtrj)
    >>> # Number of transitions starting from the 1st, 2nd or 3rd state
    >>> hist_start
    array([3, 2, 3])
    >>> # Number of transitions ending in the 1st, 2nd or 3rd state
    >>> hist_end
    array([2, 3, 3])

    Iterprete first dimension as frames and second dimension as
    compounds:

    >>> hist_start, hist_end = mdt.dtrj.trans_per_state(dtrj, axis=0)
    >>> hist_start
    array([3, 3, 4])
    >>> hist_end
    array([3, 3, 4])
    """
    dtrj = np.asarray(dtrj)
    if np.any(np.modf(dtrj)[0] != 0):
        raise ValueError("At least one element of 'dtrj' is not an"
                         " integer")
    trans = (np.diff(dtrj, axis=axis) != 0)
    shape = list(trans.shape)
    shape[axis] = 1
    insertion = np.zeros(shape, dtype=bool)
    # States from which transitions start
    trans_start = np.concatenate([trans, insertion], axis=axis)
    # States in which transitions end
    trans_end = np.concatenate([insertion, trans], axis=axis)
    del trans, insertion
    bins = np.arange(np.min(dtrj), np.max(dtrj) + 2)
    hist_start = np.histogram(dtrj[trans_start], bins)[0]
    hist_end = np.histogram(dtrj[trans_end], bins)[0]
    return hist_start, hist_end
