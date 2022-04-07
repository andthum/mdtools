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
:func:`mdtools.check.dtrj`.
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
    :func:`mdtools.dtrj.trans_ix` :
        Get the frames indices of state transitions in a discrete
        trajectory

    Notes
    -----
    This function only checks if the input array is a discrete
    trajectory using :func:`mdtools.check.dtrj` and then calls
    :func:`mdtools.numpy_helper_functions.locate_item_change`.

    To get the frame indices of the state transitions for each compound,
    apply :func:`numpy.nonzero` to the output array(s) of this function.
    If you are only interested in the frame indices but not in the
    boolean arrays returned by this function, you can use
    :func:`mdtools.dtrj.trans_ix` instead.

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


def trans_per_state(
    dtrj, axis=-1, resolve_direction=False, wrap=False, tfft=False, tlft=False
):
    """
    Count the number of transitions leading into or out of a state.

    .. todo::

        Minimum image convention: Add the possibility to respect the
        minium image convention when resolving the transition direction
        (`resolve_direction` set to ``True``).  This is usefull, when
        the states continue periodically, for instance because the
        states are position bins along a box dimension with periodic
        boundary conditions.

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
        ordinary discrete trajectories with shape ``(n, f)`` or
        ``(f,)``, where ``n`` is the number of compounds and ``f`` is
        the number of frames, set `axis` to ``-1`` or to ``1``.  If you
        parse a transposed discrete trajectory of shape ``(f, n)``, set
        `axis` to ``0``.
    resolve_direction : bool, optional
        If ``True``, distinct between transitions from/into higher
        states or from/into lower states.
    wrap : bool, optional
        If ``True``, `dtrj` is assumed to be continued after the last
        frame by `dtrj` itself, like when using periodic boundary
        conditions.  Consequently, if the first and last state in the
        trajectory do not match, this is interpreted as state
        transition.  The start of this transition is the last frame of
        the trajectory and the end is the first frame.
    tfft : bool, optional
        Treat First Frame as Transition.  If ``True``, treat the first
        frame as the end of a state transition.  Must not be used
        together with `wrap` or `resolve_direction`.
    tlft : bool, optional
        Treat Last Frame as Transition.  If ``True``, treat the last
        frame as the start of a state transition.  Must not be used
        together with `wrap` or `resolve_direction`.

    Returns
    -------
    hist_start : numpy.ndarray
        Histogram counting how many state transitions **started** from a
        given state, i.e. how many transitons led out of a given state.
        If `resolve_direction` is ``True``, `hist_start` will be a
        2-dimensional array.  ``hist_start[0]`` is then the histogram
        counting how many transitions starting from a given state ended
        in an higher state.  ``hist_start[1]`` is the histogram counting
        how many transitions starting from a given state ended in a
        lower state.  If `resolve_direction` is ``False``, `hist_start`
        will be a 1-dimensional array that is formally the sum of the
        two above histograms:
        ``hist_start = np.sum(hist_start, axis=0)``.
    hist_end : numpy.ndarray
        Histogram counting how many state transitions **ended** in a
        given state, i.e. how many transitions led into a given state.
        If `resolve_direction` is ``True``, `hist_end` will be a
        2-dimensional array.  ``hist_end[0]`` is then the histogram
        counting how many transitions ending in a given state started
        from a lower state.  ``hist_end[1]`` is the histogram counting
        how many transitions ending in a given state started from an
        higher state.  If `resolve_direction` is ``False``, `hist_end`
        will be a 1-dimensional array that is formally the sum of the
        two above histograms: ``hist_end = np.sum(hist_end, axis=0)``.

    Notes
    -----
    The histograms contain the counts for all states ranging from the
    minimum to the maximum state in `dtrj` in whole numbers.  This
    means, the states corresponding to the counts in `hist_start` and
    `hist_end` are given by
    ``numpy.arange(np.min(dtrj), np.max(dtrj)+1)``.

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
    >>> hist_start_resolve, hist_end_resolve = mdt.dtrj.trans_per_state(
    ...     dtrj, resolve_direction=True
    ... )
    >>> # Number of transitions starting from the 1st, 2nd or 3rd state\
 and
    >>> #   * hist_start_resolve[0]: ending in an higher state
    >>> #   * hist_start_resolve[0]: ending in a lower state
    >>> hist_start_resolve
    array([[3, 2, 0],
           [0, 0, 3]])
    >>> np.array_equal(np.sum(hist_start_resolve, axis=0), hist_start)
    True
    >>> # Number of transitions ending in the 1st, 2nd or 3rd state and
    >>> #   * hist_end_resolve[0]: starting from a lower state
    >>> #   * hist_end_resolve[0]: starting from an higher state
    >>> hist_end_resolve
    array([[0, 2, 3],
           [2, 1, 0]])
    >>> np.array_equal(np.sum(hist_end_resolve, axis=0), hist_end)
    True
    >>> hist_start_wrap, hist_end_wrap = mdt.dtrj.trans_per_state(
    ...     dtrj, wrap=True
    ... )
    >>> hist_start_wrap
    array([4, 4, 4])
    >>> hist_end_wrap
    array([4, 4, 4])
    >>> hist_start_resolve_wrap, hist_end_resolve_wrap = \
mdt.dtrj.trans_per_state(
    ...     dtrj, resolve_direction=True, wrap=True
    ... )
    >>> hist_start_resolve_wrap
    array([[4, 3, 0],
           [0, 1, 4]])
    >>> np.array_equal(
    ...     np.sum(hist_start_resolve_wrap, axis=0), hist_start_wrap
    ... )
    True
    >>> hist_end_resolve_wrap
    array([[0, 3, 4],
           [4, 1, 0]])
    >>> np.array_equal(
    ...     np.sum(hist_end_resolve_wrap, axis=0), hist_end_wrap
    ...  )
    True
    >>> hist_start_tfft, hist_end_tfft = mdt.dtrj.trans_per_state(
    ...     dtrj, tfft=True
    ... )
    >>> hist_start_tfft
    array([3, 2, 3])
    >>> hist_end_tfft
    array([4, 4, 4])
    >>> hist_start_tlft, hist_end_tlft = mdt.dtrj.trans_per_state(
    ...     dtrj, tlft=True
    ... )
    >>> hist_start_tlft
    array([4, 4, 4])
    >>> hist_end_tlft
    array([2, 3, 3])
    >>> hist_start_tfft_tlft, hist_end_tfft_tlft = \
mdt.dtrj.trans_per_state(
    ...     dtrj, tfft=True, tlft=True
    ... )
    >>> hist_start_tfft_tlft
    array([4, 4, 4])
    >>> hist_end_tfft_tlft
    array([4, 4, 4])

    Iterprete first dimension as frames and second dimension as
    compounds:

    >>> hist_start, hist_end = mdt.dtrj.trans_per_state(dtrj, axis=0)
    >>> hist_start
    array([3, 3, 4])
    >>> hist_end
    array([3, 3, 4])
    >>> hist_start_resolve, hist_end_resolve = mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, resolve_direction=True
    ... )
    >>> hist_start_resolve
    array([[3, 3, 0],
           [0, 0, 4]])
    >>> np.array_equal(np.sum(hist_start_resolve, axis=0), hist_start)
    True
    >>> hist_end_resolve
    array([[0, 2, 4],
           [3, 1, 0]])
    >>> np.array_equal(np.sum(hist_end_resolve, axis=0), hist_end)
    True
    >>> hist_start_wrap, hist_end_wrap = mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, wrap=True
    ... )
    >>> hist_start_wrap
    array([3, 5, 6])
    >>> hist_end_wrap
    array([3, 5, 6])
    >>> hist_start_resolve_wrap, hist_end_resolve_wrap = \
mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, resolve_direction=True, wrap=True
    ... )
    >>> hist_start_resolve_wrap
    array([[3, 5, 0],
           [0, 0, 6]])
    >>> np.array_equal(
    ...     np.sum(hist_start_resolve_wrap, axis=0), hist_start_wrap
    ... )
    True
    >>> hist_end_resolve_wrap
    array([[0, 2, 6],
           [3, 3, 0]])
    >>> np.array_equal(
    ...     np.sum(hist_end_resolve_wrap, axis=0), hist_end_wrap
    ...  )
    True
    >>> hist_start_tfft, hist_end_tfft = mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, tfft=True
    ... )
    >>> hist_start_tfft
    array([3, 3, 4])
    >>> hist_end_tfft
    array([4, 5, 7])
    >>> hist_start_tlft, hist_end_tlft = mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, tlft=True
    ... )
    >>> hist_start_tlft
    array([4, 5, 7])
    >>> hist_end_tlft
    array([3, 3, 4])
    >>> hist_start_tfft_tlft, hist_end_tfft_tlft = \
mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, tfft=True, tlft=True
    ... )
    >>> hist_start_tfft_tlft
    array([4, 5, 7])
    >>> hist_end_tfft_tlft
    array([4, 5, 7])
    """
    dtrj = np.asarray(dtrj)
    if np.any(np.modf(dtrj)[0] != 0):
        raise ValueError("At least one element of 'dtrj' is not an integer")
    if resolve_direction and (tfft or tlft):
        raise ValueError(
            "'resolve_direction' must not be used together with 'tfft' of"
            " 'tlft'"
        )
    if wrap and (tfft or tlft):
        raise ValueError(
            "'wrap' must not be used together with 'tfft' of 'tlft'"
        )

    state_diff = np.diff(dtrj, axis=axis)
    if resolve_direction:
        # Distinct between transitions to higher and to lower states
        trans = np.asarray([state_diff > 0, state_diff < 0])
    else:
        # Don't discriminate between different types of transitions
        trans = np.asarray([state_diff != 0])
    # Only index `dtrj.shape` with `axis` after np.diff to get a proper
    # numpy.AxisError if `axis` is out of bounds (instead of an
    # IndexError)
    if dtrj.shape[axis] == 0:
        return tuple(
            np.squeeze(np.array([], dtype=int).reshape(len(trans), 0)),
            np.squeeze(np.array([], dtype=int).reshape(len(trans), 0)),
        )

    shape = list(state_diff.shape)
    shape[axis] = 1
    shape = tuple(shape)
    # `insertion` = array to insert after or before `trans` to bring
    # `trans` to the same shape as `dtrj` and make `trans` a mask for
    # states from which transitions start or in which transitions end.
    if wrap:
        state_diff = dtrj.take(0, axis=axis) - dtrj.take(-1, axis=axis)
        if resolve_direction:
            insertion_start = np.asarray(
                [
                    (state_diff > 0).reshape(shape),
                    (state_diff < 0).reshape(shape),
                ]
            )
        else:
            insertion_start = np.asarray([(state_diff != 0).reshape(shape)])
        insertion_end = insertion_start
    else:
        if tfft:
            insertion_end = np.ones((len(trans),) + shape, dtype=bool)
        else:
            insertion_end = np.zeros((len(trans),) + shape, dtype=bool)
        if tlft:
            insertion_start = np.ones((len(trans),) + shape, dtype=bool)
        else:
            insertion_start = np.zeros((len(trans),) + shape, dtype=bool)
    del state_diff

    # States from which transitions start or in which transitions end
    trans_start = np.zeros((len(trans),) + dtrj.shape, dtype=bool)
    trans_end = np.zeros_like(trans_start, dtype=bool)
    # Loop over different transition types (only relevant if
    # `resolve_direction` is True).
    for i, tr in enumerate(trans):
        trans_start[i] = np.concatenate([tr, insertion_start[i]], axis=axis)
        trans_end[i] = np.concatenate([insertion_end[i], tr], axis=axis)
    del trans, tr, insertion_start, insertion_end

    min_state, max_state = np.min(dtrj), np.max(dtrj)
    bins = np.arange(min_state, max_state + 2)
    hist_start = np.zeros(
        (len(trans_start), max_state - min_state + 1), dtype=int
    )
    hist_end = np.zeros_like(hist_start)
    # Loop over different transition types (only relevant if
    # `resolve_direction` is True).
    for i, tr_start in enumerate(trans_start):
        hist_start[i] = np.histogram(dtrj[tr_start], bins)[0]
        hist_end[i] = np.histogram(dtrj[trans_end[i]], bins)[0]
    hist_start = np.squeeze(hist_start)
    hist_end = np.squeeze(hist_end)
    return hist_start, hist_end
