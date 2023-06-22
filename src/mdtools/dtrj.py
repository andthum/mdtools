# This file is part of MDTools.
# Copyright (C) 2021-2023  The MDTools Development Team and all
# contributors listed in the file AUTHORS.rst
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


# Third-party libraries
import numpy as np
import psutil

# First-party libraries
import mdtools as mdt


def locate_trans(
    dtrj,
    axis=-1,
    pin="end",
    trans_type=None,
    wrap=False,
    tfft=False,
    tlft=False,
    mic=False,
    min_state=None,
    max_state=None,
):
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
        the number of frames, set `axis` to ``-1``.  If you parse a
        transposed discrete trajectory of shape ``(f, n)``, set `axis`
        to ``0``.
    pin : {"end", "start", "both"}
        Whether to return the ``"start"`` (last frame where a given
        compound is in the previous state) or ``"end"`` (first frame
        where a given compound is in the next state) of the state
        transitions.  If set to ``"both"``, two output arrays will be
        returned, one for ``"start"`` and one for ``"end"``.
    trans_type : {None, "higher", "lower", "both"} or int or \
iterable of ints, optional
        Whether to locate all state transitions without discriminating
        between different transition types (``None``) or whether to
        locate only state transitions into higher states (``"higher"``)
        or into lower states (``"lower"``).  If set to ``"both"``, two
        output arrays will be returned, one for ``"higher"`` and one for
        ``"lower"``.  If `pin` is set to ``"both"``, too, a 2x2 tuple
        will be returned.  The first index addresses `pin`, the second
        index addresses `trans_type`.  If `trans_type` is an integer,
        locate only state transitions where the difference between the
        final and first state is equal to the given integer value.  If
        `trans_type` is an iterable of integers, one output array for
        each given integer will be returned.
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
        is set to ``"start"``.  Must not be used together with
        `trans_type`, `wrap` or `mic`.
    tlft : bool, optional
        Treat Last Frame as Transition.  If ``True``, treat the last
        frame as the start of a state transition.  Has no effect if
        `pin` is set to ``"end"``.  Must not be used together with
        `trans_type`, `wrap` or `mic`.
    mic : bool, optional
        If ``True``, respect the Minimum Image Convention when
        evaluating the transition type, i.e. when evaluating whether the
        transition was in positive direction (to a higher state) or in
        negative direction (to a lower state).  Has no effect if
        `trans_type` is ``None``.  This option could be useful, when
        the states have periodic boundary conditions, for instance
        because the states are position bins along a box dimension with
        periodic boundary conditions.  In this case, the states should
        result from an equidistant binning, otherwise the MIC algorithm
        will give wrong results, because it only considers the state
        indices.
    min_state, max_state : scalar or array_like, optional
        The lower and upper bound(s) for the minimum image convention.
        Has no effect if `mic` is ``False`` or `trans_type` is ``None``.
        If ``None``, the minium and maximum value of `dtrj` is taken,
        respectively.  `min_state` must be smaller than `max_state`.  If
        the transition between two states is larger than
        ``0.5 * (max_state - min_state)``, it is wrapped back to lie
        within that range.  See also
        :func:`mdtools.numpy_helper_functions.diff_mic`.

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
        Get the frame indices of state transitions in a discrete
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
    if pin not in ("end", "start", "both"):
        raise ValueError(
            "'pin' must be either 'end', 'start' or 'both', but you gave"
            " {}".format(pin)
        )
    if pin == "end":
        pin = "after"
    elif pin == "start":
        pin = "before"
    return mdt.nph.locate_item_change(
        a=dtrj,
        axis=axis,
        pin=pin,
        change_type=trans_type,
        rtol=0,
        atol=0,
        wrap=wrap,
        tfic=tfft,
        tlic=tlft,
        mic=mic,
        amin=min_state,
        amax=max_state,
    )


def trans_ix(
    dtrj,
    axis=-1,
    pin="end",
    trans_type=None,
    wrap=False,
    tfft=False,
    tlft=False,
    mic=False,
    min_state=None,
    max_state=None,
):
    """
    Get the frame indices of state transitions in a discrete trajectory.

    Parameters
    ----------
    dtrj : array_like
        The discrete trajectory for which to get all frame indices where
        its elements change.
    axis : int
        See :func:`mdtools.dtrj.locate_trans`.
    pin : {"end", "start", "both"}
        See :func:`mdtools.dtrj.locate_trans`.
    trans_type : {None, "higher", "lower", "both"} or int or \
iterable of ints, optional
        See :func:`mdtools.dtrj.locate_trans`.
    wrap : bool, optional
        See :func:`mdtools.dtrj.locate_trans`.
    tfft : bool, optional
        See :func:`mdtools.dtrj.locate_trans`.
    tlft : bool, optional
        See :func:`mdtools.dtrj.locate_trans`.
    mic : bool, optional
        See :func:`mdtools.dtrj.locate_trans`.
    min_state, max_state : scalar or array_like, optional
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
    This function only checks if the input array is a discrete
    trajectory using :func:`mdtools.check.dtrj` and then calls
    :func:`mdtools.numpy_helper_functions.item_change_ix`.

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
    dtrj = mdt.check.dtrj(dtrj)
    if pin not in ("end", "start", "both"):
        raise ValueError(
            "'pin' must be either 'end', 'start' or 'both', but you gave"
            " {}".format(pin)
        )
    if pin == "end":
        pin = "after"
    elif pin == "start":
        pin = "before"
    return mdt.nph.item_change_ix(
        a=dtrj,
        axis=axis,
        pin=pin,
        change_type=trans_type,
        rtol=0,
        atol=0,
        wrap=wrap,
        tfic=tfft,
        tlic=tlft,
        mic=mic,
        amin=min_state,
        amax=max_state,
    )


def trans_per_state(dtrj, **kwargs):
    """
    Count the number of transitions leading into or out of a state.

    Parameters
    ----------
    dtrj : array_like
        Array containing the discrete trajectory.
    kwargs : dict, optional
        Additional keyword arguments to parse to
        :func:`mdtools.dtrj.locate_trans`.  See there for possible
        choices.  This function sets the following default values:

            * ``pin = "end"``
            * ``trans_type = None``
            * ``min_state = None``
            * ``max_state = None``

    Returns
    -------
    hist_start : numpy.ndarray or tuple
        1d array or tuple of 1d arrays (if `trans_type` is ``'both'`` or
        an iterable of ints).  The histogram counts how many state
        transitions started from a given state, i.e. how many
        transitions led out of a given state.  Is not returned if `pin`
        is ``"end"``.
    hist_end : numpy.ndarray or tuple
        1d array or tuple of 1d arrays (if `trans_type` is ``'both'`` or
        an iterable of ints).  The histogram counts how many state
        transitions ended in a given state, i.e. how many transitions
        led into a given state.  Is not returned if `pin` is
        ``"start"``.

    See Also
    --------
    :func:`mdtools.dtrj.trans_per_state_vs_time` :
        Count the number of transitions leading into or out of a state
        for each frame in a discrete trajectory
    :func:`mdtools.dtrj.locate_trans` :
        Locate the frames of state transitions inside a discrete
        trajectory

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
    >>> hist_start, hist_end = mdt.dtrj.trans_per_state(
    ...     dtrj, pin="both"
    ... )
    >>> # Number of transitions starting from the 1st, 2nd or 3rd state
    >>> hist_start
    array([3, 2, 3])
    >>> # Number of transitions ending in the 1st, 2nd or 3rd state
    >>> hist_end
    array([2, 3, 3])
    >>> hist_start_type, hist_end_type = mdt.dtrj.trans_per_state(
    ...     dtrj, pin="both", trans_type="both"
    ... )
    >>> # Number of transitions starting from the 1st, 2nd or 3rd \
state and ending in an
    >>> #   * higher state (hist_start_type[0])
    >>> #   * lower state (hist_start_type[1])
    >>> hist_start_type
    (array([3, 2, 0]), array([0, 0, 3]))
    >>> np.array_equal(np.sum(hist_start_type, axis=0), hist_start)
    True
    >>> # Number of transitions ending in the 1st, 2nd or 3rd state \
and starting from an
    >>> #   * lower state (hist_end_type[0])
    >>> #   * higher state (hist_end_type[1])
    >>> hist_end_type
    (array([0, 2, 3]), array([2, 1, 0]))
    >>> np.array_equal(np.sum(hist_end_type, axis=0), hist_end)
    True
    >>> # Number of transitions starting from the 1st, 2nd or 3rd \
state where the difference
    >>> # between the final and initial state is plus or minus one
    >>> hist_plus_one, hist_minus_one = mdt.dtrj.trans_per_state(
    ...     dtrj, pin="start", trans_type=(1, -1)
    ... )
    >>> hist_plus_one
    array([2, 2, 0])
    >>> hist_minus_one
    array([0, 0, 1])
    >>> hist_start_wrap, hist_end_wrap = mdt.dtrj.trans_per_state(
    ...     dtrj, pin="both", wrap=True
    ... )
    >>> hist_start_wrap
    array([4, 4, 4])
    >>> hist_end_wrap
    array([4, 4, 4])
    >>> hist_start_type_wrap, hist_end_type_wrap = \
mdt.dtrj.trans_per_state(
    ...     dtrj, pin="both", trans_type="both", wrap=True
    ... )
    >>> hist_start_type_wrap
    (array([4, 3, 0]), array([0, 1, 4]))
    >>> np.array_equal(
    ...     np.sum(hist_start_type_wrap, axis=0), hist_start_wrap
    ... )
    True
    >>> hist_end_type_wrap
    (array([0, 3, 4]), array([4, 1, 0]))
    >>> np.array_equal(
    ...     np.sum(hist_end_type_wrap, axis=0), hist_end_wrap
    ...  )
    True
    >>> hist_start_tfft, hist_end_tfft = mdt.dtrj.trans_per_state(
    ...     dtrj, pin="both", tfft=True
    ... )
    >>> hist_start_tfft
    array([3, 2, 3])
    >>> hist_end_tfft
    array([4, 4, 4])
    >>> hist_start_tlft, hist_end_tlft = mdt.dtrj.trans_per_state(
    ...     dtrj, pin="both", tlft=True
    ... )
    >>> hist_start_tlft
    array([4, 4, 4])
    >>> hist_end_tlft
    array([2, 3, 3])
    >>> hist_start_tfft_tlft, hist_end_tfft_tlft = \
mdt.dtrj.trans_per_state(
    ...     dtrj, pin="both", tfft=True, tlft=True
    ... )
    >>> hist_start_tfft_tlft
    array([4, 4, 4])
    >>> hist_end_tfft_tlft
    array([4, 4, 4])

    Interpret first dimension as frames and second dimension as
    compounds:

    >>> hist_start, hist_end = mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, pin="both"
    ... )
    >>> hist_start
    array([3, 3, 4])
    >>> hist_end
    array([3, 3, 4])
    >>> hist_start_type, hist_end_type = mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, pin="both", trans_type="both"
    ... )
    >>> hist_start_type
    (array([3, 3, 0]), array([0, 0, 4]))
    >>> np.array_equal(np.sum(hist_start_type, axis=0), hist_start)
    True
    >>> hist_end_type
    (array([0, 2, 4]), array([3, 1, 0]))
    >>> np.array_equal(np.sum(hist_end_type, axis=0), hist_end)
    True
    >>> hist_plus_one, hist_minus_one = mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, pin="start", trans_type=(1, -1)
    ... )
    >>> hist_plus_one
    array([2, 3, 0])
    >>> hist_minus_one
    array([0, 0, 1])
    >>> hist_start_wrap, hist_end_wrap = mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, pin="both", wrap=True
    ... )
    >>> hist_start_wrap
    array([3, 5, 6])
    >>> hist_end_wrap
    array([3, 5, 6])
    >>> hist_start_type_wrap, hist_end_type_wrap = \
mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, pin="both", trans_type="both", wrap=True
    ... )
    >>> hist_start_type_wrap
    (array([3, 5, 0]), array([0, 0, 6]))
    >>> np.array_equal(
    ...     np.sum(hist_start_type_wrap, axis=0), hist_start_wrap
    ... )
    True
    >>> hist_end_type_wrap
    (array([0, 2, 6]), array([3, 3, 0]))
    >>> np.array_equal(
    ...     np.sum(hist_end_type_wrap, axis=0), hist_end_wrap
    ...  )
    True
    >>> hist_start_tfft, hist_end_tfft = mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, pin="both", tfft=True
    ... )
    >>> hist_start_tfft
    array([3, 3, 4])
    >>> hist_end_tfft
    array([4, 5, 7])
    >>> hist_start_tlft, hist_end_tlft = mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, pin="both", tlft=True
    ... )
    >>> hist_start_tlft
    array([4, 5, 7])
    >>> hist_end_tlft
    array([3, 3, 4])
    >>> hist_start_tfft_tlft, hist_end_tfft_tlft = \
mdt.dtrj.trans_per_state(
    ...     dtrj, axis=0, pin="both", tfft=True, tlft=True
    ... )
    >>> hist_start_tfft_tlft
    array([4, 5, 7])
    >>> hist_end_tfft_tlft
    array([4, 5, 7])
    """
    pin = kwargs.setdefault("pin", "end")
    trans_type = kwargs.setdefault("trans_type", None)
    min_state = kwargs.setdefault("min_state", None)
    max_state = kwargs.setdefault("max_state", None)
    dtrj = np.asarray(dtrj)
    trans = mdt.dtrj.locate_trans(dtrj, **kwargs)

    min_state = np.min(dtrj) if min_state is None else min_state
    max_state = np.max(dtrj) if max_state is None else max_state
    if max_state <= min_state:
        raise ValueError(
            "'max_state' ({}) must be greater than 'min_state'"
            " ({})".format(max_state, min_state)
        )
    bins = np.arange(min_state, max_state + 2)

    dtrj_copy = dtrj.copy()
    try:
        (i for i in trans_type)
        trans_type_is_iterable = True
    except TypeError:  # change_type is not iterable
        trans_type_is_iterable = False
    if pin == "both" and (trans_type == "both" or trans_type_is_iterable):
        hist = [[None for tr in trns] for trns in trans]
        for i, trns in enumerate(trans):
            for j, tr in enumerate(trns):
                dtrj_copy[:] = dtrj
                dtrj_copy[~tr] = bins[0] - 1
                hist[i][j] = np.histogram(dtrj_copy, bins)[0]
            hist[i] = tuple(hist[i])
        return tuple(hist)
    elif pin == "both" or trans_type == "both" or trans_type_is_iterable:
        hist = [None for tr in trans]
        for i, tr in enumerate(trans):
            dtrj_copy[:] = dtrj
            dtrj_copy[~tr] = bins[0] - 1
            hist[i] = np.histogram(dtrj_copy, bins)[0]
        return tuple(hist)
    else:  # pin != "both" and trans_type != "both" and not iterable:
        # Prevent states where no transitions occurred from being
        # counted in the histogram.
        dtrj_copy[~trans] = bins[0] - 1
        hist = np.histogram(dtrj_copy, bins)[0]
        return hist


def _histogram(a, bins):
    """
    Compute the histogram of a dataset.

    Helper function for :func:`mdtools.dtrj.trans_per_state_vs_time`.

    Parameters
    ----------
    a : array_like
        Input data.  See :func:`numpy.histogram`.
    bins : int or sequence of scalars or str
        Binning.  See :func:`numpy.histogram`.

    Returns
    -------
    hist : numpy.ndarray
        The histogram.

    See Also
    --------
    :func:`numpy.histogram` : Compute the histogram of a dataset

    Notes
    -----
    This function simply returns ``numpy.histogram(a, bins)[0]``.  It is
    meant as a helper function for
    :func:`mdtools.dtrj.trans_per_state_vs_time` to apply
    :func:`numpy.histogram` along the compound axis of a discrete
    trajectory for each frame via :func:`numpy.apply_along_axis`.
    """
    return np.histogram(a, bins)[0]


def trans_per_state_vs_time(dtrj, time_axis=-1, cmp_axis=0, **kwargs):
    """
    Count the number of transitions leading into or out of a state for
    each frame in a discrete trajectory.

    Parameters
    ----------
    dtrj : array_like
        Array containing the discrete trajectory.
    time_axis : int
        The axis of `dtrj` that holds the trajectory frames.  This is
        the axis along which to search for state transitions.  For
        ordinary discrete trajectories with shape ``(n, f)`` or
        ``(f,)``, where ``n`` is the number of compounds and ``f`` is
        the number of frames, set `time_axis` to ``-1``.  If you parse a
        transposed discrete trajectory of shape ``(f, n)``, set
        `time_axis` to ``0``.
    cmp_axis : int
        The axis of `dtrj` that holds the compounds.  This is the axis
        along which the histograms will be computed.  For ordinary
        discrete trajectories, set `cmp_axis` to ``0``.  For transposed
        discrete trajectory, set `cmp_axis` to ``-1``.
    kwargs : dict, optional
        Additional keyword arguments to parse to
        :func:`mdtools.dtrj.locate_trans`.  See there for possible
        choices.  This function sets the following default values:

            * ``pin = "end"``
            * ``trans_type = None``
            * ``min_state = None``
            * ``max_state = None``

    Returns
    -------
    hist_start : numpy.ndarray or tuple
        2d array or tuple of 2d arrays (if `trans_type` is ``'both'`` or
        an iterable of ints) containing one histogram for each frame in
        dtrj.  The histograms count how many state transitions started
        from a given state, i.e. how many transitions led out of a given
        state.  Is not returned if `pin` is ``"end"``.
    hist_end : numpy.ndarray
        2d array or tuple of 2d arrays (if `trans_type` is ``'both'`` or
        an iterable of ints) containing one histogram for each frame in
        dtrj.  The histograms count how many state transitions ended in
        a given state, i.e. how many transitions led into a given state.
        Is not returned if `pin` is ``"start"``.

    See Also
    --------
    :func:`mdtools.dtrj.trans_per_state` :
        Count the number of transitions leading into or out of a state
        for the entire trajectory (not frame-wise).
    :func:`mdtools.dtrj.locate_trans` :
        Locate the frames of state transitions inside a discrete
        trajectory

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
    >>> hist_start, hist_end = mdt.dtrj.trans_per_state_vs_time(
    ...     dtrj, pin="both"
    ... )
    >>> # Number of transitions starting from the 1st, 2nd or 3rd \
state for each frame
    >>> hist_start
    array([[2, 0, 0, 1, 0, 0],
           [0, 1, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 0]])
    >>> # Number of transitions ending in the 1st, 2nd or 3rd state \
for each frame
    >>> hist_end
    array([[0, 0, 0, 1, 0, 1],
           [0, 1, 0, 0, 2, 0],
           [0, 1, 1, 1, 0, 0]])
    >>> hist_start_type, hist_end_type = \
mdt.dtrj.trans_per_state_vs_time(
    ...     dtrj, pin="both", trans_type="both"
    ... )
    >>> # Number of transitions starting from the
    >>> #   * 1st state (hist_start_type[0][0])
    >>> #   * 2nd state (hist_start_type[0][1])
    >>> #   * 3rd state (hist_start_type[0][2])
    >>> # and ending in an
    >>> #   * higher state (hist_start_type[0])
    >>> #   * lower state (hist_start_type[1])
    >>> hist_start_type[0]
    array([[2, 0, 0, 1, 0, 0],
           [0, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> hist_start_type[1]
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0]])
    >>> np.array_equal(np.sum(hist_start_type, axis=0), hist_start)
    True
    >>> # Number of transitions ending in the
    >>> #   * 1st state (hist_end_type[0][0])
    >>> #   * 2nd state (hist_end_type[0][1])
    >>> #   * 3rd state (hist_end_type[0][2])
    >>> # and starting from an
    >>> #   * lower state (hist_end_type[0])
    >>> #   * higher state (hist_end_type[1])
    >>> hist_end_type[0]
    array([[0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 1, 0],
           [0, 1, 1, 1, 0, 0]])
    >>> hist_end_type[1]
    array([[0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> np.array_equal(np.sum(hist_end_type, axis=0), hist_end)
    True
    >>> # Number of transitions starting from the 1st, 2nd or 3rd \
state where the difference
    >>> # between the final and initial state is plus or minus one
    >>> hist_plus_one, hist_minus_one = \
mdt.dtrj.trans_per_state_vs_time(
    ...     dtrj, pin="start", trans_type=(1, -1), wrap=True
    ... )
    >>> hist_plus_one
    array([[1, 0, 0, 1, 0, 1],
           [0, 1, 1, 0, 0, 1],
           [0, 0, 0, 0, 0, 0]])
    >>> hist_minus_one
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1],
           [0, 0, 0, 1, 0, 0]])
    """
    pin = kwargs.setdefault("pin", "end")
    trans_type = kwargs.setdefault("trans_type", None)
    min_state = kwargs.setdefault("min_state", None)
    max_state = kwargs.setdefault("max_state", None)
    dtrj = np.asarray(dtrj)
    trans = mdt.dtrj.locate_trans(dtrj, axis=time_axis, **kwargs)

    min_state = np.min(dtrj) if min_state is None else min_state
    max_state = np.max(dtrj) if max_state is None else max_state
    if max_state <= min_state:
        raise ValueError(
            "'max_state' ({}) must be greater than 'min_state'"
            " ({})".format(max_state, min_state)
        )
    bins = np.arange(min_state, max_state + 2)

    dtrj_copy = dtrj.copy()
    try:
        (i for i in trans_type)
        trans_type_is_iterable = True
    except TypeError:  # change_type is not iterable
        trans_type_is_iterable = False
    if pin == "both" and (trans_type == "both" or trans_type_is_iterable):
        hist = [[None for tr in trns] for trns in trans]
        for i, trns in enumerate(trans):
            for j, tr in enumerate(trns):
                dtrj_copy[:] = dtrj
                dtrj_copy[~tr] = bins[0] - 1
                hist[i][j] = np.apply_along_axis(
                    func1d=mdt.dtrj._histogram,
                    axis=cmp_axis,
                    arr=dtrj_copy,
                    bins=bins,
                )
            hist[i] = tuple(hist[i])
        return tuple(hist)
    elif pin == "both" or trans_type == "both" or trans_type_is_iterable:
        hist = [None for tr in trans]
        for i, tr in enumerate(trans):
            dtrj_copy[:] = dtrj
            dtrj_copy[~tr] = bins[0] - 1
            hist[i] = np.apply_along_axis(
                func1d=mdt.dtrj._histogram,
                axis=cmp_axis,
                arr=dtrj_copy,
                bins=bins,
            )
        return tuple(hist)
    else:  # pin != "both" and trans_type != "both" and not iterable:
        # Prevent states where no transitions occurred from being
        # counted in the histogram.
        dtrj_copy[~trans] = bins[0] - 1
        hist = np.apply_along_axis(
            func1d=mdt.dtrj._histogram, axis=cmp_axis, arr=dtrj_copy, bins=bins
        )
        return hist


def remain_prob(  # noqa: C901
    dtrj,
    restart=1,
    continuous=False,
    discard_neg_start=False,
    discard_all_neg=False,
    verbose=False,
):
    r"""
    Calculate the probability that a compound is in the same state as at
    time :math:`t_0` after a lag time :math:`\Delta t`

    Take a discrete trajectory and calculate the probability to find a
    compound in the same state as at time :math:`t_0` after a lag time
    :math:`\Delta t`.

    Parameters
    ----------
    dtrj : array_like
        The discrete trajectory.  Array of shape ``(n, f)``, where ``n``
        is the number of compounds and ``f`` is the number of frames.
        The shape can also be ``(f,)``, in which case the array is
        expanded to shape ``(1, f)``.   The elements of `dtrj` are
        interpreted as the indices of the states in which a given
        compound is at a given frame.
    restart : int, optional
        Restart every `restart` frames.  Determines how many restarting
        points :math:`t_0` are used for averaging.
    continuous : bool, optional
        If ``True``, compounds must continuously be in the same state
        without interruption in order to be counted (see notes).
    discard_neg_start : bool, optional
        If ``True``, discard all transitions starting from a negative
        state (see notes).  Must not be used together with
        `discard_all_neg`.
    discard_all_neg : bool, optional
        If ``True``, discard all negative states (see notes).  Must not
        be used together with `discard_neg_start`.
    verbose : bool, optional
        If ``True`` print a progress bar.

    Returns
    -------
    prop : numpy.ndarray
        Array of shape ``(f,)`` containing for all possible lag times
        the probability that a compound is in the same state as at time
        :math:`t_0` after a lag time :math:`\Delta t`.

    See Also
    --------
    :func:`mdtools.dtrj.remain_prob_discrete` :
        Calculate the probability that a compound is in the same state
        as at time :math:`t_0` after a lag time :math:`\Delta t`
        resolved with respect to the states in second discrete
        trajectory.

    Notes
    -----
    If you want to compute the probability to stay in the same state for
    each individual state in a discrete trajectory, use
    :func:`mdtools.dtrj.remain_prob_discrete` and parse the same
    discrete trajectory to `dtrj1` and `dtrj2`.

    **Continuous and Discontinuous Probability**

    If `continuous` is ``False``, this function calculates the
    probability to be *still or again* in the same state as at time
    :math:`t_0` after a lag :math:`\Delta t`:

    .. math::

        p(\xi(t_0 + \Delta t) \in S | \xi(t_0) \in S) =
        \left\langle
            \frac{N(t_0, t_0 + \Delta t)}{N(t_0)}
        \right\rangle_t

    Here, :math:`\xi(t)` is the discretized coordinate and :math:`S` is
    a given valid discrete state.  :math:`N(t_0)` is the number of
    compounds that are in a valid state at time :math:`t_0` and
    :math:`N(t_0, t_0 + \Delta t)` is the number of compounds that are
    at time :math:`t_0 + \Delta t` *still or again* in the same valid
    state as at time :math:`t_0`.  The brackets
    :math:`\langle ... \rangle_t` denote averaging over all given
    restarting times :math:`t_0`.

    If `continuous` is ``True``, this function calculates the
    probability to be *still* in the same state as at time :math:`t_0`
    after a lag time :math:`\Delta t`:

    .. math::

        p(\xi(t) \in S ~\forall t \in [t_0, t_0 + \Delta t]) =
        \left\langle
            \frac{N(\{t \in [t_0, t_0 + \Delta t]\})}{N(t_0)}
        \right\rangle_t

    :math:`N(\{t \in [t_0, t_0 + \Delta t]\})` is the number of
    compounds that *continuously* stay in their initial valid state
    *for all times* from :math:`t_0` to :math:`t_0 + \Delta t`.

    **Valid and Invalid States**

    By default, all states in the given discrete trajectory are valid.
    However, in some cases you might want to treat certain states as
    invalid, e.g. because you only want to consider specific states.
    This can be achieved with the arguments `discard_neg_start` or
    `discard_all_neg` which treat negative states (i.e. states with a
    negative state number/index) as invalid states.

    .. note::

        If you want to treat states with index zero as invalid, too,
        simply subtract one from `dtrj`.  If you want to treat all
        states below a certain cutoff as invalid, subtract this cutoff
        from `dtrj`.  If you want to treat certain states as invalid,
        make these states negative, e.g. by multiplying these states
        with minus one.

    The arguments `discard_neg_start` and `discard_all_neg` affect the
    counting of compounds in a given state, i.e. they affect
    :math:`N(t_0)`, :math:`N(t_0, t_0 + \Delta t)` and
    :math:`N(\{t \in [t_0, t_0 + \Delta t]\})`.  In general, :math:`N`
    is the sum of compounds that are in a valid state:
    :math:`N = \sum_i N_i`.  :math:`N_i` is :math:`1` if compound
    :math:`i` is in a valid state and :math:`0` otherwise.
    `discard_neg_start` and `discard_all_neg` now affect when a state is
    considered valid.

    If both, `discard_neg_start` and `discard_all_neg`, are ``False``
    (the default), all states are valid.

    .. math::

        N_i(t_0) = 1 ~\forall S_i(t_0)

    .. math::

        N_i(t_0, t_0 + \Delta t) =
        \begin{cases}
            1 & S_i(t_0 + \Delta t) = S_i(t_0) \\
            0 & \text{otherwise}
        \end{cases}

    .. math::

        N_i(\{t \in [t_0, t_0 + \Delta t]\}) =
        \begin{cases}
            1 & S_i(t) = S_i(t_0) ~\forall t \in [t_0, t_0 + \Delta t]\\
            0 & \text{otherwise}
        \end{cases}

    Here, :math:`S_i(t)` is the state of compound :math:`i` at time
    :math:`t`.

    If `discard_neg_start` is ``True``, transitions starting from
    negative states are discarded:

    .. math::

        N_i(t_0) = \begin{cases}
            1 & S_i(t_0) \geq 0 \\
            0 & S_i(t_0) < 0
        \end{cases}

    .. math::

        N_i(t_0, t_0 + \Delta t) =
        \begin{cases}
            1 & S_i(t_0) \geq 0 \text{ and } \\
              & S_i(t_0 + \Delta t) = S_i(t_0) \\
            0 & \text{otherwise}
        \end{cases}

    .. math::

        N_i(\{t \in [t_0, t_0 + \Delta t]\}) =
        \begin{cases}
            1 & S_i(t_0) \geq 0 \text{ and } \\
              & S_i(t) = S_i(t_0) ~\forall t \in [t_0, t_0 + \Delta t]\\
            0 & \text{otherwise}
        \end{cases}

    Thus, the resulting probability does not contain the probability to
    stay in a negative state.  However, transitions from positive to
    negative states are respected.  Hence, the probability to stay in a
    positive state is decreased if a transition to a negative states
    occurs (in addition to the decrease caused by transitions to other
    positive states).

    If `discard_all_neg` is ``True``, all negative states are discarded:

    .. math::

        N_i(t_0) = \begin{cases}
            1 & S_i(t) \geq 0 ~\forall t \in [t_0, t_0 + \Delta t] \\
            0 & \text{otherwise}
        \end{cases}

    .. math::

        N_i(t_0, t_0 + \Delta t) =
        \begin{cases}
            1 & S_i(t) \geq 0 ~\forall t \in [t_0, t_0 + \Delta t]
                \text{ and } \\
              & S_i(t_0 + \Delta t) = S_i(t_0) \\
            0 & \text{otherwise}
        \end{cases}

    .. math::

        N_i(\{t \in [t_0, t_0 + \Delta t]\}) =
        \begin{cases}
            1 & S_i(t) \geq 0 ~\forall t \in [t_0, t_0 + \Delta t]
                \text{ and } \\
              & S_i(t) = S_i(t_0) ~\forall t \in [t_0, t_0 + \Delta t]\\
            0 & \text{otherwise}
        \end{cases}

    This means transitions from or to negative states are completely
    discarded.  Thus, the resulting probability does neither contain the
    probability to stay in a negative state nor is it decreased by
    transitions from positive to negative states.

    **Lifetimes**

    The calculated probabilities can be used to calculate the average
    lifetime of the valid states, i.e. how long a compounds resides on
    average in a valid state.  A common approach is to fit the
    probability with a stretched exponential function :math:`f(t)` (e.g.
    using :func:`mdtools.functions.fit_kww`) and afterwards calculating
    the area below the fitted curve: [1]_:sup:`,` [2]_:sup:`,`
    [3]_:sup:`,` [4]_

    .. math::

        f(t) = \exp{\left[ -\left(\frac{t}{\tau_0}\right)^\beta \right]}

    .. math::

        \tau = \int_0^\infty f(t) \text{ d}t =
        \frac{\tau_0}{\beta} \Gamma\left(\frac{1}{\beta}\right)

    Here, :math:`\tau_0` and :math:`\beta` are the fit parameters,
    :math:`\tau` is the average lifetime and :math:`\Gamma(x)` is the
    gamma function.  For physically meaningful results, :math:`\beta`
    should be confined to :math:`0 < \beta \leq 1`.  For purely
    exponential decay (:math:`\beta = 1`), :math:`\tau = \tau_0`
    applies.  If the calculated probability fully decays to zero within
    the accessible range of lag times, one can alternatively numerically
    integrate the calculated probability directly.

    References
    ----------
    .. [1] R. Kohlrausch,
           Theorie des Elektrischen Rückstandes in der Leidener Flasche,
           Annalen der Physik, 1854, 167, 56-82.
    .. [2] G. Williams, D. C. Watts,
           `Non-Symmetrical Dielectric Relaxation Behaviour Arising from
           a Simple Empirical Decay Function
           <https://doi.org/10.1039/TF9706600080>`_,
           Transactions of the Faraday Society, 1970, 66, 80-85.
    .. [3] M. N. Berberan-Santos, E. N. Bodunov, B. Valeur,
           `Mathematical Functions for the Analysis of Luminescence
           Decays with Underlying Distributions 1. Kohlrausch Decay
           Function (Stretched Exponential)
           <https://doi.org/10.1016/j.chemphys.2005.04.006>`_,
           Chemical Physics, 2005, 315, 171-182.
    .. [4] D. C. Johnston,
           `Stretched Exponential Relaxation Arising from a Continuous
           Sum of Exponential Decays
           <https://doi.org/10.1103/PhysRevB.74.184430>`_,
           Physical Review B, 2006, 74, 184430.

    Examples
    --------
    >>> dtrj = np.array([[2, 2, 3, 3, 3]])
    >>> mdt.dtrj.remain_prob(dtrj)
    array([1.        , 0.75      , 0.33333333, 0.        , 0.        ])
    >>> mdt.dtrj.remain_prob(dtrj, continuous=True)
    array([1.        , 0.75      , 0.33333333, 0.        , 0.        ])
    >>> dtrj = np.array([[1, 3, 3, 3, 1]])
    >>> mdt.dtrj.remain_prob(dtrj)
    array([1.        , 0.5       , 0.33333333, 0.        , 1.        ])
    >>> mdt.dtrj.remain_prob(dtrj, continuous=True)
    array([1.        , 0.5       , 0.33333333, 0.        , 0.        ])

    The following examples were not checked to be mathematically
    correct!

    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1]])
    >>> mdt.dtrj.remain_prob(dtrj)
    array([1.        , 0.6       , 0.3       , 0.06666667, 0.        ,
           0.        ])
    >>> mdt.dtrj.remain_prob(dtrj, restart=3)
    array([1. , 0.5, 0.2, 0. , 0. , 0. ])
    >>> mdt.dtrj.remain_prob(dtrj, continuous=True)
    array([1.        , 0.6       , 0.3       , 0.06666667, 0.        ,
           0.        ])
    >>> mdt.dtrj.remain_prob(dtrj, discard_neg_start=True)
    array([1.        , 0.57894737, 0.375     , 0.09090909, 0.        ,
           0.        ])
    >>> mdt.dtrj.remain_prob(
    ...     dtrj, discard_neg_start=True, continuous=True
    ... )
    array([1.        , 0.57894737, 0.375     , 0.09090909, 0.        ,
           0.        ])
    >>> mdt.dtrj.remain_prob(dtrj, discard_all_neg=True)
    array([1.        , 0.73333333, 0.6       , 0.2       , 0.        ,
                  nan])
    >>> mdt.dtrj.remain_prob(
    ...     dtrj, discard_all_neg=True, continuous=True
    ... )
    array([1.        , 0.73333333, 0.6       , 0.2       , 0.        ,
                  nan])

    >>> dtrj = dtrj.T
    >>> mdt.dtrj.remain_prob(dtrj)
    array([1.        , 0.375     , 0.11111111, 0.16666667, 0.16666667])
    >>> mdt.dtrj.remain_prob(dtrj, restart=2)
    array([1.        , 0.58333333, 0.        , 0.33333333, 0.16666667])
    >>> mdt.dtrj.remain_prob(dtrj, continuous=True)
    array([1.        , 0.375     , 0.05555556, 0.        , 0.        ])
    >>> mdt.dtrj.remain_prob(dtrj, discard_neg_start=True)
    array([1.        , 0.375     , 0.16666667, 0.25      , 0.25      ])
    >>> mdt.dtrj.remain_prob(
    ...     dtrj, discard_neg_start=True, continuous=True
    ... )
    array([1.        , 0.375     , 0.08333333, 0.        , 0.        ])
    >>> mdt.dtrj.remain_prob(dtrj, discard_all_neg=True)
    array([1.        , 0.46153846, 0.28571429, 0.33333333, 0.        ])
    >>> mdt.dtrj.remain_prob(
    ...     dtrj, discard_all_neg=True, continuous=True
    ... )
    array([1.        , 0.46153846, 0.14285714, 0.        , 0.        ])
    """
    dtrj = mdt.check.dtrj(dtrj)
    n_cmps, n_frames = dtrj.shape
    dtrj = np.asarray(dtrj.T, order="C")
    if discard_neg_start and discard_all_neg:
        raise ValueError(
            "`discard_neg_start` and `discard_all_neg are` mutually exclusive"
        )

    prob = np.zeros(n_frames, dtype=np.uint32)
    if discard_neg_start:
        valid = np.zeros(n_cmps, dtype=bool)
        norm = np.zeros(n_frames, dtype=np.uint32)
    elif discard_all_neg:
        dtrj_valid = dtrj >= 0
        norm = np.zeros(n_frames, dtype=np.uint32)
    else:
        remain = np.zeros(n_cmps, dtype=bool)

    restarts = (t0 for t0 in range(0, n_frames - 1, restart))
    if verbose:
        proc = psutil.Process()
        n_restarts = int(np.ceil(n_frames / restart))
        restarts = mdt.rti.ProgressBar(restarts, total=n_restarts)
    for t0 in restarts:
        # When trying to understand the following code, read the "else"
        # parts fist.  Those are the simpler cases upon which the other
        # cases are built.
        if discard_neg_start:
            np.greater_equal(dtrj[t0], 0, out=valid)
            n_valid = np.count_nonzero(valid)
            if n_valid == 0:
                continue
            norm[1 : n_frames - t0] += n_valid
            dtrj_t0 = dtrj[t0][valid]
            if continuous:
                stay = np.ones(n_valid, dtype=bool)
                remain = np.zeros(n_valid, dtype=bool)
                for lag in range(1, n_frames - t0):
                    np.equal(dtrj_t0, dtrj[t0 + lag][valid], out=remain)
                    stay &= remain
                    n_stay = np.count_nonzero(stay)
                    if n_stay == 0:
                        break
                    prob[lag] += n_stay
            else:
                remain = np.zeros(n_valid, dtype=bool)
                for lag in range(1, n_frames - t0):
                    np.equal(dtrj_t0, dtrj[t0 + lag][valid], out=remain)
                    prob[lag] += np.count_nonzero(remain)
        elif discard_all_neg:
            valid = dtrj_valid[t0]  # This is a view, not a copy!
            if not np.any(valid):
                continue
            if continuous:
                stay = np.ones(n_cmps, dtype=bool)
                remain = np.zeros(n_cmps, dtype=bool)
                for lag in range(1, n_frames - t0):
                    valid &= dtrj_valid[t0 + lag]
                    n_valid = np.count_nonzero(valid)
                    if n_valid == 0:
                        continue
                    norm[lag] += n_valid
                    np.equal(dtrj[t0], dtrj[t0 + lag], out=remain)
                    stay &= remain
                    stay &= valid
                    prob[lag] += np.count_nonzero(stay)
                    # This loop must not be broken upon n_stay == 0,
                    # because otherwise the norm will be incorrect.
            else:
                for lag in range(1, n_frames - t0):
                    valid &= dtrj_valid[t0 + lag]
                    n_valid = np.count_nonzero(valid)
                    if n_valid == 0:
                        continue
                    norm[lag] += n_valid
                    remain = dtrj[t0][valid] == dtrj[t0 + lag][valid]
                    prob[lag] += np.count_nonzero(remain)
        else:
            if continuous:
                stay = np.ones(n_cmps, dtype=bool)
                for lag in range(1, n_frames - t0):
                    np.equal(dtrj[t0], dtrj[t0 + lag], out=remain)
                    stay &= remain
                    n_stay = np.count_nonzero(stay)
                    if n_stay == 0:
                        break
                    prob[lag] += n_stay
            else:
                for lag in range(1, n_frames - t0):
                    np.equal(dtrj[t0], dtrj[t0 + lag], out=remain)
                    prob[lag] += np.count_nonzero(remain)
        if verbose:
            restarts.set_postfix_str(
                "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
            )

    if discard_neg_start or discard_all_neg:
        if norm[0] != 0:
            raise ValueError(
                "The first element of norm is not zero but {}.  This should"
                " not have happened".format(norm[0])
            )
        norm[0] = 1
        valid = norm > 0
        prob = np.divide(prob, norm, where=valid)
        prob[~valid] = np.nan
    else:
        prob = prob / n_cmps
        prob /= mdt.dyn.n_restarts(n_frames=n_frames, restart=restart)

    if prob[0] != 0:
        raise ValueError(
            "The first element of p is not zero but {}.  This should not have"
            " happened".format(prob[0])
        )
    prob[0] = 1
    if np.any(prob > 1):
        raise ValueError(
            "At least one element of p is greater than one.  This should not"
            " have happened"
        )
    if np.any(prob < 0):
        raise ValueError(
            "At least one element of p is less than zero.  This should not"
            " have happened"
        )

    return prob


def remain_prob_discrete(  # noqa: C901
    dtrj1,
    dtrj2,
    restart=1,
    continuous=False,
    discard_neg_start=False,
    discard_all_neg=False,
    verbose=False,
):
    r"""
    Calculate the probability that a compound is in the same state as at
    time :math:`t_0` after a lag time :math:`\Delta t` resolved with
    respect to the states in second discrete trajectory.

    Take a discrete trajectory and calculate the probability to find a
    compound in the same state as at time :math:`t_0` after a lag time
    :math:`\Delta t` given that the compound was in a specific state of
    another discrete trajectory at time :math:`t_0`.

    Parameters
    ----------
    dtrj1, dtrj2 : array_like
        The discrete trajectories.  Arrays of shape ``(n, f)``, where
        ``n`` is the number of compounds and ``f`` is the number of
        frames.   The shape can also be ``(f,)``, in which case the
        array is expanded to shape ``(1, f)``.  Both arrays must have
        the same shape.   The elements of the arrays are interpreted as
        the indices of the states in which a given compound is at a
        given frame.
    restart : int, optional
        Restart every `restart` frames.  Determines how many restarting
        points :math:`t_0` are used for averaging.
    continuous : bool, optional
        If ``True``, compounds must continuously be in the same state
        without interruption in order to be counted (see notes of
        :func:`mdtools.dtrj.remain_prob`).
    discard_neg_start : bool, optional
        If ``True``, discard all transitions starting from a negative
        state (see notes of :func:`mdtools.dtrj.remain_prob`).  Must not
        be used together with `discard_all_neg`.
    discard_all_neg : bool, optional
        If ``True``, discard all negative states (see notes of
        :func:`mdtools.dtrj.remain_prob`).  Must not be used together
        with `discard_neg_start`.
    verbose : bool, optional
        If ``True`` print a progress bar.

    Returns
    -------
    prop : numpy.ndarray
        Array of shape ``(f, m)``, where ``m`` is the number of states
        in the second discrete trajectory.  The `ij`-th element of
        `prob` is the probability that a compound is in the same state
        of the first discrete trajectory as at time :math:`t_0` after a
        lag time of `i` frames given that the compound was in state `j`
        of the second discrete trajectory at time :math:`t_0`.

    See Also
    --------
    :func:`mdtools.dtrj.remain_prob` :
        Calculate the probability that a compound is in the same state
        as at time :math:`t_0` after a lag time :math:`\Delta t`

    Notes
    -----
    See :func:`mdtools.dtrj.remain_prob` for more details about the
    calculation of the probability to stay in the same state and about
    the meaning of the `continuous`, `discard_neg_start` and
    `discard_all_neg` arguments.

    If you parse the same discrete trajectory to `dtrj1` and `dtrj2` you
    will get the probability to stay in the same state for each
    individual state of the input trajectory.  If you want the average
    probability over all states, use :func:`mdtools.dtrj.remain_prob`.

    Examples
    --------
    >>> dtrj = np.array([[2, 2, 3, 3, 3]])
    >>> mdt.dtrj.remain_prob_discrete(dtrj, dtrj)
    array([[1. , 1. ],
           [0.5, 1. ],
           [0. , 1. ],
           [0. , nan],
           [0. , nan]])
    >>> mdt.dtrj.remain_prob_discrete(dtrj, dtrj, continuous=True)
    array([[1. , 1. ],
           [0.5, 1. ],
           [0. , 1. ],
           [0. , nan],
           [0. , nan]])
    >>> dtrj = np.array([[1, 3, 3, 3, 1]])
    >>> mdt.dtrj.remain_prob_discrete(dtrj, dtrj)
    array([[1.        , 1.        ],
           [0.        , 0.66666667],
           [0.        , 0.5       ],
           [0.        , 0.        ],
           [1.        ,        nan]])
    >>> mdt.dtrj.remain_prob_discrete(dtrj, dtrj, continuous=True)
    array([[1.        , 1.        ],
           [0.        , 0.66666667],
           [0.        , 0.5       ],
           [0.        , 0.        ],
           [0.        ,        nan]])

    The following examples were not checked to be mathematically
    correct!

    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1]])
    >>> # States of the discrete trajectory:
    >>> # [-2         ,-1         , 1         , 2         , 3         ]
    >>> # Because the first and second discrete trajectories are the
    >>> # same in this example, the resulting array contains as columns
    >>> # the probability to stay in the respective state as function of
    >>> # the lag time.
    >>> mdt.dtrj.remain_prob_discrete(dtrj, dtrj)
    array([[1.        , 1.        , 1.        , 1.        , 1.        ],
           [0.66666667,        nan, 0.        , 0.72727273, 0.75      ],
           [0.        ,        nan, 0.        , 0.44444444, 0.66666667],
           [0.        ,        nan, 0.        , 0.        , 0.5       ],
           [0.        ,        nan, 0.        , 0.        , 0.        ],
           [0.        ,        nan, 0.        , 0.        ,        nan]])
    >>> mdt.dtrj.remain_prob_discrete(dtrj, dtrj, restart=3)
    array([[1.  , 1.  , 1.  , 1.  , 1.  ],
           [1.  ,  nan, 0.  , 0.75, 1.  ],
           [0.  ,  nan, 0.  , 0.5 , 0.  ],
           [0.  ,  nan, 0.  , 0.  ,  nan],
           [0.  ,  nan, 0.  , 0.  ,  nan],
           [0.  ,  nan, 0.  , 0.  ,  nan]])
    >>> mdt.dtrj.remain_prob_discrete(dtrj, dtrj, continuous=True)
    array([[1.        , 1.        , 1.        , 1.        , 1.        ],
           [0.66666667,        nan, 0.        , 0.72727273, 0.75      ],
           [0.        ,        nan, 0.        , 0.44444444, 0.66666667],
           [0.        ,        nan, 0.        , 0.        , 0.5       ],
           [0.        ,        nan, 0.        , 0.        , 0.        ],
           [0.        ,        nan, 0.        , 0.        ,        nan]])
    >>> mdt.dtrj.remain_prob_discrete(
    ...     dtrj, dtrj, discard_neg_start=True
    ... )
    array([[1.        , 1.        , 1.        , 1.        , 1.        ],
           [       nan,        nan, 0.        , 0.72727273, 0.75      ],
           [       nan,        nan, 0.        , 0.44444444, 0.66666667],
           [       nan,        nan, 0.        , 0.        , 0.5       ],
           [       nan,        nan, 0.        , 0.        , 0.        ],
           [       nan,        nan, 0.        , 0.        ,        nan]])
    >>> mdt.dtrj.remain_prob_discrete(
    ...     dtrj, dtrj, discard_neg_start=True, continuous=True
    ... )
    array([[1.        , 1.        , 1.        , 1.        , 1.        ],
           [       nan,        nan, 0.        , 0.72727273, 0.75      ],
           [       nan,        nan, 0.        , 0.44444444, 0.66666667],
           [       nan,        nan, 0.        , 0.        , 0.5       ],
           [       nan,        nan, 0.        , 0.        , 0.        ],
           [       nan,        nan, 0.        , 0.        ,        nan]])
    >>> mdt.dtrj.remain_prob_discrete(dtrj, dtrj, discard_all_neg=True)
    array([[1.        , 1.        , 1.        , 1.        , 1.        ],
           [       nan,        nan, 0.        , 0.8       , 1.        ],
           [       nan,        nan, 0.        , 0.66666667, 1.        ],
           [       nan,        nan, 0.        , 0.        , 1.        ],
           [       nan,        nan, 0.        ,        nan,        nan],
           [       nan,        nan,        nan,        nan,        nan]])
    >>> mdt.dtrj.remain_prob_discrete(
    ...     dtrj, dtrj, discard_all_neg=True, continuous=True
    ... )
    array([[1.        , 1.        , 1.        , 1.        , 1.        ],
           [       nan,        nan, 0.        , 0.8       , 1.        ],
           [       nan,        nan, 0.        , 0.66666667, 1.        ],
           [       nan,        nan, 0.        , 0.        , 1.        ],
           [       nan,        nan, 0.        ,        nan,        nan],
           [       nan,        nan,        nan,        nan,        nan]])

    >>> dtrj = dtrj.T
    >>> mdt.dtrj.remain_prob_discrete(dtrj, dtrj)
    array([[1.        , 1.        , 1.        , 1.        , 1.        ],
           [0.375     ,        nan, 0.25      , 0.41666667,        nan],
           [0.        ,        nan, 0.        , 0.22222222,        nan],
           [0.        ,        nan, 0.5       , 0.16666667,        nan],
           [0.        ,        nan, 1.        , 0.        ,        nan]])
    >>> mdt.dtrj.remain_prob_discrete(dtrj, dtrj, restart=2)
    array([[1.        , 1.        , 1.        , 1.        , 1.        ],
           [0.75      ,        nan, 0.        , 0.66666667,        nan],
           [0.        ,        nan, 0.        , 0.        ,        nan],
           [0.        ,        nan, 1.        , 0.33333333,        nan],
           [0.        ,        nan, 1.        , 0.        ,        nan]])
    >>> mdt.dtrj.remain_prob_discrete(dtrj, dtrj, continuous=True)
    array([[1.        , 1.        , 1.        , 1.        , 1.        ],
           [0.375     ,        nan, 0.25      , 0.41666667,        nan],
           [0.        ,        nan, 0.        , 0.11111111,        nan],
           [0.        ,        nan, 0.        , 0.        ,        nan],
           [0.        ,        nan, 0.        , 0.        ,        nan]])
    >>> mdt.dtrj.remain_prob_discrete(
    ...     dtrj, dtrj, discard_neg_start=True
    ... )
    array([[1.        , 1.        , 1.        , 1.        , 1.        ],
           [       nan,        nan, 0.25      , 0.41666667,        nan],
           [       nan,        nan, 0.        , 0.22222222,        nan],
           [       nan,        nan, 0.5       , 0.16666667,        nan],
           [       nan,        nan, 1.        , 0.        ,        nan]])
    >>> mdt.dtrj.remain_prob_discrete(
    ...     dtrj, dtrj, discard_neg_start=True, continuous=True
    ... )
    array([[1.        , 1.        , 1.        , 1.        , 1.        ],
           [       nan,        nan, 0.25      , 0.41666667,        nan],
           [       nan,        nan, 0.        , 0.11111111,        nan],
           [       nan,        nan, 0.        , 0.        ,        nan],
           [       nan,        nan, 0.        , 0.        ,        nan]])
    >>> mdt.dtrj.remain_prob_discrete(dtrj, dtrj, discard_all_neg=True)
    array([[1.        , 1.        , 1.        , 1.        , 1.        ],
           [       nan,        nan, 0.5       , 0.45454545,        nan],
           [       nan,        nan, 0.        , 0.33333333,        nan],
           [       nan,        nan,        nan, 0.33333333,        nan],
           [       nan,        nan,        nan, 0.        ,        nan]])
    >>> mdt.dtrj.remain_prob_discrete(
    ...     dtrj, dtrj, discard_all_neg=True, continuous=True
    ... )
    array([[1.        , 1.        , 1.        , 1.        , 1.        ],
           [       nan,        nan, 0.5       , 0.45454545,        nan],
           [       nan,        nan, 0.        , 0.16666667,        nan],
           [       nan,        nan,        nan, 0.        ,        nan],
           [       nan,        nan,        nan, 0.        ,        nan]])
    """  # noqa: W505
    dtrj1 = mdt.check.dtrj(dtrj1)
    dtrj2 = mdt.check.dtrj(dtrj2)
    n_cmps, n_frames = dtrj1.shape
    if dtrj1.shape != dtrj2.shape:
        raise ValueError("Both trajectories must have the same shape")
    dtrj1 = np.asarray(dtrj1.T, order="C")
    dtrj2 = np.asarray(dtrj2.T, order="C")
    if discard_neg_start and discard_all_neg:
        raise ValueError(
            "`discard_neg_start` and `discard_all_neg are` mutually exclusive"
        )

    # Reorder the states in the second trajectory such that they start
    # at zero and no intermediate states are missing.
    dtrj2 = mdt.nph.sequenize(dtrj2, step=np.uint8(1), start=np.uint8(0))
    n_states = np.max(dtrj2) + 1
    if np.min(dtrj2) != 0:
        raise ValueError(
            "The minimum of the reordered second trajectory is not zero.  This"
            " should not have happened"
        )

    prob = np.zeros((n_frames, n_states), dtype=np.uint32)
    norm = np.zeros((n_frames, n_states), dtype=np.uint32)
    if discard_neg_start:
        valid = np.zeros(n_cmps, dtype=bool)
    elif discard_all_neg:
        dtrj1_valid = dtrj1 >= 0
    else:
        remain = np.zeros(n_cmps, dtype=bool)

    restarts = (t0 for t0 in range(0, n_frames - 1, restart))
    if verbose:
        proc = psutil.Process()
        restarts = mdt.rti.ProgressBar(restarts, total=n_frames - 2)
    for t0 in restarts:
        # When trying to understand the following code, read the "else"
        # parts fist.  Those are the simpler cases upon which the other
        # cases are built.  Also look at the code of the function
        # `mdtools.dtrj.remain_prob`.
        if discard_neg_start:
            np.greater_equal(dtrj1[t0], 0, out=valid)
            n_valid = np.count_nonzero(valid)
            if n_valid == 0:
                continue
            dtrj1_t0 = dtrj1[t0][valid]
            dtrj2_t0 = dtrj2[t0][valid]
            bin_ix_u, counts = np.unique(dtrj2_t0, return_counts=True)
            masks = dtrj2_t0 == bin_ix_u[:, None]
            norm[1 : n_frames - t0][:, bin_ix_u] += counts.astype(np.uint32)
            if continuous:
                stay = np.ones(n_valid, dtype=bool)
                remain = np.zeros(n_valid, dtype=bool)
                for lag in range(1, n_frames - t0):
                    np.equal(dtrj1_t0, dtrj1[t0 + lag][valid], out=remain)
                    stay &= remain
                    if not np.any(stay):
                        break
                    for i, b in enumerate(bin_ix_u):
                        prob[lag][b] += np.count_nonzero(stay[masks[i]])
            else:
                remain = np.zeros(n_valid, dtype=bool)
                for lag in range(1, n_frames - t0):
                    np.equal(dtrj1_t0, dtrj1[t0 + lag][valid], out=remain)
                    for i, b in enumerate(bin_ix_u):
                        prob[lag][b] += np.count_nonzero(remain[masks[i]])
        elif discard_all_neg:
            valid = dtrj1_valid[t0]  # This is a view, not a copy!
            if not np.any(valid):
                continue
            if continuous:
                stay = np.ones(n_cmps, dtype=bool)
                remain = np.zeros(n_cmps, dtype=bool)
                mask = np.zeros(n_cmps, dtype=bool)
                for lag in range(1, n_frames - t0):
                    valid &= dtrj1_valid[t0 + lag]
                    if not np.any(valid):
                        continue
                    bin_ix_u, counts = np.unique(
                        dtrj2[t0][valid], return_counts=True
                    )
                    np.equal(dtrj1[t0], dtrj1[t0 + lag], out=remain)
                    stay &= remain
                    stay &= valid
                    # This loop must not be broken upon n_stay == 0,
                    # because otherwise the norm will be incorrect.
                    for i, b in enumerate(bin_ix_u):
                        np.equal(dtrj2[t0], b, out=mask)
                        prob[lag][b] += np.count_nonzero(stay[mask])
                        norm[lag][b] += counts[i]
            else:
                for lag in range(1, n_frames - t0):
                    valid &= dtrj1_valid[t0 + lag]
                    n_valid = np.count_nonzero(valid)
                    if n_valid == 0:
                        continue
                    bin_ix_u, counts = np.unique(
                        dtrj2[t0][valid], return_counts=True
                    )
                    mask = np.zeros(n_valid, dtype=bool)
                    remain = dtrj1[t0][valid] == dtrj1[t0 + lag][valid]
                    for i, b in enumerate(bin_ix_u):
                        np.equal(dtrj2[t0][valid], b, out=mask)
                        prob[lag][b] += np.count_nonzero(remain[mask])
                        norm[lag][b] += counts[i]
        else:
            bin_ix_u, counts = np.unique(dtrj2[t0], return_counts=True)
            masks = dtrj2[t0] == bin_ix_u[:, None]
            norm[1 : n_frames - t0][:, bin_ix_u] += counts.astype(np.uint32)
            if continuous:
                stay = np.ones(n_cmps, dtype=bool)
                for lag in range(1, n_frames - t0):
                    np.equal(dtrj1[t0], dtrj1[t0 + lag], out=remain)
                    stay &= remain
                    if not np.any(stay):
                        break
                    for i, b in enumerate(bin_ix_u):
                        prob[lag][b] += np.count_nonzero(stay[masks[i]])
            else:
                for lag in range(1, n_frames - t0):
                    np.equal(dtrj1[t0], dtrj1[t0 + lag], out=remain)
                    for i, b in enumerate(bin_ix_u):
                        prob[lag][b] += np.count_nonzero(remain[masks[i]])
        if verbose:
            restarts.set_postfix_str(
                "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
            )

    if np.any(norm[0] != 0):
        raise ValueError(
            "At least one element in the first row of norm is not zero.  This"
            " should not have happened"
        )
    norm[0] = 1
    valid = norm > 0
    prob = np.divide(prob, norm, where=valid)
    prob[~valid] = np.nan

    if np.any(prob[0] != 0):
        raise ValueError(
            "At least one element in the first row of p is not zero.  This"
            " should not have happened"
        )
    prob[0] = 1
    if np.any(prob > 1):
        raise ValueError(
            "At least one element of p is greater than one.  This should not"
            " have happened"
        )
    if np.any(prob < 0):
        raise ValueError(
            "At least one element of p is less than zero.  This should not"
            " have happened"
        )

    return prob
