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
