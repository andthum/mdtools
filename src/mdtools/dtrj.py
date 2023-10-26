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


def get_ax(ax_cmp=None, ax_fr=None):
    """
    Return the frame/compound axis for a discrete trajectory given its
    compound/frame axis.

    Parameters
    ----------
    ax_cmp : {-2, -1, 0, 1} or None, optional
        Compound axis.  Either `ax_cmp` or `ax_fr` must be given.
    ax_fr : {-2, -1, 0, 1} or None, optional
        Frame axis.  Either `ax_cmp` or `ax_fr` must be given.

    Returns
    -------
    ax_cmp : int
        The compound axis for a discrete trajectory.
    ax_fr : int
        The frame axis for a discrete trajectory.

    Raises
    ------
    :exc:`numpy.exceptions.AxisError` :
        If the given compound/frame axis is out of bounds for a discrete
        trajectory with two dimensions.

    Notes
    -----
    All returned axis indices will be positive.

    Examples
    --------
    >>> mdt.dtrj.get_ax(ax_cmp=0)
    (0, 1)
    >>> mdt.dtrj.get_ax(ax_cmp=1)
    (1, 0)
    >>> mdt.dtrj.get_ax(ax_cmp=-1)
    (1, 0)
    >>> mdt.dtrj.get_ax(ax_cmp=-2)
    (0, 1)

    >>> mdt.dtrj.get_ax(ax_fr=0)
    (1, 0)
    >>> mdt.dtrj.get_ax(ax_fr=1)
    (0, 1)
    >>> mdt.dtrj.get_ax(ax_fr=-1)
    (0, 1)
    >>> mdt.dtrj.get_ax(ax_fr=-2)
    (1, 0)
    """
    ax_is_none = [ax is None for ax in (ax_cmp, ax_fr)]
    if all(ax_is_none) or not any(ax_is_none):
        raise ValueError(
            "Either `ax_fr` ({}) or `ax_cmp` ({}) must be"
            " given".format(ax_fr, ax_cmp)
        )

    dtrj_ndim = 2
    axis = ax_cmp if ax_cmp is not None else ax_fr
    if axis >= dtrj_ndim or axis < -dtrj_ndim:
        raise np.AxisError(
            axis=axis, ndim=dtrj_ndim, msg_prefix="Discrete trajectory"
        )

    dtrj_axes = list(range(dtrj_ndim))
    if ax_cmp is not None:
        compound_axis = dtrj_axes.pop(ax_cmp)
        frame_axis = dtrj_axes[0]
    elif ax_fr is not None:
        frame_axis = dtrj_axes.pop(ax_fr)
        compound_axis = dtrj_axes[0]
    else:
        raise ValueError(
            "Either `ax_fr` ({}) or `ax_cmp` ({}) must be"
            " given".format(ax_fr, ax_cmp)
        )
    return compound_axis, frame_axis


def locate_trans(
    dtrj,
    axis=-1,
    pin="end",
    discard_neg=None,
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
    axis : int, optional
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
    discard_neg : {None, "start", "end", "both"}, optional
        Whether to locate all state transitions (``None``) or whether to
        discard state transitions starting from a negative state
        (``"start"``) or ending in a negative state (``"end"``).  If set
        to ``"both"``, all state transitions starting from or ending in
        a negative state will be discarded.
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
        `discard_neg`, `trans_type`, `wrap` or `mic`.
    tlft : bool, optional
        Treat Last Frame as Transition.  If ``True``, treat the last
        frame as the start of a state transition.  Has no effect if
        `pin` is set to ``"end"``.  Must not be used together with
        `discard_neg`, `trans_type`, `wrap` or `mic`.
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
        discard_neg=discard_neg,
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
    discard_neg=None,
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
    discard_neg : {None, "start", "end", "both"}, optional
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
        discard_neg=discard_neg,
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
    except TypeError:  # `trans_type` is not iterable.
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


def trans_rate(
    dtrj,
    axis=-1,
    discard_neg_start=False,
    discard_all_neg=False,
    return_cmp_ix=False,
):
    r"""
    Calculate the transition rate for each compound averaged over all
    states.

    Parameters
    ----------
    dtrj : array_like
        Array containing the discrete trajectory.
    axis : int, optional
        The axis along which to search for state transitions.  For
        ordinary discrete trajectories with shape ``(n, f)`` or
        ``(f,)``, where ``n`` is the number of compounds and ``f`` is
        the number of frames, set `axis` to ``-1``.  If you parse a
        transposed discrete trajectory of shape ``(f, n)``, set `axis`
        to ``0``.
    discard_neg_start : bool, optional
        If ``True``, discard all transitions starting from a negative
        state (see notes).  This is equivalent to discarding the
        lifetimes of all negative states when calculating state
        lifetimes with :func:`mdtools.dtrj.lifetimes`.  Has no effect if
        `discard_all_neg` is ``True`` .
    discard_all_neg : bool, optional
        If ``True``, discard all transitions starting from or ending in
        a negative state (see notes).  This is equivalent to discarding
        the lifetimes of all negative states and of all states that are
        followed by a negative state when calculating state lifetimes
        with :func:`mdtools.dtrj.lifetimes`.
    return_cmp_ix : bool, optional
        If ``True``, return the compound indices associated with the
        returned transition rates.

    Returns
    -------
    trans_rates : numpy.ndarray
        1-dimensional array containing the transition rate for each
        compound averaged over all states.
    cmp_ix : numpy.ndarray
        1-dimensional array of the same shape as `trans_rate` containing
        the corresponding compound indices.  Only returned if
        `return_cmp_ix` is ``True``.

    See Also
    --------
    :func:`mdtools.dtrj.trans_rate_per_state` :
        Calculate the transition rate for each state averaged over all
        compounds
    :func:`mdtools.dtrj.lifetimes` :
        Calculate the state lifetimes for each compound
    :func:`mdtools.dtrj.kaplan_meier` :
        Estimate the state survival function using the Kaplan-Meier
        estimator
    :func:`mdtools.dtrj.remain_prob` :
        Calculate the probability that a compound is still (or again) in
        the same state as at time :math:`t_0` after a lag time
        :math:`\Delta t`

    Notes
    -----
    Transitions rates are calculated by simply counting the number of
    valid state transitions for each compound and dividing this number
    by the number of valid frames.

    To calculate the transition rate averaged over all states and all
    compounds simply do:

    .. code-block:: python

        n_trans_tot = np.count_nonzero(mdt.dtrj.locate_trans(dtrj))
        n_frames_tot = dtrj.size
        trans_rate = n_trans_tot / n_frames_tot

    The inverse of the transition rate gives an estimate for the average
    state lifetime.  In contrast to calculating the average lifetime by
    simply counting how many frames a given compound stays in a given
    state (as done by :func:`mdtools.dtrj.lifetimes`), this method is
    not biased to lower lifetimes.

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
    counting of state transitions and frames.

    If both, `discard_neg_start` and `discard_all_neg`, are ``False``
    (the default), all state state transitions are counted and divided
    by the total number of frames.

    If `discard_neg_start` is ``True``, transitions starting from
    negative states are discarded and the total number of frames is
    reduced by the number of frames that a given compound resides in
    negative states.

    If `discard_all_neg` is ``True``, all negative states are discarded,
    i.e. transitions starting from or ending in a negative state are
    discarded.  The total number of frames is reduced by the number of
    frames that a given compound resides in negative states and by the
    number of frames that a given compound resides in positive states
    that are followed by negative states.

    Examples
    --------
    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2]])
    >>> rates, cmp_ix = mdt.dtrj.trans_rate(dtrj, return_cmp_ix=True)
    >>> rates  # 2 transitions / 6 frames for each compound.
    array([0.33333333, 0.33333333, 0.33333333, 0.33333333])
    >>> cmp_ix
    array([0, 1, 2, 3])
    >>> rates, cmp_ix = mdt.dtrj.trans_rate(
    ...     dtrj, axis=0, return_cmp_ix=True
    ... )
    >>> rates
    array([0.75, 0.25, 0.25, 0.5 , 0.25, 0.5 ])
    >>> cmp_ix
    array([0, 1, 2, 3, 4, 5])

    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2],
    ...                  [6, 6, 6, 6, 6, 6]])
    >>> rates, cmp_ix = mdt.dtrj.trans_rate(dtrj, return_cmp_ix=True)
    >>> rates
    array([0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.        ])
    >>> cmp_ix
    array([0, 1, 2, 3, 4])
    >>> rates, cmp_ix = mdt.dtrj.trans_rate(
    ...     dtrj, axis=0, return_cmp_ix=True
    ... )
    >>> rates
    array([0.8, 0.4, 0.4, 0.6, 0.4, 0.6])
    >>> cmp_ix
    array([0, 1, 2, 3, 4, 5])

    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1,  2,  2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1],
    ...                  [-6, -6, -6, -6, -6, -6],
    ...                  [ 6,  6,  6,  6,  6,  6]])
    >>> ax = -1
    >>> rates, cmp_ix = mdt.dtrj.trans_rate(
    ...     dtrj, axis=ax, return_cmp_ix=True
    ... )
    >>> rates
    array([0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
           0.        , 0.        ])
    >>> cmp_ix
    array([0, 1, 2, 3, 4, 5, 6])
    >>> rates_start, cmp_ix_start = mdt.dtrj.trans_rate(
    ...     dtrj, axis=ax, discard_neg_start=True, return_cmp_ix=True
    ... )
    >>> rates_start
    array([0.25      , 0.25      , 0.33333333, 0.5       , 0.4       ,
           0.        ])
    >>> cmp_ix_start
    array([0, 1, 2, 3, 4, 6])
    >>> rates_all, cmp_ix_all = mdt.dtrj.trans_rate(
    ...     dtrj, axis=ax, discard_all_neg=True, return_cmp_ix=True
    ... )
    >>> rates_all
    array([0.        , 0.25      , 0.33333333, 1.        , 1.        ,
           0.        ])
    >>> cmp_ix_all
    array([0, 1, 2, 3, 4, 6])

    >>> ax = 0
    >>> rates, cmp_ix = mdt.dtrj.trans_rate(
    ...     dtrj, axis=ax, return_cmp_ix=True
    ... )
    >>> rates
    array([0.71428571, 0.57142857, 0.57142857, 0.71428571, 0.71428571,
           0.85714286])
    >>> cmp_ix
    array([0, 1, 2, 3, 4, 5])
    >>> rates_start, cmp_ix_start = mdt.dtrj.trans_rate(
    ...     dtrj, axis=ax, discard_neg_start=True, return_cmp_ix=True
    ... )
    >>> rates_start
    array([0.6       , 0.5       , 0.4       , 0.66666667, 0.6       ,
           0.75      ])
    >>> cmp_ix_start
    array([0, 1, 2, 3, 4, 5])
    >>> rates_all, cmp_ix_all = mdt.dtrj.trans_rate(
    ...     dtrj, axis=ax, discard_all_neg=True, return_cmp_ix=True
    ... )
    >>> rates_all
    array([0.5       , 0.33333333, 0.25      , 0.6       , 0.33333333,
           0.66666667])
    >>> cmp_ix_all
    array([0, 1, 2, 3, 4, 5])
    """
    dtrj = mdt.check.dtrj(dtrj)
    ax_cmp, ax_fr = mdt.dtrj.get_ax(ax_fr=axis)
    n_cmps, n_frames = dtrj.shape[ax_cmp], dtrj.shape[ax_fr]

    # Get all state transitions.
    trans_ix = mdt.dtrj.trans_ix(dtrj, axis=ax_fr, pin="start")

    if discard_all_neg and ax_cmp != 0:
        # Sort position of state transitions by compound indices.
        ix_sort = np.argsort(trans_ix[ax_cmp], kind="stable")
        trans_ix = tuple(tix[ix_sort] for tix in trans_ix)
        del ix_sort

    # Get all compounds that undergo state transitions.
    cmp_ix_trans = trans_ix[ax_cmp]
    if discard_neg_start or discard_all_neg:
        # Discard all transitions starting from a negative state.
        valid = dtrj[trans_ix] >= 0  # Valid initial states.
        # Get the number of "valid" frames for each compound, i.e. the
        # number of frames in which a compound resides in a positive
        # state.
        n_frames = np.count_nonzero(dtrj >= 0, axis=ax_fr)
        # Get compounds that are always in a negative state.
        cmp_ix_always_neg = np.flatnonzero(n_frames == 0)
        if discard_all_neg:
            # Discard all transition ending in a negative state.
            trans_ix_end = list(trans_ix)
            trans_ix_end[ax_fr] += 1
            trans_ix_end = tuple(trans_ix_end)
            valid_end = dtrj[trans_ix_end] >= 0
            # Remove frames in which a compound resides in a positive
            # state that is followed by a negative state from the number
            # of valid frames:
            # Start points of single compound trajectories.
            trj_starts = np.diff(trans_ix[ax_cmp])
            trj_starts = np.insert(trj_starts, 0, 1)
            trj_starts = np.flatnonzero(trj_starts)
            # Frame indices at which state transitions end.
            ix_end = trans_ix_end[ax_fr]
            ix_end = np.insert(ix_end, trj_starts, 0)
            # Invalid frames are those between a positive initial state
            # and a negative final state.
            invalid_lt = valid & ~valid_end
            # Lifetimes/number of frames of positive states that are
            # followed by a negative state.
            lifetimes = np.diff(ix_end)
            # Negative values in `lifetimes` indicate the start of a new
            # compound trajectory.
            lifetimes = lifetimes[lifetimes > 0]
            lifetimes = lifetimes[invalid_lt]
            np.subtract.at(n_frames, cmp_ix_trans[invalid_lt], lifetimes)
            del trj_starts, ix_end, invalid_lt, lifetimes
            # Finish discarding all transition ending in a negative
            # state.
            valid &= valid_end
            del valid_end, trans_ix_end
        cmp_ix_trans = cmp_ix_trans[valid]
        del valid
    else:
        n_frames = np.full(n_cmps, n_frames)
        cmp_ix_always_neg = np.array([], dtype=int)
    del dtrj, trans_ix

    # Number of transitions for each compound that undergoes state
    # transition(s).
    cmp_ix_trans, n_trans = np.unique(cmp_ix_trans, return_counts=True)
    if np.any(np.isin(cmp_ix_always_neg, cmp_ix_trans)):
        raise ValueError(
            "At least one compound that is considered to be always in a"
            " negative state is also listed as compound that undergoes valid"
            " state transition(s).  This should not have happened."
        )
    # Number of transitions for each compound in the discrete
    # trajectory.
    n_trans_per_cmp = np.zeros(n_cmps, dtype=np.uint32)
    n_trans_per_cmp[cmp_ix_trans] = n_trans
    n_trans_per_cmp = np.delete(n_trans_per_cmp, cmp_ix_always_neg)
    n_frames = np.delete(n_frames, cmp_ix_always_neg)

    # Transition rates.
    trans_rate_per_cmp = n_trans_per_cmp / n_frames
    if np.any(trans_rate_per_cmp > 1):
        raise ValueError(
            "At least one compound has a transition rate greater than one."
            "  This should not have happened"
        )
    if np.any(trans_rate_per_cmp < 0):
        raise ValueError(
            "At least one compound has a negative transition rate.  This"
            " should not have happened"
        )

    if return_cmp_ix:
        cmp_ix = np.delete(np.arange(n_cmps), cmp_ix_always_neg)
        return trans_rate_per_cmp, cmp_ix
    else:
        return trans_rate_per_cmp


def trans_rate_per_state(
    dtrj,
    axis=-1,
    discard_neg_start=False,
    discard_all_neg=False,
    return_states=False,
):
    r"""
    Calculate the transition rate for each state averaged over all
    compounds.

    Parameters
    ----------
    dtrj : array_like
        Array containing the discrete trajectory.
    axis : int, optional
        The axis along which to search for state transitions.  For
        ordinary discrete trajectories with shape ``(n, f)`` or
        ``(f,)``, where ``n`` is the number of compounds and ``f`` is
        the number of frames, set `axis` to ``-1``.  If you parse a
        transposed discrete trajectory of shape ``(f, n)``, set `axis`
        to ``0``.
    discard_neg_start : bool, optional
        If ``True``, discard all transitions starting from a negative
        state (see notes of :func:`mdtools.dtrj.trans_rate`).  This is
        equivalent to discarding the lifetimes of all negative states
        when calculating state lifetimes with
        :func:`mdtools.dtrj.lifetimes_per_state`.  Has no effect if
        `discard_all_neg` is ``True`` .
    discard_all_neg : bool, optional
        If ``True``, discard all transitions starting from or ending in
        a negative state (see notes :func:`mdtools.dtrj.trans_rate`).
        This is equivalent to discarding the lifetimes of all negative
        states and of all states that are followed by a negative state
        when calculating state lifetimes with
        :func:`mdtools.dtrj.lifetimes_per_state`.
    return_states : bool, optional
        If ``True``, return the state indices associated with the
        returned transition rates.

    Returns
    -------
    trans_rates : numpy.ndarray
        1-dimensional array containing the transition rates for each
        valid state.
    states : numpy.ndarray
        1-dimensional array of the same shape as `trans_rates`
        containing the corresponding state indices.  Only returned if
        `return_states` is ``True``.

    See Also
    --------
    :func:`mdtools.dtrj.trans_rate` :
        Calculate the transition rate for each compound averaged over
        all states
    :func:`mdtools.dtrj.remain_prob_discrete` :
        Calculate the probability that a compound is still (or again) in
        the same state as at time :math:`t_0` after a lag time
        :math:`\Delta t` resolved with respect to the states in second
        discrete trajectory
    :func:`mdtools.dtrj.lifetimes_per_state` :
        Calculate the state lifetimes for each state

    Notes
    -----
    Transitions rates are calculated for a given valid state by simply
    counting the number of valid transitions out of this state and
    dividing this number by the number of valid frames that compounds
    have spent in this state.

    To calculate the transition rate averaged over all states and all
    compounds simply do:

    .. code-block:: python

        n_trans_tot = np.count_nonzero(mdt.dtrj.locate_trans(dtrj))
        n_frames_tot = dtrj.size
        trans_rate = n_trans_tot / n_frames_tot

    The inverse of the transition rate gives an estimate for the average
    state lifetime.  In contrast to calculating the average lifetime by
    simply counting how many frames a given compound stays in a given
    state (as done by :func:`mdtools.dtrj.lifetimes_per_state`), this
    method is not biased to lower lifetimes.

    See :func:`mdtools.dtrj.trans_rate` for details about valid and
    invalid states (`discard_neg_start` and `discard_all_neg`).

    Examples
    --------
    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2]])
    >>> trans_rates, states = mdt.dtrj.trans_rate_per_state(
    ...     dtrj, return_states=True
    ... )
    >>> trans_rates
    array([0.75, 0.25, 0.25])
    >>> states
    array([1, 2, 3])
    >>> trans_rates, states = mdt.dtrj.trans_rate_per_state(
    ...     dtrj, axis=0, return_states=True
    ... )
    >>> trans_rates
    array([0.75      , 0.375     , 0.33333333])
    >>> states
    array([1, 2, 3])

    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2],
    ...                  [6, 6, 6, 6, 6, 6]])
    >>> trans_rates, states = mdt.dtrj.trans_rate_per_state(
    ...     dtrj, return_states=True
    ... )
    >>> trans_rates
    array([0.75, 0.25, 0.25, 0.  ])
    >>> states
    array([1, 2, 3, 6])
    >>> trans_rates, states = mdt.dtrj.trans_rate_per_state(
    ...     dtrj, axis=0, return_states=True
    ... )
    >>> trans_rates
    array([1.        , 0.625     , 0.58333333, 0.        ])
    >>> states
    array([1, 2, 3, 6])

    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1,  2,  2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1],
    ...                  [-6, -6, -6, -6, -6, -6],
    ...                  [ 6,  6,  6,  6,  6,  6]])
    >>> ax = -1
    >>> trans_rates, states = mdt.dtrj.trans_rate_per_state(
    ...     dtrj, axis=ax, return_states=True
    ... )
    >>> trans_rates
    array([0.        , 0.33333333, 0.        , 0.8       , 0.        ,
           0.25      , 0.25      , 0.        ])
    >>> states
    array([-6, -2, -1,  1,  2,  3,  4,  6])
    >>> trans_rates_start, states_start = mdt.dtrj.trans_rate_per_state(
    ...     dtrj, axis=ax, discard_neg_start=True, return_states=True
    ... )
    >>> trans_rates_start
    array([0.8 , 0.  , 0.25, 0.25, 0.  ])
    >>> states_start
    array([1, 2, 3, 4, 6])
    >>> trans_rates_all, states_all = mdt.dtrj.trans_rate_per_state(
    ...     dtrj, axis=ax, discard_all_neg=True, return_states=True
    ... )
    >>> trans_rates_all
    array([0.75      , 0.        , 0.22222222, 0.        ])
    >>> states_all
    array([1, 2, 3, 6])

    >>> ax = 0
    >>> trans_rates, states = mdt.dtrj.trans_rate_per_state(
    ...     dtrj, axis=ax, return_states=True
    ... )
    >>> trans_rates
    array([1.        , 0.83333333, 1.        , 0.8       , 1.        ,
           0.58333333, 1.        , 0.        ])
    >>> states
    array([-6, -2, -1,  1,  2,  3,  4,  6])
    >>> trans_rates_start, states_start = mdt.dtrj.trans_rate_per_state(
    ...     dtrj, axis=ax, discard_neg_start=True, return_states=True
    ... )
    >>> trans_rates_start
    array([0.8       , 1.        , 0.58333333, 1.        , 0.        ])
    >>> states_start
    array([1, 2, 3, 4, 6])

    .. The following example *sometimes* fails when running Sphinx's
        `make doctest` as GitHub Action on Ubuntu 22.04 with Python 3.9.
        This seems somehow related to the Action's cache, but we could
        not resolve this so far.  With other operating systems and
        Python versions the example succeeds.  For now, we skip this
        test when running on Linux with Python 3.9.

    .. doctest::
        :skipif: platform.system() == "Linux" and \
                 platform.python_version().startswith("3.9.")

        >>> trans_rates_all, states_all = mdt.dtrj.trans_rate_per_state(
        ...     dtrj, axis=ax, discard_all_neg=True, return_states=True
        ... )
        >>> trans_rates_all
        array([1.        , 0.58333333, 0.        ])
        >>> states_all
        array([1, 3, 6])
    """
    dtrj = mdt.check.dtrj(dtrj)
    ax_cmp, ax_fr = mdt.dtrj.get_ax(ax_fr=axis)
    n_cmps, n_frames_tot = dtrj.shape[ax_cmp], dtrj.shape[ax_fr]

    # Unique state indices.
    states_uniq = np.unique(dtrj)
    if discard_neg_start or discard_all_neg:
        # Discard all transitions starting from a negative state.
        states_uniq = states_uniq[states_uniq >= 0]
    if discard_all_neg:
        # Discard all transition ending in a negative state.
        # Invalid trajectory frames.
        dtrj_invalid = dtrj < 0
        # All state transitions.
        trans2neg = mdt.dtrj.locate_trans(dtrj, pin="end", axis=ax_fr)
        # End points of all transitions ending in a negative state.
        trans2neg &= dtrj_invalid
        # Start points of all transitions ending in a negative state.
        trans2neg = np.roll(trans2neg, shift=-1, axis=ax_fr)
        # Start points of transitions starting from a positive state and
        # ending in a negative state.
        trans2neg &= ~dtrj_invalid
        del dtrj_invalid

    trans_rates = np.full(len(states_uniq), np.nan, dtype=np.float64)
    for six, state in enumerate(states_uniq):
        # Frames at which compounds are in `state`.
        dtrj_state = dtrj == state
        # Transitions into and out of `state`.
        trans = np.diff(dtrj_state.astype(np.int8), axis=ax_fr)
        # Number of transitions that lead out of `state`.
        n_trans = np.count_nonzero(trans == -1)
        # Number of frames that compounds have spent in `state`.
        n_frames = np.count_nonzero(dtrj_state)
        if discard_all_neg:
            # Start points of transitions starting in `state` and ending
            # in a negative state.
            trans_invalid = dtrj_state & trans2neg
            # Subtract the number of invalid transitions.
            n_trans_invalid = np.count_nonzero(trans_invalid)
            if n_trans_invalid > n_trans:
                raise ValueError(
                    "The number of invalid transitions ({}) is greater than"
                    " the total number of transitions ({}).  This should not"
                    " have happened".format(n_trans_invalid, n_trans)
                )
            n_trans -= n_trans_invalid
            # Subtract the number of invalid frames, i.e. frames in
            # which a compound resides in `state` but afterwards
            # transitions to a negative state.  This is done by
            # calculating the length ("lifetimes") of all `state`
            # sequences in `dtrj` that are followed by a negative state.
            # The sum of these sequences is then subtracted from the
            # total number of frames in which compounds reside in
            # `state`.
            # Get all end and start points of all `state` sequences.
            insertion = np.zeros(n_cmps, dtype=dtrj_state.dtype)
            dtrj_state = np.insert(
                dtrj_state, n_frames_tot, insertion, axis=ax_fr
            )
            dtrj_state = np.insert(dtrj_state, 0, insertion, axis=ax_fr)
            trans = np.diff(dtrj_state.astype(np.int8), axis=ax_fr)
            trans_ix = np.nonzero(trans)
            if ax_cmp != 0:
                # Sort position of state transitions by compound indices
                ix_sort = np.argsort(trans_ix[ax_cmp])
                trans_ix = np.array([tix[ix_sort] for tix in trans_ix])
            else:
                trans_ix = np.asarray(trans_ix)
            # Make `trans_ix` indicating the start rather than the end
            # of transitions into and out of `state`, i.e. `trans_ix`
            # now indicates the end points of all `state` sequences and
            # of all preceding sequences.
            trans_ix[ax_fr] -= 1
            # Length of all `state` sequences.
            lifetimes = np.diff(trans_ix[ax_fr])
            # End points of `state` sequences that are followed by a
            # negative state.
            trans_ix_invalid = np.asarray(np.nonzero(trans_invalid))
            # Identify the lifetimes of `state` sequences that are
            # followed by a negative state.
            trans_ix_invalid = trans_ix_invalid.T
            trans_ix = trans_ix.T
            trans_ix_invalid = [
                np.all(tix_iv == trans_ix, axis=1)
                for tix_iv in trans_ix_invalid
            ]
            trans_ix_invalid = np.any(trans_ix_invalid, axis=0)
            trans_ix_invalid = np.flatnonzero(trans_ix_invalid)
            trans_ix_invalid -= 1
            lifetimes_invalid = lifetimes[trans_ix_invalid]
            if np.any(lifetimes_invalid < 1):
                raise ValueError(
                    "At least one invalid state sequence is shorter than one"
                    " frame.  This should not have happened"
                )
            # Get the number of invalid frames.
            n_frames_invalid = np.sum(lifetimes_invalid)
            if n_frames_invalid > n_frames:
                raise ValueError(
                    "The number of invalid frames ({}) is greater than the"
                    " total number of frames ({}).  This should not have"
                    " happened".format(n_frames_invalid, n_frames)
                )
            # Subtract the number of invalid frames.
            n_frames -= n_frames_invalid
            if n_frames == 0 and n_trans == 0:
                # All `state` sequences are followed by negative states.
                trans_rates[six] = np.nan
                continue
            elif n_frames == 0 and n_trans > 0:
                raise ValueError(
                    "The number of valid frames is zero but the number of"
                    " valid transitions is {}.  This should not have"
                    " happened".format(n_trans)
                )
        trans_rates[six] = n_trans / n_frames
    del dtrj_state, trans, n_trans, n_frames

    if discard_all_neg:
        valid = ~np.isnan(trans_rates)
        trans_rates = trans_rates[valid]
        states_uniq = states_uniq[valid]
    if np.any(np.isnan(trans_rates)):
        raise ValueError(
            "At least one state has a NaN transition rate.  This should not"
            " have happened"
        )

    if return_states:
        return trans_rates, states_uniq
    else:
        return trans_rates


def lifetimes(
    dtrj,
    axis=-1,
    uncensored=False,
    discard_neg_start=False,
    discard_all_neg=False,
    return_states=False,
    return_cmp_ix=False,
):
    r"""
    Calculate the state lifetimes for each compound.

    Parameters
    ----------
    dtrj : array_like
        Array containing the discrete trajectory.
    axis : int, optional
        The axis along which to search for state transitions.  For
        ordinary discrete trajectories with shape ``(n, f)`` or
        ``(f,)``, where ``n`` is the number of compounds and ``f`` is
        the number of frames, set `axis` to ``-1``.  If you parse a
        transposed discrete trajectory of shape ``(f, n)``, set `axis`
        to ``0``.
    uncensored : bool, optional
        If ``True`` only take into account uncensored states, i.e.
        states whose start and end lie within the trajectory.  In other
        words, discard the truncated (censored) states at the beginning
        and end of the trajectory.  For these states the start/end time
        is unknown.
    discard_neg_start : bool, optional
        If ``True``, discard the lifetimes of all negative states (see
        notes).  This is equivalent to discarding all transitions
        starting from a negative state when calculating transition rates
        with :func:`mdtools.dtrj.trans_rate` or remain probabilities
        with :func:`mdtools.dtrj.remain_prob`.  Has no effect if
        `discard_all_neg` is ``True``.
    discard_all_neg : bool, optional
        If ``True``, discard the lifetimes of all negative states and of
        all states that are followed by a negative state (see notes).
        This is equivalent to discarding all transitions starting from
        or ending in a negative state when calculating transition rates
        with :func:`mdtools.dtrj.trans_rate` or remain probabilities
        with :func:`mdtools.dtrj.remain_prob`.
    return_states : bool, optional
        If ``True``, return the state indices associated with the
        returned lifetimes.
    return_cmp_ix : bool, optional
        If ``True``, return the compound indices associated with the
        returned lifetimes.

    Returns
    -------
    lt : numpy.ndarray
        1-dimensional array containing the lifetimes for all compounds
        in all states.
    states : numpy.ndarray
        1-dimensional array of the same shape as `lt` containing the
        corresponding states indices.  Only returned if `return_states`
        is ``True``.
    cmp_ix : numpy.ndarray
        1-dimensional array of the same shape as `lt` containing the
        corresponding compound indices.  Only returned if
        `return_cmp_ix` is ``True``.

    See Also
    --------
    :func:`mdtools.dtrj.lifetimes_per_state` :
        Calculate the state lifetimes for each state
    :func:`mdtools.dtrj.trans_rate` :
        Calculate the transition rate for each compound averaged over
        all states
    :func:`mdtools.dtrj.kaplan_meier` :
        Estimate the state survival function using the Kaplan-Meier
        estimator
    :func:`mdtools.dtrj.remain_prob` :
        Calculate the probability that a compound is still (or again) in
        the same state as at time :math:`t_0` after a lag time
        :math:`\Delta t`

    Notes
    -----
    State lifetimes are calculated by simply counting the number of
    frames a given compound stays in a given state.  Note that lifetimes
    calculated in this way are usually biased to lower values because of
    the limited length of the trajectory and because of
    truncation/censoring at the trajectory edges: At the beginning and
    end of the trajectory it is impossible to say how long a compound
    has already been it's initial state or how long it will stay in it's
    final state.  For better estimates of the average state lifetime use
    :func:`mdtools.dtrj.kaplan_meier`, :func:`mdtools.dtrj.trans_rate`,
    or :func:`mdtools.dtrj.remain_prob`.

    If `uncensored` is ``True``, the truncated states at the trajectory
    edges are ignored.  Thus, lifetimes calculated in this way can at
    maximum be as long as the trajectory minus the first and last frame.
    Depending on the length of the trajectory and the average state
    lifetime, uncensored counting might waste a significant amount of
    the trajectory.

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
    counting of frames in which a given compound resides in a given
    state.

    If both, `discard_neg_start` and `discard_all_neg`, are ``False``
    (the default), all states are valid, i.e. all frames are counted.

    If `discard_neg_start` is ``True``, negative states are not counted,
    i.e. the lifetimes of negative states are discarded.  Thus, the
    resulting average lifetime is the average lifetime of all positive
    states.

    If `discard_all_neg` is ``True``, negative states and states that
    are followed by a negative state are not counted.  This means
    transitions from or to negative states are completely discarded.
    Thus, the resulting average lifetime is the average lifetime of
    positive states that transition to another positive state.

    Examples
    --------
    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [6, 6, 6, 1, 4, 4],
    ...                  [1, 6, 6, 6, 4, 4]])
    >>> lt, states, cmp_ix = mdt.dtrj.lifetimes(
    ...     dtrj, return_states=True, return_cmp_ix=True
    ... )
    >>> lt
    array([1, 2, 3, 2, 3, 1, 3, 1, 2, 1, 3, 2])
    >>> states
    array([1, 2, 3, 2, 3, 1, 6, 1, 4, 1, 6, 4])
    >>> cmp_ix
    array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    >>> mdt.nph.group_by(states, lt)
    [array([1, 1, 1, 1]), array([2, 2]), array([3, 3]), array([2, 2]), array([3, 3])]
    >>> mdt.nph.group_by(cmp_ix, lt, assume_sorted=True)
    [array([1, 2, 3]), array([2, 3, 1]), array([3, 1, 2]), array([1, 3, 2])]
    >>> lt, states, cmp_ix = mdt.dtrj.lifetimes(
    ...     dtrj, axis=0, return_states=True, return_cmp_ix=True
    ... )
    >>> lt
    array([1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2])
    >>> states
    array([1, 2, 6, 1, 2, 6, 2, 3, 6, 3, 1, 6, 3, 4, 3, 1, 4])
    >>> cmp_ix
    array([0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5])
    >>> mdt.nph.group_by(states, lt)
    [array([1, 1, 1, 1]), array([1, 2, 1]), array([1, 2, 2, 1]), array([2, 2]), array([1, 2, 2, 1])]
    >>> mdt.nph.group_by(cmp_ix, lt, assume_sorted=True)
    [array([1, 1, 1, 1]), array([2, 2]), array([1, 1, 2]), array([2, 1, 1]), array([2, 2]), array([1, 1, 2])]

    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2]])
    >>> lt, states, cmp_ix = mdt.dtrj.lifetimes(
    ...     dtrj, return_states=True, return_cmp_ix=True
    ... )
    >>> lt
    array([1, 2, 3, 2, 3, 1, 3, 1, 2, 1, 3, 2])
    >>> states
    array([1, 2, 3, 2, 3, 1, 3, 1, 2, 1, 3, 2])
    >>> cmp_ix
    array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    >>> mdt.nph.group_by(states, lt)
    [array([1, 1, 1, 1]), array([2, 2, 2, 2]), array([3, 3, 3, 3])]
    >>> mdt.nph.group_by(cmp_ix, lt, assume_sorted=True)
    [array([1, 2, 3]), array([2, 3, 1]), array([3, 1, 2]), array([1, 3, 2])]
    >>> np.mean(lt)
    2.0
    >>> np.std(lt, ddof=1)
    0.8528028654224418
    >>> lt_unc, states_unc, cmp_ix_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     uncensored=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_unc
    array([2, 3, 1, 3])
    >>> states_unc
    array([2, 3, 1, 3])
    >>> cmp_ix_unc
    array([0, 1, 2, 3])
    >>> ax = 0
    >>> lt, states, cmp_ix = mdt.dtrj.lifetimes(
    ...     dtrj, axis=ax, return_states=True, return_cmp_ix=True
    ... )
    >>> lt
    array([1, 1, 1, 1, 2, 2, 1, 3, 2, 1, 1, 2, 2, 1, 1, 2])
    >>> states
    array([1, 2, 3, 1, 2, 3, 2, 3, 3, 1, 3, 3, 2, 3, 1, 2])
    >>> cmp_ix
    array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5])
    >>> mdt.nph.group_by(states, lt)
    [array([1, 1, 1, 1]), array([1, 2, 1, 2, 2]), array([1, 2, 3, 2, 1, 2, 1])]
    >>> mdt.nph.group_by(cmp_ix, lt, assume_sorted=True)
    [array([1, 1, 1, 1]), array([2, 2]), array([1, 3]), array([2, 1, 1]), array([2, 2]), array([1, 1, 2])]
    >>> np.mean(lt)
    1.5
    >>> np.std(lt, ddof=1)
    0.6324555320336759
    >>> lt_unc, states_unc, cmp_ix_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     axis=ax,
    ...     uncensored=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_unc
    array([1, 1, 1, 1])
    >>> states_unc
    array([2, 3, 1, 1])
    >>> cmp_ix_unc
    array([0, 0, 3, 5])

    >>> dtrj = np.array([[ 1,  2,  2,  1,  1,  1],
    ...                  [ 2,  2,  3,  3,  3,  2],
    ...                  [-3, -3, -3, -1,  2,  2],
    ...                  [ 2,  3,  3,  3, -2, -2]])
    >>> lt, states, cmp_ix = mdt.dtrj.lifetimes(
    ...     dtrj, return_states=True, return_cmp_ix=True
    ... )
    >>> lt
    array([1, 2, 3, 2, 3, 1, 3, 1, 2, 1, 3, 2])
    >>> states
    array([ 1,  2,  1,  2,  3,  2, -3, -1,  2,  2,  3, -2])
    >>> cmp_ix
    array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    >>> mdt.nph.group_by(states, lt)
    [array([3]), array([2]), array([1]), array([1, 3]), array([2, 2, 1, 2, 1]), array([3, 3])]
    >>> mdt.nph.group_by(cmp_ix, lt, assume_sorted=True)
    [array([1, 2, 3]), array([2, 3, 1]), array([3, 1, 2]), array([1, 3, 2])]
    >>> lt_unc, states_unc, cmp_ix_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     uncensored=True,
    ...     return_states=True,
    ...     return_cmp_ix=True
    ... )
    >>> lt_unc
    array([2, 3, 1, 3])
    >>> states_unc
    array([ 2,  3, -1,  3])
    >>> cmp_ix_unc
    array([0, 1, 2, 3])
    >>> lt_start, states_start, cmp_ix_start = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     discard_neg_start=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_start
    array([1, 2, 3, 2, 3, 1, 2, 1, 3])
    >>> states_start
    array([1, 2, 1, 2, 3, 2, 2, 2, 3])
    >>> cmp_ix_start
    array([0, 0, 0, 1, 1, 1, 2, 3, 3])
    >>> mdt.nph.group_by(states_start, lt_start)
    [array([1, 3]), array([2, 2, 1, 2, 1]), array([3, 3])]
    >>> mdt.nph.group_by(cmp_ix_start, lt_start, assume_sorted=True)
    [array([1, 2, 3]), array([2, 3, 1]), array([2]), array([1, 3])]
    >>> lt_start_unc, states_start_unc, cmp_ix_start_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     uncensored=True,
    ...     discard_neg_start=True,
    ...     return_states=True,
    ...     return_cmp_ix=True
    ... )
    >>> lt_start_unc
    array([2, 3, 3])
    >>> states_start_unc
    array([2, 3, 3])
    >>> cmp_ix_start_unc
    array([0, 1, 3])
    >>> lt_all, states_all, cmp_ix_all = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     discard_all_neg=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_all
    array([1, 2, 3, 2, 3, 1, 2, 1])
    >>> states_all
    array([1, 2, 1, 2, 3, 2, 2, 2])
    >>> cmp_ix_all
    array([0, 0, 0, 1, 1, 1, 2, 3])
    >>> mdt.nph.group_by(states_all, lt_all)
    [array([1, 3]), array([2, 2, 1, 2, 1]), array([3])]
    >>> mdt.nph.group_by(cmp_ix_all, lt_all, assume_sorted=True)
    [array([1, 2, 3]), array([2, 3, 1]), array([2]), array([1])]
    >>> lt_all_unc, states_all_unc, cmp_ix_all_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     uncensored=True,
    ...     discard_all_neg=True,
    ...     return_states=True,
    ...     return_cmp_ix=True
    ... )
    >>> lt_all_unc
    array([2, 3])
    >>> states_all_unc
    array([2, 3])
    >>> cmp_ix_all_unc
    array([0, 1])
    >>> ax = 0
    >>> lt, states, cmp_ix = mdt.dtrj.lifetimes(
    ...     dtrj, axis=ax, return_states=True, return_cmp_ix=True
    ... )
    >>> lt
    array([1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1])
    >>> states
    array([ 1,  2, -3,  2,  2, -3,  3,  2,  3, -3,  3,  1,  3, -1,  3,  1,  3,
            2, -2,  1,  2, -2])
    >>> cmp_ix
    array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5])
    >>> lt_unc, states_unc, cmp_ix_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     axis=ax,
    ...     uncensored=True,
    ...     return_states=True,
    ...     return_cmp_ix=True
    ... )
    >>> lt_unc
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    >>> states_unc
    array([ 2, -3, -3,  3, -3,  3, -1,  3,  2,  2])
    >>> cmp_ix_unc
    array([0, 0, 1, 2, 2, 3, 3, 4, 4, 5])
    >>> lt_start, states_start, cmp_ix_start = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     axis=ax,
    ...     discard_neg_start=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_start
    array([1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    >>> states_start
    array([1, 2, 2, 2, 3, 2, 3, 3, 1, 3, 3, 1, 3, 2, 1, 2])
    >>> cmp_ix_start
    array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5])
    >>> lt_start_unc, states_start_unc, cmp_ix_start_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     axis=ax,
    ...     uncensored=True,
    ...     discard_neg_start=True,
    ...     return_states=True,
    ...     return_cmp_ix=True
    ... )
    >>> lt_start_unc
    array([1, 1, 1, 1, 1, 2])
    >>> states_start_unc
    array([2, 3, 3, 3, 2, 2])
    >>> cmp_ix_start_unc
    array([0, 2, 3, 4, 4, 5])
    >>> lt_all, states_all, cmp_ix_all = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     axis=ax,
    ...     discard_all_neg=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_all
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> states_all
    array([1, 2, 3, 2, 3, 1, 3, 1, 3, 1])
    >>> cmp_ix_all
    array([0, 0, 1, 2, 2, 3, 3, 4, 4, 5])
    >>> lt_all_unc, states_all_unc, cmp_ix_all_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     axis=ax,
    ...     uncensored=True,
    ...     discard_all_neg=True,
    ...     return_states=True,
    ...     return_cmp_ix=True
    ... )
    >>> lt_all_unc
    array([1])
    >>> states_all_unc
    array([3])
    >>> cmp_ix_all_unc
    array([4])

    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1]])
    >>> lt, states, cmp_ix = mdt.dtrj.lifetimes(
    ...     dtrj, return_states=True, return_cmp_ix=True
    ... )
    >>> lt
    array([1, 2, 3, 2, 3, 1, 3, 1, 2, 1, 3, 2, 1, 4, 1])
    >>> states
    array([ 1, -2,  3, -2,  3,  1,  3,  1, -2,  1,  3, -2,  1,  4, -1])
    >>> cmp_ix
    array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    >>> mdt.nph.group_by(states, lt)
    [array([2, 2, 2, 2]), array([1]), array([1, 1, 1, 1, 1]), array([3, 3, 3, 3]), array([4])]
    >>> mdt.nph.group_by(cmp_ix, lt, assume_sorted=True)
    [array([1, 2, 3]), array([2, 3, 1]), array([3, 1, 2]), array([1, 3, 2]), array([1, 4, 1])]
    >>> lt_unc, states_unc, cmp_ix_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     uncensored=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_unc
    array([2, 3, 1, 3, 4])
    >>> states_unc
    array([-2,  3,  1,  3,  4])
    >>> cmp_ix_unc
    array([0, 1, 2, 3, 4])
    >>> lt_start, states_start, cmp_ix_start = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     discard_neg_start=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_start
    array([1, 3, 3, 1, 3, 1, 1, 3, 1, 4])
    >>> states_start
    array([1, 3, 3, 1, 3, 1, 1, 3, 1, 4])
    >>> cmp_ix_start
    array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    >>> mdt.nph.group_by(states_start, lt_start)
    [array([1, 1, 1, 1, 1]), array([3, 3, 3, 3]), array([4])]
    >>> mdt.nph.group_by(cmp_ix_start, lt_start, assume_sorted=True)
    [array([1, 3]), array([3, 1]), array([3, 1]), array([1, 3]), array([1, 4])]
    >>> lt_start_unc, states_start_unc, cmp_ix_start_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     uncensored=True,
    ...     discard_neg_start=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_start_unc
    array([3, 1, 3, 4])
    >>> states_start_unc
    array([3, 1, 3, 4])
    >>> cmp_ix_start_unc
    array([1, 2, 3, 4])
    >>> lt_all, states_all, cmp_ix_all = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     discard_all_neg=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_all
    array([3, 3, 1, 3, 1, 1])
    >>> states_all
    array([3, 3, 1, 3, 1, 1])
    >>> cmp_ix_all
    array([0, 1, 1, 2, 3, 4])
    >>> mdt.nph.group_by(states_all, lt_all)
    [array([1, 1, 1]), array([3, 3, 3])]
    >>> mdt.nph.group_by(cmp_ix_all, lt_all, assume_sorted=True)
    [array([3]), array([3, 1]), array([3]), array([1]), array([1])]
    >>> lt_all_unc, states_all_unc, cmp_ix_all_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     uncensored=True,
    ...     discard_all_neg=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_all_unc
    array([3])
    >>> states_all_unc
    array([3])
    >>> cmp_ix_all_unc
    array([1])
    >>> ax = 0
    >>> lt, states, cmp_ix = mdt.dtrj.lifetimes(
    ...     dtrj, axis=ax, return_states=True, return_cmp_ix=True
    ... )
    >>> lt
    array([1, 1, 1, 2, 2, 2, 1, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1])
    >>> states
    array([ 1, -2,  3,  1, -2,  3,  4, -2,  3,  4,  3,  1,  3,  4,  3, -2,  4,
            3,  1, -2, -1])
    >>> cmp_ix
    array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5])
    >>> lt_start, states_start, cmp_ix_start = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     axis=ax,
    ...     discard_neg_start=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_start
    array([1, 1, 2, 2, 1, 3, 1, 2, 1, 1, 1, 2, 1, 1, 1])
    >>> states_start
    array([1, 3, 1, 3, 4, 3, 4, 3, 1, 3, 4, 3, 4, 3, 1])
    >>> cmp_ix_start
    array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5])
    >>> lt_start_unc, states_start_unc, cmp_ix_start_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     axis=ax,
    ...     uncensored=True,
    ...     discard_neg_start=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_start_unc
    array([1, 2, 3, 1, 1, 1])
    >>> states_start_unc
    array([3, 3, 3, 1, 3, 1])
    >>> cmp_ix_start_unc
    array([0, 1, 2, 3, 3, 5])
    >>> lt_all, states_all, cmp_ix_all = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     axis=ax,
    ...     discard_all_neg=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_all
    array([1, 2, 2, 1, 3, 1, 2, 1, 1, 1, 1, 1])
    >>> states_all
    array([3, 1, 3, 4, 3, 4, 3, 1, 3, 4, 4, 3])
    >>> cmp_ix_all
    array([0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5])
    >>> lt_all_unc, states_all_unc, cmp_ix_all_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     axis=ax,
    ...     uncensored=True,
    ...     discard_all_neg=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ... )
    >>> lt_all_unc
    array([1, 2, 3, 1, 1])
    >>> states_all_unc
    array([3, 3, 3, 1, 3])
    >>> cmp_ix_all_unc
    array([0, 1, 2, 3, 3])

    >>> dtrj = np.array([[1, 2, 2],
    ...                  [3, 3, 3]])
    >>> lt_unc, states_unc, cmp_ix_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     uncensored=True,
    ...     return_states=True,
    ...     return_cmp_ix=True
    ... )
    >>> lt_unc
    array([], dtype=int64)
    >>> states_unc
    array([], dtype=int64)
    >>> cmp_ix_unc
    array([], dtype=int64)
    >>> lt_unc, states_unc, cmp_ix_unc = mdt.dtrj.lifetimes(
    ...     dtrj,
    ...     axis=0,
    ...     uncensored=True,
    ...     return_states=True,
    ...     return_cmp_ix=True
    ... )
    >>> lt_unc
    array([], dtype=int64)
    >>> states_unc
    array([], dtype=int64)
    >>> cmp_ix_unc
    array([], dtype=int64)
    """  # noqa: W505, E501
    dtrj = mdt.check.dtrj(dtrj)
    ax_cmp, ax_fr = mdt.dtrj.get_ax(ax_fr=axis)
    n_frames = dtrj.shape[ax_fr]

    # Ensure that the end of the trajectory (i.e. the virtual frame
    # after the last frame) is treated as the end of a state
    # transition by adding a new state to the end of `dtrj`.
    dtrj = np.insert(dtrj, n_frames, np.max(dtrj) + 1, axis=ax_fr)
    # Ensures that the first frame is treated as the end of a state
    # transition by setting `tfft` to ``True``.
    trans_ix_start, trans_ix_end = mdt.dtrj.trans_ix(
        dtrj, axis=ax_fr, pin="both", tfft=True
    )

    if ax_cmp != 0:
        # Sort position of state transitions by compound indices.
        ix_sort = np.argsort(trans_ix_start[ax_cmp], kind="stable")
        trans_ix_start = tuple(tix[ix_sort] for tix in trans_ix_start)
        ix_sort = np.argsort(trans_ix_end[ax_cmp], kind="stable")
        trans_ix_end = tuple(tix[ix_sort] for tix in trans_ix_end)
        del ix_sort

    # Calculate the state lifetimes.
    lt = np.diff(trans_ix_end[ax_fr])
    # Negative values in `lt` indicate the start of a new compound
    # trajectory and are therefore of no physical relevance and are thus
    # removed.
    lt = lt[lt > 0]

    # Get state and compound indices corresponding to each lifetime.
    states = dtrj[trans_ix_start]
    cmp_ix = trans_ix_start[ax_cmp]
    valid = np.ones(cmp_ix.shape, dtype=bool)
    del dtrj

    if discard_neg_start or discard_all_neg:
        # Discard the lifetimes of all negative states.
        valid_states = states >= 0
        valid &= valid_states

    if discard_all_neg:
        # Discard the lifetimes of positive states that are followed by
        # a negative state (i.e. discard all states that do not have a
        # valid follower state).
        valid_follower = np.roll(valid_states, shift=-1)
        # States at the end of a compound trajectory do not have any
        # follower state and are therefore always regarded as valid
        # (except if they are negative).
        trj_ends = (trans_ix_start[ax_fr] + 1) == n_frames
        valid_follower[trj_ends] = True
        valid &= valid_follower
        del valid_states, valid_follower, trj_ends

    if uncensored:
        # Remove censored states at the trajectory edges.
        # Get end of each compound trajectory.
        censored_end = np.diff(cmp_ix) != 0
        censored_end = np.append(censored_end, True)
        # Get beginning of each compound trajectory.
        censored_start = np.roll(censored_end, shift=1)
        censored = np.logical_or(
            censored_start, censored_end, out=censored_start
        )
        valid &= np.invert(censored, out=censored)
        del censored_start, censored_end, censored
        if np.any(lt[valid] > n_frames - 2):
            raise ValueError(
                "At least one lifetime is greater than the number of frames in"
                " the trajectory minus two.  This should not have happened"
            )

    if np.any(lt > n_frames):
        raise ValueError(
            "At least one lifetime is greater than the number of frames in the"
            " trajectory.  This should not have happened"
        )
    if np.any(lt < 0):
        raise ValueError(
            "At least one lifetime is negative.  This should not have happened"
        )

    ret = (lt[valid],)
    if return_states:
        ret += (states[valid],)
    if return_cmp_ix:
        ret += (cmp_ix[valid],)
    if len(ret) == 1:
        ret = ret[0]
    return ret


def lifetimes_per_state(
    dtrj,
    axis=-1,
    uncensored=False,
    discard_neg_start=False,
    discard_all_neg=False,
    return_states=False,
    return_cmp_ix=False,
    return_avg=False,
    kwargs_avg=None,
    return_std=False,
    kwargs_std=None,
):
    r"""
    Calculate the state lifetimes for each state.

    Parameters
    ----------
    dtrj : array_like
        Array containing the discrete trajectory.
    axis : int, optional
        The axis along which to search for state transitions.  For
        ordinary discrete trajectories with shape ``(n, f)`` or
        ``(f,)``, where ``n`` is the number of compounds and ``f`` is
        the number of frames, set `axis` to ``-1``.  If you parse a
        transposed discrete trajectory of shape ``(f, n)``, set `axis`
        to ``0``.
    uncensored : bool, optional
        If ``True`` only take into account uncensored states, i.e.
        states whose start and end lie within the trajectory.  In other
        words, discard the truncated (censored) states at the beginning
        and end of the trajectory.  For these states the start/end time
        is unknown.
    discard_neg_start : bool, optional
        If ``True``, discard the lifetimes of all negative states (see
        notes of :func:`mdtools.dtrj.lifetimes`).  This is equivalent to
        discarding all transitions starting from a negative state when
        calculating transition rates with
        :func:`mdtools.dtrj.trans_rate_per_state` or remain
        probabilities with :func:`mdtools.dtrj.remain_prob_discrete`.
        Has no effect if `discard_all_neg` is ``True``.
    discard_all_neg : bool, optional
        If ``True``, discard the lifetimes of all negative states and of
        all states that are followed by a negative state (see notes of
        :func:`mdtools.dtrj.lifetimes`).  This is equivalent to
        discarding all transitions starting from or ending in a negative
        state when calculating transition rates with
        :func:`mdtools.dtrj.trans_rate_per_state` or remain
        probabilities with :func:`mdtools.dtrj.remain_prob_discrete`.
    return_states : bool, optional
        If ``True``, return the state indices associated with the
        returned lifetimes.
    return_cmp_ix : bool, optional
        If ``True``, return the compound indices associated with the
        returned lifetimes.
    return_avg : bool, optional
        If ``True``, return the average lifetime of each state.
    kwargs_avg : dict, optional
        Keyword arguments to parse to :func:`numpy.mean` for calculating
        the average lifetimes.  See there for possible options.
    return_std : bool, optional
        If ``True``, return the standard deviation of the lifetimes of
        each state.
    kwargs_std : dict, optional
        Keyword arguments to parse to :func:`numpy.std` for calculating
        the standard deviation of the lifetimes.  See there for possible
        options.

    Returns
    -------
    lt : list of 1D numpy.ndarray
        List of 1-dimensional arrays.  Each array contains the lifetimes
        of one state.
    states : numpy.ndarray
        1-dimensional array of shape ``(len(lt),)`` containing the
        corresponding state indices for each lifetime array in `lt`.
        Only returned if `return_states` is ``True``.
    cmp_ix : list of 1D numpy.ndarray
        List of 1-dimensional arrays of the same shape as `lt`.  Each
        array contains the corresponding compound indices.  Only
        returned if `return_cmp_ix` is ``True``.
    lt_avg : numpy.ndarray
        1-dimensional array of shape ``(len(lt),)`` containing the
        average lifetime of each state.  Only returned if `return_avg`
        is ``True``.
    lt_std : numpy.ndarray
        1-dimensional array of shape ``(len(lt),)`` containing the
        standard deviation of the lifetimes of each state.  Only
        returned if `return_std` is ``True``.

    See Also
    --------
    :func:`mdtools.dtrj.lifetimes` :
        Calculate the state lifetimes for each compound
    :func:`mdtools.dtrj.trans_rate_per_state` :
        Calculate the transition rate for each state averaged over all
        compounds
    :func:`mdtools.dtrj.remain_prob_discrete` :
        Calculate the probability that a compound is still (or again) in
        the same state as at time :math:`t_0` after a lag time
        :math:`\Delta t` resolved with respect to the states in second
        discrete trajectory

    Notes
    -----
    See :func:`mdtools.dtrj.lifetimes` for details about how state
    lifetimes are calculated and about valid and invalid states
    (`discard_neg_start` and `discard_all_neg`).

    Examples
    --------
    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [6, 6, 6, 1, 4, 4],
    ...                  [1, 6, 6, 6, 4, 4]])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([1, 1, 1, 1]), array([2, 2]), array([3, 3]), array([2, 2]), array([3, 3])]
    >>> states
    array([1, 2, 3, 4, 6])
    >>> cmp_ix
    [array([0, 1, 2, 3]), array([0, 1]), array([0, 1]), array([2, 3]), array([2, 3])]
    >>> lt_avg
    array([1., 2., 3., 2., 3.])
    >>> lt_std
    array([0., 0., 0., 0., 0.])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     axis=0,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([1, 1, 1, 1]), array([1, 2, 1]), array([1, 2, 2, 1]), array([2, 2]), array([1, 2, 2, 1])]
    >>> states
    array([1, 2, 3, 4, 6])
    >>> cmp_ix
    [array([0, 0, 3, 5]), array([0, 1, 2]), array([2, 3, 4, 5]), array([4, 5]), array([0, 1, 2, 3])]
    >>> lt_avg
    array([1.        , 1.33333333, 1.5       , 2.        , 1.5       ])
    >>> lt_std
    array([0.        , 0.47140452, 0.5       , 0.        , 0.5       ])

    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2]])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([1, 1, 1, 1]), array([2, 2, 2, 2]), array([3, 3, 3, 3])]
    >>> states
    array([1, 2, 3])
    >>> cmp_ix
    [array([0, 1, 2, 3]), array([0, 1, 2, 3]), array([0, 1, 2, 3])]
    >>> lt_avg
    array([1., 2., 3.])
    >>> lt_std
    array([0., 0., 0.])
    >>> lt_unc, states_unc, cmp_ix_unc, lt_avg_unc, lt_std_unc = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     uncensored=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt_unc
    [array([1]), array([2]), array([3, 3])]
    >>> states_unc
    array([1, 2, 3])
    >>> cmp_ix_unc
    [array([2]), array([0]), array([1, 3])]
    >>> lt_avg_unc
    array([1., 2., 3.])
    >>> lt_std_unc
    array([0., 0., 0.])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     axis=0,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([1, 1, 1, 1]), array([1, 2, 1, 2, 2]), array([1, 2, 3, 2, 1, 2, 1])]
    >>> states
    array([1, 2, 3])
    >>> cmp_ix
    [array([0, 0, 3, 5]), array([0, 1, 2, 4, 5]), array([0, 1, 2, 3, 3, 4, 5])]
    >>> lt_avg
    array([1.        , 1.6       , 1.71428571])
    >>> lt_std
    array([0.        , 0.48989795, 0.69985421])
    >>> lt_unc, states_unc, cmp_ix_unc, lt_avg_unc, lt_std_unc = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     axis=0,
    ...     uncensored=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt_unc
    [array([1, 1]), array([1]), array([1])]
    >>> states_unc
    array([1, 2, 3])
    >>> cmp_ix_unc
    [array([3, 5]), array([0]), array([0])]
    >>> lt_avg_unc
    array([1., 1., 1.])
    >>> lt_std_unc
    array([0., 0., 0.])

    >>> dtrj = np.array([[ 1,  2,  2,  1,  1,  1],
    ...                  [ 2,  2,  3,  3,  3,  2],
    ...                  [-3, -3, -3, -1,  2,  2],
    ...                  [ 2,  3,  3,  3, -2, -2]])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([3]), array([2]), array([1]), array([1, 3]), array([2, 2, 1, 2, 1]), array([3, 3])]
    >>> states
    array([-3, -2, -1,  1,  2,  3])
    >>> cmp_ix
    [array([2]), array([3]), array([2]), array([0, 0]), array([0, 1, 1, 2, 3]), array([1, 3])]
    >>> lt_avg
    array([3. , 2. , 1. , 2. , 1.6, 3. ])
    >>> lt_std
    array([0.        , 0.        , 0.        , 1.        , 0.48989795,
           0.        ])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     discard_neg_start=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([1, 3]), array([2, 2, 1, 2, 1]), array([3, 3])]
    >>> states
    array([1, 2, 3])
    >>> cmp_ix
    [array([0, 0]), array([0, 1, 1, 2, 3]), array([1, 3])]
    >>> lt_avg
    array([2. , 1.6, 3. ])
    >>> lt_std
    array([1.        , 0.48989795, 0.        ])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     discard_all_neg=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([1, 3]), array([2, 2, 1, 2, 1]), array([3])]
    >>> states
    array([1, 2, 3])
    >>> cmp_ix
    [array([0, 0]), array([0, 1, 1, 2, 3]), array([1])]
    >>> lt_avg
    array([2. , 1.6, 3. ])
    >>> lt_std
    array([1.        , 0.48989795, 0.        ])
    >>> ax = 0
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     axis=ax,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([1, 1, 1]), array([1, 1]), array([1]), array([1, 1, 1, 1]), array([1, 1, 2, 1, 1, 2]), array([1, 1, 1, 1, 1, 1])]
    >>> states
    array([-3, -2, -1,  1,  2,  3])
    >>> cmp_ix
    [array([0, 1, 2]), array([4, 5]), array([3]), array([0, 3, 4, 5]), array([0, 0, 1, 2, 4, 5]), array([1, 2, 2, 3, 3, 4])]
    >>> lt_avg
    array([1.        , 1.        , 1.        , 1.        , 1.33333333,
           1.        ])
    >>> lt_std
    array([0.        , 0.        , 0.        , 0.        , 0.47140452,
           0.        ])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     axis=ax,
    ...     discard_neg_start=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([1, 1, 1, 1]), array([1, 1, 2, 1, 1, 2]), array([1, 1, 1, 1, 1, 1])]
    >>> states
    array([1, 2, 3])
    >>> cmp_ix
    [array([0, 3, 4, 5]), array([0, 0, 1, 2, 4, 5]), array([1, 2, 2, 3, 3, 4])]
    >>> lt_avg
    array([1.        , 1.33333333, 1.        ])
    >>> lt_std
    array([0.        , 0.47140452, 0.        ])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     axis=ax,
    ...     discard_all_neg=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([1, 1, 1, 1]), array([1, 1]), array([1, 1, 1, 1])]
    >>> states
    array([1, 2, 3])
    >>> cmp_ix
    [array([0, 3, 4, 5]), array([0, 2]), array([1, 2, 3, 4])]
    >>> lt_avg
    array([1., 1., 1.])
    >>> lt_std
    array([0., 0., 0.])

    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1]])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([2, 2, 2, 2]), array([1]), array([1, 1, 1, 1, 1]), array([3, 3, 3, 3]), array([4])]
    >>> states
    array([-2, -1,  1,  3,  4])
    >>> cmp_ix
    [array([0, 1, 2, 3]), array([4]), array([0, 1, 2, 3, 4]), array([0, 1, 2, 3]), array([4])]
    >>> lt_avg
    array([2., 1., 1., 3., 4.])
    >>> lt_std
    array([0., 0., 0., 0., 0.])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     discard_neg_start=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([1, 1, 1, 1, 1]), array([3, 3, 3, 3]), array([4])]
    >>> states
    array([1, 3, 4])
    >>> cmp_ix
    [array([0, 1, 2, 3, 4]), array([0, 1, 2, 3]), array([4])]
    >>> lt_avg
    array([1., 3., 4.])
    >>> lt_std
    array([0., 0., 0.])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     discard_all_neg=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([1, 1, 1]), array([3, 3, 3])]
    >>> states
    array([1, 3])
    >>> cmp_ix
    [array([1, 3, 4]), array([0, 1, 2])]
    >>> lt_avg
    array([1., 3.])
    >>> lt_std
    array([0., 0.])
    >>> ax = 0
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     axis=ax,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([1, 2, 1, 2, 2]), array([1]), array([1, 2, 1, 1]), array([1, 2, 3, 2, 1, 2, 1]), array([1, 1, 1, 1])]
    >>> states
    array([-2, -1,  1,  3,  4])
    >>> cmp_ix
    [array([0, 1, 2, 4, 5]), array([5]), array([0, 0, 3, 5]), array([0, 1, 2, 3, 3, 4, 5]), array([1, 2, 3, 4])]
    >>> lt_avg
    array([1.6       , 1.        , 1.25      , 1.71428571, 1.        ])
    >>> lt_std
    array([0.48989795, 0.        , 0.4330127 , 0.69985421, 0.        ])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     axis=ax,
    ...     discard_neg_start=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([1, 2, 1, 1]), array([1, 2, 3, 2, 1, 2, 1]), array([1, 1, 1, 1])]
    >>> states
    array([1, 3, 4])
    >>> cmp_ix
    [array([0, 0, 3, 5]), array([0, 1, 2, 3, 3, 4, 5]), array([1, 2, 3, 4])]
    >>> lt_avg
    array([1.25      , 1.71428571, 1.        ])
    >>> lt_std
    array([0.4330127 , 0.69985421, 0.        ])
    >>> lt, states, cmp_ix, lt_avg, lt_std = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     axis=ax,
    ...     discard_all_neg=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt
    [array([2, 1]), array([1, 2, 3, 2, 1, 1]), array([1, 1, 1, 1])]
    >>> states
    array([1, 3, 4])
    >>> cmp_ix
    [array([0, 3]), array([0, 1, 2, 3, 3, 5]), array([1, 2, 3, 4])]
    >>> lt_avg
    array([1.5       , 1.66666667, 1.        ])
    >>> lt_std
    array([0.5       , 0.74535599, 0.        ])

    >>> dtrj = np.array([[1, 2, 2],
    ...                  [3, 3, 3]])
    >>> lt_unc, states_unc, cmp_ix_unc, lt_avg_unc, lt_std_unc = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     uncensored=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt_unc
    [array([], dtype=int64)]
    >>> states_unc
    array([], dtype=int64)
    >>> cmp_ix_unc
    [array([], dtype=int64)]
    >>> lt_avg_unc
    array([nan])
    >>> lt_std_unc
    array([nan])
    >>> lt_unc, states_unc, cmp_ix_unc, lt_avg_unc, lt_std_unc = mdt.dtrj.lifetimes_per_state(
    ...     dtrj,
    ...     axis=0,
    ...     uncensored=True,
    ...     return_states=True,
    ...     return_cmp_ix=True,
    ...     return_avg=True,
    ...     return_std=True,
    ... )
    >>> lt_unc
    [array([], dtype=int64)]
    >>> states_unc
    array([], dtype=int64)
    >>> cmp_ix_unc
    [array([], dtype=int64)]
    >>> lt_avg_unc
    array([nan])
    >>> lt_std_unc
    array([nan])
    """  # noqa: E501, W505
    lt, states, cmp_ix = mdt.dtrj.lifetimes(
        dtrj,
        axis=axis,
        uncensored=uncensored,
        discard_neg_start=discard_neg_start,
        discard_all_neg=discard_all_neg,
        return_states=True,
        return_cmp_ix=True,
    )
    states_uniq, lt = mdt.nph.group_by(states, lt, return_keys=True)

    ret = (lt,)
    if return_states:
        ret += (states_uniq,)
    if return_cmp_ix:
        _states_uniq, cmp_ix = mdt.nph.group_by(
            states, cmp_ix, return_keys=True
        )
        if not np.array_equal(_states_uniq, states_uniq):
            raise ValueError(
                "`_states_uniq` ({}) != `states_uniq` ({}).  This should not"
                " have happened".format(_states_uniq, states_uniq)
            )
        ret += (cmp_ix,)
    if return_avg:
        if kwargs_avg is None:
            kwargs_avg = {}
        lt_avg = np.array(
            [
                np.mean(lt_state, **kwargs_avg)
                if len(lt_state) > 0
                else np.nan
                for lt_state in lt
            ]
        )
        ret += (lt_avg,)
    if return_std:
        if kwargs_std is None:
            kwargs_std = {}
        lt_std = np.array(
            [
                np.std(lt_state, **kwargs_std) if len(lt_state) > 0 else np.nan
                for lt_state in lt
            ]
        )
        ret += (lt_std,)
    if len(ret) == 1:
        ret = ret[0]
    return ret


def remain_prob(  # noqa: C901
    dtrj,
    restart=1,
    continuous=False,
    discard_neg_start=False,
    discard_all_neg=False,
    verbose=False,
):
    r"""
    Calculate the probability that a compound is still (or again) in the
    same state as at time :math:`t_0` after a lag time :math:`\Delta t`.

    Take a discrete trajectory and calculate the probability to find a
    compound in the same state as at time :math:`t_0` after a lag time
    :math:`\Delta t`.

    .. important::

        :math:`t_0` is an arbitrary time point in the trajectory.  It is
        particularly not the time of a state transition!

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
        state (see notes).  This is equivalent to discarding the
        lifetimes of all negative states when calculating state
        lifetimes with :func:`mdtools.dtrj.lifetimes`.  Must not be used
        together with `discard_all_neg`.
    discard_all_neg : bool, optional
        If ``True``, discard all transitions starting from or ending in
        a negative state (see notes).  This is equivalent to discarding
        the lifetimes of all negative states and of all states that are
        followed by a negative state when calculating state lifetimes
        with :func:`mdtools.dtrj.lifetimes`.  Must not be used together
        with `discard_neg_start`.
    verbose : bool, optional
        If ``True`` print a progress bar.

    Returns
    -------
    prob : numpy.ndarray
        Array of shape ``(f,)`` containing for all possible lag times
        the probability that a compound is in the same state as at time
        :math:`t_0` after a lag time :math:`\Delta t`.

    See Also
    --------
    :func:`mdtools.dtrj.remain_prob_discrete` :
        Calculate the probability that a compound is still (or again) in
        the same state as at time :math:`t_0` after a lag time
        :math:`\Delta t` resolved with respect to the states in second
        discrete trajectory
    :func:`mdtools.dtrj.back_jump_prob` :
        Calculate the back-jump probability averaged over all states
    :func:`mdtools.dtrj.leave_prob` :
        Calculate the probability that a compound leaves its state after
        a lag time :math:`\Delta t` given that it has entered the state
        at time :math:`t_0`
    :func:`mdtools.dtrj.kaplan_meier` :
        Estimate the state survival function using the Kaplan-Meier
        estimator
    :func:`mdtools.dtrj.trans_rate` :
        Calculate the transition rate for each compound averaged over
        all states
    :func:`mdtools.dtrj.lifetimes` :
        Calculate the state lifetimes for each compound

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
    probability to stay in a negative state nor is it affected by
    transitions from positive to negative states.

    **Lifetimes**

    .. important::

        Because the compound might have already been in the same state
        before the start of the observation at time :math:`t_0`, the
        continuous probability is **not** the survival function of the
        underlying distribution of lifetimes, but rather the survival
        function of the underlying distribution of remaining/residual
        lifetimes.  That is the time after which a compound that is
        randomly picked from the state will leave the state, regardless
        how long it has already been in this state.  However, because
        long-lived states are sampled more often, the continuous (and
        discontinuous) probability to be still (or again) in the same
        state is biased to longer lifetimes.  Also note that the
        discontinuous probability is just a measure for the state
        relaxation but is not directly related to the state lifetime.
        [6]_:sup:`,` [7]_  Thus, the following paragraphs have to be
        read with caution!

        To calculate the survival function of the true lifetime
        distribution, use :func:`mdtools.dtrj.surv_func`.

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

        \langle \tau \rangle = \int_0^\infty f(t) \text{ d}t =
        \frac{\tau_0}{\beta} \Gamma\left(\frac{1}{\beta}\right)

    Here, :math:`\tau_0` and :math:`\beta` are the fit parameters,
    :math:`\langle \tau \rangle` is the average lifetime and
    :math:`\Gamma(x)` is the gamma function.  For physically meaningful
    results, :math:`\beta` should be confined to :math:`0 < \beta \leq
    1`.  For purely exponential decay (:math:`\beta = 1`), :math:`\tau =
    \tau_0` applies.  If the calculated probability :math:`p(\Delta t)`
    fully decays to zero within the accessible range of lag times, one
    can alternatively numerically integrate the calculated probability
    directly, instead of the fit :math:`f(t)`.

    In general, the n-th moment of the underlying distribution of
    lifetimes is given by: [3]_:sup:`,` [4]_:sup:`,` [5]_

    .. math::

        \langle \tau^n \rangle
        = \frac{1}{(n-1)!} \int_0^\infty t^{n-1} f(t) \text{ d}t
        = \frac{\tau_0^n}{\beta}
        \frac{\Gamma\left(\frac{n}{\beta}\right)}{\Gamma(n)}

    Moreover, the time-averaged lifetime is given by: [3]_

    .. math::

        \bar{\tau^n}
        = \frac{
            \int_0^\infty t^n f(t) \text{ d}t
        }{
            \int_0^\infty f(t) \text{ d}t
        }
        = \frac{\langle \tau^{n+1} \rangle}{\langle \tau \rangle} n!
        = \tau_0^n
        \frac{
            \Gamma\left(\frac{n+1}{\beta}\right)
        }{
            \Gamma\left(\frac{1}{\beta}\right)
        }

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
    .. [5] I. S. Gradshteyn, I. M. Ryzhik,
           `Table of Integrals, Series, and Products
           <https://archive.org/details/GradshteinI.S.RyzhikI.M.TablesOfIntegralsSeriesAndProducts>`_,
           Academic Press, 2007, 7th edition, p. 370, Integral 3.478
           (1.).
    .. [6] A. Luzar,
           `Resolving the hydrogen bond dynamics conundrum
           <https://doi.org/10.1063/1.1320826>`_,
           The Journal of Chemical Physics, 2000, 113, 10663-10675.
    .. [7] V. P. Voloshin1 and Yu. I. Naberukhin,
           `Hydrogen bond lifetime distributions in computer-simulated
           water <https://doi.org/10.1007/s10947-009-0010-6>`_,
           Journal of Structural Chemistry, 2009, 50, 78-89.

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
        restarts = mdt.rti.ProgressBar(
            restarts, total=n_restarts, unit="restarts"
        )
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
    Calculate the probability that a compound is still (or again) in the
    same state as at time :math:`t_0` after a lag time :math:`\Delta t`
    resolved with respect to the states in a second discrete trajectory.

    Take a discrete trajectory and calculate the probability to find a
    compound in the same state as at time :math:`t_0` after a lag time
    :math:`\Delta t` given that the compound was in a specific state of
    another discrete trajectory at time :math:`t_0`.

    .. important::

        :math:`t_0` is an arbitrary time point in the trajectory.  It is
        particularly not the time of a state transition!

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
        state (see notes of :func:`mdtools.dtrj.remain_prob`).  This is
        equivalent to discarding the lifetimes of all negative states
        when calculating state lifetimes with
        :func:`mdtools.dtrj.lifetimes_per_state`.  Must not be used
        together with `discard_all_neg`.
    discard_all_neg : bool, optional
        If ``True``, discard all transitions starting from or ending in
        a negative state (see notes of
        :func:`mdtools.dtrj.remain_prob`).  This is equivalent to
        discarding the lifetimes of all negative states and of all
        states that are followed by a negative state when calculating
        state lifetimes with :func:`mdtools.dtrj.lifetimes_per_state`.
        Must not be used together with `discard_neg_start`.
    verbose : bool, optional
        If ``True`` print a progress bar.

    Returns
    -------
    prob : numpy.ndarray
        Array of shape ``(f, m)``, where ``m`` is the number of states
        in the second discrete trajectory.  The `ij`-th element of
        `prob` is the probability that a compound is in the same state
        of the first discrete trajectory as at time :math:`t_0` after a
        lag time of `i` frames given that the compound was in state `j`
        of the second discrete trajectory at time :math:`t_0`.

    See Also
    --------
    :func:`mdtools.dtrj.remain_prob` :
        Calculate the probability that a compound is still (or again) in
        the same state as at time :math:`t_0` after a lag time
        :math:`\Delta t`
    :func:`mdtools.dtrj.leave_prob_discrete` :
        Calculate the probability that a compound leaves its state after
        a lag time :math:`\Delta t` given that it has entered the state
        at time :math:`t_0` resolved with respect to the states in a
        second discrete trajectory
    :func:`mdtools.dtrj.back_jump_prob_discrete` :
        Calculate the back-jump probability resolved with respect to the
        states a in second discrete trajectory.
    :func:`mdtools.dtrj.trans_rate_per_state` :
        Calculate the transition rate for each state averaged over all
        compounds
    :func:`mdtools.dtrj.lifetimes_per_state` :
        Calculate the state lifetimes for each state

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
    >>> # [-2         ,-1         , 1         , 3         , 4         ]
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
    # at zero and no intermediate states are missing to reduce the
    # memory required for `prob`.
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
        n_restarts = int(np.ceil(n_frames / restart))
        restarts = mdt.rti.ProgressBar(
            restarts, total=n_restarts, unit="restarts"
        )
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


def n_leaves_vs_time(
    dtrj, discard_neg_start=False, discard_all_neg=False, verbose=False
):
    r"""
    Calculate the number of compounds that leave their state after a lag
    time :math:`\Delta t` given that they have entered the state at time
    :math:`t_0`.

    Take a discrete trajectory and calculate the total number of
    compounds that leave their state at time :math:`t_0 + \Delta t`
    given that they have entered the state at time :math:`t_0`.

    Additionally, calculate the number of compounds that are at risk to
    leave the state at time :math:`t_0 + \Delta t`, i.e. the number of
    compounds that have continuously been in the state from time
    :math:`t_0` to :math:`t_0 + \Delta t`.

    Parameters
    ----------
    dtrj : array_like
        The discrete trajectory.  Array of shape ``(n, f)``, where ``n``
        is the number of compounds and ``f`` is the number of frames.
        The shape can also be ``(f,)``, in which case the array is
        expanded to shape ``(1, f)``.   The elements of `dtrj` are
        interpreted as the indices of the states in which a given
        compound is at a given frame.
    discard_neg_start : bool, optional
        If ``True``, discard all state leavings starting from a negative
        state.  This means compounds in negative states are ignored.
        They neither increase `n_leaves` nor `n_risk`.  This is
        equivalent to discarding the lifetimes of all negative states
        when calculating state lifetimes with
        :func:`mdtools.dtrj.lifetimes`.
    discard_all_neg : bool, optional
        If ``True``, discard all state leavings starting from or ending
        in a negative state.  Additionally to ignoring compounds in
        negative states, transitions from positive to negative states
        are treated as right-censored.  These transitions increase
        `n_risk` but not `n_leaves`.  This is equivalent to discarding
        the lifetimes of all negative states and of all states that are
        followed by a negative state when calculating state lifetimes
        with :func:`mdtools.dtrj.lifetimes`.
    verbose : bool, optional
        If ``True`` print a progress bar.

    Returns
    -------
    n_leaves : numpy.ndarray
        Array of shape ``(f,)`` and dtype :attr:`numpy.uint32`
        containing for all possible lag times the number of compounds
        that leave their state at time :math:`t_0 + \Delta t` given that
        they have entered the state at time :math:`t_0`.
    n_risk : numpy.ndarray
        Array of shape ``(f,)`` and dtype :attr:`numpy.uint32`
        containing for all possible lag times the number of compounds
        that are at risk of leaving the state.

    See Also
    --------
    :func:`mdtools.dtrj.leave_prob` :
        Calculate the probability that a compound leaves its state after
        a lag time :math:`\Delta t` given that it has entered the state
        at time :math:`t_0`
    :func:`mdtools.dtrj.kaplan_meier` :
        Estimate the state survival function using the Kaplan-Meier
        estimator

    Notes
    -----
    ``n_leaves / n_risk`` is the probability that a compound leaves its
    state at time :math:`t_0 + \Delta t` given that it has entered the
    state at time :math:`t_0`.

    ``np.cumprod(1 - n_leaves / n_risk)`` is the Kaplan-Meier estimate
    of the survival function of the underlying distribution of state
    lifetimes. [#]_

    References
    ----------
    .. [#] E. L. Kaplan, P. Meier,
        `Nonparametric Estimation from Incomplete Observations
        <https://doi.org/10.1080/01621459.1958.10501452>`_,
        Journal of the American Statistical Association,
        1958, 53, 282, 457-481.

    Examples
    --------
    >>> # No detectable leave, two censored lifetimes.
    >>> dtrj = np.array([2, 2, 5, 5, 5, 5, 5])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time(dtrj)
    >>> n_leaves
    array([0, 0, 0, 0, 0, 0, 0], dtype=uint32)
    >>> n_risk
    array([2, 2, 2, 1, 1, 1, 0], dtype=uint32)
    >>> # One detectable leave, two censored lifetimes.
    >>> dtrj = np.array([2, 2, 3, 3, 3, 2, 2])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time(dtrj)
    >>> n_leaves
    array([0, 0, 0, 1, 0, 0, 0], dtype=uint32)
    >>> n_risk
    array([3, 3, 3, 1, 0, 0, 0], dtype=uint32)
    >>> # Two detectable leaves, two censored lifetimes.
    >>> dtrj = np.array([1, 3, 3, 3, 1, 2, 2])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time(dtrj)
    >>> n_leaves
    array([0, 1, 0, 1, 0, 0, 0], dtype=uint32)
    >>> n_risk
    array([4, 4, 2, 1, 0, 0, 0], dtype=uint32)
    >>> dtrj = np.array([1, 3, 3, 3, 2, 2, 1])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time(dtrj)
    >>> n_leaves
    array([0, 0, 1, 1, 0, 0, 0], dtype=uint32)
    >>> n_risk
    array([4, 4, 2, 1, 0, 0, 0], dtype=uint32)
    >>> dtrj = np.array([3, 3, 3, 1, 2, 2, 1])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time(dtrj)
    >>> n_leaves
    array([0, 1, 1, 0, 0, 0, 0], dtype=uint32)
    >>> n_risk
    array([4, 4, 2, 1, 0, 0, 0], dtype=uint32)

    >>> dtrj = np.array([[2, 2, 5, 5, 5, 5, 5],
    ...                  [2, 2, 3, 3, 3, 2, 2],
    ...                  [1, 3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2, 1],
    ...                  [3, 3, 3, 1, 2, 2, 1]])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time(dtrj)
    >>> n_leaves
    array([0, 2, 2, 3, 0, 0, 0], dtype=uint32)
    >>> n_risk
    array([17, 17, 11,  5,  1,  1,  0], dtype=uint32)
    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2]])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time(dtrj)
    >>> n_leaves
    array([0, 1, 1, 2, 0, 0], dtype=uint32)
    >>> n_risk
    array([12, 12,  8,  4,  0,  0], dtype=uint32)

    Discarding negative states:

    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2],
    ...                  [1, 4, 4, 4, 4, 1]])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time(dtrj)
    >>> n_leaves
    array([0, 1, 1, 2, 1, 0], dtype=uint32)
    >>> n_risk
    array([15, 15,  9,  5,  1,  0], dtype=uint32)
    >>> n_leaves_ns, n_risk_ns = mdt.dtrj.n_leaves_vs_time(
    ...     dtrj, discard_neg_start=True
    ... )
    >>> n_leaves_ns
    array([0, 1, 1, 2, 1, 0], dtype=uint32)
    >>> n_risk_ns
    array([15, 15,  9,  5,  1,  0], dtype=uint32)
    >>> n_leaves_an, n_risk_an = mdt.dtrj.n_leaves_vs_time(
    ...     dtrj, discard_all_neg=True
    ... )
    >>> n_leaves_an
    array([0, 1, 1, 2, 1, 0], dtype=uint32)
    >>> n_risk_an
    array([15, 15,  9,  5,  1,  0], dtype=uint32)
    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1]])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time(dtrj)
    >>> n_leaves
    array([0, 1, 1, 2, 1, 0], dtype=uint32)
    >>> n_risk
    array([15, 15,  9,  5,  1,  0], dtype=uint32)
    >>> n_leaves_ns, n_risk_ns = mdt.dtrj.n_leaves_vs_time(
    ...     dtrj, discard_neg_start=True
    ... )
    >>> n_leaves_ns
    array([0, 1, 0, 2, 1, 0], dtype=uint32)
    >>> n_risk_ns
    array([10, 10,  5,  5,  1,  0], dtype=uint32)
    >>> n_leaves_an, n_risk_an = mdt.dtrj.n_leaves_vs_time(
    ...     dtrj, discard_all_neg=True
    ... )
    >>> n_leaves_an
    array([0, 0, 0, 1, 0, 0], dtype=uint32)
    >>> n_risk_ns
    array([10, 10,  5,  5,  1,  0], dtype=uint32)
    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1],
    ...                  [ 6,  6,  6,  6,  6,  6],
    ...                  [-6, -6, -6, -6, -6, -6]])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time(dtrj)
    >>> n_leaves
    array([0, 1, 1, 2, 1, 0], dtype=uint32)
    >>> n_risk
    array([17, 17, 11,  7,  3,  2], dtype=uint32)
    >>> n_leaves_ns, n_risk_ns = mdt.dtrj.n_leaves_vs_time(
    ...     dtrj, discard_neg_start=True
    ... )
    >>> n_leaves_ns
    array([0, 1, 0, 2, 1, 0], dtype=uint32)
    >>> n_risk_ns
    array([11, 11,  6,  6,  2,  1], dtype=uint32)
    >>> n_leaves_an, n_risk_an = mdt.dtrj.n_leaves_vs_time(
    ...     dtrj, discard_all_neg=True
    ... )
    >>> n_leaves_an
    array([0, 0, 0, 1, 0, 0], dtype=uint32)
    >>> n_risk_ns
    array([11, 11,  6,  6,  2,  1], dtype=uint32)
    """
    dtrj = mdt.check.dtrj(dtrj)
    n_cmps, n_frames = dtrj.shape

    # Number of compounds that leave their state at frame i given that
    # they have entered the state at frame 0.
    n_leaves = np.zeros(n_frames, dtype=np.uint32)
    # Number of compounds that are at risk to leave their state at frame
    # i, i.e. number of compounds that have not left their state and
    # have not been censored at frame i-1.
    n_risk = np.zeros_like(n_leaves)

    if verbose:
        trj = mdt.rti.ProgressBar(dtrj, total=n_cmps, unit="compounds")
    else:
        trj = dtrj
    # Loop over single compound trajectories.
    for cmp_trj in trj:
        # Frames directly *before* state transitions.
        ix_trans = np.flatnonzero(np.diff(cmp_trj))
        # Frames directly *after* state transitions.
        ix_trans += 1

        if len(ix_trans) == 0:
            # There are no state transitions => the lifetime is greater
            # than the entire trajectory length.
            if (discard_neg_start or discard_all_neg) and cmp_trj[0] < 0:
                # Discard state leavings starting from a negative state.
                pass
            else:
                n_risk[:n_frames] += 1
            continue

        # Treatment of left-truncated lifetimes, i.e. treatment of the
        # frames before the first state transition.  The lifetime is
        # greater than the number of frames up to the first state
        # transition.
        if (discard_neg_start or discard_all_neg) and cmp_trj[0] < 0:
            # Discard state leavings starting from a negative state.
            pass
        # elif discard_all_neg and cmp_trj[ix_trans[0]] < 0:
        #     # Treat state transitions into negative states as
        #     # right-censored.
        #     # Left-truncated lifetimes are treated as right-censored
        #     # anyway => we can comment out this ``elif`` statement.
        #     n_risk[: ix_trans[0] + 1] += 1
        else:
            n_risk[: ix_trans[0] + 1] += 1

        # Loop over all state transitions.
        for i, t0 in enumerate(ix_trans, start=1):
            if (discard_neg_start or discard_all_neg) and cmp_trj[t0] < 0:
                # Discard state leavings starting from a negative state.
                continue
            if i < len(ix_trans):
                # Frame at which the compound leaves the new state for
                # the first time.
                t0_next = ix_trans[i]
                event_time = t0_next - t0
                if discard_all_neg and cmp_trj[t0_next] < 0:
                    # Treat state transitions into negative states as
                    # censored => event time = censoring time.
                    pass
                else:
                    # Event = leave => event time = lifetime.
                    n_leaves[event_time] += 1
            else:
                # The compound never leaves the new state => the
                # lifetime is right-censored.  The lifetime is greater
                # than the number of frames between the last state
                # transition and the end of the trajectory.
                # Event = censoring => event time = censoring time.
                event_time = n_frames - t0
            n_risk[: event_time + 1] += 1

    if n_leaves[0] != 0:
        raise ValueError(
            "`n_leaves[0]` = {} != 0.  This should not have"
            " happened".format(n_leaves[0])
        )
    if not discard_neg_start and not discard_all_neg and n_risk[0] < n_cmps:
        raise ValueError(
            "`n_risk[0]` ({}) < `n_cmps` ({}).  This should not have"
            " happened".format(n_risk[0], n_cmps)
        )
    return n_leaves, n_risk


def n_leaves_vs_time_discrete(
    dtrj1, dtrj2, discard_neg_start=False, discard_all_neg=False, verbose=False
):
    r"""
    Calculate the number of compounds that leave their state after a lag
    time :math:`\Delta t` resolved with respect to the states in a
    second discrete trajectory.

    Take a discrete trajectory and calculate the total number of
    compounds that leave their state at time :math:`t_0 + \Delta t`
    given that they have entered the state at time :math:`t_0` and given
    that they were in a specific state of another discrete trajectory at
    time :math:`t_0`.

    Additionally, calculate the number of compounds that are at risk to
    leave the state at time :math:`t_0 + \Delta t`, i.e. the number of
    compounds that have continuously been in the state from time
    :math:`t_0` to :math:`t_0 + \Delta t`.

    States whose starting point :math:`t_0` is not known (because the
    compound has already been in its state at the beginning of the
    trajectory) are discarded, because it is not known in which state of
    the second trajectory the compound was at time :math:`t_0`.

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
    discard_neg_start : bool, optional
        If ``True``, discard all state leavings starting from a negative
        state.  This means compounds in negative states are ignored.
        They neither increase `n_leaves` nor `n_risk`.  This is
        equivalent to discarding the lifetimes of all negative states
        when calculating state lifetimes with
        :func:`mdtools.dtrj.lifetimes`.
    discard_all_neg : bool, optional
        If ``True``, discard all state leavings starting from or ending
        in a negative state.  Additionally to ignoring compounds in
        negative states, transitions from positive to negative states
        are treated as right-censored.  These transitions increase
        `n_risk` but not `n_leaves`.  This is equivalent to discarding
        the lifetimes of all negative states and of all states that are
        followed by a negative state when calculating state lifetimes
        with :func:`mdtools.dtrj.lifetimes`.
    verbose : bool, optional
        If ``True`` print a progress bar.

    Returns
    -------
    n_leaves : numpy.ndarray
        Array of shape ``(m, f)`` and dtype :attr:`numpy.uint32`.  ``m``
        is the number of different states in the second discrete
        trajectory.  The `ij`-element of `n_leaves` is the number of
        compounds that leave their state j frames after they have
        entered it, given that the compounds were in state i of the
        second discrete trajectory at time :math:`t_0`.
    n_risk : numpy.ndarray
        Array of the same shape and dtype as `n_leaves` containing the
        corresponding number of compounds that are at risk of leaving
        their state.

    See Also
    --------
    :func:`mdtools.dtrj.n_leaves_vs_time` :
        Calculate the number of compounds that leave their state after a
        lag time :math:`\Delta t` given that they have entered the state
        at time :math:`t_0`
    :func:`mdtools.dtrj.leave_prob_discrete` :
        Calculate the probability that a compound leaves its state after
        a lag time :math:`\Delta t` given that it has entered the state
        at time :math:`t_0` resolved with respect to the states in a
        second discrete trajectory

    Notes
    -----
    If you parse the same discrete trajectory to `dtrj1` and `dtrj2` you
    will get `n_leaves` and `n_risk` for each individual state of the
    input trajectory.

    ``n_leaves / n_risk`` is the probability that a compound leaves its
    state at time :math:`t_0 + \Delta t` given that it has entered the
    state at time :math:`t_0`.

    ``np.cumprod(1 - n_leaves / n_risk, axis=-1)`` is the Kaplan-Meier
    estimate of the survival function of the underlying distribution of
    state lifetimes. [#]_

    References
    ----------
    .. [#] E. L. Kaplan, P. Meier,
        `Nonparametric Estimation from Incomplete Observations
        <https://doi.org/10.1080/01621459.1958.10501452>`_,
        Journal of the American Statistical Association,
        1958, 53, 282, 457-481.

    Examples
    --------
    >>> # 0 detectable leaves, 1 left-truncation, 1 right-censoring.
    >>> dtrj = np.array([2, 2, 5, 5, 5, 5, 5])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj
    ... )
    >>> n_leaves
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint32)
    >>> n_risk
    array([[0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 0]], dtype=uint32)
    >>> # 1 detectable leave, 1 left-truncation, 1 right-censoring.
    >>> dtrj = np.array([2, 2, 3, 3, 3, 2, 2])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj
    ... )
    >>> n_leaves
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0]], dtype=uint32)
    >>> n_risk
    array([[1, 1, 1, 0, 0, 0, 0],
           [1, 1, 1, 1, 0, 0, 0]], dtype=uint32)
    >>> # 2 detectable leaves, 1 left-truncation, 1 right-censoring.
    >>> dtrj = np.array([1, 3, 3, 3, 1, 2, 2])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj
    ... )
    >>> n_leaves
    array([[0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0]], dtype=uint32)
    >>> n_risk
    array([[1, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 0, 0, 0, 0],
           [1, 1, 1, 1, 0, 0, 0]], dtype=uint32)
    >>> dtrj = np.array([1, 3, 3, 3, 2, 2, 1])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj
    ... )
    >>> n_leaves
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0]], dtype=uint32)
    >>> n_risk
    array([[1, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 0, 0, 0, 0],
           [1, 1, 1, 1, 0, 0, 0]], dtype=uint32)
    >>> dtrj = np.array([3, 3, 3, 1, 2, 2, 1])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj
    ... )
    >>> n_leaves
    array([[0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint32)
    >>> n_risk
    array([[2, 2, 0, 0, 0, 0, 0],
           [1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint32)

    >>> dtrj = np.array([[2, 2, 5, 5, 5, 5, 5],
    ...                  [2, 2, 3, 3, 3, 2, 2],
    ...                  [1, 3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2, 1],
    ...                  [3, 3, 3, 1, 2, 2, 1]])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj
    ... )
    >>> n_leaves
    array([[0, 2, 0, 0, 0, 0, 0],
           [0, 0, 2, 0, 0, 0, 0],
           [0, 0, 0, 3, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint32)
    >>> n_risk
    array([[4, 4, 0, 0, 0, 0, 0],
           [4, 4, 4, 0, 0, 0, 0],
           [3, 3, 3, 3, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 0]], dtype=uint32)
    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2]])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj
    ... )
    >>> n_leaves
    array([[0, 1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 0, 2, 0, 0]], dtype=uint32)
    >>> n_risk
    array([[2, 2, 0, 0, 0, 0],
           [3, 3, 3, 0, 0, 0],
           [3, 3, 3, 3, 0, 0]], dtype=uint32)

    Discarding negative states:

    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2],
    ...                  [1, 4, 4, 4, 4, 1]])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj
    ... )
    >>> n_leaves
    array([[0, 1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 0, 2, 0, 0],
           [0, 0, 0, 0, 1, 0]], dtype=uint32)
    >>> n_risk
    array([[3, 3, 0, 0, 0, 0],
           [3, 3, 3, 0, 0, 0],
           [3, 3, 3, 3, 0, 0],
           [1, 1, 1, 1, 1, 0]], dtype=uint32)
    >>> n_leaves_ns, n_risk_ns = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj, discard_neg_start=True
    ... )
    >>> n_leaves_ns
    array([[0, 1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 0, 2, 0, 0],
           [0, 0, 0, 0, 1, 0]], dtype=uint32)
    >>> n_risk_ns
    array([[3, 3, 0, 0, 0, 0],
           [3, 3, 3, 0, 0, 0],
           [3, 3, 3, 3, 0, 0],
           [1, 1, 1, 1, 1, 0]], dtype=uint32)
    >>> n_leaves_an, n_risk_an = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj, discard_all_neg=True
    ... )
    >>> n_leaves_an
    array([[0, 1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 0, 2, 0, 0],
           [0, 0, 0, 0, 1, 0]], dtype=uint32)
    >>> n_risk_an
    array([[3, 3, 0, 0, 0, 0],
           [3, 3, 3, 0, 0, 0],
           [3, 3, 3, 3, 0, 0],
           [1, 1, 1, 1, 1, 0]], dtype=uint32)
    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1]])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj
    ... )
    >>> n_leaves
    array([[0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 0, 2, 0, 0],
           [0, 0, 0, 0, 1, 0]], dtype=uint32)
    >>> n_risk
    array([[3, 3, 3, 0, 0, 0],
           [1, 1, 0, 0, 0, 0],
           [2, 2, 0, 0, 0, 0],
           [3, 3, 3, 3, 0, 0],
           [1, 1, 1, 1, 1, 0]], dtype=uint32)
    >>> n_leaves_ns, n_risk_ns = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj, discard_neg_start=True
    ... )
    >>> n_leaves_ns
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 0, 2, 0, 0],
           [0, 0, 0, 0, 1, 0]], dtype=uint32)
    >>> n_risk_ns
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [2, 2, 0, 0, 0, 0],
           [3, 3, 3, 3, 0, 0],
           [1, 1, 1, 1, 1, 0]], dtype=uint32)
    >>> n_leaves_an, n_risk_an = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj, discard_all_neg=True
    ... )
    >>> n_leaves_an
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0]], dtype=uint32)
    >>> n_risk_an
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [2, 2, 0, 0, 0, 0],
           [3, 3, 3, 3, 0, 0],
           [1, 1, 1, 1, 1, 0]], dtype=uint32)
    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1],
    ...                  [ 6,  6,  6,  6,  6,  6],
    ...                  [-6, -6, -6, -6, -6, -6]])
    >>> n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj
    ... )
    >>> n_leaves
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 0, 2, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0]], dtype=uint32)
    >>> n_risk
    array([[0, 0, 0, 0, 0, 0],
           [3, 3, 3, 0, 0, 0],
           [1, 1, 0, 0, 0, 0],
           [2, 2, 0, 0, 0, 0],
           [3, 3, 3, 3, 0, 0],
           [1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0]], dtype=uint32)
    >>> n_leaves_ns, n_risk_ns = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj, discard_neg_start=True
    ... )
    >>> n_leaves_ns
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 0, 2, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0]], dtype=uint32)
    >>> n_risk_ns
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [2, 2, 0, 0, 0, 0],
           [3, 3, 3, 3, 0, 0],
           [1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0]], dtype=uint32)
    >>> n_leaves_an, n_risk_an = mdt.dtrj.n_leaves_vs_time_discrete(
    ...     dtrj1=dtrj, dtrj2=dtrj, discard_all_neg=True
    ... )
    >>> n_leaves_an
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]], dtype=uint32)
    >>> n_risk_an
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [2, 2, 0, 0, 0, 0],
           [3, 3, 3, 3, 0, 0],
           [1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0]], dtype=uint32)
    """
    dtrj1 = mdt.check.dtrj(dtrj1)
    dtrj2 = mdt.check.dtrj(dtrj2)
    n_cmps, n_frames = dtrj1.shape
    if dtrj1.shape != dtrj2.shape:
        raise ValueError("Both trajectories must have the same shape")

    # Reorder the states in the second trajectory such that they start
    # at zero and no intermediate states are missing to reduce the
    # memory required for the output arrays.
    dtrj2 = mdt.nph.sequenize(dtrj2, step=np.uint8(1), start=np.uint8(0))
    n_states = np.max(dtrj2) + 1
    if np.min(dtrj2) != 0:
        raise ValueError(
            "The minimum of the reordered second trajectory is not zero.  This"
            " should not have happened"
        )

    # Number of compounds that leave their state at frame i given that
    # they have entered the state at frame 0.
    n_leaves = np.zeros((n_states, n_frames), dtype=np.uint32)
    # Number of compounds that are at risk to leave their state at frame
    # i, i.e. number of compounds that have not left their state and
    # have not been censored at frame i-1.
    n_risk = np.zeros_like(n_leaves)

    if verbose:
        trj = mdt.rti.ProgressBar(dtrj1, total=n_cmps, unit="compounds")
    else:
        trj = dtrj1
    # Loop over single compound trajectories.
    for cmp_ix, cmp_trj in enumerate(trj):
        # Frames directly *before* state transitions.
        ix_trans = np.flatnonzero(np.diff(cmp_trj))
        # Frames directly *after* state transitions.
        ix_trans += 1

        # Left-truncated lifetimes, i.e. the frames before the first
        # state transition, are ignored, because the starting point t0
        # of these lifetimes and therefore the state of the second
        # discrete trajectory `dtrj2` in which the compound was at time
        # t0 is not known.
        if len(ix_trans) == 0:
            # There are no state transitions at all.
            continue

        # Loop over all state transitions.
        for i, t0 in enumerate(ix_trans, start=1):
            if (discard_neg_start or discard_all_neg) and cmp_trj[t0] < 0:
                # Discard state leavings starting from a negative state.
                continue
            state2 = dtrj2[cmp_ix, t0]
            if i < len(ix_trans):
                # Frame at which the compound leaves the new state for
                # the first time.
                t0_next = ix_trans[i]
                event_time = t0_next - t0
                if discard_all_neg and cmp_trj[t0_next] < 0:
                    # Treat state transitions into negative states as
                    # censored => event time = censoring time.
                    pass
                else:
                    # Event = leave => event time = lifetime.
                    n_leaves[state2, event_time] += 1
            else:
                # The compound never leaves the new state => the
                # lifetime is right-censored.  The lifetime is greater
                # than the number of frames between the last state
                # transition and the end of the trajectory.
                # Event = censoring => event time = censoring time.
                event_time = n_frames - t0
            n_risk[state2, : event_time + 1] += 1

    if np.any(n_leaves[:, 0] != 0):
        raise ValueError(
            "`n_leaves[:, 0]` = {} != 0.  This should not have"
            " happened".format(n_leaves[:, 0])
        )
    return n_leaves, n_risk


def leave_prob(*args, **kwargs):
    r"""
    Calculate the probability that a compound leaves its state after a
    lag time :math:`\Delta t` given that it has entered the state at
    time :math:`t_0`.

    Take a discrete trajectory and calculate the probability that a
    compound leaves its state at time :math:`t_0 + \Delta t` given that
    it has entered the state at time :math:`t_0`.

    Parameters
    ----------
    args, kwargs :
        This function takes the same parameters as
        :func:`mdtools.dtrj.n_leaves_vs_time`.  See there for more
        information.

    Returns
    -------
    prob : numpy.ndarray
        Array of shape ``(f,)`` containing for all possible lag times
        :math:`\Delta t` the probability that a compound leaves its
        state at time :math:`t_0 + \Delta t` given that they have
        entered the state at time :math:`t_0`.

    See Also
    --------
    :func:`mdtools.dtrj.leave_prob_discrete` :
        Calculate the probability that a compound leaves its state after
        a lag time :math:`\Delta t` given that it has entered the state
        at time :math:`t_0` resolved with respect to the states in a
        second discrete trajectory
    :func:`mdtools.dtrj.n_leaves_vs_time` :
        Calculate the number of compounds that leave their state after a
        lag time :math:`\Delta t` given that they have entered the state
        at time :math:`t_0`
    :func:`mdtools.dtrj.remain_prob` :
        Calculate the probability that a compound is still (or again) in
        the same state as at time :math:`t_0` after a lag time
        :math:`\Delta t`
    :func:`mdtools.dtrj.back_jump_prob` :
        Calculate the back-jump probability averaged over all states

    Notes
    -----
    This function simply calls :func:`mdtools.dtrj.n_leaves_vs_time`,
    calculates ``n_leaves / n_risk`` and checks that the result lies
    within the interval :math:`[0, 1]`.

    .. important::

        A leaving probability of zero means that no state leaving could
        be observed at the given lag time.  However, this does not
        necessarily mean that there definitely are no state leavings at
        this lag time, because if the lag time coincides with a
        censoring time, we simply do not know what happens.

    Examples
    --------
    >>> # No detectable leave, two censored lifetimes.
    >>> dtrj = np.array([2, 2, 5, 5, 5, 5, 5])
    >>> mdt.dtrj.leave_prob(dtrj)
    array([ 0.,  0.,  0.,  0.,  0.,  0., nan])
    >>> # One detectable leave, two censored lifetimes.
    >>> dtrj = np.array([2, 2, 3, 3, 3, 2, 2])
    >>> mdt.dtrj.leave_prob(dtrj)
    array([ 0.,  0.,  0.,  1., nan, nan, nan])
    >>> # Two detectable leaves, two censored lifetimes.
    >>> dtrj = np.array([1, 3, 3, 3, 1, 2, 2])
    >>> mdt.dtrj.leave_prob(dtrj)
    array([0.  , 0.25, 0.  , 1.  ,  nan,  nan,  nan])
    >>> dtrj = np.array([1, 3, 3, 3, 2, 2, 1])
    >>> mdt.dtrj.leave_prob(dtrj)
    array([0. , 0. , 0.5, 1. , nan, nan, nan])
    >>> dtrj = np.array([3, 3, 3, 1, 2, 2, 1])
    >>> mdt.dtrj.leave_prob(dtrj)
    array([0.  , 0.25, 0.5 , 0.  ,  nan,  nan,  nan])

    >>> dtrj = np.array([[2, 2, 5, 5, 5, 5, 5],
    ...                  [2, 2, 3, 3, 3, 2, 2],
    ...                  [1, 3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2, 1],
    ...                  [3, 3, 3, 1, 2, 2, 1]])
    >>> mdt.dtrj.leave_prob(dtrj)
    array([0.        , 0.11764706, 0.18181818, 0.6       , 0.        ,
           0.        ,        nan])
    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2]])
    >>> mdt.dtrj.leave_prob(dtrj)
    array([0.        , 0.08333333, 0.125     , 0.5       ,        nan,
                  nan])

    Discarding negative states:

    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2],
    ...                  [1, 4, 4, 4, 4, 1]])
    >>> mdt.dtrj.leave_prob(dtrj)
    array([0.        , 0.06666667, 0.11111111, 0.4       , 1.        ,
                  nan])
    >>> mdt.dtrj.leave_prob(dtrj, discard_neg_start=True)
    array([0.        , 0.06666667, 0.11111111, 0.4       , 1.        ,
                  nan])
    >>> mdt.dtrj.leave_prob(dtrj, discard_all_neg=True)
    array([0.        , 0.06666667, 0.11111111, 0.4       , 1.        ,
                  nan])
    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1]])
    >>> mdt.dtrj.leave_prob(dtrj)
    array([0.        , 0.06666667, 0.11111111, 0.4       , 1.        ,
                  nan])
    >>> mdt.dtrj.leave_prob(dtrj, discard_neg_start=True)
    array([0. , 0.1, 0. , 0.4, 1. , nan])
    >>> mdt.dtrj.leave_prob(dtrj, discard_all_neg=True)
    array([0. , 0. , 0. , 0.2, 0. , nan])
    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1],
    ...                  [ 6,  6,  6,  6,  6,  6],
    ...                  [-6, -6, -6, -6, -6, -6]])
    >>> mdt.dtrj.leave_prob(dtrj)
    array([0.        , 0.05882353, 0.09090909, 0.28571429, 0.33333333,
           0.        ])
    >>> mdt.dtrj.leave_prob(dtrj, discard_neg_start=True)
    array([0.        , 0.09090909, 0.        , 0.33333333, 0.5       ,
           0.        ])
    >>> mdt.dtrj.leave_prob(dtrj, discard_all_neg=True)
    array([0.        , 0.        , 0.        , 0.16666667, 0.        ,
           0.        ])
    """
    n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time(*args, **kwargs)
    prob = np.full_like(n_leaves, np.nan, dtype=np.float64)
    prob = np.divide(n_leaves, n_risk, where=(n_risk != 0), out=prob)
    del n_leaves, n_risk
    if prob[0] != 0:
        raise ValueError(
            "`prob[0]` = {} != 0.  This should not have"
            " happened".format(prob[0])
        )
    if np.any(prob < 0):
        raise ValueError(
            "At least one element of `prob` is less than zero.  This should"
            " not have happened"
        )
    if np.any(prob > 1):
        raise ValueError(
            "At least one element of `prob` is greater than one.  This should"
            " not have happened"
        )
    return prob


def leave_prob_discrete(*args, **kwargs):
    r"""
    Calculate the probability that a compound leaves its state after a
    lag time :math:`\Delta t` given that it has entered the state at
    time :math:`t_0` resolved with respect to the states in a second
    discrete trajectory.

    Take a discrete trajectory and calculate the probability that a
    compound leaves its state at time :math:`t_0 + \Delta t` given that
    it has entered the state at time :math:`t_0` and given that it was
    in a specific state of another discrete trajectory at time
    :math:`t_0`.

    States whose starting point :math:`t_0` is not known (because the
    compound has already been in its state at the beginning of the
    trajectory) are discarded, because it is not known in which state of
    the second trajectory the compound was at time :math:`t_0`.

    Parameters
    ----------
    args, kwargs :
        This function takes the same parameters as
        :func:`mdtools.dtrj.n_leaves_vs_time_discrete`.  See there for
        more information.

    Returns
    -------
    prob : numpy.ndarray
        Array of shape ``(m, f)`` where ``m`` is the number of different
        states in the second discrete trajectory.  The `ij`-element of
        `prob` is the probability that a compound leaves its state j
        frames after it has entered it, given that the compound was in
        state i of the second discrete trajectory at time :math:`t_0`.

    See Also
    --------
    :func:`mdtools.dtrj.leave_prob` :
        Calculate the probability that a compound leaves its state after
        a lag time :math:`\Delta t` given that it has entered the state
        at time :math:`t_0`
    :func:`mdtools.dtrj.n_leaves_vs_time_discrete` :
        Calculate the number of compounds that leave their state after a
        lag time :math:`\Delta t` resolved with respect to the states in
        a second discrete trajectory
    :func:`mdtools.dtrj.remain_prob_discrete` :
        Calculate the probability that a compound is still (or again) in
        the same state as at time :math:`t_0` after a lag time
        :math:`\Delta t` resolved with respect to the states in a second
        discrete trajectory
    :func:`mdtools.dtrj.back_jump_prob_discrete` :
        Calculate the back-jump probability resolved with respect to the
        states in a second discrete trajectory

    Notes
    -----
    If you parse the same discrete trajectory to `dtrj1` and `dtrj2` you
    will get the probability that a compound leaves its state after a
    lag time :math:`\Delta t` for each individual state of the input
    trajectory.

    This function simply calls
    :func:`mdtools.dtrj.n_leaves_vs_time_discrete`, calculates
    ``n_leaves / n_risk`` and checks that the result lies within the
    interval :math:`[0, 1]`.

    .. important::

        A leaving probability of zero means that no state leaving could
        be observed at the given lag time.  However, this does not
        necessarily mean that there definitely are no state leavings at
        this lag time, because if the lag time coincides with a
        censoring time, we simply do not know what happens.

    Examples
    --------
    >>> # 0 detectable leaves, 1 left-truncation, 1 right-censoring.
    >>> dtrj = np.array([2, 2, 5, 5, 5, 5, 5])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj)
    array([[nan, nan, nan, nan, nan, nan, nan],
           [ 0.,  0.,  0.,  0.,  0.,  0., nan]])
    >>> # 1 detectable leave, 1 left-truncation, 1 right-censoring.
    >>> dtrj = np.array([2, 2, 3, 3, 3, 2, 2])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj)
    array([[ 0.,  0.,  0., nan, nan, nan, nan],
           [ 0.,  0.,  0.,  1., nan, nan, nan]])
    >>> # 2 detectable leaves, 1 left-truncation, 1 right-censoring.
    >>> dtrj = np.array([1, 3, 3, 3, 1, 2, 2])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj)
    array([[ 0.,  1., nan, nan, nan, nan, nan],
           [ 0.,  0.,  0., nan, nan, nan, nan],
           [ 0.,  0.,  0.,  1., nan, nan, nan]])
    >>> dtrj = np.array([1, 3, 3, 3, 2, 2, 1])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj)
    array([[ 0.,  0., nan, nan, nan, nan, nan],
           [ 0.,  0.,  1., nan, nan, nan, nan],
           [ 0.,  0.,  0.,  1., nan, nan, nan]])
    >>> dtrj = np.array([3, 3, 3, 1, 2, 2, 1])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj)
    array([[0. , 0.5, nan, nan, nan, nan, nan],
           [0. , 0. , 1. , nan, nan, nan, nan],
           [nan, nan, nan, nan, nan, nan, nan]])

    >>> dtrj = np.array([[2, 2, 5, 5, 5, 5, 5],
    ...                  [2, 2, 3, 3, 3, 2, 2],
    ...                  [1, 3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2, 1],
    ...                  [3, 3, 3, 1, 2, 2, 1]])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj)
    array([[0. , 0.5, nan, nan, nan, nan, nan],
           [0. , 0. , 0.5, nan, nan, nan, nan],
           [0. , 0. , 0. , 1. , nan, nan, nan],
           [0. , 0. , 0. , 0. , 0. , 0. , nan]])
    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2]])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj)
    array([[0.        , 0.5       ,        nan,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.33333333,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.66666667,        nan,
                   nan]])

    Discarding negative states:

    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2],
    ...                  [1, 4, 4, 4, 4, 1]])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj)
    array([[0.        , 0.33333333,        nan,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.33333333,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.66666667,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.        , 1.        ,
                   nan]])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj, discard_neg_start=True)
    array([[0.        , 0.33333333,        nan,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.33333333,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.66666667,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.        , 1.        ,
                   nan]])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj, discard_all_neg=True)
    array([[0.        , 0.33333333,        nan,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.33333333,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.66666667,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.        , 1.        ,
                   nan]])
    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1]])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj)
    array([[0.        , 0.        , 0.33333333,        nan,        nan,
                   nan],
           [0.        , 0.        ,        nan,        nan,        nan,
                   nan],
           [0.        , 0.5       ,        nan,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.66666667,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.        , 1.        ,
                   nan]])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj, discard_neg_start=True)
    array([[       nan,        nan,        nan,        nan,        nan,
                   nan],
           [       nan,        nan,        nan,        nan,        nan,
                   nan],
           [0.        , 0.5       ,        nan,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.66666667,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.        , 1.        ,
                   nan]])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj, discard_all_neg=True)
    array([[       nan,        nan,        nan,        nan,        nan,
                   nan],
           [       nan,        nan,        nan,        nan,        nan,
                   nan],
           [0.        , 0.        ,        nan,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.33333333,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
                   nan]])
    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1],
    ...                  [ 6,  6,  6,  6,  6,  6],
    ...                  [-6, -6, -6, -6, -6, -6]])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj)
    array([[       nan,        nan,        nan,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.33333333,        nan,        nan,
                   nan],
           [0.        , 0.        ,        nan,        nan,        nan,
                   nan],
           [0.        , 0.5       ,        nan,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.66666667,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.        , 1.        ,
                   nan],
           [       nan,        nan,        nan,        nan,        nan,
                   nan]])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj, discard_neg_start=True)
    array([[       nan,        nan,        nan,        nan,        nan,
                   nan],
           [       nan,        nan,        nan,        nan,        nan,
                   nan],
           [       nan,        nan,        nan,        nan,        nan,
                   nan],
           [0.        , 0.5       ,        nan,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.66666667,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.        , 1.        ,
                   nan],
           [       nan,        nan,        nan,        nan,        nan,
                   nan]])
    >>> mdt.dtrj.leave_prob_discrete(dtrj, dtrj, discard_all_neg=True)
    array([[       nan,        nan,        nan,        nan,        nan,
                   nan],
           [       nan,        nan,        nan,        nan,        nan,
                   nan],
           [       nan,        nan,        nan,        nan,        nan,
                   nan],
           [0.        , 0.        ,        nan,        nan,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.33333333,        nan,
                   nan],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
                   nan],
           [       nan,        nan,        nan,        nan,        nan,
                   nan]])
    """
    n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time_discrete(*args, **kwargs)
    prob = np.full_like(n_leaves, np.nan, dtype=np.float64)
    prob = np.divide(n_leaves, n_risk, where=(n_risk != 0), out=prob)
    del n_leaves, n_risk
    if np.any(prob[:, 0][~np.isnan(prob[:, 0])] != 0):
        raise ValueError(
            "`prob[:, 0]` = {} != 0.  This should not have"
            " happened".format(prob[:, 0])
        )
    if np.any(prob < 0):
        raise ValueError(
            "At least one element of `prob` is less than zero.  This should"
            " not have happened"
        )
    if np.any(prob > 1):
        raise ValueError(
            "At least one element of `prob` is greater than one.  This should"
            " not have happened"
        )
    return prob


def kaplan_meier(*args, **kwargs):
    r"""
    Estimate the state survival function using the Kaplan-Meier
    estimator.

    Take a discrete trajectory and calculate the probability that a
    compound is still in the new state at time :math:`t_0 + \Delta t`
    given that a state transition has occurred at time :math:`t_0`.

    Parameters
    ----------
    args, kwargs :
        This function takes the same parameters as
        :func:`mdtools.dtrj.n_leaves_vs_time`.  See there for more
        information.

    Returns
    -------
    sf : numpy.ndarray
        Array of shape ``(f,)`` containing the values of the survival
        function for each possible lag time :math:`\Delta t`.  The i-th
        element of the array is the probability that a compound is still
        in the new state i frames after it has entered the state.
    sf_var : numpy.ndarray
        Array of the same shape as `sf` containing the corresponding
        variance values of the survival function calculated using
        Greenwood's formula.

    See Also
    --------
    :func:`mdtools.dtrj.n_leaves_vs_time` :
        Calculate the number of compounds that leave their state after a
        lag time :math:`\Delta t` given that they have entered the state
        at time :math:`t_0`
    :func:`mdtools.dtrj.remain_prob` :
        Calculate the probability that a compound is still (or again) in
        the same state as at time :math:`t_0` after a lag time
        :math:`\Delta t`
    :func:`mdtools.dtrj.trans_rate` :
        Calculate the transition rate for each compound averaged over
        all states
    :func:`mdtools.dtrj.lifetimes` :
        Calculate the state lifetimes for each compound

    Notes
    -----
    The survival function :math:`S(t)` is the probability that a given
    compound has a state lifetime greater than or equal to :math:`t`.
    Alternatively, one might say :math:`S(t)` percent of the compounds
    have a state lifetime greater than or equal to :math:`t`.

    The survival function is related to the underlying distribution of
    state lifetimes :math:`f(t)` by

    .. math::

        S(t) = \int_t^\infty f(u) \text{ d}u = 1 - F(t)

    where :math:`F(t) = \int_0^t f(u) \text{ d}u` is the cumulative
    distribution function. [#]_

    The survival function can be used to calculate the raw-moments of
    the distribution of state lifetimes via the alternative expectation
    formula: [#]_

    .. math::

        \langle t^n \rangle
        = \int_0^\infty t^n f(t) \text{ d}t
        = n \int_0^\infty t^{n-1} S(t) \text{ d}t

    The Kaplan-Meier estimator :math:`\hat{S}(t)` of the survival
    function :math:`S(t)` is given by [#]_

    .. math::

        \hat{S}(t) =
        \Pi_{i: t_i \leq t} \left( 1 - \frac{d_i}{n_i} \right)

    where :math:`d_i` is the number of compounds that leave their state
    at time :math:`t_i` and :math:`n_i` is the number of compounds that
    are at risk of leaving their state at time :math:`t_i`, i.e. the
    number of compounds that have not left their state and have not been
    censored up to time :math:`t_i`.

    The estimated variance :math:`\hat{\text{Var}}(\hat{S}(t))` of the
    Kaplan-Meier estimator :math:`\hat{S}(t)` is calculated by
    Greenwood's formula: [#]_

    .. math::

        \hat{\text{Var}}(\hat{S}(t)) =
        \hat{S}^2(t) \sum_{i: t_i \leq t} \frac{d_i}{n_i (n_i - d_i)}

    References
    ----------
    .. [#] J. F. Lawless,
        `Statistical Models and Methods for Lifetime Data
        <https://doi.org/10.1002/9781118033005>`_,
        John Wiley & Sons, 2002, Hoboken,
        Chapter 1.2 "Lifetime Distributions, pp. 8-16.
    .. [#] S. Chakraborti, F. Jardim, E. Epprecht,
        `Higher-Order Moments Using the Survival Function: The
        Alternative Expectation Formula
        <https://doi.org/10.1080/00031305.2017.1356374>`_,
        The American Statistician, 2019, 73, 2, 191-194.
    .. [#] E. L. Kaplan, P. Meier,
        `Nonparametric Estimation from Incomplete Observations
        <https://doi.org/10.1080/01621459.1958.10501452>`_,
        Journal of the American Statistical Association,
        1958, 53, 282, 457-481.
    .. [#] M. Greenwood,
        `The natural duration of cancer
        <https://www.cabdirect.org/cabdirect/abstract/19272700028>`_,
        Reports on public health and medical subjects, 1926, 33

    Examples
    --------
    >>> # No detectable leave, two censored lifetimes.
    >>> dtrj = np.array([2, 2, 5, 5, 5, 5, 5])
    >>> sf, sf_var = mdt.dtrj.kaplan_meier(dtrj)
    >>> sf
    array([ 1.,  1.,  1.,  1.,  1.,  1., nan])
    >>> sf_var
    array([ 0.,  0.,  0.,  0.,  0.,  0., nan])
    >>> # One detectable leave, two censored lifetimes.
    >>> dtrj = np.array([2, 2, 3, 3, 3, 2, 2])
    >>> sf, sf_var = mdt.dtrj.kaplan_meier(dtrj)
    >>> sf
    array([ 1.,  1.,  1.,  0., nan, nan, nan])
    >>> sf_var
    array([ 0.,  0.,  0.,  0., nan, nan, nan])
    >>> # Two detectable leaves, two censored lifetimes.
    >>> dtrj = np.array([1, 3, 3, 3, 1, 2, 2])
    >>> sf, sf_var = mdt.dtrj.kaplan_meier(dtrj)
    >>> sf
    array([1.  , 0.75, 0.75, 0.  ,  nan,  nan,  nan])
    >>> sf_var
    array([0.      , 0.046875, 0.046875, 0.      ,      nan,      nan,
                nan])
    >>> dtrj = np.array([1, 3, 3, 3, 2, 2, 1])
    >>> sf, sf_var = mdt.dtrj.kaplan_meier(dtrj)
    >>> sf
    array([1. , 1. , 0.5, 0. , nan, nan, nan])
    >>> sf_var
    array([0.   , 0.   , 0.125, 0.   ,   nan,   nan,   nan])
    >>> dtrj = np.array([3, 3, 3, 1, 2, 2, 1])
    >>> sf, sf_var = mdt.dtrj.kaplan_meier(dtrj)
    >>> sf
    array([1.   , 0.75 , 0.375, 0.375,   nan,   nan,   nan])
    >>> sf_var
    array([0.        , 0.046875  , 0.08203125, 0.08203125,        nan,
                  nan,        nan])

    >>> dtrj = np.array([[2, 2, 5, 5, 5, 5, 5],
    ...                  [2, 2, 3, 3, 3, 2, 2],
    ...                  [1, 3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2, 1],
    ...                  [3, 3, 3, 1, 2, 2, 1]])
    >>> sf, sf_var = mdt.dtrj.kaplan_meier(dtrj)
    >>> sf
    array([1.        , 0.88235294, 0.72192513, 0.28877005, 0.28877005,
           0.28877005,        nan])
    >>> sf_var
    array([0.        , 0.00610625, 0.01461646, 0.02735508, 0.02735508,
           0.02735508,        nan])
    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2]])
    >>> sf, sf_var = mdt.dtrj.kaplan_meier(dtrj)
    >>> sf
    array([1.        , 0.91666667, 0.80208333, 0.40104167,        nan,
                  nan])
    >>> sf_var
    array([0.        , 0.00636574, 0.01636194, 0.04429909,        nan,
                  nan])

    Discarding negative states:

    >>> dtrj = np.array([[1, 2, 2, 3, 3, 3],
    ...                  [2, 2, 3, 3, 3, 1],
    ...                  [3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2],
    ...                  [1, 4, 4, 4, 4, 1]])
    >>> sf, sf_var = mdt.dtrj.kaplan_meier(dtrj)
    >>> sf
    array([1.        , 0.93333333, 0.82962963, 0.49777778, 0.        ,
                  nan])
    >>> sf_var
    array([0.        , 0.00414815, 0.01283707, 0.03765904, 0.        ,
                  nan])
    >>> sf_ns, sf_var_ns = mdt.dtrj.kaplan_meier(
    ...     dtrj, discard_neg_start=True
    ... )
    >>> sf_ns
    array([1.        , 0.93333333, 0.82962963, 0.49777778, 0.        ,
                  nan])
    >>> sf_var_ns
    array([0.        , 0.00414815, 0.01283707, 0.03765904, 0.        ,
                  nan])
    >>> sf_an, sf_var_an = mdt.dtrj.kaplan_meier(
    ...     dtrj, discard_all_neg=True
    ... )
    >>> sf_an
    array([1.        , 0.93333333, 0.82962963, 0.49777778, 0.        ,
                  nan])
    >>> sf_var_an
    array([0.        , 0.00414815, 0.01283707, 0.03765904, 0.        ,
                  nan])
    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1]])
    >>> sf, sf_var = mdt.dtrj.kaplan_meier(dtrj)
    >>> sf
    array([1.        , 0.93333333, 0.82962963, 0.49777778, 0.        ,
                  nan])
    >>> sf_var
    array([0.        , 0.00414815, 0.01283707, 0.03765904, 0.        ,
                  nan])
    >>> sf_ns, sf_var_ns = mdt.dtrj.kaplan_meier(
    ...     dtrj, discard_neg_start=True
    ... )
    >>> sf_ns
    array([1.  , 0.9 , 0.9 , 0.54, 0.  ,  nan])
    >>> sf_var_ns
    array([0.     , 0.009  , 0.009  , 0.04212, 0.     ,     nan])
    >>> sf_an, sf_var_an = mdt.dtrj.kaplan_meier(
    ...     dtrj, discard_all_neg=True
    ... )
    >>> sf_an
    array([1. , 1. , 1. , 0.8, 0.8, nan])
    >>> sf_var_an
    array([0.   , 0.   , 0.   , 0.032, 0.032,   nan])
    >>> dtrj = np.array([[ 1, -2, -2,  3,  3,  3],
    ...                  [-2, -2,  3,  3,  3,  1],
    ...                  [ 3,  3,  3,  1, -2, -2],
    ...                  [ 1,  3,  3,  3, -2, -2],
    ...                  [ 1,  4,  4,  4,  4, -1],
    ...                  [ 6,  6,  6,  6,  6,  6],
    ...                  [-6, -6, -6, -6, -6, -6]])
    >>> sf, sf_var = mdt.dtrj.kaplan_meier(dtrj)
    >>> sf
    array([1.        , 0.94117647, 0.85561497, 0.61115355, 0.4074357 ,
           0.4074357 ])
    >>> sf_var
    array([0.        , 0.00325667, 0.0093467 , 0.02611208, 0.03927268,
           0.03927268])
    >>> sf_ns, sf_var_ns = mdt.dtrj.kaplan_meier(
    ...     dtrj, discard_neg_start=True
    ... )
    >>> sf_ns
    array([1.        , 0.90909091, 0.90909091, 0.60606061, 0.3030303 ,
           0.3030303 ])
    >>> sf_var_ns
    array([0.        , 0.00751315, 0.00751315, 0.0339483 , 0.05440076,
           0.05440076])
    >>> sf_an, sf_var_an = mdt.dtrj.kaplan_meier(
    ...     dtrj, discard_all_neg=True
    ... )
    >>> sf_an
    array([1.        , 1.        , 1.        , 0.83333333, 0.83333333,
           0.83333333])
    >>> sf_var_an
    array([0.        , 0.        , 0.        , 0.02314815, 0.02314815,
           0.02314815])
    """
    n_leaves, n_risk = mdt.dtrj.n_leaves_vs_time(*args, **kwargs)

    # sf = np.cumprod(1 - n_leaves / n_risk)
    sf = np.full_like(n_leaves, np.nan, dtype=np.float64)
    sf = np.divide(n_leaves, n_risk, where=(n_risk != 0), out=sf)
    sf = np.subtract(1, sf, out=sf)
    sf = np.cumprod(sf, out=sf)

    # sf_var =
    # sf**2 * np.cumsum(n_leaves / (n_risk * (n_risk - n_leaves)))
    sf_var = np.full_like(n_leaves, np.nan, dtype=np.float64)
    sf_var = np.subtract(n_risk, n_leaves, out=sf_var)
    sf_var *= n_risk
    sf_var = np.divide(n_leaves, sf_var, where=(sf_var != 0), out=sf_var)
    sf_var = np.cumsum(sf_var, out=sf_var)
    sf_var *= sf**2

    del n_leaves, n_risk
    if sf[0] != 1:
        raise ValueError(
            "`sf[0]` = {} != 1.  This should not have happened".format(sf[0])
        )
    if np.any(sf < 0):
        raise ValueError(
            "At least one element of `sf` is less than zero.  This should not"
            " have happened"
        )
    if np.any(sf > 1):
        raise ValueError(
            "At least one element of `sf` is greater than one.  This should"
            " not have happened"
        )
    grad = np.diff(sf)
    grad = grad[~np.isnan(grad)]
    if np.any(grad > 0):
        raise ValueError(
            "`sf` is not monotonically decreasing.  This should not have"
            " happened"
        )
    return sf, sf_var


def back_jump_prob(dtrj, continuous=False, verbose=False):
    r"""
    Calculate the back-jump probability averaged over all states.

    Take a discrete trajectory and calculate the probability to return
    back to the initial state at time :math:`t_0 + \Delta t`, given that
    a state transition has occurred at time :math:`t_0`.

    Parameters
    ----------
    dtrj : array_like
        The discrete trajectory.  Array of shape ``(n, f)``, where ``n``
        is the number of compounds and ``f`` is the number of frames.
        The shape can also be ``(f,)``, in which case the array is
        expanded to shape ``(1, f)``.   The elements of `dtrj` are
        interpreted as the indices of the states in which a given
        compound is at a given frame.
    continuous : bool
        If ``False``, calculate the probability that a compound returns
        back to its initial state at time :math:`t_0 + \Delta t`.  This
        probability might be regarded as the "discontinuous" or
        "intermittent" back-jump probability.

        If ``True``, calculate the probability that a compound returns
        back to its initial state at time :math:`t_0 + \Delta t` under
        the condition that it has *continuously* been in the new state
        from time :math:`t_0` until :math:`t_0 + \Delta t`, i.e. that
        the compound does not visit other states in between.  This
        probability might be regarded as the "continuous" or "true"
        back-jump probability.
    verbose : bool, optional
        If ``True`` print a progress bar.

    Returns
    -------
    bj_prob : numpy.ndarray
        Array of shape ``(f-1,)`` containing the back-jump probability
        for each possible lag time :math:`\Delta t`.  The k-th element
        of the array is the probability that a compound returns back to
        its initial state k frames after a state transition has
        occurred.  Another interpretation could be that the k-th element
        of the array is the percentage of compounds that return back to
        there initial state k frames after a state transition has
        occurred.

    See Also
    --------
    :func:`mdtools.dtrj.back_jump_prob_discrete` :
        Calculate the back-jump probability resolved with respect to the
        states a in second discrete trajectory
    :func:`mdtools.dtrj.remain_prob` :
        Calculate the probability that a compound is still (or again) in
        the same state as at time :math:`t_0` after a lag time
        :math:`\Delta t`
    :func:`mdtools.dtrj.leave_prob` :
        Calculate the probability that a compound leaves its state after
        a lag time :math:`\Delta t` given that it has entered the state
        at time :math:`t_0`

    Examples
    --------
    >>> dtrj = np.array([1, 3, 3, 3, 1, 2, 2])
    >>> mdt.dtrj.back_jump_prob(dtrj)
    array([0., 0., 0., 1., 0., 0.])
    >>> mdt.dtrj.back_jump_prob(dtrj, continuous=True)
    array([0., 0., 0., 1., 0., 0.])
    >>> dtrj = np.array([1, 3, 3, 3, 2, 2, 1])
    >>> mdt.dtrj.back_jump_prob(dtrj)
    array([0., 0., 0., 0., 0., 1.])
    >>> mdt.dtrj.back_jump_prob(dtrj, continuous=True)
    array([0., 0., 0., 0., 0., 0.])

    >>> dtrj = np.array([[1, 3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2, 1]])
    >>> mdt.dtrj.back_jump_prob(dtrj)
    array([0. , 0. , 0. , 0.5, 0. , 0.5])
    >>> mdt.dtrj.back_jump_prob(dtrj, continuous=True)
    array([0. , 0. , 0. , 0.5, 0. , 0. ])
    >>> dtrj = np.array([[1, 2, 2, 1, 3, 3, 3],
    ...                  [1, 2, 2, 3, 3, 3, 1]])
    >>> mdt.dtrj.back_jump_prob(dtrj)
    array([0. , 0. , 0.2, 0. , 0. , 0.5])
    >>> mdt.dtrj.back_jump_prob(dtrj, continuous=True)
    array([0. , 0. , 0.2, 0. , 0. , 0. ])
    >>> dtrj = np.array([[1, 2, 2, 1, 3, 3, 3],
    ...                  [1, 5, 5, 5, 5, 5, 1]])
    >>> mdt.dtrj.back_jump_prob(dtrj)
    array([0.  , 0.  , 0.25, 0.  , 0.  , 0.5 ])
    >>> mdt.dtrj.back_jump_prob(dtrj, continuous=True)
    array([0.  , 0.  , 0.25, 0.  , 0.  , 0.5 ])
    >>> dtrj = np.array([[1, 2, 2, 1, 2, 2],
    ...                  [2, 2, 1, 2, 2, 1]])
    >>> mdt.dtrj.back_jump_prob(dtrj)
    array([0. , 0.4, 0.5, 0. , 0. ])
    >>> mdt.dtrj.back_jump_prob(dtrj, continuous=True)
    array([0. , 0.4, 0.5, 0. , 0. ])
    """
    dtrj = mdt.check.dtrj(dtrj)
    n_cmps, n_frames = dtrj.shape

    bj_prob = np.zeros(n_frames - 1, dtype=np.uint32)
    norm = np.zeros_like(bj_prob)

    if verbose:
        trj = mdt.rti.ProgressBar(dtrj, total=n_cmps, unit="compounds")
    else:
        trj = dtrj
    # Loop over single compound trajectories.
    for cmp_trj in trj:
        # Frames directly *before* state transitions.
        ix_trans = np.flatnonzero(np.diff(cmp_trj))
        # Loop over all state transitions.
        for t0 in ix_trans:
            # Trajectory after the state transition.
            cmp_trj_a = cmp_trj[t0 + 1 :]
            # Maximum possible lag time for a back jump.
            max_lag = len(cmp_trj_a)
            norm[:max_lag] += 1
            # Frame at which the compound returns to the initial state
            # for the first time.
            # Here, `numpy.argmax` returns the first occurrence of
            # ``True``.  If the compound never returns back to the
            # initial state, all elements are ``False`` and
            # `numpy.argmax` returns ``0``.
            ix_back = np.argmax(cmp_trj_a == cmp_trj[t0])
            if continuous:
                # Frame at which the compound leaves the new state for
                # the first time.
                # Here, `numpy.argmin` returns the first occurrence of
                # ``False``.  If the compound never leaves the new
                # state, all elements are ``True`` and `numpy.argmin`
                # returns ``0``.
                ix_trans2 = np.argmin(cmp_trj_a == cmp_trj_a[0])
                # For a "continuous" back jump the compound must return
                # directly from the new state to the initial state
                # without visiting any other states in between,  i.e.
                # `ix_back` must be equal to `ix_trans2`.
                ix_back = ix_back if ix_back == ix_trans2 else 0
            if ix_back > 0:
                bj_prob[ix_back] += 1

    bj_prob = bj_prob / norm
    del norm
    if bj_prob[0] != 0:
        raise ValueError(
            "`bj_prob[0]` = {} != 0.  This should not have"
            " happened".format(bj_prob[0])
        )
    if np.any(bj_prob < 0):
        raise ValueError(
            "At least one element of `bj_prob` is less than zero.  This should"
            " not have happened"
        )
    if np.any(bj_prob > 1):
        raise ValueError(
            "At least one element of `bj_prob` is greater than one.  This"
            " should not have happened"
        )
    return bj_prob


def back_jump_prob_discrete(dtrj1, dtrj2, continuous=False, verbose=False):
    r"""
    Calculate the back-jump probability resolved with respect to the
    states in a second discrete trajectory.

    Take a discrete trajectory and calculate the probability to return
    back to the initial state at time :math:`t_0 + \Delta t`, given that
    a state transition has occurred at time :math:`t_0` and given that
    the compound was in a specific state of another discrete trajectory
    at time :math:`t_0`.

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
    continuous : bool
        If ``False``, calculate the probability that a compound returns
        back to its initial state at time :math:`t_0 + \Delta t`.  This
        probability might be regarded as the "discontinuous" or
        "intermittent" back-jump probability.

        If ``True``, calculate the probability that a compound returns
        back to its initial state at time :math:`t_0 + \Delta t` under
        the condition that it has *continuously* been in the new state
        from time :math:`t_0` until :math:`t_0 + \Delta t`, i.e. that
        the compound does not visit other states in between.  This
        probability might be regarded as the "continuous" or "true"
        back-jump probability.
    verbose : bool, optional
        If ``True`` print a progress bar.

    Returns
    -------
    bj_prob : numpy.ndarray
        Array of shape ``(m, f-1)``, where ``m`` is the number of states
        in the second discrete trajectory.  The `ij`-th element of
        `bj_prob` is the probability that a compound returns back to its
        initial state j frames after a state transition has occurred,
        given that the compound was in state i of the second discrete
        trajectory at time :math:`t_0`.

    See Also
    --------
    :func:`mdtools.dtrj.back_jump_prob` :
        Calculate the back-jump probability averaged over all states
    :func:`mdtools.dtrj.remain_prob_discrete` :
        Calculate the probability that a compound is still (or again) in
        the same state as at time :math:`t_0` after a lag time
        :math:`\Delta t` resolved with respect to the states in a second
        discrete trajectory
    :func:`mdtools.dtrj.leave_prob_discrete` :
        Calculate the probability that a compound leaves its state after
        a lag time :math:`\Delta t` given that it has entered the state
        at time :math:`t_0` resolved with respect to the states in a
        second discrete trajectory

    Notes
    -----
    If you parse the same discrete trajectory to `dtrj1` and `dtrj2` you
    will get the back-jump probability for each individual state of the
    input trajectory.

    Examples
    --------
    >>> dtrj = np.array([1, 3, 3, 3, 1, 2, 2])
    >>> mdt.dtrj.back_jump_prob_discrete(dtrj, dtrj)
    array([[ 0.,  0.,  0.,  1.,  0.,  0.],
           [nan, nan, nan, nan, nan, nan],
           [ 0.,  0.,  0., nan, nan, nan]])
    >>> mdt.dtrj.back_jump_prob_discrete(dtrj, dtrj, continuous=True)
    array([[ 0.,  0.,  0.,  1.,  0.,  0.],
           [nan, nan, nan, nan, nan, nan],
           [ 0.,  0.,  0., nan, nan, nan]])
    >>> dtrj = np.array([1, 3, 3, 3, 2, 2, 1])
    >>> mdt.dtrj.back_jump_prob_discrete(dtrj, dtrj)
    array([[ 0.,  0.,  0.,  0.,  0.,  1.],
           [ 0., nan, nan, nan, nan, nan],
           [ 0.,  0.,  0., nan, nan, nan]])
    >>> mdt.dtrj.back_jump_prob_discrete(dtrj, dtrj, continuous=True)
    array([[ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0., nan, nan, nan, nan, nan],
           [ 0.,  0.,  0., nan, nan, nan]])

    >>> dtrj = np.array([[1, 3, 3, 3, 1, 2, 2],
    ...                  [1, 3, 3, 3, 2, 2, 1]])
    >>> mdt.dtrj.back_jump_prob_discrete(dtrj, dtrj)
    array([[0. , 0. , 0. , 0.5, 0. , 0.5],
           [0. , nan, nan, nan, nan, nan],
           [0. , 0. , 0. , nan, nan, nan]])
    >>> mdt.dtrj.back_jump_prob_discrete(dtrj, dtrj, continuous=True)
    array([[0. , 0. , 0. , 0.5, 0. , 0. ],
           [0. , nan, nan, nan, nan, nan],
           [0. , 0. , 0. , nan, nan, nan]])
    >>> dtrj = np.array([[1, 2, 2, 1, 3, 3, 3],
    ...                  [1, 2, 2, 3, 3, 3, 1]])
    >>> mdt.dtrj.back_jump_prob_discrete(dtrj, dtrj)
    array([[0.        , 0.        , 0.33333333, 0.        , 0.        ,
            0.5       ],
           [0.        , 0.        , 0.        , 0.        ,        nan,
                   nan],
           [0.        ,        nan,        nan,        nan,        nan,
                   nan]])
    >>> mdt.dtrj.back_jump_prob_discrete(dtrj, dtrj, continuous=True)
    array([[0.        , 0.        , 0.33333333, 0.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , 0.        ,        nan,
                   nan],
           [0.        ,        nan,        nan,        nan,        nan,
                   nan]])
    >>> dtrj = np.array([[1, 2, 2, 1, 3, 3, 3],
    ...                  [1, 5, 5, 5, 5, 5, 1]])
    >>> mdt.dtrj.back_jump_prob_discrete(dtrj, dtrj)
    array([[0.        , 0.        , 0.33333333, 0.        , 0.        ,
            0.5       ],
           [0.        , 0.        , 0.        , 0.        ,        nan,
                   nan],
           [       nan,        nan,        nan,        nan,        nan,
                   nan],
           [0.        ,        nan,        nan,        nan,        nan,
                   nan]])
    >>> mdt.dtrj.back_jump_prob_discrete(dtrj, dtrj, continuous=True)
    array([[0.        , 0.        , 0.33333333, 0.        , 0.        ,
            0.5       ],
           [0.        , 0.        , 0.        , 0.        ,        nan,
                   nan],
           [       nan,        nan,        nan,        nan,        nan,
                   nan],
           [0.        ,        nan,        nan,        nan,        nan,
                   nan]])
    >>> dtrj = np.array([[1, 2, 2, 1, 2, 2],
    ...                  [2, 2, 1, 2, 2, 1]])
    >>> mdt.dtrj.back_jump_prob_discrete(dtrj, dtrj)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  0.,  0., nan]])
    >>> mdt.dtrj.back_jump_prob_discrete(dtrj, dtrj, continuous=True)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  0.,  0., nan]])
    """
    dtrj1 = mdt.check.dtrj(dtrj1)
    dtrj2 = mdt.check.dtrj(dtrj2)
    n_cmps, n_frames = dtrj1.shape
    if dtrj1.shape != dtrj2.shape:
        raise ValueError("Both trajectories must have the same shape")

    # Reorder the states in the second trajectory such that they start
    # at zero and no intermediate states are missing to reduce the
    # memory required for `bj_prob`.
    dtrj2 = mdt.nph.sequenize(dtrj2, step=np.uint8(1), start=np.uint8(0))
    n_states = np.max(dtrj2) + 1
    if np.min(dtrj2) != 0:
        raise ValueError(
            "The minimum of the reordered second trajectory is not zero.  This"
            " should not have happened"
        )

    bj_prob = np.zeros((n_states, n_frames - 1), dtype=np.uint32)
    norm = np.zeros_like(bj_prob)

    if verbose:
        trj = mdt.rti.ProgressBar(dtrj1, total=n_cmps, unit="compounds")
    else:
        trj = dtrj1
    # Loop over single compound trajectories.
    for cmp_ix, cmp_trj in enumerate(trj):
        # Frames directly *before* state transitions.
        ix_trans = np.flatnonzero(np.diff(cmp_trj))
        # Loop over all state transitions.
        for t0 in ix_trans:
            # Trajectory after the state transition.
            cmp_trj_a = cmp_trj[t0 + 1 :]
            # State that the compound occupies in the second discrete
            # trajectory at time t0.
            state2 = dtrj2[cmp_ix, t0]
            # Maximum possible lag time for a back jump.
            max_lag = len(cmp_trj_a)
            norm[state2, :max_lag] += 1
            # Frame at which the compound returns to the initial state
            # for the first time.
            # Here, `numpy.argmax` returns the first occurrence of
            # ``True``.  If the compound never returns back to the
            # initial state, all elements are ``False`` and
            # `numpy.argmax` returns ``0``.
            ix_back = np.argmax(cmp_trj_a == cmp_trj[t0])
            if continuous:
                # Frame at which the compound leaves the new state for
                # the first time.
                # Here, `numpy.argmin` returns the first occurrence of
                # ``False``.  If the compound never leaves the new
                # state, all elements are ``True`` and `numpy.argmin`
                # returns ``0``.
                ix_trans2 = np.argmin(cmp_trj_a == cmp_trj_a[0])
                # For a "continuous" back jump the compound must return
                # directly from the new state to the initial state
                # without visiting any other states in between,  i.e.
                # `ix_back` must be equal to `ix_trans2`.
                ix_back = ix_back if ix_back == ix_trans2 else 0
            if ix_back > 0:
                bj_prob[state2, ix_back] += 1

    if np.any(bj_prob[:, 0] != 0):
        raise ValueError(
            "`bj_prob[:, 0]` = {} != 0.  This should not have"
            " happened".format(bj_prob[:, 0])
        )
    bj_prob = bj_prob / norm
    del norm
    if np.any(bj_prob < 0):
        raise ValueError(
            "At least one element of `bj_prob` is less than zero.  This should"
            " not have happened"
        )
    if np.any(bj_prob > 1):
        raise ValueError(
            "At least one element of `bj_prob` is greater than one.  This"
            " should not have happened"
        )
    return bj_prob
