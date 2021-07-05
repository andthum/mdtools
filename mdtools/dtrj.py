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
    bins = np.arange(np.min(dtrj), np.max(dtrj)+2)
    hist_start = np.histogram(dtrj[trans_start], bins)[0]
    hist_end = np.histogram(dtrj[trans_end], bins)[0]
    return hist_start, hist_end
