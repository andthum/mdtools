# This file is part of MDTools.
# Copyright (C) 2021, 2022  The MDTools Development Team and all
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
Auxiliary functions for :class:`numpy.ndarrays <numpy.ndarray>` and
general mathematical functions.

This module can be called from :mod:`mdtools` via the shortcut ``nph``::

    import mdtools as mdt
    mdt.nph  # instead of mdt.numpy_helper_functions

.. todo::

    Finish/revise the implementation of the functions and their
    docstrings.  Revision is finished until
    :func:`mdtools.numpy_helper_functions.ceil_divide` (in source code
    order, not the alphabetical documentation order).  Basically, this
    means that all functions having an 'Examples' section in their
    docstring are revised, all other functions are not.
"""


# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter

# First-party libraries
import mdtools as mdt


def excel_colname(n, upper=True):
    """
    Convert an integer to its corresponding Excel-style column name.

    Parameters
    ----------
    n : int
        Column number to convert to column string.  If `n` is negative,
        it will be multiplied by ``-1``.  If `n` is zero, the return
        value will be ``'0'``.
    upper : bool, optional
        If ``True`` (default), return upper case column name.

    Returns
    -------
    col_name : str
        Excel-style column name.

    See Also
    --------
    :func:`mdtools.numpy_helper_functions.excel_colnum` :
        Inverse function

    Examples
    --------
    >>> mdt.nph.excel_colname(0)
    '0'

    >>> mdt.nph.excel_colname(1)
    'A'

    >>> mdt.nph.excel_colname(1, upper=False)
    'a'

    >>> mdt.nph.excel_colname(-1)
    'A'

    >>> mdt.nph.excel_colname(26)
    'Z'

    >>> mdt.nph.excel_colname(27)
    'AA'

    >>> mdt.nph.excel_colname(27, upper=False)
    'aa'
    """
    n = abs(n)
    if n == 0:
        return "0"
    if upper:
        first_letter = "A"
    else:
        first_letter = "a"
    col_name = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        col_name = chr(ord(first_letter) + remainder) + col_name
    return col_name


def excel_colnum(name):
    """
    Convert an Excel-style column name to its corresponding integer
    number.

    Parameters
    ----------
    name : str
        Excel-style column name to convert to column number.  Also
        ``name = '0'`` is accepted, in wich case the return value will
        be ``0``.

    Returns
    -------
    col_num : int
        Column number.

    See Also
    --------
    :func:`mdtools.numpy_helper_functions.excel_colname` :
        Inverse function

    Examples
    --------
    >>> mdt.nph.excel_colnum('0')
    0

    >>> mdt.nph.excel_colnum('A')
    1

    >>> mdt.nph.excel_colnum('a')
    1

    >>> mdt.nph.excel_colnum('Z')
    26

    >>> mdt.nph.excel_colnum('AA')
    27

    >>> mdt.nph.excel_colnum('aa')
    27

    >>> mdt.nph.excel_colnum('Aa')
    27

    >>> mdt.nph.excel_colnum('aA')
    27
    """
    if name == "0":
        return 0
    col_num = 0
    for char in name:
        if char.isupper():
            col_num = col_num * 26 + 1 + ord(char) - ord("A")
        else:
            col_num = col_num * 26 + 1 + ord(char) - ord("a")
    return col_num


def trailing_digits(n, digit=0):
    """
    Get the number of trailing digits in an integer.

    Parameters
    ----------
    n : int
        The integer for which to count the trailing digits.
    digit : int, optional
        The trailing digit to count.  Must be positive.

    Returns
    -------
    n_trailing : int
        The number of trailing digits.

    Examples
    --------
    >>> mdt.nph.trailing_digits(1, digit=0)
    0

    >>> mdt.nph.trailing_digits(1, digit=1)
    1

    >>> mdt.nph.trailing_digits(10, digit=0)
    1

    >>> mdt.nph.trailing_digits(10, digit=1)
    0

    >>> mdt.nph.trailing_digits(100, digit=0)
    2

    >>> mdt.nph.trailing_digits(100, digit=1)
    0

    >>> mdt.nph.trailing_digits(-100, digit=0)
    2
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError("n ({}) must be an integer".format(n))
    if not isinstance(digit, (int, np.integer)) or digit < 0:
        raise TypeError("digit ({}) must be a positive integer".format(digit))
    s = str(n)
    return len(s) - len(s.rstrip(str(digit)))


def take(a, start=None, stop=None, step=None, axis=None):
    """
    Take a slice from an array along an axis.

    For example, ``take(a, axis=2, start, stop, step)`` is equivalent to
    ``a[:,:,start:stop:step,...]``.

    Parameters
    ----------
    a : array_like
        The input array.
    start, stop, step : int or None, optional
        Start, stop and step of the slice.  See the built-in function
        :func:`slice` for more details.
    axis : int or None, optional
        The axis over which to select values.  By default, the flattened
        input array is used.

    Returns
    -------
    a_slice : numpy.ndarray
        A view of the requested slice along the requested axis.

    See Also
    --------
    :func:`numpy.take` :
        Similar function but supports fancy indexing and therefore
        always returns a copy instead of a view.

    Notes
    -----
    This function is similar to :func:`numpy.take`.  The difference is
    that this function only supports slicing and not fancy indexing but
    therefore always returns a view instead of a copy.

    References
    ----------
    Inspired by Leland Hepworth (https://stackoverflow.com/a/64436208)

    Examples
    --------
    >>> a = np.arange(6).reshape(2,3)
    >>> x = mdt.nph.take(a)
    >>> y = np.take(a, indices=np.arange(a.size))
    >>> np.array_equal(x, y)
    True
    >>> x
    array([0, 1, 2, 3, 4, 5])
    >>> x[0] = 6  # x is a view of a, i.e. changing x changes a.
    >>> x
    array([6, 1, 2, 3, 4, 5])
    >>> a
    array([[6, 1, 2],
           [3, 4, 5]])
    >>> y[0] = 7  # y is a copy of a, i.e. changing y does not affect a.
    >>> y
    array([7, 1, 2, 3, 4, 5])
    >>> a
    array([[6, 1, 2],
           [3, 4, 5]])
    >>> x = mdt.nph.take(a, axis=0)
    >>> y = np.take(a, indices=np.arange(a.shape[0]), axis=0)
    >>> np.array_equal(x, y)
    True
    >>> x
    array([[6, 1, 2],
           [3, 4, 5]])

    >>> a = np.arange(24).reshape(2,3,4)
    >>> mdt.nph.take(a, axis=0)
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    <BLANKLINE>
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> mdt.nph.take(a, start=1, axis=1)
    array([[[ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    <BLANKLINE>
           [[16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> mdt.nph.take(a, stop=3, axis=2)
    array([[[ 0,  1,  2],
            [ 4,  5,  6],
            [ 8,  9, 10]],
    <BLANKLINE>
           [[12, 13, 14],
            [16, 17, 18],
            [20, 21, 22]]])
    >>> np.array_equal(
    ...     mdt.nph.take(a, stop=3, axis=2),
    ...     mdt.nph.take(a, stop=-1, axis=2)
    ... )
    True
    >>> np.array_equal(
    ...     mdt.nph.take(a, stop=3, axis=2),
    ...     mdt.nph.take(a, stop=3, axis=-1)
    ... )
    True
    >>> np.array_equal(
    ...     mdt.nph.take(a, axis=2, stop=3),
    ...     np.take(a, indices=np.arange(a.shape[2] - 1), axis=2)
    ... )
    True
    >>> mdt.nph.take(a, step=2, axis=2)
    array([[[ 0,  2],
            [ 4,  6],
            [ 8, 10]],
    <BLANKLINE>
           [[12, 14],
            [16, 18],
            [20, 22]]])
    """
    a = np.asarray(a)
    if axis is None:
        return a.reshape(-1)[start:stop:step]
    else:
        if axis >= a.ndim or axis < -a.ndim:
            raise np.AxisError(axis=axis, ndim=a.ndim)
        # The modulo operator % ensures proper handling of negative axes
        sl = (slice(None),) * (axis % a.ndim) + (slice(start, stop, step),)
        return a[sl]


def ix_along_axis_to_global_ix(ix, axis):
    """
    Construct a tuple of index arrays suitable to directly index an
    array `a` from an array of indices along an axis of `a`.

    Parameters
    ----------
    ix : array_like
        Array of indices along a given `axis` of an array `a`.  The
        shape of `ix` must be equal to ``a.shape`` with the dimension
        along `axis` removed.  Otherwise, the resulting index array
        cannot be used to index `a`.
    axis : int
        The axis of `a` along which the indices `ix` were taken.

    Returns
    -------
    ix_global : tuple of numpy.ndarrays
        A tuple of index arrays that can be used to directly index `a`.

    See Also
    --------
    :func:`numpy.unravel_index` :
        Similar function for indices of a flattened version of `a`
    :func:`numpy.take` :
        Take elements from an array along an axis
    :func:`numpy.take_along_axis` :
        Take values from an array by matching 1d index and data slices
    :func:`mdtools.numpy_helper_functions.take` :
        Take a slice from an array along an axis


    Notes
    -----
    Functions like :func:`numpy.argmin` or :func:`numpy.argmax` return
    an "array of indices into the array [`a`].  It has the same shape as
    ``a.shape`` with the dimension along `axis` removed".  This index
    array can generally **not** be used to get the minimum or maximum
    values of `a` by simply doing ``a[ix]``.  Rather, you have to do
    ``np.squeeze(numpy.take_along_axis(a, numpy.expand_dims(ix, axis=axis), axis=axis))``
    (in this example, you can of course also use :func:`numpy.amin` and
    :func:`numpy.amax`).  But sometimes, the indices themselves, which
    would make ``a[ix]`` possible, are required.
    :func:`numpy.take_along_axis` and :func:`numpy.expand_dims` do not
    help in this case.

    This is where
    :func:`mdtools.numpy_helper_functions.ix_along_axis_to_global_ix` is
    useful.  It constructs a tuple of index arrays, `ix_global`, from
    `ix` such that ``a[ix_global]`` and
    ``np.squeeze(numpy.take_along_axis(a, numpy.expand_dims(ix, axis=axis), axis=axis))``
    give the same result.

    Examples
    --------
    If `a` is an 1-dimensional array, it can be directly indexed with
    `ix`.  In this case,
    :func:`mdtools.numpy_helper_functions.ix_along_axis_to_global_ix` is
    overkill.

    1-dimensional example:

    >>> a = np.array([1, 0, 2])
    >>> ax = 0

    >>> ix = np.argmin(a, axis=ax)
    >>> ix
    1
    >>> a[ix]
    0

    >>> ix_global = mdt.nph.ix_along_axis_to_global_ix(ix, ax)
    >>> ix_global
    array(1)
    >>> a[ix_global]
    0

    >>> np.squeeze(
    ...     np.take_along_axis(a, np.expand_dims(ix, axis=ax), axis=ax)
    ... )
    array(0)

    >>> np.min(a, axis=ax)
    0

    Only for for multidimensional arrays,
    :func:`mdtools.numpy_helper_functions.ix_along_axis_to_global_ix` is
    really useful, since in this case ``a[ix]`` yields a wrong result or
    an error.

    2-dimensional example:
    First axis:

    >>> b = np.array([[0, 1, 2],
    ...               [1, 2, 0]])
    >>> ax = 0

    >>> ix = np.argmin(b, axis=ax)
    >>> ix
    array([0, 0, 1])
    >>> b[ix]
    array([[0, 1, 2],
           [0, 1, 2],
           [1, 2, 0]])

    >>> ix_global = mdt.nph.ix_along_axis_to_global_ix(ix, ax)
    >>> ix_global
    (array([0, 0, 1]), array([0, 1, 2]))
    >>> b[ix_global]
    array([0, 1, 0])

    >>> np.squeeze(
    ...     np.take_along_axis(b, np.expand_dims(ix, axis=ax), axis=ax)
    ... )
    array([0, 1, 0])

    >>> np.min(b, axis=0)
    array([0, 1, 0])

    Second axis:

    >>> b
    array([[0, 1, 2],
           [1, 2, 0]])
    >>> ax = 1

    >>> ix = np.argmin(b, axis=ax)
    >>> ix
    array([0, 2])
    >>> b[ix]
    Traceback (most recent call last):
    ...
    IndexError: index 2 is out of bounds for axis 0 with size 2

    >>> ix_global = mdt.nph.ix_along_axis_to_global_ix(ix, ax)
    >>> ix_global
    (array([0, 1]), array([0, 2]))
    >>> b[ix_global]
    array([0, 0])

    >>> np.squeeze(
    ...     np.take_along_axis(b, np.expand_dims(ix, axis=ax), axis=ax)
    ... )
    array([0, 0])

    >>> np.min(b, axis=ax)
    array([0, 0])

    3-dimensional example:
    First axis:

    >>> c = np.array([[[0, 1, 2],
    ...                [1, 2, 0]],
    ...
    ...               [[2, 0, 1],
    ...                [0, 2, 1]]])
    >>> ax = 0

    >>> ix = np.argmin(c, axis=ax)
    >>> ix
    array([[0, 1, 1],
           [1, 0, 0]])
    >>> c[ix]
    array([[[[0, 1, 2],
             [1, 2, 0]],
    <BLANKLINE>
            [[2, 0, 1],
             [0, 2, 1]],
    <BLANKLINE>
            [[2, 0, 1],
             [0, 2, 1]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[2, 0, 1],
             [0, 2, 1]],
    <BLANKLINE>
            [[0, 1, 2],
             [1, 2, 0]],
    <BLANKLINE>
            [[0, 1, 2],
             [1, 2, 0]]]])

    >>> ix_global = mdt.nph.ix_along_axis_to_global_ix(ix, ax)
    >>> ix_global
    (array([[0, 1, 1],
           [1, 0, 0]]), array([[0, 0, 0],
           [1, 1, 1]]), array([[0, 1, 2],
           [0, 1, 2]]))
    >>> c[ix_global]
    array([[0, 0, 1],
           [0, 2, 0]])

    >>> np.squeeze(
    ...     np.take_along_axis(c, np.expand_dims(ix, axis=ax), axis=ax)
    ... )
    array([[0, 0, 1],
           [0, 2, 0]])

    >>> np.min(c, axis=ax)
    array([[0, 0, 1],
           [0, 2, 0]])

    Second axis:

    >>> c
    array([[[0, 1, 2],
            [1, 2, 0]],
    <BLANKLINE>
           [[2, 0, 1],
            [0, 2, 1]]])
    >>> ax = 1

    >>> ix = np.argmin(c, axis=ax)
    >>> ix
    array([[0, 0, 1],
           [1, 0, 0]])
    >>> c[ix]
    array([[[[0, 1, 2],
             [1, 2, 0]],
    <BLANKLINE>
            [[0, 1, 2],
             [1, 2, 0]],
    <BLANKLINE>
            [[2, 0, 1],
             [0, 2, 1]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[2, 0, 1],
             [0, 2, 1]],
    <BLANKLINE>
            [[0, 1, 2],
             [1, 2, 0]],
    <BLANKLINE>
            [[0, 1, 2],
             [1, 2, 0]]]])

    >>> ix_global = mdt.nph.ix_along_axis_to_global_ix(ix, ax)
    >>> ix_global
    (array([[0, 0, 0],
           [1, 1, 1]]), array([[0, 0, 1],
           [1, 0, 0]]), array([[0, 1, 2],
           [0, 1, 2]]))
    >>> c[ix_global]
    array([[0, 1, 0],
           [0, 0, 1]])

    >>> np.squeeze(
    ...     np.take_along_axis(c, np.expand_dims(ix, axis=ax), axis=ax)
    ... )
    array([[0, 1, 0],
           [0, 0, 1]])
    >>>
    >>> np.min(c, axis=ax)
    array([[0, 1, 0],
           [0, 0, 1]])

    Third axis:

    >>> c
    array([[[0, 1, 2],
            [1, 2, 0]],
    <BLANKLINE>
           [[2, 0, 1],
            [0, 2, 1]]])
    >>> ax = 2

    >>> ix = np.argmin(c, axis=ax)
    >>> ix
    array([[0, 2],
           [1, 0]])
    >>> c[ix]
    Traceback (most recent call last):
    ...
    IndexError: index 2 is out of bounds for axis 0 with size 2

    >>> ix_global = mdt.nph.ix_along_axis_to_global_ix(ix, ax)
    >>> ix_global
    (array([[0, 0],
           [1, 1]]), array([[0, 1],
           [0, 1]]), array([[0, 2],
           [1, 0]]))
    >>> c[ix_global]
    array([[0, 0],
           [0, 0]])

    >>> np.squeeze(
    ...     np.take_along_axis(c, np.expand_dims(ix, axis=ax), axis=ax)
    ... )
    array([[0, 0],
           [0, 0]])

    >>> np.min(c, axis=ax)
    array([[0, 0],
           [0, 0]])

    Edge cases:

    >>> ix = np.array([], dtype=int)
    >>> for ax in range(3):
    ...     mdt.nph.ix_along_axis_to_global_ix(ix, ax)
    (array([], dtype=int64), array([], dtype=int64))
    (array([], dtype=int64), array([], dtype=int64))
    (array([], dtype=int64), array([], dtype=int64))

    >>> ix = np.array(0)
    >>> for ax in range(3):
    ...     mdt.nph.ix_along_axis_to_global_ix(ix, ax)
    array(0)
    array(0)
    array(0)
    """  # noqa: W505,E501
    ix = np.asarray(ix)
    if ix.ndim == 0:
        return ix
    ix_global = np.split(np.indices(ix.shape), ix.ndim)
    ix_global.insert(axis, ix)
    return tuple(np.squeeze(ig) for ig in ix_global)


def find_nearest(a, val, axis=None, return_index=False):
    """
    Find the values in an array which are closest to a given value along
    an axis.

    Parameters
    ----------
    a : array_like
        The input array.
    val : scalar or array_like
        The value(s) to search for.  If a search value itself is not
        contained in `a` along the search axis, the value closest to
        `val` is returned.  If `val` is an array, `a` and `val` must be
        broadcastable.
    axis : None or int, optional
        The axis along which to search.  If `axis` is ``None``
        (default), a global search over the flattened array will be
        performed.
    return_index : bool, optional
        If ``True``, also return the indices of the first values in `a`
        that are closest to `val` along the search axis.

    Returns
    -------
    nearest : scalar or numpy.ndarray
        The first value(s) in `a` closest to `val` along `axis`.
    ix : int or numpy.ndarray of ints
        The indices of the first values in `a` nearest to `val` along
        the given axis.  Only returned if `return_index` is ``True``.

    See Also
    --------
    :func:`numpy.searchsorted` :
        Find indices where elements should be inserted to maintain order
    :func:`numpy.unravel_index` :
        Convert the returned index `ix` to a tuple of index arrays
        suitable to index a multidimensional input array `a` if `axis`
        was ``None``
    :func:`mdtools.numpy_helper_functions.ix_along_axis_to_global_ix` :
        Same as :func:`numpy.unravel_index`, but to be used when `axis`
        was not ``None``

    Notes
    -----
    In case of multiple occurrences of the closest values, only the
    *first* occurrence (and its index) is returned.

    Examples
    --------
    >>> a = np.arange(-1, 1.5, 0.5)
    >>> a
    array([-1. , -0.5,  0. ,  0.5,  1. ])
    >>> mdt.nph.find_nearest(a, 0)
    0.0
    >>> mdt.nph.find_nearest(a, 0, return_index=True)
    (0.0, 2)

    :func:`mdtools.numpy_helper_functions.find_nearest` returns the
    *first* value that is closest to the given value.

    >>> mdt.nph.find_nearest(a, -0.75, return_index=True)
    (-1.0, 0)
    >>> mdt.nph.find_nearest(a, 0.75, return_index=True)
    (0.5, 3)

    >>> b = np.array([0, 2, 4, 2])
    >>> mdt.nph.find_nearest(b, 2, return_index=True)
    (2, 1)

    If `axis` is ``None``,
    :func:`mdtools.numpy_helper_functions.find_nearest` searches
    globally in the flattened array for the value closest to the given
    value and returns its first occurrence.  To get an object suitable
    to index the input array, use :func:`numpy.unravel_index`.

    >>> c = np.array([[0, 1, 2],
    ...               [1, 2, 0]])
    >>> n, ix = mdt.nph.find_nearest(c, 1.8, return_index=True)
    >>> n
    2
    >>> ix
    2
    >>> ix_global = np.unravel_index(ix, c.shape)
    >>> ix_global
    (0, 2)
    >>> c[ix_global]
    2

    If `axis` is not ``None``,
    :func:`mdtools.numpy_helper_functions.find_nearest` returns the
    first occurrences along the given axis.  To get an object suitable
    to index the input array, use
    :func:`mdtools.numpy_helper_functions.ix_along_axis_to_global_ix`.

    >>> mdt.nph.find_nearest(c, 1.8, axis=0, return_index=True)
    (array([1, 2, 2]), array([1, 1, 0]))
    >>> n, ix = mdt.nph.find_nearest(c, 1.8, axis=1, return_index=True)
    >>> n
    array([2, 2])
    >>> ix
    array([2, 1])
    >>> ix_global = mdt.nph.ix_along_axis_to_global_ix(ix, axis=1)
    >>> ix_global
    (array([0, 1]), array([2, 1]))
    >>> c[ix_global]
    array([2, 2])

    3-dimensional example:

    >>> d = np.array([[[0, 1, 2],
    ...                [1, 2, 0]],
    ...
    ...               [[2, 0, 1],
    ...                [0, 2, 1]]])
    >>> n, ix = mdt.nph.find_nearest(d, 1.8, axis=0, return_index=True)
    >>> n
    array([[2, 1, 2],
           [1, 2, 1]])
    >>> ix
    array([[1, 0, 0],
           [0, 0, 1]])
    >>> n, ix = mdt.nph.find_nearest(d, 1.8, axis=1, return_index=True)
    >>> n
    array([[1, 2, 2],
           [2, 2, 1]])
    >>> ix
    array([[1, 1, 0],
           [0, 1, 0]])
    >>> n, ix = mdt.nph.find_nearest(d, 1.8, axis=2, return_index=True)
    >>> n
    array([[2, 2],
           [2, 2]])
    >>> ix
    array([[2, 1],
           [0, 1]])

    You can also specify an array of values to search for.

    >>> c
    array([[0, 1, 2],
           [1, 2, 0]])
    >>> x = np.array([1, 1, 0])
    >>> mdt.nph.find_nearest(c, x, axis=0, return_index=True)
    (array([1, 1, 0]), array([1, 0, 1]))

    Edge cases:

    >>> x = np.array([])
    >>> mdt.nph.find_nearest(x, 0, return_index=True)
    (array([], dtype=float64), array([], dtype=int64))
    >>> mdt.nph.find_nearest(x, 0, axis=0, return_index=True)
    (array([], dtype=float64), array([], dtype=int64))

    >>> x = np.array(0)
    >>> mdt.nph.find_nearest(x, 0, return_index=True)
    Traceback (most recent call last):
    ...
    ValueError: The dimension of a must be greater than zero
    """
    a = np.asarray(a)
    if a.ndim == 0:
        raise ValueError("The dimension of 'a' must be greater than zero")
    if (axis is not None and a.shape[axis] == 0) or a.size == 0:
        return (np.array([], dtype=a.dtype), np.array([], dtype=int))
    diff = np.subtract(a, val)
    np.abs(diff, out=diff)
    ix = np.argmin(diff, axis=axis)
    if a.ndim > 1:
        if axis is None:
            ix_global = np.unravel_index(ix, a.shape)
        else:
            ix_global = mdt.nph.ix_along_axis_to_global_ix(ix=ix, axis=axis)
    else:
        ix_global = ix
    if return_index:
        return a[ix_global], ix
    else:
        return a[ix_global]


def ix_of_item_change_1d(a):
    """
    Get the indices of a 1-dimensional array where its elements change.

    .. deprecated:: 0.0.0.dev0
        :func:`mdtools.numpy_helper_functions.ix_of_item_change_1d`
        might be removed in a future release.  It is replaced by
        :func:`mdtools.numpy_helper_functions.ix_of_item_change`,
        because this function works for arrays with arbitrary dimensions
        and provides additional functionality.
        :func:`mdtools.numpy_helper_functions.ix_of_item_change` with
        `prepend_zero` set to ``True`` is equivalent to
        :func:`mdtools.numpy_helper_functions.ix_of_item_change_1d`.

    .. todo::
        Check for scripts using this function before removing it.

    Parameters
    ----------
    a : array_like
        1-dimensional array for which to get each index where its
        elements change.

    Returns
    -------
    ix : numpy.ndarray
        The indices where the elements of `a` change.  The first element
        of `ix` is always zero.

    See Also
    --------
    :func:`mdtools.numpy_helper_functions.ix_of_item_change` :
        Same function for arrays of arbitrary dimension

    Examples
    --------
    >>> a = np.arange(3)
    >>> a
    array([0, 1, 2])
    >>> mdt.nph.ix_of_item_change_1d(a)
    array([0, 1, 2])

    >>> b = np.array([1, 2, 2, 3, 3, 3, 1])
    >>> mdt.nph.ix_of_item_change_1d(b)
    array([0, 1, 3, 6])

    >>> c = np.ones(3)
    >>> mdt.nph.ix_of_item_change_1d(c)
    array([0])

    Edge cases:

    >>> x = np.array([1])
    >>> mdt.nph.ix_of_item_change_1d(x)
    array([0])

    >>> x = np.array([])
    >>> ix = mdt.nph.ix_of_item_change_1d(x)
    >>> ix
    array([0])
    >>> x[ix]
    Traceback (most recent call last):
    ...
    IndexError: index 0 is out of bounds for axis 0 with size 0
    """
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(
            "The dimension of a must be 1 but is {}".format(a.ndim)
        )
    ix = np.flatnonzero(a[:-1] != a[1:])
    ix += 1
    ix = np.insert(ix, 0, 0)
    return ix


def ix_of_item_change(a, axis=-1, wrap=False, prepend_zero=False):
    """
    Get the indices of an array where its elements change along a given
    axis.

    .. deprecated:: 0.0.0.dev1
        :func:`mdtools.numpy_helper_functions.ix_of_item_change` might
        be removed in a future release.  It is replaced by
        :func:`mdtools.numpy_helper_functions.item_change_ix`, because
        this function provides additional functionality.
        :func:`mdtools.numpy_helper_functions.item_change_ix` with
        `pin` set to ``"after"`` is equivalent to
        :func:`mdtools.numpy_helper_functions.ix_of_item_change`.  The
        option `tfic` is equivalent to `prepend_zero`.

    .. todo::
        Check for modules and scripts using this function before
        removing it.

    Parameters
    ----------
    a : array_like
        Array for which to get each index where its elements change
        along the given axis.
    axis : int, optional
        The axis along which to track changing elements.
    wrap : bool, optional
        If ``True``, the array `a` is assumed to be continued after the
        last element along `axis` by `a` itself, like when using
        periodic boundary conditions.  Consequently, if the first and
        the last element of `a` do not match, this is interpreted as
        item change and the position (index) of this item change is
        regarded as zero.
    prepend_zero : bool, optional
        If ``True``, regard the first element of `a` along the given
        axis as item change and prepend a ``0`` to the returned array of
        indices.  Must not be used together with `wrap`.

    Returns
    -------
    ix : tuple
        Tuple of arrays, one for each dimension of `a`, containing the
        indices where the elements of `a` change.  `ix` can be used to
        index `a` and get the values of the changed elements.

    See Also
    --------
    :func:`mdtools.numpy_helper_functions.item_change_ix` :
        Equivalent function with additional features.

    Examples
    --------
    >>> a = np.arange(3)
    >>> a
    array([0, 1, 2])
    >>> mdt.nph.ix_of_item_change(a)
    (array([1, 2]),)
    >>> mdt.nph.ix_of_item_change(a, wrap=True)
    (array([0, 1, 2]),)
    >>> ix = mdt.nph.ix_of_item_change(a, prepend_zero=True)
    >>> ix
    (array([0, 1, 2]),)
    >>> a[ix]
    array([0, 1, 2])

    >>> b = np.array([1, 2, 2, 3, 3, 3, 1])
    >>> mdt.nph.ix_of_item_change(b)
    (array([1, 3, 6]),)
    >>> mdt.nph.ix_of_item_change(b, wrap=True)
    (array([1, 3, 6]),)
    >>> ix = mdt.nph.ix_of_item_change(b, prepend_zero=True)
    >>> ix
    (array([0, 1, 3, 6]),)
    >>> b[ix]
    array([1, 2, 3, 1])

    >>> c = np.ones(3)
    >>> ix = mdt.nph.ix_of_item_change(c)
    >>> ix
    (array([], dtype=int64),)
    >>> c[ix]
    array([], dtype=float64)
    >>> mdt.nph.ix_of_item_change(c, wrap=True)
    (array([], dtype=int64),)
    >>> mdt.nph.ix_of_item_change(c, prepend_zero=True)
    (array([0]),)

    2-dimensional example:

    >>> d = np.array([[1, 3, 3, 3],
    ...               [3, 1, 3, 3],
    ...               [3, 3, 1, 3]])
    >>> ax = 0
    >>> mdt.nph.ix_of_item_change(d, axis=ax)
    (array([1, 1, 2, 2]), array([0, 1, 1, 2]))
    >>> mdt.nph.ix_of_item_change(d, axis=ax, wrap=True)
    (array([0, 0, 1, 1, 2, 2]), array([0, 2, 0, 1, 1, 2]))
    >>> ix = mdt.nph.ix_of_item_change(d, axis=ax, prepend_zero=True)
    >>> ix
    (array([0, 0, 0, 0, 1, 1, 2, 2]), array([0, 1, 2, 3, 0, 1, 1, 2]))
    >>> d[ix]
    array([1, 3, 3, 3, 3, 1, 3, 1])

    >>> ax = 1
    >>> mdt.nph.ix_of_item_change(d, axis=ax)
    (array([0, 1, 1, 2, 2]), array([1, 1, 2, 2, 3]))
    >>> mdt.nph.ix_of_item_change(d, axis=ax, wrap=True)
    (array([0, 0, 1, 1, 2, 2]), array([0, 1, 1, 2, 2, 3]))
    >>> mdt.nph.ix_of_item_change(d, axis=ax, prepend_zero=True)
    (array([0, 0, 1, 1, 1, 2, 2, 2]), array([0, 1, 0, 1, 2, 0, 2, 3]))

    3-dimensional example:

    >>> e = np.array([[[1, 2, 2],
    ...                [2, 2, 1]],
    ...
    ...               [[2, 2, 1],
    ...                [1, 2, 2]]])
    >>> ax = 0
    >>> mdt.nph.ix_of_item_change(e, axis=ax)
    (array([1, 1, 1, 1]), array([0, 0, 1, 1]), array([0, 2, 0, 2]))
    >>> mdt.nph.ix_of_item_change(e, axis=ax, wrap=True)
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 2, 0, 2, 0, 2, 0, 2]))
    >>> ix = mdt.nph.ix_of_item_change(e, axis=ax, prepend_zero=True)
    >>> ix
    (array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1]), \
array([0, 1, 2, 0, 1, 2, 0, 2, 0, 2]))
    >>> e[ix]
    array([1, 2, 2, 2, 2, 1, 2, 1, 1, 2])

    >>> ax = 1
    >>> mdt.nph.ix_of_item_change(e, axis=ax)
    (array([0, 0, 1, 1]), array([1, 1, 1, 1]), array([0, 2, 0, 2]))
    >>> mdt.nph.ix_of_item_change(e, axis=ax, wrap=True)
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 2, 0, 2, 0, 2, 0, 2]))
    >>> mdt.nph.ix_of_item_change(e, axis=ax, prepend_zero=True)
    (array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), \
array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1]), \
array([0, 1, 2, 0, 2, 0, 1, 2, 0, 2]))

    >>> ax = 2
    >>> mdt.nph.ix_of_item_change(e, axis=ax)
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([1, 2, 2, 1]))
    >>> mdt.nph.ix_of_item_change(e, axis=ax, wrap=True)
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 1, 0, 2, 0, 2, 0, 1]))
    >>> mdt.nph.ix_of_item_change(e, axis=ax, prepend_zero=True)
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 1, 0, 2, 0, 2, 0, 1]))

    Edge cases:

    >>> x = np.array([1])
    >>> mdt.nph.ix_of_item_change(x)
    (array([], dtype=int64),)
    >>> mdt.nph.ix_of_item_change(x, wrap=True)
    (array([], dtype=int64),)
    >>> mdt.nph.ix_of_item_change(x, prepend_zero=True)
    (array([0]),)

    >>> x = np.array([])
    >>> mdt.nph.ix_of_item_change(x)
    (array([], dtype=int64),)
    >>> mdt.nph.ix_of_item_change(x, wrap=True)
    (array([], dtype=int64),)
    >>> ix = mdt.nph.ix_of_item_change(x, prepend_zero=True)
    >>> ix
    (array([], dtype=int64),)
    >>> x[ix]
    array([], dtype=float64)

    >>> x = np.array(0)
    >>> mdt.nph.ix_of_item_change(x)
    Traceback (most recent call last):
    ...
    ValueError: The dimension of a must be greater than zero
    """
    if wrap and prepend_zero:
        raise ValueError("wrap and prepend_zero must not be used together")
    a = np.asarray(a)
    if a.ndim == 0:
        raise ValueError("The dimension of a must be greater than zero")
    if a.shape[axis] == 0:
        return (np.array([], dtype=int),)
    d = np.diff(a, axis=axis)
    if wrap:
        insertion = a.take(0, axis=axis) - a.take(-1, axis=axis)
        d = np.insert(d, 0, insertion, axis=axis)
    elif prepend_zero:
        insertion = np.ones_like(a.take(0, axis=axis))
        d = np.insert(d, 0, insertion, axis=axis)
    ix = list(np.nonzero(d))
    if not wrap and not prepend_zero:
        ix[axis] += 1
    return tuple(ix)


def subtract_mic(x1, x2, amin=None, amax=None, **kwargs):
    """
    Subtract two arrays element-wise respecting the minium image
    convention.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be subtracted from each other.  If
        ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    amin, amax : scalar or array_like, optional
        The lower and upper bound(s) for the minimum image convention.
        If ``None``, the minium and maximum value of `a` is taken,
        respectively.  If `amin` and/or `amax` is an array, they must be
        broadcastable to a common shape and the difference
        ``amax - amin`` must be broadcastable to the shape of
        ``numpy.subtract(x1, x2, **kwargs)``.  `amin` must be smaller
        than `amax`.  If the difference between the `x1` and `x2` is
        larger than ``0.5 * (amax - amin)``, it is wrapped back to lie
        within that range (see Notes section).
    kwargs : dict, optional
        Keyword arguments to parse to :func:`numpy.subtract`.  Note that
        the keyword argument `dtype` is always set to ``numpy.float64``.

    Returns
    -------
    diff : numpy.ndarray
        The element-wise difference ``x1 - x2``.

    See Also
    --------
    :func:`numpy.subtract` :
        Subtract two arrays element-wise
    :func:`mdtools.numpy_helper_functions.diff_mic` :
        Calculate the difference between array elements respecting the
        minimum image convention.
    :func:`mdtools.box.vdist` :
        Calculate the distance vectors between two position arrays

    Notes
    -----
    This function is just a wrapper around :func:`numpy.subtract` that
    respects the minimum image convention.  The minimum image convention
    is taken into account using algorithm C4 from Deiters [#]_
    (``diff -= numpy.floor(diff / (amax - amin) + 0.5) *
    (amax - amin)``).

    In contrast to :func:`numpy.subtract`, the output array of this
    function always has dtype ``numpy.float64``.

    If you want to calculate distance vectors between particle
    positions, use :func:`mdtools.box.vdist`, because this function
    interprets position and box arrays correctly.

    References
    ----------
    .. [#] U. K. Deiters, `"Efficient Coding of the Minimum Image
            Convention" <https://doi.org/10.1524/zpch.2013.0311>`_,
            Zeitschrift für Physikalische Chemie, 2013, 227, 345-352.

    Examples
    --------
    >>> x1 = np.array([0, 2, 4])
    >>> x2 = np.array([5, 3, 1])
    >>> np.subtract(x1, x2)
    array([-5, -1,  3])
    >>> mdt.nph.subtract_mic(x1, x2)
    array([ 0., -1., -2.])
    >>> mdt.nph.subtract_mic(x1, x2, amin=0, amax=[3, 2, 2])
    array([ 1., -1., -1.])

    >>> x1 = np.array([[0, 2, 4],
    ...                [5, 3, 1]])
    >>> x2 = np.array([[5, 3, 1],
    ...                [0, 2, 4]])
    >>> np.subtract(x1, x2)
    array([[-5, -1,  3],
           [ 5,  1, -3]])
    >>> mdt.nph.subtract_mic(x1, x2)
    array([[ 0., -1., -2.],
           [ 0.,  1.,  2.]])
    >>> mdt.nph.subtract_mic(x1, x2, amin=0, amax=[3, 2, 2])
    array([[ 1., -1., -1.],
           [-1., -1., -1.]])
    >>> mdt.nph.subtract_mic(
    ...     x1, x2, amin=0, amax=[[3, 2, 2], [2, 3, 4]]
    ... )
    array([[ 1., -1., -1.],
           [-1.,  1.,  1.]])

    >>> x1 = np.array([[[0, 2, 4],
    ...                 [5, 3, 1]],
    ...
    ...                [[4, 0, 2],
    ...                 [5, 3, 1]]])
    >>> x2 = np.array([[[5, 3, 1],
    ...                 [0, 2, 4]],
    ...
    ...                [[5, 3, 1],
    ...                 [4, 0, 2]]])
    >>> np.subtract(x1, x2)
    array([[[-5, -1,  3],
            [ 5,  1, -3]],
    <BLANKLINE>
           [[-1, -3,  1],
            [ 1,  3, -1]]])
    >>> mdt.nph.subtract_mic(x1, x2)
    array([[[ 0., -1., -2.],
            [ 0.,  1.,  2.]],
    <BLANKLINE>
           [[-1.,  2.,  1.],
            [ 1., -2., -1.]]])
    >>> mdt.nph.subtract_mic(x1, x2, amin=0, amax=[3, 2, 2])
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [ 1., -1., -1.]]])
    >>> # The following behavior is different to `mdtools.box.vdist`!
    >>> mdt.nph.subtract_mic(
    ...     x1, x2, amin=0, amax=[[3, 2, 2], [2, 3, 4]]
    ... )
    array([[[ 1., -1., -1.],
            [-1.,  1.,  1.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [-1.,  0., -1.]]])

    >>> x1 = np.array([0, 2, 4])
    >>> x2 = np.array([[5, 3, 1],
    ...                [0, 2, 4]])
    >>> np.subtract(x1, x2)
    array([[-5, -1,  3],
           [ 0,  0,  0]])
    >>> mdt.nph.subtract_mic(x1, x2)
    array([[ 0., -1., -2.],
           [ 0.,  0.,  0.]])
    >>> mdt.nph.subtract_mic(x1, x2, amin=0, amax=[3, 2, 2])
    array([[ 1., -1., -1.],
           [ 0.,  0.,  0.]])
    >>> mdt.nph.subtract_mic(
    ...     x1, x2, amin=0, amax=[[3, 2, 2], [2, 3, 4]]
    ... )
    array([[ 1., -1., -1.],
           [ 0.,  0.,  0.]])

    >>> x1 = np.array([0, 2, 4])
    >>> x2 = np.array([[[5, 3, 1],
    ...                 [0, 2, 4]],
    ...
    ...                [[5, 3, 1],
    ...                 [4, 0, 2]]])
    >>> np.subtract(x1, x2)
    array([[[-5, -1,  3],
            [ 0,  0,  0]],
    <BLANKLINE>
           [[-5, -1,  3],
            [-4,  2,  2]]])
    >>> mdt.nph.subtract_mic(x1, x2)
    array([[[ 0., -1., -2.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[ 0., -1., -2.],
            [ 1.,  2.,  2.]]])
    >>> mdt.nph.subtract_mic(x1, x2, amin=0, amax=[3, 2, 2])
    array([[[ 1., -1., -1.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[ 1., -1., -1.],
            [-1.,  0.,  0.]]])
    >>> # The following behavior is different to `mdtools.box.vdist`!
    >>> mdt.nph.subtract_mic(
    ...     x1, x2, amin=0, amax=[[3, 2, 2], [2, 3, 4]]
    ... )
    array([[[ 1., -1., -1.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[ 1., -1., -1.],
            [ 0., -1., -2.]]])

    >>> x1 = np.array([[0, 2, 4],
    ...                [5, 3, 1]])
    >>> x2 = np.array([[[5, 3, 1],
    ...                 [0, 2, 4]],
    ...
    ...                [[5, 3, 1],
    ...                 [4, 0, 2]]])
    >>> np.subtract(x1, x2)
    array([[[-5, -1,  3],
            [ 5,  1, -3]],
    <BLANKLINE>
           [[-5, -1,  3],
            [ 1,  3, -1]]])
    >>> mdt.nph.subtract_mic(x1, x2)
    array([[[ 0., -1., -2.],
            [ 0.,  1.,  2.]],
    <BLANKLINE>
           [[ 0., -1., -2.],
            [ 1., -2., -1.]]])
    >>> mdt.nph.subtract_mic(x1, x2, amin=0, amax=[3, 2, 2])
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[ 1., -1., -1.],
            [ 1., -1., -1.]]])
    >>> # The following behavior is different to `mdtools.box.vdist`!
    >>> mdt.nph.subtract_mic(
    ...     x1, x2, amin=0, amax=[[3, 2, 2], [2, 3, 4]]
    ... )
    array([[[ 1., -1., -1.],
            [-1.,  1.,  1.]],
    <BLANKLINE>
           [[ 1., -1., -1.],
            [-1.,  0., -1.]]])
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    amin = min(np.min(x1), np.min(x2)) if amin is None else amin
    amax = max(np.max(x1), np.max(x2)) if amax is None else amax
    # np.subtract keeps the dtype of the input arrays => If the dtype of
    # the input arrays is an unsigned integer type, negative differences
    # are not possible.
    gap = np.subtract(amax, amin, dtype=np.float64, casting="safe")
    if np.any(gap <= 0):
        raise ValueError(
            "'amax' ({}) must be greater than 'amin' ({})".format(amax, amin)
        )
    kwargs.pop("dtype", None)
    casting = kwargs.pop("casting", "safe")
    diff = np.subtract(x1, x2, dtype=np.float64, casting=casting, **kwargs)
    diff -= np.floor(diff / gap + 0.5) * gap
    return diff


def diff_mic(a, amin=None, amax=None, **kwargs):
    """
    Calculate the difference between array elements respecting the
    minimum image convention.

    Parameters
    ----------
    a : array_like
        Input array.
    amin, amax : scalar or array_like, optional
        The lower and upper bound(s) for the minimum image convention.
        If ``None``, the minium and maximum value of `a` is taken,
        respectively.  If `amin` and/or `amax` is an array, they must be
        broadcastable to a common shape and the difference
        ``amax - amin`` must be broadcastable to the shape of
        ``numpy.diff(a, **kwargs)``.  `amin` must be smaller than
        `amax`.  If the difference between two elements in `a` is larger
        than ``0.5 * (amax - amin)``, it is wrapped back to lie within
        that range (see Notes section).
    kwargs : dict, optional
        Keyword arguments to parse to :func:`numpy.diff`.  Note that the
        keyword argument `n` of :func:`numpy.diff` is not supported by
        this function.

    Returns
    -------
    diff : numpy.ndarray
        The differences between the elements of the input array along
        the given axis.  The shape of `diff` is the same as the shape of
        `a` except along the given axis where the dimension is smaller
        by 1.  The dtype of `diff` is ``numpy.float64``.

    See Also
    --------
    :func:`numpy.diff` :
        Calculate the n-th discrete difference along a given axis
    :func:`mdtools.numpy_helper_functions.subtract_mic` :
        Subtract two arrays element-wise respecting the minium image
        convention.
    :func:`mdtools.box.vdist` :
        Calculate the distance vectors between two position arrays

    Notes
    -----
    This function is just a wrapper around :func:`numpy.diff` that
    respects the minimum image convention.  The minimum image convention
    is taken into account using algorithm C4 from Deiters [#]_
    (``diff -= numpy.floor(diff / (amax - amin) + 0.5) *
    (amax - amin)``).

    In contrast to :func:`numpy.diff`, the output array of this function
    always has dtype ``numpy.float64``.

    References
    ----------
    .. [#] U. K. Deiters, `"Efficient Coding of the Minimum Image
            Convention" <https://doi.org/10.1524/zpch.2013.0311>`_,
            Zeitschrift für Physikalische Chemie, 2013, 227, 345-352.

    Examples
    --------
    >>> a = np.array([5, 0, 3, 2, 1, 4])
    >>> np.diff(a)
    array([-5,  3, -1, -1,  3])
    >>> mdt.nph.diff_mic(a)
    array([ 0., -2., -1., -1., -2.])
    >>> mdt.nph.diff_mic(a, amin=0, amax=[3, 1.5, 2, 0.5, 2])
    array([ 1.,  0., -1.,  0., -1.])

    >>> a = np.array([[5, 3, 1],
    ...               [0, 2, 4]])
    >>> axis = 0
    >>> np.diff(a, axis=axis)
    array([[-5, -1,  3]])
    >>> mdt.nph.diff_mic(a, axis=axis)
    array([[ 0., -1., -2.]])
    >>> mdt.nph.diff_mic(a, axis=axis, amin=0, amax=[3, 2, 2])
    array([[ 1., -1., -1.]])
    >>> axis = 1
    >>> np.diff(a, axis=axis)
    array([[-2, -2],
           [ 2,  2]])
    >>> mdt.nph.diff_mic(a, axis=axis)
    array([[-2., -2.],
           [ 2.,  2.]])
    >>> mdt.nph.diff_mic(a, axis=axis, amin=0, amax=[3, 2])
    array([[ 1.,  0.],
           [-1.,  0.]])

    >>> a = np.array([[[0, 2, 4],
    ...                [5, 3, 1]],
    ...
    ...               [[4, 0, 2],
    ...                [5, 3, 1]]])
    >>> axis = 0
    >>> np.diff(a, axis=axis)
    array([[[ 4, -2, -2],
            [ 0,  0,  0]]])
    >>> mdt.nph.diff_mic(a, axis=axis)
    array([[[-1., -2., -2.],
            [ 0.,  0.,  0.]]])
    >>> mdt.nph.diff_mic(a, axis=axis, amin=0, amax=[3, 2, 2])
    array([[[1., 0., 0.],
            [0., 0., 0.]]])
    >>> axis = 1
    >>> np.diff(a, axis=axis)
    array([[[ 5,  1, -3]],
    <BLANKLINE>
           [[ 1,  3, -1]]])
    >>> mdt.nph.diff_mic(a, axis=axis)
    array([[[ 0.,  1.,  2.]],
    <BLANKLINE>
           [[ 1., -2., -1.]]])
    >>> mdt.nph.diff_mic(a, axis=axis, amin=0, amax=[3, 2, 2])
    array([[[-1., -1., -1.]],
    <BLANKLINE>
           [[ 1., -1., -1.]]])
    >>> axis = 2
    >>> np.diff(a, axis=axis)
    array([[[ 2,  2],
            [-2, -2]],
    <BLANKLINE>
           [[-4,  2],
            [-2, -2]]])
    >>> mdt.nph.diff_mic(a, axis=axis)
    array([[[ 2.,  2.],
            [-2., -2.]],
    <BLANKLINE>
           [[ 1.,  2.],
            [-2., -2.]]])
    >>> mdt.nph.diff_mic(a, axis=axis, amin=0, amax=[3, 2])
    array([[[-1.,  0.],
            [ 1.,  0.]],
    <BLANKLINE>
           [[-1.,  0.],
            [ 1.,  0.]]])
    """
    # np.diff and np.subtract keep the dtype of the input array(s) => If
    # the dtype of the input array(s) is an unsigned integer type,
    # negative differences are not possible.
    a = np.asarray(a).astype(dtype=np.float64, casting="safe")
    if "n" in kwargs:
        raise TypeError(
            "The keyword argument 'n' of numpy.diff is not supported by this"
            " function"
        )
    amin = np.min(a) if amin is None else amin
    amax = np.max(a) if amax is None else amax
    gap = np.subtract(amax, amin, dtype=np.float64, casting="safe")
    if np.any(gap <= 0):
        raise ValueError(
            "'amax' ({}) must be greater than 'amin' ({})".format(amax, amin)
        )
    diff = np.diff(a, **kwargs)
    diff -= np.floor(diff / gap + 0.5) * gap
    return diff


def locate_item_change(  # noqa: C901
    a,
    axis=-1,
    pin="after",
    change_type=None,
    rtol=1e-05,
    atol=1e-08,
    wrap=False,
    tfic=False,
    tlic=False,
    mic=False,
    amin=None,
    amax=None,
):
    """
    Locate the positions of item changes in an array.

    Get a boolean array of the same shape as `a` that indicates at which
    positions the elements of `a` change their value.

    .. todo::

        * Probably, this function should be split into several smaller
          functions, e.g.

            * `locate_item_change_start(a, axis, wrap, tfic, tlic, mic,
              amin, amax)`
              corresponds to ``pin="before", change_type="None"``.
            * `locate_item_change_start_pos(a, axis, wrap, tfic, tlic,
              mic, amin, amax)`
              corresponds to ``pin="before", change_type="higher"``.
            * `locate_item_change_start_neg(a, axis, wrap, tfic, tlic,
              mic, amin, amax)`
              corresponds to ``pin="before", change_type="lower"``.
            * `locate_item_change_end(a, axis, wrap, tfic, tlic, mic,
              amin, amax)`
              corresponds to ``pin="after", change_type="None"``.
            * `locate_item_change_end_pos(a, axis, wrap, tfic, tlic,
              mic, amin, amax)`
              corresponds to ``pin="after", change_type="higher"``.
            * `locate_item_change_end_neg(a, axis, wrap, tfic, tlic,
              mic, amin, amax)`
              corresponds to ``pin="after", change_type="lower"``.

        * Write examples/test cases for ``mic=True``.

    Parameters
    ----------
    a : array_like
        Array for which to get the positions where its elements change.
    axis : int, optional
        The axis along which to search for changing elements.  By
        default, the search is performed along the last axis.
    pin : {"after", "before", "both"}
        Whether to return the last position ``"before"`` an item change
        or the first position ``"after"`` an item change.  If set to
        ``"both"``, two output arrays will be returned, one for
        ``"before"`` and one for ``"after"``.
    change_type : {None, "higher", "lower", "both"} or float or \
iterable of floats, optional
        Whether to locate all item changes without discriminating
        between different change types (``None``) or whether to locate
        only item changes to higher values (``"higher"``) or to lower
        values (``"lower"``).  If set to ``"both"``, two output arrays
        will be returned, one for ``"higher"`` and one for ``"lower"``.
        If `pin` is set to ``"both"``, too, a 2x2 tuple will be
        returned.  The first index addresses `pin`, the second index
        addresses `change_type`.  If ``change_type`` is a float, locate
        only item changes where the difference between the final and
        first item of the change is equal to the given float value
        within a certain tolerance.  The tolerance is set by `rtol` and
        `atol`.  If ``change_type`` is an iterable of floats, one output
        array for each given float will be returned.
    rtol, atol : float, optional
        The relative and absolute tolerance for comparing floating point
        numbers.  If `change_type` is a float, locate only those item
        changes that satisfy
        ``absolute((final_value - initial_value) - change_type) <= \
(atol + rtol * absolute(change_type))`` (see :func:`numpy.isclose`).
        Has no effect if `change_type` is not a float.
    wrap : bool, optional
        If ``True``, the array `a` is assumed to be continued after the
        last element along `axis` by `a` itself, like when using
        periodic boundary conditions.  Consequently, if the first and
        the last element of `a` do not match, this is interpreted as
        item change.  In this case, the last element of `a` is
        considered to be the last position before an item change and the
        first element of `a` is considered to be the first position
        after an item change.
    tfic : bool, optional
        Treat First Item as Change.  If ``True``, treat the first item
        as the first position after an item change.  Has no effect if
        `pin` is set to ``"before"``.  Must not be used together with
        `change_type`, `wrap` or `mic`.
    tlic : bool, optional
        Treat Last Item as Change.  If ``True``, treat the last item as
        the last position before an item change.  Has no effect if `pin`
        is set to ``"after"``.  Must not be used together with
        `change_type`, `wrap` or `mic`.
    mic : bool, optional
        If ``True``, respect the Minimum Image Convention when
        calculating the difference between the elements of `a`.  Has no
        effect if `change_type` is ``None``.
    amin, amax : scalar or array_like, optional
        The lower and upper bound(s) for the minimum image convention.
        See :func:`mdtools.numpy_helper_functions.diff_mic`.  Has no
        effect if `mic` is ``False`` or `change_type` is ``None``.

    Returns
    -------
    item_change_before : numpy.ndarray
        Boolean array of the same shape as `a`.  Elements that evaluate
        to ``True`` indicate the last positions before an item change in
        `a`.  Can be used to index `a` and get its values right before
        the item change.  Is not returned if `pin` is ``"after"``.  If
        `change_type` is ``"both"``, `item_change_before` will be a
        tuple containing two boolean arrays of the same shape as `a`.
        The first array indicates the last positions before item changes
        to higher values, the second array indicates the last positions
        before item changes to lower values.
    item_change_after : numpy.ndarray
        Boolean array of the same shape as `a`.  Elements that evaluate
        to ``True`` indicate the first positions after an item change in
        `a`.  Can be used to index `a` and get its values right after
        the item change.  Is not returned if `pin` is ``"before"``.  If
        `change_type` is ``"both"``, `item_change_after` will be a tuple
        containing two boolean arrays of the same shape as `a`.  The
        first array indicates the first positions after item changes to
        higher values, the second array indicates the first positions
        after item changes to lower values.

    See Also
    --------
    :func:`mdtools.numpy_helper_functions.item_change_ix` :
        Get the indices of item changes in an array
    :func:`mdtools.dtrj.locate_trans` :
        Same function but specifically for discrete trajectories

    Notes
    -----
    To get the indices of the item changes, apply :func:`numpy.nonzero`
    to the output array(s) of this function.  If you are only interested
    in the indices but not in the boolean arrays returned by this
    function, you can use
    :func:`mdtools.numpy_helper_functions.item_change_ix` instead.

    If `mic` is ``True``, this function uses
    :func:`mdtools.numpy_helper_functions.diff_mic` to calculate the
    difference between consecutive array elements.  Otherwise,
    :func:`numpy.diff` is used.

    Examples
    --------
    >>> a = np.array([1, 2, 2, 3, 3, 3])
    >>> before, after = mdt.nph.locate_item_change(a, pin="both")
    >>> before
    array([ True, False,  True, False, False, False])
    >>> after
    array([False,  True, False,  True, False, False])
    >>> before_type, after_type = mdt.nph.locate_item_change(
    ...     a, pin="both", change_type="both"
    ... )
    >>> before_type[0]  # Changes to higher values
    array([ True, False,  True, False, False, False])
    >>> before_type[1]  # Changes to lower values
    array([False, False, False, False, False, False])
    >>> np.array_equal(np.sum(before_type, axis=0), before)
    True
    >>> after_type[0]  # Changes to higher values
    array([False,  True, False,  True, False, False])
    >>> after_type[1]  # Changes to lower values
    array([False, False, False, False, False, False])
    >>> np.array_equal(np.sum(after_type, axis=0), after)
    True
    >>> mdt.nph.locate_item_change(a, pin="before", change_type=1)
    array([ True, False,  True, False, False, False])
    >>> before_type_float, after_type_float = \
mdt.nph.locate_item_change(
    ...     a, pin="both", change_type=(1, -1)
    ... )
    >>> before_type_float[0]  # Changes where final-initial=1
    array([ True, False,  True, False, False, False])
    >>> before_type_float[1]  # Changes where final-initial=-1
    array([False, False, False, False, False, False])
    >>> after_type_float[0]  # Changes where final-initial=1
    array([False,  True, False,  True, False, False])
    >>> after_type_float[1]  # Changes where final-initial=-1
    array([False, False, False, False, False, False])
    >>> before_wrap, after_wrap = mdt.nph.locate_item_change(
    ...     a, pin="both", wrap=True
    ... )
    >>> before_wrap
    array([ True, False,  True, False, False,  True])
    >>> after_wrap
    array([ True,  True, False,  True, False, False])
    >>> before_type_wrap, after_type_wrap = mdt.nph.locate_item_change(
    ...     a, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap[0]
    array([ True, False,  True, False, False, False])
    >>> before_type_wrap[1]
    array([False, False, False, False, False,  True])
    >>> np.array_equal(np.sum(before_type_wrap, axis=0), before_wrap)
    True
    >>> after_type_wrap[0]
    array([False,  True, False,  True, False, False])
    >>> after_type_wrap[1]
    array([ True, False, False, False, False, False])
    >>> np.array_equal(np.sum(after_type_wrap, axis=0), after_wrap)
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.locate_item_change(
    ...     a, pin="both", change_type=(1, -1), wrap=True
    ... )
    >>> before_type_float_wrap[0]
    array([ True, False,  True, False, False, False])
    >>> before_type_float_wrap[1]
    array([False, False, False, False, False, False])
    >>> after_type_float_wrap[0]
    array([False,  True, False,  True, False, False])
    >>> after_type_float_wrap[1]
    array([False, False, False, False, False, False])
    >>> before_tfic, after_tfic = mdt.nph.locate_item_change(
    ...     a, pin="both", tfic=True
    ... )
    >>> before_tfic
    array([ True, False,  True, False, False, False])
    >>> after_tfic
    array([ True,  True, False,  True, False, False])
    >>> before_tlic, after_tlic = mdt.nph.locate_item_change(
    ...     a, pin="both", tlic=True
    ... )
    >>> before_tlic
    array([ True, False,  True, False, False,  True])
    >>> after_tlic
    array([False,  True, False,  True, False, False])
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.locate_item_change(
    ...     a, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    array([ True, False,  True, False, False,  True])
    >>> after_tfic_tlic
    array([ True,  True, False,  True, False, False])

    >>> a = np.array([1, 2, 2, 3, 3, 3, 1])
    >>> before, after = mdt.nph.locate_item_change(a, pin="both")
    >>> before
    array([ True, False,  True, False, False,  True, False])
    >>> after
    array([False,  True, False,  True, False, False,  True])
    >>> before_type, after_type = mdt.nph.locate_item_change(
    ...     a, pin="both", change_type="both"
    ... )
    >>> before_type[0]
    array([ True, False,  True, False, False, False, False])
    >>> before_type[1]
    array([False, False, False, False, False,  True, False])
    >>> np.array_equal(np.sum(before_type, axis=0), before)
    True
    >>> after_type[0]
    array([False,  True, False,  True, False, False, False])
    >>> after_type[1]
    array([False, False, False, False, False, False,  True])
    >>> np.array_equal(np.sum(after_type, axis=0), after)
    True
    >>> before_type_float, after_type_float = \
mdt.nph.locate_item_change(
    ...     a, pin="both", change_type=(1, -1)
    ... )
    >>> before_type_float[0]
    array([ True, False,  True, False, False, False, False])
    >>> before_type_float[1]
    array([False, False, False, False, False, False, False])
    >>> after_type_float[0]
    array([False,  True, False,  True, False, False, False])
    >>> after_type_float[1]
    array([False, False, False, False, False, False, False])
    >>> before_wrap, after_wrap = mdt.nph.locate_item_change(
    ...     a, pin="both", wrap=True
    ... )
    >>> before_wrap
    array([ True, False,  True, False, False,  True, False])
    >>> after_wrap
    array([False,  True, False,  True, False, False,  True])
    >>> before_type_wrap, after_type_wrap = mdt.nph.locate_item_change(
    ...     a, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap[0]
    array([ True, False,  True, False, False, False, False])
    >>> before_type_wrap[1]
    array([False, False, False, False, False,  True, False])
    >>> np.array_equal(np.sum(before_type_wrap, axis=0), before_wrap)
    True
    >>> after_type_wrap[0]
    array([False,  True, False,  True, False, False, False])
    >>> after_type_wrap[1]
    array([False, False, False, False, False, False,  True])
    >>> np.array_equal(np.sum(after_type_wrap, axis=0), after_wrap)
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.locate_item_change(
    ...     a, pin="both", change_type=(1, -1), wrap=True
    ... )
    >>> before_type_float_wrap[0]
    array([ True, False,  True, False, False, False, False])
    >>> before_type_float_wrap[1]
    array([False, False, False, False, False, False, False])
    >>> after_type_float_wrap[0]
    array([False,  True, False,  True, False, False, False])
    >>> after_type_float_wrap[1]
    array([False, False, False, False, False, False, False])
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.locate_item_change(
    ...     a, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    array([ True, False,  True, False, False,  True,  True])
    >>> after_tfic_tlic
    array([ True,  True, False,  True, False, False,  True])

    2-dimensional example:

    >>> a = np.array([[1, 3, 3, 3],
    ...               [3, 1, 3, 3],
    ...               [3, 3, 1, 3]])
    >>> ax = 0
    >>> before, after = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both"
    ... )
    >>> before
    array([[ True,  True, False, False],
           [False,  True,  True, False],
           [False, False, False, False]])
    >>> after
    array([[False, False, False, False],
           [ True,  True, False, False],
           [False,  True,  True, False]])
    >>> before_type, after_type = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type[0]
    array([[ True, False, False, False],
           [False,  True, False, False],
           [False, False, False, False]])
    >>> before_type[1]
    array([[False,  True, False, False],
           [False, False,  True, False],
           [False, False, False, False]])
    >>> np.array_equal(np.sum(before_type, axis=0), before)
    True
    >>> after_type[0]
    array([[False, False, False, False],
           [ True, False, False, False],
           [False,  True, False, False]])
    >>> after_type[1]
    array([[False, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> np.array_equal(np.sum(after_type, axis=0), after)
    True
    >>> before_type_float, after_type_float = \
mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type=(2, -2)
    ... )
    >>> before_type_float[0]
    array([[ True, False, False, False],
           [False,  True, False, False],
           [False, False, False, False]])
    >>> before_type_float[1]
    array([[False,  True, False, False],
           [False, False,  True, False],
           [False, False, False, False]])
    >>> after_type_float[0]
    array([[False, False, False, False],
           [ True, False, False, False],
           [False,  True, False, False]])
    >>> after_type_float[1]
    array([[False, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> before_wrap, after_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    array([[ True,  True, False, False],
           [False,  True,  True, False],
           [ True, False,  True, False]])
    >>> after_wrap
    array([[ True, False,  True, False],
           [ True,  True, False, False],
           [False,  True,  True, False]])
    >>> before_type_wrap, after_type_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap[0]
    array([[ True, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> before_type_wrap[1]
    array([[False,  True, False, False],
           [False, False,  True, False],
           [ True, False, False, False]])
    >>> np.array_equal(np.sum(before_type_wrap, axis=0), before_wrap)
    True
    >>> after_type_wrap[0]
    array([[False, False,  True, False],
           [ True, False, False, False],
           [False,  True, False, False]])
    >>> after_type_wrap[1]
    array([[ True, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> np.array_equal(np.sum(after_type_wrap, axis=0), after_wrap)
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type=(2, -2), wrap=True
    ... )
    >>> before_type_float_wrap[0]
    array([[ True, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> before_type_float_wrap[1]
    array([[False,  True, False, False],
           [False, False,  True, False],
           [ True, False, False, False]])
    >>> after_type_float_wrap[0]
    array([[False, False,  True, False],
           [ True, False, False, False],
           [False,  True, False, False]])
    >>> after_type_float_wrap[1]
    array([[ True, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    array([[ True,  True, False, False],
           [False,  True,  True, False],
           [ True,  True,  True,  True]])
    >>> after_tfic_tlic
    array([[ True,  True,  True,  True],
           [ True,  True, False, False],
           [False,  True,  True, False]])

    >>> ax = 1
    >>> before, after = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both"
    ... )
    >>> before
    array([[ True, False, False, False],
           [ True,  True, False, False],
           [False,  True,  True, False]])
    >>> after
    array([[False,  True, False, False],
           [False,  True,  True, False],
           [False, False,  True,  True]])
    >>> before_type, after_type = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type[0]
    array([[ True, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> before_type[1]
    array([[False, False, False, False],
           [ True, False, False, False],
           [False,  True, False, False]])
    >>> np.array_equal(np.sum(before_type, axis=0), before)
    True
    >>> after_type[0]
    array([[False,  True, False, False],
           [False, False,  True, False],
           [False, False, False,  True]])
    >>> after_type[1]
    array([[False, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> np.array_equal(np.sum(after_type, axis=0), after)
    True
    >>> before_type_float, after_type_float = \
mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type=(2, -2)
    ... )
    >>> before_type_float[0]
    array([[ True, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> before_type_float[1]
    array([[False, False, False, False],
           [ True, False, False, False],
           [False,  True, False, False]])
    >>> after_type_float[0]
    array([[False,  True, False, False],
           [False, False,  True, False],
           [False, False, False,  True]])
    >>> after_type_float[1]
    array([[False, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> before_wrap, after_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    array([[ True, False, False,  True],
           [ True,  True, False, False],
           [False,  True,  True, False]])
    >>> after_wrap
    array([[ True,  True, False, False],
           [False,  True,  True, False],
           [False, False,  True,  True]])
    >>> before_type_wrap, after_type_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap[0]
    array([[ True, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> before_type_wrap[1]
    array([[False, False, False,  True],
           [ True, False, False, False],
           [False,  True, False, False]])
    >>> np.array_equal(np.sum(before_type_wrap, axis=0), before_wrap)
    True
    >>> after_type_wrap[0]
    array([[False,  True, False, False],
           [False, False,  True, False],
           [False, False, False,  True]])
    >>> after_type_wrap[1]
    array([[ True, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> np.array_equal(np.sum(after_type_wrap, axis=0), after_wrap)
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type=(2, -2), wrap=True
    ... )
    >>> before_type_float_wrap[0]
    array([[ True, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> before_type_float_wrap[1]
    array([[False, False, False,  True],
           [ True, False, False, False],
           [False,  True, False, False]])
    >>> after_type_float_wrap[0]
    array([[False,  True, False, False],
           [False, False,  True, False],
           [False, False, False,  True]])
    >>> after_type_float_wrap[1]
    array([[ True, False, False, False],
           [False,  True, False, False],
           [False, False,  True, False]])
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    array([[ True, False, False,  True],
           [ True,  True, False,  True],
           [False,  True,  True,  True]])
    >>> after_tfic_tlic
    array([[ True,  True, False, False],
           [ True,  True,  True, False],
           [ True, False,  True,  True]])

    3-dimensional example:

    >>> a = np.array([[[1, 2, 2],
    ...                [2, 2, 1]],
    ...
    ...               [[2, 2, 1],
    ...                [1, 2, 2]]])
    >>> ax = 0
    >>> before, after = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both"
    ... )
    >>> before
    array([[[ True, False,  True],
            [ True, False,  True]],
    <BLANKLINE>
           [[False, False, False],
            [False, False, False]]])
    >>> after
    array([[[False, False, False],
            [False, False, False]],
    <BLANKLINE>
           [[ True, False,  True],
            [ True, False,  True]]])
    >>> before_type, after_type = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type[0]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False, False],
            [False, False, False]]])
    >>> before_type[1]
    array([[[False, False,  True],
            [ True, False, False]],
    <BLANKLINE>
           [[False, False, False],
            [False, False, False]]])
    >>> np.array_equal(np.sum(before_type, axis=0), before)
    True
    >>> after_type[0]
    array([[[False, False, False],
            [False, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False, False,  True]]])
    >>> after_type[1]
    array([[[False, False, False],
            [False, False, False]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> np.array_equal(np.sum(after_type, axis=0), after)
    True
    >>> before_type_float, after_type_float = \
mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type=(1, -1)
    ... )
    >>> before_type_float[0]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False, False],
            [False, False, False]]])
    >>> before_type_float[1]
    array([[[False, False,  True],
            [ True, False, False]],
    <BLANKLINE>
           [[False, False, False],
            [False, False, False]]])
    >>> after_type_float[0]
    array([[[False, False, False],
            [False, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False, False,  True]]])
    >>> after_type_float[1]
    array([[[False, False, False],
            [False, False, False]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> before_wrap, after_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    array([[[ True, False,  True],
            [ True, False,  True]],
    <BLANKLINE>
           [[ True, False,  True],
            [ True, False,  True]]])
    >>> after_wrap
    array([[[ True, False,  True],
            [ True, False,  True]],
    <BLANKLINE>
           [[ True, False,  True],
            [ True, False,  True]]])
    >>> before_type_wrap, after_type_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap[0]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> before_type_wrap[1]
    array([[[False, False,  True],
            [ True, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False, False,  True]]])
    >>> np.array_equal(np.sum(before_type_wrap, axis=0), before_wrap)
    True
    >>> after_type_wrap[0]
    array([[[False, False,  True],
            [ True, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False, False,  True]]])
    >>> after_type_wrap[1]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> np.array_equal(np.sum(after_type_wrap, axis=0), after_wrap)
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type=(1, -1), wrap=True
    ... )
    >>> before_type_float_wrap[0]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> before_type_float_wrap[1]
    array([[[False, False,  True],
            [ True, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False, False,  True]]])
    >>> after_type_float_wrap[0]
    array([[[False, False,  True],
            [ True, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False, False,  True]]])
    >>> after_type_float_wrap[1]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    array([[[ True, False,  True],
            [ True, False,  True]],
    <BLANKLINE>
           [[ True,  True,  True],
            [ True,  True,  True]]])
    >>> after_tfic_tlic
    array([[[ True,  True,  True],
            [ True,  True,  True]],
    <BLANKLINE>
           [[ True, False,  True],
            [ True, False,  True]]])

    >>> ax = 1
    >>> before, after = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both"
    ... )
    >>> before
    array([[[ True, False,  True],
            [False, False, False]],
    <BLANKLINE>
           [[ True, False,  True],
            [False, False, False]]])
    >>> after
    array([[[False, False, False],
            [ True, False,  True]],
    <BLANKLINE>
           [[False, False, False],
            [ True, False,  True]]])
    >>> before_type, after_type = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type[0]
    array([[[ True, False, False],
            [False, False, False]],
    <BLANKLINE>
           [[False, False,  True],
            [False, False, False]]])
    >>> before_type[1]
    array([[[False, False,  True],
            [False, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False, False, False]]])
    >>> np.array_equal(np.sum(before_type, axis=0), before)
    True
    >>> after_type[0]
    array([[[False, False, False],
            [ True, False, False]],
    <BLANKLINE>
           [[False, False, False],
            [False, False,  True]]])
    >>> after_type[1]
    array([[[False, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False, False],
            [ True, False, False]]])
    >>> np.array_equal(np.sum(after_type, axis=0), after)
    True
    >>> before_type_float, after_type_float = \
mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type=(1, -1)
    ... )
    >>> before_type_float[0]
    array([[[ True, False, False],
            [False, False, False]],
    <BLANKLINE>
           [[False, False,  True],
            [False, False, False]]])
    >>> before_type_float[1]
    array([[[False, False,  True],
            [False, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False, False, False]]])
    >>> after_type_float[0]
    array([[[False, False, False],
            [ True, False, False]],
    <BLANKLINE>
           [[False, False, False],
            [False, False,  True]]])
    >>> after_type_float[1]
    array([[[False, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False, False],
            [ True, False, False]]])
    >>> before_wrap, after_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    array([[[ True, False,  True],
            [ True, False,  True]],
    <BLANKLINE>
           [[ True, False,  True],
            [ True, False,  True]]])
    >>> after_wrap
    array([[[ True, False,  True],
            [ True, False,  True]],
    <BLANKLINE>
           [[ True, False,  True],
            [ True, False,  True]]])
    >>> before_type_wrap, after_type_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap[0]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> before_type_wrap[1]
    array([[[False, False,  True],
            [ True, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False, False,  True]]])
    >>> np.array_equal(np.sum(before_type_wrap, axis=0), before_wrap)
    True
    >>> after_type_wrap[0]
    array([[[False, False,  True],
            [ True, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False, False,  True]]])
    >>> after_type_wrap[1]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> np.array_equal(np.sum(after_type_wrap, axis=0), after_wrap)
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type=(1, -1), wrap=True
    ... )
    >>> before_type_float_wrap[0]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> before_type_float_wrap[1]
    array([[[False, False,  True],
            [ True, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False, False,  True]]])
    >>> after_type_float_wrap[0]
    array([[[False, False,  True],
            [ True, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False, False,  True]]])
    >>> after_type_float_wrap[1]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    array([[[ True, False,  True],
            [ True,  True,  True]],
    <BLANKLINE>
           [[ True, False,  True],
            [ True,  True,  True]]])
    >>> after_tfic_tlic
    array([[[ True,  True,  True],
            [ True, False,  True]],
    <BLANKLINE>
           [[ True,  True,  True],
            [ True, False,  True]]])

    >>> ax = 2
    >>> before, after = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both"
    ... )
    >>> before
    array([[[ True, False, False],
            [False,  True, False]],
    <BLANKLINE>
           [[False,  True, False],
            [ True, False, False]]])
    >>> after
    array([[[False,  True, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [False,  True, False]]])
    >>> before_type, after_type = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type[0]
    array([[[ True, False, False],
            [False, False, False]],
    <BLANKLINE>
           [[False, False, False],
            [ True, False, False]]])
    >>> before_type[1]
    array([[[False, False, False],
            [False,  True, False]],
    <BLANKLINE>
           [[False,  True, False],
            [False, False, False]]])
    >>> np.array_equal(np.sum(before_type, axis=0), before)
    True
    >>> after_type[0]
    array([[[False,  True, False],
            [False, False, False]],
    <BLANKLINE>
           [[False, False, False],
            [False,  True, False]]])
    >>> after_type[1]
    array([[[False, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [False, False, False]]])
    >>> np.array_equal(np.sum(after_type, axis=0), after)
    True
    >>> before_type_float, after_type_float = \
mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type=(1, -1)
    ... )
    >>> before_type_float[0]
    array([[[ True, False, False],
            [False, False, False]],
    <BLANKLINE>
           [[False, False, False],
            [ True, False, False]]])
    >>> before_type_float[1]
    array([[[False, False, False],
            [False,  True, False]],
    <BLANKLINE>
           [[False,  True, False],
            [False, False, False]]])
    >>> np.array_equal(np.sum(before_type, axis=0), before)
    True
    >>> after_type_float[0]
    array([[[False,  True, False],
            [False, False, False]],
    <BLANKLINE>
           [[False, False, False],
            [False,  True, False]]])
    >>> after_type_float[1]
    array([[[False, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [False, False, False]]])
    >>> before_wrap, after_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    array([[[ True, False,  True],
            [False,  True,  True]],
    <BLANKLINE>
           [[False,  True,  True],
            [ True, False,  True]]])
    >>> after_wrap
    array([[[ True,  True, False],
            [ True, False,  True]],
    <BLANKLINE>
           [[ True, False,  True],
            [ True,  True, False]]])
    >>> before_type_wrap, after_type_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap[0]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> before_type_wrap[1]
    array([[[False, False,  True],
            [False,  True, False]],
    <BLANKLINE>
           [[False,  True, False],
            [False, False,  True]]])
    >>> np.array_equal(np.sum(before_type_wrap, axis=0), before_wrap)
    True
    >>> after_type_wrap[0]
    array([[[False,  True, False],
            [ True, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False,  True, False]]])
    >>> after_type_wrap[1]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> np.array_equal(np.sum(after_type_wrap, axis=0), after_wrap)
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type=(1, -1), wrap=True
    ... )
    >>> before_type_float_wrap[0]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> before_type_float_wrap[1]
    array([[[False, False,  True],
            [False,  True, False]],
    <BLANKLINE>
           [[False,  True, False],
            [False, False,  True]]])
    >>> np.array_equal(np.sum(before_type_wrap, axis=0), before_wrap)
    True
    >>> after_type_float_wrap[0]
    array([[[False,  True, False],
            [ True, False, False]],
    <BLANKLINE>
           [[ True, False, False],
            [False,  True, False]]])
    >>> after_type_float_wrap[1]
    array([[[ True, False, False],
            [False, False,  True]],
    <BLANKLINE>
           [[False, False,  True],
            [ True, False, False]]])
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    array([[[ True, False,  True],
            [False,  True,  True]],
    <BLANKLINE>
           [[False,  True,  True],
            [ True, False,  True]]])
    >>> after_tfic_tlic
    array([[[ True,  True, False],
            [ True, False,  True]],
    <BLANKLINE>
           [[ True, False,  True],
            [ True,  True, False]]])

    Edge cases:

    >>> a = np.array([[1, 2, 2]])
    >>> ax = 0
    >>> before_type, after_type = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type
    (array([[False, False, False]]), array([[False, False, False]]))
    >>> after_type
    (array([[False, False, False]]), array([[False, False, False]]))
    >>> before_type_wrap, after_type_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap
    (array([[False, False, False]]), array([[False, False, False]]))
    >>> after_type_wrap
    (array([[False, False, False]]), array([[False, False, False]]))
    >>> mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    (array([[ True,  True,  True]]), array([[ True,  True,  True]]))
    >>> ax = 1
    >>> before_type, after_type = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type
    (array([[ True, False, False]]), array([[False, False, False]]))
    >>> after_type
    (array([[False,  True, False]]), array([[False, False, False]]))
    >>> before_type_wrap, after_type_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap
    (array([[ True, False, False]]), array([[False, False,  True]]))
    >>> after_type_wrap
    (array([[False,  True, False]]), array([[ True, False, False]]))
    >>> mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    (array([[ True, False,  True]]), array([[ True,  True, False]]))

    >>> a = np.array([]).reshape(2,0)
    >>> ax = 0
    >>> before, after = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both"
    ... )
    >>> before
    array([], shape=(2, 0), dtype=bool)
    >>> after
    array([], shape=(2, 0), dtype=bool)
    >>> before_type, after_type = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type
    (array([], shape=(2, 0), dtype=bool), \
array([], shape=(2, 0), dtype=bool))
    >>> after_type
    (array([], shape=(2, 0), dtype=bool), \
array([], shape=(2, 0), dtype=bool))
    >>> before_wrap, after_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    array([], shape=(2, 0), dtype=bool)
    >>> after_wrap
    array([], shape=(2, 0), dtype=bool)
    >>> before_type_wrap, after_type_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap
    (array([], shape=(2, 0), dtype=bool), \
array([], shape=(2, 0), dtype=bool))
    >>> after_type_wrap
    (array([], shape=(2, 0), dtype=bool), \
array([], shape=(2, 0), dtype=bool))
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    array([], shape=(2, 0), dtype=bool)
    >>> after_tfic_tlic
    array([], shape=(2, 0), dtype=bool)
    >>> ax = 1
    >>> before, after = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both"
    ... )
    >>> before
    array([], shape=(2, 0), dtype=bool)
    >>> after
    array([], shape=(2, 0), dtype=bool)
    >>> before_type, after_type = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type
    (array([], shape=(2, 0), dtype=bool), \
array([], shape=(2, 0), dtype=bool))
    >>> after_type
    (array([], shape=(2, 0), dtype=bool), \
array([], shape=(2, 0), dtype=bool))
    >>> before_wrap, after_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    array([], shape=(2, 0), dtype=bool)
    >>> after_wrap
    array([], shape=(2, 0), dtype=bool)
    >>> before_type_wrap, after_type_wrap = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap
    (array([], shape=(2, 0), dtype=bool), \
array([], shape=(2, 0), dtype=bool))
    >>> after_type_wrap
    (array([], shape=(2, 0), dtype=bool), \
array([], shape=(2, 0), dtype=bool))
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.locate_item_change(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    array([], shape=(2, 0), dtype=bool)
    >>> after_tfic_tlic
    array([], shape=(2, 0), dtype=bool)

    >>> a = np.array(0)
    >>> mdt.nph.locate_item_change(a, pin="both")
    Traceback (most recent call last):
    ...
    ValueError: The input array must be at least one dimensional

    >>> mdt.nph.locate_item_change(a, pin="both", wrap=True, tfic=True)
    Traceback (most recent call last):
    ...
    ValueError: 'wrap' must not be used together with 'tfic' of 'tlic'
    """
    a = np.asarray(a)
    if a.ndim == 0:
        raise ValueError("The input array must be at least 1-dimensional")
    if tfic or tlic:
        if change_type is not None:
            raise ValueError(
                "'tfic' and 'tlic' must not be used together with"
                " 'change_type'"
            )
        if wrap:
            raise ValueError(
                "'tfic' and 'tlic' must not be used together with 'wrap'"
            )
        if mic:
            raise ValueError(
                "'tfic' and 'tlic' must not be used together with 'mic'"
            )

    if mic:
        # Boundaries for the minimum image convention
        # (Don't remove, also needed further below)
        amin = np.min(a) if amin is None else amin
        amax = np.max(a) if amax is None else amax
        # Item differences according to the minimum image convention
        item_diffs = mdt.nph.diff_mic(a, amin=amin, amax=amax, axis=axis)
    else:
        if change_type is not None and np.issubdtype(
            a.dtype, np.unsignedinteger
        ):
            # np.diff keeps the dtype of the input array => If the dtype
            # of the input array is an unsigned integer type, negative
            # differences are not possible.
            a = a.astype(np.int64, casting="safe")
        item_diffs = np.diff(a, axis=axis)
    if change_type is None:  # All changes
        operators = (np.not_equal,)
    elif change_type == "higher":  # Changes to higher values
        operators = (np.greater,)
    elif change_type == "lower":  # Changes to lower values
        operators = (np.less,)
    elif change_type == "both":  # Changes to higher and to lower values
        operators = (np.greater, np.less)
    elif isinstance(change_type, str):
        raise ValueError(
            "'change_type' must be either None, 'higher', 'lower', 'both', a"
            " float or an iterable of floats, but you gave"
            " '{}'".format(change_type)
        )
    else:
        try:
            operators = tuple(
                lambda x, _, ct=ct: np.isclose(x, ct, rtol, atol)
                for ct in change_type
            )
            if len(operators) == 0:
                raise IndexError(
                    "'change_type' ({}) must not be an empty"
                    " iterable".format(change_type)
                )
        except TypeError:  # change_type is not iterable
            operators = (lambda x, _: np.isclose(x, change_type, rtol, atol),)
    items_changed = tuple(op(item_diffs, 0) for op in operators)

    # Only index `a.shape` with `axis` after np.diff(a, axis) to get a
    # proper numpy.AxisError if `axis` is out of bounds (instead of an
    # IndexError)
    if a.shape[axis] == 0:
        item_change = np.zeros_like(a, dtype=bool)
        if len(items_changed) > 1:
            item_change = tuple(item_change for itm_chngd in items_changed)
        if pin == "both":  # equivalent to `pin` = before *and* after
            return item_change, item_change
        else:  # `pin` = before *or* after
            return item_change

    # Construct an insertion array which will be inserted after or
    # before `items_changed` to bring `items_changed` to the same shape
    # as `a` and make `items_changed` a mask for items of `a` which are
    # right before or right after an item change.
    shape = list(item_diffs.shape)
    shape[axis] = 1
    shape = tuple(shape)
    if wrap:
        if mic:
            item_diffs = mdt.nph.subtract_mic(
                a.take(0, axis=axis),
                a.take(-1, axis=axis),
                amin=amin,
                amax=amax,
            )
        else:
            item_diffs = a.take(0, axis=axis) - a.take(-1, axis=axis)
        insertion_before = tuple(
            op(item_diffs, 0).reshape(shape) for op in operators
        )
        insertion_after = insertion_before
    else:
        if tfic:
            insertion_after = tuple(
                np.ones(shape, dtype=bool) for op in operators
            )
        else:
            insertion_after = tuple(
                np.zeros(shape, dtype=bool) for op in operators
            )
        if tlic:
            insertion_before = tuple(
                np.ones(shape, dtype=bool) for op in operators
            )
        else:
            insertion_before = tuple(
                np.zeros(shape, dtype=bool) for op in operators
            )
    del item_diffs, operators

    if pin in ("before", "both"):
        # Items of `a` right before an item change
        item_change_before = [
            np.zeros(a.shape, dtype=bool) for itm_chngd in items_changed
        ]
        for i, itm_chngd in enumerate(items_changed):
            item_change_before[i] = np.concatenate(
                (itm_chngd, insertion_before[i]), axis=axis
            )
        if len(item_change_before) > 1:
            item_change_before = tuple(item_change_before)
        else:
            item_change_before = item_change_before[0]
        if pin == "before":
            return item_change_before
    if pin in ("after", "both"):
        # Items of `a` right after an item change
        item_change_after = [
            np.zeros(a.shape, dtype=bool) for itm_chngd in items_changed
        ]
        for i, itm_chngd in enumerate(items_changed):
            item_change_after[i] = np.concatenate(
                (insertion_after[i], itm_chngd), axis=axis
            )
        if len(item_change_after) > 1:
            item_change_after = tuple(item_change_after)
        else:
            item_change_after = item_change_after[0]
        if pin == "after":
            return item_change_after
    else:
        return ValueError(
            "'pin' must be either 'after', 'before' or 'both' but you gave"
            " {}".format(pin)
        )
    return item_change_before, item_change_after


def item_change_ix(a, axis=-1, *args, **kwargs):
    """
    Get the indices of item changes in an array.

    Parameters
    ----------
    a : array_like
        Array for which to get all indices where its elements change.
    axis : int, optional
        The axis along which to search for changing elements.  By
        default, the search is performed along the last axis.
    kwargs : tuple, optional
        Additional keyword arguments to parse to
        :func:`mdtools.numpy_helper_functions.locate_item_change`.  See
        there for possible choices.  Note that by default `pin` is set
        to ``"after"`` and `change_type` is set to ``None``.

    Returns
    -------
    ix_before : tuple of numpy.ndarray
        Tuple of arrays, one for each dimension of `a`, containing the
        indices where the elements of `a` will change.  Can be used to
        index `a` and get its values right before the item change.  Is
        not returned if `pin` is ``"after"``.
    ix_after : tuple of numpy.ndarray
        Tuple of arrays, one for each dimension of `a`, containing the
        indices where the elements of `a` have changed.  Can be used to
        index `a` and get its values right after the item change.  Is
        not returned if `pin` is ``"before"``.

    See Also
    --------
    :func:`mdtools.numpy_helper_functions.locate_item_change` :
        Locate the positions of item changes in an array
    :func:`mdtools.dtrj.trans_ix` :
        Same function but specifically for discrete trajectories

    Notes
    -----
    This function only applies :func:`numpy.nonzero` to the output of
    :func:`mdtools.numpy_helper_functions.locate_item_change`.

    Examples
    --------
    >>> a = np.array([1, 2, 2, 3, 3, 3])
    >>> before, after = mdt.nph.item_change_ix(a, pin="both")
    >>> before
    (array([0, 2]),)
    >>> after
    (array([1, 3]),)
    >>> before_type, after_type = mdt.nph.item_change_ix(
    ...     a, pin="both", change_type="both"
    ... )
    >>> before_type[0]  # Changes to higher values
    (array([0, 2]),)
    >>> before_type[1]  # Changes to lower values
    (array([], dtype=int64),)
    >>> for i, bt in enumerate(zip(*before_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(bt)), np.sort(before[i])
    ...     )
    True
    >>> after_type[0]  # Changes to higher values
    (array([1, 3]),)
    >>> after_type[1]  # Changes to lower values
    (array([], dtype=int64),)
    >>> for i, at in enumerate(zip(*after_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(at)), np.sort(after[i])
    ...     )
    True
    >>> mdt.nph.item_change_ix(a, pin="before", change_type=1)
    (array([0, 2]),)
    >>> before_type_float, after_type_float = mdt.nph.item_change_ix(
    ...     a, pin="both", change_type=(1, -1)
    ... )
    >>> before_type_float[0]  # Changes where final-initial=1
    (array([0, 2]),)
    >>> before_type_float[1]  # Changes where final-initial=-1
    (array([], dtype=int64),)
    >>> after_type_float[0]  # Changes where final-initial=1
    (array([1, 3]),)
    >>> after_type_float[1]  # Changes where final-initial=-1
    (array([], dtype=int64),)
    >>> before_wrap, after_wrap = mdt.nph.item_change_ix(
    ...     a, pin="both", wrap=True
    ... )
    >>> before_wrap
    (array([0, 2, 5]),)
    >>> after_wrap
    (array([0, 1, 3]),)
    >>> before_type_wrap, after_type_wrap = mdt.nph.item_change_ix(
    ...     a, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap[0]
    (array([0, 2]),)
    >>> before_type_wrap[1]
    (array([5]),)
    >>> for i, btw in enumerate(zip(*before_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(btw)), np.sort(before_wrap[i])
    ...     )
    True
    >>> after_type_wrap[0]
    (array([1, 3]),)
    >>> after_type_wrap[1]
    (array([0]),)
    >>> for i, atw in enumerate(zip(*after_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(atw)), np.sort(after_wrap[i])
    ...     )
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.item_change_ix(
    ...     a, pin="both", change_type=(1, -1), wrap=True
    ... )
    >>> before_type_float_wrap[0]
    (array([0, 2]),)
    >>> before_type_float_wrap[1]
    (array([], dtype=int64),)
    >>> after_type_float_wrap[0]
    (array([1, 3]),)
    >>> after_type_float_wrap[1]
    (array([], dtype=int64),)
    >>> before_tfic, after_tfic = mdt.nph.item_change_ix(
    ...     a, pin="both", tfic=True
    ... )
    >>> before_tfic
    (array([0, 2]),)
    >>> after_tfic
    (array([0, 1, 3]),)
    >>> before_tlic, after_tlic = mdt.nph.item_change_ix(
    ...     a, pin="both", tlic=True
    ... )
    >>> before_tlic
    (array([0, 2, 5]),)
    >>> after_tlic
    (array([1, 3]),)
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.item_change_ix(
    ...     a, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    (array([0, 2, 5]),)
    >>> after_tfic_tlic
    (array([0, 1, 3]),)

    >>> a = np.array([1, 2, 2, 3, 3, 3, 1])
    >>> before, after = mdt.nph.item_change_ix(a, pin="both")
    >>> before
    (array([0, 2, 5]),)
    >>> after
    (array([1, 3, 6]),)
    >>> before_type, after_type = mdt.nph.item_change_ix(
    ...     a, pin="both", change_type="both"
    ... )
    >>> before_type[0]
    (array([0, 2]),)
    >>> before_type[1]
    (array([5]),)
    >>> for i, bt in enumerate(zip(*before_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(bt)), np.sort(before[i])
    ...     )
    True
    >>> after_type[0]
    (array([1, 3]),)
    >>> after_type[1]
    (array([6]),)
    >>> for i, at in enumerate(zip(*after_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(at)), np.sort(after[i])
    ...     )
    True
    >>> before_type_float, after_type_float = mdt.nph.item_change_ix(
    ...     a, pin="both", change_type=(1, -1)
    ... )
    >>> before_type_float[0]
    (array([0, 2]),)
    >>> before_type_float[1]
    (array([], dtype=int64),)
    >>> after_type_float[0]
    (array([1, 3]),)
    >>> after_type_float[1]
    (array([], dtype=int64),)
    >>> before_wrap, after_wrap = mdt.nph.item_change_ix(
    ...     a, pin="both", wrap=True
    ... )
    >>> before_wrap
    (array([0, 2, 5]),)
    >>> after_wrap
    (array([1, 3, 6]),)
    >>> before_type_wrap, after_type_wrap = mdt.nph.item_change_ix(
    ...     a, pin="both", change_type="both"
    ... )
    >>> before_type_wrap[0]
    (array([0, 2]),)
    >>> before_type_wrap[1]
    (array([5]),)
    >>> for i, btw in enumerate(zip(*before_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(btw)), np.sort(before_wrap[i])
    ...     )
    True
    >>> after_type_wrap[0]
    (array([1, 3]),)
    >>> after_type_wrap[1]
    (array([6]),)
    >>> for i, atw in enumerate(zip(*after_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(atw)), np.sort(after_wrap[i])
    ...     )
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.item_change_ix(
    ...     a, pin="both", change_type=(1, -1)
    ... )
    >>> before_type_float_wrap[0]
    (array([0, 2]),)
    >>> before_type_float_wrap[1]
    (array([], dtype=int64),)
    >>> after_type_float_wrap[0]
    (array([1, 3]),)
    >>> after_type_float_wrap[1]
    (array([], dtype=int64),)
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.item_change_ix(
    ...     a, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    (array([0, 2, 5, 6]),)
    >>> after_tfic_tlic
    (array([0, 1, 3, 6]),)

    2-dimensional example:

    >>> a = np.array([[1, 3, 3, 3],
    ...               [3, 1, 3, 3],
    ...               [3, 3, 1, 3]])
    >>> ax = 0
    >>> before, after = mdt.nph.item_change_ix(a, axis=ax, pin="both")
    >>> before
    (array([0, 0, 1, 1]), array([0, 1, 1, 2]))
    >>> after
    (array([1, 1, 2, 2]), array([0, 1, 1, 2]))
    >>> before_type, after_type = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type[0]
    (array([0, 1]), array([0, 1]))
    >>> before_type[1]
    (array([0, 1]), array([1, 2]))
    >>> for i, bt in enumerate(zip(*before_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(bt)), np.sort(before[i])
    ...     )
    True
    True
    >>> after_type[0]
    (array([1, 2]), array([0, 1]))
    >>> after_type[1]
    (array([1, 2]), array([1, 2]))
    >>> for i, at in enumerate(zip(*after_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(at)), np.sort(after[i])
    ...     )
    True
    True
    >>> before_type_float, after_type_float = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type=(2, -2)
    ... )
    >>> before_type_float[0]
    (array([0, 1]), array([0, 1]))
    >>> before_type_float[1]
    (array([0, 1]), array([1, 2]))
    >>> after_type_float[0]
    (array([1, 2]), array([0, 1]))
    >>> after_type_float[1]
    (array([1, 2]), array([1, 2]))
    >>> before_wrap, after_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    (array([0, 0, 1, 1, 2, 2]), array([0, 1, 1, 2, 0, 2]))
    >>> after_wrap
    (array([0, 0, 1, 1, 2, 2]), array([0, 2, 0, 1, 1, 2]))
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_type_wrap, after_type_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap[0]
    (array([0, 1, 2]), array([0, 1, 2]))
    >>> before_type_wrap[1]
    (array([0, 1, 2]), array([1, 2, 0]))
    >>> for i, btw in enumerate(zip(*before_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(btw)), np.sort(before_wrap[i])
    ...     )
    True
    True
    >>> after_type_wrap[0]
    (array([0, 1, 2]), array([2, 0, 1]))
    >>> after_type_wrap[1]
    (array([0, 1, 2]), array([0, 1, 2]))
    >>> for i, atw in enumerate(zip(*after_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(atw)), np.sort(after_wrap[i])
    ...     )
    True
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type=(2, -2), wrap=True
    ... )
    >>> before_type_float_wrap[0]
    (array([0, 1, 2]), array([0, 1, 2]))
    >>> before_type_float_wrap[1]
    (array([0, 1, 2]), array([1, 2, 0]))
    >>> after_type_float_wrap[0]
    (array([0, 1, 2]), array([2, 0, 1]))
    >>> after_type_float_wrap[1]
    (array([0, 1, 2]), array([0, 1, 2]))
    >>> before_tfic_tlic
    (array([0, 0, 1, 1, 2, 2, 2, 2]), array([0, 1, 1, 2, 0, 1, 2, 3]))
    >>> after_tfic_tlic
    (array([0, 0, 0, 0, 1, 1, 2, 2]), array([0, 1, 2, 3, 0, 1, 1, 2]))

    >>> ax = 1
    >>> before, after = mdt.nph.item_change_ix(a, axis=ax, pin="both")
    >>> before
    (array([0, 1, 1, 2, 2]), array([0, 0, 1, 1, 2]))
    >>> after
    (array([0, 1, 1, 2, 2]), array([1, 1, 2, 2, 3]))
    >>> before_type, after_type = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type[0]
    (array([0, 1, 2]), array([0, 1, 2]))
    >>> before_type[1]
    (array([1, 2]), array([0, 1]))
    >>> for i, bt in enumerate(zip(*before_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(bt)), np.sort(before[i])
    ...     )
    True
    True
    >>> after_type[0]
    (array([0, 1, 2]), array([1, 2, 3]))
    >>> after_type[1]
    (array([1, 2]), array([1, 2]))
    >>> for i, at in enumerate(zip(*after_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(at)), np.sort(after[i])
    ...     )
    True
    True
    >>> before_type_float, after_type_float = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type=(2, -2)
    ... )
    >>> before_type_float[0]
    (array([0, 1, 2]), array([0, 1, 2]))
    >>> before_type_float[1]
    (array([1, 2]), array([0, 1]))
    >>> after_type_float[0]
    (array([0, 1, 2]), array([1, 2, 3]))
    >>> after_type_float[1]
    (array([1, 2]), array([1, 2]))
    >>> before_wrap, after_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    (array([0, 0, 1, 1, 2, 2]), array([0, 3, 0, 1, 1, 2]))
    >>> after_wrap
    (array([0, 0, 1, 1, 2, 2]), array([0, 1, 1, 2, 2, 3]))
    >>> before_type_wrap, after_type_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap[0]
    (array([0, 1, 2]), array([0, 1, 2]))
    >>> before_type_wrap[1]
    (array([0, 1, 2]), array([3, 0, 1]))
    >>> for i, btw in enumerate(zip(*before_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(btw)), np.sort(before_wrap[i])
    ...     )
    True
    True
    >>> after_type_wrap[0]
    (array([0, 1, 2]), array([1, 2, 3]))
    >>> after_type_wrap[1]
    (array([0, 1, 2]), array([0, 1, 2]))
    >>> for i, atw in enumerate(zip(*after_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(atw)), np.sort(after_wrap[i])
    ...     )
    True
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type=(2, -2), wrap=True
    ... )
    >>> before_type_float_wrap[0]
    (array([0, 1, 2]), array([0, 1, 2]))
    >>> before_type_float_wrap[1]
    (array([0, 1, 2]), array([3, 0, 1]))
    >>> after_type_float_wrap[0]
    (array([0, 1, 2]), array([1, 2, 3]))
    >>> after_type_float_wrap[1]
    (array([0, 1, 2]), array([0, 1, 2]))
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    (array([0, 0, 1, 1, 1, 2, 2, 2]), array([0, 3, 0, 1, 3, 1, 2, 3]))
    >>> after_tfic_tlic
    (array([0, 0, 1, 1, 1, 2, 2, 2]), array([0, 1, 0, 1, 2, 0, 2, 3]))

    3-dimensional example:

    >>> a = np.array([[[1, 2, 2],
    ...                [2, 2, 1]],
    ...
    ...               [[2, 2, 1],
    ...                [1, 2, 2]]])
    >>> ax = 0
    >>> before, after = mdt.nph.item_change_ix(a, axis=ax, pin="both")
    >>> before
    (array([0, 0, 0, 0]), array([0, 0, 1, 1]), array([0, 2, 0, 2]))
    >>> after
    (array([1, 1, 1, 1]), array([0, 0, 1, 1]), array([0, 2, 0, 2]))
    >>> before_type, after_type = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type[0]
    (array([0, 0]), array([0, 1]), array([0, 2]))
    >>> before_type[1]
    (array([0, 0]), array([0, 1]), array([2, 0]))
    >>> for i, bt in enumerate(zip(*before_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(bt)), np.sort(before[i])
    ...     )
    True
    True
    True
    >>> after_type[0]
    (array([1, 1]), array([0, 1]), array([0, 2]))
    >>> after_type[1]
    (array([1, 1]), array([0, 1]), array([2, 0]))
    >>> for i, at in enumerate(zip(*after_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(at)), np.sort(after[i])
    ...     )
    True
    True
    True
    >>> before_type_float, after_type_float = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type=(1, -1)
    ... )
    >>> before_type_float[0]
    (array([0, 0]), array([0, 1]), array([0, 2]))
    >>> before_type_float[1]
    (array([0, 0]), array([0, 1]), array([2, 0]))
    >>> after_type_float[0]
    (array([1, 1]), array([0, 1]), array([0, 2]))
    >>> after_type_float[1]
    (array([1, 1]), array([0, 1]), array([2, 0]))
    >>> before_wrap, after_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 2, 0, 2, 0, 2, 0, 2]))
    >>> after_wrap
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 2, 0, 2, 0, 2, 0, 2]))
    >>> before_type_wrap, after_type_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap[0]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([0, 2, 2, 0]))
    >>> before_type_wrap[1]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([2, 0, 0, 2]))
    >>> for i, btw in enumerate(zip(*before_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(btw)), np.sort(before_wrap[i])
    ...     )
    True
    True
    True
    >>> after_type_wrap[0]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([2, 0, 0, 2]))
    >>> after_type_wrap[1]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([0, 2, 2, 0]))
    >>> for i, atw in enumerate(zip(*after_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(atw)), np.sort(after_wrap[i])
    ...     )
    True
    True
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type=(1, -1), wrap=True
    ... )
    >>> before_type_float_wrap[0]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([0, 2, 2, 0]))
    >>> before_type_float_wrap[1]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([2, 0, 0, 2]))
    >>> after_type_float_wrap[0]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([2, 0, 0, 2]))
    >>> after_type_float_wrap[1]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([0, 2, 2, 0]))
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    (array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1]), \
array([0, 2, 0, 2, 0, 1, 2, 0, 1, 2]))
    >>> after_tfic_tlic
    (array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1]), \
array([0, 1, 2, 0, 1, 2, 0, 2, 0, 2]))

    >>> ax = 1
    >>> before, after = mdt.nph.item_change_ix(a, axis=ax, pin="both")
    >>> before
    (array([0, 0, 1, 1]), array([0, 0, 0, 0]), array([0, 2, 0, 2]))
    >>> after
    (array([0, 0, 1, 1]), array([1, 1, 1, 1]), array([0, 2, 0, 2]))
    >>> before_type, after_type = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type[0]
    (array([0, 1]), array([0, 0]), array([0, 2]))
    >>> before_type[1]
    (array([0, 1]), array([0, 0]), array([2, 0]))
    >>> for i, bt in enumerate(zip(*before_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(bt)), np.sort(before[i])
    ...     )
    True
    True
    True
    >>> after_type[0]
    (array([0, 1]), array([1, 1]), array([0, 2]))
    >>> after_type[1]
    (array([0, 1]), array([1, 1]), array([2, 0]))
    >>> for i, at in enumerate(zip(*after_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(at)), np.sort(after[i])
    ...     )
    True
    True
    True
    >>> before_type_float, after_type_float = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type=(1, -1)
    ... )
    >>> before_type_float[0]
    (array([0, 1]), array([0, 0]), array([0, 2]))
    >>> before_type_float[1]
    (array([0, 1]), array([0, 0]), array([2, 0]))
    >>> after_type_float[0]
    (array([0, 1]), array([1, 1]), array([0, 2]))
    >>> after_type_float[1]
    (array([0, 1]), array([1, 1]), array([2, 0]))
    >>> before_wrap, after_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 2, 0, 2, 0, 2, 0, 2]))
    >>> after_wrap
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 2, 0, 2, 0, 2, 0, 2]))
    >>> before_type_wrap, after_type_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap[0]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([0, 2, 2, 0]))
    >>> before_type_wrap[1]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([2, 0, 0, 2]))
    >>> for i, btw in enumerate(zip(*before_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(btw)), np.sort(before_wrap[i])
    ...     )
    True
    True
    True
    >>> after_type_wrap[0]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([2, 0, 0, 2]))
    >>> after_type_wrap[1]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([0, 2, 2, 0]))
    >>> for i, atw in enumerate(zip(*after_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(atw)), np.sort(after_wrap[i])
    ...     )
    True
    True
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type=(1, -1), wrap=True
    ... )
    >>> before_type_float_wrap[0]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([0, 2, 2, 0]))
    >>> before_type_float_wrap[1]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([2, 0, 0, 2]))
    >>> after_type_float_wrap[0]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([2, 0, 0, 2]))
    >>> after_type_float_wrap[1]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([0, 2, 2, 0]))
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    (array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1]), \
array([0, 2, 0, 1, 2, 0, 2, 0, 1, 2]))
    >>> after_tfic_tlic
    (array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), \
array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1]), \
array([0, 1, 2, 0, 2, 0, 1, 2, 0, 2]))

    >>> ax = 2
    >>> before, after = mdt.nph.item_change_ix(a, axis=ax, pin="both")
    >>> before
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([0, 1, 1, 0]))
    >>> after
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([1, 2, 2, 1]))
    >>> before_type, after_type = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type[0]
    (array([0, 1]), array([0, 1]), array([0, 0]))
    >>> before_type[1]
    (array([0, 1]), array([1, 0]), array([1, 1]))
    >>> for i, bt in enumerate(zip(*before_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(bt)), np.sort(before[i])
    ...     )
    True
    True
    True
    >>> after_type[0]
    (array([0, 1]), array([0, 1]), array([1, 1]))
    >>> after_type[1]
    (array([0, 1]), array([1, 0]), array([2, 2]))
    >>> for i, at in enumerate(zip(*after_type)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(at)), np.sort(after[i])
    ...     )
    True
    True
    True
    >>> before_type_float, after_type_float = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type=(1, -1)
    ... )
    >>> before_type_float[0]
    (array([0, 1]), array([0, 1]), array([0, 0]))
    >>> before_type_float[1]
    (array([0, 1]), array([1, 0]), array([1, 1]))
    >>> after_type_float[0]
    (array([0, 1]), array([0, 1]), array([1, 1]))
    >>> after_type_float[1]
    (array([0, 1]), array([1, 0]), array([2, 2]))
    >>> before_wrap, after_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 2, 1, 2, 1, 2, 0, 2]))
    >>> after_wrap
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 1, 0, 2, 0, 2, 0, 1]))
    >>> before_type_wrap, after_type_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap[0]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([0, 2, 2, 0]))
    >>> before_type_wrap[1]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([2, 1, 1, 2]))
    >>> for i, btw in enumerate(zip(*before_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(btw)), np.sort(before_wrap[i])
    ...     )
    True
    True
    True
    >>> after_type_wrap[0]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([1, 0, 0, 1]))
    >>> after_type_wrap[1]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([0, 2, 2, 0]))
    >>> for i, atw in enumerate(zip(*after_type_wrap)):
    ...     np.array_equal(
    ...         np.sort(np.concatenate(atw)), np.sort(after_wrap[i])
    ...     )
    True
    True
    True
    >>> before_type_float_wrap, after_type_float_wrap = \
mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type=(1, -1), wrap=True
    ... )
    >>> before_type_float_wrap[0]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([0, 2, 2, 0]))
    >>> before_type_float_wrap[1]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([2, 1, 1, 2]))
    >>> after_type_float_wrap[0]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([1, 0, 0, 1]))
    >>> after_type_float_wrap[1]
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([0, 2, 2, 0]))
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 2, 1, 2, 1, 2, 0, 2]))
    >>> after_tfic_tlic
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 1, 0, 2, 0, 2, 0, 1]))

    Edge cases:

    >>> a = np.array([[1, 2, 2]])
    >>> ax = 0
    >>> before_type, after_type = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type
    ((array([], dtype=int64), array([], dtype=int64)), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> after_type
    ((array([], dtype=int64), array([], dtype=int64)), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> before_type_wrap, after_type_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap
    ((array([], dtype=int64), array([], dtype=int64)), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> after_type_wrap
    ((array([], dtype=int64), array([], dtype=int64)), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    (array([0, 0, 0]), array([0, 1, 2]))
    >>> after_tfic_tlic
    (array([0, 0, 0]), array([0, 1, 2]))
    >>> ax = 1
    >>> before_type, after_type = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type
    ((array([0]), array([0])), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> after_type
    ((array([0]), array([1])), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> before_type_wrap, after_type_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap
    ((array([0]), array([0])), (array([0]), array([2])))
    >>> after_type_wrap
    ((array([0]), array([1])), (array([0]), array([0])))
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    (array([0, 0]), array([0, 2]))
    >>> after_tfic_tlic
    (array([0, 0]), array([0, 1]))

    >>> a = np.array([]).reshape(2,0)
    >>> ax = 0
    >>> before, after = mdt.nph.item_change_ix(a, axis=ax, pin="both")
    >>> before
    (array([], dtype=int64), array([], dtype=int64))
    >>> after
    (array([], dtype=int64), array([], dtype=int64))
    >>> before_type, after_type = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type
    ((array([], dtype=int64), array([], dtype=int64)), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> after_type
    ((array([], dtype=int64), array([], dtype=int64)), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> before_wrap, after_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    (array([], dtype=int64), array([], dtype=int64))
    >>> after_wrap
    (array([], dtype=int64), array([], dtype=int64))
    >>> before_type_wrap, after_type = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap
    ((array([], dtype=int64), array([], dtype=int64)), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> after_type
    ((array([], dtype=int64), array([], dtype=int64)), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    (array([], dtype=int64), array([], dtype=int64))
    >>> after_tfic_tlic
    (array([], dtype=int64), array([], dtype=int64))
    >>> ax = 1
    >>> before, after = mdt.nph.item_change_ix(a, axis=ax, pin="both")
    >>> before
    (array([], dtype=int64), array([], dtype=int64))
    >>> after
    (array([], dtype=int64), array([], dtype=int64))
    >>> before_type, after_type = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both"
    ... )
    >>> before_type
    ((array([], dtype=int64), array([], dtype=int64)), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> after_type
    ((array([], dtype=int64), array([], dtype=int64)), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> before_wrap, after_wrap = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", wrap=True
    ... )
    >>> before_wrap
    (array([], dtype=int64), array([], dtype=int64))
    >>> after_wrap
    (array([], dtype=int64), array([], dtype=int64))
    >>> before_type_wrap, after_type = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", change_type="both", wrap=True
    ... )
    >>> before_type_wrap
    ((array([], dtype=int64), array([], dtype=int64)), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> after_type
    ((array([], dtype=int64), array([], dtype=int64)), \
(array([], dtype=int64), array([], dtype=int64)))
    >>> before_tfic_tlic, after_tfic_tlic = mdt.nph.item_change_ix(
    ...     a, axis=ax, pin="both", tfic=True, tlic=True
    ... )
    >>> before_tfic_tlic
    (array([], dtype=int64), array([], dtype=int64))
    >>> after_tfic_tlic
    (array([], dtype=int64), array([], dtype=int64))

    >>> a = np.array(0)
    >>> mdt.nph.item_change_ix(a, pin="both")
    Traceback (most recent call last):
    ...
    ValueError: The input array must be at least one dimensional

    >>> mdt.nph.item_change_ix(a, pin="both", wrap=True, tfic=True)
    Traceback (most recent call last):
    ...
    ValueError: 'wrap' must not be used together with 'tfic' of 'tlic'

    Examples from
    :func:`mdtools.numpy_helper_functions.ix_of_item_change` mapped to
    :func:`mdtools.numpy_helper_functions.item_change_ix`:

    >>> a = np.arange(3)
    >>> a
    array([0, 1, 2])
    >>> mdt.nph.item_change_ix(a, pin="after")
    (array([1, 2]),)
    >>> mdt.nph.item_change_ix(a, pin="after", wrap=True)
    (array([0, 1, 2]),)
    >>> ix = mdt.nph.item_change_ix(a, pin="after", tfic=True)
    >>> ix
    (array([0, 1, 2]),)
    >>> a[ix]
    array([0, 1, 2])

    >>> b = np.array([1, 2, 2, 3, 3, 3, 1])
    >>> mdt.nph.item_change_ix(b, pin="after")
    (array([1, 3, 6]),)
    >>> mdt.nph.item_change_ix(b, pin="after", wrap=True)
    (array([1, 3, 6]),)
    >>> ix = mdt.nph.item_change_ix(b, pin="after", tfic=True)
    >>> ix
    (array([0, 1, 3, 6]),)
    >>> b[ix]
    array([1, 2, 3, 1])

    >>> c = np.ones(3)
    >>> ix = mdt.nph.item_change_ix(c, pin="after")
    >>> ix
    (array([], dtype=int64),)
    >>> c[ix]
    array([], dtype=float64)
    >>> mdt.nph.item_change_ix(c, pin="after", wrap=True)
    (array([], dtype=int64),)
    >>> mdt.nph.item_change_ix(c, pin="after", tfic=True)
    (array([0]),)

    2-dimensional example:

    >>> d = np.array([[1, 3, 3, 3],
    ...               [3, 1, 3, 3],
    ...               [3, 3, 1, 3]])
    >>> ax = 0
    >>> mdt.nph.item_change_ix(d, axis=ax, pin="after")
    (array([1, 1, 2, 2]), array([0, 1, 1, 2]))
    >>> mdt.nph.item_change_ix(d, axis=ax, pin="after", wrap=True)
    (array([0, 0, 1, 1, 2, 2]), array([0, 2, 0, 1, 1, 2]))
    >>> ix = mdt.nph.item_change_ix(d, axis=ax, pin="after", tfic=True)
    >>> ix
    (array([0, 0, 0, 0, 1, 1, 2, 2]), array([0, 1, 2, 3, 0, 1, 1, 2]))
    >>> d[ix]
    array([1, 3, 3, 3, 3, 1, 3, 1])

    >>> ax = 1
    >>> mdt.nph.item_change_ix(d, axis=ax, pin="after")
    (array([0, 1, 1, 2, 2]), array([1, 1, 2, 2, 3]))
    >>> mdt.nph.item_change_ix(d, axis=ax, pin="after", wrap=True)
    (array([0, 0, 1, 1, 2, 2]), array([0, 1, 1, 2, 2, 3]))
    >>> mdt.nph.item_change_ix(d, axis=ax, pin="after", tfic=True)
    (array([0, 0, 1, 1, 1, 2, 2, 2]), array([0, 1, 0, 1, 2, 0, 2, 3]))

    3-dimensional example:

    >>> e = np.array([[[1, 2, 2],
    ...                [2, 2, 1]],
    ...
    ...               [[2, 2, 1],
    ...                [1, 2, 2]]])
    >>> ax = 0
    >>> mdt.nph.item_change_ix(e, axis=ax, pin="after")
    (array([1, 1, 1, 1]), array([0, 0, 1, 1]), array([0, 2, 0, 2]))
    >>> mdt.nph.item_change_ix(e, axis=ax, pin="after", wrap=True)
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 2, 0, 2, 0, 2, 0, 2]))
    >>> ix = mdt.nph.item_change_ix(e, axis=ax, pin="after", tfic=True)
    >>> ix
    (array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1]), \
array([0, 1, 2, 0, 1, 2, 0, 2, 0, 2]))
    >>> e[ix]
    array([1, 2, 2, 2, 2, 1, 2, 1, 1, 2])

    >>> ax = 1
    >>> mdt.nph.item_change_ix(e, axis=ax, pin="after")
    (array([0, 0, 1, 1]), array([1, 1, 1, 1]), array([0, 2, 0, 2]))
    >>> mdt.nph.item_change_ix(e, axis=ax, pin="after", wrap=True)
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 2, 0, 2, 0, 2, 0, 2]))
    >>> mdt.nph.item_change_ix(e, axis=ax, pin="after", tfic=True)
    (array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), \
array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1]), \
array([0, 1, 2, 0, 2, 0, 1, 2, 0, 2]))

    >>> ax = 2
    >>> mdt.nph.item_change_ix(e, axis=ax, pin="after")
    (array([0, 0, 1, 1]), array([0, 1, 0, 1]), array([1, 2, 2, 1]))
    >>> mdt.nph.item_change_ix(e, axis=ax, pin="after", wrap=True)
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 1, 0, 2, 0, 2, 0, 1]))
    >>> mdt.nph.item_change_ix(e, axis=ax, pin="after", tfic=True)
    (array([0, 0, 0, 0, 1, 1, 1, 1]), \
array([0, 0, 1, 1, 0, 0, 1, 1]), \
array([0, 1, 0, 2, 0, 2, 0, 1]))

    Edge cases:

    >>> x = np.array([1])
    >>> mdt.nph.item_change_ix(x, pin="after")
    (array([], dtype=int64),)
    >>> mdt.nph.item_change_ix(x, pin="after", wrap=True)
    (array([], dtype=int64),)
    >>> mdt.nph.item_change_ix(x, pin="after", tfic=True)
    (array([0]),)

    >>> x = np.array([])
    >>> mdt.nph.item_change_ix(x, pin="after")
    (array([], dtype=int64),)
    >>> mdt.nph.item_change_ix(x, pin="after", wrap=True)
    (array([], dtype=int64),)
    >>> ix = mdt.nph.item_change_ix(x, pin="after", tfic=True)
    >>> ix
    (array([], dtype=int64),)
    >>> x[ix]
    array([], dtype=float64)

    >>> x = np.array(0)
    >>> mdt.nph.item_change_ix(x, pin="after")
    Traceback (most recent call last):
    ...
    ValueError: The dimension of a must be greater than zero
    """
    pin = kwargs.setdefault("pin", "after")
    change_type = kwargs.setdefault("change_type", None)
    try:
        (i for i in change_type)
        ct_is_iterable = True
    except TypeError:  # change_type is not iterable
        ct_is_iterable = False
    item_changes = mdt.nph.locate_item_change(a, axis=axis, **kwargs)
    if pin == "both" and (change_type == "both" or ct_is_iterable):
        return tuple(
            tuple(np.nonzero(ic) for ic in ics) for ics in item_changes
        )
    elif pin == "both" or change_type == "both" or ct_is_iterable:
        return tuple(np.nonzero(ic) for ic in item_changes)
    else:  # pin != "both" and change_type != "both" and not iterable
        return np.nonzero(item_changes)


def argmin_last(a, axis=None, out=None):
    """
    Get the indices of the minimum values along an axis.

    Contrarily to :func:`numpy.argmin`, this function returns the
    indices of the last occurrence of the minimum value.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : numpy.ndarray, optional
        If provided, the result will be inserted into this array.  It
        should be of the appropriate shape and dtype.

    Returns
    -------
    index_array : numpy.ndarray of ints
        Array of indices into the array.  It has the same shape as
        ``a.shape`` with the dimension along `axis` removed.

    See Also
    --------
    :func:`numpy.argmin` :
        Returns indices of the first occurrence of the minium value
    :func:`argmax_last` :
        Same as :func:`argmin_last` but for maximum values
    :func:`numpy.unravel_index` :
        Convert the returned index `ix` to a tuple of index arrays
        suitable to index a multidimensional input array `a` if `axis`
        was ``None``
    :func:`mdtools.numpy_helper_functions.ix_along_axis_to_global_ix` :
        Same as :func:`numpy.unravel_index`, but to be used when `axis`
        was not ``None``.

    Notes
    -----
    In case of multiple occurrences of the minimum values, the indices
    corresponding to the last occurrence are returned.  This is the
    opposite of what :func:`numpy.argmin` does.

    Examples
    --------
    >>> a = -np.arange(4)
    >>> a[1] = -3
    >>> a
    array([ 0, -3, -2, -3])
    >>> mdt.nph.argmin_last(a)
    3
    >>> np.argmin(a)
    1

    2-dimensional case:

    >>> a = -np.eye(3, 4, 0, dtype=int) - np.eye(3, 4, 2, dtype=int)
    >>> a
    array([[-1,  0, -1,  0],
           [ 0, -1,  0, -1],
           [ 0,  0, -1,  0]])
    >>> mdt.nph.argmin_last(a)
    10
    >>> mdt.nph.argmin_last(a, axis=0)
    array([0, 1, 2, 1])
    >>> mdt.nph.argmin_last(a, axis=1)
    array([2, 3, 2])

    3-dimensional case:

    >>> a = -np.array([[[1, 1, 0],
    ...                 [0, 1, 1]],
    ...
    ...                [[1, 0, 1],
    ...                 [1, 1, 0]]])
    >>> mdt.nph.argmin_last(a)
    10
    >>> mdt.nph.argmin_last(a, axis=0)
    array([[1, 0, 1],
           [1, 1, 0]])
    >>> mdt.nph.argmin_last(a, axis=1)
    array([[0, 1, 1],
           [1, 1, 0]])
    >>> mdt.nph.argmin_last(a, axis=2)
    array([[1, 2],
           [2, 1]])
    """
    b = np.flip(a, axis=axis)
    ix = np.argmin(b, axis=axis, out=out)
    ix *= -1
    if axis is None:
        ix += b.size - 1
    else:
        ix += b.shape[axis] - 1
    return ix


def argmax_last(a, axis=None, out=None):
    """
    Get the indices of the maximum values along an axis.

    Contrarily to :func:`numpy.argmax`, this function returns the
    indices of the last occurrence of the maximum value.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : numpy.ndarray, optional
        If provided, the result will be inserted into this array.  It
        should be of the appropriate shape and dtype.

    Returns
    -------
    index_array : numpy.ndarray of ints
        Array of indices into the array.  It has the same shape as
        ``a.shape`` with the dimension along `axis` removed.

    See Also
    --------
    :func:`numpy.argmax` :
        Returns indices of the first occurrence of the maximum value
    :func:`argmin_last` :
        Same as :func:`argmax_last` but for minimum values
    :func:`numpy.unravel_index` :
        Convert the returned index `ix` to a tuple of index arrays
        suitable to index a multidimensional input array `a` if `axis`
        was ``None``
    :func:`mdtools.numpy_helper_functions.ix_along_axis_to_global_ix` :
        Same as :func:`numpy.unravel_index`, but to be used when `axis`
        was not ``None``.

    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the last occurrence are returned.  This is the
    opposite of what :func:`numpy.argmax` does.

    Examples
    --------
    >>> a = np.arange(4)
    >>> a[1] = 3
    >>> a
    array([0, 3, 2, 3])
    >>> mdt.nph.argmax_last(a)
    3
    >>> np.argmax(a)
    1

    2-dimensional case:

    >>> a = np.eye(3, 4, 0, dtype=int) + np.eye(3, 4, 2, dtype=int)
    >>> a
    array([[1, 0, 1, 0],
           [0, 1, 0, 1],
           [0, 0, 1, 0]])
    >>> mdt.nph.argmax_last(a)
    10
    >>> mdt.nph.argmax_last(a, axis=0)
    array([0, 1, 2, 1])
    >>> mdt.nph.argmax_last(a, axis=1)
    array([2, 3, 2])

    3-dimensional case:

    >>> a = np.array([[[1, 1, 0],
    ...                [0, 1, 1]],
    ...
    ...               [[1, 0, 1],
    ...                [1, 1, 0]]])
    >>> mdt.nph.argmax_last(a)
    10
    >>> mdt.nph.argmax_last(a, axis=0)
    array([[1, 0, 1],
           [1, 1, 0]])
    >>> mdt.nph.argmax_last(a, axis=1)
    array([[0, 1, 1],
           [1, 1, 0]])
    >>> mdt.nph.argmax_last(a, axis=2)
    array([[1, 2],
           [2, 1]])
    """
    b = np.flip(a, axis=axis)
    ix = np.argmax(b, axis=axis, out=out)
    ix *= -1
    if axis is None:
        ix += b.size - 1
    else:
        ix += b.shape[axis] - 1
    return ix


def ceil_divide(x1, x2, **kwargs):
    """
    Calculate element-wise ceil division.

    Return the smallest integer greater or equal to the division of the
    inputs.  It is equivalent to ``-(-x1 // x2)`` and can bee regarded
    as the opposite of :func:`numpy.floor_divide`.

    Parameters
    ----------
    x1 : array_like
        Numerator.
    x2 : array_like
        Denominator.  If ``x1.shape != x2.shape``, they must be
        broadcastable to a common shape (which becomes the shape of the
        output).
    kwargs : dict, optional
        Additional keyword arguments to parse to :func:`numpy.multiply`
        and :func:`numpy.floor_divide`.  See there for possible choices.

    Returns
    -------
    y : numpy.ndarray
        The element-wise ceil division of `x1` and `x2`, formally
        ``y = -(-x1 // x2)``.  This is a scalar if both `x1` and `x2`
        are scalars.

    See Also
    --------
    :func:`numpy.floor_divide` : Element-wise floor division
    :func:`numpy.ceil` : Element-wise ceiling.

    Examples
    --------
    >>> mdt.nph.ceil_divide(3, 2)
    2
    >>> np.floor_divide(3, 2)
    1
    >>> mdt.nph.ceil_divide(3.5, 2)
    2.0
    >>> np.floor_divide(3.5, 2)
    1.0

    >>> mdt.nph.ceil_divide(np.arange(3, 6), np.arange(1, 4))
    array([3, 2, 2])
    >>> np.floor_divide(np.arange(3, 6), np.arange(1, 4))
    array([3, 2, 1])

    >>> result = np.zeros(3, dtype=int)
    >>> mask = np.array([True, True, False])
    >>> mdt.nph.ceil_divide(
    ...     [5, 5, 5],
    ...     [2, 3, 3],
    ...     out=result,
    ...     where=mask
    ... )
    array([3, 2, 0])
    """
    if kwargs:
        y = np.multiply(x1, -1, **kwargs)
        kwargs.setdefault("out", y)
        y = np.floor_divide(y, x2, **kwargs)
        y = np.multiply(y, -1, **kwargs)
    else:
        y = -x1
        y //= x2
        y *= -1
    return y


def digitize_dd(
    sample,
    bins,
    right=False,
    expand_binnumbers=False,
    raise_if_less=False,
    raise_if_greater=False,
):
    """
    Return the indices of the multi-dimensional bins to which each value
    in the input array belongs.

    This function extends :func:`numpy.digitize` to multi-dimensional
    data.

    If values in `sample` are beyond the bounds of `bins`, ``0`` or
    ``len(bins)`` is returned as appropriate.

    Parameters
    ----------
    sample : array_like
        Data to be digitized.  Either a sequence of ``N`` arrays each of
        length ``D`` or a 2-dimensional array of shape ``(N, D)``.  Each
        of the ``N`` arrays/rows is interpreted as a set of coordinates
        in ``D``-dimensional space.  If `sample` has less than 2
        dimensions, simply ``np.digitize(sample, bins, right=right)`` is
        returned.
    bins : array_like or sequence of array_likes
        Array of bin edges to use for each of the ``D`` dimensions.
        Must be either a sequence of ``D`` monotonic arrays (one for
        each dimension) or a single monotonic array that is used for all
        dimensions.  If `sample` has less than 2 dimensions, `bins` must
        be a 1-dimensional, monotonic array.
    right : bool, optional
        Indicating whether the bin intervals include the right or the
        left bin edge.  If ``True``, the bin intervals include the right
        bin edge, i.e. the bin intervals are left-open and right-closed:
        (a, b] -> a < x <= b.  If ``False``, the bin intervals include
        the left bin edge, i.e. the bin intervals are left-closed and
        right-open: [a, b) -> a <= x < b.  See :func:`numpy.digitize`
        for more details.
    expand_binnumbers : bool, optional
        If ``True``, the returned index array is unraveled into an array
        of shape ``(D, N)`` where each row gives the bin numbers
        of the elements of `sample` along the corresponding dimension.
        If ``False``, the returned index array has shape ``(N,)`` and
        maps each element of `sample` to its corresponding linearized
        bin number (using row-major ordering).  Note that the returned
        linearized bin indices index into an array containing two extra
        bins at the outer bin edges to capture values outside of the
        defined bin bounds.  See also
        :func:`scipy.stats.binned_statistic_dd`.
    raise_if_less, raise_if_greater : bool, optional
        If ``True``, raise an :exc:`ValueError` if a value of `sample`
        lies below/above the first/last bin.  Otherwise, if values in
        `sample` lie below or above the bounds of `bins`, ``0`` or
        ``len(bins)`` is returned, respectively.

    Returns
    -------
    bin_ix : numpy.ndarray
        Array of indices.  This array assigns to each element of
        `sample` an integer that represents the bin number to which this
        element belongs.  The representation depends on the
        `expand_binnumbers` argument.

    See Also
    --------
    :func:`numpy.digitize` :
        Return the indices of the 1-dimensional bins to which each value
        of the input array belongs
    :func:`scipy.stats.binned_statistic_dd` :
        Compute a multi-dimensional binned statistic for a set of data
    :func:`numpy.unravel_index` :
        Convert an array of flat indices into a tuple of coordinate
        arrays

    Examples
    --------
    >>> mdt.nph.digitize_dd(0.5, [0, 1])
    1
    >>> mdt.nph.digitize_dd([-1, 0, 1], [0, 1])
    array([0, 1, 2])
    >>> mdt.nph.digitize_dd([-1, 0, 1], [0, 1], right=True)
    array([0, 0, 1])

    >>> a = np.array([[-1, 0.5],
    ...               [ 0, 1.5],
    ...               [ 1, 2.5]])
    >>> bins_x = np.arange(-1, 2)
    >>> bins_y = np.arange(1, 4)
    >>> mdt.nph.digitize_dd(a, [bins_x, bins_y])
    array([ 4,  9, 14])
    >>> mdt.nph.digitize_dd(a, [bins_x, bins_y], expand_binnumbers=True)
    array([[1, 2, 3],
           [0, 1, 2]])

    >>> from scipy.stats import binned_statistic_dd
    >>> a = np.random.rand(4,3) * 5
    >>> bins_x = np.arange(5)
    >>> bins_y = np.arange(-1, 5.1, 2)
    >>> bins_z = np.arange(0, 6.1, 2)
    >>> bins = (bins_x, bins_y, bins_z)
    >>> bin_ix = mdt.nph.digitize_dd(a, bins)
    >>> ret = binned_statistic_dd(a, values=a.T, bins=bins)
    >>> np.array_equal(bin_ix, ret.binnumber)
    True
    >>> bin_ix = mdt.nph.digitize_dd(a, bins, expand_binnumbers=True)
    >>> ret = binned_statistic_dd(
    ...    a, values=a.T, bins=bins, expand_binnumbers=True
    ... )
    >>> np.array_equal(bin_ix, ret.binnumber)
    True
    """
    sample = np.asarray(sample)
    if sample.ndim <= 1:
        bin_ix = np.digitize(sample, bins, right=right)
        if raise_if_less and bin_ix <= 0:
            raise ValueError("A value lies below the first bin")
        if raise_if_greater and bin_ix >= len(bins):
            raise ValueError("A value lies above the last bin")
        return bin_ix
    elif sample.ndim == 2:
        ndims = sample.shape[1]
        try:
            len(bins)
        except TypeError:
            # `bins` is not a sequence.
            # https://docs.python.org/3/glossary.html#term-sequence
            raise TypeError(
                "'bins' must be either a 1-dimensional array or a sequence of"
                " {} 1-dimensional arrays".format(ndims)
            )
        try:
            len(bins[0])
        except TypeError:
            # `bins` is a 1-dimensional sequence.  Make it to a sequence
            # of `ndims` 1-dimensional arrays.
            bins = (bins,) * ndims
        else:
            # `bins` is a sequence of sequences.
            try:
                len(bins[0][0])
            except TypeError:
                # `bins` is a sequence of 1-dimensional sequences.
                if len(bins) != ndims:
                    raise TypeError(
                        "'bins' must be either a 1-dimensional array or a"
                        " sequence of {} 1-dimensional arrays".format(ndims)
                    )
            else:
                # `bins` is a sequence of sequences of sequences.
                raise TypeError(
                    "'bins' must be either a 1-dimensional array or a sequence"
                    " of {} 1-dimensional arrays".format(ndims)
                )

        sample = sample.T
        nbins_nd, bin_ix_nd = [], []
        for i, arr in enumerate(sample):
            nbins_nd.append(len(bins[i]) + 1)
            bin_ix = np.digitize(arr, bins[i], right=right)
            bin_ix_nd.append(bin_ix)
            if raise_if_less and bin_ix <= 0:
                raise ValueError(
                    "A value in the {}-th dimension lies below the first"
                    " bin".format(i)
                )
            if raise_if_greater and bin_ix >= len(bins):
                raise ValueError(
                    "A value in the {}-th dimension lies above the last"
                    " bin".format(i)
                )
        del bin_ix
        if expand_binnumbers:
            bin_ix_nd = np.asarray(bin_ix_nd)
            if bin_ix_nd.shape != sample.shape:
                raise ValueError(
                    "'bin_ix' must have shape {} but has shape {}.  This"
                    " should not have"
                    " happened".format(sample.shape, bin_ix_nd.shape)
                )
        else:
            bin_ix_nd = np.ravel_multi_index(bin_ix_nd, nbins_nd)
            if bin_ix_nd.shape != (sample.shape[1],):
                raise ValueError(
                    "'bin_ix' must have shape {} but has shape {}.  This"
                    " should not have"
                    " happened".format((sample.shape[1],), bin_ix_nd.shape)
                )
        return bin_ix_nd
    else:
        raise ValueError(
            "The number of dimensions of 'sample' must be less than 3 but is"
            " {}".format(sample.ndim)
        )


def split_into_contig_seqs(
    a, step=1, step_tol=1e-08, sort=False, return_ix=False
):
    """
    Split an array into subarrays of contiguous sequences.

    Parameters
    ----------
    a : array_like
        1-dimensional array which to split into subarrays of contiguous
        sequences with step size `step`.
    step : scalar, optional
        The step size defining the contiguous sequence.
    step_tol : scalar, optional
        A tolerance value for the step size.  Values in `a` are
        considered to form a contiguous sequence if their difference
        lies within ``step +/- step_tol``.
    sort : bool, optional
        If ``True``, sort `a` before searching for contiguous sequences.
    return_ix : bool, optional
        If ``True``, return the indices where `a` was split to create
        the subarrays.  If `sort` is ``True``, these are the indices
        where the sorted input array was split.

    Returns
    -------
    contig_seqs : list
        List of subarrays, one for each contiguous sequence of numbers
        with step size `step` in `a`.
    ix_split : numpy.ndarray
        Array of indices at which `a` was split.  If `sort` is ``True``,
        these are the indices at which the sorted input array was split.

    See Also
    --------
    :func:`get_const_seqs` :
        Get all sequences of constant values in an array

    References
    ----------
    Adapted from https://stackoverflow.com/a/7353335.

    Examples
    --------
    >>> a = np.array([-3, -2,  0,  3,  2,  4])
    >>> contig_seqs, ix_split = mdt.nph.split_into_contig_seqs(
    ...     a, return_ix=True
    ... )
    >>> contig_seqs
    [array([-3, -2]), array([0]), array([3]), array([2]), array([4])]
    >>> ix_split
    array([2, 3, 4, 5])
    >>> contig_seqs, ix_split = mdt.nph.split_into_contig_seqs(
    ...     a, sort=True, return_ix=True
    ... )
    >>> contig_seqs
    [array([-3, -2]), array([0]), array([2, 3, 4])]
    >>> ix_split
    array([2, 3])

    >>> a = np.array([0, 1, 2, 5, 7, 9])
    >>> mdt.nph.split_into_contig_seqs(a, step=2)
    [array([0]), array([1]), array([2]), array([5, 7, 9])]
    >>> a = np.array([0, 1, 2, 2, 1, 0])
    >>> mdt.nph.split_into_contig_seqs(a, step=-1)
    [array([0]), array([1]), array([2]), array([2, 1, 0])]
    >>> a = np.array([1, 2, 2, 3, 3, 3])
    >>> mdt.nph.split_into_contig_seqs(a, step=0)
    [array([1]), array([2, 2]), array([3, 3, 3])]

    Edge cases:

    >>> a = np.array([])
    >>> mdt.nph.split_into_contig_seqs(a, sort=True, return_ix=True)
    ([array([], dtype=float64)], array([], dtype=int64))
    """
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(
            "`a` has {} dimension(s) but must be 1-dimensional".format(a.ndim)
        )

    if sort:
        a = np.sort(a)

    is_contiguous = np.isclose(np.diff(a), step, rtol=0, atol=step_tol)
    np.bitwise_not(is_contiguous, out=is_contiguous)
    ix_split = np.flatnonzero(is_contiguous)
    ix_split += 1
    contiguous_seqs = np.split(a, ix_split)
    if return_ix:
        return contiguous_seqs, ix_split
    else:
        return contiguous_seqs


def get_const_seqs(x, tol=1e-08, sort=False):
    """
    Get all sequences of constant values in an array.

    Parameters
    ----------
    x : array_like
        1-dimensional array for which to get all sequences of constant
        values.
    tol : scalar, optional
        Tolerance value within which consecutive elements of `x` are
        considered to be equal.
    sort : bool, optional
        If ``True``, sort `x` before searching for sequences of constant
        values.

    Returns
    -------
    seq_starts : numpy.ndarray
        Index array indicating the start of each constant sequence in
        `x`.
    eq_lengths : numpy.ndarray
        The length (i.e. the number of values) of each constant
        sequence.
    vals : numpy.ndarray
        The value of each constant sequence.  Because consecutive values
        can differ by `tol`, this is the mean value of each constant
        sequence.

    See Also
    --------
    :func:`split_into_contig_seqs` :
        Split an array into subarrays of contiguous sequences.
    :func:`find_const_seq_n` :
        Find the first sequence of at least `n` constant values in an
        array
    :func:`find_const_seq_long` :
        Find the longest sequence of constant values in an array

    Examples
    --------
    >>> a = np.array([0, 2, 2, 4, 4, 4, 3, 3, 1])
    >>> mdt.nph.get_const_seqs(a)
    (array([0, 1, 3, 6, 8]), array([1, 2, 3, 2, 1]), array([0., 2., 4., 3., 1.]))
    >>> mdt.nph.get_const_seqs(a, sort=True)
    (array([0, 1, 2, 4, 6]), array([1, 1, 2, 2, 3]), array([0., 1., 2., 3., 4.]))
    >>> np.sort(a)
    array([0, 1, 2, 2, 3, 3, 4, 4, 4])

    Edge cases:

    >>> a = np.array([])
    >>> mdt.nph.get_const_seqs(a, sort=True)
    (array([], dtype=int64), array([0]), array([], dtype=float64))
    """  # noqa: E501, W505
    seqs, seq_starts = mdt.nph.split_into_contig_seqs(
        x, step=0, step_tol=tol, sort=sort, return_ix=True
    )
    seq_lengths = np.array([len(seq) for seq in seqs])
    if len(seq_starts) > 0:
        seq_starts = np.insert(seq_starts, 0, 0)
        vals = np.array([np.mean(seq) for seq in seqs])
    else:
        vals = seqs[0]
    return seq_starts, seq_lengths, vals


def find_const_seq_n(x, n, tol=1e-08, sort=False):
    """
    Find the first sequence of at least `n` constant values in an array.

    Parameters
    ----------
    x : array_like
        1-dimensional array in which to find the first sequence of at
        least `n` constant values.
    tol : scalar, optional
        Tolerance value within which consecutive elements of `x` are
        considered to be equal.
    sort : bool, optional
        If ``True``, sort `x` before searching for sequences of constant
        values.

    Returns
    -------
    start_ix : int
        Index indicating the start of the first sequence of at least `n`
        constant values in `x`.
    length : int
        The actual length of this sequence.
    val : scalar
        The value of this sequence.  Because consecutive values can
        differ by `tol`, this is the mean value of the sequence.

    See Also
    --------
    :func:`get_const_seqs` :
        Get all sequences of constant values in an array
    :func:`find_const_seq_long` :
        Find the longest sequence of constant values in an array

    Notes
    -----
    If the array does not contain a sequence of at least `n` constant
    values, `(array([]), 0, array([]))`` is returned.

    Examples
    --------
    >>> a = np.array([0, 4, 4, 4, 2, 2])
    >>> mdt.nph.find_const_seq_n(a, n=2)
    (1, 3, 4.0)
    >>> mdt.nph.find_const_seq_n(a, n=2, sort=True)
    (1, 2, 2.0)
    >>> np.sort(a)
    array([0, 2, 2, 4, 4, 4])

    >>> a = np.array([0, 2, 2, 4, 4, 4, 2, 2, 0])
    >>> mdt.nph.find_const_seq_n(a, n=2)
    (1, 2, 2.0)
    >>> mdt.nph.find_const_seq_n(a, n=3)
    (3, 3, 4.0)
    >>> mdt.nph.find_const_seq_n(a, n=2, sort=True)
    (0, 2, 0.0)
    >>> np.sort(a)
    array([0, 0, 2, 2, 2, 2, 4, 4, 4])

    Edge cases:

    >>> a = np.array([1, 2, 2, 3, 3, 3])
    >>> mdt.nph.find_const_seq_n(a, n=4)
    (array([], dtype=int64), 0, array([], dtype=float64))
    >>> mdt.nph.find_const_seq_n(a, n=0)
    (0, 1, 1.0)
    >>> mdt.nph.find_const_seq_n(a, n=-1)
    (0, 1, 1.0)

    >>> a = np.array([])
    >>> mdt.nph.find_const_seq_n(a, n=1, sort=True)
    (array([], dtype=int64), 0, array([], dtype=float64))
    """
    seq_starts, seq_lengths, vals = mdt.nph.get_const_seqs(
        x, tol=tol, sort=sort
    )
    if len(seq_starts) == 0:
        return seq_starts, seq_lengths[0], vals
    else:
        ix = np.argmax(seq_lengths >= n)
    if seq_lengths[ix] < n:
        # This means the array does not contain a sequence of at least
        # `n` constant values, i.e. ``not np.any(seq_lengths >= n)``.
        return np.array([], dtype=np.int64), 0, np.array([], dtype=vals.dtype)
    else:
        return seq_starts[ix], seq_lengths[ix], vals[ix]


def find_const_seq_long(x, tol=1e-08, sort=False):
    """
    Find the longest sequence of constant values in an array.

    If there exist multiple sequences with the longest length, the first
    of them is returned.

    Parameters
    ----------
    x : array_like
        1-dimensional array in which to find the longest sequence of
        constant values.
    tol : scalar, optional
        Tolerance value within which consecutive elements of `x` are
        considered to be equal.
    sort : bool, optional
        If ``True``, sort `x` before searching for sequences of constant
        values.

    Returns
    -------
    start_ix : int
        Index indicating the start of the longest sequence of constant
        values in `x`.
    length : int
        The actual length of this sequence.
    val : scalar
        The value of this sequence.  Because consecutive values can
        differ by `tol`, this is the mean value of the sequence.

    See Also
    --------
    :func:`get_const_seqs` :
        Get all sequences of constant values in an array.
    :func:`find_const_seq_n`
        Find the first sequence of at least `n` constant values in an
        array

    Examples
    --------
    >>> a = np.array([0, 4, 4, 4, 2, 2])
    >>> mdt.nph.find_const_seq_long(a)
    (1, 3, 4.0)
    >>> mdt.nph.find_const_seq_long(a, sort=True)
    (3, 3, 4.0)
    >>> np.sort(a)
    array([0, 2, 2, 4, 4, 4])
    """
    seq_starts, seq_lengths, vals = mdt.nph.get_const_seqs(
        x, tol=tol, sort=sort
    )
    ix = np.argmax(seq_lengths)
    return seq_starts[ix], seq_lengths[ix], vals[ix]


def sequenize(a, step=1, start=0):
    """
    "Sequenize" the values of an array.

    Assign to each different value in an array `a` a constant offset in
    such a way that subtracting the resulting offset array from `a`
    yields a new array whose unique sorted values yield a contiguous
    sequence with a given step width.

    This might for example be useful if you have an array containing
    some indices and one or more intermediate indices are missing but
    you want the indices to form a contiguous sequence.

    Parameters
    ----------
    a : array_like
        The array whose values should yield a contiguous sequence with
        a given step width when computing the unique sorted values of
        the array.
    step : scalar, optional
        The step width of this sequence. The data type of `a` and `step`
        must be castable.
    start : scalar, optional
        The value at which this sequence should start. The data type of
        `a` and `start` must be castable.

    Returns
    -------
    b : numpy.ndarray
        A 'sequenized' version of `a`. When calling ``numpy.unique`` on
        `b`, the returned array will be a contiguous sequence with step
        width `step`.

    Examples
    --------
    >>> a = np.array([-5, -3, -1,  2,  4,  6])
    >>> mdt.nph.sequenize(a)
    array([0, 1, 2, 3, 4, 5])
    >>> mdt.nph.sequenize(a, step=3)
    array([ 0,  3,  6,  9, 12, 15])
    >>> mdt.nph.sequenize(a, step=-3)
    array([  0,  -3,  -6,  -9, -12, -15])
    >>> mdt.nph.sequenize(a, start=-(a.size - 1))
    array([-5, -4, -3, -2, -1,  0])

    >>> a = np.array([[-5, -3, -1],
    ...               [ 2,  4,  6]])
    >>> mdt.nph.sequenize(a)
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mdt.nph.sequenize(a, step=3)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> mdt.nph.sequenize(a, step=-3)
    array([[  0,  -3,  -6],
           [ -9, -12, -15]])
    >>> mdt.nph.sequenize(a, start=-(a.size - 1))
    array([[-5, -4, -3],
           [-2, -1,  0]])

    >>> a = np.arange(6)
    >>> np.all(mdt.nph.sequenize(a) == a)
    True
    >>> mdt.nph.sequenize(a, step=2)
    array([ 0,  2,  4,  6,  8, 10])
    >>> mdt.nph.sequenize(a, step=-2)
    array([  0,  -2,  -4,  -6,  -8, -10])
    >>> mdt.nph.sequenize(a, start=-(a.size - 1))
    array([-5, -4, -3, -2, -1,  0])

    >>> a = np.arange(6).reshape(2, 3)
    >>> np.all(mdt.nph.sequenize(a) == a)
    True
    >>> mdt.nph.sequenize(a, step=2)
    array([[ 0,  2,  4],
           [ 6,  8, 10]])
    >>> mdt.nph.sequenize(a, step=-2)
    array([[  0,  -2,  -4],
           [ -6,  -8, -10]])
    >>> mdt.nph.sequenize(a, start=-(a.size - 1))
    array([[-5, -4, -3],
           [-2, -1,  0]])
    """
    a = np.asarray(a)
    u, ix = np.unique(a, return_inverse=True)
    d = np.diff(u, prepend=start)
    d[1:] -= step
    np.cumsum(d, out=d)
    u -= d
    return u[ix].reshape(a.shape, order="A")


def group_by(
    keys, values, assume_sorted=False, return_keys=False, debug=False
):
    """
    Group the elements of `values` by their `keys`.

    Parameters
    ----------
    keys : array_like
        1-dimensional array of keys used to group the element of
        `values`.
    values : array_like
        The values (elements) to be grouped by `keys`.
    assume_sorted : bool, optional
        If ``True``, the key-value pairs are assumed to be sorted with
        respect to the keys, which can speed up the calculation.
    return_keys : bool, optional
        If ``True``, return the unique, sorted keys.
    debug : bool, optional
        If ``True``, check the input arguments.

    Returns
    -------
    unique_keys : numpy.ndarray
        The unique, sorted keys. Only returned, if `return_keys` is
        ``True``.
    grouped_values : list of numpy.ndarrays
        List of numpy.ndarrays, one array for all elements of `values`
        belonging to the same key. The arrays in the list are sorted
        according to the unique, sorted keys.
    """
    keys = np.asarray(keys)
    values = np.asarray(values)
    if debug:
        mdt.check.array(keys, dim=1)
        if len(keys) != len(values):
            raise ValueError("keys and values must have the same length")

    if not assume_sorted:
        ix_sort = np.argsort(keys)
        keys = keys[ix_sort]
        values = values[ix_sort]

    unique_keys, counts = np.unique(keys, return_counts=True)
    sections = np.cumsum(counts[:-1])

    if return_keys:
        return unique_keys, np.split(values, sections)
    else:
        return np.split(values, sections)


def get_middle(a, n=1, debug=False):
    """
    Get the `n` innermost elements of a 1-dimensional array.

    Parameters
    ----------
    a : array_like
        The 1-dimensional array from which to get the n middle elements.
    n : int, optional
        Number of elements to write out.
    debug : bool, optional
        If ``True``, check the input arguments.

    Returns
    -------
    b : array_like
        Array of the `n` innermost elements of `a`.
    """
    if debug:
        mdt.check.array(a, dim=1)

    if n > len(a):
        raise ValueError("n exceeds the length of a")

    b = a[(len(a) - n) // 2 : len(a) - (len(a) - n) // 2]
    if len(b) > n:
        b = b[:n]
    else:
        raise ValueError(
            "The length of b ({}) is less than n ({}). This should not have"
            " happened".format(len(b), n)
        )

    return b


def symmetrize(a, lower=True, debug=False):
    """
    Create a symmetrized copy of a 2-dimensional array.

    Parameters
    ----------
    a : array_like
        2-dimensional array to symmetrize.
    lower : bool, optional
        If ``True``, mirror the lower triangle of `a` to the upper one.
        If ``False``, mirror the upper triangle to the lower one. The
        diagonal of `a` is unchanged.
    debug : bool, optional
        If ``True``, check the input arguments.

    Returns
    -------
    a_sym : numpy.ndarray
        A symmetrized copy of `a`.
    """
    if debug:
        mdt.check.array(a, dim=2)

    if lower:
        a_sym = np.tril(a)
    else:
        a_sym = np.triu(a)
    a_sym += a_sym.T
    return a_sym - np.diag(np.diagonal(a))


def tilt_diagonals(a, clockwise=True, diagpos=None, debug=False):
    """
    Tilt a 2-dimensional array by 45 degrees so that the diagonals will
    either be converted to columns (if tilted clockwise) or to rows (if
    tilted counterclockwise).

    Parameters
    ----------
    a : array_like
        2-dimensional array of shape ``(m, n)``
    clockwise : bool, optional
        If ``True``, tilt the diagonals clockwise, if ``False`` tilt
        counterclockwise.
    diagpos : int, optional
        Index of the column (if tilted clockwise) or row (if tilted
        counterclockwise) where the main diagonal will be located after
        tilting. The default is to put the diagonal in the middle
        column/row of the tilted array.
    debug : bool, optional
        If ``True``, check the input arguments.

    Returns
    -------
    a_tilted : numpy.ndarray
        2-dimensional array of shape ``(m, n)`` (if tilted clockwise) or
        ``(n, m)`` (if tilted counterclockwise).
    """
    a = np.asarray(a)
    if debug:
        mdt.check.array(a, dim=2)

    if a.shape[0] > a.shape[1]:
        reps = int(np.ceil((a.shape[1] + a.shape[0]) / a.shape[1]))
        diags = np.tile(a, reps)
        diags = np.array(
            [np.diagonal(diags, offset=i) for i in range(a.shape[1])]
        )
    else:
        diags = np.array(
            [
                np.concatenate(
                    (
                        np.diagonal(a, offset=i),
                        np.diagonal(a, offset=-a.shape[1] + i),
                    )
                )
                for i in range(a.shape[1])
            ]
        )

    if diagpos is None:
        diagpos = int(a.shape[1] / 2)

    if clockwise:
        return np.roll(diags, shift=diagpos, axis=0).T
    else:
        diagpos += 1
        return np.roll(diags[::-1], shift=diagpos, axis=0)


def extend(a, length, axis=-1, fill_value=0):
    """
    Extend an axis of an array to a given length by padding a given
    value at the end of the axis.

    Parameters
    ----------
    a : array_like
        The array to extend.
    length : int
        The desired length of `a`.
    axis : int, optional
        The axis to extend. Default: last axis
    fill_value : scalar, optional
        The fill value to use to extend the given axis of `a` to the
        desired length. Note that you might have to change the data type
        of the input array to fit the type of `fill_value` before
        parsing the array to this function.

    Returns
    -------
    a_new : numpy.ndarray
        A copy of the input array with the given axis padded with
        `fill_value` in such a wax that it has the desired length. If
        the given axis already had the desired length or is even longer,
        just the input array is returned.

    Examples
    --------
    >>> a = np.arange(3)
    >>> mdt.nph.extend(a, 2)
    array([0, 1, 2])
    >>> mdt.nph.extend(a, 5)
    array([0, 1, 2, 0, 0])
    >>> mdt.nph.extend(a, 5, fill_value=9)
    array([0, 1, 2, 9, 9])

    >>> a = np.arange(6).reshape(2, 3)
    >>> mdt.nph.extend(a, 5)
    array([[0, 1, 2, 0, 0],
           [3, 4, 5, 0, 0]])
    >>> mdt.nph.extend(a, 4, axis=0)
    array([[0, 1, 2],
           [3, 4, 5],
           [0, 0, 0],
           [0, 0, 0]])

    >>> a = np.arange(12).reshape(2, 2, 3)
    >>> mdt.nph.extend(a, 5)
    array([[[ 0,  1,  2,  0,  0],
            [ 3,  4,  5,  0,  0]],
    <BLANKLINE>
           [[ 6,  7,  8,  0,  0],
            [ 9, 10, 11,  0,  0]]])
    >>> mdt.nph.extend(a, 3, axis=0)
    array([[[ 0,  1,  2],
            [ 3,  4,  5]],
    <BLANKLINE>
           [[ 6,  7,  8],
            [ 9, 10, 11]],
    <BLANKLINE>
           [[ 0,  0,  0],
            [ 0,  0,  0]]])
    >>> mdt.nph.extend(a, 3, axis=1)
    array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 0,  0,  0]],
    <BLANKLINE>
           [[ 6,  7,  8],
            [ 9, 10, 11],
            [ 0,  0,  0]]])
    """
    a = np.asarray(a)
    try:
        if a.shape[axis] >= length:
            return a
    except IndexError:
        raise np.AxisError(axis, ndim=a.ndim)
    pad_width = [[0, 0] for i in range(a.ndim)]
    pad_width[axis][1] = length - a.shape[axis]
    return np.pad(a, pad_width, mode="constant", constant_values=fill_value)


def match_shape(a, b, fill_value=0, transfer_to_new_dim=True):
    """
    Bring two arrays to the same shape by padding a given value at the
    end of of the axes of the shorter array.

    Parameters
    ----------
    a, b : array_like
        The arrays to bring to the same shape.
    fill_value : scalar, optional
        The fill value to use if the arrays must be extended in one or
        more dimensions to get them to the same shape. Note that you
        might have to change the data type of the input arrays to fit
        the type of `fill_value` before parsing them to this function.
    transfer_to_new_dim : bool, optional
        If a new dimension needs to be added to one of the arrays,
        transfer the values of the original array also to this new
        dimension. Default: ``True``

    Returns
    -------
    a_new, b_new : numpy.ndarray
        Copies of the input arrays padded with `fill_value` in such a
        way that they have the same shape. If `a` and `b` already had
        the same shape, just the input arrays are returned.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape == b.shape:
        return a, b

    if a.ndim == b.ndim:
        shape = np.maximum(a.shape, b.shape)
    elif a.ndim < b.ndim:
        shape = np.maximum(a.shape, b.shape[-a.ndim :])
        shape = np.insert(shape, 0, b.shape[: a.ndim])
    else:
        shape = np.maximum(a.shape[-b.ndim :], b.shape)
        shape = np.insert(shape, 0, a.shape[: b.ndim])

    if transfer_to_new_dim:
        slice_a = [slice(0, None, None)] * len(shape)
    else:
        slice_a = [0] * len(shape)
    slice_b = slice_a.copy()

    slice_a[len(slice_a) - a.ndim :] = [
        slice(None, a.shape[i], None) for i in range(a.ndim)
    ]
    slice_a = tuple(slice_a)
    a_new = np.full(shape, fill_value, dtype=a.dtype)
    a_new[slice_a] = a

    slice_b[len(slice_b) - b.ndim :] = [
        slice(None, b.shape[i], None) for i in range(b.ndim)
    ]
    slice_b = tuple(slice_b)
    b_new = np.full(shape, fill_value, dtype=b.dtype)
    b_new[slice_b] = b

    return a_new, b_new


def cross_section(z, x, y, line, num=None, order=1):
    """
    Extract a cross section from a 2-dimensional array along a given
    line using :func:`scipy.ndimage.map_coordinates`. See also
    https://stackoverflow.com/questions/18920614/plot-cross-section-through-heat-map

    Parameters
    ----------
    z : array_like
        Input array of shape ``(m, n)`` from which to extract the cross
        section. The data representation follows the standard matrix
        convention, i.e. columns represent the `x` values and rows the
        `y` values.
    x, y : array_like
        The sample points corresponding to the z values. `x` must be of
        shape ``(n,)`` and `y` must be of shape ``(m,)``.
    line : array_like
        Start point and end point of the line along which to extract the
        cross section. `line` must be given in the following format:
        ``[(x0, y0), (x1, y1)]`` with ``(x0, y0)`` being the start point
        and ``(x1, y1)`` being the end point. Note that the start point
        and end point must lie within the data range given by `x` and
        `y`.
    num : int, optional
        Number of points to use for sampling the cross section along the
        given line. If ``None`` (default), approximately as many sample
        points are used as real data points are on the line.
    order : int, optional
        Interpolation order.

    Returns
    -------
    cs : numpy.ndarray
        (Interpolated) values of `z` along the given line.
    r : numpy.ndarray
        The sample points along the given line.
    """
    z = np.asarray(z)
    x = np.asarray(x)
    y = np.asarray(y)
    line = np.asarray(line)

    if z.ndim != 2:
        raise ValueError("z must be a 2d array")
    if len(x) != z.shape[1]:
        raise ValueError(
            "The length of x does not match the number of columns of z"
        )
    if len(y) != z.shape[0]:
        raise ValueError(
            "The length of y does not match the number of rows of z"
        )
    if line.shape != (2, 2):
        raise ValueError("line must be given as [(x0, y0), (x1, y1)]")

    # Convert line from world coordinates to pixel coordinates
    x_world, y_world = np.asarray(list(zip(*line)))
    l_world = np.hypot(x_world[1] - x_world[0], y_world[1] - y_world[0])
    x_world_min = np.min(x)
    y_world_min = np.min(y)
    if np.any(x_world < x_world_min) or np.any(x_world > np.max(x)):
        raise ValueError("The line is outside the x-data range")
    if np.any(y_world < y_world_min) or np.any(y_world > np.max(y)):
        raise ValueError("The line is outside the y-data range")
    x_pix = z.shape[1] * (x_world - x_world_min) / x.ptp()  # ptp = max-min
    y_pix = z.shape[0] * (y_world - y_world_min) / y.ptp()

    # Interpolate the line at "num" points
    if num is None:
        _, x0_ix = find_nearest(x, x_world[0], return_index=True)
        _, x1_ix = find_nearest(x, x_world[1], return_index=True)
        _, y0_ix = find_nearest(y, y_world[0], return_index=True)
        _, y1_ix = find_nearest(y, y_world[1], return_index=True)
        num = max(abs(x1_ix - x0_ix), abs(y1_ix - y0_ix))
    if num <= 0:
        raise ValueError("num must be positive")
    cols = np.linspace(x_pix[0], x_pix[1], num)
    rows = np.linspace(y_pix[0], y_pix[1], num)
    r = np.linspace(0, l_world, num)
    cs = map_coordinates(z, np.vstack((rows, cols)), order=order)

    return cs, r


def cross_section2d(z, ax, x=None, y=None, width=1, mean="arithmetic"):
    """
    Extract a cross section from a 2-dimensional array along a given
    axis.

    Note
    ----
    This function is probably not what you want, since the cross section
    is always averaged either vertically (if `y` is given) or
    horizontally (if `x` is given), but not perpendicular to the given
    axis as it should be. The problem might get clear, if you uncomment
    the ``print(mask)`` statement in the ``for`` loop of this function.
    For extracting proper cross sections from 2-dimensional arrays use
    :func:`cross_section`.

    Parameters
    ----------
    z : array_like
        Input array of shape ``(m, n)`` from which to extract the cross
        section. The data representation follows the standard matrix
        convention, i.e. columns represent the `x` values and rows the
        `y` values.
    ax : array_like
        The axis along which to extract the cross section from `z`.
        Usually `ax` is a straight line as function of `x`, but actually
        it can be any function of `x` or `y`. If `y` is given, `ax` is
        interpreted as a function of `x` and must be of shape ``(n,)``.
        If `x` is given, it is interpreted as a function of `y` and must
        be of shape ``(m,)``.
    x, y : array_like
        The sample points corresponding to the z values. Either `x` or
        `y` must be given, but not both. `x` must be of shape ``(n,)``
        if given and `y` must be of shape ``(m,)`` if given.
    width : scalar, optional
        The width of the cross section. Must be greater than zero and
        should be chosen sufficiently large to avoid numerical
        instabilities. If `y` is given, the cross section is computed
        from all `z` values that fall within the range
        ``(y >= ax-width/2) & (y <= ax+width/2)``. If `x` is given, the
        cross section is computed from all `z` values that fall within
        the range ``(x >= ax-width/2) & (x <= ax+width/2)``.
    mean : str, optional
        How to average over the width of the cross section to obtain the
        actual value of the cross section at the current `x` (if `y` is
        given) or `y` (if `x` is given) position. Must be one of the
        following choices:

        * ``'arithmetic'``: Use arithmetic mean
        * ``'integral'``: Integrate over the current width slice and
          divide by the width

    Returns
    -------
    cross_sec : numpy.ndarray
        Cross section as function of `x` and shape ``(n,)`` if `y` is
        given or as function of `y` and shape ``(m,)`` if `y` is given.
    """
    z = np.asarray(z)
    if z.ndim != 2:
        raise ValueError("z must be a 2d array")
    if x is None and y is not None:
        var = np.asarray(y)
        zax = 1
        zax_var = 0
    elif x is not None and y is None:
        var = np.asarray(x)
        zax = 0
        zax_var = 1
    else:
        raise ValueError("Either x or y must be given")
    if len(var) != z.shape[zax_var]:
        raise ValueError("The length of x/y does not match the size of z")
    if len(ax) != z.shape[zax]:
        raise ValueError("The length of ax does not match the size of z")
    if width < 0:
        raise ValueError("width must not be negative")
    if width == 0 and mean != "arithmetic":
        print("Setting mean to 'arithmetic', because width is zero")
    if mean != "arithmetic" and mean != "integral":
        raise ValueError("mean must be either 'arithmetic' or 'integral'")

    width2 = width / 2
    cross_sec = np.full(z.shape[zax], np.nan, dtype=np.float64)
    for i, zz in enumerate(z if zax == 0 else z.T):
        if width == 0:
            mask = var == ax[i]
        else:
            mask = (var >= ax[i] - width2) & (var <= ax[i] + width2)
        # print(mask)
        if mean == "arithmetic":
            cross_sec[i] = np.mean(zz[mask])
        elif mean == "integral":
            cross_sec[i] = np.trapz(zz[mask], var[mask]) / (width)

    return cross_sec


def trapz2d(z, x=None, y=None, dx=1, dy=1, xlim=None, ylim=None):  # noqa: C901
    """
    Integrate `z(x,y)` within the given limits using ``numpy.trapz``
    twice, first in `y` and then in `x` direction.

    Parameters
    ----------
    z : array_like
        Input array to integrate.
    x, y : array_like, optional
        The sample points corresponding to the z values. If ``None``
        (default), the sample points are assumed to be evenly spaced
        `dx` and `dy` apart.
    dx, dy : scalar, optional
        The spacing between sample points when `x` or `y` is ``None``.
    xlim : array_like, optional
        Array of length `2` containing an upper and a lower integration
        boundary for integration in `x` direction. If ``None``
        (default), no limits are applied. If you only want to apply an
        upper (lower) limit, set the lower (upper) limit to ``-np.inf``
        (``np.inf``).
    ylim : array_like, optional
        Array of length `2` containing upper and lower integration
        boundaries for integration in `y` direction. The boundaries can
        be either constant scalars or arrays of the same shape as `x`.
        The possibility to parse arrays can e.g. be used to parse `y`
        limits that depend on `x`. If ``None`` (default), no limits are
        applied. If you only want to apply an upper (lower) limit, set
        the lower (upper) limit to ``-np.inf`` (``np.inf``).

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule.
    """
    z = np.asarray(z)
    if x is None:
        x = np.arange(0, z.shape[0] * dx, dx)
    else:
        x = np.asarray(x)
    if y is None:
        y = np.arange(0, z.shape[1] * dy, dy)
    else:
        y = np.asarray(y)

    if z.ndim != 2:
        raise ValueError("z must be a 2d array")
    if len(x) != z.shape[0]:
        raise ValueError(
            "The length of x does not match the first dimension of z"
        )
    if len(y) != z.shape[1]:
        raise ValueError(
            "The length of y does not match the second dimension of z"
        )
    if xlim is not None:
        if np.ndim(xlim) == 0:
            raise ValueError(
                "xlim must be either None or an array-like object of length 2"
            )
        if len(xlim) != 2:
            raise ValueError(
                "xlim must be either None or an array-like object of length 2"
            )
        if np.ndim(xlim[0]) != 0:
            raise ValueError("xlim[0] must be a scalar")
        if np.ndim(xlim[1]) != 0:
            raise ValueError("xlim[1] must be a scalar")
        if xlim[1] <= xlim[0]:
            raise ValueError("xlim[1] must be greater than xlim[0]")
    if ylim is not None:
        if np.ndim(ylim) == 0:
            raise ValueError(
                "ylim must be either None or an array-like object of length 2"
                " in the first dimension"
            )
        if len(ylim) != 2:
            raise ValueError(
                "ylim must be either None or an array-like object of length 2"
                " in the first dimension"
            )
        if np.ndim(ylim[0]) != 0 and ylim[0].shape != x.shape:
            raise ValueError(
                "ylim[0] must be either a scalar or an array of the same shape"
                " as x"
            )
        if np.ndim(ylim[1]) != 0 and ylim[1].shape != x.shape:
            raise ValueError(
                "ylim[1] must be either a scalar or an array of the same shape"
                " as x"
            )
        if np.any(ylim[1] <= ylim[0]):
            raise ValueError("ylim[1] must be greater than ylim[0]")

    yint = np.zeros(z.shape[0])
    if ylim is None:
        for i, zy in enumerate(z):
            yint[i] = np.trapz(y=zy, x=y)
    elif np.ndim(ylim[0]) == 0 and np.ndim(ylim[1]) == 0:
        mask = (y >= ylim[0]) & (y <= ylim[1])
        y = y[mask]
        for i, zy in enumerate(z):
            yint[i] = np.trapz(y=zy[mask], x=y)
    elif np.ndim(ylim[0]) == 0 and np.ndim(ylim[1]) != 0:
        for i, zy in enumerate(z):
            mask = (y >= ylim[0]) & (y <= ylim[1][i])
            yint[i] = np.trapz(y=zy[mask], x=y[mask])
    elif np.ndim(ylim[0]) != 0 and np.ndim(ylim[1]) != 0:
        for i, zy in enumerate(z):
            mask = (y >= ylim[0][i]) & (y <= ylim[1][i])
            yint[i] = np.trapz(y=zy[mask], x=y[mask])

    if xlim is None:
        trapz = np.trapz(y=yint, x=x)
    else:
        mask = (x >= xlim[0]) & (x <= xlim[1])
        trapz = np.trapz(y=yint[mask], x=x[mask])

    return trapz


def find_linear_region(
    data,
    window_length,
    polyorder=2,
    delta=1,
    tol=0.1,
    axis=-1,
    correct_intermittency=None,
    visualize=False,
):
    """
    Find linear regions in a data array. This is done by applying a
    Savitzky-Golay differentiation filter to the array using
    :func:`scipy.signal.savgol_filter`. The linear regions are
    identified by checking where the second derivative is zero within a
    given tolerance. The sample points of the data must be equidistant.
    For a brief introduction to Savitzky-Golay filters see for instance
    https://eigenvector.com/wp-content/uploads/2020/01/SavitzkyGolay.pdf

    Parameters
    ----------
    data : array_like
        The data in which to identify linear regions.
    window_length : int
        The length of the filter window. `window_length` must be a
        positive odd integer and it must be less than or equal to the
        size of `data`. Note that the window size should be smaller than
        the smallest characteristic you want to resolve. Otherwise these
        characteristics will be smoothed out. Keep in mind that the
        purpose of smoothing is to get rid of random noise while
        (ideally) preserving the true signal.
    polyorder : int, optional
        The order of the polynomial used to fit the samples. `polyorder`
        must be less than `window_length` but at least two. Keep in mind
        that you are fitting `window_length` data points with a
        polynomial of order `polyorder`. Thus, in order to avoid
        overfitting, `polyorder` should in fact be considerably smaller
        than `window_length`. If `polyorder` is too high, no smoothing
        effect will be seen. `polyorder` must be at least two, because
        otherwise the second derivative of the polynomial will always be
        zero.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
    tol : float, optional
        Tolerance within which the second derivative is considered to be
        zero.
    axis : int, optional
        The axis of the array `data` along which the filter is to be
        applied. Default is the last axis.
    correct_intermittency : int, optional
        Apply :func:`mdtools.dynamics.correct_intermittency_1d` to the
        result array. A value smaller than or equal to zero will not
        change the result. A value higher than zero can be used to
        eliminate numerical fluctuations in the result. E.g. if the
        result array is T,T,F,T,T it will be changed to T,T,T,T,T if
        `correct_intermittency` is one or higher. The default for
        `correct_intermittency` is ``window_length//2``. See
        :func:`mdtools.dynamics.correct_intermittency_1d` for more
        details.
    visualize : bool, optional
        Visualize the result of applying the Savitzky-Golay filter to
        the data. This should only be used in interactive algorithms,
        since the plot window must be manually closed by the user.

    Returns
    -------
    linear : numpy.ndarray
        Boolean array. A ``True`` element indicates that the second
        derivative of `data` is zero within the given tolerance at the
        respective position. Therefore, the data can be considered
        linear at this position.
    """
    if polyorder < 2:
        raise ValueError("polyorder must be at least 2")

    grad2 = savgol_filter(
        x=data,
        window_length=window_length,
        polyorder=polyorder,
        deriv=2,
        delta=delta,
        axis=axis,
    )
    linear = np.abs(grad2) <= tol
    if correct_intermittency is None:
        correct_intermittency = window_length // 5
    mdt.dynamics.correct_intermittency_1d(
        a=linear, intermittency=correct_intermittency
    )

    if visualize:
        grad0 = savgol_filter(
            x=data,
            window_length=window_length,
            polyorder=polyorder,
            deriv=0,
            delta=delta,
            axis=axis,
        )
        grad1 = savgol_filter(
            x=data,
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            delta=delta,
            axis=axis,
        )
        plt.figure(clear=True)
        plt.axhline(y=0, color="black")
        plt.plot(data, linestyle="", marker="o", label="Data")
        plt.plot(grad0, label="Gradient 0")
        plt.plot(grad1, label="Gradient 1")
        plt.plot(grad2, label="Gradient 2")
        plt.plot(linear, linewidth=4, label="Linear (1=True)")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    return linear
