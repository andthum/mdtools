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
Auxiliary functions for :mod:`SciPy <scipy>` routines.

This module can be called from :mod:`mdtools` via the shortcut ``sph``::

    import mdtools as mdt
    mdt.sph  # insetad of mdt.scipy_helper_functions

"""


def multiple_multiply(*args):
    """
    Multiply multiple objects that have a :meth:`.mutiply` method at
    once.

    Parameters
    ----------
    args : object
        Arbitrary number of objects that should be multiplied.  The
        objects must have a :meth:`.mutiply` method.

    Returns
    -------
    product : object
        The product of all input objects as returned by the objects'
        :meth:`.mutiply` method.

    Raises
    ------
    ValueError
        If :func:`mdtools.scipy_helper_functions.multiple_multiply` is
        called without arguments.
    AttributeError or TypeError
        If the input ojects don't have a :meth:`.mutiply` method.

    Notes
    -----
    This function was originally designed to multiply multiple
    :mod:`SciPy sparse matrices <scipy.sparse>` at once element-wise.
    With e.g. :meth:`scipy.sparse.csr_matrix.multiply` you can multiply
    only two matrices at once, but :func:`multiple_multiply` takes an
    arbitrary number of arguments.

    Examples
    --------
    >>> from scipy import sparse
    >>> a = np.arange(6).reshape(2,3)
    >>> a = sparse.csr_matrix(a)
    >>> a.toarray()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mdt.sph.multiple_multiply(a).toarray()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mdt.sph.multiple_multiply(a, a).toarray()
    array([[ 0,  1,  4],
           [ 9, 16, 25]])
    >>> mdt.sph.multiple_multiply(a, a, a).toarray()
    array([[  0,   1,   8],
           [ 27,  64, 125]])
    >>> mdt.sph.multiple_multiply(*[a,]*5).toarray()
    array([[   0,    1,   32],
           [ 243, 1024, 3125]])
    """
    if len(args) == 0:
        raise ValueError("The function was called without arguments")
    elif len(args) == 1:
        return args[0]
    else:
        return multiple_multiply(args[0].multiply(args[1]), *args[2:])
