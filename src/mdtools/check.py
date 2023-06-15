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
Functions for debugging and checking user input or arguments parsed to
other functions, classes, etc.
"""


# Standard libraries
# Third party libraries
import decimal as dec
import warnings
from datetime import datetime

# Third-party libraries
import numpy as np

# First-party libraries
# Local application/library specific imports
import mdtools as mdt


def array(
    a, shape=None, dim=None, axis=None, amin=None, amax=None, dtype=None
):
    """
    Check if the input array satisfies the given conditions.

    The array must meet the following requirements:

        * Must be of an `array_like` type, i.e. a type that can be
          converted to a :class:`numpy.ndarray`.
        * Must have a shape equal to the supplied `shape`.
        * Must have a dimension equal to the supplied `dim`.
        * Must contain the supplied `axis`.
        * All array elements must be greater than or equal to `amin` (if
          supplied).
        * All array elements must be less than or equal to `amax` (if
          supplied).
        * Must have a data type equal to the supplied `dtype`.

    Parameters
    ----------
    a : array_like
        The array to check.
    shape : tuple, optional
        The shape expected for `a`.  Default is ``None``, which means
        that the shape of `a` is not checked.
    dim : int, optional
        The dimension expected for `a`.  Default is ``None``, which
        means that the dimension of `a` is not checked.
    axis : int, optional
        Index of an axis that should be contained in `a`.  Default is
        ``None``, which means that it is not checked whether `a`
        contains the given axis.
    amin : scalar or array_like
        Minimum values that the elements of `a` must not undermine.  If
        `amin` and `a` do not have the same shape, they must be
        broadcastable to a common shape.
    amax : scalar or array_like
        Maximum values that the elements of `a` must not exceed.  If
        `amax` and `a` do not have the same shape, they must be
        broadcastable to a common shape.  `amax` must be greater than
        `amin`.
    dtype : type, optional
        The data type expected for `a`.  Default is ``None``, which
        means that the data type of `a` is not checked.

    Returns
    -------
    a : numpy.ndarray
        The input array as :class:`numpy.ndarray`.

    Raises
    ------
    ValueError
        If
        `shape` is not ``None`` and the shape of `a` is not `shape`;
        `dim` is not ``None`` and the dimension of `a` is not `dim`;
        `a` contains elements that are less than `amin`;
        `a` contains elements that are greater than `amax`;

        Or if
        the given combination of `shape`, `dim` and `axis` is invalid;
        `amax` is less than `amin`.
    :exc:`numpy.AxisError`
        If `axis` is not ``None`` and the axis is out of bounds for `a`.
    TypeError
        If `dtype` is not ``None`` and the data type of `a` is not
        `dtype`.
    RuntimeError
        If all arguments (except `a`) are ``None``.

    See Also
    --------
    :func:`mdtools.check.pos_array` :
        Check if an array is a suitable position array
    :func:`mdtools.check.box` :
        Check if an array is a suitable simulation box array
    :func:`mdtools.check.dtrj` :
        Check if an array is a suitable discrete trajectory array
    """
    a = np.asarray(a)
    # Check input parameters:
    if all(arg is None for arg in (shape, dim, axis, amin, amax, dtype)):
        raise RuntimeError("No arguments provided to check `a`")
    if dim is not None and shape is not None and dim != len(shape):
        raise ValueError(
            "Invalid combination of `dim` ({}) and `shape` ({}).  `dim` must"
            " be equal to len(shape)".format(dim, shape)
        )
    if axis is not None and shape is not None:
        if axis >= len(shape) or axis < -len(shape):
            raise ValueError(
                "Invalid combination of `axis` ({}) and `shape` ({}).  `axis`"
                " must lie in the interval"
                " [-len(shape), len(shape)[".format(axis, shape)
            )
    if axis is not None and dim is not None:
        if axis >= dim or axis < -dim:
            raise ValueError(
                "Invalid combination of `axis` ({}) and `dim` ({}).  `axis`"
                " must lie in the interval [-dim, dim[".format(axis, dim)
            )
    if amin is not None and amax is not None and np.any(amax < amin):
        raise ValueError(
            "`amax` ({}) must be greater than `amin` ({})".format(amax, amin)
        )

    # Check array:
    if shape is not None and a.shape != shape:
        raise ValueError(
            "`a` has shape {} but must have shape {}".format(a.shape, shape)
        )
    if dim is not None and a.ndim != dim:
        raise ValueError(
            "`a` has {} dimension(s) but must have {}"
            " dimension(s)".format(a.ndim, dim)
        )
    if axis is not None:
        try:
            a.shape[axis]
        except IndexError:
            raise np.AxisError(axis=axis, ndim=a.ndim)
    if amin is not None and np.any(a < amin):
        raise ValueError(
            "At least one element of `a` is less than `amin` ({})".format(amin)
        )
    if amax is not None and np.any(a > amax):
        raise ValueError(
            "At least one element of `a` is greater than `amax`"
            " ({})".format(amax)
        )
    if dtype is not None and a.dtype != dtype:
        raise TypeError(
            "The data type of `a` is {} but must be {}".format(a.dtype, dtype)
        )
    return a


def pos_array(
    pos_array, shape=None, dim=None, amin=None, amax=None, dtype=None
):
    """
    Check if the input array satisfies the conditions for a position
    array.

    Arrays that contain positions (spatial coordinates) of particles
    must meet the following requirements:

        * Must be of an `array_like` type, i.e. a type that can be
          converted to a :class:`numpy.ndarray`.
        * Must be of shape ``(3,)`` or ``(n, 3)`` or ``(k, n, 3)``,
          where ``n`` is the number of particles and ``k`` is the number
          of frames.

    Parameters
    ----------
    pos_array : array_like
        The array to check.
    shape : tuple, optional
        The shape expected for `pos_array`.  Default is ``None``, which
        means that the shape of `pos_array` is not checked.  If
        provided, the length of `shape` must be either one, two or three
        and the last element of shape must be 3.
    dim : {None, 1, 2, 3}, optional
        The dimension expected for `pos_array`.  Default is ``None``,
        which means that the dimension of `pos_array` is not checked.
    amin : scalar or array_like
        Minimum values that the elements of `pos_array` must not
        undermine.  If `amin` and `pos_array` do not have the same
        shape, they must be broadcastable to a common shape.
    amax : scalar or array_like
        Maximum values that the elements of `pos_array` must not exceed.
        If `amax` and `pos_array` do not have the same shape, they must
        be broadcastable to a common shape.  `amax` must be greater than
        `amin`.
    dtype : type, optional
        The data type expected for `pos_array`.  The default is
        ``None``, which means that the data type of `pos_array` is not
        checked.

    Returns
    -------
    pos_array : numpy.ndarray
        The input array as :class:`numpy.ndarray`.

    Raises
    ------
    ValueError
        If
        the last dimension of `pos_array` is not of shape ``(3,)``;
        The dimension of `pos_array` is neither one, two or three.

        Or if
        `shape` is not ``None`` and the length of `shape` is neither
        one, two or three;
        `dim` is not ``None`` and neither one, two or three.

        See :func:`mdtools.check.array` for further potentially raised
        exceptions.

    See Also
    --------
    :func:`mdtools.check.array` :
        Check if an array meets given requirements
    :func:`mdtools.check.box` :
        Check if an array is a suitable simulation box array
    """
    pos_array = np.asarray(pos_array)
    allowed_dims = (1, 2, 3)
    # Check input parameters:
    if shape is not None and len(shape) not in allowed_dims and shape[-1] != 3:
        raise ValueError(
            "`shape` ({}) must be either (3,) or (n, 3) or"
            " (k, n, 3)".format(shape)
        )
    if dim is not None and dim not in allowed_dims:
        raise ValueError("`dim` ({}) must be in {}".format(dim, allowed_dims))

    # Check position array:
    if pos_array.ndim not in allowed_dims or pos_array.shape[-1] != 3:
        raise ValueError(
            "`pos_array` has shape {} but must have shape (3,) or (n, 3) or"
            " (k, n, 3)".format(pos_array.shape)
        )
    if any(arg is not None for arg in (shape, dim, amin, amax, dtype)):
        mdt.check.array(
            pos_array, shape=shape, dim=dim, amin=amin, amax=amax, dtype=dtype
        )
    return pos_array


def box(
    box,
    with_angles=None,
    orthorhombic=False,
    allow_negative=False,
    allow_zero=False,
    dim=None,
    dtype=None,
):
    """
    Check if the input array satisfies the conditions for a simulation
    box array in length-angle representation.

    Arrays that contain the dimensions of simulation boxes in the
    length-angle representation must meet the following requirements:

        * Must be of an `array_like` type, i.e. a type that can be
          converted to a :class:`numpy.ndarray`.
        * Must have a data type equal to the supplied `dtype`.
        * If the array does not contain the box angles:

            * Must be of shape ``(3,)`` or ``(k, 3)``, where ``k`` is
              the number of frames.

        * If the array contains the box angles:

            * Must be of shape ``(6,)`` or ``(k, 6)``.
            * The angles must be greater than 0 and less than 180°.

    Parameters
    ----------
    box : array_like
        The array to check.
    with_angles : None or bool, optional
        If ``None`` (default), whether the box contains angles or not is
        inferred from the shape of the last dimension of `box`.  If the
        last dimension has shape ``(3,)``, `box` is considered to not
        contain angles, if it has shape ``(6,)``, `box` is considered to
        contain angles.  If `with_angles` is ``True`` or ``False`` the
        box must or must not contain angles, otherwise an exception is
        raised.
    orthorhombic : bool, optional
        If ``True`` and `with_angles` evaluates to ``True``, all angles
        of `box` must be 90°.  Is ignored if `with_angles` is ``False``
        or if `box` does not contain angle information.
    allow_negative : bool, optional
        If ``True``, allow negative box lengths (but not zero).
    allow_zero : bool, optional
        If ``True``, allow box lengths to be zero.
    dim : {None, 1, 2}, optional
        The dimension expected for `box`.  Default is ``None``, which
        means that the dimension of `box` must be either 1 or 2.  If
        `box` should contain the box parameters for only one frame, set
        `dim` to 1.  If `box` should contain the box parameters for
        multiple frames, set `dim` to 2.
    dtype : type, optional
        The data type expected for `box`.  Default is ``None``, which
        means that the data type of `box` is not checked.

    Returns
    -------
    box : numpy.ndarray
        The input array as :class:`numpy.ndarray`.

    Raises
    ------
    ValueError
        If `box` has an
        incorrect shape;
        incorrect dimensionality;
        an invalid angle;
        an invalid box length.

        Or if `dim` is not ``None`` and neither one or two.

        See :func:`mdtools.check.array` for further potentially raised
        exceptions.

    See Also
    --------
    :func:`mdtools.check.box_mat` :
        Check if the input array satisfies the conditions for a
        simulation box matrix.
    :func:`mdtools.check.array` :
        Check if an array meets given requirements
    :func:`mdtools.check.pos_array` :
        Check if an array is a suitable position array
    """
    box = np.asarray(box)
    allowed_dims = (1, 2)
    # Check input parameters:
    if dim is not None and dim not in allowed_dims:
        raise ValueError("`dim` ({}) must be in {}".format(dim, allowed_dims))

    # Check box array:
    if box.ndim not in allowed_dims:
        raise ValueError(
            "`box` has shape {} but must have shape (3,) or (6,) or (k, 3) or"
            " (k, 6)".format(box.shape)
        )

    if with_angles is None:
        if box.shape[-1] == 3:
            with_angles = False
        elif box.shape[-1] == 6:
            with_angles = True
        else:
            raise ValueError(
                "`box` has shape {} but must have shape (3,) or (6,) or (k, 3)"
                " or (k, 6)".format(box.shape)
            )
    else:
        if with_angles and box.shape[-1] != 6:
            raise ValueError(
                "'box' has shape {} but must have shape (6,) or"
                " (k, 6)".format(box.shape)
            )
        if not with_angles and box.shape[-1] != 3:
            raise ValueError(
                "`box` has shape {} but must have shape (3,) or"
                " (k, 3)".format(box.shape)
            )
    if with_angles:
        angles = mdt.nph.take(box, start=3, axis=-1)
        if np.any(angles <= 0):
            raise ValueError(
                "At least one angle is less than or equal to zero"
            )
        if np.any(angles >= 180):
            raise ValueError(
                "At least one angle is greater than or equal to 180 degrees"
            )
        if orthorhombic and not np.allclose(angles, 90, rtol=0):
            raise ValueError(
                "`orthorhombic` is True but at least one angle is not 90"
                " degrees"
            )

    lengths = mdt.nph.take(box, start=0, stop=3, axis=-1)
    if not allow_negative and np.any(lengths < 0):
        raise ValueError("At least one box length is negative")
    if not allow_zero and np.any(np.isclose(lengths, 0)):
        raise ValueError("At least one box length is zero")

    if any(arg is not None for arg in (dim, dtype)):
        mdt.check.array(box, dim=dim, dtype=dtype)
    return box


def box_mat(
    box_mat, orthorhombic=False, x_aligned=False, dim=None, dtype=None
):
    """
    Check if the input array satisfies the conditions for a simulation
    box matrix.

    Arrays that contain simulation box matrices must meet the following
    requirements:

        * Must be of an `array_like` type, i.e. a type that can be
          converted to a :class:`numpy.ndarray`.
        * Must be of shape ``(3, 3)`` or ``(k, 3, 3)``, where ``k`` is
          the number of frames.

    Parameters
    ----------
    box_mat : array_like
        The array to check.  It is assumed that box matrices contain the
        box vectors as rows not as columns!
    orthorhombic : bool, optional
        If ``True``, the simulation boxes must be orthorhombic, i.e. all
        given box matrices must be diagonal.
    x_aligned : bool, optional
        If ``True``, the simulation boxes must align with the x-axis and
        the xy-plane, i.e. the first box vector must have the form
        ``[lx, 0, 0]`` and the second box vector must have the form
        ``[ly1, ly2, 0]``.
    dim : {None, 2, 3}, optional
        The dimension expected for `box_mat`.  Default is ``None``,
        which means that the dimension of `box_mat` is not checked.  If
        `box_mat` should contain the box vectors for only one frame, set
        `dim` to 2.  If `box_mat` should contain the box vectors for
        multiple frames, set `dim` to 3.
    dtype : type, optional
        The data type expected for `box_mat`.  Default is ``None``,
        which means that the data type of `box_mat` is not checked.

    Returns
    -------
    box_mat : numpy.ndarray
        The input array as :class:`numpy.ndarray`.

    Raises
    ------
    ValueError
        If `box_mat` has an incorrect shape or an incorrect
        dimensionality.

        Or if `orthorhombic` is ``True`` but the matrices have non-zero
        off-diagonal elements.

        Or if `x_aligned` is ``True`` but the values in the upper
        triangle of the matrices are not zero.

        Or if `dim` is not ``None`` and neither two or three.

        See :func:`mdtools.check.array` for further potentially raised
        exceptions.

    See Also
    --------
    :func:`mdtools.check.box` :
        Check if the input array satisfies the conditions for a
        simulation box array.
    :func:`mdtools.check.array` :
        Check if an array meets given requirements
    :func:`mdtools.check.pos_array` :
        Check if an array is a suitable position array
    """
    box_mat = np.asarray(box_mat)
    allowed_dims = (2, 3)
    # Check input parameters:
    if dim is not None and dim not in allowed_dims:
        raise ValueError("`dim` ({}) must be in {}".format(dim, allowed_dims))

    # Check box matrix:
    if box_mat.ndim not in allowed_dims or box_mat.shape[-2:] != (3, 3):
        raise ValueError(
            "`box_mat` has shape {} but must have shape (3, 3) or"
            " (k, 3, 3)".format(box_mat.shape)
        )
    if orthorhombic:
        diagonals = np.diagonal(box_mat, axis1=-2, axis2=-1)
        box_mat_diag = np.apply_along_axis(np.diag, -1, diagonals)
        if not np.allclose(box_mat, box_mat_diag, rtol=0):
            raise ValueError(
                "At least one of the given box(es) is not orthorhombic"
            )
    if x_aligned and not np.allclose(np.triu(box_mat, 1), 0, rtol=0):
        raise ValueError(
            "At least one of the given box(es) is not aligned with the"
            " x-axis and the xy-plane"
        )
    if any(arg is not None for arg in (dim, dtype)):
        mdt.check.array(box_mat, dim=dim, dtype=dtype)
    return box_mat


def dtrj(dtrj, shape=None, amin=None, amax=None, dtype=None):
    """
    Check if the input array satisfies the conditions for a discrete
    trajectory.

    Arrays that serve as discrete trajectory must meet the following
    requirements:

        * Must be of an `array_like` type, i.e. a type that can be
          converted to a :class:`numpy.ndarray`.
        * Must be of shape ``(f,)`` or ``(n, f)``, where ``n`` is the
          number of compounds and ``f`` is the number of frames.
          Transposed discrete trajectories of shape ``(f, n)`` are also
          valid.
        * Must contain only integers or floats whose fractional part is
          zero, because the elements of a discrete trajectory are
          interpreted as the indices of the states in which a given
          compound is at a given frame.

    Parameters
    ----------
    dtrj : array_like
        The array to check.
    shape : tuple, optional
        The shape expected for `dtrj`.  Default is ``None``, which means
        that the shape of `dtrj` is not checked.  If provided, the
        length of `shape` must be two.
    amin : scalar or array_like
        Minimum values that the elements of `dtrj` must not undermine.
        If `amin` and `dtrj` do not have the same shape, they must be
        broadcastable to a common shape.
    amax : scalar or array_like
        Maximum values that the elements of `dtrj` must not exceed.  If
        `amax` and `dtrj` do not have the same shape, they must be
        broadcastable to a common shape.  `amax` must be greater than
        `amin`.
    dtype : type, optional
        The data type expected for `dtrj`.  The default is ``None``,
        which means that the data type of `dtrj` is not checked.

    Returns
    -------
    dtrj : numpy.ndarray
        The input array as :class:`numpy.ndarray` with two dimensions
        and in row-major order (C-style memory layout).   No copy is
        performed if the input is already a 2-dimensional
        :class:`numpy.ndarray` with matching order.

    Raises
    ------
    ValueError
        If
        `dtrj` has more than two dimensions;
        Any element of `dtrj` is not an integer.

        Or if
        `shape` is not ``None`` and the length of `shape` is not two.

        See :func:`mdtools.check.array` for further potentially raised
        exceptions.

    See Also
    --------
    :func:`mdtools.check.array` :
        Check if an array meets given requirements

    Notes
    -----
    If the input array has only shape ``(f,)``, it is expanded to shape
    ``(1, f)``.  Hence, the output array will always have two
    dimensions.
    """
    dtrj = np.asarray(dtrj, order="C")
    # Check input parameters:
    if shape is not None and len(shape) != 2:
        raise ValueError(
            "The length of `shape` is {} but must be 2".format(len(shape))
        )

    # Check discrete trajectory array:
    if dtrj.ndim == 1:
        dtrj = np.expand_dims(dtrj, axis=0)
    elif dtrj.ndim != 2:
        raise ValueError(
            "`dtrj` has {} dimensions but must have one or two"
            " dimensions".format(dtrj.ndim)
        )
    if np.any(np.modf(dtrj)[0] != 0):
        raise ValueError("At least one element of 'dtrj' is not an integer")
    if any(arg is not None for arg in (shape, amin, amax, dtype)):
        mdt.check.array(dtrj, shape=shape, amin=amin, amax=amax, dtype=dtype)
    return dtrj


def list_of_cms(
        cms, shape=None, dim=None, amin=None, amax=None, dtype=None):
    """
    Check if the input is a sequence of contact matrices (or arrays).

    A sequence of contact matrices must meet the following requirements:

        * The sequence must be a tuple, list or :class:`numpy.ndarray`
          instance containing the contact matrices either as
          :class:`NumPy arrays <numpy.ndarray>` or as
          :mod:`SciPy sparse matrices <scipy.sparse>`.  A mix of
          :class:`NumPy arrays <numpy.ndarray>` and
          :mod:`SciPy sparse matrices <scipy.sparse>` is not permitted.
        * All arrays in the sequence must have the same shape.  If
          `shape` is supplied, their shape must be equal to `shape`.
        * All arrays in the sequence must have a dimension equal to the
          supplied `dim` (usually 2 for contact matrices).
        * All array elements must be greater than or equal to `amin` (if
          supplied; usually 0 for contact matrices).
        * All array elements must be less than or equal to `amax` (if
          supplied; usually 1 for contact matrices).
        * All arrays in the sequence must have a data type equal to the
          supplied `dtype`.

    Parameters
    ----------
    cms : tuple or list or numpy.ndarray
        Sequence of contact matrices (or generally arrays) as e.g.
        generated with :func:`mdtools.structure.contact_matrices`.  The
        contact matrices itself can be either
        :class:`NumPy arrays <numpy.ndarray>` or
        :mod:`SciPy sparse matrices <scipy.sparse>`.  A mix of
        :class:`NumPy arrays <numpy.ndarray>` and
        :mod:`SciPy sparse matrices <scipy.sparse>` is not permitted.
    shape : tuple, optional
        The shape expected for the arrays in `cms`.  Default is
        ``None``, which means that it is only checked whether all arrays
        in `cms` have the same shape, but not what shape that is.
    dim : int, optional
        The dimension expected for the arrays in `cms`.  If ``None``,
        the dimension is not checked.
    amin : scalar
        Minimum values that the elements of the arrays in `cms` must not
        undermine.
    amax : scalar
        Maximum values that the elements of the arrays in `cms` must not
        exceed.
    dtype : type, optional
        The data type expected for the arrays in `cms`.  Default is
        ``None``, which means that the data type is not checked.

    Returns
    -------
    cms : tuple or list or numpy.ndarray
        The unmodified input.

    Raises
    ------
    ValueError
        If
        `shape` is not ``None`` and the shape of the arrays in `cms` is
        not `shape`;
        `dim` is not ``None`` and the dimension of the arrays in `cms`
        is not `dim`;
        The arrays in `cms` contain elements that are less than `amin`;
        The arrays in `cms` contain elements that are greater than
        `amax`.

        Or if
        the given combination of `shape` and `dim` is not satisfiable by
        any array;
        `amax` is less than `amin`.
    TypeError
        If
        `cms` is not a sequence of arrays;
        `dtype` is not ``None`` and the data type of the arrays in
        `cms` is not `dtype`;
        Not all arrays in `cms` are of the same instance.
    """
    # Check input parameters:
    if dim is not None and shape is not None and dim != len(shape):
        raise ValueError("The given combination of 'dim' ({}) and"
                         " 'shape' ({}) is not satisfiable by any array"
                         .format(dim, shape))
    if amin is not None and amax is not None and amax < amin:
        raise ValueError("'amax' ({}) must be greater than 'amin' ({})"
                         .format(amax, amin))
    # Check list of contact matrices:
    if not isinstance(cms, (tuple, list, np.ndarray)):
        raise TypeError("'cms' ({}) is not a tuple, list or"
                        " numpy.ndarray instance".format(type(cms)))
    try:
        cm_shape = cms[0].shape
        cm_dim = cms[0].ndim
    except (IndexError, AttributeError, TypeError):
        raise TypeError("'cms' is not a sequence of numpy.ndarrays")
    if shape is not None and cm_shape != shape:
        raise ValueError("The first array in 'cms' has shape {} but must"
                         " have shape {}".format(cm_shape, shape))
    if dim is not None and cm_dim != dim:
        raise ValueError("The first array in 'cms' has dimension {} but"
                         " must have dimension {}".format(cm_dim, dim))
    cm_type = type(cms[0])
    cm_size = np.prod(cm_shape)
    check_amin = (amin is not None and cm_size > 0)
    check_amax = (amax is not None and cm_size > 0)
    for i, cm in enumerate(cms):
        if cm.shape != cm_shape:
            raise ValueError("'cms[{}]' has a different shape ({}) than"
                             " the previous arrays in 'cms' ({})"
                             .format(i, cm.shape, cm_shape))
        if check_amin and cm.min() < amin:
            raise ValueError("The minimum value of 'cms[{}]' ({}) is"
                             " less than 'amin' ({})"
                             .format(i, cm.min(), amin))
        if check_amax and cm.max() > amax:
            raise ValueError("The maximum value of 'cms[{}]' ({}) is"
                             " greater than 'amax' ({})"
                             .format(i, cm.max(), amax))
        if dtype is not None and cm.dtype != dtype:
            raise TypeError("The data type of 'cms[{}]' is {} but must"
                            " be {}".format(i, cm.dtype, dtype))
        if not isinstance(cm, cm_type):
            raise TypeError("'cms[{}]' is of another instance ({}) than"
                            " the previous arrays in 'cms' ({})"
                            .format(i, type(cm), cm_type))
    return cms


def bins(  # noqa: C901
    start,
    stop,
    step=None,
    num=None,
    amin=0,
    amax=None,
    precision=9,
    verbose=True,
):
    """
    Check if start point, end point and step width or number of bins are
    chosen properly for creating bin edges.

    `start`, `stop`, `step` and `num` must meet the following
    requirements:

        * `start` must not be less than `amin` (if supplied).
        * `stop` must be greater than `start`, but not greater than
          `amax` (if supplied).
        * `step` must be greater than zero, but not greater than
          ``stop - start``.
        * `num` must be an integer and greater than zero.

    Parameters
    ----------
    start : scalar
        Start of interval.  The interval includes this value.
    stop : scalar
        End of interval.  The interval includes(!) this value.
    step : scalar, optional
        Step width.
    num : int, optional
        Number of steps between start and stop.  Note that either `step`
        or `num` or both of them must be given.
    amin : scalar, optional
        A minimum value that start must not undermine.
    amax : scalar, optional
        A maximum value that stop must not exceed.  If given, `amax`
        must be greater than `amin`.
    precision : int, optional
        Precision to use for floating point arithmetics.  Default is 9
        decimal places.
    verbose : bool, optional
        If ``True`` (default), any changes of the input parameters are
        printed to standard output.

    Returns
    -------
    start : float
        The input or a corrected start point.
    stop : float
        The input or a corrected end point.
    step : float
        The input or a corrected step width.
    num : int
        The input or a corrected number of bins.  Note: If both, `step`
        and `num`, are given, `step` is tried to be kept fixed while
        `num` is changed according to ``stop - start == num * step``.

    Raises
    ------
    ValueError
        If a corrected value cannot be inferred from the given input.

    See Also
    --------
    :func:`mdtools.check.distance_bins` :
        Check if the input is appropriate for binning distances that
        obey the minimum image convention
    :func:`mdtools.check.bin_edges` :
        Check if bin edges are chosen properly for binning a given
        interval

    Notes
    -----
    To check whether the binning also fulfills the minimum image
    convention, use :func:`mdtools.check.distance_bins`.

    If you want to ensure that **all** bins are equidistant, do not
    parse `step` but only `num`.  If you parse `step`, this function
    tries to keep the input `step` fixed, even if this requires that the
    **last** bin is smaller than all other bins.
    """
    # Check input parameters:
    if step is None and num is None:
        raise ValueError(
            "Either `step` ({}) or `num` ({}) or both of them must be"
            " given".format(step, num)
        )
    if amin is not None and amax is not None and amax <= amin:
        raise ValueError(
            "`amax` ({}) must be greater than `amin` ({})".format(amax, amin)
        )
    # Setting precision for floating point comparison:
    digits = int(max(len(str(int(start))), len(str(int(stop)))))
    if step is not None:
        digits = int(max(digits, len(str(int(step)))))
    if num is not None:
        digits = int(max(digits, len(str(int(num)))))
    if amin is not None:
        digits = int(max(digits, len(str(int(amin)))))
    if amax is not None:
        digits = int(max(digits, len(str(int(amax)))))
    if precision > 0:
        dec.getcontext().prec = digits + precision
    else:
        dec.getcontext().prec = digits
    dec.getcontext().rounding = dec.ROUND_DOWN
    start = dec.Decimal(str(start))
    stop = dec.Decimal(str(stop))
    if step is not None:
        step = dec.Decimal(str(step))
    if num is not None:
        num = dec.Decimal(str(num))
    if amin is not None:
        amin = dec.Decimal(str(amin))
    if amax is not None:
        amax = dec.Decimal(str(amax))
    # Check bins:
    if num is not None and int(num) != num:
        num = int(num)
        if verbose:
            print("`mdtools.check.bins()` set `num` to {}".format(num))
    if amin is not None and start < amin:
        start = amin
        if verbose:
            print("`mdtools.check.bins()` set `start` to {}".format(start))
    if amax is not None and stop > amax:
        stop = amax
        if verbose:
            print("`mdtools.check.bins()` set `stop` to {}".format(stop))
    if stop <= start:
        if amax is not None and step is not None:
            if start + step <= amax:
                stop = start + step
            else:
                stop = amax
        elif amax is not None and step is None:
            stop = amax
        elif amax is None and step is not None:
            stop = start + step
        elif amax is None and step is None:
            raise ValueError(
                "`stop` ({}) is equal to or less than `start`"
                " ({})".format(stop, start)
            )
        if verbose:
            print("`mdtools.check.bins()` set `stop` to {}".format(stop))
    if step is not None and (step > stop - start or step <= 0):
        if num is not None:
            step = (stop - start) / num
        else:
            step = stop - start
        if verbose:
            print("`mdtools.check.bins()` set `step` to {}".format(step))
    elif step is None:
        step = (stop - start) / num
        if verbose:
            print("`mdtools.check.bins()` set `step` to {}".format(step))
    if num is None or num != (stop - start) / step:
        num = round((stop - start) / step)
        if verbose:
            print("`mdtools.check.bins()` set `num` to {}".format(num))
    return float(start), float(stop), float(step), int(num)


def distance_bins(
        box, start, stop, step=None, num=None, precision=9,
        verbose=True):
    """
    Check if start point, end point and step width or number of bins are
    chosen properly for binning distances that obey the minimum image
    convention.

    `start`, `stop`, `step` and `num` must meet the following
    requirements:

        * `start` must be equal to or greater than zero.
        * `stop` must be greater than `start`, but less than half the
          box diagonal.
        * `step` must be greater than zero, but not greater than
          ``stop - start``.
        * `num` must be an integer and greater than zero.

    .. todo::

        This function is not implemented, yet!

    Parameters
    ----------
    box : array_like, optional
        TODO: Docstring
    start : scalar
        Start of interval.  The interval includes this value.
    stop : scalar
        End of interval.  The interval includes(!) this value.
    step : scalar, optional
        Step width.
    num : int, optional
        Number of steps between start and stop.  Note that either `step`
        or `num` or both of them must be given.
    precision : int, optional
        Precision to use for floating point arithmetics.  Default is 9
        decimal places.
    verbose : bool, optional
        If ``True`` (default), any changes of the input parameters are
        printed to standard output.

    Returns
    -------
    start : float
        The input or a corrected start point.
    stop : float
        The input or a corrected end point.
    step : float
        The input or a corrected step width.
    num : int
        The input or a corrected number of bins.  Note: If both, `step`
        and `num`, are given, `step` is tried to be kept fixed while
        `num` is changed according to ``stop - start == num * step``.

    Raises
    ------
    ValueError
        If a corrected value cannot be inferred from the given input.

    Warnings
    --------
    Works currently only for orthogonal simulation boxes.

    See Also
    --------
    :func:`mdtools.check.bins` :
        Check if the input is appropriate for creating bin edges
    :func:`mdtools.check.bin_edges` :
        Check if bin edges are chosen properly for binning a given
        interval

    Notes
    -----
    To check binning that does not need to fulfill the minimum image
    convention, use :func:`mdtools.check.bins`.

    If you want to ensure that **all** bins are equidistant, do not
    parse `step` but only `num`.  If you parse `step`, this function
    tries to keep the input `step` fixed, even if this implies that the
    **last** bin is smaller than all other bins.
    """
    raise NotImplementedError("This function is not implemented, yet")


def bin_edges(bins, amin=0, amax=1, right=False, tol=1e-6, verbose=True):
    """
    Check if bin edges are chosen properly for binning a given interval.

    The bin edges must meet the following requirements:

        * There must be at least one bin edge.
        * The first bin edge must not be less than `amin`.
        * The last bin edge must not be greater than `amax`.

    Parameters
    ----------
    bins : array_like
        Array containing the bin edges.  The bin edges are sorted in
        ascending order and duplicates are removed.
    amin : scalar, optional
        A minimum value that the first bin edge must not undermine.
    amax : scalar, optional
        A maximum value that the last bin edge must not exceed.
    right : bool, optional
        Indicating whether the bin intervals include the right or the
        left bin edge.  If ``True``, the bin intervals include the right
        bin edge, i.e. the bin intervals are left-open and right-closed:
        (a, b] -> a < x <= b.  If ``False`` (default), the bin intervals
        include the left bin edge, i.e. the bin intervals are
        left-closed and right-open: [a, b) -> a <= x < b.
    tol : scalar, optional
        A tolerance value subtracted from `amin` or added to `amax`.  If
        `right` is ``True``, the first bin edge will be set to
        ``amin - tol`` to account for the left-open bin interval.  If
        `right` is ``False``, the last bin edge will be set to
        ``amax + tol`` to account for the right-open bin interval.  In
        this way, values of `amin` or `amax` will be properly binned
        into the first or last bin, respectively, when using
        :func:`numpy.digitize`.  Set `tol` to zero if you do not want to
        account for the first/last half-open bin interval.
    verbose : bool, optional
        If ``True`` (default), any changes of the bin edges are printed
        to standard output.

    Returns
    -------
    bins : numpy.ndarray
        Unique and sorted copy of the input bin edges.  If `bins` did
        not already contain `amin` and/or `amax` these values are
        inserted with `tol` subtracted or added according to the value
        of `right`.  If `bins` already contained `amin` and/or `amax`,
        `tol` will be subtracted or added according to the value of
        `right`.

    Raises
    ------
    ValueError
        If ``len(bins)`` is zero;
        ``bins[0]`` is less than `amin` or ``amin - tol`` (if `right` is
        ``False`` or ``True``);
        ``bins[-1]`` is greater than ``amax + tol`` or `amax` (if
        `right` is ``False`` or ``True``).

    See Also
    --------
    :func:`mdtools.check.bins` :
        Check if the input is appropriate for creating bin edges
    :func:`mdtools.check.distance_bins` :
        Check if the input is appropriate for binning distances that
        obey the minimum image convention
    :func:`numpy.digitize` :
        Return the indices of the bins to which each value in the input
        array belongs

    Notes
    -----
    The bin edges returned by this function can for instance be used to
    discretize the positions of compounds that are wrapped into the
    primary unit cell with :func:`numpy.digitize`.
    """
    bins = np.unique(bins)
    if len(bins) == 0:
        raise ValueError("The number of bin edges is zero")
    if amax <= amin:
        raise ValueError(
            "`amax` ({}) must be greater than `amin` ({})".format(amax, amin)
        )
    if right:
        first_bin_edge = amin - tol
        last_bin_edge = amax
    else:
        first_bin_edge = amin
        last_bin_edge = amax + tol
    if np.isclose(bins[0], amin, rtol=0, atol=tol):
        bins[0] = first_bin_edge
        if verbose:
            print(
                "`mdtools.check.bin_edges()` set `bins[0]` to"
                " {}".format(bins[0])
            )
    elif bins[0] > amin:
        bins = np.insert(bins, 0, first_bin_edge)
        if verbose:
            print(
                "`mdtools.check.bin_edges()` prepended `bins` with"
                " {}".format(bins[0])
            )
    elif bins[0] < amin:
        raise ValueError(
            "The first bin edge ({}) must not be less than `amin - tol`"
            " ({})".format(bins[0], amin - tol)
        )
    if np.isclose(bins[-1], amax, rtol=0, atol=tol):
        bins[-1] = last_bin_edge
        if verbose:
            print(
                "`mdtools.check.bin_edges()` set `bins[-1]` to"
                " {}".format(bins[-1])
            )
    elif bins[-1] < amax:
        bins = np.append(bins, last_bin_edge)
        if verbose:
            print(
                "`mdtools.check.bin_edges()` appended `bins` with"
                " {}".format(bins[-1])
            )
    elif bins[-1] > amax:
        raise ValueError(
            "The last bin edge ({}) must not be greater than `amax + tol`"
            " ({})".format(bins[-1], amax + tol))
    return bins


def frame_slicing(start, stop, step, n_frames_tot=None, verbose=True):
    """
    Check if the input parameters are suitable for slicing |MDA_trjs|.

    Basically, the same rules as for `slicing numpy arrays`_ apply with
    the following limitations:

        * `start` must be positive or zero, but smaller than `stop`.
        * `stop` must be positive and greater than `start` but not
          greater than `n_frames_tot` (if supplied).
        * `step` must be positive but not greater than ``stop - start``.
        * `n_frames_tot` must be positive (if supplied).

    In fact, the limitations for `slicing MDAnalysis trajectories`_ are
    not that strict, but to ensure consistent user input for MDTools
    scripts, we have set up the above limitations.

    .. _slicing numpy arrays:
        https://numpy.org/doc/stable/reference/arrays.indexing.html
    .. _slicing MDAnalysis trajectories:
        https://userguide.mdanalysis.org/stable/trajectories/slicing_trajectories.html

    Parameters
    ----------
    start : int
        Start of the interval.  The interval includes this value.  Note
        that frame numbering starts at zero.
    stop : int
        End of the interval.  The interval excludes(!) this value.
    step : int
        Step width.
    n_frames_tot : int, optional
        A total value that `stop` must not exceed (usually the length of
        the :attr:`~MDAnalysis.core.universe.Universe.trajectory`).
        Must be positive.
    verbose : bool, optional
        If ``True`` (default), any changes of the input parameters are
        printed to standard output.

    Returns
    -------
    start : int
        The input or a corrected start point.
    stop : int
        The input or a corrected end point.
    step : int
        The input or a corrected step width.
    n_frames : int
        The number of selected frames.

    See Also
    --------
    :func:`mdtools.check.block_averaging` :
        Check if the number of blocks for block averaging is chosen
        properly
    :func:`mdtools.check.frame_lag` :
        Check if a frame lag ('lag time') is chosen properly
    :func:`mdtools.check.time_step` :
        Check whether all frames in a |MDA_trj| have the same
        :attr:`time step <MDAnalysis.coordinates.base.Timestep.dt>`
    :meth:`MDAnalysis.coordinates.base.ReaderBase.check_slice_indices` :
        Check if frame indices are valid to slice a |MDA_trj|
    :class:`MDAnalysis.coordinates.base.FrameIteratorSliced` :
        Iterable over the frames of a trajectory on the basis of a slice
    """
    if start < 0:
        start = 0
        if verbose:
            print("'mdtools.check.frame_slicing()' set 'start' to {}"
                  .format(start))
    if step < 1:
        step = 1
        if verbose:
            print("'mdtools.check.frame_slicing()' set 'step' to {}"
                  .format(step))
    if n_frames_tot is not None:
        if n_frames_tot < 1:
            raise ValueError("'n_frames_tot' ({}) is less than 1"
                             .format(n_frames_tot))
        if stop <= 0 or stop > n_frames_tot:
            stop = n_frames_tot
            if verbose:
                print("'mdtools.check.frame_slicing()' set 'stop' to {}"
                      .format(stop))
    if stop <= 0:
        stop = start + step
        if verbose:
            print("'mdtools.check.frame_slicing()' set 'stop' to {}"
                  .format(stop))
    elif start >= stop and stop - step >= 0:
        start = stop - step
        if verbose:
            print("'mdtools.check.frame_slicing()' set 'start' to {}"
                  .format(start))
    elif start >= stop and stop - step < 0:
        start = 0
        step = stop - start
        if verbose:
            print("'mdtools.check.frame_slicing()' set 'start' to {}"
                  .format(start))
            print("'mdtools.check.frame_slicing()' set 'step' to {}"
                  .format(step))
    elif step > stop - start:
        step = stop - start
        if verbose:
            print("'mdtools.check.frame_slicing()' set 'step' to {}"
                  .format(step))
    elif (stop - start) % step > 1:
        stop -= (stop - start) % step - 1
        if verbose:
            print("'mdtools.check.frame_slicing()' set 'stop' to {}"
                  .format(stop))
    n_frames = int(np.ceil((stop - start) / step))
    return start, stop, step, n_frames


def block_averaging(n_blocks, n_frames, check_CPUs=False, verbose=True):
    """
    Check if the number of blocks for block averaging is chosen
    properly.

    The number of blocks must be greater than zero, but less than the
    number of available frames.

    Parameters
    ----------
    n_blocks : int
        Number of blocks for block averaging.
    n_frames : int
        The number of frames available for analysis, i.e. the number of
        frames that are/were actually read, **not** the total number of
        frames in the trajectory.
    check_CPUs : bool, optional
        If ``True``, check if the number of available CPUs is a multiple
        of the number of blocks (or the other way round).  If this is
        not the case, a warning is raised.
    verbose : bool, optional
        If ``True`` (default), any changes to `n_blocks` are printed to
        standard output.

    Returns
    -------
    n_blocks : int
        The input or a corrected number of blocks.
    block_size : int
        The number of frames available for analysis per block.

    Warns
    -----
    RuntimeWarning
        If `check_CPUs` is ``True`` and the number of available CPUs is
        not an integer multiple of `n_blocks` (or the other way round).

    See Also
    --------
    :func:`mdtools.check.frame_slicing` :
        Check if the input parameters are suitable for slicing
        |MDA_trjs|
    :func:`mdtools.check.frame_lag` :
        Check if a frame lag ('lag time') is chosen properly
    :func:`mdtools.check.time_step` :
        Check whether all frames in a |MDA_trj| have the same
        :attr:`time step <MDAnalysis.coordinates.base.Timestep.dt>`
    """
    # Check input parameters:
    if n_frames < 1:
        raise ValueError("'n_frames' ({}) must be greater than zero"
                         .format(n_frames))
    # Check blocks:
    if n_blocks < 1:
        n_blocks = 1
        if verbose:
            print("'mdtools.check.block_averaging()' set 'n_blocks' to"
                  " {}".format(n_blocks))
    elif n_blocks > n_frames:
        n_blocks = n_frames
        if verbose:
            print("'mdtools.check.block_averaging()' set 'n_blocks' to"
                  " {}".format(n_blocks))
    if verbose and n_frames % n_blocks != 0:
        print("Note: The number of blocks for block averaging ({}) is\n"
              "  not a multiple of the number of frames available for\n"
              "  analysis ({}). This means that {} frame(s) will not\n"
              "  used."
              .format(n_blocks, n_frames, n_frames % n_blocks))
    if check_CPUs:
        num_CPUs = mdt.rti.get_num_CPUs()
        if ((num_CPUs < n_blocks and n_blocks % num_CPUs != 0) or
                (num_CPUs > n_blocks and num_CPUs % n_blocks != 0)):
            warnings.warn(
                "The number of available CPUs ({}) is not a multiple of"
                " the number of blocks for block averaging ({}). This"
                " will probably lead to performance degradation."
                .format(num_CPUs, n_blocks))
    block_size = int(n_frames / n_blocks)
    return n_blocks, block_size


def restarts(
        restart_every_nth_frame, read_every_nth_frame, n_frames,
        verbose=True):
    """
    Check if the number of frames between restarting points is chosen
    properly.

    .. deprecated:: 0.0.dev0
        :func:`mdtools.check.restarts` might be removed in a future
        version due to duplicate functionality.  Use
        :func:`mdtools.check.frame_lag` instead.

    Different restarting points are usually used when calculating time
    averaged quantities (like mean square displacements or
    autocorrelation functions)

    Parameters
    ----------
    restart_every_nth_frame : int
        Number of frames between restarting points.
    read_every_nth_frame : int
        Intervall for reading the trajectory, i.e. only
        every `read_every_nth_frame` frame of the trajectory is/was
        actually read.
    n_frames : int
        The number of frames available for analysis, i.e. the number of
        frames that are/were actually read, **not** the total number of
        frames in the trajectory.  If the trajectory is divided into
        blocks, this is the number of available frames per block.
    verbose : bool, optional
        If ``True`` (default), any changes of the input parameters are
        printed to standard output.

    Returns
    -------
    restart_every_nth_frame : int
        The input or a corrected value.
    effective_restart : int
        The effective number of frames between restarting points.  If
        e.g. only every tenth frame of the trajectory is read and you
        want restarts every fifty frames, you effectively restart every
        fifths frame of the read frames.

    See Also
    --------
    :func:`mdtools.check.frame_lag` :
        Check if a frame lag ('lag time') is chosen properly
    :func:`mdtools.check.frame_slicing` :
        Check if the input parameters are suitable for slicing
        |MDA_trjs|
    :func:`mdtools.check.block_averaging` :
        Check if the number of blocks for block averaging is chosen
        properly
    :func:`mdtools.check.time_step` :
        Check whether all frames in a |MDA_trj| have the same
        :attr:`time step <MDAnalysis.coordinates.base.Timestep.dt>`
    """
    # Check input parameters:
    if n_frames < 1:
        raise ValueError("'n_frames' ({}) must be greater than zero"
                         .format(n_frames))
    if read_every_nth_frame < 1:
        raise ValueError("'read_every_nth_frame' ({}) must be greater"
                         " than zero".format(n_frames))
    # Check restarts:
    if restart_every_nth_frame > n_frames * read_every_nth_frame:
        restart_every_nth_frame = n_frames * read_every_nth_frame
        if verbose:
            print("'mdtools.check.restarts()' set"
                  " 'restart_every_nth_frame' to {}"
                  .format(restart_every_nth_frame))
    elif restart_every_nth_frame < read_every_nth_frame:
        restart_every_nth_frame = read_every_nth_frame
        if verbose:
            print("'mdtools.check.restarts()' set"
                  " 'restart_every_nth_frame' to {}"
                  .format(restart_every_nth_frame))
    elif restart_every_nth_frame % read_every_nth_frame != 0:
        restart_every_nth_frame -= (restart_every_nth_frame %
                                    read_every_nth_frame)
        if verbose:
            print("'mdtools.check.restarts()' set"
                  " 'restart_every_nth_frame' to {}"
                  .format(restart_every_nth_frame))
    effective_restart = int(restart_every_nth_frame / read_every_nth_frame)
    return restart_every_nth_frame, effective_restart


def frame_lag(
        lag, every, n_frames_tot=None, allow_zero=False, verbose=True):
    """
    Check if a frame lag ('lag time') is chosen properly.

    Check if a frame lag ('lag time') is chosen properly with respect to
    the total number of frames in the trajectory and the interval
    between frames that are/were actually read.

    The frame lag must meet the following requirements:

        * Must be greater than zero (if `allow_zero` is ``True``, zero
          is also possible)
        * Must not be greater than `n_frames_tot` (if supplied).
        * Must be an integer multiple of `every`.

    Parameters
    ----------
    lag : int
        The frame lag, e.g. the number of frames between restarting
        points.
    every : int
        Intervall used while reading the trajectory, i.e. only every
        n-th frame of the trajectory is/was actually read.
    n_frames_tot : int, optional
        The total number of frames in the trajectory.
    allow_zero : bool, optional
        If ``True``, `lag` can also be zero.
    verbose : bool, optional
        If ``True`` (default), any changes to `lag` are printed to
        standard output.

    Returns
    -------
    lag : int
        The input or a corrected value.
    effective_lag : int
        An effective frame lag.  If e.g. only every tenth frame of the
        trajectory is read and you want a frame lag of fifty, the
        effective frame lag is five.

    See Also
    --------
    :func:`mdtools.check.frame_slicing` :
        Check if the input parameters are suitable for slicing
        |MDA_trjs|
    :func:`mdtools.check.block_averaging` :
        Check if the number of blocks for block averaging is chosen
        properly
    :func:`mdtools.check.time_step` :
        Check whether all frames in a |MDA_trj| have the same
        :attr:`time step <MDAnalysis.coordinates.base.Timestep.dt>`
    """
    # Check input parameters:
    if n_frames_tot < 1:
        raise ValueError("'n_frames_tot' ({}) must be greater than zero"
                         .format(n_frames_tot))
    if every < 1:
        raise ValueError("'every' ({}) must be greater than zero"
                         .format(every))
    if every > n_frames_tot:
        raise ValueError("'every' ({}) must not be greater than"
                         " 'n_frames_tot' ({})"
                         .format(every, n_frames_tot))

    # Check frame lag:
    if allow_zero and lag == 0:
        return lag, 0
    if n_frames_tot is not None and lag > n_frames_tot:
        lag = n_frames_tot
        if verbose:
            print("'mdtools.check.frame_lag()' set 'lag' to {}"
                  .format(lag))
    if lag < every:
        lag = every
        if verbose:
            print("'mdtools.check.frame_lag()' set 'lag' to {}"
                  .format(lag))
    elif lag % every != 0:
        lag -= (lag % every)
        if verbose:
            print("'mdtools.check.frame_lag()' set 'lag' to {}"
                  .format(lag))
    effective_lag = lag // every
    if lag != int(lag):
        raise ValueError("'lag' ist not an integer")
    if effective_lag != int(effective_lag):
        raise ValueError("'effective_lag' ist not an integer")

    return int(lag), int(effective_lag)


def time_step(trj, verbose=True):
    """
    Check whether all frames in a |MDA_trj| have the same
    :attr:`time step <MDAnalysis.coordinates.base.Timestep.dt>`.

    Parameters
    ----------
    trj : MDAnalysis.coordinates.base.ReaderBase or \
        MDAnalysis.coordinates.base.FrameIteratorBase
        The |MDA_trj| to check.
    verbose : bool, optional
        If ``True``, print progress information to standard output.

    Raises
    ------
    ValueError
        If not all frames have the same
        :attr:`time step <MDAnalysis.coordinates.base.Timestep.dt>`.
    """
    init_dt = trj[0].dt
    if verbose:
        print("Checking time step equality...")
        timer = datetime.now()
        trj = mdt.rti.ProgressBar(trj)
    time_steps = np.asarray([ts.dt for ts in trj])
    if not np.all(time_steps == init_dt):
        raise ValueError("The time steps in the trajectory vary")
    if verbose:
        trj.close()
        print("All frames in 'trj' have the same time step")
        print("Elapsed time:         {}".format(datetime.now() - timer))
        print("Current memory usage: {:.2f} MiB"
              .format(mdt.rti.mem_usage(unit='MiB')))


def masses(ag, flash_test=True):
    """
    Check atom masses.

    .. deprecated:: 0.0.0.dev0
        :func:`mdtools.check.masses` will be replaced by
        :func:`mdtools.check.masses_new` in a future release.

    `MDAnalysis always guesses atom masses`_ from the atom type, even if
    the input file contains the masses.  This might result in wrong mass
    assignments.  If the mass cannot be guessed at all, a mass of 0 is
    assigned.  Hence, it is important to check the correct assignment of
    masses before calculating mass dependent quantities like the center
    of mass.

    .. _MDAnalysis always guesses atom masses:
        https://userguide.mdanalysis.org/formats/guessing.html

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` whose
        masses shall be checked.
    flash_test : bool, optional
        If ``True`` (default), only check if any atom type in `ag` was
        assigned a zero mass.  If ``False``, print a list of all
        different atom types in `ag` with their corresponding masses
        guessed by MDAnalysis so that the user can check whether the
        masses were guessed correctly.

    Raises
    ------
    ValueError
        If one or more atoms of `ag` have zero mass.

    Note
    ----
    This function only checks if any atom type was assigned a zero mass.
    This functions checks *not* whether non-zero masses were assigned
    correctly.  This can be done manually by the user, when `flash_test`
    is set to ``False``.
    """
    if not flash_test:
        n_cols = 72
        print(n_cols * "!")
        print("You are going to calculate mass dependent quantities.\n"
              "Since MDAnalysis always only guesses the mass from the\n"
              "atom type instead of reading it from the input files,\n"
              "take a while and carefully check the mass assignment\n"
              "for correctness.\n")
        _, ix = np.unique(ag.types, return_index=True)
        ix.sort()
        types = ag.types[ix]
        masses = ag.masses[ix]
        print("  Type  Mass")
        for i in range(len(types)):
            print("  {:<4s}  {:>9.6f}".format(types[i], masses[i]))
        print(n_cols * "!")
    if np.any(ag.masses == 0):
        raise ValueError("One or more atoms of ag have zero mass")


def masses_new(ag, verbose=False):
    """
    Check :attr:`atom masses <MDAnalysis.core.groups.AtomGroup.masses>`.

    .. note::

        :func:`mdtools.check.masses_new` will replace
        :func:`mdtools.check.masses` in a future release.  When doing
        so, :func:`masses_new` will be renamed to
        :func:`mdtools.check.masses` (hence lines containing
        :func:`mdtools.check.masses_new` can have 76 characters instead
        of 72).

    `MDAnalysis always guesses atom masses`_ from the atom types, even
    if the input file contains the masses.  This might result in wrong
    mass assignments.  If the mass cannot be guessed at all, a mass of 0
    is assigned.  Hence, it is important to check the correct assignment
    of masses before calculating mass dependent quantities like the
    center of mass.

    .. _MDAnalysis always guesses atom masses:
        https://userguide.mdanalysis.org/formats/guessing.html

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` whose
        masses shall be checked.
    verbose : bool, optional
        If ``True``, print a list of all different atom types in `ag`
        with their corresponding masses as guessed by MDAnalysis to
        standard output so that the user can check whether the masses
        were guessed correctly.

    Raises
    ------
    ValueError
        If at least one :class:`~MDAnalysis.core.groups.Atom` of `ag`
        has a mass equal to or less than zero.

    Note
    ----
    This function only checks if any atom type was assigned a mass less
    than or equal to zero.  This functions checks **not** whether
    positive masses were assigned correctly.  This can be done manually
    by the user, when `verbose` is ``True``.
    """
    if verbose:
        N_COLS = 72
        print(N_COLS * "!")
        print("! Please check the mass assignment for correctness:"
              " {:>20s}".format("!"))
        _, ix = np.unique(ag.types, return_index=True)
        ix.sort()
        types = ag.types[ix]
        masses = ag.masses[ix]
        print("! {:<4s}  {:>9s} {:>54s}".format("Type", "Mass", "!"))
        for i, typ in enumerate(types):
            print("! {:<4s}  {:>9.6f} {:>54s}"
                  .format(typ, masses[i], "!"))
        print(N_COLS * "!")
    if np.any(ag.masses <= 0):
        raise ValueError("At least one atom of 'ag' has a mass equal to"
                         " or less than zero")
