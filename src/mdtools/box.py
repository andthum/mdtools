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
import warnings
from datetime import datetime

# Third-party libraries
import MDAnalysis as mda
import MDAnalysis.lib.distances as mdadist
import MDAnalysis.lib.mdamath as mdamath
import numpy as np
import psutil

# First-party libraries
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
    :func:`mdtools.box.volume` :
        Calculate the volume of (multiple) orthogonal box(es)
    :func:`MDAnalysis.lib.mdamath.box_volume` :
        Calculate the volume of a single (triclinic) box

    Notes
    -----
    This function is just a wrapper around
    :func:`MDAnalysis.lib.mdamath.box_volume` that raises an exception
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

    kwargs : dict, optional
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
    :func:`mdtools.box.box_volume` :
       Calculate the volume of a single (triclinic) box
    :func:`MDAnalysis.lib.mdamath.box_volume` :
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


def triclinic_box(box_mat, dtype=None):
    r"""
    Convert the matrix representation of a simulation box to the
    length-angle representation.

    Convert the matrix representation
    ``[[lx1, lx2, lx3], [[lz1, lz2, lz3]], [[lz1, lz2, lz3]]]`` (as
    returned by
    :attr:`MDAnalysis.coordinates.base.Timestep.triclinic_dimensions`)
    to the length-angle representation
    ``[lx, ly, lz, alpha, beta, gamma]`` (as returned by
    :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`).

    Parameters
    ----------
    box_mat : array_like
        The unit cell dimensions of the system, which can be orthogonal
        or triclinic and must be provided in the same format as returned
        by
        :attr:`MDAnalysis.coordinates.base.Timestep.triclinic_dimensions`:
        ``[[lx1, lx2, lx3], [[lz1, lz2, lz3]], [[lz1, lz2, lz3]]]``.
        `box_mat` can also be an array of box matrices with an arbitrary
        number of dimensions as long as the last two dimensions have
        shape ``(3, 3)``.  Note that each box matrix must contain the
        box vectors as rows!
    dtype : type, optional
        The data type of the output array.  If ``None``, the data type
        is inferred from the input array.

    Returns
    -------
    box : numpy.ndarray of dtype numpy.float32
        The unit cell dimensions of the system in the same format as
        returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.  `box` will have one
        dimension less than `box_mat` and its last dimension will have
        shape ``(6,)``.

        .. list-table:: Shapes
            :align: left
            :header-rows: 1

            *   - `box_mat`
                - `box`
            *   - ``(3, 3)``
                - ``(6,)``
            *   - ``(k, 3, 3)``
                - ``(k, 6)``
            *   - ``(j, k, 3, 3)``
                - ``(j, k, 6)``
            *   - ``(i, j, k, 3, 3)``
                - ``(i, j, k, 6)``
            *   - ...
                - ...

    See Also
    --------
    :func:`mdtools.box.triclinic_vectors` :
        Inverse function: Convert the length-angle representation of a
        simulation box to the matrix representation.
    :func:`MDAnalysis.lib.mdamath.triclinic_box` :
        Similar function: Convert the matrix representation of a
        simulation box to the length-angle representation.

    Notes
    -----
    The box vectors are converted to their lengths and angles using the
    following formulas:

    .. math::

        l_x = |\mathbf{l_x}| = \sqrt{\sum_{i=1}^3 l_{x,i}^2}

    .. math::

        \alpha = \arccos{\frac{\mathbf{l_y}\mathbf{l_z}}{l_y l_z}}

    .. math::

        \beta = \arccos{\frac{\mathbf{l_z}\mathbf{l_x}}{l_z l_x}}

    .. math::

        \gamma = \arccos{\frac{\mathbf{l_x}\mathbf{l_y}}{l_x l_y}}

    The angles are returned in degrees and are ensured to lie within the
    interval :math:`]0, 180[`.

    If one of the resulting length-angle representations is invalid, a
    warning will be raised and a zero vector will be returned for all
    boxes.

    Examples
    --------
    >>> box_mat = np.eye(3) * 2
    >>> mdt.box.triclinic_box(box_mat)
    array([ 2.,  2.,  2., 90., 90., 90.])
    >>> box_mat = [box_mat, np.eye(3) * 3]
    >>> mdt.box.triclinic_box(box_mat)
    array([[ 2.,  2.,  2., 90., 90., 90.],
           [ 3.,  3.,  3., 90., 90., 90.]])
    >>> box_mat = [box_mat, [np.eye(3) * 4, np.eye(3) * 5]]
    >>> mdt.box.triclinic_box(box_mat)
    array([[[ 2.,  2.,  2., 90., 90., 90.],
            [ 3.,  3.,  3., 90., 90., 90.]],
    <BLANKLINE>
           [[ 4.,  4.,  4., 90., 90., 90.],
            [ 5.,  5.,  5., 90., 90., 90.]]])

    >>> import MDAnalysis.lib.mdamath as mdamath
    >>> box_mat = np.eye(3) * 2
    >>> box1 = mdt.box.triclinic_box(box_mat)
    >>> box1
    array([ 2.,  2.,  2., 90., 90., 90.])
    >>> box2 = mdamath.triclinic_box(*box_mat)
    >>> box2
    array([ 2.,  2.,  2., 90., 90., 90.], dtype=float32)
    >>> np.allclose(box1, box2)
    True
    >>> box_mat = np.arange(9).reshape((3, 3))
    >>> box1 = mdt.box.triclinic_box(box_mat)
    >>> box1
    array([ 2.23606798,  7.07106781, 12.20655562,  4.88390747, 32.57846892,
           27.69456145])
    >>> box2 = mdamath.triclinic_box(*box_mat)
    >>> box2
    array([ 2.236068 ,  7.071068 , 12.206555 ,  4.8839073, 32.57847  ,
           27.694561 ], dtype=float32)
    >>> np.allclose(box1, box2)
    True
    >>> box_mat = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
    >>> box1 = mdt.box.triclinic_box(box_mat)
    >>> box1
    array([ 1.        ,  3.60555128,  8.77496439, 43.36781871, 62.88085723,
           56.30993247])
    >>> box2 = mdamath.triclinic_box(*box_mat)
    >>> box2
    array([ 1.       ,  3.6055512,  8.774964 , 43.367817 , 62.880856 ,
           56.309933 ], dtype=float32)
    >>> np.allclose(box1, box2)
    True
    """  # noqa: W505
    box_mat = np.asarray(box_mat, dtype=dtype)
    # Actually, this function works with any input arrays that have at
    # least 2 dimensions.  This if statement only checks for physically
    # meaningful inputs and could be replaced by ``if box_mat.ndim < 2``
    # without breaking functionality.
    if box_mat.shape[-2:] != (3, 3):
        raise ValueError(
            "`box_mat` has shape {}, but the last two dimensions of `box_mat`"
            " must have shape (3, 3)".format(box_mat.shape)
        )

    # Box lengths: lx, ly, lz.
    box_lengths = np.linalg.norm(box_mat, axis=-1)
    # Products of box lengths: lx*ly, ly*lz, lz*lx.
    shift = -1
    box_lengths_prods = box_lengths * np.roll(box_lengths, shift, axis=-1)
    # Dot products of box vectors: lx*ly, ly*lz, lz*lx.
    dot_prods = box_mat * np.roll(box_mat, shift, axis=-2)
    dot_prods = np.sum(dot_prods, axis=-1)
    # Box angles: gamma, alpha, beta.
    angles = dot_prods / box_lengths_prods
    angles = np.arccos(angles, out=angles)
    angles = np.rad2deg(angles, out=angles)
    # Box angles: alpha, beta, gamma.
    angles = np.roll(angles, shift, axis=-1)
    # Box in length-angle representation: lx, ly, lz, alpha, beta, gamma
    box = np.concatenate([box_lengths, angles], axis=-1)

    if box.shape[-1] != np.sum(box_mat.shape[-2:]):
        raise ValueError(
            "`box` has shape {}, but the last dimension of `box` must have"
            " shape ({},).  This should not have"
            " happened.".format(box.shape, np.sum(box_mat.shape[-2:]))
        )
    # Only positive box lengths and angles in ]0, 180[ are allowed:
    if np.all(box > 0) and np.all(mdt.nph.take(box, start=3, axis=-1) < 180):
        return box
    else:
        # invalid box, return zero vector.
        warnings.warn(
            "Invalid box.  At least one box length is not greater than zero"
            " and/or at leas one angle lies not in ]0, 180[",
            RuntimeWarning,
        )
        return np.zeros(box_mat.shape[:-2] + (6,), dtype=dtype)


def triclinic_vectors(box, dtype=None):
    """
    Convert the length-angle representation of a simulation box to the
    matrix representation.

    Convert the length-angle representation
    ``[lx, ly, lz, alpha, beta, gamma]`` (as returned by
    :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`) to the
    the matrix representation
    ``[[lx1, lx2, lx3], [[lz1, lz2, lz3]], [[lz1, lz2, lz3]]]`` (as
    returned by
    :attr:`MDAnalysis.coordinates.base.Timestep.triclinic_dimensions`).

    Parameters
    ----------
    box : array_like
        The unit cell dimensions of the system, which can be orthogonal
        or triclinic and must be provided in the same format as returned
        by :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.  `box` can also be an
        array of boxes of shape ``(k, 6)`` (one box for each frame).
    dtype : type, optional
        The data type of the output array.  If ``None``, the data type
        is inferred from the input array.

    Returns
    -------
    box_mat : numpy.ndarray
        The unit cell dimensions of the system in the same format as
        returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.triclinic_dimensions`:
        ``[[lx1, lx2, lx3], [[lz1, lz2, lz3]], [[lz1, lz2, lz3]]]``
        `box_mat` will be an array of shape ``(k, 3, 3)`` if `box` had
        shape ``(k, 6)``.  Each 3x3 box matrix contains the box vectors
        as *rows*!

        .. list-table:: Shapes
            :align: left
            :header-rows: 1

            *   - `box`
                - `box_mat`
            *   - ``(6,)``
                - ``(3, 3)``
            *   - ``(k, 6)``
                - ``(k, 3, 3)``

    See Also
    --------
    :func:`mdtools.box.triclinic_box` :
        Inverse function: Convert the matrix representation of a
        simulation box to the length-angle representation.
    :func:`MDAnalysis.lib.mdamath.triclinic_vectors` :
        Same function: Convert the length-angle representation of a
        simulation box to the matrix representation.

    Notes
    -----
    * The first vector is guaranteed to point along the x-axis, i.e., it
      has the form ``(lx, 0, 0)``.
    * The second vector is guaranteed to lie in the x/y-plane, i.e., its
      z-component is guaranteed to be zero.
    * If any box length is negative or zero, or if any box angle is
      zero, the box is treated as invalid and an all-zero-matrix is
      returned.

    This function is just a wrapper around
    :func:`MDAnalysis.lib.mdamath.triclinic_vectors` that also allows to
    parse arrays of boxes.

    Examples
    --------
    >>> box = [1, 2, 3, 90, 90, 90]
    >>> mdt.box.triclinic_vectors(box)
    array([[1., 0., 0.],
           [0., 2., 0.],
           [0., 0., 3.]])
    >>> box = [box, [4, 5, 6, 90, 90, 90]]
    >>> mdt.box.triclinic_vectors(box)
    array([[[1., 0., 0.],
            [0., 2., 0.],
            [0., 0., 3.]],
    <BLANKLINE>
           [[4., 0., 0.],
            [0., 5., 0.],
            [0., 0., 6.]]])

    >>> import MDAnalysis.lib.mdamath as mdamath
    >>> box = [1, 2, 3, 40, 60, 80]
    >>> box1 = mdt.box.triclinic_vectors(box)
    >>> box1
    array([[1.        , 0.        , 0.        ],
           [0.34729636, 1.96961551, 0.        ],
           [1.5       , 2.06909527, 1.57125579]])
    >>> box2 = mdamath.triclinic_vectors(box)
    >>> box2
    array([[1.        , 0.        , 0.        ],
           [0.34729636, 1.9696155 , 0.        ],
           [1.5       , 2.0690954 , 1.5712558 ]], dtype=float32)
    >>> np.allclose(box1, box2)
    True
    >>> box = [4, 5, 6, 100, 110, 120]
    >>> box1 = mdt.box.triclinic_vectors(box)
    >>> box1
    array([[ 4.        ,  0.        ,  0.        ],
           [-2.5       ,  4.33012702,  0.        ],
           [-2.05212086, -2.3878624 ,  5.10753494]])
    >>> box2 = mdamath.triclinic_vectors(box)
    >>> box2
    array([[ 4.       ,  0.       ,  0.       ],
           [-2.5      ,  4.3301272,  0.       ],
           [-2.052121 , -2.3878624,  5.107535 ]], dtype=float32)
    >>> np.allclose(box1, box2)
    True
    """
    box = mdt.check.box(box, with_angles=True)
    # ATTENTION: The array that is returned by
    # `mdamath.triclinic_vectors` contains the box vectors as rows, not
    # as columns!
    if box.ndim == 1:
        box_mat = mdamath.triclinic_vectors(box, dtype=dtype)
    elif box.ndim == 2:
        box_mat = np.asarray(
            [mdamath.triclinic_vectors(b, dtype=dtype) for b in box]
        )
    else:
        # This else clause should never be entered, because this error
        # should already be raised by `mdt.check.box(box)`.
        raise ValueError(
            "'box' must have shape (6,) or (k, 6), but has shape"
            " {}".format(box.shape)
        )
    return box_mat


def box2cart(pos, box, out=None, dtype=None):
    r"""
    Transform box coordinates to Cartesian coordinates.

    Parameters
    ----------
    pos : array_like
        The position array which to transform from box coordinates to
        Cartesian coordinates.  Must be either of shape ``(3,)``,
        ``(n, 3)`` or ``(k, n, 3)``, where ``n`` is the number of
        particles and ``k`` is the number of frames.
    box : array_like, optional
        The unit cell dimensions of the system in Cartesian coordinates.
        Can be orthogonal or triclinic and must be provided in the same
        format as returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.  `box` can also be an
        array of boxes of shape ``(k, 6)`` (one box for each frame).  If
        `box` has shape ``(k, 6)`` and ``pos`` has shape ``(n, 3)``, the
        latter will be broadcast to ``(k, n, 3)``.  If `box` has shape
        ``(k, 6)`` and ``pos`` has shape ``(3,)``, the latter will be
        broadcast to ``(k, 1, 3)``.

        Alternatively, `box` can be provided in the same format as
        returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.triclinic_dimensions`:
        ``[[lx1, lx2, lx3], [[lz1, lz2, lz3]], [[lz1, lz2, lz3]]]``.
        `box` can also be an array of boxes of shape ``(k, 3, 3)`` (one
        box for each frame).  Equivalent broadcasting rules as above
        apply.  Providing `box` already in matrix representation avoids
        the conversion step from the length-angle to the matrix
        representation.
    out : None or numpy.ndarray, optional
        Preallocated array of the given dtype and appropriate shape into
        which the result is stored.
    dtype : type, optional
        The data type of the output array.  If ``None``, the data type
        is inferred from the input arrays.

    Returns
    -------
    pos_cart : numpy.ndarray
        The position array in Cartesian coordinates.

        .. list-table:: Shapes
            :align: left
            :header-rows: 1

            *   - `pos`
                - `box`
                - `pos_cart`
            *   - ``(3,)``
                - ``(6,)`` or ``(3, 3)``
                - ``(3,)``
            *   - ``(3,)``
                - ``(k, 6)`` or ``(k, 3, 3)``
                - ``(k, 1, 3)``
            *   - ``(n, 3)``
                - ``(6,)`` or ``(3, 3)``
                - ``(n, 3)``
            *   - ``(n, 3)``
                - ``(k, 6)`` or ``(k, 3, 3)``
                - ``(k, n, 3)``
            *   - ``(k, n, 3)``
                - ``(6,)`` or ``(3, 3)``
                - ``(k, n, 3)``
            *   - ``(k, n, 3)``
                - ``(k, 6)`` or ``(k, 3, 3)``
                - ``(k, n, 3)``

    See Also
    --------
    :func:`mdtools.box.cart2box`
        Inverse function: Transform Cartesian coordinates to box
        coordinates.

    Notes
    -----
    The formula for the transformation of a vector
    :math:`\hat{\mathbf{r}}` given in box coordinates to the vector
    :math:`\mathbf{r}` given in Cartesian coordinates is

    .. math::

        \mathbf{r} = \mathbf{L} \hat{\mathbf{r}}

    where :math:`\mathbf{L} =
    \left( \mathbf{L}_x | \mathbf{L}_y | \mathbf{L}_z \right)`
    is the matrix of the (triclinic) box vectors in Cartesian
    coordinates.

    Examples
    --------
    `pos` has shape ``(3,)``:

    >>> pos = np.array([1, 2, 3])
    >>> box = np.array([1, 2, 3, 90, 90, 90])
    >>> mdt.box.box2cart(pos, box)
    array([1., 4., 9.])
    >>> box_mat = np.array([[1, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 3]])
    >>> mdt.box.box2cart(pos, box_mat)
    array([1, 4, 9])
    >>> box = np.array([[1, 2, 3, 90, 90, 90],
    ...                 [2, 4, 6, 90, 90, 90]])
    >>> mdt.box.box2cart(pos, box)
    array([[[ 1.,  4.,  9.]],
    <BLANKLINE>
           [[ 2.,  8., 18.]]])
    >>> box_mat = np.array([[[1, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 3]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 4, 0],
    ...                      [0, 0, 6]]])
    >>> mdt.box.box2cart(pos, box_mat)
    array([[[ 1,  4,  9]],
    <BLANKLINE>
           [[ 2,  8, 18]]])

    `pos` has shape ``(n, 3)``:

    >>> pos = np.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> box = np.array([1, 2, 3, 90, 90, 90])
    >>> mdt.box.box2cart(pos, box)
    array([[ 1.,  4.,  9.],
           [ 4., 10., 18.]])
    >>> box_mat = np.array([[1, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 3]])
    >>> mdt.box.box2cart(pos, box_mat)
    array([[ 1,  4,  9],
           [ 4, 10, 18]])
    >>> box = np.array([[1, 2, 3, 90, 90, 90],
    ...                 [2, 4, 6, 90, 90, 90]])
    >>> mdt.box.box2cart(pos, box)
    array([[[ 1.,  4.,  9.],
            [ 4., 10., 18.]],
    <BLANKLINE>
           [[ 2.,  8., 18.],
            [ 8., 20., 36.]]])
    >>> box_mat = np.array([[[1, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 3]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 4, 0],
    ...                      [0, 0, 6]]])
    >>> mdt.box.box2cart(pos, box_mat)
    array([[[ 1,  4,  9],
            [ 4, 10, 18]],
    <BLANKLINE>
           [[ 2,  8, 18],
            [ 8, 20, 36]]])

    `pos` has shape ``(k, n, 3)``:

    >>> pos = np.array([[[ 1,  2,  3],
    ...                  [ 4,  5,  6]],
    ...
    ...                 [[ 7,  8,  9],
    ...                  [10, 11, 12]]])
    >>> box = np.array([1, 2, 3, 90, 90, 90])
    >>> mdt.box.box2cart(pos, box)
    array([[[ 1.,  4.,  9.],
            [ 4., 10., 18.]],
    <BLANKLINE>
           [[ 7., 16., 27.],
            [10., 22., 36.]]])
    >>> box_mat = np.array([[1, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 3]])
    >>> mdt.box.box2cart(pos, box_mat)
    array([[[ 1,  4,  9],
            [ 4, 10, 18]],
    <BLANKLINE>
           [[ 7, 16, 27],
            [10, 22, 36]]])
    >>> box = np.array([[1, 2, 3, 90, 90, 90],
    ...                 [2, 4, 6, 90, 90, 90]])
    >>> mdt.box.box2cart(pos, box)
    array([[[ 1.,  4.,  9.],
            [ 4., 10., 18.]],
    <BLANKLINE>
           [[14., 32., 54.],
            [20., 44., 72.]]])
    >>> box_mat = np.array([[[1, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 3]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 4, 0],
    ...                      [0, 0, 6]]])
    >>> mdt.box.box2cart(pos, box_mat)
    array([[[ 1,  4,  9],
            [ 4, 10, 18]],
    <BLANKLINE>
           [[14, 32, 54],
            [20, 44, 72]]])

    Triclinic box:

    >>> pos = np.array([[ 7,  8,  9],
    ...                 [10, 11, 12]])
    >>> box_mat = np.array([[1, 0, 0],
    ...                     [2, 3, 0],
    ...                     [4, 5, 6]])
    >>> mdt.box.box2cart(pos, box_mat)
    array([[59, 69, 54],
           [80, 93, 72]])
    >>> box = mdt.box.triclinic_box(box_mat)
    >>> np.round(box, 3)
    array([ 1.   ,  3.606,  8.775, 43.368, 62.881, 56.31 ])
    >>> mdt.box.box2cart(pos, box)
    array([[59., 69., 54.],
           [80., 93., 72.]])
    >>> pos_cart = mdt.box.box2cart(box_mat, box_mat)
    >>> pos_cart
    array([[ 1,  0,  0],
           [ 8,  9,  0],
           [38, 45, 36]])
    >>> np.allclose(
    ...     pos_cart, np.linalg.matrix_power(box_mat, 2), rtol=0
    ... )
    True

    `out` argument:

    >>> pos = np.array([1, 2, 3])
    >>> box = np.array([1, 2, 3, 90, 90, 90])
    >>> out = np.full_like(box[:3], np.nan, dtype=np.float64)
    >>> pos_cart = mdt.box.box2cart(pos, box, out=out)
    >>> pos_cart
    array([1., 4., 9.])
    >>> out
    array([1., 4., 9.])
    >>> pos_cart is out
    True
    >>> box_mat = np.array([[1, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 3]])
    >>> out = np.full(box_mat.shape[:-1], np.nan, dtype=np.float64)
    >>> pos_cart = mdt.box.box2cart(pos, box_mat, out=out)
    >>> pos_cart
    array([1., 4., 9.])
    >>> out
    array([1., 4., 9.])
    >>> pos_cart is out
    True
    >>> box = np.array([[1, 2, 3, 90, 90, 90],
    ...                 [2, 4, 6, 90, 90, 90]])
    >>> out = np.full_like(box[:,:3], np.nan, dtype=np.float64)
    >>> pos_cart = mdt.box.box2cart(pos, box, out=out)
    >>> pos_cart
    array([[[ 1.,  4.,  9.]],
    <BLANKLINE>
           [[ 2.,  8., 18.]]])
    >>> out
    array([[[ 1.,  4.,  9.]],
    <BLANKLINE>
           [[ 2.,  8., 18.]]])
    >>> pos_cart is out
    True
    >>> box_mat = np.array([[[1, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 3]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 4, 0],
    ...                      [0, 0, 6]]])
    >>> out = np.full(box_mat.shape[:-1], np.nan, dtype=np.float64)
    >>> pos_cart = mdt.box.box2cart(pos, box_mat, out=out)
    >>> pos_cart
    array([[[ 1.,  4.,  9.]],
    <BLANKLINE>
           [[ 2.,  8., 18.]]])
    >>> out
    array([[[ 1.,  4.,  9.]],
    <BLANKLINE>
           [[ 2.,  8., 18.]]])
    >>> pos_cart is out
    True
    """
    pos = mdt.check.pos_array(pos)
    if box.ndim == 0:
        raise ValueError(
            "`box` ({}) must be at least 1-dimensional.".format(box)
        )
    elif box.shape[-1] == 6:
        # Get matrix of box vectors.
        # ATTENTION: The array that is returned by
        # `mdt.box.triclinic_vectors` contains the box vectors as rows,
        # not as columns like in the algebraic formula for coordinate
        # transformations!
        box_mat = mdt.box.triclinic_vectors(box, dtype=dtype)
        box_was_mat = False
    elif box.shape[-1] == 3:
        # `box`` is already given in matrix representation.
        box_mat = box
        box_was_mat = True
    else:
        raise ValueError("Invalid box (box.shape = {})".format(box.shape))

    # Transform the given position vector from box coordinates to
    # Cartesian coordinates.
    # About `np.einsum`: https://stackoverflow.com/a/33641428
    if pos.ndim == 1:
        # If the matrix contained the box vectors as columns, we had to
        # use ``subscripts = "...ij,...j->...i"``.  But the matrix
        # contains the box vectors as rows, therefore we have to use the
        # "transpose" of the subscripts.
        subscripts = "...ij,...i->...j"
    else:
        # If the matrix contained the box vectors as columns:
        # ``subscripts = "...ij,...kj->...ki"``.
        subscripts = "...ij,...ki->...kj"
    pos_cart = np.einsum(subscripts, box_mat, pos, out=out, dtype=dtype)
    if pos.ndim == 1 and (
        (not box_was_mat and box.ndim > 1) or (box_was_mat and box.ndim > 2)
    ):
        # Reshape the result array to (k, 1, 3) if `box` had shape
        # (k, 6) or (k, 3, 3) but `pos` only had shape (3,).
        pos_cart.shape = (box.shape[0], 1, pos.shape[0])
        if out is not None:
            out.shape = pos_cart.shape
    return pos_cart


def cart2box(pos, box, out=None, dtype=None):
    r"""
    Transform Cartesian coordinates to box coordinates.

    Parameters
    ----------
    pos : array_like
        The position array which to transform from Cartesian coordinates
        to box coordinates.  Must be either of shape ``(3,)``, ``(n,
        3)`` or ``(k, n, 3)``, where ``n`` is the number of particles
        and ``k`` is the number of frames.
    box : array_like, optional
        The unit cell dimensions of the system in Cartesian coordinates.
        Can be orthogonal or triclinic and must be provided in the same
        format as returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.  `box` can also be an
        array of boxes of shape ``(k, 6)`` (one box for each frame).  If
        `box` has shape ``(k, 6)`` and ``pos`` has shape ``(n, 3)``, the
        latter will be broadcast to ``(k, n, 3)``.  If `box` has shape
        ``(k, 6)`` and ``pos`` has shape ``(3,)``, the latter will be
        broadcast to ``(k, 1, 3)``.

        Alternatively, `box` can be provided in the same format as
        returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.triclinic_dimensions`:
        ``[[lx1, lx2, lx3], [[lz1, lz2, lz3]], [[lz1, lz2, lz3]]]``.
        `box` can also be an array of boxes of shape ``(k, 3, 3)`` (one
        box for each frame).  Equivalent broadcasting rules as above
        apply.  Providing `box` already in matrix representation avoids
        the conversion step from the length-angle to the matrix
        representation.
    out : None or numpy.ndarray, optional
        Preallocated array of the given dtype and appropriate shape into
        which the result is stored.
    dtype : type, optional
        The data type of the output array.  If ``None``, the data type
        is inferred from the input arrays.

    Returns
    -------
    pos_box : numpy.ndarray
        The position array in box coordinates.

        .. list-table:: Shapes
            :align: left
            :header-rows: 1

            *   - `pos`
                - `box`
                - `pos_box`
            *   - ``(3,)``
                - ``(6,)`` or ``(3, 3)``
                - ``(3,)``
            *   - ``(3,)``
                - ``(k, 6)`` or ``(k, 3, 3)``
                - ``(k, 1, 3)``
            *   - ``(n, 3)``
                - ``(6,)`` or ``(3, 3)``
                - ``(n, 3)``
            *   - ``(n, 3)``
                - ``(k, 6)`` or ``(k, 3, 3)``
                - ``(k, n, 3)``
            *   - ``(k, n, 3)``
                - ``(6,)`` or ``(3, 3)``
                - ``(k, n, 3)``
            *   - ``(k, n, 3)``
                - ``(k, 6)`` or ``(k, 3, 3)``
                - ``(k, n, 3)``

    See Also
    --------
    :func:`mdtools.box.box2cart`
        Inverse function: Transform box coordinates to Cartesian
        coordinates.

    Notes
    -----
    The formula for the transformation of a vector :math:`\mathbf{r}`
    given in Cartesian coordinates to the vector
    :math:`\hat{\mathbf{r}}` given in box coordinates is

    .. math::

        \hat{\mathbf{r}} = \mathbf{L}^{-1} \mathbf{r}

    where :math:`\mathbf{L} =
    \left( \mathbf{L}_x | \mathbf{L}_y | \mathbf{L}_z \right)`
    is the matrix of the (triclinic) box vectors in Cartesian
    coordinates.  Note that in box coordinates all boxes are cubes with
    length one.

    Examples
    --------
    `pos` has shape ``(3,)``:

    >>> pos = np.array([1, 2, 3])
    >>> box = np.array([1, 2, 3, 90, 90, 90])
    >>> mdt.box.cart2box(pos, box)
    array([1., 1., 1.])
    >>> box_mat = np.array([[1, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 3]])
    >>> mdt.box.cart2box(pos, box_mat)
    array([1., 1., 1.])
    >>> box = np.array([[1, 2, 3, 90, 90, 90],
    ...                 [2, 4, 6, 90, 90, 90]])
    >>> mdt.box.cart2box(pos, box)
    array([[[1. , 1. , 1. ]],
    <BLANKLINE>
           [[0.5, 0.5, 0.5]]])
    >>> box_mat = np.array([[[1, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 3]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 4, 0],
    ...                      [0, 0, 6]]])
    >>> mdt.box.cart2box(pos, box_mat)
    array([[[1. , 1. , 1. ]],
    <BLANKLINE>
           [[0.5, 0.5, 0.5]]])

    `pos` has shape ``(n, 3)``:

    >>> pos = np.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> box = np.array([1, 2, 3, 90, 90, 90])
    >>> mdt.box.cart2box(pos, box)
    array([[1. , 1. , 1. ],
           [4. , 2.5, 2. ]])
    >>> box_mat = np.array([[1, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 3]])
    >>> mdt.box.cart2box(pos, box_mat)
    array([[1. , 1. , 1. ],
           [4. , 2.5, 2. ]])
    >>> box = np.array([[1, 2, 3, 90, 90, 90],
    ...                 [2, 4, 6, 90, 90, 90]])
    >>> mdt.box.cart2box(pos, box)
    array([[[1.  , 1.  , 1.  ],
            [4.  , 2.5 , 2.  ]],
    <BLANKLINE>
           [[0.5 , 0.5 , 0.5 ],
            [2.  , 1.25, 1.  ]]])
    >>> box_mat = np.array([[[1, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 3]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 4, 0],
    ...                      [0, 0, 6]]])
    >>> mdt.box.cart2box(pos, box_mat)
    array([[[1.  , 1.  , 1.  ],
            [4.  , 2.5 , 2.  ]],
    <BLANKLINE>
           [[0.5 , 0.5 , 0.5 ],
            [2.  , 1.25, 1.  ]]])

    `pos` has shape ``(k, n, 3)``:

    >>> pos = np.array([[[ 1,  2,  3],
    ...                  [ 4,  5,  6]],
    ...
    ...                 [[ 7,  8,  9],
    ...                  [10, 11, 12]]])
    >>> box = np.array([1, 2, 3, 90, 90, 90])
    >>> mdt.box.cart2box(pos, box)
    array([[[ 1. ,  1. ,  1. ],
            [ 4. ,  2.5,  2. ]],
    <BLANKLINE>
           [[ 7. ,  4. ,  3. ],
            [10. ,  5.5,  4. ]]])
    >>> box_mat = np.array([[1, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 3]])
    >>> mdt.box.cart2box(pos, box_mat)
    array([[[ 1. ,  1. ,  1. ],
            [ 4. ,  2.5,  2. ]],
    <BLANKLINE>
           [[ 7. ,  4. ,  3. ],
            [10. ,  5.5,  4. ]]])
    >>> box = np.array([[1, 2, 3, 90, 90, 90],
    ...                 [2, 4, 6, 90, 90, 90]])
    >>> mdt.box.cart2box(pos, box)
    array([[[1.  , 1.  , 1.  ],
            [4.  , 2.5 , 2.  ]],
    <BLANKLINE>
           [[3.5 , 2.  , 1.5 ],
            [5.  , 2.75, 2.  ]]])
    >>> box_mat = np.array([[[1, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 3]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 4, 0],
    ...                      [0, 0, 6]]])
    >>> mdt.box.cart2box(pos, box_mat)
    array([[[1.  , 1.  , 1.  ],
            [4.  , 2.5 , 2.  ]],
    <BLANKLINE>
           [[3.5 , 2.  , 1.5 ],
            [5.  , 2.75, 2.  ]]])

    Triclinic box:

    >>> pos = np.array([[ 7,  8,  9],
    ...                 [10, 11, 12]])
    >>> box_mat = np.array([[1, 0, 0],
    ...                     [2, 3, 0],
    ...                     [4, 5, 6]])
    >>> mdt.box.cart2box(pos, box_mat)
    array([[0.66666667, 0.16666667, 1.5       ],
           [1.33333333, 0.33333333, 2.        ]])
    >>> box = mdt.box.triclinic_box(box_mat)
    >>> np.round(box, 3)
    array([ 1.   ,  3.606,  8.775, 43.368, 62.881, 56.31 ])
    >>> mdt.box.cart2box(pos, box)
    array([[0.66666667, 0.16666667, 1.5       ],
           [1.33333333, 0.33333333, 2.        ]])
    >>> pos_box = mdt.box.cart2box(box_mat, box_mat)
    >>> np.round(pos_box, 3)
    array([[ 1., -0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> np.allclose(pos_box, np.eye(3), rtol=0)
    True

    `out` argument:

    >>> pos = np.array([1, 2, 3])
    >>> box = np.array([1, 2, 3, 90, 90, 90])
    >>> out = np.full_like(box[:3], np.nan, dtype=np.float64)
    >>> pos_box = mdt.box.cart2box(pos, box, out=out)
    >>> pos_box
    array([1., 1., 1.])
    >>> out
    array([1., 1., 1.])
    >>> pos_box is out
    True
    >>> box_mat = np.array([[1, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 3]])
    >>> out = np.full(box_mat.shape[:-1], np.nan, dtype=np.float64)
    >>> pos_box = mdt.box.cart2box(pos, box_mat, out=out)
    >>> pos_box
    array([1., 1., 1.])
    >>> out
    array([1., 1., 1.])
    >>> pos_box is out
    True
    >>> box = np.array([[1, 2, 3, 90, 90, 90],
    ...                 [2, 4, 6, 90, 90, 90]])
    >>> out = np.full_like(box[:,:3], np.nan, dtype=np.float64)
    >>> pos_box = mdt.box.cart2box(pos, box, out=out)
    >>> pos_box
    array([[[1. , 1. , 1. ]],
    <BLANKLINE>
           [[0.5, 0.5, 0.5]]])
    >>> out
    array([[[1. , 1. , 1. ]],
    <BLANKLINE>
           [[0.5, 0.5, 0.5]]])
    >>> pos_box is out
    True
    >>> box_mat = np.array([[[1, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 3]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 4, 0],
    ...                      [0, 0, 6]]])
    >>> out = np.full(box_mat.shape[:-1], np.nan, dtype=np.float64)
    >>> pos_box = mdt.box.cart2box(pos, box_mat, out=out)
    >>> pos_box
    array([[[1. , 1. , 1. ]],
    <BLANKLINE>
           [[0.5, 0.5, 0.5]]])
    >>> out
    array([[[1. , 1. , 1. ]],
    <BLANKLINE>
           [[0.5, 0.5, 0.5]]])
    >>> pos_box is out
    True
    """
    pos = mdt.check.pos_array(pos)
    if box.ndim == 0:
        raise ValueError(
            "`box` ({}) must be at least 1-dimensional.".format(box)
        )
    elif box.shape[-1] == 6:
        # Get matrix of box vectors.
        # ATTENTION: The array that is returned by
        # `mdt.box.triclinic_vectors` contains the box vectors as rows,
        # not as columns like in the algebraic formula for coordinate
        # transformations!
        box_mat = mdt.box.triclinic_vectors(box, dtype=dtype)
        box_was_mat = False
    elif box.shape[-1] == 3:
        # `box` is already given in matrix representation.
        box_mat = box
        box_was_mat = True
    else:
        raise ValueError("Invalid box (box.shape = {})".format(box.shape))
    # Calculate the inverse matrix, because the inverse matrix is the
    # transform matrix for the change from Cartesian coordinates to box
    # coordinates.
    # Note: The inverse matrix of the transpose is the
    # transpose of the inverse matrix.
    box_mat_inv = np.linalg.inv(box_mat)

    # Transform the given position vector from Cartesian coordinates to
    # box coordinates.
    # About `np.einsum`: https://stackoverflow.com/a/33641428
    if pos.ndim == 1:
        # If the matrix contained the box vectors as columns, we had to
        # use ``subscripts = "...ij,...j->...i"``.  But the matrix
        # contains the box vectors as rows, therefore we have to use the
        # "transpose" of the subscripts.
        subscripts = "...ij,...i->...j"
    else:
        # If the matrix contained the box vectors as columns:
        # ``subscripts = "...ij,...kj->...ki"``.
        subscripts = "...ij,...ki->...kj"
    pos_box = np.einsum(subscripts, box_mat_inv, pos, out=out, dtype=dtype)
    if pos.ndim == 1 and (
        (not box_was_mat and box.ndim > 1) or (box_was_mat and box.ndim > 2)
    ):
        # Reshape the result array to (k, 1, 3) if `box` had shape
        # (k, 6) or (k, 3, 3) but `pos` only had shape (3,).
        pos_box.shape = (box.shape[0], 1, pos.shape[0])
        if out is not None:
            out.shape = pos_box.shape
    return pos_box


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
        :attr:`~MDAnalysis.coordinates.base.Timestep.dimensions` of the
        current :class:`~MDAnalysis.coordinates.base.Timestep` will be
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
    :func:`mdtools.box.make_whole` :
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
    function.  See :func:`mdtools.box.make_whole` for a possibility to
    make compounds whole that are split across periodic boundaries.

    Because :mod:`MDAnalysis` will pull
    :attr:`~MDAnalysis.core.universe.Universe.trajectory` data directly
    from the file it is reading from, changes to
    :class:`~MDAnalysis.core.groups.Atom` coordinates and box
    :attr:`~MDAnalysis.coordinates.base.Timestep.dimensions` will not
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
        the specified compounds of `ag` were made whole.

    See Also
    --------
    :meth:`MDAnalysis.core.groups.AtomGroup.unwrap` :
        Move
        :class:`Atoms <MDAnalysis.core.groups.Atom>`
        in the :class:`~MDAnalysis.core.groups.AtomGroup` such that
        bonds within the group's compounds aren't split across periodic
        boundaries
    :func:`MDAnalysis.lib.mdamath.make_whole` :
        Move all :class:`Atoms <MDAnalysis.core.groups.Atom>` in a
        MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` such that
        bonds don't split over periodic images
    :func:`mdtools.box.wrap` :
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
    :attr:`~MDAnalysis.coordinates.base.Timestep.dimensions` will not
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


def vdist(pos1, pos2, box=None, out=None, out_tmp=None, dtype=np.float64):
    r"""
    Calculate the distance vectors between two position arrays.

    Parameters
    ----------
    pos1, pos2 : array_like
        The two position arrays between which to calculate the distance
        vectors.  Position arrays must be of shape ``(3,)``, ``(n, 3)``
        or ``(k, n, 3)`` where ``n`` is the number of particles and
        ``k`` is the number of frames.  If `pos1` and `pos2` do not have
        the same shape, they must be broadcastable to a common shape.
        The user is responsible to provide inputs that result in a
        physically meaningful broadcasting!
    box : None or array_like, optional
        The unit cell dimensions of the system, which can be orthogonal
        or triclinic and must be provided in the same format as returned
        by :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.  If supplied, the minimum
        image convention is taken into account.  `box` can also be an
        array of boxes of shape ``(k, 6)`` (one box for each frame).  If
        `box` has shape ``(k, 6)`` and the result of ``pos1 - pos2`` has
        shape ``(n, 3)``, the latter will be broadcast to ``(k, n, 3)``.
        If `box` has shape ``(k, 6)`` and the result of ``pos1 - pos2``
        has shape ``(3,)``, the latter will be broadcast to
        ``(k, 1, 3)``.

        Alternatively, `box` can be provided in the same format as
        returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.triclinic_dimensions`:
        ``[[lx1, lx2, lx3], [[lz1, lz2, lz3]], [[lz1, lz2, lz3]]]``.
        `box` can also be an array of boxes of shape ``(k, 3, 3)`` (one
        box for each frame).  Equivalent broadcasting rules as above
        apply.  Providing `box` already in matrix representation avoids
        the conversion step from the length-angle to the matrix
        representation.
    out : None or numpy.ndarray, optional
        Preallocated array of the given dtype and appropriate shape into
        which the result is stored.  If `box` is ``None``, `out` is
        ignored the final result is stored in `out_tmp`.
    out_tmp : None or numpy.ndarray, optional
        Preallocated array of the given dtype into which temporary
        results are stored.  If provided, it must have the same shape as
        the result of ``pos1 - pos2``.  Providing `out_tmp` can speed up
        the calculated if this function is called many times.
    dtype : type, optional
        The data type of the output array.  If ``None``, the data type
        is inferred from the input arrays.

    Returns
    -------
    dist_vecs : numpy.ndarray
        Distance array containing the result of ``pos1 - pos2`` with the
        minimum image convention taken into account if `box` was
        supplied.

        .. list-table:: Shapes
            :align: left
            :header-rows: 1

            *   - ``pos1 - pos2``
                - `box`
                - `dist_vecs`
            *   - ``(3,)``
                - ``(6,)`` or ``(3, 3)``
                - ``(3,)``
            *   - ``(3,)``
                - ``(k, 6)`` or ``(k, 3, 3)``
                - ``(k, 1, 3)``
            *   - ``(n, 3)``
                - ``(6,)`` or ``(3, 3)``
                - ``(n, 3)``
            *   - ``(n, 3)``
                - ``(k, 6)`` or ``(k, 3, 3)``
                - ``(k, n, 3)``
            *   - ``(k, n, 3)``
                - ``(6,)`` or ``(3, 3)``
                - ``(k, n, 3)``
            *   - ``(k, n, 3)``
                - ``(k, 6)`` or ``(k, 3, 3)``
                - ``(k, n, 3)``

    See Also
    --------
    :func:`MDAnalysis.lib.distances.minimize_vectors` :
        Apply the minimum image convention to an array of vectors
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
    :func:`mdtools.numpy_helper_functions.subtract_mic` :
        Subtract two arrays element-wise respecting the minium image
        convention
    :func:`mdtools.numpy_helper_functions.diff_mic` :
        Calculate the difference between array elements respecting the
        minimum image convention.

    Notes
    -----
    If your are only interested in the distances itself (i.e. the norm
    of the distance vectors), consider using
    :func:`MDAnalysis.analysis.distances.dist` or
    :func:`MDAnalysis.lib.distances.calc_bonds` instead.

    The function :func:`MDAnalysis.lib.distances.minimize_vectors` (new
    in MDAnalysis version 2.1) has similar functionality as this
    function but only accepts distance vectors of shape ``(n, 3)`` and
    boxes of shape ``(6,)``.

    If `box` is provided, the minimum image convention is taken into
    account using algorithm C4 from Deiters [#]_.  This algorithm not
    only works for wrapped coordinates but for any image coordinates.
    Mathematically, it can be described by the following formula:

    .. math::

        \Delta\mathbf{r}^{mic} = \Delta\mathbf{r} - \mathbf{L}
        \biggl\lfloor
        \mathbf{L}^{-1} \Delta\mathbf{r} + \mathbf{\frac{1}{2}}
        \biggr\rfloor

    Here, :math:`\Delta\mathbf{r} = \mathbf{r}_1 - \mathbf{r}_2` is the
    difference of the position vectors and
    :math:`\mathbf{L} =
    \left( \mathbf{L}_x | \mathbf{L}_y | \mathbf{L}_z \right)`
    is the matrix of the (triclinic) box vectors.

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
    >>> # Shape of box is (6,), shape of output is (3,).
    >>> box = np.array([3, 2, 2, 90, 90, 90])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([ 1., -1., -1.])
    >>> # Shape of box is (3, 3), shape of output is (3,).
    >>> box_mat = np.array([[3, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 2]])
    >>> mdt.box.vdist(pos1, pos2, box=box_mat)
    array([ 1., -1., -1.])
    >>> # Shape of box is (k, 6), shape of output is (k, 1, 3).
    >>> box = np.array([[3, 2, 2, 90, 90, 90],
    ...                 [2, 3, 4, 90, 90, 90]])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.]]])
    >>> # Shape of box is (k, 3, 3), shape of output is (k, 1, 3).
    >>> box_mat = np.array([[[3, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 2]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 3, 0],
    ...                      [0, 0, 4]]])
    >>> mdt.box.vdist(pos1, pos2, box=box_mat)
    array([[[ 1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.]]])

    Shape of `pos1` and `pos2` is ``(n, 3)``:

    >>> pos1 = np.array([[0, 2, 4],
    ...                  [5, 3, 1]])
    >>> pos2 = np.array([[5, 3, 1],
    ...                  [0, 2, 4]])
    >>> mdt.box.vdist(pos1, pos2)
    array([[-5., -1.,  3.],
           [ 5.,  1., -3.]])
    >>> # Shape of box is (6,), shape of output is (n, 3).
    >>> box = np.array([3, 2, 2, 90, 90, 90])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[ 1., -1., -1.],
           [-1., -1., -1.]])
    >>> # Shape of box is (3, 3), shape of output is (n, 3).
    >>> box_mat = np.array([[3, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 2]])
    >>> mdt.box.vdist(pos1, pos2, box=box_mat)
    array([[ 1., -1., -1.],
           [-1., -1., -1.]])
    >>> # Shape of box is (k, 6), shape of output is (k, n, 3).
    >>> box = np.array([[3, 2, 2, 90, 90, 90],
    ...                 [2, 3, 4, 90, 90, 90]])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [-1.,  1.,  1.]]])
    >>> # Shape of box is (k, 3, 3), shape of output is (k, n, 3).
    >>> box_mat = np.array([[[3, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 2]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 3, 0],
    ...                      [0, 0, 4]]])
    >>> mdt.box.vdist(pos1, pos2, box=box_mat)
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [-1.,  1.,  1.]]])

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
    >>> # Shape of box is (6,), shape of output is (k, n, 3).
    >>> box = np.array([3, 2, 2, 90, 90, 90])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [ 1., -1., -1.]]])
    >>> # Shape of box is (3, 3), shape of output is (k, n, 3).
    >>> box_mat = np.array([[3, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 2]])
    >>> mdt.box.vdist(pos1, pos2, box=box_mat)
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [ 1., -1., -1.]]])
    >>> # Shape of box is (k, 6), shape of output is (k, n, 3).
    >>> box = np.array([[3, 2, 2, 90, 90, 90],
    ...                 [2, 3, 4, 90, 90, 90]])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[-1.,  0.,  1.],
            [-1.,  0., -1.]]])
    >>> # Shape of box is (k, 3, 3), shape of output is (k, n, 3).
    >>> box_mat = np.array([[[3, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 2]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 3, 0],
    ...                      [0, 0, 4]]])
    >>> mdt.box.vdist(pos1, pos2, box=box_mat)
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
    >>> # Shape of box is (6,), shape of output is (n, 3).
    >>> box = np.array([3, 2, 2, 90, 90, 90])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[ 1., -1., -1.],
           [ 0.,  0.,  0.]])
    >>> # Shape of box is (3, 3), shape of output is (n, 3).
    >>> box_mat = np.array([[3, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 2]])
    >>> mdt.box.vdist(pos1, pos2, box=box_mat)
    array([[ 1., -1., -1.],
           [ 0.,  0.,  0.]])
    >>> # Shape of box is (k, 6), shape of output is (k, n, 3).
    >>> box = np.array([[3, 2, 2, 90, 90, 90],
    ...                 [2, 3, 4, 90, 90, 90]])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [ 0.,  0.,  0.]]])
    >>> # Shape of box is (k, 3, 3), shape of output is (k, n, 3).
    >>> box_mat = np.array([[[3, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 2]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 3, 0],
    ...                      [0, 0, 4]]])
    >>> mdt.box.vdist(pos1, pos2, box=box_mat)
    array([[[ 1., -1., -1.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [ 0.,  0.,  0.]]])

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
    >>> # Shape of box is (6,), shape of output is (k, n, 3).
    >>> box = np.array([3, 2, 2, 90, 90, 90])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[ 1., -1., -1.],
            [-1.,  0.,  0.]]])
    >>> # Shape of box is (3, 3), shape of output is (k, n, 3).
    >>> box_mat = np.array([[3, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 2]])
    >>> mdt.box.vdist(pos1, pos2, box=box_mat)
    array([[[ 1., -1., -1.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[ 1., -1., -1.],
            [-1.,  0.,  0.]]])
    >>> # Shape of box is (k, 6), shape of output is (k, n, 3).
    >>> box = np.array([[3, 2, 2, 90, 90, 90],
    ...                 [2, 3, 4, 90, 90, 90]])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [ 0.,  0.,  0.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [ 0., -1., -2.]]])
    >>> # Shape of box is (k, 3, 3), shape of output is (k, n, 3).
    >>> box_mat = np.array([[[3, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 2]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 3, 0],
    ...                      [0, 0, 4]]])
    >>> mdt.box.vdist(pos1, pos2, box=box_mat)
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
    >>> # Shape of box is (6,), shape of output is (k, n, 3).
    >>> box = np.array([3, 2, 2, 90, 90, 90])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[ 1., -1., -1.],
            [ 1., -1., -1.]]])
    >>> # Shape of box is (3, 3), shape of output is (k, n, 3).
    >>> box_mat = np.array([[3, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 2]])
    >>> mdt.box.vdist(pos1, pos2, box=box_mat)
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[ 1., -1., -1.],
            [ 1., -1., -1.]]])
    >>> # Shape of box is (k, 6), shape of output is (k, n, 3).
    >>> box = np.array([[3, 2, 2, 90, 90, 90],
    ...                 [2, 3, 4, 90, 90, 90]])
    >>> mdt.box.vdist(pos1, pos2, box=box)
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [-1.,  0., -1.]]])
    >>> # Shape of box is (k, 3, 3), shape of output is (k, n, 3).
    >>> box_mat = np.array([[[3, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 2]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 3, 0],
    ...                      [0, 0, 4]]])
    >>> mdt.box.vdist(pos1, pos2, box=box_mat)
    array([[[ 1., -1., -1.],
            [-1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.],
            [-1.,  0., -1.]]])

    Triclinic boxes:

    >>> pos1 = np.array([[0, 2, 4],
    ...                  [5, 3, 1]])
    >>> pos2 = np.array([[5, 3, 1],
    ...                  [0, 2, 4]])
    >>> mdt.box.vdist(pos1, pos2)
    array([[-5., -1.,  3.],
           [ 5.,  1., -3.]])
    >>> box = np.array([3, 2, 2, 80, 90, 100])
    >>> np.round(mdt.box.vdist(pos1, pos2, box=box), 3)
    array([[ 0.653,  0.264, -0.937],
           [-0.653, -0.264,  0.937]])
    >>> box_mat = np.array([[1, 0, 0],
    ...                     [2, 3, 0],
    ...                     [4, 5, 6]])
    >>> mdt.box.vdist(pos1, pos2, box=box_mat)
    array([[-2., -3., -3.],
           [-2., -2., -3.]])

    `out` and `out_tmp` arguments:

    >>> # box is None.
    >>> pos1 = np.array([0, 2, 4])
    >>> pos2 = np.array([5, 3, 1])
    >>> out_tmp = np.full_like(pos1, np.nan, dtype=np.float64)
    >>> dist_vecs = mdt.box.vdist(pos1, pos2, out_tmp=out_tmp)
    >>> dist_vecs
    array([-5., -1.,  3.])
    >>> out_tmp
    array([-5., -1.,  3.])
    >>> dist_vecs is out_tmp
    True
    >>> # Shape of box is (6,), shape of output is (3,).
    >>> box = np.array([3, 2, 2, 90, 90, 90])
    >>> out = np.full_like(box[:3], np.nan, dtype=np.float64)
    >>> dist_vecs = mdt.box.vdist(pos1, pos2, box=box, out=out)
    >>> dist_vecs
    array([ 1., -1., -1.])
    >>> out
    array([ 1., -1., -1.])
    >>> dist_vecs is out
    True
    >>> # Shape of box is (3, 3), shape of output is (3,).
    >>> box_mat = np.array([[3, 0, 0],
    ...                     [0, 2, 0],
    ...                     [0, 0, 2]])
    >>> out = np.full(box_mat.shape[:-1], np.nan, dtype=np.float64)
    >>> dist_vecs = mdt.box.vdist(pos1, pos2, box=box_mat, out=out)
    >>> dist_vecs
    array([ 1., -1., -1.])
    >>> out
    array([ 1., -1., -1.])
    >>> dist_vecs is out
    True
    >>> # Shape of box is (k, 6), shape of output is (k, 1, 3).
    >>> box = np.array([[3, 2, 2, 90, 90, 90],
    ...                 [2, 3, 4, 90, 90, 90]])
    >>> out = np.full_like(box[:,:3], np.nan, dtype=np.float64)
    >>> out_tmp = np.full_like(pos1, np.nan, dtype=np.float64)
    >>> dist_vecs = mdt.box.vdist(
    ...     pos1, pos2, box=box, out=out, out_tmp=out_tmp
    ... )
    >>> dist_vecs
    array([[[ 1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.]]])
    >>> out
    array([[[ 1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.]]])
    >>> dist_vecs is out
    True
    >>> # Shape of box is (k, 3, 3), shape of output is (k, n, 3).
    >>> box_mat = np.array([[[3, 0, 0],
    ...                      [0, 2, 0],
    ...                      [0, 0, 2]],
    ...
    ...                     [[2, 0, 0],
    ...                      [0, 3, 0],
    ...                      [0, 0, 4]]])
    >>> out = np.full(box_mat.shape[:-1], np.nan, dtype=np.float64)
    >>> out_tmp = np.full_like(pos1, np.nan, dtype=np.float64)
    >>> dist_vecs = mdt.box.vdist(
    ...     pos1, pos2, box=box_mat, out=out, out_tmp=out_tmp
    ... )
    >>> dist_vecs
    array([[[ 1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.]]])
    >>> out
    array([[[ 1., -1., -1.]],
    <BLANKLINE>
           [[-1., -1., -1.]]])
    >>> dist_vecs is out
    True

    >>> import MDAnalysis.lib.distances as mdadist
    >>> import numpy as np
    >>> pos1 = np.array([[0, 2, 4],
    ...                  [5, 3, 1]])
    >>> pos2 = np.array([[5, 3, 1],
    ...                  [0, 2, 4]])
    >>> box = np.array([3, 2, 2, 90, 90, 90])
    >>> dist_vecs1 = mdt.box.vdist(pos1, pos2, box)
    >>> dists1 = np.linalg.norm(dist_vecs1, axis=-1)
    >>> dists2 = mdadist.calc_bonds(pos1, pos2, box)
    >>> np.allclose(dists1, dists2, rtol=0, atol=1e-6)
    True
    >>> box = np.array([3, 2, 2, 80, 90, 100])
    >>> dist_vecs1 = mdt.box.vdist(pos1, pos2, box)
    >>> dists1 = np.linalg.norm(dist_vecs1, axis=-1)
    >>> dists2 = mdadist.calc_bonds(pos1, pos2, box)
    >>> np.allclose(dists1, dists2, rtol=0, atol=1e-5)
    True
    """
    if out_tmp is not None and out_tmp is out:
        raise ValueError("`out_tmp` must not point to `out`")
    pos1 = mdt.check.pos_array(pos1)
    pos2 = mdt.check.pos_array(pos2)
    dist_vecs = np.subtract(pos1, pos2, out=out_tmp, dtype=dtype)
    if box is not None:
        if box.ndim == 0:
            raise ValueError(
                "`box` ({}) must be at least 1-dimensional.".format(box)
            )
        elif box.shape[-1] == 6:
            # Convert length-angle representation to matrix
            # representation.
            box = mdt.box.triclinic_vectors(box, dtype=dtype)
        elif box.shape[-1] == 3:
            # `box` is already given in matrix representation.
            box = box
        else:
            raise ValueError("Invalid box (box.shape = {})".format(box.shape))
        # Transform to box coordinates.
        dist_vecs_prime = mdt.box.cart2box(dist_vecs, box, out=out)
        dist_vecs_prime += 0.5
        dist_vecs_prime = np.floor(dist_vecs_prime, out=dist_vecs_prime)
        # Transform back to the Cartesian coordinate system.
        dist_vecs_prime = mdt.box.box2cart(
            dist_vecs_prime, box, out=dist_vecs_prime
        )
        if dist_vecs.shape != dist_vecs_prime.shape:
            # `dist_vecs_prime` gets additional axes prepended if `box`
            # had shape ``(k, 6)`` or ``(k, 3, 3)`` but the result of
            # ``pos1 - pos2`` only had shape ``(n, 3)`` or ``(3,)``.  To
            # enable inplace subtraction in all other cases, resize
            # `dist_vecs` to the new shape.  Alternatively, we could use
            # ``dist_vecs = dist_vecs - dist_vecs_prime``.
            dist_vecs = np.resize(dist_vecs, dist_vecs_prime.shape)
        dist_vecs_prime *= -1
        dist_vecs_prime += dist_vecs
        return dist_vecs_prime
    else:
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
    the previous atom positions to build the unwrapped trajectory.

    The main difference to :func:`mdtools.box.unwrap_trj` is that
    :func:`mdtools.box.unwrap_trj` unwraps the complete trajectory while
    this function only unwraps a single frame based on the unwrapped
    coordinates of the previous frame.

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
    the previous atom positions to build the unwrapped trajectory.

    The main difference to :func:`mdtools.box.unwrap` is that
    :func:`mdtools.box.unwrap` only unwraps a single frame while this
    function unwraps the complete trajectory.

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
        the unwrapping procedure is done, you can create a new universe
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
        ``'fragments'``. See :func:`mdtools.box.make_whole` for more
        details.  Is meaningless if `make_whole` and `keep_whole` are
        ``False``.
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
