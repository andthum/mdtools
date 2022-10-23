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


"""Functions for the statistical analysis of data."""


# Third-party libraries
import numpy as np

# First-party libraries
import mdtools as mdt


def acf(x, axis=None, dt=1, dtau=1, tau_max=None, center=True, unbiased=False):
    r"""
    Calculate the autocorrelation function of an array.

    Parameters
    ----------
    x : array_like
        The input array.
    axis : int or None, optional
        The axis along which to calculate the ACF.  By default, the
        flattened input array is used.
    dt : int or None, optional
        Distance between restarting times (actually restarting indices)
        :math:`t_0`.  If ``None`` is given, `dt` is set to the current
        lag time :math:`\tau` which results in independent sampling
        windows.
    dtau : int, optional
        Distance between evaluated lag times (actually lag indices)
        :math:`\tau`.
    tau_max : int or None, optional
        Maximum lag time :math:`\tau` upon which to compute the ACF.
        Because the first lag time is 0, the last evaluated lag time is
        actually ``tau_max - 1``.  By default, `tau_max` is set to
        ``x.size`` is `axis` is ``None`` and to ``x.shape[axis]``
        otherwise.  `tau_max` must lie within the interval
        ``[0, len(x)]``.
    center : bool, optional
        If ``True``, center the input array around its mean, i.e.
        subtract the sample mean :math:`\bar{X}` when calculating the
        variance and covariance.
    unbiased : bool, optional
        If ``True``, the covariance
        :math:`\text{Cov}[X_{t_0}, X_{t_0 + \tau}]` is normed by the
        acutal number of sample points :math:`t_0` (which depends on the
        lag time :math:`\tau`).  If ``False``, the covariance is for all
        lag times normed by the number of sampling points at
        :math:`t_0 = 0` (which is equal to the length of the input array
        `x`).  Note that setting `unbiased` to ``True`` might result in
        values of the ACF that lie outside the interval [-1, 1],
        especially for lag times larger than ``len(x) // 2`` if `center`
        is ``True``.

    Returns
    -------
    ac : numpy.ndarray
        Autocorrelation function of `x`.  If `axis` is ``None``, the
        output array has shape ``(n,)`` where ``n`` is
        ``int(np.ceil(tau_max / dtau))``.  If a specific axis is given,
        `ac` has the same shape as `x` except along the given axis where
        the length is given by ``n``.

    Raises
    ------
    ValueError :
        If the first element of `ac` is not unity or one or more
        elements of `ac` fall outside the interval [-1, 1].

    See Also
    --------
    :func:`mdtools.statistics.acf_np` :
        Different implementation of the ACF using
        :func:`numpy.correlate`

    Notes
    -----
    This function computes the autocorrelation function (ACF) of the
    input array `x` assuming that `x` describes a `weakly stationary
    stochastic process`_.  That means, the mean does not vary with time,
    the variance is finite for all times and the autocovariance does not
    depend on the starting time :math:`t_0` but only on the lag time
    :math:`\tau`.  In this case the ACF is given by: [1]_:sup:`,` [2]_

    .. math::

        C(\tau) = \frac{\text{Cov}[X_{t_0}, X_{t_0 + \tau}]}{\text{Var}[X]}
        = \frac{\langle (X_{t_0} - \mu) \cdot (X_{t_0 + \tau} - \mu) \rangle}{\langle (X - \mu)^2 \rangle}

    The ACF strictly fulfills
    :math:`C(\tau) \in [-1, 1] \text{ } \forall \text{ } \tau \in
    \mathbb{R}`.  Numerically, the ACF is calculated using the following
    formula: [3]_:sup:`,` [4]_

    .. math::

        C_\tau = \frac{\frac{1}{T^*} \sum_{t_0=0}^{T-\tau} (X_{t_0} - \bar{X}) (X_{t_0 + \tau} - \bar{X})}{\frac{1}{T} \sum_{t_0=0}^T (X_{t_0} - \bar{X})^2}
        = \frac{\frac{1}{T^*} \sum_{t_0=0}^{T-\tau} (X_{t_0} - \bar{X}) (X_{t_0 + \tau} - \bar{X})}{C_0}

    The increment of the sums in the formula of :math:`C_\tau` is given
    by `dt`.  :math:`\bar{X} = \frac{1}{T} \sum_{t=1}^T X_t` is the
    sample mean.  For the sample mean, the increment of the sum is
    always 1, independent of the choice of `dt`.  Note that **the sample
    mean is only subtracted when** `center` **is** ``True``.  `T` is
    always the length of the input array `x`, independent of the choice
    of `tau_max`.  :math:`T^*` is either :math:`T - \tau` if `unbiased`
    is ``True`` or simply :math:`T` if `unbiased` is ``False``.

    For the sake of completness, the ACF of a non-stationary
    stochasitic process is given by: [1]_:sup:`,` [2]_

    .. math::

        C(t_1, t_2) = \frac{\text{Cov}[X_{t_1}, X_{t_2}]}{\sqrt{\text{Var}[X_{t_1}]} \cdot \sqrt{\text{Var}[X_{t_2}]}}
        = \frac{\langle (X_{t_1} - \mu_{t_1}) \cdot (X_{t_2} - \mu_{t_2}) \rangle}{\sqrt{\langle (X_{t_1} - \mu_{t_1})^2 \rangle} \cdot \sqrt{\langle (X_{t_2} - \mu_{t_2})^2 \rangle}}

    .. _weakly stationary stochastic process:
        https://en.wikipedia.org/wiki/Stationary_process#Weak_or_wide-sense_stationarity

    References
    ----------
    .. [1] Wikipedia `Autocorrelation
        <https://en.wikipedia.org/wiki/Autocorrelation#Normalization>`_

    .. [2] Kun Il Park, `Fundamentals of Probability and Stochastic
        Processes with Applications to Communications
        <https://doi.org/10.1007/978-3-319-68075-0>`_, Springer, 2018,
        p. 169, Holmdel, New Jersey, USA, ISBN 978-3-319-68074-3

    .. [3] Julius O. Smith, `Unbiased Cross-Correlation
        <https://www.dsprelated.com/freebooks/mdft/Unbiased_Cross_Correlation.html>`_.
        In Mathematics of the Discrete Fourier Transform (DFT), 2002,
        p. 189, Stanford, California, USA

    .. [4] Wikipedia `Correlogram
        <https://en.wikipedia.org/wiki/Correlogram#Estimation_of_autocorrelations>`_

    Examples
    --------
    >>> a = np.array([-2,  2, -2,  2, -2])
    >>> mdt.stats.acf(a)
    array([ 1.        , -0.8       ,  0.56666667, -0.4       ,  0.13333333])
    >>> mdt.stats.acf(a, dt=None)
    array([ 1.        , -0.8       ,  0.26666667, -0.2       ,  0.13333333])
    >>> mdt.stats.acf(a, unbiased=True)
    array([ 1.        , -1.        ,  0.94444444, -1.        ,  0.66666667])
    >>> mdt.stats.acf(a, unbiased=True, dt=None)
    array([ 1.        , -1.        ,  0.66666667, -1.        ,  0.66666667])
    >>> mdt.stats.acf(a, center=False)
    array([ 1. , -0.8,  0.6, -0.4,  0.2])
    >>> mdt.stats.acf(a, center=False, dtau=2)
    array([1. , 0.6, 0.2])
    >>> mdt.stats.acf(a, center=False, tau_max=len(a)//2)
    array([ 1. , -0.8])
    >>> mdt.stats.acf(a, center=False, dt=None)
    array([ 1. , -0.8,  0.4, -0.2,  0.2])
    >>> mdt.stats.acf(a, center=False, unbiased=True)
    array([ 1., -1.,  1., -1.,  1.])
    >>> mdt.stats.acf(a, center=False, unbiased=True, dt=None)
    array([ 1., -1.,  1., -1.,  1.])

    >>> a = np.array([-2, -2, -2,  2,  2])
    >>> mdt.stats.acf(a)
    array([ 1.        ,  0.36666667, -0.26666667, -0.4       , -0.2       ])
    >>> mdt.stats.acf(a, dt=None)
    array([ 1.        ,  0.36666667, -0.06666667, -0.2       , -0.2       ])
    >>> mdt.stats.acf(a, unbiased=True)
    array([ 1.        ,  0.45833333, -0.44444444, -1.        , -1.        ])
    >>> mdt.stats.acf(a, unbiased=True, dt=None)
    array([ 1.        ,  0.45833333, -0.16666667, -1.        , -1.        ])
    >>> mdt.stats.acf(a, center=False)
    array([ 1. ,  0.4, -0.2, -0.4, -0.2])
    >>> mdt.stats.acf(a, center=False, dt=None)
    array([ 1. ,  0.4,  0. , -0.2, -0.2])
    >>> mdt.stats.acf(a, center=False, unbiased=True)
    array([ 1.        ,  0.5       , -0.33333333, -1.        , -1.        ])
    >>> mdt.stats.acf(a, center=False, unbiased=True, dt=None)
    array([ 1. ,  0.5,  0. , -1. , -1. ])

    >>> a = np.array([[-2,  2, -2,  2, -2],
    ...               [-2,  2, -2,  2, -2],
    ...               [-2, -2, -2,  2,  2],
    ...               [-2, -2, -2,  2,  2]])
    >>> mdt.stats.acf(a, center=False)
    array([ 1.  , -0.15,  0.  , -0.25, -0.2 ,  0.55,  0.  ,  0.15, -0.1 ,
           -0.25,  0.1 ,  0.05,  0.1 ,  0.05, -0.2 ,  0.05,  0.  ,  0.05,
            0.  , -0.05])
    >>> mdt.stats.acf(a, center=False, axis=0)
    array([[ 1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
           [ 0.75,  0.25,  0.75,  0.75,  0.25],
           [ 0.5 , -0.5 ,  0.5 ,  0.5 , -0.5 ],
           [ 0.25, -0.25,  0.25,  0.25, -0.25]])
    >>> mdt.stats.acf(a, center=False, axis=0, unbiased=True)
    array([[ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
           [ 1.        ,  0.33333333,  1.        ,  1.        ,  0.33333333],
           [ 1.        , -1.        ,  1.        ,  1.        , -1.        ],
           [ 1.        , -1.        ,  1.        ,  1.        , -1.        ]])
    >>> mdt.stats.acf(a, center=False, axis=1)
    array([[ 1. , -0.8,  0.6, -0.4,  0.2],
           [ 1. , -0.8,  0.6, -0.4,  0.2],
           [ 1. ,  0.4, -0.2, -0.4, -0.2],
           [ 1. ,  0.4, -0.2, -0.4, -0.2]])
    >>> mdt.stats.acf(a, center=False, axis=1, unbiased=True)
    array([[ 1.        , -1.        ,  1.        , -1.        ,  1.        ],
           [ 1.        , -1.        ,  1.        , -1.        ,  1.        ],
           [ 1.        ,  0.5       , -0.33333333, -1.        , -1.        ],
           [ 1.        ,  0.5       , -0.33333333, -1.        , -1.        ]])

    >>> a = np.array([[[-2,  2, -2,  2, -2],
    ...                [-2,  2, -2,  2, -2]],
    ...
    ...               [[-2, -2, -2,  2,  2],
    ...                [-2, -2, -2,  2,  2]]])
    >>> mdt.stats.acf(a, center=False)
    array([ 1.  , -0.15,  0.  , -0.25, -0.2 ,  0.55,  0.  ,  0.15, -0.1 ,
           -0.25,  0.1 ,  0.05,  0.1 ,  0.05, -0.2 ,  0.05,  0.  ,  0.05,
            0.  , -0.05])
    >>> mdt.stats.acf(a, center=False, axis=0)
    array([[[ 1. ,  1. ,  1. ,  1. ,  1. ],
            [ 1. ,  1. ,  1. ,  1. ,  1. ]],
    <BLANKLINE>
           [[ 0.5, -0.5,  0.5,  0.5, -0.5],
            [ 0.5, -0.5,  0.5,  0.5, -0.5]]])
    >>> mdt.stats.acf(a, center=False, axis=0, unbiased=True)
    array([[[ 1.,  1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.,  1.]],
    <BLANKLINE>
           [[ 1., -1.,  1.,  1., -1.],
            [ 1., -1.,  1.,  1., -1.]]])

    >>> mdt.stats.acf(a, center=False, axis=1)
    array([[[1. , 1. , 1. , 1. , 1. ],
            [0.5, 0.5, 0.5, 0.5, 0.5]],
    <BLANKLINE>
           [[1. , 1. , 1. , 1. , 1. ],
            [0.5, 0.5, 0.5, 0.5, 0.5]]])
    >>> mdt.stats.acf(a, center=False, axis=1, unbiased=True)
    array([[[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]],
    <BLANKLINE>
           [[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]]])
    >>> mdt.stats.acf(a, center=False, axis=2)
    array([[[ 1. , -0.8,  0.6, -0.4,  0.2],
            [ 1. , -0.8,  0.6, -0.4,  0.2]],
    <BLANKLINE>
           [[ 1. ,  0.4, -0.2, -0.4, -0.2],
            [ 1. ,  0.4, -0.2, -0.4, -0.2]]])
    >>> mdt.stats.acf(a, center=False, axis=2, unbiased=True)
    array([[[ 1.        , -1.        ,  1.        , -1.        ,
              1.        ],
            [ 1.        , -1.        ,  1.        , -1.        ,
              1.        ]],
    <BLANKLINE>
           [[ 1.        ,  0.5       , -0.33333333, -1.        ,
             -1.        ],
            [ 1.        ,  0.5       , -0.33333333, -1.        ,
             -1.        ]]])
    """  # noqa: E501, W505
    x = np.array(x, dtype=np.float64, copy=center)
    if center:
        x = mdt.stats.center(x, axis=axis, dtype=np.float64, inplace=True)
    # Maximum possible tau value.
    if axis is None:
        tau_max_real = x.size
    else:
        if abs(axis) >= x.ndim:
            raise np.AxisError(axis=axis, ndim=x.ndim)
        tau_max_real = x.shape[axis]

    if dt is not None and dt < 1:
        raise ValueError(
            "'dt' ({}) must be a positive integer or None".format(dt)
        )
    if dtau < 1:
        raise ValueError(
            "'dtau' ({}) must be a positive integer or None".format(dtau)
        )
    if tau_max is None:
        tau_max = tau_max_real
    elif tau_max < 0 or tau_max > len(x):
        raise ValueError(
            "'tau_max' ({}) must lie within the interval"
            " [0, {}]".format(tau_max, len(x))
        )

    ac = []
    for tau in range(0, tau_max, dtau):
        if dt is None:
            t_step = max(1, tau)  # Slice step cannot be zero
        else:
            t_step = dt
        t0_max = tau_max_real - tau
        cov = np.copy(mdt.nph.take(x, stop=t0_max, step=t_step, axis=axis))
        cov *= mdt.nph.take(x, start=tau, step=t_step, axis=axis)
        ac.append(np.sum(cov, axis=axis))
        if unbiased:
            if axis is None:
                ac[-1] /= cov.size
            else:
                ac[-1] /= cov.shape[axis]
    if axis is None:
        ac = np.concatenate(ac, axis=axis)
    else:
        ac = np.stack(ac, axis=axis)
    ac0 = mdt.nph.take(ac, start=0, stop=1, axis=axis)
    ac /= ac0

    # Consistency check.
    if axis is None:
        shape = (int(np.ceil(tau_max / dtau)),)
    else:
        shape = list(x.shape)
        shape[axis] = int(np.ceil(tau_max / dtau))
        shape = tuple(shape)
    if ac.shape != shape:
        raise ValueError(
            "'ac' has shape {} but should have shape {}.  This should not have"
            " happened".format(ac.shape, shape)
        )
    if not np.allclose(
        mdt.nph.take(ac, start=0, stop=1, axis=axis), 1, rtol=0
    ):
        raise ValueError(
            "The first element of the ACF is not unity but {}.  This should"
            " not have"
            " happened".format(mdt.nph.take(ac, start=0, stop=1, axis=axis))
        )
    if np.any(np.abs(ac) > 1):
        raise ValueError(
            "At least one element of the ACF is absolutely seen greater than"
            " unity.  This should not have happened.  min(ACF) = {}.  max(ACF)"
            " = {}".format(np.min(ac), np.max(ac))
        )
    return ac


def acf_np(x, center=True, unbiased=False):
    r"""
    Calculate the autocorrelation function of a 1-dimensional array.

    Parameters
    ----------
    x : array_like
        1-dimensional input array.
    center : bool, optional
        If ``True``, center the input array around its mean, i.e.
        subtract the sample mean :math:`\bar{X}` when calculating the
        variance and covariance.
    unbiased : bool, optional
        If ``True``, the covariance
        :math:`\text{Cov}[X_{t_0}, X_{t_0 + \tau}]` is normed by the
        acutal number of sample points :math:`t_0` (which depends on the
        lag time :math:`\tau`).  If ``False``, the covariance is for all
        lag times normed by the number of sampling points at
        :math:`t_0 = 0` (which is equal to the length of the input array
        `x`).  Note that setting `unbiased` to ``True`` might result in
        values of the ACF that lie outside the interval [-1, 1],
        especially for lag times larger than ``len(x) // 2`` if `center`
        is ``True``.

    Returns
    -------
    ac : numpy.ndarray
        Array of the same shape as `x` containing the autocorrelation
        function of `x`.

    Raises
    ------
    ValueError :
        If the first element of `ac` is not unity or one or more
        elements of `ac` fall outside the interval [-1, 1].

    See Also
    --------
    :func:`mdtools.statistics.acf` :
        Different tmplementation of the ACF using using a for loop

    Notes
    -----
    See :func:`mdtools.statistics.acf` for the mathematical details of
    computing the autocorrelation function (ACF).

    This function uses :func:`numpy.correlate` to calculate the
    autocorrelation function whereas :func:`mdtools.statistics.acf` uses
    an explicit for loop.  Therefore, :func:`mdtools.statistics.acf`
    can handle arbitrarily shaped input arrays and offers more options.

    Examples
    --------
    >>> a = np.random.normal(loc=3, scale=7, size=11)
    >>> np.allclose(mdt.stats.acf_np(a), mdt.stats.acf(a), rtol=0)
    True
    >>> np.allclose(
    ...     mdt.stats.acf_np(a, center=False),
    ...     mdt.stats.acf(a, center=False),
    ...     rtol=0
    ... )
    True
    """
    x = np.array(x, dtype=np.float64, copy=center)
    if x.ndim != 1:
        raise ValueError(
            "'x' must be 1-dimensional but is {}-dimensional".format(x.ndim)
        )

    if center:
        x -= np.mean(x, dtype=np.float64)
    ac = np.correlate(x, x, mode="full")
    ac = ac[-len(x) :]
    if unbiased:
        ac /= np.arange(len(ac), 0, -1)
    ac /= ac[0]

    if not np.isclose(ac[0], 1, rtol=0):
        raise ValueError(
            "The first element of the ACF is not unity but {}.  This should"
            " not have happened".format(ac[0])
        )
    if np.any(np.abs(ac) > 1):
        raise ValueError(
            "At least one element of the ACF is absolutely seen greater than"
            " unity.  This should not have happened.  min(ACF) = {}.  max(ACF)"
            " = {}".format(np.min(ac), np.max(ac))
        )
    return ac


def center(x, axis=None, dtype=None, inplace=False):
    """
    Center a distribution around its sample mean.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or None, optional
        The axis along which to compute the sample mean.  By default,
        the flattened input array is used.
    dtype : data-type, optional
        The dtype of the output array.  This dtype is also used to
        compute the mean.  Note that computing means can be inaccurate
        when using the floating-point dtype ``numpy.float32`` or less.
        See :func:`numpy.mean` for more details.
    inplace : bool, optional
        If ``True``, change the input array inplace if possible (the
        given dtype must be castable to the dtype of `x` and `x` must
        already be a :class:`numpy.ndarray`).

    Returns
    -------
    x_centered : numpy.ndarray
        The centered input distribution: ``x - np.mean(x)``.

    See Also
    --------
    :func:`numpy.mean`
        Compute the arithmetic mean along the specified axis
    :func:`mdtools.statistics.standardize`
        Standardize a distribution

    Examples
    --------
    >>> a = np.arange(24).reshape(2,3,4)
    >>> mdt.stats.center(a)
    array([[[-11.5, -10.5,  -9.5,  -8.5],
            [ -7.5,  -6.5,  -5.5,  -4.5],
            [ -3.5,  -2.5,  -1.5,  -0.5]],
    <BLANKLINE>
           [[  0.5,   1.5,   2.5,   3.5],
            [  4.5,   5.5,   6.5,   7.5],
            [  8.5,   9.5,  10.5,  11.5]]])
    >>> mdt.stats.center(a, axis=0)
    array([[[-6., -6., -6., -6.],
            [-6., -6., -6., -6.],
            [-6., -6., -6., -6.]],
    <BLANKLINE>
           [[ 6.,  6.,  6.,  6.],
            [ 6.,  6.,  6.,  6.],
            [ 6.,  6.,  6.,  6.]]])
    >>> mdt.stats.center(a, axis=1)
    array([[[-4., -4., -4., -4.],
            [ 0.,  0.,  0.,  0.],
            [ 4.,  4.,  4.,  4.]],
    <BLANKLINE>
           [[-4., -4., -4., -4.],
            [ 0.,  0.,  0.,  0.],
            [ 4.,  4.,  4.,  4.]]])
    >>> mdt.stats.center(a, axis=2)
    array([[[-1.5, -0.5,  0.5,  1.5],
            [-1.5, -0.5,  0.5,  1.5],
            [-1.5, -0.5,  0.5,  1.5]],
    <BLANKLINE>
           [[-1.5, -0.5,  0.5,  1.5],
            [-1.5, -0.5,  0.5,  1.5],
            [-1.5, -0.5,  0.5,  1.5]]])

    >>> a = np.ones(3, dtype=np.float64)
    >>> a
    array([1., 1., 1.])
    >>> b = mdt.stats.center(a, inplace=True)
    >>> b
    array([0., 0., 0.])
    >>> b is a
    True
    >>> b[0] = 1
    >>> a
    array([1., 0., 0.])
    """
    x = np.asarray(x, dtype=dtype)
    mean = np.mean(x, axis=axis, dtype=dtype)
    if axis is not None:
        mean = np.expand_dims(mean, axis=axis)
    if inplace:
        x -= mean
    else:
        x = x - mean
    return x


def standardize(x, axis=None, dtype=None, inplace=False, ddof=0):
    """
    Standardize a distribution.

    Standardize a distribution such that its mean is zero and its
    standard deviation is one.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or None, optional
        The axis along which to compute the sample mean and standard
        deviation.  By default, the flattened input array is used.
    dtype : data-type, optional
        The dtype of the output array.  This dtype is also used to
        compute the mean and standard deviation.  Note that computing
        means and standard deviations can be inaccurate when using the
        floating-point dtype ``numpy.float32`` or less.  See
        :func:`numpy.mean` for more details.
    inplace : bool, optional
        If ``True``, change the input array inplace if possible (the
        given dtype must be castable to the dtype of `x` and `x` must
        already be a :class:`numpy.ndarray`).
    ddof : int, optional
        Delta Degrees of Freedom.  When calculating the standard
        deviation, the used divisor is ``N - ddof``, where ``N``
        represents the number of elements.  See :func:`numpy.std` for
        more details.

    Returns
    -------
    x_standardized : numpy.ndarray
        The standardized input distribution:
        ``(x - np.mean(x)) / np.std(x)``.

    See Also
    --------
    :func:`numpy.mean`
        Compute the arithmetic mean along the specified axis
    :func:`numpy.std`
        Compute the standard deviation along the specified axis
    :func:`mdtools.statistics.center`
        Center a distribution around its sample mean

    Examples
    --------
    >>> a = np.arange(24).reshape(2,3,4)
    >>> mdt.stats.standardize(a)
    array([[[-1.66132477, -1.51686175, -1.37239873, -1.2279357 ],
            [-1.08347268, -0.93900965, -0.79454663, -0.65008361],
            [-0.50562058, -0.36115756, -0.21669454, -0.07223151]],
    <BLANKLINE>
           [[ 0.07223151,  0.21669454,  0.36115756,  0.50562058],
            [ 0.65008361,  0.79454663,  0.93900965,  1.08347268],
            [ 1.2279357 ,  1.37239873,  1.51686175,  1.66132477]]])
    >>> mdt.stats.standardize(a, axis=0)
    array([[[-1., -1., -1., -1.],
            [-1., -1., -1., -1.],
            [-1., -1., -1., -1.]],
    <BLANKLINE>
           [[ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.]]])
    >>> mdt.stats.standardize(a, axis=1)
    array([[[-1.22474487, -1.22474487, -1.22474487, -1.22474487],
            [ 0.        ,  0.        ,  0.        ,  0.        ],
            [ 1.22474487,  1.22474487,  1.22474487,  1.22474487]],
    <BLANKLINE>
           [[-1.22474487, -1.22474487, -1.22474487, -1.22474487],
            [ 0.        ,  0.        ,  0.        ,  0.        ],
            [ 1.22474487,  1.22474487,  1.22474487,  1.22474487]]])
    >>> mdt.stats.standardize(a, axis=2)
    array([[[-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079],
            [-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079],
            [-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079]],
    <BLANKLINE>
           [[-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079],
            [-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079],
            [-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079]]])

    >>> a = np.arange(3, dtype=np.float64)
    >>> a
    array([0., 1., 2.])
    >>> b = mdt.stats.standardize(a, inplace=True)
    >>> b
    array([-1.22474487,  0.        ,  1.22474487])
    >>> b is a
    True
    >>> b[0] = 1
    >>> a
    array([1.        , 0.        , 1.22474487])
    """
    x = np.asarray(x, dtype=dtype)
    x = mdt.stats.center(x, axis=axis, dtype=dtype, inplace=inplace)
    std = np.std(x, axis=axis, dtype=dtype, ddof=ddof)
    if axis is not None:
        std = np.expand_dims(std, axis=axis)
    if inplace:
        x /= std
    else:
        x = x / std
    return x


def gaussian(x, mu=0, sigma=1):
    """
    Gaussian distribution:

    .. math::
        f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}

    Parameters
    ----------
    x : scalar or array_like
        Array of :math:`x` values for wich to evaluate :math:`f(x)`.
    mu : scalar, optional
        Mean of the distribution.
    sigma : scalar, optional
        Standard deviation of the distribution.

    Returns
    -------
    f : scalar or array_like
        :math:`f(x)`.
    """

    s2 = 2 * sigma**2
    return np.exp(-(x - mu)**2 / s2) / np.sqrt(np.pi * s2)


def non_gaussian_parameter(x, d=1, is_squared=False, axis=None):
    """
    Non-Gaussian parameter as first introduced by Rahman, Singwi, and
    Sjölander in Phys. Rev., 1962, 126 to test if the distribution of
    particle displacements is Gaussian (test for Gaussian diffusion).

    The definition used here is taken from Song et al., PNAS, 2019, 116,
    12733-12742:

    .. math::
        A_{n,d}(t) = \frac{\langle x^{2n}(t) \rangle}{(1+\frac{2}{d}) \cdot \langle x^2 \rangle^n} - 1

    where :math:`d` is the number of spatial dimensions and :math:`t` is
    the time. If the distribution of :math:`x` is Gaussian,
    :math:`A_{n,d}(t)` is zero.

    Parameters
    ----------
    x : array_like
        Array of data for which to calculate the non-Gaussian parameter.
    d : int, optional
        The number of spatial dimensions of the data.
    is_squared : bool, optional
        If ``True``, `x` is assumed to be already squared. If ``False``,
        `x` will be squared before calculating the moments.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default
        (``None``) is to compute the mean of the flattened array. If
        `axis` is a tuple of ints, a the non-Gaussian parameter is
        calculated over multiple axes.

    Returns
    -------
    a : scalar
        The non-Gaussian parameter :math:`A_{n,d}(t)`.
    """

    if is_squared:
        x2 = x
    else:
        x2 = x**2
    x4 = np.mean(x2**2, axis=axis)
    x2 = np.mean(x2, axis=axis)
    return x4 / ((1 + 2 / d) * x2**2) - 1


def exp_dist(x, rate=1):
    """
    Exponential distribution:

    .. math::
        f(x) = \lambda e^{-\lambda x}

    The mean and standard deviation are given by :math:`1/\lambda`.

    Parameters
    ----------
    x : scalar or array_like
        Array of :math:`x` values for wich to evaluate :math:`f(x)`.
    rate : scalar, optional
        The value for :math:`lambda`.

    Returns
    -------
    f : scalar or array_like
        :math:`f(x)`.
    """

    return rate * np.exp(-rate * x)


def exp_dist_log(x, rate):
    """
    Logarithm of the exponential distribution (see also :func:`exp_dist`):

    .. math::
        f(x) = \ln(\lambda) - \lambda x

    Parameters
    ----------
    x : scalar or array_like
        Array of :math:`x` values for wich to evaluate :math:`f(x)`.
    rate : scalar, optional
        The value for :math:`lambda`.

    Returns
    -------
    f : scalar or array_like
        :math:`f(x)`.
    """

    return np.log(rate) - rate * x


def var_weighted(a, weights=None, axis=None, return_average=True):
    """
    Compute a weighted variance along the specified axis according to

    .. math::
        \sigma^2 = \sum_i (a_i - \mu)**2 \cdot p(a_i)

    with

    .. math::
        p(a_i) = \frac{\sum_i a_i * weight_i}{\sum_j weight_j}

    and

    .. math::
        \mu = \sum_i a_i \cdot p(a_i)

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired.
    weights : array_like, optional
        An array of weights associated with the values in `a`. Each
        value in `a` contributes to the variance according to its
        associated weight. The weights array can either be 1-D (in which
        case its length must be the size of `a` along the given axis) or
        of the same shape as `a`. If `weights=None`, then all data in
        `a` are assumed to have a weight equal to one. The only
        constraint on `weights` is that `sum(weights)` must not be 0.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed. The default
        is to compute the variance of the flattened array. If `axis` is
        a tuple of ints, a variance is performed over multiple axes.
    return_average : bool, optional
        If ``True`` (default), also return the weighted average
        (:math:`\mu`) used to calculate the weighted variance.

    Returns
    -------
    average : scalar
        Weighted average.
    var : scalar
        Weighted variance.
    """

    average = np.average(a, axis=axis, weights=weights)
    var = np.average(np.abs(a - average)**2, axis=axis, weights=weights)
    return average, var


def std_weighted(a, weights=None, axis=None, return_average=True):
    """
    Compute a weighted standard deviation along the specified axis
    according to

    .. math::
        \sigma = \sqrt{\sum_i (a_i - \mu)**2 \cdot p(a_i)}

    with

    .. math::
        p(a_i) = \frac{\sum_i a_i * weight_i}{\sum_j weight_j}

    and

    .. math::
        \mu = \sum_i a_i \cdot p(a_i)

    Parameters
    ----------
    a : array_like
        Array containing numbers whose standard deviation is desired.
    weights : array_like, optional
        An array of weights associated with the values in `a`. Each
        value in `a` contributes to the standard deviation according to
        its associated weight. The weights array can either be 1-D (in
        which case its length must be the size of `a` along the given
        axis) or of the same shape as `a`. If `weights=None`, then all
        data in `a` are assumed to have a weight equal to one. The only
        constraint on `weights` is that `sum(weights)` must not be 0.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The
        default is to compute the standard deviation of the flattened
        array. If `axis` is a tuple of ints, a standard deviation is
        performed over multiple axes.
    return_average : bool, optional
        If ``True`` (default), also return the weighted average
        (:math:`\mu`) used to calculate the weighted standard deviation.

    Returns
    -------
    average : scalar
        Weighted average.
    std : scalar
        Weighted standard deviation.
    """

    average = np.average(a, axis=axis, weights=weights)
    std = np.sqrt(np.average(np.abs(a - average)**2,
                             axis=axis,
                             weights=weights))
    return average, std


def running_average(a, axis=None, out=None):
    """
    Calculate the running average

    .. math::
        \mu_n = \frac{1}{n} \sum_{i=1}^{n} a_i

    Parameters
    ----------
    a : array_like
        Array of values for which to calculate the running average.
    axis : None or int, optional
        Axis along which the running average is computed. The default
        (``None``) is to compute the running average over the flattened
        array
    out : array_like
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output but
        the type will be cast if necessary.

    Returns
    -------
    rav : numpy.ndarray
        The running average along the specified axis. The result has the
        same size as `a`, and also the same shape as `a` if `axis` is
        not ``None`` or `a` is a 1-d array.
    """

    a = np.asarray(a)
    rav = np.cumsum(a, axis=axis, out=out)
    if axis is None:
        norm = np.arange(1, a.size + 1)
    else:
        s = [1 for i in range(a.ndim)]
        s[axis] = a.shape[axis]
        norm = np.arange(1, a.shape[axis] + 1, dtype=np.uint32).reshape(s)
    rav /= norm
    return rav


def block_average(data, axis=0, ddof=0, dtype=np.float64):
    """
    Calculate the arithmetic mean values and their standard deviations
    over multiple series of measurement.

    Parameters
    ----------
    data : array_like
        The array that holds the measured values. Must have at least two
        dimensions, one dimension containing the different series of
        measurement and at least one other dimension containing the
        measured values per series.
    axis : int, optional
        Axis along which the means and their standard deviations are
        computed.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in calculating
        the standard deviation is ``N-ddof``, where ``N`` is the number
        of measurements.
    dtpye : type, optional
        The data type of the output arrays and the data type to use in
        computing the means and standard deviations. Note: Computing
        means and standard deviation using ``numpy.float32`` can be
        inaccurate.

    Returns
    -------
    mean : numpy.ndarray
        The mean values of the measurements.
    sd : numpy.ndarray
        The standard deviations of the mean values.
    """

    data = np.asarray(data)
    num_measurments = data.shape[axis]
    mean = np.mean(data, axis=axis, dtype=dtype)
    sd = np.std(data, axis=axis, ddof=ddof, dtype=dtype)
    # Standard deviation of the mean value after n series of measurement:
    # sd_n = sd / sqrt(n)
    # The mean value remains always the same inside the error region.
    np.divide(sd, np.sqrt(num_measurments), out=sd, dtype=dtype)

    return mean, sd
