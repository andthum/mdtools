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


# Standard libraries
import warnings

# Third-party libraries
import numpy as np
from scipy import constants, special
from scipy.stats import norm

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
        actual number of sample points :math:`t_0` (which depends on the
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
    :func:`mdtools.statistics.acf_se` :
        Calculate the standard errors of an autocorrelation function
    :func:`mdtools.statistics.acf_confint` :
        Calculate the confidence intervals of an autocorrelation
        function

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

    For the sake of completeness, the ACF of a non-stationary
    stochastic process is given by: [1]_:sup:`,` [2]_

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
        actual number of sample points :math:`t_0` (which depends on the
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
        Different implementation of the ACF using using a for loop
    :func:`mdtools.statistics.acf_se` :
        Calculate the standard errors of an autocorrelation function
    :func:`mdtools.statistics.acf_confint` :
        Calculate the confidence intervals of an autocorrelation
        function

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


def acf_se(x, axis=None, n=None):
    r"""
    Calculate the standard errors of an autocorrelation function.

    Parameters
    ----------
    x : array_like
        The values of the ACF.  Intermediate lag times must not be
        missing.  That means if you used e.g.
        :func:`mdtools.statistics.acf` to compute the ACF, the argument
        `dtau` must have been ``1``.
    axis : int or None, optional
        The axis along which to calculate the SE.  By default, the
        flattened input array is used.
    n : int, optional
        Sample size (i.e. number of recorded time points) of the
        underlying time series.  By default, `n` is set to the number of
        lag times :math:`\tau` in the given ACF.  This is valid for ACFs
        that were computed at all possible lag times from the underlying
        time series.  That means if you used e.g.
        :func:`mdtools.statistics.acf` to compute the ACF, the argument
        `dtau` must have been ``1`` and `tau_max` must have been
        ``None`` or the length of `x` along the given axis..

    Returns
    -------
    se : numpy.ndarray
        Array of the same shape as `x` containing the standard error of
        the ACF at each lag time.

    See Also
    --------
    :func:`mdtools.statistics.acf` :
        Calculate the autocorrelation function of an array
    :func:`mdtools.statistics.acf_confint` :
        Calculate the confidence intervals of an autocorrelation
        function

    Notes
    -----
    The standard error (SE) of the autocorrelation function (ACF)
    :math:`C_\tau` at lag time :math:`\tau` is estimated according to
    Bartlett's formula for MA(l) processes: [#]_:sup:`,` [#]_

    .. math::

        SE(C_\tau) = \sqrt{\frac{1 + 2 \sum_{\tau^{'} = 1}^{\tau - 1} C_{\tau^{'}}}{N}}

    for :math:`\tau \gt 1`.  For :math:`\tau = 1` the standard error is
    estimated by :math:`SE(C_1) = \frac{1}{\sqrt{N}}` and for
    :math:`\tau = 0` the standard error is :math:`SE(C_0) = 0`.

    References
    ----------
    .. [#] Wikipedia `Correlogram
        <https://en.wikipedia.org/wiki/Correlogram#Statistical_inference_with_correlograms>`_

    .. [#] Wikipedia `Autoregressive-moving-average model
        <https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model#Moving-average_model>`_

    Examples
    --------
    >>> mdt.stats.acf_se([3])
    array([0.])
    >>> a = np.arange(4)
    >>> mdt.stats.acf_se(a)
    array([0.       , 0.5      , 0.8660254, 1.6583124])
    >>> b = np.column_stack([a, a])
    >>> mdt.stats.acf_se(b, axis=0)
    array([[0.       , 0.       ],
           [0.5      , 0.5      ],
           [0.8660254, 0.8660254],
           [1.6583124, 1.6583124]])
    >>> b = np.row_stack([a, a])
    >>> mdt.stats.acf_se(b, axis=1)
    array([[0.       , 0.5      , 0.8660254, 1.6583124],
           [0.       , 0.5      , 0.8660254, 1.6583124]])
    >>> c = np.array([b, b])
    >>> mdt.stats.acf_se(c, axis=2)
    array([[[0.       , 0.5      , 0.8660254, 1.6583124],
            [0.       , 0.5      , 0.8660254, 1.6583124]],
    <BLANKLINE>
           [[0.       , 0.5      , 0.8660254, 1.6583124],
            [0.       , 0.5      , 0.8660254, 1.6583124]]])
    """  # noqa: E501, W505
    x = np.asarray(x)
    if n is None and axis is None:
        n = x.size
    elif n is None:  # and axis is not None
        if abs(axis) >= x.ndim:
            raise np.AxisError(axis=axis, ndim=x.ndim)
        n = x.shape[axis]
    elif n <= 0:
        raise ValueError("'n' ({}) must be a positive integer".format(n))

    # SE(lag) = \sqrt{1 + 2 * \sum_{i=1}^{lag-1} ACF(i) / N}.
    # Sum starts at lag=1.
    acf = mdt.nph.take(x, start=1, axis=axis)
    se = np.cumsum(acf**2, axis=axis, dtype=np.float64)
    # The sum must only run until lag-1 but now runs until lag => Shift
    # the array by 1 to the right.
    se = np.roll(se, shift=1, axis=axis)
    se *= 2
    se += 1
    se /= n
    # SE(lag=1) = 1/\sqrt{N}.
    se1 = mdt.nph.take(se, start=0, stop=1, axis=axis)
    se1[:] = 1 / n
    se = np.sqrt(se, out=se)
    # SE(lag=0) = 0 because ACF(lag=0) is always 1 and has zero error.
    se = np.insert(se, 0, 0, axis=axis)

    # Consistency check
    if se.shape != x.shape:
        raise ValueError(
            "'se' has shape {} but should have shape {}.  This should not have"
            " happened".format(se.shape, x.shape)
        )
    if np.any(mdt.nph.take(se, start=0, stop=1, axis=axis) != 0):
        raise ValueError(
            "The first element of the standard error of the ACF is not zero"
            " but {}.  This should not have"
            " happened".format(mdt.nph.take(se, start=0, stop=1, axis=axis))
        )
    if not np.allclose(
        mdt.nph.take(se, start=1, stop=2, axis=axis), 1 / np.sqrt(n), rtol=0
    ):
        raise ValueError(
            "The second element of the standard error of the ACF is not"
            " 1/sqrt(n) = {} but {}.  This should not have happened".format(
                1 / np.sqrt(n),
                mdt.nph.take(se, start=1, stop=2, axis=axis),
            )
        )
    if (axis is None and np.any(np.diff(se) < 0)) or (
        axis is not None and np.any(np.diff(se, axis=axis) < 0)
    ):
        raise ValueError("'se' is not monotonically increasing")
    return se


def acf_confint(x, axis=None, alpha=0.05, n=None):
    r"""
    Calculate the confidence intervals of an autocorrelation function.

    Parameters
    ----------
    x : array_like
        The values of the ACF.  Intermediate lag times must not be
        missing.  That means if you used e.g.
        :func:`mdtools.statistics.acf` to compute the ACF, the argument
        `dtau` must have been ``1``.
    axis : int or None, optional
        The axis along which to calculate the confidence interval.  By
        default, the flattened input array is used.
    alpha : scalar, optional
        The significance level (also known as probability of error).
        The significance level is the maximum probability for rejecting
        the null hypothesis although its true (Type 1 error).  Here, the
        null hypothesis is that the underlying time series has no
        autocorrelation.  Typical values for the significance level are
        0.01 or 0.05.  The smaller the significance level (probability
        for Type 1 error), the higher the probability of a Type 2 error
        (i.e. the null hypothesis is not rejected although it is wrong).
    n : int, optional
        Sample size (i.e. number of recorded time points) of the
        underlying time series.  By default, `n` is set to the number of
        lag times :math:`\tau` in the given ACF.  This is valid for ACFs
        that were computed at all possible lag times from the underlying
        time series.  That means if you used e.g.
        :func:`mdtools.statistics.acf` to compute the ACF, the argument
        `dtau` must have been ``1`` and `tau_max` must have been
        ``None`` or the length of `x` along the given axis..

    Returns
    -------
    confint : numpy.ndarray
        Array of the same shape as `x` containing the upper limit of the
        confidence interval of the ACF at each lag time.  The lower
        limit is simply given by ``-confint``.  The confidence interval
        is centered at zero.  To get the confidence interval around the
        actual ACF values compute ``x + confint`` and ``x - confint``.

    See Also
    --------
    :func:`mdtools.statistics.acf` :
        Calculate the autocorrelation function of an array
    :func:`mdtools.statistics.acf_se` :
        Calculate the standard errors of an autocorrelation function

    Notes
    -----
    The confidence interval of the autocorrelation function (ACF)
    :math:`C_\tau` at lag time :math:`\tau` is estimated according to:
    [1]_

    .. math::

        B(C_\tau) = z_{1-\alpha/2} SE(C_\tau)

    with the significance level :math:`\alpha`, the quantile function
    :math:`z_p` of the *standard* normal distribution
    (:math:`\sigma = 1`) and the standard error :math:`SE(C_\tau)`.

    The quantile function :math:`z_p` is also known as the inverse
    cumulative distribution function :math:`F^{-1}(p)`.  The cumulative
    distribution function :math:`F(X)` returns the probability
    :math:`p` to gain a value below or equal to :math:`X`.  Hence, the
    inverse cumulative distribution function :math:`F^{-1}(p)` returns
    the value :math:`X` such that the probability of gaining a value
    below or equal to :math:`X` is given by :math:`p`.  Thus, the
    probability for gaining a value higher than :math:`X` is
    :math:`1-p`.

    A normal random variable :math:`X` will exceed the value
    :math:`\mu + z_p \sigma` with probability :math:`1-p` and will lie
    outside the interval :math:`\mu \pm z_p \sigma` with probability
    :math:`2(1-p)` (for :math:`p \ge 0.5` or :math:`2p` for
    :math:`p \le 0.5`).  In particular, the quantile :math:`z_{0.975}`
    is :math:`1.96`.  Therefore, a normal random variable will lie
    outside the interval :math:`\mu \pm 1.96\sigma` in only 5% of cases.
    [2]_  Here :math:`\mu` is the mean of the random variable and
    :math:`\sigma` is its standard deviation.  Note that for the normal
    distribution :math:`z_p - \mu = -(z_{1-p} - \mu)` holds, because it
    is symmetric around the mean.

    The standard error of the ACF is estimated according to Bartlett's
    formula for MA(l) processes: [1]_:sup:`,` [3]_

    .. math::

        SE(C_\tau) = \sqrt{\frac{1 + 2 \sum_{\tau^{'} = 1}^{\tau - 1} C_{\tau^{'}}}{N}}

    for :math:`\tau \gt 1`.  For :math:`\tau = 1` the standard error is
    estimated by :math:`SE(C_1) = \frac{1}{\sqrt{N}}` and for
    :math:`\tau = 0` the standard error is :math:`SE(C_0) = 0`.

    If the ACF :math:`C_\tau` is zero within the confidence interval,
    the null hypothesis that there is no autocorrelation at the given
    lag time :math:`\tau` is rejected with a significance level of
    :math:`\alpha`.  This is an approximate test that assumes that the
    underlying time series is normally distributed. [1]_

    References
    ----------
    .. [1] Wikipedia `Correlogram
        <https://en.wikipedia.org/wiki/Correlogram#Statistical_inference_with_correlograms>`_

    .. [2] Wikipedia `Normal distribution
        <https://en.wikipedia.org/wiki/Normal_distribution#Quantile_function>`_

    .. [3] Wikipedia `Autoregressive-moving-average model
        <https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model#Moving-average_model>`_

    Examples
    --------
    >>> mdt.stats.acf_confint([3])
    array([0.])
    >>> a = np.arange(4)
    >>> se = mdt.stats.acf_se(a)
    >>> ci = mdt.stats.acf_confint(a)
    >>> ci
    array([0.        , 0.97998199, 1.6973786 , 3.25023257])
    >>> np.allclose(ci, 1.959960*se, rtol=0, atol=1e-5)
    True
    >>> ci = mdt.stats.acf_confint(a, alpha=0.01)
    >>> ci
    array([0.        , 1.28791465, 2.23073361, 4.27152966])
    >>> np.allclose(ci, 2.575830*se, rtol=0, atol=1e-5)
    True

    >>> b = np.column_stack([a, a])
    >>> mdt.stats.acf_confint(b, axis=0)
    array([[0.        , 0.        ],
           [0.97998199, 0.97998199],
           [1.6973786 , 1.6973786 ],
           [3.25023257, 3.25023257]])
    >>> b = np.row_stack([a, a])
    >>> mdt.stats.acf_confint(b, axis=1)
    array([[0.        , 0.97998199, 1.6973786 , 3.25023257],
           [0.        , 0.97998199, 1.6973786 , 3.25023257]])
    >>> c = np.array([b, b])
    >>> mdt.stats.acf_confint(c, axis=2)
    array([[[0.        , 0.97998199, 1.6973786 , 3.25023257],
            [0.        , 0.97998199, 1.6973786 , 3.25023257]],
    <BLANKLINE>
           [[0.        , 0.97998199, 1.6973786 , 3.25023257],
            [0.        , 0.97998199, 1.6973786 , 3.25023257]]])
    """  # noqa: E501, W505
    rv = norm(loc=0, scale=1)
    quantile = rv.ppf(1 - alpha / 2)
    confint = mdt.stats.acf_se(x, axis=axis, n=n)
    confint *= quantile
    return confint


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
    :func:`numpy.mean` :
        Compute the arithmetic mean along the specified axis
    :func:`mdtools.statistics.standardize` :
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
    :func:`numpy.mean` :
        Compute the arithmetic mean along the specified axis
    :func:`numpy.std` :
        Compute the standard deviation along the specified axis
    :func:`mdtools.statistics.center` :
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
    r"""
    Gaussian distribution.

    .. math::

        f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}

    Parameters
    ----------
    x : scalar or array_like
        Array of :math:`x` values for which to evaluate :math:`f(x)`.
    mu : scalar or array_like, optional
        Mean of the distribution.  If an array of means is given, it
        must be broadcastable to a common shape with `x`.
    sigma : scalar or array_like, optional
        Standard deviation of the distribution.  If an array of standard
        deviations is given, it must be broadcastable to a common shape
        with `x`.

    Returns
    -------
    f : scalar or array_like
        The function values :math:`f(x)` at the given :math:`x` values.

    See Also
    --------
    :obj:`scipy.stats.norm` :
        Normal continuous random variable.

    Examples
    --------
    >>> from scipy.stats import norm
    >>> x = np.linspace(-1, 3, 5)
    >>> y1 = mdt.stats.gaussian(x, mu=1, sigma=1)
    >>> y2 = norm.pdf(x, loc=1, scale=1)
    >>> np.allclose(y1, y2, rtol=0)
    True
    >>> y1
    array([0.05399097, 0.24197072, 0.39894228, 0.24197072, 0.05399097])
    """  # noqa: E501, W505
    s2 = 2 * sigma**2
    return np.exp(-((x - mu) ** 2) / s2) / np.sqrt(np.pi * s2)


def ngp(x, axis=None, d=1, center=False, is_squared=False):
    r"""
    Compute the non-Gaussian parameter of a distribution of random
    variables.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes of `x` along which to compute the NGP.  By default,
        the flattened input array is used.
    d : int, optional
        Dimensionality of the random variables (this is *not* the
        dimensionality of the input array).  Originally, this option is
        provided to compute the NGP of a distribution of particle
        displacements :math:`\Delta\mathbf{r}` that were calculated in
        one, two or three dimensional euclidean space.
    center : bool, optional
        If ``True``, center the input distribution around its mean, i.e.
        use central moments :math:`\langle (X - \bar{X})^n \rangle` for
        computing the NGP (with :math:`\bar{X} = \sum_{i=1}^N X_i` being
        the sample mean).  If ``False``, use the raw moments
        :math:`\langle X^n \rangle` without subtracting the sample mean.
        Note that the central moments of the sample are biased
        estimators for the central moments of the population, whereas
        the raw moments of the sample are always equal to the raw
        moments of the population.  See Wikipedia
        `Moment (mathematics) § Sample moments
        <https://en.wikipedia.org/wiki/Moment_(mathematics)#Sample_moments>`_
        `center` must not be used together with `is_squared`, because we
        cannot estimate the original sample mean if `x` is already
        squared.
    is_squared : bool, optional
        If ``True``, `x` is assumed to be already squared.  If ``False``
        (default), `x` will be squared before calculating the moments.
        Setting `is_squared` to ``True`` is for example useful if `x`
        is the distribution of squared particle displacements
        :math:`\Delta\mathbf{r}^2(\tau)` of an ensemble of particles at
        a given lag time :math:`\tau`.  `is_squared` must not be used
        together with `center`.

    Returns
    -------
    a : scalar
        The non-Gaussian parameter :math:`\alpha_{2,d}`.

    Notes
    -----
    The non-Gaussian parameter (NGP) was first introduced by Rahman,
    Singwi, and Sjölander. [#]_:sup:`,` [#]_  Since then it has become a
    standard test to check for Gaussian diffusion, i.e. whether the
    particle displacements :math:`\Delta\mathbf{r}(\tau)` follow a
    Gaussian distribution.  For a Gaussian distribution the NGP is zero.

    This function uses the definition of the NGP in
    :math:`d`-dimensional euclidean space given in the text on page
    12735 of Reference: [#]_

    .. math::

        \alpha_{2,d} = \frac{1}{1+\frac{2}{d}} \frac{\langle X^4 \rangle}{\langle X^2 \rangle^2} - 1

    A more general definition of the NGP :math:`\alpha_{n,d}` in
    :math:`d`-dimensional euclidean space was given by Huang, Wang and
    Yu in Equation (14) of Reference: [#]_

    .. math::

        \alpha_{n,d} = \frac{(d-2)!! d^n}{(2n + d - 2)!!} \frac{\langle X^{2n} \rangle}{\langle X^2 \rangle^n} - 1

    References
    ----------
    .. [#] A. Rahman, K. S. Singwi, A. Sjölander, `Theory of Slow
        Neutron Scattering by Liquids. I
        <https://doi.org/10.1103/PhysRev.126.986>`_, Physical Review,
        1962, 126, 986.

    .. [#] A. Rahman, `Correlations in the Motion of Atoms in Liquid
        Argon <https://doi.org/10.1103/PhysRev.136.A405>`_, Physical
        Review, 1964, 136, A405.

    .. [#] Sanggeun Song, Seong Jun Park, Minjung Kim, Jun Soo Kim,
        Bong June Sung, Sangyoub Lee, Ji-Hyun Kim, Jaeyoung Sung,
        `Transport dynamics of complex fluids
        <https://doi.org/10.1073/pnas.1900239116>`_, Proceedings of the
        National Academy of Sciences, 2019, 116, 26, 12733-12742

    .. [#] Zihan Huang, Gaoming Wang, Zhao Yu, `Non-Gaussian Parameter
        in k-Dimensional Euclidean Space
        <https://doi.org/10.48550/arXiv.1511.06672>`_, arXiv preprint
        arXiv:1511.06672v1, 2015.

    Examples
    --------
    >>> mdt.stats.ngp(np.array([-2, -1, -1,  0,  0,  0,  1,  1,  2]))
    -0.25
    >>> a = mdt.stats.gaussian(np.linspace(-5, 5, 11))
    >>> mdt.stats.ngp(a)
    0.48352183005980653
    >>> a = np.arange(24).reshape(2,3,4)
    >>> mdt.stats.ngp(a)
    -0.3876040703052728
    >>> mdt.stats.ngp(a, center=True)
    -0.401391304347826
    >>> mdt.stats.ngp(a, axis=0)
    array([[-0.33333333, -0.34113033, -0.35946667, -0.382643  ],
           [-0.4071511 , -0.43103845, -0.45333333, -0.47363871],
           [-0.49187475, -0.50812525, -0.52254957, -0.53533412]])
    >>> mdt.stats.ngp(a, axis=0, center=True)
    array([[-0.66666667, -0.66666667, -0.66666667, -0.66666667],
           [-0.66666667, -0.66666667, -0.66666667, -0.66666667],
           [-0.66666667, -0.66666667, -0.66666667, -0.66666667]])
    >>> mdt.stats.ngp(a, axis=1)
    array([[-0.32      , -0.37225959, -0.42285714, -0.46559096],
           [-0.6152    , -0.62068471, -0.62535515, -0.62936154]])
    >>> mdt.stats.ngp(a, axis=1, center=True)
    array([[-0.5, -0.5, -0.5, -0.5],
           [-0.5, -0.5, -0.5, -0.5]])
    >>> mdt.stats.ngp(a, axis=2)
    array([[-0.33333333, -0.61552028, -0.64866075],
           [-0.65763599, -0.66126512, -0.66307898]])
    >>> mdt.stats.ngp(a, axis=2, center=True)
    array([[-0.45333333, -0.45333333, -0.45333333],
           [-0.45333333, -0.45333333, -0.45333333]])
    """  # noqa: E501, W505
    if is_squared and center:
        raise ValueError(
            "'center' must not be used together with 'is_squared'"
        )

    x2 = np.array(x, dtype=np.float64, copy=not is_squared)
    if not is_squared:
        if center:
            x2 = mdt.stats.center(
                x2, axis=axis, dtype=np.float64, inplace=True
            )
        x2 **= 2

    x4_mean = np.mean(x2**2, axis=axis, dtype=np.float64)
    x2_mean = np.mean(x2, axis=axis, dtype=np.float64)
    return x4_mean / ((1 + 2 / d) * x2_mean**2) - 1


def exp_dist(x, rate=1):
    r"""
    Exponential distribution.

    .. math::

        f(x) = \lambda e^{-\lambda x}

    The mean and standard deviation of an exponential distribution are
    given by :math:`1/\lambda`.

    Parameters
    ----------
    x : scalar or array_like
        Array of :math:`x` values for which to evaluate :math:`f(x)`.
    rate : scalar or array_like, optional
        The value(s) for :math:`\lambda`.  If an array is provided, it
        must be broadcastable to a common shape with `x`.

    Returns
    -------
    f : scalar or array_like
        The function values :math:`f(x)` at the given :math:`x` values.

    See Also
    --------
    :obj:`scipy.stats.expon` :
        Exponential continuous random variable

    Examples
    --------
    >>> from scipy.stats import expon
    >>> x = np.linspace(0, 4, 5)
    >>> y1 = mdt.stats.exp_dist(x, rate=2)
    >>> y2 = expon.pdf(x, loc=0, scale=1/2)
    >>> np.allclose(y1, y2, rtol=0)
    True
    >>> y1
    array([2.00000000e+00, 2.70670566e-01, 3.66312778e-02, 4.95750435e-03,
           6.70925256e-04])
    """  # noqa: W505
    return rate * np.exp(-rate * x)


def exp_dist_log(x, rate):
    r"""
    Logarithm of the exponential distribution.

    .. math::

        f(x) = \ln(\lambda) - \lambda x

    Parameters
    ----------
    x : scalar or array_like
        Array of :math:`x` values for which to evaluate :math:`f(x)`.
    rate : scalar or array_like, optional
        The value(s) for :math:`\lambda`.  If an array is provided, it
        must be broadcastable to a common shape with `x`.

    Returns
    -------
    f : scalar or array_like
        The function values :math:`f(x)` at the given :math:`x` values.

    See Also
    --------
    :func:`exp_dist` :
        Exponential distribution

    Examples
    --------
    >>> x = np.linspace(0, 4, 5)
    >>> y1 = mdt.stats.exp_dist_log(x, rate=2)
    >>> y2 = mdt.stats.exp_dist(x, rate=2)
    >>> np.allclose(y1, np.log(y2), rtol=0)
    True
    >>> y1
    array([ 0.69314718, -1.30685282, -3.30685282, -5.30685282, -7.30685282])
    """  # noqa: W505
    return np.log(rate) - rate * x


def mb_dist(v, temp=None, mass=None, var=None, drift=0, d=3):
    r"""
    Maxwell-Boltzmann speed distribution in d-dimensional space. [#]_

    .. math::

        p(v) = S_d \cdot (v - u)^{d-1}
        \left(\frac{1}{2\pi\sigma^2}\right)^{\frac{d}{2}}
        \exp{\left[-\frac{(v - u)^2}{2\sigma^2} \right]}

    With the speed

    .. math::

        v = \sqrt{\sum_{i=1}^d v_i^2},

    a drift speed :math:`u`, the variance of the gaussian part (note
    that this is not the variance of :math:`p(v)`)

    .. math::

        \sigma^2 = \frac{k_B T}{m}

    and the surface area of a :math:`d`-dimensional unit sphere [#]_

    .. math::

        S_d = \frac{2\pi^{\frac{d}{2}}}{\Gamma\left(\frac{d}{2}\right)}
            =
            \begin{cases}
                2      , & d = 1\\
                2\pi   , & d = 2\\
                4\pi   , & d = 3\\
                2\pi^2 , & d = 4\\
                \vdots &
            \end{cases}

    where :math:`\Gamma` is the `gamma function
    <https://en.wikipedia.org/wiki/Gamma_function>`_.

    Parameters
    ----------
    v : array_like
        Array of speed values for which to evaluate :math:`p(v)`.
    temp : scalar or None, optional
        Temperature of the particles.  Must be provided if `var` is
        ``None``.
    mass : scalar or None, optional
        Mass of the particles.  Must be provided if `var` is ``None``.
    var : scalar or None, optional
        Variance :math:`\sigma^2` of the Maxwell-Boltzmann distribution.
        If ``None``, it is calculated as :math:`\sigma^2 = k_B T / m`,
        where :math:`k_B` is the Boltzmann constant, :math:`T` is the
        temperature and :math:`m` is the mass of the particles.  If
        both, `temp` and `mass`, are provided, `var` is ignored.
    drift : scalar, optional
        Drift speed of the particles.
    d : int, optional
        Number of spatial dimensions in which the particles can move.

    Returns
    -------
    p : scalar or array_like
        The function values :math:`p(v)` at the given :math:`v` values.

    Notes
    -----
    Be aware of the units of your input values!  The unit of the
    Boltzmann constant :math:`k_B` is J/K.  Thus, temperatures must
    be given in K, masses in kg and velocities in m/s.  If you parse
    `var` directly instead of `temp` and `mass`, ``v**2`` and `var` must
    have the same units.

    Note that all arguments should be positive to get a physically
    meaningful output.  However, it is also possible to parse negative
    values to all input arguments.

    Note that a (positive) drift velocity only leads to a shift of the
    distribution along the x axis (to higher values).  Therefore, parts
    of the distribution that, without drift, would lie in the region of
    (unphysical) negative speeds now appear at positive speed values.
    The user is responsible to take care of this unphysical artifact.
    Generally, it's easiest to calculate the Maxwell-Boltzmann
    distribution without drift at positive speed values and afterwards
    shift the outcome by the drift speed.

    References
    ----------
    .. [#] Wikipedia `Maxwell-Boltzmann distribution § In n-dimensional
        space
        <https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution#In_n-dimensional_space>`_
    .. [#] Wikipedia `n-sphere § Closed forms
        <https://en.wikipedia.org/wiki/N-sphere#Closed_forms>`_

    Examples
    --------
    >>> x = np.linspace(0, 4, 5)
    >>> mdt.stats.mb_dist(x, var=1)
    array([0.        , 0.48394145, 0.43192773, 0.07977327, 0.00428257])
    >>> mdt.stats.mb_dist(x, var=2)
    array([0.        , 0.21969564, 0.4151075 , 0.26759315, 0.08266794])
    >>> mdt.stats.mb_dist(x, var=1, d=1)
    array([7.97884561e-01, 4.83941449e-01, 1.07981933e-01, 8.86369682e-03,
           2.67660452e-04])
    >>> mdt.stats.mb_dist(x, var=1, d=2)
    array([0.        , 0.60653066, 0.27067057, 0.03332699, 0.00134185])
    >>> mdt.stats.mb_dist(x, var=1, d=4)
    array([0.        , 0.30326533, 0.54134113, 0.14997145, 0.0107348 ])
    >>> mdt.stats.mb_dist(x, var=1, drift=1)
    array([0.48394145, 0.        , 0.48394145, 0.43192773, 0.07977327])
    >>> np.allclose(
    ...     mdt.stats.mb_dist(x, var=1)[:-1],
    ...     mdt.stats.mb_dist(x, var=1, drift=1)[1:],
    ...     rtol=0,
    ... )
    True

    >>> from scipy import constants
    >>> temp, mass = 273.15, 1.660e-27
    >>> var = constants.k * temp / mass
    >>> np.allclose(
    ...     mdt.stats.mb_dist(x, temp=temp, mass=mass),
    ...     mdt.stats.mb_dist(x, var=var),
    ...     rtol=0,
    ... )
    True
    >>> np.allclose(
    ...     mdt.stats.mb_dist(x, temp=temp, mass=mass, d=2),
    ...     mdt.stats.mb_dist(x, var=var, d=2),
    ...     rtol=0,
    ... )
    True
    >>> np.allclose(
    ...     mdt.stats.mb_dist(x, temp=temp, mass=mass, drift=2),
    ...     mdt.stats.mb_dist(x, var=var, drift=2),
    ...     rtol=0,
    ... )
    True

    The area below the Maxwell-Boltzmann distribution is one:

    >>> x = np.linspace(0, 20, 200)
    >>> for d in range(1, 6):
    ...     np.isclose(
    ...         np.trapz(mdt.stats.mb_dist(x, var=1, d=d), x),
    ...         1,
    ...         rtol=0,
    ...         atol=1e-3,
    ...     )
    True
    True
    True
    True
    True
    """  # noqa: E501, W505
    if var is None:
        if temp is None and mass is None:
            raise ValueError(
                "'mass' and 'temp' must be provided if 'var' is None"
            )
        var = constants.k * temp / mass
    else:
        if temp is not None and mass is not None:
            warnings.warn(
                "'var' ({}) is ignored, because 'temp' and 'mass' are given",
                UserWarning,
            )
    dist = norm.pdf(v, loc=drift, scale=np.sqrt(var))
    dist *= (v - drift) ** (d - 1)
    # Remaining factor: S_d * [1/(2*pi*sigma^2)]**(d/2), but
    # [1/(2*pi*sigma^2)]**(1/2) is already contained in `norm.pdf`
    # => The remaining factor is actually
    #    S_d * [1/(2*pi*sigma^2)]**((d-1)/2)
    sd = 2 * np.pi ** (d / 2)
    sd /= special.gamma(d / 2)
    dist *= sd
    dist /= (2 * np.pi * var) ** ((d - 1) / 2)
    return dist


def var_weighted(a, weights=None, axis=None, return_mean=False):
    r"""
    Compute the weighted variance along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    weights : array_like or None, optional
        An array of weights associated with the values in `a`. Each
        value in `a` contributes to the variance according to its
        associated weight.  The weights array can either be
        1-dimensional (in which case its length must be the size of `a`
        along the given axis) or of the same shape as `a`.  If `weights`
        is ``None``, then all data in `a` are assumed to have a weight
        equal to one.  The only constraint on `weights` is that
        ``np.sum(weights)`` must not be 0 along the given axis.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed.  The default
        is to compute the variance of the flattened array.
    return_mean : bool, optional
        If ``True``, also return the weighted sample mean
        :math:`\bar{X}` used to calculate the weighted variance.

    Returns
    -------
    var : scalar
        Weighted variance.
    mean : scalar
        Weighted sample mean.  Only returned if `return_mean` is
        ``True``.

    See Also
    --------
    :func:`numpy.average` :
        Compute the weighted average along the specified axis
    :func:`numpy.var` :
        Compute the variance along the specified axis
    :func:`mdtools.statistics.std_weighted` :
        Compute the weighted standard deviation along a given axis

    Notes
    -----
    The weighted variance is given by

    .. math::

        \sigma^2 = \sum_{i=1}^N (X_i - \bar{X})^2 \cdot p(X_i)

    with the weighted sample mean

    .. math::

        \bar{X} = \sum_{i=1}^N X_i \cdot p(X_i)

    the probability of element :math:`X_i`

    .. math::

        p(X_i) = \frac{\sum_{i=1}^N X_i w_i}{\sum_{j=1}^N w_j}

    and the weight :math:`w_i` of the :math:`i`-th element.
    """
    mean = np.average(a, axis=axis, weights=weights)
    var = np.average(np.abs(a - mean) ** 2, axis=axis, weights=weights)
    if return_mean:
        return var, mean
    else:
        return var


def std_weighted(a, weights=None, axis=None, return_mean=False):
    r"""
    Compute the weighted standard deviation along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    weights : array_like or None, optional
        An array of weights associated with the values in `a`.  Each
        value in `a` contributes to the standard deviation according to
        its associated weight.  The weights array can either be
        1-dimensional (in which case its length must be the size of `a`
        along the given axis) or of the same shape as `a`.  If `weights`
        is ``None``, then all data in `a` are assumed to have a weight
        equal to one.  The only constraint on `weights` is that
        ``np.sum(weights)`` must not be 0 along the given axis.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed.
        The default is to compute the standard deviation of the
        flattened array.
    return_mean : bool, optional
        If ``True``, also return the weighted sample mean
        :math:`\bar{X}` used to calculate the weighted standard
        deviation.

    Returns
    -------
    std : scalar
        Weighted standard deviation.
    mean : scalar
        Weighted sample mean.  Only returned if `return_mean` is
        ``True``.

    See Also
    --------
    :func:`numpy.average` :
        Compute the weighted average along the specified axis
    :func:`numpy.std` :
        Compute the standard deviation along the specified axis
    :func:`mdtools.statistics.var_weighted` :
        Compute the weighted variance along a given axis

    Notes
    -----
    The weighted standard deviation is given by

    .. math::

        \sigma = \sqrt{\sum_{i=1}^N (X_i - \bar{X})^2 \cdot p(X_i)}

    with the weighted sample mean

    .. math::

        \bar{X} = \sum_{i=1}^N X_i \cdot p(X_i)

    the probability of element :math:`X_i`

    .. math::

        p(X_i) = \frac{\sum_{i=1}^N X_i w_i}{\sum_{j=1}^N w_j}

    and the weight :math:`w_i` of the :math:`i`-th element.
    """
    var, mean = mdt.stats.var_weighted(
        a, weights=weights, axis=axis, return_mean=True
    )
    std = np.sqrt(var)
    if return_mean:
        return std, mean
    else:
        return std


def cumav(a, axis=None, out=None):
    r"""
    Calculate the cumulative average.

    Parameters
    ----------
    a : array_like
        Array of values for which to calculate the cumulative average.
    axis : None or int, optional
        Axis along which the cumulative average is computed.  The
        default is to compute the cumulative average of the flattened
        array.
    out : array_like
        Alternative output array in which to place the result.  It must
        have the same shape and buffer length as the expected output but
        the type will be cast if necessary.  See :func:`numpy.cumsum`
        for more details.

    Returns
    -------
    rav : numpy.ndarray
        The cumulative average along the specified axis.  The result has
        the same size as `a`, and also the same shape as `a` if `axis`
        is not ``None`` or if `a` is a 1-d array.

    See Also
    --------
    :func:`numpy.cumsum` :
        Return the cumulative sum of the elements along a given axis

    Notes
    -----
    The cumulative average at the :math:`n`-th position is given by

    .. math::

        \mu_n = \frac{1}{n} \sum_{i=1}^{n} a_i

    """
    a = np.asarray(a)
    cav = np.cumsum(a, axis=axis, out=out)
    if axis is None:
        norm = np.arange(1, a.size + 1)
    else:
        s = [1 for i in range(a.ndim)]
        s[axis] = a.shape[axis]
        norm = np.arange(1, a.shape[axis] + 1, dtype=np.uint32).reshape(s)
    cav /= norm
    return cav


def block_average(data, axis=0, ddof=0, dtype=np.float64):
    """
    Calculate the sample means and their standard deviations over
    multiple series of measurement.

    Parameters
    ----------
    data : array_like
        The array that holds the measured values.  Must have at least
        two dimensions, one dimension containing the different series
        of measurement and at least one other dimension containing the
        measured values per series.
    axis : int, optional
        Axis along which the means and their standard deviations are
        computed.
    ddof : int, optional
        Delta Degrees of Freedom.  The divisor used in calculating the
        standard deviation is ``N-ddof``, where ``N`` is the number of
        measurements.
    dtype : type, optional
        The data type of the output arrays and the data type to use in
        computing the means and standard deviations.  Note: Computing
        means and standard deviation using ``numpy.float32`` or lower
        can be inaccurate.  See :func:`numpy.mean` for more details.

    Returns
    -------
    mean : numpy.ndarray
        The mean values of the measurements.
    sd : numpy.ndarray
        The standard deviations of the mean values.

    See Also
    --------
    :func:`numpy.mean` :
        Compute the arithmetic mean along the specified axis
    :func:`numpy.std` :
        Compute the standard deviation along the specified axis
    """
    data = np.asarray(data)
    num_measurements = data.shape[axis]
    mean = np.mean(data, axis=axis, dtype=dtype)
    sd = np.std(data, axis=axis, ddof=ddof, dtype=dtype)
    # Standard deviation of the mean value after n series of
    # measurement: sd_n = sd / sqrt(n).  The mean value remains always
    # the same inside the error region.
    np.divide(sd, np.sqrt(num_measurements), out=sd, dtype=dtype)
    return mean, sd
