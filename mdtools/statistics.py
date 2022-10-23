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
