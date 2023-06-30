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


"""General mathematical and physical functions."""


# Standard libraries
import warnings

# Third-party libraries
import numpy as np
from scipy import optimize


# `scipy.optimize.curve_fit` can not handle zero standard deviations.
# Therefore, replace zeros with a very small value.
CURVE_FIT_ZERO_SD = 1e-30


def g(x, m=1, c=0):
    """
    Straight line:

    .. math::

        g(x) = mx + c

    Parameters
    ----------
    x : scalar or array_like
        Value(s) at which to evaluate :math:`g(x)`.
    m : scalar or array_like, optional
        Slope(s).
    c : scalar or array_like, optional
        y-axis intercept(s).

    Returns
    -------
    g : scalar or numpy.ndarray
        The outcome of :math:`g(x)`.  :math:`g(x)` is evaluated
        element-wise if at least one of the input arguments is an array.

    See Also
    --------
    :func:`mdtools.functions.g_inverse` :
        Inverse of a straight line

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    return m * x + c


def g_inverse(y, m=1, c=0):
    r"""
    Inverse of a straight line:

    .. math::

        g(y) = \frac{y-c}{m}

    Parameters
    ----------
    y : scalar or array_like
        Value(s) at which to evaluate :math:`g(y)`.
    m : scalar or array_like, optional
        Slope(s) of the original straight line.
    c : scalar or array_like, optional
        y-axis intercept(s) of the original straight line.

    Returns
    -------
    g : scalar or numpy.ndarray
        The outcome of :math:`g(y)`.  :math:`g(y)` is evaluated
        element-wise if at least one of the input arguments is an array.

    See Also
    --------
    :func:`mdtools.functions.g_inverse` :
        Straight line

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    return (y - c) / m


def exp_decay(t, k=1):
    """
    Exponential decay function:

    .. math::

        f(t) = e^{-kt}

    Parameters
    ----------
    t : scalar or array_like
        Value(s) at which to evaluate :math:`f(t)`.
    k : scalar or array_like
        Rate constant(s).

    See Also
    --------
    :func:`mdtools.functions.exp_decay_log` :
        Logarithm of an exponential decay function
    :func:`mdtools.functions.fit_exp_decay_log` :
        Fit an exponential decay function

    Returns
    -------
    decay : scalar or numpy.ndarray
        The outcome of :math:`f(t)`.  :math:`f(t)` is evaluated
        element-wise if `t` and/or `k` are arrays.

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    return np.exp(-k * t)


def exp_decay_log(t, k=1):
    r"""
    Logarithm of an exponential decay function:

    .. math::

        f(t) = -k \cdot t

    Parameters
    ----------
    t : scalar or array_like
        Value(s) at which to evaluate :math:`f(t)`.
    k : scalar or array_like, optional
        Rate constant(s).

    See Also
    --------
    :func:`mdtools.functions.exp_decay` :
        Exponential decay function
    :func:`mdtools.functions.fit_exp_decay_log` :
        Fit an exponential decay function

    Returns
    -------
    decay : scalar or numpy.ndarray
        The outcome of :math:`f(t)`.  :math:`f(t)` is evaluated
        element-wise if `t` and/or `k` are arrays.

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    return -k * t


def fit_exp_decay_log(xdata, ydata, ysd=None, return_valid=False):
    r"""
    Fit an exponential decay function to `ydata` using
    :func:`scipy.optimize.curve_fit`.

    Instead of fitting the data directly with an exponential decay
    function :math:`e^{-kt}`, the logarithm of the y-data is fitted by

    .. math::

        f(t) = -k \cdot t

    :math:`k` is bound to :math:`[0, \infty)`.

    Parameters
    ----------
    xdata : array_like
        The independent variable where the data is measured.
    ydata : array_like
        The dependent data.  Must have the same shape as `xdata`.  Only
        data points with `0 < ydata <= 1` will be considered.  NaN's and
        infinite values will not be considered, either.
    ysd : array_like, optional
        The standard deviation of `ydata`.  Must have the same shape as
        `ydata`.
    return_valid : bool, optional
        If ``True``, return a boolean array of the same shape as `ydata`
        that indicates which elements of `ydata` meet the requirements
        given above.

    Returns
    -------
    popt : numpy.ndarray
        Optimal values for the parameters so that the sum of the squared
        residuals is minimized.  The first and only element of `popt` is
        the optimal :math:`k` value.
    perr : numpy.ndarray
        Standard deviation of the optimal parameters.
    valid : numpy.ndarray
        Boolean array of the same shape as `ydata` indicating, which
        elements of `ydata` were used for the fit.  Only returned if
        `return_valid` is ``True``.

    See Also
    --------
    :func:`mdtools.functions.exp_decay` :
        Exponential decay function
    :func:`mdtools.functions.exp_decay_log` :
        Logarithm of an exponential decay function
    """
    ydata = np.asarray(ydata)
    valid = np.isfinite(ydata) & (ydata > 0) & (ydata <= 1)
    if not np.all(valid):
        warnings.warn(
            "{} elements of ydata do not fulfill the requirement"
            " 0 < ydata <= 1 and are discarded for"
            " fitting".format(len(valid) - np.count_nonzero(valid)),
            RuntimeWarning,
        )
    if not np.any(valid):
        warnings.warn(
            "None of the y-data fulfills the requirement 0 < ydata <= 1."
            "  Setting fit parameters to numpy.nan",
            RuntimeWarning,
        )
        if return_valid:
            return np.array([np.nan]), np.array([np.nan]), valid
        else:
            return np.array([np.nan]), np.array([np.nan])
    if ysd is not None:
        ysd = ysd[valid]
        # Propagation of uncertainty when taking the logarithm of the
        # data.  See
        # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
        ysd /= ydata[valid]
        ysd[ysd == 0] = CURVE_FIT_ZERO_SD

    try:
        popt, pcov = optimize.curve_fit(
            f=exp_decay_log,
            xdata=xdata[valid],
            ydata=np.log(ydata[valid]),
            sigma=ysd,
            absolute_sigma=True,
            p0=1 / (len(ydata) * np.mean(ydata)),
            bounds=(0, np.inf),
        )
    except (ValueError, RuntimeError) as err:
        print(flush=True)
        print("An error has occurred during fitting:", flush=True)
        print("{}".format(err), flush=True)
        print("Setting fit parameters to numpy.nan", flush=True)
        print(flush=True)
        if return_valid:
            return np.array([np.nan]), np.array([np.nan]), valid
        else:
            return np.array([np.nan]), np.array([np.nan])
    except optimize.OptimizeWarning as warn:
        print(flush=True)
        print("A warning has occurred during fitting:", flush=True)
        print("{}".format(warn), flush=True)
        perr = np.sqrt(np.diag(pcov))
    else:
        perr = np.sqrt(np.diag(pcov))

    if return_valid:
        return popt, perr, valid
    else:
        return popt, perr


def kww(t, tau=1, beta=1):
    r"""
    Stretched exponential function, also known as Kohlrausch-Williams-
    Watts (KWW) function:

    .. math::

        f(t) = \exp{\left[ -\left( \frac{t}{\tau} \right)^\beta \right]}

    In physics, this function is often used as a phenomenological
    description of relaxation in disordered systems when the relaxation
    cannot be described by a simple exponential decay.  Usually, it is
    then assumed that the system does not have a single relaxation
    mechanism with a single relaxation time, but the total relaxation is
    a superposition of multiple relaxation mechanisms with different
    relaxation times.  The mean relaxation time is then given by

    .. math::

        \langle \tau \rangle = \int_0^\infty \text{d}t
        \exp{\left[ -\left( \frac{t}{\tau} \right)^\beta \right]}
        = \frac{\tau}{\beta} \Gamma(\frac{1}{\beta})

    with :math:`\Gamma(x)` being the gamma function.

    Parameters
    ----------
    t : scalar or array_like
        Value(s) at which to evaluate :math:`f(t)`.
    tau : scalar or array_like, optional
        Relaxation time(s).
    beta : scalar or array_like, optional
        Stretching exponent(s).

    Returns
    -------
    kww : scalar or numpy.ndarray
        The outcome of :math:`f(t)`.  :math:`f(t)` is evaluated
        element-wise if at least one of the input arguments is an array.

    See Also
    --------
    :func:`mdtools.functions.fit_kww` :
        Fit a stretched exponential function

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    return np.exp(-((t / tau) ** beta))


def fit_kww(xdata, ydata, ysd=None, return_valid=False):
    r"""
    Fit a stretched exponential function, also known as Kohlrausch-
    Williams-Watts (KWW) function, to `ydata` using
    :func:`scipy.optimize.curve_fit`.

    The KWW function reads:

    .. math::

        f(t) = \exp{\left[ -\left( \frac{t}{\tau} \right)^\beta \right]}

    :math:`\beta` is bound to :math:`[0, 1]`, :math:`\tau` is bound to
    :math:`[0, \infty)`.

    Parameters
    ----------
    xdata : array_like
        The independent variable where the data is measured.
    ydata : array_like
        The dependent data.  Must have the same shape as `xdata`.  Only
        data points with `0 < ydata <= 1` will be considered.  NaN's and
        infinite values will not be considered, either.
    ysd : array_like, optional
        The standard deviation of `ydata`.  Must have the same shape as
        `ydata`.
    return_valid : bool, optional
        If ``True``, return a boolean array of the same shape as `ydata`
        that indicates which elements of `ydata` meet the requirements
        given above.

    Returns
    -------
    popt : numpy.ndarray
        Optimal values for the parameters so that the sum of the squared
        residuals is minimized.  The first element of `popt` is the
        optimal :math:`\tau` value, the second element is the optimal
        :math:`\beta` value.
    perr : numpy.ndarray
        Standard deviation of the optimal parameters.
    valid : numpy.ndarray
        Boolean array of the same shape as `ydata` indicating, which
        elements of `ydata` were used for the fit.  Only returned if
        `return_valid` is ``True``.
    """
    ydata = np.asarray(ydata)
    valid = np.isfinite(ydata) & (ydata > 0) & (ydata <= 1)
    if not np.all(valid):
        warnings.warn(
            "{} elements of ydata do not fulfill the requirement"
            " 0 < ydata <= 1 and are discarded for"
            " fitting".format(len(valid) - np.count_nonzero(valid)),
            RuntimeWarning,
        )
    if not np.any(valid):
        warnings.warn(
            "None of the y-data fulfills the requirement 0 < ydata <= 1."
            "  Setting fit parameters to numpy.nan",
            RuntimeWarning,
        )
        if return_valid:
            return np.full(2, np.nan), np.full(2, np.nan), valid
        else:
            return np.full(2, np.nan), np.full(2, np.nan)
    if ysd is not None:
        ysd = ysd[valid]
        # `scipy.optimize.curve_fit` can not handle zero standard
        # deviations.  Therefore, replace zeros with a very small value.
        ysd[ysd == 0] = CURVE_FIT_ZERO_SD

    try:
        popt, pcov = optimize.curve_fit(
            f=kww,
            xdata=xdata[valid],
            ydata=ydata[valid],
            sigma=ysd,
            absolute_sigma=True,
            p0=[len(ydata) * np.mean(ydata), 1],
            bounds=([0, 0], [np.inf, 1]),
        )
    except (ValueError, RuntimeError) as err:
        print(flush=True)
        print("An error has occurred during fitting:", flush=True)
        print("{}".format(err), flush=True)
        print("Setting fit parameters to numpy.nan", flush=True)
        print(flush=True)
        if return_valid:
            return np.full(2, np.nan), np.full(2, np.nan), valid
        else:
            return np.full(2, np.nan), np.full(2, np.nan)
    except optimize.OptimizeWarning as warn:
        print(flush=True)
        print("A warning has occurred during fitting:", flush=True)
        print("{}".format(warn), flush=True)
        perr = np.sqrt(np.diag(pcov))
    else:
        perr = np.sqrt(np.diag(pcov))

    if return_valid:
        return popt, perr, valid
    else:
        return popt, perr
