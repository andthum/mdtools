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


def line(x, m=1, c=0):
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
    :func:`mdtools.functions.line_inv` :
        Inverse of a straight line

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    return m * x + c


def line_inv(y, m=1, c=0):
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
    :func:`mdtools.functions.line` :
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


def fit_exp_decay_log(xdata, ydata, ysd=None, return_valid=False, **kwargs):
    r"""
    Fit an exponential decay function to `ydata` using
    :func:`scipy.optimize.curve_fit`.

    Instead of fitting the data directly with an exponential decay
    function :math:`e^{-kt}`, the logarithm of the y-data is fitted by

    .. math::

        f(t) = -k \cdot t

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
    kwargs : dict, optional
        Additional keyword arguments (besides `xdata`, `ydata` and
        `sigma`) to parse to :func:`scipy.optimize.curve_fit`.  See
        there for possible options.  By default, `absolute_sigma` is set
        to ``True``, `p0` is set to ``1 / (len(ydata) *
        np.mean(ydata))`` and `bounds` is set to ``(0, np.inf)``.

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
    valid = np.isfinite(ydata)
    valid &= ydata > 0
    valid &= ydata <= 1
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

    kwargs.setdefault("absolute_sigma", True)
    kwargs.setdefault("p0", 1 / (len(ydata) * np.mean(ydata)))
    kwargs.setdefault("bounds", (0, np.inf))

    try:
        popt, pcov = optimize.curve_fit(
            f=exp_decay_log,
            xdata=xdata[valid],
            ydata=np.log(ydata[valid]),
            sigma=ysd,
            **kwargs,
        )
    except (ValueError, RuntimeError) as err:
        print()
        print("An error has occurred during fitting:")
        print("{}".format(err))
        print("Setting fit parameters to `numpy.nan`")
        print()
        if return_valid:
            return np.array([np.nan]), np.array([np.nan]), valid
        else:
            return np.array([np.nan]), np.array([np.nan])
    except optimize.OptimizeWarning as warn:
        print()
        print("A warning has occurred during fitting:")
        print("{}".format(warn))
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


def fit_kww(xdata, ydata, ysd=None, return_valid=False, **kwargs):
    r"""
    Fit a stretched exponential function, also known as Kohlrausch-
    Williams-Watts (KWW) function, to `ydata` using
    :func:`scipy.optimize.curve_fit`.

    The KWW function reads:

    .. math::

        f(t) = \exp{\left[ -\left( \frac{t}{\tau} \right)^\beta \right]}

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
    kwargs : dict, optional
        Additional keyword arguments (besides `xdata`, `ydata` and
        `sigma`) to parse to :func:`scipy.optimize.curve_fit`.  See
        there for possible options.  By default, `absolute_sigma` is set
        to ``True``, `p0` is set to ``[tau_init, 1]`` and `bounds` is
        set to ``([0, 0], [np.inf, 1])``.  ``tau_init`` is the point at
        which `ydata` falls below :math:`1/e`.

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

    See Also
    --------
    :func:`mdtools.functions.kww` :
        Stretched exponential function
    """
    ydata = np.asarray(ydata)
    valid = np.isfinite(ydata)
    valid &= ydata > 0
    valid &= ydata <= 1
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
    xdata = xdata[valid]
    ydata = ydata[valid]
    if ysd is not None:
        ysd = ysd[valid]
        ysd[ysd == 0] = CURVE_FIT_ZERO_SD

    if kwargs.get("p0", None) is not None:
        # Initial guess for tau.
        # If beta = 1, f(t=tau) = 1/e  =>  tau = f^(-1)(1/e)
        thresh = 1 / np.e
        ix = np.argmin(ydata <= thresh)
        if ydata[ix] > thresh:
            # `ydata` never falls below `thresh`.
            # Linearly extrapolate `ydata` to `thresh`.
            tau_init = (1 + ydata[-1] - thresh) * xdata[-1]
        else:
            tau_init = xdata[ix]
        # Ensure that `tau_init` is greater than zero.
        tau_init = max(tau_init, CURVE_FIT_ZERO_SD)
    else:
        # `p0` is already given, the value of `tau_init` is ignored.
        tau_init = xdata[-1]
    kwargs.setdefault("absolute_sigma", True)
    kwargs.setdefault("p0", [tau_init, 1])
    kwargs.setdefault("bounds", ([0, 0], [np.inf, 1]))

    try:
        popt, pcov = optimize.curve_fit(
            f=kww, xdata=xdata, ydata=ydata, sigma=ysd, **kwargs
        )
    except (ValueError, RuntimeError) as err:
        print()
        print("An error has occurred during fitting:")
        print("{}".format(err))
        print("Setting fit parameters to `numpy.nan`")
        print()
        if return_valid:
            return np.full(2, np.nan), np.full(2, np.nan), valid
        else:
            return np.full(2, np.nan), np.full(2, np.nan)
    except optimize.OptimizeWarning as warn:
        print()
        print("A warning has occurred during fitting:")
        print("{}".format(warn))
        perr = np.sqrt(np.diag(pcov))
    else:
        perr = np.sqrt(np.diag(pcov))

    if return_valid:
        return popt, perr, valid
    else:
        return popt, perr


def burr12_sf(t, tau=1, beta=1, delta=1):
    r"""
    Survival function of the Burr Type XII distribution function:

    .. math::

        S(t) =
        \frac{
            1
        }{
            \left[
                1 + \left( \frac{t}{\tau} \right)^\beta
            \right]^\delta
        }

    For :math:`\beta = 1`, the Burr Type XII distribution becomes the
    Lomax distribution and for :math:`\delta = 1` it becomes the
    log-logistic distribution.  The survival function of the Lomax
    distribution is equal to the Becquerel decay law.

    Parameters
    ----------
    t : scalar or array_like
        Value(s) at which to evaluate :math:`f(t)`.
    tau : scalar or array_like, optional
        Scale parameter(s).
    beta : scalar or array_like, optional
        Shape parameter(s).
    delta : scalar or array_like, optional
        Shape parameter(s).

    Returns
    -------
    sf : scalar or numpy.ndarray
        The outcome of :math:`S(t)`.  :math:`S(t)` is evaluated
        element-wise if at least one of the input arguments is an array.

    See Also
    --------
    :func:`mdtools.functions.burr12_sf_alt` :
        Alternative parameterization of the Burr Type XII survival
        function
    :func:`mdtools.functions.fit_burr12_sf` :
        Fit the survival function of the Burr Type XII distribution

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    return (1 + (t / tau) ** beta) ** (-delta)


def fit_burr12_sf(xdata, ydata, ysd=None, return_valid=False, **kwargs):
    r"""
    Fit the survival function of the Burr Type XII distribution to
    `ydata` using :func:`scipy.optimize.curve_fit`.

    The survival function of the Burr Type XII distribution is given by

    .. math::

        S(t) =
        \frac{
            1
        }{
            \left[
                1 + \left( \frac{t}{\tau} \right)^\beta
            \right]^\delta
        }

    For :math:`\beta = 1`, the Burr Type XII distribution becomes the
    Lomax distribution and for :math:`\delta = 1` it becomes the
    log-logistic distribution.  The survival function of the Lomax
    distribution is equal to the Becquerel decay law.

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
    kwargs : dict, optional
        Additional keyword arguments (besides `xdata`, `ydata` and
        `sigma`) to parse to :func:`scipy.optimize.curve_fit`.  See
        there for possible options.  By default, `absolute_sigma` is set
        to ``True``, `p0` is set to
        ``[tau_init, 1, 1]`` and `bounds` is set to
        ``([0, 0, 1], [np.inf, 1, 2])``.  ``tau_init`` is the point at
        which `ydata` falls below :math:`1/2`.

    Returns
    -------
    popt : numpy.ndarray
        Optimal values for the parameters so that the sum of the squared
        residuals is minimized.  The first element of `popt` is the
        optimal :math:`\tau` value, the second element is the optimal
        :math:`\beta` value, the third value is the optimal
        :math:`\delta` value.
    perr : numpy.ndarray
        Standard deviation of the optimal parameters.
    valid : numpy.ndarray
        Boolean array of the same shape as `ydata` indicating, which
        elements of `ydata` were used for the fit.  Only returned if
        `return_valid` is ``True``.

    See Also
    --------
    :func:`mdtools.functions.fit_burr12_sf_alt` :
        Fit the survival function of the Burr Type XII distribution
        using an alternative parameterization
    :func:`mdtools.functions.burr12_sf` :
        Survival function of the Burr Type XII distribution function
    """
    ydata = np.asarray(ydata)
    valid = np.isfinite(ydata)
    valid &= ydata > 0
    valid &= ydata <= 1
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
            return np.full(3, np.nan), np.full(3, np.nan), valid
        else:
            return np.full(3, np.nan), np.full(3, np.nan)
    xdata = xdata[valid]
    ydata = ydata[valid]
    if ysd is not None:
        ysd = ysd[valid]
        ysd[ysd == 0] = CURVE_FIT_ZERO_SD

    if kwargs.get("p0", None) is not None:
        # Initial guess for tau.
        # If delta = 1, S(t=tau) = 0.5  =>  tau = S^(-1)(0.5).
        thresh = 0.5
        ix = np.argmin(ydata <= thresh)
        if ydata[ix] > thresh:
            # `ydata` never falls below `thresh`.
            # Linearly extrapolate `ydata` to `thresh`.
            tau_init = (1 + ydata[-1] - thresh) * xdata[-1]
        else:
            tau_init = xdata[ix]
        # Ensure that `tau_init` is greater than zero.
        tau_init = max(tau_init, CURVE_FIT_ZERO_SD)
    else:
        # `p0` is already given, the value of `tau_init` is ignored.
        tau_init = xdata[-1]
    kwargs.setdefault("absolute_sigma", True)
    kwargs.setdefault("p0", [tau_init, 1, 1])
    kwargs.setdefault("bounds", ([0, 0, 1], [np.inf, 1, 2]))

    try:
        popt, pcov = optimize.curve_fit(
            f=burr12_sf, xdata=xdata, ydata=ydata, sigma=ysd, **kwargs
        )
    except (ValueError, RuntimeError) as err:
        print()
        print("An error has occurred during fitting:")
        print("{}".format(err))
        print("Setting fit parameters to `numpy.nan`")
        print()
        if return_valid:
            return np.full(3, np.nan), np.full(3, np.nan), valid
        else:
            return np.full(3, np.nan), np.full(3, np.nan)
    except optimize.OptimizeWarning as warn:
        print()
        print("A warning has occurred during fitting:")
        print("{}".format(warn))
        perr = np.sqrt(np.diag(pcov))
    else:
        perr = np.sqrt(np.diag(pcov))

    if return_valid:
        return popt, perr, valid
    else:
        return popt, perr


def burr12_sf_alt(t, tau=1, beta=1, d=1):
    r"""
    Survival function of the Burr Type XII distribution function:

    .. math::

        S(t) =
        \frac{
            1
        }{
            \left[
                1 + \left( \frac{t}{\tau} \right)^\beta
            \right]^{\frac{d}{\beta}}
        }

    This function uses an alternative parameterization compared to
    :func:`burr12_sf` by setting :math:`d = \beta \delta`.  The
    advantage is that one can know specify bounds for
    :math:`\beta \delta` when fitting data with this function.  One
    might want to do this, because the n-th raw moment of the Burr Type
    XII distribution, which is

    .. math::

        \langle t^n \rangle =
        \tau^n
        \frac{
            \Gamma\left( \delta - \frac{n}{\beta}\right)
            \Gamma\left( 1      + \frac{b}{\beta}\right)
        }{
            \Gamma(\delta)
        },

    only exists if :math:`\beta \delta > n`.

    Parameters
    ----------
    t : scalar or array_like
        Value(s) at which to evaluate :math:`f(t)`.
    tau : scalar or array_like, optional
        Scale parameter(s).
    beta : scalar or array_like, optional
        Shape parameter(s).
    d : scalar or array_like, optional
        Shape parameter(s).

    Returns
    -------
    sf : scalar or numpy.ndarray
        The outcome of :math:`S(t)`.  :math:`S(t)` is evaluated
        element-wise if at least one of the input arguments is an array.

    See Also
    --------
    :func:`mdtools.functions.burr12_sf` :
        Original parameterization of the Burr Type XII survival function
    :func:`mdtools.functions.fit_burr12_sf_alt` :
        Fit the survival function of the Burr Type XII distribution
        using the alternative parameterization

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    return (1 + (t / tau) ** beta) ** (-(d / beta))


def fit_burr12_sf_alt(xdata, ydata, ysd=None, return_valid=False, **kwargs):
    r"""
    Fit the survival function of the Burr Type XII distribution to
    `ydata` using :func:`scipy.optimize.curve_fit`.

    The survival function of the Burr Type XII distribution is given by

    .. math::

        S(t) =
        \frac{
            1
        }{
            \left[
                1 + \left( \frac{t}{\tau} \right)^\beta
            \right]^{\frac{d}{\beta}}
        }

    This function uses an alternative parameterization compared to
    :func:`fit_burr12_sf` by setting :math:`d = \beta \delta`.  The
    advantage is that one can know specify bounds for
    :math:`\beta \delta`.  One might want to do this, because the n-th
    raw moment of the Burr Type XII distribution, which is

    .. math::

        \langle t^n \rangle =
        \tau^n
        \frac{
            \Gamma\left( \delta - \frac{n}{\beta}\right)
            \Gamma\left( 1      + \frac{b}{\beta}\right)
        }{
            \Gamma(\delta)
        },

    only exists if :math:`\beta \delta > n`.

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
    kwargs : dict, optional
        Additional keyword arguments (besides `xdata`, `ydata` and
        `sigma`) to parse to :func:`scipy.optimize.curve_fit`.  See
        there for possible options.  By default, `absolute_sigma` is set
        to ``True``, `p0` is set to
        ``[tau_init, 1, 1 + tol_d]`` and `bounds` is set to
        ``([0, 0, 1 + tol_d], [np.inf, 1, 2 + tol_d])`` where ``tol_d``
        is set to ``1e-6``.  ``tau_init`` is the point at which `ydata`
        falls below :math:`1/2`.

    Returns
    -------
    popt : numpy.ndarray
        Optimal values for the parameters so that the sum of the squared
        residuals is minimized.  The first element of `popt` is the
        optimal :math:`\tau` value, the second element is the optimal
        :math:`\beta` value, the third value is the optimal :math:`d`
        value.
    perr : numpy.ndarray
        Standard deviation of the optimal parameters.
    valid : numpy.ndarray
        Boolean array of the same shape as `ydata` indicating, which
        elements of `ydata` were used for the fit.  Only returned if
        `return_valid` is ``True``.

    See Also
    --------
    :func:`mdtools.functions.fit_burr12_sf` :
        Fit the survival function of the Burr Type XII distribution
        using the original parameterization
    :func:`mdtools.functions.burr12_sf_alt` :
        Survival function of the Burr Type XII distribution function
        using the alternative parameterization
    """
    ydata = np.asarray(ydata)
    valid = np.isfinite(ydata)
    valid &= ydata > 0
    valid &= ydata <= 1
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
            return np.full(3, np.nan), np.full(3, np.nan), valid
        else:
            return np.full(3, np.nan), np.full(3, np.nan)
    xdata = xdata[valid]
    ydata = ydata[valid]
    if ysd is not None:
        ysd = ysd[valid]
        ysd[ysd == 0] = CURVE_FIT_ZERO_SD

    if kwargs.get("p0", None) is not None:
        # Initial guess for tau.
        # If beta = d = 1, S(t=tau) = 0.5  =>  tau = S^(-1)(0.5).
        thresh = 0.5
        ix = np.argmin(ydata <= thresh)
        if ydata[ix] > thresh:
            # `ydata` never falls below `thresh`.
            # Linearly extrapolate `ydata` to `thresh`.
            tau_init = (1 + ydata[-1] - thresh) * xdata[-1]
        else:
            tau_init = xdata[ix]
        # Ensure that `tau_init` is greater than zero.
        tau_init = max(tau_init, CURVE_FIT_ZERO_SD)
    else:
        # `p0` is already given, the value of `tau_init` is ignored.
        tau_init = xdata[-1]
    tol_d = 1e-6
    kwargs.setdefault("absolute_sigma", True)
    kwargs.setdefault("p0", [tau_init, 1, 1 + tol_d])
    kwargs.setdefault(
        "bounds",
        ([0, 0, 1 + tol_d], [np.inf, 1, 2 + tol_d]),
    )

    try:
        popt, pcov = optimize.curve_fit(
            f=burr12_sf_alt, xdata=xdata, ydata=ydata, sigma=ysd, **kwargs
        )
    except (ValueError, RuntimeError) as err:
        print()
        print("An error has occurred during fitting:")
        print("{}".format(err))
        print("Setting fit parameters to `numpy.nan`")
        print()
        if return_valid:
            return np.full(3, np.nan), np.full(3, np.nan), valid
        else:
            return np.full(3, np.nan), np.full(3, np.nan)
    except optimize.OptimizeWarning as warn:
        print()
        print("A warning has occurred during fitting:")
        print("{}".format(warn))
        perr = np.sqrt(np.diag(pcov))
    else:
        perr = np.sqrt(np.diag(pcov))

    if return_valid:
        return popt, perr, valid
    else:
        return popt, perr
