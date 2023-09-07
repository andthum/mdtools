#!/usr/bin/env python3

# This file is part of MDTools.
# Copyright (C) 2021-2023  The MDTools Development Team and all
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


r"""
Plot the PDF, CDF, Survival and Hazard function of the generalized gamma
distribution or the Burr Type XII distribution for a given set of scale
and shape parameters.

See Also
--------
:func:`scipy.stats.gengamma` :
    A generalized gamma continuous random variable.
:func:`scipy.stats.burr12` :
    A Burr Type XII continuous random variable.
`Generalized Gamma Distribution`_ :
    Information about the generalized gamma distribution as implemented
    in SciPy.
`Burr12 Distribution`_ :
    Information about the Burr Type XII distribution as implemented in
    SciPy.


.. _Generalized Gamma Distribution:
    https://docs.scipy.org/doc/scipy/tutorial/stats/continuous_gengamma.html
.. _Burr12 Distribution:
    https://docs.scipy.org/doc/scipy/tutorial/stats/continuous_burr12.html

Notes
-----
Generalized Gamma Distribution

    Probability density function (PDF):

    .. math::

        f(t) =
        \frac{1}{\Gamma\left( \frac{\delta}{\beta} \right)}
        \frac{\beta}{\tau_0}
        \left( \frac{t}{\tau_0} \right)^{\delta - 1}
        \exp{\left[ -\left( \frac{t}{\tau_0} \right)^\beta \right]}

    with shape parameters :math:`\beta, \delta > 0`, scale parameter
    :math:`\tau_0 > 0` and :math:`t \geq 0`.

    n-th raw moment:

    .. math::

        \langle t^n \rangle =
        \int_0^\infty t^n f(t) dt =
        n \int_0^\infty t^{n-1} S(t) dt =
        \tau_0^n
        \frac{
            \Gamma\left( \frac{\delta + n}{\beta} \right)
        }{
            \Gamma\left( \frac{\delta}{\beta} \right)
        }

    Cumulative distribution function (CDF):

    .. math::

        F(t) = \int_0^t f(t') dt'

    Survival function:

    .. math::

        S(t) = 1 - F(t)

    Hazard function:

    .. math::

        h(t) =
        \frac{f(t)}{S(t)} =
        -\frac{S'(t)}{S(t)} =
        -\frac{\text{d } \ln{S(t)}}{\text{d}t}

    The PDF implemented by
    :func:`scipy.stats.gengamma(x, a, c, loc, scale)` is

    .. math::

        f(y) =
        \frac{|c| y^{ca-1}}{scale \cdot \Gamma(a)}
        \exp{\left(-y^c\right)}

    with :math:`y = (x - loc) / scale`.  To bring this form of the PDF
    to the one stated above, the following conversions must be applied:

       * :math:`x` -> :math:`t`
       * :math:`a` -> :math:`\delta / \beta`
       * :math:`c` -> :math:`\beta`
       * :math:`loc` -> :math:`0`
       * :math:`scale` -> :math:`\tau_0`

Burr Type XII Distribution

    Probability density function (PDF):

    .. math::

        f(t) =
        \frac{\beta \delta}{\tau_0}
        \left( \frac{t}{\tau_0} \right)^{\beta - 1}
        \frac{
            1
        }{
            \left[
                1 + \left( \frac{t}{\tau_0} \right)^\beta
            \right]^{\delta - 1}
        }

    with shape parameters :math:`\beta, \delta > 0`, scale parameter
    :math:`\tau_0 > 0` and :math:`t \geq 0`.

    n-th raw moment:

    .. math::

        \langle t^n \rangle =
        \int_0^\infty t^n f(t) dt =
        n \int_0^\infty t^{n-1} S(t) dt =
        \tau_0^n
        \frac{
            \Gamma\left( \delta - \frac{n}{\beta} \right)
            \Gamma\left( 1      + \frac{n}{\beta} \right)
        }{
            \Gamma(\delta)
        }

    only exists if :math:`n < \beta \delta`.

    Cumulative distribution function (CDF):

    .. math::

        F(t) =
        \int_0^t f(t') dt' =
        1 -
        \frac{
            1
        }{
            \left[
                1 + \left( \frac{t}{\tau_0} \right)^\beta
            \right]^\delta
        }

    Survival function:

    .. math::

        S(t) =
        1 - F(t) =
        \frac{
            1
        }{
            \left[
                1 + \left( \frac{t}{\tau_0} \right)^\beta
            \right]^\delta
        }

    Hazard function:

    .. math::

        h(t) =
        \frac{f(t)}{S(t)} =
        -\frac{S'(t)}{S(t)} =
        -\frac{\text{d } \ln{S(t)}}{\text{d}t} =
        \frac{\beta \delta}{\tau_0}
        \left( \frac{t}{\tau_0} \right)^{beta-1}
        \frac{1}{1 + \left( \frac{t}{\tau_0} \right)^\beta}

    The PDF implemented by
    :func:`scipy.stats.burr12(x, c, d, loc, scale)` is

    .. math::

        f(y) =
        \frac{c d}{scale}
        \frac{x^{c-1}}{\left( 1 + y^c \right)^{d+1}}

    with :math:`y = (x - loc) / scale`.  To bring this form of the PDF
    to the one stated above, the following conversions must be applied:

       * :math:`x` -> :math:`t`
       * :math:`c` -> :math:`\beta`
       * :math:`d` -> :math:`\delta`
       * :math:`loc` -> :math:`0`
       * :math:`scale` -> :math:`\tau_0`

"""


__author__ = "Andreas Thum"


# Standard libraries
import argparse
import os
import sys

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import burr12, gengamma

# First-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # noqa: F401; Import MDTools plot style


def get_colors(labels):
    """Get colors for the given labels."""
    cmap = plt.get_cmap()
    c_vals = np.arange(len(labels))
    c_norm = len(labels) - 1
    colors = cmap(c_vals / c_norm)
    return colors


def replot_xlogscale(out, ax, x):
    """Re-plot the given axes in x-log scale."""
    ax.set_xlim((x[1], x[-1]))
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    out.savefig()


def plot_pdf(out, x, random_variables, labels, legend_title, ylim=(0, 1.6)):
    """Plot probability density function."""
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=get_colors(labels))
    for i, rv in enumerate(random_variables):
        ax.plot(x, rv.pdf(x), label=labels[i])
    ax.set(xlabel="Time", ylabel="PDF", xlim=(x[0], x[-1]), ylim=ylim)
    legend = ax.legend(
        title=legend_title, loc="upper right", **mdtplt.LEGEND_KWARGS_XSMALL
    )
    legend.get_title().set_multialignment("center")
    out.savefig()
    replot_xlogscale(out, ax, x)
    plt.close()


def plot_cdf(out, x, random_variables, labels, legend_title, ylim=(0, 1.05)):
    """Plot cumulative distribution function."""
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=get_colors(labels))
    for i, rv in enumerate(random_variables):
        ax.plot(x, rv.cdf(x), label=labels[i])
    ax.set(xlabel="Time", ylabel="CDF", xlim=(x[0], x[-1]), ylim=ylim)
    legend = ax.legend(
        title=legend_title, loc="lower right", **mdtplt.LEGEND_KWARGS_XSMALL
    )
    legend.get_title().set_multialignment("center")
    out.savefig()
    replot_xlogscale(out, ax, x)
    plt.close()


def plot_sf(out, x, random_variables, labels, legend_title, ylim=(0, 1.05)):
    """Plot survival function."""
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=get_colors(labels))
    for i, rv in enumerate(random_variables):
        ax.plot(x, rv.sf(x), label=labels[i])
    ax.set(
        xlabel="Time",
        ylabel="Survival Function",
        xlim=(x[0], x[-1]),
        ylim=ylim,
    )
    legend = ax.legend(
        title=legend_title, loc="upper right", **mdtplt.LEGEND_KWARGS_XSMALL
    )
    legend.get_title().set_multialignment("center")
    out.savefig()
    replot_xlogscale(out, ax, x)
    plt.close()


def plot_hz(out, x, random_variables, labels, legend_title, ylim=(0, 8)):
    """Plot hazard function."""
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=get_colors(labels))
    for i, rv in enumerate(random_variables):
        ax.plot(x, rv.pdf(x) / rv.sf(x), label=labels[i])
    ax.set(
        xlabel="Time", ylabel="Hazard Function", xlim=(x[0], x[-1]), ylim=ylim
    )
    legend = ax.legend(
        title=legend_title, loc="upper center", **mdtplt.LEGEND_KWARGS_XSMALL
    )
    legend.get_title().set_multialignment("center")
    out.savefig()
    replot_xlogscale(out, ax, x)
    plt.close()


def plot_all(*args, **kwargs):
    """Plot all functions."""
    plot_pdf(*args, **kwargs)
    plot_cdf(*args, **kwargs)
    plot_sf(*args, **kwargs)
    plot_hz(*args, **kwargs)


if __name__ == "__main__":
    print(mdt.rti.run_time_info_str())

    parser = argparse.ArgumentParser(
        description=(
            "Plot the PDF, CDF, Survival and Hazard function of the"
            " generalized gamma distribution for a given set of shape and"
            " scale parameters."
        )
    )
    parser.add_argument(
        "--param-set",
        dest="PARAM_SET",
        type=str,
        choices=(
            "generalized_gamma",
            "exponential",
            "stretched_exponential",
            "weibull",
            "gamma",
            "chi",
            "burr12",
            "log_logistic",
            "lomax",
            "all",
        ),
        required=False,
        default="generalized_gamma",
        help="Name of the parameter set to use.",
    )
    args = parser.parse_args()
    outfile = args.PARAM_SET + "_distribution.pdf"

    parameters = np.array([0.25, 0.50, 1.00, 2.00, 4.00])

    # if args.PARAM_SET == "generalized_gamma":
    #     t_min, t_max = 0, 4
    #     ylim_pdf = (0, 1.05)
    #     ylim_hz = (0, 4)
    # elif args.PARAM_SET == "exponential":
    #     t_min, t_max = 0, 8
    #     ylim_pdf = (0, 1.2)
    #     ylim_hz = (0, t_max)
    # elif args.PARAM_SET == "stretched_exponential":
    #     t_min, t_max = 0, 10
    #     ylim_pdf = (0, 1.2)
    #     ylim_hz = (0, t_max)
    # elif args.PARAM_SET == "weibull":
    #     t_min, t_max = 0, 4
    #     ylim_pdf = (0, 1.6)
    #     ylim_hz = (0, t_max)
    # elif args.PARAM_SET == "gamma":
    #     t_min, t_max = 0, 20
    #     ylim_pdf = (0, 1.6)
    #     ylim_hz = (0, 8)
    # elif args.PARAM_SET == "chi":
    #     t_min, t_max = 0, 4
    #     ylim_pdf = (0, 1.05)
    #     ylim_hz = (0, 4)
    # elif args.PARAM_SET == "burr12":
    #     t_min, t_max = 0, 4
    #     ylim_pdf = (0, 1.05)
    #     ylim_hz = (0, 4)
    # elif args.PARAM_SET == "log_logistic":
    #     t_min, t_max = 0, 4
    #     ylim_pdf = (0, 1.2)
    #     ylim_hz = (0, 3)
    # elif args.PARAM_SET == "lomax":
    #     t_min, t_max = 0, 4
    #     ylim_pdf = (0, 1.05)
    #     ylim_hz = (0, 4)
    # else:
    #     t_min, t_max = 0, 8
    #     ylim_pdf = (0, 1.6)
    #     ylim_hz = (0, 8)

    t_min, t_max = 0, 8
    n_samples = (t_max - t_min) * 100 + 1
    times = np.linspace(t_min, t_max, n_samples)

    xlabel = r"$t$"
    xlim = (t_min, t_max)

    print("\n")
    print("Creating plot(s)...")
    mdt.fh.backup(outfile)
    with PdfPages(outfile) as out:
        if args.PARAM_SET in ("generalized_gamma", "all"):
            ############################################################
            # Generalized gamma distribution
            # beta = 0.5, delta = 0.5, tau0 = 1.0     => Weibull
            # beta = 0.5, delta = 1.0, tau0 = 1.0     => Stretched Exp.
            # beta = 1.0, delta = 0.5, tau0 = 2.0     => Gamma/Chi^2
            # beta = 1.0, delta = 1.0, tau0 = 1.0     => Exponential
            # beta = 2.0, delta = 1.0, tau0 = sqrt(2) => Half-normal/Chi
            # beta = 2.0, delta = 2.0, tau0 = sqrt(2) => Rayleigh/Chi
            ############################################################
            print("Generalized gamma distribution")
            betas = np.array([0.5, 0.5, 1.0, 1.0, 2.0, 2.0])
            deltas = np.array([0.5, 1.0, 0.5, 1.0, 1.0, 2.0])
            tau0s = np.array([1.0, 1.0, 2.0, 1.0, np.sqrt(2), np.sqrt(2)])
            rv = [
                gengamma(a=deltas[i] / beta, c=beta, loc=0, scale=tau0s[i])
                for i, beta in enumerate(betas)
            ]
            labels = [
                r"$\beta = %.2f$, $\delta = %.2f$, $\tau_0 = %.2f$"
                % (beta, deltas[i], tau0s[i])
                for i, beta in enumerate(betas)
            ]
            legend_title = "Generalized Gamma Dist."
            plot_all(out, times, rv, labels=labels, legend_title=legend_title)

        if args.PARAM_SET in ("exponential", "all"):
            ############################################################
            # Dependency on tau0
            # beta = delta = 1 => Exponential distribution
            ############################################################
            print("Exponential distribution")
            beta = 1.0
            delta = 1.0
            tau0s = parameters
            # tau0s = np.array([1e1, 1e2, 1e3, 1e4, 1e5])
            rv = [
                gengamma(a=delta / beta, c=beta, loc=0, scale=tau0)
                for tau0 in tau0s
            ]
            labels = [r"$%.2f$" % tau0 for tau0 in tau0s]
            # labels = [r"$10^{%d}$" % np.log10(tau0) for tau0 in tau0s]
            legend_title = (
                "Exp. Dist.\n"
                + r"$\beta = \delta = %.2f$" % beta
                + "\n"
                + r"$\tau_0$"
            )
            plot_all(out, times, rv, labels=labels, legend_title=legend_title)

        if args.PARAM_SET in ("stretched_exponential", "all"):
            ############################################################
            # Dependency on beta
            # delta = 1 => Stretched exponential distribution
            ############################################################
            print("Stretched exponential distribution")
            betas = parameters
            delta = 1.0
            tau0 = 1.0
            rv = [
                gengamma(a=delta / beta, c=beta, loc=0, scale=tau0)
                for beta in betas
            ]
            labels = [r"$%.2f$" % beta for beta in betas]
            legend_title = (
                "Str. Exp. Dist.\n"
                + r"$\delta = %.2f$, $\tau_0 = %.2f$" % (delta, tau0)
                + "\n"
                + r"$\beta$"
            )
            plot_all(out, times, rv, labels=labels, legend_title=legend_title)

        if args.PARAM_SET in ("weibull", "all"):
            ############################################################
            # Dependency on beta
            # delta = beta => Weibull distribution
            ############################################################
            print("Weibull distribution")
            betas = parameters
            deltas = betas
            tau0 = 1.0
            rv = [
                gengamma(a=deltas[i] / beta, c=beta, loc=0, scale=tau0)
                for i, beta in enumerate(betas)
            ]
            labels = [r"$%.2f$" % beta for beta in betas]
            legend_title = (
                "Weibull Dist.\n"
                + r"$\tau_0 = %.2f$" % tau0
                + "\n"
                + r"$\delta = \beta$"
            )
            plot_all(out, times, rv, labels=labels, legend_title=legend_title)

        if args.PARAM_SET in ("gamma", "all"):
            ############################################################
            # Dependency on delta
            # beta = 1 => Gamma distribution
            ############################################################
            print("Gamma distribution")
            beta = 1.0
            deltas = parameters
            tau0 = 1.0
            rv = [
                gengamma(a=delta / beta, c=beta, loc=0, scale=tau0)
                for delta in deltas
            ]
            labels = [r"$%.2f$" % delta for delta in deltas]
            legend_title = (
                "Gamma Dist.\n"
                + r"$\beta = %.2f$, $\tau_0 = %.2f$" % (beta, tau0)
                + "\n"
                + r"$\delta$"
            )
            plot_all(out, times, rv, labels=labels, legend_title=legend_title)

        if args.PARAM_SET in ("chi", "all"):
            ############################################################
            # Dependency on delta
            # beta = 2, tau0 = sqrt(2) => Chi distribution
            ############################################################
            print("Chi distribution")
            beta = 2.0
            deltas = parameters
            tau0 = np.sqrt(2)
            rv = [
                gengamma(a=delta / beta, c=beta, loc=0, scale=tau0)
                for delta in deltas
            ]
            labels = [r"$%.2f$" % delta for delta in deltas]
            legend_title = (
                "Chi Dist.\n"
                + r"$\beta = %.2f$, $\tau_0 = \sqrt{2}$" % beta
                + "\n"
                + r"$\delta$"
            )
            plot_all(out, times, rv, labels=labels, legend_title=legend_title)

        if args.PARAM_SET in ("burr12", "all"):
            ############################################################
            # Burr Type XII Distribution
            # beta = 0.5, delta = 0.5, tau0 = 1.0
            # beta = 0.5, delta = 1.0, tau0 = 1.0 => log-logistic
            # beta = 1.0, delta = 0.5, tau0 = 2.0 => Lomax
            # beta = 1.0, delta = 1.0, tau0 = 1.0 => Lomax/log-logistic
            # beta = 2.0, delta = 1.0, tau0 = sqrt(2) => log-logistic
            # beta = 2.0, delta = 2.0, tau0 = sqrt(2)
            ############################################################
            print("Burr Type XII Distribution")
            betas = np.array([0.5, 0.5, 1.0, 1.0, 2.0, 2.0])
            deltas = np.array([0.5, 1.0, 0.5, 1.0, 1.0, 2.0])
            tau0s = np.array([1.0, 1.0, 2.0, 1.0, np.sqrt(2), np.sqrt(2)])
            rv = [
                burr12(c=beta, d=deltas[i], loc=0, scale=tau0s[i])
                for i, beta in enumerate(betas)
            ]
            labels = [
                r"$\beta = %.2f$, $\delta = %.2f$, $\tau_0 = %.2f$"
                % (beta, deltas[i], tau0s[i])
                for i, beta in enumerate(betas)
            ]
            legend_title = "Burr Type XII Dist."
            plot_all(out, times, rv, labels=labels, legend_title=legend_title)

        if args.PARAM_SET in ("log_logistic", "all"):
            ############################################################
            # Dependency on beta
            # delta = 1 => Log-logistic distribution
            ############################################################
            print("Log-logistic distribution")
            betas = parameters
            delta = 1.0
            tau0 = 1.0
            rv = [burr12(c=beta, d=delta, loc=0, scale=tau0) for beta in betas]
            labels = [r"$%.2f$" % beta for beta in betas]
            legend_title = (
                "Log-Lgstc. Dist.\n"
                + r"$\delta = %.2f$, $\tau_0 = %.2f$" % (delta, tau0)
                + "\n"
                + r"$\beta$"
            )
            plot_all(out, times, rv, labels=labels, legend_title=legend_title)

        if args.PARAM_SET in ("lomax", "all"):
            ############################################################
            # Dependency on delta
            # beta = 1 => Lomax distribution
            ############################################################
            print("Lomax distribution")
            beta = 1.0
            deltas = parameters
            tau0 = 1.0
            rv = [
                burr12(c=beta, d=delta, loc=0, scale=tau0) for delta in deltas
            ]
            labels = [r"$%.2f$" % delta for delta in deltas]
            legend_title = (
                "Lomax Dist.\n"
                + r"$\beta = %.2f$, $\tau_0 = %.2f$" % (beta, tau0)
                + "\n"
                + r"$\delta$"
            )
            plot_all(out, times, rv, labels=labels, legend_title=legend_title)

    print("Created {}".format(outfile))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
