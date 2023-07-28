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
distribution for a given set of shape and scale parameters.

See Also
--------
:func:`scipy.stats.gengamma` :
    A generalized gamma continuous random variable.  Function used to
    sample the PDF.
`SciPy User Guide`_ :
    Information about the generalized gamma distribution.

.. _SciPy User Guide:
    https://docs.scipy.org/doc/scipy/tutorial/stats/continuous_gengamma.html

Notes
-----
Probability density function (PDF) of the generalized gamma function:

.. math::

    f(t) =
    \frac{1}{\Gamma\left( \frac{\delta}{\beta} \right)}
    \frac{\beta}{\theta}
    \left( \frac{t}{\theta} \right)^{\delta - 1}
    \exp{\left[ \left( \frac{t}{\theta} \right)^\beta \right]}

with :math:`\delta, \beta, \theta > 0` and :math:`t \geq 0`.

n-th raw moment:

.. math::

    \langle t^n \rangle =
    \int_0^\infty t^n f(t) dt =
    \theta^n
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
:func:`scipy.stats.gengamma(x, a, c, loc=0, scale)` is

.. math::

    f(y) = |c| * y^(ca-1) * exp(-y^c) / [Gamma(a) * scale]

with :math:`y = (x - loc) / scale`.  To bring this to this form of the
PDF to the one stated above, the following conversions must be applied:

   * x -> t
   * a -> delta / beta
   * c -> beta
   * loc -> 0
   * scale -> theta

"""


__author__ = "Andreas Thum"


# Standard libraries
import os
import sys

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gengamma

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


# Good `ylim`: (0, 1.05), (0, 1.2), (0, 1.6)
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


# Good `ylim`: (0, `t_max`)
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

    outfile = "gen_gamma_dist.pdf"
    parameters = np.array([0.25, 0.50, 1.00, 2.00, 4.00])

    # Good `t_max`: 4, 8, 10, 20
    t_min, t_max = 0, 8
    n_samples = (t_max - t_min) * 100 + 1
    times = np.linspace(t_min, t_max, n_samples)

    xlabel = r"$t$"
    xlim = (t_min, t_max)

    print("\n")
    print("Creating plot(s)...")
    mdt.fh.backup(outfile)
    with PdfPages(outfile) as out:
        ################################################################
        # Different distributions
        # delta = 0.5, beta = 0.5, theta = 1.0     => Weibull
        # delta = 0.5, beta = 1.0, theta = 2.0     => Gamma/Chi-Squared
        # delta = 1.0, beta = 0.5, theta = 1.0     => Stretched Exp.
        # delta = 1.0, beta = 1.0, theta = 1.0     => Exponential
        # delta = 1.0, beta = 2.0, theta = sqrt(2) => Half-normal / Chi
        # delta = 2.0, beta = 2.0, theta = sqrt(2) => Rayleigh / Chi
        ################################################################
        print("Different distributions")
        deltas = np.array([0.5, 0.5, 1.0, 1.0, 1.0, 2.0])
        betas = np.array([0.5, 1.0, 0.5, 1.0, 2.0, 2.0])
        thetas = np.array([1.0, 2.0, 1.0, 1.0, np.sqrt(2), np.sqrt(2)])
        rv_gengamma = [
            gengamma(a=deltas[i] / beta, c=beta, loc=0, scale=thetas[i])
            for i, beta in enumerate(betas)
        ]
        labels = [
            r"$\delta = %.2f$, $\beta = %.2f$, $\theta = %.2f$"
            % (deltas[i], beta, thetas[i])
            for i, beta in enumerate(betas)
        ]
        legend_title = ""
        plot_all(
            out, times, rv_gengamma, labels=labels, legend_title=legend_title
        )

        ################################################################
        # Dependency on beta
        # delta = 1 => Stretched-exponential distribution
        ################################################################
        print("Stretched-exponential distribution")
        delta = 1.0
        betas = parameters
        theta = 1.0
        rv_gengamma = [
            gengamma(a=delta / beta, c=beta, loc=0, scale=theta)
            for beta in betas
        ]
        labels = [r"$%.2f$" % beta for beta in betas]
        legend_title = (
            r"$\delta = %.2f$, $\theta = %.2f$" % (delta, theta)
            + "\n"
            + r"$\beta$"
        )
        plot_all(
            out, times, rv_gengamma, labels=labels, legend_title=legend_title
        )

        ################################################################
        # Dependency on delta
        # beta = 1 => Gamma distribution
        ################################################################
        print("Gamma distribution")
        deltas = parameters
        beta = 1.0
        theta = 1.0
        rv_gengamma = [
            gengamma(a=delta / beta, c=beta, loc=0, scale=theta)
            for delta in deltas
        ]
        labels = [r"$%.2f$" % delta for delta in deltas]
        legend_title = (
            r"$\beta = %.2f$, $\theta = %.2f$" % (beta, theta)
            + "\n"
            + r"$\delta$"
        )
        plot_all(
            out, times, rv_gengamma, labels=labels, legend_title=legend_title
        )

        ################################################################
        # Dependency on delta
        # beta = 2, theta = sqrt(2) => Chi distribution
        ################################################################
        print("Chi distribution")
        deltas = parameters
        beta = 2.0
        theta = np.sqrt(2)
        rv_gengamma = [
            gengamma(a=delta / beta, c=beta, loc=0, scale=theta)
            for delta in deltas
        ]
        labels = [r"$%.2f$" % delta for delta in deltas]
        legend_title = (
            r"$\beta = %.2f$, $\theta = \sqrt{2}$" % beta + "\n" + r"$\delta$"
        )
        plot_all(
            out, times, rv_gengamma, labels=labels, legend_title=legend_title
        )

        ################################################################
        # Dependency on delta
        # beta = delta => Weibull distribution
        ################################################################
        print("Weibull distribution")
        deltas = parameters
        betas = deltas
        theta = 1.0
        rv_gengamma = [
            gengamma(a=deltas[i] / beta, c=beta, loc=0, scale=theta)
            for i, beta in enumerate(betas)
        ]
        labels = [r"$%.2f$" % delta for delta in deltas]
        legend_title = r"$\theta = %.2f$" % theta + "\n" + r"$\delta = \beta$"
        plot_all(
            out, times, rv_gengamma, labels=labels, legend_title=legend_title
        )

    print("Created {}".format(outfile))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
