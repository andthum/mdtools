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
Plot statistics about the distribution of energy terms contained in an
`.edr file
<https://manual.gromacs.org/documentation/current/reference-manual/file-formats.html#edr>`_
(Gromacs energy file).

For each energy term selected with \--observables the following plots
are created:

    * The full evolution of the energy term with time including the
      cumulative average and the centered moving average.
    * A cutout of the above plot for the last \--num-points data points.
    * A histogram showing the distribution of the energy values.
    * The autocorrelation function (ACF) of the energy term with
      confidence intervals given by :math:`1 - \alpha`.
    * The power spectrum of the energy term, i.e. the absolute square of
      its discrete Fourier transform.

Additionally, the following characteristics of the distributions of the
selected energy terms are written to file:

    * Number of data points.
    * Sample mean.
    * Median of the sample.
    * Unbiased sample variance.
    * Minimum value of the sample.
    * Maximum value of the sample.
    * Unbiased sample skewness (Fisher-Pearson).
    * Unbiased excess sample kurtosis (Fisher).
    * Biased non-Gaussian parameter.
    * p-value from D'Agostino's and Pearson's test for normality.

Options
-------
-f
    The name of the .edr file to read.
--plot-out
    Output file name for the file that contains the plots.
--stats-out
    Output file name for the file that contains the characteristics of
    the distributions of the selected energy terms.
-b
    First frame to use from the .edr file.  Frame numbering starts at
    zero.  Default: ``0``.
-e
    Last frame to use from the .edr file.  This is exclusive, i.e. the
    last frame read is actually ``END - 1``.  A value of ``-1`` means to
    use the very last frame.  Default: ``-1``.
--every
    Use every n-th frame from the .edr file.  Default: ``1``.
--gzipped
    If given, the input file is assumed to be compressed with gzip and
    will be decompressed before processing.  Afterwards, the
    decompressed file is removed.
--observables
    A space separated list of energy terms to select.  The energy terms
    must be present in the .edr file.  If an energy term contains a
    space, like 'Kinetic En.', put it in quotes.  ``'Time'`` is not
    allowed as selection.  Default:
    ``["Potential", "Kinetic En.", "Total Energy"]``
--print-obs
    Only print all energy terms contained in the .edr file and exit.
--diff
    Use the difference between consecutive values of the energy term for
    the analysis rather than the energy term itself.
--wlen
    Window length for calculating the centered moving average.  Should
    be odd for a real *centered* moving average.  Default: ``501``.
--num-points
    The cutout plot show the last `NUM_POINTS` data points of the
    selected energy term(s) vs. time.  Must not be negative.  If
    `NUM_POINTS` is greater then the actual number of available data
    points or ``None``, it is set to the maximum number of available
    data points.  Default: ``500``.
--alpha
    Significance level for D'Agostino's and Pearson's K-squared test for
    normality of the distribution of energy values (see
    :func:`scipy.stats.normaltest`) and for the confidence intervals of
    the ACF (see :func:`mdtools.statistics.acf_confint`).  The K-squared
    test requires a sample size of more than 20 data points.  Typical
    values for :math:`\alpha` are 0.01 or 0.05.  In some cases it is set
    to 0.1 to reduce the probability of a Type 2 error, i.e. the null
    hypothesis is not rejected although it is wrong.  Here, the null
    hypothesis is that the data are normally distributed (in case of the
    K-squared test) or have no autocorrelation (in case of the ACF).
    For more details about the significance level see
    :func:`mdtools.statistics.acf_confint`.  Default: ``0.1``

See Also
--------
:func:`scipy.stats.skew` :
    Compute the sample skewness of a data set
:func:`scipy.stats.kurtosis` :
    Compute the kurtosis of a dataset
:func:`scipy.stats.normaltest` :
    Test whether a sample differs from a normal distribution
:func:`mdtools.statistics.ngp` :
    Compute the non-Gaussian parameter of a data set
:func:`mdtools.plot.correlogram`
    Create and plot a correlogram for a given data set

Notes
-----
The produced plots and distribution characteristics can be used to judge
whether the distribution of energy terms is reasonable for the simulated
ensemble.

For instance, in the canonical (:math:`NVT`) ensemble, the variance of
the total energy :math:`\sigma_{E_{tot}}^2` is related to the heat
capacity :math:`C_V` of the system at constant volume. [1]_

.. math::

    \sigma_{E_{tot}}^2 = k_B T^2 C_V

Here, :math:`k_B` is the Boltzmann constant and :math:`T` is the
temperature.  Furthermore, the variance of the kinetic energy
:math:`\sigma_{E_{kin}}^2` is related to the number of degrees of
freedom :math:`f` of the system.

.. math::

    \sigma_{E_{kin}}^2 = \frac{f}{2} \left( k_B T \right)^2

The total number of degrees of freedom of :math:`N` molecules in
:math:`d`-dimensional space is usually :math:`f = Nd`.  However, in
molecular dynamics simulations, the total momentum of the system and the
length of bonds to hydrogen atoms are typically kept constant.  Thus,
the number of degrees of freedom reduces by :math:`d` (for the total
momentum) and the number of constraints :math:`N_c`, such that :math:`f`
becomes :math:`f = d(N-1) - N_c`.

In the :math:`NpT` ensemble, the ratio of the variance of the simulation
box volume :math:`\sigma_V^2` to the average simulation box volume
:math:`\langle V \rangle` is related to the isothermal compressibility
:math:`\kappa_T` of the system. [1]_

.. math::

    \frac{\sigma_V^2}{\langle V \rangle} = k_B T \kappa_T

References
----------
.. [1] Terrell I. Hill,
    `An Introduction to Statistical Thermodynamics
    <https://www.eng.uc.edu/~beaucag/Classes/AdvancedMaterialsThermodynamics/Books/Terrell%20L.%20Hill%20-%20Introduction%20to%20Statistical%20Thermodynamics-Addison-Wesley%20Educational%20Publishers%20Inc%20(1960).pdf>`_,
    Addison-Wesley Publishing Company Inc., Reading, Massachusetts, USA,
    1960, Chapter 2-1, Pages 33-38.
"""


__author__ = "Andreas Thum"


# Standard libraries
import argparse
import gzip
import os
import shutil
import sys
import uuid
import warnings
from datetime import datetime, timedelta

# Third-party libraries
import numpy as np
import psutil
import pyedr
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText
from scipy import stats

# First-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt


if __name__ == "__main__":  # noqa: C901
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage.
    parser = argparse.ArgumentParser(
        description=(
            "Plot statistics about the distribution of energy terms contained"
            " in an Gromacs energy file (.edr).  For more information, refer"
            " to the documentation of this script."
        )
    )
    parser.add_argument(
        "-f",
        dest="EDRFILE",
        type=str,
        required=True,
        help="Gromacs energy file (.edr).",
    )
    parser.add_argument(
        "--plot-out",
        dest="PLOT_OUT",
        type=str,
        required=True,
        help="Output filename for the file that contains the plots.",
    )
    parser.add_argument(
        "--stats-out",
        dest="STATS_OUT",
        type=str,
        required=True,
        help=(
            "Output filename for the file that contains the characteristics of"
            " the distributions of the selected energy terms."
        ),
    )
    parser.add_argument(
        "-b",
        dest="BEGIN",
        type=int,
        required=False,
        default=0,
        help=(
            "First frame to use from the .edr file.  Frame numbering starts"
            " at zero.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "-e",
        dest="END",
        type=int,
        required=False,
        default=-1,
        help=(
            "Last frame to use from the .edr file (exclusive).  Default:"
            " %(default)s."
        ),
    )
    parser.add_argument(
        "--every",
        dest="EVERY",
        type=int,
        required=False,
        default=1,
        help="Use every n-th frame from the .edr file.  Default: %(default)s.",
    )
    parser.add_argument(
        "--gzipped",
        dest="GZIPPED",
        required=False,
        default=False,
        action="store_true",
        help="Input file is compressed with gzip.",
    )
    parser.add_argument(
        "--observables",
        dest="OBSERVABLES",
        type=str,
        nargs="+",
        required=False,
        default=("Potential", "Kinetic En.", "Total Energy"),
        help=(
            "Space separated list of energy terms to select.  Default:"
            " %(default)s."
        ),
    )
    parser.add_argument(
        "--print-obs",
        dest="PRINT_OBS",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Only print all energy terms contained in the .edr file and exit."
        ),
    )
    parser.add_argument(
        "--diff",
        dest="DIFF",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Use the difference between consecutive values of the energy term"
            " for the analysis rather than the energy term itself."
        ),
    )
    parser.add_argument(
        "--wlen",
        dest="WLEN",
        type=int,
        required=False,
        default=501,
        help=(
            "Window length for calculating the centered moving average."
            "  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--num-points",
        dest="NUM_POINTS",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=float),
        required=False,
        default=500,
        help=(
            "Use the last `NUM_POINTS` data points for the cutout plot."
            "  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--alpha",
        dest="ALPHA",
        type=float,
        required=False,
        default=0.1,
        help=(
            "Significance level for statistical tests and confidence"
            " intervals.  Default: %(default)s."
        ),
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    if "Time" in args.OBSERVABLES:
        raise ValueError("Illegal value for option --observables: 'Time'")
    if args.ALPHA < 0 or args.ALPHA > 1:
        raise ValueError(
            "--alpha ({}) must lie between 0 and 1".format(args.ALPHA)
        )

    print("\n")
    print("Reading input file...")
    timer = datetime.now()
    if args.GZIPPED:
        timestamp = datetime.now()
        file_decompressed = (
            "."
            + args.EDRFILE[:-7]
            + "_uuid_"
            + str(uuid.uuid4())
            + "_date_"
            + str(timestamp.strftime("%Y-%m-%d_%H-%M-%S"))
            + ".edr"
        )
        with gzip.open(args.EDRFILE, "rb") as file_in:
            with open(file_decompressed, "wb") as file_out:
                shutil.copyfileobj(file_in, file_out)
        args.EDRFILE = file_decompressed
    data = pyedr.edr_to_dict(args.EDRFILE, verbose=True)
    units = pyedr.get_unit_dictionary(args.EDRFILE)
    if args.GZIPPED:
        # Remove decompressed file.
        os.remove(file_decompressed)
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    if args.PRINT_OBS:
        print("\n")
        print("The .edr files contains the following energy terms:")
        for key in data.keys():
            print(key)
        print("\n")
        print("{} done".format(os.path.basename(sys.argv[0])))
        print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
        _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
        print("CPU time:             {}".format(_cpu_time))
        print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
        print(
            "Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc))
        )
        sys.exit(0)

    print("\n")
    print("Extracting observables...")
    timer = datetime.now()
    times = data.pop("Time")
    N_FRAMES_TOT = len(times)
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=N_FRAMES_TOT,
    )
    print("Total number of frames: {:>8d}".format(N_FRAMES_TOT))
    print("Frames to use:          {:>8d}".format(N_FRAMES))
    print("First frame to use:     {:>8d}".format(BEGIN))
    print("Last frame to use:      {:>8d}".format(END - 1))
    print("Use every n-th frame:   {:>8d}".format(EVERY))
    times = times[BEGIN:END:EVERY]
    if args.DIFF:
        times = times[1:]
    time_unit = units.pop("Time")
    for key in tuple(data.keys()):
        if key not in args.OBSERVABLES:
            data.pop(key)
            units.pop(key)
        elif args.DIFF:
            data[key] = np.diff(data[key][BEGIN:END:EVERY])
        else:
            data[key] = data[key][BEGIN:END:EVERY]
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    time_diffs = np.diff(times)
    time_diff = time_diffs[0]
    if not np.all(time_diffs >= 0):
        raise ValueError("The times in the .edr file are not sorted")
    if not np.allclose(time_diffs, time_diff, rtol=0):
        raise ValueError("The times in the .edr file are not evenly spaced")
    del time_diffs
    lag_times = np.arange(0, time_diff * len(times), time_diff)

    if args.WLEN < 1:
        raise ValueError("--wlen ({}) must be at least 1".format(args.WLEN))
    elif args.WLEN > len(times):
        raise ValueError(
            "--wlen ({}) must not be greater than the number of frames"
            " ({})".format(args.WLEN, len(times))
        )
    if args.NUM_POINTS is None:
        args.NUM_POINTS = len(times)
    elif args.NUM_POINTS < 0:
        raise ValueError(
            "--num-points ({}) must be positive'".format(args.NUM_POINTS)
        )
    args.NUM_POINTS = min(args.NUM_POINTS, len(times))

    print("\n")
    print("Calculating characteristics of the distributions...")
    timer = datetime.now()
    dist_props = {}
    for key, val in data.items():
        nobs, minmax, mean, var, skew, kurt = stats.describe(
            val, ddof=1, bias=False
        )
        median = np.median(val)
        ngp = mdt.stats.ngp(val, center=True)
        if len(val) > 20:
            # D'Agostino's and Pearson's K-squared test for
            # normality.  The test tests the null hypothesis that
            # the data are distributed normally.  If the returned
            # p-value is less than the chosen significance level
            # alpha (given by --alpha), the null hypothesis must be
            # rejected.  The test is only valid for sample sizes
            # >20.
            _, pval = stats.normaltest(val)
        else:
            pval = np.nan
            warnings.warn(
                "Could not perform D'Agostino's and Pearson's K-squared"
                " normality test for {}, because the number of samples is less"
                " than 20".format(key),
                UserWarning,
            )
        dist_props[key] = {
            "N": nobs,  # Number of data points.
            "Mean": mean,  # Sample mean.
            "Median": median,  # Median of the sample.
            "Var": var,  # Unbiased sample variance.
            "Min": minmax[0],  # Minimum value of the sample.
            "Max": minmax[1],  # Maximum value of the sample.
            "Skewness": skew,  # Unbiased sample skewness.
            "Kurtosis": kurt,  # Unbiased excess sample kurtosis.
            "NGP": ngp,  # Biased non-Gaussian parameter.
            "p-value": pval,  # p-value of the normality test.
        }
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating plots...")
    timer = datetime.now()
    if args.DIFF:
        key_prefix = r"$\Delta$"
    else:
        key_prefix = ""
    mdt.fh.backup(args.PLOT_OUT)
    with PdfPages(args.PLOT_OUT) as pdf:
        for key, val in data.items():
            print()
            print("Plotting {} vs. Time...".format(key))
            cumav = mdt.stats.cumav(val)
            movav = mdt.stats.movav(val, args.WLEN)
            fig, ax = plt.subplots(clear=True)
            lines = ax.plot(
                times, val, linestyle="", marker=".", rasterized=True
            )
            ax.plot(
                times[args.WLEN // 2 : args.WLEN // 2 + len(movav)],
                movav,
                label="Mov. Av. ({})".format(args.WLEN),
            )
            ax.plot(times, cumav, label="Cum. Av.")
            ax.set(
                xlabel="Time / " + time_unit,
                ylabel=key_prefix + key + " / " + units[key],
                xlim=(times[0], times[-1]),
            )
            ax.legend(
                loc="upper center", ncols=2, **mdtplt.LEGEND_KWARGS_XSMALL
            )
            pdf.savefig()

            print("Plotting Cutout of {} vs. Time...".format(key))
            lines[0].set_linestyle("solid")
            ax.set(xlim=(times[len(times) - args.NUM_POINTS], times[-1]))
            pdf.savefig()
            plt.close()

            print("Plotting Histogram of {}...".format(key))
            mean_fit, std_fit = stats.norm.fit(
                val,
                loc=dist_props[key]["Mean"],
                scale=np.sqrt(dist_props[key]["Var"]),
            )
            rv = stats.norm(loc=mean_fit, scale=std_fit)
            fig, ax = plt.subplots(clear=True)
            hist, bin_edges, patches = ax.hist(
                val, bins="auto", density=True, rasterized=True
            )
            bin_mids = bin_edges[1:] - np.diff(bin_edges) / 2
            lines = ax.plot(bin_mids, rv.pdf(bin_mids))
            ax.set(
                xlabel=key_prefix + key + " / " + units[key],
                ylabel="Probability",
                ylim=(0, None),
            )
            at_data = AnchoredText(
                (
                    "Data (n = {})\n".format(dist_props[key]["N"])
                    + "Mean: {:.3f}\n".format(dist_props[key]["Mean"])
                    + "Median: {:.3f}\n".format(dist_props[key]["Median"])
                    + "Var: {:.3f}\n".format(dist_props[key]["Var"])
                    + "Min: {:.3f}\n".format(dist_props[key]["Min"])
                    + "Max: {:.3f}\n".format(dist_props[key]["Max"])
                    + "Skew.: {:.3f}\n".format(dist_props[key]["Skewness"])
                    + "Kurt.: {:.3f}\n".format(dist_props[key]["Kurtosis"])
                    + "NGP: {:.3f}\n".format(dist_props[key]["NGP"])
                    + "p-value: {:.3f}".format(dist_props[key]["p-value"])
                ),
                loc="upper left",
                prop={
                    "fontsize": "xx-small",
                    "color": patches[0].get_facecolor(),
                },  # Text properties
            )
            at_data.patch.set(alpha=0.75, edgecolor="lightgrey")
            ax.add_artist(at_data)
            at_fit = AnchoredText(
                (
                    "Gaussian Fit\n"
                    + "Mean: {:.3f}\n".format(mean_fit)
                    + "Median = Mean \n"
                    + "Var: {:.3f}\n".format(std_fit**2)
                    + "Skew.: 0\n"
                    + "Kurt.: 0\n"
                    + "NGP: 0"
                ),
                loc="upper right",
                prop={
                    "fontsize": "xx-small",
                    "color": lines[0].get_color(),
                },  # Text properties
            )
            at_fit.patch.set(alpha=0.75, edgecolor="lightgrey")
            ax.add_artist(at_fit)
            pdf.savefig()
            plt.close()

            print("Plotting ACF of {}...".format(key))
            fig, ax = plt.subplots(clear=True)
            mdtplt.correlogram(
                ax, val, lag_times, siglev=args.ALPHA, rasterized=True
            )
            ax.set_xscale("log", base=10, subs=np.arange(2, 10))
            ax.set(
                xlabel="Lag Time / " + time_unit,
                ylabel="ACF of " + key_prefix + key,
                xlim=(lag_times[1], lag_times[-1]),
                ylim=(None, 1),
            )
            ax.legend(loc="upper right")
            pdf.savefig()
            plt.close()

            print("Plotting Power Spectrum of {}...".format(key))
            # The zero-frequency term is the sum of the signal => Remove
            # the mean to get a zero-frequency term that is zero.
            amplitudes = np.abs(np.fft.rfft(val - dist_props[key]["Mean"]))
            amplitudes **= 2
            frequencies = np.fft.rfftfreq(len(val), time_diff)
            fig, ax = plt.subplots(clear=True)
            ax.plot(frequencies, amplitudes, rasterized=True)
            ax.set_xscale("log", base=10, subs=np.arange(2, 10))
            ax.set(
                xlabel="Frequency / 1/" + time_unit,
                ylabel="Pow. Spec. of " + key_prefix + key,
                xlim=(frequencies[1], frequencies[-1]),
                ylim=(0, None),
            )
            pdf.savefig()
            plt.close()
    print()
    print("Created {}".format(args.PLOT_OUT))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    labels = list(list(dist_props.values())[0].keys())
    data = [list(sub_dct.values()) for sub_dct in dist_props.values()]
    data = np.array([labels] + data, dtype=object)
    data = np.column_stack(data)
    fmt = ("%-12s",) + ("%16.9e",) * (data.shape[1] - 1)
    header = (
        "Total number of frames: {:>8d}\n".format(N_FRAMES_TOT)
        + "Used frames:            {:>8d}\n".format(N_FRAMES)
        + "First used frame:       {:>8d}\n".format(BEGIN)
        + "Last used frame:        {:>8d}\n".format(END - 1)
        + "Used every n-th frame:  {:>8d}\n".format(EVERY)
        + "\n"
        + "N: Number of data points.\n"
        + "Mean: Sample mean.\n"
        + "  1/N sum_{i=1}^N x_i\n"
        + "Median: Median of the sample.\n"
        + "  For normally distributed data, mean and median are the same.\n"
        + "Var: Unbiased sample variance.\n"
        + "  s^2 = 1/(N-1) sum_{i=1}^N (x_i - mean)^2\n"
        + "Min: Minimum value of the sample.\n"
        + "Max: Maximum value of the sample.\n"
        + "Skewness: Unbiased sample skewness (Fisher-Pearson).\n"
        + "  G_1 = sqrt{N*(N-1)}/(N-2) * m_3 / m_2^(3/2)\n"
        + "      = N/[(N-1)(N-2)] sum_{i=1}^N [(x_i - mean)/s]^3\n"
        + "  with the k-th biased central moment\n"
        + "  m_k = 1/N sum_{i=1}^N (x_i - mean)^k\n"
        + "  For normally distributed data, the skewness is zero.\n"
        + "Kurtosis: Unbiased excess sample kurtosis (Fisher).\n"
        + "  G_2 = (N-1)/[(N-2)(N-3)] [(N+1) m_4/m_2^2 - 3 (N-1)]\n"
        + "      = N(N+1)/[(N-1)(N-2)(N-3)] sum_{i=1}^N [(x_i - mean)/s]^4 - 3 (N-1)^2/[(N-2)(N-3)]\n"  # noqa: E501
        + "  For normally distributed data, the Kurtosis is zero.\n"
        + "NGP: Biased non-Gaussian parameter.\n"
        + "  a = 1/3 m_4/m_2^2 - 1\n"
        + "  For normally distributed data, the NGP is zero.\n"
        + "p-value: p-value from D'Agostino's and Pearson's test for\n"
        + "  normality.  If the p-value is below the chosen significance\n"
        + "  level (--alpha), the null hypothesis the data are distributed\n"
        + "  normally must be rejected.\n"
        + "\n"
    )
    if args.DIFF:
        header += (
            "\n"
            "The given values are calculated from the difference between\n"
            "consecutive values of the given observables (--diff).\n"
        )
    col_names = ["{:>16s}".format(key) for key in dist_props.keys()]
    col_units = ["{:>16s}".format(val) for val in units.values()]
    header += "\n" + "{:<11s}".format("Observable") + " ".join(col_names)
    header += "\n" + "{:<11s}".format("Unit") + " ".join(col_units)
    mdt.fh.savetxt(args.STATS_OUT, data, fmt=fmt, header=header)
    print("Created {}".format(args.STATS_OUT))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
