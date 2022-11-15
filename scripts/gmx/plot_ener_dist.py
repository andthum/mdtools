#!/usr/bin/env python3

# This file is part of MDTools.
# Copyright (C) 2021, 2022  The MDTools Development Team and all
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

    * The evolution of the energy term with time.
    * A histogram showing the distribution of the energy values.
    * The autocorrelation function (ACF) of the energy term with
      confidence intervals given by :math:`1 - \alpha`.
    * The power spectrum of the energy term, i.e. the absolute square of
      its discrete Fourier transform.

Options
-------
-f
    The name of the .edr file to read.
-o
    Output file name.
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
    ``["Potential", "Kinetic En.", "Pressure"]``
--print-obs
    Only print all energy terms contained in the .edr file and exit.
--diff
    Use the difference between consecutive values of the energy term for
    the analysis rather than the energy term itself.
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
--num-points
    Use only the last `NUM_POINTS` data points when ploting the energy
    terms vs. time.  Must not be negative.  If `NUM_POINTS` is greater
    then the actual number of available data points or ``None``, it is
    set to the maximum number of available data points.  Default:
    ``None``

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
The produced plots can be used to judge wether the distribution of
energy terms is reasonable for the simulated ensemble.
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
            " to the documetation of this script."
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
        "-o",
        dest="OUTFILE",
        type=str,
        required=True,
        help="Output filename.",
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
        default=("Potential", "Kinetic En.", "Pressure"),
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
    parser.add_argument(
        "--num-points",
        dest="NUM_POINTS",
        type=int,
        required=False,
        default=None,
        help=(
            "Use only the last `NUM_POINTS` data points when ploting the"
            " energy terms vs. time.  Default: %(default)s."
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
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=len(times),
    )
    print("Total number of frames: {:>8d}".format(len(times)))
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

    if args.NUM_POINTS is None:
        args.NUM_POINTS = len(times)
    elif args.NUM_POINTS < 0:
        raise ValueError(
            "--num-points ({}) must be positive'".format(args.NUM_POINTS)
        )
    args.NUM_POINTS = min(args.NUM_POINTS, len(times))
    if args.NUM_POINTS > 1000:
        # Force rasterized (bitmap) drawing for vector graphics output.
        # This leads to smaller files for plots with many data points.
        rasterized = True
    else:
        rasterized = False

    print("\n")
    print("Processing data and creating plots...")
    timer = datetime.now()
    if args.DIFF:
        key_prefix = r"$\Delta$"
    else:
        key_prefix = ""
    n_gauss_warnings = 0
    non_gaussian_observables = []
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        for key, val in data.items():
            print()
            print("Plotting {} vs. Time...".format(key))
            fig, ax = plt.subplots(clear=True)
            ax.plot(times, val, rasterized=rasterized)
            ax.set(
                xlabel="Time / " + time_unit,
                ylabel=key_prefix + key + " / " + units[key],
                xlim=(times[len(times) - args.NUM_POINTS], times[-1]),
            )
            pdf.savefig()
            plt.close()

            print("Plotting histogram of {}...".format(key))
            # Statistical analysis.
            nobs, minmax, mean, var, skewness, kurtosis = stats.describe(
                val, ddof=1, bias=False
            )
            std = np.sqrt(var)
            median = np.median(val)
            ngp = mdt.stats.ngp(val, center=True)
            if len(val) > 20:
                # D'Agostino's and Pearson's K-squared test for
                # normality.  The test is only valid for sample sizes
                # > 20.
                _, pval = stats.normaltest(val)
                if pval < args.ALPHA:
                    n_gauss_warnings += 1
                    non_gaussian_observables.append(key)
                    warnings.warn(
                        "The null hypothesis that the {} is normally"
                        " distributed is rejected (p-value: {}, significance"
                        " level: {})".format(key, pval, args.ALPHA),
                        UserWarning,
                    )
            else:
                pval = np.nan
                n_gauss_warnings += 1
                warnings.warn(
                    "Could not perform the normality test the {}, because the"
                    " number of samples is less than 20".format(key),
                    UserWarning,
                )
            mean_fit, std_fit = stats.norm.fit(val, loc=mean, scale=std)
            rv = stats.norm(loc=mean_fit, scale=std_fit)
            # Plot figure.
            fig, ax = plt.subplots(clear=True)
            hist, bin_edges, patches = ax.hist(
                val, bins="auto", density=True, rasterized=True
            )
            bin_mids = bin_edges[1:] - np.diff(bin_edges)
            lines = ax.plot(bin_mids, rv.pdf(bin_mids))
            ax.set(
                xlabel=key_prefix + key + " / " + units[key],
                ylabel="Probability",
                ylim=(0, None),
            )
            at_data = AnchoredText(
                (
                    "Data (n = {})\n".format(nobs)
                    + "Mean: {:.3f}\n".format(mean)
                    + "Median: {:.3f}\n".format(median)
                    + "Min: {:.3f}\n".format(minmax[0])
                    + "Max: {:.3f}\n".format(minmax[1])
                    + "StD: {:.3f}\n".format(std)
                    + "Skew.: {:.3f}\n".format(skewness)
                    + "Kurt.: {:.3f}\n".format(kurtosis)
                    + "NGP: {:.3f}\n".format(ngp)
                    + "p-value: {:.3f}".format(pval)
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
                    + "StD: {:.3f}\n".format(std_fit)
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
            # The zero-frequence term is the sum of the signal => Remove
            # the mean to get a zero-frequence term that is zero.
            amplitudes = np.abs(np.fft.rfft(val - mean)) ** 2
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
    print("Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    if n_gauss_warnings > 0:
        print("\n")
        print("!" * 72)
        print(
            "{} observable(s) follow(s) a non-Gaussian"
            " distribution:".format(n_gauss_warnings)
        )
        print("{}".format(", ".join(non_gaussian_observables)))
        print("!" * 72)

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
