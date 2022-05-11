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


"""
Plot the number of transitions leading into or out of a given state for
each frame in a discrete trajectory as function of time.

Options
-------
-f          Trajectory file.  File containing the discrete trajectory
            stored as :class:`numpy.ndarray` in the binary :file:`.npy`
            format.  The array must be of shape ``(n, f)``, where ``n``
            is the number of compounds and ``f`` is the number of
            frames.  The shape can also be ``(f,)``, in which case the
            array is expanded to shape ``(1, f)``.  All elements of the
            array must be integers or floats whose fractional part is
            zero, because they are interpreted as the indices of the
            states in which a given compound is at a given frame.
-o          Output filename.
-b          First frame to read from the trajectory.  Frame numbering
            starts at zero.  Default: ``0``.
-e          Last frame to read from the trajectory.  This is exclusive,
            i.e. the last frame read is actually ``END - 1``.  A value
            of ``-1`` means to read the very last frame.  Default:
            ``-1``.
--every     Read every n-th frame from the trajectory.  Default: ``1``.
--cumsum    Calculate and plot the cumulative sum of transitions per
            state over time.
--fit       Fit the number of transitions leading into or out of the
            central state by a power law.
--labels    One label for each state in the discrete trajectory.  The
            labels will be shown in the legend of the plot.  If
            provided, you must give one label for each state.  By
            default the states are numbered consecutively from
            ``MIN_STATE`` to ``MAX_STATE``.  Default: ``None``.
--xlim      Left and right limit of the x-axis in data coordinates.
            Pass 'None' to adjust the limit(s) automatically.  Default:
            ``[None, None]``.
--ylim      Lower and upper limit of the y-axis in data coordinates.
            Pass 'None' to adjust the limit(s) automatically.  Default:
            ``[None, None]``.
--log       Whether to use a logarithmic scale for the x- and/or y-axis.
            Default:  ``[False, False]``.

See Also
--------
:func:`mdtools.dtrj.trans_per_state_vs_time` :
    Function that counts the number of transitions leading into or out
    of a state for each frame in the discrete trajectory

Examples
--------
TODO
"""


__author__ = "Andreas Thum"


# Standard libraries
import argparse
import os
import sys
from datetime import datetime, timedelta

# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import psutil
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# Local application/library specific imports
import mdtools as mdt
import mdtools.plot as mdtplt  # noqa: F401; Import MDTools plot style


def power_law(x, a, b):
    """
    Power law to use as fit function.

    Parameters
    ----------
    x : scalar or numpy.ndarray
        The sample point(s) for the function
    a : scalar or numpy.ndarray
        Prefactor.
    b : scalar or numpy.ndarray
        Exponent.

    Returns
    -------
    result : scalar or numpy.ndarray
        ``a * x**b``
    """
    return a * x**b


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Plot the number of transitions leading into or out of a given"
            " state for each frame in a discrete trajectory as function of"
            " time"
        )
    )
    parser.add_argument(
        "-f",
        dest="TRJFILE",
        type=str,
        required=True,
        help=(
            "Trajectory file containing the discrete trajectory stored as"
            " numpy.ndarray in the binary .npy format."
        ),
    )
    parser.add_argument(
        "-o",
        dest="OUTFILE",
        type=str,
        required=True,
        help=("Output filename."),
    )
    parser.add_argument(
        "-b",
        dest="BEGIN",
        type=int,
        required=False,
        default=0,
        help=(
            "First frame to read from the trajectory.  Frame numbering starts"
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
            "Last frame to read from the trajectory (exclusive).  Default:"
            " %(default)s."
        ),
    )
    parser.add_argument(
        "--every",
        dest="EVERY",
        type=int,
        required=False,
        default=1,
        help=(
            "Read every n-th frame from the trajectory.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--cumsum",
        dest="CUMSUM",
        required=False,
        default=False,
        action="store_true",
        help="Plot the cumulative sum of transitions per state over time.",
    )
    parser.add_argument(
        "--fit",
        dest="FIT",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Fit the number of transitions leading into or out of the central"
            " state by a power law."
        ),
    )
    parser.add_argument(
        "--labels",
        dest="LABELS",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help=(
            "A label for each state in the discrete trajectory.  Default:"
            " %(default)s"
        ),
    )
    parser.add_argument(
        "--xlim",
        dest="XLIM",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=float),
        nargs=2,
        required=False,
        default=[0, None],
        help=(
            "Left and right limit of the x-axis in data coordinates.  Default:"
            " %(default)s"
        ),
    )
    parser.add_argument(
        "--ylim",
        dest="YLIM",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=float),
        nargs=2,
        required=False,
        default=[0, None],
        help=(
            "Lower and upper limit of the y-axis in data coordinates."
            "  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--log",
        dest="LOG",
        type=mdt.fh.str2bool,
        nargs=2,
        required=False,
        default=[False, False],
        help=(
            "Whether to use a logarithmic scale for the x- and/or y-axis."
            "  Default: %(default)s"
        ),
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

    print("\n")
    print("Loading discrete trajectory")
    timer = datetime.now()
    dtrj = mdt.fh.load_dtrj(args.TRJFILE)
    N_CMPS, N_FRAMES_TOT = dtrj.shape
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=N_FRAMES_TOT,
    )
    dtrj = dtrj[:, BEGIN:END:EVERY]
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Calculating transition histograms as function of time")
    timer = datetime.now()
    hist_start, hist_end = mdt.dtrj.trans_per_state_vs_time(
        dtrj, pin="both", trans_type="both"
    )
    min_state = np.min(dtrj)
    max_state = np.max(dtrj)
    assert hist_start[0].shape[0] == max_state - min_state + 1
    times = np.arange(0, hist_start[0].shape[1])
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating plot...")
    timer = datetime.now()
    if args.LABELS is None:
        args.LABELS = np.arange(min_state, max_state + 1)
    if len(args.LABELS) != max_state - min_state + 1:
        raise ValueError(
            "You must give {} labels, one for each"
            " state".format(max_state - min_state + 1)
        )
    legend_titles = ("Initial States",) * 2 + ("Final States",) * 2
    if args.CUMSUM:
        ylabels = (
            "Cum. Sum of pos. Trans.",
            "Cum. Sum of neg. Trans.",
        ) * 2
        linestyle, marker, rasterized = "-", "", False
    else:
        ylabels = ("No. of pos. Trans.", "No. of neg. Trans.") * 2
        linestyle, marker, rasterized = "", "o", True
    cmap = plt.get_cmap()
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        for i, hist in enumerate(hist_start + hist_end):
            if args.CUMSUM:
                hist = np.cumsum(hist, axis=-1)
            if args.FIT:
                popt, pcov = curve_fit(
                    power_law, xdata=times, ydata=hist[len(hist) // 2]
                )
                fit = power_law(times, *popt)
            fig, ax = plt.subplots(clear=True)
            ax.set_prop_cycle(
                color=[
                    cmap(i / len(args.LABELS)) for i in range(len(args.LABELS))
                ]
            )
            ax.plot(
                times,
                hist.T,
                linestyle=linestyle,
                marker=marker,
                label=args.LABELS,
                rasterized=rasterized,
            )
            if args.FIT:
                ax.plot(
                    times,
                    fit,
                    color="black",
                    linestyle="--",
                    label=(
                        r"$"
                        + str("%.2f" % popt[0])
                        + r" \cdot x^{"
                        + str("%.2f" % popt[1])
                        + r"}$"
                    ),
                )
            if args.LOG[0]:
                ax.set_xscale("log", base=10, subs=np.arange(2, 10))
            if args.LOG[1]:
                ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax.set(
                xlabel="Time in Trajectory Steps",
                ylabel=ylabels[i],
                xlim=args.XLIM,
                ylim=args.YLIM,
            )
            ax.legend(
                title=legend_titles[i],
                ncol=2,  # Default: 1
                labelspacing=0.25,  # Default: 0.5
                handlelength=1,  # Default: 2.0
                columnspacing=1,  # Default: 2.0
            )
            pdf.savefig()
        plt.close()
    print("Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
