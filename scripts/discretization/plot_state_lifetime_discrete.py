#!/usr/bin/env python3

# This file is part of MDTools.
# Copyright (C) 2021  The MDTools Development Team and all contributors
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


r"""
Plot the lifetime autocorrelation function of discrete states as
function of another set of discrete states.

.. todo::

    Finish docstring

This script is designed to plot the output of
:mod:`scripts.discretization.state_lifetime_discrete`.

Options
-------
-f          Input filename.  The output file of
            :mod:`scripts.discretization.state_lifetime_discrete`.
-o          Output filename pattern.  There are created two or three
            output files:

                1. <OUTFILE>.pdf containing the lifetime autocorrelation
                   functions as function of lag time and secondary state
                   as well as the average lifetimes as function of the
                   secondary states.
                2. <OUTFILE>_fit.pdf containing the lifetime
                   autocorrelation functions for each secondary state as
                   function of lag time with its respective fit.
                3. <OUTFILE>_fit.txt containing the fit parameters for
                   each secondary state.  Only created if \--refit is
                   given.
--cols      The columns of the input file that should be plotted.
            Column numbering starts at zero.  Note that all given
            columns are treated as y data.  The 0-th column is always
            read automatically and treated as x data.  Default: ``None``
            (this means read all columns).
--refit     Do not read the fitting parameters from the input file but
            redo the fitting of the data.
--end-fit   End time for fitting the remain probability in data units.
            This is inclusive, i.e. the time given here is still
            included in the fit.  Is meaningless if \--refit is not
            given.  Default: ``None`` (this means end at 90% of the lag
            times).
--stop-fit  Stop fitting the remain probability as soon as it falls
            below this value.  The fitting is stopped by whatever
            happens earlier: \--end-fit or \--stop-fit.  Is meaningless
            if \--refit is not given.  Default: ``0.01``.
--xlim      Left and right limit of the x-axis in data coordinates.  If
            the right limit is ``None``, plot until the maximum lag
            time.  Default: ``[0, None]``.
--ylim      Lower and upper limit of the y-axis in data coordinates.
            Pass 'None' to adjust the limit(s) automatically.  Default:
            ``[0, 1]``.
--time-conv
            Multiply all times by this factor.  Default: ``1``.
--time-unit
            Time unit.  Default: ``'steps'``.

See Also
--------
:mod:`plot_state_lifetime` :
    Plot the lifetime autocorrelation function of discrete states as
    function of lag time

Examples
--------
TODO
"""


__author__ = "Andreas Thum"


# Standard libraries
import sys
import os
import warnings
import argparse
from datetime import datetime, timedelta

# Third party libraries
import psutil
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Local application/library specific imports
import mdtools as mdt
import mdtools.plot as mdtplt


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Plot the lifetime autocorrelation function of discrete states as"
            " function of another set of discrete states.  This script is"
            " designed to plot the output of state_lifetime_discrete.py.  For"
            " more information, refer to the documetation of this script."
        )
    )
    parser.add_argument(
        "-f",
        dest="INFILE",
        type=str,
        required=True,
        help=(
            "Input filename.  The output file of state_lifetime_discrete.py."
        ),
    )
    parser.add_argument(
        "-o",
        dest="OUTFILE",
        type=str,
        required=True,
        help=(
            "Output filename pattern.  There are created two or three output"
            " files:"
            "  <OUTFILE>.pdf containing the remain probabilities as function"
            " of lag time and secondary state as well as the average lifetimes"
            " as function of the secondary states;"
            "  <OUTFILE>_fit.pdf containing the remain probability for each"
            " secondary state as function of lag time with its respective fit;"
            "  <OUTFILE>_fit.txt containing the fit parameters for each"
            " secondary state.  Only created if --refit is given."
        ),
    )
    parser.add_argument(
        "--cols",
        dest="COLS",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help=(
            "The columns of the input file that should be plotted.  Column"
            " numbering starts at zero.  Note that all given columns are"
            " treated as y data.  The 0-th column is always read automatically"
            " and treated as x data.  Default: %(default)s  (this means read"
            " all columns)."
        ),
    )
    parser.add_argument(
        "--refit",
        dest="REFIT",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Do not read the fitting parameters from the input file but redo"
            " the fitting of the data."
        ),
    )
    parser.add_argument(
        "--end-fit",
        dest="ENDFIT",
        type=float,
        required=False,
        default=None,
        help=(
            "End time for fitting the remain probability in data units.  This"
            " is inclusive, i.e. the time given here is still included in the"
            " fit.  Is meaningless if --refit is not given.  Default:"
            " %(default)s (this means end at 90%% of the lag times)."
        ),
    )
    parser.add_argument(
        "--stop-fit",
        dest="STOPFIT",
        type=float,
        required=False,
        default=0.01,
        help=(
            "Stop fitting the remain probability as soon as it falls below"
            " this value.  The fitting is stopped by whatever happens earlier:"
            " --end-fit or --stop-fit.  Is meaningless if --refit is not"
            " given.  Default: %(default)s"
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
            "Left and right limit of the x-axis in data coordinates.  If the"
            " right limit is None, plot until the maximum lag time.  Default:"
            " %(default)s"
        ),
    )
    parser.add_argument(
        "--ylim",
        dest="YLIM",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=float),
        nargs=2,
        required=False,
        default=[0, 1],
        help=(
            "Lower and upper limit of the y-axis in data coordinates."
            "  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--time-conv",
        dest="TCONV",
        type=float,
        required=False,
        default=1,
        help=("Multiply all times by this factor.  Default: %(default)s"),
    )
    parser.add_argument(
        "--time-unit",
        dest="TUNIT",
        type=str,
        required=False,
        default="steps",
        help=("Time unit.  Default: %(default)s"),
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

    print("\n")
    print("Reading input file...")
    timer = datetime.now()
    if args.COLS is not None:
        args.COLS.insert(0, 0)
    p = np.loadtxt(args.INFILE, usecols=args.COLS)
    times = p[1:, 0] * args.TCONV
    states = p[0, 1:]
    p = p[1:, 1:]
    if np.any(p < 0) or np.any(p > 1):
        warnings.warn(
            "Your input is not a proper probability function, some values lie"
            " outside the interval [0, 1]",
            RuntimeWarning,
        )
    fractional_part = np.modf(states)[0]
    if np.all(fractional_part == 0):
        states = states.astype(np.int32)
    del fractional_part
    if not args.REFIT:
        if args.COLS is None:
            args.COLS = np.arange(len(states) + 1)
        with open(args.INFILE, "r") as f:
            header = []
            for line in f:
                if line[0] == "#":
                    header.append(line[1:])
                else:
                    break
        header = header[-8:]
        fit_data = []
        for i, head in enumerate(header):
            fd = head.split()
            if i == 5:  # beta
                fd = np.array(fd[1:], dtype=np.float32)
            else:
                fd = np.array(fd[2:], dtype=np.float32)
            fd = fd[args.COLS[1:] - 1]
            if i < 2:  # fit_start and fit_stop
                if np.any(np.modf(fd)[0] != 0):
                    raise ValueError(
                        "The input file contains non-integer values for"
                        " 'fit_start' and/or 'fit_stop'"
                    )
                fd = fd.astype(np.uint32)
            elif i >= 2 and i < 5:  # tau_mean, tau and tau_sd
                fd *= args.TCONV
            fit_data.append(fd)
            if len(fit_data[i]) != len(fit_data[0]):
                raise ValueError(
                    "len(fit_data[i]) != len(fit_data[0]).  This should not"
                    " have happened"
                )
        del header
        # fit_data now contains the values for
        #   fit_start [in steps]
        #   fit_stop  [in steps]
        #   tau_mean  [in the unit given by --time-unit]
        #   tau       [in the unit given by --time-unit]
        #   tau_sd    [in the unit given by --time-unit]
        #   beta
        #   beta_sd
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    if args.REFIT:
        print("\n")
        print("Re-fitting remain probabilities")
        timer = datetime.now()
        if args.ENDFIT is None:
            endfit = int(0.9 * len(times))
        else:
            _, endfit = mdt.nph.find_nearest(
                times, args.ENDFIT, return_index=True
            )
        endfit += 1  # To make args.ENDFIT inclusive
        fit_start = np.zeros(len(states), dtype=np.uint32)  # inclusive
        fit_stop = np.zeros(len(states), dtype=np.uint32)  # exclusive
        popt = np.full((len(states), 2), np.nan, dtype=np.float32)
        perr = np.full((len(states), 2), np.nan, dtype=np.float32)
        for i in range(len(states)):
            stopfit = np.argmax(p[:, i] < args.STOPFIT)
            if stopfit == 0 and p[:, i][stopfit] >= args.STOPFIT:
                stopfit = len(p[:, i])
            elif stopfit < 2:
                stopfit = 2
            fit_stop[i] = min(endfit, stopfit)
            popt[i], perr[i] = mdt.func.fit_kww(
                xdata=times[fit_start[i] : fit_stop[i]],
                ydata=p[:, i][fit_start[i] : fit_stop[i]],
            )
        tau = popt[:, 0]
        beta = popt[:, 1]
        tau_mean = tau / beta * gamma(1 / beta)
        fit_data = [
            fit_start,
            fit_stop,
            tau_mean,
            tau,
            perr[:, 0],
            beta,
            perr[:, 1],
        ]
        print("Elapsed time:         {}".format(datetime.now() - timer))
        print(
            "Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc))
        )

    print("\n")
    print("Creating plot...")
    timer = datetime.now()
    if args.REFIT:
        header = (
            "Refitting of the remain probabilities from the input file with\n"
            "a stretched exponential decay function.\n"
            "\n"
            "\n"
            "The columns contain:\n"
            "  1 Secondary state\n"
            "  2 Mean relaxation time <tau> (in {unit})\n"
            "  3 Start fit                  (in {unit})\n"
            "  4 Stop fit                   (in {unit})\n"
            "  5 Fit parameter tau          (in {unit})\n"
            "  6 Standard deviation of tau  (in {unit})\n"
            "  7 Fit parameter beta\n"
            "  8 Standard deviation of beta\n"
            "\n"
            "Column number:\n"
            "{:>14d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d}"
            " {:>16d}\n".format(1, 2, 3, 4, 5, 6, 7, 8, unit=args.TUNIT)
        )
        data = np.column_stack(
            [
                states,
                tau_mean,
                fit_start * args.TCONV,
                fit_stop * args.TCONV,
                tau,
                perr[:, 0],
                beta,
                perr[:, 1],
            ]
        )
        outfile = args.OUTFILE + "_fit.txt"
        mdt.fh.savetxt(outfile, data=data, header=header)
        print("  Created {}".format(outfile))

    if args.XLIM[1] is None:
        args.XLIM[1] = np.nanmax(times)
    if args.XLIM[0] is not None and args.XLIM[0] > 0:
        xmin = args.XLIM[0]
    else:
        xmin = np.nanmin(times[times > 0])
    if args.XLIM[1] is not None and args.XLIM[1] > 0:
        xmax = args.XLIM[1]
    else:
        xmax = np.nanmax(times[times > 0])
    if args.YLIM[0] is not None and args.YLIM[0] > 0:
        ymin = np.log(args.YLIM[0])
    else:
        ymin = None
    if args.YLIM[1] is not None and args.YLIM[1] > 0:
        ymax = np.log(args.YLIM[1])
    else:
        ymax = None
    cmap = plt.get_cmap()
    outfile = args.OUTFILE + ".pdf"
    mdt.fh.backup(outfile)
    with PdfPages(outfile) as pdf:
        # Remain probability vs lag time
        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(
            color=[cmap(i / len(states)) for i in range(len(states))],
        )
        for i, s in enumerate(states):
            ax.plot(times, p[:, i], label=str(s))
        ax.set(
            xlabel=r"$\Delta t$ / " + args.TUNIT,
            ylabel=r"$p(\Delta t, S^\prime)$",
            xlim=args.XLIM,
            ylim=args.YLIM,
        )
        ax.legend(
            title=r"$S^\prime$",
            ncol=1 + len(states) // 9,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        pdf.savefig()
        plt.close()
        # ln(p) vs lag time
        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(
            color=[cmap(i / len(states)) for i in range(len(states))],
        )
        for i, s in enumerate(states):
            mask = p[:, i] > 0
            ax.plot(times[mask], np.log(p[:, i][mask]), label=str(s))
        ax.set(
            xlabel=r"$\Delta t$ / " + args.TUNIT,
            ylabel=r"$\ln{p(\Delta t, S^\prime)}$",
            xlim=args.XLIM,
            ylim=(ymin, ymax),
        )
        ax.legend(
            title=r"$S^\prime$",
            ncol=1 + len(states) // 9,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        pdf.savefig()
        plt.close()
        # Remain probability vs log(lag time)
        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(
            color=[cmap(i / len(states)) for i in range(len(states))],
        )
        for i, s in enumerate(states):
            ax.plot(times, p[:, i], label=str(s))
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set(
            xlabel=r"$\Delta t$ / " + args.TUNIT,
            ylabel=r"$p(\Delta t, S^\prime)$",
            xlim=(xmin, xmax),
            ylim=args.YLIM,
        )
        ax.legend(
            title=r"$S^\prime$",
            ncol=1 + len(states) // 9,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        pdf.savefig()
        plt.close()
        # Mean relaxation time vs secondary state
        fig, ax = plt.subplots(clear=True)
        ax.plot(states, fit_data[2], marker="o")
        ax.set_xlabel(r"$S^\prime$")
        ax.set_ylabel(
            r"Mean relaxation time $\langle \tau \rangle$ / " + args.TUNIT
        )
        pdf.savefig()
        plt.close()
        # Fit parameter tau and beta vs secondary state
        ylabels = (r"$\tau$ / " + args.TUNIT, r"$\beta$")
        for i, j in enumerate([3, 5]):
            fig, ax = plt.subplots(clear=True)
            ax.errorbar(
                x=states, y=fit_data[j], yerr=fit_data[j + 1], marker="o"
            )
            ax.set_xlabel(r"$S^\prime$")
            ax.set_ylabel(r"Fit parameter " + ylabels[i])
            pdf.savefig()
            plt.close()
        # Fit start and stop vs secondary state
        fig, ax = plt.subplots(clear=True)
        labels = (r"start", r"stop")
        markers = ("^", "v")
        for i, j in enumerate([0, 1]):
            ax.plot(
                states,
                fit_data[j] * args.TCONV,
                label=labels[i],
                marker=markers[i],
            )
        ax.set_xlabel(r"$S^\prime$")
        ax.set_ylabel(r"Fit range / " + args.TUNIT)
        ax.legend()
        pdf.savefig()
        plt.close()
    print("  Created {}".format(outfile))

    outfile = args.OUTFILE + "_fit.pdf"
    mdt.fh.backup(outfile)
    with PdfPages(outfile) as pdf:
        for i, s in enumerate(states):
            fit = mdt.func.kww(
                t=times, tau=fit_data[3][i], beta=fit_data[5][i]
            )
            fit_region = np.zeros_like(fit, dtype=bool)
            fit_region[fit_data[0][i] : fit_data[1][i]] = True
            # ln(p) vs lag time with fit
            fig, ax = plt.subplots(clear=True)
            mask = p[:, i] > 0
            ax.plot(
                times[mask],
                np.log(p[:, i][mask]),
                label=r"$S^\prime =" + str(s) + r"$",
            )
            mask = fit_region & (fit > 0)
            ax.plot(
                times[mask], np.log(fit[mask]), linestyle="--", label="Fit"
            )
            ax.set(
                xlabel=r"$\Delta t$ / " + args.TUNIT,
                ylabel=r"$\ln{p(\Delta t, S^\prime)}$",
                xlim=args.XLIM,
                ylim=(ymin, ymax),
            )
            ax.legend()
            pdf.savefig()
            plt.close()
            # Remain probability vs log(lag time) with fit
            fig, ax = plt.subplots(clear=True)
            ax.plot(times, p[:, i], label=r"$S^\prime =" + str(s) + r"$")
            ax.plot(
                times[fit_region], fit[fit_region], linestyle="--", label="Fit"
            )
            ax.set_xscale("log", base=10, subs=np.arange(2, 10))
            ax.set(
                xlabel=r"$\Delta t$ / " + args.TUNIT,
                ylabel=r"$p(\Delta t, S^\prime)$",
                xlim=(xmin, xmax),
                ylim=args.YLIM,
            )
            ax.legend()
            pdf.savefig()
            plt.close()
    print("  Created {}".format(outfile))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
