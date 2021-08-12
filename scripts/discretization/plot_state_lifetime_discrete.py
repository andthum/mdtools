#!/usr/bin/env python3


# This file is part of MDTools.
# Copyright (C) 2020  Andreas Thum
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


import sys
import os
from datetime import datetime
import psutil
import argparse
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mdtools as mdt


if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process()

    parser = argparse.ArgumentParser(
        description=(
            "Plot selected columns from the output of"
            " state_lifetime_discrete.py."
        )
    )

    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=True,
        help="The output file of state_lifetime_discrete.py."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename pattern. There are created two or three"
             " output files:"
             " <OUTFILE>.pdf containing the remain probabilities as"
             " function of lag time and secondary state as well as the"
             " average lifetimes as function of the secondary states;"
             " <OUTFILE>_fit.pdf containing the remain probability for"
             " each secondary state as function of lag time with its"
             " respective fit;"
             " <OUTFILE>_fit.txt containing the fit parameters for each"
             " secondary state. Only created if --refit is given."
             " Plots are optimized for PDF format with TeX support."
    )
    parser.add_argument(
        '-c',
        dest='COLS',
        type=int,
        nargs='+',
        required=False,
        default=None,
        help="Space separated list of columns to plot. Default is to"
             " plot all columns. Column 0 contains the lag times and is"
             " therefore always selected automatically."
    )

    parser.add_argument(
        '--refit',
        dest='REFIT',
        required=False,
        default=False,
        action='store_true',
        help="Do not read the fitting parameters from the input file but"
             " redo the fitting of the data."
    )
    parser.add_argument(
        '--end-fit',
        dest='ENDFIT',
        type=float,
        required=False,
        default=None,
        help="End time for fitting the remain probability (in the unit"
             " given by --time-unit). Inclusive, i.e. the time given"
             " here is still included in the fit. Is meaningless if"
             " --refit is not given. Default: End at 90%% of the lag"
             " times."
    )
    parser.add_argument(
        '--stop-fit',
        dest='STOPFIT',
        type=float,
        required=False,
        default=0.01,
        help="Stop fitting the remain probability as soon as it falls"
             " below this value. The fitting is stopped by whatever"
             " happens earlier: --end-fit or --stop-fit. Is meaningless"
             " if --refit is not given. Default: 0.01"
    )

    parser.add_argument(
        '--xmin',
        dest='XMIN',
        type=float,
        required=False,
        default=None,
        help="Minimum x-range of the plot. Default: Minimum time"
    )
    parser.add_argument(
        '--xmax',
        dest='XMAX',
        type=float,
        required=False,
        default=None,
        help="Maximum x-range of the plot. Default: Maximum time"
    )
    parser.add_argument(
        '--ymin',
        dest='YMIN',
        type=float,
        required=False,
        default=0,
        help="Minimum y-range of the plot. Default: 0"
    )
    parser.add_argument(
        '--ymax',
        dest='YMAX',
        type=float,
        required=False,
        default=1,
        help="Maximum y-range of the plot. Default: 1"
    )

    parser.add_argument(
        '--time-conv',
        dest='TCONV',
        type=float,
        required=False,
        default=1,
        help="Multiply times by this factor. Default: 1, which results"
             " in 'trajectory steps'"
    )
    parser.add_argument(
        '--time-unit',
        dest='TUNIT',
        type=str,
        required=False,
        default="steps",
        help="Time unit. Default: 'steps'"
    )

    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()

    if args.COLS is not None:
        args.COLS = np.unique(args.COLS)
        if 0 not in args.COLS:
            args.COLS = np.insert(args.COLS, 0, 0)
        if len(args.COLS) < 2:
            raise ValueError("You must give at least one column"
                             " different from zero with -c")
    p = np.loadtxt(fname=args.INFILE, usecols=args.COLS)
    times = p[1:, 0] * args.TCONV
    states = p[0, 1:]
    p = p[1:, 1:]

    if np.any(p > 1):
        raise ValueError("At least one element of the remain probability"
                         " is greater than one. This means your input is"
                         " not a proper probability function")
    if np.any(p < 0):
        raise ValueError("At least one element of the remain probability"
                         " is less than zero. This means your input is"
                         " not a proper probability function")

    fractional_part = np.modf(states)[0]
    if np.all(fractional_part == 0):
        states = states.astype(np.int32)
    del fractional_part

    if not args.REFIT:
        if args.COLS is None:
            args.COLS = np.arange(len(states) + 1)
        with open(args.INFILE, 'r') as f:
            header = []
            for line in f:
                if line[0] == '#':
                    header.append(line[1:])
                else:
                    break
        header = header[-8:]
        fit_data = []
        for i in range(8 - 1):
            fd = header[i].split()
            if i == 5:  # beta
                fd = np.array(fd[1:], dtype=np.float32)
            else:
                fd = np.array(fd[2:], dtype=np.float32)
            fd = fd[args.COLS[1:] - 1]
            if i < 2:  # fit_start and fit_stop
                if np.any(np.modf(fd)[0] != 0):
                    raise ValueError("The input file contains"
                                     " non-integer values for fit_start"
                                     " and/or fit_stop")
                fd = fd.astype(np.uint32)
            elif i >= 2 and i < 5:  # tau_mean, tau and tau_sd
                fd *= args.TCONV
            fit_data.append(fd)
            if len(fit_data[i]) != len(fit_data[0]):
                raise ValueError("len(fit_data[i]) != len(fit_data[0])."
                                 " This should not have happened")
        del header
        # fit_data now contains the values for
        #   fit_start [in steps]
        #   fit_stop  [in steps]
        #   tau_mean  [in the unit given by --time-unit]
        #   tau       [in the unit given by --time-unit]
        #   tau_sd    [in the unit given by --time-unit]
        #   beta
        #   beta_sd

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    if args.REFIT:
        print("\n\n\n", flush=True)
        print("Re-fitting remain probabilities", flush=True)
        timer = datetime.now()

        if args.ENDFIT is None:
            endfit = int(0.9 * len(times))
        else:
            _, endfit = mdt.nph.find_nearest(times,
                                             args.ENDFIT,
                                             return_index=True)
        endfit += 1  # To make args.ENDFIT inclusive

        fit_start = np.zeros(len(states), dtype=np.uint32)  # inclusive
        fit_stop = np.zeros(len(states), dtype=np.uint32)   # exclusive
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
                xdata=times[fit_start[i]:fit_stop[i]],
                ydata=p[:, i][fit_start[i]:fit_stop[i]])
        tau = popt[:, 0]
        beta = popt[:, 1]
        tau_mean = tau / beta * gamma(1 / beta)
        fit_data = [fit_start, fit_stop,
                    tau_mean,
                    tau, perr[:, 0],
                    beta, perr[:, 1]]

        print("Elapsed time:         {}"
              .format(datetime.now() - timer),
              flush=True)
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss / 2**20),
              flush=True)

    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()

    if args.REFIT:
        header = (
            "Refitting of the remain probabilities from the input file\n"
            "with a stretched exponential decay function.\n"
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
            "{:>14d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d}\n"
            .format(1, 2, 3, 4, 5, 6, 7, 8, unit=args.TUNIT)
        )
        data = np.column_stack([states,
                                tau_mean,
                                fit_start * args.TCONV,
                                fit_stop * args.TCONV,
                                tau, perr[:, 0],
                                beta, perr[:, 1]])
        outfile = args.OUTFILE + "_fit.txt"
        mdt.fh.savetxt(fname=outfile, data=data, header=header)
        print("  Created {}".format(outfile))

    fontsize_legend = 24

    if args.XMIN is None:
        args.XMIN = np.nanmin(times)
    if args.XMAX is None:
        args.XMAX = np.nanmax(times)
    # Line style "cycler"
    ls = ['-', '--', '-.', ':']
    ls *= (1 + len(states) // len(ls))

    outfile = args.OUTFILE + ".pdf"
    mdt.fh.backup(outfile)
    with PdfPages(outfile) as pdf:
        # Remain probability vs lag time
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        for i, s in enumerate(states):
            mdt.plot.plot(ax=axis,
                          x=times,
                          y=p[:, i],
                          xmin=args.XMIN,
                          xmax=args.XMAX,
                          ymin=args.YMIN,
                          ymax=args.YMAX,
                          xlabel=r'$\Delta t$ / ' + args.TUNIT,
                          ylabel=r'$p(\Delta t, S^\prime)$',
                          label=str(s),
                          linestyle=ls[i])
        axis.legend(loc='best',
                    title=r'$S^\prime$',
                    title_fontsize=fontsize_legend,
                    fontsize=fontsize_legend,
                    numpoints=1,
                    ncol=1 + len(states) // 9,
                    labelspacing=0.2,
                    columnspacing=1.4,
                    handletextpad=0.5,
                    frameon=False)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # ln(p) vs lag time
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        ymin = args.YMIN if args.YMIN < 0 else np.nanmin(np.log(p[p > 0]))
        ymax = args.YMAX if args.YMAX <= 0 else 0
        for i, s in enumerate(states):
            mask = (p[:, i] > 0)
            mdt.plot.plot(ax=axis,
                          x=times[mask],
                          y=np.log(p[:, i][mask]),
                          xmin=args.XMIN,
                          xmax=args.XMAX,
                          ymin=ymin,
                          ymax=ymax,
                          xlabel=r'$\Delta t$ / ' + args.TUNIT,
                          ylabel=r'$\ln{p(\Delta t, S^\prime)}$',
                          label=str(s),
                          linestyle=ls[i])
        axis.legend(loc='best',
                    title=r'$S^\prime$',
                    title_fontsize=fontsize_legend,
                    fontsize=fontsize_legend,
                    numpoints=1,
                    ncol=1 + len(states) // 9,
                    labelspacing=0.2,
                    columnspacing=1.4,
                    handletextpad=0.5,
                    frameon=False)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Remain probability vs log(lag time)
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        xmin = args.XMIN if args.XMIN > 0 else np.nanmin(times[times > 0])
        xmax = args.XMAX if args.XMAX > 0 else np.nanmax(times[times > 0])
        for i, s in enumerate(states):
            mdt.plot.plot(ax=axis,
                          x=times,
                          y=p[:, i],
                          xmin=xmin,
                          xmax=xmax,
                          ymin=args.YMIN,
                          ymax=args.YMAX,
                          logx=True,
                          xlabel=r'$\Delta t$ / ' + args.TUNIT,
                          ylabel=r'$p(\Delta t, S^\prime)$',
                          label=str(s),
                          linestyle=ls[i])
        axis.legend(loc='best',
                    title=r'$S^\prime$',
                    title_fontsize=fontsize_legend,
                    fontsize=fontsize_legend,
                    numpoints=1,
                    ncol=1 + len(states) // 9,
                    labelspacing=0.2,
                    columnspacing=1.4,
                    handletextpad=0.5,
                    frameon=False)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Mean relaxation time vs secondary state
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(
            ax=axis,
            x=states,
            y=fit_data[2],
            ymin=0,
            xlabel=r'$S^\prime$',
            ylabel=r'Mean relaxation time $\langle \tau \rangle$ / ' + args.TUNIT,
            marker='o')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Fit parameter tau and bete vs secondary state
        ylabels = (r'$\tau$ / ' + args.TUNIT, r'$\beta$')
        for i, j in enumerate([3, 5]):
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            mdt.plot.errorbar(ax=axis,
                              x=states,
                              y=fit_data[j],
                              yerr=fit_data[j + 1],
                              ymin=0,
                              ymax=1 if j == 5 else None,
                              xlabel=r'$S^\prime$',
                              ylabel=r'Fit parameter ' + ylabels[i],
                              marker='o')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Fit start and stop vs secondary state
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        labels = (r'start', r'stop')
        markers = ('o', 's')
        for i, j in enumerate([0, 1]):
            mdt.plot.plot(ax=axis,
                          x=states,
                          y=fit_data[j] * args.TCONV,
                          ymin=0,
                          ymax=1.05 * np.nanmax(fit_data[1]) * args.TCONV,
                          xlabel=r'$S^\prime$',
                          ylabel=r'Fit range / ' + args.TUNIT,
                          label=labels[i],
                          marker=markers[i])
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    print("  Created {}".format(outfile))

    outfile = args.OUTFILE + "_fit.pdf"
    mdt.fh.backup(outfile)
    with PdfPages(outfile) as pdf:
        for i, s in enumerate(states):
            fit = mdt.func.kww(t=times,
                               tau=fit_data[3][i],
                               beta=fit_data[5][i])
            fit_region = np.zeros_like(fit, dtype=bool)
            fit_region[fit_data[0][i]:fit_data[1][i]] = True

            # ln(p) vs lag time with fit
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            mask = (p[:, i] > 0)
            mdt.plot.plot(ax=axis,
                          x=times[mask],
                          y=np.log(p[:, i][mask]),
                          xmin=args.XMIN,
                          xmax=args.XMAX,
                          ymax=0,
                          xlabel=r'$\Delta t$ / ' + args.TUNIT,
                          ylabel=r'$\ln{p(\Delta t, S^\prime)}$',
                          label=r'$S^\prime =' + str(s) + r'$')
            mask = fit_region & (fit > 0)
            mdt.plot.plot(ax=axis,
                          x=times[mask],
                          y=np.log(fit[mask]),
                          xmin=args.XMIN,
                          xmax=args.XMAX,
                          ymax=0,
                          xlabel=r'$\Delta t$ / ' + args.TUNIT,
                          ylabel=r'$\ln{p(\Delta t, S^\prime)}$',
                          linestyle='--',
                          label="Fit")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Remain probability vs log(lag time)
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            xmin = args.XMIN if args.XMIN > 0 else np.nanmin(times[times > 0])
            xmax = args.XMAX if args.XMAX > 0 else np.nanmax(times[times > 0])
            mdt.plot.plot(ax=axis,
                          x=times,
                          y=p[:, i],
                          xmin=xmin,
                          xmax=xmax,
                          ymin=args.YMIN,
                          ymax=args.YMAX,
                          logx=True,
                          xlabel=r'$\Delta t$ / ' + args.TUNIT,
                          ylabel=r'$p(\Delta t, S^\prime)$',
                          label=r'$S^\prime =' + str(s) + r'$')
            mdt.plot.plot(ax=axis,
                          x=times[fit_region],
                          y=fit[fit_region],
                          xmin=xmin,
                          xmax=xmax,
                          ymin=args.YMIN,
                          ymax=args.YMAX,
                          logx=True,
                          xlabel=r'$\Delta t$ / ' + args.TUNIT,
                          ylabel=r'$p(\Delta t, S^\prime)$',
                          linestyle='--',
                          label="Fit")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print("  Created {}".format(outfile))

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print("\n\n\n{} done".format(os.path.basename(sys.argv[0])))
    print("Elapsed time:         {}"
          .format(datetime.now() - timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)
