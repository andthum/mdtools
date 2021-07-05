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


import os
import sys
from datetime import datetime
import psutil
import argparse
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mdtools as mdt


if __name__ == "__main__":

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())

    parser = argparse.ArgumentParser(
        description=(
            "Fit mean square displacements with MSD(t)=2d*D*t,"
            " where D is the fitting parameter (diffusion"
            " coefficient) and d is the dimensionalitiy of the"
            " diffusive motion."
        )
    )

    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=True,
        help="Input file."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename pattern. There will be created two files:"
             "<OUTFILE>.pdf containing the the graph and the fit and"
             "<OUTFILE>.txt containing the fitting parameters."
    )
    parser.add_argument(
        '-c',
        dest='COLS',
        type=int,
        nargs='+',
        required=False,
        default=[0, 1],
        help="Space separated list of columns to read from the input"
             " file. The first column given is treated as time (x"
             " values), all other columns as MSDs (y values)."
             " Default: '0 1'"
    )

    parser.add_argument(
        '-b',
        dest='BEGINFIT',
        type=int,
        nargs='+',
        required=False,
        default=None,
        help="Start time for fitting the MSD in the time unit given with"
             " --t-unit. If not given, the fit will start at 10 %% of"
             " the data. Give either one start time for all MSDs or one"
             " for each MSD."
    )
    parser.add_argument(
        "-e",
        dest="ENDFIT",
        type=int,
        nargs='+',
        required=False,
        default=None,
        help="End time for fitting the MSD in the time unit given with"
             " --t-unit. If not given, the fit will end at 90 %% of the"
             " data. Give either one end time for all MSDs or one for"
             " each MSD."
    )
    parser.add_argument(
        '-d',
        dest='NDIM',
        type=int,
        default=3,
        help="Number of dimensions of the diffusive motion. Default: 3"
    )

    parser.add_argument(
        '--xmin',
        dest='XMIN',
        type=float,
        default=None,
        help="Lower limit of x axis."
    )
    parser.add_argument(
        '--xmax',
        dest='XMAX',
        type=float,
        default=None,
        help="Upper limit of x axis."
    )
    parser.add_argument(
        '--ymin',
        dest='YMIN',
        type=float,
        default=None,
        help="Lower limit of y axis."
    )
    parser.add_argument(
        '--ymax',
        dest='YMAX',
        type=float,
        default=None,
        help="Upper limit of y axis."
    )
    parser.add_argument(
        '--labels',
        dest='LABELS',
        type=str,
        nargs='+',
        required=False,
        default=None,
        help="Space separated list of labels, one for each MSD column"
             " given with -c"
    )

    parser.add_argument(
        '--t-unit',
        dest='TUNIT',
        type=str,
        default="ns",
        help="Time unit. Default: ns"
    )
    parser.add_argument(
        '--l-unit',
        dest='LUNIT',
        type=str,
        default="nm",
        help="Length unit. Default: nm"
    )
    parser.add_argument(
        '--t-conv',
        dest='TCONV',
        type=float,
        default=1e-3,
        help="Time conversion factor. All times will be multiplied by"
             " this factor. Default: 1"
    )
    parser.add_argument(
        '--l-conv',
        dest='LCONV',
        type=float,
        default=1,
        help="Length convertion factor. All lengths will be multiplied"
             " by this factor. Default: 1"
    )

    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

    if len(args.COLS) < 2:
        raise ValueError("You must give at least two columns")
    if (args.BEGINFIT is not None and
        len(args.BEGINFIT) != 1 and
            len(args.BEGINFIT) != len(args.COLS) - 1):
        raise ValueError("You have to give either one start time for"
                         " fitting or exactly as many as MSDs to plot")
    if (args.ENDFIT is not None and
        len(args.ENDFIT) != 1 and
            len(args.ENDFIT) != len(args.COLS) - 1):
        raise ValueError("You have to give either one end time for"
                         " fitting or exactly as many as MSDs to plot")
    if args.LABELS is not None and len(args.LABELS) != len(args.COLS) - 1:
        raise ValueError("You have to give exactly as many labels as"
                         " MSDs to plot")

    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()

    msds = np.loadtxt(fname=args.INFILE,
                      comments=['#', '@'],
                      usecols=args.COLS,
                      ndmin=2)
    times = msds[:, 0] * args.TCONV
    msds = msds[:, 1:] * args.LCONV**2
    if len(times) < 2:
        raise ValueError("The input must contain at least two rows")

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    if args.BEGINFIT is None:
        beginfit = np.full(len(msds[0]), int(0.1 * len(times)))
    else:
        beginfit = np.searchsorted(times, args.BEGINFIT)
        if len(beginfit) == 1:
            beginfit = np.full(len(msds[0]), beginfit[0])
    if np.any(beginfit > len(times) - 2):
        beginfit[beginfit > len(times) - 2] = len(times) - 2
    if args.ENDFIT is None:
        endfit = np.full(len(msds[0]), int(0.9 * len(times)) + 1)
    else:
        endfit = np.searchsorted(times, args.ENDFIT) + 1
        if len(endfit) == 1:
            endfit = np.full(len(msds[0]), endfit[0])
    if np.any(endfit > len(times)):
        endfit[endfit > len(times)] = len(times)

    if args.XMIN is None:
        args.XMIN = np.min(times)
    if args.XMAX is None:
        args.XMAX = np.max(times)
    if args.YMIN is None:
        args.YMIN = np.min(msds)
    if args.YMAX is None:
        args.YMAX = np.max(msds)

    if args.XMIN <= 0:
        xmin_log = np.min(times[times > 0])
    else:
        xmin_log = args.XMIN
    if args.YMIN <= 0:
        ymin_log = np.min(msds[msds > 0])
    else:
        ymin_log = args.YMIN

    if args.LABELS is None:
        args.LABELS = [None, ] * len(msds[0])

    print("\n\n\n", flush=True)
    print("Fitting curve(s)", flush=True)
    timer = datetime.now()

    popt = np.zeros(len(msds[0]))
    pcov = np.zeros(len(msds[0]))
    fit = [None, ] * len(msds[0])
    ymin_fit = np.zeros(len(msds[0]))
    ymax_fit = np.zeros(len(msds[0]))
    for i, msd in enumerate(msds.T):
        popt[i], pcov[i] = curve_fit(f=lambda t, D: mdt.dyn.msd(t=t,
                                                                D=D,
                                                                d=args.NDIM),
                                     xdata=times[beginfit[i]:endfit[i]],
                                     ydata=msd[beginfit[i]:endfit[i]])
        fit[i] = mdt.dyn.msd(t=times[beginfit[i]:endfit[i]],
                             D=popt[i],
                             d=args.NDIM)
        ymin_fit[i] = np.min([np.min(msd[beginfit[i]:endfit[i]]),
                              np.min(fit[i])])
        ymax_fit[i] = np.max([np.max(msd[beginfit[i]:endfit[i]]),
                              np.max(fit[i])])
    xmin_fit = np.min(times[np.min(beginfit):np.max(endfit)])
    xmax_fit = np.max(times[np.min(beginfit):np.max(endfit)])
    ymin_fit = np.min(ymin_fit)
    ymax_fit = np.max(ymax_fit)

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print("\n\n\n", flush=True)
    print("Writing output", flush=True)
    timer = datetime.now()

    mdt.fh.write_header(args.OUTFILE + ".txt")
    with open(args.OUTFILE + ".txt", 'a') as outfile:
        outfile.write("# Diffusion coefficient(s) D\n")
        outfile.write("# Fitted from Mean Square Displacement (MSD) via\n")
        outfile.write("#   MSD(t) = {}*D*t\n".format(2 * args.NDIM))
        outfile.write("#\n")
        outfile.write("#\n")
        outfile.write("# The columns contain:\n")
        outfile.write("#   1 The column number of the input file from which the MSD data were read\n")
        outfile.write("#   2 The label given with --labels\n")
        outfile.write("#   3 D ({}^2 {}^-1)\n".format(args.LUNIT, args.TUNIT))
        outfile.write("#   4 Standard deviation of D resulting from the fit\n")
        outfile.write("#   5 Start time for fitting ({})\n".format(args.TUNIT))
        outfile.write("#   6 End time for fitting ({})\n".format(args.TUNIT))
        outfile.write("#\n")
        outfile.write('# Column number:\n')
        outfile.write("# {:>3d} {:>12d} {:>16d} {:>16d} {:>12d} {:>12d}\n".format(1, 2, 3, 4, 5, 6))
        outfile.write("# {:>3s}".format("Num"))
        outfile.write(" {:>12s}".format("Label"))
        outfile.write(" {:>16s}".format("D"))
        outfile.write(" {:>16s}".format("SD"))
        outfile.write(" {:>12s}".format("Start"))
        outfile.write(" {:>12s}\n".format("End"))
        for i in range(len(popt)):
            outfile.write("  {:>3d}".format(args.COLS[i + 1]))
            outfile.write(" {:>12s}".format(str(args.LABELS[i])))
            outfile.write(" {:16.9e}".format(popt[i]))
            outfile.write(" {:16.9e}".format(np.sqrt(pcov[i])))
            outfile.write(" {:>12.3f}".format(times[beginfit[i]]))
            outfile.write(" {:>12.3f}\n".format(times[endfit[i] - 1]))

    print("  Created {}".format(args.OUTFILE + ".txt"))
    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print("\n\n\n", flush=True)
    print("Creating plots", flush=True)
    timer = datetime.now()

    outfile = args.OUTFILE + ".pdf"
    mdt.fh.backup(outfile)
    with PdfPages(outfile) as pdf:

        # Loglog plot without fit
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True,
                                 num=1)
        axis.ticklabel_format(axis='both',
                              style='scientific',
                              scilimits=(0, 0),
                              useOffset=False)
        for i, msd in enumerate(msds.T):
            mdt.plot.plot(
                ax=axis,
                x=times,
                y=msd,
                xmin=xmin_log,
                xmax=args.XMAX,
                ymin=ymin_log,
                ymax=args.YMAX,
                logx=True,
                logy=True,
                xlabel=r'$\Delta t$ / {}'.format(args.TUNIT),
                ylabel=r'$\langle \Delta r^2(\Delta t) \rangle$ / {}$^2$'.format(args.LUNIT),
                label=args.LABELS[i])
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Loglog plot with fit
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True,
                                 num=1)
        axis.ticklabel_format(axis='both',
                              style='scientific',
                              scilimits=(0, 0),
                              useOffset=False)
        for msd in msds.T:
            mdt.plot.plot(
                ax=axis,
                x=times,
                y=msd,
                xmin=xmin_log,
                xmax=args.XMAX,
                ymin=ymin_log,
                ymax=args.YMAX,
                logx=True,
                logy=True,
                xlabel=r'$\Delta t$ / {}'.format(args.TUNIT),
                ylabel=r'$\langle \Delta r^2(\Delta t) \rangle$ / {}$^2$'.format(args.LUNIT))
        for i in range(len(msds[0])):
            mdt.plot.plot(
                ax=axis,
                x=times[beginfit[i]:endfit[i]],
                y=fit[i],
                xmin=xmin_log,
                xmax=args.XMAX,
                ymin=ymin_log,
                ymax=args.YMAX,
                logx=True,
                logy=True,
                xlabel=r'$\Delta t$ / {}'.format(args.TUNIT),
                ylabel=r'$\langle \Delta r^2(\Delta t) \rangle$ / {}$^2$'.format(args.LUNIT),
                label=((args.LABELS[i] + ", " if args.LABELS[i] is not None else "") +
                       r'$D={:<.4f}$'.format(popt[i])))
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Loglog plot with fit, zoomed in fitted region
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True,
                                 num=1)
        axis.ticklabel_format(axis='both',
                              style='scientific',
                              scilimits=(0, 0),
                              useOffset=False)
        for msd in msds.T:
            mdt.plot.plot(
                ax=axis,
                x=times,
                y=msd,
                xmin=xmin_fit if xmin_fit > 0 else xmin_log,
                xmax=xmax_fit,
                ymin=ymin_fit if ymin_fit > 0 else ymin_log,
                ymax=ymax_fit,
                logx=True,
                logy=True,
                xlabel=r'$\Delta t$ / {}'.format(args.TUNIT),
                ylabel=r'$\langle \Delta r^2(\Delta t) \rangle$ / {}$^2$'.format(args.LUNIT))
        for i in range(len(msds[0])):
            mdt.plot.plot(
                ax=axis,
                x=times[beginfit[i]:endfit[i]],
                y=fit[i],
                xmin=xmin_fit if xmin_fit > 0 else xmin_log,
                xmax=xmax_fit,
                ymin=ymin_fit if ymin_fit > 0 else ymin_log,
                ymax=ymax_fit,
                logx=True,
                logy=True,
                xlabel=r'$\Delta t$ / {}'.format(args.TUNIT),
                ylabel=r'$\langle \Delta r^2(\Delta t) \rangle$ / {}$^2$'.format(args.LUNIT),
                label=r'$D={:<.4f}$'.format(popt[i]))
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Fit residuals
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True,
                                 num=1)
        axis.axhline(color='black')
        for i, msd in enumerate(msds.T):
            mdt.plot.plot(
                ax=axis,
                x=times[beginfit[i]:endfit[i]],
                y=msd[beginfit[i]:endfit[i]] - fit[i],
                xmin=xmin_fit,
                xmax=xmax_fit,
                xlabel=r'$\Delta t$ / {}'.format(args.TUNIT),
                ylabel=r'(Data $-$ Fit) / {}$^2$'.format(args.LUNIT),
                label=args.LABELS[i])
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Semi log plot without fit
        for i in range(len(msds[0])):
            msds[:, i][times != 0] /= (6 * times[times != 0])
        ymin_semilog = np.min(msds)
        ymax_semilog = np.max(msds)

        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True,
                                 num=1)
        axis.ticklabel_format(axis='x',
                              style='scientific',
                              scilimits=(0, 0),
                              useOffset=False)
        for i, msd in enumerate(msds.T):
            mdt.plot.plot(
                ax=axis,
                x=times,
                y=msd,
                xmin=xmin_log,
                xmax=args.XMAX,
                ymin=ymin_semilog,
                ymax=ymax_semilog,
                logx=True,
                xlabel=r'$\Delta t$ / {}'.format(args.TUNIT),
                ylabel=(r'$\langle \Delta r^2(\Delta t) \rangle / 6 \Delta t$ / ' +
                        args.LUNIT +
                        r'$^2$ ' +
                        args.TUNIT +
                        r'$^{-1}$'),
                label=args.LABELS[i])
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Semi log plot with fit
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True,
                                 num=1)
        axis.ticklabel_format(axis='x',
                              style='scientific',
                              scilimits=(0, 0),
                              useOffset=False)
        for i, msd in enumerate(msds.T):
            mdt.plot.plot(
                ax=axis,
                x=times,
                y=msd,
                xmin=xmin_log,
                xmax=args.XMAX,
                ymin=ymin_semilog,
                ymax=ymax_semilog,
                logx=True,
                xlabel=r'$\Delta t$ / {}'.format(args.TUNIT),
                ylabel=(r'$\langle \Delta r^2(\Delta t) \rangle / 6 \Delta t$ / ' +
                        args.LUNIT +
                        r'$^2$ ' +
                        args.TUNIT +
                        r'$^{-1}$'))
        for i in range(len(msds[0])):
            mdt.plot.plot(
                ax=axis,
                x=times[beginfit[i]:endfit[i]],
                y=fit[i] / (6 * times[beginfit[i]:endfit[i]]),
                xmin=xmin_log,
                xmax=args.XMAX,
                ymin=ymin_semilog,
                ymax=ymax_semilog,
                logx=True,
                xlabel=r'$\Delta t$ / {}'.format(args.TUNIT),
                ylabel=(r'$\langle \Delta r^2(\Delta t) \rangle / 6 \Delta t$ / ' +
                        args.LUNIT +
                        r'$^2$ ' +
                        args.TUNIT +
                        r'$^{-1}$'),
                label=((args.LABELS[i] + ", " if args.LABELS[i] is not None else "") +
                       r'$D={:<.4f}$'.format(popt[i])))
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Semi log plot with fit, zoomed in fitted region
        ymin_fit_semilog = np.min(msds[np.min(beginfit):np.max(endfit[i])])
        ymax_fit_semilog = np.max(msds[np.min(beginfit):np.max(endfit[i])])

        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True,
                                 num=1)
        axis.ticklabel_format(axis='x',
                              style='scientific',
                              scilimits=(0, 0),
                              useOffset=False)
        for msd in msds.T:
            mdt.plot.plot(
                ax=axis,
                x=times,
                y=msd,
                xmin=xmin_fit if xmin_fit > 0 else xmin_log,
                xmax=xmax_fit,
                ymin=ymin_fit_semilog,
                ymax=ymax_fit_semilog,
                logx=True,
                xlabel=r'$\Delta t$ / {}'.format(args.TUNIT),
                ylabel=(r'$\langle \Delta r^2(\Delta t) \rangle / 6 \Delta t$ / ' +
                        args.LUNIT +
                        r'$^2$ ' +
                        args.TUNIT +
                        r'$^{-1}$'))
        for i in range(len(msds[0])):
            mdt.plot.plot(
                ax=axis,
                x=times[beginfit[i]:endfit[i]],
                y=fit[i] / (6 * times[beginfit[i]:endfit[i]]),
                xmin=xmin_fit if xmin_fit > 0 else xmin_log,
                xmax=xmax_fit,
                ymin=ymin_fit_semilog,
                ymax=ymax_fit_semilog,
                logx=True,
                xlabel=r'$\Delta t$ / {}'.format(args.TUNIT),
                ylabel=(r'$\langle \Delta r^2(\Delta t) \rangle / 6 \Delta t$ / ' +
                        args.LUNIT +
                        r'$^2$ ' +
                        args.TUNIT +
                        r'$^{-1}$'),
                label=r'$D={:<.4f}$'.format(popt[i]))
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

    print("\n\n\n", flush=True)
    print("{} done".format(os.path.basename(sys.argv[0])), flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now() - timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)
