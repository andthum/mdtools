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
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mdtools as mdt


if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process()

    parser = argparse.ArgumentParser(
        description=(
            "Read a trajectory of renewal events as e.g."
            " generated with extract_renewal_events.py and plot"
            " the displacement versus the renewal time as"
            " scatter plot. Additionally, a bin-wise average"
            " and variance are computed and plotted."
            " Furthermore, the displacement distribution in"
            " each time bin is fitted by a Gaussian function"
            " and the non-Gaussian parameter is calculated."
        )
    )

    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=True,
        help="Trajectory of renewal events as e.g. generated with"
             " extract_renewal_events.py."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename. Plots are optimized for PDF format with"
             " TeX support."
    )
    parser.add_argument(
        '--sel',
        dest='SEL',
        required=False,
        default=False,
        action='store_true',
        help="Use the selection compounds instead of the reference"
             " compounds."
    )
    parser.add_argument(
        '-dcolor',
        dest='DCOLOR',
        type=str,
        required=False,
        default=None,
        help="The scatter points can be colored according to the initial"
             " position of the compounds. Must be either x, y or z."
             " Default: No coloring"
    )

    parser.add_argument(
        '--tbin-start',
        dest='TBIN_START',
        type=float,
        required=False,
        default=0,
        help="Time to start binning the renewal times. Default: 0"
    )
    parser.add_argument(
        '--tbin-end',
        dest='TBIN_STOP',
        type=float,
        required=False,
        default=None,
        help="Time to end binning the renewal times. Default: Maximum"
             " renewal time"
    )
    parser.add_argument(
        '--tbin-num',
        dest='TBIN_NUM',
        type=int,
        required=False,
        default=50,
        help="Number of bins to use for binning the renewal times."
             " Default: 50"
    )
    parser.add_argument(
        '--tbins',
        dest='TBINFILE',
        type=str,
        required=False,
        default=None,
        help="ASCII formatted text file containing custom bin edges for"
             " binning the renewal times. Bin edges are read from the"
             " first column, lines starting with '#' are ignored. Bins"
             " do not need to be equidistant."
    )
    parser.add_argument(
        '--displbin-start',
        dest='DISPLBIN_START',
        type=float,
        required=False,
        default=None,
        help="Displacement to start binning the displacements. Default:"
             " Minimum displacement"
    )
    parser.add_argument(
        '--displbin-end',
        dest='DISPLBIN_STOP',
        type=float,
        required=False,
        default=None,
        help="Displacement to end binning the displacements. Default:"
             " Maximum displacement"
    )
    parser.add_argument(
        '--displbin-num',
        dest='DISPLBIN_NUM',
        type=int,
        required=False,
        default=50,
        help="Number of bins to use for binning the displacements."
             " Default: 50"
    )

    parser.add_argument(
        '--xmin',
        dest='XMIN',
        type=float,
        required=False,
        default=0,
        help="Minimum x-range of the plot. Default: 0"
    )
    parser.add_argument(
        '--xmax',
        dest='XMAX',
        type=float,
        required=False,
        default=None,
        help="Maximum x-range of the plot. By default detected"
             " automatically."
    )
    parser.add_argument(
        '--ymin',
        dest='YMIN',
        type=float,
        required=False,
        default=None,
        help="Minimum y-range of the plot. By default detected"
             " automatically."
    )
    parser.add_argument(
        '--ymax',
        dest='YMAX',
        type=float,
        required=False,
        default=None,
        help="Maximum y-range of the plot. By default detected"
             " automatically."
    )

    parser.add_argument(
        '--time-conv',
        dest='TCONV',
        type=float,
        required=False,
        default=1,
        help="Multiply times by this factor. Default: 1, which results"
             " in ps"
    )
    parser.add_argument(
        '--time-unit',
        dest='TUNIT',
        type=str,
        required=False,
        default="ps",
        help="Time unit. Default: ps"
    )
    parser.add_argument(
        '--length-conv',
        dest='LCONV',
        type=float,
        required=False,
        default=1,
        help="Multiply lengths by this factor. Default: 1, which results"
             " in Angstroms"
    )
    parser.add_argument(
        '--length-unit',
        dest='LUNIT',
        type=str,
        required=False,
        default="A",
        help="Lengh unit. Default: A"
    )

    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

    if (args.DCOLOR is not None and
        args.DCOLOR != 'x' and
        args.DCOLOR != 'y' and
            args.DCOLOR != 'z'):
        raise ValueError("--dcolor must be either 'x', 'y' or 'z', but"
                         " you gave {}".format(args.DCOLOR))
    dim = {'x': 0, 'y': 1, 'z': 2}

    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()

    trenew = np.loadtxt(fname=args.INFILE, usecols=3)
    trenew *= args.TCONV
    if args.SEL:
        cols = (13, 14, 15)
    else:
        cols = (10, 11, 12)
    displ = np.loadtxt(fname=args.INFILE, usecols=cols)
    displ *= args.LCONV
    if args.DCOLOR is not None:
        if args.SEL:
            cols = 7 + dim[args.DCOLOR]
        else:
            cols = 4 + dim[args.DCOLOR]
        pos_t0 = np.loadtxt(fname=args.INFILE, usecols=cols)
        pos_t0 *= args.LCONV

    if args.TBINFILE is None:
        if args.TBIN_START is None or args.TBIN_START > np.min(trenew):
            args.TBIN_START = np.min(trenew)
        if args.TBIN_STOP is None or args.TBIN_STOP < np.max(trenew):
            args.TBIN_STOP = np.max(trenew)
        tbins = np.linspace(args.TBIN_START,
                            args.TBIN_STOP,
                            args.TBIN_NUM)
    else:
        tbins = np.loadtxt(args.TBINFILE, usecols=0)
        tbins = np.unique(tbins)
        if len(tbins) == 0:
            raise ValueError("Invalid tbins")
        if tbins[0] > np.min(trenew):
            tbins = np.insert(tbins, 0, np.min(trenew))
        if tbins[-1] < np.max(trenew):
            tbins = np.append(tbins, np.max(trenew))
    t = tbins[1:] - np.diff(tbins) / 2

    tbin_ix = np.digitize(trenew, tbins)
    # In np.histogram the last bin is closed, but in np.digitize all
    # bins are half-open. Make the last bin closed:
    tbin_ix[tbin_ix == len(tbins)] = len(tbins) - 1
    if np.any(tbin_ix == 0):
        raise ValueError("At least one element of tbin_ix is zero. This"
                         " should not have happened.")
    nevents = np.full(len(tbins), np.nan)
    displ_mean = np.full((len(tbins), displ.shape[1]), np.nan)
    displ_std = np.full((len(tbins), displ.shape[1]), np.nan)
    msd_mean = np.full((len(tbins), displ.shape[1]), np.nan)
    non_gaus = np.full((len(tbins), displ.shape[1]), np.nan)
    for i in np.unique(tbin_ix):
        mask = (tbin_ix == i)
        nevents[i] = np.count_nonzero(mask)
        if nevents[i] > 0:
            displ_mean[i] = np.mean(displ[mask], axis=0)
            displ_std[i] = np.std(displ[mask], axis=0)
            msd_mean[i] = np.mean(displ[mask]**2, axis=0)
            non_gaus[i] = mdt.stats.non_gaussian_parameter(
                displ[mask],
                d=1,
                is_squared=False,
                axis=0)
    if not np.isnan(nevents[0]):
        raise ValueError("The first element of nevents is not NaN. This"
                         " should not have happened")
    if not np.all(np.isnan(displ_mean[0])):
        raise ValueError("Not all first elements of displ_mean are NaN."
                         " This should not have happened")
    if not np.all(np.isnan(displ_std[0])):
        raise ValueError("Not all first elements of displ_std are NaN."
                         " This should not have happened")
    if not np.all(np.isnan(msd_mean[0])):
        raise ValueError("Not all first elements of msd_mean are NaN."
                         " This should not have happened")
    if not np.all(np.isnan(non_gaus[0])):
        raise ValueError("Not all first elements of non_gaus are NaN."
                         " This should not have happened")
    valid = (nevents > 0)

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print("\n\n\n", flush=True)
    print("Fitting histograms", flush=True)
    timer = datetime.now()

    if args.DISPLBIN_START is None or args.DISPLBIN_START > np.min(displ):
        args.DISPLBIN_START = np.min(displ)
    if args.DISPLBIN_STOP is None or args.DISPLBIN_STOP < np.max(displ):
        args.DISPLBIN_STOP = np.max(displ)

    popt_displ = np.full((len(tbins), displ.shape[1], 2), np.nan)
    perr_displ = np.full((len(tbins), displ.shape[1], 2), np.nan)
    for i in np.unique(tbin_ix):
        for j, data in enumerate(displ[tbin_ix == i].T):
            try:
                displhist, displbins = np.histogram(
                    data,
                    bins=args.DISPLBIN_NUM,
                    range=(args.DISPLBIN_START,
                           args.DISPLBIN_STOP),
                    density=True)
                x = displbins[1:] - np.diff(displbins) / 2
                popt, pcov = opt.curve_fit(f=mdt.stats.gaussian,
                                           xdata=x,
                                           ydata=displhist,
                                           p0=(np.mean(data),
                                               np.std(data)))
                popt_displ[i][j] = popt
                perr_displ[i][j] = np.sqrt(np.diag(pcov))
            except (ValueError, RuntimeError, opt.OptimizeWarning) as err:
                print(flush=True)
                print("  An error has occurred while fitting the"
                      " displacement histograms:",
                      flush=True)
                print("  {}".format(err), flush=True)
                print("  Setting fit parameters to numpy.nan",
                      flush=True)
                print(flush=True)

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print("\n\n\n", flush=True)
    print("Creating plot", flush=True)
    timer = datetime.now()

    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    label_pad = 16

    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        # Number of renewal events per time bins
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(ax=axis,
                      x=t,
                      y=nevents[1:],
                      xmin=args.XMIN,
                      xmax=args.XMAX,
                      xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                      ylabel=r'$N_{renew}$',
                      color='black',
                      marker='o')
        mdt.plot.vlines(ax=axis,
                        x=tbins,
                        start=axis.get_ylim()[0],
                        stop=axis.get_ylim()[1],
                        xmin=args.XMIN,
                        xmax=args.XMAX,
                        color='black',
                        linestyle='dotted')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Displacements vs renewal time
        ylabel = ('x', 'y', 'z')
        for i in range(displ.shape[1]):
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            axis.axhline(y=0, color='black')
            if args.DCOLOR is None:
                mdt.plot.scatter(
                    ax=axis,
                    x=trenew,
                    y=displ.T[i],
                    xmin=args.XMIN,
                    xmax=args.XMAX,
                    ymin=args.YMIN,
                    ymax=args.YMAX,
                    xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                    ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                    marker='x')
            else:
                img = mdt.plot.scatter(
                    ax=axis,
                    x=trenew,
                    y=displ.T[i],
                    c=pos_t0,
                    xmin=args.XMIN,
                    xmax=args.XMAX,
                    ymin=args.YMIN,
                    ymax=args.YMAX,
                    xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                    ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                    marker='x',
                    cmap='plasma')
                cbar = plt.colorbar(img, ax=axis)
                cbar.set_label(label=r'${}(t_0)$ / {}'.format(args.DCOLOR, args.LUNIT),
                               fontsize=fontsize_labels)
                cbar.ax.yaxis.labelpad = label_pad
                cbar.ax.yaxis.offsetText.set(size=fontsize_ticks)
                cbar.ax.tick_params(which='major',
                                    direction='out',
                                    length=tick_length,
                                    labelsize=fontsize_ticks)
                cbar.ax.tick_params(which='minor',
                                    direction='out',
                                    length=0.5 * tick_length,
                                    labelsize=0.8 * fontsize_ticks)
            mdt.plot.vlines(ax=axis,
                            x=tbins,
                            start=axis.get_ylim()[0],
                            stop=axis.get_ylim()[1],
                            xmin=args.XMIN,
                            xmax=args.XMAX,
                            ymin=args.YMIN,
                            ymax=args.YMAX,
                            color='black',
                            linestyle='dotted')
            mdt.plot.plot(
                ax=axis,
                x=t,
                y=displ_mean.T[i][1:],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                label=r'$\langle \Delta ' + ylabel[i] + r' \rangle$',
                color='red',
                marker='o')
            mdt.plot.plot(
                ax=axis,
                x=t,
                y=np.sqrt(msd_mean.T[i][1:]),
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                label=r'$\sqrt{\langle \Delta ' + ylabel[i] + r'^2 \rangle}$',
                color='blue',
                marker='^')
            mdt.plot.plot(
                ax=axis,
                x=t,
                y=displ_std.T[i][1:],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                label=r'$\sqrt{\langle \Delta ' + ylabel[i] + r'^2 \rangle - \langle \Delta ' + ylabel[i] + r' \rangle ^2}$',
                color='green',
                marker='s')
            axis.legend(loc='upper center',
                        bbox_to_anchor=(0.5, 1.32),
                        ncol=2,
                        numpoints=1,
                        fontsize=fontsize_legend,
                        frameon=True,
                        framealpha=1,
                        edgecolor='black',
                        fancybox=False)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Statistics
        fig, axis = plt.subplots(figsize=(11.69, 8.27),
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.axis('off')
        fontsize = 26

        xpos = 0.05
        ypos = 0.95
        if args.SEL:
            plt.text(x=xpos,
                     y=ypos,
                     s="Selection compound displacements",
                     fontsize=fontsize)
        else:
            plt.text(x=xpos,
                     y=ypos,
                     s="Reference compound displacements",
                     fontsize=fontsize)
        ypos -= 0.10
        plt.text(x=xpos, y=ypos, s="Statistics:", fontsize=fontsize)
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Tot. counts: ${:>16d}$'.format(displ.shape[0]),
                 fontsize=fontsize)
        xpos += 0.10
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Mean / ' + args.LUNIT,
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Std. dev. / ' + args.LUNIT,
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$N(\Delta a = 0)$',
                 fontsize=fontsize)
        for i, data in enumerate(displ.T):
            xpos = 0.05
            ypos -= 0.05
            plt.text(x=xpos,
                     y=ypos,
                     s=r'$\Delta ' + ylabel[i] + r'$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>+16.9e}$'.format(np.mean(data)),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(np.std(data)),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16d}$'.format(np.count_nonzero(data == 0)),
                     fontsize=fontsize)

        # Histogram parameters
        xpos = 0.05
        ypos -= 0.10
        plt.text(x=xpos,
                 y=ypos,
                 s="Histogram parameters:",
                 fontsize=fontsize)
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s="First bin edge / " + args.LUNIT,
                 fontsize=fontsize)
        xpos += 0.40
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>+16.9e}$'.format(args.DISPLBIN_START),
                 fontsize=fontsize)
        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s="Last bin edge / " + args.LUNIT,
                 fontsize=fontsize)
        xpos += 0.40
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>+16.9e}$ '.format(args.DISPLBIN_STOP),
                 fontsize=fontsize)
        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos, y=ypos, s=r'$N$ bins', fontsize=fontsize)
        xpos += 0.40
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16d}$'.format(args.DISPLBIN_NUM),
                 fontsize=fontsize)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Displacements vs renewal time, Gaussian fit parameters
        ylabel = ('x', 'y', 'z')
        for i in range(displ.shape[1]):
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            axis.axhline(y=0, color='black')
            if args.DCOLOR is None:
                img = mdt.plot.scatter(
                    ax=axis,
                    x=trenew,
                    y=displ.T[i],
                    xmin=args.XMIN,
                    xmax=args.XMAX,
                    ymin=args.YMIN,
                    ymax=args.YMAX,
                    xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                    ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                    marker='x')
            else:
                img = mdt.plot.scatter(
                    ax=axis,
                    x=trenew,
                    y=displ.T[i],
                    c=pos_t0,
                    xmin=args.XMIN,
                    xmax=args.XMAX,
                    ymin=args.YMIN,
                    ymax=args.YMAX,
                    xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                    ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                    marker='x',
                    cmap='plasma')
                cbar = plt.colorbar(img, ax=axis)
                cbar.set_label(label=r'${}(t_0)$ / {}'.format(args.DCOLOR, args.LUNIT),
                               fontsize=fontsize_labels)
                cbar.ax.yaxis.labelpad = label_pad
                cbar.ax.yaxis.offsetText.set(size=fontsize_ticks)
                cbar.ax.tick_params(which='major',
                                    direction='out',
                                    length=tick_length,
                                    labelsize=fontsize_ticks)
                cbar.ax.tick_params(which='minor',
                                    direction='out',
                                    length=0.5 * tick_length,
                                    labelsize=0.8 * fontsize_ticks)
            mdt.plot.vlines(ax=axis,
                            x=tbins,
                            start=axis.get_ylim()[0],
                            stop=axis.get_ylim()[1],
                            xmin=args.XMIN,
                            xmax=args.XMAX,
                            ymin=args.YMIN,
                            ymax=args.YMAX,
                            color='black',
                            linestyle='dotted')
            mdt.plot.plot(
                ax=axis,
                x=t,
                y=popt_displ.T[0][i][1:],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                label=r'$\mu$',
                color='red',
                marker='o')
            mdt.plot.plot(
                ax=axis,
                x=t,
                y=popt_displ.T[1][i][1:],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                label=r'$\sigma$',
                color='blue',
                marker='^')
            mdt.plot.plot(
                ax=axis,
                x=t,
                y=non_gaus.T[i][1:],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                label=r'$A$',
                color='green',
                marker='s')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Fit parameters
        fig, axis = plt.subplots(figsize=(11.69, 8.27),
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.axis('off')
        fontsize = 26

        xpos = 0.05
        ypos = 0.95
        plt.text(x=xpos, y=ypos, s="Fit parameters:", fontsize=fontsize)
        ypos -= 0.08
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Fit function: $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$',
                 fontsize=fontsize)
        ypos -= 0.08
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Non-Gaussian parameter: $A(t) = \frac{\langle \Delta a^4(t) \rangle}{(1+\frac{2}{d}) \cdot \langle \Delta a^2(t) \rangle^2} - 1$',
                 fontsize=fontsize)
        ypos -= 0.08
        plt.text(x=xpos,
                 y=ypos,
                 s="Average of all bins with data points, not weighted:",
                 fontsize=fontsize)
        xpos += 0.10
        ypos -= 0.08
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$\langle \mu \rangle$ / ' + args.LUNIT,
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Std. dev. / ' + args.LUNIT,
                 fontsize=fontsize)
        for i in range(displ.shape[1]):
            xpos = 0.05
            ypos -= 0.05
            plt.text(x=xpos,
                     y=ypos,
                     s=r'$\Delta ' + ylabel[i] + r'$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>+16.9e}$'.format(np.nanmean(popt_displ.T[0][i][1:])),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(np.nanstd(popt_displ.T[0][i][1:])),
                     fontsize=fontsize)

        xpos = 0.15
        ypos -= 0.08
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$\langle \sigma \rangle$ / ' + args.LUNIT,
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Std. dev. / ' + args.LUNIT,
                 fontsize=fontsize)
        for i in range(displ.shape[1]):
            xpos = 0.05
            ypos -= 0.05
            plt.text(x=xpos,
                     y=ypos,
                     s=r'$\Delta ' + ylabel[i] + r'$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(np.nanmean(popt_displ.T[1][i][1:])),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(np.nanstd(popt_displ.T[1][i][1:])),
                     fontsize=fontsize)

        xpos = 0.15
        ypos -= 0.08
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$\langle A \rangle$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos, y=ypos, s=r'Std. dev.', fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos, y=ypos, s=r'$d$', fontsize=fontsize)
        for i in range(displ.shape[1]):
            xpos = 0.05
            ypos -= 0.05
            plt.text(x=xpos,
                     y=ypos,
                     s=r'$\Delta ' + ylabel[i] + r'$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>+16.9e}$'.format(np.nanmean(non_gaus.T[i][1:])),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(np.nanstd(non_gaus.T[i][1:])),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos, y=ypos, s=r'$1$', fontsize=fontsize)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        fig, axis = plt.subplots(figsize=(11.69, 8.27),
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.axis('off')
        fontsize = 26

        _, counts = np.unique(tbin_ix, return_counts=True)
        mean, std = mdt.stats.std_weighted(popt_displ[valid],
                                           weights=counts,
                                           axis=0)

        xpos = 0.05
        ypos = 0.95
        plt.text(x=xpos,
                 y=ypos,
                 s="Average over all bins with data points,",
                 fontsize=fontsize)
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s="weighted by the number of data points per bin:",
                 fontsize=fontsize)
        xpos += 0.10
        ypos -= 0.08
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$\langle \mu \rangle$ / ' + args.LUNIT,
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Std. dev. / ' + args.LUNIT,
                 fontsize=fontsize)
        for i in range(displ.shape[1]):
            xpos = 0.05
            ypos -= 0.05
            plt.text(x=xpos,
                     y=ypos,
                     s=r'$\Delta ' + ylabel[i] + r'$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>+16.9e}$'.format(mean.T[0][i]),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(std.T[0][i]),
                     fontsize=fontsize)

        xpos = 0.15
        ypos -= 0.08
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$\langle \sigma \rangle$ / ' + args.LUNIT,
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Std. dev. / ' + args.LUNIT,
                 fontsize=fontsize)
        for i in range(displ.shape[1]):
            xpos = 0.05
            ypos -= 0.05
            plt.text(x=xpos,
                     y=ypos,
                     s=r'$\Delta ' + ylabel[i] + r'$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(mean.T[1][i]),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(std.T[1][i]),
                     fontsize=fontsize)

        mean, std = mdt.stats.std_weighted(non_gaus[valid],
                                           weights=counts,
                                           axis=0)
        xpos = 0.15
        ypos -= 0.08
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$\langle A \rangle$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos, y=ypos, s=r'Std. dev.', fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos, y=ypos, s=r'$d$', fontsize=fontsize)
        for i in range(displ.shape[1]):
            xpos = 0.05
            ypos -= 0.05
            plt.text(x=xpos,
                     y=ypos,
                     s=r'$\Delta ' + ylabel[i] + r'$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>+16.9e}$'.format(mean.T[i]),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(std.T[i]),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos, y=ypos, s=r'$1$', fontsize=fontsize)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print("  Created {}".format(args.OUTFILE))
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
