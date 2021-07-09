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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mdtools as mdt


if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())

    parser = argparse.ArgumentParser(
        description=(
            "Read a trajectory of renewal events as e.g."
            " generated with extract_renewal_events.py and plot"
            " the displacement versus the inital position as"
            " scatter plot. Additionally, a bin-wise average"
            " and variance are computed and plotted."
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
        '-d',
        dest='DIRECTION',
        type=str,
        required=False,
        default='z',
        help="The spatial direction to use for the inital position. Must"
             " be either x, y or z. Default: z"
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
        '--bin-start',
        dest='START',
        type=float,
        required=False,
        default=None,
        help="Point on the chosen spatial direction to start binning."
             " Default: Minimum position in the given direction."
    )
    parser.add_argument(
        '--bin-end',
        dest='STOP',
        type=float,
        required=False,
        default=None,
        help="Point on the chosen spatial direction to stop binning."
             " Default: Maximum position in the given direction."
    )
    parser.add_argument(
        '--bin-num',
        dest='NUM',
        type=int,
        required=False,
        default=50,
        help="Number of bins to use. Default: 50"
    )
    parser.add_argument(
        '--bins',
        dest='BINFILE',
        type=str,
        required=False,
        default=None,
        help="ASCII formatted text file containing custom bin edges. Bin"
             " edges are read from the first column, lines starting with"
             " '#' are ignored. Bins do not need to be equidistant."
    )

    parser.add_argument(
        '--f2',
        dest='INFILE2',
        type=str,
        required=False,
        default=None,
        help="An optional second input file providing additional"
             " 1-dimensional data as a function of the spatial direction"
             " given with -d, e.g. a density profile. This data will be"
             " plotted above the other plot."
    )
    parser.add_argument(
        '-c',
        dest='COLS',
        type=int,
        nargs=2,
        required=False,
        default=[0, 1],
        help="From which columns of INFILE2 to read additional data."
             " Column numbering starts at 0. The first given number"
             " determines the column containing the x values, the second"
             " is for the y values. Default: '0 1'"
    )

    parser.add_argument(
        '--xmin',
        dest='XMIN',
        type=float,
        required=False,
        default=None,
        help="Minimum x-range of the plot. By default detected"
             " automatically."
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

    if (args.DIRECTION != 'x' and
        args.DIRECTION != 'y' and
            args.DIRECTION != 'z'):
        raise ValueError("-d must be either 'x', 'y' or 'z', but you"
                         " gave {}".format(args.DIRECTION))
    dim = {'x': 0, 'y': 1, 'z': 2}

    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()

    if args.SEL:
        cols = (3, 7 + dim[args.DIRECTION])
    else:
        cols = (3, 4 + dim[args.DIRECTION])
    trenew, pos_t0 = np.loadtxt(fname=args.INFILE,
                                usecols=cols,
                                unpack=True)
    trenew *= args.TCONV
    pos_t0 *= args.LCONV
    if args.SEL:
        cols = (13, 14, 15)
    else:
        cols = (10, 11, 12)
    displ = np.loadtxt(fname=args.INFILE, usecols=cols)
    displ *= args.LCONV

    if args.BINFILE is None:
        if args.START is None or args.START > np.min(pos_t0):
            args.START = np.min(pos_t0)
        if args.STOP is None or args.STOP <= np.max(pos_t0):
            args.STOP = np.max(pos_t0) + (np.max(pos_t0) - args.START) / args.NUM
        bins = np.linspace(args.START, args.STOP, args.NUM)
    else:
        bins = np.loadtxt(args.BINFILE, usecols=0)
        bins = np.unique(bins)
        if len(bins) == 0:
            raise ValueError("Invalid bins")
        if bins[0] > np.min(pos_t0):
            bins = np.insert(bins, 0, np.min(pos_t0))
        if bins[-1] <= np.max(pos_t0):
            bins = np.append(bins, np.max(pos_t0) + (np.max(pos_t0) - bins[0]) / len(bins))

    bin_ix = np.digitize(pos_t0, bins)
    displ_means = np.full((len(bins), displ.shape[1]), np.nan)
    displ_vars = np.full((len(bins), displ.shape[1]), np.nan)
    msd_means = np.full((len(bins), displ.shape[1]), np.nan)
    for i in np.unique(bin_ix):
        displ_tmp = displ[bin_ix == i]
        displ_means[i] = np.mean(displ_tmp, axis=0)
        displ_vars[i] = np.var(displ_tmp, axis=0)
        msd_means[i] = np.mean(displ_tmp**2, axis=0)
    if not np.all(np.isnan(displ_means[0])):
        raise ValueError("Not all first elements of displ_means are NaN."
                         " This should not have happened")
    if not np.all(np.isnan(displ_vars[0])):
        raise ValueError("Not all first elements of displ_vars are NaN."
                         " This should not have happened")
    if not np.all(np.isnan(msd_means[0])):
        raise ValueError("Not all first elements of msd_means are NaN."
                         " This should not have happened")

    if args.INFILE2 is not None:
        data = np.loadtxt(fname=args.INFILE2,
                          comments=['#', '@'],
                          usecols=args.COLS)

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
        ylabel = ('x', 'y', 'z')
        for i in range(displ.shape[1]):
            if args.INFILE2 is None:
                fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                         frameon=False,
                                         clear=True,
                                         tight_layout=True)
            else:
                fig, axes = plt.subplots(nrows=2,
                                         sharex=True,
                                         figsize=(11.69, 8.27 + 8.27 / 5),
                                         frameon=False,
                                         clear=True,
                                         constrained_layout=True,
                                         gridspec_kw={'height_ratios': [1 / 5, 1]})
                axis = axes[1]

            axis.axhline(y=0, color='black')
            img = mdt.plot.scatter(
                ax=axis,
                x=pos_t0,
                y=displ.T[i],
                c=trenew,
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                xlabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
                ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                marker='x',
                cmap='plasma')
            cbar = plt.colorbar(img, ax=axis)
            cbar.set_label(label=r'$\tau_{renew}$ / ' + args.TUNIT,
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
                            x=bins,
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
                x=bins[1:] - np.diff(bins) / 2,
                y=displ_means.T[i][1:],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                xlabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
                ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                label=r'$\langle \Delta ' + ylabel[i] + r' \rangle$',
                color='red',
                marker='o')

            mdt.plot.plot(
                ax=axis,
                x=bins[1:] - np.diff(bins) / 2,
                y=msd_means.T[i][1:],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                xlabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
                ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                label=r'$\langle \Delta ' + ylabel[i] + r'^2 \rangle$',
                color='blue',
                marker='^')

            mdt.plot.plot(
                ax=axis,
                x=bins[1:] - np.diff(bins) / 2,
                y=displ_vars.T[i][1:],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                xlabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
                ylabel=r'$\Delta ' + ylabel[i] + r'(\tau_{renew})$ / ' + args.LUNIT,
                label=r'$\langle \Delta ' + ylabel[i] + r'^2 \rangle - \langle \Delta ' + ylabel[i] + r' \rangle ^2$',
                color='green',
                marker='s')

            axis.legend(loc='lower left',
                        numpoints=1,
                        frameon=True,
                        fontsize=fontsize_legend)

            if args.INFILE2 is not None:
                mdt.plot.plot(ax=axes[0],
                              x=data[:, 0],
                              y=data[:, 1],
                              xmin=args.XMIN,
                              xmax=args.XMAX,
                              ymin=np.min(data[:, 1]),
                              ymax=np.max(data[:, 1]),
                              color='black')
                axes[0].xaxis.set_visible(False)
                axes[0].yaxis.set_visible(False)
                axes[0].spines['bottom'].set_visible(False)
                axes[0].spines['top'].set_visible(False)
                axes[0].spines['left'].set_visible(False)
                axes[0].spines['right'].set_visible(False)

            if args.INFILE2 is None:
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
