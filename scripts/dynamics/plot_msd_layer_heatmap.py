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
import mdtools as mdt


if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())

    parser = argparse.ArgumentParser(
        description=(
            "Plot the output of msd_layer_serial.py (or"
            " msd_layer_parallel.py) as heatmap."
        )
    )

    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=True,
        help="One of the output files of msd_layer_serial.py (or"
             " msd_layer_parallel.py)."
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
        '--d1',
        dest='MSD_DIRECTION',
        type=str,
        required=False,
        default='z',
        help="Which component of the MSD is contained in the input file."
             " Must be either r, x, y or z. Default: r"
    )
    parser.add_argument(
        '--d2',
        dest='BIN_DIRECTION',
        type=str,
        required=False,
        default='z',
        help="The spatial direction used to dicretize the MSD. Must be"
             " either x, y or z. Default: z"
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
             " plotted above the heatmap."
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
        help="Minimum x-range of the plot. Default: The first bin edge"
    )
    parser.add_argument(
        '--xmax',
        dest='XMAX',
        type=float,
        required=False,
        default=None,
        help="Maximum x-range of the plot. Default: The last bin edge"
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
        '--vmin',
        dest='VMIN',
        type=float,
        required=False,
        default=None,
        help="Minimum data range of the colorbar. By default detected"
             " automatically."
    )
    parser.add_argument(
        '--vmax',
        dest='VMAX',
        type=float,
        required=False,
        default=None,
        help="Maximum data range of the colorbar. By default detected"
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

    if (args.MSD_DIRECTION != '3' and
        args.MSD_DIRECTION != 'x' and
        args.MSD_DIRECTION != 'y' and
            args.MSD_DIRECTION != 'z'):
        raise ValueError("--d1 must be either 'r, 'x', 'y' or 'z', but"
                         " you gave {}".format(args.MSD_DIRECTION))
    if (args.BIN_DIRECTION != 'x' and
        args.BIN_DIRECTION != 'y' and
            args.BIN_DIRECTION != 'z'):
        raise ValueError("--d2 must be either 'x', 'y' or 'z', but you"
                         " gave {}".format(args.BIN_DIRECTION))
    dim = {'x': 0, 'y': 1, 'z': 2}

    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()

    msd = np.loadtxt(fname=args.INFILE)
    bins = msd[0] * args.LCONV
    times = msd[1:, 0] * args.TCONV
    msd = msd[1:, 1:] * args.LCONV**2

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

    times = np.append(times, times[-1] + (times[-1] - times[-2]))
    dt = np.append(np.diff(times), times[-1] - times[-2])
    times -= dt / 2
    mdt.plot.pcolormesh(
        ax=axis,
        x=bins,
        y=times,
        z=msd,
        xmin=args.XMIN,
        xmax=args.XMAX,
        ymin=args.YMIN,
        ymax=args.YMAX,
        xlabel=r'${}$ / {}'.format(args.BIN_DIRECTION, args.LUNIT),
        ylabel=r'$\Delta t$ / {}'.format(args.TUNIT),
        cbarlabel=r'$\langle \Delta ' + args.BIN_DIRECTION + r'^2(\Delta t) \rangle$ / ' + args.LUNIT + r'$^2$' + args.TUNIT + r'$^{-1}$',
        cmap='plasma')

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

    mdt.fh.backup(args.OUTFILE)
    if args.INFILE2 is None:
        plt.tight_layout()
    plt.savefig(args.OUTFILE)
    plt.close(fig)

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
