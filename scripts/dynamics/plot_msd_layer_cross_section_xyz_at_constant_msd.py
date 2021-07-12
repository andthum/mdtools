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
    proc = psutil.Process()

    parser = argparse.ArgumentParser(
        description=(
            "Plot cross sections as function of the layer"
            " position at constant MSDs from the output of"
            " msd_layer_serial.py (or msd_layer_parallel.py)"
            " for all three spatial components x, y and z in"
            " one plot. In other words, plot the time needed to"
            " reach a given MSD value as function of the layer."
        )
    )
    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        '-f',
        dest='INFILES',
        type=str,
        nargs=3,
        required=True,
        help="The output files of msd_layer_serial.py (or"
             " msd_layer_parallel.py) for all three spatial directions"
             " as space separated list in the order x y z."
    )
    parser.add_argument(
        '--fmd',
        dest='MDFILES',
        type=str,
        nargs=3,
        required=False,
        default=None,
        help="The corresponding output files of msd_layer_serial.py (or"
             " msd_layer_parallel.py) that contain the mean"
             " displacements. If provided the square of the"
             " mean displacements will be subtracted from the mean"
             " square displacements to correct for a potentially"
             " drifting system by calculating the variance"
             " <r^2> - <r>^2."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename. Plots are optimized for PDF format with"
             " TeX support."
    )
    group.add_argument(
        '--msd',
        dest='MSD',
        type=float,
        required=False,
        default=1,
        help="MSD for which to plot the times across all layers in the"
             " length unit given by TUNIT. Default: 1"
    )
    group.add_argument(
        '--msd-eq-bins',
        dest='MSD_EQ_BINS',
        required=False,
        default=False,
        action='store_true',
        help="Set MSD from --msd for each individual bin to the square"
             " of the respective bin width. Must not be used together"
             " with --msd."
    )
    parser.add_argument(
        '-d',
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
             " given with --d2, e.g. a density profile. This data will"
             " be plotted above the other plot."
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
        help="Minimum x-range of the plot. Default: Leftmost bin edge"
    )
    parser.add_argument(
        '--xmax',
        dest='XMAX',
        type=float,
        required=False,
        default=None,
        help="Maximum x-range of the plot. Default: Rightmost bin edge"
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
        '--logy',
        dest='LOGY',
        required=False,
        default=False,
        action='store_true',
        help="Use logarithmic scale for y-axis."
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

    if (args.BIN_DIRECTION != 'x' and
        args.BIN_DIRECTION != 'y' and
            args.BIN_DIRECTION != 'z'):
        raise ValueError("--d2 must be either 'x', 'y' or 'z', but you"
                         " gave {}".format(args.BIN_DIRECTION))
    dim = {'x': 0, 'y': 1, 'z': 2}

    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()

    msd = [None, ] * len(args.INFILES)
    times = [None, ] * len(args.INFILES)
    bins = [None, ] * len(args.INFILES)
    for i, infile in enumerate(args.INFILES):
        msd[i] = np.loadtxt(fname=infile)
        times[i] = msd[i][1:, 0] * args.TCONV
        bins[i] = msd[i][0] * args.LCONV
        msd[i] = msd[i][1:, 1:] * args.LCONV**2
    for i in range(len(args.INFILES)):
        if times[i].shape != times[0].shape:
            raise ValueError("The number of lag times in the"
                             " different input files does not match")
        if not np.allclose(times[i], times[0]):
            raise ValueError("The lag times of the different input"
                             " files do not match")
        if bins[i].shape != bins[0].shape:
            raise ValueError("The number of bin edges in the"
                             " different input files does not match")
        if not np.allclose(bins[i], bins[0]):
            raise ValueError("The bin edges of the different input"
                             " files do not match")
    times = np.asarray(times)
    bins = bins[0]
    msd = np.asarray(msd)

    if args.MDFILES is not None:
        md = [None, ] * len(args.MDFILES)
        for i, mdfile in enumerate(args.MDFILES):
            md[i] = np.loadtxt(fname=mdfile)
            times_md = md[i][1:, 0] * args.TCONV
            bins_md = md[i][0] * args.LCONV
            md[i] = md[i][1:, 1:] * args.LCONV
            if times_md.shape != times[0].shape:
                raise ValueError("The number of lag times in the"
                                 " different input files does not match")
            if not np.allclose(times_md, times[0]):
                raise ValueError("The lag times of the different input"
                                 " files do not match")
            if bins_md.shape != bins.shape:
                raise ValueError("The number of bin edges in the"
                                 " different input files does not match")
            if not np.allclose(bins_md, bins):
                raise ValueError("The bin edges of the different input"
                                 " files do not match")
        del times_md, bins_md
        md = np.asarray(md)
        np.square(md, out=md)
        msd -= md
        del md

    if args.MSD_EQ_BINS:
        args.MSD = np.diff(bins)**2
    msd, ix = mdt.nph.find_nearest(msd,
                                   args.MSD,
                                   axis=1,
                                   return_index=True)
    times = np.take_along_axis(times, ix, axis=1)

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

    fontsize_legend = 24
    labels = [r'$x$', r'$y$', r'$z$']
    markers = ['s', 'D', 'o']

    if args.XMIN is None:
        args.XMIN = np.min(bins)
    if args.XMAX is None:
        args.XMAX = np.max(bins)

    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
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

        if args.YMIN is None:
            args.YMIN = np.nanmin(times)
        if args.YMAX is None:
            args.YMAX = np.nanmax(times)
        for i in range(len(args.MDFILES)):
            mdt.plot.plot(
                ax=axis,
                x=bins[1:] - np.diff(bins) / 2,
                y=times[i],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                logy=args.LOGY,
                xlabel=r'${}(t_0)$ / {}'.format(args.BIN_DIRECTION,
                                                args.LUNIT),
                ylabel=r'$\Delta t$ / {}'.format(args.TUNIT),
                label=labels[i],
                marker=markers[i])
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
        if args.MDFILES is not None and not args.MSD_EQ_BINS:
            legend_title = (r'Var$[\Delta a(\Delta t)] =' +
                            str(args.MSD) + r'$ ' +
                            args.LUNIT + r'$^2$')
        elif args.MDFILES is not None and args.MSD_EQ_BINS:
            legend_title = r'Var$[\Delta a(\Delta t)] =$ (Bin Width)$^2$'
        elif args.MDFILES is None and not args.MSD_EQ_BINS:
            legend_title = (r'$\langle \Delta a^2(\Delta t) \rangle =' +
                            str(args.MSD) + r'$ ' +
                            args.LUNIT + r'$^2$')
        elif args.MDFILES is None and args.MSD_EQ_BINS:
            legend_title = r'$\langle \Delta a^2(\Delta t) \rangle =$ (Bin Width)$^2$'
        axis.legend(loc='upper center',
                    title=legend_title,
                    fontsize=fontsize_legend,
                    title_fontsize=fontsize_legend,
                    numpoints=1,
                    ncol=3,
                    labelspacing=0.2,
                    columnspacing=1.4,
                    handletextpad=0.5,
                    frameon=True,
                    fancybox=False)

        if args.INFILE2 is not None:
            mdt.plot.plot(ax=axes[0],
                          x=data[:, 0],
                          y=data[:, 1],
                          xmin=args.XMIN,
                          xmax=args.XMAX,
                          ymin=np.nanmin(data[:, 1]),
                          ymax=np.nanmax(data[:, 1]),
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

        if args.MDFILES is not None:
            ylabel = (r'True Var$[\Delta a(\Delta t)]$ / ' +
                      args.LUNIT + r'$^2$')
        else:
            ylabel = (r'True $\langle \Delta a^2(\Delta t) \rangle$ / ' +
                      args.LUNIT + r'$^2$')
        for i in range(len(args.MDFILES)):
            mdt.plot.plot(
                ax=axis,
                x=bins[1:] - np.diff(bins) / 2,
                y=msd[i],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=np.nanmin(msd),
                ymax=np.nanmax(msd),
                xlabel=r'${}(t_0)$ / {}'.format(args.BIN_DIRECTION,
                                                args.LUNIT),
                ylabel=ylabel,
                label=labels[i],
                marker=markers[i])
        mdt.plot.vlines(ax=axis,
                        x=bins,
                        start=axis.get_ylim()[0],
                        stop=axis.get_ylim()[1],
                        xmin=args.XMIN,
                        xmax=args.XMAX,
                        ymin=np.nanmin(msd),
                        ymax=np.nanmax(msd),
                        color='black',
                        linestyle='dotted')
        axis.legend(loc='best',
                    fontsize=fontsize_legend,
                    numpoints=1,
                    ncol=3,
                    labelspacing=0.2,
                    columnspacing=1.4,
                    handletextpad=0.5,
                    frameon=True,
                    fancybox=False)
        axis.ticklabel_format(axis='y', useOffset=False)

        if args.INFILE2 is not None:
            mdt.plot.plot(ax=axes[0],
                          x=data[:, 0],
                          y=data[:, 1],
                          xmin=args.XMIN,
                          xmax=args.XMAX,
                          ymin=np.nanmin(data[:, 1]),
                          ymax=np.nanmax(data[:, 1]),
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
