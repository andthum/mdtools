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


def as_si(x, ndp):
    """
    https://stackoverflow.com/questions/31453422/displaying-numbers-with-x-instead-of-e-scientific-notation-in-matplotlib/31453961
    """
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())

    parser = argparse.ArgumentParser(
        description=(
            "Plot cross sections as function of the layer"
            " position at constant times from the output of"
            " msd_layer_serial.py (or msd_layer_parallel.py)."
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
        '--fmd',
        dest='MDFILE',
        type=str,
        nargs='+',
        required=False,
        default=None,
        help="The output files of msd_layer_serial.py (or"
             " msd_layer_parallel.py) that contain the mean displacement"
             " (either one or all three). If provided the square of the"
             " mean displacement will be subtracted from the mean square"
             " displacement to correct for a potentially drifting system"
             " by calculating the variance <r^2> - <r>^2."
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
        '-t',
        dest='TIMES',
        type=float,
        nargs='+',
        required=False,
        default=[1e-2, 1e-1, 1e+0, 1e+1, 1e+2],
        help="Space separated list of times for which to plot the MSDs"
             " across all layers. Default: 1e-2 1e-1 1e+0 1e+1 1e+2"
    )
    parser.add_argument(
        '--d1',
        dest='MSD_DIRECTION',
        type=str,
        required=False,
        default='r',
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

    if (args.MSD_DIRECTION != 'r' and
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
    times = msd[1:, 0] * args.TCONV
    bins = msd[0] * args.LCONV
    msd = msd[1:, 1:] * args.LCONV**2

    TIMES = np.unique(args.TIMES)
    if len(TIMES) <= 0:
        raise ValueError("Invalid --times")
    tix = [None, ] * len(TIMES)
    for i, t in enumerate(args.TIMES):
        _, tix[i] = mdt.nph.find_nearest(times, t, return_index=True)
    tix = np.asarray(tix)
    msd = msd[tix]

    if args.MDFILE is not None:
        if len(args.MDFILE) != 1 and len(args.MDFILE) != 3:
            raise ValueError("You must provide either one or three"
                             " additional input files with --fmd")
        md = [None, ] * len(args.MDFILE)
        for i, mdfile in enumerate(args.MDFILE):
            md[i] = np.loadtxt(fname=mdfile)
            times_md = md[i][1:, 0] * args.TCONV
            bins_md = md[i][0] * args.LCONV
            md[i] = md[i][1:, 1:] * args.LCONV
            if times_md.shape != times.shape:
                raise ValueError("The number of lag times in the"
                                 " different input files does not match")
            if not np.allclose(times_md, times):
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
        md = md[:, tix]
        msd -= np.sum(md**2, axis=0)
        md = np.sum(md, axis=0)
    times = times[tix]

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

    if args.XMIN is None:
        args.XMIN = np.min(bins)
    if args.XMAX is None:
        args.XMAX = np.max(bins)

    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        if args.MDFILE is not None:
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
                ymin = np.min(md)
            else:
                ymin = args.YMIN
            if args.YMAX is None:
                ymax = np.max(md)
                ymax += 0.1 * ymax
            else:
                ymax = args.YMAX
            if args.YMIN is None:
                ymin -= 0.1 * ymax
            if args.MSD_DIRECTION != 'r':
                ylabel = (r'$\langle \Delta ' + args.MSD_DIRECTION +
                          r'(\Delta t) \rangle$ / ' + args.LUNIT)
            else:
                ylabel = (r'$\langle \Delta x(\Delta t) \rangle' +
                          r'+ \langle \Delta y(\Delta t) \rangle' +
                          r'+ \langle \Delta z(\Delta t) \rangle$ / ' +
                          args.LUNIT)
            axis.axhline(y=0, color='black')
            for i, time in enumerate(times):
                mdt.plot.plot(
                    ax=axis,
                    x=bins[1:] - np.diff(bins) / 2,
                    y=md[i],
                    xmin=args.XMIN,
                    xmax=args.XMAX,
                    ymin=ymin,
                    ymax=ymax,
                    xlabel=r'${}(t_0)$ / {}'.format(args.BIN_DIRECTION,
                                                    args.LUNIT),
                    ylabel=ylabel,
                    label=r'${}$'.format(as_si(time, 2)),
                    marker='o')
            mdt.plot.vlines(ax=axis,
                            x=bins,
                            start=axis.get_ylim()[0],
                            stop=axis.get_ylim()[1],
                            xmin=args.XMIN,
                            xmax=args.XMAX,
                            ymin=ymin,
                            ymax=ymax,
                            color='black',
                            linestyle='dotted')
            axis.legend(loc='best',
                        title=r'$\Delta t$ / {}'.format(args.TUNIT),
                        title_fontsize=fontsize_legend,
                        fontsize=fontsize_legend,
                        numpoints=1,
                        ncol=1 + len(times) // 6,
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
            ymin = np.min(msd)
        else:
            ymin = args.YMIN
        if args.YMAX is None:
            ymax = np.max(msd)
            ymax += 0.1 * ymax
        else:
            ymax = args.YMAX
        if args.YMIN is None and not args.LOGY:
            ymin -= 0.1 * ymax
        elif args.YMIN is None and args.LOGY:
            ymin -= 0.1 * ymin
        if args.YMIN and args.LOGY <= 0:
            ymin = np.min(msd)
        if args.MDFILE is not None:
            ylabel = (r'$[\langle \Delta ' + args.MSD_DIRECTION +
                      r'^2(\Delta t) \rangle - ' +
                      r'\langle \Delta ' + args.MSD_DIRECTION +
                      r'(\Delta t) \rangle^2]$' +
                      r' / ' + args.LUNIT + r'$^2$')
        else:
            ylabel = (r'$\langle \Delta ' + args.MSD_DIRECTION +
                      r'^2(\Delta t) \rangle$' +
                      r' / ' + args.LUNIT + r'$^2$')
        for i, time in enumerate(times):
            mdt.plot.plot(
                ax=axis,
                x=bins[1:] - np.diff(bins) / 2,
                y=msd[i],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=ymin,
                ymax=ymax,
                logy=args.LOGY,
                xlabel=r'${}(t_0)$ / {}'.format(args.BIN_DIRECTION,
                                                args.LUNIT),
                ylabel=ylabel,
                label=r'${}$'.format(as_si(time, 2)),
                marker='o')
        mdt.plot.vlines(ax=axis,
                        x=bins,
                        start=axis.get_ylim()[0],
                        stop=axis.get_ylim()[1],
                        xmin=args.XMIN,
                        xmax=args.XMAX,
                        ymin=ymin,
                        ymax=ymax,
                        color='black',
                        linestyle='dotted')
        axis.legend(loc='best',
                    title=r'$\Delta t$ / {}'.format(args.TUNIT),
                    title_fontsize=fontsize_legend,
                    fontsize=fontsize_legend,
                    numpoints=1,
                    ncol=1 + len(times) // 6,
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
