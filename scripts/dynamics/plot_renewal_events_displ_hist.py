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
from scipy.optimize import curve_fit
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
            " the displacement histograms for all three spatial"
            " directions."
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
        '--bin-start',
        dest='START',
        type=float,
        required=False,
        default=None,
        help="Displacement to start binning. Default: Minimum"
             " displacement"
    )
    parser.add_argument(
        '--bin-end',
        dest='STOP',
        type=float,
        required=False,
        default=None,
        help="Displacement to end binning. Default: Maximum displacement"
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
        default=0,
        help="Minimum y-range of the plot. Default:0"
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

    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()

    if args.SEL:
        cols = (13, 14, 15)
    else:
        cols = (10, 11, 12)
    displ = np.loadtxt(fname=args.INFILE, usecols=cols)
    displ *= args.LCONV

    mean = np.mean(displ, axis=0)
    std = np.std(displ, axis=0)
    non_gaus = mdt.stats.non_gaussian_parameter(displ,
                                                d=1,
                                                is_squared=False,
                                                axis=0)
    non_gaus_tot = mdt.stats.non_gaussian_parameter(
        np.sum(displ**2, axis=1),
        d=3,
        is_squared=True)

    if args.START is None:
        args.START = np.min(displ)
    if args.STOP is None:
        args.STOP = np.max(displ)
    hist = [None for i in range(displ.shape[1])]
    bins = [None for i in range(displ.shape[1])]
    for i, data in enumerate(displ.T):
        hist[i], bins[i] = np.histogram(data,
                                        bins=args.NUM,
                                        range=(args.START, args.STOP),
                                        density=True)

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print("\n\n\n", flush=True)
    print("Fitting histogram", flush=True)
    timer = datetime.now()

    x = [None for i in range(displ.shape[1])]
    popt = np.zeros((displ.shape[1], 2))
    pcov = np.zeros((displ.shape[1], 2, 2))
    for i in range(displ.shape[1]):
        x[i] = bins[i][1:] - np.diff(bins[i]) / 2
        popt[i], pcov[i] = curve_fit(f=mdt.stats.gaussian,
                                     xdata=x[i],
                                     ydata=hist[i],
                                     p0=(mean[i], std[i]))

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print("\n\n\n", flush=True)
    print("Creating plot", flush=True)
    timer = datetime.now()

    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        xlabel = (r'\Delta x', r'\Delta y', r'\Delta z')
        for i, data in enumerate(displ.T):
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            mdt.plot.hist(ax=axis,
                          x=data,
                          xmin=args.XMIN,
                          xmax=args.XMAX,
                          ymin=args.YMIN,
                          ymax=args.YMAX,
                          xlabel=r'$' + xlabel[i] + r'$ / ' + args.LUNIT,
                          ylabel=r'$p(' + xlabel[i] + r')$',
                          bins=bins[i],
                          range=(args.START, args.STOP),
                          density=True)
            mdt.plot.plot(
                ax=axis,
                x=x[i],
                y=mdt.stats.gaussian(x=x[i],
                                     mu=popt[i][0],
                                     sigma=popt[i][1]),
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                xlabel=r'$' + xlabel[i] + r'$ / ' + args.LUNIT,
                ylabel=r'$p(' + xlabel[i] + r')$',
                label="Fit",
                color='black')
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
                     s=r'$' + xlabel[i] + r'$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>+16.9e}$'.format(mean[i]),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(std[i]),
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
                 s=r'${:>+16.9e}$'.format(args.START),
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
                 s=r'${:>+16.9e}$'.format(args.STOP),
                 fontsize=fontsize)
        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos, y=ypos, s=r'$N$ bins', fontsize=fontsize)
        xpos += 0.40
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16d}$'.format(args.NUM),
                 fontsize=fontsize)

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
                 s=r'Non-Gaussian parameter: $A = \frac{\langle \Delta a^4 \rangle}{(1+\frac{2}{d}) \cdot \langle \Delta a^2 \rangle^2} - 1$',
                 fontsize=fontsize)
        xpos += 0.10
        ypos -= 0.08
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$\mu$ / ' + args.LUNIT,
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
                     s=r'$' + xlabel[i] + r'$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>+16.9e}$'.format(popt[i][0]),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(np.sqrt(np.diag(pcov[i])[0])),
                     fontsize=fontsize)

        xpos = 0.15
        ypos -= 0.08
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$\sigma$ / ' + args.LUNIT,
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
                     s=r'$' + xlabel[i] + r'$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(popt[i][1]),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(np.sqrt(np.diag(pcov[i])[1])),
                     fontsize=fontsize)

        xpos = 0.15
        ypos -= 0.08
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$A$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos, y=ypos, s=r'$d$', fontsize=fontsize)
        for i in range(displ.shape[1]):
            xpos = 0.05
            ypos -= 0.05
            plt.text(x=xpos,
                     y=ypos,
                     s=r'$' + xlabel[i] + r'$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>+16.9e}$'.format(non_gaus[i]),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos, y=ypos, s=r'$1$', fontsize=fontsize)
        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$\Delta r$',
                 fontsize=fontsize)
        xpos += 0.10
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>+16.9e}$'.format(non_gaus_tot),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos, y=ypos, s=r'$3$', fontsize=fontsize)

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
