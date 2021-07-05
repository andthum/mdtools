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
            " the mean square displacement (MSD) versus the"
            " renewal time as scatter plot. Additionaly, a"
            " running-average is computed and plotted."
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
    msd = np.loadtxt(fname=args.INFILE, usecols=cols)
    msd *= args.LCONV
    msd *= msd
    msd_tot = np.sum(msd, axis=1)
    if args.DCOLOR is not None:
        if args.SEL:
            cols = 7 + dim[args.DCOLOR]
        else:
            cols = 4 + dim[args.DCOLOR]
        pos_t0 = np.loadtxt(fname=args.INFILE, usecols=cols)
        pos_t0 *= args.LCONV

    sort_ix = np.argsort(trenew)
    running_average = mdt.stats.running_average(msd[sort_ix], axis=0)
    running_average_tot = mdt.stats.running_average(msd_tot[sort_ix])

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Fitting MSDs", flush=True)
    timer = datetime.now()

    displ0 = np.zeros(4, dtype=np.uint32)
    popt = np.zeros(4)
    pcov = np.zeros(4)
    for i, data in enumerate(msd.T):
        popt[i], pcov[i] = curve_fit(
            f=lambda t, D: mdt.dyn.msd(t=t, D=D, d=1),
            xdata=trenew,
            ydata=data)
    popt[-1], pcov[-1] = curve_fit(
        f=lambda t, D: mdt.dyn.msd(t=t, D=D, d=3),
        xdata=trenew,
        ydata=msd_tot)

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
    tick_length = 10
    label_pad = 16

    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        displ0[-1] = np.count_nonzero(msd_tot == 0)
        mask = (msd_tot > 0)
        if args.DCOLOR is None:
            mdt.plot.scatter(
                ax=axis,
                x=trenew[mask],
                y=msd_tot[mask],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                logx=True,
                logy=True,
                xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                ylabel=r'$\Delta r^2(\tau_{renew})$ / ' + args.LUNIT + r'$^2$',
                marker='x')
        else:
            img = mdt.plot.scatter(
                ax=axis,
                x=trenew[mask],
                y=msd_tot[mask],
                c=pos_t0[mask],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                logx=True,
                logy=True,
                xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                ylabel=r'$\Delta r^2(\tau_{renew})$ / ' + args.LUNIT + r'$^2$',
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
        mask = (running_average_tot > 0)
        mdt.plot.plot(
            ax=axis,
            x=trenew[mask][sort_ix[mask]],
            y=running_average_tot[mask],
            xmin=args.XMIN,
            xmax=args.XMAX,
            ymin=args.YMIN,
            ymax=args.YMAX,
            logx=True,
            logy=True,
            xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
            ylabel=r'$\Delta r^2(\tau_{renew})$ / ' + args.LUNIT + r'$^2$',
            label="Running average",
            color='red')
        fit = mdt.dyn.msd(t=trenew, D=popt[-1], d=3)
        mask = (fit > 0)
        mdt.plot.plot(
            ax=axis,
            x=trenew[mask],
            y=fit[mask],
            xmin=args.XMIN,
            xmax=args.XMAX,
            ymin=args.YMIN,
            ymax=args.YMAX,
            logx=True,
            logy=True,
            xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
            ylabel=r'$\Delta r^2(\tau_{renew})$ / ' + args.LUNIT + r'$^2$',
            label="Fit",
            color='black')
        plt.tight_layout()
        pdf.savefig()
        plt.close()


        ylabel = ('x', 'y', 'z')
        for i, data in enumerate(msd.T):
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            displ0[i] = np.count_nonzero(data == 0)
            mask = (data > 0)
            if args.DCOLOR is None:
                mdt.plot.scatter(
                    ax=axis,
                    x=trenew[mask],
                    y=data[mask],
                    xmin=args.XMIN,
                    xmax=args.XMAX,
                    ymin=args.YMIN,
                    ymax=args.YMAX,
                    logx=True,
                    logy=True,
                    xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                    ylabel=r'$\Delta ' + ylabel[i] + r'^2(\tau_{renew})$ / ' + args.LUNIT + r'$^2$',
                    marker='x')
            else:
                img = mdt.plot.scatter(
                    ax=axis,
                    x=trenew[mask],
                    y=data[mask],
                    c=pos_t0[mask],
                    xmin=args.XMIN,
                    xmax=args.XMAX,
                    ymin=args.YMIN,
                    ymax=args.YMAX,
                    logx=True,
                    logy=True,
                    xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                    ylabel=r'$\Delta ' + ylabel[i] + r'^2(\tau_{renew})$ / ' + args.LUNIT + r'$^2$',
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
            mask = (running_average.T[i] > 0)
            mdt.plot.plot(
                ax=axis,
                x=trenew[mask][sort_ix[mask]],
                y=running_average.T[i][mask],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                logx=True,
                logy=True,
                xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                ylabel=r'$\Delta ' + ylabel[i] + r'^2(\tau_{renew})$ / ' + args.LUNIT + r'$^2$',
                label="Running average",
                color='red')
            fit = mdt.dyn.msd(t=trenew, D=popt[i], d=1)
            mask = (fit > 0)
            mdt.plot.plot(
                ax=axis,
                x=trenew[mask],
                y=fit[mask],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                logx=True,
                logy=True,
                xlabel=r'$\tau_{renew}$ / ' + args.TUNIT,
                ylabel=r'$\Delta ' + ylabel[i] + r'^2(\tau_{renew})$ / ' + args.LUNIT + r'$^2$',
                label="Fit",
                color='black')
            plt.tight_layout()
            pdf.savefig()
            plt.close()


        # Statistics
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
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
                     s="Selection compound MSDs",
                     fontsize=fontsize)
        else:
            plt.text(x=xpos,
                     y=ypos,
                     s="Reference compound MSDs",
                     fontsize=fontsize)
        ypos -= 0.10
        plt.text(x=xpos, y=ypos, s="Statistics:", fontsize=fontsize)
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Tot. counts: {:>16d}'.format(len(msd_tot)),
                 fontsize=fontsize)
        xpos += 0.10
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Mean / ' + args.LUNIT + r'$^2$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Std. dev. / ' + args.LUNIT + r'$^2$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$N(\Delta a^2 = 0)$',
                 fontsize=fontsize)

        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos, y=ypos, s=r'$\Delta r^2$', fontsize=fontsize)
        xpos += 0.10
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(np.mean(msd_tot)),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(np.std(msd_tot)),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16d}$'.format(displ0[-1]),
                 fontsize=fontsize)
        for i, data in enumerate(msd.T):
            xpos = 0.05
            ypos -= 0.05
            plt.text(x=xpos,
                     y=ypos,
                     s=r'$\Delta ' + ylabel[i] + r'^2$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(np.mean(data)),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(np.std(data)),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16d}$'.format(displ0[i]),
                     fontsize=fontsize)

        # Fit parameters
        xpos = 0.05
        ypos -= 0.10
        plt.text(x=xpos, y=ypos, s="Fit parameters:", fontsize=fontsize)
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Fit function: $msd(t) = 2d \cdot D \cdot t$',
                 fontsize=fontsize)
        xpos += 0.10
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$D$ / ' + args.LUNIT + r'$^2$ ' + args.TUNIT + r'$^{-1}$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Std. dev. / ' + args.LUNIT + r'$^2$ ' + args.TUNIT + r'$^{-1}$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos, y=ypos, s=r'$d$', fontsize=fontsize)

        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos, y=ypos, s=r'$\Delta r^2$', fontsize=fontsize)
        xpos += 0.10
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(popt[-1]),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(np.sqrt(pcov[-1])),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16d}$'.format(3),
                 fontsize=fontsize)
        for i in range(3):
            xpos = 0.05
            ypos -= 0.05
            plt.text(x=xpos,
                     y=ypos,
                     s=r'$\Delta ' + ylabel[i] + r'^2$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(popt[i]),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(np.sqrt(pcov[i])),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16d}$'.format(1),
                     fontsize=fontsize)

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
