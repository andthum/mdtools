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
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mdtools as mdt




if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())


    parser = argparse.ArgumentParser(
        description=(
            "Read a trajectory of renewal events as e.g."
            " generated with extract_renewal_events.py and"
            " correlate the mean square displacements (MSDs)"
            " along two different spatial directions in a"
            " scatter plot."
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
        '--d1',
        dest='DIRECTION1',
        type=str,
        required=True,
        help="The first spatial direction. Must be either x, y or z."
    )
    parser.add_argument(
        '--d2',
        dest='DIRECTION2',
        type=str,
        required=True,
        help="The second spatial direction. Must be either x, y or z."
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
        '--d3',
        dest='DIRECTION3',
        type=str,
        required=False,
        default=None,
        help="You can restrict the plot to compounds whoose initial"
             " position in the direction given with --d3 falls into a"
             " specific region."
    )
    parser.add_argument(
        '--pos-t0-min',
        dest='POS_T0_MIN',
        type=float,
        required=False,
        default=None,
        help="Only consider compounds whose initial position in the"
             " direction given with --d3 is equal to or higher than this"
             " value. Is meaningless if --d3 is not given. Default:"
             " Minimum position"
    )
    parser.add_argument(
        '--pos-t0-max',
        dest='POS_T0_MAX',
        type=float,
        required=False,
        default=None,
        help="Only consider compounds whose initial position in the"
             " direction given with --d3 is less than this value. Is"
             " meaningless if --d3 is not given. Default: Maximum"
             " position plus one"
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


    if (args.DIRECTION1 != 'x' and
        args.DIRECTION1 != 'y' and
            args.DIRECTION1 != 'z'):
        raise ValueError("--d1 must be either 'x', 'y' or 'z', but you"
                         " gave {}".format(args.DIRECTION))
    if (args.DIRECTION2 != 'x' and
        args.DIRECTION2 != 'y' and
            args.DIRECTION2 != 'z'):
        raise ValueError("--d2 must be either 'x', 'y' or 'z', but you"
                         " gave {}".format(args.DIRECTION))
    if (args.DIRECTION3 is not None and
        args.DIRECTION3 != 'x' and
        args.DIRECTION3 != 'y' and
            args.DIRECTION3 != 'z'):
        raise ValueError("--d3 must be either 'x', 'y' or 'z', but you"
                         " gave {}".format(args.DIRECTION3))
    dim = {'x': 0, 'y': 1, 'z': 2}




    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()

    if args.SEL:
        cols = (3, 13+dim[args.DIRECTION1], 13+dim[args.DIRECTION2])
    else:
        cols = (3, 10+dim[args.DIRECTION1], 10+dim[args.DIRECTION2])
    trenew, displ1, displ2 = np.loadtxt(fname=args.INFILE,
                                        usecols=cols,
                                        unpack=True)
    trenew *= args.TCONV
    displ1 *= args.LCONV
    displ2 *= args.LCONV
    msd1 = displ1**2
    msd2 = displ2**2

    if args.DIRECTION3 is not None:
        if args.SEL:
            cols = 7+dim[args.DIRECTION3]
        else:
            cols = 4+dim[args.DIRECTION3]
        pos_t0 = np.loadtxt(fname=args.INFILE, usecols=cols)
        pos_t0 *= args.LCONV
        if args.POS_T0_MIN is not None and args.POS_T0_MAX is None:
            valid = (pos_t0 >= args.POS_T0_MIN)
        elif args.POS_T0_MIN is None and args.POS_T0_MAX is not None:
            valid = (pos_t0 < args.POS_T0_MAX)
        elif args.POS_T0_MIN is not None and args.POS_T0_MAX is not None:
            valid = ((pos_t0 >= args.POS_T0_MIN) &
                     (pos_t0 < args.POS_T0_MAX))
        if args.POS_T0_MIN is not None or args.POS_T0_MAX is not None:
            if not np.any(valid):
                raise ValueError("There is no compound that fulfills"
                                 " your --pos-t0-min --pos-t0-max"
                                 " condition")
            trenew = trenew[valid]
            displ1 = displ1[valid]
            displ2 = displ2[valid]
            msd1 = msd1[valid]
            msd2 = msd2[valid]
            pos_t0 = pos_t0[valid]

    r, p = pearsonr(x=msd1, y=msd2)

    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
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
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)

        mask = (msd1 > 0) & (msd2 > 0)
        img = mdt.plot.scatter(
            ax=axis,
            x=msd1[mask],
            y=msd2[mask],
            c=trenew[mask],
            xmin=args.XMIN,
            xmax=args.XMAX,
            ymin=args.YMIN,
            ymax=args.YMAX,
            logx=True,
            logy=True,
            xlabel=r'$\Delta '+args.DIRECTION1+r'^2(\tau_{renew})$ / '+args.LUNIT+r'$^2$',
            ylabel=r'$\Delta '+args.DIRECTION2+r'^2(\tau_{renew})$ / '+args.LUNIT+r'$^2$',
            marker='x',
            cmap='plasma')
        cbar = plt.colorbar(img, ax=axis)
        cbar.set_label(label=r'$\tau_{renew}$ / '+args.TUNIT,
                       fontsize=fontsize_labels)
        cbar.ax.yaxis.labelpad = label_pad
        cbar.ax.yaxis.offsetText.set(size=fontsize_ticks)
        cbar.ax.tick_params(which='major',
                            direction='out',
                            length=tick_length,
                            labelsize=fontsize_ticks)
        cbar.ax.tick_params(which='minor',
                            direction='out',
                            length=0.5*tick_length,
                            labelsize=0.8*fontsize_ticks)

        diagonal = np.linspace(*axis.get_xlim())
        mdt.plot.plot(
            ax=axis,
            x=diagonal,
            y=diagonal,
            xmin=args.XMIN,
            xmax=args.XMAX,
            ymin=args.YMIN,
            ymax=args.YMAX,
            logx=True,
            logy=True,
            xlabel=r'$\Delta '+args.DIRECTION1+r'^2(\tau_{renew})$ / '+args.LUNIT+r'$^2$',
            ylabel=r'$\Delta '+args.DIRECTION2+r'^2(\tau_{renew})$ / '+args.LUNIT+r'$^2$',
            color='black',
            linestyle='--')

        axis.axvline(x=np.mean(msd1),
                     color='blue',
                     linestyle='-.',
                     label=r'$\langle \Delta a^2 \rangle$')
        axis.axvline(x=np.var(displ1),
                     color='green',
                     linestyle='dotted',
                     label=r'$\langle \Delta a^2 \rangle - \langle \Delta a \rangle ^2$')
        axis.axhline(y=np.mean(msd2), color='blue', linestyle='-.')
        axis.axhline(y=np.var(displ2), color='green', linestyle='dotted')

        axis.legend(loc='upper center',
                    bbox_to_anchor=(0.5, 1.2),
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
                     s="Selection compound MSDs",
                     fontsize=fontsize)
        else:
            plt.text(x=xpos,
                     y=ypos,
                     s="Reference compound MSDs",
                     fontsize=fontsize)

        # Input
        if args.DIRECTION3 is not None:
            ypos -= 0.10
            plt.text(x=xpos, y=ypos, s="Input:", fontsize=fontsize)
            ypos -= 0.05
            if args.POS_T0_MIN is None:
                plt.text(x=xpos,
                         y=ypos,
                         s=r'$'+args.DIRECTION3+r' \geq None$',
                         fontsize=fontsize)
            else:
                plt.text(x=xpos,
                         y=ypos,
                         s=(r'$'+args.DIRECTION3 +
                            r' \geq {:>16.9e}$'.format(args.POS_T0_MIN) +
                            r' '+args.LUNIT),
                         fontsize=fontsize)
            ypos -= 0.05
            if args.POS_T0_MAX is None:
                plt.text(x=xpos,
                         y=ypos,
                         s=r'$'+args.DIRECTION3+r' < None$',
                         fontsize=fontsize)
            else:
                plt.text(x=xpos,
                         y=ypos,
                         s=(r'$'+args.DIRECTION3 +
                            r' < {:>16.9e}$'.format(args.POS_T0_MAX) +
                            r' '+args.LUNIT),
                         fontsize=fontsize)

        # Statistics
        ypos -= 0.10
        plt.text(x=xpos, y=ypos, s="Statistics:", fontsize=fontsize)
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Tot. counts: ${:>16d}$'.format(len(msd1)),
                 fontsize=fontsize)
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Pearson correlation: ${:>16.9e}$'.format(r),
                 fontsize=fontsize)
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'p-value: ${:>16.9e}$'.format(p),
                 fontsize=fontsize)
        xpos += 0.10
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Mean / '+args.LUNIT+r'$^2$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Std. dev. / '+args.LUNIT+r'$^2$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$N(\Delta a^2 = 0)$',
                 fontsize=fontsize)

        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$\Delta '+args.DIRECTION1+r'^2$',
                 fontsize=fontsize)
        xpos += 0.10
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(np.mean(msd1)),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(np.std(msd1)),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16d}$'.format(np.count_nonzero(msd1==0)),
                 fontsize=fontsize)

        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$\Delta '+args.DIRECTION2+r'^2$',
                 fontsize=fontsize)
        xpos += 0.10
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(np.mean(msd2)),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(np.std(msd2)),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16d}$'.format(np.count_nonzero(msd2==0)),
                 fontsize=fontsize)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print("  Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)




    print("\n\n\n{} done".format(os.path.basename(sys.argv[0])))
    print("Elapsed time:         {}"
          .format(datetime.now()-timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
