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
from matplotlib.ticker import MaxNLocator
import mdtools as mdt




if __name__ == "__main__":

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())


    parser = argparse.ArgumentParser(
        description=(
            "Plot discretized trajectories generated with"
            " discrete_pos.py."
        )
    )

    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="File containing the discretized trajectories created by"
             " discrete_pos.py"
    )
    parser.add_argument(
        '--bins',
        dest='BINFILE',
        type=str,
        required=False,
        default=None,
        help="File containing the bin edges used to generate the"
             " discretized trajectory created by discrete_pos.py. If"
             " provided, the bins will be shown on a secondary y-axis."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename. Output is optimized for PDF format with"
             " TeX support."
    )

    parser.add_argument(
        '-t',
        dest="TRAJ_IX",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help="Space separated list of indices of particles for which to"
             " plot the discretized trajectories. Indexing starts at"
             " zero. Negative indices are counted backwards. By default,"
             " the trajectories of all particles plotted."
    )

    parser.add_argument(
        '--ylabel',
        dest='YLABEL',
        type=str,
        nargs="+",
        required=False,
        default=['$z$', '/', 'A'],
        help="String to use as secondary y-axis label. Is meaningless"
             " if --bins is not given. Note that you have to use TeX"
             " syntax. Default: '$z$ / A' (Note that you must either"
             " leave a space after dollar signs or enclose the"
             " expression in single quotes to avoid bash's variable"
             " expansion)."
    )
    parser.add_argument(
        '--decs',
        dest='DECS',
        type=int,
        required=False,
        default=1,
        help="Number of decimal places for the tick labels of the"
             " secondary y-axis. Is meaningless if --bins is not given."
             " Default: 1"
    )

    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())




    print("\n\n\n", flush=True)
    print("Loading discrete trajectories", flush=True)
    timer = datetime.now()

    dtrajs = np.load(args.TRJFILE)
    if dtrajs.ndim == 1:
        n_particles = 1
    elif dtrajs.ndim == 2:
        n_particles = dtrajs.shape[0]
    else:
        raise ValueError("dtrajs has more than two dimensions ({})"
                         .format(dtrajs.ndim))

    if args.TRAJ_IX is None:
        TRAJ_IX = np.arange(n_particles)
    else:
        TRAJ_IX = np.asarray(args.TRAJ_IX)
        pos = np.unique(TRAJ_IX[TRAJ_IX >= 0])
        neg = np.unique(TRAJ_IX[TRAJ_IX < 0])
        TRAJ_IX = np.concatenate((pos, neg))

    if (np.max(TRAJ_IX) >= n_particles or
            np.min(TRAJ_IX) < -n_particles):
        print("\n\n\n", flush=True)
        print("The particle indices you gave exceed the maximum number"
              " of particles in the input file ({}). Note that indexing"
              " starts at zero".format(n_particles), flush=True)
        TRAJ_IX = TRAJ_IX[TRAJ_IX < n_particles]
        TRAJ_IX = TRAJ_IX[TRAJ_IX >= -n_particles]
        print("Set TRAJ_IX to: {}".format(TRAJ_IX), flush=True)
    if TRAJ_IX.size == 0:
        raise ValueError("TRAJ_IX is empty. No trajectories selected")

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Generating y-axis", flush=True)
    timer = datetime.now()

    states = np.arange(np.max(dtrajs))

    if args.BINFILE is not None:
        print("  Loading bins from {}".format(args.BINFILE), flush=True)

        try:
            bins = np.load(args.BINFILE).astype(float)
            if bins[0] != 0:
                bin_width_first = bins[0] - 0
            else:
                bin_width_first = bins[1] - bins[0]
            bin_width_last = bins[-1] - bins[-2]

            print("    Start:            {:>12.6f}"
                  .format(0),
                  flush=True)
            print("    Stop:             {:>12.6f}"
                  .format(bins[-1]),
                  flush=True)
            print("    First bin width:  {:>12.6f}"
                  .format(bin_width_first),
                  flush=True)
            print("    Last bin width:   {:>12.6f}"
                  .format(bin_width_last),
                  flush=True)
            print("    Equidistant bins: {:>5s}"
                  .format(str(np.all(np.isclose(np.diff(bins),
                                                bin_width_last))
                              and np.isclose(bin_width_first,
                                             bin_width_last))),
                  flush=True)
            print("    Number of bins:   {:>5d}"
                  .format(len(bins)),
                  flush=True)

            bin_widths = np.insert(np.diff(bins), 0, bin_width_first)
            bins -= 0.5 * bin_widths

            YLABEL = ' '.join(args.YLABEL)
            YLABEL = "r'%s'" % YLABEL
            YLABEL = YLABEL[2:-1]

            states = np.arange(len(bins))

        except IOError:
            print("    {} not found".format(args.BINFILE), flush=True)
            print("    Will not plot secondary x-axis", flush=True)
            args.BINFILE = None

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Creating plots", flush=True)
    timer = datetime.now()

    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))

    offset = 0
    for i in TRAJ_IX:
        mdt.plot.plot(ax=axis,
                      x=np.arange(len(dtrajs[i])) + offset,
                      y=dtrajs[i],
                      xmin=0,
                      xmax=offset + len(dtrajs[i]),
                      ymin=states[0] - 0.5,
                      ymax=states[-1] + 0.5,
                      xlabel=r'Time / steps',
                      ylabel=r'State $i$')
        offset += len(dtrajs[i])
    axis.ticklabel_format(axis='x',
                          style='scientific',
                          scilimits=(0, 0),
                          useOffset=False)

    if args.BINFILE is not None:
        img, ax2 = mdt.plot.plot_2nd_yaxis(
            ax=axis,
            x=np.arange(len(dtrajs[i])) + offset - len(dtrajs[i]),
            y=dtrajs[i],
            ymin=states[0] - 0.5,
            ymax=states[-1] + 0.5,
            ylabel=YLABEL,
            alpha=0)
        ylim = axis.get_ylim()
        yticks = axis.get_yticks().astype(int)
        yticks = yticks[np.logical_and(yticks >= ylim[0], yticks <= ylim[1])]
        ax2.get_yaxis().set_ticks(yticks)
        yticklabels = np.around(bins[ax2.get_yticks()],
                                decimals=args.DECS)
        if args.DECS == 0:
            yticklabels = yticklabels.astype(int)
        if args.DECS < 0:
            yticklabels = [str(int(l))[:args.DECS] for l in yticklabels]
            ylabel = r'$' + args.DIRECTION + r'$ / $10^{' + str(abs(args.DECS)) + r'}$ A'
            ax2.set_ylabel(ylabel=ylabel)
        ax2.set_yticklabels(yticklabels)
        axis.set_xlim(xmin=0, xmax=offset, auto=True)

    mdt.fh.backup(args.OUTFILE)
    plt.tight_layout()
    plt.savefig(args.OUTFILE)
    plt.close(fig)
    print("  Created " + args.OUTFILE, flush=True)

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
