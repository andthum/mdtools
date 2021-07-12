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
    proc = psutil.Process()

    parser = argparse.ArgumentParser(
        description=(
            "Read a trajectory of renewal events as e.g."
            " generated with extract_renewal_events.py and plot"
            " the new starting time after a renewal event"
            " versus the end time of the preceding renewal"
            " event as scatter plot."
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
        default=0,
        help="Minimum y-range of the plot. Default: 0"
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

    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()

    if args.SEL:
        cols = (1, 2, 3)
    else:
        cols = (0, 2, 3)
    compound_ix, t0, trenew = np.loadtxt(fname=args.INFILE,
                                         usecols=cols,
                                         unpack=True)
    t0 *= args.TCONV
    trenew *= args.TCONV

    sort_ix = np.lexsort((t0, compound_ix))
    compound_ix = compound_ix[sort_ix]
    t0 = t0[sort_ix]
    trenew = trenew[sort_ix]

    t0_new_event = []
    tend_preceding_event = []
    for i, cix in enumerate(compound_ix[1:], 1):
        if cix != compound_ix[i - 1]:
            continue
        t0_new_event.append(t0[i])
        tend_preceding_event.append(t0[i - 1] + trenew[i - 1])
    t0_new_event = np.asarray(t0_new_event)
    tend_preceding_event = np.asarray(tend_preceding_event)

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print("\n\n\n", flush=True)
    print("Creating plot", flush=True)
    timer = datetime.now()

    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)

    img = mdt.plot.scatter(
        ax=axis,
        x=t0_new_event,
        y=tend_preceding_event,
        xmin=args.XMIN,
        xmax=args.XMAX,
        ymin=args.YMIN,
        ymax=args.YMAX,
        xlabel=r'$t_0 + \tau_{renew}$ / ' + args.TUNIT,
        ylabel=r'$t_0^\prime$ / ' + args.TUNIT,
        marker='x')

    diagonal = np.linspace(*axis.get_xlim())
    mdt.plot.plot(
        ax=axis,
        x=diagonal,
        y=diagonal,
        xmin=args.XMIN,
        xmax=args.XMAX,
        ymin=args.YMIN,
        ymax=args.YMAX,
        xlabel=r'$t_0 + \tau_{renew}$ / ' + args.TUNIT,
        ylabel=r'$t_0^\prime$ / ' + args.TUNIT,
        color='black')

    mdt.fh.backup(args.OUTFILE)
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
