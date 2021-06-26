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
                     "Plot implied timescales from a txt file as created"
                     " by pyemma_its.py"
                 )
    )
    
    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        nargs="+",
        required=True,
        help="Input file contatinng as first column the lag times and in"
             " the following columns the corresponding implied"
             " timescales. Lines starting with '#' are ignored. You can"
             " also give multiple input files, to compare the first"
             " --nits timescales of different models."
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
        '--nits',
        dest='NITS',
        type=int,
        required=False,
        default=None,
        help="Number of implied timescales to plot. Must not exceed the"
             " number of implied timescales in the input file(s)."
             " Default: None, which means plot all timescales."
    )
    parser.add_argument(
        '--errors',
        dest='ERRORS',
        required=False,
        default=False,
        action='store_true',
        help="Plot error estimates. If set, every second column of the"
             " columns contatinng the implied timescales is assumed to"
             " contain the error estimate of the foregoing column."
    )
    
    parser.add_argument(
        '--xmin',
        dest='XMIN',
        type=float,
        required=False,
        default=0,
        help="Minimum x-range of plot. Default: 0."
    )
    parser.add_argument(
        '--xmax',
        dest='XMAX',
        type=float,
        required=False,
        default=None,
        help="Maximum x-range of plot. By default detected automatically."
    )
    parser.add_argument(
        '--ymin',
        dest='YMIN',
        type=float,
        required=False,
        default=0,
        help="Minimum y-range of plot. Default: 0."
    )
    parser.add_argument(
        '--ymax',
        dest='YMAX',
        type=float,
        required=False,
        default=None,
        help="Maximum y-range of plot. By default detected automatically."
    )
    parser.add_argument(
        '--logx',
        dest='LOGX',
        required=False,
        default=False,
        action='store_true',
        help="Plot x-axis in logscale. If set, --xmin will be set to 0.9"
             " if it is not positive."
    )
    parser.add_argument(
        '--logy',
        dest='LOGY',
        required=False,
        default=False,
        action='store_true',
        help="Plot y-axis in logscale. If set, --ymin will be set to the"
             " minium non-negative value if it is not positive."
    )
    parser.add_argument(
        '--label',
        dest='LABEL',
        type=str,
        nargs="+",
        required=False,
        default=None,
        help="Give a label for each input file. Default: None"
    )
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    
    
    
    print("\n\n\n", flush=True)
    print("Loading text file", flush=True)
    timer = datetime.now()
    
    if args.NITS is None:
        cols = None
    elif args.ERRORS:
        cols = np.arange(args.NITS*2+1)
    else:
        cols = np.arange(args.NITS+1)
    
    lags = []
    its = []
    err = []
    for infile in args.INFILE:
        data = np.genfromtxt(infile, usecols=cols)
        data = data.T
        lags.append(data[0])
        if args.ERRORS:
            its.append(data[1::2])
            err.append(data[2::2])
        else:
            its.append(data[1:])
    del data
    lags = np.array(lags)
    its = np.array(its)
    err = np.array(err)
    
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
    tick_length = 10
    tick_pad = 12
    label_pad = 16
    
    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.ticklabel_format(axis='both',
                          style='scientific',
                          scilimits=(0,0),
                          useOffset=False)
    if args.LOGX and args.XMIN <= 0:
        args.XMIN = 0.9
    if args.LOGY and args.YMIN <= 0:
        args.YMIN = np.min(its[its>0])
    ls = ['-', '--', '-.', ':',  '-', '--', '-.', ':']
    if args.LABEL is None:
        args.LABEL = [None,] * len(its)
    for i in range(len(its)):
        axis.set_prop_cycle(None)
        for j in range(len(its[i])):
            if args.ERRORS:
                axis.fill_between(x=lags[i],
                                  y1=its[i][j]-err[i][j],
                                  y2=its[i][j]+err[i][j],
                                  alpha=0.5)
            mdt.plot.plot(ax=axis,
                          x=lags[i],
                          y=its[i][j],
                          xmin=args.XMIN,
                          xmax=args.XMAX,
                          ymin=args.YMIN,
                          ymax=args.YMAX,
                          logx=args.LOGX,
                          logy=args.LOGY,
                          xlabel=r'Lag time / steps',
                          ylabel=r'Timescale / steps',
                          label=args.LABEL[i] if j==0 else None,
                          marker='o',
                          linestyle=ls[i])
    axis.set_xlim(xmin=args.XMIN, xmax=args.XMAX)
    axis.set_ylim(ymin=args.YMIN, ymax=args.YMAX)
    
    mdt.fh.backup(args.OUTFILE)
    plt.tight_layout()
    plt.savefig(args.OUTFILE)
    plt.close(fig)
    print("  Created "+args.OUTFILE, flush=True)
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("{} done".format(os.path.basename(sys.argv[0])), flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
