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
                     "Read a trajectory of renewal evenst as e.g."
                     " generated with extract_renewal_events.py and plot"
                     " the number of events versus the inital positions"
                     " as a bin-wise average. For the error estimation"
                     " it is assumed that the renewal events are"
                     " distributed in time according to a Poisson"
                     " distribution. In this case, the standard"
                     " deviation is simply the square root of the mean."
                     )
    )
    
    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=True,
        help="Trajectory of renewal evenst as e.g. generated with"
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
        '--box',
        dest='BOX',
        type=float,
        nargs=2,
        required=False,
        default=[1, 1],
        help="Box dimensions in the two other directions not given by -d."
             " In a third plot, the data read from INFILE2 will be"
             " interpreted as number density profile of the compounds"
             " along the direction given with -d. The number of events"
             " will be divided by the number of compounds per bin, which"
             " is calculated from the integral of the density profile in"
             " this bin multiplied by the box dimensions in the"
             " remaining two directions. Default: '1 1'"
    )
    parser.add_argument(
        '--name',
        dest='NAME',
        type=str,
        required=False,
        default="Compound",
        help="Name of the compound to use in the y-axis label of the"
             " second and third plot. These plots are only created if"
             " --f2 is provided. Default: 'Compound'"
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
        cols = 7+dim[args.DIRECTION]
    else:
        cols = 4+dim[args.DIRECTION]
    pos_t0 = np.loadtxt(fname=args.INFILE, usecols=cols)
    pos_t0 *= args.LCONV
    
    if args.BINFILE is None:
        if args.START is None or args.START > np.min(pos_t0):
            args.START = np.min(pos_t0)
        if args.STOP is None or args.STOP <= np.max(pos_t0):
            args.STOP = np.max(pos_t0) + (np.max(pos_t0)-args.START)/args.NUM
        bins = np.linspace(args.START, args.STOP, args.NUM)
    else:
        bins = np.loadtxt(args.BINFILE, usecols=0)
        bins = np.unique(bins)
        if len(bins) == 0:
            raise ValueError("Invalid bins")
        if bins[0] > np.min(pos_t0):
            bins = np.insert(bins, 0, np.min(pos_t0))
        if bins[-1] <= np.max(pos_t0):
            bins = np.append(bins, np.max(pos_t0) + (np.max(pos_t0)-bins[0])/len(bins))
    
    bin_ix = np.digitize(pos_t0, bins)
    nevents = np.zeros(len(bins), dtype=np.uint32)
    for i in np.unique(bin_ix):
        nevents[i] = np.count_nonzero(bin_ix==i)
    if nevents[0] != 0:
        raise ValueError("The first element of nevents is not zero. This"
                         " should not have happened")
    
    if args.INFILE2 is not None:
        xdata, ydata = np.loadtxt(fname=args.INFILE2,
                                  comments=['#', '@'],
                                  usecols=args.COLS,
                                  unpack=True)
        n_compounds_per_bin = np.zeros(len(bins), dtype=np.float64)
        for i, b in enumerate(bins[1:], 1):
            mask = (xdata >= bins[i-1]) & (xdata < b)
            n_compounds_per_bin[i] = np.trapz(x=xdata[mask],
                                              y=ydata[mask])
        if n_compounds_per_bin[0] != 0:
            raise ValueError("The first element of n_compounds_per_bin"
                             " is not zero. This should not have"
                             " happened")
        n_compounds_per_bin *= args.BOX[0] * args.BOX[1]
        print("  Total number of compounds: {}"
              .format(np.sum(n_compounds_per_bin)),
              flush=True)
    
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
    label_pad = 16
    
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        logy = (False, True)
        ymin = (args.YMIN, None)
        for i in range(len(logy)):
            # Number of renewal events per bin
            if args.INFILE2 is None:
                fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                         frameon=False,
                                         clear=True,
                                         tight_layout=True)
            else:
                fig, axes = plt.subplots(nrows=2,
                                         sharex=True,
                                         figsize=(11.69, 8.27+8.27/5),
                                         frameon=False,
                                         clear=True,
                                         constrained_layout=True,
                                         gridspec_kw={'height_ratios': [1/5, 1]})
                axis = axes[1]
            
            mdt.plot.errorbar(
                ax=axis,
                x=bins[1:]-np.diff(bins)/2,
                y=nevents[1:],
                yerr=np.sqrt(nevents[1:]),
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=ymin[i],
                ymax=args.YMAX,
                logy=logy[i],
                xlabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
                ylabel=r'$N_{renew}$',
                color='black',
                marker='o')
            
            mdt.plot.vlines(ax=axis,
                            x=bins,
                            start=axis.get_ylim()[0],
                            stop=axis.get_ylim()[1],
                            xmin=args.XMIN,
                            xmax=args.XMAX,
                            ymin=ymin[i],
                            ymax=args.YMAX,
                            color='black',
                            linestyle='dotted')
            
            if args.INFILE2 is not None:
                mdt.plot.plot(ax=axes[0],
                              x=xdata,
                              y=ydata,
                              xmin=args.XMIN,
                              xmax=args.XMAX,
                              ymin=np.min(ydata),
                              ymax=np.max(ydata),
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
            
            
            if args.INFILE2 is not None:
                # Number of compounds per bin
                fig, axes = plt.subplots(nrows=2,
                                         sharex=True,
                                         figsize=(11.69, 8.27+8.27/5),
                                         frameon=False,
                                         clear=True,
                                         constrained_layout=True,
                                         gridspec_kw={'height_ratios': [1/5, 1]})
                
                mdt.plot.plot(
                    ax=axes[1],
                    x=bins[1:]-np.diff(bins)/2,
                    y=n_compounds_per_bin[1:],
                    xmin=args.XMIN,
                    xmax=args.XMAX,
                    ymin=ymin[i],
                    ymax=args.YMAX,
                    logy=logy[i],
                    xlabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
                    ylabel=r'$\langle N_{'+args.NAME+r'} \rangle$',
                    color='black',
                    marker='o')
                
                mdt.plot.vlines(ax=axes[1],
                                x=bins,
                                start=axes[1].get_ylim()[0],
                                stop=axes[1].get_ylim()[1],
                                xmin=args.XMIN,
                                xmax=args.XMAX,
                                ymin=ymin[i],
                                ymax=args.YMAX,
                                color='black',
                                linestyle='dotted')
                
                mdt.plot.plot(ax=axes[0],
                              x=xdata,
                              y=ydata,
                              xmin=args.XMIN,
                              xmax=args.XMAX,
                              ymin=np.min(ydata),
                              ymax=np.max(ydata),
                              color='black')
                axes[0].xaxis.set_visible(False)
                axes[0].yaxis.set_visible(False)
                axes[0].spines['bottom'].set_visible(False)
                axes[0].spines['top'].set_visible(False)
                axes[0].spines['left'].set_visible(False)
                axes[0].spines['right'].set_visible(False)
                
                pdf.savefig()
                plt.close()
                
                
                # Number of renewal events / Number of  compounds per bin
                fig, axes = plt.subplots(nrows=2,
                                         sharex=True,
                                         figsize=(11.69, 8.27+8.27/5),
                                         frameon=False,
                                         clear=True,
                                         constrained_layout=True,
                                         gridspec_kw={'height_ratios': [1/5, 1]})
                
                mdt.plot.errorbar(
                    ax=axes[1],
                    x=bins[1:]-np.diff(bins)/2,
                    y=nevents[1:]/n_compounds_per_bin[1:],
                    yerr=np.sqrt(nevents[1:])/n_compounds_per_bin[1:],
                    xmin=args.XMIN,
                    xmax=args.XMAX,
                    ymin=ymin[i],
                    ymax=args.YMAX,
                    logy=logy[i],
                    xlabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
                    ylabel=r'$N_{renew}$ / $\langle N_{'+args.NAME+r'} \rangle$',
                    color='black',
                    marker='o')
                
                mdt.plot.vlines(ax=axes[1],
                                x=bins,
                                start=axes[1].get_ylim()[0],
                                stop=axes[1].get_ylim()[1],
                                xmin=args.XMIN,
                                xmax=args.XMAX,
                                ymin=ymin[i],
                                ymax=args.YMAX,
                                color='black',
                                linestyle='dotted')
                
                mdt.plot.plot(ax=axes[0],
                              x=xdata,
                              y=ydata,
                              xmin=args.XMIN,
                              xmax=args.XMAX,
                              ymin=np.min(ydata),
                              ymax=np.max(ydata),
                              color='black')
                axes[0].xaxis.set_visible(False)
                axes[0].yaxis.set_visible(False)
                axes[0].spines['bottom'].set_visible(False)
                axes[0].spines['top'].set_visible(False)
                axes[0].spines['left'].set_visible(False)
                axes[0].spines['right'].set_visible(False)
                
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
