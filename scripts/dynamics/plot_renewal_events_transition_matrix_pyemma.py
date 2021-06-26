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
import warnings
from datetime import datetime
import psutil
import argparse
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import pyemma.msm as msm
import mdtools as mdt




if __name__ == '__main__':
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read a trajectory of renewal events as e.g."
                     " generated with extract_renewal_events.py,"
                     " discretize a given spatial direction and create"
                     " and plot a row-stochastic transition matrix"
                     " similar to the transition matrix of a Markov"
                     " model. The matrix element T_ij represents the"
                     " probability that a renewal event which starts in"
                     " bins i ends in bin j."
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
        '-d',
        dest='DIRECTION',
        type=str,
        required=False,
        default='z',
        help="The spatial direction to dicretize. Must be either x, y"
             " or z. Default: z"
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
        '--min',
        dest='MIN',
        type=float,
        required=False,
        default=None,
        help="Minimum x- and y-range of the plot. By default detected"
             " automatically."
    )
    parser.add_argument(
        '--max',
        dest='MAX',
        type=float,
        required=False,
        default=None,
        help="Maximum x- and y-range of the plot. By default detected"
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
    
    t0, trenew = np.loadtxt(fname=args.INFILE,
                            usecols=(2, 3),
                            unpack=True)
    if args.SEL:
        cols = (1, 7+dim[args.DIRECTION], 13+dim[args.DIRECTION])
    else:
        cols = (0, 4+dim[args.DIRECTION], 10+dim[args.DIRECTION])
    compound_ix, pos_t0, pos_trenew = np.loadtxt(fname=args.INFILE,
                                                 usecols=cols,
                                                 unpack=True)
    pos_t0 *= args.LCONV
    pos_trenew *= args.LCONV
    pos_trenew += pos_t0
    
    sort_ix = np.lexsort((t0, compound_ix))
    compound_ix = compound_ix[sort_ix]
    t0 = t0[sort_ix]
    trenew = trenew[sort_ix]
    pos_t0 = pos_t0[sort_ix]
    pos_trenew = pos_trenew[sort_ix]
    
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
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Generating discretized single-particle trajectories",
          flush=True)
    timer = datetime.now()
    
    t1 = t0 + trenew
    bin_ix_t0 = np.digitize(pos_t0, bins) - 1
    bin_ix_trenew = np.digitize(pos_trenew, bins) - 1
    if np.any(bin_ix_t0 < 0):
        raise ValueError("At least one element of bin_ix_t0 is less"
                         " than zero. This should not have happened")
    if np.any(bin_ix_trenew < 0):
        raise ValueError("At least one element of bin_ix_trenew is less"
                         " than zero. This should not have happened")
    dtrajs = []
    dtraj = [bin_ix_t0[0]]
    for i, cix in enumerate(compound_ix[:-1]):
        dtraj.append(bin_ix_trenew[i])
        if (cix != compound_ix[i+1] or
            not np.isclose(t1[i], t0[i+1]) or
            not np.isclose(pos_trenew[i], pos_t0[i+1])):
            dtrajs.append(np.asarray(dtraj))
            dtraj = [bin_ix_t0[i+1]]
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating Markov model", flush=True)
    timer = datetime.now()
    
    mm = msm.estimate_markov_model(dtrajs=dtrajs, lag=1)
    del dtrajs
    active_set = np.arange(len(bins)-1, dtype=np.uint32)
    inactive = np.setdiff1d(active_set, mm.active_set, assume_unique=True)
    if not np.allclose(np.sum(mm.transition_matrix, axis=1), 1):
        raise ValueError("Not all rows of the transition matrix sum up"
                         " to unity")
    if not np.isclose(np.sum(mm.stationary_distribution), 1):
        raise ValueError("The sum of the stationary distribution ({})"
                         " is not unity"
                         .format(np.sum(mm.stationary_distribution)))
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating plot", flush=True)
    timer = datetime.now()
    
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        # Transition matrix as function of bin number
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        xy = np.append(mm.active_set, mm.active_set[-1]+1) + 0.5
        mdt.plot.pcolormesh(ax=axis,
                            x=xy,
                            y=xy,
                            z=mm.transition_matrix,
                            xmin=0.5,
                            xmax=len(bins)-0.5,
                            ymin=0.5,
                            ymax=len(bins)-0.5,
                            xlabel=r'Bin $j$',
                            ylabel=r'Bin $i$',
                            cbarlabel=r'Transition probability $T_{ij}$')
        
        axis.invert_yaxis()
        axis.xaxis.set_label_position('top')
        axis.xaxis.labelpad = 22
        axis.xaxis.tick_top()
        axis.tick_params(axis='x', which='both', pad=6)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        # Stationary distribution as function of bin number
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        mdt.plot.plot(ax=axis,
                      x=mm.active_set+1,
                      y=mm.stationary_distribution,
                      xmin=0.5,
                      xmax=len(bins)-0.5,
                      ymin=0,
                      xlabel=r'Bin $i$',
                      ylabel="Stationary distribution",
                      color='black',
                      marker='o')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        
        
        # Transition matrix as function of spatial direction
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        
        if len(inactive) > 0:
            transition_matrix = np.insert(mm.transition_matrix.astype(np.float64),
                                          inactive,
                                          np.nan,
                                          axis=0)
            transition_matrix = np.insert(transition_matrix,
                                          inactive,
                                          np.nan,
                                          axis=1)
        else:
            transition_matrix = mm.transition_matrix
        mdt.plot.pcolormesh(ax=axis,
                            x=bins,
                            y=bins,
                            z=transition_matrix,
                            xmin=args.MIN,
                            xmax=args.MAX,
                            ymin=args.MIN,
                            ymax=args.MAX,
                            xlabel=r'$'+args.DIRECTION+r'(t_0 + \tau_{renew})$ / '+args.LUNIT,
                            ylabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
                            cbarlabel=r'Transition probability')
        
        yticks = np.array(axis.get_yticks())
        mask = ((yticks >= axis.get_xlim()[0]) &
                (yticks <= axis.get_xlim()[1]))
        axis.set_xticks(yticks[mask])
        
        axis.invert_yaxis()
        axis.xaxis.set_label_position('top')
        axis.xaxis.labelpad = 22
        axis.xaxis.tick_top()
        axis.tick_params(axis='x', which='both', pad=6)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        # Stationary distribution as function of spatial direction
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        if len(inactive) > 0:
            stationary_distribution = np.insert(mm.stationary_distribution,
                                                inactive,
                                                np.nan,
                                                axis=0)
        else:
            stationary_distribution = mm.stationary_distribution
        mdt.plot.plot(ax=axis,
                      x=bins[1:]-np.diff(bins)/2,
                      y=stationary_distribution,
                      xmin=args.MIN,
                      xmax=args.MAX,
                      ymin=0,
                      xlabel=r'${}$ / {}'.format(args.DIRECTION, args.LUNIT),
                      ylabel="Stationary distribution",
                      color='black',
                      marker='o')
        mdt.plot.vlines(ax=axis,
                        x=bins,
                        start=axis.get_ylim()[0],
                        stop=axis.get_ylim()[1],
                        xmin=args.MIN,
                        xmax=args.MAX,
                        ymin=0,
                        color='black',
                        linestyle='dotted')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        # Stationary distribution as function of spatial direction
        # divided by bin width
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        mdt.plot.plot(ax=axis,
                      x=bins[1:]-np.diff(bins)/2,
                      y=stationary_distribution/np.diff(bins),
                      xmin=args.MIN,
                      xmax=args.MAX,
                      ymin=0,
                      xlabel=r'${}$ / {}'.format(args.DIRECTION, args.LUNIT),
                      ylabel="Stat. dist. / Bin width",
                      color='black',
                      marker='o')
        mdt.plot.vlines(ax=axis,
                        x=bins,
                        start=axis.get_ylim()[0],
                        stop=axis.get_ylim()[1],
                        xmin=args.MIN,
                        xmax=args.MAX,
                        ymin=0,
                        color='black',
                        linestyle='dotted')
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
