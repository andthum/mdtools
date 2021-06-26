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
import pyemma
import mdtools as mdt




if __name__ == "__main__":
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read a pyemma.msm.MaximumLikelihoodMSM or"
                     " pyemma.msm.BayesianMSM object from file (which"
                     " must have been created by the object's save"
                     " method) and plot selected eigenvectors of the"
                     " transition matrix."
                 )
    )
    
    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=True,
        help="Input file containing the pyemma.msm.MaximumLikelihoodMSM"
             " or pyemma.msm.BayesianMSM object in HDF5 format as"
             " created by the object's save method."
    )
    parser.add_argument(
        '--bins',
        dest='BINFILE',
        type=str,
        required=False,
        default=None,
        help="File containing the bins used to generate the discretized"
             " trajectory which was used to estimate the Markov model"
             " stored as numpy.ndarray in .npy format. If provided, the"
             " bins will be shown on a secondary x-axis."
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
        '-v',
        dest='VEC_IX',
        type=int,
        nargs="+",
        required=False,
        default=[0, 1, 2, 3],
        help="Space separated list of eigenvectors to plot. Note that"
             " indexing starts at zero and is in the order of descending"
             " corresponding eigenvalues. I.e. eigenvector 0 is the"
             " stationary distribution (corresponding to the eigenvalue"
             " 1). Negative indices are counted backwards, i.e."
             " eigenvector -1 is the last eigenvector with the smallest"
             " corresponding eigenvalue. By default, the first four"
             " eigenvectors are plotted."
    )
    parser.add_argument(
        '--right',
        dest='RIGHT',
        required=False,
        default=False,
        action='store_true',
        help="Plot right eigenvectors (column vectors) instead of left"
             " eigenvectors (row vectors) of the transition matrix."
             " Since the transition matrix is row-stochastic, the left"
             " eigenvectors are real probability densities, whereas the"
             " right eigenvectors are probability densities reweighted"
             " by the stationary distribution."
    )
    
    parser.add_argument(
        '--xlabel',
        dest='XLABEL',
        type=str,
        nargs="+",
        required=False,
        default=['$z$', '/', 'A'],
        help="String to use as secondary x-axis label. Is meaningless"
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
             " secondary x-axis. Is meaningless if --bins is not given."
             " Default: 1"
    )
    parser.add_argument(
        '--every-n-ticks',
        dest='EVERY_N_TICKS',
        type=int,
        required=False,
        default=2,
        help="Set for every n ticks of the primary x-axis one tick on"
             " the secondary x-axis. Is meaningless if --bins is not"
             " given. Default: 2"
    )
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    
    
    
    print("\n\n\n", flush=True)
    print("Loading Markov model", flush=True)
    timer = datetime.now()
    
    mm = pyemma.load(args.INFILE)
    print("  Lag time in trajectory steps:                               {:>6d}".format(mm.lag), flush=True)
    print("  Lag time in real time units:                                {:>11s}".format(str(mm.dt_model)), flush=True)
    print("  Largest implied timescale in trajectory steps:              {:>11.4f}".format(mm.timescales()[0]))
    print("  2nd largest implied timescale in trajectory steps:          {:>11.4f}".format(mm.timescales()[1]))
    print("  Number of active states (reversible connected):             {:>6d}".format(mm.nstates), flush=True)
    print("  First active state:                                         {:>6d}".format(mm.active_set[0]), flush=True)
    print("  Last active state:                                          {:>6d}".format(mm.active_set[-1]), flush=True)
    print("  Total number of states in the discrete trajectories:        {:>6d}".format(mm.nstates_full), flush=True)
    print("  Fraction of states in the largest reversible connected set: {:>11.4f}".format(mm.active_state_fraction), flush=True)
    print("  Fraction of counts in the largest reversible connected set: {:>11.4f}".format(mm.active_count_fraction), flush=True)
    
    if not np.all(np.isclose(np.sum(mm.transition_matrix, axis=1), 1)):
        raise ValueError("Not all rows of the transition matrix sum up"
                         " to unity")
    if not np.isclose(np.sum(mm.stationary_distribution), 1):
        raise ValueError("The sum of the stationary distribution ({})"
                         " is not unity"
                         .format(np.sum(mm.stationary_distribution)))
    
    if mm.nstates < 2:
        print(flush=True)
        print("  Active set of states:", flush=True)
        print(mm.active_set, flush=True)
        print(flush=True)
        print("  Largest reversible connected set of states:",
              flush=True)
        print(mm.largest_connected_set, flush=True)
        print(flush=True)
        print("  Reversible connected sets of states:", flush=True)
        print('\n'.join(str(s) for s in mm.connected_sets), flush=True)
        raise ValueError("The active set of the Markov model must"
                         " contain at least 2 states")
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Computing eigenvectors", flush=True)
    timer = datetime.now()
    
    if args.RIGHT:
        eigvecs = mm.eigenvectors_right()
    else:
        eigvecs = mm.eigenvectors_left()
    
    VEC_IX = np.array(args.VEC_IX)
    pos = np.unique(VEC_IX[VEC_IX>=0])
    neg = np.unique(VEC_IX[VEC_IX<0])
    VEC_IX = np.concatenate((pos, neg))
    
    if eigvecs.ndim < 2:
        n_eigvecs = 1
        if len(VEC_IX) > 1:
            raise ValueError("The transition matrix has only one"
                             " eigenvector, but you want to plot {}"
                             " eigenvectors".format(len(VEC_IX)))
        if VEC_IX[0] != 0:
            raise ValueError("The transition matrix has only one"
                             " eigenvector, but you want to plot the"
                             " {} eigenvector. Note that indexing starts"
                             " at zero".format(VEC_IX[0]+1))
    else:
        n_eigvecs = len(eigvecs)
        if (np.max(args.VEC_IX) > 0 and
            np.max(args.VEC_IX) >= len(eigvecs)):
            raise ValueError("The highest eigenvector index you gave"
                             " ({}) is out of range of the number of"
                             " eigenvectors ({}). Note that indexing"
                             " starts at zero"
                             .format(np.max(args.VEC_IX), len(eigvecs)))
        if (np.min(args.VEC_IX) < 0 and
            abs(np.min(args.VEC_IX)) > len(eigvecs)):
            raise ValueError("The lowest eigenvector index you gave"
                             " ({}) is out of range of the number of"
                             " eigenvectors ({}). Note that backward"
                             " indexing starts with -1"
                             .format(np.max(args.VEC_IX), len(eigvecs)))
        eigvecs = eigvecs[args.VEC_IX]
    
    print("  Total number of eigenvectors: {}"
          .format(n_eigvecs), flush=True)
    if n_eigvecs != mm.nstates:
        raise ValueError("The number of eigenvectors ({}) does not match"
                         " the number of active states ({})"
                         .format(n_eigvecs, mm.nstates))
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Generating x-axis", flush=True)
    timer = datetime.now()
    
    states = np.arange(mm.active_set[-1] + mm.active_set[0] + 1)
    
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
            
            # Markov states in reals space
            bin_widths = np.insert(np.diff(bins), 0, bin_width_first)
            bins -= 0.5 * bin_widths
            
            XLABEL = ' '.join(args.XLABEL)
            XLABEL = "r'%s'" % XLABEL
            XLABEL = XLABEL[2:-1]
            
            states = states[:len(bins)]
            
        except IOError:
            print("    {} not found".format(args.BINFILE), flush=True)
            print("    Will not plot secondary x-axis", flush=True)
            args.BINFILE = None
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating plots", flush=True)
    timer = datetime.now()
    
    fig, axes = plt.subplots(nrows=len(VEC_IX),
                             squeeze=False,
                             sharex=True,
                             sharey=True,
                             figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True)
    axes = axes[:,0]
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(len(VEC_IX)):
        if VEC_IX[i] < 0:
            vec_num = n_eigvecs + VEC_IX[i] + 1
        else:
            vec_num = VEC_IX[i] + 1
        if args.RIGHT:
            ylabel = r'$\psi_{'+str(vec_num)+r'}$'
        else:
            ylabel = r'$\phi_{'+str(vec_num)+r'}$'
        if i == len(VEC_IX)-1:
            xlabel = r'State $i$'
        else:
            xlabel = None
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[i].fill_between(x=mm.active_set,
                             y1=eigvecs[i],
                             color=colors[i],
                             alpha=0.5)
        mdt.plot.plot(ax=axes[i],
                      x=mm.active_set,
                      y=eigvecs[i],
                      xmin=states[0]-0.5,
                      xmax=states[-1]+0.5,
                      xlabel=xlabel,
                      ylabel=ylabel,
                      color=colors[i])
        axes[i].set_yticks([])
        axes[i].set_xlim(xmin=states[0]-0.5, xmax=states[-1]+0.5)
    
    if args.BINFILE is not None:
        img, ax2 = mdt.plot.plot_2nd_xaxis(ax=axes[0],
                                           x=mm.active_set,
                                           y=eigvecs[0],
                                           xmin=states[0]-0.5,
                                           xmax=states[-1]+0.5,
                                           xlabel=XLABEL,
                                           alpha=0)
        xlim = axes[0].get_xlim()
        ax2.set_xlim(xlim)
        xticks = axes[0].get_xticks().astype(int)
        xticks = xticks[np.logical_and(xticks>=xlim[0], xticks<=xlim[1])]
        ax2.set_xticks(xticks[::args.EVERY_N_TICKS])
        xticklabels = np.around(bins[ax2.get_xticks()],
                                decimals=args.DECS)
        if args.DECS == 0:
            xticklabels = xticklabels.astype(int)
        if args.DECS < 0:
            xticklabels = [str(int(l))[:args.DECS] for l in xticklabels]
            ix = XLABEL.find('/')
            XLABEL = (XLABEL[:ix+1] +
                      r' $10^{'+str(abs(args.DECS))+r'}$' +
                      XLABEL[ix+1:])
            ax2.set_xlabel(xlabel=XLABEL)
        ax2.set_xticklabels(xticklabels)
    
    mdt.fh.backup(args.OUTFILE)
    plt.tight_layout(h_pad=0)
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
