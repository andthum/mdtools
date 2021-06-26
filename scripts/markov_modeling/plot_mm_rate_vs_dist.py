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
import matplotlib.ticker as ticker
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
                     " method) and plot transition probabilities for"
                     " selected Markov states as function of the jump"
                     " distance."
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
             " bins will be shown on a secondary x-axis. Works only for"
             " equidistant bins."
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
        "-s",
        dest="STATE_IX",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help="Space separated list of indices of the Markov states for"
             " which to plot the jump probabilities. Indexing starts at"
             " zero. Negative indices are counted backwards, e.g. -1 is"
             " the last state in the active set. By default, the jump"
             " rates are plotted for 5 equally spaced different states,"
             " starting with the first and ending with the last state of"
             " the active set."
    )
    parser.add_argument(
        '--pbc',
        dest='PBC',
        required=False,
        default=False,
        action='store_true',
        help="Assume periodic boundary conditions. I.e. jumps over more"
             " than half the Markov states are actually jumps in the"
             " opposite direction over n_tot - n_jump states (if"
             " n_jump > n_tot/2)."
    )
    
    parser.add_argument(
        '-l',
        dest='XLABEL',
        type=str,
        nargs="+",
        required=False,
        default=['$\Delta', 'z$', '/', 'A'],
        help="String to use as secondary x-axis label. Is meaningless"
             " if --bins is not given. Note that you have to use TeX"
             " syntax. Default: '$\Delta z$ / A' (Note that you must"
             " either leave a space after dollar signs or enclose the"
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
    print("  Lag time in real time units:                                   {}".format(mm.dt_model), flush=True)
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
    
    if not np.all(np.diff(mm.active_set) == mm.active_set[1]-mm.active_set[0]):
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
        raise ValueError("This script works only for Markov models where"
                         " all states in the active set are equally"
                         " spaced")
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    if args.STATE_IX is None:
        STATE_IX = np.unique(np.linspace(0, mm.nstates-1, 5, dtype=int))
    else:
        STATE_IX = np.array(args.STATE_IX)
        pos = np.unique(STATE_IX[STATE_IX>=0])
        neg = np.unique(STATE_IX[STATE_IX<0])
        STATE_IX = np.concatenate((pos, neg))
    
    if (np.max(STATE_IX) >= mm.nstates or
        np.min(STATE_IX) < -mm.nstates):
        print("\n\n\n", flush=True)
        print("The state indices you gave exceed the maximum number of"
              " Markov states in the active set ({}). Note that indexing"
              " starts at zero even if the first state in the active set"
              " is not zero".format(mm.nstates), flush=True)
        STATE_IX = STATE_IX[STATE_IX<mm.nstates]
        STATE_IX = STATE_IX[STATE_IX>=-mm.nstates]
        print("Set STATE_IX to: {}".format(STATE_IX), flush=True)
    if STATE_IX.size == 0:
        raise ValueError("STATE_IX is empty. No states selected")
    
    
    
    
    print("\n\n\n", flush=True)
    print("Generating x-axis", flush=True)
    timer = datetime.now()
    
    
    # Maximum forward and backward jump distances with non-zero jump
    # probability in overleaped Markov states
    tm = mdt.nph.tilt_diagonals(mm.transition_matrix, diagpos=0)
    max_forward = np.max(np.argmax(np.isclose(tm[STATE_IX],
                                               0,
                                               atol=1e-4),
                                    axis=1))
    max_backward = np.max(np.argmax(np.isclose(tm[STATE_IX][:,-1:0:-1],
                                                0,
                                                atol=1e-4),
                                     axis=1))
    max_backward += 1
    
    if max_forward == 0:
        max_forward = 1
    if max_backward == 0:
        max_backward = 1
    
    # Symmetrize x-axis if max_forward and max_backward are close
    diff = abs(max_forward - max_backward)
    if np.isclose(diff, 0, atol=0.075*mm.nstates):
        max_forward = max(max_forward, max_backward)
        max_backward = max_forward
    
    if args.PBC:
        if max_forward > int(mm.nstates/2):
            max_forward = int(mm.nstates/2)
        if max_backward > int(mm.nstates/2) and mm.nstates % 2 != 0:
            max_backward = int(mm.nstates/2)
        elif max_backward >= int(mm.nstates/2) and mm.nstates % 2 == 0:
            max_backward = int(mm.nstates/2) - 1
    
    # All jump distances with non-zero jump probability in multiples of
    # Markov states
    jump_width = mm.active_set[1] - mm.active_set[0]
    jump_dists = np.arange(0, jump_width*max_forward+1, jump_width)
    jump_dists = np.concatenate((jump_dists,
                                 -np.arange(jump_width,
                                            jump_width*max_backward+1,
                                            jump_width)[::-1]))
    
    
    if args.BINFILE is not None:
        print("  Loading bins from {}".format(args.BINFILE), flush=True)
        
        try:
            bins = np.load(args.BINFILE).astype(float)
            bin_width = bins[1] - bins[0]
            
            print("    Start:          {:>12.6f}"
                  .format(bins[0] - bin_width),
                  flush=True)
            print("    Stop:           {:>12.6f}"
                  .format(bins[-1]),
                  flush=True)
            print("    Bin width:      {:>12.6f}"
                  .format(bin_width),
                  flush=True)
            print("    Number of bins: {:>5d}"
                  .format(len(bins)),
                  flush=True)
            
            if not np.all(np.isclose(np.diff(bins), bin_width)):
                print("    Bins:", flush=True)
                print(bins, flush=True)
                print("    Bin widths:", flush=True)
                print(np.diff(bins), flush=True)
                raise ValueError("This script works only for equidistant"
                                 " bins. Consider running this script"
                                 " without --bins")
            
            # Jump distances in real space
            bin_dists = np.cumsum(np.diff(bins))
            bin_dists = np.concatenate((bin_dists, -bin_dists[::-1]))
            bin_dists = np.insert(bin_dists, 0, 0)
            
            XLABEL = ' '.join(args.XLABEL)
            XLABEL = "r'%s'" % XLABEL
            XLABEL = XLABEL[2:-1]
            
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
    
    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axis.axvline(color='black', linestyle='--')
    
    for state in STATE_IX:
        if args.PBC:
            jump_ix = np.arange(-max_backward, max_forward+1)
        else:
            if state < 0:
                state = mm.nstates + state
            jump_ix = np.arange(mm.nstates) - state
            jump_ix = jump_ix[np.logical_and(jump_ix>=-max_backward,
                                             jump_ix<=max_forward)]
        mdt.plot.plot(ax=axis,
                      x=jump_dists[jump_ix],
                      y=tm[state][jump_ix],
                      xmin=jump_dists[-max_backward]-0.5*jump_width,
                      xmax=jump_dists[max_forward]+0.5*jump_width,
                      ymin=0,
                      label=r'${:d}$'.format(mm.active_set[state]),
                      xlabel=r'Jump distance $x$ / states',
                      ylabel=r'Transition probability $T_{i,i+x}$')
    
    fontsize_legend = 28
    legend = axis.legend(frameon=False,
                         loc='best',
                         ncol=int(np.ceil(len(STATE_IX)/7)),
                         title=r'State $i$',
                         fontsize=fontsize_legend)
    plt.setp(legend.get_title(), fontsize=fontsize_legend)
    
    if args.BINFILE is not None:
        img, ax2 = mdt.plot.plot_2nd_xaxis(
                       ax=axis,
                       x=jump_dists[jump_ix],
                       y=tm[state][jump_ix],
                       xmin=jump_dists[-max_backward]-0.5*jump_width,
                       xmax=jump_dists[max_forward]+0.5*jump_width,
                       xlabel=XLABEL,
                       alpha=0)
        xlim = axis.get_xlim()
        xticks = axis.get_xticks().astype(int)
        xticks = xticks[np.logical_and(xticks>=xlim[0], xticks<=xlim[1])]
        ax2.get_xaxis().set_ticks(xticks[xticks[0]%2::args.EVERY_N_TICKS])
        xticklabels = np.around(bin_dists[ax2.get_xticks()],
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
        axis.set_ylim(ymin=0, ymax=None, auto=True)
    
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
