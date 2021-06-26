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
                     " method) and plot transition probabilities for"
                     " selected jump distances as function of the Markov"
                     " states."
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
        '-d',
        dest="JUMP_IX",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help="Space separated list of jump distances counted in"
             " overleaped Markov states for which to plot the jump"
             " rates. By default, the jump rates are plotted for 7"
             " different jump distances ranging from the largest"
             " backward jump with a non-zero probability to the largest"
             " forward jump with a non-zero probability (and including"
             " zero)."
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
    
    parser.add_argument(
        '--split-legend',
        dest='SPLIT_LEGEND',
        required=False,
        default=False,
        action='store_true',
        help="Split legend in positive and negative jump distances"
             " (Recommended for better clarity of the plot if both,"
             " positive and negative, jump distances are plotted)."
    )
    parser.add_argument(
        '--ymin',
        dest='YMIN',
        type=float,
        required=False,
        default=None,
        help="Minimum y-range of plot. By default detected automatically."
    )
    parser.add_argument(
        '--ymax',
        dest='YMAX',
        type=float,
        required=False,
        default=None,
        help="Maximum y-range of plot. By default detected automatically."
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
    
    
    
    
    # Highest theoretically possible forward and backward jump widths in
    # overleaped Markov states
    if args.PBC:
        jump_ix_max = int(mm.nstates/2)
        if mm.nstates % 2 != 0:
            jump_ix_min = -int(mm.nstates/2)
        elif mm.nstates % 2 == 0:
            jump_ix_min = -int(mm.nstates/2) + 1
    else:
        jump_ix_max = mm.nstates - 1
        jump_ix_min = -mm.nstates
    
    
    
    
    # Transition matrices containing only the theoretically possible
    # forward and backward jumps
    tm = mdt.nph.tilt_diagonals(mm.transition_matrix, diagpos=0)
    if args.PBC:
        tm_forward = tm[:,:jump_ix_max+1]
        tm_backward = tm[:,jump_ix_max+1:]
        if not np.all(np.isclose(np.hstack((tm_forward, tm_backward)),
                                 tm)):
            raise RuntimeError("Error while splitting the transition"
                               " matrix in forward and backward jumps")
    else:
        tm_forward = np.triu(tm[:,::-1])[:,::-1]
        tm_backward = np.tril(tm[:,::-1], k=-1)[:,::-1]
        if not np.all(np.isclose(tm_forward+tm_backward, tm)):
            raise RuntimeError("Error while splitting the transition"
                               " matrix in forward and backward jumps")
    del tm
    
    
    
    
    # Maximum forward and backward jump widths with non-zero jump
    # probability in overleaped Markov states
    max_forward = np.max(np.argmax(np.isclose(tm_forward,
                                               0,
                                               atol=1e-4),
                                    axis=1))
    max_forward -= 1
    max_backward = np.max(np.argmax(np.isclose(tm_backward[:,::-1],
                                                0,
                                                atol=1e-4),
                                     axis=1))
    
    if max_forward == 0:
        max_forward = 1
    if max_backward == 0:
        max_backward = 1
    
    if max_forward > jump_ix_max:
        max_forward = jump_ix_max
    if max_backward > abs(jump_ix_min):
        max_backward = abs(jump_ix_min)
    
    
    
    
    # Select jump distances for which to plot the jump probabilities.
    # JUMP_IX is just the number of overleaped states. The actual jump
    # distance in multiples of Markov states is JUMP_IX * jump_width.
    # with jump_width = mm.active_set[1] - mm.active_set[0]
    if args.JUMP_IX is None:
        JUMP_IX = np.unique(np.linspace(0, max_forward, 4, dtype=int))
        JUMP_IX = JUMP_IX[JUMP_IX>0]
        JUMP_IX = np.append(JUMP_IX,
                            -np.unique(np.linspace(0,
                                                   max_backward,
                                                   4,
                                                   dtype=int)))
    else:
        JUMP_IX = np.array(args.JUMP_IX)
        pos = np.unique(JUMP_IX[JUMP_IX>0])
        neg = np.unique(JUMP_IX[JUMP_IX<=0])[::-1]
        JUMP_IX = np.concatenate((pos, neg))
    
    if args.PBC:
        if (np.max(JUMP_IX) > jump_ix_max or
            np.min(JUMP_IX) < jump_ix_min):
            print("\n\n\n", flush=True)
            print("You selected --pbc, but the jump indices you gave"
                  " exceed half the maximum number of Markov states the"
                  " in active set ({}). Note that the jump distances"
                  " given with -d are counted in overleaped Markov"
                  " states not in multiples of Markov states."
                  .format(int(mm.nstates/2)), flush=True)
            JUMP_IX = JUMP_IX[JUMP_IX<=jump_ix_max]
            JUMP_IX = JUMP_IX[JUMP_IX>=jump_ix_min]
            print("Set JUMP_IX to: {}".format(JUMP_IX), flush=True)
    else:
        if (np.max(JUMP_IX) >= mm.nstates or
            np.min(JUMP_IX) < -mm.nstates):
            print("\n\n\n", flush=True)
            print("The jump indices you gave exceed the maximum number"
                  " of Markov states the in active set ({}). Note that"
                  " the jump distances given with -d are counted in"
                  " overleaped Markov states not in multiples of Markov"
                  " states.".format(mm.nstates), flush=True)
            JUMP_IX = JUMP_IX[JUMP_IX<mm.nstates]
            JUMP_IX = JUMP_IX[JUMP_IX>=-mm.nstates]
            print("Set JUMP_IX to: {}".format(JUMP_IX), flush=True)
    if JUMP_IX.size == 0:
        raise ValueError("JUMP_IX is empty. No jump distances selected")
    
    
    
    
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
    
    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    img = [None,] * len(JUMP_IX)
    jump_prev = JUMP_IX[0]
    for i, jump in enumerate(JUMP_IX):
        if np.sign(jump) != np.sign(jump_prev):
            axis.set_prop_cycle(None)
        if jump == 0:
            axis.set_prop_cycle(color='black')
        jump_prev = jump
        img[i], = mdt.plot.plot(
                      ax=axis,
                      x=mm.active_set,
                      y=tm_forward[:,jump] if jump >= 0 else -tm_backward[:,jump],
                      xmin=states[0]-0.5,
                      xmax=states[-1]+0.5,
                      label=r'${:d}$'.format(jump),
                      xlabel=r'State $i$',
                      ylabel=r'Transition probability $T_{i,i+x}$')
    axis.axhline(color='black')
    axis.set_xlim(xmin=states[0]-0.5, xmax=states[-1]+0.5)
    
    fontsize_legend = 24
    if args.SPLIT_LEGEND:
        legend_pos = plt.legend(
                         handles=img[:np.argmax(JUMP_IX<0)],
                         frameon=False,
                         loc='upper center',
                         ncol=int(np.ceil(len(img[:np.argmax(JUMP_IX<0)])/2)),
                         title=r'$x$',
                         fontsize=fontsize_legend)
        plt.setp(legend_pos.get_title(),fontsize=fontsize_legend)
        axis.add_artist(legend_pos)
        legend_neg = plt.legend(
                         handles=img[np.argmax(JUMP_IX<0):],
                         frameon=False,
                         loc='lower center',
                         ncol=int(np.ceil(len(img[np.argmax(JUMP_IX<0):])/2)),
                         title=r'$x$',
                         fontsize=fontsize_legend)
        plt.setp(legend_neg.get_title(),fontsize=fontsize_legend)
    else:
        legend = axis.legend(frameon=False,
                             loc='best',
                             ncol=int(np.ceil(len(JUMP_IX)/4)),
                             title=r'$x$',
                             fontsize=fontsize_legend)
        plt.setp(legend.get_title(),fontsize=fontsize_legend)
    
    if args.BINFILE is not None:
        img, ax2 = mdt.plot.plot_2nd_xaxis(
                       ax=axis,
                       x=mm.active_set,
                       y=tm_forward[:,JUMP_IX[0]] if JUMP_IX[0] >= 0 else -tm_backward[:,JUMP_IX[0]],
                       xmin=states[0]-0.5,
                       xmax=states[-1]+0.5,
                       xlabel=XLABEL,
                       alpha=0)
        xlim = axis.get_xlim()
        ax2.set_xlim(xlim)
        xticks = axis.get_xticks().astype(int)
        xticks = xticks[np.logical_and(xticks>=xlim[0], xticks<=xlim[1])]
        ax2.get_xaxis().set_ticks(xticks[::args.EVERY_N_TICKS])
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
    
    if np.min(JUMP_IX) >= 0:
        axis.set_ylim(ymin=0, ymax=args.YMAX, auto=True)
    elif np.max(JUMP_IX) < 0:
        axis.set_ylim(ymin=args.YMIN, ymax=0, auto=True)
    else:
        axis.set_ylim(ymin=args.YMIN, ymax=args.YMAX, auto=True)
    
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
