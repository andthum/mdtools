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
import pyemma.msm as msm
import mdtools as mdt




if __name__ == "__main__":
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read a discretized trajectory as e.g. created by"
                     " discrete_pos.py and estimate a (Bayesian) Markov"
                     " model using pyemma.msm.estimate_markov_model() or"
                     " pyemma.msm.bayesian_markov_model()."
                 )
    )
    
    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="File containing the discretized trajectory stored as"
             " integer numpy.ndarray in .npy format. It is possible to"
             " load multiple trajectories if they are stored in .npy"
             " format as a two dimensional numpy.ndarray. Each"
             " trajectory represents a single particle."
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
        help="Output filename pattern. There will be created eigth files:"
             " <OUTFILE>_mm.h5 containing the created"
             " pyemma.msm.MaximumLikelihoodMSM or pyemma.msm.BayesianMSM"
             " object;"
             " <OUTFILE>_mm_tm.txt containing the transition matrix."
             " Note that the transition matrix is row-stochastic;"
             " <OUTFILE>_mm_eigval.txt containing the eigenvalue"
             " spectrum with corresponding implied timescales."
             " <OUTFILE>_mm_sd.txt containing the stationary"
             " distribution;"
             " <OUTFILE>_mm_tm.pdf containing a heatmap plot of the"
             " transition matrix;"
             " <OUTFILE>_eigval.pdf containing a plot of the eigenvalue"
             " spectrum with corresponding implied timescales;"
             " <OUTFILE>_mm_sd.pdf containing a plot of the stationary"
             " distribution;"
             " <OUTFILE>_mm_fe.pdf containing a plot of the free energy"
             " landscape."
             " Plots are optimized for PDF format with TeX support."
    )
    parser.add_argument(
        '--no-plots',
        dest='NOPLOTS',
        required=False,
        default=False,
        action='store_true',
        help="Do not create plots."
    )
    
    parser.add_argument(
        '--lag',
        dest='LAGTIME',
        type=int,
        default=1,
        help="Lag time (in trajectory steps) at which transitions are"
             " counted and the transition matrix is estimated. Must be"
             " smaller than the total number of frames in the"
             " trajectory. Note that the lag time should be smaller than"
             " the largest implied timescale in order for the Markov"
             " model to be useful. The timescales of interest should be"
             " converged at the chosen lag time. Default: 1"
    )
    parser.add_argument(
        '--bayes',
        dest='BAYES',
        required=False,
        default=False,
        action='store_true',
        help="Estimate a Bayesian Markov model. In contrast to a normal"
             " Markov model, a Bayesian Markov model also contains"
             " information about the uncertainties of the model. Note"
             " that this will increase the computational cost and memory"
             " consumption significantly."
    )
    parser.add_argument(
        '--dt',
        dest='DT',
        type=str,
        nargs="+",
        required=False,
        default=['1', 'step'],
        help="Real time corresponding to one trajectory step, specified"
             " by a number, whitespace and the time unit, e.g. '2 ps'."
             " Permitted units are 'fs', 'ps', 'ns', 'us', 'ms', 's' and"
             " 'step'. Useful to give the Markov model a real time"
             " information, but actually not really necessary."
             " Default: '1 step'"
    )
    parser.add_argument(
        '-T',
        dest='TEMP',
        type=float,
        required=False,
        default=273,
        help="Average temperature (in K) during the simulation. Only"
             " used to calculate the free energy for plotting from the"
             " stationary distribution of the Markov model via"
             " F=-R*T*ln(stat). Is meaningless if --no-plots is set."
             " Default: 273"
    )
    
    parser.add_argument(
        '--xlabel',
        dest='XLABEL',
        type=str,
        nargs="+",
        required=False,
        default=['$z$', '/', 'A'],
        help="String to use as secondary x-axis label. Is meaningless"
             " if --bins is not given or --no-plots is set. Note that"
             " you have to use TeX syntax. Default: '$z$ / A' (Note that"
             " you must either leave a space after dollar signs or"
             " enclose the expression in single quotes to avoid bash's"
             " variable expansion)."
    )
    parser.add_argument(
        '--decs',
        dest='DECS',
        type=int,
        required=False,
        default=1,
        help="Number of decimal places for the tick labels of the"
             " secondary x-axis. Is meaningless if --bins is not given"
             " or --no-plots is set. Default: 1"
    )
    parser.add_argument(
        '--every-n-ticks',
        dest='EVERY_N_TICKS',
        type=int,
        required=False,
        default=2,
        help="Set for every n ticks of the primary x-axis one tick on"
             " the secondary x-axis. Is meaningless if --bins is not"
             " given or --no-plots is set. Default: 2"
    )
    
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    
    
    
    print("\n\n\n", flush=True)
    print("Loading discrete trajectories", flush=True)
    timer = datetime.now()
    
    dtrajs = np.load(args.TRJFILE)
    n_frames = dtrajs.shape[-1]
    if n_frames < 2:
        raise ValueError("Trajectories must contain at least two frames"
                         " in order to estimate a Markov model")
    
    if dtrajs.ndim == 1:
        n_trajs = 1
    elif dtrajs.ndim == 2:
        # PyEMMA takes multiple trajectories only as list of
        # numpy.ndarrays, not as 2-dimensional numpy.ndarray
        n_trajs = dtrajs.shape[0]
        dtrajs = [dtraj for dtraj in dtrajs]
        if len(dtrajs) == 1:
            dtrajs = dtrajs[0]
        if len(dtrajs) != n_trajs:
            raise RuntimeError("Unexpected error: len(dtrajs) != n_trajs")
    else:
        raise ValueError("dtrajs has more than two dimensions ({})"
                         .format(dtrajs.ndim))
    
    traj_info = ("  Number of single particle trajectories: {:>9d}\n"
                 "  Number of frames per trajectory:        {:>9d}\n"
                 "  First populated state:                  {:>9d}\n"
                 "  Last populated state:                   {:>9d}"
                 .format(n_trajs,
                         n_frames,
                         np.min(dtrajs),
                         np.max(dtrajs)))
    print("{}".format(traj_info), flush=True)
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating Markov model", flush=True)
    timer = datetime.now()
    
    if args.LAGTIME >= n_frames:
        args.LAGTIME = n_frames - 1
        print("  Set lag time to {}".format(args.LAGTIME))
    if args.LAGTIME > int(n_frames/2):
        print("  WARNING: The lag time ({}) is greater than half the"
              " total number of frames in the trajectories ({})"
              .format(args.LAGTIME, int(n_frames/2)), flush=True)
    
    if args.BAYES:
        print(flush=True)
        mm = msm.bayesian_markov_model(dtrajs=dtrajs,
                                       lag=args.LAGTIME,
                                       count_mode='sample',
                                       nsamples=8,
                                       dt_traj=' '.join(args.DT),
                                       show_progress=True)
        print(flush=True)
    else:
        mm = msm.estimate_markov_model(dtrajs=dtrajs,
                                       lag=args.LAGTIME,
                                       count_mode='sliding',
                                       dt_traj=' '.join(args.DT))
    
    del dtrajs
    
    mm_info = (
        "  Lag time in trajectory steps:                               {:>6d}\n"
        "  Lag time in real time units:                                {:>11s}\n"
        "  Largest implied timescale in trajectory steps:              {:>11.4f}\n"
        "  2nd largest implied timescale in trajectory steps:          {:>11.4f}\n"
        "  Number of active states (reversible connected):             {:>6d}\n"
        "  First active state:                                         {:>6d}\n"
        "  Last active state:                                          {:>6d}\n"
        "  Total number of states in the discrete trajectories:        {:>6d}\n"
        "  Fraction of states in the largest reversible connected set: {:>11.4f}\n"
        "  Fraction of counts in the largest reversible connected set: {:>11.4f}"
        .format(mm.lag,
                str(mm.dt_model),
                mm.timescales()[0],
                mm.timescales()[1],
                mm.nstates,
                mm.active_set[0],
                mm.active_set[-1],
                mm.nstates_full,
                mm.active_state_fraction,
                mm.active_count_fraction))
    print("{}".format(mm_info), flush=True)
    #print(flush=True)
    #print("  Active set of states:", flush=True)
    #print(mm.active_set, flush=True)
    #print(flush=True)
    #print("  Largest reversible connected set of states:", flush=True)
    #print(mm.largest_connected_set, flush=True)
    #print(flush=True)
    #print("  Reversible connected sets of states:", flush=True)
    #print('\n'.join(str(s) for s in mm.connected_sets), flush=True)
    #print(flush=True)
    #print("  Count matrix on the active set of states:", flush=True)
    #print(mm.count_matrix_active, flush=True)
    #print(flush=True)
    #print("  Count matrix on the full set of states:", flush=True)
    #print(mm.count_matrix_full, flush=True)
    #print(flush=True)
    #print("  Statistically uncorrelated transition counts within the active set of states:", flush=True)
    #print(mm.effective_count_matrix, flush=True)
    #print(flush=True)
    #print("  Transition matrix on the active set:", flush=True)
    #print(mm.transition_matrix, flush=True)
    #print(flush=True)
    #print("  Stationary_distribution:", flush=True)
    #print(mm.stationary_distribution, flush=True)
    
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
    
    eigvals = mm.eigenvalues()
    if len(eigvals) != mm.nstates:
        raise ValueError("The number of eigenvalues ({}) does not match"
                         " the number of active states ({})"
                         .format(len(eigvals), mm.nstates))
    
    mask = np.logical_and(eigvals>0, eigvals<1)
    timescales = -mm.lag / np.log(eigvals, where=mask)
    timescales[np.logical_not(mask)] = np.nan
    timescales[np.isclose(eigvals, 1)] = np.inf
    
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
            dim = XLABEL[:XLABEL.find('/')].strip()
            
            states = states[:len(bins)]
            
        except IOError:
            print("    {} not found".format(args.BINFILE), flush=True)
            print("    Will not plot secondary x-axis", flush=True)
            args.BINFILE = None
    
    if args.BINFILE is None:
        dim = "$i$"
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()
    
    
    # pyemma.msm.MaximumLikelihoodMSM or pyemma.msm.BayesianMSM object
    mdt.fh.backup(args.OUTFILE+"_mm.h5")
    mm.save(file_name=args.OUTFILE+"_mm.h5")
    print("  Created "+args.OUTFILE+"_mm.h5", flush=True)
    
    
    # Transition matrix
    mdt.fh.savetxt_matrix(
        fname=args.OUTFILE+"_mm_tm.txt",
        data=mm.transition_matrix,
        var1=mm.active_set,
        var2=mm.active_set,
        header=("Markov state model:\n"
                + mm_info + "\n\n"
                "Discrete trajectories used for construction:\n"
                + traj_info + "\n\n"
                "Transition matrix (row-stochastic)\n"
                "First column: State i\n"
                "First row:    State j\n"))
    print("  Created "+args.OUTFILE+"_mm_tm.txt", flush=True)
    
    
    # Eigenvalue spectrum
    mdt.fh.savetxt(
        fname=args.OUTFILE+"_mm_eigval.txt",
        data=np.column_stack((np.arange(1, len(eigvals)+1),
                              eigvals,
                              timescales)),
        header=("Markov state model:\n"
                + mm_info + "\n\n"
                "Discrete trajectories used for construction:\n"
                + traj_info + "\n\n"
                "The columns contain:\n"
                "  1 Index i\n"
                "  2 Eigenvalue lambda_i\n"
                "  3 Corresponding timescale t_i = -tau/ln(lambda_i)\n\n"
                "Column number:\n"
                  "{:>14d} {:>16d} {:>16d}".format(1, 2, 3)))
    print("  Created "+args.OUTFILE+"_mm_eigval.txt", flush=True)
    
    
    # Stationary distribution
    if args.BINFILE is not None:
        sd = np.column_stack((mm.active_set,
                              bins[mm.active_set],
                              mm.stationary_distribution))
        xlabel_txt = XLABEL.replace("$", "")
        header = ("  2 "+xlabel_txt+"\n"
                  "  3 Stationary distribution\n\n"
                  "Column number:\n"
                  "{:>14d} {:>16d} {:>16d}".format(1, 2, 3))
    elif args.BINFILE is None:
        sd = np.column_stack((mm.active_set,mm.stationary_distribution))
        header = ("  2 Stationary distribution\n\n"
                  "Column number:\n"
                  "{:>14d} {:>16d}".format(1, 2))
    mdt.fh.savetxt(
        fname=args.OUTFILE+"_mm_sd.txt",
        data=sd,
        header=("Markov state model:\n"
                + mm_info + "\n\n"
                "Discrete trajectories used for construction:\n"
                + traj_info + "\n\n"
                "The columns contain:\n"
                "  1 State i\n"
                + header))
    print("  Created "+args.OUTFILE+"_mm_sd.txt", flush=True)
    del sd
    
    
    if not args.NOPLOTS:
        
        # Transition matrix
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        state_widths = np.insert(np.diff(mm.active_set),
                                 0,
                                 mm.active_set[1]-mm.active_set[0])
        xy = mm.active_set - 0.5*state_widths
        xy = np.append(xy, mm.active_set[-1]+0.5*state_widths[-1])
        
        mdt.plot.pcolormesh(ax=axis,
                            x=xy,
                            y=xy,
                            z=mm.transition_matrix,
                            xmin=states[0]-0.5,
                            xmax=states[-1]+0.5,
                            ymin=states[0]-0.5,
                            ymax=states[-1]+0.5,
                            xlabel=r'State $j$',
                            ylabel=r'State $i$',
                            cbarlabel=r'Transition probability $T_{ij}$')
        
        axis.invert_yaxis()
        axis.xaxis.set_label_position('top')
        axis.xaxis.labelpad = 22
        axis.xaxis.tick_top()
        axis.tick_params(axis='x', which='both', pad=6)
        
        mdt.fh.backup(args.OUTFILE+"_mm_tm.pdf")
        plt.tight_layout()
        plt.savefig(args.OUTFILE+"_mm_tm.pdf")
        plt.close(fig)
        print("  Created "+args.OUTFILE+"_mm_tm.pdf", flush=True)
        del state_widths, xy
        
        
        # Eigenvalue spectrum
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        axis.axhline(color='black')
        axis.axhline(y=1, color='black', linestyle='--')
        mdt.plot.plot(ax=axis,
                      x=np.arange(1, len(eigvals)+1),
                      y=eigvals,
                      linestyle='--',
                      marker='o',
                      xlabel=r'Index $i$',
                      ylabel=r'Eigenvalue $\lambda_i$')
        
        img, ax2 = mdt.plot.plot_2nd_yaxis(
                       ax=axis,
                       x=np.arange(1, len(eigvals)+1),
                       y=eigvals,
                       ylabel=r'Implied timescale $t_i$ / steps',
                       alpha=0)
        ax2.axhline(color='black', alpha=0)
        if np.min(eigvals) >= 0:
            axis.set_ylim(ymin=0)
        ax2.set_ylim(ymin=axis.get_ylim()[0], ymax=axis.get_ylim()[1])
        mask = np.logical_and(axis.get_yticks()>0, axis.get_yticks()<1)
        labels = -mm.lag / np.log(axis.get_yticks(), where=mask)
        labels = np.around(labels).astype(int).astype(str)
        labels[np.logical_not(mask)] = ""
        labels[axis.get_yticks()==0] = 0
        labels[axis.get_yticks()==1] = r'$\infty$'
        ax2.set_yticklabels(labels)
        
        mdt.fh.backup(args.OUTFILE+"_mm_eigval.pdf")
        plt.tight_layout()
        plt.savefig(args.OUTFILE+"_mm_eigval.pdf")
        plt.close(fig)
        print("  Created "+args.OUTFILE+"_mm_eigval.pdf", flush=True)
        
        
        # Stationary distribution
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        mdt.plot.plot(
            ax=axis,
            x=mm.active_set,
            y=mm.stationary_distribution*100,  # *100 to convert to %
            xmin=states[0]-0.5,
            xmax=states[-1]+0.5,
            ymin=0,
            xlabel=r'State $i$',
            ylabel=r'Stationary distribution $\mathbf{\pi}$ / \%')
        
        if args.BINFILE is not None:
            img, ax2 = mdt.plot.plot_2nd_xaxis(
                           ax=axis,
                           x=mm.active_set,
                           y=mm.stationary_distribution*100,
                           xmin=states[0]-0.5,
                           xmax=states[-1]+0.5,
                           xlabel=XLABEL,
                           alpha=0)
            xlim = axis.get_xlim()
            xticks = axis.get_xticks().astype(int)
            xticks = xticks[np.logical_and(xticks>=xlim[0],
                                           xticks<=xlim[1])]
            ax2.get_xaxis().set_ticks(xticks[::args.EVERY_N_TICKS])
            xticklabels = np.around(bins[ax2.get_xticks()],
                                    decimals=args.DECS)
            if args.DECS == 0:
                xticklabels = xticklabels.astype(int)
            if args.DECS < 0:
                xticklabels = [str(int(l))[:args.DECS]
                               for l in xticklabels]
                ix = XLABEL.find('/')
                XLABEL = (XLABEL[:ix+1] +
                          r' $10^{'+str(abs(args.DECS))+r'}$' +
                          XLABEL[ix+1:])
                ax2.set_xlabel(xlabel=XLABEL)
            ax2.set_xticklabels(xticklabels)
            axis.set_ylim(ymin=0)
        
        mdt.fh.backup(args.OUTFILE+"_mm_sd.pdf")
        plt.tight_layout()
        plt.savefig(args.OUTFILE+"_mm_sd.pdf")
        plt.close(fig)
        print("  Created "+args.OUTFILE+"_mm_sd.pdf", flush=True)
        
        
        # Free energy landscape
        # NA = 6.02214076e23    # mol^-1
        # kb = 1.380649e-23     # J K^-1
        R = 8.31446261815324    # J K^-1 mol^-1, R=NA*kb
        free_energy = -R * args.TEMP * np.log(mm.stationary_distribution)
        
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        mdt.plot.plot(ax=axis,
                      x=mm.active_set,
                      y=free_energy/1000,  # /1000 to convert J ot kJ
                      xmin=states[0]-0.5,
                      xmax=states[-1]+0.5,
                      xlabel=r'State $i$',
                      ylabel=r'$F($'+dim+r'$)$ / kJ mol$^{-1}$')
        
        if args.BINFILE is not None:
            img, ax2 = mdt.plot.plot_2nd_xaxis(ax=axis,
                                               x=mm.active_set,
                                               y=free_energy/1000,
                                               xmin=states[0]-0.5,
                                               xmax=states[-1]+0.5,
                                               xlabel=XLABEL,
                                               alpha=0)
            xlim = axis.get_xlim()
            xticks = axis.get_xticks().astype(int)
            xticks = xticks[np.logical_and(xticks>=xlim[0],
                                           xticks<=xlim[1])]
            ax2.get_xaxis().set_ticks(xticks[::args.EVERY_N_TICKS])
            xticklabels = np.around(bins[ax2.get_xticks()],
                                    decimals=args.DECS)
            if args.DECS == 0:
                xticklabels = xticklabels.astype(int)
            if args.DECS < 0:
                xticklabels = [str(int(l))[:args.DECS]
                               for l in xticklabels]
                ix = XLABEL.find('/')
                XLABEL = (XLABEL[:ix+1] +
                          r' $10^{'+str(abs(args.DECS))+r'}$' +
                          XLABEL[ix+1:])
                ax2.set_xlabel(xlabel=XLABEL)
            ax2.set_xticklabels(xticklabels)
        
        mdt.fh.backup(args.OUTFILE+"_mm_fe.pdf")
        plt.tight_layout()
        plt.savefig(args.OUTFILE+"_mm_fe.pdf")
        plt.close(fig)
        print("  Created "+args.OUTFILE+"_mm_fe.pdf", flush=True)
    
    
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
