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
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import pyemma.msm as msm
import mdtools as mdt




if __name__ == "__main__":
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read a discretized trajectory as e.g. created by"
                     " discrete_pos.py, estimate Markov models at"
                     " different lag times and conduct a Chapman-"
                     "Kolmogorow test."
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
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename pattern. There will be created sixteen"
             " files:"
             " <OUTFILE>_cktest_tm_diff.txt containing a matrix"
             " representation of the total absolute differences of all"
             " transition matrices normalized by the number of states;"
             " <OUTFILE>_tm_diff.pdf containing a heatmap plot of the"
             " total absolute differences of all transition matrices"
             " normalized by the number of states;"
             " <OUTFILE>_cktest_eigval_comp.pdf comparing the eigenvalue"
             " spectra of the transition matrices;"
             " <OUTFILE>_cktest_eigval_self.pdf comparing the eigenvalue"
             " spectra of a transition matrix with exponentiated"
             " versions of itself;"
             " <OUTFILE>_cktest_eigvec_comp<1-4>.pdf comparing the first"
             " to fourth eigenvectors of the transition matrices;"
             " <OUTFILE>_cktest_eigvec_self<1-4>.pdf comparing the first"
             " to fourth eigenvectors of a transition matrix with"
             " exponentiated version of itself;"
             " <OUTFILE>_cktest_dist<1-4>.pdf comparing the final"
             " distributions when propagating four different initial"
             " distributions with the difference transition matrices."
             " Plots are optimized for PDF format with TeX support."
    )
    
    parser.add_argument(
        '--fist-lag',
        dest='START',
        type=float,
        required=False,
        default=1,
        help="First lag time (in trajectory steps) to use. Default: 1"
    )
    parser.add_argument(
        "--last-lag",
        dest="STOP",
        type=float,
        required=False,
        default=None,
        help="Last lag time (in trajectory steps) to use. Must be"
             " greater than --first-lag. Note that for a meaningful"
             " Chapman-Kolmogorow test the lag times should not exceed"
             " the highest implied timescale. Default: None, which means"
             " use half the maximum number of steps in the trajectory"
             " (Might lead to a meaningless Chapman-Kolmogorow test!)."
    )
    parser.add_argument(
        '--num-lags',
        dest='NUM',
        type=np.uint16,
        required=False,
        default=8,
        help="Number of lag times to use. Lag times will be spaced"
             " linearly from --first-lag to --last-lag. Must be at least"
             " 2. The lag times should be a multiple of each other. Thus,"
             " it is probably better, to specify the lag times yourself"
             " with --lags. Default: 8"
    )
    parser.add_argument(
        "--lags",
        dest="LAGS",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help="Space separated list of lag times to use. If supplied,"
             " this takes precedence over --fist-lag, --last-lag and"
             " --num-lags. The lag times should be a multiple of each"
             " other and should not exceed the highest implied"
             " timescale."
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
    
    if dtrajs.ndim > 2:
        raise ValueError("dtrajs has more than two dimensions ({})"
                         .format(dtrajs.ndim))
    elif dtrajs.ndim == 2:
        # PyEMMA takes multiple trajectories only as list of
        # numpy.ndarrays, not as 2-dimensional numpy.ndarray
        n_trajs = dtrajs.shape[0]
        dtrajs = [signle_part_traj for signle_part_traj in dtrajs]
        if len(dtrajs) == 1:
            dtrajs = dtrajs[0]
    else:
        n_trajs = 1
    
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
    
    
    
    
    if args.LAGS is None:
        if args.STOP is None:
            STOP = int(n_frames/2)
        else:
            STOP = args.STOP
        if args.NUM < 2:
            raise ValueError("--num-lags ({}) must be greater than one"
                             .format(args.NUM))
        START, STOP, STEP, NUM = mdt.check.bins(start=args.START,
                                                stop=STOP,
                                                num=args.NUM,
                                                amin=1,
                                                amax=n_frames-1)
        lags = np.linspace(START, STOP, NUM, dtype=int)
    else:
        lags = np.sort(args.LAGS)
        START, STOP, STEP, NUM = mdt.check.bins(start=lags[0],
                                                stop=lags[-1],
                                                num=len(lags),
                                                amin=1,
                                                amax=n_frames-1)
        if lags[0] != START:
            raise ValueError("The first lag time ({}) must be at least 1"
                             .format(lags[0]))
        elif lags[-1] != STOP:
            raise ValueError("The last lag time ({}) must be less than"
                             " the total number of frames in the"
                             " trajectory ({})"
                             .format(lags[-1], n_frames))
        elif len(lags) != NUM:
            raise ValueError("Illegal choice of lag times")
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating Markov models", flush=True)
    timer = datetime.now()
    
    mms_orig = []
    for i in range(len(lags)):
        print(flush=True)
        print("  Model {:>2d}:".format(i), flush=True)
        mms_orig.append(msm.estimate_markov_model(dtrajs=dtrajs, lag=lags[i]))
        print("    Lag time in trajectory steps:                               {:>6d}".format(mms_orig[i].lag), flush=True)
        print("    Lag time in real time units:                                {:>11s}".format(str(mms_orig[i].dt_model)), flush=True)
        print("    Largest implied timescale in trajectory steps:              {:>11.4f}".format(mms_orig[i].timescales()[0]))
        print("    2nd largest implied timescale in trajectory steps:          {:>11.4f}".format(mms_orig[i].timescales()[1]))
        print("    Number of active states (reversible connected):             {:>6d}".format(mms_orig[i].nstates), flush=True)
        print("    First active state:                                         {:>6d}".format(mms_orig[i].active_set[0]), flush=True)
        print("    Last active state:                                          {:>6d}".format(mms_orig[i].active_set[-1]), flush=True)
        print("    Total number of states in the discrete trajectories:        {:>6d}".format(mms_orig[i].nstates_full), flush=True)
        print("    Fraction of states in the largest reversible connected set: {:>11.4f}".format(mms_orig[i].active_state_fraction), flush=True)
        print("    Fraction of counts in the largest reversible connected set: {:>11.4f}".format(mms_orig[i].active_count_fraction), flush=True)
        
        if not np.all(np.isclose(np.sum(mms_orig[i].transition_matrix, axis=1), 1)):
            raise ValueError("Not all rows of the transition matrix sum up"
                             " to unity")
        if not np.isclose(np.sum(mms_orig[i].stationary_distribution), 1):
            raise ValueError("The sum of the stationary distribution ({})"
                             " is not unity"
                             .format(np.sum(mms_orig[i].stationary_distribution)))
    
    print(flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Comparing Markov models", flush=True)
    timer = datetime.now()
    
    # Transition matrix, eigenvalues, eigenvectors
    tm_diff = np.full((len(lags), len(lags)), np.nan, dtype=np.float32)
    eigvals = [[None for i in range(len(lags))]
               for j in range(len(lags))]
    neigvecs = 4
    eigvecs = [[[None for i in range(neigvecs)]
                for j in range(len(lags))]
               for k in range(len(lags))]
    
    # Test distributions
    peak_ix, _ = find_peaks(mms_orig[-1].stationary_distribution,
                            distance=3)
    ndists = neigvecs if neigvecs <= len(peak_ix) else len(peak_ix)
    if ndists == 0:
        ndists = 1
    ix = np.argsort(mms_orig[-1].stationary_distribution[peak_ix])[::-1]
    peak_ix = peak_ix[ix[:ndists-1]]
    peak_ix = np.append(peak_ix, int(mms_orig[-1].nstates/2))
    test_dists = [[[None for i in range(ndists)]
                   for j in range(len(lags))]
                  for k in range(len(lags))]
    del ix, _
    
    # Structure of the comparison matrices eigvals, eigvecs and
    # test_dists. Element ij compares Model i with model j, where the
    # models need to be exponentiated accordingly to reach the same
    # lag time. In the matrix element ij the (exponentiated) model j is
    # stored (in the scheme below the stored model is denoted by |x|).
    # eigvecs and test_dists consist of neigvecs and ndists such
    # matrices, one for each eigenvector or test distribution under
    # consideration.
    #
    # Model i\j 0       1         2         3      ...
    #       0   0|0  |  0^1|1  |  0^2|2  |  0^3|3|
    #       1   1|0^1|  1  |1  |  1^2|2  |  1^3|3|
    #       2   2|0^2|  2  |1^2|  2  |2  |  2^3|3|
    #       3   3|0^3|  3  |1^3|  3  |2^3|  3  |3|
    #       :
    
    for i in range(len(lags)):
        print(flush=True)
        print("  Model {:>2d}:".format(i), flush=True)
        
        for j in range(len(lags)):
            print("    Compared to Model {:>2d}:".format(j), flush=True)
            lowest_common_lag = np.lcm(lags[i], lags[j])
            print("      Lowest common lag time:  {:>6d}"
                  .format(lowest_common_lag),
                  flush=True)
            mms_comp = mdt.msm.align_active_sets(mm1=mms_orig[i],
                                                 mm2=mms_orig[j])
            mms_comp = mdt.msm.match_lag_time(mms=mms_comp,
                                              lags=lags[[i, j]])
            
            # Transition matrix
            if not np.isnan(tm_diff[i][j]):
                raise ValueError("The element [{}][{}] of tm_diff is"
                                 " already filled. This should not have"
                                 " happened".format(i, j))
            tm_diff[i][j] = np.sum(np.abs(mms_comp[0].transition_matrix-
                                          mms_comp[1].transition_matrix))
            tm_diff[i][j] /= mms_comp[1].nstates
            print("      Total absolute difference of the transition\n"
                  "      matrices normalized by the number of states: {}"
                  .format(tm_diff[i][j]),
                  flush=True)
            
            # Eigenvalues
            if eigvals[i][j] is not None:
                raise ValueError("The element [{}][{}] of eigvals is"
                                 " already filled. This should not have"
                                 " happened".format(i, j))
            eigvals[i][j] = mms_comp[1].eigenvalues()
            if not np.isclose(eigvals[i][j][0], 1):
                raise ValueError("The first eigenvalue ({}) of the"
                                 " exponentiated Model {} is not unity"
                                 .format(eigvals[i][j][0], j))
            
            # Eigenvectors
            if not np.all(np.array(eigvecs[i][j]) == None):
                raise ValueError("The element [{}][{}] of eigvecs is"
                                 " already filled. This should not have"
                                 " happened".format(i, j))
            if args.RIGHT:
                eigvecs[i][j] = mms_comp[1].eigenvectors_right(neigvecs)
            else:
                eigvecs[i][j] = mms_comp[1].eigenvectors_left(neigvecs)
            for k in range(len(eigvecs[i][j])):
                if not np.isclose(np.sum(eigvecs[i][j]), 1):
                    raise ValueError("The sum ({}) of eigenvector {} of"
                                     " the exponentiated Model {} is not"
                                     " unity"
                                     .format(np.sum(eigvecs[i][j]),
                                             k,
                                             j))
            
            # Test distributions
            if not np.all(np.array(test_dists[i][j]) == None):
                raise ValueError("The element [{}][{}] of test_dists is"
                                 " already filled. This should not have"
                                 " happened".format(i, j))
            for k in range(ndists):
                test_dists[i][j][k] = np.zeros(mms_comp[1].nstates,
                                               dtype=float)
                test_dists[i][j][k][peak_ix[k]] = 1
                if np.sum(test_dists[i][j][k]) != 1:
                    raise ValueError("The sum ({}) of test distribution"
                                     " {} of Model {} is not unity"
                                     .format(np.sum(test_dists[i][j][k]),
                                             k,
                                             j))
                test_dists[i][j][k] = mms_comp[1].propagate(
                                          test_dists[i][j][k],
                                          1)
                if not np.isclose(np.sum(test_dists[i][j][k]),
                                  1):
                    raise ValueError("The sum ({}) of the propagated"
                                     " test distribution {} of Model {}"
                                     " is not unity"
                                     .format(np.sum(test_dists[i][j][k]),
                                             k,
                                             j))
    
    if not np.all(np.isclose(tm_diff, mdt.nph.symmetrize(tm_diff))):
        print(flush=True)
        print("tm_diff =", flush=True)
        print(tm_diff, flush=True)
        raise ValueError("tm_diff is not symmetric. This should not have"
                         " happened")
    
    print(flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()
    
    
    fontsize_legend = 24
    
    
    # Difference of transition matrices (text file)
    filename = args.OUTFILE+"_cktest_tm_diff.txt"
    mdt.fh.savetxt_matrix(
        fname=filename,
        data=tm_diff,
        var1=lags,
        var2=lags,
        header=(
            "Markov state models\n\n"
            "Discrete trajectories used for construction:\n"
            + traj_info + "\n\n"
            "Chapman-Kolmogorow test\n"
            "  1/N_i * sum_{i,j}(|T_{ij}(tau2)^{lcm(tau1,tau2)/tau2} -\n"
            "                     T_{ij}(tau1)^{lcm(tau1,tau2)/tau1}|)\n"
            "  with T   = transition matrix\n"
            "       tau = lag time\n"
            "       lcm = lowest common multiple\n\n"
            "First column: Lag time tau1 in trajectory steps\n"
            "First row:    Lag time tau2 in trajectory steps\n"))
    print("  Created " + filename, flush=True)
    
    
    # Difference of transition matrices (plot)
    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    
    trailing_zeros = min([mdt.nph.trailing_digits(lag) for lag in lags])
    scale = 10**trailing_zeros
    ticklabels = (lags / scale).astype(int)
    if trailing_zeros == 1:
        steps = r'$10$'.format(trailing_zeros)
    elif trailing_zeros > 1:
        steps = r'$10^{:d}$'.format(trailing_zeros)
    else:
        steps = ''
    xy = np.arange(1, len(lags)+2) - 0.5
    axis.set_xticks(np.arange(1, len(lags)+1))
    axis.set_yticks(np.arange(1, len(lags)+1))
    axis.set_xticklabels(ticklabels)
    axis.set_yticklabels(ticklabels)
    
    heatmap = mdt.plot.pcolormesh(
                  ax=axis,
                  x=xy,
                  y=xy,
                  z=tm_diff,
                  xmin=xy[0],
                  xmax=xy[-1],
                  ymin=xy[0],
                  ymax=xy[-1],
                  xlabel=r"Lag time $\tau$ / {:s} steps".format(steps),
                  ylabel=r"Lag time $\tau'$ / {:s} steps".format(steps))
    heatmap.colorbar.set_label(
        r"$\frac{1}{N_i} \sum\limits_{i,j} \left| [T_{ij}(\tau')]^\frac{\mathrm{lcm}(\tau,\tau')}{\tau'} - [T_{ij}(\tau)]^\frac{\mathrm{lcm}(\tau,\tau')}{\tau} \right|$",
        y=1.0,
        horizontalalignment='right')
    
    mdt.plot.annotate_heatmap(im=heatmap,
                              data=tm_diff,
                              xpos=xy[:-1] + 0.5,
                              ypos=xy[:-1] + 0.5,
                              fmt='{x:.2f}',
                              textcolors=["red", "red"])
    
    filename = args.OUTFILE+"_cktest_tm_diff.pdf"
    mdt.fh.backup(filename)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("  Created " + filename, flush=True)
    
    
    # Eigenvalue spectrum
    # Complete overview
    #rows = len(lags)
    #cols = len(lags)
    #fig, axes = plt.subplots(nrows=rows,
                             #ncols=cols,
                             #squeeze=False,
                             #sharex=True,
                             #figsize=(cols*11.69, rows*8.27),
                             #frameon=False,
                             #clear=True,
                             #tight_layout=True)
    
    #for i in range(len(lags)):
        #xlabel = r'Index $i$'
        #for j in range(len(lags)):
            #ylabel = r'Eigenvalue $\lambda_i$'
            #axes[i][j].xaxis.set_tick_params(labelbottom=True)
            #axes[i][j].xaxis.offsetText.set_visible(True)
            #axes[i][j].xaxis.set_major_locator(MaxNLocator(integer=True))
            #axes[i][j].axhline(y=1, color='black', linestyle='--')
            #axes[i][j].axhline(color='black')
            #lowest_common_lag = np.lcm(lags[i], lags[j])
            #label = r'$\mathbf{T}(' + str("%d" %lags[j]) + r')$'
            #if lowest_common_lag/lags[j] > 1:
                #label = r'$[$' + label
                #label += (r'$]^{' +
                          #str("%d" %(lowest_common_lag/lags[j])) +
                          #r'}$')
            #mdt.plot.plot(ax=axes[i][j],
                          #x=np.arange(1, len(eigvals[i][j])+1),
                          #y=eigvals[i][j],
                          #linestyle='--',
                          #marker='o',
                          #xlabel=xlabel,
                          #ylabel=ylabel,
                          #label=label,
                          #legend_loc='upper right')
            
            #if j == len(lags)-1:
                #ylabel = r'Implied timescale $t_i$ / steps'
            #else:
                #ylabel = ''
            #img, ax2 = mdt.plot.plot_2nd_yaxis(
                           #ax=axes[i][j],
                           #x=np.arange(1, len(eigvals[i][j])+1),
                           #y=eigvals[i][j],
                           #ylabel=ylabel,
                           #alpha=0)
            #ax2.axhline(y=1, color='black', linestyle='--', alpha=0)
            #ax2.axhline(color='black', alpha=0)
            #a = np.array([a.tolist() for a in eigvals[i][:i+1]],
            #             dtype=object)
            #if np.min(np.concatenate(a.ravel(), axis=None)) >= 0:
                #axes[i][j].set_ylim(ymin=0)
            #ax2.set_ylim(ymin=axes[i][j].get_ylim()[0],
                         #ymax=axes[i][j].get_ylim()[1])
            #mask = np.logical_and(axes[i][j].get_yticks()>0,
                                  #axes[i][j].get_yticks()<1)
            #labels = -lowest_common_lag / np.log(axes[i][j].get_yticks(),
                                                 #where=mask)
            #labels = np.around(labels).astype(int).astype(str)
            #labels[np.logical_not(mask)] = ""
            #labels[axes[i][j].get_yticks()==0] = 0
            #labels[axes[i][j].get_yticks()==1] = r'$\infty$'
            #ax2.set_yticklabels(labels)
    
    #filename = args.OUTFILE+"_cktest_eigval_overview.pdf"
    #mdt.fh.backup(filename)
    #plt.tight_layout()
    #plt.savefig(filename)
    #plt.close()
    #print("  Created " + filename, flush=True)
    
    
    # Eigenvalue spectrum
    # Compare exponentiated models with each other
    filename = args.OUTFILE+"_cktest_eigval_comp.pdf"
    mdt.fh.backup(filename)
    with PdfPages(filename) as pdf:
        for i in range(1, len(lags)):
            fig, axis = plt.subplots(figsize=(11.69, 8.27),
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.axhline(y=1, color='black', linestyle='--')
            axis.axhline(color='black')
            
            lowest_common_lag = np.lcm.reduce(lags[:i+1])
            for j in range(i+1):
                label = r'$\mathbf{T}(' + str("%d" %lags[j]) + r')$'
                if lowest_common_lag/lags[j] > 1:
                    label = r'$[$' + label
                    label += (r'$]^{' +
                              str("%d" %(lowest_common_lag/lags[j])) +
                              r'}$')
                mdt.plot.plot(ax=axis,
                              x=np.arange(1, len(eigvals[i][j])+1),
                              y=eigvals[i][j],
                              xlabel=r'Index $i$',
                              ylabel=r'Eigenvalue $\lambda_i$',
                              label=label,
                              legend_loc='upper right',
                              linestyle='--',
                              marker='o',)
            axis.legend(loc='upper right',
                        numpoints=1,
                        ncol=1 + j//8,
                        frameon=False,
                        fontsize=fontsize_legend)
            
            img, ax2 = mdt.plot.plot_2nd_yaxis(
                           ax=axis,
                           x=np.arange(1, len(eigvals[i][0])+1),
                           y=eigvals[i][0],
                           ylabel=r'Implied timescale $t_i$ / steps',
                           alpha=0)
            ax2.axhline(y=1, color='black', linestyle='--', alpha=0)
            ax2.axhline(color='black', alpha=0)
            a = np.array([a.tolist() for a in eigvals[i][:i+1]],
                         dtype=object)
            if np.min(np.concatenate(a.ravel(), axis=None)) >= 0:
                axis.set_ylim(ymin=0)
            ax2.set_ylim(ymin=axis.get_ylim()[0],
                         ymax=axis.get_ylim()[1])
            mask = np.logical_and(axis.get_yticks()>0,
                                  axis.get_yticks()<1)
            labels = -lowest_common_lag / np.log(axis.get_yticks(),
                                                 where=mask)
            labels = np.around(labels).astype(int).astype(str)
            labels[np.logical_not(mask)] = ""
            labels[axis.get_yticks()==0] = 0
            labels[axis.get_yticks()==1] = r'$\infty$'
            ax2.set_yticklabels(labels)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print("  Created " + filename, flush=True)
    
    
    # Eigenvalue spectrum
    # Compare exponentiated models with original models
    filename = args.OUTFILE+"_cktest_eigval_self.pdf"
    mdt.fh.backup(filename)
    with PdfPages(filename) as pdf:
        for i in range(len(lags)-1):
            fig, axis = plt.subplots(figsize=(11.69, 8.27),
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.axhline(y=1, color='black', linestyle='--')
            axis.axhline(color='black')
            
            for j in range(i, len(lags)):
                lowest_common_lag = np.lcm(lags[i], lags[j])
                label = r'$\mathbf{T}(' + str("%d" %lags[i]) + r')$'
                if lowest_common_lag/lags[i] > 1:
                    label = r'$[$' + label
                    label += (r'$]^{' +
                              str("%d" %(lowest_common_lag/lags[i])) +
                              r'}$')
                mdt.plot.plot(ax=axis,
                              x=np.arange(1, len(eigvals[j][i])+1),
                              y=eigvals[j][i],
                              xlabel=r'Index $i$',
                              ylabel=r'Eigenvalue $\lambda_i$',
                              label=label,
                              legend_loc='upper right',
                              linestyle='--',
                              marker='o')
            axis.legend(loc='upper right',
                        numpoints=1,
                        ncol=1 + (len(lags)-i)//8,
                        frameon=False,
                        fontsize=fontsize_legend)
            a = np.array([a.tolist() for a in eigvals[i][i:]],
                         dtype=object)
            if np.min(np.concatenate(a.ravel(), axis=None)) >= 0:
                axis.set_ylim(ymin=0)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print("  Created " + filename, flush=True)
    
    
    # Eigenvectors
    # Compare exponentiated models with each other
    for i in range(neigvecs):
        if args.RIGHT:
            ylabel = r'Eigenvector $\psi_{'+str(i+1)+r'}$ / \%'
        else:
            ylabel = r'Eigenvector $\phi_{'+str(i+1)+r'}$ / \%'
        
        filename = args.OUTFILE+"_cktest_eigvec_comp"+str(i+1)+".pdf"
        mdt.fh.backup(filename)
        with PdfPages(filename) as pdf:
            for j in range(1, len(lags)):
                fig, axis = plt.subplots(figsize=(11.69, 8.27),
                                         frameon=False,
                                         clear=True,
                                         tight_layout=True)
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                axis.axhline(color='black', linestyle='--')
                
                small = np.argmin([mm.nstates for mm in mms_orig[:j+1]])
                lowest_common_lag = np.lcm.reduce(lags[:j+1])
                for k in range(j+1):
                    label = r'$\mathbf{T}(' + str("%d" %lags[k]) + r')$'
                    if lowest_common_lag/lags[k] > 1:
                        label = r'$[$' + label
                        label += (r'$]^{' +
                                  str("%d"%(lowest_common_lag/lags[k]))+
                                  r'}$')
                    mdt.plot.plot(
                        ax=axis,
                        x=mms_orig[small].active_set,
                        y=eigvecs[j][k][i]*100,  # *100 to convert to %
                        xmin=mms_orig[small].active_set[0]-0.5,
                        xmax=mms_orig[small].active_set[-1]+0.5,
                        xlabel=r'State $i$',
                        ylabel=ylabel,
                        label=label)
                axis.legend(loc='best',
                            numpoints=1,
                            ncol=1 + j//4,
                            frameon=False,
                            fontsize=fontsize_legend)
                a = np.array([a.tolist() for a in eigvecs[j][:j+1]],
                             dtype=object)
                if (np.min(np.concatenate(a.ravel(), axis=None)) >= 0
                    or i == 0):
                    axis.set_ylim(ymin=0)
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        print("  Created " + filename, flush=True)
    
    
    # Eigenvectors
    # Compare exponentiated models with original models
    for i in range(neigvecs):
        if args.RIGHT:
            ylabel = r'Eigenvector $\psi_{'+str(i+1)+r'}$ / \%'
        else:
            ylabel = r'Eigenvector $\phi_{'+str(i+1)+r'}$ / \%'
        
        filename = args.OUTFILE+"_cktest_eigvec_self"+str(i+1)+".pdf"
        mdt.fh.backup(filename)
        with PdfPages(filename) as pdf:
            for j in range(len(lags)-1):
                fig, axis = plt.subplots(figsize=(11.69, 8.27),
                                         frameon=False,
                                         clear=True,
                                         tight_layout=True)
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                axis.axhline(color='black', linestyle='--')
                
                for k in range(j, len(lags)):
                    lowest_common_lag = np.lcm(lags[j], lags[k])
                    label = r'$\mathbf{T}(' + str("%d" %lags[j]) + r')$'
                    if lowest_common_lag/lags[j] > 1:
                        label = r'$[$' + label
                        label += (r'$]^{' +
                                  str("%d"%(lowest_common_lag/lags[j]))+
                                  r'}$')
                    states = mdt.nph.get_middle(a=mms_orig[j].active_set,
                                                n=len(eigvecs[k][j][i]))
                    mdt.plot.plot(ax=axis,
                                  x=states,
                                  y=eigvecs[k][j][i]*100,  # *100 to convert to %
                                  xmin=states[0]-0.5,
                                  xmax=states[-1]+0.5,
                                  xlabel=r'State $i$',
                                  ylabel=ylabel,
                                  label=label)
                axis.legend(loc='best',
                            numpoints=1,
                            ncol=1 + (len(lags)-1-j)//4,
                            frameon=False,
                            fontsize=fontsize_legend)
                a = np.array([a.tolist() for a in eigvecs[j][j:]],
                             dtype=object)
                if (np.min(np.concatenate(a.ravel(), axis=None)) >= 0
                    or i == 0):
                    axis.set_ylim(ymin=0)
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        print("  Created " + filename, flush=True)
    
    
    # Test distributions
    small = np.argmin([mm.nstates for mm in mms_orig])
    for i in range(ndists):
        ylabel = (r'Probability $p_{' + 
                  str("%d"%peak_ix[i]) +
                  r'\rightarrow i}$ / \%')
        
        filename = args.OUTFILE+"_cktest_dist"+str(i+1)+".pdf"
        mdt.fh.backup(filename)
        with PdfPages(filename) as pdf:
            fig, axis = plt.subplots(figsize=(11.69, 8.27),
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            mdt.plot.plot(
                ax=axis,
                x=mms_orig[small].active_set,
                y=mms_orig[small].stationary_distribution*100,  # *100 to convert to %
                xmin=mms_orig[small].active_set[0]-0.5,
                xmax=mms_orig[small].active_set[-1]+0.5,
                ymin=0,
                xlabel=r'State $i$',
                ylabel=r'Stationary distribution $\mathbf{\pi}$ / \%',
                label=r'$\mathbf{T}('+str("%d" %lags[-1])+r')$')
            mdt.plot.plot(
                ax=axis,
                x=mms_orig[small].active_set[peak_ix[i]],
                y=mms_orig[small].stationary_distribution[peak_ix[i]]*100,  # *100 to convert to %
                xmin=mms_orig[small].active_set[0]-0.5,
                xmax=mms_orig[small].active_set[-1]+0.5,
                ymin=0,
                xlabel=r'State $i$',
                ylabel=r'Stationary distribution $\mathbf{\pi}$ / \%',
                marker='x',
                markersize=12,
                markeredgewidth=2,
                linestyle='')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            
            for j in range(1, len(lags)):
                fig, axis = plt.subplots(figsize=(11.69, 8.27),
                                         frameon=False,
                                         clear=True,
                                         tight_layout=True)
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                axis.axhline(color='black', linestyle='--')
                
                small = np.argmin([mm.nstates for mm in mms_orig[:j+1]])
                lowest_common_lag = np.lcm.reduce(lags[:j+1])
                for k in range(j+1):
                    label = r'$\mathbf{T}(' + str("%d" %lags[k]) + r')$'
                    if lowest_common_lag/lags[k] > 1:
                        label = r'$[$' + label
                        label += (r'$]^{' +
                                  str("%d" %(lowest_common_lag/lags[k])) +
                                  r'}$')
                    mdt.plot.plot(
                        ax=axis,
                        x=mms_orig[small].active_set,
                        y=test_dists[j][k][i]*100,  # *100 to convert to %
                        xmin=mms_orig[small].active_set[0]-0.5,
                        xmax=mms_orig[small].active_set[-1]+0.5,
                        ymin=0,
                        xlabel=r'State $i$',
                        ylabel=ylabel,
                        label=label)
                axis.legend(loc='best',
                            numpoints=1,
                            ncol=1 + j//10,
                            frameon=False,
                            fontsize=fontsize_legend)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        
        print("  Created " + filename, flush=True)
    
    
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
