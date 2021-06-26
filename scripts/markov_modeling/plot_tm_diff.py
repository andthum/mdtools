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
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
import pyemma
import mdtools as mdt




if __name__ == "__main__":
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read two pyemma.msm.MaximumLikelihoodMSM or"
                     " pyemma.msm.BayesianMSM objects from file (which"
                     " must have been created by the object's save"
                     " method). Exponentiate the transition matrices by"
                     " the lowest common multiple of the lag times of"
                     " the two models divided by the model's respective"
                     " lag time. Then plot the difference of the"
                     " transition matrices (second minus first), their"
                     " eigenvalue spectra and their first four"
                     " eigenvectors. If the two Markov models were"
                     " estimated from the same system but with different"
                     " lag times, this corresponds to a simple Chapman-"
                     "Kolmogorow test. If the estimated models are"
                     " really Markovian, the exponentiated transition"
                     " matrices (and all quantities derived therefrom)"
                     " must be the same within numerical uncertainty."
                 )
    )
    
    parser.add_argument(
        '--f1',
        dest='INFILE1',
        type=str,
        required=True,
        help="Input file 1 containing the first"
             " pyemma.msm.MaximumLikelihoodMSM or pyemma.msm.BayesianMSM"
             " object in HDF5 format as created by the object's save"
             " method."
    )
    parser.add_argument(
        '--f2',
        dest='INFILE2',
        type=str,
        required=True,
        help="Input file 2 containing the second"
             " pyemma.msm.MaximumLikelihoodMSM or pyemma.msm.BayesianMSM"
             " object in HDF5 format as created by the object's save"
             " method."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename pattern. There will be created four files:"
             " <OUTFILE>_tm_diff.pdf containing a heatmap plot of the"
             " difference of the exponentiated transition matrices;"
             " <OUTFILE>_tm_diff_eigval.pdf containing a plot of the"
             " eigenvalue spectra of the exponentiated transition"
             " matrices with corresponding implied timescales;"
             " <OUTFILE>_tm_diff_eigvec.pdf containing a plot of the"
             " 1st-4th eigenvectors of the exponentiated transition"
             " matrices."
             " <OUTFILE>_tm_diff_dist.pdf comparing the final"
             " distributions when propagating a well-defined initial"
             " distribution with the exponentiated transition matrices;"
             " Plots are optimized for PDF format with TeX support."
    )
    parser.add_argument(
        '--plot-exp',
        dest='PLOT_EXP',
        required=False,
        default=False,
        action='store_true',
        help="Create additional plots comparing the original models"
             " to the exponentiated models. The plots are only created,"
             " if the model was really exponentiated."
             " <OUTFILE>_tm_diff_tm<i>.pdf containing a heatmap plot of"
             " the exponentiated transition matrices of the i-th input"
             " model;"
             " <OUTFILE>_tm_diff_eigval<i>.pdf containing a plot"
             " comparing the eigenvalue spectra of the original and"
             " exponentiated transition matrix;"
             " <OUTFILE>_tm_diff_eigvec<i>.pdf containing a plot"
             " comparing the 1st-4th eigenvectors of the original and"
             " exponentiated transition matrix;"
    )
    parser.add_argument(
        '--right',
        dest='RIGHT',
        required=False,
        default=False,
        action='store_true',
        help="Plot right eigenvectors (column vectors) instead of left"
             " eigenvectors (row vectors) of the transition matrices."
             " Since transition matrices are row-stochastic, the left"
             " eigenvectors are real probability densities, whereas the"
             " right eigenvectors are probability densities reweighted"
             " by the stationary distribution."
    )
    parser.add_argument(
        '--fine2coarse',
        dest='FINE2COARSE',
        type=int,
        required=False,
        default=None,
        help="If the two models to compare were created using different"
             " discretizations, you can give here the number of states"
             " of the fine model corresponding to one state of the"
             " coarse model. Comparing models with a different number of"
             " states requires that both models were created in exactly"
             " the same way, except that the discretization of one model"
             " is coarser. However, the number of bins used for the fine"
             " model must be an integer multiple of the number of bins"
             " used for the coarse model. If --fine2coarse is not given"
             " and the active sets of the models differ, a guess how to"
             " compare the models is made. This functionality is not"
             " robust and you should know what you are comparing and"
             " what you are expecting."
    )
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    
    
    
    print("\n\n\n", flush=True)
    print("Loading Markov models", flush=True)
    timer = datetime.now()
    
    infile = [args.INFILE1, args.INFILE2]
    mms_orig = [None,] * len(infile)
    lags = np.zeros(len(infile), dtype=int)
    for i in range(len(mms_orig)):
        print(flush=True)
        print("  Model {:>2d}:".format(i), flush=True)
        mms_orig[i] = pyemma.load(infile[i])
        lags[i] = mms_orig[i].lag
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
    
    lowest_common_lag = np.lcm(lags[0], lags[1])
    print(flush=True)
    print("  Lowest common lag time:  {:>6d}"
          .format(lowest_common_lag),
          flush=True)
    mms_comp = mdt.msm.match_active_sets(mm1=mms_orig[0],
                                         mm2=mms_orig[1],
                                         fine2coarse=args.FINE2COARSE)
    mms_comp = mdt.msm.match_lag_time(mms=mms_comp, lags=lags)
    tm_diff = (mms_comp[1].transition_matrix -
               mms_comp[0].transition_matrix)
    print(flush=True)
    print("  Total absolute difference of the transition\n"
          "  matrices normalized by the number of states: {}"
          .format(np.sum(np.abs(tm_diff))/mms_comp[0].nstates),
          flush=True)
    
    neigvecs = 4
    eigvals_orig = [None for i in range(len(mms_orig))]
    eigvecs_orig = [[None for i in range(neigvecs)]
                    for j in range(len(mms_orig))]
    eigvals_comp = np.full((len(mms_comp), mms_comp[0].nstates),
                           np.nan,
                           dtype=np.float32)
    eigvecs_comp = np.full((len(mms_comp),neigvecs,mms_comp[0].nstates),
                           np.nan,
                           dtype=np.float32)
    
    old = np.argmax(lags)
    peak_ix, _ = find_peaks(mms_comp[old].stationary_distribution,
                            distance=3)
    ndists = 3 if 3 <= len(peak_ix) else len(peak_ix)
    if ndists == 0:
        ndists = 1
    ix = np.argsort(mms_comp[old].stationary_distribution[peak_ix])[::-1]
    peak_ix = peak_ix[ix[:ndists-1]]
    peak_ix = np.append(peak_ix, int(mms_comp[old].nstates/2))
    test_dists = np.full((len(mms_comp), ndists, mms_comp[old].nstates),
                         np.nan,
                         dtype=np.float32)
    del ix, _
    
    for i in range(len(mms_orig)):
        # Eigenvalues
        if (eigvals_orig[i] is not None or
            not np.all(np.isnan(eigvals_comp[i]))):
            raise ValueError("The element [{}] of eigvals is already"
                             " filled. This should not have happened"
                             .format(i))
        eigvals_orig[i] = mms_orig[i].eigenvalues()
        eigvals_comp[i] = mms_comp[i].eigenvalues()
        if not np.isclose(eigvals_orig[i][0], 1):
            raise ValueError("The first eigenvalue ({}) of the original"
                             "  Model {} is not unity"
                             .format(eigvals_orig[i][0], i))
        if not np.isclose(eigvals_comp[i][0], 1):
            raise ValueError("The first eigenvalue ({}) of the"
                             " exponentiated Model {} is not unity"
                             .format(eigvals_comp[i][0], i))
        
        # Eigenvectors
        if (not np.all(np.array(eigvecs_orig[i]) == None) or
            not np.all(np.isnan(eigvecs_comp[i]))):
            raise ValueError("The element [{}] of eigvecs is already"
                             " filled. This should not have happened"
                             .format(i))
        if args.RIGHT:
            eigvecs_orig[i] = mms_orig[i].eigenvectors_right(neigvecs)
            eigvecs_comp[i] = mms_comp[i].eigenvectors_right(neigvecs)
        else:
            eigvecs_orig[i] = mms_orig[i].eigenvectors_left(neigvecs)
            eigvecs_comp[i] = mms_comp[i].eigenvectors_left(neigvecs)
        for j in range(len(eigvecs_orig[i])):
            if not np.isclose(np.sum(eigvecs_orig[i]), 1):
                raise ValueError("The sum ({}) of eigenvector {} of the"
                                 " original Model {} is not unity"
                                 .format(np.sum(eigvecs_orig[i]), j, i))
        for j in range(len(eigvecs_comp[i])):
            if not np.isclose(np.sum(eigvecs_comp[i]), 1):
                raise ValueError("The sum ({}) of eigenvector {} of the"
                                 " original Model {} is not unity"
                                 .format(np.sum(eigvecs_comp[i]), j, i))
        
        # Test distributions
        if not np.all(np.isnan(test_dists[i])):
            raise ValueError("The element [{}] of test_dists is already"
                             " filled. This should not have happened"
                             .format(i))
        for j in range(ndists):
            test_dists[i][j] = np.zeros(mms_comp[old].nstates,
                                        dtype=np.float32)
            test_dists[i][j][peak_ix[j]] = 1
            if np.sum(test_dists[i][j]) != 1:
                raise ValueError("The sum ({}) of test distribution {}"
                                 " of Model {} is not unity"
                                 .format(np.sum(test_dists[i][j]),j,i))
            test_dists[i][j] = mms_comp[i].propagate(test_dists[i][j],1)
            if not np.isclose(np.sum(test_dists[i][j]), 1):
                raise ValueError("The sum ({}) of the propagated test"
                                 " distribution {} of Model {} is not"
                                 " unity"
                                 .format(np.sum(test_dists[i][j]),j,i))
    
    print(flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating plots", flush=True)
    timer = datetime.now()
    
    
    fontsize_legend = 28
    
    
    small = np.argmin([mm.nstates for mm in mms_orig])
    states = np.arange(mms_orig[small].active_set[0],
                       mms_orig[small].active_set[-1]+1)
    xy = mms_orig[small].active_set - 0.5
    xy = np.append(xy, mms_orig[small].active_set[-1]+0.5)
    
    
    # Exponentiated transition matrices
    for i in range(len(mms_comp)):
        if args.PLOT_EXP and lowest_common_lag/lags[i] > 1:
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            cbarlabel = r'$T_{ij}(' + str("%d" %lags[i]) + r')$'
            cbarlabel = r'$[$' + cbarlabel
            cbarlabel += (r'$]^{' +
                          str("%d" %(lowest_common_lag/lags[i])) +
                          r'}$')
            
            mdt.plot.pcolormesh(
                ax=axis,
                x=xy,
                y=xy,
                z=mms_comp[i].transition_matrix,
                xmin=states[0]-0.5,
                xmax=states[-1]+0.5,
                ymin=states[0]-0.5,
                ymax=states[-1]+0.5,
                xlabel=r'State $j$',
                ylabel=r'State $i$',
                cbarlabel=cbarlabel)
            
            axis.invert_yaxis()
            axis.xaxis.set_label_position('top')
            axis.xaxis.labelpad = 22
            axis.xaxis.tick_top()
            axis.tick_params(axis='x', which='both', pad=6)
            
            mdt.fh.backup(args.OUTFILE+"_tm_diff_tm"+str(i+1)+".pdf")
            plt.tight_layout()
            plt.savefig(args.OUTFILE+"_tm_diff_tm"+str(i+1)+".pdf")
            plt.close(fig)
            print("  Created "+args.OUTFILE+"_tm_diff_tm"+str(i+1)+".pdf",
                  flush=True)
    
    
    # Difference of exponentiated transition matrices
    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    cbarlabel = [None, None]
    for i in range(len(cbarlabel)):
        cbarlabel[i] = r'$T_{ij}(' + str("%d" %lags[i]) + r')$'
        if lowest_common_lag/lags[i] > 1:
            cbarlabel[i] = r'$[$' + cbarlabel[i]
            cbarlabel[i] += (r'$]^{' +
                             str("%d" %(lowest_common_lag/lags[i])) +
                             r'}$')
    cbarlabel = cbarlabel[1] + r'$ - $' + cbarlabel[0]
    
    # TODO: Update matplotlib and remove the first if clause (only keep
    # else clause)
    import matplotlib
    if matplotlib.__version__ <= '3.1.0':
        warnings.warn("Your matplotlib version ({}) is less than 3.1.0"
                      .format(matplotlib.__version__),
                      DeprecationWarning)
        mdt.plot.pcolormesh(
            ax=axis,
            x=xy,
            y=xy,
            z=tm_diff,
            xmin=states[0]-0.5,
            xmax=states[-1]+0.5,
            ymin=states[0]-0.5,
            ymax=states[-1]+0.5,
            xlabel=r'State $j$',
            ylabel=r'State $i$',
            cmap='bwr',
            norm=mdt.plot.MidpointNormalize(midpoint=0.0),
            cbarlabel=cbarlabel)
    else:
        if np.min(tm_diff) < 0 and np.max(tm_diff) > 0:
            colors.DivergingNorm(vmin=np.min(tm_diff),
                                 vcenter=0.0,
                                 vmax=np.max(tm_diff))
        mdt.plot.pcolormesh(
            ax=axis,
            x=xy,
            y=xy,
            z=tm_diff,
            xmin=states[0]-0.5,
            xmax=states[-1]+0.5,
            ymin=states[0]-0.5,
            ymax=states[-1]+0.5,
            vmin=np.min(tm_diff),
            vmax=np.max(tm_diff),
            xlabel=r'State $j$',
            ylabel=r'State $i$',
            cmap='bwr',
            cbarlabel=cbarlabel)
    
    axis.invert_yaxis()
    axis.xaxis.set_label_position('top')
    axis.xaxis.labelpad = 22
    axis.xaxis.tick_top()
    axis.tick_params(axis='x', which='both', pad=6)
    
    mdt.fh.backup(args.OUTFILE+"_tm_diff.pdf")
    plt.tight_layout()
    plt.savefig(args.OUTFILE+"_tm_diff.pdf")
    plt.close(fig)
    print("  Created "+args.OUTFILE+"_tm_diff.pdf", flush=True)
    
    
    # Eigenvalue spectrum
    # Compare exponentiated models with each other
    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    axis.axhline(y=1, color='black', linestyle='--')
    axis.axhline(color='black')
    for i in range(len(mms_comp)):
        label = r'$\mathbf{T}(' + str("%d" %lags[i]) + r')$'
        if lowest_common_lag/lags[i] > 1:
            label = r'$[$' + label
            label += (r'$]^{' +
                      str("%d" %(lowest_common_lag/lags[i])) +
                      r'}$')
        mdt.plot.plot(ax=axis,
                      x=np.arange(1, len(eigvals_comp[i])+1),
                      y=eigvals_comp[i],
                      linestyle='--',
                      marker='o',
                      xlabel=r'Index $i$',
                      ylabel=r'Eigenvalue $\lambda_i$',
                      label=label)
    
    img, ax2 = mdt.plot.plot_2nd_yaxis(
                   ax=axis,
                   x=np.arange(1, len(eigvals_comp[0])+1),
                   y=eigvals_comp[0],
                   ylabel=r'Implied timescale $t_i$ / steps',
                   alpha=0)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0)
    ax2.axhline(color='black', alpha=0)
    if np.min(np.concatenate(eigvals_comp)) >= 0:
        axis.set_ylim(ymin=0)
    ax2.set_ylim(ymin=axis.get_ylim()[0], ymax=axis.get_ylim()[1])
    mask = np.logical_and(axis.get_yticks()>0, axis.get_yticks()<1)
    labels = -lowest_common_lag / np.log(axis.get_yticks(), where=mask)
    labels = np.around(labels).astype(int).astype(str)
    labels[np.logical_not(mask)] = ""
    labels[axis.get_yticks()==0] = 0
    labels[axis.get_yticks()==1] = r'$\infty$'
    ax2.set_yticklabels(labels)
    
    mdt.fh.backup(args.OUTFILE+"_tm_diff_eigval.pdf")
    plt.tight_layout()
    plt.savefig(args.OUTFILE+"_tm_diff_eigval.pdf")
    plt.close(fig)
    print("  Created "+args.OUTFILE+"_tm_diff_eigval.pdf", flush=True)
    
    
    # Eigenvalue spectrum
    # Compare exponentiated models with original models
    for i in range(len(mms_comp)):
        if args.PLOT_EXP and lowest_common_lag/lags[i] > 1:
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            label_orig = r'$\mathbf{T}(' + str("%d" %lags[i]) + r')$'
            label_comp = r'$[$' + label_orig
            label_comp += (r'$]^{' +
                           str("%d" %(lowest_common_lag/lags[i])) +
                           r'}$')
            
            axis.axhline(y=1, color='black', linestyle='--')
            axis.axhline(color='black')
            mdt.plot.plot(ax=axis,
                          x=np.arange(1, len(eigvals_orig[i])+1),
                          y=eigvals_orig[i],
                          linestyle='--',
                          marker='o',
                          label=label_orig)
            mdt.plot.plot(ax=axis,
                          x=np.arange(1, len(eigvals_comp[i])+1),
                          y=eigvals_comp[i],
                          linestyle='--',
                          marker='o',
                          xlabel=r'Index $i$',
                          ylabel=r'Eigenvalue $\lambda_i$',
                          label=label_comp)
            
            mdt.fh.backup(args.OUTFILE+"_tm_diff_eigval"+str(i+1)+".pdf")
            plt.tight_layout()
            plt.savefig(args.OUTFILE+"_tm_diff_eigval"+str(i+1)+".pdf")
            plt.close(fig)
            print("  Created "+args.OUTFILE+"_tm_diff_eigval"+str(i+1)+".pdf",
                  flush=True)
    
    
    # Eigenvectors
    # Compare exponentiated models with each other
    fig, axes = plt.subplots(nrows=neigvecs,
                             squeeze=False,
                             sharex=True,
                             sharey=True,
                             figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True)
    axes = axes[:,0]
    
    for i in range(neigvecs):
        if args.RIGHT:
            ylabel = r'$\psi_{'+str(i+1)+r'}$/\%'
        else:
            ylabel = r'$\phi_{'+str(i+1)+r'}$/\%'
        if i == neigvecs-1:
            xlabel = r'State $i$'
        else:
            xlabel = ''
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        for j in range(len(mms_comp)):
            if i == 0:
                label = r'$\mathbf{T}(' + str("%d" %lags[j]) + r')$'
                if lowest_common_lag/lags[j] > 1:
                    label = r'$[$' + label
                    label += (r'$]^{' +
                          str("%d" %(lowest_common_lag/lags[j])) +
                          r'}$')
            else:
                label = ''
            axes[i].fill_between(x=states,
                                 y1=eigvecs_comp[j][i]*100,  # *100 to convert to %
                                 alpha=0.5)
            mdt.plot.plot(ax=axes[i],
                          x=states,
                          y=eigvecs_comp[j][i]*100,  # *100 to convert to %
                          xmin=states[0]-0.5,
                          xmax=states[-1]+0.5,
                          xlabel=xlabel,
                          ylabel=ylabel,
                          label=label)
        axes[i].legend(ncol=len(mms_comp),
                       loc='lower center',
                       numpoints=1,
                       frameon=False,
                       fontsize=fontsize_legend)
    
    mdt.fh.backup(args.OUTFILE+"_tm_diff_eigvec.pdf")
    plt.tight_layout(h_pad=0)
    plt.savefig(args.OUTFILE+"_tm_diff_eigvec.pdf")
    plt.close(fig)
    print("  Created "+args.OUTFILE+"_tm_diff_eigvec.pdf", flush=True)
    
    # Eigenvectors
    # Compare exponentiated models with original models
    for j in range(len(mms_comp)):
        if args.PLOT_EXP and lowest_common_lag/lags[j] > 1:
            fig, axes = plt.subplots(nrows=neigvecs,
                                     squeeze=False,
                                     sharex=True,
                                     sharey=True,
                                     figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True)
            axes = axes[:,0]
            
            for i in range(neigvecs):
                if args.RIGHT:
                    ylabel = r'$\psi_{'+str(i+1)+r'}$/\%'
                else:
                    ylabel = r'$\phi_{'+str(i+1)+r'}$/\%'
                if i == neigvecs-1:
                    xlabel = r'State $i$'
                else:
                    xlabel = ''
                axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
                if i == 0:
                    label_orig = r'$\mathbf{T}(' + str("%d" %lags[i]) + r')$'
                    label_comp = r'$[$' + label_orig
                    label_comp += (r'$]^{' +
                                   str("%d" %(lowest_common_lag/lags[i])) +
                                   r'}$')
                else:
                    label_orig = None
                    label_comp = None
                axes[i].fill_between(x=mms_orig[j].active_set,
                                     y1=eigvecs_orig[j][i]*100,  # *100 to convert to %
                                     alpha=0.5)
                mdt.plot.plot(ax=axes[i],
                              x=mms_orig[j].active_set,
                              y=eigvecs_orig[j][i]*100,  # *100 to convert to %
                              xmin=states[0]-0.5,
                              xmax=states[-1]+0.5,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              label=label_orig)
                axes[i].fill_between(x=states,
                                     y1=eigvecs_comp[j][i]*100,  # *100 to convert to %
                                     alpha=0.5)
                mdt.plot.plot(ax=axes[i],
                              x=states,
                              y=eigvecs_comp[j][i]*100,  # *100 to convert to %
                              xmin=states[0]-0.5,
                              xmax=states[-1]+0.5,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              label=label_comp)
                axes[i].legend(ncol=len(mms_comp),
                               loc='lower center',
                               numpoints=1,
                               frameon=False,
                               fontsize=fontsize_legend)
            
            mdt.fh.backup(args.OUTFILE+"_tm_diff_eigvec"+str(j+1)+".pdf")
            plt.tight_layout() #h_pad=0
            plt.savefig(args.OUTFILE+"_tm_diff_eigvec"+str(j+1)+".pdf")
            plt.close(fig)
            print("  Created "+args.OUTFILE+"_tm_diff_eigvec"+str(j+1)+".pdf",
                  flush=True)
    
    
    # Test distributions
    fig, axes = plt.subplots(nrows=ndists+1,
                             squeeze=False,
                             sharex=True,
                             figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True)
    axes = axes[:,0]
    
    mdt.plot.plot(ax=axes[0],
                  x=states,
                  y=mms_comp[old].stationary_distribution*100,  # *100 to convert to %
                  xmin=states[0]-0.5,
                  xmax=states[-1]+0.5,
                  ymin=0,
                  xlabel='',
                  ylabel=r'$\mathbf{\pi}$/\%',
                  label=r'$\mathbf{T}(' + str("%d" %lags[old]) + r')$')
    mdt.plot.plot(ax=axes[0],
                  x=states[peak_ix],
                  y=mms_comp[old].stationary_distribution[peak_ix]*100,  # *100 to convert to %
                  xmin=states[0]-0.5,
                  xmax=states[-1]+0.5,
                  ymin=0,
                  xlabel='',
                  ylabel=r'$\mathbf{\pi}$/\%',
                  marker='x',
                  markersize=12,
                  markeredgewidth=2,
                  linestyle='')
    
    for i in range(1, ndists+1):
        ylabel = (r'$p_{' +
                  str("%d" %states[peak_ix[i-1]]) +
                  r'\rightarrow i}$/\%')
        if i == ndists:
            xlabel = r'State $i$'
        else:
            xlabel = ''
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        for j in range(len(mms_comp)):
            if i == 1:
                label = r'$\mathbf{T}(' + str("%d" %lags[j]) + r')$'
                if lowest_common_lag/lags[j] > 1:
                    label = r'$[$' + label
                    label += (r'$]^{' +
                          str("%d" %(lowest_common_lag/lags[j])) +
                          r'}$')
            else:
                label = ''
            mdt.plot.plot(ax=axes[i],
                          x=states,
                          y=test_dists[j][i-1]*100,  # *100 to convert to %
                          xmin=states[0]-0.5,
                          xmax=states[-1]+0.5,
                          ymin=0,
                          xlabel=xlabel,
                          ylabel=ylabel,
                          label=label)
    
    mdt.fh.backup(args.OUTFILE+"_tm_diff_dist.pdf")
    plt.tight_layout() #h_pad=0
    plt.savefig(args.OUTFILE+"_tm_diff_dist.pdf")
    plt.close(fig)
    print("  Created "+args.OUTFILE+"_tm_diff_dist.pdf", flush=True)
    
    
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
