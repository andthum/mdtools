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
from matplotlib.ticker import MaxNLocator
import pyemma
import mdtools as mdt




if __name__ == "__main__":
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read one or more pyemma.msm.MaximumLikelihoodMSM"
                     " or pyemma.msm.BayesianMSM objects from file"
                     " (which must have been created by the objects'"
                     " save method) and propagate a given initial"
                     " distribution k times. If the lag time of the"
                     " transition matrix is tau, this will provide the"
                     " probability distribution at time k*tau. If"
                     " multiple Markov models are given, they all must"
                     " have the same lag time. This script is e.g."
                     " useful to propagate a given initial dristribution"
                     " with a single model or to compare the final"
                     " distributions when propagating different models"
                     " created with different discretizations."
                 )
    )
    
    parser.add_argument(
        '-f',
        dest='INFILES',
        type=str,
        nargs="+",
        required=True,
        help="Input file or space separated list of input files each"
             " containing a pyemma.msm.MaximumLikelihoodMSM or"
             " pyemma.msm.BayesianMSM object in HDF5 format as created"
             " by the objects' save method."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output file. Plots are optimized for PDF format with TeX"
             " support."
    )
    
    parser.add_argument(
        '-k',
        dest='K',
        type=int,
        required=False,
        default=1,
        help="How many times to propagate the initial distribution."
             " Default: 1"
    )
    parser.add_argument(
        '-s',
        dest='INIT_STATE',
        type=int,
        nargs="+",
        required=False,
        default=None,
        help="Initial distribution. Give the index of a state from the"
             " active set (indexing starts at zero) of the model. The"
             " intial distribution will have 100 %% population in this"
             " state and no population in all other states. It will be"
             " marked by a cross in the plot of the stationary"
             " distribution of the model. If multiple models are given,"
             " you can give as many initial states as models or only"
             " one inital state. If only one initial state is given for"
             " all models, it is treated as initial state for the first"
             " model. The initial states for the other models are"
             " created at the corresponding relative position of their"
             " respective active sets."
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
    print("Loading Markov models", flush=True)
    timer = datetime.now()
    
    n_models = len(args.INFILES)
    print("  Number of models to be loaded: {}"
          .format(n_models),
          flush=True)
    
    mms = [None,] * n_models
    for i in range(n_models):
        print(flush=True)
        print("  Model {:>2d}:".format(i), flush=True)
        mms[i] = pyemma.load(args.INFILES[i])
        print("    Lag time in trajectory steps:                               {:>6d}".format(mms[i].lag), flush=True)
        print("    Lag time in real time units:                                {:>11s}".format(str(mms[i].dt_model)), flush=True)
        print("    Largest implied timescale in trajectory steps:              {:>11.4f}".format(mms[i].timescales()[0]))
        print("    2nd largest implied timescale in trajectory steps:          {:>11.4f}".format(mms[i].timescales()[1]))
        print("    Number of active states (reversible connected):             {:>6d}".format(mms[i].nstates), flush=True)
        print("    First active state:                                         {:>6d}".format(mms[i].active_set[0]), flush=True)
        print("    Last active state:                                          {:>6d}".format(mms[i].active_set[-1]), flush=True)
        print("    Total number of states in the discrete trajectories:        {:>6d}".format(mms[i].nstates_full), flush=True)
        print("    Fraction of states in the largest reversible connected set: {:>11.4f}".format(mms[i].active_state_fraction), flush=True)
        print("    Fraction of counts in the largest reversible connected set: {:>11.4f}".format(mms[i].active_count_fraction), flush=True)
        
        if not np.all(np.isclose(np.sum(mms[i].transition_matrix, axis=1), 1)):
            raise ValueError("Not all rows of the transition matrix sum up"
                             " to unity")
        if not np.isclose(np.sum(mms[i].stationary_distribution), 1):
            raise ValueError("The sum of the stationary distribution ({})"
                             " is not unity"
                             .format(np.sum(mms[i].stationary_distribution)))
    
    if not np.all(np.array([mm.lag for mm in mms]) == mms[0].lag):
        raise ValueError("If multiple Markov models are given, they all"
                         " must have the same lag time")
    
    print(flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Propagate initial distribution", flush=True)
    timer = datetime.now()
    
    if args.INIT_STATE is None:
        args.INIT_STATE = np.zeros(n_models, dtype=int)
        for i in range(n_models):
            args.INIT_STATE[i] = int(mms[i].nstates / 2)
    elif len(args.INIT_STATE) != n_models:
        if len(args.INIT_STATE) != 1:
            raise ValueError("You must either give as many inital states"
                             " as Markov models or exactly one inital"
                             " state")
        rel_pos = args.INIT_STATE[0] / mms[0].nstates
        for i in range(1, n_models):
            args.INIT_STATE.append(int(rel_pos * mms[i].nstates))
    
    dists = [None,] * n_models
    for i in range(n_models):
        dists[i] = np.zeros(mms[i].nstates, dtype=np.float32)
        dists[i][args.INIT_STATE[i]] = 1
        if np.sum(dists[i]) != 1:
            raise ValueError("The sum ({}) of the initial distribution"
                             " to propagate with Model {} is not unity"
                             .format(np.sum(dists[i]), i))
        dists[i] = mms[i].propagate(dists[i], args.K)
        if not np.isclose(np.sum(dists[i]), 1):
            raise ValueError("The sum ({}) of the distribution"
                             " propagated with Model {} is not unity"
                             .format(np.sum(dists[i]), i))
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating plots", flush=True)
    timer = datetime.now()
    
    
    if args.LABEL is None:
        args.LABEL = [''] * n_models
    
    if n_models == 1:
        xlabel = r'State $i$'
        xmin = mms[0].active_set[0]-0.5
        xmax = mms[0].active_set[-1]+0.5
        xdata = [mms[0].active_set]
        ydata = [dists[0]]
        stat_dist = [mms[0].stationary_distribution]
    else:
        xlabel = ''
        xmin = 0
        xmax = 1
        xdata = [None,] * n_models
        stat_dist = [None,] * n_models
        for i in range(n_models):
            xdata[i] = np.linspace(xmin, xmax, mms[i].nstates)
            integral = np.trapz(y=dists[i], x=xdata[i])
            dists[i] /= integral
            if not np.isclose(np.trapz(y=dists[i], x=xdata[i]), 1):
                raise ValueError("The propagated distribution of Model"
                                 " {} does not sum up to unitiy"
                                 .format(i))
            integral = np.trapz(y=mms[i].stationary_distribution,
                                x=xdata[i])
            stat_dist[i] = mms[i].stationary_distribution / integral
            if not np.isclose(np.trapz(y=stat_dist[i], x=xdata[i]), 1):
                raise ValueError("The stationary distribution of Model"
                                 " {} does not sum up to unitiy"
                                 .format(i))
    
    
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        
        if n_models == 1:
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        for i in range(n_models):
            mdt.plot.plot(
                ax=axis,
                x=xdata[i],
                y=stat_dist[i],
                xmin=xmin,
                xmax=xmax,
                ymin=0,
                xlabel=xlabel,
                ylabel=r'Stationary distribution $\mathbf{\pi}$',
                label=args.LABEL[i])
            mdt.plot.plot(
                ax=axis,
                x=xdata[i][args.INIT_STATE[i]],
                y=stat_dist[i][args.INIT_STATE[i]],
                xmin=xmin,
                xmax=xmax,
                ymin=0,
                xlabel=xlabel,
                ylabel=r'Stationary distribution $\mathbf{\pi}$',
                marker='x',
                markersize=12,
                markeredgewidth=2,
                linestyle='')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        
        if n_models == 1:
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            ylabel = (r'Probability $p_{' + 
                      str("%d"%args.INIT_STATE) +
                      r'\rightarrow i}$')
        else:
            ylabel = (r'$p_{' +
                      str("%d"%args.K) +
                      r'\tau} = p_{t_0}^T \cdot [T(\tau=' +
                      str("%d"%mms[i].lag) +
                      r')]^{' +
                      str("%d"%args.K) +
                      r'}$')
        
        for i in range(n_models):
            mdt.plot.plot(
                ax=axis,
                x=xdata[i],
                y=dists[i],
                xmin=xmin,
                xmax=xmax,
                ymin=0,
                xlabel=xlabel,
                ylabel=ylabel,
                label=args.LABEL[i])
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    print("  Created " + args.OUTFILE, flush=True)
    
    
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
