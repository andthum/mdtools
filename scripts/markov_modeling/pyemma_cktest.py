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
from scipy.signal import find_peaks
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pyemma
import pyemma.plots as mplt
import mdtools as mdt




if __name__ == "__main__":
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    n_cpus = mdt.rti.get_num_CPUs()
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read a pyemma.msm.MaximumLikelihoodMSM or"
                     " pyemma.msm.BayesianMSM object from file (which"
                     " must have been created by the object's save"
                     " method) and conduct a Chapman-Kolmogorow test on"
                     " it."
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
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename pattern. There will be created five files:"
             " <OUTFILE>_cktest.h5 containing the created"
             " pyemma.msm.ChapmanKolmogorovValidator object;"
             " <OUTFILE>_cktest_assignments.txt containing the assigment"
             " of each active state to a metastable set from PCCA++"
             " (Perron Cluster Cluster Analysis);"
             " <OUTFILE>_cktest.pdf containing a visualization of the"
             " Chapman-Kolmogorow test result;"
             " <OUTFILE>_cktest_memberships.pdf containing a plot of the"
             " probability for each active state i to belong to a given"
             " set A;"
             " <OUTFILE>_cktest_assignments.pdf containing a plot of the"
             " PCCA++ assigments;"
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
        '--nsets',
        dest='NSETS',
        type=int,
        required=False,
        default=None,
        help="Number of sets to test on. The test will be done on"
             " metastable (i.e. long-lived) sets which are identified"
             " with PCCA++ (Perron Cluster Cluster Analysis). By default,"
             " the number of sets to use is inferred from the number of"
             " local maxima of the stationary distribution."
    )
    parser.add_argument(
        '--nlags',
        dest='NLAGS',
        type=int,
        required=False,
        default=20,
        help="Highest multiple k of lag time tau up to which to test the"
             " model. Note that for a meaningful Chapman-Kolmogorow test"
             " k*tau should not exceed the second largest implied"
             " timescale. It also cannot exceed the maximum trajectory"
             " length. Default: 20"
    )
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    
    
    
    print("\n\n\n", flush=True)
    print("Loading Markov model", flush=True)
    timer = datetime.now()
    
    mm = pyemma.load(args.INFILE)
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
    
    if not np.all(np.isclose(np.sum(mm.transition_matrix, axis=1), 1)):
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
    print("Calculating metastable sets with PCCA++", flush=True)
    timer = datetime.now()
    
    if args.NSETS is None:
        peaks, _ = find_peaks(
                       mm.stationary_distribution,
                       height=0.66*np.max(mm.stationary_distribution),
                       distance=3)
        args.NSETS = len(peaks)
    if args.NSETS > mm.nstates:
        print("  The number of sets to test on ({}) exceeds the number"
              " of states ({}) in the Markov model"
              .format(args.NSETS, mm.nstates),
              flush=True)
        args.NSETS = mm.nstates
        print("  Set it to {}".format(args.NSETS), flush=True)
    if args.NSETS < 2:
        print("  The number of sets to test on ({}) is less than two"
              .format(args.NSETS, mm.nstates), flush=True)
        args.NSETS = 2
        print("  Set it to {}".format(args.NSETS), flush=True)
    print("  Number of sets to test on: {:>2d}"
          .format(args.NSETS),
          flush=True)
    mm.pcca(args.NSETS)
    if not np.all(np.isclose(np.sum(mm.metastable_memberships, axis=1),
                             1)):
        raise ValueError("The membership functions do not sum up to"
                         " unity for each active state")
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Conducting Chapman-Kolmogorow test", flush=True)
    timer = datetime.now()
    print("  CPUs found: {}".format(n_cpus), flush=True)
    
    if args.NLAGS is not None and args.NLAGS*mm.lag > mm.timescales()[1]:
        print("  Warning: The highest lag time for the test is larger\n"
              "  than the second largest implied timescale. The\n"
              "  Chapman-Kolmogorow test is only meaningful for lag\n"
              "  times lower than the second largest implied\n"
              "  timescale, because the stationary distribution is\n"
              "  almost always reproduced correctly, even with bad\n"
              "   models", flush=True)
    
    print(flush=True)
    ckvalidator = mm.cktest(nsets=args.NSETS,
                            mlags=args.NLAGS,
                            n_jobs=n_cpus,
                            show_progress=True)
    print(flush=True)
    
    if not np.all(np.isclose(np.sum(ckvalidator.memberships, axis=1), 1)):
        raise ValueError("The membership functions do not sum up to"
                         " unity for each active state")
    if not np.all(np.isclose(mm.metastable_memberships,
                             ckvalidator.memberships)):
        raise ValueError("The metastable memberships of the"
                         " pyemma.msm.ChapmanKolmogorovValidator object"
                         " are not the same as the ones of the input"
                         " object")
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()
    
    
    # pyemma.msm.ChapmanKolmogorovValidator object
    mdt.fh.backup(args.OUTFILE+"_cktest.h5")
    ckvalidator.save(file_name=args.OUTFILE+"_cktest.h5")
    print("  Created "+args.OUTFILE+"_cktest.h5", flush=True)
    
    
    # Fuzzy memberships (probability that state i belongs to set A)
    #data = np.column_stack((mm.active_set, ckvalidator.memberships))
    #header = ("Markov state model:\n"
              #+ mm_info + "\n\n"
              #"Fuzzy memberships from PCCA++"
              #" (Perron Cluster Cluster Analysis)\n"
              #"= Probability that state i belongs to set A\n"
              #"Number of sets: {:>2d}\n\n"
              #"The columns contain:\n"
              #"  1 State i\n"
              #"  2-{} Membership function for set {}-{}\n\n"
              #"Column number:\n"
              #.format(ckvalidator.memberships.shape[1],
                      #data.shape[1],
                      #0,
                      #ckvalidator.memberships.shape[1]-1))
    #cols = ' '.join("{:>16d}"
                    #.format(i) for i in range(2, data.shape[1]+1))
    #cols = "{:>14d} ".format(1) + cols
    #header += cols
    #mdt.fh.savetxt(fname=args.OUTFILE+"_cktest_memberships.txt",
                   #data=data,
                   #header=header)
    #print("  Created "+args.OUTFILE+"_cktest_memberships.txt",
          #flush=True)
    
    
    # Crisp/sharp assignments
    data = np.column_stack((mm.active_set, mm.metastable_assignments))
    header = ("Markov state model:\n"
              + mm_info + "\n\n"
              "Crisp/sharp membership assignments from PCCA++"
              " (Perron Cluster Cluster Analysis)\n"
              "= Assignment of state i to set A\n"
              "Number of sets: {:d}\n\n"
              "The columns contain:\n"
              "  1 State i\n"
              "  2 Set A to which state i is assigned\n\n"
              "Column number:\n"
              "{:>14d} {:>16d}"
              .format(ckvalidator.memberships.shape[1], 1, 2))
    mdt.fh.savetxt(fname=args.OUTFILE+"_cktest_assignments.txt",
                   data=data,
                   header=header)
    print("  Created "+args.OUTFILE+"_cktest_assignments.txt",
          flush=True)
    del data, header
    
    
    
    if not args.NOPLOTS:
        fontsize_labels = 36
        fontsize_ticks = 32
        fontsize_legend = 28
        tick_length = 10
        tick_pad = 12
        label_pad = 16
    
        states = np.arange(mm.active_set[-1] + mm.active_set[0] + 1)
    
    
        # Chapman-Kolmogorow test results
        fig, axes = mplt.plot_cktest(cktest=ckvalidator,
                                     figsize=(args.NSETS * 11.69,
                                              args.NSETS * 8.27),
                                     y01=False,
                                     marker='o')
        
        legends = [c for c in fig.get_children()
                   if isinstance(c, matplotlib.legend.Legend)]
        for lgd in legends:
            lgd.remove()
        
        for i in range(len(axes)):
            for j in range(len(axes[i])):
                axes[i][j].texts.remove(axes[i][j].texts[0])
                axes[i][j].set_xlim(xmin=0)
                if axes[i][j].get_ylim()[0] < 0:
                    axes[i][j].set_ylim(ymin=0)
                if axes[i][j].get_ylim()[1] > 1:
                    axes[i][j].set_ylim(ymax=1)
                
                axes[i][j].xaxis.set_tick_params(labelbottom=True)
                axes[i][j].xaxis.offsetText.set_visible(True)
                axes[i][j].xaxis.set_major_locator(
                                     MaxNLocator(integer=True))
                axes[i][j].ticklabel_format(axis='x',
                                            style='scientific',
                                            scilimits=(0,0),
                                            useOffset=False)
                
                axes[i][j].set_xlabel(xlabel=r'Lag time $k\tau$ / steps',
                                      fontsize=fontsize_labels)
                axes[i][j].set_ylabel(
                    ylabel=r'Probability $p_{A \rightarrow B}(k\tau)$',
                    fontsize=fontsize_labels)
                axes[i][j].xaxis.labelpad = label_pad
                axes[i][j].yaxis.labelpad = label_pad
                axes[i][j].xaxis.offsetText.set_fontsize(fontsize_ticks)
                axes[i][j].yaxis.offsetText.set_fontsize(fontsize_ticks)
                axes[i][j].tick_params(which='major',
                                       direction='in',
                                       top=True,
                                       right=True,
                                       length=tick_length,
                                       labelsize=fontsize_ticks,
                                       pad=tick_pad)
                axes[i][j].tick_params(which='minor',
                                       direction='in',
                                       top=True,
                                       right=True,
                                       length=0.5*tick_length,
                                       labelsize=0.8*fontsize_ticks,
                                       pad=tick_pad)
                legend = axes[i][j].legend(
                             labels=['Estimation', 'Prediction'],
                             title=r'${} \rightarrow {}$'.format(i, j),
                             loc='best',
                             numpoints=1,
                             frameon=False,
                             fontsize=fontsize_legend)
                plt.setp(legend.get_title(), fontsize=fontsize_legend)
        
        mdt.fh.backup(args.OUTFILE+"_cktest.pdf")
        plt.tight_layout()
        plt.savefig(args.OUTFILE+"_cktest.pdf")
        plt.close(fig)
        print("  Created "+args.OUTFILE+"_cktest.pdf", flush=True)
        
        
        # Fuzzy memberships (probability that state i belongs to set A)
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        for i in range(len(ckvalidator.memberships.T)):
            mdt.plot.plot(ax=axis,
                          x=mm.active_set,
                          y=ckvalidator.memberships.T[i],
                          xmin=states[0]-0.5,
                          xmax=states[-1]+0.5,
                          ymin=0,
                          xlabel=r'State $i$',
                          ylabel=r'$\chi_A(i)$',
                          label=r'${}$'.format(i))
        
        if axis.get_ylim()[1] > 1:
            axis.set_ylim(ymin=0, ymax=1)
        legend = axis.legend(frameon=False,
                             loc='best',
                             ncol=1+args.NSETS//4,
                             title=r'Set $A$',
                             fontsize=fontsize_legend)
        plt.setp(legend.get_title(), fontsize=fontsize_legend)
        
        mdt.fh.backup(args.OUTFILE+"_cktest_memberships.pdf")
        plt.tight_layout()
        plt.savefig(args.OUTFILE+"_cktest_memberships.pdf")
        plt.close(fig)
        print("  Created "+args.OUTFILE+"_cktest_memberships.pdf",
              flush=True)
        
        
        # Crisp/sharp assignments
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        mdt.plot.plot(ax=axis,
                      x=mm.active_set,
                      y=mm.metastable_assignments,
                      xmin=states[0]-0.5,
                      xmax=states[-1]+0.5,
                      ymin=-0.5,
                      ymax=np.max(mm.metastable_assignments)+0.5,
                      xlabel=r'State $i$',
                      ylabel=r'Set $A$')
        
        mdt.fh.backup(args.OUTFILE+"_cktest_assignments.pdf")
        plt.tight_layout()
        plt.savefig(args.OUTFILE+"_cktest_assignments.pdf")
        plt.close(fig)
        print("  Created "+args.OUTFILE+"_cktest_assignments.pdf",
              flush=True)
    
    
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
