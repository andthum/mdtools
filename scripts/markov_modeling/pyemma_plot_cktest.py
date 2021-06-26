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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pyemma
import pyemma.plots as mplt
import mdtools as mdt




if __name__ == "__main__":
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read a pyemma.msm.ChapmanKolmogorovValidator"
                     " object from file (which must have been created by"
                     " the object's save method) and visualize the"
                     " result of the Chapman-Kolmogorov test."
                 )
    )
    
    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=True,
        help="Input file containing the"
             " pyemma.msm.ChapmanKolmogorovValidator object in HDF5"
             " format as created by the object's save method."
    )
    parser.add_argument(
        '--f2',
        dest='INFILE2',
        type=str,
        required=False,
        default=None,
        help="Text file containing the assigment of each active state to"
             " a metastable set from PCCA++ (Perron Cluster Cluster"
             " Analysis). The first column must contain the active"
             " states and the second column their assigments to the PCCA"
             " sets. Lines starting with '#' are ignored. If provided,"
             " these assigments and also the fuzzy  memberships (i.e."
             " the probability that state i belongs to set A) are"
             " plotted."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename pattern. There will be created three files:"
             " <OUTFILE>_cktest.pdf containing a visualization of the"
             " Chapman-Kolmogorow test result;"
             " <OUTFILE>_cktest_memberships.pdf containing a plot of the"
             " probability for each active state i to belong to a given"
             " set A;"
             " <OUTFILE>_cktest_assignments.pdf containing a plot of the"
             " PCCA++ assigments of states to sets;"
             " Plots are optimized for PDF format with TeX support."
    )
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    
    
    
    print("\n\n\n", flush=True)
    print("Loading input file(s)",
          flush=True)
    timer = datetime.now()
    
    ckvalidator = pyemma.load(args.INFILE)
    nsets = ckvalidator.memberships.shape[1]
    if args.INFILE2 is not None:
        active_set, pcca_assignments = np.genfromtxt(args.INFILE2,
                                                     usecols=(0, 1),
                                                     unpack=True)
        states = np.arange(active_set[-1] + active_set[0] + 1)
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating plots", flush=True)
    timer = datetime.now()
    
    
    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16
    
    
    # Chapman-Kolmogorow test results
    fig, axes = mplt.plot_cktest(cktest=ckvalidator,
                                 figsize=(nsets*11.69, nsets*8.27),
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
            axes[i][j].xaxis.set_major_locator(MaxNLocator(integer=True))
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
    
    
    if args.INFILE2 is not None:
        # Fuzzy memberships (probability that state i belongs to set A)
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        for i in range(len(ckvalidator.memberships.T)):
            mdt.plot.plot(ax=axis,
                          x=active_set,
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
                             ncol=1+nsets//4,
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
                      x=active_set,
                      y=pcca_assignments,
                      xmin=states[0]-0.5,
                      xmax=states[-1]+0.5,
                      ymin=-0.5,
                      ymax=np.max(pcca_assignments)+0.5,
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
