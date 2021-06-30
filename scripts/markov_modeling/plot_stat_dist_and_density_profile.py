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
                     " method) and compare its stationary distribution"
                     " to the corresponding density profile along the"
                     " discretized coordinate directly obtained from"
                     " the MD simulation."
                 )
    )
    
    parser.add_argument(
        '--mm',
        dest='MM',
        type=str,
        required=True,
        help="Input file containing the pyemma.msm.MaximumLikelihoodMSM"
             " or pyemma.msm.BayesianMSM object in HDF5 format as"
             " created by the object's save method."
    )
    parser.add_argument(
        '--md',
        dest='MD',
        type=str,
        required=True,
        help="Text file containing the density profile along the"
             " discretized coordinate directly calculated from the MD"
             " simulation. Lines starting with '#' or '@' are ignored."
    )
    parser.add_argument(
        '--bins',
        dest='BINFILE',
        type=str,
        required=True,
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
        help="Output filename. Plots are optimized for PDF format with"
        " TeX support."
    )
    
    parser.add_argument(
        '--cols',
        dest='COLS',
        type=str,
        nargs="+",
        required=False,
        default=[0, 2],
        help="Space separated list of columns to read from the text file"
             " parsed to --md. Column numbering starts at 0. Exactly two"
             " columns must be given. The first columns contains the"
             " bins used to calculate the density profile from the MD"
             " simulation, the second column contains the density"
             " profile. Default: 0 2"
    )
    parser.add_argument(
        '--conv',
        dest='CONV',
        type=int,
        required=False,
        default=10,
        help="Conversion factor to convert the MD bins to MM bins."
             " Default: 10"
    )
    
    parser.add_argument(
        '--xmin',
        dest='XMIN',
        type=float,
        required=False,
        default=0,
        help="Minimum x-range of plot. Default: 0."
    )
    parser.add_argument(
        '--xmax',
        dest='XMAX',
        type=float,
        required=False,
        default=None,
        help="Maximum x-range of plot. By default detected automatically."
    )
    
    parser.add_argument(
        '--xlabel',
        dest='XLABEL',
        type=str,
        nargs="+",
        required=False,
        default=['$z$', '/', 'A'],
        help="String to use as secondary x-axis label. Note that you"
             " have to use TeX syntax. Default: '$z$ / A' (Note that you"
             " must either leave a space after dollar signs or enclose"
             " the expression in single quotes to avoid bash's variable"
             " expansion)."
    )
    parser.add_argument(
        '--decs',
        dest='DECS',
        type=int,
        required=False,
        default=1,
        help="Number of decimal places for the tick labels of the"
             " secondary x-axis. Default: 1"
    )
    parser.add_argument(
        '--every-n-ticks',
        dest='EVERY_N_TICKS',
        type=int,
        required=False,
        default=2,
        help="Set for every n ticks of the primary x-axis one tick on"
             " the secondary x-axis. Default: 2"
    )
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    
    
    
    print("\n\n\n", flush=True)
    print("Loading Markov model", flush=True)
    timer = datetime.now()
    
    mm = pyemma.load(args.MM)
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
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Loading Markov bins", flush=True)
    timer = datetime.now()
    
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
    
    # Markov states in real space
    bin_widths = np.insert(np.diff(bins), 0, bin_width_first)
    bins -= 0.5 * bin_widths
    
    stat_dist = mm.stationary_distribution
    integral = np.trapz(y=stat_dist, x=bins[mm.active_set])
    stat_dist /= integral
    print("bins.shape                =", bins.shape)
    print("bins[mm.active_set].shape =", bins[mm.active_set].shape)
    print("stat_dist.shape           =", stat_dist.shape)
    print("integral                  =", integral)
    
    XLABEL = ' '.join(args.XLABEL)
    XLABEL = "r'%s'" % XLABEL
    XLABEL = XLABEL[2:-1]
    dim = XLABEL[:XLABEL.find('/')].strip()
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Loading density profile", flush=True)
    timer = datetime.now()
    
    if len(args.COLS) != 2:
        raise ValueError("Exactly two columns must be given, but you"
                         " gave {} ({})"
                         .format(len(args.COLS), args.COLS))
    
    bins_md, density_profile = np.loadtxt(args.MD,
                                          comments=['#', '@'],
                                          usecols=args.COLS,
                                          unpack=True)
    
    bins_md *= args.CONV
    integral = np.trapz(y=density_profile, x=bins_md)
    density_profile /= integral
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating plot", flush=True)
    timer = datetime.now()
    
    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    mdt.plot.plot(ax=axis,
                  x=bins_md,
                  y=density_profile,
                  xmin=args.XMIN,
                  xmax=args.XMAX,
                  ymin=0,
                  xlabel=XLABEL,
                  ylabel=r'Stationary distribution',
                  label=r'MD simulation')
    mdt.plot.plot(ax=axis,
                  x=bins[mm.active_set],
                  y=mm.stationary_distribution,
                  xmin=args.XMIN,
                  xmax=args.XMAX,
                  ymin=0,
                  xlabel=XLABEL,
                  ylabel=r'Stationary distribution',
                  label=r'Markov model')
    axis.set_xlim(xmin=args.XMIN, xmax=args.XMAX)
    axis.set_yticklabels([])
    
    filename = args.OUTFILE
    mdt.fh.backup(filename)
    plt.tight_layout()
    plt.savefig(filename)
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
