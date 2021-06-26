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
import matplotlib.pyplot as plt
import pyemma
import pyemma.plots as mplt
import mdtools as mdt




if __name__ == "__main__":
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read a pyemma.msm.ImpliedTimescales object from"
                     " file (which must have been created by the"
                     " object's save method) and plot the implied"
                     " timescales."
                 )
    )
    
    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=True,
        help="Input file containing the pyemma.msm.ImpliedTimescales"
             " object in HDF5 format as created by the object's save"
             " method."
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
        '--nits',
        dest='NITS',
        type=int,
        required=False,
        default=10,
        help="Number of implied timescales to plot. Parse -1 to plot all"
             " timescales. Default: 10"
    )
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    
    
    
    print("\n\n\n", flush=True)
    print("Loading pyemma.msm.ImpliedTimescales object", flush=True)
    timer = datetime.now()
    its = pyemma.load(args.INFILE)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating plot", flush=True)
    timer = datetime.now()
    
    fontsize_labels = 36
    fontsize_ticks = 32
    tick_length = 10
    tick_pad = 12
    label_pad = 16
    
    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    axis.set_xlabel(xlabel='Lag time / steps', fontsize=fontsize_labels)
    axis.set_ylabel(ylabel='Timescale / steps', fontsize=fontsize_labels)
    axis.xaxis.labelpad = label_pad
    axis.yaxis.labelpad = label_pad
    axis.xaxis.offsetText.set_fontsize(fontsize_ticks)
    axis.yaxis.offsetText.set_fontsize(fontsize_ticks)
    axis.tick_params(which='major',
                     direction='in',
                     top=True,
                     right=True,
                     length=tick_length,
                     labelsize=fontsize_ticks,
                     pad=tick_pad)
    axis.tick_params(which='minor',
                     direction='in',
                     top=True,
                     right=True,
                     length=0.5*tick_length,
                     labelsize=0.8*fontsize_ticks,
                     pad=tick_pad)
    
    mdt.fh.backup(args.OUTFILE)
    mplt.plot_implied_timescales(ITS=its,
                                 ax=axis,
                                 outfile=args.OUTFILE,
                                 nits=args.NITS)
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
