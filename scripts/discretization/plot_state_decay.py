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
import mdtools as mdt




if __name__ == '__main__':
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=("Plot the output of state_decay.py.")
    )
    
    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=True,
        help="The output file of state_decay.py."
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
        '--xmin',
        dest='XMIN',
        type=float,
        required=False,
        default=0,
        help="Minimum x-range of the plot. Default: 0"
    )
    parser.add_argument(
        '--xmax',
        dest='XMAX',
        type=float,
        required=False,
        default=None,
        help="Maximum x-range of the plot. Default: Maximum time"
    )
    parser.add_argument(
        '--ymin',
        dest='YMIN',
        type=float,
        required=False,
        default=0,
        help="Minimum y-range of the plot. Default: 0"
    )
    parser.add_argument(
        '--ymax',
        dest='YMAX',
        type=float,
        required=False,
        default=1,
        help="Maximum y-range of the plot. Default: 1"
    )
    
    parser.add_argument(
        '--time-conv',
        dest='TCONV',
        type=float,
        required=False,
        default=1,
        help="Multiply times by this factor. Default: 1, which results"
             " in 'trajectory steps'"
    )
    parser.add_argument(
        '--time-unit',
        dest='TUNIT',
        type=str,
        required=False,
        default="steps",
        help="Time unit. Default: 'steps'"
    )
    
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    
    
    
    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()
    
    times, decay, fit = np.loadtxt(fname=args.INFILE, unpack=True)
    times *= args.TCONV
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating plot", flush=True)
    timer = datetime.now()
    
    if args.XMAX is None:
        args.XMAX = np.max(times)
    
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(ax=axis,
                      x=times,
                      y=decay,
                      xmin=args.XMIN,
                      xmax=args.XMAX,
                      ymin=args.YMIN,
                      ymax=args.YMAX,
                      xlabel=r'$\Delta t$ / '+args.TUNIT,
                      ylabel=r'$p[\xi(t_0 + \Delta t) \in S_i | \xi(t_0) \in S_i]$',
                      #ylabel=r'$p[r_{xy}(t_0 + \Delta t) \in S_i | r_{xy}(t_0) \in S_i]$',
                      linestyle='',
                      marker='x')
        mdt.plot.plot(ax=axis,
                      x=times,
                      y=fit,
                      xmin=args.XMIN,
                      xmax=args.XMAX,
                      ymin=args.YMIN,
                      ymax=args.YMAX,
                      xlabel=r'$\Delta t$ / '+args.TUNIT,
                      ylabel=r'$p[\xi(t_0 + \Delta t) \in S_i | \xi(t_0) \in S_i]$',
                      #ylabel=r'$p[r_{xy}(t_0 + \Delta t) \in S_i | r_{xy}(t_0) \in S_i]$',
                      label="Fit")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mask = (decay > 0)
        mdt.plot.plot(ax=axis,
                      x=times[mask],
                      y=np.log(decay[mask]),
                      xmin=args.XMIN,
                      xmax=args.XMAX,
                      ymin=np.min(np.log(decay[mask])),
                      ymax=0,
                      xlabel=r'$\Delta t$ / '+args.TUNIT,
                      ylabel=r'$\ln{p[\xi(t_0 + \Delta t) \in S_i | \xi(t_0) \in S_i]}$',
                      #ylabel=r'$\ln{p[r_{xy}(t_0 + \Delta t) \in S_i | r_{xy}(t_0) \in S_i]}$',
                      linestyle='',
                      marker='x')
        mask = (fit > 0)
        mdt.plot.plot(ax=axis,
                      x=times[mask],
                      y=np.log(fit[mask]),
                      xmin=args.XMIN,
                      xmax=args.XMAX,
                      ymin=np.min(np.log(fit[mask])),
                      ymax=0,
                      xlabel=r'$\Delta t$ / '+args.TUNIT,
                      ylabel=r'$\ln{p[\xi(t_0 + \Delta t) \in S_i | \xi(t_0) \in S_i]}$',
                      #ylabel=r'$\ln{p[r_{xy}(t_0 + \Delta t) \in S_i | r_{xy}(t_0) \in S_i]}$',
                      label="Fit")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    print("  Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n{} done".format(os.path.basename(sys.argv[0])))
    print("Elapsed time:         {}"
          .format(datetime.now()-timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
