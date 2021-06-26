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
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mdtools as mdt




if __name__ == '__main__':
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read a trajectory of renewal events as e.g."
                     " generated with extract_renewal_events.py and plot"
                     " the renewal time histogram (renewal time"
                     " distribution). The histogram is fitted by an"
                     " exponential distribution"
                     )
    )
    
    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=True,
        help="Trajectory of renewal events as e.g. generated with"
             " extract_renewal_events.py."
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
        '--bin-start',
        dest='START',
        type=float,
        required=False,
        default=0,
        help="Time to start binning. Default: 0"
    )
    parser.add_argument(
        '--bin-end',
        dest='STOP',
        type=float,
        required=False,
        default=None,
        help="Time to end binning. Default: Maximum time"
    )
    parser.add_argument(
        '--bin-num',
        dest='NUM',
        type=int,
        required=False,
        default=50,
        help="Number of bins to use. Default: 50"
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
        help="Maximum x-range of the plot. By default detected"
             " automatically."
    )
    parser.add_argument(
        '--ymin',
        dest='YMIN',
        type=float,
        required=False,
        default=0,
        help="Minimum y-range of the plot. Default:0"
    )
    parser.add_argument(
        '--ymax',
        dest='YMAX',
        type=float,
        required=False,
        default=None,
        help="Maximum y-range of the plot. By default detected"
             " automatically."
    )
    
    parser.add_argument(
        '--time-conv',
        dest='TCONV',
        type=float,
        required=False,
        default=1,
        help="Multiply times by this factor. Default: 1, which results"
             " in ps"
    )
    parser.add_argument(
        '--time-unit',
        dest='TUNIT',
        type=str,
        required=False,
        default="ps",
        help="Time unit. Default: ps"
    )
    
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    
    
    
    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()
    
    trenew = np.loadtxt(fname=args.INFILE, usecols=3)
    trenew *= args.TCONV
    
    if args.STOP is None:
        args.STOP = np.max(trenew)
    hist, bins = np.histogram(trenew,
                              bins=args.NUM,
                              range=(args.START, args.STOP),
                              density=True)
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Fitting histogram", flush=True)
    timer = datetime.now()
    
    x = bins[1:] - np.diff(bins)/2
    valid = (hist > 0)
    if not np.any(valid):
        raise ValueError("All histogram elements are zero")
    popt, pcov = curve_fit(f=mdt.stats.exp_dist_log,
                           xdata=x[valid],
                           ydata=np.log(hist[valid]),
                           p0=1/np.mean(hist),
                           bounds=(0, np.inf))
    popt = popt[0]
    pcov = np.diag(pcov)[0]
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating plot", flush=True)
    timer = datetime.now()
    
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        # Renewal time histogram, counts
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.hist(ax=axis,
                      x=trenew,
                      xmin=args.XMIN,
                      xmax=args.XMAX,
                      ymin=args.YMIN,
                      ymax=args.YMAX,
                      xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
                      ylabel="Counts",
                      bins=bins,
                      range=(args.START, args.STOP),
                      density=False)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        # Renewal time histogram, density
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.hist(ax=axis,
                      x=trenew,
                      xmin=args.XMIN,
                      xmax=args.XMAX,
                      ymin=args.YMIN,
                      ymax=args.YMAX,
                      xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
                      ylabel=r'$p(\tau_{renew})$',
                      bins=bins,
                      range=(args.START, args.STOP),
                      density=True)
        mdt.plot.plot(ax=axis,
                      x=x,
                      y=mdt.stats.exp_dist(x=x, rate=popt),
                      xmin=args.XMIN,
                      xmax=args.XMAX,
                      ymin=args.YMIN,
                      ymax=args.YMAX,
                      xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
                      ylabel=r'$p(\tau_{renew})$',
                      label="Fit",
                      color='black')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        # Statistics
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.axis('off')
        fontsize = 26
        
        xpos = 0.05
        ypos = 0.95
        plt.text(x=xpos, y=ypos, s="Statistics:", fontsize=fontsize)
        xpos += 0.10
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s="Mean / "+args.TUNIT,
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s="Std. dev. / "+args.TUNIT,
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos, y=ypos, s="Tot. counts", fontsize=fontsize)
        
        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos, y=ypos, s=r'$\tau_{renew}$', fontsize=fontsize)
        xpos += 0.10
        plt.text(x=xpos,
                 y=ypos, 
                 s=r'${:>16.9e}$'.format(np.mean(trenew)),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(np.std(trenew)),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16d}$'.format(len(trenew)),
                 fontsize=fontsize)
        
        
        # Histogram parameters
        xpos = 0.05
        ypos -= 0.10
        plt.text(x=xpos,
                 y=ypos,
                 s="Histogram parameters:",
                 fontsize=fontsize)
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s="First bin edge / "+args.TUNIT,
                 fontsize=fontsize)
        xpos += 0.40
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(args.START),
                 fontsize=fontsize)
        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s="Last bin edge / "+args.TUNIT,
                 fontsize=fontsize)
        xpos += 0.40
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(args.STOP),
                 fontsize=fontsize)
        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos, y=ypos, s=r'$N$ bins', fontsize=fontsize)
        xpos += 0.40
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16d}$'.format(args.NUM),
                 fontsize=fontsize)
        
        
        # Fit parameters
        xpos = 0.05
        ypos -= 0.10
        plt.text(x=xpos, y=ypos, s="Fit parameters:", fontsize=fontsize)
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Fit function: $p(t) = \lambda e^{-\lambda t}$',
                 fontsize=fontsize)
        xpos += 0.10
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Opt. param. / '+args.TUNIT+r'$^{-1}$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Std. dev. / '+args.TUNIT+r'$^{-1}$',
                 fontsize=fontsize)
        
        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos, y=ypos, s=r'$\lambda$', fontsize=fontsize)
        xpos += 0.10
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(popt),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(np.sqrt(pcov)),
                 fontsize=fontsize)
        
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
