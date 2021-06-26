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
from scipy.special import gamma
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mdtools as mdt




if __name__ == '__main__':
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Plot single columns from the output of"
                     " state_decay_discrete.py."
                     )
    )
    
    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=True,
        help="The output file of state_decay_discrete.py."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename pattern. There are created two or three"
             " output files:"
             " <OUTFILE>.pdf containing the decay functions as function"
             " of lag time and state as well as the lifetimes as"
             " function of state;"
             " <OUTFILE>_fit.pdf containing the decay function for each"
             " state as function of lag time with its respective fit;"
             " <OUTFILE>_fit.txt containing the fit parameters for each"
             " secondary state. Only created if --refit is given."
             " Plots are optimized for PDF format with TeX support."
    )
    parser.add_argument(
        '-c',
        dest='COLS',
        type=int,
        nargs='+',
        required=False,
        default=None,
        help="Space separated list of columns to plot. Default is to"
             " plot all columns. Column 0 contains the lag times and is"
             " therefore always selected automatically."
    )
    
    parser.add_argument(
        '--refit',
        dest='REFIT',
        required=False,
        default=False,
        action='store_true',
        help="Do not read the fitting parameters from the input file but"
             " redo the fitting of the data."
    )
    parser.add_argument(
        '--end-fit',
        dest='ENDFIT',
        type=float,
        required=False,
        default=None,
        help="End time for fitting the decay function (in the unit given"
             " by --time-unit). Inclusive, i.e. the time given here is"
             " still included in the fit. Is meaningless if --refit is"
             " not given. Default: End at 90%% of the lag times."
    )
    parser.add_argument(
        '--stop-fit',
        dest='STOPFIT',
        type=float,
        required=False,
        default=0.01,
        help="Stop fitting the decay function as soon as it falls below"
             " this value. The fitting is stopped by whatever happens"
             " earlier: --end-fit or --stop-fit. Is meaningless if"
             " --refit is not given. Default: 0.01"
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
    
    if args.COLS is not None:
        args.COLS = np.unique(args.COLS)
        if 0 not in args.COLS:
            args.COLS = np.insert(args.COLS, 0, 0)
        if len(args.COLS) < 2:
            raise ValueError("You must give at least one column"
                             " different from zero with -c")
    decay = np.loadtxt(fname=args.INFILE, usecols=args.COLS)
    times = decay[1:,0] * args.TCONV
    states = decay[0,1:]
    decay = decay[1:,1:]
    
    if np.any(decay > 1):
        raise ValueError("At least one element of decay is greater than"
                         " one. This means your input is not a proper"
                         " decay function")
    if np.any(decay < 0):
        raise ValueError("At least one element of decay is less than"
                         " zero. This means your input is not a proper"
                         " decay function")
    
    fractional_part = np.modf(states)[0]
    if np.all(fractional_part == 0):
        states = states.astype(np.int32)
    del fractional_part
    
    if not args.REFIT:
        if args.COLS is None:
            args.COLS = np.arange(len(states)+1)
        with open(args.INFILE, 'r') as f:
            header = []
            for line in f:
                if line[0] == '#':
                    header.append(line[1:])
                else:
                    break
        k = header[-9]
        k = k.split()
        k = np.array(k[3:], dtype=np.float32)
        k = k[args.COLS[1:]-1]
        k /= args.TCONV
        fit_start = header[-3]
        fit_start = fit_start.split()
        fit_start = np.array(fit_start[2:], dtype=np.uint32)
        fit_start = fit_start[args.COLS[1:]-1]
        fit_stop = header[-2]
        fit_stop = fit_stop.split()
        fit_stop = np.array(fit_stop[2:], dtype=np.uint32)
        fit_stop = fit_stop[args.COLS[1:]-1]
        del header
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    if args.REFIT:
        print("\n\n\n", flush=True)
        print("Re-fitting decay functions", flush=True)
        timer = datetime.now()
        
        if args.ENDFIT is None:
            endfit = int(0.9 * len(times))
        else:
            _, endfit = mdt.nph.find_nearest(times,
                                             args.ENDFIT,
                                             return_index=True)
        
        fit_start = np.zeros(len(states), dtype=np.uint32)
        fit_stop = np.zeros(len(states), dtype=np.uint32)
        popt = np.full((len(states), 2), np.nan, dtype=np.float32)
        perr = np.full((len(states), 2), np.nan, dtype=np.float32)
        for i in range(len(states)):
            stopfit = np.argmax(decay[:,i] < args.STOPFIT)
            if stopfit == 0 and decay[:,i][stopfit] >= args.STOPFIT:
                stopfit = len(decay[:,i])
            elif stopfit < 2:
                stopfit = 2
            valid = np.isfinite(decay[:,i]) & (decay[:,i] > 0)
            valid[min(endfit+1, stopfit):] = False
            fit_start[i] = np.argmax(valid)
            fit_stop[i] = len(valid)-1 - np.argmax(valid[::-1])
            if np.any(valid):
                popt[i], perr[i] = mdt.func.fit_kww(
                                       xdata=times[valid],
                                       ydata=decay[:,i][valid])
        tau_mean = popt[:,0]/popt[:,1] * gamma(1/popt[:,1])
        
        print("Elapsed time:         {}"
              .format(datetime.now()-timer),
              flush=True)
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()
    
    if args.REFIT:
        header = (
            "Refitting of the decay functions from the input file.\n"
            "Fit function:\n"
            "  f(t) = exp[-(t/tau)^beta]\n"
            "Mean relaxation time:\n"
            "  <tau> = integral_0^infty exp[-(t/tau)^beta] dt\n"
            "        = tau/beta * Gamma(1/beta)\n"
            "  Gamma(x) = Gamma function\n"
            "  If beta=1, <tau>=tau\n"
            "\n"
            "\n"
            "The columns contain:\n"
            "  1 Secondary state\n"
            "  2 Start fit                  (in {unit})\n"
            "  3 Stop fit                   (in {unit})\n"
            "  4 Fit parameter tau          (in {unit})\n"
            "  5 Standard deviation of tau  (in {unit})\n"
            "  6 Fit parameter beta\n"
            "  7 Standard deviation of beta\n"
            "  8 Mean relaxation time <tau> (in {unit})\n"
            "\n"
            "Column number:\n"
            "{:>14d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d}\n"
            .format(1, 2, 3, 4, 5, 6, 7, 8, unit=args.TUNIT)
        )
        data = np.column_stack([states,
                                fit_start*args.TCONV,
                                fit_stop*args.TCONV,
                                popt[:,0], perr[:,0],
                                popt[:,1], perr[:,1],
                                tau_mean])
        outfile = args.OUTFILE + "_fit.txt"
        mdt.fh.savetxt(fname=outfile, data=data, header=header)
        print("  Created {}".format(outfile))
    
    
    
    
    fontsize_legend = 24
    
    if args.XMAX is None:
        args.XMAX = np.max(times)
    # Line style "cycler"
    ls = ['-', '--', '-.', ':']
    ls *= (1 + len(states)//len(ls))
    
    
    outfile = args.OUTFILE + ".pdf"
    mdt.fh.backup(outfile)
    with PdfPages(outfile) as pdf:
        # Decay vs lag time
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        for i, s in enumerate(states):
            mdt.plot.plot(
                ax=axis,
                x=times,
                y=decay[:,i],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                xlabel=r'$\Delta t$ / '+args.TUNIT,
                ylabel=r'$p[\xi(t_0 + \Delta t) \in S_i | \xi(t_0) \in S_i, S_j^\prime]$',
                #ylabel=r'$p[z(t_0 + \Delta t) \in S_i | z(t_0) \in S_i]$',
                label=str(s),
                linestyle=ls[i])
        axis.legend(loc='best',
                    title=r'$S_j^\prime$',
                    #title=r'$S_i$',
                    title_fontsize=fontsize_legend,
                    fontsize=fontsize_legend,
                    numpoints=1,
                    ncol=1 + len(states)//9,
                    labelspacing = 0.2,
                    columnspacing = 1.4,
                    handletextpad = 0.5,
                    frameon=False)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        # ln(decay) vs lag time
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        ymin = args.YMIN if args.YMIN < 0 else np.min(np.log(decay[decay>0]))
        ymax = args.YMAX if args.YMAX <= 0 else 0
        for i, s in enumerate(states):
            mask = (decay[:,i] > 0)
            mdt.plot.plot(
                ax=axis,
                x=times[mask],
                y=np.log(decay[:,i][mask]),
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=ymin,
                ymax=ymax,
                xlabel=r'$\Delta t$ / '+args.TUNIT,
                ylabel=r'$\ln{p[\xi(t_0 + \Delta t) \in S_i | \xi(t_0) \in S_i, S_j^\prime]}$',
                #ylabel=r'$\ln{p[z(t_0 + \Delta t) \in S_i | z(t_0) \in S_i]}$',
                label=str(s),
                linestyle=ls[i])
        axis.legend(loc='best',
                    title=r'$S_j^\prime$',
                    #title=r'$S_i$',
                    title_fontsize=fontsize_legend,
                    fontsize=fontsize_legend,
                    numpoints=1,
                    ncol=1 + len(states)//9,
                    labelspacing = 0.2,
                    columnspacing = 1.4,
                    handletextpad = 0.5,
                    frameon=False)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        # Decay vs log(lag time)
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        xmin = args.XMIN if args.XMIN > 0 else np.min(times[times>0])
        xmax = args.XMAX if args.XMAX > 0 else np.max(times[times>0])
        for i, s in enumerate(states):
            mdt.plot.plot(
                ax=axis,
                x=times,
                y=decay[:,i],
                xmin=xmin,
                xmax=xmax,
                ymin=args.YMIN,
                ymax=args.YMAX,
                logx=True,
                xlabel=r'$\Delta t$ / '+args.TUNIT,
                ylabel=r'$p[\xi(t_0 + \Delta t) \in S_i | \xi(t_0) \in S_i, S_j^\prime]$',
                #ylabel=r'$p[z(t_0 + \Delta t) \in S_i | z(t_0) \in S_i]$',
                label=str(s),
                linestyle=ls[i])
        axis.legend(loc='best',
                    title=r'$S_j^\prime$',
                    #title=r'$S_i$',
                    title_fontsize=fontsize_legend,
                    fontsize=fontsize_legend,
                    numpoints=1,
                    ncol=1 + len(states)//9,
                    labelspacing = 0.2,
                    columnspacing = 1.4,
                    handletextpad = 0.5,
                    frameon=False)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        # Mean relaxation time vs secondary state
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(
            ax=axis,
            x=states,
            y=tau_mean,
            ymin=0,
            xlabel=r'$S_j^\prime$',
            #xlabel=r'$S_i$',
            ylabel=r'Mean relaxation time $\langle \tau \rangle$ / '+args.TUNIT,
            marker='o')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    print("  Created {}".format(outfile))
    
    
    
    
    outfile = args.OUTFILE + "_fit.pdf"
    mdt.fh.backup(outfile)
    with PdfPages(outfile) as pdf:
        for i, s in enumerate(states):
            # ln(decay) vs lag time with fit
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            mask = (decay[:,i] > 0)
            mdt.plot.plot(
                ax=axis,
                x=times[mask],
                y=np.log(decay[:,i][mask]),
                xmin=args.XMIN,
                ymax=0,
                xlabel=r'$\Delta t$ / '+args.TUNIT,
                ylabel=r'$\ln{p[\xi(t_0 + \Delta t) \in S_i | \xi(t_0) \in S_i, S_j^\prime]}$',
                #ylabel=r'$\ln{p[z(t_0 + \Delta t) \in S_i | z(t_0) \in S_i]}$',
                label=r'$S_j^\prime =' + str(s) + r'$')
                #label=r'$S_i =' + str(s) + r'$')
            fit = mdt.func.kww(t=times, tau=popt[i,0], beta=popt[i,1])
            mask = np.zeros_like(fit, dtype=bool)
            mask[fit_start[i]:fit_stop[i]] = True
            mask &= (fit > 0)
            mdt.plot.plot(
                ax=axis,
                x=times[mask],
                y=np.log(fit[mask]),
                xmin=args.XMIN,
                ymax=0,
                xlabel=r'$\Delta t$ / '+args.TUNIT,
                ylabel=r'$\ln{p[\xi(t_0 + \Delta t) \in S_i | \xi(t_0) \in S_i, S_j^\prime]}$',
                #ylabel=r'$\ln{p[z(t_0 + \Delta t) \in S_i | z(t_0) \in S_i]}$',
                linestyle='--',
                label="Fit")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print("  Created {}".format(outfile))
    
    
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
