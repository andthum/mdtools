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
                     " the mean square displacement (MSD) versus the"
                     " renewal time as scatter plot. Additionally, a"
                     " bin-wise average and variance are computed and"
                     " plotted. Furthermore, the non-Gaussian parameter"
                     " is calculated for each time bin."
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
        '--sel',
        dest='SEL',
        required=False,
        default=False,
        action='store_true',
        help="Use the selection compounds instead of the reference"
             " compounds."
    )
    parser.add_argument(
        '-dcolor',
        dest='DCOLOR',
        type=str,
        required=False,
        default=None,
        help="The scatter points can be colored according to the initial"
             " position of the compounds. Must be either x, y or z."
             " Default: No coloring"
    )
    
    parser.add_argument(
        '--bin-start',
        dest='START',
        type=float,
        required=False,
        default=0,
        help="Time to start binning the renewal times. Default: 0"
    )
    parser.add_argument(
        '--bin-end',
        dest='STOP',
        type=float,
        required=False,
        default=None,
        help="Time to end binning the renewal times. Default: Maximum"
             " renewal time"
    )
    parser.add_argument(
        '--bin-num',
        dest='NUM',
        type=int,
        required=False,
        default=50,
        help="Number of bins to use for binning the renewal times."
             " Default: 50"
    )
    parser.add_argument(
        '--bins',
        dest='BINFILE',
        type=str,
        required=False,
        default=None,
        help="ASCII formatted text file containing custom bin edges. Bin"
             " edges are read from the first column, lines starting with"
             " '#' are ignored. Bins do not need to be equidistant."
    )
    
    parser.add_argument(
        '--xmin',
        dest='XMIN',
        type=float,
        required=False,
        default=None,
        help="Minimum x-range of the plot. By default detected"
             " automatically."
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
        default=None,
        help="Minimum y-range of the plot. By default detected"
             " automatically."
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
    parser.add_argument(
        '--length-conv',
        dest='LCONV',
        type=float,
        required=False,
        default=1,
        help="Multiply lengths by this factor. Default: 1, which results"
             " in Angstroms"
    )
    parser.add_argument(
        '--length-unit',
        dest='LUNIT',
        type=str,
        required=False,
        default="A",
        help="Lengh unit. Default: A"
    )
    
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    
    if (args.DCOLOR is not None and
        args.DCOLOR != 'x' and
        args.DCOLOR != 'y' and
        args.DCOLOR != 'z'):
        raise ValueError("--dcolor must be either 'x', 'y' or 'z', but"
                         " you gave {}".format(args.DCOLOR))
    dim = {'x': 0, 'y': 1, 'z': 2}
    
    
    
    
    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()
    
    trenew = np.loadtxt(fname=args.INFILE, usecols=3)
    trenew *= args.TCONV
    if args.SEL:
        cols = (13, 14, 15)
    else:
        cols = (10, 11, 12)
    msd = np.loadtxt(fname=args.INFILE, usecols=cols)
    msd *= args.LCONV
    msd *= msd
    msd_tot = np.sum(msd, axis=1)
    if args.DCOLOR is not None:
        if args.SEL:
            cols = 7+dim[args.DCOLOR]
        else:
            cols = 4+dim[args.DCOLOR]
        pos_t0 = np.loadtxt(fname=args.INFILE, usecols=cols)
        pos_t0 *= args.LCONV
    
    if args.BINFILE is None:
        if args.START is None or args.START > np.min(trenew):
            args.START = np.min(trenew)
        if args.STOP is None or args.STOP < np.max(trenew):
            args.STOP = np.max(trenew)
        tbins = np.linspace(args.START, args.STOP, args.NUM)
    else:
        tbins = np.loadtxt(args.BINFILE, usecols=0)
        tbins = np.unique(tbins)
        if len(tbins) == 0:
            raise ValueError("Invalid tbins")
        if tbins[0] > np.min(trenew):
            tbins = np.insert(tbins, 0, np.min(trenew))
        if tbins[-1] < np.max(trenew):
            tbins = np.append(tbins, np.max(trenew))
    t = tbins[1:] - np.diff(tbins)/2
    
    tbin_ix = np.digitize(trenew, tbins)
    # In np.histogram the last bin is closed, but in np.digitize all
    # bins are half-open. Make the last bin closed:
    tbin_ix[tbin_ix==len(tbins)] = len(tbins) - 1
    if np.any(tbin_ix == 0):
        raise ValueError("At least one element of tbin_ix is zero. This"
                         " should not have happened.")
    nevents = np.full(len(tbins), np.nan)
    msd_mean = np.full((len(tbins), msd.shape[1]), np.nan)
    msd_std = np.full((len(tbins), msd.shape[1]), np.nan)
    msd_tot_mean = np.full(len(tbins), np.nan)
    msd_tot_std = np.full(len(tbins), np.nan)
    msd_non_gaus = np.full((len(tbins), msd.shape[1]), np.nan)
    msd_tot_non_gaus = np.full(len(tbins), np.nan)
    for i in np.unique(tbin_ix):
        mask = (tbin_ix == i)
        nevents[i] = np.count_nonzero(mask)
        if nevents[i] > 0:
            msd_mean[i] = np.mean(msd[mask], axis=0)
            msd_std[i] = np.std(msd[mask], axis=0) / np.sqrt(nevents[i])
            msd_tot_mean[i] = np.mean(msd_tot[mask])
            msd_tot_std[i] = np.std(msd_tot[mask]) / np.sqrt(nevents[i])
            msd_non_gaus[i] = mdt.stats.non_gaussian_parameter(
                                  msd[mask],
                                  d=1,
                                  is_squared=True,
                                  axis=0)
            msd_tot_non_gaus[i] = mdt.stats.non_gaussian_parameter(
                                      msd_tot[mask],
                                      d=3,
                                      is_squared=True,
                                      axis=0)
    if not np.isnan(nevents[0]):
        raise ValueError("The first element of nevents is not NaN. This"
                         " should not have happened")
    if not np.all(np.isnan(msd_mean[0])):
        raise ValueError("Not all first elements of msd_mean are NaN."
                         " This should not have happened")
    if not np.isnan(msd_tot_mean[0]):
        raise ValueError("The first element of msd_tot_mean is not NaN."
                         " This should not have happened")
    if not np.all(np.isnan(msd_non_gaus[0])):
        raise ValueError("Not all first elements of msd_non_gaus are NaN."
                         " This should not have happened")
    if not np.isnan(msd_tot_non_gaus[0]):
        raise ValueError("The first element of msd_tot_non_gaus is not"
                         " NaN. This should not have happened")
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Fitting MSDs", flush=True)
    timer = datetime.now()
    
    displ0 = np.zeros(4, dtype=np.uint32)
    popt_msd = np.zeros(4)
    pcov_msd = np.zeros(4)
    for i, data in enumerate(msd.T):
        popt_msd[i], pcov_msd[i] = curve_fit(
                                       f=lambda t, D: mdt.dyn.msd(t=t, D=D, d=1),
                                       xdata=trenew,
                                       ydata=data)
    popt_msd[-1], pcov_msd[-1] = curve_fit(
                                     f=lambda t, D: mdt.dyn.msd(t=t, D=D, d=3),
                                     xdata=trenew,
                                     ydata=msd_tot)
    
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
    fontsize_legend = 28
    tick_length = 10
    label_pad = 16
    
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        # Number of renewal events per time bins
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(ax=axis,
                      x=tbins[1:]-np.diff(tbins)/2,
                      y=nevents[1:],
                      xmin=0,
                      xmax=args.XMAX,
                      xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
                      ylabel=r'$N_{renew}$',
                      color='black',
                      marker='o')
        mdt.plot.vlines(ax=axis,
                        x=tbins,
                        start=axis.get_ylim()[0],
                        stop=axis.get_ylim()[1],
                        xmin=args.XMIN,
                        xmax=args.XMAX,
                        color='black',
                        linestyle='dotted')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        
        
        # MSDs vs renewal time
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        displ0[-1] = np.count_nonzero(msd_tot == 0)
        mask = (msd_tot > 0)
        if args.DCOLOR is None:
            mdt.plot.scatter(
                ax=axis,
                x=trenew[mask],
                y=msd_tot[mask],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                logx=True,
                logy=True,
                xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
                ylabel=r'$\Delta r^2(\tau_{renew})$ / '+args.LUNIT+r'$^2$',
                marker='x')
        else:
            img = mdt.plot.scatter(
                      ax=axis,
                      x=trenew[mask],
                      y=msd_tot[mask],
                      c=pos_t0[mask],
                      xmin=args.XMIN,
                      xmax=args.XMAX,
                      ymin=args.YMIN,
                      ymax=args.YMAX,
                      logx=True,
                      logy=True,
                      xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
                      ylabel=r'$\Delta r^2(\tau_{renew})$ / '+args.LUNIT+r'$^2$',
                      marker='x',
                      cmap='plasma')
            cbar = plt.colorbar(img, ax=axis)
            cbar.set_label(label=r'${}(t_0)$ / {}'.format(args.DCOLOR, args.LUNIT),
                           fontsize=fontsize_labels)
            cbar.ax.yaxis.labelpad = label_pad
            cbar.ax.yaxis.offsetText.set(size=fontsize_ticks)
            cbar.ax.tick_params(which='major',
                                direction='out',
                                length=tick_length,
                                labelsize=fontsize_ticks)
            cbar.ax.tick_params(which='minor',
                                direction='out',
                                length=0.5*tick_length,
                                labelsize=0.8*fontsize_ticks)
        mdt.plot.vlines(ax=axis,
                        x=tbins,
                        start=axis.get_ylim()[0],
                        stop=axis.get_ylim()[1],
                        xmin=args.XMIN,
                        xmax=args.XMAX,
                        ymin=args.YMIN,
                        ymax=args.YMAX,
                        color='black',
                        linestyle='dotted')
        mask = ~(msd_tot_mean[1:] < 0)
        mdt.plot.errorbar(
            ax=axis,
            x=t[mask],
            y=msd_tot_mean[1:][mask],
            yerr=msd_tot_std[1:][mask],
            xmin=args.XMIN,
            xmax=args.XMAX,
            ymin=args.YMIN,
            ymax=args.YMAX,
            logx=True,
            logy=True,
            xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
            ylabel=r'$\Delta r^2(\tau_{renew})$ / '+args.LUNIT+r'$^2$',
            label=r'$\langle \Delta r^2 \rangle$',
            color='red',
            marker='o')
        fit = mdt.dyn.msd(t=trenew, D=popt_msd[-1], d=3)
        mask = (fit > 0)
        mdt.plot.plot(
            ax=axis,
            x=trenew[mask],
            y=fit[mask],
            xmin=args.XMIN,
            xmax=args.XMAX,
            ymin=args.YMIN,
            ymax=args.YMAX,
            logx=True,
            logy=True,
            xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
            ylabel=r'$\Delta r^2(\tau_{renew})$ / '+args.LUNIT+r'$^2$',
            label="Fit",
            color='black')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        ylabel = ('x', 'y', 'z')
        for i, data in enumerate(msd.T):
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            displ0[i] = np.count_nonzero(data == 0)
            mask = (data > 0)
            if args.DCOLOR is None:
                mdt.plot.scatter(
                    ax=axis,
                    x=trenew[mask],
                    y=data[mask],
                    xmin=args.XMIN,
                    xmax=args.XMAX,
                    ymin=args.YMIN,
                    ymax=args.YMAX,
                    logx=True,
                    logy=True,
                    xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
                    ylabel=r'$\Delta '+ylabel[i]+r'^2(\tau_{renew})$ / '+args.LUNIT+r'$^2$',
                    marker='x')
            else:
                img = mdt.plot.scatter(
                          ax=axis,
                          x=trenew[mask],
                          y=data[mask],
                          c=pos_t0[mask],
                          xmin=args.XMIN,
                          xmax=args.XMAX,
                          ymin=args.YMIN,
                          ymax=args.YMAX,
                          logx=True,
                          logy=True,
                          xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
                          ylabel=r'$\Delta '+ylabel[i]+r'^2(\tau_{renew})$ / '+args.LUNIT+r'$^2$',
                          marker='x',
                          cmap='plasma')
                cbar = plt.colorbar(img, ax=axis)
                cbar.set_label(label=r'${}(t_0)$ / {}'.format(args.DCOLOR, args.LUNIT),
                               fontsize=fontsize_labels)
                cbar.ax.yaxis.labelpad = label_pad
                cbar.ax.yaxis.offsetText.set(size=fontsize_ticks)
                cbar.ax.tick_params(which='major',
                                    direction='out',
                                    length=tick_length,
                                    labelsize=fontsize_ticks)
                cbar.ax.tick_params(which='minor',
                                    direction='out',
                                    length=0.5*tick_length,
                                    labelsize=0.8*fontsize_ticks)
            mdt.plot.vlines(ax=axis,
                            x=tbins,
                            start=axis.get_ylim()[0],
                            stop=axis.get_ylim()[1],
                            xmin=args.XMIN,
                            xmax=args.XMAX,
                            ymin=args.YMIN,
                            ymax=args.YMAX,
                            color='black',
                            linestyle='dotted')
            mask = ~(msd_mean.T[i][1:] < 0)
            mdt.plot.errorbar(
                ax=axis,
                x=t[mask],
                y=msd_mean.T[i][1:][mask],
                yerr=msd_std.T[i][1:][mask],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                logx=True,
                logy=True,
                xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
                ylabel=r'$\Delta '+ylabel[i]+r'^2(\tau_{renew})$ / '+args.LUNIT+r'$^2$',
                label=r'$\langle \Delta '+ylabel[i]+r'^2 \rangle$',
                color='red',
                marker='o')
            fit = mdt.dyn.msd(t=trenew, D=popt_msd[i], d=1)
            mask = (fit > 0)
            mdt.plot.plot(
                ax=axis,
                x=trenew[mask],
                y=fit[mask],
                xmin=args.XMIN,
                xmax=args.XMAX,
                ymin=args.YMIN,
                ymax=args.YMAX,
                logx=True,
                logy=True,
                xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
                ylabel=r'$\Delta '+ylabel[i]+r'^2(\tau_{renew})$ / '+args.LUNIT+r'$^2$',
                label="Fit",
                color='black')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        
        # Statistics
        fig, axis = plt.subplots(figsize=(11.69, 8.27),
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.axis('off')
        fontsize = 26
        
        xpos = 0.05
        ypos = 0.95
        if args.SEL:
            plt.text(x=xpos,
                     y=ypos,
                     s="Selection compound MSDs",
                     fontsize=fontsize)
        else:
            plt.text(x=xpos,
                     y=ypos,
                     s="Reference compound MSDs",
                     fontsize=fontsize)
        ypos -= 0.10
        plt.text(x=xpos, y=ypos, s="Statistics:", fontsize=fontsize)
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Tot. counts: ${:>16d}$'.format(msd.shape[0]),
                 fontsize=fontsize)
        xpos += 0.10
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Mean / '+args.LUNIT+r'$^2$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Std. dev. / '+args.LUNIT+r'$^2$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$N(\Delta a^2 = 0)$',
                 fontsize=fontsize)
        
        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos, y=ypos, s=r'$\Delta r^2$', fontsize=fontsize)
        xpos += 0.10
        plt.text(x=xpos,
                 y=ypos, 
                 s=r'${:>16.9e}$'.format(np.mean(msd_tot)),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(np.std(msd_tot)),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16d}$'.format(displ0[-1]),
                 fontsize=fontsize)
        for i, data in enumerate(msd.T):
            xpos = 0.05
            ypos -= 0.05
            plt.text(x=xpos,
                     y=ypos,
                     s=r'$\Delta '+ylabel[i]+r'$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(np.mean(data)),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(np.std(data)),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16d}$'.format(np.count_nonzero(data==0)),
                     fontsize=fontsize)
        
        # Fit parameters
        xpos = 0.05
        ypos -= 0.10
        plt.text(x=xpos, y=ypos, s="Fit parameters:", fontsize=fontsize)
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Fit function: $msd(t) = 2d \cdot D \cdot t$',
                 fontsize=fontsize)
        xpos += 0.10
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$D$ / '+args.LUNIT+r'$^2$ '+args.TUNIT+r'$^{-1}$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Std. dev. / '+args.LUNIT+r'$^2$ '+args.TUNIT+r'$^{-1}$',
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos, y=ypos, s=r'$d$', fontsize=fontsize)
        
        xpos = 0.05
        ypos -= 0.05
        plt.text(x=xpos,
                 y=ypos,
                 s=r'$\Delta r^2$',
                 fontsize=fontsize)
        xpos += 0.10
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(popt_msd[-1]),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16.9e}$'.format(np.sqrt(pcov_msd[-1])),
                 fontsize=fontsize)
        xpos += 0.30
        plt.text(x=xpos,
                 y=ypos,
                 s=r'${:>16d}$'.format(3),
                 fontsize=fontsize)
        for i in range(3):
            xpos = 0.05
            ypos -= 0.05
            plt.text(x=xpos,
                     y=ypos,
                     s=r'$\Delta '+ylabel[i]+r'^2$',
                     fontsize=fontsize)
            xpos += 0.10
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(popt_msd[i]),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16.9e}$'.format(np.sqrt(pcov_msd[i])),
                     fontsize=fontsize)
            xpos += 0.30
            plt.text(x=xpos,
                     y=ypos,
                     s=r'${:>16d}$'.format(1),
                     fontsize=fontsize)
        
        xpos = 0.05
        ypos -= 0.10
        plt.text(x=xpos,
                 y=ypos,
                 s=r'Non-Gaussian parameter: $A(t) = \frac{\langle \Delta a^4(t) \rangle}{(1+\frac{2}{d}) \cdot \langle \Delta a^2(t) \rangle^2} - 1$',
                 fontsize=fontsize)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        
        
        # Non-Gaussian parameters vs renewal time
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        
        axis.axhline(y=0, color='black')
        mdt.plot.plot(
            ax=axis,
            x=t,
            y=msd_tot_non_gaus[1:],
            xmin=0,
            xmax=args.XMAX,
            ymin=min(np.nanmin(msd_tot_non_gaus), np.nanmin(msd_non_gaus)),
            ymax=max(np.nanmax(msd_tot_non_gaus), np.nanmax(msd_non_gaus)),
            xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
            ylabel=r'$A(\tau_{renew})$',
            label=r'$\Delta r^2$',
            marker='o')
        
        mdt.plot.vlines(ax=axis,
                        x=tbins,
                        start=axis.get_ylim()[0],
                        stop=axis.get_ylim()[1],
                        xmin=args.XMIN,
                        xmax=args.XMAX,
                        ymin=args.YMIN,
                        ymax=args.YMAX,
                        color='black',
                        linestyle='dotted')
        
        label = (r'$\Delta x^2$', r'$\Delta y^2$', r'$\Delta z^2$')
        marker = ('^', 's', 'D')
        for i, data in enumerate(msd_non_gaus.T):
            mdt.plot.plot(
                ax=axis,
                x=t,
                y=data[1:],
                xmin=0,
                xmax=args.XMAX,
                ymin=min(np.nanmin(msd_tot_non_gaus), np.nanmin(msd_non_gaus)),
                ymax=max(np.nanmax(msd_tot_non_gaus), np.nanmax(msd_non_gaus)),
                xlabel=r'$\tau_{renew}$ / '+args.TUNIT,
                ylabel=r'$A(\tau_{renew})$',
                label=label[i],
                marker=marker[i])
        
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
