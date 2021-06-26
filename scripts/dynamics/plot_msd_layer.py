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
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mdtools as mdt




if __name__ == '__main__':
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Plot selected columns from the output of"
                     " msd_layer_serial.py (or msd_layer_parallel.py)."
                     )
    )
    
    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=True,
        help="One of the output files of msd_layer_serial.py (or"
             " msd_layer_parallel.py)."
    )
    parser.add_argument(
        '--fmd',
        dest='MDFILE',
        type=str,
        nargs='+',
        required=False,
        default=None,
        help="The output files of msd_layer_serial.py (or"
             " msd_layer_parallel.py) that contain the mean displacement"
             " (either one or all three). If provided the square of the"
             " mean displacement will be subtracted from the mean square"
             " displacement to correct for a potentially drifting system"
             " by calculating the variance <r^2> - <r>^2."
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
        '-l',
        dest='LAYER',
        type=int,
        nargs='+',
        required=False,
        default=None,
        help="Space separated list of layers for which to plot the MSD."
             " Numbering starts at one. Default is to plot the MSD for"
             " all layers."
    )
    parser.add_argument(
        '--d1',
        dest='MSD_DIRECTION',
        type=str,
        required=False,
        default='r',
        help="Which component of the MSD is contained in the input file."
             " Must be either r, x, y or z. Default: r"
    )
    parser.add_argument(
        '--d2',
        dest='BIN_DIRECTION',
        type=str,
        required=False,
        default='z',
        help="The spatial direction used to dicretize the MSD. Must be"
             " either x, y or z. Default: z"
    )
    
    parser.add_argument(
        '--xmin',
        dest='XMIN',
        type=float,
        required=False,
        default=None,
        help="Minimum x-range of the plot. Default: Minimum time greater"
             " than zero"
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
        default=None,
        help="Minimum y-range of the plot. Default: Minimum MSD"
    )
    parser.add_argument(
        '--ymax',
        dest='YMAX',
        type=float,
        required=False,
        default=None,
        help="Maximum y-range of the plot. Default: Maximum MSD"
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
    
    
    if (args.MSD_DIRECTION != 'r' and
        args.MSD_DIRECTION != 'x' and
        args.MSD_DIRECTION != 'y' and
        args.MSD_DIRECTION != 'z'):
        raise ValueError("--d1 must be either 'r, 'x', 'y' or 'z', but"
                         " you gave {}".format(args.MSD_DIRECTION))
    if (args.BIN_DIRECTION != 'x' and
        args.BIN_DIRECTION != 'y' and
        args.BIN_DIRECTION != 'z'):
        raise ValueError("--d2 must be either 'x', 'y' or 'z', but you"
                         " gave {}".format(args.BIN_DIRECTION))
    dim = {'x': 0, 'y': 1, 'z': 2}
    
    
    
    
    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()
    
    if args.LAYER is None:
        cols = None
    else:
        cols = [0] + np.unique(args.LAYER).tolist()
    msd = np.loadtxt(fname=args.INFILE, usecols=cols)
    times = msd[1:,0] * args.TCONV
    msd = msd[1:,1:] * args.LCONV**2
    if args.LAYER is None:
        layer = np.arange(1, msd.shape[1]+1)
    else:
        layer = np.unique(args.LAYER)
    
    if args.MDFILE is not None:
        if len(args.MDFILE) != 1 and len(args.MDFILE) != 3:
            raise ValueError("You must provide either one or three"
                             " additional input files with --fmd")
        md = [None,] * len(args.MDFILE)
        for i, mdfile in enumerate(args.MDFILE):
            md[i] = np.loadtxt(fname=mdfile, usecols=cols)
            times_md = md[i][1:,0] * args.TCONV
            if times_md.shape != times.shape:
                raise ValueError("The number of lag times in the"
                                 " different input files does not match")
            if not np.allclose(times_md, times):
                raise ValueError("The lag times of the different input"
                                 " files do not match")
            md[i] = md[i][1:,1:] * args.LCONV
            if md[i].shape != msd.shape:
                raise ValueError("The number of displacements in the"
                                 " different input files does not match")
        del times_md
        md = np.asarray(md)
        msd -= np.sum(md**2, axis=0)
        md = np.sum(md, axis=0)
    
    # Line with slope 1 in log-log plot indicating the diffusive regime
    if args.MSD_DIRECTION == 'r':
        ndim = 3
    else:
        ndim = 1
    try:
        popt, pcov = opt.curve_fit(
                         f=lambda t, D: mdt.dyn.msd(t=t, D=D, d=ndim),
                         xdata=times,
                         ydata=msd[:,len(layer)//2])
        fit_successful = True
    except (ValueError, RuntimeError, opt.OptimizeWarning):
        fit_successful = False
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating plot", flush=True)
    timer = datetime.now()
    
    fontsize_legend = 24
    
    if args.XMIN is None:
        args.XMIN = np.min(times[times>0])
    if args.XMAX is None:
        args.XMAX = np.max(times[times>0])
    
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        if args.MDFILE is not None:
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            axis.axhline(y=0, color='black')
            if args.MSD_DIRECTION != 'r':
                ylabel=(r'$\langle \Delta ' + args.MSD_DIRECTION +
                        r'(\Delta t) \rangle$ / ' + args.LUNIT)
            else:
                ylabel=(r'$\langle \Delta x(\Delta t) \rangle' +
                        r'+ \langle \Delta y(\Delta t) \rangle' +
                        r'+ \langle \Delta z(\Delta t) \rangle$ / ' +
                        args.LUNIT)
            for i, l in enumerate(layer):
                mdt.plot.plot(ax=axis,
                              x=times,
                              y=md[:,i],
                              xmin=args.XMIN,
                              xmax=args.XMAX,
                              ymin=np.min(md[np.isfinite(md)]),
                              ymax=np.max(md[np.isfinite(md)]),
                              logx=True,
                              xlabel=r'$\Delta t$ / '+args.TUNIT,
                              ylabel=ylabel,
                              label=str(l))
            axis.legend(loc='best',
                        bbox_to_anchor=(0, 0, 0.7, 1),
                        title="Bin number",
                        title_fontsize=fontsize_legend,
                        fontsize=fontsize_legend,
                        numpoints=1,
                        ncol=1 + len(layer)//6,
                        labelspacing = 0.2,
                        columnspacing = 1.4,
                        handletextpad = 0.5,
                        frameon=False)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        if args.YMIN is None:
            args.YMIN = np.min(msd[1:][np.isfinite(msd[1:])])
        if args.YMAX is None:
            args.YMAX = np.max(msd[1:][np.isfinite(msd[1:])])
        if args.MDFILE is not None:
            ylabel=(r'$[\langle \Delta ' + args.MSD_DIRECTION +
                    r'^2(\Delta t) \rangle - ' +
                    r'\langle \Delta ' + args.MSD_DIRECTION +
                    r'(\Delta t) \rangle^2]$' +
                    r' / ' + args.LUNIT + r'$^2$')
        else:
            ylabel=(r'$\langle \Delta ' + args.MSD_DIRECTION +
                    r'^2(\Delta t) \rangle$' +
                    r' / ' + args.LUNIT + r'$^2$')
        #cmap = plt.get_cmap('gist_rainbow')
        #axis.set_prop_cycle(color=[cmap(i/len(layer))
                                   #for i in range(len(layer))])
        #ls = ['-', '--', '-.', ':']
        #ls *= (1 + len(layer)//len(ls))
        for i, l in enumerate(layer):
            mdt.plot.plot(ax=axis,
                          x=times,
                          y=msd[:,i],
                          xmin=args.XMIN,
                          xmax=args.XMAX,
                          ymin=args.YMIN,
                          ymax=args.YMAX,
                          logx=True,
                          logy=True,
                          xlabel=r'$\Delta t$ / '+args.TUNIT,
                          ylabel=ylabel,
                          label=str(l))
        if fit_successful:
            mdt.plot.plot(ax=axis,
                          x=times,
                          y=mdt.dyn.msd(t=times, D=popt, d=ndim),
                          xmin=args.XMIN,
                          xmax=args.XMAX,
                          ymin=args.YMIN,
                          ymax=args.YMAX,
                          logx=True,
                          logy=True,
                          xlabel=r'$\Delta t$ / '+args.TUNIT,
                          ylabel=ylabel,
                          label=r'$\propto \Delta t$',
                          color='black',
                          linestyle='--')
        axis.legend(loc='upper left',
                    title="Bin number",
                    title_fontsize=fontsize_legend,
                    fontsize=fontsize_legend,
                    numpoints=1,
                    ncol=1 + len(layer)//10,
                    labelspacing = 0.2,
                    columnspacing = 1.4,
                    handletextpad = 0.5,
                    frameon=False)
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
