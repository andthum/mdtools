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
import mdtools as mdt




if __name__ == '__main__':
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Plot a discretized single particle trajectory"
                     " generated with discrete_hex.py"
                     )
    )
    
    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="File containing the discretized trajectories created by"
             " discrete_hex.py"
    )
    parser.add_argument(
        '--lf',
        dest='LATFACE',
        type=str,
        required=True,
        help="File containing the hexagonal lattice faces created by"
             " discrete_hex.py"
    )
    parser.add_argument(
        '--lv',
        dest='LATVERT',
        type=str,
        required=True,
        help="File containing the hexagonal lattice vertices created by"
             " discrete_hex.py"
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
        '-i',
        dest='IX',
        type=int,
        required=False,
        default=None,
        help="The index of the particle for which to plot the"
             " discretized trajectory. Indexing starts at zero. Default:"
             " The particle that has the fewest negative elements in its"
             " discretized trajectory"
    )
    
    parser.add_argument(
        '--every',
        dest='EVERY',
        type=int,
        required=False,
        default=1,
        help="Plot only every n-th frame. Default: 1"
    )
    parser.add_argument(
        '--msize',
        dest='MSIZE',
        required=False,
        default=False,
        action='store_true',
        help="If given, the size of the scatter points will be"
             " proportional to the number of times the particle visited"
             " the respective lattice face."
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
    
    
    
    
    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()
    
    dtraj = np.load(args.TRJFILE)
    latfaces = np.load(args.LATFACE)
    latverts = np.load(args.LATVERT)
    latfaces *= args.LCONV
    latverts *= args.LCONV
    
    if args.IX is None:
        args.IX = np.argmax(np.count_nonzero(dtraj > 0, axis=1))
        print("  Chosen particle {}".format(args.IX), flush=True)
    if args.IX < 0:
        raise ValueError("The particle index ({}) is less than zero"
                         .format(args.IX))
    elif args.IX > len(dtraj) - 1:
        raise ValueError("The particle index ({}) is out of range"
                         .format(args.IX))
    dtraj = dtraj[args.IX]
    valid = (dtraj > 0)
    if not np.any(valid):
        raise ValueError("The selected particle is never in the"
                         " [ZMIN; ZMAX) interval used while generating"
                         " the discrete trajectory")
    
    n_frames = len(dtraj)
    frames = np.arange(n_frames)
    
    if args.MSIZE:
        u, ix, c = np.unique(dtraj,
                             return_inverse=True,
                             return_counts=True)
        markersize = c[ix]
        del u, ix, c
    else:
        markersize = None
    
    dtraj = dtraj[valid][::args.EVERY]
    frames = frames[valid][::args.EVERY]
    if markersize is not None:
        markersize = markersize[valid][::args.EVERY]
    
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
    label_pad = 16
    
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.ticklabel_format(axis='x',
                              style='scientific',
                              scilimits=(0,0),
                              useOffset=False)
        mdt.plot.scatter(ax=axis,
                         x=frames,
                         y=dtraj,
                         xmin=0,
                         xmax=n_frames,
                         ymin=np.min(dtraj),
                         ymax=np.max(dtraj),
                         xlabel="Frame",
                         ylabel="Lattice site",
                         marker='.')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        
        
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        
        mdt.plot.scatter(ax=axis,
                         x=latverts[:,0],
                         y=latverts[:,1],
                         xmin=args.XMIN,
                         xmax=args.XMAX,
                         ymin=args.YMIN,
                         ymax=args.YMAX,
                         xlabel=r'$x$ / {}'.format(args.LUNIT),
                         ylabel=r'$y$ / {}'.format(args.LUNIT),
                         marker='.',
                         color='black')
        
        img = mdt.plot.scatter(ax=axis,
                               x=latfaces[dtraj][:,0],
                               y=latfaces[dtraj][:,1],
                               c=frames,
                               s=markersize,
                               xmin=args.XMIN,
                               xmax=args.XMAX,
                               ymin=args.YMIN,
                               ymax=args.YMAX,
                               xlabel=r'$x$ / {}'.format(args.LUNIT),
                               ylabel=r'$y$ / {}'.format(args.LUNIT),
                               marker='o',
                               cmap='plasma',
                               vmin=0,
                               vmax=n_frames)
        cbar = fig.colorbar(img, ax=axis)
        cbar.set_label(label="Frame", fontsize=fontsize_labels)
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
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        cbar.ax.ticklabel_format(axis='y',
                                 style='scientific',
                                 scilimits=(0,0),
                                 useOffset=False)
        
        xsize = abs(axis.get_xlim()[0] - axis.get_xlim()[1])
        ysize = abs(axis.get_ylim()[0] - axis.get_ylim()[1])
        axis.set_aspect(ysize / xsize)
        
        yticks = np.array(axis.get_yticks())
        mask = ((yticks >= axis.get_xlim()[0]) &
                (yticks <= axis.get_xlim()[1]))
        axis.set_xticks(yticks[mask])
        
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        
        
        
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        
        mdt.plot.scatter(ax=axis,
                         x=latverts[:,0],
                         y=latverts[:,1],
                         xmin=args.XMIN,
                         xmax=args.XMAX,
                         ymin=args.YMIN,
                         ymax=args.YMAX,
                         xlabel=r'$x$ / {}'.format(args.LUNIT),
                         ylabel=r'$y$ / {}'.format(args.LUNIT),
                         marker='.',
                         color='black')
        
        latface_num = np.arange(len(latfaces))
        for i, txt in enumerate(latface_num):
            axis.annotate(txt,
                          xy=(latfaces[:,0][i], latfaces[:,1][i]),
                          horizontalalignment='center',
                          verticalalignment='center')
        
        xsize = abs(axis.get_xlim()[0] - axis.get_xlim()[1])
        ysize = abs(axis.get_ylim()[0] - axis.get_ylim()[1])
        axis.set_aspect(ysize / xsize)
        
        yticks = np.array(axis.get_yticks())
        mask = ((yticks >= axis.get_xlim()[0]) &
                (yticks <= axis.get_xlim()[1]))
        axis.set_xticks(yticks[mask])
        
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
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
