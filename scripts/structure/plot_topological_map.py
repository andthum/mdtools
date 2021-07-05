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


# This python script is inspired by the work of Florian Müller-Plathe
# and Wilfred F. van Gunsteren which was adapted by Oleg Borodin and
# Grant D. Smith
# * F. Müller-Plathe, W.F. van Gunsteren, JCP, 1995, 103, 4745-4756
# * O. Borodin, G.D. Smith, Macromolecules, 2006, 39, 1620-1629
# * O. Borodin, G.D. Smith, Macromolecules, 2007, 40, 1252-1258




import sys
import os
from datetime import datetime
import psutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import mdtools as mdt


# Matplotlib: Nested Gridspecs
# https://matplotlib.org/3.2.2/gallery/subplots_axes_and_figures/gridspec_nested.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-nested-py


plt.rc('lines', linewidth=2)
plt.rc('axes', linewidth=2)
plt.rc('xtick.major', width=2)
plt.rc('xtick.minor', width=2)
plt.rc('ytick.major', width=2)
plt.rc('ytick.minor', width=2)
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preview'] = True
plt.rc('font',**{'family' : 'serif', 'serif' : 'Times'})

fontsize_labels = 36
fontsize_ticks = 32
tick_length = 10




if __name__ == "__main__":

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())


    parser = argparse.ArgumentParser(
        description=(
            "Plot the outcome of topological_map.py. Different"
            " segment types are plotted on separate pages,"
            " different residue types in separate subplots and"
            " different atom types in separate subsubplots."
            " Different residues of the same type are separated"
            " by dashed lines in the respective subplot. The"
            " atoms are renumbered to give a continuous"
            " sequence starting at one. The y labels of the"
            " plots are read from the column headers of the"
            " input file. If you want to change the y labels,"
            " change the column headers (but white space is"
            " not allowed)"
        )
    )

    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=True,
        help="Input file (output of topological_map.py)."
    )
    parser.add_argument(
        '--f2',
        dest='INFILE2',
        type=str,
        required=False,
        default=None,
        help="An optional second input file providing additional"
             " 1-dimensional data as a function of the spatial direction"
             " given with -d, e.g. a density profile. If -d is given,"
             " this data will be plotted above the colorbar, otherwise"
             " it is ignored."
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
        '-b',
        dest='BEGIN',
        type=int,
        required=False,
        default=0,
        help="First frame to read. Frame numbering starts at zero."
             " Default: 0"
    )
    parser.add_argument(
        "-e",
        dest="END",
        type=int,
        required=False,
        default=-1,
        help="Last frame to read (exclusive, i.e. the last frame read is"
             " actually END-1). Default: -1 (means read the very last"
             " frame of the trajectory)"
    )
    parser.add_argument(
        '--every',
        dest='EVERY',
        type=int,
        required=False,
        default=1,
        help="Read every n-th frame. Default: 1"
    )
    parser.add_argument(
        '--plot-every',
        dest='PLOTEVERY',
        type=int,
        required=False,
        default=1,
        help="Plot every n-th frame. Must be an integer multiple of"
             " --every. Default: 1"
    )
    parser.add_argument(
        '--min-bound',
        dest='MINBOUND',
        type=int,
        required=False,
        default=1,
        help="Only plot residues that are bound for at least --min-bound"
             " consecutive frames to the reference position. Must be an"
             " integer multiple of --every. Only frames that are"
             " actually read in are considered. Default: 1"
    )

    parser.add_argument(
        '--time-conv',
        dest='TIME_CONV',
        type=float,
        required=False,
        default=1,
        help="Multiply the simulation time (x axis) by this factor."
             " Default: 1, which results in ps"
    )
    parser.add_argument(
        '--time-unit',
        dest='TIME_UNIT',
        type=str,
        required=False,
        default="ps",
        help="Time unit. Default: ps"
    )
    parser.add_argument(
        '--t0',
        dest='T0',
        type=float,
        required=False,
        default=0,
        help="Start time. The times read from the input file are shifted"
             " such that the first time equals this value. The start"
             " time must be given in the units specified with"
             " --time-unit Default: 0"
    )
    parser.add_argument(
        '--length-conv',
        dest='LENGTH_CONV',
        type=float,
        required=False,
        default=1,
        help="Multiply lengths by this factor. Meaningless if -d is not"
             " given. Default: 1, which results in Angstroms"
    )
    parser.add_argument(
        '--length-unit',
        dest='LENGTH_UNIT',
        type=str,
        required=False,
        default="A",
        help="Lengh unit. Default: A"
    )
    parser.add_argument(
        '--nxbins',
        dest='NXBINS',
        type=int,
        default=10,
        help="Maximum number of x tick intervalls to use for the plot."
             " Default: 10"
    )
    parser.add_argument(
        '--ylabel',
        dest='YLABEL',
        type=str,
        default="Coordinating",
        help="ylabel prefix. The ylabel is constructed from this prefix"
             " and the corresponding column header from the input file."
             " Default prefix: 'Coordinating'"
    )

    parser.add_argument(
        '-d',
        dest='DIRECTION',
        type=str,
        default=None,
        help="The data points in the plot can be colored according to"
             " the x, y or z coordinate of the reference position."
             " Default: No coloring."
    )
    parser.add_argument(
        '-c',
        dest='COLS',
        type=int,
        nargs=2,
        required=False,
        default=[0, 1],
        help="From which columns of INFILE2 to read additional data."
             " Column numbering starts at 0. The first given number"
             " determines the column containing the x values, the second"
             " is for the y values. Default: '0 1'"
    )
    parser.add_argument(
        '--cmap',
        dest='CMAP',
        type=str,
        default="gnuplot",
        help="Name of the matplotlib colormap used to decode the"
             " position coloring. Meaningless if -d is not set."
             " Default: gnuplot. For directions with periodic boundary"
             " conditions, a cyclic colormap like 'twilight' might be"
             " more suitable."
    )
    parser.add_argument(
        '--vmin',
        dest='VMIN',
        type=float,
        default=None,
        help="Lower limit of the colorbar and the data value that"
             " defines 0.0 in the normalization of the colormap."
             " Meaningless if -d is not set."
    )
    parser.add_argument(
        '--vmax',
        dest='VMAX',
        type=float,
        default=None,
        help="Upper limit of the colorbar and the data value that"
             " defines 1.0 in the normalization of the colormap."
             " Meaningless if -d is not set."
    )
    parser.add_argument(
        '--vcenter',
        dest='VCENTER',
        type=float,
        default=None,
        help="The data value that defines 0.5 in the normalization of"
             " the colormap. Useful for diverging colormaps to set the"
             " point of divergence. Meaningless if -d is not set."
    )

    parser.add_argument(
        '--verbose',
        dest='VERBOSE',
        required=False,
        default=False,
        action='store_true',
        help="Print progress information."
    )
    parser.add_argument(
        '--debug',
        dest='DEBUG',
        required=False,
        default=False,
        action='store_true',
        help="Run in debug mode."
    )


    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())


    if (args.DIRECTION is not None and
        args.DIRECTION != 'x' and
        args.DIRECTION != 'y' and
            args.DIRECTION != 'z'):
        raise ValueError("-d must be either 'x', 'y' or 'z', but you"
                         " gave {}".format(args.DIRECTION))
    pos_col = {'x': 1, 'y': 2, 'z': 3}




    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()


    with open(args.INFILE, 'r') as f:
        headers = None
        for line in f:
            if line[0] != '#':
                break
            else:
                headers = line[1:]

    if headers is None:
        raise ValueError("Could not read headers")
    if len(headers) <= 4:
        raise ValueError("headers does not contain segment, residue or"
                         " atom type names")

    headers = headers.split()
    headers = np.array(headers[4:], dtype=str)
    ix_cols = np.arange(4, len(headers)+4)


    times = np.genfromtxt(fname=args.INFILE, usecols=0, dtype=np.float32)
    if len(times) <= 1:
        raise ValueError("The input file must contain at least two"
                         " frames")
    BEGIN, END, EVERY, n_frames = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=len(times))
    PLOTEVERY, effective_plotevery = mdt.check.restarts(
        restart_every_nth_frame=args.PLOTEVERY,
        read_every_nth_frame=EVERY,
        n_frames=n_frames)
    MINBOUND, effective_minbound = mdt.check.restarts(
        restart_every_nth_frame=args.MINBOUND,
        read_every_nth_frame=EVERY,
        n_frames=n_frames)
    times -= times[0]
    times += args.T0
    times = times[BEGIN:END:EVERY]
    times *= args.TIME_CONV


    ix = np.genfromtxt(fname=args.INFILE, usecols=ix_cols, dtype=str)
    seg_cols = np.any(np.char.isalpha(ix), axis=0)
    atm_cols = np.all(np.char.isnumeric(ix), axis=0)
    res_cols = np.equal(seg_cols, atm_cols)
    ix = ix[BEGIN:END:EVERY]


    if args.DIRECTION is None:
        pos = None
        vmin = None
        vmax = None
        norm = None
        cmap = None
    else:
        pos = np.genfromtxt(fname=args.INFILE,
                            usecols=pos_col[args.DIRECTION],
                            dtype=np.float32)
        pos = pos[BEGIN:END:EVERY]
        pos *= args.LENGTH_CONV
        vmin = np.min(pos)
        vmax = np.max(pos)
        if args.INFILE2 is None:
            data = None
            label_pad = 16
        else:
            data = np.loadtxt(fname=args.INFILE2,
                              comments=['#', '@'],
                              usecols=args.COLS,
                              ndmin=2,
                              dtype=np.float32)
            if np.min(data[:,0]) < vmin:
                vmin = np.min(data[:,0])
            if np.max(data[:,0]) > vmax:
                vmax = np.max(data[:,0])
            label_pad = 8
        if args.VMIN is not None:
            vmin = args.VMIN
        if args.VMAX is not None:
            vmax = args.VMAX
        if args.VCENTER is not None:
            norm = colors.TwoSlopeNorm(vmin=vmin,
                                       vcenter=args.VCENTER,
                                       vmax=vmax)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        if args.CMAP == 'discrete':
            cmap = colors.ListedColormap(colors=['blue', 'red'],
                                         name='discrete')
            cmap.set_over('orange')
            cmap.set_under('darkviolet')
            if args.VCENTER is not None:
                bounds = [vmin, args.VCENTER, vmax]
            else:
                bounds = [vmin, np.mean((vmin, vmax)), vmax]
            norm = colors.BoundaryNorm(boundaries=bounds,
                                       ncolors=cmap.N)
            spacing='proportional'
        else:
            cmap = args.CMAP
            spacing='uniform'


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
        seg_blocks = np.append(np.nonzero(seg_cols)[0], len(seg_cols))
        for sb in range(len(seg_blocks)-1):
            sb_headers = headers[seg_blocks[sb]:seg_blocks[sb+1]]
            print(flush=True)
            print("  Segment: {:>12s}"
                  .format(sb_headers[0]),
                  flush=True)
            sb_ix = ix[:,seg_blocks[sb]:seg_blocks[sb+1]]
            #sb_seg_ix = np.array([mdt.nph.excel_colnum(i)  # not needed
            #for i in sb_ix[:,0]],
            #dtype=np.float16)
            sb_seg_cols = seg_cols[seg_blocks[sb]:seg_blocks[sb+1]]
            sb_atm_cols = np.all(np.char.isnumeric(sb_ix), axis=0)
            sb_atm_cols &= atm_cols[seg_blocks[sb]:seg_blocks[sb+1]]
            sb_res_cols = np.equal(sb_seg_cols, sb_atm_cols)
            sb_res_cols &= res_cols[seg_blocks[sb]:seg_blocks[sb+1]]
            res_blocks = np.append(np.nonzero(sb_res_cols)[0],
                                   len(sb_res_cols))
            res_blocks_same = mdt.nph.ix_of_item_change_1d(sb_headers[sb_res_cols])
            res_blocks_same = res_blocks_same[1:] - 1
            if len(res_blocks_same) == 0 and len(res_blocks) > 2:
                res_blocks = res_blocks[[0, -1]]
            else:
                res_blocks = np.delete(res_blocks, res_blocks_same)

            if pos is None:
                nrows = len(res_blocks) - 1
                grid_heights = np.ones(nrows)
                fig_size = 0
            else:
                nrows = len(res_blocks)
                grid_heights = np.ones(nrows)
                if data is None:
                    grid_heights[0] = 0.1
                    fig_size = 0
                else:
                    grid_heights[0] = 0.3
                    fig_size = 8.27 * (grid_heights[0] + 0.05)
            fig = plt.figure(figsize=(11.69,
                                      8.27*(len(res_blocks)-1)+fig_size),
                             frameon=False,
                             clear=True)
            grid_outer = gridspec.GridSpec(nrows=nrows,
                                           ncols=1,
                                           figure=fig,
                                           height_ratios=grid_heights)

            for rb in range(len(res_blocks)-1):
                rb_headers = sb_headers[res_blocks[rb]:res_blocks[rb+1]]
                print("    Residue: {:>10s}"
                      .format(rb_headers[0]),
                      flush=True)
                rb_ix = sb_ix[:,res_blocks[rb]:res_blocks[rb+1]].astype(np.float32)
                rb_res_ix = -rb_ix[:,sb_res_cols[res_blocks[rb]:res_blocks[rb+1]]]
                rb_res_ix = mdt.dyn.replace_short_sequences_global(
                    list_of_arrays=rb_res_ix,
                    min_len=effective_minbound,
                    val=0,
                    verbose=args.VERBOSE,
                    debug=args.DEBUG)
                rb_res_ix_sorted = np.unique(rb_res_ix[rb_res_ix>0])
                rb_atm_cols = np.all(rb_ix >= 0, axis=0)
                rb_atm_cols &= atm_cols[res_blocks[rb]:res_blocks[rb+1]]
                atm_types = np.unique(rb_headers[rb_atm_cols])
                atm_blocks = np.zeros((len(atm_types), len(rb_headers)),
                                      dtype=bool)
                for i, at in enumerate(atm_types):
                    atm_blocks[i][rb_headers==at] = True

                if pos is None:
                    grid_cell = rb
                else:
                    grid_cell = rb + 1
                grid_inner = gridspec.GridSpecFromSubplotSpec(
                    nrows=len(atm_types),
                    ncols=1,
                    subplot_spec=grid_outer[grid_cell],
                    hspace=0)
                axis_prev = None

                for i, ab in enumerate(atm_blocks):
                    ab_headers = rb_headers[ab]
                    print("      Atom type: {:>6s}"
                          .format(ab_headers[0]),
                          flush=True)
                    ab_ix = rb_ix[:,ab]
                    rb_res_ix_invalid = np.equal(rb_res_ix, 0)
                    rb_res_ix_invalid = np.repeat(
                        rb_res_ix_invalid,
                        len(ab_ix[0])//len(rb_res_ix[0]),
                        axis=1)
                    ab_ix[rb_res_ix_invalid] = 0
                    ab_ix_valid = ab_ix > 0
                    if np.any(ab_ix_valid):
                        ab_ix_min = np.min(ab_ix[ab_ix_valid])
                        ab_ix[ab_ix_valid] -= ab_ix_min - 1
                    else:
                        ab_ix_min = 0
                    ab_ix_sorted = np.unique(ab_ix[ab_ix_valid])
                    ab_ix_sorted_diff = np.diff(ab_ix_sorted)
                    ab_ix_sorted_diff_big = np.nonzero(ab_ix_sorted_diff>1)[0]
                    for d in ab_ix_sorted_diff_big:
                        ab_ix[ab_ix>=ab_ix_sorted[d+1]] -= ab_ix_sorted_diff[d]-1
                        ab_ix_sorted[d+1:] -= ab_ix_sorted_diff[d] - 1
                    ab_ix_in_r_min = np.zeros(len(rb_res_ix_sorted),
                                              dtype=np.uint32)
                    for j, r in enumerate(rb_res_ix_sorted):
                        r_mask = np.equal(rb_res_ix, r)
                        r_mask = np.repeat(
                            r_mask,
                            len(ab_ix[0])//len(rb_res_ix[0]),
                            axis=1)
                        ab_ix_in_r = np.unique(ab_ix[r_mask])
                        if not np.all(ab_ix_in_r == 0):
                            ab_ix_in_r_min[j] = ab_ix_in_r[ab_ix_in_r>0][0]
                    ab_ix_in_r_min = np.unique(ab_ix_in_r_min[ab_ix_in_r_min>0])
                    ab_ix_max = np.max(ab_ix)
                    ab_ix[np.logical_not(ab_ix_valid)] = np.nan

                    axis = fig.add_subplot(grid_inner[i],
                                           sharex=axis_prev)
                    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
                    if args.NXBINS is not None:
                        axis.locator_params(nbins=args.NXBINS, axis='x')
                    axis.hlines(y=ab_ix_in_r_min-0.5,
                                xmin=times[0],
                                xmax=times[-1],
                                linestyles='--')

                    for a_ix in ab_ix.T:
                        mdt.plot.scatter(
                            ax=axis,
                            x=times[::effective_plotevery],
                            y=a_ix[::effective_plotevery],
                            c=pos[::effective_plotevery],
                            xmin=times[0],
                            xmax=times[-1],
                            ymin=0.5,
                            ymax=ab_ix_max+0.5 if ab_ix_max>1 else ab_ix_max+1.5,
                            xlabel=r'Time / {}'.format(args.TIME_UNIT),
                            ylabel=r'{} {}'.format(args.YLABEL,
                                                   ab_headers[0]),
                            marker='.',
                            color='black' if pos is None else None,
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                            norm=norm)
                    if rb < len(res_blocks)-2:
                        axis.tick_params(labelbottom=False)
                        axis.set_xlabel('')
                    elif i < len(atm_types)-1:
                        axis.tick_params(labelbottom=False)
                        axis.set_xlabel('')
                    else:
                        axis.ticklabel_format(axis='x',
                                              style='scientific',
                                              scilimits=(0,0),
                                              useOffset=False)
                    axis_prev = axis

            if pos is not None:
                if data is not None:
                    grid_inner = gridspec.GridSpecFromSubplotSpec(
                        nrows=2,
                        ncols=1,
                        subplot_spec=grid_outer[0],
                        hspace=0.1,
                        height_ratios=[0.7, 0.3])
                    axis = fig.add_subplot(grid_inner[0])
                    _, vmin_ix = mdt.nph.find_nearest(data[:,0],
                                                      val=vmin,
                                                      return_index=True,
                                                      debug=args.DEBUG)
                    _, vmax_ix = mdt.nph.find_nearest(data[:,0],
                                                      val=vmax,
                                                      return_index=True,
                                                      debug=args.DEBUG)
                    if vmin_ix == vmax_ix:
                        vmin_ix = 0
                        vmax_ix = len(data[:,1])
                    mdt.plot.plot(ax=axis,
                                  x=data[:,0],
                                  y=data[:,1],
                                  xmin=vmin,
                                  xmax=vmax,
                                  ymin=np.min(data[vmin_ix:vmax_ix, 1]),
                                  ymax=np.max(data[vmin_ix:vmax_ix, 1]),
                                  color='black')
                    axis.xaxis.set_visible(False)
                    axis.yaxis.set_visible(False)
                    axis.spines['bottom'].set_visible(False)
                    axis.spines['top'].set_visible(False)
                    axis.spines['left'].set_visible(False)
                    axis.spines['right'].set_visible(False)
                    axis = fig.add_subplot(grid_inner[1])
                else:
                    axis = fig.add_subplot(grid_outer[0])
                axis.axis('off')
                mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
                cbar = fig.colorbar(mappable=mappable,
                                    ax=axis,
                                    orientation='horizontal',
                                    fraction=1,
                                    spacing=spacing)
                cbar.set_label(label=r'${}$ / {}'.format(args.DIRECTION,
                                                         args.LENGTH_UNIT),
                               fontsize=fontsize_labels)
                cbar.ax.xaxis.labelpad = label_pad
                cbar.ax.xaxis.offsetText.set(size=fontsize_ticks)
                cbar.ax.tick_params(which='major',
                                    direction='out',
                                    length=tick_length,
                                    labelsize=fontsize_ticks)
                cbar.ax.tick_params(which='minor',
                                    direction='out',
                                    length=0.5*tick_length,
                                    labelsize=0.8*fontsize_ticks)
                if data is None:
                    cbar.ax.xaxis.set_ticks_position('top')
                    cbar.ax.xaxis.set_label_position('top')

            grid_outer.tight_layout(fig)
            pdf.savefig()
            plt.close()

        # Parameters for creating the plot(s)
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.axis('off')
        ypos = 0.95
        plt.text(x=0.05,
                 y=ypos,
                 s=r'Parameters for creating the plot(s)',
                 fontsize=28)
        ypos -= 0.05
        plt.text(x=0.05,
                 y=ypos,
                 s=r'--begin = ${}$'.format(args.BEGIN),
                 fontsize=28)
        ypos -= 0.05
        plt.text(x=0.05,
                 y=ypos,
                 s=r'--end = ${}$'.format(args.END),
                 fontsize=28)
        ypos -= 0.05
        plt.text(x=0.05,
                 y=ypos,
                 s=r'--every = ${}$'.format(args.EVERY),
                 fontsize=28)
        ypos -= 0.05
        plt.text(x=0.05,
                 y=ypos,
                 s=r'--plot-every = ${}$'.format(args.PLOTEVERY),
                 fontsize=28)
        ypos -= 0.05
        plt.text(x=0.05,
                 y=ypos,
                 s=r'--min-bound = ${}$'.format(args.MINBOUND),
                 fontsize=28)
        ypos -= 0.05
        plt.text(x=0.05,
                 y=ypos,
                 s=r'--cmap = ${}$'.format(args.CMAP),
                 fontsize=28)
        ypos -= 0.05
        plt.text(x=0.05,
                 y=ypos,
                 s=r'--vmin = ${}$'.format(args.VMIN),
                 fontsize=28)
        ypos -= 0.05
        plt.text(x=0.05,
                 y=ypos,
                 s=r'--vmax = ${}$'.format(args.VMAX),
                 fontsize=28)
        ypos -= 0.05
        plt.text(x=0.05,
                 y=ypos,
                 s=r'--vcenter = ${}$'.format(args.VCENTER),
                 fontsize=28)
        ypos -= 0.05
        plt.text(x=0.05,
                 y=ypos,
                 s=r'--f2 = ${}$'.format(args.INFILE2),
                 fontsize=28)
        ypos -= 0.05
        plt.text(x=0.05,
                 y=ypos,
                 s=r'-c = ${}$'.format(args.COLS),
                 fontsize=28)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print(flush=True)
    print("  Created {}".format(args.OUTFILE))
    print(flush=True)
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
