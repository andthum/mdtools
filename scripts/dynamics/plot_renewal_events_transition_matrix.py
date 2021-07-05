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
from scipy.linalg import eig
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import mdtools as mdt




if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())


    parser = argparse.ArgumentParser(
        description=(
            "Read a trajectory of renewal events as e.g."
            " generated with extract_renewal_events.py,"
            " discretize a given spatial direction and create"
            " and plot a row-stochastic transition matrix"
            " similar to the transition matrix of a Markov"
            " model. The matrix element T_ij represents the"
            " probability that a renewal event which starts in"
            " bins i ends in bin j."
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
        '-d',
        dest='DIRECTION',
        type=str,
        required=False,
        default='z',
        help="The spatial direction to dicretize. Must be either x, y"
             " or z. Default: z"
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
        '--bin-start',
        dest='START',
        type=float,
        required=False,
        default=None,
        help="Point on the chosen spatial direction to start binning."
             " Default: Minimum position in the given direction."
    )
    parser.add_argument(
        '--bin-end',
        dest='STOP',
        type=float,
        required=False,
        default=None,
        help="Point on the chosen spatial direction to stop binning."
             " Default: Maximum position in the given direction."
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
        '--f2',
        dest='INFILE2',
        type=str,
        required=False,
        default=None,
        help="An optional second input file providing additional"
             " 1-dimensional data as a function of the spatial direction"
             " given with -d, e.g. a density profile. This data will be"
             " plotted above the other plots."
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
        '--box',
        dest='BOX',
        type=float,
        nargs=2,
        required=False,
        default=[1, 1],
        help="Box dimensions in the two other directions not given by -d."
             " In a further plot, the data read from INFILE2 will be"
             " interpreted as number density profile of the compounds"
             " along the direction given with -d. The number of events"
             " will be divided by the number of compounds per bin, which"
             " is calculated from the integral of the density profile in"
             " this bin multiplied by the box dimensions in the"
             " remaining two directions. Default: '1 1'"
    )
    parser.add_argument(
        '--name',
        dest='NAME',
        type=str,
        required=False,
        default="Compound",
        help="Name of the compound to use in the y-axis label. Is"
             " meaningless if --f2 is not given. Default: 'Compound'"
    )

    parser.add_argument(
        '--min',
        dest='MIN',
        type=float,
        required=False,
        default=None,
        help="Minimum x- and y-range of the plot. By default detected"
             " automatically."
    )
    parser.add_argument(
        '--max',
        dest='MAX',
        type=float,
        required=False,
        default=None,
        help="Maximum x- and y-range of the plot. By default detected"
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


    if (args.DIRECTION != 'x' and
        args.DIRECTION != 'y' and
            args.DIRECTION != 'z'):
        raise ValueError("-d must be either 'x', 'y' or 'z', but you"
                         " gave {}".format(args.DIRECTION))
    dim = {'x': 0, 'y': 1, 'z': 2}




    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()

    if args.SEL:
        cols = (3, 7+dim[args.DIRECTION], 13+dim[args.DIRECTION])
    else:
        cols = (3, 4+dim[args.DIRECTION], 10+dim[args.DIRECTION])
    trenew, pos_t0, pos_trenew = np.loadtxt(fname=args.INFILE,
                                            usecols=cols,
                                            unpack=True)
    trenew *= args.TCONV
    pos_t0 *= args.LCONV
    pos_trenew *= args.LCONV
    pos_trenew += pos_t0

    if args.BINFILE is None:
        if args.START is None or args.START > np.min(pos_t0):
            args.START = np.min(pos_t0)
        if args.STOP is None or args.STOP <= np.max(pos_t0):
            args.STOP = np.max(pos_t0) + (np.max(pos_t0)-args.START)/args.NUM
        bins = np.linspace(args.START, args.STOP, args.NUM)
    else:
        bins = np.loadtxt(args.BINFILE, usecols=0)
        bins = np.unique(bins)
        if len(bins) == 0:
            raise ValueError("Invalid bins")
        if bins[0] > np.min(pos_t0):
            bins = np.insert(bins, 0, np.min(pos_t0))
        if bins[-1] <= np.max(pos_t0):
            bins = np.append(bins, np.max(pos_t0) + (np.max(pos_t0)-bins[0])/len(bins))

    if args.INFILE2 is not None:
        xdata, ydata = np.loadtxt(fname=args.INFILE2,
                                  comments=['#', '@'],
                                  usecols=args.COLS,
                                  unpack=True)
        n_compounds_per_bin = np.zeros(len(bins), dtype=np.float64)
        for i, b in enumerate(bins[1:], 1):
            mask = (xdata >= bins[i-1]) & (xdata < b)
            n_compounds_per_bin[i] = np.trapz(x=xdata[mask],
                                              y=ydata[mask])
        if n_compounds_per_bin[0] != 0:
            raise ValueError("The first element of n_compounds_per_bin"
                             " is not zero. This should not have"
                             " happened")
        n_compounds_per_bin *= args.BOX[0] * args.BOX[1]
        print("  Total number of compounds: {}"
              .format(np.sum(n_compounds_per_bin)),
              flush=True)

    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Creating transition matrix", flush=True)
    timer = datetime.now()

    bin_ix_t0 = np.digitize(pos_t0, bins) - 1
    bin_ix_trenew = np.digitize(pos_trenew, bins) - 1
    if np.any(bin_ix_t0 < 0):
        raise ValueError("At least one element of bin_ix_t0 is less"
                         " than zero. This should not have happened")
    if np.any(bin_ix_trenew < 0):
        raise ValueError("At least one element of bin_ix_trenew is less"
                         " than zero. This should not have happened")

    count_matrix = np.zeros((len(bins)-1, len(bins)-1), dtype=np.uint32)
    trenew_matrix = np.full((len(bins)-1, len(bins)-1), np.nan)
    nevents = np.zeros(len(bins)-1, dtype=np.uint32)
    for i in np.unique(bin_ix_t0):
        mask = (bin_ix_t0 == i)
        nevents[i] = np.count_nonzero(mask)
        bin_ix_trenew_unq, counts = np.unique(bin_ix_trenew[mask],
                                              return_counts=True)
        count_matrix[i][bin_ix_trenew_unq] += counts.astype(np.uint32)
        for j in bin_ix_trenew_unq:
            mask2 = mask & (bin_ix_trenew == j)
            if np.any(mask2):
                trenew_matrix[i][j] = np.mean(trenew[mask2])

    ####################################################################
    #print("count_matrix =")
    #print(count_matrix)
    #count_matrix[-1][-1] -= 4
    #count_matrix[-1][-2] += 2
    #count_matrix[-1][-3] += 2

    #count_matrix[-2][-1] -= 1
    #count_matrix[-2][-3] += 1

    #count_matrix[-4][-1] -= 1
    #count_matrix[-4][-4] -= 1
    #count_matrix[-4][-2] += 1
    #count_matrix[-4][-3] += 1

    #count_matrix[-5][-1] -= 1
    #count_matrix[-5][-2] += 1
    #print("count_matrix =")
    #print(count_matrix)
    ####################################################################

    active_set = np.arange(1, len(bins), dtype=np.uint32)
    inactive = np.flatnonzero(np.sum(count_matrix, axis=1) == 0)
    count_matrix = np.delete(count_matrix, inactive, axis=0)
    count_matrix = np.delete(count_matrix, inactive, axis=1)
    if np.sum(count_matrix) != np.sum(nevents):
        raise ValueError("The elements of the count matrix do not sum up"
                         " to the total number of renewal events.")
    trenew_matrix = np.delete(trenew_matrix, inactive, axis=0)
    trenew_matrix = np.delete(trenew_matrix, inactive, axis=1)
    active_set = np.delete(active_set, inactive)

    norm = np.sum(count_matrix, axis=1)
    transition_matrix = count_matrix / norm[:,None]
    if not np.allclose(np.sum(transition_matrix, axis=1), 1):
        raise ValueError("Not all rows of the row-stochastic transition"
                         " matrix sum up to unity")
    if np.max(transition_matrix) > 1:
        raise ValueError("At least one element of the transition matrix"
                         " is greater than one")
    if np.min(transition_matrix) < 0:
        raise ValueError("At least one element of the transition matrix"
                         " is less than zero")

    eigvals, eigvecs = eig(transition_matrix, left=True, right=False)
    mask = np.isclose(eigvals, 1)
    if not np.any(mask):
        raise ValueError("The transition matrix has no eigenvalue close"
                         " to unity")
    eigvals = np.abs(np.real(eigvals[mask]))
    eigvecs = np.abs(np.real(eigvecs[:,mask]))
    eigvecs /= np.sum(eigvecs, axis=0)
    if not np.allclose(np.sum(eigvecs, axis=0), 1):
        raise ValueError("The stationary distribution does not sum up to"
                         " unity")
    if np.max(eigvecs) > 1:
        raise ValueError("At least one element of the stationary"
                         " distribution is greater than one")
    if np.min(eigvecs) < 0:
        raise ValueError("At least one element of the stationary"
                         " distribution is less than zero")

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
        # Count/Transition/Renewal time matrix as function of bin number
        matrix = [count_matrix, transition_matrix, trenew_matrix]
        cbarlabel = ("Counts",
                     r'Transition probability $T_{ij}$',
                     r'$\langle \tau_{renew} \rangle$')
        cmap = ('Greys', 'Greys', 'plasma')
        for i in range(len(cbarlabel)):
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))

            xy = np.append(active_set, active_set[-1]+1) - 0.5
            mdt.plot.pcolormesh(
                ax=axis,
                x=xy,
                y=xy,
                z=matrix[i],
                xmin=0.5,
                xmax=len(bins)-0.5,
                ymin=0.5,
                ymax=len(bins)-0.5,
                xlabel=r'Bin $j$',
                ylabel=r'Bin $i$',
                cbarlabel=cbarlabel[i],
                cmap=cmap[i])

            axis.invert_yaxis()
            axis.xaxis.set_label_position('top')
            axis.xaxis.labelpad = 22
            axis.xaxis.tick_top()
            axis.tick_params(axis='x', which='both', pad=6)

            plt.tight_layout()
            pdf.savefig()
            plt.close()


        # Stationary distribution as function of bin number
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        mdt.plot.plot(ax=axis,
                      x=active_set,
                      y=eigvecs,
                      xmin=0.5,
                      xmax=len(bins)-0.5,
                      ymin=0,
                      xlabel=r'Bin $i$',
                      ylabel="Stationary distribution",
                      color='black',
                      marker='o')
        plt.tight_layout()
        pdf.savefig()
        plt.close()




        # Count/Transition/Renewal time matrix as function of spatial
        # direction
        for i in range(len(cbarlabel)):
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)

            matrix[i] = np.insert(matrix[i].astype(np.float32),
                                  inactive,
                                  np.nan,
                                  axis=0)
            matrix[i] = np.insert(matrix[i].astype(np.float32),
                                  inactive,
                                  np.nan,
                                  axis=1)
            mdt.plot.pcolormesh(
                ax=axis,
                x=bins,
                y=bins,
                z=matrix[i],
                xmin=args.MIN,
                xmax=args.MAX,
                ymin=args.MIN,
                ymax=args.MAX,
                xlabel=r'$'+args.DIRECTION+r'(t_0 + \tau_{renew})$ / '+args.LUNIT,
                ylabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
                cbarlabel=cbarlabel[i],
                cmap=cmap[i])

            yticks = np.array(axis.get_yticks())
            mask = ((yticks >= axis.get_xlim()[0]) &
                    (yticks <= axis.get_xlim()[1]))
            axis.set_xticks(yticks[mask])

            axis.invert_yaxis()
            axis.xaxis.set_label_position('top')
            axis.xaxis.labelpad = 22
            axis.xaxis.tick_top()
            axis.tick_params(axis='x', which='both', pad=6)

            plt.tight_layout()
            pdf.savefig()
            plt.close()


        # Stationary distribution as function of spatial direction
        if args.INFILE2 is None:
            fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                     frameon=False,
                                     clear=True,
                                     tight_layout=True)
        else:
            fig, axes = plt.subplots(
                nrows=2,
                sharex=True,
                figsize=(11.69, 8.27+8.27/5),
                frameon=False,
                clear=True,
                constrained_layout=True,
                gridspec_kw={'height_ratios': [1/5, 1]})
            axis = axes[1]

        eigvecs = np.insert(eigvecs, inactive, np.nan, axis=0)
        mdt.plot.plot(
            ax=axis,
            x=bins[1:]-np.diff(bins)/2,
            y=eigvecs,
            xmin=args.MIN,
            xmax=args.MAX,
            ymin=0,
            xlabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
            ylabel="Stationary distribution",
            color='black',
            marker='o')
        mdt.plot.vlines(ax=axis,
                        x=bins,
                        start=axis.get_ylim()[0],
                        stop=axis.get_ylim()[1],
                        xmin=args.MIN,
                        xmax=args.MAX,
                        ymin=0,
                        color='black',
                        linestyle='dotted')

        if args.INFILE2 is not None:
            mdt.plot.plot(ax=axes[0],
                          x=xdata,
                          y=ydata,
                          xmin=args.MIN,
                          xmax=args.MAX,
                          ymin=np.min(ydata),
                          ymax=np.max(ydata),
                          color='black')
            axes[0].xaxis.set_visible(False)
            axes[0].yaxis.set_visible(False)
            axes[0].spines['bottom'].set_visible(False)
            axes[0].spines['top'].set_visible(False)
            axes[0].spines['left'].set_visible(False)
            axes[0].spines['right'].set_visible(False)

        if args.INFILE2 is None:
            plt.tight_layout()
        pdf.savefig()
        plt.close()


        logy = (False, True)
        ymin = (0, None)
        for i in range(len(logy)):
            # Stationary distribution as function of spatial direction
            # together with number of events per bin
            if args.INFILE2 is None:
                fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                         frameon=False,
                                         clear=True,
                                         tight_layout=True)
            else:
                fig, axes = plt.subplots(
                    nrows=2,
                    sharex=True,
                    figsize=(11.69, 8.27+8.27/5),
                    frameon=False,
                    clear=True,
                    constrained_layout=True,
                    gridspec_kw={'height_ratios': [1/5, 1]})
                axis = axes[1]

            mdt.plot.errorbar(
                ax=axis,
                x=bins[1:]-np.diff(bins)/2,
                y=nevents,
                yerr=np.sqrt(nevents),
                xmin=args.MIN,
                xmax=args.MAX,
                ymin=ymin[i],
                logy=logy[i],
                xlabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
                ylabel=r'$N_{renew}$',
                label="Simulation",
                marker='o')
            mdt.plot.vlines(ax=axis,
                            x=bins,
                            start=axis.get_ylim()[0],
                            stop=axis.get_ylim()[1],
                            xmin=args.MIN,
                            xmax=args.MAX,
                            ymin=ymin[i],
                            color='black',
                            linestyle='dotted')
            mdt.plot.plot(
                ax=axis,
                x=bins[1:]-np.diff(bins)/2,
                y=eigvecs*np.sum(nevents),
                xmin=args.MIN,
                xmax=args.MAX,
                ymin=ymin[i],
                logy=logy[i],
                xlabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
                ylabel=r'$N_{renew}$',
                label="Model",
                marker='s')

            if args.INFILE2 is not None:
                mdt.plot.plot(ax=axes[0],
                              x=xdata,
                              y=ydata,
                              xmin=args.MIN,
                              xmax=args.MAX,
                              ymin=np.min(ydata),
                              ymax=np.max(ydata),
                              color='black')
                axes[0].xaxis.set_visible(False)
                axes[0].yaxis.set_visible(False)
                axes[0].spines['bottom'].set_visible(False)
                axes[0].spines['top'].set_visible(False)
                axes[0].spines['left'].set_visible(False)
                axes[0].spines['right'].set_visible(False)

            if args.INFILE2 is None:
                plt.tight_layout()
            pdf.savefig()
            plt.close()


            # Stationary distribution as function of spatial direction
            # together with number of events per bin
            # divided by the number of compounds per bin
            fig, axes = plt.subplots(
                nrows=2,
                sharex=True,
                figsize=(11.69, 8.27+8.27/5),
                frameon=False,
                clear=True,
                constrained_layout=True,
                gridspec_kw={'height_ratios': [1/5, 1]})
            mdt.plot.errorbar(
                ax=axes[1],
                x=bins[1:]-np.diff(bins)/2,
                y=nevents/n_compounds_per_bin[1:],
                yerr=np.sqrt(nevents)/n_compounds_per_bin[1:],
                xmin=args.MIN,
                xmax=args.MAX,
                ymin=ymin[i],
                logy=logy[i],
                xlabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
                ylabel=r'$N_{renew}$ / $\langle N_{'+args.NAME+r'} \rangle$',
                label="Simulation",
                marker='o')
            mdt.plot.vlines(ax=axes[1],
                            x=bins,
                            start=axes[1].get_ylim()[0],
                            stop=axes[1].get_ylim()[1],
                            xmin=args.MIN,
                            xmax=args.MAX,
                            ymin=ymin[i],
                            color='black',
                            linestyle='dotted')
            mdt.plot.plot(
                ax=axes[1],
                x=bins[1:]-np.diff(bins)/2,
                y=eigvecs*np.sum(nevents)/n_compounds_per_bin[1:][:,None],
                xmin=args.MIN,
                xmax=args.MAX,
                ymin=ymin[i],
                logy=logy[i],
                xlabel=r'${}(t_0)$ / {}'.format(args.DIRECTION, args.LUNIT),
                ylabel=r'$N_{renew}$ / $\langle N_{'+args.NAME+r'} \rangle$',
                label="Model",
                marker='s')
            mdt.plot.plot(ax=axes[0],
                          x=xdata,
                          y=ydata,
                          xmin=args.MIN,
                          xmax=args.MAX,
                          ymin=np.min(ydata),
                          ymax=np.max(ydata),
                          color='black')
            axes[0].xaxis.set_visible(False)
            axes[0].yaxis.set_visible(False)
            axes[0].spines['bottom'].set_visible(False)
            axes[0].spines['top'].set_visible(False)
            axes[0].spines['left'].set_visible(False)
            axes[0].spines['right'].set_visible(False)
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
