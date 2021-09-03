#!/usr/bin/env python3

# This file is part of MDTools.
# Copyright (C) 2021  The MDTools Development Team and all contributors
# listed in the file AUTHORS.rst
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


r"""
Plot the x-, y- and z-component of the MSD as function of the initial
particle position at a constant diffusion time.

.. todo::

    Finish docstring

This script is designed to plot cross sections from the output of
:mod:`scripts.dynamics.msd_layer_serial` (or
:mod:`scripts.dynamics.msd_layer_parallel`) for all three spatial
components x, y and z in one plot.

Options
-------
-f          Input files.  The three output files of
            :mod:`scripts.dynamics.msd_layer_serial` (or
            :mod:`scripts.dynamics.msd_layer_parallel`) containing the
            MSDs for all three spatial directions as space separated
            list in the order x y z.
--fmd       Optional further input files.  The three output files of
            :mod:`scripts.dynamics.msd_layer_serial` (or
            :mod:`scripts.dynamics.msd_layer_parallel`) containing the
            mean displacements for all three spatial directions as space
            separated list in the order x y z.  If provided, the square
            of the mean displacements will be subtracted from the MSDs
            to correct for a potential particle drift by calculating the
            variance :math:`\langle \Delta\mathbf{r}^2 \rangle -
            \langle \Delta\mathbf{r} \rangle^2`.
-o          Output filename.
-t          Diffusion time in data units for which to plot the MSDs as
            function of the initial particle position.  If no data are
            present at the given diffusion time, the next nearest
            diffusion time for which data are present is used.  Default:
            ``1``.
-d          The spatial direction that was used to dicretize the MSD.
            Default: ``'z'``.
--f2        An optional further input file providing additional
            1-dimensional data as a function of the spatial direction
            given with -d, e.g. a density profile.  This data will be
            plotted above the original plot.
--cols      From which columns of ``INFILE2`` to read the additional
            data.  Column numbering starts at zero.  The first given
            number determines the column containing the x values, the
            second is for the y values.  Default: ``[0, 1]``.
--xlim      Left and right limit of the x-axis in data coordinates.
            Pass 'None' to set the limits to the leftmost and rightmost
            bin edges.  Default:``[None, None]``.
--ylim      Lower and upper limit of the y-axis in data coordinates.
            Pass 'None' to adjust the limit(s) automatically.  Default:
            ``[None, None]``.
--logy      Use logarithmic scale for the y-axis.
--time-conv
            Multiply all times by this factor.  Default: ``1``.
--time-unit
            Time unit.  Default: ``'ps'``.
--length-conv
            Multiply all lengths by this factor.  Default:  ``1``.
--length-unit
            Lengh unit.  Default: ``A``.

See Also
--------
:mod:`plot_msd_layer` :
    Plot the MSD as function of diffusion time for different initial
    particle positions
:mod:`plot_msd_layer_heatmap` :
    Plot the MSD as function of the initial particle position and the
    diffusion time in a heatmap
:mod:`plot_msd_layer_cross_section_at_constant_time` :
    Plot (one component of) the MSD as function of the initial particle
    position at a constant diffusion time(s)
:mod:`plot_msd_layer_cross_section_xyz_at_constant_msd` :
    Plot the diffusion time at which the x-, y- and z-component of the
    MSD reach a certain value as function of the initial particle
    position

Examples
--------
TODO
"""


__author__ = "Andreas Thum"


# Standard libraries
import sys
import os
import argparse
from datetime import datetime, timedelta

# Third party libraries
import psutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Local application/library specific imports
import mdtools as mdt
import mdtools.plot as mdtplt  # noqa: F401; Import MDTools plot style


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Plot the x-, y- and z-component of the MSD as function of the"
            " initial particle position at a constant diffusion time.  This"
            " script is designed to plot cross sections from the output of"
            " msd_layer_serial.py (or msd_layer_parallel.py).  For more"
            " information, refer to the documetation of this script."
        )
    )
    parser.add_argument(
        "-f",
        dest="INFILES",
        type=str,
        nargs=3,
        required=True,
        help=(
            "Input files.  The three output files of msd_layer_serial.py (or"
            " msd_layer_parallel.py) containing the MSDs for all three spatial"
            " directions as space separated list in the order x y z."
        ),
    )
    parser.add_argument(
        "--fmd",
        dest="MDFILES",
        type=str,
        nargs=3,
        required=False,
        default=None,
        help=(
            "Optional further input files.  The three output files of"
            " msd_layer_serial.py (or msd_layer_parallel.py) containing the"
            " mean displacements for all three spatial directions as space"
            " separated list in the order x y z.  If provided, the square of"
            " the mean displacements will be subtracted from the MSDs to"
            " correct for a potential particle drift by calculating the"
            " variance <r^2> - <r>^2."
        ),
    )
    parser.add_argument(
        "-o",
        dest="OUTFILE",
        type=str,
        required=True,
        help=("Output filename."),
    )
    parser.add_argument(
        "-t",
        dest="TIME",
        type=float,
        required=False,
        default=1,
        help=(
            "Diffusion time in data units for which to plot the MSDs as"
            " function of the initial particle position.  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "-d",
        dest="BIN_DIRECTION",
        type=str,
        choices=("x", "y", "z"),
        required=False,
        default="z",
        help=(
            "The spatial direction that was used to dicretize the MSD."
            "  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--f2",
        dest="INFILE2",
        type=str,
        required=False,
        default=None,
        help=(
            "An optional further input file providing additional 1-dimensional"
            " data as a function of the spatial direction given with -d, e.g."
            " a density profile.  This data will be plotted above the original"
            " plot."
        ),
    )
    parser.add_argument(
        "--cols",
        dest="COLS",
        type=int,
        nargs=2,
        required=False,
        default=[0, 1],
        help=(
            "From which columns of INFILE2 to read the additional data."
            "  Column numbering starts at zero.  The first given number"
            " determines the column containing the x values, the second is for"
            " the y values.  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--xlim",
        dest="XLIM",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=float),
        nargs=2,
        required=False,
        default=[None, None],
        help=(
            "Left and right limit of the x-axis in data coordinates.  Default:"
            " %(default)s (this means the leftmost and rightmost bin edges)."
        ),
    )
    parser.add_argument(
        "--ylim",
        dest="YLIM",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=float),
        nargs=2,
        required=False,
        default=[None, None],
        help=(
            "Lower and upper limit of the y-axis in data coordinates."
            "  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--logy",
        dest="LOGY",
        required=False,
        default=False,
        action="store_true",
        help=("Use logarithmic scale for the y-axis."),
    )
    parser.add_argument(
        "--time-conv",
        dest="TCONV",
        type=float,
        required=False,
        default=1,
        help=("Multiply all times by this factor.  Default: %(default)s"),
    )
    parser.add_argument(
        "--time-unit",
        dest="TUNIT",
        type=str,
        required=False,
        default="ps",
        help=("Time unit.  Default: %(default)s"),
    )
    parser.add_argument(
        "--length-conv",
        dest="LCONV",
        type=float,
        required=False,
        default=1,
        help=("Multiply all lengths by this factor.  Default: %(default)s"),
    )
    parser.add_argument(
        "--length-unit",
        dest="LUNIT",
        type=str,
        required=False,
        default="A",
        help=("Lengh unit.  Default: %(default)s"),
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    dim = {"x": 0, "y": 1, "z": 2}

    print("\n")
    print("Reading input files...")
    timer = datetime.now()
    msd = [None] * len(args.INFILES)
    times = [None] * len(args.INFILES)
    bins = [None] * len(args.INFILES)  # Bin edges
    for i, infile in enumerate(args.INFILES):
        msd[i] = np.loadtxt(infile)
        times[i] = msd[i][1:, 0] * args.TCONV
        bins[i] = msd[i][0] * args.LCONV
        msd[i] = msd[i][1:, 1:] * args.LCONV ** 2
    for i in range(len(args.INFILES)):
        if times[i].shape != times[0].shape:
            raise ValueError(
                "All input files must contain the same number of lag times"
            )
        if not np.allclose(times[i], times[0]):
            raise ValueError(
                "The lag times must be the same in all input files"
            )
        if bins[i].shape != bins[0].shape:
            raise ValueError(
                "All input files must contain the same number of bins"
            )
        if not np.allclose(bins[i], bins[0]):
            raise ValueError(
                "The bin edges must be the same in all input files"
            )
        if msd[i].shape != msd[0].shape:
            raise ValueError(
                "The number of displacements must be the same in all input"
                " files"
            )
    times = times[0]
    bins = bins[0]
    bin_centers = bins[1:] - np.diff(bins) / 2
    msd = np.asarray(msd)
    _, tix = mdt.nph.find_nearest(times, args.TIME, return_index=True)
    msd = msd[:, tix]
    if args.MDFILES is not None:
        md = [None] * len(args.MDFILES)
        for i, mdfile in enumerate(args.MDFILES):
            md[i] = np.loadtxt(mdfile)
            times_md = md[i][1:, 0] * args.TCONV
            bins_md = md[i][0] * args.LCONV
            md[i] = md[i][1:, 1:] * args.LCONV
            if times_md.shape != times.shape:
                raise ValueError(
                    "All input files must contain the same number of lag times"
                )
            if not np.allclose(times_md, times):
                raise ValueError(
                    "The lag times must be the same in all input files"
                )
            if bins_md.shape != bins.shape:
                raise ValueError(
                    "All input files must contain the same number of bins"
                )
            if not np.allclose(bins_md, bins):
                raise ValueError(
                    "The bin edges must be the same in all input files"
                )
            if md[i].shape[1] != len(msd[i]):
                raise ValueError(
                    "md[{}] ({}) has not the same length as msd[{}]"
                    " ({})".format(i, md[i].shape[1], i, len(msd[i]))
                )
        del times_md, bins_md
        md = np.asarray(md)
        md = md[:, tix]
        msd -= md ** 2
        if np.any(msd < 0):
            raise ValueError(
                "At least one displacement variance is less than zero. Are you"
                " sure you have provided correct intput files?"
            )
    if args.INFILE2 is not None:
        data = np.loadtxt(args.INFILE2, comments=["#", "@"], usecols=args.COLS)
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating plot...")
    timer = datetime.now()
    labels = (r"$x$", r"$y$", r"$z$")
    markers = ("s", "D", "o")
    if args.XLIM[0] is None:
        args.XLIM[0] = np.nanmin(bins)
    if args.XLIM[1] is None:
        args.XLIM[1] = np.nanmax(bins)
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        if args.MDFILES is not None:
            # Plot mean displacement as function of initial position
            if args.INFILE2 is None:
                fig, ax1 = plt.subplots(clear=True)
            else:
                height_ratios = (0.2, 1)
                fig, axs = plt.subplots(
                    clear=True,
                    nrows=2,
                    sharex=True,
                    gridspec_kw={"height_ratios": height_ratios},
                )
                fig.set_figheight(fig.get_figheight() * sum(height_ratios))
                ax2, ax1 = axs
            ax1.axhline(y=0, color="black")
            for i in range(len(args.MDFILES)):
                ax1.plot(
                    bin_centers, md[i], label=labels[i], marker=markers[i]
                )
            ax1.set(
                xlabel=r"${}(t_0)$ / {}".format(
                    args.BIN_DIRECTION, args.LUNIT
                ),
                ylabel=(
                    r"$\langle \Delta a("
                    + str(times[tix])
                    + r"$ "
                    + args.TUNIT
                    + r"$) \rangle$ / "
                    + args.LUNIT
                ),
                xlim=args.XLIM,
            )
            for bin_edge in bins:
                ax1.axvline(x=bin_edge, color="black", linestyle="dotted")
            ax1.legend(ncol=3)
            if args.INFILE2 is not None:
                ax2.plot(data[:, 0], data[:, 1], color="black")
                ax2.set(
                    xlim=ax1.get_xlim(),
                    ylim=(np.nanmin(data[:, 1]), np.nanmax(data[:, 1])),
                )
                ax2.xaxis.set_visible(False)
                ax2.yaxis.set_visible(False)
                ax2.spines["bottom"].set_visible(False)
                ax2.spines["top"].set_visible(False)
                ax2.spines["left"].set_visible(False)
                ax2.spines["right"].set_visible(False)
            pdf.savefig()
            plt.close()
        # Plot MSD as function of initial position
        if args.INFILE2 is None:
            fig, ax1 = plt.subplots(clear=True)
        else:
            height_ratios = (0.2, 1)
            fig, axs = plt.subplots(
                clear=True,
                nrows=2,
                sharex=True,
                gridspec_kw={"height_ratios": height_ratios},
            )
            fig.set_figheight(fig.get_figheight() * sum(height_ratios))
            ax2, ax1 = axs
        for i in range(len(args.INFILES)):
            ax1.plot(bin_centers, msd[i], label=labels[i], marker=markers[i])
        if args.LOGY:
            ax1.set_yscale("log", base=10, subs=np.arange(2, 10))
        if args.MDFILES is not None:
            ylabel = (
                r"Var$[\Delta a("
                + str(times[tix])
                + r"$ "
                + args.TUNIT
                + r"$)]$ / "
                + args.LUNIT
                + r"$^2$"
            )
        else:
            ylabel = (
                r"$\langle \Delta a^2("
                + str(times[tix])
                + r"$ "
                + args.TUNIT
                + r"$) \rangle$ / "
                + args.LUNIT
                + r"$^2$"
            )
        ax1.set(
            xlabel=r"${}(t_0)$ / {}".format(args.BIN_DIRECTION, args.LUNIT),
            ylabel=ylabel,
            xlim=args.XLIM,
            ylim=args.YLIM,
        )
        for bin_edge in bins:
            ax1.axvline(x=bin_edge, color="black", linestyle="dotted")
        ax1.legend(loc="lower center", ncol=3)
        if args.INFILE2 is not None:
            ax2.plot(data[:, 0], data[:, 1], color="black")
            ax2.set(
                xlim=ax1.get_xlim(),
                ylim=(np.nanmin(data[:, 1]), np.nanmax(data[:, 1])),
            )
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
            ax2.spines["bottom"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.spines["right"].set_visible(False)
        pdf.savefig()
        plt.close()
    print("Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
