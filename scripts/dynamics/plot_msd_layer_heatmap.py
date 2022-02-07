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
Plot the MSD as function of the initial particle position and the
diffusion time in a heatmap.

.. todo::

    Finish docstring

This script is designed to plot the output of
:mod:`scripts.dynamics.msd_layer_serial` (or
:mod:`scripts.dynamics.msd_layer_parallel`).

Options
-------
-f          Input file.  One of the output files of
            :mod:`scripts.dynamics.msd_layer_serial` (or
            :mod:`scripts.dynamics.msd_layer_parallel`).
--fmd       Optional further input file(s).  The output file(s) of
            :mod:`scripts.dynamics.msd_layer_serial` (or
            :mod:`scripts.dynamics.msd_layer_parallel`) that contain(s)
            the mean displacement.  If provided the square of the mean
            displacement will be subtracted from the MSD to correct for
            a potential particle drift by calculating the variance
            :math:`\langle \Delta\mathbf{r}^2 \rangle -
            \langle \Delta\mathbf{r} \rangle^2`.  You can either provide
            the mean displacement in a single spatial direction or in
            all three spatial directions.
-o          Output filename.
--d1        {'r', 'x', 'y', 'z'}

            The component of the MSD that is contained in the input
            file.  ``'r'`` stands for the entire displacement vector.
            Default: ``'r'``.
--d2        The spatial direction that was used to dicretize the MSD.
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
            Pass 'None' to adjust the limit(s) automatically.  Default:
            ``[None, None]``.
--ylim      Lower and upper limit of the y-axis in data coordinates.
            Pass 'None' to adjust the limit(s) automatically.  Default:
            ``[None, None]``.
--vlim      The colorbar range.  Pass 'None' to adjust the limit(s)
            automatically.  Default: ``[None, None]``.
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
:mod:`plot_msd_layer_cross_section_at_constant_time` :
    Plot (one component of) the MSD as function of the initial particle
    position at a constant diffusion time(s).
:mod:`plot_msd_layer_cross_section_xyz_at_constant_time` :
    Plot the x-, y- and z-component of the MSD as function of the
    initial particle position at a constant diffusion time
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
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages

# Local application/library specific imports
import mdtools as mdt
import mdtools.plot as mdtplt


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Plot the MSD as function of the initial particle position and the"
            " diffusion time in a heatmap.  This script is designed to plot"
            " the output of msd_layer_serial.py (or msd_layer_parallel.py)."
            "  For more information, refer to the documetation of this script."
        )
    )
    parser.add_argument(
        "-f",
        dest="INFILE",
        type=str,
        required=True,
        help=(
            "Input file.  One of the output files of msd_layer_serial.py (or"
            " msd_layer_parallel.py)."
        ),
    )
    parser.add_argument(
        "--fmd",
        dest="MDFILES",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help=(
            "Optional further input file(s).  The output file(s) of"
            " msd_layer_serial.py (or msd_layer_parallel.py) that contain(s)"
            " the mean displacement.  If provided the square of the mean"
            " displacement will be subtracted from the MSD to correct for a"
            " potential particle drift by calculating the variance"
            " <r^2> - <r>^2.  You can either provide the mean displacement in"
            " a single spatial direction or in all three spatial directions."
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
        "--d1",
        dest="MSD_DIRECTION",
        type=str,
        choices=("r", "x", "y", "z"),
        required=False,
        default="r",
        help=(
            "The component of the MSD that is contained in the input file."
            "  r stands for the entire displacement vector.  Default:"
            " %(default)s"
        ),
    )
    parser.add_argument(
        "--d2",
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
            " %(default)s."
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
        "--vlim",
        dest="VLIM",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=float),
        nargs=2,
        required=False,
        default=[None, None],
        help=("The colorbar range.  Default: %(default)s"),
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
    if args.MDFILES is not None and len(args.MDFILES) not in (1, 3):
        raise ValueError(
            "--fmd takes either one or three input files, not"
            " {}".format(len(args.MDFILES))
        )
    dim = {"x": 0, "y": 1, "z": 2}

    print("\n")
    print("Reading input files...")
    timer = datetime.now()
    msd = np.loadtxt(args.INFILE)
    times = msd[1:, 0] * args.TCONV
    bins = msd[0] * args.LCONV
    msd = msd[1:, 1:] * args.LCONV**2
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
            if md[i].shape != msd.shape:
                raise ValueError(
                    "The number of displacements must be the same in all input"
                    " files"
                )
        del times_md, bins_md
        md = np.asarray(md)
        msd -= np.sum(md**2, axis=0)
        md = np.sum(md, axis=0)
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
    times = np.append(times, times[-1] + (times[-1] - times[-2]))
    dt = np.append(np.diff(times), times[-1] - times[-2])
    times -= dt / 2
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        if args.MDFILES is not None:
            # Plot mean displacement as function of initial position and
            # diffusion time
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
            pcm, cbar = mdtplt.pcolormesh_new(
                bins,  # X
                times,  # Y
                md,  # Z
                cmap="coolwarm",
                norm=colors.TwoSlopeNorm(vcenter=0),
                ax=ax1,
            )
            ax1.set(
                xlabel=r"${}(t_0)$ / {}".format(
                    args.BIN_DIRECTION, args.LUNIT
                ),
                ylabel=r"$\Delta t$ / {}".format(args.TUNIT),
                xlim=args.XLIM,
                ylim=args.YLIM,
            )
            if args.MSD_DIRECTION != "r":
                cbarlabel = (
                    r"$\langle \Delta "
                    + args.MSD_DIRECTION
                    + r"(\Delta t, "
                    + args.BIN_DIRECTION
                    + r"(t_0)) \rangle$ / "
                    + args.LUNIT
                )
            else:
                cbarlabel = (
                    r"$\langle \Delta x \rangle"
                    + r"+ \langle \Delta y \rangle"
                    + r"+ \langle \Delta z \rangle$ / "
                    + args.LUNIT
                )
            cbar.set_label(cbarlabel)
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
        # Plot MSD as function of initial position and diffusion time
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
        pcm, cbar = mdtplt.pcolormesh_new(
            bins,  # X
            times,  # Y
            msd,  # Z
            vmin=args.VLIM[0],
            vmax=args.VLIM[1],
            ax=ax1,
        )
        ax1.set(
            xlabel=r"${}(t_0)$ / {}".format(args.BIN_DIRECTION, args.LUNIT),
            ylabel=r"$\Delta t$ / {}".format(args.TUNIT),
            xlim=args.XLIM,
            ylim=args.YLIM,
        )
        if args.MDFILES is not None:
            cbarlabel = (
                r"Var$[\Delta "
                + args.MSD_DIRECTION
                + r"(\Delta t, "
                + args.BIN_DIRECTION
                + r"(t_0))]$ / "
                + args.LUNIT
                + r"$^2$"
            )
        else:
            cbarlabel = (
                r"$\langle \Delta "
                + args.MSD_DIRECTION
                + r"^2(\Delta t, "
                + args.BIN_DIRECTION
                + r"(t_0)) \rangle$ / "
                + args.LUNIT
                + r"$^2$"
            )
        cbar.set_label(cbarlabel)
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
