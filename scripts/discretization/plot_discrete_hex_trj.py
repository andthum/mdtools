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


"""
Plot on which discrete hexagonal site a compounds resides as function of
time and space in a scatter plot.

.. todo::

    Finish docstring

This script is designed to plot the output of
:mod:`scripts.discretization.discrete_hex`.

Options
-------
-f          Input filename.  The output file of
            :mod:`scripts.discretization.discrete_hex` containing the
            discrete trajectory.
--lf        Second input file.  The output file of
            :mod:`scripts.discretization.discrete_hex` containing the
            hexagonal lattice faces.
--lv        Third input file.  The output file of
            :mod:`scripts.discretization.discrete_hex` containing the
            hexagonal lattice vertices.
-o          Output filename.
-i          The index of the compound for which to plot the discrete
            trajectory.  Indexing starts at zero.  Default: The compound
            that has the fewest negative elements in its discrete
            trajectory.
--every     Plot only every n-th frame.  Default: ``1``.
--msize     If given, the size of the scatter points will be
            proportional to the number of times the compound visited the
            respective lattice face.
--xlim      Left and right limit of the x-axis in data coordinates.
            Pass 'None' to adjust the limit(s) automatically.  Default:
            ``[0, None]``.
--ylim      Lower and upper limit of the y-axis in data coordinates.
            Pass 'None' to adjust the limit(s) automatically.  Default:
            ``[0, None]``.
--length-conv
            Multiply all lengths by this factor.  Default:  ``1``.
--length-unit
            Lengh unit.  Default: ``A``.

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
from matplotlib.ticker import MaxNLocator

# Local application/library specific imports
import mdtools as mdt
import mdtools.plot as mdtplt

plt.rc("lines", markersize=3, markeredgewidth=0.5)
plt.rc("markers", fillstyle="full")


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Plot on which discrete hexagonal site a compounds resides as"
            " function of time and space in a scatter plot.  This script is"
            " designed to plot the output of discrete_hex.py.    For more"
            " information, refer to the documetation of this script."
        )
    )
    parser.add_argument(
        "-f",
        dest="TRJFILE",
        type=str,
        required=True,
        help=(
            "Input filename.  The output file of discrete_hex.py containing"
            " the discrete trajectory."
        ),
    )
    parser.add_argument(
        "--lf",
        dest="LATFACE",
        type=str,
        required=True,
        help=(
            "Second input file.  The output file of discrete_hex.py containing"
            " the hexagonal lattice faces."
        ),
    )
    parser.add_argument(
        "--lv",
        dest="LATVERT",
        type=str,
        required=True,
        help=(
            "Third input file.  The output file of discrete_hex.py containing"
            " the hexagonal lattice vertices."
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
        "-i",
        dest="IX",
        type=int,
        required=False,
        default=None,
        help=(
            "The index of the compound for which to plot the discrete"
            " trajectory.  Indexing starts at zero.  Default: The compound"
            " that has the fewest negative elements in its discrete"
            " trajectory."
        ),
    )
    parser.add_argument(
        "--every",
        dest="EVERY",
        type=int,
        required=False,
        default=1,
        help=("Plot only every n-th frame.  Default: %(default)s"),
    )
    parser.add_argument(
        "--msize",
        dest="MSIZE",
        required=False,
        default=False,
        action="store_true",
        help=(
            "If given, the size of the scatter points will be proportional to"
            " the number of times the compound visited the respective lattice"
            " face."
        ),
    )
    parser.add_argument(
        "--xlim",
        dest="XLIM",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=float),
        nargs=2,
        required=False,
        default=[0, None],
        help=(
            "Left and right limit of the x-axis in data coordinates.  Default:"
            " %(default)s"
        ),
    )
    parser.add_argument(
        "--ylim",
        dest="YLIM",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=float),
        nargs=2,
        required=False,
        default=[0, None],
        help=(
            "Lower and upper limit of the y-axis in data coordinates."
            "  Default: %(default)s"
        ),
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

    print("\n")
    print("Reading input files...")
    timer = datetime.now()
    dtrj = mdt.fh.load_dtrj(args.TRJFILE)
    latfaces = np.load(args.LATFACE)
    latverts = np.load(args.LATVERT)
    latfaces *= args.LCONV
    latverts *= args.LCONV
    if args.IX is None:
        args.IX = np.argmax(np.count_nonzero(dtrj > 0, axis=1))
        print("Chosen compound {}".format(args.IX))
    if args.IX < 0:
        raise ValueError(
            "The compound index ({}) is less than zero".format(args.IX)
        )
    elif args.IX > len(dtrj) - 1:
        raise ValueError(
            "The compound index ({}) is out of range".format(args.IX)
        )
    dtrj = dtrj[args.IX]
    valid = dtrj > 0
    if not np.any(valid):
        raise ValueError(
            "The selected compound is never in the [ZMIN; ZMAX) interval that"
            " was used while generating the discrete trajectory"
        )
    n_frames = len(dtrj)
    frames = np.arange(n_frames)
    if args.MSIZE:
        u, ix, c = np.unique(dtrj, return_inverse=True, return_counts=True)
        markersize = c[ix]
        del u, ix, c
    else:
        markersize = None
    dtrj = dtrj[valid][:: args.EVERY]
    frames = frames[valid][:: args.EVERY]
    if markersize is not None:
        markersize = markersize[valid][:: args.EVERY]
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating plot...")
    timer = datetime.now()
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        # Plot lattice sites as function of time (frame number)
        fig, ax = plt.subplots(clear=True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.ticklabel_format(
            axis="x", style="scientific", scilimits=(0, 0), useOffset=False
        )
        ax.scatter(x=frames, y=dtrj, marker=".")
        ax.set(
            xlabel="Frame",
            ylabel="Lattice site",
            xlim=(0, n_frames),
            ylim=(np.nanmin(dtrj), np.nanmax(dtrj)),
        )
        pdf.savefig()
        plt.close()
        # Plot lattice site as function of space and time
        fig, ax = plt.subplots(clear=True)
        ax.scatter(
            x=latverts[:, 0], y=latverts[:, 1], marker=".", color="black"
        )
        img = ax.scatter(
            x=latfaces[dtrj][:, 0],
            y=latfaces[dtrj][:, 1],
            s=markersize,
            c=frames,
            marker="o",
            vmin=0,
            vmax=n_frames,
        )
        ax.set(
            xlabel=r"$x$ / {}".format(args.LUNIT),
            ylabel=r"$y$ / {}".format(args.LUNIT),
            xlim=args.XLIM,
            ylim=args.YLIM,
        )
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Frame")
        cbar.ax.tick_params(which="both", direction="out")
        cbar.ax.ticklabel_format(
            axis="y", style="scientific", scilimits=(0, 0), useOffset=False
        )
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        cbar.ax.yaxis.offsetText.set_horizontalalignment(
            mdtplt.CBAR_YAX_HALIGN
        )
        cbar.ax.yaxis.offsetText.set_verticalalignment(mdtplt.CBAR_YAX_VALIGN)
        xsize = abs(ax.get_xlim()[0] - ax.get_xlim()[1])
        ysize = abs(ax.get_ylim()[0] - ax.get_ylim()[1])
        ax.set_aspect(ysize / xsize)
        yticks = np.asarray(ax.get_yticks())
        mask = (yticks >= ax.get_xlim()[0]) & (yticks <= ax.get_xlim()[1])
        ax.set_xticks(yticks[mask])
        pdf.savefig(bbox_inches="tight")
        plt.close()
        # Lattice site numbering
        fig, ax = plt.subplots(clear=True)
        ax.scatter(
            x=latverts[:, 0], y=latverts[:, 1], marker=".", color="black"
        )
        ax.set(
            xlabel=r"$x$ / {}".format(args.LUNIT),
            ylabel=r"$y$ / {}".format(args.LUNIT),
            xlim=args.XLIM,
            ylim=args.YLIM,
        )
        latface_num = np.arange(len(latfaces))
        for i, txt in enumerate(latface_num):
            ax.annotate(
                txt,
                xy=(latfaces[:, 0][i], latfaces[:, 1][i]),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=6,
            )
        xsize = abs(ax.get_xlim()[0] - ax.get_xlim()[1])
        ysize = abs(ax.get_ylim()[0] - ax.get_ylim()[1])
        ax.set_aspect(ysize / xsize)
        yticks = np.array(ax.get_yticks())
        mask = (yticks >= ax.get_xlim()[0]) & (yticks <= ax.get_xlim()[1])
        ax.set_xticks(yticks[mask])
        pdf.savefig(bbox_inches="tight")
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
