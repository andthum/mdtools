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
Plot in which discrete position bin a compounds resides as function of
time.

.. todo::

    Finish docstring

This script is designed to plot the output of
:mod:`scripts.discretization.discrete_pos`.

Options
-------
-f          Input filename.  The output file of
            :mod:`scripts.discretization.discrete_pos` containing the
            discrete trajectory.
--bins      Optional second input file.  The output file of
            :mod:`scripts.discretization.discrete_pos` containing the
            bin edges that were used to generate the discrete
            trajectory.  If provided, the bins will be shown on a
            secondary y-axis.
-o          Output filename.
-t          Space separated list of indices of compounds for which to
            plot the discrete trajectory.  Indexing starts at zero.
            Negative indices are counted backwards.  By default, the
            trajectories of all compounds are plotted.
--ylabel    Label for the secondary y-axis.  Is meaningless if \--bins
            is not given.  Default: ``r"$z$ / A"``.
--decs      Number of decimal places for the tick labels of the
            secondary y-axis.  Is meaningless if \--bins is not given.
            Default: ``1``.

Examples
--------
TODO
"""


__author__ = "Andreas Thum"


# Standard libraries
import sys
import os
import warnings
import argparse
from datetime import datetime, timedelta

# Third party libraries
import psutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Local application/library specific imports
import mdtools as mdt
import mdtools.plot as mdtplt  # noqa: F401; Import MDTools plot style


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Plot in which discrete position bin a compounds resides as"
            " function of time.  This script is designed to plot the output of"
            " discrete_pos.py.    For more information, refer to the"
            " documetation of this script."
        )
    )
    parser.add_argument(
        "-f",
        dest="TRJFILE",
        type=str,
        required=True,
        help=(
            "Input filename.  The output file of discrete_pos.py containing"
            " the discrete trajectory created by discrete_pos.py"
        ),
    )
    parser.add_argument(
        "--bins",
        dest="BINFILE",
        type=str,
        required=False,
        default=None,
        help=(
            "Optional second input file.  The output file of discrete_pos.py"
            " containing the bin edges that were used to generate the discrete"
            " trajectory.  If provided, the bins will be shown on a secondary"
            " y-axis."
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
        dest="TRJ_IX",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help=(
            "Space separated list of indices of compounds for which to plot"
            " the discrete trajectory.  Indexing starts at zero.  Negative"
            " indices are counted backwards.  By default, the trajectories of"
            " all compounds are plotted."
        ),
    )
    parser.add_argument(
        "--ylabel",
        dest="YLABEL",
        type=str,
        nargs="+",
        required=False,
        default=r"$z$ / A",
        help=(
            "Label for the secondary y-axis.  Is meaningless if --bins is not"
            " given.  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--decs",
        dest="DECS",
        type=int,
        required=False,
        default=1,
        help=(
            "Number of decimal places for the tick labels of the secondary"
            " y-axis.  Is meaningless if --bins is not given.  Default:"
            " %(default)s"
        ),
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

    print("\n")
    print("Loading discrete trajectory")
    timer = datetime.now()
    # TODO: Make dtrj compliant with MDTools discrete trajectories (see
    # :func:`mdtools.check.dtrj`)
    dtrj = np.load(args.TRJFILE)
    if dtrj.ndim == 1:
        n_compounds = 1
    elif dtrj.ndim == 2:
        n_compounds = dtrj.shape[0]
    else:
        raise ValueError(
            "dtrj has more than two dimensions ({})".format(dtrj.ndim)
        )
    if args.TRJ_IX is None:
        TRJ_IX = np.arange(n_compounds)
    else:
        TRJ_IX = np.asarray(args.TRJ_IX)
        pos = np.unique(TRJ_IX[TRJ_IX >= 0])
        neg = np.unique(TRJ_IX[TRJ_IX < 0])
        TRJ_IX = np.concatenate((pos, neg))
    if np.max(TRJ_IX) >= n_compounds or np.min(TRJ_IX) < -n_compounds:
        TRJ_IX = TRJ_IX[TRJ_IX < n_compounds]
        TRJ_IX = TRJ_IX[TRJ_IX >= -n_compounds]
        warnings.warn(
            "The compound indices you gave exceed the maximum number of"
            " compounds in the input file ({}).  Note that indexing starts at"
            " zero.  Set TRJ_IX to: {}".format(n_compounds, TRJ_IX),
            UserWarning,
        )
    if TRJ_IX.size == 0:
        raise ValueError("TRJ_IX is empty.  No trajectories selected")
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Generating y-axis")
    timer = datetime.now()
    states = np.arange(np.max(dtrj))
    if args.BINFILE is not None:
        print("Loading bins from {}".format(args.BINFILE), flush=True)
        try:
            bins = np.load(args.BINFILE).astype(float)
            if bins[0] != 0:
                bin_width_first = bins[0] - 0
            else:
                bin_width_first = bins[1] - bins[0]
            bin_width_last = bins[-1] - bins[-2]
            print("Start:            {:>12.6f}".format(0))
            print("Stop:             {:>12.6f}".format(bins[-1]))
            print("First bin width:  {:>12.6f}".format(bin_width_first))
            print("Last bin width:   {:>12.6f}".format(bin_width_last))
            print(
                "Equidistant bins: {:>5s}".format(
                    str(
                        np.all(np.isclose(np.diff(bins), bin_width_last))
                        and np.isclose(bin_width_first, bin_width_last)
                    )
                )
            )
            print("Number of bins:   {:>5d}".format(len(bins)))
            bin_widths = np.insert(np.diff(bins), 0, bin_width_first)
            bins -= 0.5 * bin_widths
            states = np.arange(len(bins))
        except IOError:
            warnings.warn(
                "{} not found. Will not plot secondary y-axis".format(
                    args.BINFILE
                )
            )
            args.BINFILE = None
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating plot...")
    timer = datetime.now()
    fig, ax = plt.subplots(clear=True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    offset = 0
    for i in TRJ_IX:
        ax.plot(np.arange(len(dtrj[i])) + offset, dtrj[i])
        offset += len(dtrj[i])
    ax.set(
        xlabel=r"Time / steps",
        ylabel=r"State $i$",
        xlim=(0, offset),
        ylim=(states[0] - 0.5, states[-1] + 0.5),
    )
    ax.ticklabel_format(
        axis="x", style="scientific", scilimits=(0, 0), useOffset=False
    )
    if args.BINFILE is not None:
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(dtrj[i])) + offset - len(dtrj[i]), dtrj[i])
        ax2.set(ylabel=args.YLABEL, ylim=ax.get_ylim())
        ylim = ax.get_ylim()
        yticks = ax.get_yticks().astype(int)
        yticks = yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])]
        ax2.set_yticks(yticks)
        yticklabels = np.around(bins[ax2.get_yticks()], decimals=args.DECS)
        if args.DECS == 0:
            yticklabels = yticklabels.astype(int)
        ax2.set_yticklabels(yticklabels)
    mdt.fh.backup(args.OUTFILE)
    plt.savefig(args.OUTFILE)
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
