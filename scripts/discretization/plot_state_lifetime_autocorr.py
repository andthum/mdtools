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
Plot the lifetime autocorrelation function of discrete states.

This script is designed to plot the output of
:mod:`scripts.discretization.state_lifetime`.

.. todo::

    Finish docstring

Options
-------
-f          Input filename.  The output file of
            :mod:`scripts.discretization.state_lifetime`.
-o          Output filename.
--xlim      Left and right limit of the x-axis in data coordinates.  If
            the right limit is ``None``, plot until the maximum lag
            time.  Default: ``[0, None]``.
--ylim      Lower and upper limit of the y-axis in data coordinates.
            Pass 'None' to adjust the limit(s) automatically.  Default:
            ``[0, 1]``.
--time-conv
            Multiply all times by this factor.  Default: ``1``.
--time-unit
            Time unit.  Default: ``'steps'``.

See Also
--------
:mod:`plot_state_lifetime_discrete` : TODO

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
            "Plot the lifetime autocorrelation function of discrete states."
            "  This script is designed to plot the output of"
            " state_lifetime.py.  For more information, refer to the"
            " documetation of this script."
        )
    )
    parser.add_argument(
        "-f",
        dest="INFILE",
        type=str,
        required=True,
        help=("Input filename.  The output file of state_lifetime.py."),
    )
    parser.add_argument(
        "-o",
        dest="OUTFILE",
        type=str,
        required=True,
        help=("Output filename."),
    )
    parser.add_argument(
        "--xlim",
        dest="XLIM",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=float),
        nargs=2,
        required=False,
        default=[0, None],
        help=(
            "Left and right limit of the x-axis in data coordinates.  If the"
            " right limit is None, plot until the maximum lag time.  Default:"
            " %(default)s"
        ),
    )
    parser.add_argument(
        "--ylim",
        dest="YLIM",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=float),
        nargs=2,
        required=False,
        default=[0, 1],
        help=(
            "Lower and upper limit of the y-axis in data coordinates."
            "  Default: %(default)s"
        ),
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
        default="steps",
        help=("Time unit.  Default: %(default)s"),
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

    print("\n")
    print("Reading input file...")
    timer = datetime.now()
    times, autocorr, fit = np.loadtxt(fname=args.INFILE, unpack=True)
    times *= args.TCONV
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating plot...")
    timer = datetime.now()
    if args.XLIM[0] is None:
        args.XLIM[0] = np.nanmax(times)
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        # Autocorrelation function vs lag time
        fig, ax = plt.subplots(clear=True)
        ax.plot(times, autocorr, linestyle="", marker="x")
        mdt.plot.plot(times, fit, label="Fit")
        ax.set(
            xlim=args.XLIM,
            ylim=args.YLIM,
            xlabel=r"$\Delta t$ / " + args.TUNIT,
            ylabel=r"$C(\Delta t)$",
        )
        pdf.savefig()
        plt.close()
        # ln(Autocorrelation function) vs lag time
        fig, ax = plt.subplots(clear=True)
        mask = autocorr > 0
        mask_fit = fit > 0
        if args.YLIM[0] < 0:
            ymin = args.YLIM[0]
        else:
            ymin = min(
                np.nanmin(np.log(autocorr[mask])),
                np.nanmin(np.log(fit[mask_fit])),
            )
        ymax = args.YLIM[1] if args.YLIM[1] <= 0 else 0
        ax.plot(times[mask], np.log(autocorr[mask]), linestyle="", marker="x")
        ax.plot(times[mask_fit], np.log(fit[mask_fit]), label="Fit")
        ax.set(
            xlim=args.XLIM,
            ylim=(ymin, ymax),
            xlabel=r"$\Delta t$ / " + args.TUNIT,
            ylabel=r"$\ln{C(\Delta t)}$",
        )
        pdf.savefig()
        plt.close()
        # Autocorrelation function vs log(lag time)
        fig, ax = plt.subplots(clear=True)
        mask = times > 0
        xmin = args.XLIM[0] if args.XLIM[0] > 0 else np.nanmin(times[mask])
        xmax = args.XLIM[1] if args.XLIM[1] > 0 else np.nanmax(times[mask])
        ax.plot(times, autocorr, linestyle="", marker="x")
        ax.plot(times, fit, label="Fit")
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set(
            xlim=(xmin, xmax),
            ylim=args.YLIM,
            ymax=args.YMAX,
            xlabel=r"$\Delta t$ / " + args.TUNIT,
            ylabel=r"$C(\Delta t)$",
        )
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
