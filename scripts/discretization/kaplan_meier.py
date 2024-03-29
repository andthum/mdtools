#!/usr/bin/env python3

# This file is part of MDTools.
# Copyright (C) 2021-2023  The MDTools Development Team and all
# contributors listed in the file AUTHORS.rst
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
Calculate the state survival function using the Kaplan-Meier estimator.

Given that a state transition occurred at time :math:`t_0`, calculate
the probability that a compound is still in the new state at time
:math:`t_0 + \Delta t`.

Options
-------
-f
    Name of the file containing the discrete trajectory.  The discrete
    trajectory must be stored as :class:`numpy.ndarray` either in a
    binary NumPy |npy_file| or in a (compressed) NumPy |npz_archive|.
    See :func:`mdtools.file_handler.load_dtrj` for more information
    about the requirements for the input file.
-o
    Output filename.
-b
    First frame to read from the discrete trajectory.  Frame numbering
    starts at zero.  Default: ``0``.
-e
    Last frame to read from the discrete trajectory.  This is exclusive,
    i.e. the last frame read is actually ``END - 1``.  A value of ``-1``
    means to read the very last frame.  Default: ``-1``.
--every
    Read every n-th frame from the discrete trajectory.  Default: ``1``.
--intermittency
    Maximum number of frames a compound is allowed to leave its state
    whilst still being considered to be in this state provided that it
    returns to this state after the given number of frames.  In other
    words, a compound is only considered to have left its state if it
    has left it for at least the given number of frames.
--discard-neg-start
    If provided, discard state leavings starting from negative states.
    Transitions from positive to negative states are regarded as proper
    state leaving.
--discard-all-neg
    If provided, discard all state leavings starting from or ending in a
    negative state.  This is different to \--discard-neg-start in the
    sense that transitions from positive to negative states are treated
    as censored.

See Also
--------
:func:`mdtools.dtrj.kaplan_meier` :
    The underlying function that calculates the Kaplan-Meier estimate of
    the survival function
:mod:`scripts.discretization.kaplan_meier_discrete` :
    Calculate the state survival function using the Kaplan-Meier
    estimator resolved with respect to the states in the second discrete
    trajectory

Notes
-----
For more information about the survival function and the Kaplan-Meier
estimator refer to :func:`mdtools.dtrj.kaplan_meier`.
"""


__author__ = "Andreas Thum"


# Standard libraries
import argparse
import os
import sys
from datetime import datetime, timedelta

# Third-party libraries
import numpy as np
import psutil

# First-party libraries
import mdtools as mdt


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage.
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the state survival function using the Kaplan-Meier"
            " estimator.  For more information, refer to the documentation of"
            " this script."
        )
    )
    parser.add_argument(
        "-f",
        dest="TRJFILE",
        type=str,
        required=True,
        help=(
            "File containing the discrete trajectory stored as numpy.ndarray"
            " in the binary .npy format or as .npz archive."
        ),
    )
    parser.add_argument(
        "-o",
        dest="OUTFILE",
        type=str,
        required=True,
        help="Output filename.",
    )
    parser.add_argument(
        "-b",
        dest="BEGIN",
        type=int,
        required=False,
        default=0,
        help=(
            "First frame to read from the trajectory.  Frame numbering starts"
            " at zero.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "-e",
        dest="END",
        type=int,
        required=False,
        default=-1,
        help=(
            "Last frame to read from the trajectory (exclusive).  Default:"
            " %(default)s."
        ),
    )
    parser.add_argument(
        "--every",
        dest="EVERY",
        type=int,
        required=False,
        default=1,
        help=(
            "Read every n-th frame from the trajectory.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--intermittency",
        dest="INTERMITTENCY",
        type=int,
        required=False,
        default=0,
        help=(
            "Maximum number of frames a compound is allowed to leave its state"
            " whilst still being considered to be in this state provided that"
            " it returns to this state after the given number of frames."
        ),
    )
    parser.add_argument(
        "--discard-neg-start",
        dest="DISCARD_NEG_START",
        required=False,
        default=False,
        action="store_true",
        help=(
            "If provided, discard state leavings starting from negative"
            " states."
        ),
    )
    parser.add_argument(
        "--discard-all-neg",
        dest="DISCARD_ALL_NEG",
        required=False,
        default=False,
        action="store_true",
        help=(
            "If provided, discard all state leavings starting from or ending"
            " in a negative state."
        ),
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    if args.INTERMITTENCY < 0:
        raise ValueError(
            "--intermittency ({}) must be equal to or greater than"
            " zero".format(args.INTERMITTENCY)
        )

    print("\n")
    print("Loading trajectory...")
    timer = datetime.now()
    dtrj = mdt.fh.load_dtrj(args.TRJFILE)
    N_CMPS, N_FRAMES_TOT = dtrj.shape
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=N_FRAMES_TOT,
    )
    dtrj = dtrj[:, BEGIN:END:EVERY]
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    if args.INTERMITTENCY > 0:
        print("\n")
        print("Correcting for intermittency...")
        dtrj = mdt.dyn.correct_intermittency(
            dtrj.T, args.INTERMITTENCY, inplace=True, verbose=True
        )
        dtrj = dtrj.T
    trans_info_str = mdt.rti.dtrj_trans_info_str(dtrj)

    print("\n")
    print("Calculating survival function...")
    print("Number of compounds:    {:>8d}".format(N_CMPS))
    print("Total number of frames: {:>8d}".format(N_FRAMES_TOT))
    print("Frames to read:         {:>8d}".format(N_FRAMES))
    print("First frame to read:    {:>8d}".format(BEGIN))
    print("Last frame to read:     {:>8d}".format(END - 1))
    print("Read every n-th frame:  {:>8d}".format(EVERY))
    timer = datetime.now()
    sf, sf_var = mdt.dtrj.kaplan_meier(
        dtrj,
        discard_neg_start=args.DISCARD_NEG_START,
        discard_all_neg=args.DISCARD_ALL_NEG,
        verbose=True,
    )
    del dtrj
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    header = (
        "Kaplan-Meier estimate of the survival function:  Probability that a\n"
        + "compound is still in the new state at time t0+dt given that a\n"
        + "state transition has occurred at time t0.\n"
        + "\n\n"
        + "intermittency:     {}\n".format(args.INTERMITTENCY)
        + "discard_neg_start: {}\n".format(args.DISCARD_NEG_START)
        + "discard_all_neg:   {}\n".format(args.DISCARD_ALL_NEG)
        + "\n\n"
    )
    header += trans_info_str
    header += "\n\n"
    header += (
        "The columns contain:\n"
        + "  1 Lag time (in trajectory steps)\n"
        + "  2 Survival function\n"
        + "  3 Variance of the survival function (Greenwood's formula)\n"
        + "\n"
        + "Column number:\n"
        + "{:>14d} {:>16d} {:>16d}".format(1, 2, 3)
    )
    lag_times = np.arange(0, len(sf) * EVERY, EVERY, dtype=np.uint32)
    data = np.column_stack([lag_times, sf, sf_var])
    mdt.fh.savetxt(args.OUTFILE, data, header=header)
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
