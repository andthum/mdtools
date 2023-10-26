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
Calculate the state survival function using the Kaplan-Meier estimator
resolved with respect to the states in the second discrete trajectory.

Given that a state transition occurred at time :math:`t_0`, calculate
the probability that a compound is still in the new state at time
:math:`t_0 + \Delta t` given that the compound was in a specific state
of another discrete trajectory at time :math:`t_0`.

Options
-------
--f1
    Name of the file containing the first discrete trajectory.  The
    discrete trajectory must be stored as :class:`numpy.ndarray` either
    in a binary NumPy |npy_file| or in a (compressed) NumPy
    |npz_archive|.  See :func:`mdtools.file_handler.load_dtrj` for more
    information about the requirements for the input file.
--f2
    Name of the file containing the second discrete trajectory.  The
    second discrete trajectory must have the same shape as the first
    discrete trajectory.
--o-sf
    Output filename for the file containing the values of the survival
    function.
--o-var
    Output filename for the file containing the corresponding variance
    values.
-b
    First frame to read from the discrete trajectory.  Frame numbering
    starts at zero.  Default: ``0``.
-e
    Last frame to read from the discrete trajectory.  This is exclusive,
    i.e. the last frame read is actually ``END - 1``.  A value of ``-1``
    means to read the very last frame.  Default: ``-1``.
--every
    Read every n-th frame from the discrete trajectory.  Default: ``1``.
--intermittency1
    Allowed intermittency for the first discrete trajectory:  Maximum
    number of frames a compound is allowed to leave its state whilst
    still being considered to be in this state provided that it returns
    to this state after the given number of frames.  In other words, a
    compound is only considered to have left its state if it has left it
    for at least the given number of frames.
--intermittency2
    Allowed intermittency for the second discrete trajectory.
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
:func:`mdtools.dtrj.kaplan_meier_discrete` :
    The underlying function that calculates the Kaplan-Meier estimate of
    the survival function
:mod:`scripts.discretization.kaplan_meier` :
    Calculate the state survival function using the Kaplan-Meier
    estimator

Notes
-----
For more information about the survival function and the Kaplan-Meier
estimator refer to :func:`mdtools.dtrj.kaplan_meier`.

If you parse the same discrete trajectory to \--f1 and \--f2 you will
get the survival function for each individual state of the input
trajectory.
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
            " estimator resolved with respect to the states in the second"
            " discrete trajectory.  For more information, refer to the"
            " documentation of this script."
        )
    )
    parser.add_argument(
        "--f1",
        dest="TRJFILE1",
        type=str,
        required=True,
        help=(
            "File containing the first discrete trajectory stored as"
            " numpy.ndarray in the binary .npy format or as .npz archive."
        ),
    )
    parser.add_argument(
        "--f2",
        dest="TRJFILE2",
        type=str,
        required=True,
        help=(
            "File containing the second discrete trajectory stored as"
            " numpy.ndarray in the binary .npy format or as .npz archive."
        ),
    )
    parser.add_argument(
        "--o-sf",
        dest="OUTFILE_SF",
        type=str,
        required=True,
        help=(
            "Output filename for the file containing the values of the"
            " survival function."
        ),
    )
    parser.add_argument(
        "--o-var",
        dest="OUTFILE_VAR",
        type=str,
        required=True,
        help=(
            "Output filename for the file containing the corresponding"
            " variance values."
        ),
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
        "--intermittency1",
        dest="INTERMITTENCY1",
        type=int,
        required=False,
        default=0,
        help=(
            "Allowed intermittency for the first discrete trajectory:  Maximum"
            " number of frames a compound is allowed to leave its state whilst"
            " still being considered to be in this state provided that it"
            " returns to this state after the given number of frames."
        ),
    )
    parser.add_argument(
        "--intermittency2",
        dest="INTERMITTENCY2",
        type=int,
        required=False,
        default=0,
        help="Allowed intermittency for the second discrete trajectory",
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
    if args.INTERMITTENCY1 < 0:
        raise ValueError(
            "--intermittency1 ({}) must be equal to or greater than"
            " zero".format(args.INTERMITTENCY1)
        )
    if args.INTERMITTENCY2 < 0:
        raise ValueError(
            "--intermittency2 ({}) must be equal to or greater than"
            " zero".format(args.INTERMITTENCY2)
        )

    print("\n")
    print("Loading trajectory...")
    timer = datetime.now()
    dtrj1 = mdt.fh.load_dtrj(args.TRJFILE1)
    dtrj2 = mdt.fh.load_dtrj(args.TRJFILE2)
    N_CMPS, N_FRAMES_TOT = dtrj1.shape
    if dtrj1.shape != dtrj2.shape:
        raise ValueError("Both trajectories must have the same shape")
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=N_FRAMES_TOT,
    )
    dtrj1 = dtrj1[:, BEGIN:END:EVERY]
    dtrj2 = dtrj2[:, BEGIN:END:EVERY]
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    if args.INTERMITTENCY1 > 0:
        print("\n")
        print("Correcting the first discrete trajectory for intermittency...")
        dtrj1 = mdt.dyn.correct_intermittency(
            dtrj1.T, args.INTERMITTENCY1, inplace=True, verbose=True
        )
        dtrj1 = dtrj1.T
    dtrj1_trans_info_str = mdt.rti.dtrj_trans_info_str(dtrj1)
    if args.INTERMITTENCY2 > 0:
        print("\n")
        print("Correcting the second discrete trajectory for intermittency...")
        dtrj2 = mdt.dyn.correct_intermittency(
            dtrj2.T, args.INTERMITTENCY2, inplace=True, verbose=True
        )
        dtrj2 = dtrj2.T
    dtrj2_states = np.unique(dtrj2)

    print("\n")
    print("Calculating survival function...")
    print("Number of compounds:    {:>8d}".format(N_CMPS))
    print("Total number of frames: {:>8d}".format(N_FRAMES_TOT))
    print("Frames to read:         {:>8d}".format(N_FRAMES))
    print("First frame to read:    {:>8d}".format(BEGIN))
    print("Last frame to read:     {:>8d}".format(END - 1))
    print("Read every n-th frame:  {:>8d}".format(EVERY))
    timer = datetime.now()
    sf, sf_var = mdt.dtrj.kaplan_meier_discrete(
        dtrj1,
        dtrj2,
        discard_neg_start=args.DISCARD_NEG_START,
        discard_all_neg=args.DISCARD_ALL_NEG,
        verbose=True,
    )
    del dtrj1, dtrj2
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    # Kaplan-Meier estimate of the survival function.
    header = (
        "Kaplan-Meier estimate of the survival function:  Probability that a\n"
        + "compound is still in the new state at time t0+dt given that a\n"
        + "state transition has occurred at time t0 resolved with respect to\n"
        + "the states in a second discrete trajectory.\n"
        + "\n\n"
        + "intermittency_1:   {}\n".format(args.INTERMITTENCY1)
        + "intermittency_2:   {}\n".format(args.INTERMITTENCY2)
        + "discard_neg_start: {}\n".format(args.DISCARD_NEG_START)
        + "discard_all_neg:   {}\n".format(args.DISCARD_ALL_NEG)
        + "\n\n"
    )
    header += dtrj1_trans_info_str
    header += "\n\n"
    header += (
        "The first column contains the lag times (in trajectory steps).\n"
        "The first row contains the states of the second discrete trajectory\n"
        "that were used to discretize the survival function.\n"
        "The remaining matrix elements are the values of the survival\n"
        "function.\n"
    )
    lag_times = np.arange(0, sf.shape[1] * EVERY, EVERY, dtype=np.uint32)
    mdt.fh.savetxt_matrix(
        args.OUTFILE_SF, sf.T, var1=lag_times, var2=dtrj2_states, header=header
    )
    print("Created {}".format(args.OUTFILE_SF))

    # Variance of the Kaplan-Meier estimate.
    header = (
        "Variance of the Kaplan-Meier estimate of the survival function\n"
        + "according to Greenwood's formula\n"
        + "\n\n"
        + "intermittency_1:   {}\n".format(args.INTERMITTENCY1)
        + "intermittency_2:   {}\n".format(args.INTERMITTENCY2)
        + "discard_neg_start: {}\n".format(args.DISCARD_NEG_START)
        + "discard_all_neg:   {}\n".format(args.DISCARD_ALL_NEG)
        + "\n\n"
    )
    header += dtrj1_trans_info_str
    header += "\n\n"
    header += (
        "The first column contains the lag times (in trajectory steps).\n"
        "The first row contains the states of the second discrete trajectory\n"
        "that were used to discretize the survival function.\n"
        "The remaining matrix elements are the variance values of the\n"
        "survival function.\n"
    )
    mdt.fh.savetxt_matrix(
        args.OUTFILE_VAR,
        sf_var.T,
        var1=lag_times,
        var2=dtrj2_states,
        header=header,
    )
    print("Created {}".format(args.OUTFILE_VAR))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
