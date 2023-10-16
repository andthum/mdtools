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
Calculate the survival function and the back-jump probability.

Calculate the probability to be (continuously) in the same state as
directly after a state transition and the probability to return to the
state directly before the state transition as function of the time that
passed since the transition.

Given that at time :math:`t_0` a state transition occurred, compute the
probability that a compound

    * is at time :math:`t_0 + \Delta t` in the same state as directly
      after the state transition ('prob_a_as_a').  This probability
      might be regarded as a a "discontinuous" survival function.
    * is at time :math:`t_0 + \Delta t` in the same state as directly
      before the state transition ('prob_a_as_b').
    * is from time :math:`t_0` until :math:`t_0 + \Delta t`
      *continuously* in the same state ('prob_a_as_a_con').  This
      probability is the survival function of the underlying
      distribution of state lifetimes.
    * is at time :math:`t_0 + \Delta t` in the same state as directly
      before the state transition under the condition that it has
      *continuously* been in the same state as directly after the state
      transition from time :math:`t_0` until :math:`t_0 + \Delta t`
      ('prob_a_as_b_con').

    * returns at time :math:`t_0 + \Delta t` back to the same state as
      directly before the state transition ('prob_back').  This
      probability might be regarded as a a "discontinuous" back-jump
      probability.
    * returns at time :math:`t_0 + \Delta t` back to the same state as
      directly before the state transition under the condition that it
      has *continuously* been in the same state as directly after the
      state transition from time :math:`t_0` until
      :math:`t_0 + \Delta t` ('prob_back_con').  This is the probability
      for back jumps.

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


See Also
--------
:mod:`scripts.structure.lig_change_at_pos_change_blocks` :
    Compare the coordination environment of reference compounds before
    and after they have changed their position
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Calculate the probability to be in the same state as directly"
            " before or after a state transition as function of time passed"
            " since the transition.  For more information, refer to the"
            " documentation of this script."
        ),
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
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

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
    trans_info_str = mdt.rti.dtrj_trans_info_str(dtrj)
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    # Given that at time t0 a state transition occurred...
    #
    # `prob_b_as_b`: Probability that a compound is at time t0-dt in the
    #     same state as directly before the state transition.
    # `prob_b_as_a`: Probability that a compound is at time t0-dt in the
    #     same state as directly after the state transition.
    # `prob_b_as_b_con`: Probability that a compound is from time t0-dt
    #     until t0 *continuously* in the same state.
    # `prob_b_as_a_con`: Probability that a compound is at time t0-dt in
    #     the same state as directly after the state transition under
    #     the condition that it has *continuously* been in the same
    #     state as directly before the state transition from time t0-dt
    #     until t0.
    #
    # `prob_a_as_a`: Probability that a compound is at time t0+dt in the
    #     same state as directly after the state transition.
    # `prob_a_as_b`: Probability that a compound is at time t0+dt in the
    #     same state as directly before the state transition.
    # `prob_a_as_a_con`: Probability that a compound is from time t0
    #     until t0+dt *continuously* in the same state.
    # `prob_a_as_b_con`: Probability that a compound is at time t0+dt in
    #     the same state as directly before the state transition under
    #     the condition that it has *continuously* been in the same
    #     state as directly after the state transition from time t0
    #     until t0+dt.
    #
    # `prob_back`: Probability that a compound returns at time t0+dt
    #     back to the same state as directly before the state
    #     transition.
    # `prob_back_con`: Probability that a compound returns at time t0+dt
    #     back to the same state as directly before the state transition
    #     under the condition that it has *continuously* been in the
    #     same state as directly after the state transition from time t0
    #     until t0+dt.
    prob_b_as_b = np.zeros(N_FRAMES - 1, dtype=np.uint32)
    prob_b_as_a = np.zeros_like(prob_b_as_b)
    prob_b_as_b_con = np.zeros_like(prob_b_as_b)
    prob_b_as_a_con = np.zeros_like(prob_b_as_b)
    norm_b = np.zeros_like(prob_b_as_b)

    prob_a_as_a = np.zeros_like(prob_b_as_b)
    prob_a_as_b = np.zeros_like(prob_a_as_a)
    prob_a_as_a_con = np.zeros_like(prob_a_as_a)
    prob_a_as_b_con = np.zeros_like(prob_a_as_a)
    norm_a = np.zeros_like(prob_a_as_a)

    prob_back = np.zeros_like(prob_b_as_b)
    prob_back_con = np.zeros_like(prob_back)
    norm_back = np.zeros_like(prob_back)

    print("\n")
    print("Reading trajectory...")
    print("Number of compounds:    {:>8d}".format(N_CMPS))
    print("Total number of frames: {:>8d}".format(N_FRAMES_TOT))
    print("Frames to read:         {:>8d}".format(N_FRAMES))
    print("First frame to read:    {:>8d}".format(BEGIN))
    print("Last frame to read:     {:>8d}".format(END - 1))
    print("Read every n-th frame:  {:>8d}".format(EVERY))
    timer = datetime.now()
    dtrj = mdt.rti.ProgressBar(dtrj, unit="compounds")
    for cmp_trj in dtrj:  # Loop over single compound trajectories
        # Frames directly *before* state transitions.
        trans = np.flatnonzero(np.diff(cmp_trj))
        # Frames directly *after* state transitions.
        trans += 1
        norm_back += len(trans)
        for t0 in trans:
            # Trajectory before the state transition, time reversed.
            cmp_trj_b = cmp_trj[t0 - 1 :: -1]
            # Trajectory after the state transition.
            cmp_trj_a = cmp_trj[t0:]

            # Calculate `prob_b_as_b` and `prob_b_as_b_con`.
            max_lag = len(cmp_trj_b)
            norm_b[:max_lag] += 1
            same_t0 = cmp_trj_b == cmp_trj_b[0]
            prob_b_as_b[:max_lag] += same_t0
            same_t0_con = same_t0  # This is a view, not a copy!
            ix_trans = np.argmin(same_t0_con)
            ix_trans = ix_trans if ix_trans > 0 else len(same_t0_con)
            same_t0_con[ix_trans:] = False
            prob_b_as_b_con[:max_lag] += same_t0_con
            # Calculate `prob_b_as_a` and `prob_b_as_a_con`.
            same_t0 = cmp_trj_b == cmp_trj_a[0]
            prob_b_as_a[:max_lag] += same_t0
            ix_trans = np.argmin(same_t0_con ^ same_t0)
            same_t0[ix_trans:] = False
            prob_b_as_a_con[:max_lag] += same_t0

            # Calculate `prob_a_as_a` and `prob_a_as_a_con`.
            max_lag = len(cmp_trj_a)
            norm_a[:max_lag] += 1
            same_t0 = cmp_trj_a == cmp_trj_a[0]
            prob_a_as_a[:max_lag] += same_t0
            same_t0_con = same_t0  # This is a view, not a copy!
            ix_trans = np.argmin(same_t0_con)
            ix_trans = ix_trans if ix_trans > 0 else len(same_t0_con)
            same_t0_con[ix_trans:] = False
            prob_a_as_a_con[:max_lag] += same_t0_con
            # Calculate `prob_a_as_b`, `prob_a_as_b_con`, `prob_back`
            # and `prob_back_con`.
            same_t0 = cmp_trj_a == cmp_trj_b[0]
            prob_a_as_b[:max_lag] += same_t0
            ix_back = np.argmax(same_t0)
            if ix_back > 0:
                prob_back[ix_back] += 1
            ix_trans = np.argmin(same_t0_con ^ same_t0)
            same_t0[ix_trans:] = False
            prob_a_as_b_con[:max_lag] += same_t0
            ix_back_con = np.argmax(same_t0)
            if ix_back_con > 0:
                prob_back_con[ix_back_con] += 1

        # ProgressBar update:
        progress_bar_mem = proc.memory_info().rss / 2**20
        dtrj.set_postfix_str(
            "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
        )
    dtrj.close()
    del same_t0, same_t0_con
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    prob_b_as_b = prob_b_as_b / norm_b
    prob_b_as_a = prob_b_as_a / norm_b
    prob_b_as_b_con = prob_b_as_b_con / norm_b
    prob_b_as_a_con = prob_b_as_a_con / norm_b
    del norm_b

    prob_a_as_a = prob_a_as_a / norm_a
    prob_a_as_b = prob_a_as_b / norm_a
    prob_a_as_a_con = prob_a_as_a_con / norm_a
    prob_a_as_b_con = prob_a_as_b_con / norm_a
    del norm_a

    prob_back = prob_back / norm_back
    prob_back_con = prob_back_con / norm_back
    del norm_back

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    lag_times = np.arange(len(prob_b_as_b), dtype=np.uint32)
    data = np.column_stack(
        [
            lag_times,  # 1
            #
            prob_b_as_b,  # 2
            prob_b_as_a,  # 3
            prob_b_as_b_con,  # 4
            prob_b_as_a_con,  # 5
            #
            prob_a_as_a,  # 6
            prob_a_as_b,  # 7
            prob_a_as_a_con,  # 8
            prob_a_as_b_con,  # 9
            #
            prob_back,  # 10
            prob_back_con,  # 11
        ]
    )
    header = (
        "Probability to be in the same state a certain time before and after\n"
        "a state transition.\n"
        "\n"
        "\n" + trans_info_str + "\n"
        "\n"
        "Given that at time t0 a state transition occurred...\n"
        "The columns contain:\n"
        "  1 Lag time dt (in trajectory steps)\n"
        "\n"
        "  2 prob_b_as_b: Probability that a compound is at time t0-dt in\n"
        "      the same state as directly before the state transition\n"
        "  3 prob_b_as_a: Probability that a compound is at time t0-dt in\n"
        "      the same state as directly after the state transition\n"
        "  4 prob_b_as_b_con: Probability that a compound is from time t0-dt\n"
        "      until t0 *continuously* in the same state\n"
        "  5 prob_b_as_a_con: Probability that a compound is at time t0-dt\n"
        "      in the same state as directly after the state transition\n"
        "      under the condition that it has *continuously* been in the\n"
        "      same state as directly before the state transition from time\n"
        "      t0-dt until t0\n"
        "\n"
        "  6 prob_a_as_a: Probability that a compound is at time t0+dt in\n"
        "      the same state as directly after the state transition\n"
        "  7 prob_a_as_b: Probability that a compound is at time t0+dt in\n"
        "      the same state as directly before the state transition\n"
        "  8 prob_a_as_a_con: Probability that a compound is from time t0\n"
        "      until t0+dt *continuously* in the same state\n"
        "  9 prob_a_as_b_con: Probability that a compound is at time t0+dt\n"
        "      in the same state as directly before the state transition\n"
        "      under the condition that it has *continuously* been in the\n"
        "      same state as directly after the state transition during the\n"
        "      time t0+dt\n"
        "\n"
        " 10 prob_back`: Probability that a compound returns at time t0+dt\n"
        "      back to the same state as directly before the state\n"
        "      transition\n"
        " 11 prob_back_con`: Probability that a compound returns at time\n"
        "      t0+dt back to the same state as directly before the state\n"
        "      transition under the condition that it has *continuously*\n"
        "      been in the same state as directly after the state transition\n"
        "      from time t0 until t0+dt\n"
        "\n"
        "Column number:\n"
    )
    header += "{:>14d}".format(1)
    for i in range(2, data.shape[-1] + 1):
        header += " {:>16d}".format(i)
    mdt.fh.savetxt(args.OUTFILE, data, header=header)
    print("Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Checking output for consistency...")
    timer = datetime.now()
    names = (
        "prob_b_as_b",
        "prob_b_as_a",
        "prob_b_as_b_con",
        "prob_b_as_a_con",
        #
        "prob_a_as_a",
        "prob_a_as_b",
        "prob_a_as_a_con",
        "prob_a_as_b_con",
        #
        "prob_back",
        "prob_back_con",
    )
    first_values = (1, 0, 1, 0, 1, 0, 1, 0, 0, 0)
    for i, prob in enumerate(data.T[1:]):
        if not np.isclose(prob[0], first_values[i]):
            raise ValueError(
                "The first element of '{}' ({}) is not"
                " {}".format(names[i], prob[0], first_values[i])
            )
        if np.any(prob > 1):
            raise ValueError(
                "At least one element of '{}' is greater than"
                " one.".format(names[i])
            )
        if np.any(prob < 0):
            raise ValueError(
                "At least one element of '{}' is less than"
                " zero.".format(names[i])
            )
    for i in (0, 1, 4, 5):
        prob = data.T[1:][i]
        prob_con = data.T[1:][i + 2]
        if np.any(prob_con > prob):
            raise ValueError(
                "At leas one element of '{}' is greater than the corresponding"
                " element of '{}'".format(names[i + 2], names[i])
            )
    if np.any(prob_back_con > prob_back):
        raise ValueError(
            "At leas one element of 'prob_back_con' is greater than the"
            " corresponding element of 'prob_back'"
        )
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
