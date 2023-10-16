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
Calculate the probability to be in the same state as directly before or
after a state transition as function of time passed since the
transition.

Given that at time :math:`t_0` a state transition occurred, compute the
probability

    * that a compound is at time :math:`t_0 - \Delta t` in the same
      state as directly before the state transition ('prob_b_as_b').
    * that a compound is at time :math:`t_0 - \Delta t` in the same
      state as directly after the state transition ('prob_b_as_a').
    * that a compound is at time :math:`t_0 + \Delta t` in the same
      state as directly after the state transition ('prob_a_as_a').
    * that a compound is at time :math:`t_0 + \Delta t` in the same
      state as directly before the state transition ('prob_a_as_b').
    * that a compound is at time :math:`t_0 - \Delta t` in the same
      state as at time :math:`t_0 + \Delta t` ('prob_same_sym').

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
    # `prob_sym`: Probability that a compound is at time t0-dt in the
    #     same state as at time t0+dt.
    # `prob_b_as_b`: Probability that a compound is at time t0-dt in the
    #     same state as directly before the state transition.
    # `prob_b_as_a`: Probability that a compound is at time t0-dt in the
    #     same state as directly after the state transition.
    # `prob_a_as_a`: Probability that a compound is at time t0+dt in the
    #     same state as directly after the state transition.
    # `prob_a_as_b`: Probability that a compound is at time t0+dt in the
    #     same state as directly before the state transition.
    prob_sym = np.zeros(N_FRAMES // 2, dtype=np.uint32)
    norm_sym = np.zeros_like(prob_sym)
    prob_b_as_b = np.zeros(N_FRAMES - 1, dtype=np.uint32)
    prob_b_as_a = np.zeros_like(prob_b_as_b)
    norm_b = np.zeros_like(prob_b_as_b)
    prob_a_as_a = np.zeros_like(prob_b_as_b)
    prob_a_as_b = np.zeros_like(prob_a_as_a)
    norm_a = np.zeros_like(prob_a_as_a)

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
        for t0 in trans:
            # Trajectory before the state transition, time reversed.
            cmp_trj_b = cmp_trj[t0 - 1 :: -1]
            # Trajectory after the state transition.
            cmp_trj_a = cmp_trj[t0:]
            # Calculate `prob_sym`.
            max_lag = min(t0, N_FRAMES - t0)
            prob_sym[:max_lag] += cmp_trj_b[:max_lag] == cmp_trj_a[:max_lag]
            norm_sym[:max_lag] += 1
            # Calculate `prob_b_as_b` and `prob_b_as_a`.
            max_lag = len(cmp_trj_b)
            prob_b_as_b[:max_lag] += cmp_trj_b == cmp_trj_b[0]
            prob_b_as_a[:max_lag] += cmp_trj_b == cmp_trj_a[0]
            norm_b[:max_lag] += 1
            # Calculate `prob_a_as_a` and `prob_a_as_b`.
            max_lag = len(cmp_trj_a)
            prob_a_as_a[:max_lag] += cmp_trj_a == cmp_trj_a[0]
            prob_a_as_b[:max_lag] += cmp_trj_a == cmp_trj_b[0]
            norm_a[:max_lag] += 1
        # ProgressBar update:
        progress_bar_mem = proc.memory_info().rss / 2**20
        dtrj.set_postfix_str(
            "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
        )
    dtrj.close()
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    prob_sym = prob_sym / norm_sym
    prob_b_as_b = prob_b_as_b / norm_b
    prob_b_as_a = prob_b_as_a / norm_b
    prob_a_as_a = prob_a_as_a / norm_a
    prob_a_as_b = prob_a_as_b / norm_a
    del norm_sym, norm_b, norm_a

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    lag_times = np.arange(len(prob_b_as_b), dtype=np.uint32)
    prob_sym = mdt.nph.extend(prob_sym, len(prob_b_as_b), fill_value=np.nan)
    data = np.column_stack(
        [
            lag_times,
            prob_sym,
            prob_b_as_b,
            prob_b_as_a,
            prob_a_as_a,
            prob_a_as_b,
        ]
    )
    del prob_sym, prob_b_as_b, prob_b_as_a, prob_a_as_a, prob_a_as_b
    header = (
        "Probability to be in the same state a certain time before and after\n"
        "a state transition.\n"
        "\n"
        "\n" + trans_info_str + "\n"
        "\n"
        "Given that at time t0 a state transition occurred...\n"
        "The columns contain:\n"
        "  1 Lag time dt (in trajectory steps)\n"
        "  2 prob_sym:    Probability that a compound is at time t0-dt in\n"
        "      the same state as at time t0+dt\n"
        "  3 prob_b_as_b: Probability that a compound is at time t0-dt in\n"
        "      the same state as directly before the state transition\n"
        "  4 prob_b_as_a: Probability that a compound is at time t0-dt in\n"
        "      the same state as directly after the state transition\n"
        "  5 prob_a_as_a: Probability that a compound is at time t0+dt in\n"
        "      the same state as directly after the state transition\n"
        "  6 prob_a_as_b: Probability that a compound is at time t0+dt in\n"
        "      the same state as directly before the state transition\n"
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
        "prob_sym",
        "prob_b_as_b",
        "prob_b_as_a",
        "prob_a_as_a",
        "prob_a_as_b",
    )
    first_values = (0, 1, 0, 1, 0)
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
