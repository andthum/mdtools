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
Calculate the average lifetime of the states in a discrete trajectory
resolved with respect to the states in a second discrete trajectory.

Calculate the average residence time for how long a compound resides in
a specific state before it changes states given that it was in a
specific state of a second discrete trajectory at time :math:`t_0`.
This is done by computing the probability to be in the same state as at
time :math:`t_0` after a lag time :math:`\Delta t` as function of the
states in the second discrete trajectory.  Afterwards, these
probabilities are fitted by stretched exponential functions, whose
integrals from zero to infinity are the average lifetimes of the states
in the first discrete trajectory.  See also
:func:`mdtools.dtrj.remain_prob_discrete`.

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
--restart
    Number of frames between restarting points for calculating the
    remain probability.  Must be an integer multiple of \--every.
    Default: ``100``.
--continuous
    If given, compounds must continuously be in the same state without
    interruption in order to be counted (see notes section of
    :func:`mdtools.dtrj.remain_prob`).
--discard-neg-start
    Discard all transitions starting from a negative state (see notes
    section of :func:`mdtools.dtrj.remain_prob`).  Must not be used
    together with \--discard-all-neg.
--discard-all-neg
    Discard all negative states (see notes section of
    :func:`mdtools.dtrj.remain_prob`).  Must not be used together with
    \--discard-neg-start.
--end-fit
    End time for fitting the remain probability (in trajectory steps).
    This is inclusive, i.e. the time given here is still included in the
    fit.  If ``None``, the fit ends at 90% of the lag times.  Default:
    ``None``.
--stop-fit
    Stop fitting the remain probability as soon as it falls below the
    given value.  The fitting is stopped by whatever happens earlier:
    \--end-fit or \--stop-fit.  Default: ``0.01``.

See Also
--------
:func:`mdtools.dtrj.remain_prob_discrete` :
    The underlying function to calculate the remain probabilities
:mod:`scripts.discretization.state_lifetime` :
    Calculate the average lifetime of the states in a discrete
    trajectory

Notes
-----
If you parse the same discrete trajectory to \--f1 and \--f2 you will
get the lifetime of each individual state in the input trajectory.  If
you want the average lifetime of all states, use
:mod:`scripts.discretization.state_lifetime`.

See :func:`mdtools.dtrj.remain_prob_discrete` for further details.
"""


# Standard libraries
import argparse
import os
import sys
from datetime import datetime, timedelta

# Third-party libraries
import numpy as np
import psutil
from scipy.special import gamma

# First-party libraries
import mdtools as mdt


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage.
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the average lifetime of the states in a discrete"
            " trajectory resolved with respect to the states in a second"
            " discrete trajectory."
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
        "--restart",
        dest="RESTART",
        type=int,
        default=100,
        help=(
            "Number of frames between restarting points for calculating the"
            " remain probability.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--continuous",
        dest="CONTINUOUS",
        required=False,
        default=False,
        action="store_true",
        help=(
            "If given, compounds must continuously be in the same state"
            " without interruption in order to be counted."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--discard-neg-start",
        dest="DISCARD_NEG_START",
        required=False,
        default=False,
        action="store_true",
        help="Discard all transitions starting from a negative state.",
    )
    group.add_argument(
        "--discard-all-neg",
        dest="DISCARD_ALL_NEG",
        required=False,
        default=False,
        action="store_true",
        help="Discard all negative states.",
    )
    parser.add_argument(
        "--end-fit",
        dest="ENDFIT",
        type=float,
        required=False,
        default=None,
        help=(
            "End time for fitting the remain probability in trajectory"
            " steps (inclusive).  If None, the fit ends at 90%% of the"
            " lag times   Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--stop-fit",
        dest="STOPFIT",
        type=float,
        required=False,
        default=0.01,
        help=(
            "Stop fitting the remain probability as soon as it falls below the"
            " given value.  The fitting is stopped by whatever happens"
            " earlier: --end-fit or --stop-fit.  Default: %(default)s"
        ),
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

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
    RESTART, effective_restart = mdt.check.restarts(
        restart_every_nth_frame=args.RESTART,
        read_every_nth_frame=EVERY,
        n_frames=N_FRAMES_TOT,
    )
    dtrj1 = dtrj1[:, BEGIN:END:EVERY]
    dtrj2 = dtrj2[:, BEGIN:END:EVERY]
    dtrj1_trans_info = mdt.rti.dtrj_trans_info_str(dtrj1)
    dtrj2_states = np.unique(dtrj2)
    dtrj2_n_states = len(dtrj2_states)
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Calculating remain probability...")
    timer = datetime.now()
    prob = mdt.dtrj.remain_prob_discrete(
        dtrj1=dtrj1,
        dtrj2=dtrj2,
        restart=effective_restart,
        continuous=args.CONTINUOUS,
        discard_neg_start=args.DISCARD_NEG_START,
        discard_all_neg=args.DISCARD_ALL_NEG,
        verbose=True,
    )
    del dtrj1, dtrj2
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Fitting remain probabilities...")
    timer = datetime.now()
    lag_times = np.arange(N_FRAMES_TOT, dtype=np.uint32)

    if args.ENDFIT is None:
        endfit = int(0.9 * len(lag_times))
    else:
        _, endfit = mdt.nph.find_nearest(
            lag_times, args.ENDFIT, return_index=True
        )
    endfit += 1  # To make args.ENDFIT inclusive

    fit_start = np.zeros(dtrj2_n_states, dtype=np.uint32)  # inclusive
    fit_stop = np.zeros(dtrj2_n_states, dtype=np.uint32)  # exclusive
    popt = np.full((dtrj2_n_states, 2), np.nan, dtype=np.float32)
    perr = np.full((dtrj2_n_states, 2), np.nan, dtype=np.float32)
    for i in range(dtrj2_n_states):
        stopfit = np.argmax(prob[:, i] < args.STOPFIT)
        if stopfit == 0 and prob[:, i][stopfit] >= args.STOPFIT:
            stopfit = len(prob[:, i])
        elif stopfit < 2:
            stopfit = 2
        fit_stop[i] = min(endfit, stopfit)
        popt[i], perr[i] = mdt.func.fit_kww(
            xdata=lag_times[fit_start[i] : fit_stop[i]],
            ydata=prob[:, i][fit_start[i] : fit_stop[i]],
        )
    tau_mean = popt[:, 0] / popt[:, 1] * gamma(1 / popt[:, 1])
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    header = (
        "Average lifetime of all valid discrete states in the given discrete\n"
        + "trajectory as function of another set of discrete states\n"
        + "\n\n"
        + "continuous:        {}\n".format(args.CONTINUOUS)
        + "discard_neg_start: {}\n".format(args.DISCARD_NEG_START)
        + "discard_all_neg:   {}\n".format(args.DISCARD_ALL_NEG)
        + "\n\n"
    )
    header += dtrj1_trans_info
    header += "\n\n"
    header += (
        "The first column contains the lag times (in trajectory steps).\n"
        "The first row contains the states of the second discrete trajectory\n"
        "that were used to discretize the remain probability.\n"
        "\n"
        "Fit:\n"
    )
    header += "Start (steps):"
    for i in range(dtrj2_n_states):
        header += " {:>16.9e}".format(fit_start[i])
    header += "\n"
    header += "Stop (steps): "
    for i in range(dtrj2_n_states):
        header += " {:>16.9e}".format(fit_stop[i])
    header += "\n"
    header += "<tau> (steps):"
    for i in range(dtrj2_n_states):
        header += " {:>16.9e}".format(tau_mean[i])
    header += "\n"
    header += "tau (steps):  "
    for i in range(dtrj2_n_states):
        header += " {:>16.9e}".format(popt[i][0])
    header += "\n"
    header += "Std. dev.:    "
    for i in range(dtrj2_n_states):
        header += " {:>16.9e}".format(perr[i][0])
    header += "\n"
    header += "beta:         "
    for i in range(dtrj2_n_states):
        header += " {:>16.9e}".format(popt[i][1])
    header += "\n"
    header += "Std. dev.:    "
    for i in range(dtrj2_n_states):
        header += " {:>16.9e}".format(perr[i][1])
    header += "\n"
    mdt.fh.savetxt_matrix(
        args.OUTFILE, prob, var1=lag_times, var2=dtrj2_states, header=header
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
