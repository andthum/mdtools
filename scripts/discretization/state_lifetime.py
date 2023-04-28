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
Calculate the average lifetime of the states in a discrete trajectory.

Calculate the average residence time for how long a compound resides in
a specific state before it changes states.  This is done by computing
the probability to be in the same state as at time :math:`t_0` after a
lag time :math:`\Delta t`.  Afterwards, this probability is fitted by a
stretched exponential function, whose integral from zero to infinity is
the average lifetime of all states in the discrete trajectory.  See also
the notes section of :func:`mdtools.dtrj.remain_prob`.

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
--nblocks
    Number of blocks for block averaging.  The trajectory will be split
    in ``NBLOCKS`` equally sized blocks, which will be analyzed
    independently, like if they were different trajectories.  Finally,
    the average and standard deviation over all blocks will be
    calculated.  Default: ``1``."
--restart
    Number of frames between restarting points for calculating the
    remain probability.  Must be an integer multiple of \--every.
    Default: ``100``.
--intermittency
    Maximum number of frames a compound is allowed to leave its state
    whilst still being considered to be in this state provided that it
    returns to this state after the given number of frames.  In other
    words, a compound is only considered to have left its state if it
    has left it for at least the given number of frames.
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
:func:`mdtools.dtrj.remain_prob` :
    The underlying function to calculate the remain probability
:mod:`scripts.discretization.state_lifetime_discrete` :
    Calculate the average lifetime of the states in a discrete
    trajectory resolved with respect to the states in a second discrete
    trajectory

Notes
-----
If you want to compute the lifetime of each individual state in a
discrete trajectory, use
:mod:`scripts.discretization.state_lifetime_discrete` and parse the same
discrete trajectory to \--f1 and \--f2.

See :func:`mdtools.dtrj.remain_prob` for further details.
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
            " trajectory."
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
        "--nblocks",
        dest="NBLOCKS",
        type=int,
        required=False,
        default=1,
        help="Number of blocks for block averaging.  Default: %(default)s.",
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
    NBLOCKS, blocksize = mdt.check.block_averaging(
        n_blocks=args.NBLOCKS, n_frames=N_FRAMES_TOT
    )
    RESTART, effective_restart = mdt.check.restarts(
        restart_every_nth_frame=args.RESTART,
        read_every_nth_frame=EVERY,
        n_frames=blocksize,
    )
    dtrj = dtrj[:, BEGIN:END:EVERY]
    dtrj_trans_info = mdt.rti.dtrj_trans_info_str(
        dtrj[:, : NBLOCKS * blocksize]
    )
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    if args.INTERMITTENCY > 0:
        print("\n")
        print("Correcting for intermittency...")
        dtrj = mdt.dyn.correct_intermittency(
            dtrj.T, args.INTERMITTENCY, inplace=True, verbose=True
        )
        dtrj = dtrj.T

    print("\n")
    print("Calculating remain probability...")
    timer = datetime.now()
    timer_block = datetime.now()
    prob = np.full((NBLOCKS, blocksize), np.nan, dtype=float)
    for block in range(NBLOCKS):
        if block % 10 ** (len(str(block)) - 1) == 0 or block == NBLOCKS - 1:
            print("Block   {:12d} of {:12d}".format(block, NBLOCKS - 1))
            print(
                "Elapsed time:         {}".format(datetime.now() - timer_block)
            )
            print(
                "Current memory usage: {:.2f} MiB".format(
                    mdt.rti.mem_usage(proc)
                )
            )
            timer_block = datetime.now()
        prob[block] = mdt.dtrj.remain_prob(
            dtrj=dtrj[:, block * blocksize : (block + 1) * blocksize],
            restart=effective_restart,
            continuous=args.CONTINUOUS,
            discard_neg_start=args.DISCARD_NEG_START,
            discard_all_neg=args.DISCARD_ALL_NEG,
            verbose=True,
        )
    del dtrj
    if NBLOCKS > 1:
        prob, prob_sd = mdt.stats.block_average(prob)
    else:
        prob = np.squeeze(prob)
        prob_sd = None
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Fitting remain probability...")
    timer = datetime.now()
    lag_times = np.arange(blocksize, dtype=np.uint32)

    if args.ENDFIT is None:
        endfit = int(0.9 * len(lag_times))
    else:
        _, endfit = mdt.nph.find_nearest(
            lag_times, args.ENDFIT, return_index=True
        )
    endfit += 1  # Make `endfit` inclusive.

    stopfit = np.argmax(prob < args.STOPFIT)
    if stopfit == 0 and prob[stopfit] >= args.STOPFIT:
        stopfit = len(prob)
    elif stopfit < 2:
        stopfit = 2

    fit_start = 0  # Inclusive.
    fit_stop = min(endfit, stopfit)  # Exclusive.
    if prob_sd is None:
        popt, perr = mdt.func.fit_kww(
            xdata=lag_times[fit_start:fit_stop], ydata=prob[fit_start:fit_stop]
        )
    else:
        popt, perr = mdt.func.fit_kww(
            xdata=lag_times[fit_start:fit_stop],
            ydata=prob[fit_start:fit_stop],
            ysd=prob_sd[fit_start:fit_stop],
        )
    tau_mean = popt[0] / popt[1] * gamma(1 / popt[1])
    fit = mdt.func.kww(t=lag_times, tau=popt[0], beta=popt[1])
    fit[:fit_start] = np.nan
    fit[fit_stop:] = np.nan
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    header = (
        "Average lifetime of all valid discrete states in the given discrete\n"
        + "trajectory\n"
        + "\n\n"
        + "intermittency:     {}\n".format(args.INTERMITTENCY)
        + "continuous:        {}\n".format(args.CONTINUOUS)
        + "discard_neg_start: {}\n".format(args.DISCARD_NEG_START)
        + "discard_all_neg:   {}\n".format(args.DISCARD_ALL_NEG)
        + "\n\n"
    )
    header += dtrj_trans_info
    header += "\n\n"
    if NBLOCKS == 1:
        header += (
            "The columns contain:\n"
            + "  1 Lag time (in trajectory steps)\n"
            + "  2 Remain probability\n"
            + "  3 Stretched exponential fit of the remain probability\n"
            + "\n"
            + "Column number:\n"
            + "{:>14d} {:>16d} {:>16d}\n".format(1, 2, 3)
            + "\n"
            + "Fit:\n"
            + "Fit start (steps):               {:>15.9e}\n".format(fit_start)
            + "Fit stop  (steps):               {:>15.9e}\n".format(fit_stop)
            + "Average lifetime <tau> (steps):  {:>15.9e}\n".format(tau_mean)
            + "Relaxation time tau (steps):     {:>15.9e}\n".format(popt[0])
            + "Std. dev. of tau (steps):        {:>15.9e}\n".format(perr[0])
            + "Stretching exponent beta:        {:>15.9e}\n".format(popt[1])
            + "Standard deviation of beta:      {:>15.9e}\n".format(perr[1])
        )
        data = np.column_stack([lag_times, prob, fit])
    else:
        header += (
            "The columns contain:\n"
            + "  1 Lag time (in trajectory steps)\n"
            + "  2 Remain probability\n"
            + "  3 Standard deviation of the remain probability\n"
            + "  4 Stretched exponential fit of the remain probability\n"
            + "\n"
            + "Column number:\n"
            + "{:>14d} {:>16d} {:>16d} {:>16d}\n".format(1, 2, 3, 4)
            + "\n"
            + "Fit:\n"
            + "Fit start (steps):               {:>32.9e}\n".format(fit_start)
            + "Fit stop  (steps):               {:>32.9e}\n".format(fit_stop)
            + "Average lifetime <tau> (steps):  {:>32.9e}\n".format(tau_mean)
            + "Relaxation time tau (steps):     {:>32.9e}\n".format(popt[0])
            + "Std. dev. of tau (steps):        {:>32.9e}\n".format(perr[0])
            + "Stretching exponent beta:        {:>32.9e}\n".format(popt[1])
            + "Standard deviation of beta:      {:>32.9e}\n".format(perr[1])
        )
        data = np.column_stack([lag_times, prob, prob_sd, fit])
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
