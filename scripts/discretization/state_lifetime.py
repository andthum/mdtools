#!/usr/bin/env python3


# This file is part of MDTools.
# Copyright (C) 2020  Andreas Thum
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




# Take a look at the notes from 17.10.2020 and 24.01.2021 when writing
# the documentation of this script.

# TODO: add intermittency flag




import sys
import os
import warnings
from datetime import datetime
import psutil
import argparse
import numpy as np
from scipy.special import gamma
import mdtools as mdt




# Also used by state_lifetime_discrete.py
def dtraj_transition_info(dtraj):
    """
    Get basic information about the transitions between different states
    in a discretized trajectory.

    Parameters
    ----------
    dtraj : array_like
        The discretized trajectory.  Array of shape ``(f, n)`` where
        ``f`` is the number of frames and ``n`` is the number of
        compounds.  The elements of `dtraj` are interpreted as the
        indices of the states in which a given compound is at a given
        frame.

    Returns
    -------
    n_stay : int
        Number of compounds that stay in the same state during the
        entire trajectory.
    always_neg : int
        Number of compounds that are always in a negative state during
        the entire trajectory.
    never_neg : int
        Number of compounds that are never in a negative state during
        the entire trajectory.
    n_frames_neg : int
        Total number of frames with negative states (summed over all
        compounds).
    n_trans : int
        Total number of state transitions.
    pos2pos : int
        Number of Positive -> Positive transitions (transitions from one
        state with a positive state index to another state with a
        positive state index).
    pos2neg : int
        Number of Positive -> Negative transitions.
    neg2pos : int
        Number of Negative -> Positive transitions.
    neg2neg : int
        Number of Negative -> Negative transitions.

    Note
    ----
    Positive states are states with a state index equal(!) to or greater
    than zero.  Negative states are states with a state index less than
    zero.
    """
    dtraj = np.asarray(dtraj)
    if dtraj.ndim != 2:
        raise ValueError("dtraj must have two dimensions")
    if np.any(np.modf(dtraj)[0] != 0):
        warnings.warn("At least one element of the discrete trajectory"
                      " is not an integer", RuntimeWarning)

    n_stay = np.count_nonzero(np.all(dtraj==dtraj[0], axis=0))
    always_neg = np.count_nonzero(np.all(dtraj<0, axis=0))
    never_neg = np.count_nonzero(np.all(dtraj>=0, axis=0))
    n_frames_neg = np.count_nonzero(dtraj<0)

    n_compounds = dtraj.shape[1]
    transitions = (np.diff(dtraj, axis=0) != 0)
    trans_init = np.vstack([transitions, np.zeros(n_compounds, dtype=bool)])
    trans_final = np.insert(transitions, 0, np.zeros(n_compounds), axis=0)
    n_trans = np.count_nonzero(transitions)
    if np.count_nonzero(trans_init) != n_trans:
        raise ValueError("The number of transitions in trans_init is not"
                         " the same as in transitions. This should not"
                         " have happened")
    if np.count_nonzero(trans_final) != n_trans:
        raise ValueError("The number of transitions in trans_final is"
                         " not the same as in transitions. This should"
                         " not have happened")
    pos2pos = np.count_nonzero((dtraj[trans_init] >= 0) &
                               (dtraj[trans_final] >= 0))
    pos2neg = np.count_nonzero((dtraj[trans_init] >= 0) &
                               (dtraj[trans_final] < 0))
    neg2pos = np.count_nonzero((dtraj[trans_init] < 0) &
                               (dtraj[trans_final] >= 0))
    neg2neg = np.count_nonzero((dtraj[trans_init] < 0) &
                               (dtraj[trans_final] < 0))
    if pos2pos + pos2neg + neg2pos + neg2neg != n_trans:
        raise ValueError("The sum of Positive <-> Negative transitions"
                         " ({}) is not equal to the total number of"
                         " transitions ({}). This should not have"
                         " happened"
                         .format(pos2pos+pos2neg+neg2pos+neg2neg,
                                 n_trans))

    return (n_stay, always_neg, never_neg, n_frames_neg,
            n_trans, pos2pos, pos2neg, neg2pos, neg2neg)




def remain_prob(dtraj, restart=1, continuous=False,
                discard_neg_start=False, discard_all_neg=False):
    """
    .. todo::
       Write docstring

    Parameters
    ----------
    dtraj : array_like
        The discretized trajectory.  Array of shape ``(f, n)`` where
        ``f`` is the number of frames and ``n`` is the number of
        compounds.  The elements of `dtraj` are interpreted as the
        indices of the states in which a given compound is at a given
        frame.
    restart : int, optional
        Number of frames between restarting points :math:`t_0`.
    continuous : bool, optional
        Compounds must be in the same state without interruption in
        order to be counted.
    discard_neg_start : bool, optional
        Discard all transitions starting from a state with a negative
        index (Transitions from positive to negative states are still
        counted).  `discard_neg_start` and `discard_all_neg` are
        mutually exclusive.
    discard_all_neg : bool, optional
        Discard all transitions starting from or ending in a state with
        a negative index.  `discard_neg_start` and `discard_all_neg` are
        mutually exclusive.

    Returns
    -------
    p : numpy.ndarray
        Array of shape ``f`` containing the remain probability at each
        lag time.

    Notes
    -----
    """
    dtraj = np.asarray(dtraj)
    if dtraj.ndim != 2:
        raise ValueError("dtraj must have two dimensions")
    if np.any(np.modf(dtraj)[0] != 0):
        warnings.warn("At least one element of the discrete trajectory"
                      " is not an integer", RuntimeWarning)
    if discard_neg_start and discard_all_neg:
        raise ValueError("discard_neg_start and discard_all_neg are"
                         " mutually exclusive")

    n_frames = dtraj.shape[0]
    n_compounds = dtraj.shape[1]
    p = np.zeros(n_frames, dtype=np.uint32)
    if discard_neg_start:
        valid = np.zeros(n_compounds, dtype=bool)
        norm = np.zeros(n_frames, dtype=np.uint32)
    elif discard_all_neg:
        dtraj_valid = (dtraj >= 0)
        norm = np.zeros(n_frames, dtype=np.uint32)
    else:
        remain = np.zeros(n_compounds, dtype=bool)

    proc = psutil.Process(os.getpid())
    timer = datetime.now()
    for t0 in range(0, n_frames-1, restart):
        if t0 % 10**(len(str(t0))-1) == 0 or t0 == n_frames-2:
            print("  Restart {:12d} of {:12d}"
                  .format(t0, n_frames-2),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now()-timer),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss/2**20),
                  flush=True)
            timer = datetime.now()

        # When trying to read and under stand this code, always read the
        # "else" parts fist.  Those are the simpler cases upon which the
        # other cases are built.
        if discard_neg_start:
            np.greater_equal(dtraj[t0], 0, out=valid)
            n_valid = np.count_nonzero(valid)
            if n_valid == 0:
                continue
            norm[1:n_frames-t0] += n_valid
            dtraj_t0 = dtraj[t0][valid]
            if continuous:
                stay = np.ones(n_valid, dtype=bool)
                remain = np.zeros(n_valid, dtype=bool)
                for lag in range(1, n_frames-t0):
                    np.equal(dtraj_t0, dtraj[t0+lag][valid], out=remain)
                    stay &= remain
                    n_stay = np.count_nonzero(stay)
                    if n_stay == 0:
                        break
                    p[lag] += n_stay
            else:
                remain = np.zeros(n_valid, dtype=bool)
                for lag in range(1, n_frames-t0):
                    np.equal(dtraj_t0, dtraj[t0+lag][valid], out=remain)
                    p[lag] += np.count_nonzero(remain)
        elif discard_all_neg:
            valid = dtraj_valid[t0]  # This is a view, not a copy!
            if not np.any(valid):
                continue
            if continuous:
                stay = np.ones(n_compounds, dtype=bool)
                remain = np.zeros(n_compounds, dtype=bool)
                for lag in range(1, n_frames-t0):
                    valid &= dtraj_valid[t0+lag]
                    n_valid = np.count_nonzero(valid)
                    if n_valid == 0:
                        continue
                    norm[lag] += n_valid
                    np.equal(dtraj[t0], dtraj[t0+lag], out=remain)
                    stay &= remain
                    stay &= valid
                    p[lag] += np.count_nonzero(stay)
                    # This loop must not be broken upon n_stay == 0,
                    # since otherwise the norm will be incorrect.
            else:
                for lag in range(1, n_frames-t0):
                    valid &= dtraj_valid[t0+lag]
                    n_valid = np.count_nonzero(valid)
                    if n_valid == 0:
                        continue
                    norm[lag] += n_valid
                    remain = (dtraj[t0][valid] == dtraj[t0+lag][valid])
                    p[lag] += np.count_nonzero(remain)
        else:
            if continuous:
                stay = np.ones(n_compounds, dtype=bool)
                for lag in range(1, n_frames-t0):
                    np.equal(dtraj[t0], dtraj[t0+lag], out=remain)
                    stay &= remain
                    n_stay = np.count_nonzero(stay)
                    if n_stay == 0:
                        break
                    p[lag] += n_stay
            else:
                for lag in range(1, n_frames-t0):
                    np.equal(dtraj[t0], dtraj[t0+lag], out=remain)
                    p[lag] += np.count_nonzero(remain)

    if discard_neg_start or discard_all_neg:
        if norm[0] != 0:
            raise ValueError("The first element of norm is not zero but"
                             " {}. This should not have happened"
                             .format(norm[0]))
        norm[0] = 1
        p = p / norm
    else:
        p = p / n_compounds
        p /= mdt.dyn.n_restarts(n_frames=n_frames, restart=restart)
    if p[0] != 0:
        raise ValueError("The first element of p is not zero but {}."
                         " This should not have happened".format(p[0]))
    p[0] = 1
    if np.any(p > 1):
        raise ValueError("At least one element of p is greater than one."
                         " This should not have happened")
    if np.any(p < 0):
        raise ValueError("At least one element of p is less than zero."
                         " This should not have happened")

    return p








if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())


    parser = argparse.ArgumentParser(
        description=(
            "Calculate the average lifetime of the discrete"
            " states in a discretized trajectory. I.e."
            " calculate the average residence time for how long"
            " a compound resides in a specific state before it"
            " changes states. This is done by computing the"
            " probability to be in the same state as at time t0"
            " after a lag time tau. This 'remain probability'"
            " is then fitted by a stretched exponential"
            " function, whose integral from zero to infinity is"
            " the averave lifetime of all states in the"
            " discretized trajectory."
        )
    )
    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="File containing the discretized trajectory stored as"
             " numpy.ndarray in .npy format. The array must be of shape"
             " (n, f) where n is the number of compounds and f is the"
             " number of frames. The elements of the array are"
             " interpreted as the indices of the states in which a given"
             " compound is at a given frame."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename."
    )

    parser.add_argument(
        '-b',
        dest='BEGIN',
        type=int,
        required=False,
        default=0,
        help="First frame to read. Frame numbering starts at zero."
             " Default: 0"
    )
    parser.add_argument(
        '-e',
        dest='END',
        type=int,
        required=False,
        default=-1,
        help="Last frame to read (exclusive, i.e. the last frame read is"
             " actually END-1). Default: -1 (means read the very last"
             " frame of the trajectory)"
    )
    parser.add_argument(
        '--every',
        dest='EVERY',
        type=int,
        required=False,
        default=1,
        help="Read every n-th frame. Default: 1"
    )
    parser.add_argument(
        '--nblocks',
        dest='NBLOCKS',
        type=int,
        required=False,
        default=1,
        help="Number of blocks for block averaging. The trajectory will"
             " be split in NBLOCKS equally sized blocks, which will be"
             " analyzed independently, like if they were different"
             " trajectories. Finally, the average and standard deviation"
             " over all blocks will be calculated. Default: 1"
    )
    parser.add_argument(
        '--restart',
        dest='RESTART',
        type=int,
        default=100,
        help="Number of frames between restarting points for calculating"
             " the remain probability. This must be an integer multiple"
             " of --every. Default: 100"
    )

    parser.add_argument(
        '--continuous',
        dest='CONTINUOUS',
        required=False,
        default=False,
        action='store_true',
        # TODO: help text
        help="Compounds must be in the same state without interruption"
             " in order to be counted."
    )
    group.add_argument(
        '--discard-neg-start',
        dest='DISCARD_NEG_START',
        required=False,
        default=False,
        action='store_true',
        # TODO: help text
        help="Discard all transitions starting from a state with a"
             " negative index (Transitions from positive to negative"
             " states are still counted)."
    )
    group.add_argument(
        '--discard-all-neg',
        dest='DISCARD_ALL_NEG',
        required=False,
        default=False,
        action='store_true',
        # TODO: help text
        help="Discard all transitions starting from or ending in a state"
             " with a negative index. Warning: Experimental feature!"
    )

    parser.add_argument(
        '--end-fit',
        dest='ENDFIT',
        type=float,
        required=False,
        default=None,
        help="End time for fitting the remain probability (in trajectory"
             " steps). Inclusive, i.e. the time given here is still"
             " included in the fit. Default: End at 90%% of the"
             " trajectory."
    )
    parser.add_argument(
        '--stop-fit',
        dest='STOPFIT',
        type=float,
        required=False,
        default=0.01,
        help="Stop fitting the remain probability as soon as it falls"
             " below this value. The fitting is stopped by whatever"
             " happens earlier: --end-fit or --stop-fit. Default: 0.01"
    )


    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())




    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()

    dtrajs = np.load(args.TRJFILE)
    if dtrajs.ndim == 1:
        dtrajs = np.expand_dims(dtrajs, axis=0)
    elif dtrajs.ndim > 2:
        raise ValueError("The discrete trajectory must have one or two"
                         " dimensions")
    dtrajs = np.asarray(dtrajs.T, order='C')
    n_frames = dtrajs.shape[0]
    n_compounds = dtrajs.shape[1]
    print("  Number of frames:    {:>9d}".format(n_frames), flush=True)
    print("  Number of compounds: {:>9d}"
          .format(n_compounds),
          flush=True)
    if np.any(np.modf(dtrajs)[0] != 0):
        warnings.warn("At least one element of the discrete trajectory"
                      " is not an integer", RuntimeWarning)

    BEGIN, END, EVERY, n_frames = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=n_frames)
    NBLOCKS, blocksize = mdt.check.block_averaging(n_blocks=args.NBLOCKS,
                                                   n_frames=n_frames)
    RESTART, effective_restart = mdt.check.restarts(
        restart_every_nth_frame=args.RESTART,
        read_every_nth_frame=EVERY,
        n_frames=blocksize)
    dtrajs = dtrajs[BEGIN:END:EVERY]

    trans_info = dtraj_transition_info(dtraj=dtrajs[:NBLOCKS*blocksize])

    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Calculating remain probability", flush=True)
    timer = datetime.now()
    timer_block = datetime.now()

    p = [None,] * NBLOCKS
    for block in range(NBLOCKS):
        if block % 10**(len(str(block))-1) == 0 or block == NBLOCKS-1:
            print(flush=True)
            print("  Block   {:12d} of {:12d}"
                  .format(block, NBLOCKS-1),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now()-timer_block),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss/2**20),
                  flush=True)
            timer_block = datetime.now()
        p[block] = remain_prob(
            dtraj=dtrajs[block*blocksize:(block+1)*blocksize],
            restart=effective_restart,
            continuous=args.CONTINUOUS,
            discard_neg_start=args.DISCARD_NEG_START,
            discard_all_neg=args.DISCARD_ALL_NEG)
    del dtrajs

    p = np.asarray(p)
    if NBLOCKS > 1:
        p, p_sd = mdt.stats.block_average(p)
    else:
        p = np.squeeze(p)
        p_sd = None

    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Fitting remain probability", flush=True)
    timer = datetime.now()

    lag_times = np.arange(blocksize, dtype=np.uint32)

    if args.ENDFIT is None:
        endfit = int(0.9 * len(lag_times))
    else:
        _, endfit = mdt.nph.find_nearest(lag_times,
                                         args.ENDFIT,
                                         return_index=True)
    endfit += 1  # To make args.ENDFIT inclusive

    stopfit = np.argmax(p < args.STOPFIT)
    if stopfit == 0 and p[stopfit] >= args.STOPFIT:
        stopfit = len(p)
    elif stopfit < 2:
        stopfit = 2

    fit_start = 0                    # inclusive
    fit_stop = min(endfit, stopfit)  # exclusive
    if p_sd is None:
        popt, perr = mdt.func.fit_kww(xdata=lag_times[fit_start:fit_stop],
                                      ydata=p[fit_start:fit_stop])
    else:
        popt, perr = mdt.func.fit_kww(xdata=lag_times[fit_start:fit_stop],
                                      ydata=p[fit_start:fit_stop],
                                      ysd=p_sd[fit_start:fit_stop])
    tau_mean = popt[0]/popt[1] * gamma(1/popt[1])
    fit = mdt.func.kww(t=lag_times, tau=popt[0], beta=popt[1])
    fit[:fit_start] = np.nan
    fit[fit_stop:] = np.nan

    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()

    header = (
        "Average lifetime of all discrete states in the discretized\n"
        "trajectory\n"
        "\n"
        "\n"
        "continuous =        {}\n"
        "discard_neg_start = {}\n"
        "discard_all_neg =   {}\n"
        ""
        "\n"
        "Number of frames (per compound):                         {:>12d}\n"
        "Number of compounds:                                     {:>12d}\n"
        "Number of compounds that never leave their state:        {:>12d}\n"
        "Number of compounds that are always in a negative state: {:>12d}\n"
        "Number of compounds that are never  in a negative state: {:>12d}\n"
        "Total number of frames with negative states:             {:>12d}\n"
        "\n"
        "Total number of state transitions:          {:>12d}\n"
        "Number of Positive -> Positive transitions: {:>12d}  ({:>8.4f} %)\n"
        "Number of Positive -> Negative transitions: {:>12d}  ({:>8.4f} %)\n"
        "Number of Negative -> Positive transitions: {:>12d}  ({:>8.4f} %)\n"
        "Number of Negative -> Negative transitions: {:>12d}  ({:>8.4f} %)\n"
        "Positive states are states with a state index >= 0\n"
        "Negative states are states with a state index <  0\n"
        "\n"
        "\n"
        .format(args.CONTINUOUS,
                args.DISCARD_NEG_START,
                args.DISCARD_ALL_NEG,
                n_frames, n_compounds,
                trans_info[0], trans_info[1],
                trans_info[2], trans_info[3],
                trans_info[4],
                trans_info[5], 100*trans_info[5]/trans_info[4],
                trans_info[6], 100*trans_info[6]/trans_info[4],
                trans_info[7], 100*trans_info[7]/trans_info[4],
                trans_info[8], 100*trans_info[8]/trans_info[4]
                )
    )
    if NBLOCKS == 1:
        header += (
            "The columns contain:\n"
            "  1 Lag time (in trajectory steps)\n"
            "  2 Remain probability\n"
            "  3 Stretched exponential fit of the remain probability\n"
            "\n"
            "Column number:\n"
            "{:>14d} {:>16d} {:>16d}\n"
            "\n"
            "Fit:\n"
            "Fit start (steps):               {:>15.9e}\n"
            "Fit stop  (steps):               {:>15.9e}\n"
            "Average lifetime <tau> (steps):  {:>15.9e}\n"
            "Relaxation time tau (steps):     {:>15.9e}\n"
            "Std. dev. of tau (steps):        {:>15.9e}\n"
            "Stretching exponent beta:        {:>15.9e}\n"
            "Standard deviation of beta:      {:>15.9e}\n"
            .format(1, 2, 3,
                    fit_start, fit_stop,
                    tau_mean, popt[0], perr[0], popt[1], perr[1])
        )
        data = np.column_stack([lag_times, p, fit])
    else:
        header += (
            "The columns contain:\n"
            "  1 Lag time (in trajectory steps)\n"
            "  2 Remain probability\n"
            "  3 Standard deviation of the remain probability\n"
            "  4 Stretched exponential fit of the remain probability\n"
            "\n"
            "Column number:\n"
            "{:>14d} {:>16d} {:>16d} {:>16d}\n"
            "\n"
            "Fit:\n"
            "Fit start (steps):               {:>32.9e}\n"
            "Fit stop  (steps):               {:>32.9e}\n"
            "Average lifetime <tau> (steps):  {:>32.9e}\n"
            "Relaxation time tau (steps):     {:>32.9e}\n"
            "Std. dev. of tau (steps):        {:>32.9e}\n"
            "Stretching exponent beta:        {:>32.9e}\n"
            "Standard deviation of beta:      {:>32.9e}\n"
            .format(1, 2, 3, 4,
                    fit_start, fit_stop,
                    tau_mean, popt[0], perr[0], popt[1], perr[1])
        )
        data = np.column_stack([lag_times, p, p_sd, fit])

    mdt.fh.savetxt(fname=args.OUTFILE, data=data, header=header)

    print("  Created {}".format(args.OUTFILE), flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)




    print("\n\n\n{} done".format(os.path.basename(sys.argv[0])))
    print("Elapsed time:         {}"
          .format(datetime.now()-timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
