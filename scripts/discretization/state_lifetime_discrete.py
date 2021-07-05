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
from state_lifetime import dtraj_transition_info




def remain_prob_discrete(dtraj1, dtraj2, restart=1, continuous=False,
                         discard_neg_start=False, discard_all_neg=False):
    """
    .. todo::
       Write docstring

    Parameters
    ----------
    dtraj1 : array_like
        The first discretized trajectory.  Array of shape ``(f, n)``
        where ``f`` is the number of frames and ``n`` is the number of
        compounds.  The elements of `dtraj1` are interpreted as the
        indices of the states in which a given compound is at a given
        frame.
    dtraj2 : array_like
        The second discretized trajectory.  Must have the same shape as
        `dtraj1`.
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
        Array of shape ``(f, m)``, where ``m`` is the number of states
        in the second discretized trajectory.  The `ij`-th element of `p`
        is the remain probability at a lag time of `i` frames given that
        the compound was in the secondary state `j` at time :math:`t_0`.

    Notes
    -----
    """
    dtraj1 = np.asarray(dtraj1)
    dtraj2 = np.asarray(dtraj2)
    if dtraj1.ndim != 2:
        raise ValueError("dtraj1 must have two dimensions")
    if dtraj1.shape != dtraj2.shape:
        raise ValueError("Both trajectories must have the same shape")
    if np.any(np.modf(dtraj1)[0] != 0):
        warnings.warn("At least one element of the first discrete"
                      " trajectory is not an integer", RuntimeWarning)
    if np.any(np.modf(dtraj2)[0] != 0):
        warnings.warn("At least one element of the second discrete"
                      " trajectory is not an integer", RuntimeWarning)
    if discard_neg_start and discard_all_neg:
        raise ValueError("discard_neg_start and discard_all_neg are"
                         " mutually exclusive")

    dtraj2 = mdt.nph.sequenize(dtraj2,
                               step=np.uint32(1),
                               start=np.uint32(0))
    n_states = np.max(dtraj2) + 1
    if np.min(dtraj2) != 0:
        raise ValueError("The minimum of the reordered second trajectory"
                         " is not zero. This should not have happened")

    n_frames = dtraj1.shape[0]
    n_compounds = dtraj1.shape[1]
    p = np.zeros((n_frames, n_states), dtype=np.uint32)
    norm = np.zeros((n_frames, n_states), dtype=np.uint32)
    if discard_neg_start:
        valid = np.zeros(n_compounds, dtype=bool)
    elif discard_all_neg:
        dtraj1_valid = (dtraj1 >= 0)
    else:
        remain = np.zeros(n_compounds, dtype=bool)

    proc = psutil.Process(os.getpid())
    timer = datetime.now()
    for t0 in range(0, n_frames - 1, restart):
        if t0 % 10**(len(str(t0)) - 1) == 0 or t0 == n_frames - 2:
            print("  Restart {:12d} of {:12d}"
                  .format(t0, n_frames - 2),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now() - timer),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss / 2**20),
                  flush=True)
            timer = datetime.now()

        # When trying to read and under stand this code, always read the
        # "else" parts fist.  Those are the simpler cases upon which the
        # other cases are built.
        if discard_neg_start:
            np.greater_equal(dtraj1[t0], 0, out=valid)
            n_valid = np.count_nonzero(valid)
            if n_valid == 0:
                continue
            dtraj1_t0 = dtraj1[t0][valid]
            dtraj2_t0 = dtraj2[t0][valid]
            bin_ix_u, counts = np.unique(dtraj2_t0, return_counts=True)
            masks = (dtraj2_t0 == bin_ix_u[:, None])
            norm[1:n_frames - t0][:, bin_ix_u] += counts.astype(np.uint32)
            if continuous:
                stay = np.ones(n_valid, dtype=bool)
                remain = np.zeros(n_valid, dtype=bool)
                for lag in range(1, n_frames - t0):
                    np.equal(dtraj1_t0, dtraj1[t0 + lag][valid], out=remain)
                    stay &= remain
                    if not np.any(stay):
                        break
                    for i, b in enumerate(bin_ix_u):
                        p[lag][b] += np.count_nonzero(stay[masks[i]])
            else:
                remain = np.zeros(n_valid, dtype=bool)
                for lag in range(1, n_frames - t0):
                    np.equal(dtraj1_t0, dtraj1[t0 + lag][valid], out=remain)
                    for i, b in enumerate(bin_ix_u):
                        p[lag][b] += np.count_nonzero(remain[masks[i]])
        elif discard_all_neg:
            valid = dtraj1_valid[t0]  # This is a view, not a copy!
            if not np.any(valid):
                continue
            if continuous:
                stay = np.ones(n_compounds, dtype=bool)
                remain = np.zeros(n_compounds, dtype=bool)
                mask = np.zeros(n_compounds, dtype=bool)
                for lag in range(1, n_frames - t0):
                    valid &= dtraj1_valid[t0 + lag]
                    if not np.any(valid):
                        continue
                    bin_ix_u, counts = np.unique(dtraj2[t0][valid],
                                                 return_counts=True)
                    np.equal(dtraj1[t0], dtraj1[t0 + lag], out=remain)
                    stay &= remain
                    stay &= valid
                    # This loop must not be broken upon n_stay == 0,
                    # since otherwise the norm will be incorrect.
                    for i, b in enumerate(bin_ix_u):
                        np.equal(dtraj2[t0], b, out=mask)
                        p[lag][b] += np.count_nonzero(stay[mask])
                        norm[lag][b] += counts[i]
            else:
                for lag in range(1, n_frames - t0):
                    valid &= dtraj1_valid[t0 + lag]
                    n_valid = np.count_nonzero(valid)
                    if n_valid == 0:
                        continue
                    bin_ix_u, counts = np.unique(dtraj2[t0][valid],
                                                 return_counts=True)
                    mask = np.zeros(n_valid, dtype=bool)
                    remain = (dtraj1[t0][valid] == dtraj1[t0 + lag][valid])
                    for i, b in enumerate(bin_ix_u):
                        np.equal(dtraj2[t0][valid], b, out=mask)
                        p[lag][b] += np.count_nonzero(remain[mask])
                        norm[lag][b] += counts[i]
        else:
            bin_ix_u, counts = np.unique(dtraj2[t0], return_counts=True)
            masks = (dtraj2[t0] == bin_ix_u[:, None])
            norm[1:n_frames - t0][:, bin_ix_u] += counts.astype(np.uint32)
            if continuous:
                stay = np.ones(n_compounds, dtype=bool)
                for lag in range(1, n_frames - t0):
                    np.equal(dtraj1[t0], dtraj1[t0 + lag], out=remain)
                    stay &= remain
                    if not np.any(stay):
                        break
                    for i, b in enumerate(bin_ix_u):
                        p[lag][b] += np.count_nonzero(stay[masks[i]])
            else:
                for lag in range(1, n_frames - t0):
                    np.equal(dtraj1[t0], dtraj1[t0 + lag], out=remain)
                    for i, b in enumerate(bin_ix_u):
                        p[lag][b] += np.count_nonzero(remain[masks[i]])

    if np.any(norm[0] != 0):
        raise ValueError("At least one element in the first row of norm"
                         " is not zero. This should not have happened")
    norm[0] = 1
    p = p / norm
    if np.any(p[0] != 0):
        raise ValueError("At least one element in the first row of p is"
                         " not zero. This should not have happened")
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
            " states in a discretized trajectory as function of"
            " the states in another discretized trajectory. I.e."
            " calculate the average residence time for how long"
            " a compound resides in a specific state before it"
            " changes states, given that it was in a specific"
            " state of the second discretized trajectory at"
            " time t0. This is done by computing the"
            " probability to be in the same state as at time t0"
            " after a lag time tau as function of the states in"
            " the second discretized trajectory. These 'remain"
            " probabilities' are then fitted by a stretched"
            " exponential function, whose integral from zero to"
            " infinity is the averave lifetime of the states in"
            " the first discretized trajectory."
        )
    )
    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        '--f1',
        dest='TRJFILE1',
        type=str,
        required=True,
        help="File containing the first discretized trajectory stored as"
             " integer numpy.ndarray in .npy format. The array must be"
             " of shape (n, f) where n is the number of compounds and f"
             " is the number of frames. The elements of the array are"
             " interpreted as the indices of the states in which a given"
             " compound is at a given frame."
    )
    parser.add_argument(
        '--f2',
        dest='TRJFILE2',
        type=str,
        required=True,
        help="File containing the second discretized trajectory, which"
             " must be of the same shape as the first trajectory."
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
             " in order to counted."
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

    dtrajs1 = np.load(args.TRJFILE1)
    dtrajs2 = np.load(args.TRJFILE2)
    if dtrajs1.shape != dtrajs2.shape:
        raise ValueError("Both trajectories must have the same shape")
    if dtrajs1.ndim == 1:
        dtrajs1 = np.expand_dims(dtrajs1, axis=0)
        dtrajs2 = np.expand_dims(dtrajs2, axis=0)
    elif dtrajs1.ndim > 2:
        raise ValueError("The discrete trajectories must have one or two"
                         " dimensions")
    dtrajs1 = np.asarray(dtrajs1.T, order='C')
    dtrajs2 = np.asarray(dtrajs2.T, order='C')
    n_frames = dtrajs1.shape[0]
    n_compounds = dtrajs1.shape[1]
    print("  Number of frames:    {:>9d}".format(n_frames), flush=True)
    print("  Number of compounds: {:>9d}"
          .format(n_compounds),
          flush=True)
    if np.any(np.modf(dtrajs1)[0] != 0):
        warnings.warn("At least one element of the first discrete"
                      " trajectory is not an integer", RuntimeWarning)
    if np.any(np.modf(dtrajs2)[0] != 0):
        warnings.warn("At least one element of the second discrete"
                      " trajectory is not an integer", RuntimeWarning)

    states = np.unique(dtrajs2)
    n_states = len(states)

    BEGIN, END, EVERY, n_frames = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=n_frames)
    RESTART, effective_restart = mdt.check.restarts(
        restart_every_nth_frame=args.RESTART,
        read_every_nth_frame=EVERY,
        n_frames=n_frames)
    dtrajs1 = dtrajs1[BEGIN:END:EVERY]
    dtrajs2 = dtrajs2[BEGIN:END:EVERY]

    trans_info = dtraj_transition_info(dtraj=dtrajs1)

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Calculating remain probability", flush=True)
    timer = datetime.now()
    timer_block = datetime.now()

    p = remain_prob_discrete(dtraj1=dtrajs1,
                             dtraj2=dtrajs2,
                             restart=effective_restart,
                             continuous=args.CONTINUOUS,
                             discard_neg_start=args.DISCARD_NEG_START,
                             discard_all_neg=args.DISCARD_ALL_NEG)
    del dtrajs1, dtrajs2

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Fitting remain probability", flush=True)
    timer = datetime.now()

    lag_times = np.arange(n_frames, dtype=np.uint32)

    if args.ENDFIT is None:
        endfit = int(0.9 * len(lag_times))
    else:
        _, endfit = mdt.nph.find_nearest(lag_times,
                                         args.ENDFIT,
                                         return_index=True)
    endfit += 1  # To make args.ENDFIT inclusive

    fit_start = np.zeros(n_states, dtype=np.uint32)  # inclusive
    fit_stop = np.zeros(n_states, dtype=np.uint32)   # exclusive
    popt = np.full((n_states, 2), np.nan, dtype=np.float32)
    perr = np.full((n_states, 2), np.nan, dtype=np.float32)
    for i in range(n_states):
        stopfit = np.argmax(p[:, i] < args.STOPFIT)
        if stopfit == 0 and p[:, i][stopfit] >= args.STOPFIT:
            stopfit = len(p[:, i])
        elif stopfit < 2:
            stopfit = 2
        fit_stop[i] = min(endfit, stopfit)
        popt[i], perr[i] = mdt.func.fit_kww(
            xdata=lag_times[fit_start[i]:fit_stop[i]],
            ydata=p[:, i][fit_start[i]:fit_stop[i]])
    tau_mean = popt[:, 0] / popt[:, 1] * gamma(1 / popt[:, 1])

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()

    header = (
        "Average lifetime of all discrete states in the discretized\n"
        "trajectory as function of another set of discrete states\n"
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
        "The first colum contains the lag times (in trajectory steps).\n"
        "The first row contains the states of the second trajectory\n"
        "used to discretize the remain probability.\n"
        "\n"
        "Fit:\n"
        "Start (steps):"
        .format(args.CONTINUOUS,
                args.DISCARD_NEG_START,
                args.DISCARD_ALL_NEG,
                n_frames, n_compounds,
                trans_info[0], trans_info[1],
                trans_info[2], trans_info[3],
                trans_info[4],
                trans_info[5], 100 * trans_info[5] / trans_info[4],
                trans_info[6], 100 * trans_info[6] / trans_info[4],
                trans_info[7], 100 * trans_info[7] / trans_info[4],
                trans_info[8], 100 * trans_info[8] / trans_info[4]
                )
    )
    for i in range(n_states):
        header += " {:>16.9e}".format(fit_start[i])
    header += ("\n"
               "Stop (steps): ")
    for i in range(n_states):
        header += " {:>16.9e}".format(fit_stop[i])

    header += ("\n"
               "<tau> (steps):")
    for i in range(n_states):
        header += " {:>16.9e}".format(tau_mean[i])

    header += ("\n"
               "tau (steps):  ")
    for i in range(n_states):
        header += " {:>16.9e}".format(popt[i][0])
    header += ("\n"
               "Std. dev.:    ")
    for i in range(n_states):
        header += " {:>16.9e}".format(perr[i][0])

    header += ("\n"
               "beta:         ")
    for i in range(n_states):
        header += " {:>16.9e}".format(popt[i][1])
    header += ("\n"
               "Std. dev.:    ")
    for i in range(n_states):
        header += " {:>16.9e}".format(perr[i][1])
    header += "\n"

    mdt.fh.savetxt_matrix(fname=args.OUTFILE,
                          data=p,
                          var1=lag_times,
                          var2=states,
                          header=header)

    print("  Created {}".format(args.OUTFILE), flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)




    print("\n\n\n{} done".format(os.path.basename(sys.argv[0])))
    print("Elapsed time:         {}"
          .format(datetime.now() - timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)
