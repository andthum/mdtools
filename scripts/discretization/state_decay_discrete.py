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




import sys
import os
from datetime import datetime
import psutil
import argparse
import numpy as np
from scipy import optimize
import mdtools as mdt
from state_decay import fit_exp_decay




def state_decay_discrete(dtraj1, dtraj2, restart=1, cut=False,
        cut_and_merge=False):
    """
    Take two discretized trajectories as input and calculate the
    propability
    :math:`p[\\xi(t_0 + \Delta t) \in S_i | \\xi(t_0) \in S_i, S_j^\prime]`
    that a compound, whose discrete states are tracked in
    the first trajectory, is still in the same state, in which it was at
    time :math:`t_0`, after a lag time :math:`\Delta t`. This decay
    function is calculated as function of the states :math:`S_i^\prime`
    of the second trajectory.
    
    Parameters
    ----------
    dtraj1 : array_like
        The first discretized trajectory. Array of shape ``(f, n)``
        where ``f`` is the number of frames and ``n`` is the number of
        compounds. The elements of `dtraj1` are interpreted as the
        indices of the states in which a given compound is at a given
        frame.
    dtraj2 : array_like
        The second discretized trajectory. Must have the same shape as
        `dtraj1`.
    restart : int, optional
        Number of frames between restarting points :math:`t_0`.
    cut : int, optional
        If ``True``, states with negative indices are effectively cut
        out of the trajectory. The cutting edges are not merged so that
        you effectively get multiple smaller trajectories. This means,
        negative states are ignored completely. Even transitions from
        positive to negative states will not be counted and hence will
        not influence the propability :math:`p` to stay in the same
        state. Practically seen, you discard all restarting and end
        points where the state index is negative.
    cut_and_merge : int, optional
        If ``True``, states with negative indices are effectively cut
        out of the trajectory. The cutting edges are merged to one new
        trajectory. This means, transitions from positive to negative
        states will still be counted and decrease the propability
        :math:`p` to stay in the same state. But otherwise, negative
        states are completely ignored and do not influence the
        propability :math:`p`. Practically seen, you discard all
        restarting points where the state index is negative. In short,
        the difference between `cut` and `cut_and_merge` is that a
        transition from a positive to a negative state is not counted
        when using `cut`, whereas it is counted when using
        `cut_and_merge`. In both cases all transitions starting from
        negative states are ignored, as well as compounds that stay in
        the same negative state. `cut` and `cut_and_merge` are mutually
        exclusive.
    
    Returns
    -------
    decay : numpy.ndarray
        Array of shape ``(f, n)``. The `ij`-th element of `decay` is the
        propability that a compound is still in its initial state after
        `i` frames given that it was in state `j` of `dtraj2` at time
        :math:`t_0`.
    """
    
    dtraj1 = np.asarray(dtraj1)
    dtraj2 = np.array(dtraj2, copy=True)
    if dtraj1.ndim != 2:
        raise ValueError("dtraj1 must have two dimensions")
    if dtraj1.shape != dtraj2.shape:
        raise ValueError("Both trajectories must have the same shape")
    
    if cut and cut_and_merge:
        raise ValueError("cut and cut_and_merge are mutually exclusive")
    if cut or cut_and_merge:
        valid = np.any(dtraj1>=0, axis=0)
        if np.sum(valid)/len(valid) < 0.9:
            # Trade-off between reduction of computations and memory
            # consumption
            dtraj1 = np.asarray(dtraj1[:,valid], order='C')
            dtraj2 = np.asarray(dtraj2[:,valid], order='C')
    
    # TODO: Enable second trajectories with intermediate unoccupied states
    dtraj2 -= np.min(dtraj2)
    n_states = len(np.unique(dtraj2))
    
    n_frames = dtraj1.shape[0]
    n_compounds = dtraj1.shape[1]
    transitions = (np.diff(dtraj1, axis=0) != 0)
    transitions = np.insert(transitions, 0, np.zeros(n_compounds), axis=0)
    n_trans = np.zeros(n_compounds, dtype=np.uint32)
    decay = np.zeros((n_frames, n_states), dtype=np.float32)
    norm = np.zeros((n_frames, n_states), dtype=np.uint32)
    mask = np.ones(n_compounds, dtype=bool)
    
    no_cut_at_all = not any([cut, cut_and_merge])
    if cut or cut_and_merge:
        valid = np.zeros(n_compounds, dtype=bool)
    if cut:
        valid2 = np.zeros(n_compounds, dtype=bool)
        valid3 = np.zeros(n_compounds, dtype=bool)
    
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
        
        if cut or cut_and_merge:
            np.greater_equal(dtraj1[t0], 0, out=valid)
            bin_ix_u, counts = np.unique(dtraj2[t0][valid],
                                         return_counts=True)
        else:
            bin_ix_u, counts = np.unique(dtraj2[t0], return_counts=True)
        if no_cut_at_all or cut_and_merge:
            norm[1:n_frames-t0][:,bin_ix_u] += counts.astype(np.uint32)
        for lag in range(1, n_frames-t0):
            n_trans += transitions[t0+lag]
            if cut:
                np.greater_equal(dtraj1[t0+lag], 0, out=valid2)  # True means valid, but False does not necessarily mean invalid
                # If the first transition that a particle makes is a
                # valid transition, this particle contributes to the
                # norm until the end of the trajectory ("forever")
                np.greater_equal(n_trans, 2, out=valid3)  # True means valid, but False does not necessarily mean invalid
                valid2 |= valid3  # Now True means valid and False means invalid
                valid &= valid2
                if not np.any(valid):
                    break
            else:
                if np.all(n_trans > 0):
                    break
            for b in bin_ix_u:
                np.equal(dtraj2[t0], b, out=mask)
                if cut or cut_and_merge:
                    mask &= valid
                decay[lag][b] += np.count_nonzero(n_trans[mask]==0)
                if cut:
                    norm[lag][b] += np.count_nonzero(mask)
        n_trans[:] = 0
    
    del dtraj1, dtraj2, transitions, n_trans, mask
    
    if not np.all(norm[0] == 0):
        raise ValueError("The first element of norm is not zero. This"
                         " should not have happened")
    norm[0] = 1
    decay /= norm
    
    if not np.all(decay[0] == 0):
        raise ValueError("The first element of decay is not zero. This"
                         " should not have happened")
    decay[0,:] = 1
    if np.any(decay > 1):
        raise ValueError("At least one element of decay is greater than"
                         " one. This should not have happened")
    if np.any(decay < 0):
        raise ValueError("At least one element of decay is less than"
                         " zero. This should not have happened")
    
    return decay








if __name__ == '__main__':
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read two discretized trajectories, as e.g."
                     " generated by discrete_coord.py or discrete_hex.py."
                     " Calculate the propability that a compound, whose"
                     " discrete states are tracked in the first"
                     " trajectory, is still in same the state, in which"
                     " it was at time t0, after a lag time t. This decay"
                     " function is calculated as function of the states"
                     " of the second trajectory. Finally, an exponential"
                     " fit is applied to the data."
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
             " the decay function. Default: 100"
    )
    
    group.add_argument(
        '--cut',
        dest='CUT',
        required=False,
        default=False,
        action='store_true',
        help="If set, states with negative indices are effectively cut"
             " out of the first trajectory. The cutting edges are not"
             " merged so that you effectively get multiple smaller"
             " trajectories. This means, negative states are ignored"
             " completely. Even transitions from positive to negative"
             " states will not be counted and hence will not influence"
             " the propability p to stay in the same state. Practically"
             " seen, you discard all restarting and end points where the"
             " state index is negative."
    )
    group.add_argument(
        '--cut-and-merge',
        dest='CUT_AND_MERGE',
        required=False,
        default=False,
        action='store_true',
        help="If set, states with negative indices are effectively cut"
             " out of the first trajectory. The cutting edges are merged"
             " to one new trajectory. This means, transitions from"
             " positive to negative states will still be counted and"
             " decrease the propability p to stay in the same state. But"
             " otherwise, negative states are completely ignored and do"
             " not influence the propability p. Practically seen, you"
             " discard all restarting points where the state index is"
             " negative. In short, the difference between --cut and"
             " --cut-and-merge is that a transition from a positive to a"
             " negative state is not counted when using --cut, whereas"
             " it is counted when using --cut-and-merge. In both cases"
             " all transitions starting from negative states are"
             " ignored, as well as compounds that stay in the same"
             " negative state. --cut and --cut-and-merge are mutually"
             " exclusive."
    )
    
    parser.add_argument(
        '--end-fit',
        dest='ENDFIT',
        type=float,
        required=False,
        default=None,
        help="End time for fitting the decay function (in trajectory"
             " steps). Default: End at 90%% of the trajectory."
    )
    parser.add_argument(
        '--stop-fit',
        dest='STOPFIT',
        type=float,
        required=False,
        default=0.01,
        help="Stop fitting the decay function as soon as it falls below"
             " this value. The fitting is stopped by whatever happens"
             " earlier: --end-fit or --stop-fit. Default: 0.01"
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
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Calculating decay function", flush=True)
    timer = datetime.now()
    
    decay = state_decay_discrete(dtraj1=dtrajs1,
                                 dtraj2=dtrajs2,
                                 restart=args.RESTART,
                                 cut=args.CUT,
                                 cut_and_merge=args.CUT_AND_MERGE)
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    print("\n\n\n", flush=True)
    print("Fitting decay function", flush=True)
    timer = datetime.now()
    
    lag_times = np.arange(n_frames, dtype=np.uint32)
    
    if args.ENDFIT is None:
        endfit = int(0.9 * len(lag_times))
        args.ENDFIT = lag_times[endfit]
    else:
        _, endfit = mdt.nph.find_nearest(lag_times,
                                         args.ENDFIT,
                                         return_index=True)
    
    stopfit = [None,] * n_states
    k = np.full(n_states, np.nan, dtype=np.float32)
    k_sd = np.full(n_states, np.nan, dtype=np.float32)
    for i in range(n_states):
        stopfit[i] = np.nonzero(decay[:,i] < args.STOPFIT)[0]
        if len(stopfit[i]) == 0:
            stopfit[i] = len(decay[:,i])
        else:
            stopfit[i] = stopfit[i][0]
        # TODO: Only fit linear region
        valid = np.isfinite(decay[:,i])
        valid[min(endfit, stopfit[i]):] = False
        if np.any(valid):
            k[i], k_sd[i] = fit_exp_decay(xdata=lag_times[valid],
                                          ydata=decay[:,i][valid])
    stopfit = np.asarray(stopfit)
    
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
        "Propability that a compound is still in the same state, in\n"
        "which it was at time t0, after a lag time t as function of\n"
        "another set of states."
        "\n"
        "\n"
        "Number of frames (per compound):                         {:>9d}\n"
        "Number of compounds:                                     {:>9d}\n"
        "Number of compounds that never leave their state:        {:>9d}\n"
        "Number of compounds that are always in a negative state: {:>9d}\n"
        "Number of compounds that are never  in a negative state: {:>9d}\n"
        "Total number of frames with negative states:             {:>9d}\n"
        "\n"
        "\n"
        "The first colum contains the lag times (in trajectory steps).\n"
        "The first row contains the states of the second trajectory\n"
        "used for discretizing the decay function of the first\n"
        "trajectory.\n"
        "\n"
        "Exponential fit of the decay function: f(t) = exp(-k*t):\n"
        "k (in 1/step):"
        .format(n_frames, n_compounds,
                np.sum(np.all(dtrajs1==dtrajs1[0], axis=0)),
                np.sum(np.all(dtrajs1<0, axis=0)),
                np.sum(np.all(dtrajs1>=0, axis=0)),
                np.count_nonzero(dtrajs1<0),
                1, 2, 3
        )
    )
    for i in range(n_states):
        header += " {:>16.9e}".format(k[i])
    header += ("\n"
               "Std. dev.:    ")
    for i in range(n_states):
        header += " {:>16.9e}".format(k_sd[i])
    
    header += ("\n"
               "t2 = ln(2)/k: ")
    for i in range(n_states):
        header += " {:>16.9e}".format(np.log(2)/k[i])
    header += ("\n"
               "Std. dev.:    ")
    for i in range(n_states):
        header += " {:>16.9e}".format(np.sqrt((np.log(2)/k[i])**2 * (k_sd[i]/k[i])**2))
    
    header += ("\n"
               "tau = 1/k:    ")
    for i in range(n_states):
        header += " {:>16.9e}".format(1/k[i])
    header += ("\n"
               "Std. dev.:    ")
    for i in range(n_states):
        header += " {:>16.9e}".format(np.sqrt((1/k[i])**2 * (k_sd[i]/k[i])**2))
    
    header += ("\n"
               "Start fit:    ")
    for i in range(n_states):
        header += " {:>16d}".format(0)
    header += ("\n"
               "Stop  fit:    ")
    for i in range(n_states):
        header += " {:>16d}".format(min(endfit, stopfit[i]))
    header += "\n"
    
    mdt.fh.savetxt_matrix(fname=args.OUTFILE,
                          data=decay,
                          var1=lag_times,
                          var2=states,
                          header=header)
    
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
