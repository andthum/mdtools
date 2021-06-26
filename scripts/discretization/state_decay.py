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
import warnings
from datetime import datetime
import psutil
import argparse
import numpy as np
from scipy import optimize
import mdtools as mdt




def state_decay(dtraj, restart=1, cut=False, cut_and_merge=False):
    """
    Take a discretized trajectory as input and calculate the propability
    :math:`p[\\xi(t_0 + \Delta t) \in S_i | \\xi(t_0) \in S_i]` that a
    compound is still in the same state, in which it was at time
    :math:`t_0`, after a lag time :math:`\Delta t`:
    
    Note
    ----
    You probably want to use :func:`state_decay_fast` instead of this
    function, since it gives you the same result in less time.
    
    Parameters
    ----------
    dtraj : array_like
        The discretized trajectory. Array of shape ``(f, n)`` where ``f``
        is the number of frames and ``n`` is the number of compounds.
        The elements of `dtraj` are interpreted as the indices of the
        states in which a given compound is at a given frame.
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
        Array of shape ``(f,)``. The `i`-th element of `decay` is the
        propability that a compound is still in its initial state after
        `i` frames.
    """
    
    dtraj = np.asarray(dtraj)
    if dtraj.ndim != 2:
        raise ValueError("dtraj must have two dimensions")
    
    if cut and cut_and_merge:
        raise ValueError("cut and cut_and_merge are mutually exclusive")
    if cut or cut_and_merge:
        valid = np.any(dtraj>=0, axis=0)
        if np.sum(valid)/len(valid) < 0.9:
            # Trade-off between reduction of computations and memory
            # consumption
            dtraj = np.asarray(dtraj[:,valid], order='C')
    
    n_frames = dtraj.shape[0]
    n_compounds = dtraj.shape[1]
    transitions = (np.diff(dtraj, axis=0) != 0)
    transitions = np.insert(transitions, 0, np.zeros(n_compounds), axis=0)
    n_trans = np.zeros(n_compounds, dtype=np.uint32)
    decay = np.zeros(n_frames, dtype=np.float32)
    
    no_cut_at_all = not any([cut, cut_and_merge])
    if cut or cut_and_merge:
        valid = np.zeros(n_compounds, dtype=bool)
        norm = np.zeros(n_frames, dtype=np.uint32)
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
            np.greater_equal(dtraj[t0], 0, out=valid)
        for lag in range(1, n_frames-t0):
            n_trans += transitions[t0+lag]
            if no_cut_at_all:
                n_stay = np.count_nonzero(n_trans==0)
                if n_stay == 0:
                    break
            else:  # cut or cut_and_merge
                if cut:
                    np.greater_equal(dtraj[t0+lag], 0, out=valid2)  # True means valid, but False does not necessarily mean invalid
                    # If the first transition that a particle makes is a
                    # valid transition, this particle contributes to the
                    # norm until the end of the trajectory ("forever")
                    np.greater_equal(n_trans, 2, out=valid3)  # True means valid, but False does not necessarily mean invalid
                    valid2 |= valid3  # Now True means valid and False means invalid
                    valid &= valid2
                n_valid = np.count_nonzero(valid)
                if n_valid == 0:
                    break
                norm[lag] += n_valid
                n_stay = np.count_nonzero(n_trans[valid]==0)
            decay[lag] += n_stay
        n_trans[:] = 0
    
    del dtraj, transitions, n_trans
    
    
    if no_cut_at_all:
        n_restarts = np.arange(n_frames, 0, -1, dtype=np.float32)
        n_restarts /= restart
        np.ceil(n_restarts, out=n_restarts)
        decay /= n_restarts
        decay /= n_compounds
    else:
        if norm[0] != 0:
            raise ValueError("The first element of norm is not zero."
                             " This should not have happened")
        norm[0] = 1
        decay /= norm
    
    if decay[0] != 0:
        raise ValueError("The first element of decay is not zero. This"
                         " should not have happened")
    decay[0] = 1
    if np.any(decay > 1):
        raise ValueError("At least one element of decay is greater than"
                         " one. This should not have happened")
    if np.any(decay < 0):
        raise ValueError("At least one element of decay is less than"
                         " zero. This should not have happened")
    
    return decay




def state_decay_fast(dtraj, restart=1, cut=False, cut_and_merge=False):
    """
    Take a discretized trajectory as input and calculate the propability
    :math:`p[\\xi(t_0 + \Delta t) \in S_i | \\xi(t_0) \in S_i]` that a
    compound is still in the same state, in which it was at time
    :math:`t_0`, after a lag time :math:`\Delta t`:
    
    Note
    ----
    This is a faster version of :func:`state_decay`. However, in the
    case that the number of compounds is greater than the number of
    frames, :func:`state_decay` might be faster.
    
    Parameters
    ----------
    dtraj : array_like
        The discretized trajectory. Array of shape ``(f, n)`` where ``f``
        is the number of frames and ``n`` is the number of compounds.
        The elements of `dtraj` are interpreted as the indices of the
        states in which a given compound is at a given frame.
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
        Array of shape ``(f,)``. The `i`-th element of `decay` is the
        propability that a compound is still in its initial state after
        `i` frames.
    """
    
    dtraj = np.asarray(dtraj)
    if dtraj.ndim != 2:
        raise ValueError("dtraj must have two dimensions")
    
    if cut and cut_and_merge:
        raise ValueError("cut and cut_and_merge are mutually exclusive")
    if cut or cut_and_merge:
        valid = np.any(dtraj>=0, axis=0)
        if np.sum(valid)/len(valid) < 0.9:
            # Trade-off between reduction of computations and memory
            # consumption
            dtraj = np.asarray(dtraj[:,valid], order='C')
    
    n_frames = dtraj.shape[0]
    n_compounds = dtraj.shape[1]
    transitions = (np.diff(dtraj, axis=0) != 0)
    transitions = np.insert(transitions, 0, np.zeros(n_compounds), axis=0)
    compound_ix = np.arange(n_compounds, dtype=np.uint32)
    lag = np.zeros(n_compounds, dtype=np.uint32)
    lag_inf = np.zeros(n_compounds, dtype=bool)
    decay = np.zeros(n_frames, dtype=np.float32)
    
    no_cut_at_all = not any([cut, cut_and_merge])
    if cut or cut_and_merge:
        valid = np.zeros(n_compounds, dtype=bool)
    if cut_and_merge:
        norm = np.zeros(int(np.ceil((n_frames-1)/restart)),
                        dtype=np.uint32)
    if cut:
        norm = np.zeros(n_frames, dtype=np.uint32)
        norm_mask = np.zeros(n_compounds, dtype=bool)
        mask = np.zeros(n_compounds, dtype=bool)
        lag_tmp = np.zeros(n_compounds, dtype=np.uint32)
    
    
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
        
        np.argmax(transitions[t0+1:], axis=0, out=lag)
        lag += 1
        np.invert(transitions[t0:][lag,compound_ix], out=lag_inf)
        if no_cut_at_all:
            ix = range(n_compounds)
        else:  # cut or cut_and_merge
            np.greater_equal(dtraj[t0], 0, out=valid)
            if cut:
                np.less(dtraj[t0+lag,compound_ix], 0, out=mask)  # invalid
                # All particles starting from valid states contribute to
                # the norm until an invalid transition
                np.bitwise_and(valid, mask, out=norm_mask)  # valid start and invalid first transition
                for i in np.flatnonzero(norm_mask):
                    norm[:lag[i]] += 1
                # If the first transition that a particle makes is a
                # valid transition, this particle contributes to the
                # norm until the end of the trajectory ("forever")
                np.invert(mask, out=norm_mask)
                norm_mask &= valid  # valid start and valid first transition
                for i in np.flatnonzero(norm_mask):
                    norm[:n_frames-t0] += 1
                np.subtract(lag, mask, out=lag_tmp)
                np.greater(lag_tmp, 0, out=mask)  # valid lag times
                valid &= mask
            ix = np.flatnonzero(valid)
            if cut_and_merge:
                # Number of valid compounds per restarting point
                norm[t0//restart] += len(ix)
        lag[lag_inf] = n_frames - t0
        for i in ix:
            decay[:lag[i]] += 1
    
    del transitions, compound_ix, lag, lag_inf
    
    n_restarts = np.arange(n_frames, 0, -1, dtype=np.float32)
    n_restarts /= restart
    np.ceil(n_restarts, out=n_restarts)
    if no_cut_at_all:
        decay /= n_restarts
        decay /= n_compounds
    elif cut_and_merge:
        n_restarts -= 1
        n_restarts = n_restarts.astype(np.uint32)
        np.cumsum(norm, out=norm)
        # Which restarting points were visited by which lag times?
        # = Number of valid compounds per lag time
        decay[1:] /= norm[n_restarts[1:]]
        restarts = np.arange(0, n_frames, restart, dtype=np.uint32)
        decay[0] /= np.count_nonzero(dtraj[restarts] >= 0)
        del dtraj, restarts
    elif cut:
        decay /= norm
    
    if not np.isclose(decay[0], 1, atol=0.01, rtol=0):
        warnings.warn("The first element of decay is not one, but {}"
                      .format(decay[0]), RuntimeWarning)
    decay[0] = 1
    if np.any(decay > 1):
        raise ValueError("At least one element of decay is greater than"
                         " one. This should not have happened")
    if np.any(decay < 0):
        raise ValueError("At least one element of decay is less than"
                         " zero. This should not have happened")
    
    return decay




def exp_decay(t, k=1):
    """
    Exponential decay fuction:
    
    .. math:
        f(t) = e^{-kt}
    
    Parameters
    ----------
    t : scalar or array_like
        Time(s) at which to evaluate :math:`f(t)`.
    k : scalar
        Rate constant.
    
    Returns
    -------
    decay : scalar or numpy.ndarray
        :math:`f(t)`.
    """
    
    return np.exp(-k * t)




def fit_exp_decay(xdata, ydata, ysd=None):
    """
    Fit an exponential decay function to `ydata` using
    :func:`scipy.optimize.curve_fit`. Fit function:
    
    .. math::
        f(t) = e^{-kt}
    
    Parameters
    ----------
    xdata : array_like
        The independent variable where the data is measured.
    ydata : array_like
        The dependent data. Must have the same shape as `xdata`.
    ysd : array_like, optional
        The standard deviation of `ydata`. Must have the same shape as
        `ydata`.
    
    Returns
    -------
    k : scalar
        Rate constant.
    k_sd : scalar
        Standard deviation of `k` from fitting.
    """
    
    try:
        popt, pcov = optimize.curve_fit(f=exp_decay,
                                        xdata=xdata,
                                        ydata=ydata,
                                        sigma=ysd,
                                        absolute_sigma=True)
    except (ValueError, RuntimeError, optimize.OptimizeWarning) as err:
        print(flush=True)
        print("An error has occurred during fitting:", flush=True)
        print("{}".format(err), flush=True)
        print("Setting fit parameters to numpy.nan", flush=True)
        print(flush=True)
        popt = np.array([np.nan])
        perr = np.array([np.nan])
    else:
        perr = np.sqrt(np.diag(pcov))
    
    return popt[0], perr[0]








if __name__ == '__main__':
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read a discretized trajectory, as e.g. generated"
                     " by discrete_coord.py or discrete_hex.py, and"
                     " calculate the propability that a compound, whose"
                     " discrete states are tracked in this trajectory,"
                     " is still in the same state, in which it was at"
                     " time t0, after a lag time t. The resulting decay"
                     " is fitted by an exponential decay function."
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
             " the decay function. Default: 100"
    )
    
    group.add_argument(
        '--cut',
        dest='CUT',
        required=False,
        default=False,
        action='store_true',
        help="If set, states with negative indices are effectively cut"
             " out of the trajectory. The cutting edges are not merged"
             " so that you effectively get multiple smaller trajectories."
             " This means, negative states are ignored completely. Even"
             " transitions from positive to negative states will not be"
             " counted and hence will not influence the propability p to"
             " stay in the same state. Practically seen, you discard all"
             " restarting and end points where the state index is"
             " negative."
    )
    group.add_argument(
        '--cut-and-merge',
        dest='CUT_AND_MERGE',
        required=False,
        default=False,
        action='store_true',
        help="If set, states with negative indices are effectively cut"
             " out of the trajectory. The cutting edges are merged to"
             " one new trajectory. This means, transitions from positive"
             " to negative states will still be counted and decrease the"
             " propability p to stay in the same state. But otherwise,"
             " negative states are completely ignored and do not"
             " influence the propability p. Practically seen, you"
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
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Calculating decay function", flush=True)
    timer = datetime.now()
    timer_block = datetime.now()
    
    decay = [None,] * NBLOCKS
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
        decay[block] = state_decay_fast(
                           dtraj=dtrajs[block*blocksize:(block+1)*blocksize],
                           restart=effective_restart,
                           cut=args.CUT,
                           cut_and_merge=args.CUT_AND_MERGE)
    
    decay = np.asarray(decay)
    if NBLOCKS > 1:
        decay, decay_sd = mdt.stats.block_average(decay)
    else:
        decay = np.squeeze(decay)
        decay_sd = None
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Fitting decay function", flush=True)
    timer = datetime.now()
    
    lag_times = np.arange(blocksize, dtype=np.uint32)
    
    if args.ENDFIT is None:
        endfit = int(0.9 * len(lag_times))
        args.ENDFIT = lag_times[endfit]
    else:
        _, endfit = mdt.nph.find_nearest(lag_times,
                                         args.ENDFIT,
                                         return_index=True)
    
    stopfit = np.nonzero(decay < args.STOPFIT)[0]
    if len(stopfit) == 0:
        stopfit = len(decay)
    else:
        stopfit = stopfit[0]
    
    # TODO: Only fit linear region
    valid = np.isfinite(decay)
    valid[min(endfit, stopfit):] = False
    if not np.any(valid):
        k = np.nan
        k_sd = np.nan
    elif decay_sd is None:
        k, k_sd = fit_exp_decay(xdata=lag_times[valid],
                                ydata=decay[valid])
    else:
        decay_sd[decay_sd==0] = 1e-20
        k, k_sd = fit_exp_decay(xdata=lag_times[valid],
                                ydata=decay[valid],
                                ysd=decay_sd[valid])
    fit = exp_decay(t=lag_times, k=k)
    fit[~valid] = np.nan
    del valid
    
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
        "which it was at time t0, after a lag time t.\n"
        "\n"
        "\n"
        "Number of frames (per compound):                         {:>9d}\n"
        "Number of compounds:                                     {:>9d}\n"
        "Number of compounds that never leave their state:        {:>9d}\n"
        "Number of compounds that are always in a negative state: {:>9d}\n"
        "Number of compounds that are never  in a negative state: {:>9d}\n"
        "Total number of frames with negative states:             {:>9d}\n"
        .format(n_frames, n_compounds,
                np.sum(np.all(dtrajs[:NBLOCKS*blocksize]==dtrajs[0], axis=0)),
                np.sum(np.all(dtrajs[:NBLOCKS*blocksize]<0, axis=0)),
                np.sum(np.all(dtrajs[:NBLOCKS*blocksize]>=0, axis=0)),
                np.count_nonzero(dtrajs[:NBLOCKS*blocksize]<0)
        )
    )
    if NBLOCKS > 1:
        header += (
            "\n"
            "\n"
            "The columns contain:\n"
            "  1 Lag time t (in trajectory steps)\n"
            "  2 Decay function\n"
            "  3 Standard deviation of the decay function\n"
            "  4 Exponential fit of the decay function: f(t) = exp(-k*t)\n"
            "\n"
            "Column number:\n"
            "{:>14d} {:>16d} {:>16d} {:>16d}\n"
            .format(1, 2, 3, 4)
        )
        data = np.column_stack([lag_times, decay, decay_sd, fit])
    else:
        header += (
            "\n"
            "\n"
            "The columns contain:\n"
            "  1 Lag time t (in trajectory steps)\n"
            "  2 Decay function\n"
            "  3 Exponential fit of the decay function: f(t) = exp(-k*t)\n"
            "\n"
            "Column number:\n"
            "{:>14d} {:>16d} {:>16d}\n"
            .format(1, 2, 3)
        )
        data = np.column_stack([lag_times, decay, fit])
    header += (
        "\n"
        "Fit:\n"
        "Rate constant k (in 1/step):    {:>16.9e}\n"
        "Std. dev. of  k (in 1/step):    {:>16.9e}\n"
        "Half-life t2 = ln(2)/k (steps): {:>16.9e}\n"
        "Std. dev. of t2        (steps): {:>16.9e}\n"
        "Lifetime tau = 1/k (steps):     {:>16.9e}\n"
        "Std. dev. of tau   (steps):     {:>16.9e}\n"
        "Start fit (steps):              {:>16d}\n"
        "Stop  fit (steps):              {:>16d}\n"
        .format(k, k_sd,
                np.log(2)/k, np.sqrt((np.log(2)/k)**2 * (k_sd/k)**2),
                1/k, np.sqrt((1/k)**2 * (k_sd/k)**2),
                0, min(endfit, stopfit)
        )
    )
    
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
