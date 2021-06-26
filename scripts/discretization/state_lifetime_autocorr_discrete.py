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
from scipy.special import gamma
import mdtools as mdt
from state_lifetime_autocorr import dtraj_transition_info




def autocorr_dtraj(dtraj1, dtraj2, restart=1):
    """
    Calculate the autocorrelation function of a first discretized
    trajectory, with the speciality that the autocovariance is set to
    one, if the state :math:`S` after a lag time :math:`\tau` is the
    same as at time :math:`t_0`. Otherwise, the autocovariance is zero.
    Additonally, this autocorrelation function is resolved with respect
    to the states :math:`S^\prime` in a second discretized trajectory.
    That means the calculated quantity is
    
    .. math::
        C(\tau, S^\prime) = \langle \frac{S(t_0) S(t_0+\tau)}{S(t_0) S(t_0)} \delta_{S^\prime(t_0), S^\prime} \rangle
    
    with :math:`\delta_{S^\prime(t_0), S^\prime}` as well as
    :math:`S(t) \cdot S(t')` being the Kronecker delta:
    
    .. math::
        S(t) S(t') = \delta_{S(t),S(t')}
    
    The brackets :math:`\langle ... \rangle` denote averaging over all
    states :math:`S` in the first trajectory and over all possible
    starting times :math:`t_0`. You can interprete
    :math:`C(\tau, S^\prime)` as the percentage of primary states that
    have not changed or have come back to the initial primary state
    after a lag time :math:`\tau` given that the compound was in the
    secondary state :math:`S^\prime` at time :math:`t_0`.
    
    Between this function and :func:`state_decay_discrete` from
    state_decay_discrete.py exists a subtle but distinct difference.
    Compounds that return into their initial state after being in at
    least one other state, increase the autocorrelation function, since
    :math:`S(t_0) S(t_0+\tau)` will be one again. However, in
    :func:`state_decay_discrete` :math:`S(t_0) S(t_0+\tau)` will stay
    zero once it has become zero. That means :func:`state_decay_discrete`
    is insensitive of compounds that return into their initial states.
    
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
    
    Returns
    -------
    autocorr : numpy.ndarray
        Array of shape ``(f, m)``, where ``m`` is the number of
        secondary states (states in the second discretized trajectory).
        The `ij`-th element of `autocorr` is the autocorrelation
        function at a lag time of `i` frames given that the compound
        was in the secondary state `j` at time :math:`t_0`.
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
    
    dtraj2 = mdt.nph.sequenize(dtraj2,
                               step=np.uint32(1),
                               start=np.uint32(0))
    n_states = np.max(dtraj2) + 1
    if np.min(dtraj2) != 0:
        raise ValueError("The minimum of the reordered second trajectory"
                         " is not zero. This should not have happened")
    
    n_frames = dtraj1.shape[0]
    n_compounds = dtraj1.shape[1]
    autocorr = np.zeros((n_frames, n_states), dtype=np.float32)
    autocov = np.zeros(n_compounds, dtype=bool)
    norm = np.zeros((n_frames, n_states), dtype=np.uint32)
    
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
        
        bin_ix_u, counts = np.unique(dtraj2[t0], return_counts=True)
        masks = (dtraj2[t0] == bin_ix_u[:,None])
        norm[1:n_frames-t0][:,bin_ix_u] += counts.astype(np.uint32)
        for lag in range(1, n_frames-t0):
            np.equal(dtraj1[t0], dtraj1[t0+lag], out=autocov)
            for i, b in enumerate(bin_ix_u):
                autocorr[lag][b] += np.count_nonzero(autocov[masks[i]])
    
    del dtraj1, dtraj2, autocov, masks
    
    if not np.all(norm[0] == 0):
        raise ValueError("The first element of norm is not zero. This"
                         " should not have happened")
    norm[0] = 1
    autocorr /= norm
    
    if not np.all(autocorr[0] == 0):
        raise ValueError("The first element of autocorr is not zero."
                         " This should not have happened")
    autocorr[0,:] = 1
    if np.any(autocorr > 1):
        raise ValueError("At least one element of autocorr is greater"
                         " than one. This should not have happened")
    if np.any(autocorr < 0):
        raise ValueError("At least one element of autocorr is less than"
                         " zero. This should not have happened")
    
    return autocorr








if __name__ == '__main__':
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read two discretized trajectories, as e.g."
                     " generated by discrete_coord.py or discrete_hex.py."
                     " and calculate the average lifetime of the"
                     " discrete states in the first trajectory as"
                     " function of the states in the second trajectory."
                     " This is done by computing the autocorrelation"
                     " function of the first discrete trajectory, with"
                     " the speciality that the autocovariance is set to"
                     " one if the state after a lag time tau is the same"
                     " as at time t0. Otherwise, the autocovariance is"
                     " set to zero. Finally, the autocorrelation"
                     " function (normalized autocovariance) is fitted by"
                     " a stretched exponential function, whose integral"
                     " from zero to infinity is the averave lifetime of"
                     " all states in the first discretized trajectory."
                     " Between this script state_decay_discrete.py"
                     " exists a subtle but distinct difference."
                     " Compounds that return into their initial state"
                     " after being in at least one other state will"
                     " increase the autocorrelation function again."
                     " However, in state_decay_discrete.py compounds"
                     " that return into their initial states will not"
                     " have any influence."
                     )
    )
    
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
             " the autocorrelation function. This must be an integer"
             " multiple of --every. Ideally, RESTART should be larger"
             " than the longest lifetime to ensure independence of each"
             " restart window. Default: 100"
    )
    
    parser.add_argument(
        '--end-fit',
        dest='ENDFIT',
        type=float,
        required=False,
        default=None,
        help="End time for fitting the autocorrelation function (in"
             " trajectory steps). Inclusive, i.e. the time given here is"
             " still included in the fit. Default: End at 90%% of the"
             " trajectory."
    )
    parser.add_argument(
        '--stop-fit',
        dest='STOPFIT',
        type=float,
        required=False,
        default=0.01,
        help="Stop fitting the autocorrelation function as soon as it"
             " falls below this value. The fitting is stopped by"
             " whatever happens earlier: --end-fit or --stop-fit."
             " Default: 0.01"
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
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Calculating autocorrelation function", flush=True)
    timer = datetime.now()
    
    autocorr = autocorr_dtraj(dtraj1=dtrajs1,
                              dtraj2=dtrajs2,
                              restart=effective_restart)
    del dtrajs1, dtrajs2
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Fitting autocorrelation function", flush=True)
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
        stopfit = np.argmax(autocorr[:,i] < args.STOPFIT)
        if stopfit == 0 and autocorr[:,i][stopfit] >= args.STOPFIT:
            stopfit = len(autocorr[:,i])
        elif stopfit < 2:
            stopfit = 2
        fit_stop[i] = min(endfit, stopfit)
        popt[i], perr[i] = mdt.func.fit_kww(
                               xdata=lag_times[fit_start[i]:fit_stop[i]],
                               ydata=autocorr[:,i][fit_start[i]:fit_stop[i]])
    tau_mean = popt[:,0]/popt[:,1] * gamma(1/popt[:,1])
    
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
        "trajectory as function of another set of discrete states\n"
        "\n"
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
        "The average lifetime <tau> is estimated from the integral of a\n"
        "stretched exponential fit of the autocorrelation function of\n"
        "the discrete trajectory. The autocorrelation of the discrete\n"
        "trajectory is calculated with the speciality that the\n"
        "autocovariance is set to one if the state S after a lag time\n"
        "t is the same as at time t0. Otherwise, the autocovariance is\n"
        "set to zero.\n"
        "\n"
        "Autocovariance:\n"
        "  S(t0)*S(t0+t) = 1, if S(t0)=S(t0+t)\n"
        "  S(t0)*S(t0+t) = 0, otherwise\n"
        "\n"
        "Autocorrelation function:\n"
        "  C(t) = < S(t0)*S(t0+t) / S(t0)*S(t0) >\n"
        "  <...> = Average over all states S in the trajectory and\n"
        "          over all possible starting times t0\n"
        "  You can interprete C(t) as the percentage of states that\n"
        "  have not changed or have come back to the initial state\n"
        "  after a lag time t\n"
        "\n"
        "The autocorrelation is fitted using a stretched exponential\n"
        "function, also known as Kohlrausch-Williams-Watts (KWW)\n"
        "function:\n"
        "  f(t) = exp[-(t/tau)^beta]\n"
        "  beta is constrained to the intervall [0, 1]\n"
        "  tau must be positive\n"
        "\n"
        "The average lifetime <tau> is calculated as the integral of\n"
        "the KWW function from zero to infinity:\n"
        "  <tau> = integral_0^infty exp[-(t/tau)^beta] dt\n"
        "        = tau/beta * Gamma(1/beta)\n"
        "  Gamma(x) = Gamma function\n"
        "  If beta=1, <tau>=tau\n"
        "\n"
        "\n"
        "The first colum contains the lag times (in trajectory steps).\n"
        "The first row contains the states of the second trajectory\n"
        "used for discretizing the autocorrelation function of the first\n"
        "trajectory.\n"
        "\n"
        "Fit:\n"
        "Start (steps):"
        .format(n_frames, n_compounds,
                trans_info[0], trans_info[1],
                trans_info[2], trans_info[3],
                trans_info[4],
                trans_info[5], 100*trans_info[5]/trans_info[4],
                trans_info[6], 100*trans_info[6]/trans_info[4],
                trans_info[7], 100*trans_info[7]/trans_info[4],
                trans_info[8], 100*trans_info[8]/trans_info[4]
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
                          data=autocorr,
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
