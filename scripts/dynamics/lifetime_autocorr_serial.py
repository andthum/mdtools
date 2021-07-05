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
from scipy import special
import mdtools as mdt




# This function is also used by: lifetime_autocorr_parallel.py
def parse_user_input(add_description=""):
    description=("Compute the average lifetime of reference-selection"
                 " complexes, i.e. the mean residence time how long a"
                 " compound of the selection group is within the given"
                 " cutoff of a compound of the reference group. This is"
                 " done by first evaluating the existence function"
                 " (which is one, if a contact exists, and zero"
                 " otherwise) for each compound at each frame and"
                 " afterwards calculating its autocorrelation function."
                 " The autocorrelation function is then fitted by a"
                 " stretched exponential function, whose integral from"
                 " zero to infinity is the averave lifetime."
                 " Additionally the lifetime of unbound reference and"
                 " selection compounds is calculated.")
    parser = argparse.ArgumentParser(
        description=description+add_description
    )

    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="Trajectory file [<.trr/.xtc/.gro/.pdb/.xyz/.mol2/...>]."
             " See supported coordinate formats of MDAnalysis."
    )
    parser.add_argument(
        '-s',
        dest='TOPFILE',
        type=str,
        required=True,
        help="Topology file [<.top/.tpr/.gro/.pdb/.xyz/.mol2/...>]. See"
             " supported topology formats of MDAnalysis."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename."
    )

    parser.add_argument(
        '--ref',
        dest='REF',
        type=str,
        nargs='+',
        required=True,
        help="Reference group. See MDAnalysis selection commands for"
             " possible choices. E.g. 'name Li'"
    )
    parser.add_argument(
        "--sel",
        dest="SEL",
        type=str,
        nargs="+",
        required=True,
        help="Selection group. See MDAnalysis selection commands for"
             " possible choices. E.g. 'type OE'"
    )
    parser.add_argument(
        '-c',
        dest='CUTOFF',
        type=float,
        required=True,
        help="Cutoff distance in Angstrom. A reference and selection"
             " atom are considered to be in contact, if their distance"
             " is less than or equal to this cutoff."
    )
    parser.add_argument(
        '--compound',
        dest='COMPOUND',
        type=str,
        required=False,
        default="atoms",
        help="Contacts between the reference and selection group can be"
             " computed either for individual 'atoms', 'segments',"
             " 'residues', or 'fragments'. Refer to the MDAnalysis user"
             " guide for the meaning of these terms"
             " (https://userguide.mdanalysis.org/1.0.0/groups_of_atoms.html)."
             " Default: 'atoms'"
    )
    parser.add_argument(
        '--min-contacts',
        dest='MINCONTACTS',
        type=int,
        required=False,
        default=1,
        help="Compounds of the reference and selection group are only"
             " considered to be in contact, if there are at least"
             " MINCONTACTS contacts between the atoms of the compounds."
             " --min-contacts is ignored if --compound is set to"
             " 'atoms'. Default: 1"
    )
    parser.add_argument(
        '--intermittency',
        dest='INTERMITTENCY',
        type=int,
        required=False,
        default=0,
        help="Maximum numer of frames a selection atom is allowed to"
             " leave the cutoff range of a reference atom whilst still"
             " being considered to be bound to the reference atom,"
             " provided that it is indeed bound again to the reference"
             " atom after this unbound period. The other way round, a"
             " selection atom is only considered to be bound to a"
             " reference atom, if it has been bound to it for at least"
             " this number of consecutive frames. Default: 0"
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
        "-e",
        dest="END",
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
        default=1000,
        help="Number of frames between restarting points for calculating"
             " the autocorrelation function. This must be an integer"
             " multiply of --every. Ideally, RESTART should be larger"
             " than the longest lifetime to ensure independence of each"
             " restart window. Default: 1000"
    )

    parser.add_argument(
        '--end-fit',
        dest='ENDFIT',
        type=float,
        required=False,
        default=None,
        help="End time for fitting the autocorrelation function in ps."
             " Default: End at 90%% of the lag times."
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

    parser.add_argument(
        '--debug',
        dest='DEBUG',
        required=False,
        default=False,
        action='store_true',
        help="Run in debug mode."
    )

    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

    if args.CUTOFF <= 0:
        raise ValueError("-c must be greater than zero, but you gave {}"
                         .format(args.CUTOFF))
    if (args.COMPOUND != 'atoms' and
        args.COMPOUND != 'segments' and
        args.COMPOUND != 'residues' and
            args.COMPOUND != 'fragments'):
        raise ValueError("--compound must be either 'atoms', 'segments',"
                         " 'residues' or 'fragments', but you gave {}"
                         .format(args.COMPOUND))
    if args.MINCONTACTS < 1:
        raise ValueError("--min-contacts must be greater than zero, but"
                         " you gave {}"
                         .format(args.MINCONTACTS))
    if args.MINCONTACTS > 1 and args.COMPOUND == 'atoms':
        args.MINCONTACTS = 1
        print("\n\n\n", flush=True)
        print("Note: Setting --min-contacts to {}, because --compound\n"
              "  is set to 'atoms'".format(args.MINCONTACTS), flush=True)
    if args.INTERMITTENCY < 0:
        raise ValueError("--intermittency must be equal to or greater"
                         " than zero, but you gave {}"
                         .format(args.INTERMITTENCY))

    return args




# This function is also used by: liftime_hist_serial.py
def unbound_compounds(contact_matrix, axis, debug=False):
    """
    Get all compounds of a reference (selection) group that are not in
    contact with any atom of a selection (reference) group.

    Parameters
    ----------
    contact_matrix : numpy.ndarray or scipy.sparse.spmatrix
        Boolean contact matrix as e.g. generated with
        :func:`mdtools.structure.contact_matrix`. It is assumed that
        each row stands for a reference compound and each column for a
        selection compound.
    axis : int, optional
        Must be either ``0`` or ``1``. If ``0``, unbound selection
        compounds will be returned (columns in which all elements are
        ``False``), if ``1`` unbound reference compounds will be
        returned (rows in which all elements are ``False``).
    debug : bool
        If ``True``, check the input arguments.

    Returns
    -------
    unbound : numpy.ndarray
        1-dimensional boolean array containing one element for each
        compound. If compound ``i`` is unbound, the corresponding array
        element ``i`` evaluates to ``True``.
    """

    if debug:
        if contact_matrix.ndim != 2:
            warnings.warn("contact_matrix seems not to be a contact"
                          " matrix, since it has not 2 dimensions",
                          RuntimeWarning)
        if axis >= contact_matrix.ndim:
            raise np.AxisError("axis {} is out of bounds for"
                               " contact_matrix of dimension {}"
                               .format(axis, contact_matrix.ndim))

    unbound = contact_matrix.sum(axis=axis)
    unbound = np.squeeze(np.asarray(unbound))
    return unbound == 0




# This function is also used by: lifetime_autocorr_parallel.py
def autocorr_bound(cms, restart=1, debug=False):
    """
    Analogue of :func:`MDAnalysis.lib.correlations.autocorrelation` for
    lists of contact matrices as e.g. generated with
    :func:`mdtools.structure.contact_matrix`.

    .. todo::
       Adapt above paragraph. In contrast to this function,
       :func:`MDAnalysis.lib.correlations.autocorrelation` only takes
       into account contacts that **continuously** survive from
       :math:`t_0` to :math:`t_0 + \tau`

    Calculate the autocorrelation of the existence function
    :math:`S_{ij}(t)`. The existence function is defined as
    :math:`S_{ij}(t) = 1`, if at time :math:`t` a contact exists between
    compound :math:`j` of the selection group and compound :math:`i` of
    the reference group, otherwise :math:`S_{ij}(t) = 0`. The
    autocorrelation is calculated as

    .. math::
        C(\tau) = \langle \frac{S_{ij}(t_0) S_{ij}(t_0+\tau)}{S_{ij}(t_0) S_{ij}(t_0)} \rangle

    where :math:`\langle ... \rangle` denotes averaging over all
    existing contacts :math:`ij` within the given cutoff and over all
    possible starting times :math:`t_0`. You can interprete
    :math:`C(\tau)` as the percentage of contacts that still exist
    or exist again after a lag time :math:`\tau`.

    Parameters
    ----------
    cms : list or tuple
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be given as scipy sparse matrices.
    restart : int, optional
        Number of frames between restarting points :math:`t_0`.
    debug : bool
        If ``True``, check the input arguments.

    Returns
    -------
    autocorr : numpy.ndarray
        Array of shape ``(len(cms),)`` containing the values of
        :math:`C(\tau)` for each lag time :math:`\tau`.
    """

    if debug:
        if restart >= len(cms):
            warnings.warn("The number of frames between restarting"
                          " points ({}) is equal to or larger than the"
                          " total number of frames in cms ({})"
                          .format(restart, len(cms)),
                          RuntimeWarning)
        for i, cm in enumerate(cms):
            if cm.ndim != 2:
                warnings.warn("cms seems not to be a list of contact"
                              " matrices, since its {}-th element has"
                              " not 2 dimensions".format(i),
                              RuntimeWarning)
            if cm.shape != cms[0].shape:
                raise ValueError("All arrays in cms must have the same"
                                 " shape")
            if type(cm) != type(cms[0]):
                raise TypeError("All arrays in cms must be of the same"
                                " type")

    proc = psutil.Process(os.getpid())
    n_frames = len(cms)
    autocorr = np.full(n_frames, np.nan, dtype=np.float32)

    timer_lag = datetime.now()
    for lag in range(n_frames):
        if lag % 10**(len(str(lag))-1) == 0 or lag == n_frames-1:
            print("  Lag     {:12d} of {:12d}"
                  .format(lag, n_frames-1),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now()-timer_lag),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss/2**20),
                  flush=True)
            timer_lag = datetime.now()

        norm = 0
        corr = 0
        for t0 in range(0, n_frames-lag, restart):
            norm += cms[t0].sum()  # = cms[t0].multiply(cms[t0]).sum(), because 1*1=1 and 0*0=0
            corr += cms[t0].multiply(cms[t0+lag]).sum()
        if norm: # != 0:
            autocorr[lag] = corr / norm

    return autocorr




# This function is also used by: lifetime_autocorr_parallel.py
def autocorr_unbound(cms, axis, restart=1, debug=False):
    """
    Analogue of :func:`autocorr_bound` for unbound compounds.

    Calculate the autocorrelation of the existence function
    :math:`S_{ij}(t)`. In this case, the existence function is defined
    as :math:`S_{ij}(t) = 1`, if at time :math:`t` compound :math:`i` of
    a reference group is not in contact with any atom of a selection
    group, otherwise :math:`S_{ij}(t) = 0`. The autocorrelation is
    calculated as

    .. math::
        C(\tau) = \langle \frac{S_{ij}(t_0) S_{ij}(t_0+\tau)}{S_{ij}(t_0) S_{ij}(t_0)} \rangle

    where :math:`\langle ... \rangle` denotes averaging over all
    existing contacts :math:`ij` within the given cutoff and over all
    possible starting times :math:`t_0`. You can interprete
    :math:`C(\tau)` as the percentage of reference compounds that are
    still unbound or again unbound after a lag time :math:`\tau`.

    Parameters
    ----------
    cms : list or tuple
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be either :class:`numpy.ndarray`s or
        scipy sparse matrices. It is assumed that each row stands for a
        reference compound and each column for a selection compound.
    axis : int, optional
        Must be either ``0`` or ``1``. If ``0``, unbound selection
        compounds will be returned (columns in which all elements are
        ``False``), if ``1`` unbound reference compounds will be
        returned (rows in which all elements are ``False``).
    restart : int, optional
        Number of frames between restarting points :math:`t_0`.
    debug : bool
        If ``True``, check the input arguments.

    Returns
    -------
    autocorr : numpy.ndarray
        Array of shape ``(len(cms),)`` containing the values of
        :math:`C(\tau)` for each lag time :math:`\tau`.
    """

    if debug:
        if restart >= len(cms):
            warnings.warn("The number of frames between restarting"
                          " points ({}) is equal to or larger than the"
                          " total number of frames in cms ({})"
                          .format(restart, len(cms)),
                          RuntimeWarning)
        for i, cm in enumerate(cms):
            if cm.ndim != 2:
                warnings.warn("cms seems not to be a list of contact"
                              " matrices, since its {}-th element has"
                              " not 2 dimensions".format(i),
                              RuntimeWarning)
            if axis >= cm.ndim:
                raise np.AxisError("axis {} is out of bounds for array"
                                   " of dimension {}"
                                   .format(axis, cm.ndim))
            if cm.shape != cms[0].shape:
                raise ValueError("All arrays in cms must have the same"
                                 " shape")
            if type(cm) != type(cms[0]):
                raise TypeError("All arrays in cms must be of the same"
                                " type")

    proc = psutil.Process(os.getpid())
    n_frames = len(cms)
    autocorr = np.full(n_frames, np.nan, dtype=np.float32)

    timer_lag = datetime.now()
    for lag in range(n_frames):
        if lag % 10**(len(str(lag))-1) == 0 or lag == n_frames-1:
            print("  Lag     {:12d} of {:12d}"
                  .format(lag, n_frames-1),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now()-timer_lag),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss/2**20),
                  flush=True)
            timer_lag = datetime.now()

        norm = 0
        corr = 0
        for t0 in range(0, n_frames-lag, restart):
            unbound_t0 = unbound_compounds(contact_matrix=cms[t0],
                                           axis=axis,
                                           debug=debug)
            norm += np.sum(unbound_t0)  # = np.sum(unbound_t0*unbound_t0), because 1*1=1 and 0*0=0
            unbound_lag = unbound_compounds(contact_matrix=cms[t0+lag],
                                            axis=axis,
                                            debug=debug)
            corr += np.sum(unbound_t0 * unbound_lag)
        if norm: # != 0:
            autocorr[lag] = corr / norm

    return autocorr




# This function is also used by: lifetime_autocorr_parallel.py
def kww(t, tau, beta):
    """
    Stretched exponential function, also known as Kohlrausch-Williams-
    Watts (KWW) function.

    Parameters
    ----------
    t : array_like or scalar
        Independet variable.
    tau : scalar
        Relaxation time.
    beta : scalar
        Stretching exponent.

    Returns
    -------
    .. math::
        f(t) = \exp \left[ -\left( \frac{t}{\tau} \right)^\beta \right]
    """

    return np.exp(-(t/tau)**beta)




# This function is also used by: lifetime_autocorr_parallel.py
def fit_kww(xdata, ydata, ysd=None):
    """
    Fit a stretched exponential function, also known as Kohlrausch-
    Williams-Watts (KWW) function, to `ydata` using
    :func:`scipy.optimize.curve_fit`. Fit function:

    .. math::
        f(t) = \exp \left[ -\left( \frac{t}{\tau} \right)^\beta \right]

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
    tau : scalar
        Relaxation time.
    tau_sd : scalar
        Standard deviation of `tau` from fitting.
    beta : scalar
        Stretching exponent.
    beta_sd : scalar
        Standard deviation of `beta` from fitting.
    """

    try:
        popt, pcov = optimize.curve_fit(f=kww,
                                        xdata=xdata,
                                        ydata=ydata,
                                        sigma=ysd,
                                        absolute_sigma=True,
                                        check_finite=False,
                                        p0=[len(ydata)*np.mean(ydata), 1],
                                        bounds=([0, 0], [np.inf, 1]))
    except (ValueError, RuntimeError, optimize.OptimizeWarning) as err:
        print(flush=True)
        print("An error has occurred during fitting:", flush=True)
        print("{}".format(err), flush=True)
        print("Setting fit parameters to numpy.nan", flush=True)
        print(flush=True)
        popt = np.array([np.nan, np.nan])
        perr = np.array([np.nan, np.nan])
    else:
        perr = np.sqrt(np.diag(pcov))

    tau = popt[0]
    tau_sd = perr[0]
    beta = popt[1]
    beta_sd = perr[1]

    return tau, tau_sd, beta, beta_sd








if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())


    args = parse_user_input()




    print("\n\n\n", flush=True)
    u = mdt.select.universe(top=args.TOPFILE,
                            trj=args.TRJFILE,
                            verbose=True)


    print("\n\n\n", flush=True)
    print("Creating selections", flush=True)
    timer = datetime.now()

    ref = u.select_atoms(' '.join(args.REF))
    sel = u.select_atoms(' '.join(args.SEL))
    print("  Reference group: '{}'"
          .format(' '.join(args.REF)),
          flush=True)
    print(mdt.rti.ag_info_str(ag=ref, indent=4))
    print(flush=True)
    print("  Selection group: '{}'"
          .format(' '.join(args.SEL)),
          flush=True)
    print(mdt.rti.ag_info_str(ag=sel, indent=4))

    if ref.n_atoms <= 0:
        raise ValueError("The reference group contains no atoms")
    if sel.n_atoms <= 0:
        raise ValueError("The selection group contains no atoms")

    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)




    BEGIN, END, EVERY, n_frames = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=u.trajectory.n_frames)
    last_frame = u.trajectory[END-1].frame
    NBLOCKS, blocksize = mdt.check.block_averaging(n_blocks=args.NBLOCKS,
                                                   n_frames=n_frames)
    RESTART, effective_restart = mdt.check.restarts(
        restart_every_nth_frame=args.RESTART,
        read_every_nth_frame=EVERY,
        n_frames=blocksize)
    if args.DEBUG:
        print("\n\n\n", flush=True)
        mdt.check.time_step(trj=u.trajectory[BEGIN:END], verbose=True)
    timestep = u.trajectory[BEGIN].dt




    print("\n\n\n", flush=True)
    print("Reading trajectory", flush=True)
    print("  Total number of frames in trajectory: {:>9d}"
          .format(u.trajectory.n_frames),
          flush=True)
    print("  Time step per frame:                  {:>9} (ps)\n"
          .format(u.trajectory[0].dt),
          flush=True)
    timer = datetime.now()

    cms = mdt.strc.contact_matrices(topfile=args.TOPFILE,
                                    trjfile=args.TRJFILE,
                                    ref=args.REF,
                                    sel=args.SEL,
                                    cutoff=args.CUTOFF,
                                    compound=args.COMPOUND,
                                    min_contacts=args.MINCONTACTS,
                                    begin=BEGIN,
                                    end=END,
                                    every=EVERY,
                                    verbose=True,
                                    debug=args.DEBUG)
    if len(cms) != n_frames:
        raise ValueError("The number of contact matrices does not equal"
                         " the number of frames to read. This should not"
                         " have happened")

    print(flush=True)
    print("Frames read: {}".format(n_frames), flush=True)
    print("First frame: {:>12d}    Last frame: {:>12d}    "
          "Every Nth frame: {:>12d}"
          .format(u.trajectory[BEGIN].frame, last_frame, EVERY),
          flush=True)
    print("Start time:  {:>12}    End time:   {:>12}    "
          "Every Nth time:  {:>12} (ps)"
          .format(u.trajectory[BEGIN].time,
                  u.trajectory[END-1].time,
                  u.trajectory[0].dt * EVERY),
          flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)




    if args.INTERMITTENCY > 0:
        print("\n\n\n", flush=True)
        print("Correcting for intermittency", flush=True)
        timer = datetime.now()

        cms = mdt.dyn.correct_intermittency(
            list_of_arrays=cms,
            intermittency=args.INTERMITTENCY,
            verbose=True,
            debug=args.DEBUG)

        print("Elapsed time:         {}"
              .format(datetime.now()-timer),
              flush=True)
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)




    print("\n\n\n", flush=True)
    print("Calculating autocorrelation function", flush=True)
    timer = datetime.now()
    timer_block = datetime.now()

    print("  Bound reference-selection complexes", flush=True)
    acorr_bound = [None,] * NBLOCKS
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
        acorr_bound[block] = autocorr_bound(
            cms=cms[block*blocksize:(block+1)*blocksize],
            restart=effective_restart,
            debug=args.DEBUG)

    print("\n", flush=True)
    print("  Unbound reference compounds", flush=True)
    acorr_unbound_ref = [None,] * NBLOCKS
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
        acorr_unbound_ref[block] = autocorr_unbound(
            cms=cms[block*blocksize:(block+1)*blocksize],
            axis=1,
            restart=effective_restart,
            debug=args.DEBUG)

    print("\n", flush=True)
    print("  Unbound selection compounds", flush=True)
    acorr_unbound_sel = [None,] * NBLOCKS
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
        acorr_unbound_sel[block] = autocorr_unbound(
            cms=cms[block*blocksize:(block+1)*blocksize],
            axis=0,
            restart=effective_restart,
            debug=args.DEBUG)

    del cms
    acorr_bound = np.vstack(acorr_bound)
    acorr_unbound_ref = np.vstack(acorr_unbound_ref)
    acorr_unbound_sel = np.vstack(acorr_unbound_sel)

    if NBLOCKS > 1:
        acorr_bound, acorr_bound_sd = mdt.stats.block_average(acorr_bound)
        acorr_bound_sd_fit = np.copy(acorr_bound_sd)
        acorr_bound_sd_fit[acorr_bound_sd_fit==0] = 1e-20
        acorr_unbound_ref, acorr_unbound_ref_sd = mdt.stats.block_average(acorr_unbound_ref)
        acorr_unbound_ref_sd_fit = np.copy(acorr_unbound_ref_sd)
        acorr_unbound_ref_sd_fit[acorr_unbound_ref_sd_fit==0] = 1e-20
        acorr_unbound_sel, acorr_unbound_sel_sd = mdt.stats.block_average(acorr_unbound_sel)
        acorr_unbound_sel_sd_fit = np.copy(acorr_unbound_sel_sd)
        acorr_unbound_sel_sd_fit[acorr_unbound_sel_sd_fit==0] = 1e-20
    else:
        acorr_bound = np.squeeze(acorr_bound)
        acorr_bound_sd_fit = None
        acorr_unbound_ref = np.squeeze(acorr_unbound_ref)
        acorr_unbound_ref_sd_fit = None
        acorr_unbound_sel = np.squeeze(acorr_unbound_sel)
        acorr_unbound_sel_sd_fit = None

    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Fitting autocorrelation function", flush=True)
    timer = datetime.now()

    lag_times = np.arange(0,
                          timestep*blocksize*EVERY,
                          timestep*EVERY,
                          dtype=np.float32)
    if args.ENDFIT is None:
        endfit = int(0.9 * len(lag_times))
        args.ENDFIT = lag_times[endfit]
    else:
        _, endfit = mdt.nph.find_nearest(lag_times,
                                         args.ENDFIT,
                                         return_index=True,
                                         debug=args.DEBUG)


    stopfit = np.nonzero(acorr_bound < args.STOPFIT)[0]
    if stopfit.size == 0:
        stopfit = len(acorr_bound)
    else:
        stopfit = stopfit[0]
    valid = np.isfinite(acorr_bound)
    valid[min(endfit, stopfit):] = False
    if acorr_bound_sd_fit is None:
        ysd = None
    else:
        ysd = acorr_bound_sd_fit[valid]
    if np.any(valid):
        fit_bound = fit_kww(xdata=lag_times[valid],
                            ydata=acorr_bound[valid],
                            ysd=ysd)
    else:
        fit_bound = np.full(4, np.nan)
    lifetime_bound = (fit_bound[0]/fit_bound[2] *
                      special.gamma(1/fit_bound[2]))
    kww_bound = kww(t=lag_times, tau=fit_bound[0], beta=fit_bound[2])
    kww_bound[~valid] = np.nan


    stopfit = np.nonzero(acorr_unbound_ref < args.STOPFIT)[0]
    if stopfit.size == 0:
        stopfit = len(acorr_unbound_ref)
    else:
        stopfit = stopfit[0]
    valid = np.isfinite(acorr_unbound_ref)
    valid[min(endfit, stopfit):] = False
    if acorr_unbound_ref_sd_fit is None:
        ysd = None
    else:
        ysd = acorr_unbound_ref_sd_fit[valid]
    if np.any(valid):
        fit_unbound_ref = fit_kww(xdata=lag_times[valid],
                                  ydata=acorr_unbound_ref[valid],
                                  ysd=ysd)
    else:
        fit_unbound_ref = np.full(4, np.nan)
    lifetime_unbound_ref = (fit_unbound_ref[0]/fit_unbound_ref[2] *
                            special.gamma(1/fit_unbound_ref[2]))
    kww_unbound_ref = kww(t=lag_times,
                          tau=fit_unbound_ref[0],
                          beta=fit_unbound_ref[2])
    kww_unbound_ref[~valid] = np.nan


    stopfit = np.nonzero(acorr_unbound_sel < args.STOPFIT)[0]
    if stopfit.size == 0:
        stopfit = len(acorr_unbound_sel)
    else:
        stopfit = stopfit[0]
    valid = np.isfinite(acorr_unbound_sel)
    valid[min(endfit, stopfit):] = False
    if acorr_unbound_sel_sd_fit is None:
        ysd = None
    else:
        ysd = acorr_unbound_sel_sd_fit[valid]
    if np.any(valid):
        fit_unbound_sel = fit_kww(xdata=lag_times[valid],
                                  ydata=acorr_unbound_sel[valid],
                                  ysd=ysd)
    else:
        fit_unbound_sel = np.full(4, np.nan)
    lifetime_unbound_sel = (fit_unbound_sel[0]/fit_unbound_sel[2] *
                            special.gamma(1/fit_unbound_sel[2]))
    kww_unbound_sel = kww(t=lag_times,
                          tau=fit_unbound_sel[0],
                          beta=fit_unbound_sel[2])
    kww_unbound_sel[~valid] = np.nan


    del acorr_bound_sd_fit
    del acorr_unbound_ref_sd_fit
    del acorr_unbound_sel_sd_fit
    del valid
    del ysd

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
        "Average lifetime of reference-selection complexes and unbound\n"
        "reference and selection compounds\n"
        "\n"
        "\n"
        "Cutoff (Angstrom)     = {}\n"
        "Compound              = {}\n"
        "Minimum contacts      = {}\n"
        "Allowed intermittency = {}\n"
        "End fit at            = {} ps\n"
        "Only fit C(t)        >= {}\n"
        "\n"
        "\n"
        "Reference: '{}'\n"
        "  Segments:               {}\n"
        "    Different segments:   {}\n"
        "    Segment name(s):      '{}'\n"
        "  Residues:               {}\n"
        "    Different residues:   {}\n"
        "    Residue name(s):      '{}'\n"
        "  Atoms:                  {}\n"
        "    Different atom names: {}\n"
        "    Atom name(s):         '{}'\n"
        "    Different atom types: {}\n"
        "    Atom type(s):         '{}'\n"
        "  Fragments:              {}\n"
        "\n"
        "Selection: '{}'\n"
        "  Segments:               {}\n"
        "    Different segments:   {}\n"
        "    Segment name(s):      '{}'\n"
        "  Residues:               {}\n"
        "    Different residues:   {}\n"
        "    Residue name(s):      '{}'\n"
        "  Atoms:                  {}\n"
        "    Different atom names: {}\n"
        "    Atom name(s):         '{}'\n"
        "    Different atom types: {}\n"
        "    Atom type(s):         '{}'\n"
        "  Fragments:              {}\n"
        "\n"
        "\n"
        "The average lifetime tau is estimated from the integral of a\n"
        "stretched exponential fit of the autocorrelation of the\n"
        "existence function\n"
        "\n"
        "Existence function:\n"
        "  S_ij(t) = 1, if at time t a contact exists between compound\n"
        "               j of the selection group and compound i of the\n"
        "               reference group\n"
        "  S_ij(t) = 0, otherwise\n"
        "\n"
        "For calculating the lifetime of unbound compounds, the\n"
        "existence function is redefined as\n"
        "  S_i(t) = 1, if at time t the reference (selection) compound\n"
        "              i is not in contact with any atom of the\n"
        "              selection (reference) group\n"
        "  S_i(t) = 0, otherwise\n"
        "\n"
        "Autocorrelation of the existence function:\n"
        "  C(t) = < S_ij(t0)*S_ij(t0+t) / S_ij(t0)*S_ij(t0) >\n"
        "  <...> = Average over all existing contacts ij and over all\n"
        "          possible starting times t0\n"
        "  You can interprete C(t) as the percentage of contacts that\n"
        "  still exist or exist again after a lag time t\n"
        "\n"
        "The autocorrelation is fitted using a stretched exponential\n"
        "function, also known as Kohlrausch-Williams-Watts (KWW)\n"
        "function:\n"
        "  f(t) = exp[-(t/tau')^beta]\n"
        "  beta is constrained to the intervall [0, 1]\n"
        "  tau' must be positive\n"
        "\n"
        "The average lifetime tau is calculated as the integral of the\n"
        "KWW function from zero to infinity:\n"
        "  tau = integral_0^infty exp[-(t/tau')^beta] dt\n"
        "      = tau'/beta * Gamma(1/beta)\n"
        "  Gamma(x) = Gamma function\n"
        "  If beta=1, tau=tau'\n"
        "\n"
        "\n"
        .format(args.CUTOFF,
                args.COMPOUND,
                args.MINCONTACTS,
                args.INTERMITTENCY,
                args.ENDFIT,
                args.STOPFIT,

                ' '.join(args.REF),
                ref.n_segments,
                len(np.unique(ref.segids)),
                '\' \''.join(i for i in np.unique(ref.segids)),
                ref.n_residues,
                len(np.unique(ref.resnames)),
                '\' \''.join(i for i in np.unique(ref.resnames)),
                ref.n_atoms,
                len(np.unique(ref.names)),
                '\' \''.join(i for i in np.unique(ref.names)),
                len(np.unique(ref.types)),
                '\' \''.join(i for i in np.unique(ref.types)),
                len(ref.fragments),

                ' '.join(args.SEL),
                sel.n_segments,
                len(np.unique(sel.segids)),
                '\' \''.join(i for i in np.unique(sel.segids)),
                sel.n_residues,
                len(np.unique(sel.resnames)),
                '\' \''.join(i for i in np.unique(sel.resnames)),
                sel.n_atoms,
                len(np.unique(sel.names)),
                '\' \''.join(i for i in np.unique(sel.names)),
                len(np.unique(sel.types)),
                '\' \''.join(i for i in np.unique(sel.types)),
                len(sel.fragments),
                )
    )

    if NBLOCKS == 1:
        columns = (
            "The columns contain:\n"
            "  1 Lag time t (ps)\n"
            "  2 Autocorrelation for bound reference-selection complexes\n"
            "  3 Fit of column 2\n"
            "  4 Autocorrelation for unbound reference compounds\n"
            "  5 Fit of column 4\n"
            "  6 Autocorrelation for unbound selection compounds\n"
            "  7 Fit of column 6\n"
            "\n"
            "Column number:\n"
            "{:>14d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d}\n"
            "\n"
            "Fit:\n"
            "Lifetime tau (ps):               {:>15.9e} {:>33.9e} {:>33.9e}\n"
            "Relaxation time tau' (ps):       {:>15.9e} {:>33.9e} {:>33.9e}\n"
            "Standard deviation of tau' (ps): {:>15.9e} {:>33.9e} {:>33.9e}\n"
            "Stretching exponent beta:        {:>15.9e} {:>33.9e} {:>33.9e}\n"
            "Standard deviation of beta:      {:>15.9e} {:>33.9e} {:>33.9e}\n"
            .format(1, 2, 3, 4, 5, 6, 7,

                    lifetime_bound,
                    lifetime_unbound_ref,
                    lifetime_unbound_sel,
                    fit_bound[0], fit_unbound_ref[0], fit_unbound_sel[0],
                    fit_bound[1], fit_unbound_ref[1], fit_unbound_sel[1],
                    fit_bound[2], fit_unbound_ref[2], fit_unbound_sel[2],
                    fit_bound[3], fit_unbound_ref[3], fit_unbound_sel[3]
                    )
        )
        data = np.column_stack([lag_times,

                                acorr_bound,
                                kww_bound,

                                acorr_unbound_ref,
                                kww_unbound_ref,

                                acorr_unbound_sel,
                                kww_unbound_sel])

    else:
        columns = (
            "The columns contain:\n"
            "   1 Lag time t (ps)\n"
            "   2 Autocorrelation for bound reference-selection complexes\n"
            "   3 Standard deviation of column 2\n"
            "   4 Fit of column 2\n"
            "   5 Autocorrelation for unbound reference compounds\n"
            "   6 Standard deviation of column 5\n"
            "   7 Fit of column 5\n"
            "   8 Autocorrelation for unbound selection compounds\n"
            "   9 Standard deviation of column 8\n"
            "  10 Fit of column 8\n"
            "\n"
            "Column number:\n"
            "{:>14d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d}\n"
            "\n"
            "Fit:\n"
            "Lifetime tau (ps):               {:>32.9e} {:>50.9e} {:>50.9e}\n"
            "Relaxation time tau' (ps):       {:>32.9e} {:>50.9e} {:>50.9e}\n"
            "Standard deviation of tau' (ps): {:>32.9e} {:>50.9e} {:>50.9e}\n"
            "Stretching exponent beta:        {:>32.9e} {:>50.9e} {:>50.9e}\n"
            "Standard deviation of beta:      {:>32.9e} {:>50.9e} {:>50.9e}\n"
            .format(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,

                    lifetime_bound,
                    lifetime_unbound_ref,
                    lifetime_unbound_sel,
                    fit_bound[0], fit_unbound_ref[0], fit_unbound_sel[0],
                    fit_bound[1], fit_unbound_ref[1], fit_unbound_sel[1],
                    fit_bound[2], fit_unbound_ref[2], fit_unbound_sel[2],
                    fit_bound[3], fit_unbound_ref[3], fit_unbound_sel[3]
                    )
        )
        data = np.column_stack([lag_times,

                                acorr_bound,
                                acorr_bound_sd,
                                kww_bound,

                                acorr_unbound_ref,
                                acorr_unbound_ref_sd,
                                kww_unbound_ref,

                                acorr_unbound_sel,
                                acorr_unbound_sel_sd,
                                kww_unbound_sel])

    mdt.fh.savetxt(fname=args.OUTFILE,
                   data=data,
                   header=header+columns)

    print("  Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("{} done".format(os.path.basename(sys.argv[0])), flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
