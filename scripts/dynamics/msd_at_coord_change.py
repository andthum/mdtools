#!/usr/bin/env python3

# This file is part of MDTools.
# Copyright (C) 2021, 2022  The MDTools Development Team and all
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


import sys
import os
import warnings
from datetime import datetime
import psutil
import argparse
import numpy as np
import mdtools as mdt


# This function is also used by: extract_renewal_events.py
def get_pos(universe, atm_grp, frame, compound='atoms', debug=False):
    """
    Get the positions of the compounds of a MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup` at a given frame in the
    :attr:`~MDAnalysis.core.universe.Universe.trajectory`.

    Note
    ----
    Broken molecules are not made whole before center of mass
    calculation!

    Parameters
    ----------
    universe : MDAnalysis.core.universe.Universe
        The MDAnalysis :class:`~MDAnalysis.core.universe.Universe` which
        holds the :class:`~MDAnalysis.core.groups.AtomGroup`.
    ref : MDAnalysis.core.groups.AtomGroup
        The :class:`~MDAnalysis.core.groups.AtomGroup` from which to get
        the compound positions
    frame : int
        The frame for which to get the positions.
    compound : str, optional
        The compound for which to get the positions. Must be either
        ``'atoms'``, ``'group'``, ``'segments'``, ``'residues'``,
        ``'molecules'`` or ``'fragments'``. If not ``'atoms'``, the
        center of mass positions of the respective compounds are
        returned.
    debug : bool, optional
        If ``True``, check the input arguments.

    Returns
    -------
    pos : numpy.ndarray
        Position vector(s) of the compounds of `atm_grp` at the given
        `frame`.
    """

    if debug:
        if (compound != 'atoms' and
            compound != 'group' and
            compound != 'segments' and
            compound != 'residues' and
            compound != 'molecules' and
                compound != 'fragments'):
            raise ValueError("compound must be either 'atoms', 'group',"
                             " 'segments', 'residues', 'molecules' or"
                             " 'fragments', but you gave '{}'"
                             .format(compound))

    universe.trajectory[frame]
    if compound == 'atoms':
        return atm_grp.positions
    else:
        return mdt.strc.com(ag=atm_grp,
                            pbc=False,
                            compound=compound,
                            make_whole=False,
                            debug=debug)


def msd_at_coordination_change(universe, ref, cms, compound='atoms',
                               coord_change='any', n_det=1, n_att=1, begin=0, every=1,
                               verbose=False, debug=False):
    """
    Calculate lifetime histograms of reference compounds until their
    coordination to selection compounds changes. Additionally, the mean
    square displacements (MSDs) of the reference compounds are computed
    as a function of these lifetimese. MSDs are calculated from the
    center of  mass of the reference compounds.

    Parameters
    ----------
    universe : MDAnalysis.core.universe.Universe
        The MDAnalysis :class:`~MDAnalysis.core.universe.Universe` which
        holds the unwrapped and whole reference compounds.
    ref : MDAnalysis.core.groups.AtomGroup
        The reference :class:`~MDAnalysis.core.groups.AtomGroup`.
    cms : array_like
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be given as scipy sparse matrices.
        The rows must stand for the reference compounds, the columns for
        the selection compounds.
    compound : str, optional
        For which type of components the contact matrices in `cms` were
        calculated. Must be either ``'atoms'``, ``'segments'``,
        ``'residues'``, or ``'fragments'``.
    coord_change : str, optional
        You track one of the following four changes in the reference-
        selection coordination: Any change; detachment of at least
        `n_det` selection compounds at once; attachment of at least
        `n_att` selection compounds at once; detachment of at least
        `n_det` selection compounds and attachment of at least `n_att`
        selection compounds at once. 'At once' means here from one
        frame to the next frame. The choose one of these four change
        types parse either ``'any'``, ``'detachment'``, ``'attachment'``
        of ``'exchange'``.
    n_det : int, optional
        How many selection compounds must be detached at once, to count
        the change as detachment event.
    n_att : int, optional
        How many selection compounds must be attached at once, to count
        the change as attachment event.
    begin : int, optional
        The frame number of the
        :attr:`~MDAnalysis.core.universe.Universe.trajectory` to which
        the first contact matrix in `cms` correspond to.
    every : int, optional
        To how many frames of the
        :attr:`~MDAnalysis.core.universe.Universe.trajectory` one step
        in `cms` corresponds to.
    verbose : bool, optional
        If ``True``, print progress information to standard output.
    debug : bool, optional
        If ``True``, check the input arguments.

    Returns
    -------
    dt_counts : numpy.ndarray
        Array of shape ``(len(cms), )`` containing the counts for how
        many reference compounds lived for the corresponding number of
        frames until their coordination to selection compounds was
        changed in the way determinded by `coord_change`.
    msd : numpy.ndarray
        Array of shape ``(len(cms), )`` containing containing the
        corresponding MSDs of these reference compounds during their
        lifetime.
    """

    if debug:
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
        if (compound != 'atoms' and
            compound != 'segments' and
            compound != 'residues' and
                compound != 'fragments'):
            raise ValueError("compound must be either 'atoms',"
                             " 'segments', 'residues' or 'fragments',"
                             " but you gave '{}'"
                             .format(compound))
        if ((compound == 'atoms' and ref.n_atoms != cms[0].shape[0]) or
            (compound == 'segments' and ref.n_segments != cms[0].shape[0]) or
            (compound == 'residues' and ref.n_residues != cms[0].shape[0]) or
                (compound == 'fragments' and ref.n_fragments != cms[0].shape[0])):
            raise ValueError("The number of reference compounds does not"
                             " fit to the number of columns of the"
                             " contact matrices")
        if (coord_change != 'any' and
            coord_change != 'detachment' and
            coord_change != 'attachment' and
                coord_change != 'exchange'):
            raise ValueError("coord_change must be either 'any',"
                             " 'detachment', 'attachment' or 'exchange',"
                             " but you gave '{}'"
                             .format(coord_change))
        if n_det <= 0:
            warnings.warn("n_det <= 0. Counting every event as"
                          " detachment event", RuntimeWarning)
        elif n_det > cms[0].shape[1]:
            warnings.warn("n_det is larger than the total number of"
                          " selection compounds. No detachment events"
                          " will be found", RuntimeWarning)
        if n_att <= 0:
            warnings.warn("n_att <= 0. Counting every event as"
                          " attachment event", RuntimeWarning)
        elif n_att > cms[0].shape[1]:
            warnings.warn("n_att is larger than the total number of"
                          " selection compounds. No attachment events"
                          " will be found", RuntimeWarning)
        if begin < 0:
            raise ValueError("begin must not be negative")
        if every <= 0:
            raise ValueError("every must be positive")
        mdt.check.time_step(trj=universe.trajectory[begin:len(cms) * every])

    if verbose:
        timer = datetime.now()
        proc = psutil.Process()
        print("  Frame   {:12d} of {:12d}"
              .format(0, len(cms) - 1),
              flush=True)
        print("    Elapsed time:             {}"
              .format(datetime.now() - timer),
              flush=True)
        print("    Current memory usage: {:18.2f} MiB"
              .format(proc.memory_info().rss / 2**20),
              flush=True)
        timer = datetime.now()

    msd = np.zeros(len(cms), dtype=np.float32)
    dt = np.ones(cms[0].shape[0], dtype=np.uint32)
    dt_counts = np.zeros(len(cms), dtype=np.uint32)
    refpos = get_pos(universe=universe,
                     atm_grp=ref,
                     frame=begin,
                     compound=compound,
                     debug=debug)

    for i, cm in enumerate(cms[1:], 1):
        if (verbose and
                (i % 10**(len(str(i)) - 1) == 0 or i == len(cms) - 1)):
            print("  Frame   {:12d} of {:12d}"
                  .format(i, len(cms) - 1),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now() - timer),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss / 2**20),
                  flush=True)
            timer = datetime.now()

        if coord_change == 'any':
            # Consider any change in the reference-selection coordination
            changes = (cm != cms[i - 1])
            if changes.count_nonzero() == 0:
                dt += 1
                continue
            else:
                affected_ref = np.flatnonzero(changes.sum(axis=1))
        elif coord_change == 'detachment':
            # Consider only detachments of at least n_det selection compounds at once
            bound_in_both = cm.multiply(cms[i - 1])
            detached = cms[i - 1] - bound_in_both
            affected_ref = np.flatnonzero(detached.sum(axis=1) >= n_det)
            if affected_ref.size == 0:
                dt += 1
                continue
        elif coord_change == 'attachment':
            # Consider only attachments of at least n_att selection compounds at once
            bound_in_both = cm.multiply(cms[i - 1])
            attached = cm - bound_in_both
            affected_ref = np.flatnonzero(attached.sum(axis=1) >= n_att)
            if affected_ref.size == 0:
                dt += 1
                continue
        elif coord_change == 'exchange':
            # Consider only detachments of at least n_det selection compounds
            # and attachments of at least n_att selection compounds at once
            bound_in_both = cm.multiply(cms[i - 1])
            detached = cms[i - 1] - bound_in_both
            attached = cm - bound_in_both
            affected_ref = np.intersect1d(
                np.flatnonzero(detached.sum(axis=1) >= n_det),
                np.flatnonzero(attached.sum(axis=1) >= n_att))
            if affected_ref.size == 0:
                dt += 1
                continue

        newpos = get_pos(universe=universe,
                         atm_grp=ref,
                         frame=begin + i * every,
                         compound=compound,
                         debug=debug)
        newpos = newpos[affected_ref]
        msd_affected_ref = np.sum((newpos - refpos[affected_ref])**2,
                                  axis=1)
        refpos[affected_ref] = newpos
        dt_unique, dt_unique_counts = np.unique(dt[affected_ref],
                                                return_counts=True)
        dt_counts[dt_unique] += dt_unique_counts.astype(np.uint32)
        if not np.all(dt_unique_counts == 1):
            msd_affected_ref = [np.sum(msd_affected_ref,
                                       where=(dt[affected_ref] == t))
                                for t in dt_unique]
            msd_affected_ref = np.asarray(msd_affected_ref)
        msd[dt_unique] += msd_affected_ref
        dt[affected_ref] = 0
        dt += 1

    if dt_counts[0] != 0:
        raise ValueError("The number of counts with zero diffusion time"
                         " is not zero. This should not have happened")
    if msd[0] != 0:
        raise ValueError("The MSD at zero diffusion time is not zero."
                         " This should not have happened")

    valid = (dt_counts != 0)
    np.divide(msd, dt_counts, where=valid, out=msd)
    msd[~valid] = np.nan
    msd[0] = 0

    return dt_counts, msd


if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process()

    parser = argparse.ArgumentParser(
        description=(
            "Calculate the lifetime of reference compounds"
            " until their coordination to selection compounds"
            " changes from lifetime histograms. Additionally,"
            " the mean square displacements (MSDs) of the"
            " reference compounds during these lifetimes is"
            " computed. MSDs are calculated from the center of"
            " mass of the reference compounds. The following"
            " four changes in the reference-selection"
            " coordination are tracked: Any change; detachment"
            " of at least --ndet selection compounds at once;"
            " attachment of at least --natt selection compounds"
            " at once; detachment of at least --ndet selection"
            " compounds and attachment of at least --natt"
            " selection compounds at once. 'At once' means here"
            " from one trajectory frame to the next frame."
        )
    )

    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="Trajectory file [<.trr/.xtc/.gro/.pdb/.xyz/.mol2/...>]."
             " See supported coordinate formats of MDAnalysis. IMPORTANT:"
             " At least the reference compounds must be whole and"
             " unwrapped in order to get the correct MSDs. You can use"
             " 'unwrap_trj' to unwrap a wrapped trajectory and make"
             " broken molecules whole."
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
             " This option also affects how displacements are calculated."
             " If not set to 'atoms', the center of mass of the compound"
             " is used to calculate its displacement. Default: 'atoms'"
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
        '--ndet',
        dest='NDET',
        type=int,
        required=False,
        default=1,
        help="How many selection compounds must be detached at once, to"
             " count the change as detachment event. Default: 1"
    )
    parser.add_argument(
        '--natt',
        dest='NATT',
        type=int,
        required=False,
        default=1,
        help="How many selection compounds must be attached at once, to"
             " count the change as attachment event. Default: 1"
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
        raise ValueError("The reference atom group contains no atoms")
    if sel.n_atoms <= 0:
        raise ValueError("The selection atom group contains no atoms")

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    BEGIN, END, EVERY, n_frames = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=u.trajectory.n_frames)
    last_frame = u.trajectory[END - 1].frame
    if args.DEBUG:
        print("\n\n\n", flush=True)
        mdt.check.time_step(trj=u.trajectory[BEGIN:END], verbose=True)
    timestep = u.trajectory[BEGIN].dt

    print("\n\n\n", flush=True)
    print("Calculating contact matrices", flush=True)
    print("  Total number of frames in trajectory: {:>9d}"
          .format(u.trajectory.n_frames),
          flush=True)
    print("  Time step per frame:                  {:>9} (ps)\n"
          .format(u.trajectory[0].dt),
          flush=True)
    timer = datetime.now()
    if mdt.rti.get_num_CPUs() > 1:
        mdabackend = "OpenMP"
    else:
        mdabackend = "serial"
    print()
    cms = mdt.strc.contact_matrices(
        ref=" ".join(args.REF),
        sel=" ".join(args.SEL),
        cutoff=args.CUTOFF,
        topfile=args.TOPFILE,
        trjfile=args.TRJFILE,
        begin=BEGIN,
        end=END,
        every=EVERY,
        compound=args.COMPOUND,
        min_contacts=args.MINCONTACTS,
        mdabackend=mdabackend,
        verbose=True,
    )
    print()
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
                  u.trajectory[END - 1].time,
                  u.trajectory[0].dt * EVERY),
          flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
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
              .format(datetime.now() - timer),
              flush=True)
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss / 2**20),
              flush=True)

    print("\n\n\n", flush=True)
    print("Calculating lifetime histograms and MSDs", flush=True)
    timer = datetime.now()

    print(flush=True)
    print("  Lifetimes and MSDs until any change in the reference-\n"
          "  selection coordination", flush=True)
    timer_msd = datetime.now()
    hist_any, msd_any = msd_at_coordination_change(
        universe=u,
        ref=ref,
        cms=cms,
        compound=args.COMPOUND,
        coord_change='any',
        begin=BEGIN,
        every=EVERY,
        verbose=True,
        debug=args.DEBUG)
    print("  Elapsed time:         {}"
          .format(datetime.now() - timer_msd),
          flush=True)
    print("  Current memory usage: {:14.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print(flush=True)
    print("  Lifetimes and MSDs until detachment of at least {}\n"
          "  selection compound(s) at once"
          .format(args.NDET),
          flush=True)
    timer_msd = datetime.now()
    hist_det, msd_det = msd_at_coordination_change(
        universe=u,
        ref=ref,
        cms=cms,
        compound=args.COMPOUND,
        coord_change='detachment',
        n_det=args.NDET,
        begin=BEGIN,
        every=EVERY,
        verbose=True,
        debug=args.DEBUG)
    print("  Elapsed time:         {}"
          .format(datetime.now() - timer_msd),
          flush=True)
    print("  Current memory usage: {:14.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print(flush=True)
    print("  Lifetimes and MSDs until attachment of at least {}\n"
          "  selection compound(s) at once"
          .format(args.NATT),
          flush=True)
    timer_msd = datetime.now()
    hist_att, msd_att = msd_at_coordination_change(
        universe=u,
        ref=ref,
        cms=cms,
        compound=args.COMPOUND,
        coord_change='attachment',
        n_att=1,
        begin=BEGIN,
        every=EVERY,
        verbose=True,
        debug=args.DEBUG)
    print("  Elapsed time:         {}"
          .format(datetime.now() - timer_msd),
          flush=True)
    print("  Current memory usage: {:14.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print(flush=True)
    print("  Lifetimes and MSDs until detachment of at least {}\n"
          "  and attachment of at least {} selection compound(s) at\n"
          "  once"
          .format(args.NDET, args.NATT),
          flush=True)
    timer_msd = datetime.now()
    hist_exc, msd_exc = msd_at_coordination_change(
        universe=u,
        ref=ref,
        cms=cms,
        compound=args.COMPOUND,
        coord_change='exchange',
        n_det=1,
        n_att=1,
        begin=BEGIN,
        every=EVERY,
        verbose=True,
        debug=args.DEBUG)
    print("  Elapsed time:         {}"
          .format(datetime.now() - timer_msd),
          flush=True)
    print("  Current memory usage: {:14.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    del cms

    lag_times = np.arange(0,
                          timestep * n_frames * EVERY,
                          timestep * EVERY,
                          dtype=np.float32)

    tot_counts_any = np.sum(hist_any)
    tot_counts_det = np.sum(hist_det)
    tot_counts_att = np.sum(hist_att)
    tot_counts_exc = np.sum(hist_exc)
    if tot_counts_any < 0:
        raise ValueError("The total number of events is less than zero."
                         " This should not have happend")
    if tot_counts_det < 0:
        raise ValueError("The total number of detachment events is less"
                         " than zero. This should not have happend")
    if tot_counts_att < 0:
        raise ValueError("The total number of attachment events is less"
                         " than zero. This should not have happend")
    if tot_counts_exc < 0:
        raise ValueError("The total number of exchange events is less"
                         " than zero. This should not have happend")
    if tot_counts_det > tot_counts_any:
        raise ValueError("The total number of detachment events is"
                         " higher than the total number of events. This"
                         " should not have happend")
    if tot_counts_att > tot_counts_any:
        raise ValueError("The total number of attachment events is"
                         " higher than the total number of events. This"
                         " should not have happend")
    if tot_counts_exc > tot_counts_det:
        raise ValueError("The total number of exchange events is higher"
                         " than the total number of detachment events."
                         " This should not have happend")
    if tot_counts_exc > tot_counts_att:
        raise ValueError("The total number of exchange events is higher"
                         " than the total number of attachment events"
                         " This should not have happend")

    hist_any = hist_any / tot_counts_any
    hist_det = hist_det / tot_counts_det
    hist_att = hist_att / tot_counts_att
    hist_exc = hist_exc / tot_counts_exc

    mean_any = np.sum(lag_times * hist_any)
    mean_det = np.sum(lag_times * hist_det)
    mean_att = np.sum(lag_times * hist_att)
    mean_exc = np.sum(lag_times * hist_exc)

    sd_any = np.sum((lag_times - mean_any)**2 * hist_any)
    sd_det = np.sum((lag_times - mean_det)**2 * hist_det)
    sd_att = np.sum((lag_times - mean_att)**2 * hist_att)
    sd_exc = np.sum((lag_times - mean_exc)**2 * hist_exc)

    mean_msd_any = np.nansum(msd_any * hist_any)
    mean_msd_det = np.nansum(msd_det * hist_det)
    mean_msd_att = np.nansum(msd_att * hist_att)
    mean_msd_exc = np.nansum(msd_exc * hist_exc)

    sd_msd_any = np.nansum((msd_any - mean_msd_any)**2 * hist_any)
    sd_msd_det = np.nansum((msd_det - mean_msd_det)**2 * hist_det)
    sd_msd_att = np.nansum((msd_att - mean_msd_att)**2 * hist_att)
    sd_msd_exc = np.nansum((msd_exc - mean_msd_exc)**2 * hist_exc)

    print(flush=True)
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
        "Lifetime of reference compounds until their coordination to\n"
        "selection compounds changes. Additionally, the mean square\n"
        "displacements (MSDs) of the reference compounds during these\n"
        "lifetimes is given.\n"
        "\n"
        "\n"
        "Cutoff (Angstrom)     = {}\n"
        "Compound              = {}\n"
        "Minimum contacts      = {}\n"
        "Allowed intermittency = {}\n"
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
        "The columns contain:\n"
        "   1 Lifetime / diffusion time (ps)\n"
        "   2 Lifetime histogram for reference compounds until any\n"
        "     in their coordination to selection compounds\n"
        "   3 MSD (A^2) of these reference compounds over their\n"
        "     complete lifetime\n"
        "   4 Lifetime histogram for reference compounds until\n"
        "     detachment of at least {} selection compound(s) at once\n"
        "   5 MSD (A^2) of these reference compounds over their\n"
        "     complete lifetime\n"
        "   6 Lifetime histogram for reference compounds until\n"
        "     attachment of at least {} selection compound(s) at once\n"
        "   7 MSD (A^2) of these reference compounds over their\n"
        "     complete lifetime\n"
        "   8 Lifetime histogram for reference compounds until\n"
        "     detachment of at least {} selection compound(s) and\n"
        "     attachment of at least {} selection compound(s) at once\n"
        "   9 MSD (A^2) of these reference compounds over their\n"
        "     complete lifetime\n"
        "\n"
        "Column number:\n"
        "{:>14d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d}\n"
        "\n"
        "{:>31s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s}\n"
        "Mean:          {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e}\n"
        "Std. Dev.:     {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e}\n"
        "Tot. counts:   {:>16d} {:>33d} {:>33d} {:>33d}\n"
        .format(args.CUTOFF,
                args.COMPOUND,
                args.MINCONTACTS,
                args.INTERMITTENCY,

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

                args.NDET,
                args.NATT,
                args.NDET, args.NATT,

                1, 2, 3, 4, 5, 6, 7, 8, 9,

                "Lifetime (ps)", "MSD (A^2)",
                "Lifetime (ps)", "MSD (A^2)",
                "Lifetime (ps)", "MSD (A^2)",
                "Lifetime (ps)", "MSD (A^2)",
                mean_any, mean_msd_any,
                mean_det, mean_msd_det,
                mean_att, mean_msd_att,
                mean_exc, mean_msd_exc,
                sd_any, sd_msd_any,
                sd_det, sd_msd_det,
                sd_att, sd_msd_att,
                sd_exc, sd_msd_exc,
                tot_counts_any,
                tot_counts_det,
                tot_counts_att,
                tot_counts_exc
                )
    )

    mdt.fh.savetxt(fname=args.OUTFILE,
                   data=np.column_stack([lag_times,
                                         hist_any, msd_any,
                                         hist_det, msd_det,
                                         hist_att, msd_att,
                                         hist_exc, msd_exc]),
                   header=header)

    print("  Created {}".format(args.OUTFILE))
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
