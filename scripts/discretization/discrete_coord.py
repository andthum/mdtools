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
import mdtools as mdt




def discrete_coord(cms, verbose=False, debug=False):
    """
    Generate a trajectory containing for each single reference compound
    for each frame the index of the selection compound that is
    continuously bound the longest to the reference compound. This
    function is closely related to :func:`extract_renewal_events` from
    extract_renewal_events.py.

    Parameters
    ----------
    cms : array_like
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be given as scipy sparse matrices.
        The rows must stand for the reference compounds, the columns for
        the selection compounds.
    verbose : bool, optional
        If ``True``, print progress information to standard output.
    debug : bool, optional
        If ``True``, check the input arguments.

    Returns
    -------
    traj : numpy.ndarray
        Array of shape ``(len(cms), n)``, where ``n`` is the number of
        reference compounds. The elements of the array are the indices
        of the selection compounds that are continuously bound the
        longest to the given reference compound. Indexing starts at zero.
        An index of -1 indicates that in the given frame no selection
        compound was bound to the given reference compound.
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


    refix2refix_t0 = -np.ones(cms[0].shape[0], dtype=np.int32)
    # If refix2refix_t0[rix] < 0, reference compound rix was not bound
    # to any selection compound
    refix_t0 = []
    selix_t0 = []
    t0 = []
    trenew = []


    if verbose:
        timer = datetime.now()
        proc = psutil.Process(os.getpid())
        print("  Frame   {:12d} of {:12d}"
              .format(0, len(cms)-1),
              flush=True)
        print("    Elapsed time:             {}"
              .format(datetime.now()-timer),
              flush=True)
        print("    Current memory usage: {:18.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
        timer = datetime.now()

    refix, selix = cms[0].nonzero()
    refix_unique, selix = mdt.nph.group_by(
        keys=refix.astype(np.uint32),
        values=selix.astype(np.uint32),
        assume_sorted=True,
        return_keys=True)
    refix_t0.extend(refix_unique)
    refix2refix_t0[refix_unique] = np.arange(len(refix_t0))
    selix_t0.extend(selix)
    t0.extend(np.uint32(0) for i in refix_t0)
    trenew.extend(np.int32(-1) for i in refix_t0)


    for i, cm in enumerate(cms[1:], 1):
        if (verbose and
                (i % 10**(len(str(i))-1) == 0 or i == len(cms)-1)):
            print("  Frame   {:12d} of {:12d}"
                  .format(i, len(cms)-1),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now()-timer),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss/2**20),
                  flush=True)
            timer = datetime.now()

        bound_now_and_before = cm.multiply(cms[i-1])
        attached = cm - bound_now_and_before
        refix, selix = attached.nonzero()
        if len(refix) > 0 and np.any(refix2refix_t0[refix] < 0):
            refix_unique, selix = mdt.nph.group_by(
                keys=refix.astype(np.uint32),
                values=selix.astype(np.uint32),
                assume_sorted=True,
                return_keys=True)
            for j, rix in enumerate(refix_unique):
                if refix2refix_t0[rix] >= 0:
                    # Reference compound rix is already bound to a
                    # selection compound
                    continue
                refix2refix_t0[rix] = len(refix_t0)
                refix_t0.append(rix)
                selix_t0.append(selix[j])
                t0.append(np.uint32(i))
                trenew.append(np.int32(-1))

        detached = cms[i-1] - bound_now_and_before
        refix, selix = detached.nonzero()
        if len(refix) > 0:
            refix_unique, selix = mdt.nph.group_by(
                keys=refix.astype(np.uint32),
                values=selix.astype(np.uint32),
                assume_sorted=True,
                return_keys=True)
            for j, rix in enumerate(refix_unique):
                rix_t0 = refix2refix_t0[rix]
                remain = np.isin(selix_t0[rix_t0],
                                 test_elements=selix[j],
                                 assume_unique=True,
                                 invert=True)
                if not np.any(remain):
                    # Last selection compound(s) that was (were) bound to
                    # reference compound rix at time t0 got detached
                    selix_t0[rix_t0] = selix_t0[rix_t0][0]
                    trenew[rix_t0] = np.uint32(i - t0[rix_t0])
                    selix_bound = cm[rix].nonzero()[1]
                    if len(selix_bound) == 0:
                        refix2refix_t0[rix] = -1
                    else:
                        refix2refix_t0[rix] = len(refix_t0)
                        refix_t0.append(rix)
                        selix_t0.append(selix_bound)
                        t0.append(np.uint32(i))
                        trenew.append(np.int32(-1))
                else:
                    selix_t0[rix_t0] = selix_t0[rix_t0][remain]


    del refix, selix, refix_unique, refix2refix_t0
    del cm, bound_now_and_before, attached, detached

    if len(selix_t0) != len(refix_t0):
        raise ValueError("The number of selection indices does not match"
                         " the number of reference indices. This should"
                         " not have happened")
    if len(t0) != len(refix_t0):
        raise ValueError("The number of start times does not match the"
                         " number of reference indices. This should not"
                         " have happened")
    if len(trenew) != len(refix_t0):
        raise ValueError("The number of renewal times does not match the"
                         " number of reference indices. This should not"
                         " have happened")


    refix_t0 = np.asarray(refix_t0, dtype=np.uint32)
    selix_t0 = np.asarray(selix_t0, dtype=object)
    t0 = np.asarray(t0, dtype=np.uint32)
    trenew = np.asarray(trenew, dtype=np.int32)

    # Set the renewal time for the last renewal event for each compound
    # where a start time t0 is already set but a renewal event was
    # actually not seen, to "end of trajectory - t0"
    mask = (trenew < 0)
    trenew[mask] = len(cms) - t0[mask]
    for i in np.flatnonzero(mask):
        selix_t0[i] = selix_t0[i][0]
    selix_t0 = selix_t0.astype(np.uint32)
    del mask

    ix_sort = np.lexsort((selix_t0, trenew, t0, refix_t0))
    refix_t0 = refix_t0[ix_sort]
    selix_t0 = selix_t0[ix_sort]
    t0 = t0[ix_sort]
    trenew = trenew[ix_sort]
    del ix_sort

    traj = -np.ones((cms[0].shape[0], len(cms)), dtype=np.int32)
    for i, rix in enumerate(refix_t0):
        start = t0[i]
        stop = start + trenew[i]
        traj[rix][start:stop] = selix_t0[i]

    return traj








if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())


    parser = argparse.ArgumentParser(
        description=(
            "Generate a trajectory containing for each single"
            " reference compound for each frame the index of"
            " the selection compound that is continuously bound"
            " the longest to the reference compound. This"
            " script is closely related to"
            " extract_renewal_events.py. In fact,"
            " extract_renewal_events.py gives you the same"
            " output when using the --dtraj flag."
        )
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
        help="Output filename. The ouput will be a numpy.ndarray of"
             " type numpy.int32 and shape (n, f) stored in the binary"
             " .npy format. n is the number of reference compounds and"
             " f is the number of frames. The elements of the array are"
             " the indices of the selection compounds that are"
             " continuously bound the longest to the given reference"
             " compound. Indexing starts at zero. An index of -1"
             " indicates that in the given frame no selection compound"
             " was bound to the given reference compound."
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
        '--sel',
        dest='SEL',
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




    print("\n\n\n", flush=True)
    print("Calculating contact matrices", flush=True)
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
    print("Extracting renewal events", flush=True)
    timer = datetime.now()

    dtrajs = discrete_coord(cms=cms, verbose=True, debug=args.DEBUG)
    del cms

    print(flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()

    mdt.fh.backup(args.OUTFILE)
    np.save(args.OUTFILE, dtrajs, allow_pickle=False)

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
