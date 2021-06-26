#!/usr/bin/env python3

# This file is part of MDTools.
# Copyright (C) 2021  The MDTools Development Team and all contributors
# listed in the file AUTHORS.rst
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
Compare the coordination environment of reference compounds before and
after they have changed their position.

.. todo::
    
    * Properly account for periodic boundary conditions in the direction
      given with \--d.  At the moment the assignment 'left to right' or
      'right to left' is wrong for jumps accross periodic boundaries.
    * Allow to choose between center of mass and center of geometry
      (this feature has to be implemented in
      :func:`mdtools.structure.discrete_pos_trj`).
    * Finish docstring.

Thereby distinct between three different types of postion changes:
'unsuccessful', 'left to right' or 'right to left'.  Position changes
are identified as transitions between blocks of consecutive frames in
which the reference compounds stay at the same position.

Options
-------
-f          Trajectory file.  See |supported_coordinate_formats| of
            MDAnalysis.
-s          Topology file.  See |supported_topology_formats| of
            MDAnalysis.
-o          Output filename.  Name of the text file to which to write
            the statistics about the coordination environment of the
            reference compounds before and after the position change.
--dtrj-out  Output filename for the discrete trajectory (optional).  If
            provided, the discrete trajectory is written to a binary
            :file:`.npy` file of the given filename.  The discrete
            trajectory is stored as :class:`numpy.ndarray` of dtype
            :attr:`numpy.uint32` and shape ``(n, f)``, where ``n`` is
            the number of reference compounds and ``f`` is the number of
            frames.  The elements of the discrete trajectory are the
            states in which a given compound resides at a given frame.
--bins-out  Output filename for the bin edges (optional).  If provided,
            the (average) bin edges used for creating the discrete
            trajectory are written to a text file of the given filename.
-b          First frame to read from the trajectory.  Frame numbering
            starts at zero.  Default: ``0``.
-e          Last frame to read from the trajectory.  This is exclusive,
            i.e. the last frame read is actually ``END - 1``.  A value
            of ``-1`` means to read the very last frame.  Default:
            ``-1``.
--every     Read every n-th frame from the trajectory.  Default: ``1``.
--ref       Selection string to select the reference group.  See
            MDAnalysis' |selection_syntax| for possible choices.
--sel       Selection string to select the selection group.  See
            MDAnalysis' |selection_syntax| for possible choices.
-c          Cutoff distance in Angstrom.  A reference and selection atom
            are considered to be in contact, if their distance is less
            than or equal to this cutoff.
--refcmp    {'group', 'segments', 'residues', 'fragments', 'atoms'}
            
            The compounds of the reference group whose center of mass
            positions should be discretized and whose contacts to
            selection atoms and compounds should be compared before and
            after a position change.  Reference compounds can be 'group'
            (the entire reference group), 'segments', 'residues',
            'fragments', or 'atoms'.  Refer to the MDAnalysis' user
            guide for an |explanation_of_these_terms|.  Compounds are
            made whole before calculating their centers of mass.  The
            centers of mass are wrapped back into the primary unit cell
            before discretizing their positions.  Note that in any case,
            even if ``REFCMP`` is e.g. 'residues', only the atoms
            belonging to the reference group are taken into account for
            contact calculation, even if the compound might comprise
            additional atoms that are not contained in the reference
            group.  However, the center of mass calculation is done
            considering all atoms of a compound, including those that
            are not part of the reference group.  Default: ``'atoms'``
--selcmp    {'group', 'segments', 'residues', 'fragments'}
            
            The compounds of the selection group to use for calculating
            the contact histograms.  Contacts between reference
            compounds and selection atoms are always counted.
            Additionally, contacts between reference and selection
            compounds are counted, too.  Specify here, which compounds
            to use for the selection group.  Note that in any case, even
            if ``SELCMP`` is e.g. 'residues', only the atoms belonging
            to the selection group are taken into account, even if the
            compound might comprise additional atoms that are not
            contained in the selection group.  Default: ``'residues'``
--lag       The lag time :math:`\tau` (in trajectory frames).  The
            coordination environment of the reference compounds is
            compared ``LAG`` frames before they leave their block and
            ``LAG`` frames after they have entered the new block.
            ``LAG`` must be equal to or greater than zero, but not
            greater than half of the total number of frames in the
            trajectory.  ``LAG`` must be an integer multiple of
            ``EVERY``.  Default: ``0``
--min-block-size
            Minimum block size (in trajectory frames).  Blocks of
            consecutive frames in which a given reference compound stays
            in the same position bin must comprise at least this many
            frames in order to be counted as valid block.
            ``MIN_BLOCK_SIZE`` must be greater than ``LAG`` and an
            integer multiple of ``EVERY``.  Default: ``2*LAG`` if ``LAG``
            is not zero, otherwise ``2*EVERY``.
--max-gap-size
            Maximum gap size (in trajectory frames).  The gap between
            two following valid blocks must not be greater than this
            many frames in order to count the transition between the two
            valid blocks as valid transition.  ``MAX_GAP_SIZE`` must be
            equal to or greater than zero, but should be less than
            ``MIN_BLOCK_SIZE``.  It must be an integer multiple of
            ``EVERY``.  Default: ``LAG//(2*EVERY)``
-d          {'x', 'y', 'z'}
            
            Direction.  The spatial direction in which to bin the
            positions of the reference compounds.  Default: ``'z'``
--bin-start
            Point (in Angstrom) on the chosen spatial direction to start
            binning.  Note that binning naturally starts at zero (origin
            of the simulation box).  If parsing a start value greater
            than zero, the first bin interval will be ``[0, START)``.
            In this way you can determine the width of the first bin
            independently from the other bins.  Note that ``START`` must
            lie within the simulation box obtained from the first frame
            read and it must be smaller than ``STOP``.  Default: ``0``
--bin-end   Point (in Angstrom) on the chosen spatial direction to stop
            binning.  Note that binning naturally ends at ``lbox + tol``
            (length of the simulation box in the given spatial direction
            plus a small tolerance to account for the right-open bin
            interval).  If parsing a value less than ``lbox``, the last
            bin interval will be ``[STOP, lbox+tol)``.  In this way you
            can determine the width of the last bin independently from
            the other bins.  Note that ``STOP`` must lie within the
            simulation box obtained from the first frame read and it
            must be greater than ``START``.  Default: ``lbox + tol``
--bin-num   Number of equidistant bins (not bin edges!) to use for
            discretizing the given spatial direction between ``START``
            and ``STOP``.  Note that two additional bins, ``[0, START)``
            and ``[STOP, lbox+tol)``, are created if ``START`` is not
            zero and ``STOP`` is not ``lbox``.  Default: ``10``
--bins      Text file containing custom bin edges (in Angstrom).  Bin
            edges are read from the first column, characters following a
            '#' are ignored.  Bins do not need to be equidistant.  All
            bin edges must lie within the simulation box as obtained
            from the first frame read.  If \--bins is given, it takes
            precedence over all other \--bin* flags.
--debug     Run in :ref:`debug mode <debug-mode-label>`.

See Also
--------
:mod:`lig_change_at_pos_change_blocks_hist` :
    Like
    :mod:`lig_change_at_pos_change_blocks`, but additionally keeps
    track of the position history of the reference compounds
:mod:`contact_hist` :
    Calculate the number of contacts between two MDAnalysis
    :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`
:func:`mdtools.structure.discrete_pos_trj` :
    Function that creates a discrete posotion trajectory

Notes
-----
This scripts works as follows:

In a first step, the center of mass positions of the reference compounds
are discretized along the given spatial direction using
:func:`mdtools.structure.discrete_pos_trj`.  The result is a discrete
position trajectory for every single reference compound.

In a second step, the discrete position trajectories are searched for
blocks of at least ``MIN_BLOCK_SIZE`` consecutive frames in which a
given reference compound stays in the same position bin.  A position
change of the reference compound is defined as a transition between two
such blocks that are not more than ``MAX_GAP_SIZE`` frames appart from
each other (therefore a position change is also called 'block transition'
within this script).

There are three different types of position changes:
    
    1. The index of the position bin in which the reference compound
       resides is the same in both blocks.  These position changes are
       classified as 'unsuccessful', because the reference compound
       resides at the same position in both blocks.
    2. The bin index in the first block is less than in the second block.
       These position changes are classified as 'left to right' (or 'in
       positive direction').
    3. The bin index in the first block is greater than in the second
       block.  These position changes are classified as 'right to left'
       (or 'in negative direction').

After identification and classification of the position changes, the
coordination environment of the reference compounds are compared a given
lag time :math:`\tau` before and after the position change.  Precisely,
if the first block ends at time (frame) :math:`t_0^{(1)}` and the second
block starts at :math:`t_0^{(2)}`, the coordination environment is
compared at times :math:`t_0^{(1)} - \tau` and :math:`t_0^{(2)} + \tau`.
To chose an appropriate lag time :math:`\tau`, you can for instance
first apply :mod:`state_probs_around_trans` on the output of
:mod:`discrete_pos`.

Further notes:

The **simulation box must be orthogonal**, otherwise the discretization
of the center of mass positions of the reference compounds does not work.
For more details about the discretization see the Notes section of
:func:`mdtools.structure.discrete_pos_trj`.

Examples
--------
TODO
"""


__author__ = "Andreas Thum"


# Standard libraries
import sys
import os
import warnings
import argparse
from datetime import datetime, timedelta
# Third party libraries
import psutil
import numpy as np
# Local application/library specific imports
import mdtools as mdt


if __name__ == '__main__':
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Compare the coordination environment of reference compounds"
            " before and after they have changed their position.  For"
            " more information, refer to the documetation of this script."
        )
    )
    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="Trajectory file."
    )
    parser.add_argument(
        '-s',
        dest='TOPFILE',
        type=str,
        required=True,
        help="Topology file."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename."
    )
    parser.add_argument(
        '--dtrj-out',
        dest='OUTFILE_DTRJ',
        type=str,
        required=False,
        default=None,
        help="Output filename for the discrete trajectory (optional)."
    )
    parser.add_argument(
        '--bins-out',
        dest='OUTFILE_BINS',
        type=str,
        required=False,
        default=None,
        help="Output filename for the bin edges (optional)."
    )
    parser.add_argument(
        '-b',
        dest='BEGIN',
        type=int,
        required=False,
        default=0,
        help="First frame to read from the trajectory.  Frame numbering"
             " starts at zero.  Default: %(default)s."
    )
    parser.add_argument(
        '-e',
        dest='END',
        type=int,
        required=False,
        default=-1,
        help="Last frame to read from the trajectory (exclusive)."
             "  Default: %(default)s."
    )
    parser.add_argument(
        '--every',
        dest='EVERY',
        type=int,
        required=False,
        default=1,
        help="Read every n-th frame from the trajectory.  Default:"
             " %(default)s."
    )
    parser.add_argument(
        '--ref',
        dest='REF',
        type=str,
        nargs='+',
        required=True,
        help="Selection string for the reference group."
    )
    parser.add_argument(
        '--sel',
        dest='SEL',
        type=str,
        nargs='+',
        required=True,
        help="Selection string for the selection group."
    )
    parser.add_argument(
        '-c',
        dest='CUTOFF',
        type=float,
        required=True,
        help="Cutoff distance in Angstrom."
    )
    parser.add_argument(
        '--refcmp',
        dest='REFCMP',
        type=str,
        required=False,
        choices=('group', 'segments', 'residues', 'fragments', 'atoms'),
        default='atoms',
        help="Reference compound.  Default: %(default)s"
    )
    parser.add_argument(
        '--selcmp',
        dest='SELCMP',
        type=str,
        required=False,
        choices=('group', 'segments', 'residues', 'fragments'),
        default='residues',
        help="Selection compound.  Default: %(default)s"
    )
    parser.add_argument(
        '--lag',
        dest='LAG',
        type=int,
        required=False,
        default=0,
        help="Lag time (in trajectory frames).  Default: %(default)s"
    )
    parser.add_argument(
        '--min-block-size',
        dest='MIN_BLOCK_SIZE',
        type=int,
        required=False,
        default=None,
        help="Minimum block size (in trajectory frames).  Default:"
             " 2*LAG if LAG is not zero, otherwise 2*EVERY."
    )
    parser.add_argument(
        '--max-gap-size',
        dest='MAX_GAP_SIZE',
        type=int,
        required=False,
        default=None,
        help="Maximum gap size (in trajectory frames).  Default:"
             " LAG//(2*EVERY)"
    )
    parser.add_argument(
        '-d',
        dest='DIRECTION',
        type=str,
        required=False,
        choices=('x', 'y', 'z'),
        default='z',
        help="Direction for binning.  Default: %(default)s"
    )
    parser.add_argument(
        '--bin-start',
        dest='START',
        type=float,
        required=False,
        default=0,
        help="Point (in Angstrom) on the chosen spatial direction to"
             " start binning.  Default: %(default)s"
    )
    parser.add_argument(
        '--bin-end',
        dest='STOP',
        type=float,
        required=False,
        default=None,
        help="Point (in Angstrom) on the chosen spatial direction to"
             " stop binning.  Default: lbox+tol"
    )
    parser.add_argument(
        '--bin-num',
        dest='NUM',
        type=int,
        required=False,
        default=10,
        help="Number of equidistant bins (not bin edges!) to use for"
             " discretizing the given spatial direction between START"
             " and STOP.  Default: %(default)s"
    )
    parser.add_argument(
        '--bins',
        dest='BINFILE',
        type=str,
        required=False,
        default=None,
        help="Text file containing custom bin edges (in Angstrom).  If"
             " --bins is given, it takes precedence over all other"
             " --bin* flags."
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
        raise ValueError("-c ({}) must be positive".format(args.CUTOFF))
    if args.LAG < 0:
        raise ValueError("--lag ({}) must not be negative"
                         .format(args.LAG))
    if (args.MIN_BLOCK_SIZE is not None and
        args.MIN_BLOCK_SIZE <= args.LAG):
        raise ValueError("--min-block-size ({}) must be greater than"
                         " --lag ({})"
                         .format(args.MIN_BLOCK_SIZE, args.LAG))
    if args.MAX_GAP_SIZE is not None and args.MAX_GAP_SIZE < 0:
        raise ValueError("--max-gap-size ({}) must not be negative"
                         .format(args.MAX_GAP_SIZE))
    dim = {'x': 0, 'y': 1, 'z': 2}
    ixd = dim[args.DIRECTION]
    if mdt.rti.get_num_CPUs() > 1:
        mdabackend = 'OpenMP'
    else:
        mdabackend = 'serial'
    
    print("\n")
    u = mdt.select.universe(top=args.TOPFILE, trj=args.TRJFILE)
    print("\n")
    print("Creating selections...")
    timer = datetime.now()
    ref = u.select_atoms(' '.join(args.REF))
    sel = u.select_atoms(' '.join(args.SEL))
    if ref.n_atoms == 0:
        raise ValueError("The reference group contains no atoms")
    if sel.n_atoms == 0:
        raise ValueError("The selection group contains no atoms")
    print("Reference group: '{}'".format(' '.join(args.REF)))
    print(mdt.rti.ag_info_str(ag=ref, indent=2))
    print("Selection group: '{}'".format(' '.join(args.SEL)))
    print(mdt.rti.ag_info_str(ag=sel, indent=2))
    print("Elapsed time:         {}".format(datetime.now()-timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20))
    print("\n")
    print("Checking frame slice...")
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=u.trajectory.n_frames
    )
    if args.DEBUG:
        print("\n")
        mdt.check.time_step(trj=u.trajectory[BEGIN:END])
    time_step = u.trajectory[BEGIN].dt
    md_trj = u.trajectory[BEGIN:END:EVERY]
    print("\n")
    print("Checking 'LAG'...")
    LAG, LAG_EFF = mdt.check.frame_lag(
        lag=args.LAG,
        every=EVERY,
        n_frames_tot=u.trajectory.n_frames,
        allow_zero=True
    )
    if LAG_EFF > N_FRAMES // 2:
        LAG_EFF = N_FRAMES // 2
        LAG = LAG_EFF * EVERY
        print("Set 'LAG' to {}".format(LAG))
    print("\n")
    print("Checking 'MIN_BLOCK_SIZE'...")
    if args.MIN_BLOCK_SIZE is None:
        if LAG != 0:
            args.MIN_BLOCK_SIZE = 2 * LAG
        else:
            args.MIN_BLOCK_SIZE = 2 * EVERY
    MIN_BLOCK_SIZE, MIN_BLOCK_SIZE_EFF = mdt.check.frame_lag(
        lag=args.MIN_BLOCK_SIZE,
        every=EVERY,
        n_frames_tot=u.trajectory.n_frames,
    )
    if MIN_BLOCK_SIZE <= LAG:
        MIN_BLOCK_SIZE = LAG + 1
        MIN_BLOCK_SIZE_EFF = LAG_EFF + EVERY
        print("Set 'MIN_BLOCK_SIZE' to {}".format(MIN_BLOCK_SIZE))
    print("\n")
    print("Checking 'MAX_GAP_SIZE'...")
    if args.MAX_GAP_SIZE is None:
        args.MAX_GAP_SIZE = LAG // (2 * EVERY)
    MAX_GAP_SIZE, MAX_GAP_SIZE_EFF = mdt.check.frame_lag(
        lag=args.MAX_GAP_SIZE,
        every=EVERY,
        n_frames_tot=u.trajectory.n_frames,
        allow_zero=True
    )
    if MAX_GAP_SIZE >= MIN_BLOCK_SIZE:
        warnings.warn("'MAX_GAP_SIZE' ({}) should be less than"
                      " 'MIN_BLOCK_SIZE' ({})"
                      .format(MAX_GAP_SIZE, MIN_BLOCK_SIZE),
                      RuntimeWarning)
    
    # Reference group containing *all* atoms of the given compound (for
    # creating the discrete center of mass trajectory)
    if args.REFCMP == 'group':
        refcmp = ref
        N_REFCMPS = 1
    elif args.REFCMP == 'segments':
        refcmp = ref.segments.atoms
        N_REFCMPS = ref.n_segments
    elif args.REFCMP == 'residues':
        refcmp = ref.residues.atoms
        N_REFCMPS = ref.n_residues
    elif args.REFCMP == 'fragments':
        refcmp = ref.fragments.atoms
        N_REFCMPS = ref.n_fragments
    elif args.REFCMP == 'atoms':
        refcmp = ref
        N_REFCMPS = ref.n_atoms
    else:
        raise ValueError("--refcmp must be either 'group', 'segments',"
                         " 'residues', 'fragments', or 'atoms', but you"
                         " gave {}".format(args.REFCMP))
    # Number of atoms per compound (needed later in step 2 for the
    # calculation of contact matrices):
    natms_per_refcmp = mdt.strc.natms_per_cmp(ag=ref,
                                              compound=args.REFCMP,
                                              return_array=True,
                                              check_contiguos=True)
    if len(natms_per_refcmp) != N_REFCMPS:
        raise ValueError("'len(natms_per_refcmp)' ({}) != 'N_REFCMPS'"
                         " ({}). This should not have happened"
                         .format(len(natms_per_refcmp), N_REFCMPS))
    refcmp_slices = np.cumsum(natms_per_refcmp, dtype=np.uint32)
    refcmp_slices = np.insert(refcmp_slices, 0, 0)
    del natms_per_refcmp
    natms_per_selcmp = mdt.strc.natms_per_cmp(ag=sel,
                                              compound=args.SELCMP,
                                              check_contiguos=True)
    if np.ndim(natms_per_selcmp) == 0:
        N_SELCMPS = sel.n_atoms // natms_per_selcmp
    else:
        N_SELCMPS = len(natms_per_selcmp)
    
    print("\n")
    print("Step 1/2:")
    # Creating discrete position trajectory...
    if args.BINFILE is None:
        bins = None
    else:
        bins = np.loadtxt(args.BINFILE, usecols=0)
    dtrj, bins, lbox_av = mdt.strc.discrete_pos_trj(
        sel=refcmp,
        trj=md_trj,
        compound=args.REFCMP,
        direction=args.DIRECTION,
        bin_start=args.START,
        bin_stop=args.STOP,
        bin_num=args.NUM,
        bins=bins,
        return_bins=True,
        return_lbox=True,
        dtype=np.uint32,
        verbose=True,
        debug=args.DEBUG
    )
    
    if args.OUTFILE_BINS is not None or args.OUTFILE_DTRJ is not None:
        print("\n")
        print("Creating output...")
        timer = datetime.now()
    # Discrete trajectory:
    if args.OUTFILE_DTRJ is not None:
        mdt.fh.backup(args.OUTFILE_DTRJ)
        np.save(args.OUTFILE_DTRJ, dtrj, allow_pickle=False)
        print("Created {}".format(args.OUTFILE_DTRJ))
    # Bin edges:
    if args.OUTFILE_BINS is not None:
        header = ("Bin edges in Angstrom\n"
                  "Number of bin edges:                  {:<d}\n"
                  "Number of bins:                       {:<d}\n"
                  "Discretized spatial dimension:        {:<s}\n"
                  "Average box length in this direction: {:<.9e} A\n"
                  .format(len(bins),
                          len(bins)-1,
                          args.DIRECTION,
                          lbox_av))
        mdt.fh.savetxt(fname=args.OUTFILE_BINS, data=bins, header=header)
        print("Created {}".format(args.OUTFILE_BINS))
    if args.OUTFILE_BINS is not None or args.OUTFILE_DTRJ is not None:
        print("Elapsed time:         {}".format(datetime.now()-timer))
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss/2**20))
    del bins
    
    print("\n")
    print("Step 2/2:")
    print("Comparing coordination environments...")
    print("Number of compounds:    {:>8d}".format(N_REFCMPS))
    print("Total number of frames: {:>8d}".format(u.trajectory.n_frames))
    print("Frames to read:         {:>8d}".format(N_FRAMES))
    print("First frame to read:    {:>8d}".format(BEGIN))
    print("Last frame to read:     {:>8d}".format(END-1))
    print("Read every n-th frame:  {:>8d}".format(EVERY))
    print("Time first frame:       {:>12.3f} (ps)"
          .format(u.trajectory[BEGIN].time))
    print("Time last frame:        {:>12.3f} (ps)"
          .format(u.trajectory[END-1].time))
    print("Time step first frame:  {:>12.3f} (ps)"
          .format(u.trajectory[BEGIN].dt))
    print("Time step last frame:   {:>12.3f} (ps)"
          .format(u.trajectory[END-1].dt))
    timer = datetime.now()
    # Total number of state transitions
    n_state_trans = 0
    # Total number of blocks (any block size)
    n_blocks = 0
    # Total number of valid blocks (block_size >= MIN_BLOCK_SIZE)
    n_valid_blocks = 0
    # Average size of all blocks
    block_size_av = dtrj.size  # must be normalized by n_blocks
    # Average size of valid blocks
    valid_block_size_av = 0
    # Transition types
    trans_types = ("unsuccessful", "left2right", "right2left")
    # Total number of block transitions (any gap_size) between valid
    # blocks
    n_block_trans = np.zeros(len(trans_types), dtype=np.uint32)
    # Total number of valid block transitions (gap_size <= MAX_GAP_SIZE)
    # between valid blocks (i.e. total number of actually counted
    # position changes)
    n_valid_block_trans = np.zeros_like(n_block_trans)
    # Average size of all gaps (any gap size) between valid blocks
    gap_size_av = np.zeros_like(n_block_trans)
    # Average size of valid gaps (gap_size <= MAX_GAP_SIZE) between
    # valid blocks
    valid_gap_size_av = np.zeros_like(n_valid_block_trans)
    # Number of refcmps bound to selatms before a valid block transition
    # (only needed to normalize selix_stats_b)
    n_refcmps_bound_b = np.zeros_like(n_valid_block_trans)
    # Number of refcmps bound to selatms after a valid block transition
    # (only needed to normalize selix_stats_a)
    n_refcmps_bound_a = np.zeros_like(n_refcmps_bound_b)
    # Number of refcmps bound to selatms before and after a valid block
    # transition (only needed to normalize selix_stats_diff)
    n_refcmps_bound_ba = np.zeros_like(n_refcmps_bound_b)
    # For contancts between refcmps and
    #  0. selatms,
    #  1. selcmps,
    # [^ -> Indices for the 1st dimension]
    # compute the...
    # ...number of contacts between refcmps and selatms/selcmps
    #  0. before a valid block transition
    #  1. after a valid block transition
    #  2. common/remaining contacts
    # [^ -> Indices for the 3rd dimension]
    n_contacts = np.zeros((2, len(n_valid_block_trans), 3),
                          dtype=np.uint32)
    # ...average
    #  0. average
    #  1. variance
    #  2. minimum
    #  3. maximum
    # [^ -> Indices for the 3rd dimension]
    # of the indices of selatms/selcmps that are bound to refcmps
    selix_stats_b = np.zeros((2, len(n_refcmps_bound_b), 4),
                             dtype=np.float64)
    selix_stats_a = np.zeros((2, len(n_refcmps_bound_a), 4),
                             dtype=np.float64)
    # ...difference (before and after a valid block transition) of the
    #  0. average
    #  1. variance
    #  2. minimum
    #  3. maximum
    # [^ -> Indices for the 3rd dimension]
    # of the indices of selatms/selcmps that are bound to refcmps
    selix_stats_diff = np.zeros((2, len(n_refcmps_bound_ba), 4),
                                dtype=np.float64)
    # Loop over single compound trajectories:
    dtrj = mdt.rti.ProgressBar(dtrj, unit="compounds")
    for rc, cmp_trj in enumerate(dtrj):
        # Frame indices directly after state transitions:
        trans = np.flatnonzero(np.diff(cmp_trj))
        trans += 1
        n_state_trans += len(trans)
        # Sizes (number of frames) of blocks of same states
        block_sizes = np.diff(trans, prepend=0, append=len(cmp_trj))
        del trans
        n_blocks += len(block_sizes)
        valid_blocks = (block_sizes >= MIN_BLOCK_SIZE_EFF)
        n_valid_blocks_tmp = np.count_nonzero(valid_blocks)
        if n_valid_blocks_tmp == 0:
            continue
        n_valid_blocks += n_valid_blocks_tmp
        valid_block_size_av += np.sum(block_sizes[valid_blocks])
        # MDAnalysis AtomGroup containing a single reference compound:
        refcmp = ref[refcmp_slices[rc]:refcmp_slices[rc+1]]
        # Frame index where the current block ends:
        frame = -1
        # Loop over all blocks:
        for b, b_size in enumerate(block_sizes[:-1]):
            frame += b_size
            if not valid_blocks[b]:
                # Current block is too short
                continue
            # Index of next valid block:
            next_valid = b + 1 + np.argmax(valid_blocks[b+1:])
            if not valid_blocks[next_valid]:
                # All following blocks are too short
                break
            # Number of frames until the next valid block:
            gap_size = np.sum(block_sizes[b+1:next_valid])
            if cmp_trj[frame] == cmp_trj[frame+gap_size+1]:
                # "unsuccessful" transition type
                tt = 0
            elif cmp_trj[frame] < cmp_trj[frame+gap_size+1]:
                # "left to right" transition type
                tt = 1
            else:
                # "right to left" transition type
                tt = 2
            n_block_trans[tt] += 1
            gap_size_av[tt] += gap_size
            if gap_size > MAX_GAP_SIZE_EFF:
                # Gap between two consecutive valid blocks is too large
                continue
            valid_gap_size_av[tt] += gap_size
            n_valid_block_trans[tt] += 1
            # refcmp-selatm contact matrix at t0_b-dt (b = "before"):
            ts = md_trj[frame-LAG_EFF]
            cm_b = mdt.strc.contact_matrix(
                ref=refcmp,
                sel=sel,
                cutoff=args.CUTOFF,
                compound=(args.REFCMP, 'atoms'),
                box=ts.dimensions,
                mdabackend=mdabackend
            )
            # refcmp-selatm contact matrix at t0_a+dt (a = "after"):
            ts = md_trj[frame+gap_size+1+LAG_EFF]
            cm_a = mdt.strc.contact_matrix(
                ref=refcmp,
                sel=sel,
                cutoff=args.CUTOFF,
                compound=(args.REFCMP, 'atoms'),
                box=ts.dimensions,
                mdabackend=mdabackend
            )
            for sc in range(2):  # sc=0 -> selatm,  sc=1 -> selcmp
                if sc == 1:
                    # refcmp-selcmp contact matrices
                    cm_b = mdt.strc.cmp_contact_matrix(
                        cm=cm_b,
                        natms_per_selcmp=natms_per_selcmp
                    )
                    cm_a = mdt.strc.cmp_contact_matrix(
                        cm=cm_a,
                        natms_per_selcmp=natms_per_selcmp
                    )
                n_contacts[sc][tt] += mdt.strc.cms_n_contacts(
                    (cm_b, cm_a),
                    dtype=n_contacts.dtype
                )
                # Contact statistics before block transition
                selix_stats_b_tmp = np.squeeze(mdt.strc.cm_selix_stats(cm_b))
                if selix_stats_b_tmp[0] > 0:  # selix_stats_b_tmp[0] = n_sel
                    if sc == 0:
                        n_refcmps_bound_b[tt] += 1
                    # selix_stats_b_tmp[1:] = mean, var, min, max
                    selix_stats_b[sc][tt] += selix_stats_b_tmp[1:]
                # Contact statistics after block transition
                selix_stats_a_tmp = np.squeeze(mdt.strc.cm_selix_stats(cm_a))
                if selix_stats_a_tmp[0] > 0:
                    if sc == 0:
                        n_refcmps_bound_a[tt] += 1
                    selix_stats_a[sc][tt] += selix_stats_a_tmp[1:]
                # Contact statistics difference
                if selix_stats_b_tmp[0] > 0 and selix_stats_a_tmp[0] > 0:
                    if sc == 0:
                        n_refcmps_bound_ba[tt] += 1
                    selix_stats_diff[sc][tt] += np.abs(
                        selix_stats_b_tmp[1:] - selix_stats_a_tmp[1:]
                    )
        # ProgressBar update:
        progress_bar_mem = proc.memory_info().rss / 2**20
        dtrj.set_postfix_str("{:>7.2f}MiB".format(progress_bar_mem),
                             refresh=False)
    dtrj.close()
    del dtrj, cmp_trj, block_sizes, valid_blocks, natms_per_selcmp
    print("Elapsed time:         {}".format(datetime.now()-timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20))
    
    if n_state_trans == 0:
        warnings.warn("The reference compounds did not change their"
                      " position bins. The output will be meaningless."
                      " Try another discretization", RuntimeWarning)
    if n_valid_blocks == 0:
        warnings.warn("The number of valid blocks is zero. The output"
                      " will be meaningless. Try to decrease the minimum"
                      " block size", RuntimeWarning)
    if np.all(n_valid_block_trans == 0):
        warnings.warn("The number of valid block transitions is zero."
                      " The output will be meaningless. Try to increase"
                      " the maximum gap size", RuntimeWarning)
    elif np.any(n_valid_block_trans == 0):
        warnings.warn("For at least one transition type ('unsuccessful',"
                      " 'left2right' or 'right2left') the number of"
                      " valid block transitions is zero. The"
                      " corresponding output will be meaningless. Try to"
                      " increase the maximum gap size", RuntimeWarning)
    
    # Effective frame numbers -> real frame numbers
    block_size_av *= EVERY
    valid_block_size_av *= EVERY
    gap_size_av *= EVERY
    valid_gap_size_av *= EVERY
    
    # Compute averages:
    block_size_av /= n_blocks
    valid_block_size_av /= n_valid_blocks
    gap_size_av = gap_size_av / n_block_trans
    valid_gap_size_av = valid_gap_size_av / n_valid_block_trans
    n_contacts = n_contacts / n_valid_block_trans[:,None]
    selix_stats_b /= n_refcmps_bound_b[:,None]
    selix_stats_a /= n_refcmps_bound_a[:,None]
    selix_stats_diff /= n_refcmps_bound_ba[:,None]
    
    # n_detached = n_contacts_before - n_remain
    # n_attached = n_contacts_after - n_remain
    for sc in range(2):
        for tt in range(len(trans_types)):
            n_contacts[sc][tt][:-1] -= n_contacts[sc][tt][-1]
    
    print("\n")
    print("Creating output...")
    timer = datetime.now()
    # Contact statistics:
    mdt.fh.write_header(args.OUTFILE)
    with open(args.OUTFILE, 'a') as outfile:
        outfile.write("# \n")
        outfile.write("# \n")
        outfile.write("# Reference: '{}'\n".format(' '.join(args.REF)))
        outfile.write(mdt.fh.indent(text=mdt.rti.ag_info_str(ag=ref),
                                    amount=1,
                                    char="#   ")
                      + "\n")
        outfile.write("# \n")
        outfile.write("# Selection: '{}'\n".format(' '.join(args.SEL)))
        outfile.write(mdt.fh.indent(text=mdt.rti.ag_info_str(ag=sel),
                                    amount=1,
                                    char="#   ")
                      + "\n")
        outfile.write("# \n")
        outfile.write("# \n")
        outfile.write("# Contacts before and after position change\n")
        outfile.write("# Cutoff (Angstrom):     {}\n".format(args.CUTOFF))
        outfile.write("# Reference compound:   '{}'\n".format(args.REFCMP))
        outfile.write("# Selection compound:   '{}'\n".format(args.SELCMP))
        outfile.write("# Lag time:              {:.3f} ps\n".format(LAG*time_step))
        outfile.write("# Minimum block size     {:.3f} ps\n".format(MIN_BLOCK_SIZE*time_step))
        outfile.write("# Maximum gap size       {:.3f} ps\n".format(MAX_GAP_SIZE*time_step))
        outfile.write("# Discretized dimension: {}\n".format(args.DIRECTION))
        outfile.write("# \n")
        outfile.write("# \n")
        outfile.write("# No. of analyzed frames:                        {:>12d}\n".format(N_FRAMES))
        outfile.write("# No. of reference compounds (refcmps):          {:>12d}\n".format(N_REFCMPS))
        outfile.write("# No. of selection atoms     (selatms):          {:>12d}\n".format(sel.n_atoms))
        outfile.write("# No. of selection compounds (selcmps):          {:>12d}\n".format(N_SELCMPS))
        outfile.write("# Tot. No. of transitions between position bins: {:>12d}\n".format(n_state_trans))
        outfile.write("# Tot. No. of       blocks (   any block size):  {:>12d}\n".format(n_blocks))
        outfile.write("# Tot. No. of valid blocks (>= min block size):  {:>12d}\n".format(n_valid_blocks))
        outfile.write("# Av. size of all   blocks (   any block size):  {:>16.3f} ps\n".format(block_size_av*time_step))
        outfile.write("# Av. size of valid blocks (>= min block size):  {:>16.3f} ps\n".format(valid_block_size_av*time_step))
        outfile.write("# \n")
        outfile.write("# \n")
        outfile.write("# The COLUMNS contain:\n")
        outfile.write("#   1 The row numbers\n")
        outfile.write("#   2 The row names\n")
        outfile.write("#   The following columns contain contact statistics about\n")
        outfile.write("#     refcmp-selatm contacts of refcmps that undergo\n")
        outfile.write("#       3 'unsuccessful'  transitions\n")
        outfile.write("#       4 'left to right' transitions\n")
        outfile.write("#       5 'right to left' transitions\n")
        outfile.write("#     refcmp-selcmp contacts of refcmps that undergo\n")
        outfile.write("#       6 'unsuccessful'  transitions\n")
        outfile.write("#       7 'left to right' transitions\n")
        outfile.write("#       8 'right to left' transitions\n")
        outfile.write("# \n")
        outfile.write("# The ROWS contain:\n")
        outfile.write("#    1 Tot. No. of       block transitions (any gap size)\n")
        outfile.write("#    2 Tot. No. of valid block transitions (gap size <= max gap size)\n")
        outfile.write("#    3 Av. size of all   gaps (   any gap size) (in ps)\n")
        outfile.write("#    4 Av. size of valid gaps (<= max gap size) (in ps)\n")
        outfile.write("#    5 Av. No. of selatms/selcmps      detached     from a refcmp  during a valid block transition\n")
        outfile.write("#    6 Av. No. of selatms/selcmps      attached     to   a refcmp  during a valid block transition\n")
        outfile.write("#    7 Av. No. of selatms/selcmps that remain bound to   a refcmp  during a valid block transition\n")
        outfile.write("#    8 Av. average  of indices of selatms/selcmps that are bound to refcmps before a valid block transition\n")
        outfile.write("#    9 Av. average  of indices of selatms/selcmps that are bound to refcmps after  a valid block transition\n")
        outfile.write("#   10 Av. variance of indices of selatms/selcmps that are bound to refcmps before a valid block transition\n")
        outfile.write("#   11 Av. variance of indices of selatms/selcmps that are bound to refcmps after  a valid block transition\n")
        outfile.write("#   12 Av. minimum     index   of selatms/selcmps that are bound to refcmps before a valid block transition\n")
        outfile.write("#   13 Av. minimum     index   of selatms/selcmps that are bound to refcmps after  a valid block transition\n")
        outfile.write("#   14 Av. maximum     index   of selatms/selcmps that are bound to refcmps before a valid block transition\n")
        outfile.write("#   15 Av. maximum     index   of selatms/selcmps that are bound to refcmps after  a valid block transition\n")
        outfile.write("#   16 Av. absolute change of the average  of indices of selatms/selcmps that are bound to refcmps\n")
        outfile.write("#   17 Av. absolute change of the variance of indices of selatms/selcmps that are bound to refcmps\n")
        outfile.write("#   18 Av. absolute change of the minimum     index   of selatms/selcmps that are bound to refcmps\n")
        outfile.write("#   19 Av. absolute change of the maximum     index   of selatms/selcmps that are bound to refcmps\n")
        outfile.write("# \n")
        col_names = ("row_num", "row_name") + 2*trans_types
        row_names = ("n_block_trans", "n_valid_block_trans",
                     "gap_size", "gap_size_valid",
                     "n_detached", "n_attached", "n_remain",
                     "ix_av_before", "ix_av_after",
                     "ix_var_before", "ix_var_after",
                     "ix_min_before", "ix_min_after",
                     "ix_max_before", "ix_max_after",
                     "ix_av_diff", "ix_var_diff",
                     "ix_min_diff", "ix_max_diff")
        # Column numbers:
        outfile.write("# {:>7d}".format(1))
        outfile.write(" {:<19d}".format(2))
        for col_num in range(3, len(col_names)+1):
            if col_num in (3, 6):
                outfile.write("  ")
            outfile.write(" {:>16d}".format(col_num))
        outfile.write("\n")
        # Column "header headers":
        outfile.write("# {:<28s}".format(" "))
        outfile.write("   {:<50s}".format("refcmp_selatm"))
        outfile.write("   {:<s}\n".format("refcmp_selcmp"))
        # Column headers:
        outfile.write("# {:>7s}".format(col_names[0]))
        outfile.write(" {:<19s}".format(col_names[1]))
        for col_num, col_name in enumerate(col_names[2:], start=3):
            if col_num in (3, 6):
                outfile.write("  ")
            outfile.write(" {:>16s}".format(col_name))
        outfile.write("\n")
        # Data:
        # n_block_trans
        row = 0
        outfile.write("{:>9d}".format(row+1))
        outfile.write(" {:<19s}".format(row_names[row]))
        for sc in range(2):
            outfile.write("  ")
            for tt in range(len(trans_types)):
                outfile.write(" {:>16d}".format(n_block_trans[tt]))
        outfile.write("\n")
        # n_valid_block_trans
        row += 1
        outfile.write("{:>9d}".format(row+1))
        outfile.write(" {:<19s}".format(row_names[row]))
        for sc in range(2):
            outfile.write("  ")
            for tt in range(len(trans_types)):
                outfile.write(" {:>16d}".format(n_valid_block_trans[tt]))
        outfile.write("\n")
        # gap_size_av
        row += 1
        outfile.write("{:>9d}".format(row+1))
        outfile.write(" {:<19s}".format(row_names[row]))
        for sc in range(2):
            outfile.write("  ")
            for tt in range(len(trans_types)):
                outfile.write(" {:>16.9e}".format(gap_size_av[tt] *
                                                  time_step))
        outfile.write("\n")
        # valid_gap_size_av
        row += 1
        outfile.write("{:>9d}".format(row+1))
        outfile.write(" {:<19s}".format(row_names[row]))
        for sc in range(2):
            outfile.write("  ")
            for tt in range(len(trans_types)):
                outfile.write(" {:>16.9e}".format(valid_gap_size_av[tt] *
                                                  time_step))
        outfile.write("\n")
        # n_contacts (detached, attached, remain)
        for i in range(n_contacts.shape[-1]):
            row += 1
            outfile.write("{:>9d}".format(row+1))
            outfile.write(" {:<19s}".format(row_names[row]))
            for sc in range(2):
                outfile.write("  ")
                for tt in range(len(trans_types)):
                    outfile.write(" {:>16.9e}".format(n_contacts[sc][tt][i]))
            outfile.write("\n")
        # selix_stats_b and selix_stats_a
        for i in range(selix_stats_b.shape[-1]):
            # selix_stats_b
            row += 1
            outfile.write("{:>9d}".format(row+1))
            outfile.write(" {:<19s}".format(row_names[row]))
            for sc in range(2):
                outfile.write("  ")
                for tt in range(len(trans_types)):
                    outfile.write(" {:>16.9e}".format(selix_stats_b[sc][tt][i]))
            outfile.write("\n")
            # selix_stats_a
            row += 1
            outfile.write("{:>9d}".format(row+1))
            outfile.write(" {:<19s}".format(row_names[row]))
            for sc in range(2):
                outfile.write("  ")
                for tt in range(len(trans_types)):
                    outfile.write(" {:>16.9e}".format(selix_stats_a[sc][tt][i]))
            outfile.write("\n")
        # selix_stats_diff
        for i in range(selix_stats_diff.shape[-1]):
            row += 1
            outfile.write("{:>9d}".format(row+1))
            outfile.write(" {:<19s}".format(row_names[row]))
            for sc in range(2):
                outfile.write("  ")
                for tt in range(len(trans_types)):
                    outfile.write(" {:>16.9e}".format(selix_stats_diff[sc][tt][i]))
            outfile.write("\n")
    print("Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}".format(datetime.now()-timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20))
    
    print("\n")
    print("Checking output for consistency...")
    timer = datetime.now()
    if n_state_trans > N_FRAMES * N_REFCMPS:
        raise ValueError("'n_state_trans' ({}) > 'N_FRAMES' ({}) *"
                         " 'N_REFCMPS' ({})"
                         .format(n_state_trans, N_FRAMES, N_REFCMPS))
    if n_blocks > N_REFCMPS * N_FRAMES:
        raise ValueError("'n_blocks' ({}) > 'N_REFCMPS' ({}) * 'N_FRAMES'"
                         " ({})".format(n_blocks, N_REFCMPS, N_FRAMES))
    if n_blocks < N_REFCMPS:
        raise ValueError("'n_blocks' ({}) < 'N_REFCMPS' ({})"
                         .format(n_blocks, N_REFCMPS))
    if n_valid_blocks > N_REFCMPS * N_FRAMES // MIN_BLOCK_SIZE_EFF:
        raise ValueError("'n_valid_blocks' ({}) > {} ('N_REFCMPS' {} *"
                         " 'N_FRAMES' {} // 'MIN_BLOCK_SIZE_EFF' {})"
                         .format(n_valid_blocks,
                                 N_REFCMPS*N_FRAMES//MIN_BLOCK_SIZE_EFF,
                                 N_REFCMPS,
                                 N_FRAMES,
                                 MIN_BLOCK_SIZE_EFF))
    if n_valid_blocks > n_blocks:
        raise ValueError("'n_valid_blocks' ({}) > 'n_blocks' ({})"
                         .format(n_valid_blocks, n_blocks))
    if block_size_av > u.trajectory.n_frames:
        raise ValueError("'block_size_av' ({}) > 'u.trajectory.n_frames'"
                         " ({})"
                         .format(block_size_av, u.trajectory.n_frames))
    if valid_block_size_av > u.trajectory.n_frames:
        raise ValueError("'valid_block_size_av' ({}) >"
                         " 'u.trajectory.n_frames' ({})"
                         .format(valid_block_size_av,
                                 u.trajectory.n_frames))
    if valid_block_size_av < MIN_BLOCK_SIZE:
        raise ValueError("'valid_block_size_av' ({}) < 'MIN_BLOCK_SIZE'"
                         " ({})"
                         .format(valid_block_size_av, MIN_BLOCK_SIZE))
    if valid_block_size_av < block_size_av:
        raise ValueError("'valid_block_size_av' ({}) < 'block_size_av'"
                         " ({})"
                         .format(valid_block_size_av, block_size_av))
    if np.any(n_block_trans > n_state_trans):
        raise ValueError("'np.any(n_block_trans > n_state_trans)'"
                         "'n_block_trans' = {}\n"
                         "'n_state_trans' = {}\n"
                         .format(n_block_trans, n_state_trans))
    if np.sum(n_block_trans) > n_valid_blocks - 1:
        raise ValueError("'np.sum(n_block_trans) > n_valid_blocks - 1'\n"
                         "'n_block_trans' = {}\n"
                         "'n_valid_blocks' = {}"
                         .format(n_block_trans, n_valid_blocks))
    if np.any(n_valid_block_trans > n_block_trans):
        raise ValueError("'np.any(n_valid_block_trans > n_block_trans)'"
                         "'n_valid_block_trans' = {}\n"
                         "'n_block_trans' = {}\n"
                         .format(n_valid_block_trans, n_block_trans))
    if np.any(gap_size_av > u.trajectory.n_frames):
        raise ValueError("'np.any(gap_size_av > u.trajectory.n_frames)'\n"
                         "'gap_size_av' = {}\n"
                         "'u.trajectory.n_frames' = {}\n"
                         .format(gap_size_av, u.trajectory.n_frames))
    if np.any(valid_gap_size_av > MAX_GAP_SIZE):
        raise ValueError("'np.any(valid_gap_size_av > MAX_GAP_SIZE)'\n"
                         "'valid_gap_size_av' = {}\n"
                         "'MAX_GAP_SIZE' = {}\n"
                         .format(valid_gap_size_av, MAX_GAP_SIZE))
    if np.any(valid_gap_size_av > gap_size_av):
        raise ValueError("'np.any(valid_gap_size_av > gap_size_av)'\n"
                         "'valid_gap_size_av' = {}\n"
                         "'gap_size_av' = {}\n"
                         .format(valid_gap_size_av, gap_size_av))
    if np.any(n_refcmps_bound_b > N_FRAMES * N_REFCMPS):
        raise ValueError("'np.any(n_refcmps_bound_b > N_FRAMES * N_REFCMPS)'\n"
                         "'n_refcmps_bound_b' = {}\n"
                         "'N_FRAMES' = {}\n"
                         "'N_REFCMPS' = {}\n"
                         .format(n_refcmps_bound_b, N_FRAMES, N_REFCMPS))
    if np.any(n_refcmps_bound_a > N_FRAMES * N_REFCMPS):
        raise ValueError("'np.any(n_refcmps_bound_a > N_FRAMES * N_REFCMPS)'\n"
                         "'n_refcmps_bound_a' = {}\n"
                         "'N_FRAMES' = {}\n"
                         "'N_REFCMPS' = {}\n"
                         .format(n_refcmps_bound_a, N_FRAMES, N_REFCMPS))
    if np.any(n_refcmps_bound_ba > n_refcmps_bound_b):
        raise ValueError("'np.any(n_refcmps_bound_ba > n_refcmps_bound_b)'\n"
                         "'n_refcmps_bound_ba' = {}\n"
                         "'n_refcmps_bound_b' = {}\n"
                         .format(n_refcmps_bound_ba, n_refcmps_bound_b))
    if np.any(n_refcmps_bound_ba > n_refcmps_bound_a):
        raise ValueError("'np.any(n_refcmps_bound_ba > n_refcmps_bound_a)'\n"
                         "'n_refcmps_bound_ba' = {}\n"
                         "'n_refcmps_bound_a' = {}\n"
                         .format(n_refcmps_bound_ba, n_refcmps_bound_a))
    for sc in range(2):
        for tt in range(len(trans_types)):
            if selix_stats_b[sc][tt][2] > selix_stats_b[sc][tt][3]:
                raise ValueError("Minimum index ({}) is greater than"
                                 " maximum index ({})"
                                 .format(selix_stats_b[sc][tt][2],
                                         selix_stats_b[sc][tt][3]))
    for sc in range(2):
        for tt in range(len(trans_types)):
            if selix_stats_a[sc][tt][2] > selix_stats_a[sc][tt][3]:
                raise ValueError("Minimum index ({}) is greater than"
                                 " maximum index ({})"
                                 .format(selix_stats_a[sc][tt][2],
                                         selix_stats_a[sc][tt][3]))
    print("Elapsed time:         {}".format(datetime.now()-timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20))
    
    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now()-timer_tot))
    print("CPU time:             {}"
          .format(timedelta(seconds=sum(proc.cpu_times()[:4]))))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20))
