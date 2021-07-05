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
Calculate how many selection :class:`Atoms <MDAnalysis.core.groups.Atom>`
and compounds stay attached to a reference compound that changed its
position.

Given that at time :math:`t_0` a reference compound changed its position
bin, compare how many and which selection
:class:`Atoms <MDAnalysis.core.groups.Atom>` and compounds are attached
to that reference compound at time :math:`t_0 - \Delta t` and
:math:`t_0 + \Delta t`.  Thereby, distinct between reference compounds
that are at time :math:`t_0 - \Delta t` and :math:`t_0 + \Delta t` in
different bins or in the same bin.

Additionally, do the same for reference compounds that did not change
their position at time :math:`t_0`.

See Also
--------
:mod:`contact_hist`
:mod:`contact_hist_at_pos_change`
:mod:`discrete_pos`
:mod:`state_probs_around_trans`

Notes
-----
The simulation box must be orthogonal, otherwise the discretization of
the center of mass positions of the reference compounds does not work.

Compounds are asigned to bins according to their center of mass position.
Compounds are made whole before calculating their centers of mass.  The
centers of mass are wrapped back into the primary unit cell before
discretizing their positions.

The discretization of the compounds' positions is done in relative box
coordinates.  The final output is scaled by the average box length in
the given spatial direction.  Doing so accounts for possible
fluctuations of the simulation box (e.g. due to pressure scaling).  Note
that :mod:`MDAnalysis` always sets the origin of the simulation box to
the origin of the cartesian coordinate system.

All bin intervals are left-closed and right-open, i.e. [a, b) ->
a <= x < b.  The first bin edge is always zero.  The last bin edge is
always the (average) box length in the chosen spatial direction (i.e. 1
in relative box coordinates) plus a small tolerance to account for the
right-open bin interval.

To chose an appropriate lag time :math:`\Delta t`, you can first apply
:mod:`state_probs_around_trans` on the output of :mod:`discrete_pos`.

.. todo::
    
    * Allow choice between center of mass and center of geometry.
    * Instead of calculating all contact matrices for all frames at once,
      only calculate the contact matrices in the window
      :math:`t_0 - \Delta t` to :math:`t_0 + \Delta t`.  This is much
      more memory efficient while the computational cost should be the
      same.
    * Same for the discrete position trajectory.
    * Finish docstring.

"""


__author__ = "Andreas Thum"


# Standard libraries
import sys
import os
import argparse
from datetime import datetime, timedelta
# Third party libraries
import psutil
import numpy as np
from scipy import sparse
# Local application/library specific imports
import mdtools as mdt


if __name__ == '__main__':
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            """
Calculate how many selection atoms and compounds stay attached to a
reference compound that changed its position.

Given that at time t0 a reference compound changed its position bin,
compare how many and which selection atoms and compounds are attached to
that reference compound at time t0-dt and t0+dt.  Thereby, distinct
between reference compounds that are at time t0-dt and t0+dt in
different bins or in the same bin.

Additionally, do the same for reference compounds that did not change
their position at time t0.
"""))
    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="Trajectory file.  See supported coordinate formats of"
             " MDAnalysis."
    )
    parser.add_argument(
        '-s',
        dest='TOPFILE',
        type=str,
        required=True,
        help="Topology file.  See supported topology formats of"
             " MDAnalysis."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename pattern.  There will be created two files:"
             "  <OUTFILE_contacts.txt> contains ... TODO;"
             "  <OUTFILE_bins.txt> contains the bin edges used for"
             " defining the position bins."
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
        help="Last frame to read from the trajectory.  This is"
             " exclusive, i.e. the last frame read is actually END-1."
             "  A value of -1 means to read the very last frame."
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
        help="Selection string for the reference group.  See MDAnalysis'"
             " selection syntax for possible choices.  Example:"
             " 'type Li'."
    )
    parser.add_argument(
        '--sel',
        dest='SEL',
        type=str,
        nargs='+',
        required=True,
        help="Selection string for the selection group.  See MDAnalysis'"
             " selection syntax for possible choices.  Example:"
             " 'type OE'."
    )
    parser.add_argument(
        '-c',
        dest='CUTOFF',
        type=float,
        required=True,
        help="Cutoff distance in Angstrom.  A contact between a"
             " reference and selection atom is counted, if their"
             " distance is less than or equal to this cutoff."
    )
    parser.add_argument(
        '--refcmp',
        dest='REFCMP',
        type=str,
        required=False,
        choices=('group', 'segments', 'residues', 'fragments', 'atoms'),
        default='atoms',
        help="Choose the compounds of the reference group whose center"
             " of mass positions should be discretized and whose"
             " contacts to selection atoms and compounds should be"
             " compared before and after a position change.  Reference"
             " compounds can be 'group' (the entire reference group),"
             " 'segments', 'residues', 'fragments' or 'atoms'.  Refer to"
             " the MDAanalysis user guide for an explanation of these"
             " terms.  Compounds are made whole before calculating their"
             " centers of mass.  The centers of mass are wrapped back"
             " into the primary unit cell before discretizing their"
             " positions.  Note that in any case, even if REFCMP is e.g."
             " 'residues', only the atoms belonging to the reference"
             " group are taken into account for contact calculation,"
             " even if the compound might comprise additional atoms that"
             " are not contained in the reference group.  However, the"
             " center of mass calculation is done considering all atoms"
             " of a compound, including those that are not part of the"
             " reference group.  Default: %(default)s"
    )
    parser.add_argument(
        '--selcmp',
        dest='SELCMP',
        type=str,
        required=False,
        choices=('group', 'segments', 'residues', 'fragments'),
        default='residues',
        help="Contacts between reference compounds and selection atoms"
             " are always counted, but also contacts between reference"
             " and selection compounds.  Specify here, which compounds"
             " to use for the selection group.  Note that in any case,"
             " even if SELCMP is e.g. 'residues', only the atoms"
             " belonging to the selection group are taken into account,"
             " even if the compound might comprise additional atoms that"
             " are not contained in the selection group.  Default:"
             " %(default)s"
    )
    parser.add_argument(
        '--lag',
        dest='LAG',
        type=int,
        required=False,
        default=0,
        help="The lag time dt (in trajectory frames).  The coordination"
             " environment of the reference compounds is compared dt"
             " frames before they change their position bin and dt"
             " frames afterwards.  Note that the position change"
             " naturally lies between two frames.  Thus, a lag time of"
             " 0 means to compare the frame directly before the position"
             " change with the frame directly afterwards.  Must be equal"
             " to or greather than zero, but not greater than half of"
             " the total number of frames in the trajectory.  LAG must"
             " be an integer multiple of EVERY.  Default: %(default)s"
    )
    parser.add_argument(
        '-d',
        dest='DIRECTION',
        type=str,
        required=False,
        choices=('x', 'y', 'z'),
        default='z',
        help="The spatial direction in which to bin the positions of"
             " the reference compounds.  Default: %(default)s"
    )
    parser.add_argument(
        '--bin-start',
        dest='START',
        type=float,
        required=False,
        default=0,
        help="Point (in Angstrom) on the chosen spatial direction to"
             " start binning.  Note that binning naturally starts at 0"
             " (origin of the simulation box).  If parsing a start value"
             " greater than zero, the first bin interval will be"
             " [0, START).  In this way you can determine the width of"
             " the first bin independently from the other bins.  Note"
             " that START must lie within the simulation box obtained"
             " from the first frame read and it must be smaller than"
             " STOP.  Default: %(default)s"
    )
    parser.add_argument(
        '--bin-end',
        dest='STOP',
        type=float,
        required=False,
        default=None,
        help="Point (in Angstrom) on the chosen spatial direction to"
             " stop binning.  Note that binning naturally ends at"
             " lbox+tol (length of the simulation box in the given"
             " spatial direction plus a small tolerance to account for"
             " the right-open bin interval).  If parsing a value less"
             " than lbox, the last bin interval will be"
             " [STOP, lbox+tol).  In this way you can determine the"
             " width of the last bin independently from the other bins."
             "  Note that STOP must lie within the simulation box"
             " obtained from the first frame read and it must be greater"
             " than START.  Default: lbox+tol"
    )
    parser.add_argument(
        '--bin-num',
        dest='NUM',
        type=int,
        required=False,
        default=10,
        help="Number of equidistant bins (not bin edges!) to use for"
             " discretizing the given spatial direction between START"
             " and STOP.  Note that two additional bins, [0, START) and"
             " [STOP, lbox+tol), are created if START is not zero and"
             " STOP is not lbox.  Default: %(default)s"
    )
    parser.add_argument(
        '--bins',
        dest='BINFILE',
        type=str,
        required=False,
        default=None,
        help="Text file containing custom bin edges (in Angstrom).  Bin"
             " edges are read from the first column, characters"
             " following a '#' are ignored.  Bins do not need to be"
             " equidistant.  All bin edges must lie within the"
             " simulation box as obtained from the first frame read."
             "  If --bins is given, it takes precedence over all other"
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
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))
    print("\n")
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=u.trajectory.n_frames
    )
    print("\n")
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
    if args.DEBUG:
        print("\n")
        mdt.check.time_step(trj=u.trajectory[BEGIN:END])

    print("\n")
    print("Creating/checking bins...")
    timer = datetime.now()
    lbox = u.trajectory[BEGIN].dimensions[ixd]
    if lbox <= 0:
        raise ValueError("Invalid simulation box: The box length ({}) in"
                         " the given spatial direction ({}) is less than"
                         " or equal to zero".format(lbox, args.DIRECTION))
    if args.BINFILE is None:
        if args.STOP is None:
            STOP = lbox
        else:
            STOP = args.STOP
        START, STOP, STEP, NUM = mdt.check.bins(start=args.START / lbox,
                                                stop=STOP / lbox,
                                                num=args.NUM,
                                                amin=0,
                                                amax=1)
        bins = np.linspace(START, STOP, NUM + 1)
    else:
        bins = np.loadtxt(args.BINFILE, usecols=0)
        bins = np.unique(bins) / lbox
    mdt.check.bin_edges(bins=bins, amin=0, amax=1, tol=1e-6)
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))

    print("\n")
    print("Step 1/3:")
    # Creating contact matrices...
    cms = mdt.strc.contact_matrices(topfile=args.TOPFILE,
                                    trjfile=args.TRJFILE,
                                    ref=' '.join(args.REF),
                                    sel=' '.join(args.SEL),
                                    cutoff=args.CUTOFF,
                                    begin=BEGIN,
                                    end=END,
                                    every=EVERY,
                                    compound=(args.REFCMP, 'atoms'),
                                    mdabackend=mdabackend,
                                    verbose=True)
    # TODO
    print()
    print("cms[0].shape =", cms[0].shape)

    print("\n")
    print("Step 2/3:")
    # Creating discrete position trajectory...
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
    # Masses are already checked within :func:`mdt.strc.discrete_pos_trj`
    # if args.REFCMP != 'atoms':
    #     print("\n")
    #     mdt.check.masses_new(ag=refcmp, verbose=True)
    dtrj, bins = mdt.strc.discrete_pos_trj(topfile=args.TOPFILE,
                                           trjfile=args.TRJFILE,
                                           sel="group refcmp_ag",
                                           refcmp_ag=refcmp,
                                           begin=BEGIN,
                                           end=END,
                                           every=EVERY,
                                           compound=args.REFCMP,
                                           direction=args.DIRECTION,
                                           bins=bins * lbox,
                                           return_bins=True,
                                           dtype=np.uint32,
                                           verbose=True,
                                           debug=args.DEBUG)
    dtrj = np.asarray(dtrj.T, order='C')

    print("\n")
    print("Step 3/3:")
    print("Comparing coordination environments...")
    print("Total number of frames: {:>8d}".format(u.trajectory.n_frames))
    print("Frames to read:         {:>8d}".format(N_FRAMES))
    print("First frame to read:    {:>8d}".format(BEGIN))
    print("Last frame to read:     {:>8d}".format(END - 1))
    print("Read every n-th frame:  {:>8d}".format(EVERY))
    print("Time first frame:       {:>12.3f} (ps)"
          .format(u.trajectory[BEGIN].time))
    print("Time last frame:        {:>12.3f} (ps)"
          .format(u.trajectory[END - 1].time))
    print("Time step first frame:  {:>12.3f} (ps)"
          .format(u.trajectory[BEGIN].dt))
    print("Time step last frame:   {:>12.3f} (ps)"
          .format(u.trajectory[END - 1].dt))
    timer = datetime.now()
    # refcmps can
    #   0. stay in their bin and be at t0-dt in the same bin as at t0+dt
    #   1. stay in their bin and be at t0-dt in another bin than at t0+dt
    #   2. leave their bin and be at t0-dt in the same bin as at t0+dt
    #   3. leave their bin and be at t0-dt in another bin than at t0+dt
    valid_refcmps = np.zeros((4, N_REFCMPS), dtype=bool)
    valid_refcmps_tmp = np.zeros(N_REFCMPS, dtype=bool)
    # Total number of valid refcmps
    n_refcmps_tot = np.zeros(len(valid_refcmps), dtype=np.uint32)
    # Number of refcmps bound to selatms before position change (only
    # needed to normalize selix_var_min_max_b)
    n_refcmps_bound_b = np.zeros_like(n_refcmps_tot)
    # Number of refcmps bound to selatms after position change (only
    # needed to normalize selix_var_min_max_a)
    n_refcmps_bound_a = np.zeros_like(n_refcmps_bound_b)
    # Number of refcmps bound to selatms before and after position
    # change (only needed to normalize selix_diff)
    n_refcmps_bound_ba = np.zeros_like(n_refcmps_bound_b)
    # For contancts between refcmps and
    #  0. selatms,
    #  1. selcmps,
    # [^ -> Indices for 1st dimension]
    # compute the...
    # ...number of contacts between refcmps and selatms/selcmps
    #  0. before position change
    #  1. after position change
    #  2. common/remaining contacts
    # [^ -> Indices for 3rd dimension]
    n_contacts = np.zeros((2, len(valid_refcmps), 3), dtype=np.uint32)
    # ...average
    #  0. variance
    #  1. minimum
    #  2. maximum
    # [^ -> Indices for 3rd dimension]
    # of the indices of selatms/selcmps that are bound to refcmps
    selix_var_min_max_b = np.zeros_like(n_contacts, dtype=np.float64)
    selix_var_min_max_a = np.zeros_like(selix_var_min_max_b)
    # ...difference (before and after the position change) of the
    #  0. average
    #  1. minimum
    #  2. maximum
    # [^ -> Indices for 3rd dimension]
    # of the indices of selatms/selcmps that are bound to refcmps
    selix_diff = np.zeros_like(selix_var_min_max_b)
    # For creation of the refcmp-selcmp contact matrices from the
    # recmp-selatm contact matrices
    natms_per_selcmp = mdt.strc.natms_per_cmp(ag=sel,
                                              compound=args.SELCMP,
                                              check_contiguos=True)
    # Read trajectory:
    trj = u.trajectory[BEGIN + EVERY + LAG:END - LAG:EVERY]
    trj = mdt.rti.ProgressBar(trj,
                              initial=(BEGIN + EVERY + LAG) // EVERY,
                              total=(END - LAG) // EVERY)
    for i, ts in enumerate(trj, start=1 + LAG_EFF):
        # Find refcmps that stayed in their position bin or left it and
        # that are at time t0-dt in the same bin as at time t0+dt or not:
        np.equal(dtrj[i - 1], dtrj[i], out=valid_refcmps[:2])
        np.invert(valid_refcmps[0], out=valid_refcmps[2:])
        np.equal(dtrj[i - 1 - LAG_EFF], dtrj[i + LAG_EFF], out=valid_refcmps_tmp)
        valid_refcmps[[0, 2]] &= valid_refcmps_tmp
        np.invert(valid_refcmps_tmp, out=valid_refcmps_tmp)
        valid_refcmps[[1, 3]] &= valid_refcmps_tmp
        if args.DEBUG and np.count_nonzero(valid_refcmps) != N_REFCMPS:
            raise ValueError("'np.count_nonzero(valid_refcmps)' ({}) !="
                             " 'N_REFCMPS' ({})"
                             .format(np.count_nonzero(valid_refcmps),
                                     N_REFCMPS))
        # Compare refcmp-selatm coordination environments:
        cm_b = cms[i - 1 - LAG_EFF]  # Contact matrix at t0-dt (b = "before")
        cm_a = cms[i + LAG_EFF]    # Contact matrix at t0+dt (a = "after")
        for s in range(2):  # s=0 -> selatm,  s=1 -> selcmp
            for v, valid in enumerate(valid_refcmps):
                if not np.any(valid):
                    continue
                cm_b_valid = cm_b[valid]
                cm_a_valid = cm_a[valid]
                if s == 1:
                    cm_b_valid = mdt.strc.cmp_contact_matrix(
                        cm=cm_b_valid.toarray(),
                        natms_per_selcmp=natms_per_selcmp)
                    cm_b_valid = sparse.csr_matrix(cm_b_valid)
                    cm_a_valid = mdt.strc.cmp_contact_matrix(
                        cm=cm_a_valid.toarray(),
                        natms_per_selcmp=natms_per_selcmp)
                    cm_a_valid = sparse.csr_matrix(cm_a_valid)
                else:
                    # If a refcmp is bound to a selatm, it is also bound
                    # to a selcmp
                    n_refcmps_tot[v] += np.count_nonzero(valid)
                n_contacts[s][v] += mdt.strc.cms_n_contacts(
                    (cm_b_valid, cm_a_valid),
                    dtype=n_contacts.dtype
                )
                # Contact statistics before position change
                selix_stats_b = mdt.strc.cm_selix_stats(cm_b_valid)
                refcmps_bound_b = selix_stats_b[:, 0].astype(bool)
                if s == 0:
                    n_refcmps_bound_b[v] += np.count_nonzero(refcmps_bound_b)
                selix_var_min_max_b[s][v] += np.sum(
                    selix_stats_b[refcmps_bound_b][:, 2:],  # var, min, max
                    axis=0
                )
                # Contact statistics after position change
                selix_stats_a = mdt.strc.cm_selix_stats(cm_a_valid)
                refcmps_bound_a = selix_stats_a[:, 0].astype(bool)
                if s == 0:
                    n_refcmps_bound_a[v] += np.count_nonzero(refcmps_bound_a)
                selix_var_min_max_a[s][v] += np.sum(
                    selix_stats_a[refcmps_bound_a][:, 2:],  # var, min, max
                    axis=0
                )
                # Contact statistics difference
                refcmps_bound_ba = refcmps_bound_b & refcmps_bound_a
                if s == 0:
                    n_refcmps_bound_ba[v] += np.count_nonzero(refcmps_bound_ba)
                selix_diff[s][v] += np.sum(
                    np.abs(
                        selix_stats_b[refcmps_bound_ba][:, [1, 3, 4]] -  # 1=mean,
                        selix_stats_a[refcmps_bound_ba][:, [1, 3, 4]]    # 3=min, 4=max
                    ),
                    axis=0
                )
        # ProgressBar update:
        progress_bar_mem = proc.memory_info().rss / 2**20
        trj.set_postfix_str("{:>7.2f}MiB".format(progress_bar_mem),
                            refresh=False)
    trj.close()
    del cms, dtrj, valid_refcmps, valid_refcmps_tmp
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))

    # Compute averages:
    N_FRAMES_EFF = N_FRAMES - 1 - 2 * LAG_EFF
    n_contacts = n_contacts / n_refcmps_tot[:, None]
    selix_var_min_max_b /= n_refcmps_bound_b[:, None]
    selix_var_min_max_a /= n_refcmps_bound_a[:, None]
    selix_diff /= n_refcmps_bound_ba[:, None]

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    if np.ndim(natms_per_selcmp) == 0:
        N_SELCMPS = sel.n_atoms // natms_per_selcmp
    else:
        N_SELCMPS = len(natms_per_selcmp)
    # Contact information:
    fname = args.OUTFILE + "_contacts.txt"
    mdt.fh.write_header(fname)
    with open(fname, 'a') as outfile:
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
        outfile.write("# Lag time:              {:.3f} ps\n".format(LAG * u.trajectory[BEGIN].dt))
        outfile.write("# Discretized dimension: {}\n".format(args.DIRECTION))
        outfile.write("# No. of frames available for computation: {:>12d}\n".format(N_FRAMES_EFF))
        outfile.write("# No. of reference compounds (refcmps):    {:>12d}\n".format(N_REFCMPS))
        outfile.write("# No. of selection atoms     (selatms):    {:>12d}\n".format(sel.n_atoms))
        outfile.write("# No. of selection compounds (selcmps):    {:>12d}\n".format(N_SELCMPS))
        outfile.write("# \n")
        outfile.write("# \n")
        outfile.write("# The ROWS contain the averages over reference compounds that:\n")
        outfile.write("#   1 stay in their bin and are at time t0-dt in the same bin as   at time t0+dt\n")
        outfile.write("#   2 stay in their bin and are at time t0-dt in another  bin than at time t0+dt\n")
        outfile.write("#   3 left    their bin and are at time t0-dt in the same bin as   at time t0+dt\n")
        outfile.write("#   4 left    their bin and are at time t0-dt in another  bin than at time t0+dt\n")
        row_names = ("stay_and_same", "stay_and_other",
                     "leave_and_same", "leave_and_other")
        outfile.write("# The COLUMNS contain:\n")
        outfile.write("#    1 Total No. of refcmps belonging the respective category\n")
        outfile.write("#    2 Av. No. of selatms/selcmps detached from a refcmp during bin change\n")
        outfile.write("#    3 Av. No. of selatms/selcmps attached to   a refcmp during bin change\n")
        outfile.write("#    4 Av. No. of selatms/selcmps that remain bound to a refcmp during bin change\n")
        outfile.write("#    5 Av. variance of indices              of selatms/selcmps that are bound to refcmps before bin change\n")
        outfile.write("#    6 Av. variance of indices              of selatms/selcmps that are bound to refcmps after  bin change\n")
        outfile.write("#    7 Av. minimum index                    of selatms/selcmps that are bound to refcmps before bin change\n")
        outfile.write("#    8 Av. minimum index                    of selatms/selcmps that are bound to refcmps after  bin change\n")
        outfile.write("#    9 Av. maximum index                    of selatms/selcmps that are bound to refcmps before bin change\n")
        outfile.write("#   10 Av. maximum index                    of selatms/selcmps that are bound to refcmps after  bin change\n")
        outfile.write("#   11 Av. absolute change of the av. index of selatms/selcmps that are bound to refcmps\n")
        outfile.write("#   12 Av. absolute change of the min index of selatms/selcmps that are bound to refcmps\n")
        outfile.write("#   13 Av. absolute change of the max index of selatms/selcmps that are bound to refcmps\n")
        col_names = ("n_refcmps", "n_detached", "n_attached", "n_remain",
                     "ix_var_before", "ix_var_after",
                     "ix_min_before", "ix_min_after",
                     "ix_max_before", "ix_max_after",
                     "ix_av_diff", "ix_min_diff", "ix_max_diff")
        for s in range(2):  # s=0 -> selatm,  s=1 -> selcmp
            # Block headers:
            outfile.write("\n")
            outfile.write("\n")
            if s == 0:
                outfile.write("# refcmp-selatm contacts\n")
            else:
                outfile.write("# refcmp-selatm contacts\n")
            # Column numbers:
            outfile.write("# {:>14d}".format(1))
            for i in range(2, len(col_names) + 1):
                outfile.write(" {:>16d}".format(i))
            outfile.write("\n")
            # Column names:
            outfile.write("# {:>14s}".format(col_names[0]))
            for cn in col_names[1:]:
                outfile.write(" {:>16s}".format(cn))
            outfile.write("\n")
            # Data:
            for v, n_refcmps in enumerate(n_refcmps_tot):
                outfile.write("  {:>14d}".format(n_refcmps))
                for contact in n_contacts[s][v][:2]:
                    outfile.write(" {:>16.9e}".format(contact - n_contacts[s][v][2]))
                outfile.write(" {:>16.9e}".format(n_contacts[s][v][2]))
                for i, vmm in enumerate(selix_var_min_max_b[s][v]):
                    outfile.write(" {:>16.9e}".format(vmm))
                    outfile.write(" {:>16.9e}".format(selix_var_min_max_a[s][v][i]))
                for i, diff in enumerate(selix_diff[s][v]):
                    outfile.write(" {:>16.9e}".format(diff))
                outfile.write("\n")
    print("Created {}".format(fname))
    # Bin edges:
    fname = args.OUTFILE + "_bins.txt"
    lbox_av = np.mean([ts.dimensions[ixd]
                       for ts in u.trajectory[BEGIN:END:EVERY]])
    header = ("Bin edges in Angstrom\n"
              "Number of bin edges:                  {:<d}\n"
              "Number of bins:                       {:<d}\n"
              "Discretized spatial dimension:        {:<s}\n"
              "Average box length in this direction: {:<.9e} A\n"
              .format(len(bins),
                      len(bins) - 1,
                      args.DIRECTION,
                      lbox_av))
    mdt.fh.savetxt(fname=fname, data=bins, header=header)
    print("Created {}".format(fname))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))

    print("\n")
    print("Checking output for consistency...")
    timer = datetime.now()
    if np.sum(n_refcmps_tot) != N_REFCMPS * N_FRAMES_EFF:
        raise ValueError("'np.sum(n_contacts_tot)' ({}) !="
                         " 'N_REFCMPS' * N_FRAMES' ({})"
                         .format(np.sum(n_refcmps_tot),
                                 N_REFCMPS * N_FRAMES))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    print("CPU time:             {}"
          .format(timedelta(seconds=sum(proc.cpu_times()[:4]))))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))
