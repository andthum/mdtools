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
Calculate the number of contacts between two MDAnalysis
:class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>` and thereby
distinct between reference
:class:`Atoms <MDAnalysis.core.groups.Atom>`/compounds that stay at
their initial position or leave it.

.. todo::

    * Account for fluctuating simulation boxes (see :mod:`discrete_pos`).
    * Add \--bin-start and \--bin-stop argument (see :mod:`discrete_pos`).
    * Finish docstring.

Contacts are binned into histograms according to how many different
selection :class:`Atoms <MDAnalysis.core.groups.Atom>`/compounds have
contact with a given reference
:class:`~MDAnalysis.core.groups.Atom`/compound and according to how many
contacts exist between a given reference-selection pair.

See Also
--------
:mod:`contact_hist` :
    Calculate the number of contacts between two MDAnalysis
    :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`

Notes
-----
The **simulation box must be orthogonal**.

The discretization is done with wrapped coordinates and not in real
space.
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
from contact_hist import cmp_contact_count_matrices


if __name__ == '__main__':
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the number of contacts between two groups of"
            " atoms and thereby distinct between reference"
            " atoms/compounds that stay at their initial position or"
            " leave it.  Contacts are binned into histograms according"
            " to how many different selection atoms/compounds have"
            " contact with a given reference atom/compound and according"
            " to how many contacts exist between a given"
            " reference-selection pair."
        )
    )
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
        help="Output filename pattern.  There will be created three"
             " files:"
             "  <OUTFILE_stay.txt> contains the histograms for reference"
             " atoms/compounds that stay in their initial position bin;"
             "  <OUTFILE_leave.txt> contains the histograms for"
             " reference compounds/atoms that leave their initial"
             " position bin;"
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
             " starts at zero.  Note that the first frame cannot be used"
             " for histogram computation, because it is not definable,"
             " whether the reference atoms/compounds stayed in their"
             " initial position bin or whether they have left it."
             "  Default: %(default)s."
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
        choices=('segments', 'residues', 'fragments'),
        default='residues',
        help="Contact histograms are always calculated for single atom"
             " contacts, but also for contacts between entire compounds."
             "  Specify here, which compounds to use for the reference"
             " group.  Compounds can be 'segments', 'residues' or"
             " 'fragments'.  Refer to the MDAanalysis user guide for an"
             " explanation of these terms.  Note that in any case, even"
             " if REFCMP is e.g. 'residues', only the atoms belonging to"
             " the reference group are taken into account for contact"
             " calculation, even if the compound might comprise"
             " additional atoms that are not contained in the reference"
             " group.  However, the center of mass calculation is done"
             " considering all atoms of a compound, including those that"
             " are not part of the reference group.  Default:"
             " %(default)s"
    )
    parser.add_argument(
        '--selcmp',
        dest='SELCMP',
        type=str,
        required=False,
        choices=('segments', 'residues', 'fragments'),
        default='residues',
        help="Compounds to use for the selection group.  Default:"
             " %(default)s"
    )
    parser.add_argument(
        '-d',
        dest='DIRECTION',
        type=str,
        required=False,
        choices=('x', 'y', 'z'),
        default='z',
        help="The spatial direction in which to bin the positions of"
             " the reference atoms and compounds.  Default: %(default)s"
    )
    parser.add_argument(
        '--bin-num',
        dest='NUM',
        type=int,
        required=False,
        default=10,
        help="Number of equidistant bins (not bin edges!) to use for"
             " discretizing the given spatial direction.  The"
             " discretization spans the entire box length in the given"
             " spatial direction as inferred from the first frame read."
             "  Note that the bins do not scale with a potentially"
             " fluctuating simulation box.  Default: %(default)s"
    )
    parser.add_argument(
        '--bins',
        dest='BINFILE',
        type=str,
        required=False,
        default=None,
        help="Text file containing custom bin edges in Angstrom.  Bin"
             " edges are read from the first column, lines starting"
             " with '#' are ignored.  Bins do not need to be"
             " equidistant.  Note that the lower left corner of"
             " simulation boxes is always set to (0, 0, 0) by"
             " MDAnalysis.  --bins takes precedence over --bin-num."
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
    if args.NUM < 1:
        raise ValueError("--bin-num ({}) must be greater than zero"
                         .format(args.NUM))
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
    print("Creating/checking bins...")
    timer = datetime.now()
    lbox = u.trajectory[BEGIN].dimensions[ixd]
    if lbox <= 0:
        raise ValueError("Invalid simulation box: The box length ({}) in"
                         " the given spatial direction ({}) is less than"
                         " or equal to zero".format(lbox, args.DIRECTION))
    tol = 1e-9
    if args.BINFILE is None:
        bins = np.linspace(0, lbox, args.NUM + 1)
    else:
        bins = np.loadtxt(args.BINFILE, usecols=0)
        bins = np.unique(bins)
    if len(bins) == 0:
        raise ValueError("The number of bin edges is zero")
    if bins[0] > 0:
        bins = np.insert(bins, 0, 0)
        print("Prepended new first bin edge: {:.6f}".format(bins[0]))
    if np.isclose(bins[-1], lbox, rtol=0, atol=tol):
        bins[-1] = lbox + tol
        print("Changed last bin edge to {:.6f}".format(bins[-1]))
        print("(Maximum box length plus small tolerance)")
    elif bins[-1] < lbox:
        bins = np.append(bins, lbox + tol)
        print("Appended new last bin edge: {:.6f}".format(bins[-1]))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))

    natms_per_refcmp = mdt.strc.natms_per_cmp(ag=ref,
                                              compound=args.REFCMP,
                                              check_contiguos=True)
    natms_per_selcmp = mdt.strc.natms_per_cmp(ag=sel,
                                              compound=args.SELCMP,
                                              check_contiguos=True)
    dist_array_tmp = np.full((ref.n_atoms, sel.n_atoms),
                             np.nan,
                             dtype=np.float64)
    # refatms/cmps can stay in their bin (index 0) or leav it (index 1)
    refatm_motion = np.zeros((2, ref.n_atoms), dtype=bool)
    if args.REFCMP == 'segments':
        refcmp = ref.segments.atoms
        refcmp_motion = np.zeros((2, ref.n_segments), dtype=bool)
    elif args.REFCMP == 'residues':
        refcmp = ref.residues.atoms
        refcmp_motion = np.zeros((2, ref.n_residues), dtype=bool)
    elif args.REFCMP == 'fragments':
        refcmp = ref.fragments.atoms
        refcmp_motion = np.zeros((2, ref.n_fragments), dtype=bool)
    else:
        raise ValueError("--refcmp must be either 'segments', 'residues'"
                         " or 'fragments', but you gave {}"
                         .format(args.REFCMP))
    print("\n")
    mdt.check.masses_new(ag=refcmp, verbose=True)

    expected_max_contacts = 16  # Will be increased if necessary
    # Histogram for refatm_selatm:
    hist_refatm_selatm = [np.zeros(expected_max_contacts, dtype=np.uint32)
                          for m in range(len(refatm_motion))]
    # Histograms for refatm_selcmp, refcmp_selatm, refcmp_selcmp:
    hists_cmp = [[np.zeros((4, expected_max_contacts), dtype=np.uint32)
                  for i in range(3)]
                 for m in range(len(refcmp_motion))]
    n_atms = np.zeros((len(refatm_motion), 2), dtype=int)
    n_cmps = np.zeros_like(n_atms)
    n_pairs = np.zeros((len(refatm_motion), len(hists_cmp[0])),
                       dtype=np.uint32)

    print("\n")
    print("Reading trajectory...")
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
    # Read first frame:
    ts = u.trajectory[BEGIN]
    refatm_bin_ix_prev = np.digitize(ref.positions[:, ixd], bins=bins)
    refcmp_coms = mdt.strc.com(ag=refcmp,
                               pbc=True,
                               compound=args.REFCMP,
                               make_whole=True,
                               debug=args.DEBUG)
    refcmp_bin_ix_prev = np.digitize(refcmp_coms[:, ixd], bins=bins)
    # Read remaining frames:
    trj = mdt.rti.ProgressBar(u.trajectory[BEGIN + EVERY:END:EVERY],
                              initial=1,
                              total=END)
    for ts in trj:
        # Generate contact matrices:
        cm_refatm_selatm = mdt.strc.contact_matrix(
            ref=ref,
            sel=sel,
            cutoff=args.CUTOFF,
            box=ts.dimensions,
            result=dist_array_tmp,
            mdabackend=mdabackend,
        )
        cmp_ccms = cmp_contact_count_matrices(
            cm=cm_refatm_selatm,
            natms_per_refcmp=natms_per_refcmp,
            natms_per_selcmp=natms_per_selcmp,
            dtype=np.uint32
        )
        # Find refatms that stayed in or left their initial position bin:
        refatm_bin_ix = np.digitize(ref.positions[:, ixd], bins=bins)
        np.equal(refatm_bin_ix, refatm_bin_ix_prev, out=refatm_motion[0])
        np.invert(refatm_motion[0], out=refatm_motion[1])
        refatm_bin_ix_prev = refatm_bin_ix
        # refatm contact histograms:
        for m, valid_refatms in enumerate(refatm_motion):
            n_atms[m] += cm_refatm_selatm[valid_refatms].shape
            # refatm_selatm:
            hists_tmp = mdt.strc.contact_hist_refcmp_diff_selcmp(
                cm=cm_refatm_selatm[valid_refatms],
                minlength=hist_refatm_selatm[m].shape[-1],
                dtype=np.uint32
            )
            hist_refatm_selatm[m], hists_tmp = mdt.nph.match_shape(
                hist_refatm_selatm[m],
                hists_tmp
            )
            hist_refatm_selatm[m] += hists_tmp
            # refatm_selcmp:
            hists_tmp = np.asarray(mdt.strc.contact_hists(
                cm=cmp_ccms[0][valid_refatms],
                minlength=hists_cmp[m][0].shape[-1],
                dtype=np.uint32
            ))
            hists_cmp[m][0], hists_tmp = mdt.nph.match_shape(
                hists_cmp[m][0],
                hists_tmp
            )
            n_pairs[m][0] += np.sum(hists_tmp[-1])
            hists_cmp[m][0] += hists_tmp
        # Find refcmps that stayed in or left their initial position bin:
        refcmp_coms = mdt.strc.com(ag=refcmp,
                                   pbc=True,
                                   compound=args.REFCMP,
                                   make_whole=True,
                                   debug=args.DEBUG)
        refcmp_bin_ix = np.digitize(refcmp_coms[:, ixd], bins=bins)
        np.equal(refcmp_bin_ix, refcmp_bin_ix_prev, out=refcmp_motion[0])
        np.invert(refcmp_motion[0], out=refcmp_motion[1])
        refcmp_bin_ix_prev = refcmp_bin_ix
        # refcmp contact histograms:
        for m, valid_refcmps in enumerate(refcmp_motion):
            n_cmps[m] += cmp_ccms[-1][valid_refcmps].shape
            for i, cmp_ccm in enumerate(cmp_ccms[1:], start=1):
                # refcmp_selatm (j=1) and refcmp_selcmp (j=2):
                hists_tmp = np.asarray(mdt.strc.contact_hists(
                    cm=cmp_ccm[valid_refcmps],
                    minlength=hists_cmp[m][i].shape[-1],
                    dtype=np.uint32
                ))
                hists_cmp[m][i], hists_tmp = mdt.nph.match_shape(
                    hists_cmp[m][i],
                    hists_tmp
                )
                n_pairs[m][i] += np.sum(hists_tmp[-1])
                hists_cmp[m][i] += hists_tmp
        # ProgressBar update:
        progress_bar_mem = proc.memory_info().rss / 2**20
        trj.set_postfix_str("{:>7.2f}MiB".format(progress_bar_mem),
                            refresh=False)
    trj.close()
    del cm_refatm_selatm, cmp_ccms, dist_array_tmp, hists_tmp
    del refatm_motion, refatm_bin_ix, refatm_bin_ix_prev, valid_refatms
    del refcmp_motion, refcmp_bin_ix, refcmp_bin_ix_prev, valid_refcmps
    del refcmp_coms
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))

    if np.any(n_atms[0] == 0):
        warnings.warn("The total number of reference or selection atoms"
                      " that stayed in their initial bin is zero. This"
                      " output will be meaningless", RuntimeWarning)
    if np.any(n_atms[1] == 0):
        warnings.warn("The total number of reference or selection atoms"
                      " that left their initial bin is zero. This output"
                      " will be meaningless", RuntimeWarning)
    if np.any(n_pairs == 0):
        warnings.warn("At least one of the pair histograms is void."
                      " This histogram will be meaningless",
                      RuntimeWarning)

    # Average contact numbers:
    # Averages of pure atom-atom histogram (refatm_selatm):
    avs_refatm_selatm = np.zeros((len(hist_refatm_selatm), 2),
                                 dtype=np.float64)
    for m, hist in enumerate(hist_refatm_selatm):
        n_refatms_bound = np.sum(hist[1:])
        tot_contacts = np.sum(hist * np.arange(len(hist)))
        # normalized by the total number of reference atoms
        avs_refatm_selatm[m][0] = tot_contacts / n_atms[m][0]
        # normalized by the number of refatms that are bound to a selatm
        avs_refatm_selatm[m][1] = tot_contacts / n_refatms_bound
    # Averages of compound histograms:
    # (refatm_selcmp, refcmp_selatm, refcmp_selcmp)
    avs_cmp = np.zeros((len(hists_cmp),
                        len(hists_cmp[0]),
                        len(hists_cmp[0][0][:-1]),
                        2),
                       dtype=np.float64)
    for m, hists_c in enumerate(hists_cmp):
        n_refatms_bound = np.sum(hist_refatm_selatm[m][1:])
        n_refcmps_bound = np.sum(hists_c[1][0][1:])  # refcmp_diff_selatm
        for i, hists in enumerate(hists_c):
            for j, hist in enumerate(hists[:-1]):
                tot_contacts = np.sum(hist * np.arange(len(hist)))
                if i == 0:  # refatm_selcmp
                    avs_cmp[m][i][j][0] = tot_contacts / n_atms[m][0]
                    avs_cmp[m][i][j][1] = tot_contacts / n_refatms_bound
                else:  # refcmp_selatm, refcmp_selcmp
                    avs_cmp[m][i][j][0] = tot_contacts / n_cmps[m][0]
                    avs_cmp[m][i][j][1] = tot_contacts / n_refcmps_bound
    # Pair histograms (refatm_selcmp, refcmp_selatm, refcmp_selcmp):
    avs_pair = np.zeros((len(n_pairs), len(n_pairs[0])),
                        dtype=np.float64)
    for m, n_ps in enumerate(n_pairs):
        for i, n in enumerate(n_ps):
            tot_contacts = np.sum(hists_cmp[m][i][-1] *
                                  np.arange(len(hists_cmp[m][i][-1])))
            avs_pair[m][i] = tot_contacts / n

    # Normalization:
    for m, hist in enumerate(hist_refatm_selatm):
        hist_refatm_selatm[m] = hist / n_atms[m][0]
    for m, hists_c in enumerate(hists_cmp):
        for i in range(len(hists_c)):
            hists_cmp[m][i] = hists_cmp[m][i].astype(np.float64)
            if i == 0:  # refatm_selcmp
                hists_cmp[m][i][:-1] /= n_atms[m][0]
            else:  # refcmp_selatm, refcmp_selcmp
                hists_cmp[m][i][:-1] /= n_cmps[m][0]
    for m, n_ps in enumerate(n_pairs):
        for i, n in enumerate(n_ps):
            hists_cmp[m][i][-1] /= n

    # Bring all histograms to the same length:
    max_length = np.zeros(len(hist_refatm_selatm), dtype=np.uint32)
    for m, hist in enumerate(hist_refatm_selatm):
        max_length[m] = max(max_length[m], len(hist))
    for m, hists_c in enumerate(hists_cmp):
        for hists in hists_c:
            max_length[m] = max(max_length[m], hists.shape[-1])
    for m, hist in enumerate(hist_refatm_selatm):
        hist_refatm_selatm[m] = mdt.nph.extend(hist, max_length[m])
    for m, hists_c in enumerate(hists_cmp):
        for i, hists in enumerate(hists_c):
            hists_cmp[m][i] = mdt.nph.extend(hists,
                                             max_length[m],
                                             axis=-1)

    # Find the last non-zero value up to which to write the histograms
    # to file:
    last_nonzero = np.zeros(len(hist_refatm_selatm), dtype=np.uint32)
    for m, hist in enumerate(hist_refatm_selatm):
        if np.any(hist != 0):
            last_nonzero[m] = max(last_nonzero[m],
                                  np.flatnonzero(hist)[-1])
    for m, hists_c in enumerate(hists_cmp):
        for hists in hists_c:
            if np.any(hists != 0):
                last_nonzero[m] = max(last_nonzero[m],
                                      np.max(np.nonzero(hists)[1]))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    # Save contact histograms:
    for m in range(len(hist_refatm_selatm)):
        if m == 0:
            fname = args.OUTFILE + "_stay.txt"
        elif m == 1:
            fname = args.OUTFILE + "_leave.txt"
        else:
            raise ValueError("'m' > 1. This should not have happened")
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
            outfile.write("# Contact histograms\n")
            outfile.write("# for reference atoms/compounds that")
            if m == 0:
                outfile.write(" STAY in")
            elif m == 1:
                outfile.write(" have LEFT")
            else:
                raise ValueError("'m' > 1. This should not have happened")
            outfile.write(" their initial position bin\n")
            outfile.write("# Discretized dimension: {}\n".format(args.DIRECTION))
            outfile.write("# Cutoff (Angstrom):     {}\n".format(args.CUTOFF))
            outfile.write("# Reference compound =  '{}'\n".format(args.REFCMP))
            outfile.write("# Selection compound =  '{}'\n".format(args.SELCMP))
            outfile.write("# Number of frames available for computation:   {:>12d}\n".format((N_FRAMES - 1)))
            outfile.write("# Total No. of reference atoms     (per frame): {:>12d}  ({:>12.3f})\n".format(n_atms[m][0], n_atms[m][0] / (N_FRAMES - 1)))
            outfile.write("# Total No. of reference compounds (per frame): {:>12d}  ({:>12.3f})\n".format(n_cmps[m][0], n_cmps[m][0] / (N_FRAMES - 1)))
            outfile.write("# Total No. of selection atoms     (per frame): {:>12d}  ({:>12.3f})\n".format(n_atms[m][1], n_atms[m][1] / (N_FRAMES - 1)))
            outfile.write("# Total No. of selection compounds (per frame): {:>12d}  ({:>12.3f})\n".format(n_cmps[m][1], n_cmps[m][1] / (N_FRAMES - 1)))
            outfile.write("# Total No. of refatm-selcmp pairs (per frame): {:>12d}  ({:>12.3f})\n".format(n_pairs[m][0], n_pairs[m][0] / (N_FRAMES - 1)))
            outfile.write("# Total No. of refcmp-selatm pairs (per frame): {:>12d}  ({:>12.3f})\n".format(n_pairs[m][1], n_pairs[m][1] / (N_FRAMES - 1)))
            outfile.write("# Total No. of refcmp-selcmp pairs (per frame): {:>12d}  ({:>12.3f})\n".format(n_pairs[m][2], n_pairs[m][2] / (N_FRAMES - 1)))
            outfile.write("# \n")
            outfile.write("# \n")
            outfile.write("# Histogram averages:\n")
            outfile.write("# Percentage of refatms bound to at least one selatm:       {:10.4e}\n".format(1 - hist_refatm_selatm[m][0]))
            outfile.write("# Percentage of refcmps bound to at least one selatm:       {:10.4e}\n".format(1 - hists_cmp[m][1][0][0]))  # refcmp_diff_selatm
            outfile.write("# \n")
            outfile.write("#  (2)* Av.       selatm coordination No. of all   refatms: {:10.4e}  (Every refatm                  has on average           contact  with this many           selatms)\n".format(avs_refatm_selatm[m][0]))
            outfile.write("#  (2)' Av.       selatm coordination No. of bound refatms: {:10.4e}  (Every refatm bound to selatms has on average           contact  with this many           selatms)\n".format(avs_refatm_selatm[m][1]))
            outfile.write("# \n")
            outfile.write("#  (3)* Av.       selcmp coordination No. of all   refatms: {:10.4e}  (Every refatm                  has on average           contact  with this many different selcmps)\n".format(avs_cmp[m][0][0][0]))
            outfile.write("#  (3)' Av.       selcmp coordination No. of bound refatms: {:10.4e}  (Every refatm bound to selatms has on average           contact  with this many different selcmps)\n".format(avs_cmp[m][0][0][1]))
            outfile.write("#  (5)* Av. total selcmp coordination No. of all   refatms: {:10.4e}  (Every refatm                  has on average this many contacts with                     selcmps)\n".format(avs_cmp[m][0][2][0]))
            outfile.write("#  (5)' Av. total selcmp coordination No. of bound refatms: {:10.4e}  (Every refatm bound to selatms has on average this many contacts with                     selcmps)\n".format(avs_cmp[m][0][2][1]))
            outfile.write("#  (6)° Av. No. of 'bonds' between refatm-selcmp pairs:     {:10.4e}  (Every refatm-selcmp pair      is  on average           connected via this many 'bonds')\n".format(avs_pair[m][0]))
            outfile.write("# \n")
            outfile.write("#  (7)* Av.       selatm coordination No. of all   refcmps: {:10.4e}  (Every refcmp                  has on average           contact  with this many different selatms)\n".format(avs_cmp[m][1][0][0]))
            outfile.write("#  (7)' Av.       selatm coordination No. of bound refcmps: {:10.4e}  (Every refcmp bound to selatms has on average           contact  with this many different selatms)\n".format(avs_cmp[m][1][0][1]))
            outfile.write("#  (9)* Av. total selatm coordination No. of all   refcmps: {:10.4e}  (Every refcmp                  has on average this many contacts with                     selatms)\n".format(avs_cmp[m][1][2][0]))
            outfile.write("#  (9)' Av. total selatm coordination No. of bound refcmps: {:10.4e}  (Every refcmp bound to selatms has on average this many contacts with                     selatms)\n".format(avs_cmp[m][1][2][1]))
            outfile.write("# (10)° Av. No. of 'bonds' between refcmp-selatm pairs:     {:10.4e}  (Every refcmp-selatm pair      is  on average           connected via this many 'bonds')\n".format(avs_pair[m][1]))
            outfile.write("# \n")
            outfile.write("# (11)* Av.       selcmp coordination No. of all   refcmps: {:10.4e}  (Every refcmp                  has on average           contact  with this many different selcmps)\n".format(avs_cmp[m][2][0][0]))
            outfile.write("# (11)' Av.       selcmp coordination No. of bound refcmps: {:10.4e}  (Every refcmp bound to selatms has on average           contact  with this many different selcmps)\n".format(avs_cmp[m][2][0][1]))
            outfile.write("# (13)* Av. total selcmp coordination No. of all   refcmps: {:10.4e}  (Every refcmp                  has on average this many contacts with                     selcmps)\n".format(avs_cmp[m][2][2][0]))
            outfile.write("# (13)' Av. total selcmp coordination No. of bound refcmps: {:10.4e}  (Every refcmp bound to selatms has on average this many contacts with                     selcmps)\n".format(avs_cmp[m][2][2][1]))
            outfile.write("# (14)° Av. No. of 'bonds' between refcmp-selcmp pairs:     {:10.4e}  (Every refcmp-selcmp pair      is  on average           connected via this many 'bonds')\n".format(avs_pair[m][2]))
            outfile.write("# \n")
            outfile.write("# *) Normalized by the total number of       reference atoms/compounds\n")
            outfile.write("# ') Normalized by the       number of bound reference atoms/compounds\n")
            outfile.write("# °) Normalized by the       number of pairs\n")
            outfile.write("# \n")
            outfile.write("# \n")
            outfile.write("# The columns contain:\n")
            outfile.write("#    1 N:                  Number of contacts between the reference and selection group\n")
            outfile.write("#    2 refatm_selatm:      % of refatms that have   contact  with N different selatms (multiple contacts with the same selatm discounted) [refatm_diff_selatm]\n")
            outfile.write("#                       [= % of refatms that have N contacts with             selatms (multiple contacts with the same selatm    counted)  refatm_selatm_tot]\n")
            outfile.write("# \n")
            outfile.write("#    3 refatm_diff_selcmp: % of refatms that have   contact  with N different selcmps (multiple contacts with the same selcmp discounted)\n")
            outfile.write("#    4 refatm_same_selcmp: % of refatms that have N contacts with   the same  selcmp  (multiple contacts with the same selcmp    counted, multiple connections to different selcmps via the same number of 'bonds' discounted)\n")
            outfile.write("#    5 refatm_selcmp_tot:  % of refatms that have N contacts with             selcmps (multiple contacts with the same selcmp    counted)\n")
            outfile.write("#    6 refatm_selcmp_pair: % of refatm-selcmp pairs connected via N 'bonds'           (first element is meaningless)\n")
            outfile.write("# \n")
            outfile.write("#    7 refcmp_diff_selatm: % of refcmps that have   contact  with N different selatms (multiple contacts with the same selatm discounted)\n")
            outfile.write("#    8 refcmp_same_selatm: % of refcmps that have N contacts with   the same  selatm  (multiple contacts with the same selatm    counted, multiple connections to different selatms via the same number of 'bonds' discounted)\n")
            outfile.write("#    9 refcmp_selatm_tot:  % of refcmps that have N contacts with             selatms (multiple contacts with the same selatm    counted)\n")
            outfile.write("#   10 refcmp_selatm_pair: % of refcmp-selatm pairs connected via N 'bonds'           (first element is meaningless)\n")
            outfile.write("# \n")
            outfile.write("#   11 refcmp_diff_selcmp: % of refcmps that have   contact  with N different selcmps (multiple contacts with the same selcmp discounted)\n")
            outfile.write("#   12 refcmp_same_selcmp: % of refcmps that have N contacts with   the same  selcmp  (multiple contacts with the same selcmp    counted, multiple connections to different selcmps via the same number of 'bonds' discounted)\n")
            outfile.write("#   13 refcmp_selcmp_tot:  % of refcmps that have N contacts with             selcmps (multiple contacts with the same selcmp    counted)\n")
            outfile.write("#   14 refcmp_selcmp_pair: % of refcmp-selcmp pairs connected via N 'bonds'           (first element is meaningless)\n")
            outfile.write("# \n")
            outfile.write("# Note that refatm_selcmp_tot (5) = refatm_selatm_tot (1) (= refatm_selatm) and refcmp_selcmp_tot (13) = refcmp_selatm_tot (9)\n")
            outfile.write("# \n")
            outfile.write('# Column number:\n')
            outfile.write("# {:3d}   {:16d}   {:16d} {:16d} {:16d} {:16d}   {:16d} {:16d} {:16d} {:16d}   {:16d} {:16d} {:16d} {:16d}\n"
                          .format(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
            outfile.write("# {:>3s}   {:>16s}   {:>16s} {:>16s} {:>16s} {:>16s}   {:>16s} {:>16s} {:>16s} {:>16s}   {:>16s} {:>16s} {:>16s} {:>16s}\n"
                          .format("N",                 # 01  \vline
                                  "refatm_selatm",     # 02  \vline
                                  "refatm_d_selcmp",   # 03
                                  "refatm_s_selcmp",   # 04
                                  "refatm_selcmp_t",   # 05
                                  "refatm_selcmp_p",   # 06  \vline
                                  "refcmp_d_selatm",   # 07
                                  "refcmp_s_selatm",   # 08
                                  "refcmp_selatm_t",   # 09
                                  "refcmp_selatm_p",   # 10  \vline
                                  "refcmp_d_selcmp",   # 11
                                  "refcmp_s_selcmp",   # 12
                                  "refcmp_selcmp_t",   # 13
                                  "refcmp_selcmp_p"))  # 14
            for i in range(last_nonzero[m] + 1):
                outfile.write("  {:3d}   {:>16.9e}"
                              .format(i, hist_refatm_selatm[m][i]))
                for hists in hists_cmp[m]:
                    outfile.write(2 * " ")
                    for hist in hists:
                        outfile.write(" {:>16.9e}".format(hist[i]))
                outfile.write("\n")
            outfile.write("\n")
            outfile.write("\n")
            outfile.write("# Sums\n")
            outfile.write("  {:3d}   {:>16.9e}"
                          .format(np.sum(np.arange(last_nonzero[m] + 1)),
                                  np.sum(hist_refatm_selatm[m])))
            for hists in hists_cmp[m]:
                outfile.write(2 * " ")
                for hist in hists:
                    outfile.write(" {:>16.9e}".format(np.sum(hist)))
            outfile.write("\n")
            outfile.flush()
        print("Created {}".format(fname))
    # Save bin edges:
    fname = args.OUTFILE + "_bins.txt"
    header = ("Bin edges in Angstrom\n"
              "Discretized dimension: {}\n".format(args.DIRECTION))
    mdt.fh.savetxt(fname=fname, data=bins, header=header)
    print("Created {}".format(fname))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))

    print("\n")
    print("Checking output for consistency...")
    timer = datetime.now()
    tol = 1e-4
    # Check histogram sums:
    for m, hist in enumerate(hist_refatm_selatm):
        if not np.isclose(np.sum(hist), 1, rtol=0, atol=tol):
            raise ValueError("The sum over 'hist_refatm_selatm[{}]' ({})"
                             " is not one".format(m, np.sum(hist)))
    for m, hists_c in enumerate(hists_cmp):
        for i, hists in enumerate(hists_c):
            for j, hist in enumerate(hists):
                if j == 1:  # refcmp_same_selcmp
                    if np.sum(hist) < 1:
                        raise ValueError(
                            "The sum over 'hists_cmp[{}][{}][{}]' ({})"
                            " is less than one"
                            .format(m, i, j, np.sum(hist))
                        )
                    if np.any(hist) > 1:
                        raise ValueError(
                            "At least one element of"
                            " 'hists_cmp[{}][{}][{}]' is greater than"
                            " one".format(m, i, j)
                        )
                else:  # refcmp_diff_selcmp, refcmp_selcmp_tot, recmp_selcmp_pair
                    if not np.isclose(np.sum(hist), 1, rtol=0, atol=tol):
                        raise ValueError(
                            "The sum over 'hists_cmp[{}][{}][{}]' ({})"
                            " is not one".format(m, i, j, np.sum(hist))
                        )
    # Check first element of each histogram:
    for m, hists_c in enumerate(hists_cmp):
        for i, hists in enumerate(hists_c):
            for j, hist in enumerate(hists[:-1]):
                if i == 0:  # refatm_selcmp
                    if not np.isclose(hist[0], hist_refatm_selatm[m][0],
                                      rtol=0, atol=tol):
                        raise ValueError(
                            "The percentage of refatms having no contact"
                            " with any selatm is not the same in"
                            " 'hists_cmp[{}][{}][{}]' ({}) and"
                            " 'hist_refatm_selatm[{}]' ({})"
                            .format(m, i, j, hist[0],
                                    m, hist_refatm_selatm[m][0]))
                else:  # refcmp_selatm, refcmp_selcmp
                    if not np.isclose(hist[0], hists_cmp[m][1][0][0],
                                      rtol=0, atol=tol):
                        raise ValueError(
                            "The percentage of refcmps having no contact"
                            " with any selatm is not the same in"
                            " 'hists_cmp[{}][{}][{}]' ({}) and"
                            " 'hists_cmp[{}][1][0]' ({})"
                            .format(m, i, j, hist[0],
                                    m, hists_cmp[m][1][0][0]))
    for m, n_ps in enumerate(n_pairs):
        for i in range(len(n_ps)):
            if not np.isclose(hists_cmp[m][i][-1][0], 0,
                              rtol=0, atol=tol):
                raise ValueError(
                    "The first element of 'hists_cmp[{}][{}][-1][0]'"
                    " ({}) is not zero"
                    .format(m, i, hists_cmp[m][i][-1][0])
                )
    # Check if refatm_selcmp_tot == refatm_selatm:
    for m, hist in enumerate(hist_refatm_selatm):
        if not np.allclose(hists_cmp[m][0][2], hist,
                           rtol=0, atol=tol, equal_nan=True):
            raise ValueError("'hists_cmp[{}][0][2]' !="
                             " 'hist_refatm_selatm[{}]'".format(m, m))
    # Check if refcmp_selcmp_tot == refcmp_selatm_tot:
    for m, hists_c in enumerate(hists_cmp):
        if not np.allclose(hists_c[2][2], hists_c[1][2],
                           rtol=0, atol=tol, equal_nan=True):
            raise ValueError("'hists_cmp[{}][2][2]' !="
                             " 'hists_cmp[{}][1][2]'".format(m, m))
    # Check averages:
    for m, avs_ra_sa in enumerate(avs_refatm_selatm):
        for i, av_refatm_selatm in enumerate(avs_ra_sa):
            if not np.isclose(avs_cmp[m][0][2][i], av_refatm_selatm):
                # average(refatm_selcmp_tot) != average(refatm_selatm)
                raise ValueError("'avs_cmp[{}][0][2][{}]' ({}) !="
                                 " 'avs_refatm_selatm[{}][{}]' ({})"
                                 .format(m, i, avs_cmp[m][0][2][i],
                                         m, i, av_refatm_selatm))
    for m, avs_c in enumerate(avs_cmp):
        for i, av_refcmp_selatm_tot in enumerate(avs_c[1][2]):
            if not np.isclose(avs_c[2][2][i], av_refcmp_selatm_tot):
                # average(refcmp_selcmp_tot) != average(refcmp_selatm_tot)
                raise ValueError("'avs_cmp[{}][0][2][{}]' ({}) !="
                                 " 'avs_cmp[{}][1][2][{}]' ({})"
                                 .format(m, i, avs_c[2][2][i],
                                         m, i, av_refcmp_selatm_tot))
    for m, avs_p in enumerate(avs_pair):
        for i, av_pair in enumerate(avs_p):
            for k in range(2):
                if not np.isclose(avs_cmp[m][i][0][k] * av_pair,
                                  avs_cmp[m][i][2][k],
                                  rtol=0, atol=tol):
                    # av(refcmp_diff_selcmp)*av(refcmp_selcmp_pair) !=
                    # av(refcmp_selcmp_tot)
                    raise ValueError(
                        "'avs_cmp[{}][{}][0][{}]*avs_pair[{}][{}]' ({}) !="
                        " 'avs_cmp[{}][{}][2][{}]' ({})"
                        .format(m, i, k, m, i, avs_cmp[m][i][0][k] * av_pair,
                                m, i, k, avs_cmp[m][i][2][k])
                    )
    for m, hist_a in enumerate(hist_refatm_selatm):
        if np.isclose(hist_a[0], 0, rtol=0, atol=tol):
            # All refatms are bound to selatms
            if not np.isclose(avs_refatm_selatm[m][0],
                              avs_refatm_selatm[m][1],
                              rtol=0, atol=tol):
                raise ValueError("All refatms are bound to selatms, but"
                                 "'avs_refatm_selatm[{}][0]' ({}) !="
                                 " 'avs_refatm_selatm[{}][1]' ({})"
                                 .format(m, avs_refatm_selatm[0],
                                         m, avs_refatm_selatm[1]))
            for j, av in enumerate(avs_cmp[m][0][:-1]):  # refatm_selcmp
                if not np.isclose(av[0], av[1], rtol=0, atol=tol):
                    raise ValueError("All refatms are bound to selatms,"
                                     " but 'avs_cmp[{}][0][{}][0]' ({})"
                                     " != 'avs_cmp[{}][0][{}][1]' ({})"
                                     .format(m, j, av[0], m, j, av[1]))
    for m, hists_c in enumerate(hists_cmp):
        if np.isclose(hists_c[1][0][0], 0, rtol=0, atol=tol):  # refcmp_diff_selatm
            # All refcmps are bound to selatms
            for i, avs in enumerate(avs_cmp[m][1:]):  # refcmp_selatm, refcmp_selcmp
                for j, av in enumerate(avs[:-1]):
                    if not np.isclose(av[0], av[1], rtol=0, atol=tol):
                        raise ValueError(
                            "All refcmps are bound to selatms, but"
                            " 'avs_cmp[{}][{}][{}][0]' ({}) !="
                            " 'avs_cmp[{}][{}][{}][1]' ({})"
                            .format(m, i, j, av[0], m, i, j, av[1])
                        )

    for m, hist_a in enumerate(hist_refatm_selatm):
        hist_refatm_same_selatm = np.array([hist_a[0], 1 - hist_a[0]])
        hist_refatm_same_selatm = mdt.nph.extend(hist_refatm_same_selatm,
                                                 len(hists_cmp[m][1][1]))
        hist_refatm_selatm_pair = np.array([0, 1])
        hist_refatm_selatm_pair = mdt.nph.extend(hist_refatm_selatm_pair,
                                                 len(hists_cmp[m][1][-1]))
        if np.all(np.equal(natms_per_refcmp, 1)):
            # refcmp == refatm
            # Check if refcmp_selatm == refatm_selatm:
            for j in (0, 2):  # refcmp_diff_selatm, refcmp_selatm_tot
                if not np.allclose(hists_cmp[m][1][j], hist_a,
                                   rtol=0, atol=tol, equal_nan=True):
                    raise ValueError(
                        "refcmp = refatm, but 'hists_cmp[{}][1][{}]' !="
                        " 'hist_refatm_selatm[{}]'".format(m, j, m)
                    )
            if not np.allclose(hists_cmp[m][1][1],
                               hist_refatm_same_selatm,
                               rtol=0, atol=tol, equal_nan=True):
                # refcmp_same_selatm != refatm_same_selatm
                raise ValueError(
                    "refcmp = refatm, but 'hists_cmp[{}][1][1]' !="
                    " 'hist_refatm_same_selatm'".format(m)
                )
            if not np.allclose(hists_cmp[m][1][-1],
                               hist_refatm_selatm_pair,
                               rtol=0, atol=tol, equal_nan=True):
                # refcmp_selatm_pair != refatm_selatm_pair
                raise ValueError(
                    "refcmp = refatm, but 'hists_cmp[{}][1][-1]' !="
                    " 'hist_refatm_selatm_pair'".format(m)
                )
            # Check if refcmp_selcmp == refatm_selcmp:
            for j, hist_c in enumerate(hists_cmp[m][2]):
                if not np.allclose(hist_c, hists_cmp[m][0][j],
                                   rtol=0, atol=tol, equal_nan=True):
                    raise ValueError(
                        "refcmp = refatm, but hists_cmp[{}][2][{}]"
                        " != hists_cmp[{}][0][{}]".format(m, j, m, j)
                    )
        if np.all(np.equal(natms_per_selcmp, 1)):
            # selcmp == selatm
            # Check if refatm_selcmp == refatm_selatm:
            for j in (0, 2):  # refatm_diff_selcmp, refatm_selcmp_tot
                if not np.allclose(hists_cmp[m][0][j], hist_a,
                                   rtol=0, atol=tol, equal_nan=True):
                    raise ValueError(
                        "selcmp = selatm, but 'hists_cmp[{}][0][{}]' !="
                        " 'hist_refatm_selatm[{}]'".format(m, j, m)
                    )
            if not np.allclose(hists_cmp[m][0][1],
                               hist_refatm_same_selatm,
                               rtol=0, atol=tol, equal_nan=True):
                # refatm_same_selcmp != refatm_same_selatm
                raise ValueError(
                    "selcmp = selatm, but 'hists_cmp[{}][0][1]' !="
                    " 'hist_refatm_same_selatm'".format(m)
                )
            if not np.allclose(hists_cmp[m][0][-1],
                               hist_refatm_selatm_pair,
                               rtol=0, atol=tol, equal_nan=True):
                # refatm_selcmp_pair != refatm_selatm_pair
                raise ValueError(
                    "selcmp = selatm, but 'hists_cmp[{}][0][-1]' !="
                    " 'hist_refatm_selatm_pair'".format(m)
                )
            # Check if refcmp_selcmp == refcmp_selatm:
            for j, hist_c in enumerate(hists_cmp[m][2]):
                if not np.allclose(hist_c, hists_cmp[m][1][j],
                                   rtol=0, atol=tol, equal_nan=True):
                    raise ValueError(
                        "selcmp = selatm, but hists_cmp[{}][2][{}]"
                        " != hists_cmp[{}][1][{}]".format(m, j, m, j)
                    )
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
