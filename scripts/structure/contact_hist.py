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


"""
Calculate the number of contacts between two MDAnalysis
:class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`.

.. todo::

    Finish docstring.

Contacts are binned into histograms according to how many different
selection :class:`Atoms <MDAnalysis.core.groups.Atom>`/compounds have
contact with a given reference
:class:`~MDAnalysis.core.groups.Atom`/compound and according to how many
contacts exist between a given reference-selection pair.

Options
-------
-f          Trajectory file.  See |supported_coordinate_formats| of
            MDAnalysis.
-s          Topology file.  See |supported_topology_formats| of
            MDAnalysis.
-o          Output filename.
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
--refcmp    {'segments', 'residues', 'fragments'}

            The compounds of the reference group to use for calculating
            the contact histograms.  Contact histograms are always
            calculated for single atom contacts, but also for contacts
            between entire compounds.  Specify here, which compounds to
            use for the reference group.  Compounds can be 'segments',
            'residues' or 'fragments'.  Refer to the MDAanalysis user
            guide for an |explanation_of_these_terms|.  Note that in any
            case, even if ``REFCMP`` is e.g. 'residues', only the atoms
            belonging to the reference group are taken into account,
            even if the compound might comprise additional atoms that
            are not contained in the reference group.  Default:
            ``'residues'``
--selcmp    {'segments', 'residues', 'fragments'}

            Same for the selection group.  Default: ``'residues'``
--updating-ref
            Use an :class:`~MDAnalysis.core.groups.UpdatingAtomGroup`
            for the reference group.  Selection expressions of
            :class:`UpdatingAtomGroups <MDAnalysis.core.groups.UpdatingAtomGroup>`
            are re-evaluated every
            :attr:`time step <MDAnalysis.coordinates.base.Timestep.dt>`.
            This is e.g. useful for position-based selections like
            'type Li and prop z <= 2.0'.
--updating-sel
            Use an :class:`~MDAnalysis.core.groups.UpdatingAtomGroup`
            for the selection group.

See Also
--------
:mod:`lig_change_at_pos_change_blocks` :
    Compare the coordination environment of reference compounds before
    and after they have changed their position

Notes
-----
TODO

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


def cmp_contact_count_matrices(
        cm, natms_per_refcmp=1, natms_per_selcmp=1, dtype=int):
    """
    Take a contact matrix and return all possible compound contact
    **count** matrices.

    A compound is usually a chemically meaningfull subgroup of an
    :class:`~MDAnalysis.core.groups.AtomGroup`.  This can e.g. be a
    :class:`~MDAnalysis.core.groups.Segment`,
    :class:`~MDAnalysis.core.groups.Residue`,
    :attr:`fragment <MDAnalysis.core.groups.AtomGroup.fragments>` or
    a single :class:`~MDAnalysis.core.groups.Atom`.
    Refer to the MDAnalysis' user guide for an
    |explanation_of_these_terms|.  Note that in any case, only
    :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
    original :class:`~MDAnalysis.core.groups.AtomGroup` are taken into
    account, even if the compound might comprise additional
    :class:`Atoms <MDAnalysis.core.groups.Atom>` that are not contained
    in the original :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    cm : array_like
        (Boolean) contact matrix of shape ``(m, n)`` as e.g. generated
        by :func:`mdtools.structure.contact_matrix`, where ``m`` is the
        number of reference :class:`Atoms <MDAnalysis.core.groups.Atom>`
        and ``n`` is the number of selection
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.
    natms_per_refcmp : int or array_like, optional
        Number of :class:`Atoms <MDAnalysis.core.groups.Atom>` per
        reference compound.  Can be a single integer or an array of
        integers.  If `natms_per_refcmp` is a single integer, all
        reference compounds are assumed to contain the same number of
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.  In this case,
        `natms_per_refcmp` must be an integer divisor of ``cm.shape[0]``.
        If `natms_per_refcmp` is an array of integers, it must contain
        the number of reference
        :class:`Atoms <MDAnalysis.core.groups.Atom>` for each single
        reference compound.  In this case,
        ``numpy.sum(natms_per_refcmp)`` must be equal to ``cm.shape[0]``.
    natms_per_selcmp : int or array_like, optional
        Same for selection compounds (`natms_per_selcmp` is checked
        against ``cm.shape[1]``).
    dtype : dtype, optional
        Data type of the output arrays.

    Returns
    -------
    ccm_refatm_selcmp : numpy.ndarray
        Contact **count** matrix indicating how many contacts exsit
        between a reference :class:`~MDAnalysis.core.groups.Atom` and a
        selection compound.
    ccm_refcmp_selatm : numpy.ndarray
        Same for reference compounds and selection
        :class:`Atoms <MDAnalysis.core.groups.Atom>`.
    ccm_refcmp_selcmp : numpy.ndarray
        Same for reference and selection compounds.

    See Also
    --------
    :func:`mdtools.structure.cmp_contact_count_matrix` :
        Take an :class:`~MDAnalysis.core.groups.Atom` contact matrix and
        sum the contacts of all
        :class:`Atoms <MDAnalysis.core.groups.Atom>` belonging to the
        same compound.
    """
    ccm_refatm_selcmp = mdt.strc.cmp_contact_count_matrix(
        cm=cm,
        natms_per_refcmp=1,
        natms_per_selcmp=natms_per_selcmp,
        dtype=dtype
    )
    ccm_refcmp_selatm = mdt.strc.cmp_contact_count_matrix(
        cm=cm,
        natms_per_refcmp=natms_per_refcmp,
        natms_per_selcmp=1,
        dtype=dtype
    )
    ccm_refcmp_selcmp = mdt.strc.cmp_contact_count_matrix(
        cm=ccm_refcmp_selatm,
        natms_per_refcmp=1,
        natms_per_selcmp=natms_per_selcmp,
        dtype=dtype
    )
    return (ccm_refatm_selcmp, ccm_refcmp_selatm, ccm_refcmp_selcmp)


if __name__ == '__main__':
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the number of contacts between two MDAnalysis"
            " AtomGroups.  For more information, refer to the"
            " documetation of this script."
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
        help="Selection string to select the reference group"
    )
    parser.add_argument(
        '--sel',
        dest='SEL',
        type=str,
        nargs='+',
        required=True,
        help="Selection string to select the selection group."
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
        choices=('segments', 'residues', 'fragments'),
        default='residues',
        help="Reference compound.  Default: %(default)s"
    )
    parser.add_argument(
        '--selcmp',
        dest='SELCMP',
        type=str,
        required=False,
        choices=('segments', 'residues', 'fragments'),
        default='residues',
        help="Selection compound.  Default: %(default)s"
    )
    parser.add_argument(
        '--updating-ref',
        dest='UPDATING_REF',
        required=False,
        default=False,
        action='store_true',
        help="Use an UpdatingAtomGroup for the reference group."
    )
    parser.add_argument(
        '--updating-sel',
        dest='UPDATING_SEL',
        required=False,
        default=False,
        action='store_true',
        help="Use an UpdatingAtomGroup for the selection group."
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    if args.CUTOFF <= 0:
        raise ValueError("-c ({}) must be positive".format(args.CUTOFF))
    if mdt.rti.get_num_CPUs() > 1:
        mdabackend = 'OpenMP'
    else:
        mdabackend = 'serial'

    print("\n")
    u = mdt.select.universe(top=args.TOPFILE, trj=args.TRJFILE)
    print("\n")
    print("Creating selections...")
    timer = datetime.now()
    ref = u.select_atoms(' '.join(args.REF), updating=args.UPDATING_REF)
    sel = u.select_atoms(' '.join(args.SEL), updating=args.UPDATING_SEL)
    if ref.n_atoms == 0 and not args.UPDATING_REF:
        raise ValueError("The reference group contains no atoms")
    if sel.n_atoms == 0 and not args.UPDATING_SEL:
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

    if not args.UPDATING_REF:
        natms_per_refcmp = mdt.strc.natms_per_cmp(ag=ref,
                                                  compound=args.REFCMP,
                                                  check_contiguos=True)
    if not args.UPDATING_SEL:
        natms_per_selcmp = mdt.strc.natms_per_cmp(ag=sel,
                                                  compound=args.SELCMP,
                                                  check_contiguos=True)
    if not args.UPDATING_REF and not args.UPDATING_SEL:
        dist_array_tmp = np.full((ref.n_atoms, sel.n_atoms),
                                 np.nan,
                                 dtype=np.float64)
    else:
        dist_array_tmp = None

    expected_max_contacts = 16  # Will be increased if necessary
    # Histogram for refatm_selatm:
    hist_refatm_selatm = np.zeros(expected_max_contacts, dtype=np.uint32)
    # Histograms for refatm_selcmp, refcmp_selatm, refcmp_selcmp:
    hists_cmp = [np.zeros((4, expected_max_contacts), dtype=np.uint32)
                 for i in range(3)]
    n_atms = np.zeros(2, dtype=int)  # refatms and selatms
    n_cmps = np.zeros_like(n_atms)   # refcmps and selcmps
    n_pairs = np.zeros(len(hists_cmp), dtype=np.uint32)

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
    trj = mdt.rti.ProgressBar(u.trajectory[BEGIN:END:EVERY])
    for ts in trj:
        if args.UPDATING_REF:
            natms_per_refcmp = mdt.strc.natms_per_cmp(
                ag=ref,
                compound=args.REFCMP,
                check_contiguos=True
            )
        if args.UPDATING_SEL:
            natms_per_selcmp = mdt.strc.natms_per_cmp(
                ag=sel,
                compound=args.SELCMP,
                check_contiguos=True
            )
        # Generate contact matrices:
        cm_refatm_selatm = mdt.strc.contact_matrix(
            ref=ref,
            sel=sel,
            cutoff=args.CUTOFF,
            box=ts.dimensions,
            result=dist_array_tmp,
            mdabackend=mdabackend,
        )
        n_atms += cm_refatm_selatm.shape
        cmp_ccms = cmp_contact_count_matrices(
            cm=cm_refatm_selatm,
            natms_per_refcmp=natms_per_refcmp,
            natms_per_selcmp=natms_per_selcmp,
            dtype=np.uint32
        )
        n_cmps += cmp_ccms[-1].shape
        if cm_refatm_selatm.shape[0] == 0:
            # Number of reference atoms (and compounds) is zero
            continue
        # Pure atom-atom contact histogram (refatm_selatm):
        hists_tmp = mdt.strc.contact_hist_refcmp_diff_selcmp(
            cm=cm_refatm_selatm,
            minlength=hist_refatm_selatm.shape[-1],
            dtype=np.uint32
        )
        hist_refatm_selatm, hists_tmp = mdt.nph.match_shape(
            hist_refatm_selatm,
            hists_tmp
        )
        hist_refatm_selatm += hists_tmp
        # Compound histograms:
        # (refatm_selcmp, refcmp_selatm, refcmp_selcmp)
        for i, cmp_ccm in enumerate(cmp_ccms):
            hists_tmp = np.asarray(mdt.strc.contact_hists(
                cm=cmp_ccm,
                minlength=hists_cmp[i].shape[-1],
                dtype=np.uint32
            ))
            hists_cmp[i], hists_tmp = mdt.nph.match_shape(hists_cmp[i],
                                                          hists_tmp)
            n_pairs[i] += np.sum(hists_tmp[-1])
            hists_cmp[i] += hists_tmp
        # ProgressBar update:
        progress_bar_mem = proc.memory_info().rss / 2**20
        trj.set_postfix_str("{:>7.2f}MiB".format(progress_bar_mem),
                            refresh=False)
    trj.close()
    del cm_refatm_selatm, cmp_ccms, dist_array_tmp
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))

    if np.any(n_atms == 0):
        warnings.warn("The total number of reference or selection atoms"
                      " is zero. The output will be meaningless",
                      RuntimeWarning)
    if np.any(n_pairs == 0):
        warnings.warn("At least one of the pair histograms is void."
                      " This histogram will be meaningless",
                      RuntimeWarning)

    # Average contact numbers:
    n_refatms_bound = np.sum(hist_refatm_selatm[1:])
    n_refcmps_bound = np.sum(hists_cmp[1][0][1:])  # refcmp_diff_selatm
    # Averages of pure atom-atom histogram (refatm_selatm):
    avs_refatm_selatm = np.zeros(2, dtype=np.float64)
    tot_contacts = np.sum(hist_refatm_selatm *
                          np.arange(len(hist_refatm_selatm)))
    # normalized by the total number of reference atoms
    avs_refatm_selatm[0] = tot_contacts / n_atms[0]
    # normalized by the number of refatms that are bound to a selatm
    avs_refatm_selatm[1] = tot_contacts / n_refatms_bound
    # Averages of compound histograms:
    # (refatm_selcmp, refcmp_selatm, refcmp_selcmp)
    avs_cmp = np.zeros((len(hists_cmp), len(hists_cmp[0][:-1]), 2),
                       dtype=np.float64)
    for i, hists in enumerate(hists_cmp):
        for j, hist in enumerate(hists[:-1]):
            tot_contacts = np.sum(hist * np.arange(len(hist)))
            if i == 0:  # refatm_selcmp
                avs_cmp[i][j][0] = tot_contacts / n_atms[0]
                avs_cmp[i][j][1] = tot_contacts / n_refatms_bound
            else:  # refcmp_selatm, refcmp_selcmp
                avs_cmp[i][j][0] = tot_contacts / n_cmps[0]
                avs_cmp[i][j][1] = tot_contacts / n_refcmps_bound
    # Pair histograms (refatm_selcmp, refcmp_selatm, refcmp_selcmp):
    avs_pair = np.zeros(len(n_pairs), dtype=np.float64)
    for i, n in enumerate(n_pairs):
        tot_contacts = np.sum(hists_cmp[i][-1] *
                              np.arange(len(hists_cmp[i][-1])))
        avs_pair[i] = tot_contacts / n

    # Normalization:
    hist_refatm_selatm = hist_refatm_selatm / n_atms[0]
    for i in range(len(hists_cmp)):
        hists_cmp[i] = hists_cmp[i].astype(np.float64)
        if i == 0:  # refatm_selcmp
            hists_cmp[i][:-1] /= n_atms[0]
        else:  # refcmp_selatm, refcmp_selcmp
            hists_cmp[i][:-1] /= n_cmps[0]
    for i, n in enumerate(n_pairs):
        hists_cmp[i][-1] /= n

    # Bring all histograms to the same length:
    max_length = len(hist_refatm_selatm)
    for hists in hists_cmp:
        max_length = max(max_length, hists.shape[-1])
    hist_refatm_selatm = mdt.nph.extend(hist_refatm_selatm, max_length)
    for i, hists in enumerate(hists_cmp):
        hists_cmp[i] = mdt.nph.extend(hists, max_length, axis=-1)

    # Find the last non-zero value up to which to write the histograms
    # to file:
    if np.any(hist_refatm_selatm != 0):
        last_nonzero = np.flatnonzero(hist_refatm_selatm)[-1]
    else:
        last_nonzero = 0
    for hists in hists_cmp:
        if np.any(hists != 0):
            last_nonzero = max(last_nonzero,
                               np.max(np.nonzero(hists)[1]))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
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
        outfile.write("# Contact histograms\n")
        outfile.write("# Cutoff (Angstrom): {}\n".format(args.CUTOFF))
        outfile.write("# Reference compound = '{}'\n".format(args.REFCMP))
        outfile.write("# Selection compound = '{}'\n".format(args.SELCMP))
        outfile.write("# Number of frames:                             {:>12d}\n".format(N_FRAMES))
        outfile.write("# Total No. of reference atoms     (per frame): {:>12d}  ({:>12.3f})\n".format(n_atms[0], n_atms[0] / N_FRAMES))
        outfile.write("# Total No. of reference compounds (per frame): {:>12d}  ({:>12.3f})\n".format(n_cmps[0], n_cmps[0] / N_FRAMES))
        outfile.write("# Total No. of selection atoms     (per frame): {:>12d}  ({:>12.3f})\n".format(n_atms[1], n_atms[1] / N_FRAMES))
        outfile.write("# Total No. of selection compounds (per frame): {:>12d}  ({:>12.3f})\n".format(n_cmps[1], n_cmps[1] / N_FRAMES))
        outfile.write("# Total No. of refatm-selcmp pairs (per frame): {:>12d}  ({:>12.3f})\n".format(n_pairs[0], n_pairs[0] / N_FRAMES))
        outfile.write("# Total No. of refcmp-selatm pairs (per frame): {:>12d}  ({:>12.3f})\n".format(n_pairs[1], n_pairs[1] / N_FRAMES))
        outfile.write("# Total No. of refcmp-selcmp pairs (per frame): {:>12d}  ({:>12.3f})\n".format(n_pairs[2], n_pairs[2] / N_FRAMES))
        outfile.write("# \n")
        outfile.write("# \n")
        outfile.write("# Histogram averages:\n")
        outfile.write("# Percentage of refatms bound to at least one selatm:       {:10.4e}\n".format(1 - hist_refatm_selatm[0]))
        outfile.write("# Percentage of refcmps bound to at least one selatm:       {:10.4e}\n".format(1 - hists_cmp[1][0][0]))  # refcmp_diff_selatm
        outfile.write("# \n")
        outfile.write("#  (2)* Av.       selatm coordination No. of all   refatms: {:10.4e}  (Every refatm                  has on average           contact  with this many           selatms)\n".format(avs_refatm_selatm[0]))
        outfile.write("#  (2)' Av.       selatm coordination No. of bound refatms: {:10.4e}  (Every refatm bound to selatms has on average           contact  with this many           selatms)\n".format(avs_refatm_selatm[1]))
        outfile.write("# \n")
        outfile.write("#  (3)* Av.       selcmp coordination No. of all   refatms: {:10.4e}  (Every refatm                  has on average           contact  with this many different selcmps)\n".format(avs_cmp[0][0][0]))
        outfile.write("#  (3)' Av.       selcmp coordination No. of bound refatms: {:10.4e}  (Every refatm bound to selatms has on average           contact  with this many different selcmps)\n".format(avs_cmp[0][0][1]))
        outfile.write("#  (5)* Av. total selcmp coordination No. of all   refatms: {:10.4e}  (Every refatm                  has on average this many contacts with                     selcmps)\n".format(avs_cmp[0][2][0]))
        outfile.write("#  (5)' Av. total selcmp coordination No. of bound refatms: {:10.4e}  (Every refatm bound to selatms has on average this many contacts with                     selcmps)\n".format(avs_cmp[0][2][1]))
        outfile.write("#  (6)° Av. No. of 'bonds' between refatm-selcmp pairs:     {:10.4e}  (Every refatm-selcmp pair      is  on average           connected via this many 'bonds')\n".format(avs_pair[0]))
        outfile.write("# \n")
        outfile.write("#  (7)* Av.       selatm coordination No. of all   refcmps: {:10.4e}  (Every refcmp                  has on average           contact  with this many different selatms)\n".format(avs_cmp[1][0][0]))
        outfile.write("#  (7)' Av.       selatm coordination No. of bound refcmps: {:10.4e}  (Every refcmp bound to selatms has on average           contact  with this many different selatms)\n".format(avs_cmp[1][0][1]))
        outfile.write("#  (9)* Av. total selatm coordination No. of all   refcmps: {:10.4e}  (Every refcmp                  has on average this many contacts with                     selatms)\n".format(avs_cmp[1][2][0]))
        outfile.write("#  (9)' Av. total selatm coordination No. of bound refcmps: {:10.4e}  (Every refcmp bound to selatms has on average this many contacts with                     selatms)\n".format(avs_cmp[1][2][1]))
        outfile.write("# (10)° Av. No. of 'bonds' between refcmp-selatm pairs:     {:10.4e}  (Every refcmp-selatm pair      is  on average           connected via this many 'bonds')\n".format(avs_pair[1]))
        outfile.write("# \n")
        outfile.write("# (11)* Av.       selcmp coordination No. of all   refcmps: {:10.4e}  (Every refcmp                  has on average           contact  with this many different selcmps)\n".format(avs_cmp[2][0][0]))
        outfile.write("# (11)' Av.       selcmp coordination No. of bound refcmps: {:10.4e}  (Every refcmp bound to selatms has on average           contact  with this many different selcmps)\n".format(avs_cmp[2][0][1]))
        outfile.write("# (13)* Av. total selcmp coordination No. of all   refcmps: {:10.4e}  (Every refcmp                  has on average this many contacts with                     selcmps)\n".format(avs_cmp[2][2][0]))
        outfile.write("# (13)' Av. total selcmp coordination No. of bound refcmps: {:10.4e}  (Every refcmp bound to selatms has on average this many contacts with                     selcmps)\n".format(avs_cmp[2][2][1]))
        outfile.write("# (14)° Av. No. of 'bonds' between refcmp-selcmp pairs:     {:10.4e}  (Every refcmp-selcmp pair      is  on average           connected via this many 'bonds')\n".format(avs_pair[2]))
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
        for i in range(last_nonzero + 1):
            outfile.write("  {:3d}   {:16.9e}"
                          .format(i, hist_refatm_selatm[i]))
            for hists in hists_cmp:
                outfile.write(2 * " ")
                for hist in hists:
                    outfile.write(" {:16.9e}".format(hist[i]))
            outfile.write("\n")
        outfile.write("\n")
        outfile.write("\n")
        outfile.write("# Sums\n")
        outfile.write("  {:3d}   {:16.9e}"
                      .format(np.sum(np.arange(last_nonzero + 1)),
                              np.sum(hist_refatm_selatm)))
        for hists in hists_cmp:
            outfile.write(2 * " ")
            for hist in hists:
                outfile.write(" {:16.9e}".format(np.sum(hist)))
        outfile.write("\n")
        outfile.flush()
    print("Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))

    print("\n")
    print("Checking output for consistency...")
    timer = datetime.now()
    tol = 1e-6
    # Check histogram sums:
    if not np.isclose(np.sum(hist_refatm_selatm), 1, rtol=0, atol=tol):
        raise ValueError("The sum over 'hist_refatm_selatm' ({}) is not"
                         " one".format(np.sum(hist_refatm_selatm)))
    for i, hists in enumerate(hists_cmp):
        for j, hist in enumerate(hists):
            if j == 1:  # refcmp_same_selcmp
                if np.sum(hist) < 1:
                    raise ValueError("The sum over 'hists_cmp[{}][{}]'"
                                     " ({}) is less than one"
                                     .format(i, j, np.sum(hist)))
                if np.any(hist) > 1:
                    raise ValueError("At least one element of"
                                     " 'hists_cmp[{}][{}]' is greater"
                                     " than one".format(i, j))
            else:  # refcmp_diff_selcmp, refcmp_selcmp_tot, recmp_selcmp_pair
                if not np.isclose(np.sum(hist), 1, rtol=0, atol=tol):
                    raise ValueError("The sum over 'hists_cmp[{}][{}]'"
                                     " ({}) is not one"
                                     .format(i, j, np.sum(hist)))
    # Check first element of each histogram:
    for i, hists in enumerate(hists_cmp):
        for j, hist in enumerate(hists[:-1]):
            if i == 0:  # refatm_selcmp
                if not np.isclose(hist[0], hist_refatm_selatm[0],
                                  rtol=0, atol=tol):
                    raise ValueError(
                        "The percentage of refatms having no contact"
                        " with any selatm is not the same in"
                        " 'hists_cmp[{}][{}]' ({}) and"
                        " 'hist_refatm_selatm' ({})"
                        .format(i, j, hist[0], hist_refatm_selatm[0]))
            else:  # refcmp_selatm, refcmp_selcmp
                if not np.isclose(hist[0], hists_cmp[1][0][0],
                                  rtol=0, atol=tol):
                    raise ValueError(
                        "The percentage of refcmps having no contact"
                        " with any selatm is not the same in"
                        " 'hists_cmp[{}][{}]' ({}) and"
                        " 'hists_cmp[1][0]' ({})"
                        .format(i, j, hist[0], hists_cmp[1][0][0]))
    for i in range(len(n_pairs)):
        if not np.isclose(hists_cmp[i][-1][0], 0, rtol=0, atol=tol):
            raise ValueError("The first element of"
                             " 'hists_cmp[{}][-1][0]' ({}) is not zero"
                             .format(i, hists_cmp[i][-1][0]))
    # Check if refatm_selcmp_tot == refatm_selatm:
    if not np.allclose(hists_cmp[0][2], hist_refatm_selatm,
                       rtol=0, atol=tol, equal_nan=True):
        raise ValueError("'hists_cmp[0][2]' != 'hist_refatm_selatm'")
    # Check if refcmp_selcmp_tot == refcmp_selatm_tot:
    if not np.allclose(hists_cmp[2][2], hists_cmp[1][2],
                       rtol=0, atol=tol, equal_nan=True):
        raise ValueError("'hists_cmp[2][2]' != 'hists_cmp[1][2]'")
    # Check averages:
    for i, av_refatm_selatm in enumerate(avs_refatm_selatm):
        if not np.isclose(avs_cmp[0][2][i], av_refatm_selatm):
            # average(refatm_selcmp_tot) != average(refatm_selatm)
            raise ValueError("'avs_cmp[0][2][{}]' ({}) !="
                             " 'avs_refatm_selatm[{}]' ({})"
                             .format(i, avs_cmp[0][2][i],
                                     i, av_refatm_selatm))
    for i, av_refcmp_selatm_tot in enumerate(avs_cmp[1][2]):
        if not np.isclose(avs_cmp[2][2][i], av_refcmp_selatm_tot):
            # average(refcmp_selcmp_tot) != average(refcmp_selatm_tot)
            raise ValueError("'avs_cmp[0][2][{}]' ({}) !="
                             " 'avs_cmp[1][2][{}]' ({})"
                             .format(i, avs_cmp[2][2][i],
                                     i, av_refcmp_selatm_tot))
    for i, av_pair in enumerate(avs_pair):
        for k in range(2):
            if not np.isclose(avs_cmp[i][0][k] * av_pair, avs_cmp[i][2][k],
                              rtol=0, atol=tol):
                # av(refcmp_diff_selcmp)*av(refcmp_selcmp_pair) !=
                # av(refcmp_selcmp_tot)
                raise ValueError(
                    "'avs_cmp[{}][0][{}]*avs_pair[{}]' ({}) !="
                    " 'avs_cmp[{}][2][{}]' ({})"
                    .format(i, k, i, avs_cmp[i][0][k] * av_pair,
                            i, k, avs_cmp[i][2][k])
                )
    if np.isclose(hist_refatm_selatm[0], 0, rtol=0, atol=tol):
        # All refatms are bound to selatms
        if not np.isclose(avs_refatm_selatm[0], avs_refatm_selatm[1],
                          rtol=0, atol=tol):
            raise ValueError("All refatms are bound to selatms, but"
                             "'avs_refatm_selatm[0]' ({}) !="
                             " 'avs_refatm_selatm[1]' ({})"
                             .format(avs_refatm_selatm[0],
                                     avs_refatm_selatm[1]))
        for j, av in enumerate(avs_cmp[0][:-1]):  # refatm_selcmp
            if not np.isclose(av[0], av[1], rtol=0, atol=tol):
                raise ValueError("All refatms are bound to selatms, but"
                                 "'avs_cmp[0][{}][0]' ({}) !="
                                 " 'avs_cmp[0][{}][1]' ({})"
                                 .format(j, av[0], j, av[1]))
    if np.isclose(hists_cmp[1][0][0], 0, rtol=0, atol=tol):  # refcmp_diff_selatm
        # All refcmps are bound to selatms
        for i, avs in enumerate(avs_cmp[1:]):  # refcmp_selatm, refcmp_selcmp
            for j, av in enumerate(avs[:-1]):
                if not np.isclose(av[0], av[1], rtol=0, atol=tol):
                    raise ValueError("All refcmps are bound to selatms,"
                                     " but 'avs_cmp[{}][{}][0]' ({}) !="
                                     " 'avs_cmp[{}][{}][1]' ({})"
                                     .format(i, j, av[0], i, j, av[1]))

    hist_refatm_same_selatm = np.array([hist_refatm_selatm[0],
                                        1 - hist_refatm_selatm[0]])
    hist_refatm_same_selatm = mdt.nph.extend(hist_refatm_same_selatm,
                                             len(hists_cmp[1][1]))
    hist_refatm_selatm_pair = np.array([0, 1])
    hist_refatm_selatm_pair = mdt.nph.extend(hist_refatm_selatm_pair,
                                             len(hists_cmp[1][-1]))
    if not args.UPDATING_REF and np.all(np.equal(natms_per_refcmp, 1)):
        # refcmp == refatm
        # Check if refcmp_selatm == refatm_selatm:
        for j in (0, 2):  # refcmp_diff_selatm, refcmp_selatm_tot
            if not np.allclose(hists_cmp[1][j], hist_refatm_selatm,
                               rtol=0, atol=tol, equal_nan=True):
                raise ValueError("refcmp = refatm, but"
                                 " 'hists_cmp[1][{}]' !="
                                 " 'hist_refatm_selatm'".format(j))
        if not np.allclose(hists_cmp[1][1], hist_refatm_same_selatm,
                           rtol=0, atol=tol, equal_nan=True):
            # refcmp_same_selatm != refatm_same_selatm
            raise ValueError("refcmp = refatm, but 'hists_cmp[1][1]' !="
                             " 'hist_refatm_same_selatm'")
        if not np.allclose(hists_cmp[1][-1], hist_refatm_selatm_pair,
                           rtol=0, atol=tol, equal_nan=True):
            # refcmp_selatm_pair != refatm_selatm_pair
            raise ValueError("refcmp = refatm, but 'hists_cmp[1][-1]' !="
                             " 'hist_refatm_selatm_pair'")
        # Check if refcmp_selcmp == refatm_selcmp:
        for j, hist in enumerate(hists_cmp[2]):
            if not np.allclose(hist, hists_cmp[0][j],
                               rtol=0, atol=tol, equal_nan=True):
                raise ValueError("refcmp = refatm, but hists_cmp[2][{}]"
                                 " != hists_cmp[0][{}]".format(j, j))
    if not args.UPDATING_SEL and np.all(np.equal(natms_per_selcmp, 1)):
        # selcmp == selatm
        # Check if refatm_selcmp == refatm_selatm:
        for j in (0, 2):  # refatm_diff_selcmp, refatm_selcmp_tot
            if not np.allclose(hists_cmp[0][j], hist_refatm_selatm,
                               rtol=0, atol=tol, equal_nan=True):
                raise ValueError("selcmp = selatm, but"
                                 " 'hists_cmp[0][{}]' !="
                                 " 'hist_refatm_selatm'".format(j))
        if not np.allclose(hists_cmp[0][1], hist_refatm_same_selatm,
                           rtol=0, atol=tol, equal_nan=True):
            # refatm_same_selcmp != refatm_same_selatm
            raise ValueError("selcmp = selatm, but 'hists_cmp[0][1]' !="
                             " 'hist_refatm_same_selatm'")
        if not np.allclose(hists_cmp[0][-1], hist_refatm_selatm_pair,
                           rtol=0, atol=tol, equal_nan=True):
            # refatm_selcmp_pair != refatm_selatm_pair
            raise ValueError("selcmp = selatm, but 'hists_cmp[0][-1]' !="
                             " 'hist_refatm_selatm_pair'")
        # Check if refcmp_selcmp == refcmp_selatm:
        for j, hist in enumerate(hists_cmp[2]):
            if not np.allclose(hist, hists_cmp[1][j],
                               rtol=0, atol=tol, equal_nan=True):
                raise ValueError("selcmp = selatm, but hists_cmp[2][{}]"
                                 " != hists_cmp[1][{}]".format(j, j))
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
