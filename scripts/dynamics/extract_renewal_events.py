#!/usr/bin/env python3

# This file is part of MDTools.
# Copyright (C) 2021-2023  The MDTools Development Team and all
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


r"""
Extract renewal events from a molecular dynamics trajectory.

A renewal event occurs when the selection compound that was continuously
bound the longest to a reference compound dissociates from it. [#]_

A new trajectory is generated containing only these renewal events for
all reference compounds undergoing such events.  The new trajectory
contains the indices of the reference and selection compound, the time
:math:`t_0` when the selection compound started coordinating to the
reference compound, the renewal time :math:`\tau` needed till
dissociation of the selection compound, the center of mass positions of
the reference and selection compound at :math:`t_0` and the
displacements of the reference and selection compound during
:math:`\tau`.

Options
-------
-f
    Trajectory file.  See |supported_coordinate_formats| of MDAnalysis.
    **Important**: This scripts requires an unwrapped trajectory!  At
    least the reference and selection compounds must be unwrapped in
    order to get the correct displacements.  If you want to calculate
    center-of-mass displacements, the reference and selection compounds
    need to be whole, too.  You can use e.g.
    :mod:`scripts.trajectory.unwrap_trj` to unwrap a wrapped trajectory
    and make broken molecules whole.
-s
    Topology file.  See |supported_topology_formats| of MDAnalysis.
-o
    Output filename.
--dtrj
    Output filename for the discrete coordination trajectory (optional).
    If provided, generate a discrete trajectory containing for each
    single reference compound for each frame the index of the selection
    compound that is continuously bound the longest to the reference
    compound.  Indexing starts at zero.  An index of -1 indicates that
    in the given frame no selection compound was bound to the given
    reference compound.  The discrete trajectory is stored as
    :class:`numpy.ndarray` of dtype :attr:`numpy.int32` and shape
    ``(n, f)``, where ``n`` is the number of reference compounds and
    ``f`` is the number of frames.  The generated discrete trajectory is
    the same as the one created by
    :mod:`scripts.discretization.discrete_coord`.
-b
    First frame to read from the trajectory.  Frame numbering starts at
    zero.  Default: ``0``.
-e
    Last frame to read from the trajectory.  This is exclusive, i.e. the
    last frame read is actually ``END - 1``.  A value of ``-1`` means to
    read the very last frame.  Default: ``-1``.
--every
    Read every n-th frame from the trajectory.  Default: ``1``.
--ref
    Selection string for the reference
    :class:`~MDAnalysis.core.groups.AtomGroup`.  See MDAnalysis'
    |selection_syntax| for possible choices.
--sel
    Selection string for the selection
    :class:`~MDAnalysis.core.groups.AtomGroup`.  See MDAnalysis'
    |selection_syntax| for possible choices.
-c
    Contact cutoff distance in Angstrom.  A reference and selection atom
    are considered to be in contact, if their distance is less than or
    equal to the given cutoff.
--compound
    {'atoms', 'segments', 'residues', 'fragments'}

    The compounds of the reference and selection group to use for the
    analysis.  Contacts between the reference and selection group can be
    computed either for individual 'atoms', 'segments', 'residues', or
    'fragments'.  Refer to the MDAnalysis' user guide for an
    |explanation_of_these_terms|.  This option also affects how
    displacements are calculated.  If not set to 'atoms', the center of
    mass of the compound is used to calculate its displacement.  Note
    that in any case, even if ``CMP`` is e.g. 'residues', only the atoms
    belonging to the reference/selection group are taken into account,
    even if the compound might comprise additional atoms that are not
    contained in the reference/selection group.  Default: ``'atoms'``.
--min-contacts
    Compounds of the reference and selection group are only considered
    to be in contact, if there are at least ``MINCONTACTS`` contacts
    between the atoms of the compounds.  \--min-contacts is ignored if
    \--compound is set to ``'atoms'``.  Default: ``1``.
--intermittency
    Maximum number of frames a selection atom is allowed to leave the
    cutoff range of a reference atom whilst still being considered to be
    bound to the reference atom, provided that it is indeed bound again
    to the reference atom after this unbound period.  The other way
    round, a selection atom is only considered to be bound to a
    reference atom, if it has been bound to it for at least this number
    of consecutive frames.  Default: ``0``.
--debug
    Run in :ref:`debug mode <debug-mode-label>`.


See Also
--------
:mod:`scripts.discretization.discrete_coord` :
    Generate a trajectory containing for each single reference compound
    for each frame the index of the selection compound that is
    continuously bound the longest to the reference compound.

References
----------
.. [#] A. Maitra, A. Heuer,
    `Cation Transport in Polymer Electrolytes: A Microscopic Approach
    <https://doi.org/10.1103/PhysRevLett.98.227802>`_,
    Physical Review Letters, 2007, 98, 227802.
"""


__author__ = "Andreas Thum"


# Standard libraries
import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

# Third-party libraries
import numpy as np
import psutil

# First-party libraries
import mdtools as mdt

# Local imports
from msd_at_coord_change import get_pos


def extract_renewal_events(  # noqa: C901
    universe,
    ref,
    sel,
    cms,
    compound="atoms",
    begin=0,
    every=1,
    coord_traj=False,
    verbose=False,
    debug=False,
):
    r"""
    Extract renewal events from an
    :attr:`~MDAnalysis.core.universe.Universe.trajectory`.

    A renewal event occurs when the selection compound that was
    continuously bound the longest to a reference compound dissociates
    from it. [#]_ Such an event is called "trackable", if its start time
    :math:`t_0` and its renewal time :math:`\tau_{renew}` are exactly
    known.  This means, the first and last event of each reference
    compound must be discarded, because :math:`t_0` of the first event
    and :math:`\tau_{renew}` of the last event are unknown due to the
    limited trajectory length.

    Parameters
    ----------
    universe : MDAnalysis.core.universe.Universe
        The MDAnalysis :class:`~MDAnalysis.core.universe.Universe` which
        holds the unwrapped and whole reference compounds.
    ref, sel : MDAnalysis.core.groups.AtomGroup
        The reference and selection
        :class:`~MDAnalysis.core.groups.AtomGroup`.
    cms : array_like
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be given as
        :mod:`SciPy sparse matrices <scipy.sparse>`.  The rows must
        stand for the reference compounds, the columns for the selection
        compounds.
    compound : {'atoms', 'segments', 'residues', 'fragments'}, optional
        For which type of components the contact matrices in `cms` were
        calculated.  Refer to the MDAnalysis' user guide for an
        |explanation_of_these_terms|.
    begin : int, optional
        The frame number of the
        :attr:`~MDAnalysis.core.universe.Universe.trajectory` to which
        the first contact matrix in `cms` corresponds to.
    every : int, optional
        To how many frames of the
        :attr:`~MDAnalysis.core.universe.Universe.trajectory` one frame
        in `cms` corresponds to.
    coord_traj : bool, optional
        Generate a trajectory containing for each single reference
        compound for each frame the index of the selection compound that
        is continuously bound the longest to the reference compound.
        See also :func:`discrete_coord` from discrete_coord.py
    verbose : bool, optional
        If ``True``, print progress information to standard output.
    debug : bool, optional
        If ``True``, check the input arguments.

    Returns
    -------
    refix_t0 : numpy.ndarray
        The indices of the reference compounds which underwent trackable
        renewal events.  Indexing starts at zero.
    selix_t0 : numpy.ndarray
        The corresponding indices of the selection compounds that were
        continuously bound the longest to the reference compounds.
        Indexing starts at zero.
    t0 : numpy.ndarray
        The corresponding times when the selection compounds started
        coordinating to the reference compounds.
    trenew : numpy.ndarray
        The corresponding renewal times needed till dissociation of the
        selection compounds.
    refpos_t0 : numpy.ndarray
        The corresponding center of mass positions of the reference
        compounds at time `t0`.
    selpos_t0 : numpy.ndarray
        The corresponding center of mass positions of the selection
        compounds at time `t0`.
    refdispl : numpy.ndarray
        The corresponding displacements of the reference compounds
        during `trenew`.
    seldispl : numpy.ndarray
        The corresponding displacements of the selection compounds
        during `trenew`.
    traj : numpy.ndarray
        Only returned if `coord_traj` is ``True``.  Array of shape
        ``(n, len(cms))``, where ``n`` is the number of reference
        compounds.  The elements of the array are the indices of the
        selection compounds that are continuously bound the longest to
        the given reference compound.  Indexing starts at zero.  An
        index of -1 indicates that in the given frame no selection
        compound was bound to the given reference compound.

    Notes
    -----
    The output is sorted by `refix_t0` as primary sort order, `t0`
    as secondary sort order, `trenew` as tertiary sort order and
    `selix_t0` as quaternary sort order.

    References
    ----------
    .. [#] A. Maitra, A. Heuer,
        `Cation Transport in Polymer Electrolytes: A Microscopic
        Approach <https://doi.org/10.1103/PhysRevLett.98.227802>`_,
        Physical Review Letters, 2007, 98, 227802.
    """
    if compound not in ("atoms", "segments", "residues", "fragments"):
        raise ValueError(
            "`compound` must be either 'atoms', 'segments', 'residues' or"
            " 'fragments', but you gave '{}'".format(compound)
        )
    if (
        (compound == "atoms" and ref.n_atoms != cms[0].shape[0])
        or (compound == "segments" and ref.n_segments != cms[0].shape[0])
        or (compound == "residues" and ref.n_residues != cms[0].shape[0])
        or (compound == "fragments" and ref.n_fragments != cms[0].shape[0])
    ):
        raise ValueError(
            "The number of reference compounds does not fit to the number"
            " of rows of the contact matrices"
        )
    if (
        (compound == "atoms" and sel.n_atoms != cms[0].shape[1])
        or (compound == "segments" and sel.n_segments != cms[0].shape[1])
        or (compound == "residues" and sel.n_residues != cms[0].shape[1])
        or (compound == "fragments" and sel.n_fragments != cms[0].shape[1])
    ):
        raise ValueError(
            "The number of selection compounds does not fit to the number"
            " of columns of the contact matrices"
        )
    if begin < 0:
        raise ValueError("`begin` must not be negative")
    if every <= 0:
        raise ValueError("`every` must be positive")
    if debug:
        for i, cm in enumerate(cms):
            if cm.ndim != 2:
                warnings.warn(
                    "`cms` seems not to be a list of contact matrices, because"
                    " its {}-th element has not 2 dimensions".format(i),
                    RuntimeWarning,
                    stacklevel=2,
                )
            if cm.shape != cms[0].shape:
                raise ValueError(
                    "All arrays in `cms` must have the same shape"
                )
            if not isinstance(cm, type(cms[0])):
                raise TypeError("All arrays in `cms` must be of the same type")
        mdt.check.time_step(trj=universe.trajectory[begin : len(cms) * every])

    refix2refix_t0 = -np.ones(cms[0].shape[0], dtype=np.int32)
    # If refix2refix_t0[rix] < 0, reference compound rix was not bound
    # to any selection compound
    refix_t0 = []
    selix_t0 = []
    t0 = []
    trenew = []
    refpos_t0 = []
    selpos_t0 = []
    refdispl = []
    seldispl = []

    refpos = get_pos(
        universe=universe,
        atm_grp=ref,
        frame=begin,
        compound=compound,
        debug=debug,
    )
    selpos = get_pos(
        universe=universe,
        atm_grp=sel,
        frame=begin,
        compound=compound,
        debug=debug,
    )
    refix, selix = cms[0].nonzero()
    refix_unique, selix = mdt.nph.group_by(
        keys=refix.astype(np.uint32),
        values=selix.astype(np.uint32),
        assume_sorted=True,
        return_keys=True,
    )
    refix_t0.extend(refix_unique)
    refix2refix_t0[refix_unique] = np.arange(len(refix_t0))
    selix_t0.extend(selix)
    t0.extend(np.uint32(begin) for i in refix_t0)
    trenew.extend(np.int32(-1) for i in refix_t0)
    refpos_t0.extend(refpos[refix_unique])
    selpos_t0.extend(selpos[six] for six in selix)
    refdispl.extend(np.full(3, np.nan, dtype=np.float32) for i in refix_t0)
    seldispl.extend(np.full(3, np.nan, dtype=np.float32) for i in refix_t0)

    if verbose:
        proc = psutil.Process()
        contact_matrices = mdt.rti.ProgressBar(
            cms[1:], initial=1, total=len(cms)
        )
    else:
        contact_matrices = cms[1:]
    for i, cm in enumerate(contact_matrices, start=1):
        frame = np.uint32(begin + i * every)
        refpos = get_pos(
            universe=universe,
            atm_grp=ref,
            frame=frame,
            compound=compound,
            debug=debug,
        )
        selpos = get_pos(
            universe=universe,
            atm_grp=sel,
            frame=frame,
            compound=compound,
            debug=debug,
        )
        bound_now_and_before = cm.multiply(cms[i - 1])

        attached = cm - bound_now_and_before
        refix, selix = attached.nonzero()
        if len(refix) > 0 and np.any(refix2refix_t0[refix] < 0):
            refix_unique, selix = mdt.nph.group_by(
                keys=refix.astype(np.uint32),
                values=selix.astype(np.uint32),
                assume_sorted=True,
                return_keys=True,
            )
            for j, rix in enumerate(refix_unique):
                if refix2refix_t0[rix] >= 0:
                    # Reference compound rix is already bound to a
                    # selection compound
                    continue
                refix2refix_t0[rix] = len(refix_t0)
                refix_t0.append(rix)
                selix_t0.append(selix[j])
                t0.append(frame)
                trenew.append(np.int32(-1))
                refpos_t0.append(refpos[rix])
                selpos_t0.append(selpos[selix[j]])
                refdispl.append(np.full(3, np.nan, dtype=np.float32))
                seldispl.append(np.full(3, np.nan, dtype=np.float32))

        detached = cms[i - 1] - bound_now_and_before
        refix, selix = detached.nonzero()
        if len(refix) > 0:
            refix_unique, selix = mdt.nph.group_by(
                keys=refix.astype(np.uint32),
                values=selix.astype(np.uint32),
                assume_sorted=True,
                return_keys=True,
            )
            for j, rix in enumerate(refix_unique):
                rix_t0 = refix2refix_t0[rix]
                remain = np.isin(
                    selix_t0[rix_t0],
                    test_elements=selix[j],
                    assume_unique=True,
                    invert=True,
                )
                if not np.any(remain):
                    # Last selection compound(s) that was (were) bound
                    # to reference compound rix at time t0 got detached.
                    selix_t0[rix_t0] = selix_t0[rix_t0][0]
                    selpos_t0[rix_t0] = selpos_t0[rix_t0][0]
                    trenew[rix_t0] = frame - t0[rix_t0]
                    np.subtract(
                        refpos[rix], refpos_t0[rix_t0], out=refdispl[rix_t0]
                    )
                    np.subtract(
                        selpos[selix_t0[rix_t0]],
                        selpos_t0[rix_t0],
                        out=seldispl[rix_t0],
                    )
                    selix_bound = cm[rix].nonzero()[1]
                    if len(selix_bound) == 0:
                        refix2refix_t0[rix] = -1
                    else:
                        refix2refix_t0[rix] = len(refix_t0)
                        refix_t0.append(rix)
                        selix_t0.append(selix_bound)
                        t0.append(frame)
                        trenew.append(np.int32(-1))
                        refpos_t0.append(refpos[rix])
                        selpos_t0.append(selpos[selix_bound])
                        refdispl.append(np.full(3, np.nan, dtype=np.float32))
                        seldispl.append(np.full(3, np.nan, dtype=np.float32))
                else:
                    selix_t0[rix_t0] = selix_t0[rix_t0][remain]
                    selpos_t0[rix_t0] = selpos_t0[rix_t0][remain]
        if verbose:
            contact_matrices.set_postfix_str(
                "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
            )
    contact_matrices.close()
    del refix, selix, refix_unique, refix2refix_t0, refpos, selpos
    del cm, bound_now_and_before, attached, detached

    if len(selix_t0) != len(refix_t0):
        raise ValueError(
            "The number of selection indices does not match the number of"
            " reference indices.  This should not have happened"
        )
    if len(t0) != len(refix_t0):
        raise ValueError(
            "The number of start times does not match the number of reference"
            " indices.  This should not have happened"
        )
    if len(trenew) != len(refix_t0):
        raise ValueError(
            "The number of renewal times does not match the number of"
            " reference indices.  This should not have happened"
        )
    if len(refpos_t0) != len(refix_t0):
        raise ValueError(
            "The number of reference positions does not match the number of"
            " reference indices.  This should not have happened"
        )
    if len(selpos_t0) != len(refix_t0):
        raise ValueError(
            "The number of selection positions does not match the number of"
            " reference indices.  This should not have happened"
        )
    if len(refdispl) != len(refix_t0):
        raise ValueError(
            "The number of reference displacements does not match the number"
            " of reference indices.  This should not have happened"
        )
    if len(seldispl) != len(refix_t0):
        raise ValueError(
            "The number of selection displacements does not match the number"
            " of reference indices.  This should not have happened"
        )

    refix_t0 = np.asarray(refix_t0, dtype=np.uint32)
    selix_t0 = np.asarray(selix_t0, dtype=object)
    t0 = np.asarray(t0, dtype=np.uint32)
    trenew = np.asarray(trenew, dtype=np.int32)
    refpos_t0 = np.row_stack(refpos_t0)
    selpos_t0 = np.asarray(selpos_t0, dtype=object)
    refdispl = np.row_stack(refdispl)
    seldispl = np.row_stack(seldispl)

    # Remove the first renewal event for each reference compound since
    # you cannot say what is t0 for these events.  Also remove the last
    # events where a start time t0 is already set, but a renewal event
    # was actually not seen
    valid = (t0 > begin) & (trenew > 0)
    if not np.all(np.isfinite(t0)):
        raise ValueError(
            "t0 contains non-finite values.  This should not have happened"
        )
    if not np.all(np.isfinite(trenew)):
        raise ValueError(
            "trenew contains non-finite values.  This should not have happened"
        )
    if np.any(trenew == 0):
        raise ValueError(
            "At least one renewal time is zero.  This should not have happened"
        )
    if np.any((t0 <= begin) & valid):
        raise ValueError(
            "At least one renewal event was marked valid, although its start"
            " time is unknown.  This should not have happened"
        )
    unfinished = trenew < 0
    if np.any(unfinished & valid):
        raise ValueError(
            "At least one renewal event was marked valid, although a renewal"
            " event was actually not seen.  This should not have happened"
        )
    test_mask = np.asarray([np.ndim(i) for i in selix_t0]) == 1
    if not np.array_equal(unfinished, test_mask):
        raise ValueError(
            "Test 1: The mask and test mask do not match.  This should not"
            " have happened"
        )
    test_mask = np.asarray([np.ndim(i) for i in selpos_t0]) == 2
    if not np.array_equal(unfinished, test_mask):
        raise ValueError(
            "Test 2: The mask and test mask do not match.  This should not"
            " have happened"
        )
    test_mask = np.all(np.isnan(refdispl), axis=1)
    if not np.array_equal(unfinished, test_mask):
        raise ValueError(
            "Test 3: The mask and test mask do not match.  This should not"
            " have happened"
        )
    test_mask = np.all(np.isnan(seldispl), axis=1)
    if not np.array_equal(unfinished, test_mask):
        raise ValueError(
            "Test 4: The mask and test mask do not match.  This should not"
            " have happened"
        )
    del test_mask

    if coord_traj:
        # The following changes are only temporal.  Outside this if
        # statement, the "valid" mask from above applies.
        # Set the renewal time for the last renewal event for each
        # compound, where a start time t0 is already set but a renewal
        # event was actually not seen, to "end of trajectory - t0".
        trenew[unfinished] = begin + len(cms) * every - t0[unfinished]
        for i in np.flatnonzero(unfinished):
            selix_t0[i] = selix_t0[i][0]
        selix_t0 = selix_t0.astype(np.uint32)

        ix_sort = np.lexsort((selix_t0, trenew, t0, refix_t0))
        valid = valid[ix_sort]
        refix_t0 = refix_t0[ix_sort]
        selix_t0 = selix_t0[ix_sort]
        t0 = t0[ix_sort]
        trenew = trenew[ix_sort]
        refpos_t0 = refpos_t0[ix_sort]
        selpos_t0 = selpos_t0[ix_sort]
        refdispl = refdispl[ix_sort]
        seldispl = seldispl[ix_sort]
        del ix_sort

        traj = -np.ones((cms[0].shape[0], len(cms)), dtype=np.int32)
        for i, rix in enumerate(refix_t0):
            start = (t0[i] - begin) // every
            stop = start + (trenew[i] - begin) // every
            traj[rix][start:stop] = selix_t0[i]

    del unfinished
    if not np.any(valid):
        warnings.warn(
            "Could not detect any renewal event", RuntimeWarning, stacklevel=2
        )
        if coord_traj:
            return tuple([np.array([]) for i in range(8)] + [traj])
        else:
            return tuple([np.array([]) for i in range(8)])

    refix_t0 = refix_t0[valid]
    selix_t0 = selix_t0[valid].astype(np.uint32)
    t0 = t0[valid]
    trenew = trenew[valid].astype(np.uint32)
    refpos_t0 = refpos_t0[valid]
    selpos_t0 = np.row_stack(selpos_t0[valid])
    refdispl = refdispl[valid]
    seldispl = seldispl[valid]
    del valid

    ix_sort = np.lexsort((selix_t0, trenew, t0, refix_t0))
    refix_t0 = refix_t0[ix_sort]
    selix_t0 = selix_t0[ix_sort]
    t0 = t0[ix_sort]
    trenew = trenew[ix_sort]
    refpos_t0 = refpos_t0[ix_sort]
    selpos_t0 = selpos_t0[ix_sort]
    refdispl = refdispl[ix_sort]
    seldispl = seldispl[ix_sort]
    del ix_sort

    if coord_traj:
        return (
            refix_t0,
            selix_t0,
            t0,
            trenew,
            refpos_t0,
            selpos_t0,
            refdispl,
            seldispl,
            traj,
        )
    else:
        return (
            refix_t0,
            selix_t0,
            t0,
            trenew,
            refpos_t0,
            selpos_t0,
            refdispl,
            seldispl,
        )


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage.
    parser = argparse.ArgumentParser(
        description=(
            "Extract renewal events from a molecular dynamics trajectory.  For"
            " more information, refer to the documentation of this script."
        )
    )
    parser.add_argument(
        "-f",
        dest="TRJFILE",
        type=str,
        required=True,
        help=(
            "Trajectory file.  IMPORTANT: This scripts requires an unwrapped"
            " trajectory!"
        ),
    )
    parser.add_argument(
        "-s",
        dest="TOPFILE",
        type=str,
        required=True,
        help="Topology file.",
    )
    parser.add_argument(
        "-o",
        dest="OUTFILE",
        type=str,
        required=True,
        help="Output filename.",
    )
    parser.add_argument(
        "--dtrj",
        dest="DTRJFILE",
        type=str,
        required=False,
        default=None,
        help=(
            "Output filename for the discrete coordination trajectory"
            " (optional)."
        ),
    )
    parser.add_argument(
        "-b",
        dest="BEGIN",
        type=int,
        required=False,
        default=0,
        help=(
            "First frame to read from the trajectory.  Frame numbering starts"
            " at zero.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "-e",
        dest="END",
        type=int,
        required=False,
        default=-1,
        help=(
            "Last frame to read from the trajectory (exclusive).  Default:"
            " %(default)s."
        ),
    )
    parser.add_argument(
        "--every",
        dest="EVERY",
        type=int,
        required=False,
        default=1,
        help=(
            "Read every n-th frame from the trajectory.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--ref",
        dest="REF",
        type=str,
        nargs="+",
        required=True,
        help="Selection string for the reference group.",
    )
    parser.add_argument(
        "--sel",
        dest="SEL",
        type=str,
        nargs="+",
        required=True,
        help="Selection string for the selection group.",
    )
    parser.add_argument(
        "-c",
        dest="CUTOFF",
        type=float,
        required=True,
        help="Contact cutoff distance in Angstrom.",
    )
    parser.add_argument(
        "--compound",
        dest="COMPOUND",
        type=str,
        required=False,
        choices=("atoms", "segments", "residues", "fragments"),
        default="atoms",
        help=(
            "The compounds of the reference and selection group to use for the"
            " analysis.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--min-contacts",
        dest="MINCONTACTS",
        type=int,
        required=False,
        default=1,
        help=(
            "Compounds of the reference and selection group are only"
            " considered to be in contact, if there are at least MINCONTACTS"
            " contacts between the atoms of the compounds.  Default:"
            " %(default)s."
        ),
    )
    parser.add_argument(
        "--intermittency",
        dest="INTERMITTENCY",
        type=int,
        required=False,
        default=0,
        help=(
            "Maximum number of frames a selection atom is allowed to leave the"
            " cutoff range of a reference atom whilst still being considered"
            " to be bound to the reference atom, provided that it is indeed"
            " bound again to the reference atom after this unbound period."
            "  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--debug",
        dest="DEBUG",
        required=False,
        default=False,
        action="store_true",
        help="Run in debug mode.",
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    if args.CUTOFF <= 0:
        raise ValueError(
            "-c must be greater than zero, but you gave {}".format(args.CUTOFF)
        )
    if args.MINCONTACTS < 1:
        raise ValueError(
            "--min-contacts must be greater than zero, but you gave"
            " {}".format(args.MINCONTACTS)
        )
    if args.MINCONTACTS > 1 and args.COMPOUND == "atoms":
        args.MINCONTACTS = 1
        print("\n")
        print(
            "Note: Setting --min-contacts to {}, because --compound is set to"
            " 'atoms'".format(args.MINCONTACTS)
        )
    if args.INTERMITTENCY < 0:
        raise ValueError(
            "--intermittency must be equal to or greater than zero, but you"
            " gave {}".format(args.INTERMITTENCY)
        )

    print("\n")
    u = mdt.select.universe(top=args.TOPFILE, trj=args.TRJFILE)
    print("\n")
    print("Creating selections...")
    timer = datetime.now()
    ref = u.select_atoms(" ".join(args.REF))
    sel = u.select_atoms(" ".join(args.SEL))
    if ref.n_atoms == 0 and not args.UPDATING_REF:
        raise ValueError("The reference group contains no atoms")
    if sel.n_atoms == 0 and not args.UPDATING_SEL:
        raise ValueError("The selection group contains no atoms")
    print("Reference group: '{}'".format(" ".join(args.REF)))
    print(mdt.rti.ag_info_str(ag=ref, indent=2))
    print("Selection group: '{}'".format(" ".join(args.SEL)))
    print(mdt.rti.ag_info_str(ag=sel, indent=2))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
    print("\n")
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=u.trajectory.n_frames,
    )
    if args.DEBUG:
        print("\n")
        print("Checking time steps for equality...")
        timer = datetime.now()
        mdt.check.time_step(trj=u.trajectory[BEGIN:END], verbose=True)
        print("Elapsed time:         {}".format(datetime.now() - timer))
        print(
            "Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc))
        )
    timestep = u.trajectory[BEGIN].dt
    start_time = u.trajectory[BEGIN].time

    print("\n")
    print("Calculating contact matrices...")
    timer = datetime.now()
    if mdt.rti.get_num_CPUs() > 1:
        mdabackend = "OpenMP"
    else:
        mdabackend = "serial"
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
    if len(cms) != N_FRAMES:
        raise ValueError(
            "The number of contact matrices does not equal the number of"
            " frames to read.  This should not have happened"
        )
    n_ref_compounds = cms[0].shape[0]
    n_sel_compounds = cms[0].shape[1]
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    if args.INTERMITTENCY > 0:
        print("\n")
        print("Correcting for intermittency...")
        timer = datetime.now()
        cms = mdt.dyn.correct_intermittency(
            list_of_arrays=cms,
            intermittency=args.INTERMITTENCY,
            verbose=True,
            debug=args.DEBUG,
        )
        print("Elapsed time:         {}".format(datetime.now() - timer))
        print(
            "Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc))
        )

    print("\n")
    print("Extracting renewal events...")
    timer = datetime.now()
    if args.DTRJFILE is None:
        coord_traj = False
    else:
        coord_traj = True
    data = extract_renewal_events(
        universe=u,
        ref=ref,
        sel=sel,
        cms=cms,
        compound=args.COMPOUND,
        begin=BEGIN,
        every=EVERY,
        coord_traj=coord_traj,
        verbose=True,
        debug=args.DEBUG,
    )
    del cms
    if args.DTRJFILE is None:
        data = np.column_stack(data)
    else:
        dtrj = data[-1]
        data = np.column_stack(data[:-1])
    if len(data) == 0:
        raise ValueError("Could not detect any renewal event")
    data[:, 2] *= timestep
    data[:, 2] += start_time
    data[:, 3] *= timestep
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    if args.DTRJFILE is not None:
        mdt.fh.save_dtrj(args.DTRJFILE, dtrj)
        print("Created {}".format(args.DTRJFILE), flush=True)

    header = (
        "Trajectory of renewal events.  A renewal event occurs when the\n"
        "selection compound that was continuously bound the longest to\n"
        "a reference compound dissociates from it.\n"
        "\n"
        "\n"
        "Total number of renewal events:        {:>16d}\n"
        "Number of reference compounds NOT\n"
        "undergoing any renewal event:          {:>16d}\n"
        "Number of selection compounds NOT\n"
        "participating in any renewal event:    {:>16d}\n"
        "Mean square displacement r^2 the reference compounds travel\n"
        "while bound to the selection compound: {:>16.9e} (A^2)\n"
        "Standard deviation of r^2:             {:>16.9e} (A^2)\n"
        "Mean square displacement r^2 the selection compounds travel\n"
        "while bound to the reference compound: {:>16.9e} (A^2)\n"
        "Standard deviation of r^2:             {:>16.9e} (A^2)\n".format(
            len(data),
            n_ref_compounds - len(np.unique(data[:, 0])),
            n_sel_compounds - len(np.unique(data[:, 1])),
            np.mean(np.sum(data[:, 10:13] ** 2, axis=1)),
            np.std(np.sum(data[:, 10:13] ** 2, axis=1)),
            np.mean(np.sum(data[:, 13:16] ** 2, axis=1)),
            np.std(np.sum(data[:, 13:16] ** 2, axis=1)),
        )
    )
    header += (
        "\n\n"
        + "Cutoff (Angstrom)     = {}\n".format(args.CUTOFF)
        + "Compound              = {}\n".format(args.COMPOUND)
        + "Minimum contacts      = {}\n".format(args.MINCONTACTS)
        + "Allowed intermittency = {}\n".format(args.INTERMITTENCY)
        + "\n\n"
        + "Reference: '{}'\n".format(" ".join(args.REF))
        + mdt.rti.ag_info_str(ref)
        + "\n\n"
        + "Selection: '{}'\n".format(" ".join(args.SEL))
        + mdt.rti.ag_info_str(sel)
        + "\n\n\n"
        + "The columns contain:\n"
        + "   1    Index of the reference compound (indexing starts at zero)\n"
        + "   2    Index of the longest continuously bound selection compound"
        + " (indexing starts at zero)\n"
        + "   3    Start time t0 (ps) at which the selection compound starts"
        + " to coordinate to the reference compound\n"
        + "   4    Renewal time tau (ps).  Time after which the selection"
        + " compound dissociates from the reference compound\n"
        + "   5- 7 x, y and z coordinate (A) of the reference compound at t0\n"
        + "   8-10 x, y and z coordinate (A) of the selection compound at t0\n"
        + "  11-13 x, y and z displacement (A) of the reference compound"
        + " during tau\n"
        + "  14-16 x, y and z displacement (A) of the selection compound"
        + " during tau\n"
        + "\n"
        + "Column number:\n"
    )
    header += "{:>14d}".format(1)
    for col in range(2, data.shape[-1] + 1):
        header += " {:>16d}".format(col)
    header += "\n"
    header += (
        "\n"
        "              {:>51s} {:>118s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s}\n"  # noqa: E501, W505
        "Mean:          {:>50.9e} {:>118.9e} {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e}\n"  # noqa: E501, W505
        "Std. Dev.:     {:>50.9e} {:>118.9e} {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e} {:>16.9e}\n"  # noqa: E501, W505
        "\n"
        "{:>14s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s}".format(  # noqa: E501, W505
            "Renewal time (ps)",
            "MSD x^2 (A^2)",
            "MSD y^2 (A^2)",
            "MSD z^2 (A^2)",
            "MSD x^2 (A^2)",
            "MSD y^2 (A^2)",
            "MSD z^2 (A^2)",
            np.mean(data[:, 3]),
            np.mean(data[:, 10] ** 2),
            np.mean(data[:, 11] ** 2),
            np.mean(data[:, 12] ** 2),
            np.mean(data[:, 13] ** 2),
            np.mean(data[:, 14] ** 2),
            np.mean(data[:, 15] ** 2),
            np.std(data[:, 3]),
            np.std(data[:, 10] ** 2),
            np.std(data[:, 11] ** 2),
            np.std(data[:, 12] ** 2),
            np.std(data[:, 13] ** 2),
            np.std(data[:, 14] ** 2),
            np.std(data[:, 15] ** 2),
            "ref_ix",
            "sel_ix",
            "t0",
            "tau_renew",
            "x_ref(t0)",
            "y_ref(t0)",
            "z_ref(t0)",
            "x_sel(t0)",
            "y_sel(t0)",
            "z_sel(t0)",
            "dx_ref",
            "dy_ref",
            "dz_ref",
            "dx_sel",
            "dy_sel",
            "dz_sel",
        )
    )
    mdt.fh.savetxt(args.OUTFILE, data, header=header)
    print("Created {}".format(args.OUTFILE), flush=True)
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
