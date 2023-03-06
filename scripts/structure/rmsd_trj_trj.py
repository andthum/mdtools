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


r"""
Calculate the Root Mean Square Deviation (RMSD) between two
trajectories for each frame.

.. todo::

    * Implement a "superposition" functionality like in
      :func:`MDAnalysis.analysis.rms.rmsd`.

Take two groups of atoms from two trajectories and calculate the RMSD
between these groups for each frame.  You must read the same number of
frames from both trajectories and both atom groups must have the same
number of atoms.

Options
-------
-f
    First and second trajectory file.  See
    |supported_coordinate_formats| of MDAnalysis.
-s
    First and second topology file.  See |supported_topology_formats| of
    MDAnalysis.
-o
    Output filename.
-b
    First frame to read from the first and second trajectory,
    respectively.  Frame numbering starts at zero.  Default: ``[0, 0]``.
-e
    Last frame to read from the first and second trajectory,
    respectively.  This is exclusive, i.e. the last frame read is
    actually ``END - 1``.  A value of ``-1`` means to read the very last
    frame.  Default: ``[-1, -1]``.
--every
    Read every n-th frame from the first and second trajectory,
    respectively.  Default: ``[1, 1]``.
--sel1, --sel2
    Selection string to select a group of atoms from the first/second
    trajectory for which to calculate the RMSD.  See MDAnalysis'
    |selection_syntax| for possible choices.
--cmp
    {'group', 'segments', 'residues', 'molecules', 'fragments', 'atoms'}

    The compounds of both selection groups, respectively, to use for the
    RMSD calculation.  Compounds can be 'group' (the entire selection
    group), 'segments', 'residues', 'molecules', 'fragments', or
    'atoms'.  Refer to the MDAnalysis' user guide for an
    |explanation_of_these_terms|.  Note that in any case, even if
    ``CMP`` is e.g. 'residues', only the atoms belonging to the
    selection group are taken into account, even if the compound might
    comprise additional atoms that are not contained in the selection
    group.  If COMPOUND is something different than ``'atoms'``, the
    RMSD will be calculated between the centers of the compounds as
    given by \--center.  Note that broken compounds are **not** made
    whole before calculating their centers, so be sure to provide a
    trajectory with whole compounds if you want to calculate the RMSD
    between compound centers.  This decision was made, because currently
    compounds can only be made whole when wrapping the trajectory
    beforehand.  However, for calculating the RMSD between two
    trajectories, unwrapped trajectories might be desired.  Default:
    ``['atoms', 'atoms']``.
--center
    {'cog', 'com', 'coc'}

    The center of the compounds to use for the RMSD calculation,
    respectively for each trajectory.

        * ``'cog'``: Center of geometry
        * ``'com'``: Center of mass
        * ``'coc'``: Center of charge

    Note that |MDA_always_guesses_atom_masses| from the atom types, even
    if the input file contains the masses.  Default: ``['cog', 'cog']``.
--weights
    {'masses', 'charges'}

    Weight the RMSD by the given property of the compounds from the
    first trajectory.  If ``None``, all compounds are assumed to have a
    weight equal to one.  Weights must not sum up to zero.  Thus, you
    cannot weight atoms by charge in a charge neutral selection.
    Default: ``None``.
--center-pos
    Shift the reference and candidate positions by their (weighted)
    center, respectively, before calculating the RMSD.
--mic
    Take the minimum image convention into account when calculating the
    RMSD.  This works also with unwrapped trajectories.  The box
    dimensions are taken from the first trajectory.  Note that this
    might not be desireable if the two trajectories have different box
    sizes changes.
--debug
    Run in :ref:`debug mode <debug-mode-label>`.

See Also
--------
:mod:`scripts.structure.rmsd_vs_time` :
    Calculate the RMSD between a reference frame and all other frames in
    a trajectory.
:func:`mdtools.structure.rmsd` :
    The underlying function that is called by this script.

Notes
-----
TODO

Examples
--------
TODO
"""


__author__ = "Andreas Thum"


# Standard libraries
import argparse
import os
import sys
from datetime import datetime, timedelta

# Third-party libraries
import numpy as np
import psutil

# First-party libraries
import mdtools as mdt


if __name__ == "__main__":  # noqa: C901
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the Root Mean Square Deviation (RMSD) between two"
            " trajectories for each frame.  For more information, refer to the"
            " documentation of this script."
        )
    )
    parser.add_argument(
        "-f",
        dest="TRJFILE",
        type=str,
        nargs=2,
        required=True,
        help="First and second trajectory file.",
    )
    parser.add_argument(
        "-s",
        dest="TOPFILE",
        type=str,
        nargs=2,
        required=True,
        help="First and second topology file.",
    )
    parser.add_argument(
        "-o",
        dest="OUTFILE",
        type=str,
        required=True,
        help="Output filename.",
    )
    parser.add_argument(
        "-b",
        dest="BEGIN",
        type=int,
        nargs=2,
        required=False,
        default=[0, 0],
        help=(
            "First frame to read from the first and second trajectory,"
            " respectively.  Frame numbering starts at zero.  Default:"
            " %(default)s."
        ),
    )
    parser.add_argument(
        "-e",
        dest="END",
        type=int,
        nargs=2,
        required=False,
        default=[-1, -1],
        help=(
            "Last frame to read from the first and second trajectory,"
            " respectively (exclusive).  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--every",
        dest="EVERY",
        type=int,
        nargs=2,
        required=False,
        default=[1, 1],
        help=(
            "Read every n-th frame from the first and second trajectory,"
            " respectively.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--sel1",
        dest="SEL1",
        type=str,
        nargs="+",
        required=True,
        help="Selection string for the first trajectory.",
    )
    parser.add_argument(
        "--sel2",
        dest="SEL2",
        type=str,
        nargs="+",
        required=True,
        help="Selection string for the second trajectory.",
    )
    parser.add_argument(
        "--cmp",
        dest="CMP",
        type=str,
        nargs=2,
        required=False,
        choices=(
            "group",
            "segments",
            "residues",
            "molecules",
            "fragments",
            "atoms",
        ),
        default=["atoms", "atoms"],
        help=(
            "The compounds of both selection groups, respectively, to use for"
            " the RMSD calculation.  IMPORTANT: Compounds are not made whole"
            " before calculating their centers!  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--center",
        dest="CENTER",
        type=str,
        nargs=2,
        required=False,
        choices=("cog", "com", "coc"),
        default=["cog", "cog"],
        help=(
            "The center of the compounds to use for the RMSD calculation,"
            " respectively for each trajectory.  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--weights",
        dest="WEIGHTS",
        type=str,
        nargs=2,
        required=False,
        choices=("masses", "charges"),
        default=None,
        help=(
            "Weight the RMSD by the given property of the compounds from the"
            " first trajectory.  If None, all particles are assumed to have a"
            " weight equal to one.  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--center-pos",
        dest="CENTER_POS",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Shift the reference and candidate particles by their (weighted)"
            " center, respectively, before calculating the RMSD."
        ),
    )
    parser.add_argument(
        "--mic",
        dest="MIC",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Take the minimum image convention into account when calculating"
            " the RMSD.  The box dimensions are taken from the first"
            " trajectory."
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

    SEL = (args.SEL1, args.SEL2)
    u, sel, frames = [], [], []
    for i in range(len(args.TRJFILE)):
        print("\n")
        u.append(mdt.select.universe(top=args.TOPFILE[i], trj=args.TRJFILE[i]))
        print("\n")
        sel.append(mdt.select.atoms(ag=u[i], sel=" ".join(SEL[i])))
        print("\n")
        # frames = (BEGIN, END, EVERY, N_FRAMES)
        frames.append(
            mdt.check.frame_slicing(
                start=args.BEGIN[i],
                stop=args.END[i],
                step=args.EVERY[i],
                n_frames_tot=u[i].trajectory.n_frames,
            )
        )
    if any(fr[-1] != frames[0][-1] for fr in frames):
        print("\n")
        for i, fr in enumerate(frames):
            print("Trajectory {}: {} frames".format(i + 1, fr[-1]))
        raise ValueError(
            "You must read the same number of frames from all trajectories."
        )

    if args.WEIGHTS is None:
        weights = None
    else:
        weights = mdt.strc.cmp_attr(
            sel[0], cmp=args.CMP[0], attr=args.WEIGHTS[0], weights="total"
        )

    print("\n")
    print("Reading trajectories...")
    for i, fr in enumerate(frames):
        print("Trajectory {}:".format(i + 1))
        print("Frames to read:         {:>8d}".format(fr[-1]))
        print("First frame to read:    {:>8d}".format(fr[0]))
        print("Last frame to read:     {:>8d}".format(fr[1]))
        print("Read every n-th frame:  {:>8d}".format(fr[2]))
    timer = datetime.now()
    times = np.full(frames[0][-1], np.nan, dtype=np.float64)
    rmsd = np.full((frames[0][-1], 3), np.nan, dtype=np.float64)
    trj0 = mdt.rti.ProgressBar(
        u[0].trajectory[frames[0][0] : frames[0][1] : frames[0][2]]
    )
    for i, ts0 in enumerate(trj0):
        u[1].trajectory[i]
        times[i] = ts0.time
        if args.MIC:
            box = ts0.dimensions
        else:
            box = None
        pos = []
        for j, sl in enumerate(sel):
            pos.append(
                mdt.strc.center(
                    ag=sl,
                    center=args.CENTER[j],
                    pbc=False,
                    cmp=args.CMP[j],
                    make_whole=False,
                    debug=args.DEBUG,
                )
            )
        rmsd[i] = mdt.strc.rmsd(
            pos[0],
            pos[1],
            weights=weights,
            center=args.CENTER_POS,
            inplace=True,
            xyz=True,
            box=box,
        )
        # ProgressBar update:
        trj0.set_postfix_str(
            "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
        )
    trj0.close()
    del pos
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    rmsd_tot = np.sqrt(np.sum(rmsd, axis=-1))
    header = (
        "Root Mean Square Deviation (RMSD) between two trajectories for each"
        "frame.  The first trajectory is the reference trajectory.\n"
        "\n"
        "\n"
    )
    for i, sl in enumerate(SEL):
        header += (
            "Trajectory {}:"
            "Selection: '{}'\n"
            "Compound:  {}\n".format(i + 1, " ".join(sl), args.CMP[i])
            + mdt.rti.ag_info_str(ag=sel[i])
            + "\n"
        )
    header += (
        "\n"
        "\n"
        "The columns contain:\n"
        "  1) Time of the reference trajectory in [ps]\n"
        "  2) RMSD in [A]\n"
        "  3) <x^2> in [A^2] (x component of RMSD)\n"
        "  4) <y^2> in [A^2] (y component of RMSD)\n"
        "  5) <z^2> in [A^2] (z component of RMSD)\n"
        "\n"
        "{:>14d} {:>16d} {:>16d} {:>16d} {:>16d}".format(1, 2, 3, 4, 5)
    )
    data = np.column_stack([times, rmsd_tot, rmsd])
    mdt.fh.savetxt(args.OUTFILE, data, header=header)
    print("Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
