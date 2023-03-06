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
Calculate the Root Mean Square Deviation (RMSD) between a reference
frame and all other frames in the trajectory.

.. todo::

    * Implement a "superposition" functionality like in
      :func:`MDAnalysis.analysis.rms.rmsd`.

    * Provide an option to make broken compounds whole before
      calculating their centers.  Currently, making compounds whole is
      only possible by first wrapping the trajectory and then making the
      compounds whole.  However, for calculating a time-dependent RMSD,
      an unwrapped trajectory might be desired.

Take a structure from a reference frame and calculate the RMSD to its
configuration in all other frames in the trajectory.  This generates an
RMSD as function of time.

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
--ref-frame
            Frame from which to take the reference structure.  ``None``
            means take the same frame as given by -b.  Default:
            ``None``.
--sel       Selection string to select a group of atoms for which to
            calculate the RMSD as function of time.  See MDAnalysis'
            |selection_syntax| for possible choices.
--cmp       {'group', 'segments', 'residues', 'molecules', \
            'fragments', 'atoms'}

            The compounds of the selection group to use for the RMSD
            calculation.  Compounds can be 'group' (the entire selection
            group), 'segments', 'residues', 'molecules', 'fragments', or
            'atoms'.  Refer to the MDAnalysis' user guide for an
            |explanation_of_these_terms|.  Note that in any case, even
            if ``CMP`` is e.g. 'residues', only the atoms belonging to
            the selection group are taken into account, even if the
            compound might comprise additional atoms that are not
            contained in the selection group.  If COMPOUND is something
            different than ``'atoms'``, the RMSD will be calculated
            between the centers of the compounds as given by \--center.
            Note that broken compounds are **not** made whole before
            calculating their centers, so be sure to provide a
            trajectory with whole compounds if you want to calculate the
            RMSD between compound centers.  This decision was made,
            because currently compounds can only be made whole when
            wrapping the trajectory beforehand.  However, for
            calculating a time-dependent RMSD, an unwrapped trajectory
            might be desired.  Default: ``'atoms'``.
--center    {'cog', 'com', 'coc'}

            The center of the compounds to use for the RMSD calculation.

                * ``'cog'``: Center of geometry
                * ``'com'``: Center of mass
                * ``'coc'``: Center of charge

            Note that |MDA_always_guesses_atom_masses| from the atom
            types, even if the input file contains the masses.  Default:
            ``'cog'``.
--weights   {'mass', 'charge'}

            Weights to use for calculating the RMSD.  If ``None``, all
            compounds are assumed to have a weight equal to one.
            Weights must not sum up to zero.  Thus, you cannot weight
            atoms by charge in a charge neutral selection.  Default:
            ``None``.
--center-pos
            Shift the reference and candidate positions by their
            (weighted) center, respectively, before calculating the
            RMSD.
--mic       Take the minimum image convention into account when
            calculating the RMSD.  This works also with unwrapped
            trajectories.  The box dimensions are taken from the
            reference frame.  Note that this might not be desireable in
            simulations where the box size changes (like in NpT
            simulations).
--debug     Run in :ref:`debug mode <debug-mode-label>`.

See Also
--------
:func:`scripts.structure.rmsd_trj_trj` :
    Calculate the RMSD between two trajectories for each frame.
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
            "Calculate the Root Mean Square Deviation (RMSD) between a"
            " reference frame and all other frames in the trajectory.  For"
            " more information, refer to the documentation of this script."
        )
    )
    parser.add_argument(
        "-f",
        dest="TRJFILE",
        type=str,
        required=True,
        help="Trajectory file.",
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
        "--ref-frame",
        dest="REF_FRAME",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=int),
        required=False,
        default=None,
        help=(
            "Frame from which to take the reference structure.  Default:"
            " %(default)s."
        ),
    )
    parser.add_argument(
        "--sel",
        dest="SEL",
        type=str,
        nargs="+",
        required=True,
        help="Selection string.",
    )
    parser.add_argument(
        "--cmp",
        dest="CMP",
        type=str,
        required=False,
        choices=(
            "group",
            "segments",
            "residues",
            "molecules",
            "fragments",
            "atoms",
        ),
        default="atoms",
        help=(
            "The compounds of the selection group to use for the analysis."
            "  IMPORTANT: Compounds are not made whole before calculating"
            " their centers!  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--center",
        dest="CENTER",
        type=str,
        required=False,
        choices=("cog", "com", "coc"),
        default="cog",
        help=(
            "The center of the compounds to use for the analysis.  Default:"
            " %(default)s"
        ),
    )
    parser.add_argument(
        "--weights",
        dest="WEIGHTS",
        type=str,
        required=False,
        choices=("mass", "charge"),
        default=None,
        help=(
            "Weights to use for calculating the RMSD.  If None, all particles"
            " are assumed to have a weight equal to one.  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--center-pos",
        dest="CENTER_POS",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Shift the reference and candidate particles by their weighted"
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
            " the RMSD.  The box dimensions are taken from the reference"
            " frame."
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

    print("\n")
    u = mdt.select.universe(top=args.TOPFILE, trj=args.TRJFILE)
    print("\n")
    sel = mdt.select.atoms(ag=u, sel=" ".join(args.SEL))
    print("\n")
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=u.trajectory.n_frames,
    )
    first_frame_read = u.trajectory[BEGIN].copy()
    last_frame_read = u.trajectory[END - 1].copy()

    if args.REF_FRAME is None:
        args.REF_FRAME = BEGIN
    if args.REF_FRAME < 0 or args.REF_FRAME >= u.trajectory.n_frames:
        raise ValueError(
            "--ref-frame ({}) lies outside the range of available frames ([0,"
            " {}])".format(args.REF_FRAME, u.trajectory.n_frames)
        )
    ref_frame = u.trajectory[args.REF_FRAME].copy()
    if args.MIC:
        box = ref_frame.dimensions
    else:
        box = None
    ref_pos = mdt.strc.center(
        ag=sel,
        center=args.CENTER,
        pbc=False,
        cmp=args.CMP,
        make_whole=False,
        debug=args.DEBUG,
    )

    if args.WEIGHTS is None:
        weights = args.WEIGHTS
    elif args.WEIGHTS == "mass":
        if args.CMP == "group":
            weights = np.sum(sel.masses)
        elif args.CMP == "segments":
            weights = sel.segments.masses
        elif args.CMP == "residues":
            weights = sel.residues.masses
        elif args.CMP == "molecules":
            natms_per_mol = np.unique(sel.molnums, return_counts=True)[1]
            slices = np.cumsum(natms_per_mol[:-1], dtype=np.uint32)
            slices = np.insert(slices, 0, 0)
            weights = np.add.reduceat(sel.masses, slices)
        elif args.CMP == "fragments":
            weights = np.array([np.sum(frag.masses) for frag in sel.fragments])
        elif args.CMP == "atoms":
            weights = sel.masses
        else:
            raise ValueError("Invalide choice for --cmp ({})".format(args.CMP))
    elif args.WEIGHTS == "charge":
        if args.CMP == "group":
            weights = np.sum(sel.charges)
        elif args.CMP == "segments":
            weights = sel.segments.charges
        elif args.CMP == "residues":
            weights = sel.residues.charges
        elif args.CMP == "molecules":
            natms_per_mol = np.unique(sel.molnums, return_counts=True)[1]
            slices = np.cumsum(natms_per_mol[:-1], dtype=np.uint32)
            slices = np.insert(slices, 0, 0)
            weights = np.add.reduceat(sel.charges, slices)
        elif args.CMP == "fragments":
            weights = np.array(
                [np.sum(frag.charges) for frag in sel.fragments]
            )
        elif args.CMP == "atoms":
            weights = sel.charges
        else:
            raise ValueError("Invalide choice for --cmp ({})".format(args.CMP))
    else:
        raise ValueError(
            "Invalide choice for --weights ({})".format(args.WEIGHTS)
        )
    if weights is not None and len(weights) != len(ref_pos):
        raise ValueError(
            "The number of weights ({}) does not match the number of reference"
            " compounds ({}).  This should not have"
            " happened".format(len(weights), len(ref_pos))
        )

    print("\n")
    print("Reading trajectory...")
    print("Total number of frames: {:>8d}".format(u.trajectory.n_frames))
    print("Frames to read:         {:>8d}".format(N_FRAMES))
    print("First frame to read:    {:>8d}".format(BEGIN))
    print("Last frame to read:     {:>8d}".format(END - 1))
    print("Read every n-th frame:  {:>8d}".format(EVERY))
    print("Time first frame:       {:>12.3f} ps".format(first_frame_read.time))
    print("Time last frame:        {:>12.3f} ps".format(last_frame_read.time))
    print("Time step first frame:  {:>12.3f} ps".format(first_frame_read.dt))
    print("Time step last frame:   {:>12.3f} ps".format(last_frame_read.dt))
    print("Reference frame (RMSD): {:>8d}".format(args.REF_FRAME))
    timer = datetime.now()
    rmsd = np.full((N_FRAMES, 3), np.nan, dtype=np.float64)
    trj = mdt.rti.ProgressBar(u.trajectory[BEGIN:END:EVERY])
    for i, _ts in enumerate(trj):
        sel_pos = mdt.strc.center(
            ag=sel,
            center=args.CENTER,
            pbc=False,
            cmp=args.CMP,
            make_whole=False,
            debug=args.DEBUG,
        )
        rmsd[i] = mdt.strc.rmsd(
            ref_pos,
            sel_pos,
            weights=weights,
            center=args.CENTER_POS,
            inplace=True,
            xyz=True,
            box=box,
        )
        # ProgressBar update:
        trj.set_postfix_str(
            "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
        )
    trj.close()
    del ref_pos, sel_pos
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    rmsd_tot = np.sqrt(np.sum(rmsd, axis=1))
    times = np.array([ts.time for ts in u.trajectory[BEGIN:END:EVERY]])
    header = (
        "Root Mean Square Deviation (RMSD)\n"
        "\n"
        "Reference frame index: {}\n"
        "Reference frame time:  {:.3f} ps\n"
        "\n"
        "\n"
        "Selection: '{}'\n"
        "Compound:  {}\n".format(
            args.REF_FRAME, ref_frame.time, " ".join(args.SEL), args.CMP
        )
        + mdt.rti.ag_info_str(ag=sel)
        + "\n"
        "\n"
        "\n"
        "The columns contain:\n"
        "  1) Time in [ps]\n"
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
