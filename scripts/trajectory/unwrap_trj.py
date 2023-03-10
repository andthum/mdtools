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
Unwrap the given trajectory.

Unwrap the selected compounds out of the primary unit cell, i.e.
calculate their real-space coordinates.

Options
-------
-f
    Trajectory file.  See |supported_coordinate_formats| of MDAnalysis.
-s
    Topology file.  See |supported_topology_formats| of MDAnalysis.
--trj-out
    Output file name for the unwrapped trajectory.  See
    |supported_coordinate_formats| of MDAnalysis.
--top-out
    Output file name for the topology file belonging to the generated
    unwrapped trajectory (optional).  See
    |supported_topology_formats| of MDAnalysis.  If you unwrap the full
    trajectory, providing \--top-out is usually not necessary, because
    the input topology file can also be used with the generated
    unwrapped trajectory.  However, if you unwrap only a subset of atoms
    with \--sel, you probably want to generate a corresponding topology
    file.
-b
    First frame to read from the trajectory.  Frame numbering starts at
    zero.  Default: ``0``.
-e
    Last frame to read from the trajectory.  This is exclusive, i.e. the
    last frame read is actually ``END - 1``.  A value of ``-1`` means to
    read the very last frame.  Default: ``-1``.
--every
    Read every n-th frame from the trajectory.  Note that the sampling
    interval should be as small as possible to avoid unwrapping errors.
    Non of the unwrapping methods works if the particle displacement
    between two frames is larger than half of the box length.  For more
    details see the notes of :func:`mdtools.box.unwrap_frame`.  Default:
    ``1``.
--sel
    Selection string to select a group of atoms for the unwrapping.  See
    MDAnalysis' |selection_syntax| for possible choices.  Only the
    selected atoms will be unwrapped and written to the new trajectory.
    Use 'all' if you want to unwrap all atoms in the trajectory.
--cmp
    {'group', 'segments', 'residues', 'molecules', 'fragments', 'atoms'}

    The compounds of the selection group to make whole for the first
    frame before starting the actual unwrapping.  Is ignored if set to
    "atoms".  Note that all atoms within each compound must be
    interconnected by bonds, otherwise an error will be raised.  Also
    note that all atoms are wrapped back into the primary unit cell
    before making broken compounds whole.  This means, you cannot start
    from an already unwrapped configuration unless \--cmp is set to
    "atoms".

    Compounds can be 'group' (the entire selection group), 'segments',
    'residues', 'molecules', 'fragments', or 'atoms'.  Refer to the
    MDAnalysis' user guide for an |explanation_of_these_terms|.  Note
    that in any case, even if ``CMP`` is e.g. 'residues', only the atoms
    belonging to the selection group are taken into account, even if the
    compound might comprise additional atoms that are not contained in
    the selection group.  Default: ``'atoms'``.
--center
    {'cog', 'com'}

    The center of the compounds to shift to the primary unit cell before
    unwrapping the trajectory when making broken compounds whole:

        * ``'cog'``: Center of geometry
        * ``'com'``: Center of mass

    Unless \--cmp is "atoms", broken compounds are made whole for the
    first frame before unwrapping the trajectory.  Thereby, the
    compounds are shifted in such a way that their centers lie within
    the primary unit cell.  A change of the centering method might
    affect the unwrapped trajectory, depending on the chosen unwrapping
    algorithm, because the unwrapping might start from a different
    starting configuration.

    Note that |MDA_always_guesses_atom_masses| from the atom types, even
    if the input file contains the masses. Default: ``'cog'``.
--method
    {'scaling', 'heuristic', 'displacement', 'hybrid', 'in-house'}

    The unwrapping method to choose.  See the notes section of
    :func:`mdtools.box.unwrap_frame` for further details.  Default:
    ``'scaling'``.
--debug
    Run in :ref:`debug mode <debug-mode-label>`.

See Also
--------
:func:`mdtools.box.unwrap_frame` :
    Underlying function that performs the unwrapping.

Notes
-----
See :func:`mdtools.box.unwrap_frame` for a description of the different
unwrapping methods and their limitations.

References
----------
* Sören von Bülow, Jakob Tómas Bullerjahn, and Gerhard Hummer,
  `Systematic errors in diffusion coefficients from long-time molecular
  dynamics simulations at constant pressure
  <https://doi.org/10.1063/5.0008316>`_,
  The Journal of Chemical Physics, 2020, 153, 021101.
* Martin Kulke and Josh V. Vermaas,
  `Reversible Unwrapping Algorithm for Constant-Pressure Molecular
  Dynamics Simulations <https://doi.org/10.1021/acs.jctc.2c00327>`_,
  Journal of Chemical Theory and Computation, 2022, 18, 10, 6161-6171.
"""


# Standard libraries
import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

# Third-party libraries
import MDAnalysis as mda
import numpy as np
import psutil

# First-party libraries
import mdtools as mdt


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage.
    parser = argparse.ArgumentParser(
        description=(
            "Unwrap the given trajectory.  For more information, refer to the"
            " documentation of this script."
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
        "--trj-out",
        dest="TRJFILE_OUT",
        type=str,
        required=True,
        help="Output trajectory.",
    )
    parser.add_argument(
        "--top-out",
        dest="TOPFILE_OUT",
        type=str,
        required=False,
        default=None,
        help="Output topology (optional).",
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
        "--sel",
        dest="SEL",
        type=str,
        nargs="+",
        required=False,
        default=["all"],
        help="Selection string.  Default: %(default)s.",
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
            "The compounds of the selection group to make whole for the first"
            " frame before unwrapping the trajectory.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--center",
        dest="CENTER",
        type=str,
        required=False,
        choices=("cog", "com"),
        default="cog",
        help=(
            "The center of the compounds to shift to the primary unit cell"
            " before unwrapping the trajectory when making broken compounds"
            " whole (ignored if --cmp is 'atoms').  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--method",
        dest="METHOD",
        type=str,
        required=False,
        choices=("scaling", "heuristic", "displacement", "hybrid", "in-house"),
        default="scaling",
        help="Unwrapping method.  Default: %(default)s.",
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
    if " ".join(args.SEL).lower() != "all" and args.TOPFILE_OUT is None:
        warnings.warn(
            "You seem to unwrap only a subset of the trajectory (--sel is not"
            " 'all'), but you did not provide --top-out.",
            RuntimeWarning,
        )

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
    timer = datetime.now()

    ts = u.trajectory[BEGIN]
    out = np.full_like(sel.positions, np.nan, dtype=np.float64)
    out_tmp = np.full_like(out, np.nan)
    if args.CMP != "atoms":
        pos_w_prev = mdt.box.make_whole(
            ag=sel,
            compound=args.CMP,
            reference=args.CENTER,
            inplace=True,
            debug=args.DEBUG,
        )
        sel.positions = pos_w_prev
    else:
        pos_w_prev = np.copy(sel.positions)
    pos_u_prev = np.copy(pos_w_prev)
    box_prev = np.copy(ts.dimensions)

    if args.TOPFILE_OUT is not None:
        mdt.fh.backup(args.TOPFILE_OUT)
        sel.write(args.TOPFILE_OUT)
        print("Created {}".format(args.TOPFILE_OUT))
    mdt.fh.backup(args.TRJFILE_OUT)
    with mda.Writer(args.TRJFILE_OUT, sel.n_atoms) as w:
        w.write(sel)
        trj = mdt.rti.ProgressBar(
            u.trajectory[BEGIN + 1 : END : EVERY], initial=1, total=N_FRAMES
        )
        for ts in trj:
            pos_u = mdt.box.unwrap_frame(
                pos_w=sel.positions,
                pos_u_prev=pos_u_prev,
                box=ts.dimensions,
                box_prev=box_prev,
                pos_w_prev=pos_w_prev,
                method=args.METHOD,
                out=out,
                out_tmp=out_tmp,
                dtype=np.float64,
            )
            np.copyto(pos_w_prev, sel.positions)
            sel.positions = pos_u
            np.copyto(pos_u_prev, pos_u)
            np.copyto(box_prev, ts.dimensions)
            w.write(sel)
            # ProgressBar update.
            trj.set_postfix_str(
                "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
            )
        trj.close()
    print("Created {}".format(args.TRJFILE_OUT))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
