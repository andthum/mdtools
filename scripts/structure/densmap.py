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


"""
Compute a 2-dimensional number density map.

Project all selected compounds along a given spatial dimension on a
plane and calculate the number density distribution of the compounds on
the plane.

Options
-------
-f
    Trajectory file.  See |supported_coordinate_formats| of MDAnalysis.
-s
    Topology file.  See |supported_topology_formats| of MDAnalysis.
-o
    Output filename.
-b
    First frame to read from the trajectory.  Frame numbering starts at
    zero.  Default: ``0``.
-e
    Last frame to read from the trajectory.  This is exclusive, i.e. the
    last frame read is actually ``END - 1``.  A value of ``-1`` means to
    read the very last frame.  Default: ``-1``.
--every
    Read every n-th frame from the trajectory.  Default: ``1``.
--sel
    Selection string to select a group of atoms for the analysis.  See
    MDAnalysis' |selection_syntax| for possible choices.
--cmp
    {'group', 'segments', 'residues', 'fragments', 'atoms'}

    The compounds of the selection group to use for the analysis.
    Compounds can be 'group' (the entire selection group), 'segments',
    'residues', 'fragments', or 'atoms'.  Refer to the MDAnalysis' user
    guide for an |explanation_of_these_terms|.  Note that in any case,
    even if ``CMP`` is e.g. 'residues', only the atoms belonging to the
    selection group are taken into account, even if the compound might
    comprise additional atoms that are not contained in the selection
    group.  Default: ``'atoms'``.
--center
    {'cog', 'com', 'coc'}

    The center of the compounds to use for the analysis.

        * ``'cog'``: Center of geometry
        * ``'com'``: Center of mass
        * ``'coc'``: Center of charge

    Note that |MDA_always_guesses_atom_masses| from the atom types, even
    if the input file contains the masses.  Default: ``'cog'``.
--direction
    {'x', 'y', 'z'}

    The spatial direction along which to project the selected compounds
    on a plane.  By default, the density map is created in the xy plane
    and projection/averaging is done along the z axis.  Default:
    ``'z'``.
--min
    Minimum coordinate for averaging in Angstrom.  Only compounds whose
    position in the projection direction is greater than or equal to the
    minimum coordinate are taken into account.  Default: ``0``.
--max
    Maximum coordinate for averaging in Angstrom.  Only compounds whose
    position in the projection direction is less than the maximum
    coordinate are taken into account.  If ``None``, the maximum
    coordinate is set to the box length in the projection direction.
    Default: ``None``.
--grid-spacing
    Desired grid spacing in Angstrom for binning the compounds in the
    plane.  Note that the grid spacing will be adjusted such that the
    grid fits within the simulation box.  The final grid cells might not
    be squares.  Default: ``0.05``.
--debug
    Run in :ref:`debug mode <debug-mode-label>`.

Notes
-----
This script is similar to the Gromacs tool `gmx densmap
<https://manual.gromacs.org/documentation/current/onlinehelp/gmx-densmap.html>`_.

The binning is done in the box coordinate system to take into account
potential box fluctuations.

See Also
--------
:mod:`scripts.structure.plot_gmx_densmap` :
    Read up to three matrices from text files and plot them as one RGB
    matrix
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
    proc.cpu_percent()  # Initiate monitoring of CPU usage.
    parser = argparse.ArgumentParser(
        # The description should only contain the short summary from the
        # docstring and a reference to the documentation.
        description=(
            "Compute a 2-dimensional number density map.  For more"
            " information, refer to the documentation of this script."
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
        choices=("group", "segments", "residues", "fragments", "atoms"),
        default="atoms",
        help=(
            "The compounds of the selection group to use for the analysis."
            "  Default: %(default)s."
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
            " %(default)s."
        ),
    )
    parser.add_argument(
        "--direction",
        dest="DIRECTION",
        type=str,
        required=False,
        choices=("x", "y", "z"),
        default="z",
        help=(
            "The spatial direction along which to project the selected"
            " compounds on a plane.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--min",
        dest="MIN",
        type=float,
        required=False,
        default=0,
        help=(
            "Minimum coordinate for averaging in Angstrom.  Only compounds"
            " whose position in the projection direction is greater than or"
            " equal to the minimum coordinate are taken into account."
            "  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--max",
        dest="MAX",
        type=float,
        required=False,
        default=None,
        help=(
            "Maximum coordinate for averaging in Angstrom.  Only compounds"
            " whose position in the projection direction is less than the"
            " maximum coordinate are taken into account.  If ``None``, the"
            " maximum coordinate is set to the box length in the projection"
            " direction.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--grid-spacing",
        dest="GRID_SPACING",
        type=float,
        required=False,
        default=0.05,
        help=(
            "Desired grid spacing in Angstrom for binning the compounds in the"
            " plane.  Note that the grid spacing will be adjusted such that"
            " the grid fits within the simulation box.  Default: %(default)s."
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
    if args.MIN < 0:
        raise ValueError("--min ({}) must not be negative".format(args.MIN))
    if args.GRID_SPACING <= 0:
        raise ValueError(
            "--grid-spacing ({}) must be greater than"
            " zero".format(args.GRID_SPACING)
        )
    dim_prj = {"x": 0, "y": 1, "z": 2}
    dim_plane = {"x": [1, 2], "y": [0, 2], "z": [0, 1]}
    plane_dims = {"x": ["y", "z"], "y": ["x", "z"], "z": ["x", "y"]}
    ixd_prj = dim_prj[args.DIRECTION]
    ixd_plane = dim_plane[args.DIRECTION]

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
    print("Creating grid...")
    timer = datetime.now()
    tol = 1e-6
    box = u.trajectory[BEGIN].dimensions
    if np.any(box <= 0):
        raise ValueError(
            "Invalid simulation box: At least one box length or angle ({}) is"
            " less than or equal to zero".format(box)
        )

    # Check coordinate limits in the projection direction.
    lbox_prj = box[ixd_prj]
    if args.MAX is None:
        args.MAX = lbox_prj
    if args.MAX > lbox_prj:
        args.MAX = lbox_prj
        print(
            "Note: Set --max to {}, because it exceed the box length ({}) in"
            " the projection direction".format(args.MAX, lbox_prj)
        )
    if args.MIN >= args.MAX:
        raise ValueError(
            "--min ({}) must be less than --max"
            " ({})".format(args.MIN, args.MAX)
        )
    # Convert coordinate limits to box coordinates (0 to 1).
    args.MIN /= lbox_prj
    args.MAX /= lbox_prj
    if np.isclose(args.MAX, 1, rtol=0):
        args.MAX = 1 + tol

    # Create bins for the plane perpendicular to the projection
    # direction.
    lbox_plane = box[ixd_plane]
    nbins = np.round(lbox_plane / args.GRID_SPACING).astype(np.uint32)
    bins = []
    for nb in nbins:
        if nb < 1:
            raise ValueError(
                "The grid spacing ({}) is larger than the simulation box"
                " ({}).".format(args.GRID_SPACING, box)
            )
        START, STOP, STEP, NUM = mdt.check.bins(
            start=0, stop=1, num=nb, amin=0, amax=1
        )
        # Create bins in the box coordinate system (0 to 1).
        bins_tmp = np.linspace(START, STOP, NUM + 1)
        bins_tmp = mdt.check.bin_edges(bins=bins_tmp, amin=0, amax=1, tol=tol)
        bins.append(bins_tmp)
    hist = np.zeros(nbins, dtype=np.float64)
    del bins_tmp
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    if args.CMP == "group":
        valid = np.zeros(1, dtype=bool)
    elif args.CMP == "segments":
        valid = np.zeros(sel.n_segments, dtype=bool)
    elif args.CMP == "residues":
        valid = np.zeros(sel.n_residues, dtype=bool)
    elif args.CMP == "fragments":
        valid = np.zeros(sel.n_fragments, dtype=bool)
    elif args.CMP == "atoms":
        valid = np.zeros(sel.n_atoms, dtype=bool)
    else:
        raise ValueError("Unknown --cmp: {}".format(args.CMP))
    valid_tmp = np.zeros_like(valid)

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
    lbox_mean = np.zeros(3, dtype=np.float64)
    hist_tmp, bins0, bins1 = np.nan, np.nan, np.nan
    trj = mdt.rti.ProgressBar(u.trajectory[BEGIN:END:EVERY])
    for ts in trj:
        lbox_mean += ts.dimensions[:3]
        pos = mdt.strc.center(
            ag=sel,
            center=args.CENTER,
            pbc=True,
            cmp=args.CMP,
            make_whole=True,
            debug=args.DEBUG,
        )
        pos = mdt.box.cart2box(
            pos, box=ts.dimensions, out=pos, dtype=pos.dtype
        )
        valid = np.greater_equal(pos[:, ixd_prj], args.MIN, out=valid)
        valid_tmp = np.less(pos[:, ixd_prj], args.MAX, out=valid_tmp)
        valid &= valid_tmp
        if not np.any(valid):
            continue
        pos = pos[valid]
        pos = pos[:, ixd_plane]
        hist_tmp, bins0, bins1 = np.histogram2d(
            *pos.T, bins=bins, density=False
        )
        hist += hist_tmp
        if not np.allclose(bins0, bins[0]) or not np.allclose(bins1, bins[1]):
            raise ValueError(
                "The bin edges have changed.  This should not have happened"
            )
        # ProgressBar update.
        trj.set_postfix_str(
            "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
        )
    trj.close()
    del pos, valid, valid_tmp, hist_tmp, bins0, bins1
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    # Convert coordinate limits from box coordinates to Cartesian
    # coordinates.
    lbox_mean /= N_FRAMES
    args.MIN *= lbox_mean[ixd_prj]
    args.MAX *= lbox_mean[ixd_prj]
    # Convert bin edges from box coordinates to Cartesian coordinates.
    for i, lbox_plane_mean in enumerate(lbox_mean[ixd_plane]):
        bins[i] *= lbox_plane_mean

    # Calculate average number density per bin.
    bin_width_0 = bins[0][1] - bins[0][0]
    bin_width_1 = bins[1][1] - bins[1][0]
    bin_height = args.MAX - args.MIN
    bin_volume = bin_width_0 * bin_width_1 * bin_height
    hist /= N_FRAMES * bin_volume

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    header = (
        "2-dimensional number density map in {:s} plane.\n".format(
            "".join(plane_dims[args.DIRECTION])
        )
        + "\n\n"
        + "Selection string:   '{:s}'\n".format(" ".join(args.SEL))
        + "Selection compound: '{:s}'\n".format(args.CMP)
        + mdt.rti.ag_info_str(sel)
        + "\n\n\n"
        + "Projection direction:   {:s}\n".format(args.DIRECTION)
        + "Minimum coordinate:    {:>16.9e} A\n".format(args.MIN)
        + "Maximum coordinate:    {:>16.9e} A\n".format(args.MAX)
        + "Desired grid spacing:  {:>16.9e} A x {:>1.9e} A\n".format(
            args.GRID_SPACING, args.GRID_SPACING
        )
        + "Actual grid spacing:   {:>16.9e} A x {:>15.9e} A\n".format(
            bin_width_0, bin_width_1
        )
        + "(Average) bin volume:  {:>16.9e} A^3\n".format(bin_volume)
        + "Number of read frames:  {:d}\n".format(N_FRAMES)
        + "(Average) simulation box lengths:\n"
        + "  lx: {:>16.9e} A\n".format(lbox_mean[0])
        + "  ly: {:>16.9e} A\n".format(lbox_mean[1])
        + "  lz: {:>16.9e} A\n".format(lbox_mean[2])
        + "\n\n"
        + "The first column contains the (average) bin edges in Angstrom\n"
        + "used for binning the {:s} direction of the plane.\n".format(
            plane_dims[args.DIRECTION][0]
        )
        + "The first row contains the (average) bin edges in Angstrom\n"
        + "used for binning the {:s} direction of the plane.\n".format(
            plane_dims[args.DIRECTION][1]
        )
        + "The remaining matrix elements contain the number density of the\n"
        + "selection compounds in the respective bin in Angstrom^{-3}.\n"
    )
    mdt.fh.savetxt_matrix(
        args.OUTFILE,
        data=hist,
        var1=bins[0][1:],
        var2=bins[1][1:],
        upper_left=0,
        header=header,
    )
    print("Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    # Consistency checks.
    tol *= 100
    for i, bin_edges in enumerate(bins):
        if not np.isclose(bin_edges[0], 0, rtol=0):
            raise ValueError(
                "The first bin edge ({}) in {} direction is not zero.  This"
                " should not have"
                " happened".format(bin_edges[0], plane_dims[args.DIRECTION][i])
            )
        if not np.isclose(
            bin_edges[-1], lbox_mean[ixd_plane][i], rtol=0, atol=tol
        ):
            raise ValueError(
                "The last bin edge ({}) in {} direction is not equal to the"
                " (average) box length ({}) in this direction.  This should"
                " not have happened".format(
                    bin_edges[-1],
                    plane_dims[args.DIRECTION][i],
                    lbox_mean[ixd_plane][i],
                )
            )
        bin_width = bin_edges[1] - bin_edges[0]
        if not np.allclose(np.diff(bin_edges), bin_width, rtol=0, atol=tol):
            print("\n")
            print("Bin width =", bin_width)
            print("np.diff(bin_edges) =")
            print(np.diff(bin_edges))
            raise ValueError(
                "The bins in direction {} are not equidistant.  This should"
                " not have happened".format(plane_dims[args.DIRECTION][i])
            )

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
