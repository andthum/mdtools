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


"""TODO: Write docstring"""


__author__ = "Andreas Thum"


# Standard libraries
import argparse
import os
import sys
import warnings
from datetime import datetime

# Third-party libraries
import MDAnalysis.lib.distances as mdadist
import numpy as np
import psutil

# First-party libraries
import mdtools as mdt


def check_hex_lattice(verts, r0, box, flatside='x', tol=1e-3):
    r"""
    Check if a given hexagonal lattice is suited for the analyses in
    this script.

    The hexagonal lattice must

        * lie flat in xy plane
        * continue properly across periodic boundaries

    Parameters
    ----------
    verts : numpy.ndarray
        Array of shape ``(n, 3)`` containing the positions of all ``n``
        vertices of the hexagonal lattice.
    r0 : scalar
        Side length of the hexagons.  Note that the side length of the
        hexagons is related to the lattice constant `a` via
        :math:`a = 2 r_0 \sin{(60°)} = r_0 \sqrt{3}`.
    box : array_like
        The unit cell dimensions of the system, which can be orthogonal
        or triclinic and must be provided in the same format as returned
        by :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.
    flatside : {'x', 'y'}, optional
        Specify whether the edges of the hexagons align with the x or
        the y axis of the simulation box.
    tol : scalar, optional
        Two floating point numbers are regarded as equal if they deviate
        by less than the tolerance given here.

    Raises
    ------
    ValueError
        If the hexagonal lattice defined by `verts` does not meet the
        above listed requirements.
    """
    verts = mdadist.apply_PBC(verts, box=box)
    if not np.allclose(verts[:, 2], verts[0, 2], rtol=0, atol=tol):
        raise ValueError(
            "The hexagonal lattice must lie flat in xy plane"
        )
    direction = ('x', 'y')
    if flatside == 'x':
        ix0, ix1 = 0, 1
    elif flatside == 'y':
        ix0, ix1 = 1, 0
    else:
        raise ValueError(
            "`flatside` must be either 'x' or 'y', but you gave"
            " {}".format(flatside)
        )
    if not np.isclose(box[ix0] % (r0 * 3), 0, rtol=0, atol=tol):
        raise ValueError(
            "The hexagonal lattice does not continue properly across periodic"
            " boundaries in {} direction".format(direction[ix0])
        )
    if not np.isclose(box[ix1] % (r0 * np.sqrt(3)), 0, rtol=0, atol=tol):
        raise ValueError(
            "The hexagonal lattice does not continue properly across periodic"
            " boundaries in {} direction".format(direction[ix1])
        )


# A "copy"" of this function is used in discretization/discrete_hex.py
# Last modified: 2021-01-19
def hex_verts2faces(verts, r0, box, flatside='x', tol=1e-3):
    r"""
    Calculate the positions of the faces of a hexagonal lattice from the
    positions of the vertices.

    Parameters
    ----------
    verts : numpy.ndarray
        Array of shape ``(n, 3)`` containing the positions of all ``n``
        vertices of the hexagonal lattice.
    r0 : scalar, optional
        Side length of the hexagons.  Note that the side length of the
        hexagons is related to the lattice constant `a` via
        :math:`a = 2 r_0 \sin{(60°)} = r_0 \sqrt{3}`.
    box : array_like, optional
        The unit cell dimensions of the system, which can be orthogonal
        or triclinic and must be provided in the same format as returned
        by :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.
    flatside : {'x', 'y'}, optional
        Specify whether the edges of the hexagons align with the x or
        the y axis of the simulation box.
    tol : scalar, optional
        Two floating point numbers are regarded as equal if they deviate
        by less than the tolerance given here.

    Returns
    -------
    faces : numpy.ndarray
        Array of shape ``(n/2, 3)`` containing the positions of the
        faces of the hexagonal lattice.  All positions lie within the
        primary unit cell given by `box`.  The faces are sorted by the x
        position as primary sort order, the y position as secondary sort
        order and the z position as tertiary sort order.

    Notes
    -----
    The lattice must lie flat in xy plane and continue properly across
    periodic boundaries.
    """
    check_hex_lattice(verts=verts, r0=r0, box=box, flatside=flatside, tol=tol)
    verts = mdadist.apply_PBC(verts, box=box)
    faces = np.copy(verts)
    if flatside == 'x':
        faces[:, 0] += r0
    elif flatside == 'y':
        faces[:, 1] += r0
    else:
        raise ValueError(
            "`flatside` must be either x or y, but you gave"
            " {}".format(flatside)
        )
    faces = mdadist.apply_PBC(faces, box=box)
    precision = int(np.ceil(-np.log10(tol)))
    np.round(faces, precision, out=faces)
    np.round(verts, precision, out=verts)
    if flatside == 'x':
        valid = np.isin(faces[:, 0], verts[:, 0], invert=True)
    elif flatside == 'y':
        valid = np.isin(faces[:, 1], verts[:, 1], invert=True)
    faces = faces[valid]
    if 2 * len(faces) != len(verts):
        raise ValueError(
            "The number of hexagon faces ({}) is not half the number of"
            " vertices ({}).  This should not have"
            " happened".format(len(faces), len(verts))
        )
    ix_sort = np.lexsort(faces[:, ::-1].T)
    return faces[ix_sort]


def get_1st_hex_face_col(verts, r0, box, tol):
    r"""
    Get the positions of the hexagon faces in the first column of a
    hexagonal lattice.

    Parameters
    ----------
    verts : numpy.ndarray
        Array of shape ``(n, 3)`` containing the positions of all ``n``
        vertices of the hexagonal lattice.
    r0 : scalar, optional
        Side length of the hexagons.  Note that the side length of the
        hexagons is related to the lattice constant `a` via
        :math:`a = 2 r_0 \sin{(60°)} = r_0 \sqrt{3}`.
    box : array_like, optional
        The unit cell dimensions of the system, which must be
        orthogonal.  They must be provided in the same format as
        returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.
    tol : scalar, optional
        Two floating point numbers are regarded as equal if they deviate
        by less than the tolerance given here.

    Returns
    -------
    hex_face_col : numpy.ndarray
        Array of shape ``(n, 3)``, where ``n`` is the number of hexagons
        per column, containing the positions of the hexagon faces in the
        first column of the lattice.  All faces lie within the primary
        unit cell and their positions are sorted by their y coordinates.

    Notes
    -----
    The lattice must lie flat in xy plane and continue properly across
    periodic boundaries.  The edges of the hexagons of the lattice must
    align with the x axis.
    """
    mdt.check.box(box=box, with_angles=True, orthorhombic=True, dim=1)
    check_hex_lattice(verts=verts, r0=r0, box=box, flatside='x', tol=tol)
    hex_faces = hex_verts2faces(
        verts=verts, r0=r0, box=box, flatside='x', tol=tol
    )
    xmin = np.min(hex_faces[:, 0])
    ix = np.isclose(hex_faces[:, 0], xmin, rtol=0, atol=args.TOL)
    hex_face_col = hex_faces[ix]
    a0 = r0 * np.sqrt(3)  # Lattice constant
    if not np.isclose(len(hex_face_col), box[1] / a0, rtol=0, atol=tol):
        raise ValueError(
            "The number of hexagons per column is {} but should be"
            " {}".format(len(hex_face_col), box[1] / a0)
        )
    return hex_face_col


def get_1st_hex_face_rows(verts, r0, box, tol):
    r"""
    Get the positions of the hexagon faces in the first two staggered
    rows of a hexagonal lattice.

    Parameters
    ----------
    verts : numpy.ndarray
        Array of shape ``(n, 3)`` containing the positions of all ``n``
        vertices of the hexagonal lattice.
    r0 : scalar, optional
        Side length of the hexagons.  Note that the side length of the
        hexagons is related to the lattice constant `a` via
        :math:`a = 2 r_0 \sin{(60°)} = r_0 \sqrt{3}`.
    box : array_like, optional
        The unit cell dimensions of the system, which must be
        orthogonal.  They must be provided in the same format as
        returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.
    tol : scalar, optional
        Two floating point numbers are regarded as equal if they deviate
        by less than the tolerance given here.

    Returns
    -------
    hex_face_rows : numpy.ndarray
        Array of shape ``(2, n, 3)``, where ``n`` is the number of
        hexagons per row, containing the positions of the hexagon faces
        in the first two staggered rows of the lattice.  All faces lie
        within the primary unit cell and their positions are sorted by
        their x coordinates.

    Notes
    -----
    The lattice must lie flat in xy plane and continue properly across
    periodic boundaries.  The edges of the hexagons of the lattice must
    align with the x axis.
    """
    mdt.check.box(box=box, with_angles=True, orthorhombic=True, dim=1)
    check_hex_lattice(verts=verts, r0=r0, box=box, flatside='x', tol=tol)
    hex_faces = hex_verts2faces(
        verts=verts, r0=r0, box=box, flatside='x', tol=tol
    )
    a0 = r0 * np.sqrt(3)  # Lattice constant
    ymin = np.min(hex_faces[:, 1])
    ix = np.isclose(hex_faces[:, 1], ymin, rtol=0, atol=tol)
    hex_face_row1 = hex_faces[ix]
    ix = np.isclose(hex_faces[:, 1], ymin + a0 / 2, rtol=0, atol=tol)
    hex_face_row2 = hex_faces[ix]
    if not np.isclose(len(hex_face_row1), box[0] / (r0 * 3), rtol=0, atol=tol):
        raise ValueError(
            "The number of hexagons per row is {} but should be"
            " {}".format(len(hex_face_row1), box[0] / (r0 * 3))
        )
    if len(hex_face_row2) != len(hex_face_row1):
        raise ValueError(
            "The number of hexagons in row 2 ({}) is not the same as in row 1"
            " ({})".format(len(hex_face_row2), len(hex_face_row1))
        )
    return np.asarray([hex_face_row1, hex_face_row2])


if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process()

    parser = argparse.ArgumentParser(
        description=(
            "Calculate the number density of a selection group"
            " along the first-nearest neighbour axes of a"
            " hexagonal lattice. This script works only for"
            " orthogonal simulation boxes with fixed size and"
            " when the edges of the hexagons of the lattice"
            " align with the x axis."
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
        help="Output filename."
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
        '--sel',
        dest='SEL',
        type=str,
        nargs='+',
        required=True,
        help="Selection group. See MDAnalysis selection commands for"
             " possible choices. E.g. 'type OE'"
    )
    parser.add_argument(
        '--surf',
        dest='SURF',
        type=str,
        nargs='+',
        required=True,
        help="Group of atoms that define a flat hexagonal lattice"
             " (surface) in xy plane. The hexagonal lattice must"
             " continue properly across periodic boundaries. See"
             " MDAnalysis selection commands for possible choices. E.g."
             " 'resname graphene'"
    )
    parser.add_argument(
        '--com',
        dest='COM',
        type=str,
        required=False,
        default=None,
        choices=("group", "segments", "residues", "fragments"),
        help="Use the center of mass for calculating the number density"
             " along the hexagonal axes rather than each individual atom"
             " of the selection group. If 'group', the center of mass of"
             " all atoms in the selection group will be used. Else, the"
             " centers of mass of each segment, residue or fragment of"
             " the selection group will be used. See the MDAnalysis user"
             " guide"
             " (https://userguide.mdanalysis.org/groups_of_atoms.html)"
             " for the definition of these terms. Compounds will be made"
             " whole before calculating their centers of mass. Default"
             " is 'None'."
    )
    parser.add_argument(
        '--zmin',
        dest='ZMIN',
        type=float,
        required=False,
        default=0,
        help="Only consider selection atoms whose z coordinate is equal"
             " to or higher than ZMIN (in Angstroms). Default: 0"
    )
    parser.add_argument(
        '--zmax',
        dest='ZMAX',
        type=float,
        required=False,
        default=None,
        help="Only consider selections atoms whose z coordinate is less"
             " than ZMAX (in Angstroms). Default: z box length"
    )
    parser.add_argument(
        '--r0',
        dest='R0',
        type=float,
        required=False,
        default=1.42,
        help="Side length of the hexagons (in Angstrom). Default: 1.42"
             " (C-C Bond length in graphene)"
    )
    parser.add_argument(
        '--ax-width',
        dest='AX_WIDTH',
        type=float,
        required=False,
        default=None,
        help="Width of the sampling axes (in Angstrom). Should not"
             " exceed R0*3/2, because otherwise the axes overlap."
             " Default: R0/2."
    )
    parser.add_argument(
        '--bin-width',
        dest='BIN_WIDTH',
        type=float,
        required=False,
        default=None,
        help="Bin width to use to divide the sampling axes in bins (in"
             " Angstrom). Should not exceed R0*sqrt(3), because"
             " otherwise the hexagons are not resolved. Note that"
             " R0*sqrt(3) is equal to the lattice constant of the"
             " hexagonal lattice. Default: R0*sqrt(3)/50"
    )
    parser.add_argument(
        '--tol',
        dest='TOL',
        type=float,
        required=False,
        default=1e-3,
        help="Two floating point numbers are regarded as equal if they"
             " deviate by less than the tolerance given here. If you"
             " receive errors that the hexagonal lattice does not"
             " continue properly across periodic boundaries but you are"
             " sure that it does, you can try to increase the tolerance"
             " (but it should never be higher than 0.1). Default: 1e-3"
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

    if args.ZMIN < 0:
        raise ValueError("--zmin ({}) must not be negative"
                         .format(args.ZMIN))
    if args.R0 <= 0:
        raise ValueError("--r0 ({}) must be greater than zero"
                         .format(args.R0))
    A0 = args.R0 * np.sqrt(3)
    if args.AX_WIDTH is None:
        args.AX_WIDTH = args.R0 / 2
    if args.AX_WIDTH <= 0:
        raise ValueError("--ax-width ({}) must be greater than zero"
                         .format(args.AX_WIDTH))
    elif args.AX_WIDTH > args.R0 * 3 / 2:
        warnings.warn("--ax-width ({}) should not exceed {} (=R0*3/2)"
                      .format(args.AX_WIDTH, args.R0 * 3 / 2),
                      RuntimeWarning)
    if args.BIN_WIDTH is None:
        args.BIN_WIDTH = (A0 / 2) / 25
    if args.BIN_WIDTH <= 0:
        raise ValueError("--bin-width ({}) must be greater than zero"
                         .format(args.BIN_WIDTH))
    elif args.BIN_WIDTH > A0:
        warnings.warn("--bin-width ({}) should not exceed {}"
                      " (=R0*sqrt(3))"
                      .format(args.BIN_WIDTH, A0),
                      RuntimeWarning)
    if args.TOL > 0.1:
        warnings.warn("--tol ({}) should not exceed 0.1"
                      .format(args.TOL),
                      RuntimeWarning)

    print("\n\n\n", flush=True)
    u = mdt.select.universe(top=args.TOPFILE,
                            trj=args.TRJFILE,
                            verbose=True)

    print("\n\n\n", flush=True)
    print("Creating selections", flush=True)
    timer = datetime.now()

    sel = u.select_atoms(' '.join(args.SEL))
    surf = u.select_atoms(' '.join(args.SURF))

    print("  Selection group: '{}'"
          .format(' '.join(args.SEL)),
          flush=True)
    print(mdt.rti.ag_info_str(ag=sel, indent=4))
    print(flush=True)
    print("  Surface group: '{}'"
          .format(' '.join(args.SURF)),
          flush=True)
    print(mdt.rti.ag_info_str(ag=surf, indent=4))

    if sel.n_atoms <= 0:
        raise ValueError("The selection group contains no atoms")
    if surf.n_atoms <= 0:
        raise ValueError("The surface group contains no atoms")
    if args.COM is not None:
        print("\n\n\n", flush=True)
        mdt.check.masses(ag=sel, flash_test=False)

    if args.ZMAX is None:
        args.ZMAX = surf.dimensions[2]
    if args.ZMAX > surf.dimensions[2]:
        args.ZMAX = surf.dimensions[2]
        print("\n\n\n", flush=True)
        print("Note: Set --zmax to {}, because it exceeded the z box"
              " length".formt(args.ZMAX), flush=True)
    if args.ZMIN >= args.ZMAX:
        raise ValueError("--zmin ({}) must be less than --zmax ({})"
                         .format(args.ZMIN, args.ZMAX))

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=u.trajectory.n_frames)
    LAST_FRAME = u.trajectory[END - 1].frame

    # Term definitions:
    # x/y axis:       Cartesian x or y axis (enclose 90° with each
    #                 other).  The x axis is per definition the referene
    #                 axis in this script.
    # Hexagonal axis: One of the three hexgonal first-nearest neighbour
    #                 axes.  These three axes enclose angles of 30°, 90°
    #                 and 150° with the x axis, provided that the edges
    #                 of the hexagons align with the x axis like shown
    #                 below:
    #                  y
    #                  ^  ___
    #                  | /   \
    #                  | \___/
    #                  |------->x
    # Sampling axis: One specific instance (realization) of one of the
    #                three hexagonal axes.  A sampling axis is defined
    #                by the position of any hexagon face and the angle
    #                to the x axis.

    # Positions of hexagon centers (faces)
    u.trajectory[BEGIN]
    hex_face_col = get_1st_hex_face_col(verts=surf.positions,
                                        r0=args.R0,
                                        box=surf.dimensions,
                                        tol=args.TOL)
    hex_face_row = get_1st_hex_face_rows(verts=surf.positions,
                                         r0=args.R0,
                                         box=surf.dimensions,
                                         tol=args.TOL)
    hex_face_row = np.concatenate(hex_face_row, axis=0)

    # Define hexgonal first-nearest neighbour axes
    hex_ax_angles = np.array([30, 150])  # 90° needs special treatment
    hex_ax_radians = np.deg2rad(hex_ax_angles)
    hex_ax_slopes = np.tan(hex_ax_radians)
    # y = mx + c  =>  c = y0 - m*x0
    hex_ax_intercepts = (hex_face_col[:, 1][:, None] -
                         hex_ax_slopes * hex_face_col[:, 0][:, None])

    # Number of sampling axes
    n_axes_col = len(hex_face_col)
    n_axes_row = len(hex_face_row)
    n_axes_tot = len(hex_ax_slopes) * n_axes_col + n_axes_row

    # Used later for projection of compound position on hexagonal axes
    hex_ax_sin = np.sin(hex_ax_radians)
    hex_ax_cos = np.cos(hex_ax_radians)
    # hex_ax_tan = hex_ax_slopes
    # shift_x_max = abs(args.AX_WIDTH/2 * hex_ax_sin[0])

    # Maximal axis length when the hexagonal axis is allowed to leave
    # the box in y direction but not in x direction -> x-limited
    ax_lx = abs(surf.dimensions[0] / hex_ax_cos[0])
    if not np.isclose(ax_lx / A0, n_axes_row, rtol=0, atol=args.TOL):
        raise ValueError("ax_lx/A0 ({}/{}={}) is not equal to the number"
                         " of hexagons per row ({}). This means the"
                         " hexagonal lattice does not continue properly"
                         " across periodic boundaries"
                         .format(ax_lx, A0, n_axes_row))
    if args.BIN_WIDTH > ax_lx / 2:
        raise ValueError("The bin width ({}) must be less than half of"
                         " the maximal axis length ({})"
                         .format(args.BIN_WIDTH, ax_lx / 2))
    bins = np.arange(0,
                     ax_lx / 2 + args.BIN_WIDTH / 2,
                     args.BIN_WIDTH,
                     dtype=np.float32)
    # Histograms for axes with 30°, 150° and 90° to x axis
    hists = np.zeros((len(hex_ax_slopes) + 1, len(bins) - 1),
                     dtype=np.uint32)

    print("\n\n\n", flush=True)
    print("Reading trajectory", flush=True)
    print("  Total number of frames in trajectory: {:>9d}"
          .format(u.trajectory.n_frames),
          flush=True)
    print("  Time step per frame:                  {:>9} (ps)\n"
          .format(u.trajectory[0].dt),
          flush=True)
    timer = datetime.now()
    timer_frame = datetime.now()

    if args.COM is None:
        valid_z = np.zeros(sel.n_atoms, dtype=bool)
        valid_z_tmp = np.zeros(sel.n_atoms, dtype=bool)
    elif args.COM == 'group':
        valid_z = np.zeros(1, dtype=bool)
        valid_z_tmp = np.zeros(1, dtype=bool)
    elif args.COM == 'segments':
        valid_z = np.zeros(sel.n_segments, dtype=bool)
        valid_z_tmp = np.zeros(sel.n_segments, dtype=bool)
    elif args.COM == 'residues':
        valid_z = np.zeros(sel.n_residues, dtype=bool)
        valid_z_tmp = np.zeros(sel.n_residues, dtype=bool)
    elif args.COM == 'fragments':
        valid_z = np.zeros(sel.n_fragments, dtype=bool)
        valid_z_tmp = np.zeros(sel.n_fragments, dtype=bool)

    box_prev = surf.dimensions
    surf_pos_prev = surf.positions
    n_surf_moves = 0
    n_surf_moves_dangerous = 0

    for ts in u.trajectory[BEGIN:END:EVERY]:
        if (ts.frame % 10**(len(str(ts.frame)) - 1) == 0 or
                ts.frame == END - 1):
            print("  Frame   {:12d}".format(ts.frame), flush=True)
            print("    Step: {:>12}    Time: {:>12} (ps)"
                  .format(ts.data['step'], ts.data['time']),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now() - timer_frame),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss / 2**20),
                  flush=True)
            timer_frame = datetime.now()

        if not np.allclose(ts.dimensions,
                           box_prev,
                           rtol=0,
                           atol=args.TOL):
            raise ValueError("The simulation box has changed")
        box_prev = ts.dimensions

        if not np.allclose(surf.positions,
                           surf_pos_prev,
                           rtol=0,
                           atol=args.TOL):
            print(flush=True)
            print("  Note: The surface has moved in", flush=True)
            print("  Frame {:12d}".format(ts.frame), flush=True)
            print("  Step: {:>12}    Time: {:>12} (ps)"
                  .format(ts.data['step'], ts.data['time']),
                  flush=True)
            n_surf_moves += 1
            ix_sort1 = np.lexsort(surf.positions.T)
            ix_sort2 = np.lexsort(surf_pos_prev.T)
            if not np.allclose(surf.positions[ix_sort1],
                               surf_pos_prev[ix_sort2],
                               rtol=0,
                               atol=args.TOL):
                print("  This was a dangerous move!", flush=True)
                n_surf_moves_dangerous += 1
            print(flush=True)
            del ix_sort1, ix_sort2
            hex_face_col = get_1st_hex_face_col(verts=surf.positions,
                                                r0=args.R0,
                                                box=ts.dimensions,
                                                tol=args.TOL)
            hex_face_row = get_1st_hex_face_rows(verts=surf.positions,
                                                 r0=args.R0,
                                                 box=ts.dimensions,
                                                 tol=args.TOL)
            hex_face_row = np.concatenate(hex_face_row, axis=0)
        surf_pos_prev = surf.positions

        if args.COM is None:
            pos = mdt.box.wrap(ag=sel, debug=args.DEBUG)
        else:
            mdt.box.make_whole(ag=sel,
                               compound=args.COM,
                               debug=args.DEBUG)
            pos = mdt.strc.com(ag=sel,
                               pbc=True,
                               compound=args.COM,
                               debug=args.DEBUG)

        if args.DEBUG:
            mdt.check.box(box=ts.dimensions,
                          with_angles=True,
                          orthorhombic=True,
                          dim=1)
            mdt.check.pos_array(pos_array=pos,
                                amin=0,
                                amax=ts.dimensions[:3])

        np.greater_equal(pos[:, 2], args.ZMIN, out=valid_z)
        np.less(pos[:, 2], args.ZMAX, out=valid_z_tmp)
        valid_z &= valid_z_tmp
        if not np.any(valid_z):
            continue
        pos = pos[valid_z]
        on_axis = np.zeros(len(pos), dtype=bool)
        lx = ts.dimensions[0]
        ly = ts.dimensions[1]

        # Axes with 30° and 150° to x axis
        for j, hex_face in enumerate(hex_face_col):
            for k, hex_ax_slope in enumerate(hex_ax_slopes):
                hex_ax = mdt.func.line(
                    x=pos[:, 0], m=hex_ax_slope, c=hex_ax_intercepts[j][k]
                )
                # Minimum y distance to sampling axis
                dist_ax = pos[:, 1] - hex_ax
                dist_ax -= np.floor(dist_ax / ly + 0.5) * ly  # MIC
                # Distance orthogonal to sampling axis
                dist_ax *= hex_ax_cos[k]
                # Compounds that lie on the current sampling axis
                np.less(np.abs(dist_ax), args.AX_WIDTH / 2, out=on_axis)
                if not np.any(on_axis):
                    continue
                # x axis is reference axis!
                # Projection of x position on hexagonal axis (necessary
                # due to finite axis width)
                shift_x = dist_ax[on_axis] * hex_ax_sin[k]
                # Distance to reference hexagon face:
                # x component of the distance is of primary interest
                dists_x = pos[:, 0][on_axis] + shift_x
                dists_x -= hex_face[0]
                # y component is determined by x component and axis angle
                dists_y = dists_x * hex_ax_slope  # y = x * tan(phi)
                # Distance in xy plane by Pythagoras' theorem
                np.square(dists_y, out=dists_y)
                np.square(dists_x, out=dists_x)
                dists = dists_x + dists_y
                np.sqrt(dists, out=dists)
                # Minimum image convention along hexagonal axis
                dists -= np.floor(dists / ax_lx + 0.5) * ax_lx
                np.abs(dists, out=dists)
                hist, _ = np.histogram(dists, bins, density=False)
                hists[k] += hist.astype(hists.dtype, copy=False)

        # Axis with 90° to x axis
        for hex_face in hex_face_row:
            # Minimum x distance to sampling axis
            dist_ax = pos[:, 0] - hex_face[0]
            dist_ax -= np.floor(dist_ax / lx + 0.5) * lx  # MIC
            # Compounds that lie on the current sampling axis
            np.less(np.abs(dist_ax), args.AX_WIDTH / 2, out=on_axis)
            if not np.any(on_axis):
                continue
            # Distance to reference hexagon face:
            # Only the y component of the distance is of interest
            dists_y = pos[:, 1][on_axis] - hex_face[1]
            # Minimum image convention along y axis
            dists_y -= np.floor(dists_y / ly + 0.5) * ly
            np.abs(dists_y, out=dists_y)
            hist, _ = np.histogram(dists_y, bins, density=False)
            hists[-1] += hist.astype(hists.dtype, copy=False)

    if n_surf_moves > 0:
        print(flush=True)
        print("  Note: The surface has moved {} times. {} of these\n"
              "  moves were classified as dangerous (i.e. not only due\n"
              "  to wrapping around periodic boundaries)"
              .format(n_surf_moves, n_surf_moves_dangerous),
              flush=True)

    print(flush=True)
    print("Frames read: {}".format(N_FRAMES), flush=True)
    print("First frame: {:>12d}    Last frame: {:>12d}    "
          "Every Nth frame: {:>12d}"
          .format(u.trajectory[BEGIN].frame, LAST_FRAME, EVERY),
          flush=True)
    print("Start time:  {:>12}    End time:   {:>12}    "
          "Every Nth time:  {:>12} (ps)"
          .format(u.trajectory[BEGIN].time,
                  u.trajectory[END - 1].time,
                  u.trajectory[0].dt * EVERY),
          flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()

    bin_vol = args.BIN_WIDTH * args.AX_WIDTH * (args.ZMAX - args.ZMIN)
    # Due to the minimum image convention, every bin is sampled "twice"
    # (positive and negative distance)
    hists = hists / (2 * N_FRAMES * bin_vol)
    hist_tot = np.sum(hists, axis=0)
    hists[:len(hex_ax_slopes)] /= n_axes_col
    hists[-1] /= n_axes_row
    hist_tot /= n_axes_tot

    header = (
        "z_min:      {:>16.9e} A\n"
        "z_max:      {:>16.9e} A\n"
        "Axis width: {:>16.9e} A\n"
        "Bin width:  {:>16.9e} A\n"
        "Bin volume: {:>16.9e} A^3\n"
        "\n"
        "\n"
        "Selection: '{}'\n"
        "  Segments:               {}\n"
        "    Different segments:   {}\n"
        "    Segment name(s):      '{}'\n"
        "  Residues:               {}\n"
        "    Different residues:   {}\n"
        "    Residue name(s):      '{}'\n"
        "  Atoms:                  {}\n"
        "    Different atom names: {}\n"
        "    Atom name(s):         '{}'\n"
        "    Different atom types: {}\n"
        "    Atom type(s):         '{}'\n"
        "  Fragments:              {}\n"
        "\n"
        "\n"
        "Surface: '{}'\n"
        "  Segments:               {}\n"
        "    Different segments:   {}\n"
        "    Segment name(s):      '{}'\n"
        "  Residues:               {}\n"
        "    Different residues:   {}\n"
        "    Residue name(s):      '{}'\n"
        "  Atoms:                  {}\n"
        "    Different atom names: {}\n"
        "    Atom name(s):         '{}'\n"
        "    Different atom types: {}\n"
        "    Atom type(s):         '{}'\n"
        "  Fragments:              {}\n"
        "\n"
        "\n"
        "Nnumber density along the first-nearest neighbour axes of a\n"
        "hexagonal lattice. The hexagonal lattice lies flat in xy plane\n"
        "and the edges of the hexagons align with the x axis.\n"
        "\n"
        "\n"
        "The columns contain:\n"
        "  1 Bin centers (in Angstrom)\n"
        "  2 Number density along first-nearest neighbour axes with\n"
        "    30° to x axis (in 1/A^3)\n"
        "  3 Number density along first-nearest neighbour axes with\n"
        "    150° to x axis (in 1/A^3)\n"
        "  4 Number density along first-nearest neighbour axes with\n"
        "    90° to x axis (in 1/A^3)\n"
        "  5 Sum of all first-nearest neighbour axes\n"
        "\n"
        "Column number:\n"
        "{:>14d} {:>16d} {:>16d} {:>16d} {:>16d}\n"
        "Number of sampling axes:\n"
        "{:>31d} {:>16d} {:>16d} {:>16d}\n"
        .format(args.ZMIN, args.ZMAX,
                args.AX_WIDTH, args.BIN_WIDTH, bin_vol,

                ' '.join(args.SEL),
                sel.n_segments,
                len(np.unique(sel.segids)),
                '\' \''.join(i for i in np.unique(sel.segids)),
                sel.n_residues,
                len(np.unique(sel.resnames)),
                '\' \''.join(i for i in np.unique(sel.resnames)),
                sel.n_atoms,
                len(np.unique(sel.names)),
                '\' \''.join(i for i in np.unique(sel.names)),
                len(np.unique(sel.types)),
                '\' \''.join(i for i in np.unique(sel.types)),
                len(sel.fragments),

                ' '.join(args.SURF),
                surf.n_segments,
                len(np.unique(surf.segids)),
                '\' \''.join(i for i in np.unique(surf.segids)),
                surf.n_residues,
                len(np.unique(surf.resnames)),
                '\' \''.join(i for i in np.unique(surf.resnames)),
                surf.n_atoms,
                len(np.unique(surf.names)),
                '\' \''.join(i for i in np.unique(surf.names)),
                len(np.unique(surf.types)),
                '\' \''.join(i for i in np.unique(surf.types)),
                len(surf.fragments),

                1, 2, 3, 4, 5,
                n_axes_col, n_axes_col, n_axes_row, n_axes_tot
                )
    )
    data = np.column_stack([bins[1:] - np.diff(bins) / 2, hists.T, hist_tot])
    mdt.fh.savetxt(args.OUTFILE, data, header=header)

    print("  Created {}".format(args.OUTFILE), flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print("\n\n\n{} done".format(os.path.basename(sys.argv[0])))
    print("Elapsed time:         {}"
          .format(datetime.now() - timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)
