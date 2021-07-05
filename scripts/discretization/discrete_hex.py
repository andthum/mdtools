#!/usr/bin/env python3


# This file is part of MDTools.
# Copyright (C) 2020  Andreas Thum
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




import sys
import os
from datetime import datetime
import psutil
import argparse
import numpy as np
from scipy.spatial import distance_matrix
import MDAnalysis.lib.distances as mdadist
import mdtools as mdt




# A "copy" of this function is used in structure/axial_hex_distribution.py
# Last modified: much before 2021-01-19
def hex_verts2faces(verts, r0=None, flatside='x', box=None):
    """
    Generate the positions of the faces (hexagon centers) of a hexagonal
    lattice from the positions of the vertices of this lattice.

    WARNING:
    The lattice must lie flat within the xy-plane. The z coordinate is
    only used for a potential wrapping into the primary unit cell. In
    fact, if `box` is ``None``, `verts` can also have shape ``(n, 2)``
    instead of ``(n, 3)``.

    Parameters
    ----------
    verts : numpy.ndarray
        Array of shape ``(n, 3)``, where ``n`` is the number of vertices
        of the hexagonal lattice.
    r0 : scalar, optional
        Side length of the hexagons. Note that the side length of the
        hexagons is related to the lattice constant `a` via
        :math:`a = 2 r_0 \sin{(60°)} = \sqrt{3} r_0`. If `r0` is not
        given, it is guessed from the distance between the two closest
        vertices.
    flatside : string, optional
        Specify whether the edges of the hexagons align with the x- or
        y-axis of the simulation box. Default is ``'x'``. WARNING: Only
        tested for ``flatside=='x'``.
    box : array_like, optional
        The unit cell dimensions of the system, which can be orthogonal
        or triclinic and must be provided in the same format as returned
        by :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``. If provided, the lattice
        vertices will be shifted into the primary unit cell before
        calculating the lattice faces. It is ensured that all lattice
        faces will lie within the primary unit cell as well.

    Returns
    -------
    faces : numpy.ndarray
        Array of shape ``(n/2, 3)`` containing the positions of
        the faces of the hexagonal lattice. The faces are sorted by
        their position, whereas the primary sort order is the x position
        and the secondary sort order the y position.
    """
    if verts.shape[1] == 3 and np.any(verts[:, 2] != verts[0, 2]):
        raise ValueError("The lattice must lie flat in xy-plane, but the"
                         " z positions of the lattice vertices are not"
                         " all the same")
    if box is not None:
        verts = mdadist.apply_PBC(verts, box=box)
    if r0 is None:
        d = distance_matrix(verts[:, :2], verts[:, :2])
        r0 = np.min(d[d > 0])
    faces = np.copy(verts)
    if flatside == 'x':
        faces[:, 0] += r0 / 2
        faces[:, 1] += r0 * np.sqrt(3) / 2  # sin(60°) = sqrt(3)/2
    elif flatside == 'y':
        faces[:, 0] += r0 * np.sqrt(3) / 2
        faces[:, 1] += r0 / 2
    else:
        raise ValueError("flatside must be either x or y, but you gave"
                         " {}".format(flatside))
    if box is not None:
        faces = mdadist.apply_PBC(faces, box=box)
        precision = [str(b).split('.')[1] for b in box[:3]]
    else:
        precision = []
    precision.append(str(float(r0)).split('.')[1])
    precision = max(len(p) for p in precision)
    np.round(faces, precision, out=faces)
    if flatside == 'x':
        valid = np.isin(faces[:, 0],
                        np.round(verts[:, 0], precision),
                        invert=True)
    elif flatside == 'y':
        valid = np.isin(faces[:, 1],
                        np.round(verts[:, 1], precision),
                        invert=True)
    faces = faces[valid]
    if 2 * len(faces) != len(verts):
        raise ValueError("The number of hexagon faces is not half the"
                         " number of vertices. This should not have"
                         " happened")
    ix_sort = np.lexsort(faces[:, ::-1].T)
    faces = faces[ix_sort]
    return faces








if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())


    parser = argparse.ArgumentParser(
        description=(
            "Assign the atoms of a selection group to the"
            " hexagons of a hexagonal surface. The surface must"
            " lie in xy-plane. For some basic considerations on"
            " hexagonal grids see"
            " http://www-cs-students.stanford.edu/~amitp/game-programming/grids/#parts"
            " and"
            " https://www.redblobgames.com/grids/hexagons/"
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
        help="Output filename pattern. There will be created three"
             " output files:"
             " <OUTFILE>_lattice_faces.npy containing the xy-positions"
             " of the lattice faces, i.e. the mid points of the hexagons,"
             " in Angstrom stored as numpy.ndarray of shape (m, 2) in"
             " the binary .npy format. m is the mumber of lattice faces;"
             " <OUTFILE>_lattice_vertices.npy containing the"
             " xy-positions of the lattice vertices in Angstrom stored"
             " as numpy.ndarray of shape (2*m, 2) in the binary .npy"
             " format;"
             " <OUTFILE>_traj.npy containing a discretized trajectory"
             " for each particle in the selection group stored as"
             " numpy.ndarray of type numpy.int32 and shape (n, f) in the"
             " binary .npy format. n is the number of particles and f is"
             " the number of frames. The elements of the array are the"
             " indices of the lattice faces at which a given particle"
             " resides at a given time. If the particle is at a given"
             " time not within the [ZMIN; ZMAX) interval, the index will"
             " be set to -1."
    )

    parser.add_argument(
        '--sel',
        dest='SEL',
        type=str,
        nargs='+',
        required=True,
        help="Selection group. See MDAnalysis selection commands for"
             " possible choices. E.g. 'type Li'"
    )
    parser.add_argument(
        '--surf',
        dest='SURF',
        type=str,
        nargs='+',
        required=True,
        help="Atom group containing the hexagonal surface. See"
             " MDAnalysis selection commands for possible choices."
             " E.g. 'resname graphene'"
    )
    parser.add_argument(
        '--com',
        dest='COM',
        type=str,
        required=False,
        default=None,
        help="Use the center of mass for discretization rather than"
             " discretizing the spatial coordinates of each individual"
             " atom of the selection group. COM can be either 'group',"
             " 'segments', 'residues' or 'fragments'. If 'group', the"
             " center of mass of all atoms in the selection group will"
             " be used. Else, the centers of mass of each segment,"
             " residue or fragment of the selection group will be used."
             " Compounds will be made whole before calculating their"
             " centers of mass. See the MDAnalysis user guide"
             " (https://userguide.mdanalysis.org/groups_of_atoms.html)"
             " for the definition of the terms. Default is 'None'"
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
        default=10,
        help="Only consider selections atoms whose z coordinate is less"
             " than ZMAX (in Angstroms). Default: 10"
    )
    parser.add_argument(
        '--r0',
        dest='R0',
        type=float,
        required=False,
        default=1.42,
        help="Side length of the hexagons in Angstrom. Default: 1.42"
             " (C-C Bond length in graphene)"
    )
    parser.add_argument(
        '--flat-side',
        dest='FLATSIDE',
        type=str,
        required=False,
        default="x",
        help="Specify whether the edges of the hexagons align with the"
             " x- or y-axis of the simulation box. Default: x."
             " WARNING: ONLY TESTED FOR x"
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
        '--debug',
        dest='DEBUG',
        required=False,
        default=False,
        action='store_true',
        help="Run in debug mode."
    )


    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())


    if (args.COM is not None and
        args.COM != 'group' and
        args.COM != 'segments' and
        args.COM != 'residues' and
            args.COM != 'fragments'):
        raise ValueError("--com must be either 'group', 'segments',"
                         " 'residues' or 'fragments', but you gave {}"
                         .format(args.COM))
    if args.ZMIN >= args.ZMAX:
        raise ValueError("--zmin must be less than --zmax")
    if args.R0 <= 0:
        raise ValueError("--r0 must be greater than zero")
    if args.FLATSIDE != 'x' and args.FLATSIDE != 'y':
        raise ValueError("--flat-side must be either 'x', or 'y', but"
                         " you gave {}".format(args.FLATSIDE))




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

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)




    BEGIN, END, EVERY, n_frames = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=u.trajectory.n_frames)
    last_frame = u.trajectory[END - 1].frame




    # xy-positions of hexagon centers
    u.trajectory[BEGIN]
    lattice = hex_verts2faces(verts=surf.positions,
                              r0=args.R0,
                              flatside=args.FLATSIDE,
                              box=surf.dimensions)
    lattice = lattice[:, :2]




    # Discretized single-particle trajectories compatible with PyEMMA's
    # MSM model
    if args.COM is None:
        dtrajs = -np.ones((n_frames, sel.n_atoms), dtype=np.int32)
        n_particles = sel.n_atoms
    elif args.COM == 'group':
        dtrajs = -np.ones(n_frames, dtype=np.int32)
        n_particles = 1
    elif args.COM == 'segments':
        dtrajs = -np.ones((n_frames, sel.n_segments), dtype=np.int32)
        n_particles = sel.n_segments
    elif args.COM == 'residues':
        dtrajs = -np.ones((n_frames, sel.n_residues), dtype=np.int32)
        n_particles = sel.n_residues
    elif args.COM == 'fragments':
        dtrajs = -np.ones((n_frames, sel.n_fragments), dtype=np.int32)
        n_particles = sel.n_fragments




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

    surf_pos = surf.positions
    n_surf_moves = 0
    n_surf_moves_dangerous = 0
    for i, ts in enumerate(u.trajectory[BEGIN:END:EVERY]):
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

        if not np.allclose(surf.positions, surf_pos):
            print(flush=True)
            print("  Note: The surface has moved in", flush=True)
            print("  Frame {:12d}".format(ts.frame), flush=True)
            print("  Step: {:>12}    Time: {:>12} (ps)"
                  .format(ts.data['step'], ts.data['time']),
                  flush=True)
            n_surf_moves += 1
            ix_sort1 = np.lexsort(surf.positions[:, ::-1].T)
            ix_sort2 = np.lexsort(surf_pos[:, ::-1].T)
            if not np.allclose(surf.positions[ix_sort1],
                               surf_pos[ix_sort2]):
                print("  This was a dangerous move!", flush=True)
                n_surf_moves_dangerous += 1
            del ix_sort1, ix_sort2
            lattice = hex_verts2faces(verts=surf.positions,
                                      r0=args.R0,
                                      flatside=args.FLATSIDE,
                                      box=surf.dimensions)
            lattice = lattice[:, :2]
        surf_pos = surf.positions

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
                                shape=(n_particles, 3),
                                amin=0,
                                amax=ts.dimensions[:3])

        sel_pos_z = pos[:, 2]
        valid = (sel_pos_z >= args.ZMIN) & (sel_pos_z < args.ZMAX)
        sel_pos_xy = pos[:, :2][valid]
        dists = distance_matrix(sel_pos_xy, lattice)
        lattice_faces = np.argmin(dists, axis=1)
        dtrajs[i][valid] = lattice_faces

    # Trajectories must be C contiguous for PyEMMA's timescales_msm and
    # each individual atom/COM needs its individual trajectory.
    dtrajs = np.ascontiguousarray(dtrajs.T)

    surf_pos = surf_pos[:, :2]
    if n_surf_moves > 0:
        print(flush=True)
        print("  Note: The surface has moved {} times. {} of these\n"
              "  moves were classified as dangerous (i.e. not only due\n"
              "  to wrapping around periodic boundaries)"
              .format(n_surf_moves, n_surf_moves_dangerous),
              flush=True)

    print(flush=True)
    print("Frames read: {}".format(n_frames), flush=True)
    print("First frame: {:>12d}    Last frame: {:>12d}    "
          "Every Nth frame: {:>12d}"
          .format(u.trajectory[BEGIN].frame, last_frame, EVERY),
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

    # Hexagonal lattice faces
    mdt.fh.backup(args.OUTFILE + "_lattice_faces.npy")
    np.save(args.OUTFILE + "_lattice_faces.npy",
            lattice,
            allow_pickle=False)
    print("  Created " + args.OUTFILE + "_lattice_faces.npy", flush=True)

    # Hexagonal lattice vertices
    mdt.fh.backup(args.OUTFILE + "_lattice_vertices.npy")
    np.save(args.OUTFILE + "_lattice_vertices.npy",
            surf_pos,
            allow_pickle=False)
    print("  Created " + args.OUTFILE + "_lattice_vertices.npy", flush=True)

    # Discrete trajectories
    mdt.fh.backup(args.OUTFILE + "_traj.npy")
    np.save(args.OUTFILE + "_traj.npy", dtrajs, allow_pickle=False)
    print("  Created " + args.OUTFILE + "_traj.npy", flush=True)

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
