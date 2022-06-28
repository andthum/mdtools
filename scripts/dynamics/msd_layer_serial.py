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


"""TODO: Docstring"""


# Standard libraries
import argparse
import os
import sys
import warnings
from datetime import datetime

# Third-party libraries
import numpy as np
import psutil

# First-party libraries
import mdtools as mdt

# Local imports
from msd_serial import get_COMs


# This function is also used by: msd_layer_parallel.py
def parse_user_input(add_description=""):
    description = ("Calculate the mean displacement (MD) and the mean"
                   " square displacement (MSD) for compounds of a"
                   " selection group as function of the initial compound"
                   " position. The MD is computed, since this script is"
                   " usually apllied to anisotropic systems where it might"
                   " happen that the net-displacement might not be zero.")
    parser = argparse.ArgumentParser(
        description=description + add_description
    )

    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="Trajectory file [<.trr/.xtc/.gro/.pdb/.xyz/.mol2/...>]."
             " See supported coordinate formats of MDAnalysis. IMPORTANT:"
             " At least the selection compounds must be unwrapped in"
             " order to get the correct displacements. If you want to"
             " calculate center-of-mass-based displacements, the"
             " selection compounds need to be whole, too. You can use"
             " 'unwrap_trj' to unwrap a wrapped trajectory and make"
             " broken molecules whole."
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
        help="Output filename pattern. There will be created eight (or"
             " sixteen) output files:"
             "<OUTFILE>_msd_layer.txt containing the total MSD;"
             "<OUTFILE>_m[s]dx_layer.txt containing the x-component of"
             " the M[S]D;"
             "<OUTFILE>_m[s]dy_layer.txt containing the y-component of"
             " the M[S]D;"
             "<OUTFILE>_m[s]dz_layer.txt containing the z-component of"
             " the M[S]D."
             " If --nblocks is greater than one eight additional files"
             " are created containing the respective standard deviations."
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
        '--com',
        dest='COM',
        type=str,
        required=False,
        default=None,
        help="Use the center of mass rather than calculating the"
             " displacement for each individual atom of the selection"
             " group. COM can be either 'group', 'segments', 'residues'"
             " or 'fragments'. If 'group', the center of mass of all"
             " atoms in the selection group will be used. Else, the"
             " centers of mass of each segment, residue or fragment of"
             " the selection group will be used. Compounds will NOT be"
             " made whole! The user is responsible for providing a"
             " suitable trajectory. See the MDAnalysis user guide"
             " (https://userguide.mdanalysis.org/groups_of_atoms.html)"
             " for the definition of the terms. Default is 'None'"
    )

    parser.add_argument(
        '-d',
        dest='DIRECTION',
        type=str,
        required=False,
        default='z',
        help="The spatial direction in which to bin the displacements"
             " according to the starting position of the selection"
             " compounds. Must be either x, y or z. Default: z"
    )
    parser.add_argument(
        '--bin-num',
        dest='NUM',
        type=int,
        required=False,
        default=10,
        help="Number of bins to use for discretizing the given spatial"
             " direction. Binning always ranges from zero to the maximum"
             " box length in the given spatial direction. Note that the"
             " bins do not scale with a potentially fluctuating"
             " simulation box. Default: 10"
    )
    parser.add_argument(
        '--bins',
        dest='BINFILE',
        type=str,
        required=False,
        default=None,
        help="ASCII formatted text file containing custom bin edges in"
             " Angstrom. Bin edges are read from the first column, lines"
             " starting with '#' are ignored. Bins do not need to be"
             " equidistant.  --bins takes precedence over --bin-num."
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
        '--nblocks',
        dest='NBLOCKS',
        type=int,
        required=False,
        default=1,
        help="Number of blocks for block averaging. The trajectory will"
             " be split in NBLOCKS equally sized blocks, which will be"
             " analyzed independently, like if they were different"
             " trajectories. Finally, the average and standard deviation"
             " over all blocks will be calculated. Default: 1"
    )
    parser.add_argument(
        '--restart',
        dest='RESTART',
        type=int,
        default=100,
        help="Number of frames between restarting points for calculating"
             " the MD and MSD. This must be an integer multiply of"
             " --every. Default: 100"
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
    if (args.DIRECTION != 'x' and
        args.DIRECTION != 'y' and
            args.DIRECTION != 'z'):
        raise ValueError("-d must be either 'x', 'y' or 'z', but you"
                         " gave {}".format(args.DIRECTION))

    return args


# This function is also used by: msd_layer_parallel.py
def msd_layer(pos, bins, direction='z', restart=1, debug=False):
    """
    Calculate the mean displacement (MD)

    .. math:
        \langle \Delta r_i(\Delta t, z) \rangle = \langle [r_i(t_0 + \Delta t) - r_i(t_0)] \cdot \delta[z0 - z_i(t_0)]\rangle

    and the mean square displacement (MSD) as function of the initial
    particle position :math:`z_0`:

    .. math:
        \langle \Delta r_i^2(\Delta t, z) \rangle = \langle |r_i(t_0 + \Delta t) - r_i(t_0)|^2 \cdot \delta[z0 - z_i(t_0)]\rangle

    The brackets :math:`\langle ... \rangle` denote averaging over all
    particles :math:`i` and over all possible starting times :math:`t_0`.

    Parameters
    ----------
    pos : array_like
        Array of particle positions of shape ``(m, n, 3)``, where ``m``
        is the number of frames and ``n`` is the number of particles.
    bins : array_like
        1-dimensional array containing the bin edges to use for binning
        the initial particle position
    direction : str
        The spatial direction in which to bin the initial particle
        position. Must be either x, y or z.
    restart : int, optional
        Number of frames between restarting points :math:`t_0`.
    debug : bool
        If ``True``, check the input arguments.

    Returns
    -------
    md : numpy.ndarray
        Array of shape ``(m, len(bins)-1, 3)`` containing the three
        spatial components of the mean displacement
        :math:`\langle \Delta r_i(\Delta t, z_0) \rangle` for each bin
        for all possible lag times :math:`\Delta t`.
    msd : numpy.ndarray
        Array of shape ``(m, len(bins)-1, 3)`` containing the three
        spatial components of the mean square displacement
        :math:`\langle \Delta r_i^2(\Delta t, z_0) \rangle` for each bin
        for all possible lag times :math:`\Delta t`.
    bins : numpy.ndarray
        The used bin edges. This is usually the same as the input `bins`.
        However, if ``bins.min()`` is greater than the minimum position
        or ``bins.max()`` is less than the maximum position in the given
        spatial direction, then `bins` is extended by the corresponding
        value(s).
    """

    if debug:
        mdt.check.pos_array(pos, dim=3)
        mdt.check.array(bins, dim=1)
        if direction != 'x' and direction != 'y' and direction != 'z':
            raise ValueError("direction must be either 'x', 'y' or 'z'"
                             .format(direction))
        if restart >= len(pos):
            warnings.warn("The number of frames between restarting"
                          " points ({}) is equal to or larger than the"
                          " total number of frames in pos ({})"
                          .format(restart, len(pos)), RuntimeWarning)

    proc = psutil.Process()
    pos = np.asarray(pos)
    bins = np.unique(bins)

    dim = {'x': 0, 'y': 1, 'z': 2}
    d = dim[direction]
    pos_min = np.min(pos[:, :, d])
    pos_max = np.max(pos[:, :, d])
    if len(bins) == 0:
        raise ValueError("Invalid bins")
    if bins[0] > pos_min:
        bins = np.insert(bins, 0, pos_min)
        print("Note: Inserting new first bin edge: {}"
              .format(bins[0]),
              flush=True)
    if np.isclose(bins[-1], pos_max):
        bins[-1] = pos_max + 1e-9
        print("Note: Changed last bin edge to {}"
              .format(bins[-1]),
              flush=True)
    elif bins[-1] < pos_max:
        bins = np.append(bins, pos_max + 1e-9)
        print("Note: Appending new last bin edge: {}"
              .format(bins[-1]),
              flush=True)

    n_frames = pos.shape[0]
    n_particles = pos.shape[1]
    md = np.zeros((n_frames, len(bins) - 1, 3), dtype=np.float32)
    msd = np.zeros((n_frames, len(bins) - 1, 3), dtype=np.float32)
    norm = np.zeros((n_frames, len(bins) - 1), dtype=np.uint32)
    displ = np.full((n_particles, 3), np.nan, dtype=np.float32)
    mask = np.zeros(n_particles, dtype=bool)

    timer = datetime.now()
    for t0 in range(0, n_frames - 1, restart):
        if t0 % 10**(len(str(t0)) - 1) == 0 or t0 == n_frames - 2:
            print("  Restart {:12d} of {:12d}"
                  .format(t0, n_frames - 2),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now() - timer),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss / 2**20),
                  flush=True)
            timer = datetime.now()

        bin_ix = np.digitize(pos[t0][:, d], bins=bins)
        bin_ix -= 1
        bin_ix_u, counts = np.unique(bin_ix, return_counts=True)
        if np.any(bin_ix_u < 0):
            raise ValueError("At least one particle was assigned to a"
                             " negative bin number. This should not have"
                             " happened")
        if np.any(bin_ix_u >= len(bins) - 1):
            raise ValueError("At least one particle is outside the bin"
                             " range. This should not have happened")
        norm[1:n_frames - t0][:, bin_ix_u] += counts.astype(np.uint32)
        for lag in range(1, n_frames - t0):
            np.subtract(pos[t0 + lag], pos[t0], out=displ)
            for b in bin_ix_u:
                np.equal(bin_ix, b, out=mask)
                md[lag][b] += np.sum(displ[mask], axis=0)
                msd[lag][b] += np.sum(displ[mask]**2, axis=0)

    del displ, mask
    if not np.all(norm[0] == 0):
        raise ValueError("The first element of norm is not zero. This"
                         " should not have happened")
    norm[0] = 1
    md /= norm[:, None:, None]
    msd /= norm[:, None:, None]

    return md, msd, bins


if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process()

    args = parse_user_input()
    dim = {'x': 0, 'y': 1, 'z': 2}
    d = dim[args.DIRECTION]

    print("\n\n\n", flush=True)
    u = mdt.select.universe(top=args.TOPFILE,
                            trj=args.TRJFILE,
                            verbose=True)

    print("\n\n\n", flush=True)
    sel = mdt.select.atoms(ag=u,
                           sel=' '.join(args.SEL),
                           verbose=True)
    if sel.n_atoms == 0:
        raise ValueError("The selection group contains no atoms")
    if args.COM is not None:
        print("\n\n\n", flush=True)
        mdt.check.masses(ag=sel, flash_test=False)

    BEGIN, END, EVERY, n_frames = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=u.trajectory.n_frames)
    NBLOCKS, blocksize = mdt.check.block_averaging(n_blocks=args.NBLOCKS,
                                                   n_frames=n_frames)
    RESTART, effective_restart = mdt.check.restarts(
        restart_every_nth_frame=args.RESTART,
        read_every_nth_frame=EVERY,
        n_frames=blocksize)
    last_frame = u.trajectory[END - 1].frame
    if args.DEBUG:
        print("\n\n\n", flush=True)
        mdt.check.time_step(trj=u.trajectory[BEGIN:END], verbose=True)
    timestep = u.trajectory[BEGIN].dt

    print("\n\n\n", flush=True)
    print("Checking bins", flush=True)
    timer = datetime.now()

    lbox_max = [ts.dimensions[d] for ts in u.trajectory[BEGIN:END:EVERY]]
    lbox_max = np.max(lbox_max)
    if args.BINFILE is None:
        bins = np.linspace(0, lbox_max, args.NUM + 1)
    else:
        bins = np.loadtxt(args.BINFILE, usecols=0)
        bins = np.unique(bins)
    if len(bins) == 0:
        raise ValueError("Invalid bins")
    if bins[0] > 0:
        bins = np.insert(bins, 0, 0)
        print("  Inserting new first bin edge: {}"
              .format(bins[-1]),
              flush=True)
    if np.isclose(bins[-1], lbox_max):
        bins[-1] = lbox_max + 1e-9
        print("  Changed last bin edge to {}"
              .format(bins[-1]),
              flush=True)
    elif bins[-1] < lbox_max:
        bins = np.append(bins, lbox_max + 1e-9)
        print("  Appending new last bin edge: {}"
              .format(bins[-1]),
              flush=True)

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print("\n\n\n", flush=True)
    print("Reading trajectory", flush=True)
    print("  Total number of frames in trajectory: {:>9d}"
          .format(u.trajectory.n_frames),
          flush=True)
    print("  Time step per frame:                  {:>9} (ps)\n"
          .format(u.trajectory[0].dt),
          flush=True)
    timer = datetime.now()

    pos = get_COMs(topfile=args.TOPFILE,
                   trjfile=args.TRJFILE,
                   sel=args.SEL,
                   com=args.COM,
                   begin=BEGIN,
                   end=END,
                   every=EVERY,
                   debug=args.DEBUG)

    if len(pos) != n_frames:
        raise ValueError("The number of position arrays does not equal"
                         " the number of frames to read. This should not"
                         " have happened")

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
    print("Calculating MSD", flush=True)
    timer = datetime.now()
    timer_block = datetime.now()

    md = [None, ] * NBLOCKS
    msd = [None, ] * NBLOCKS
    for block in range(NBLOCKS):
        if block % 10**(len(str(block)) - 1) == 0 or block == NBLOCKS - 1:
            print(flush=True)
            print("  Block   {:12d} of {:12d}"
                  .format(block, NBLOCKS - 1),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now() - timer_block),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss / 2**20),
                  flush=True)
            timer_block = datetime.now()
        md[block], msd[block], _ = msd_layer(
            pos=pos[block * blocksize:(block + 1) * blocksize],
            bins=bins,
            direction=args.DIRECTION,
            restart=effective_restart,
            debug=args.DEBUG)

    del pos
    md = np.asarray(md)
    msd = np.asarray(msd)

    if NBLOCKS > 1:
        md, md_sd = mdt.stats.block_average(md)
        msd, msd_sd = mdt.stats.block_average(msd)
        msd_tot_sd = np.sqrt(np.sum(msd_sd**2, axis=2))  # Assuming x,y,z are uncorrelated
    else:
        md = np.squeeze(md, axis=0)
        msd = np.squeeze(msd, axis=0)
    msd_tot = np.sum(msd, axis=2)
    lag_times = np.arange(0,
                          timestep * blocksize * EVERY,
                          timestep * EVERY,
                          dtype=np.float32)

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()

    header = (
        "The brackets <...> denote averaging over all particles and\n"
        "over all possible restarting points t0. d[...] stands for the\n"
        "Dirac delta function."
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
        "The first colum contains the diffustion times (ps).\n"
        "The first row contains the bin edges used for discretizing\n"
        "the initial compound positions (Angstrom).\n"
        .format(' '.join(args.SEL),
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
                len(sel.fragments)
                )
    )

    # MSDs
    prefix = (
        "Total mean square displacement (MSD) as function of the\n"
        "initial compound position {}0:\n"
        "  <r^2(t,{})> = <|r(t0 + t) - r(t0)|^2 * d[{}-{}(t0)]>\n"
        .format(args.DIRECTION, args.DIRECTION, args.DIRECTION, args.DIRECTION)
    )
    suffix = "The remaining matrix elements contain the respective MSD values.\n"
    mdt.fh.savetxt_matrix(fname=args.OUTFILE + "_msd_layer.txt",
                          data=msd_tot,
                          var1=lag_times,
                          var2=bins[1:],
                          upper_left=bins[0],
                          header=prefix + header + suffix)
    print("  Created {}".format(args.OUTFILE + "_msd_layer.txt"))
    if args.NBLOCKS > 1:
        prefix = (
            "Standard deviation of the total mean square displacement (MSD)\n"
            "as function of the initial compound position {}0:\n"
            "  <r^2(t,{})> = <|r(t0 + t) - r(t0)|^2 * d[{}-{}(t0)]>\n"
            .format(args.DIRECTION, args.DIRECTION, args.DIRECTION, args.DIRECTION)
        )
        suffix = "The remaining matrix elements contain the respective MSD values.\n"
        mdt.fh.savetxt_matrix(fname=args.OUTFILE + "_msd_layer_sd.txt",
                              data=msd_tot_sd,
                              var1=lag_times,
                              var2=bins[1:],
                              upper_left=bins[0],
                              header=prefix + header + suffix)
        print("  Created {}".format(args.OUTFILE + "_msd_layer_sd.txt"))

    for i, x in enumerate(['x', 'y', 'z']):
        prefix = (
            "{}-component of the mean square displacement (MSD) as function\n"
            "of the initial compound position {}0:\n"
            "  <{}^2(t,{})> = <|{}(t0 + t) - {}(t0)|^2 * d[{}-{}(t0)]>\n"
            .format(x, args.DIRECTION, x, args.DIRECTION, x, x, args.DIRECTION, args.DIRECTION)
        )
        suffix = "The remaining matrix elements contain the respective MSD values.\n"
        mdt.fh.savetxt_matrix(fname=args.OUTFILE + "_msd" + x + "_layer.txt",
                              data=msd[:, :, i],
                              var1=lag_times,
                              var2=bins[1:],
                              upper_left=bins[0],
                              header=prefix + header + suffix)
        print("  Created {}".format(args.OUTFILE + "_msd" + x + "_layer.txt"))
        if args.NBLOCKS > 1:
            prefix = (
                "Standard deviation of the {}-component of the mean square\n"
                "displacement (MSD) as function of the initial compound\n"
                "position {}0:\n"
                "  <{}^2(t,{})> = <|{}(t0 + t) - {}(t0)|^2 * d[{}-{}(t0)]>\n"
                .format(x, args.DIRECTION, x, args.DIRECTION, x, x, args.DIRECTION, args.DIRECTION)
            )
            suffix = "The remaining matrix elements contain the respective MSD values.\n"
            mdt.fh.savetxt_matrix(fname=args.OUTFILE + "_msd" + x + "_layer_sd.txt",
                                  data=msd_sd[:, :, i],
                                  var1=lag_times,
                                  var2=bins[1:],
                                  upper_left=bins[0],
                                  header=prefix + header + suffix)
            print("  Created {}".format(args.OUTFILE + "_msd" + x + "_layer_sd.txt"))

    # MDs
    for i, x in enumerate(['x', 'y', 'z']):
        prefix = (
            "{}-component of the mean displacement (MD) as function\n"
            "of the initial compound position {}0:\n"
            "  <{}(t,{})> = <[{}(t0 + t) - {}(t0)] * d[{}-{}(t0)]>\n"
            .format(x, args.DIRECTION, x, args.DIRECTION, x, x, args.DIRECTION, args.DIRECTION)
        )
        suffix = "The remaining matrix elements contain the respective MD values.\n"
        mdt.fh.savetxt_matrix(fname=args.OUTFILE + "_md" + x + "_layer.txt",
                              data=md[:, :, i],
                              var1=lag_times,
                              var2=bins[1:],
                              upper_left=bins[0],
                              header=prefix + header + suffix)
        print("  Created {}".format(args.OUTFILE + "_md" + x + "_layer.txt"))
        if args.NBLOCKS > 1:
            prefix = (
                "Standard deviation of the {}-component of the mean\n"
                "displacement (MD) as function of the initial compound\n"
                "position {}0:\n"
                "  <{}(t,{})> = <[{}(t0 + t) - {}(t0)] * d[{}-{}(t0)]>\n"
                .format(x, args.DIRECTION, x, args.DIRECTION, x, x, args.DIRECTION, args.DIRECTION)
            )
            suffix = "The remaining matrix elements contain the respective MD values.\n"
            mdt.fh.savetxt_matrix(fname=args.OUTFILE + "_md" + x + "_layer_sd.txt",
                                  data=md_sd[:, :, i],
                                  var1=lag_times,
                                  var2=bins[1:],
                                  upper_left=bins[0],
                                  header=prefix + header + suffix)
            print("  Created {}".format(args.OUTFILE + "_md" + x + "_layer_sd.txt"))

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
