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
import warnings
from datetime import datetime
import psutil
import argparse
import numpy as np
import MDAnalysis as mda
import mdtools as mdt


# This function is also used by: msd_parallel.py
def parse_user_input(add_description=""):
    description = ("Calculate the mean square displacement (MSD) for"
                   " compounds of a selection group.")
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
             " order to get the correct MSDs. If you want to calculate a"
             " center-of-mass-based MSD, the selection compounds need to"
             " be whole, too. You can use 'unwrap_trj' to unwrap a"
             " wrapped trajectory and make broken molecules whole."
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
        help="Use the center of mass rather than calculating the MSD for"
             " each individual atom of the selection group. COM can be"
             " either 'group', 'segments', 'residues' or 'fragments'. If"
             " 'group', the center of mass of all atoms in the selection"
             " group will be used. Else, the centers of mass of each"
             " segment, residue or fragment of the selection group will"
             " be used. Compounds will NOT be made whole! The user is"
             " responsible for providing a suitable trajectory. See the"
             " MDAnalysis user guide"
             " (https://userguide.mdanalysis.org/groups_of_atoms.html)"
             " for the definition of the terms. Default is 'None'"
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
             " the MSD. This must be an integer multiply of --every."
             " Default: 100"
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

    return args


# This function is also used by: msd_parallel.py, msd_layer_serial.py
def get_COMs(topfile, trjfile, sel, com, begin, end, every, debug):
    """
    Read the trajectory and calculate for each frame the center of mass
    positions of the selection compounds.

    Parameters
    ----------
    See the help texts of the arguments added to
    :class:`argparse.ArgumentParser` in :func:`parse_user_input` or call
    this script with the -h option to print the help texts to standard
    output.

    Returns
    -------
    pos : numpy.ndarray
        Array of center of mass positions of shape ``(m, n, 3)``, where
        ``m`` is the number of frames and ``n`` is the number of
        selection compounds.
    """

    proc = psutil.Process(os.getpid())
    u = mda.Universe(topfile, trjfile)
    sel = u.select_atoms(' '.join(sel))

    begin, end, every, n_frames = mdt.check.frame_slicing(
        start=begin,
        stop=end,
        step=every,
        n_frames_tot=u.trajectory.n_frames)

    if com is None:
        pos = np.full((n_frames, sel.n_atoms, 3),
                      np.nan,
                      dtype=np.float32)
        n_particles = sel.n_atoms
    elif com == 'group':
        pos = np.full(n_frames, np.nan, dtype=np.float32)
        n_particles = 1
    elif com == 'segments':
        pos = np.full((n_frames, sel.n_segments, 3),
                      np.nan,
                      dtype=np.float32)
        n_particles = sel.n_segments
    elif com == 'residues':
        pos = np.full((n_frames, sel.n_residues, 3),
                      np.nan,
                      dtype=np.float32)
        n_particles = sel.n_residues
    elif com == 'fragments':
        pos = np.full((n_frames, sel.n_fragments, 3),
                      np.nan,
                      dtype=np.float32)
        n_particles = sel.n_fragments

    timer_frame = datetime.now()
    for i, ts in enumerate(u.trajectory[begin:end:every]):
        if (ts.frame % 10**(len(str(ts.frame)) - 1) == 0 or
            ts.frame == begin or
                ts.frame == end - 1):
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

        if com is None:
            pos[i] = sel.positions
        else:
            pos[i] = mdt.strc.com(ag=sel, compound=com, debug=debug)

    return pos


# This function is also used by: msd_parallel.py
def calc_msd(pos, restart=1, debug=False):
    """
    Calculate the mean square displacement (MSD):

    .. math:
        \langle \Delta r_i^2(\Delta t) \rangle = \langle |r_i(t_0 + \Delta t) - r_i(t_0)|^2 \rangle

    The brackets :math:`\langle ... \rangle` denote averaging over all
    particles :math:`i` and over all possible starting times :math:`t_0`.

    Parameters
    ----------
    pos : array_like
        Array of particle positions of shape ``(m, n, 3)``, where ``m``
        is the number of frames and ``n`` is the number of particles.
    restart : int, optional
        Number of frames between restarting points :math:`t_0`.
    debug : bool
        If ``True``, check the input arguments.

    Returns
    -------
    msd : numpy.ndarray
        Array of shape ``(m, 3)`` containing the three spatial
        components of :math:`\langle \Delta r_i^2(\Delta t) \rangle`
        for all possible lag times :math:`\Delta t`.
    """

    if debug:
        mdt.check.pos_array(pos, dim=3)
        if restart >= len(pos):
            warnings.warn("The number of frames between restarting"
                          " points ({}) is equal to or larger than the"
                          " total number of frames in pos ({})"
                          .format(restart, len(pos)), RuntimeWarning)

    proc = psutil.Process(os.getpid())
    pos = np.asarray(pos)
    n_frames = pos.shape[0]
    n_particles = pos.shape[1]
    msd = np.zeros((n_frames, 3), dtype=np.float32)
    msd_tmp = np.full((n_particles, 3), np.nan, dtype=np.float32)

    timer_lag = datetime.now()
    for lag in range(1, n_frames):
        if lag % 10**(len(str(lag)) - 1) == 0 or lag == n_frames - 1:
            print("  Lag     {:12d} of {:12d}"
                  .format(lag, n_frames - 1),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now() - timer_lag),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss / 2**20),
                  flush=True)
            timer_lag = datetime.now()

        for t0 in range(0, n_frames - lag, restart):
            np.subtract(pos[t0 + lag], pos[t0], out=msd_tmp)
            np.square(msd_tmp, out=msd_tmp)
            np.sum(msd_tmp, axis=0, out=msd_tmp[0])
            msd[lag] += msd_tmp[0]

    del msd_tmp
    n_restarts = n_frames - np.arange(n_frames, dtype=np.float32)
    n_restarts /= restart
    np.ceil(n_restarts, out=n_restarts)
    msd /= n_restarts[:, None]
    msd /= n_particles

    return msd


if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())

    args = parse_user_input()

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
        raise ValueError("The number of positions arrays does not equal"
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
        msd[block] = calc_msd(
            pos=pos[block * blocksize:(block + 1) * blocksize],
            restart=effective_restart,
            debug=args.DEBUG)

    del pos
    msd = np.asarray(msd)

    if NBLOCKS > 1:
        msd, msd_sd = mdt.stats.block_average(msd)
        msd_tot_sd = np.sqrt(np.sum(msd_sd**2, axis=1))  # Assuming x,y,z are uncorrelated
    else:
        msd = np.squeeze(msd)
    msd_tot = np.sum(msd, axis=1)
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
        "Mean square displacement (MSD):\n"
        "  <r^2(t)> = <|r(t0 + t) - r(t0)|^2>\n"
        "The brackets <...> denote averaging over all particles and\n"
        "over all possible restarting points t0\n"
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

    if NBLOCKS == 1:
        columns = (
            "The columns contain:\n"
            "  1 Diffusion time t (ps)\n"
            "  2 Total MSD <r^2(t)> (A^2)\n"
            "  3 x-component of MSD <x^2(t)> (A^2)\n"
            "  4 y-component of MSD <y^2(t)> (A^2)\n"
            "  5 z-component of MSD <z^2(t)> (A^2)\n"
            "\n"
            "Column number:\n"
            "{:>14d} {:>16d} {:>16d} {:>16d} {:>16d}"
            .format(1, 2, 3, 4, 5)
        )
        data = np.column_stack([lag_times, msd_tot, msd])
    else:
        columns = (
            "The columns contain:\n"
            "  1 Diffusion time t (ps)\n"
            "  2 Total MSD <r^2(t)> (A^2)\n"
            "  3 Standard deviation of <r^2(t)> (A^2)\n"
            "  4 x-component of MSD <x^2(t)> (A^2)\n"
            "  5 Standard deviation of <x^2(t)> (A^2)\n"
            "  6 y-component of MSD <y^2(t)> (A^2)\n"
            "  7 Standard deviation of <y^2(t)> (A^2)\n"
            "  8 z-component of MSD <z^2(t)> (A^2)\n"
            "  9 Standard deviation of <z^2(t)> (A^2)\n"
            "\n"
            "Column number:\n"
            "{:>14d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d}"
            .format(1, 2, 3, 4, 5, 6, 7, 8, 9)
        )
        data = np.column_stack([lag_times,
                                msd_tot,
                                msd_tot_sd,
                                msd[:, 0],
                                msd_sd[:, 0],
                                msd[:, 1],
                                msd_sd[:, 1],
                                msd[:, 2],
                                msd_sd[:, 2]])

    mdt.fh.savetxt(fname=args.OUTFILE,
                   data=data,
                   header=header + columns)

    print("  Created {}".format(args.OUTFILE))
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
