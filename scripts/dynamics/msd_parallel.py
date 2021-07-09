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
import numpy as np
import mdtools as mdt
from msd_serial import parse_user_input, get_COMs, calc_msd


if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())

    additional_description = (
        " This script is parallelized. The number of CPUs to use is"
        " specified (in decreasing precedence) by either one of the"
        " environment variables OMP_NUM_THREADS, SLURM_CPUS_PER_TASK,"
        " SLURM_JOB_CPUS_PER_NODE, SLURM_CPUS_ON_NODE or python intern"
        " by os.cpu_count(). Best performance is considered to be"
        " reached, when the number of used CPUs is a multiple of the"
        " number of blocks for block averaging.")
    args = parse_user_input(add_description=additional_description)
    num_CPUs = mdt.rti.get_num_CPUs()
    print("\n\n\n", flush=True)
    print("Available CPUs: {}".format(num_CPUs), flush=True)

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
                                                   n_frames=n_frames,
                                                   check_CPUs=True)
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

    nchunks = num_CPUs
    if nchunks > int(n_frames / 10):
        nchunks = int(n_frames / 10)
    pool = mdt.parallel.ProcessPool(nprocs=nchunks)

    chunk_size = int((END - BEGIN) / nchunks)
    chunk_size -= chunk_size % EVERY
    if chunk_size:  # !=0
        nchunks = int((END - BEGIN) / chunk_size)
    else:
        nchunks = 1

    for chunk in range(nchunks):
        pool.submit_task(func=get_COMs,
                         args=(args.TOPFILE,
                               args.TRJFILE,
                               args.SEL,
                               args.COM,
                               BEGIN + chunk * chunk_size,
                               BEGIN + (chunk + 1) * chunk_size,
                               EVERY,
                               args.DEBUG))
    if BEGIN + (chunk + 1) * chunk_size < END:
        chunk += 1
        pool.submit_task(func=get_COMs,
                         args=(args.TOPFILE,
                               args.TRJFILE,
                               args.SEL,
                               args.COM,
                               BEGIN + chunk * chunk_size,
                               END,
                               EVERY,
                               args.DEBUG))
    elif BEGIN + (chunk + 1) * chunk_size > END:
        raise ValueError("I've read more frames than given with -e. This"
                         " should not have happened")

    pos = []
    for result in pool.get_results():
        pos.append(result)
    del result
    pos = np.vstack(pos)
    pool.close()
    pool.join()

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

    pool = mdt.parallel.ProcessPool(nprocs=num_CPUs)
    for block in range(NBLOCKS):
        pool.submit_task(func=calc_msd,
                         args=(pos[block * blocksize:(block + 1) * blocksize],
                               effective_restart,
                               args.DEBUG))
    del pos

    msd = []
    for result in pool.get_results():
        msd.append(result)
    del result
    pool.close()
    pool.join()

    if len(msd) != NBLOCKS:
        raise ValueError("The number of MSDs does not equal the number"
                         " of blocks for block averaging. This should"
                         " not have happened")
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
