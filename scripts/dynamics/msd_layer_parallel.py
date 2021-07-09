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
from msd_serial import get_COMs
from msd_layer_serial import parse_user_input, msd_layer


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
    dim = {'x': 0, 'y': 1, 'z': 2}
    d = dim[args.DIRECTION]
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
    elif bins[-1] <= lbox_max:
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

    pool = mdt.parallel.ProcessPool(nprocs=num_CPUs)
    for block in range(NBLOCKS):
        pool.submit_task(func=msd_layer,
                         args=(pos[block * blocksize:(block + 1) * blocksize],
                               bins,
                               args.DIRECTION,
                               effective_restart,
                               args.DEBUG))
    del pos

    md = []
    msd = []
    for result in pool.get_results():
        md.append(result[0])
        msd.append(result[1])
    del result
    pool.close()
    pool.join()

    if len(md) != NBLOCKS:
        raise ValueError("The number of MDs does not equal the number"
                         " of blocks for block averaging. This should"
                         " not have happened")
    if len(msd) != NBLOCKS:
        raise ValueError("The number of MSDs does not equal the number"
                         " of blocks for block averaging. This should"
                         " not have happened")
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
