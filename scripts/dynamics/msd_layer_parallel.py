#!/usr/bin/env python3

# This file is part of MDTools.
# Copyright (C) 2023  The MDTools Development Team and all contributors
# listed in the file AUTHORS.rst
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
Calculate the mean displacement (MD) and the mean squared displacement
(MSD) of a given compound as function of its initial position.

This is a parallelized version of
:mod:`scripts.dynamics.msd_layer_serial`.  See there for further
information.

The number of CPUs to use is specified (in decreasing precedence) by the
environment variable

    * :bash:`OMP_NUM_THREADS`
    * :bash:`SLURM_CPUS_PER_TASK`
    * :bash:`SLURM_JOB_CPUS_PER_NODE`
    * :bash:`SLURM_CPUS_ON_NODE`
    * or python intern by :func:`os.cpu_count`.

Best performance is considered to be reached, when the number of used
CPUs is a multiple of the number of blocks for block averaging.
"""


__author__ = "Andreas Thum"


# Standard libraries
import os
import sys
from datetime import datetime, timedelta

# Third-party libraries
import numpy as np
import psutil

# First-party libraries
import mdtools as mdt

# Local imports
from msd_layer_serial import msd_layer, parse_user_input
from msd_serial import get_COMs


if __name__ == "__main__":  # noqa: 23
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage.
    additional_description = (
        "Calculate the mean displacement (MD) and the mean squared"
        " displacement (MSD) of a given compound as function of its initial"
        " position.  This is a parallelized version of"
        " scripts/dynamics/msd_layer_serial.py.  See there for further"
        " information."
    )
    args = parse_user_input(add_description=additional_description)
    dim = {"x": 0, "y": 1, "z": 2}
    ixd = dim[args.DIRECTION]
    n_cpus = mdt.rti.get_num_CPUs()
    print("\n")
    print("Available CPUs: {}".format(n_cpus))

    print("\n")
    u = mdt.select.universe(top=args.TOPFILE, trj=args.TRJFILE)
    print("\n")
    sel = mdt.select.atoms(ag=u, sel=" ".join(args.SEL))
    if args.COM is not None:
        print("\n")
        mdt.check.masses_new(ag=sel, verbose=args.DEBUG)
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=u.trajectory.n_frames,
    )
    first_frame_read = u.trajectory[BEGIN].copy()
    last_frame_read = u.trajectory[END - 1].copy()
    NBLOCKS, blocksize = mdt.check.block_averaging(
        n_blocks=args.NBLOCKS, n_frames=N_FRAMES
    )
    RESTART, effective_restart = mdt.check.restarts(
        restart_every_nth_frame=args.RESTART,
        read_every_nth_frame=EVERY,
        n_frames=blocksize,
    )
    if args.DEBUG:
        print("\n")
        print("Checking time steps for equality...")
        timer = datetime.now()
        mdt.check.time_step(trj=u.trajectory[BEGIN:END], verbose=True)
        print("Elapsed time:         {}".format(datetime.now() - timer))
        print(
            "Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc))
        )
    timestep = u.trajectory[BEGIN].dt

    print("\n")
    print("Creating/checking bins...")
    timer = datetime.now()
    lbox = u.trajectory[BEGIN].dimensions[ixd]
    if lbox <= 0:
        raise ValueError(
            "Invalid simulation box: The box length ({}) in the given"
            " spatial dimension ({}) is less than or equal to"
            " zero".format(lbox, args.DIRECTION)
        )
    if args.BINFILE is None:
        START, STOP, STEP, NUM = mdt.check.bins(
            start=0, stop=1, num=args.NUM, amin=0, amax=1
        )
        # Create bins in the box coordinate system (0 to 1).
        bins = np.linspace(START, STOP, NUM + 1)
    else:
        bins = np.loadtxt(args.BINFILE, usecols=0)
        bins = np.unique(bins) / lbox  # Convert bins to box coordinates
    bins = mdt.check.bin_edges(bins=bins, amin=0, amax=1)
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Calculating compound positions...")
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
    nchunks = n_cpus
    if nchunks > int(N_FRAMES / 10):
        nchunks = int(N_FRAMES / 10)
    pool = mdt.parallel.ProcessPool(nprocs=nchunks)
    chunk_size = int((END - BEGIN) / nchunks)
    chunk_size -= chunk_size % EVERY
    if chunk_size:  # !=0
        nchunks = int((END - BEGIN) / chunk_size)
    else:
        nchunks = 1

    for chunk in range(nchunks):
        pool.submit_task(
            func=get_COMs,
            args=(
                args.TOPFILE,
                args.TRJFILE,
                args.SEL,
                args.COM,
                BEGIN + chunk * chunk_size,
                BEGIN + (chunk + 1) * chunk_size,
                EVERY,
                args.DEBUG,
            ),
        )
    if BEGIN + (chunk + 1) * chunk_size < END:
        chunk += 1
        pool.submit_task(
            func=get_COMs,
            args=(
                args.TOPFILE,
                args.TRJFILE,
                args.SEL,
                args.COM,
                BEGIN + chunk * chunk_size,
                END,
                EVERY,
                args.DEBUG,
            ),
        )
    elif BEGIN + (chunk + 1) * chunk_size > END:
        raise ValueError(
            "More frames than given with -e were read.  This should not have"
            " happened"
        )

    pos = []
    for result in pool.get_results():
        pos.append(result)
    del result
    pos = np.vstack(pos)
    pool.close()
    pool.join()
    if len(pos) != N_FRAMES:
        raise ValueError(
            "The number of position arrays does not equal the number of frames"
            " to read.  This should not have happened"
        )
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Calculating MD and MSD...")
    timer = datetime.now()
    boxes = np.array([ts.dimensions for ts in u.trajectory[BEGIN:END:EVERY]])
    lbox_mean = np.nanmean(boxes[:, ixd], axis=0)
    pool = mdt.parallel.ProcessPool(nprocs=n_cpus)
    for block in range(NBLOCKS):
        pool.submit_task(
            func=msd_layer,
            args=(
                pos[block * blocksize : (block + 1) * blocksize],
                boxes,
                bins,
                args.DIRECTION,
                effective_restart,
            ),
        )
    del pos, boxes

    md = []
    msd = []
    for result in pool.get_results():
        md.append(result[0])
        msd.append(result[1])
    del result
    pool.close()
    pool.join()
    if len(md) != NBLOCKS:
        raise ValueError(
            "The number of MDs does not equal the number of blocks for block"
            " averaging.  This should not have happened"
        )
    if len(msd) != NBLOCKS:
        raise ValueError(
            "The number of MSDs does not equal the number of blocks for block"
            " averaging.  This should not have happened"
        )

    md = np.asarray(md)
    msd = np.asarray(msd)
    if NBLOCKS > 1:
        md, md_sd = mdt.stats.block_average(md)
        msd, msd_sd = mdt.stats.block_average(msd)
        # Standard deviation of the total MSD assuming the x, y and z
        # dimensions are uncorrelated.
        msd_tot_sd = np.sqrt(np.sum(msd_sd**2, axis=2))
    else:
        md = np.squeeze(md, axis=0)
        msd = np.squeeze(msd, axis=0)
    msd_tot = np.sum(msd, axis=2)
    lag_times = np.arange(
        0, timestep * blocksize * EVERY, timestep * EVERY, dtype=np.float32
    )
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    bins *= lbox_mean  # Convert bins from box to Cartesian coordinates.
    header = (
        "The brackets <...> denote averaging over all particles and\n"
        + "over all possible restarting points t0.  d[...] stands for the\n"
        + "Dirac delta function."
        + "\n"
        + "\n"
        + "Selection string:   '{:s}'\n".format(" ".join(args.SEL))
        + "Selection compound: '{}'\n".format(args.COM)
        + mdt.rti.ag_info_str(sel)
        + "\n\n\n"
        + "The first column contains the diffusion times (ps).\n"
        + "The first row contains the (average) bin edges used for\n"
        + "discretizing the initial compound positions (in Angstrom).\n"
        + "The remaining matrix elements contain the respective "
    )

    # Write MSDs to file.
    # Total MSD.
    prefix = (
        "Total mean squared displacement (MSD) as function of the initial\n"
        "compound position {bin_dim}0:\n"
        "  <r^2(t,{bin_dim})> =\n"
        "  <|r(t0 + t) - r(t0)|^2 * d[{bin_dim}-{bin_dim}(t0)]>\n".format(
            bin_dim=args.DIRECTION
        )
    )
    suffix = "MSD values (in Angstrom^2).\n"
    mdt.fh.savetxt_matrix(
        args.OUTFILE + "_msd_layer.txt",
        msd_tot,
        var1=lag_times,
        var2=bins[1:],
        upper_left=bins[0],
        header=prefix + header + suffix,
    )
    print("Created {}".format(args.OUTFILE + "_msd_layer.txt"))
    if args.NBLOCKS > 1:
        prefix = (
            "Standard deviation of the total mean squared displacement (MSD)\n"
            "as function of the initial compound position {bin_dim}0:\n"
            "  <r^2(t,{bin_dim})> =\n"
            "  <|r(t0 + t) - r(t0)|^2 * d[{bin_dim}-{bin_dim}(t0)]>\n".format(
                bin_dim=args.DIRECTION
            )
        )
        suffix = "MSD standard deviations (in Angstrom^2).\n"
        mdt.fh.savetxt_matrix(
            args.OUTFILE + "_msd_layer_sd.txt",
            msd_tot_sd,
            var1=lag_times,
            var2=bins[1:],
            upper_left=bins[0],
            header=prefix + header + suffix,
        )
        print("Created {}".format(args.OUTFILE + "_msd_layer_sd.txt"))
    # MSDs in each spatial dimension.
    for i, msd_dim in enumerate(dim.keys()):
        prefix = (
            "{msd_dim}-component of the mean squared displacement (MSD) as\n"
            "function of the initial compound position {bin_dim}0:\n"
            "  <{msd_dim}^2(t,{bin_dim})> = \n"
            "  <|{msd_dim}(t0 + t) - {msd_dim}(t0)|^2"
            "  * d[{bin_dim}-{bin_dim}(t0)]>\n".format(
                msd_dim=msd_dim, bin_dim=args.DIRECTION
            )
        )
        suffix = "MSD values (in Angstrom^2).\n"
        mdt.fh.savetxt_matrix(
            args.OUTFILE + "_msd" + msd_dim + "_layer.txt",
            msd[:, :, i],
            var1=lag_times,
            var2=bins[1:],
            upper_left=bins[0],
            header=prefix + header + suffix,
        )
        print(
            "Created {}".format(args.OUTFILE + "_msd" + msd_dim + "_layer.txt")
        )
        if args.NBLOCKS > 1:
            prefix = (
                "Standard deviation of the {msd_dim}-component of the mean\n"
                "squared displacement (MSD) as function of the initial\n"
                "compound position {bin_dim}0:\n"
                "  <{msd_dim}^2(t,{bin_dim})> =\n"
                "  <|{msd_dim}(t0 + t) - {msd_dim}(t0)|^2"
                "  * d[{bin_dim}-{bin_dim}(t0)]>\n".format(
                    msd_dim=msd_dim, bin_dim=args.DIRECTION
                )
            )
            suffix = "MSD standard deviations (in Angstrom^2).\n"
            mdt.fh.savetxt_matrix(
                args.OUTFILE + "_msd" + msd_dim + "_layer_sd.txt",
                msd_sd[:, :, i],
                var1=lag_times,
                var2=bins[1:],
                upper_left=bins[0],
                header=prefix + header + suffix,
            )
            print(
                "Created"
                " {}".format(args.OUTFILE + "_msd" + msd_dim + "_layer_sd.txt")
            )

    # Write MDs to file.
    # MDs in each spatial dimension.
    for i, md_dim in enumerate(dim.keys()):
        prefix = (
            "{md_dim}-component of the mean displacement (MD) as function of\n"
            "the initial compound position {bin_dim}0:\n"
            "  <{md_dim}(t,{bin_dim})> = \n"
            "  <[{md_dim}(t0 + t) - {md_dim}(t0)]"
            "  * d[{bin_dim}-{bin_dim}(t0)]>\n".format(
                md_dim=md_dim, bin_dim=args.DIRECTION
            )
        )
        suffix = "MD values (in Angstrom).\n"
        mdt.fh.savetxt_matrix(
            args.OUTFILE + "_md" + md_dim + "_layer.txt",
            data=md[:, :, i],
            var1=lag_times,
            var2=bins[1:],
            upper_left=bins[0],
            header=prefix + header + suffix,
        )
        print(
            "Created {}".format(args.OUTFILE + "_md" + md_dim + "_layer.txt")
        )
        if args.NBLOCKS > 1:
            prefix = (
                "Standard deviation of the {md_dim}-component of the mean\n"
                "displacement (MD) as function of the initial compound\n"
                "position {bin_dim}0:\n"
                "  <{md_dim}(t,{bin_dim})> = \n"
                "  <[{md_dim}(t0 + t) - {md_dim}(t0)]"
                "  * d[{bin_dim}-{bin_dim}(t0)]>\n".format(
                    md_dim=md_dim, bin_dim=args.DIRECTION
                )
            )
            suffix = "MD standard deviations (in Angstrom).\n"
            mdt.fh.savetxt_matrix(
                args.OUTFILE + "_md" + md_dim + "_layer_sd.txt",
                md_sd[:, :, i],
                var1=lag_times,
                var2=bins[1:],
                upper_left=bins[0],
                header=prefix + header + suffix,
            )
            print(
                "Created"
                " {}".format(args.OUTFILE + "_md" + md_dim + "_layer_sd.txt")
            )
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
