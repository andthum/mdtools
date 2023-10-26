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


r"""
Calculate the mean displacement (MD) and the mean squared displacement
(MSD) of a given compound as function of its initial position.

.. todo::

    Do the binning in box coordinates (-> all boxes are cubes with unit
    length) to account for fluctuating simulation boxes.

Options
-------
-f
    Trajectory file.  See |supported_coordinate_formats| of MDAnalysis.
    **Important**: This scripts requires an unwrapped trajectory!  At
    least the selection compounds must be unwrapped in order to get the
    correct displacements.  If you want to calculate center-of-mass
    displacements, the selection compounds need to be whole, too.  You
    can use e.g. :mod:`scripts.trajectory.unwrap_trj` to unwrap a
    wrapped trajectory and make broken molecules whole.
-s
    Topology file.  See |supported_topology_formats| of MDAnalysis.
-o
    Output filename pattern.  There will be created seven output files:

        * :bash:`OUTFILE_msd_layer.txt` contains the total MSD.
        * :bash:`OUTFILE_m[s]dx_layer.txt` contains the x-component of
          the M[S]D.
        * :bash:`OUTFILE_m[s]dy_layer.txt` contains the y-component of
          the M[S]D.
        * :bash:`OUTFILE_m[s]dz_layer.txt` contains the z-component of
          the M[S]D.

    If \--nblocks is greater than one, seven additional files are
    created that contain the respective standard deviations.
-b
    First frame to read from the trajectory.  Frame numbering starts at
    zero.  Default: ``0``.
-e
    Last frame to read from the trajectory.  This is exclusive, i.e. the
    last frame read is actually ``END - 1``.  A value of ``-1`` means to
    read the very last frame.  Default: ``-1``.
--every
    Read every n-th frame from the trajectory.  Default: ``1``.
--nblocks
    Number of blocks for block averaging.  The trajectory will be split
    in ``NBLOCKS`` equally sized blocks, which will be analyzed
    independently, like if they were different trajectories.  Finally,
    the average and standard deviation over all blocks will be
    calculated.  Default: ``1``.
--restart
    Number of frames between restarting points for calculating the MD
    and MSD.  Must be an integer multiple of \--every.  Default:
    ``100``.
--sel
    Selection string to select a group of atoms for the analysis.  See
    MDAnalysis' |selection_syntax| for possible choices.
--com
    {None, "group", "segments", "residues", "fragments"}

    If ``None``, use the displacements of each individual atom in the
    selection group for the calculation of the M(S)D's.  Otherwise,
    use the center-of-mass displacements of the given compound.
    Compounds can be 'group' (the entire selection group), 'segments',
    'residues', or 'fragments'.  Refer to the MDAnalysis' user guide for
    an |explanation_of_these_terms|.  Default: ``None``.
    TODO
-d
    {"x", "x", "z"}

    The spatial dimension in which to bin the displacements according to
    the initial position of the selection compounds.  Default: ``"z"``.
--bin-num
    Number of bins to use for discretizing the given spatial dimension.
    Binning always ranges from zero to the maximum box length in the
    given spatial dimension.  Note that the bins do not scale with a
    potentially fluctuating simulation box.  Default: ``10``.
--bins
    Text file containing custom bin edges in Angstrom.  Bin edges are
    read from the first column, lines starting with '#' are ignored.
    Bins do not need to be equidistant.  \--bins takes precedence over
    \--bin-num.
--debug
    Run in :ref:`debug mode <debug-mode-label>`.

See Also
--------
:mod:`scripts.dynamics.msd_layer_parallel` :
    A parallelized version of this script
:mod:`scripts.dynamics.plot_msd_layer` :
    Plot the MSD as function of diffusion time for different initial
    particle positions
:mod:`scripts.dynamics.plot_msd_layer_heatmap` :
    Plot the MSD as function of the initial particle position and the
    diffusion time in a heatmap
:mod:`scripts.dynamics.plot_msd_layer_cross_section_at_constant_time` :
    Plot (one component of) the MSD as function of the initial particle
    position at a constant diffusion time(s)
:mod:`scripts.dynamics.plot_msd_layer_cross_section_xyz_at_constant_time` :
    Plot the x-, y- and z-component of the MSD as function of the
    initial particle position at a constant diffusion time
:mod:`scripts.dynamics.plot_msd_layer_cross_section_xyz_at_constant_msd` :
    Plot the diffusion time at which the x-, y- and z-component of the
    MSD reach a certain value as function of the initial particle
    position

Notes
-----
The MD is computed, because it might happen that the net-displacement of
compounds in a bin is not zero.  To account for non-zero
net-displacements, the square of the MD should be subtracted from the
MSD to calculate the displacement variance:

.. math::

    \sigma_r^2 = \langle r^2 \rangle - \langle r \rangle^2

"""  # noqa: W505


__author__ = "Andreas Thum"


# Standard libraries
import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

# Third-party libraries
import MDAnalysis.lib.distances as mdadist
import numpy as np
import psutil

# First-party libraries
import mdtools as mdt

# Local imports
from msd_serial import get_COMs


# This function is also used by: msd_layer_parallel.py
def parse_user_input(add_description=""):
    """
    Parse command-line input using :mod:`argparse`.

    Parameters
    ----------
    add_description : str, optional
        Additional text to add to the description of the created
        :class:`argparse.ArgumentParser`.

    Returns
    -------
    args : argparse.Namespace
        The created :class:`argparse.Namespace` object.
    """
    description = (
        "Calculate the mean displacement (MD) and the mean squared"
        " displacement (MSD) of a given compound as function of its initial"
        " position."
    )
    parser = argparse.ArgumentParser(description=description + add_description)
    parser.add_argument(
        "-f",
        dest="TRJFILE",
        type=str,
        required=True,
        help=(
            "Trajectory file.  IMPORTANT: This scripts requires an unwrapped"
            " trajectory!"
        ),
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
        help="Output filename pattern.",
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
        "--nblocks",
        dest="NBLOCKS",
        type=int,
        required=False,
        default=1,
        help="Number of blocks for block averaging.  Default: %(default)s.",
    )
    parser.add_argument(
        "--restart",
        dest="RESTART",
        type=int,
        default=100,
        help=(
            "Number of frames between restarting points for calculating"
            " the MD and MSD.  Must be an integer multiply of --every."
            "  Default: %(default)s."
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
        "--com",
        dest="COM",
        type=str,
        required=False,
        choices=(None, "group", "segments", "residues", "fragments"),
        default=None,
        help=(
            "If 'None', use the displacements of each individual atom in the"
            " selection group for the calculation of the M(S)D's.  Otherwise,"
            " use the center-of-mass displacements of the given compound."
            "  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "-d",
        dest="DIRECTION",
        type=str,
        required=False,
        choices=("x", "x", "z"),
        default="z",
        help=(
            "The spatial dimension in which to bin the displacements according"
            " to the initial position of the selection compounds.  Default:"
            " %(default)s."
        ),
    )
    parser.add_argument(
        "--bin-num",
        dest="NUM",
        type=int,
        required=False,
        default=10,
        help=(
            "Number of bins to use for discretizing the given spatial"
            " dimension. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--bins",
        dest="BINFILE",
        type=str,
        required=False,
        default=None,
        help=(
            "Text file containing custom bin edges in Angstrom.  If provided,"
            " it takes precedence over --bin-num."
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
    return args


# This function is also used by: msd_layer_parallel.py
def msd_layer(pos, boxes, bins, direction="z", restart=1, verbose=True):
    r"""
    Calculate the mean displacement (MD) and the mean squared
    displacement (MSD) as function of the particle's initial position.

    Calculate the mean displacement (MD)

    .. math::

        \langle \Delta r_i(\Delta t, z) \rangle =
        \langle
            [r_i(t_0 + \Delta t) - r_i(t_0)] \cdot \delta[z0 - z_i(t_0)]
        \rangle

    and the mean squared displacement (MSD)

    .. math::

        \langle \Delta r_i^2(\Delta t, z) \rangle =
        \langle
            |r_i(t_0 + \Delta t) - r_i(t_0)|^2 \cdot
            \delta[z0 - z_i(t_0)]
        \rangle

    as function of the initial particle position :math:`z_0`.

    The brackets :math:`\langle ... \rangle` denote averaging over all
    particles :math:`i` and over all given restarting times :math:`t_0`.

    Parameters
    ----------
    pos : array_like
        Array of unwrapped(!) particle positions of shape ``(m, n, 3)``,
        where ``m`` is the number of frames and ``n`` is the number of
        particles.
    boxes : array_like
        Array of simulation boxes of shape ``(m, 6)``, one for each
        frame.  The simulation boxes can be orthogonal or triclinic and
        must be provided in the same format as returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.
    bins : array_like
        1-dimensional array containing the bin edges to use for binning
        the initial particle position.  For binning the particle
        positions, the positions are wrapped back into the primary unit
        cell.
    direction : {'x', 'y', 'z'}
        The spatial direction in which to bin the initial particle
        position.
    restart : int, optional
        Number of frames between restarting points :math:`t_0`.
    verbose : bool, optional
        If ``True``, print progress information to standard output.

    Returns
    -------
    md : numpy.ndarray
        Array of shape ``(m, len(bins)-1, 3)`` containing the three
        spatial components of the mean displacement
        :math:`\langle \Delta r_i(\Delta t, z_0) \rangle` for each bin
        for all possible lag times :math:`\Delta t`.
    msd : numpy.ndarray
        Array of shape ``(m, len(bins)-1, 3)`` containing the three
        spatial components of the mean squared displacement
        :math:`\langle \Delta r_i^2(\Delta t, z_0) \rangle` for each bin
        for all possible lag times :math:`\Delta t`.
    """
    pos = mdt.check.pos_array(pos, dim=3)
    boxes = mdt.check.box(boxes, with_angles=True, dim=2)
    bins = np.asarray(bins)
    if direction not in ("x", "y", "z"):
        raise ValueError(
            "`direction` must be either 'x', 'y' or 'z', but you gave"
            " '{}'".format(direction)
        )
    if restart >= len(pos):
        warnings.warn(
            "The number of frames between restarting points ({}) is equal to"
            " or greater than the total number of frames in `pos`"
            " ({})".format(restart, len(pos)),
            RuntimeWarning,
            stacklevel=2,
        )
    dim = {"x": 0, "y": 1, "z": 2}
    ixd = dim[direction]
    if mdt.rti.get_num_CPUs() > 1:
        mda_backend = "OpenMP"
    else:
        mda_backend = "serial"

    n_frames, n_cmps = pos.shape[0], pos.shape[1]
    md = np.zeros((n_frames, len(bins) - 1, 3), dtype=np.float32)
    msd = np.zeros((n_frames, len(bins) - 1, 3), dtype=np.float32)
    norm = np.zeros((n_frames, len(bins) - 1), dtype=np.uint32)
    displ = np.full((n_cmps, 3), np.nan, dtype=np.float32)
    mask = np.zeros(n_cmps, dtype=bool)

    restarts = (t0 for t0 in range(0, n_frames - 1, restart))
    if verbose:
        proc = psutil.Process()
        n_restarts = int(np.ceil(n_frames / restart))
        restarts = mdt.rti.ProgressBar(
            restarts, total=n_restarts, unit="restarts"
        )
    for t0 in restarts:
        pos_wrapped_t0 = mdadist.apply_PBC(
            pos[t0], boxes[t0], backend=mda_backend
        )
        bin_ix = np.digitize(pos_wrapped_t0[:, ixd], bins=bins)
        bin_ix -= 1
        bin_ix_u, counts = np.unique(bin_ix, return_counts=True)
        if np.any(bin_ix_u < 0) or np.any(bin_ix_u >= len(bins) - 1):
            raise ValueError(
                "At least one compound lies outside the bin range"
            )
        norm[1 : n_frames - t0][:, bin_ix_u] += counts.astype(np.uint32)
        for lag in range(1, n_frames - t0):
            np.subtract(pos[t0 + lag], pos[t0], out=displ)
            for b in bin_ix_u:
                np.equal(bin_ix, b, out=mask)
                md[lag][b] += np.sum(displ[mask], axis=0)
                msd[lag][b] += np.sum(displ[mask] ** 2, axis=0)
        if verbose:
            restarts.set_postfix_str(
                "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
            )
    del bin_ix, bin_ix_u, pos_wrapped_t0, displ, mask

    if not np.all(norm[0] == 0):
        raise ValueError(
            "The first element of norm is not zero.  This should not have"
            " happened"
        )
    norm[0] = 1
    md /= norm[:, None:, None]
    msd /= norm[:, None:, None]
    return md, msd


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage.
    args = parse_user_input()
    dim = {"x": 0, "y": 1, "z": 2}
    ixd = dim[args.DIRECTION]

    print("\n")
    u = mdt.select.universe(top=args.TOPFILE, trj=args.TRJFILE)
    print("\n")
    sel = mdt.select.atoms(ag=u, sel=" ".join(args.SEL))
    if args.COM is not None:
        print("\n")
        mdt.check.masses_new(ag=sel, verbose=args.DEBUG)
    print("\n")
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
    boxes = np.array([ts.dimensions for ts in u.trajectory[BEGIN:END:EVERY]])
    lbox_max = np.max(boxes[:, ixd])
    if lbox_max <= 0:
        raise ValueError(
            "Invalid simulation box: The box length ({}) in the given"
            " spatial dimension ({}) is less than or equal to"
            " zero".format(lbox_max, args.DIRECTION)
        )
    if args.BINFILE is None:
        bins = np.linspace(0, lbox_max, args.NUM + 1)
    else:
        bins = np.loadtxt(args.BINFILE, usecols=0)
    bins = mdt.check.bin_edges(bins=bins, amin=0, amax=lbox_max)
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
    pos = get_COMs(
        topfile=args.TOPFILE,
        trjfile=args.TRJFILE,
        sel=args.SEL,
        com=args.COM,
        begin=BEGIN,
        end=END,
        every=EVERY,
        debug=args.DEBUG,
    )
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
    timer_block = datetime.now()
    md = [None] * NBLOCKS
    msd = [None] * NBLOCKS
    for block in range(NBLOCKS):
        if block % 10 ** (len(str(block)) - 1) == 0 or block == NBLOCKS - 1:
            print("Block   {:12d} of {:12d}".format(block, NBLOCKS - 1))
            print("Elapsed time:         {}".format(datetime.now() - timer))
            print(
                "Current memory usage: {:.2f}"
                " MiB".format(mdt.rti.mem_usage(proc))
            )
            timer_block = datetime.now()
        md[block], msd[block] = msd_layer(
            pos=pos[block * blocksize : (block + 1) * blocksize],
            boxes=boxes,
            bins=bins,
            direction=args.DIRECTION,
            restart=effective_restart,
        )
    del pos, boxes
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
    header = (
        "The brackets <...> denote averaging over all particles and over all\n"
        + "possible restarting points t0.  d[...] stands for the Dirac delta\n"
        + "function."
        + "\n\n"
        + "Selection string:   '{:s}'\n".format(" ".join(args.SEL))
        + "Selection compound: '{}'\n".format(args.COM)
        + mdt.rti.ag_info_str(sel)
        + "\n\n\n"
        + "The first column contains the diffusion times (ps).\n"
        + "The first row contains the bin edges used for discretizing the\n"
        + "initial compound positions (in Angstrom).\n"
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
            md[:, :, i],
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
