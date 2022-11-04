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


r"""
Calculate the net charge of cubic subvolumes of the simulation box.

Divide the simulation box into as many cubic subvolumes of the given
cube lengths as possible and calculate the net charge of all
:class:`Atoms <MDAnalysis.core.groups.Atom>` inside these subvolumes.

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
    MDAnalysis' |selection_syntax| for possible choices.  Default:
    ``'all'``.
--updating-sel
    Use an :class:`~MDAnalysis.core.groups.UpdatingAtomGroup` for the
    analysis.  Selection expressions of UpdatingAtomGroups are
    re-evaluated every :attr:`time step
    <MDAnalysis.coordinates.base.Timestep.dt>`.  This is e.g. useful for
    position-based selections like ``'type Li and prop z <= 2.0'``.
--bin-start
    First cube length to use (in Angstrom).  If set to ``None``,
    \--bin-start is set to \--bin-step.  Default: ``None``.
--bin-stop
    Last cube length to use (in Angstrom).  If set to ``None``, go up to
    the minimum box length for each frame.  Default: ``None``.
--bin-step
    Increment the cube length by this amount (in Angstrom).  Default:
    ``1.0``.

Notes
-----
Works only with orthogonal simulation boxes.
"""


__author__ = "Andreas Thum"


# Standard libraries
import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

# Third-party libraries
import numpy as np
import psutil

# First-party libraries
import mdtools as mdt


def net_charge_of_cubes(ag, cube_lengths, box=None, mean=True):
    """
    Get the net charges of cubic subvolumes.

    Divide the simulation box into as many cubic subvolumes of the given
    cube lengths as possible and calculate the net charge of all
    :class:`Atoms <MDAnalysis.core.groups.Atom>` inside these
    subvolumes.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` or,
        :class:`~MDAnalysis.core.groups.UpdatingAtomGroup` whose
        :class:`Atoms <MDAnalysis.core.groups.Atom>` should be assigned
        to cubic subvolumes.  The net charge of a given subvolume is
        calculated as the sum of the charges of all atoms that lie
        within this subvolume.
    cube_lengths : array_like
        List of cube lengths to use.
    box : array_like, optional
        The unit cell dimensions of the system, which must be provided
        in the same format as returned by
        :attr:`MDAnalysis.coordinates.timestep.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.  If ``None``, the
        :attr:`~MDAnalysis.coordinates.base.Timestep.dimensions` of the
        current :class:`~MDAnalysis.coordinates.base.Timestep` are used.
        This function can only handle orthogonal simulation boxes.
    mean : bool, optional
        If ``True``, return the average net charge for each considered
        cube length.  If ``False``, the net charges of all cubic
        subvolumes with the same cube length are summed up and the sums
        are returned for each considered cube length.

    Returns
    -------
    cube_charges : numpy.ndarray
        1-dimensional array containing the averaged or summed (depends
        on the value of `mean`) net charges of the cubic subvolumes for
        all considered cube lengths.
    cube_charges_squared : numpy.ndarray
        1-dimensional array containing the avarage or sum (depends on
        the value of `mean`) of the squared net charges of the cubic
        subvolumes for all considered cube lengths.
    cube_nums : numpy.ndarray
        1-dimensional array of dtype ``numpy.uint64`` containing the
        number of cubic subvolumes into which the simulation box was
        divided for each considered cube length.
    atm_nums : numpy.ndarray
        1-dimensional array of dtype ``numpy.uint64`` containing the sum
        of the number of atoms in all cubic subvolumes for each
        considered cube length.
    """
    if box is None:
        box = ag.dimensions.asdtype(np.float64)
    mdt.check.box(box, with_angles=True, orthorhombic=True, dim=1)

    ag.wrap(compound="atoms", box=box, inplace=True)
    charge_sums, charge_sums_squared, cube_nums, atm_nums = [], [], [], []
    for cube_length in cube_lengths:
        bin_ix, bins, ag_valid = mdt.strc.assign_atoms_to_grid(
            ag,
            binwidth=cube_length,
            discard_below=True,
            discard_above=True,
            box=box,
            assume_wrapped=True,
            return_bins=True,
            return_ag=True,
        )
        # Total number of atoms in all cubes.
        atm_nums.append(ag_valid.n_atoms)
        # Number of bins/cubes in each spatial direction.
        n_bins = [len(bns) - 1 for bns in bins]
        # Total number of cubes in the simulation box.
        n_cubes = np.prod(n_bins, dtype=np.uint64)
        cube_nums.append(n_cubes)
        # Calculate the net charge of each cube.
        cube_q = np.zeros(n_cubes, dtype=np.float64)
        np.add.at(cube_q, bin_ix, ag_valid.charges.astype(np.float64))
        charge_sums.append(np.sum(cube_q))
        charge_sums_squared.append(np.sum(np.square(cube_q, out=cube_q)))
    del bin_ix, bins, cube_q
    charge_sums = np.asarray(charge_sums)
    charge_sums_squared = np.asarray(charge_sums_squared)
    cube_nums = np.asarray(cube_nums, dtype=np.uint64)
    if np.any(cube_nums < 0):
        raise RuntimeError("Overflow encountered in 'cube_nums'")
    atm_nums = np.asarray(atm_nums, dtype=np.uint64)
    if np.any(atm_nums < 0):
        raise RuntimeError("Overflow encountered in 'atm_nums'")
    if mean:
        charge_sums /= cube_nums
        charge_sums_squared /= cube_nums
    return charge_sums, charge_sums_squared, cube_nums, atm_nums


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage.
    parser = argparse.ArgumentParser(
        description=(
            "TODO: Summary.  For"
            " more information, refer to the documetation of this script."
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
        required=False,
        default=["all"],
        help="Selection string.  Default: %(default)s",
    )
    parser.add_argument(
        "--updating-sel",
        dest="UPDATING_SEL",
        required=False,
        default=False,
        action="store_true",
        help="Use an UpdatingAtomGroup for the analysis.",
    )
    parser.add_argument(
        "--bin-start",
        dest="BIN_START",
        type=float,
        required=False,
        default=None,
        help=(
            "First cube length to use (in Angstrom).  If set to ``None``,"
            " BIN_START is set to BIN_STEP.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--bin-stop",
        dest="BIN_STOP",
        type=float,
        required=False,
        default=None,
        help=(
            "Last cube length to use (in Angstrom).  If set to ``None``,"
            " go up to the minimum box length for each frame.  Default:"
            " %(default)s."
        ),
    )
    parser.add_argument(
        "--bin-step",
        dest="BIN_STEP",
        type=float,
        required=False,
        default=1.0,
        help=(
            "Increment the cube length by this amount (in Angstrom).  Default:"
            " %(default)s."
        ),
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

    print("\n")
    u = mdt.select.universe(top=args.TOPFILE, trj=args.TRJFILE)
    print("\n")
    sel = mdt.select.atoms(
        ag=u, sel=" ".join(args.SEL), updating=args.UPDATING_SEL
    )
    print("\n")
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=u.trajectory.n_frames,
    )
    first_frame_read = u.trajectory[BEGIN].copy()
    last_frame_read = u.trajectory[END - 1].copy()

    box = first_frame_read.dimensions.astype(np.float64)
    lbox_min = np.min(box[:3])
    if args.BIN_START is None:
        start = args.BIN_STEP
    else:
        start = args.BIN_START
    step = args.BIN_STEP
    if args.BIN_STOP is None:
        stop = lbox_min
    print("\n")
    start, stop, step, num = mdt.check.bins(
        start, stop, step, amin=0, amax=lbox_min, verbose=True
    )
    bins_init = np.array([start, step])

    # Guess for the maximum number of different subvolume sizes.  The
    # true number of different subvolume sizes might actually be larger
    # if the trajectory contains a frame with a larger simulation box
    # than in the first frame.
    n_bins_max_guess = num
    n_bins_max_true = 0
    # Net charges of the subvolumes.
    charge_sums = np.zeros(n_bins_max_guess, dtype=np.float64)
    charge_sums_squared = np.zeros_like(charge_sums)
    subvol_nums = np.zeros_like(charge_sums_squared, dtype=np.uint64)
    atm_nums = np.zeros_like(subvol_nums)
    # Net charge of the entire simulation box.
    box_charge_sum = 0
    box_charge_sum_squared = 0
    box_atm_num = 0

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
    trj = mdt.rti.ProgressBar(u.trajectory[BEGIN:END:EVERY])
    for ts in trj:
        box_net_charge = np.sum(sel.charges.astype(np.float64))
        box_charge_sum += box_net_charge
        box_charge_sum_squared += box_net_charge**2
        box_atm_num += sel.n_atoms
        box = ts.dimensions.astype(np.float64)
        lbox_min = np.min(box[:3])
        if args.BIN_STOP is None:
            stop = lbox_min
        start, stop, step, num = mdt.check.bins(
            start, stop, step, num, amin=0, amax=lbox_min, verbose=False
        )
        if not np.allclose([start, step], bins_init, rtol=0):
            raise ValueError(
                "The binning has changed.  Initial start/step: {} / {}.  New"
                " start/step: {} /"
                " {}".format(bins_init[0], bins_init[1], start, step)
            )
        if ((stop - start) / step).is_integer():
            # `numpy.arange` generates values within the half-open
            # interval `[start, stop)`, i.e. `stop` is not included.
            # To include `stop` in the case it falls within the
            # value spacing given by `step`, increase it a bit.
            #
            # The modulo operator suffers from floating point error.
            # E.g. 3.5 % 0.1 is 0.09999999999999981, which is much
            # closer to 0.1 than to the correct value of 0.0.
            # Therefore we use `((stop - start) / step).is_integer()`
            # instead of `np.isclose((stop - start) % step, 0)`.
            stop += step / 2
        cube_lengths = np.arange(start, stop, step)
        # Raise exception if `mdtools.structure.assign_atoms_to_grid`
        # changes the bin width.
        warnings.simplefilter("error", RuntimeWarning)
        charges, charges_squared, n_subvols, n_atms = net_charge_of_cubes(
            sel, cube_lengths=cube_lengths, box=box, mean=False
        )
        # Reset warnings filter.
        warnings.simplefilter("default", RuntimeWarning)
        n_bins_frame = len(charges)
        n_bins_max_true = max(n_bins_max_true, n_bins_frame)
        if n_bins_frame > n_bins_max_guess:
            n_bins_max_guess = n_bins_frame
            charge_sums = mdt.nph.extend(charge_sums, n_bins_frame)
            charge_sums_squared = mdt.nph.extend(
                charge_sums_squared, n_bins_frame
            )
            subvol_nums = mdt.nph.extend(subvol_nums, n_bins_frame)
            atm_nums = mdt.nph.extend(atm_nums, n_bins_frame)
        charge_sums[:n_bins_frame] += charges
        charge_sums_squared[:n_bins_frame] += charges_squared
        subvol_nums[:n_bins_frame] += n_subvols
        atm_nums[:n_bins_frame] += n_atms
        # ProgressBar update.
        trj.set_postfix_str(
            "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
        )
    trj.close()
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    atm_nums = atm_nums[:n_bins_max_true]
    if np.any(atm_nums < 0):
        raise RuntimeError("Overflow encountered in 'atm_nums'")
    if np.any(atm_nums > box_atm_num):
        raise ValueError(
            "Any 'atm_nums' (max: {}) is greater than 'box_atm_num' ({})."
            "  This should not have"
            " happened".format(np.max(atm_nums), box_atm_num)
        )
    subvol_nums = subvol_nums[:n_bins_max_true]
    if np.any(subvol_nums < 0):
        raise RuntimeError("Overflow encountered in 'subvol_nums'")
    # Calculate averages.  (From now on, the suffixe "_sum" is
    # missleading and should actually be "_mean".  But to save memory,
    # and speed up the calculation, we re-use the array).
    charge_sums = charge_sums[:n_bins_max_true]
    charge_sums /= subvol_nums
    charge_sums_squared = charge_sums_squared[:n_bins_max_true]
    charge_sums_squared /= subvol_nums
    box_charge_sum /= N_FRAMES
    box_charge_sum_squared /= N_FRAMES

    stop = start + (n_bins_max_true + 0.5) * step
    cube_lengths = np.arange(start, stop, step)
    while len(cube_lengths) < n_bins_max_true:
        cube_lengths.append(cube_lengths[-1] + step)
    if len(cube_lengths) > n_bins_max_true:
        cube_lengths = cube_lengths[:n_bins_max_true]

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    data = np.column_stack(
        (cube_lengths, charge_sums, charge_sums_squared, subvol_nums, atm_nums)
    )
    header = (
        "Mean net charge <q> of cubic subvolumes of the simulation\n"
        "box as function of the subvolume size.\n"
        "\n"
    )
    header += "Selection:\n"
    header += mdt.rti.ag_info_str(sel, indent=2)
    header += (
        "\n\n"
        "Mean         net charge <q>   of the whole box: {:>16.9e}\n"
        "Mean squared net charge <q^2> of the whole box: {:>16.9e}\n"
        "Number of frames read: {:>15d}\n"
        "Total number of atoms: {:>15d}\n"
        "\n".format(
            box_charge_sum, box_charge_sum_squared, N_FRAMES, box_atm_num
        )
    )
    header += (
        "The columns contain:\n"
        "  1 Cube lengths in Angstrom\n"
        "  2 Mean         net charge <q>\n"
        "  3 Mean squared net charge <q^2>\n"
        "  4 Total number of cubic subvolumes used for averaging\n"
        "  5 Total number of atoms in all subvolumes\n"
        "\n"
        "{:>14d}".format(1)
    )
    for col_num in range(2, data.shape[1] + 1 - 2):
        header += " {:>16d}".format(col_num)
    for col_num in range(data.shape[1] + 1 - 2, data.shape[1] + 1):
        header += " {:>22d}".format(col_num)
    fmt = ("%16.9e",) * (data.shape[1] - 2) + ("%22.15e",) * 2
    mdt.fh.savetxt(args.OUTFILE, data, header=header, fmt=fmt)
    print("Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
