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
Archive/Extract one or multiple NumPy
`.npy <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>`__
files in/from one or multiple NumPy
`.npz <https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html>`_
archives.

This script can:

    * Archive one or multiple .npy files in a single compressed .npz
      archive.
    * Archive each given .npy file in its individual compressed .npz
      archive (\--individual).
    * Extract all .npy files from one or more .npz archives.

Options
-------
-f
    Input file name(s).  Space separated list of input file name(s).  By
    default, all .npy files in this list will be archived in a single
    compressed .npz archive and all .npz archives in this list will be
    extracted.
-o
    Output file name.  Name of the created compressed .npz archive.
    Only meaningful when archiving multiple .npy files in a single
    compressed .npy archive.
--npy-names
    Names of the .npy files in the created compressed .npz archive(s).
    Note that the names given here determine how to access the arrays
    when loading the .npz archive with :func:`numpy.load`.  See there
    for furhter details.  If given, the number of names must match the
    number of .npy files given with -f and all names must be unique.
    When \--individual is given, it is also allowed to parse a single
    name that is used for all .npy files.  Ignored, when only extracting
    .npz archives.  Default: Use names of the input .npy files.
--individual
    Archive each given .npy file in its individual compressed .npz
    archive instead of archiving all given .npy files in a single
    compressed .npz archive.

See Also
--------
:func:`numpy.savez_compressed` :
    Save several arrays into a single file in compressed .npz format.
:func:`numpy.load` :
    Load arrays or pickled objects from .npy, .npz or pickled files.
"""  # noqa: E501, W505


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


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Archive/Extract one or multiple NumPy .npy files in/from one or"
            " multiple NumPy .npz archives."
        )
    )
    parser.add_argument(
        "-f",
        dest="INFILES",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Space separated list of input file name(s).  By default, all .npy"
            " files in this list will be archived in a single compressed .npz"
            " archive and all .npz archives in this list will be extracted."
        ),
    )
    parser.add_argument(
        "-o",
        dest="NPZ_OUT",
        type=str,
        required=False,
        default="archive.npz",
        help=(
            "Name of the created compressed .npz archive.  Only meaningful"
            " when archiving multiple .npy files in a single compressed .npy"
            " archive."
        ),
    )
    parser.add_argument(
        "--npy-names",
        dest="NPY_NAMES",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help=(
            "Names of the .npy files in the created compressed .npz"
            " archive(s).  Ignored when only extracting .npz archives."
            "  Default: Use names of the input .npy files."
        ),
    )
    parser.add_argument(
        "--individual",
        dest="INDIVIDUAL",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Archive each given .npy file in its individual compressed .npz"
            " archive instead of archiving all given .npy files in a single"
            " compressed .npz archive."
        ),
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    infiles_npy = [file for file in args.INFILES if file.endswith(".npy")]
    infiles_npz = [file for file in args.INFILES if file.endswith(".npz")]
    if len(infiles_npy) == 0 and len(infiles_npz) == 0:
        raise ValueError("No .npy or .npz files given with -f")
    if args.NPY_NAMES is not None and len(infiles_npy) > 0:
        if len(args.NPY_NAMES) != len(set(args.NPY_NAMES)):
            raise ValueError("All names given with --names must be unique")
        if args.INDIVIDUAL and len(args.NPY_NAMES) == 1:
            args.NPY_NAMES *= len(infiles_npy)
        if len(args.NPY_NAMES) != len(infiles_npy):
            raise ValueError(
                "The number of names given with --names ({}) must match the"
                " number of .npy files given with -f"
                " ({})".format(len(args.NPY_NAMES), len(infiles_npy))
            )

    if len(infiles_npy) > 0:
        print("\n")
        print("Archiving given .npy files...")
        timer = datetime.now()
        arrays = {}
        infiles = mdt.rti.ProgressBar(infiles_npy, unit="files", mininterval=1)
        for i, infile in enumerate(infiles_npy):
            array = np.load(infile, mmap_mode="r", allow_pickle=False)
            if args.NPY_NAMES is None:
                arrays_tmp = {infile[:-4]: array}
            else:
                arrays_tmp = {args.NPY_NAMES[i]: array}
            if args.INDIVIDUAL:
                outfile = infile[:-4] + ".npz"
                mdt.fh.backup(outfile)
                np.savez_compressed(outfile, **arrays_tmp)
            else:
                arrays.update(arrays_tmp)
            # ProgressBar update:
            infiles.set_postfix_str(
                "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
            )
        infiles.close()
        if not args.INDIVIDUAL and len(infiles_npy) > 0:
            mdt.fh.backup(args.NPZ_OUT)
            np.savez_compressed(args.NPZ_OUT, **arrays)
        del array, arrays, arrays_tmp
        print("Elapsed time:         {}".format(datetime.now() - timer))
        print(
            "Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc))
        )

    if len(infiles_npz) > 0:
        print("\n")
        print("Extracting given .npz files...")
        timer = datetime.now()
        infiles = mdt.rti.ProgressBar(infiles_npz, unit="files", mininterval=1)
        for infile in infiles_npz:
            with np.load(infile, mmap_mode="r", allow_pickle=False) as npz:
                for name, array in npz.items():
                    outfile = infile[:-4] + "_" + name + ".npy"
                    mdt.fh.backup(outfile)
                    np.save(outfile, array, allow_pickle=False)
            # ProgressBar update:
            infiles.set_postfix_str(
                "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
            )
        infiles.close()
        del array, npz
        print("Elapsed time:         {}".format(datetime.now() - timer))
        print(
            "Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc))
        )

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
