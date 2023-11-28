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


"""
Correct the intermittent behavior of a discrete trajectory.

Take a discrete trajectory and fill gaps between same states with a gap
size smaller or equal to a given maximum gap size with the enclosing
states.  See the Notes section of
:func:`mdtools.dynamics.correct_intermittency` for more details.

Options
-------
-f
    Name of the file containing the discrete trajectory.  The discrete
    trajectory must be stored as :class:`numpy.ndarray` either in a
    binary NumPy |npy_file| or in a (compressed) NumPy |npz_archive|.
    See :func:`mdtools.file_handler.load_dtrj` for more information
    about the requirements for the input file.
-o
    Output filename.
-b
    First frame to read from the discrete trajectory.  Frame numbering
    starts at zero.  Default: ``0``.
-e
    Last frame to read from the discrete trajectory.  This is exclusive,
    i.e. the last frame read is actually ``END - 1``.  A value of ``-1``
    means to read the very last frame.  Default: ``-1``.
--every
    Read every n-th frame from the discrete trajectory.  Default: ``1``.
--intermittency
    Maximum number of frames a compound is allowed to leave its state
    whilst still being considered to be in this state provided that it
    returns to this state after the given number of frames.  In other
    words, a compound is only considered to have left its state if it
    has left it for at least the given number of frames.
--debug
    Run in :ref:`debug mode <debug-mode-label>`.

See Also
--------
:func:`mdtools.dynamics.correct_intermittency` :
    The underlying function used to correct the intermittent behavior.
"""


__author__ = "Andreas Thum"


# Standard libraries
import argparse
import os
import sys
from datetime import datetime, timedelta

# Third-party libraries
import psutil

# First-party libraries
import mdtools as mdt


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage.
    parser = argparse.ArgumentParser(
        description=(
            "Correct the intermittent behavior of a discrete trajectory.  For"
            " more information, refer to the documentation of this script."
        )
    )
    parser.add_argument(
        "-f",
        dest="TRJFILE",
        type=str,
        required=True,
        help=(
            "File containing the discrete trajectory stored as numpy.ndarray"
            " in the binary .npy format or as .npz archive."
        ),
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
        "--intermittency",
        dest="INTERMITTENCY",
        type=int,
        required=True,
        help=(
            "Maximum number of frames a compound is allowed to leave its state"
            " whilst still being considered to be in this state provided that"
            " it returns to this state after the given number of frames."
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
    if args.INTERMITTENCY < 1:
        raise ValueError(
            "--intermittency ({}) must be a positive"
            " integer".format(args.intermittency)
        )

    print("\n")
    print("Loading trajectory...")
    timer = datetime.now()
    dtrj = mdt.fh.load_dtrj(args.TRJFILE)
    N_CMPS, N_FRAMES_TOT = dtrj.shape
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=N_FRAMES_TOT,
    )
    dtrj = dtrj[:, BEGIN:END:EVERY]
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Correcting trajectory for intermittency...")
    print("Number of compounds:    {:>8d}".format(N_CMPS))
    print("Total number of frames: {:>8d}".format(N_FRAMES_TOT))
    print("Frames to read:         {:>8d}".format(N_FRAMES))
    print("First frame to read:    {:>8d}".format(BEGIN))
    print("Last frame to read:     {:>8d}".format(END - 1))
    print("Read every n-th frame:  {:>8d}".format(EVERY))
    dtrj = mdt.dyn.correct_intermittency(
        dtrj.T,
        args.INTERMITTENCY,
        inplace=True,
        verbose=True,
        debug=args.DEBUG,
    )
    dtrj = dtrj.T

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    mdt.fh.save_dtrj(args.OUTFILE, dtrj)
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
