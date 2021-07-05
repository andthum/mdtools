#!/usr/bin/env python3

# This file is part of MDTools.
# Copyright (C) 2021  The MDTools Development Team and all contributors
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
Count the number of transitions leading into or out of a given state.

.. todo::
    
    Finish docstring

Options
-------
-f          Trajectory file.  File containing the discrete trajectory
            stored as :class:`numpy.ndarray` in the binary :file:`.npy`
            format.  The array must be of shape ``(n, f)``, where ``n``
            is the number of compounds and ``f`` is the number of
            frames.  The shape can also be ``(f,)``, in which case the
            array is expanded to shape ``(1, f)``.  All elements of the
            array must be integers or floats whose fractional part is
            zero, because they are interpreted as the indices of the
            states in which a given compound is at a given frame.
-o          Output filename.
-b          First frame to read from the trajectory.  Frame numbering
            starts at zero.  Default: ``0``.
-e          Last frame to read from the trajectory.  This is exclusive,
            i.e. the last frame read is actually ``END - 1``.  A value
            of ``-1`` means to read the very last frame.  Default:
            ``-1``.
--every     Read every n-th frame from the trajectory.  Default: ``1``.

See Also
--------
:func:`mdtools.dtrj.trans_per_state` :
    Function that counts the number of transitions leading into or out
    of a given state
"""


__author__ = "Andreas Thum"


# Standard libraries
import sys
import os
import argparse
from datetime import datetime, timedelta
# Third party libraries
import psutil
import numpy as np
# Local application/library specific imports
import mdtools as mdt


if __name__ == '__main__':
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Count the number of transitions leading into or out of a"
            " given state.  For more information, refer to the"
            " documetation of this script.")
    )
    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="File containing the discrete trajectory stored as"
             " numpy.ndarray in the binary .npy format."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename."
    )
    parser.add_argument(
        '-b',
        dest='BEGIN',
        type=int,
        required=False,
        default=0,
        help="First frame to read from the trajectory.  Frame numbering"
             " starts at zero.  Default: %(default)s."
    )
    parser.add_argument(
        '-e',
        dest='END',
        type=int,
        required=False,
        default=-1,
        help="Last frame to read from the trajectory (exclusive)."
             "  Default: %(default)s."
    )
    parser.add_argument(
        '--every',
        dest='EVERY',
        type=int,
        required=False,
        default=1,
        help="Read every n-th frame from the trajectory.  Default:"
             " %(default)s."
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

    print("\n")
    print("Loading trajectory...")
    timer = datetime.now()
    dtrj = mdt.fh.load_dtrj(args.TRJFILE)
    N_CMPS, N_FRAMES_TOT = dtrj.shape
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=N_FRAMES_TOT
    )
    dtrj = dtrj[:, BEGIN:END:EVERY]
    trans_info_str = mdt.rti.dtrj_trans_info_str(dtrj)
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))

    print("\n")
    print("Reading trajectory...")
    print("Number of compounds:    {:>8d}".format(N_CMPS))
    print("Total number of frames: {:>8d}".format(N_FRAMES_TOT))
    print("Frames to read:         {:>8d}".format(N_FRAMES))
    print("First frame to read:    {:>8d}".format(BEGIN))
    print("Last frame to read:     {:>8d}".format(END - 1))
    print("Read every n-th frame:  {:>8d}".format(EVERY))
    timer = datetime.now()
    trans = (np.diff(dtrj) != 0)
    # States from which transitions start
    trans_start = np.hstack([trans, np.zeros((N_CMPS, 1), dtype=bool)])
    # States in which transitions end
    trans_end = np.hstack([np.zeros((N_CMPS, 1), dtype=bool), trans])
    del trans
    bins = np.arange(np.min(dtrj), np.max(dtrj) + 2)
    hist_start = np.histogram(dtrj[trans_start], bins)
    hist_end = np.histogram(dtrj[trans_end], bins)

    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    # TODO: Create your output file(s).  When creating text files, use
    # mdtools.file_handler.savetxt or mdtools.file_handler.savetxt_matrix
    print("Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    print("CPU time:             {}"
          .format(timedelta(seconds=sum(proc.cpu_times()[:4]))))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20))
