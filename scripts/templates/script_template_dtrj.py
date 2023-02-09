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


# TODO: Write docstring.
r"""
Script template for scripts that process discrete trajectories.

.. deprecated:: 1.6.0

    **Example deprecation warning**.
    :mod:`scripts.script_template_dtrj` will be removed in MDTools
    2.0.0.  It is replaced by :mod:`scripts.script_template_dtrj_new`,
    because the latter has additional functionality xyz.

.. todo::

    * **Example todo list**.
    * Implement feature xyz.

Discrete trajectories must be stored in arrays.  Arrays that serve as
discrete trajectory must meet the requirements listed in
:func:`mdtools.check.dtrj`.

The following is a guide/template on how to write a docstring for a
MDTools script.  For more information see the |dev_guide| and the
|NumPy_docstring_convention|.

The first part of the docstring should contain the following paragraphs
(all separated by a blank line):

    1. One-sentence summary (preferably one line only).
    2. Potential deprecation warning.
    3. Potential todo list.
    4. Extended summary clarifying **functionality**, not implementation
       details or background theory (this goes in the Notes section).

Note that you will have to repeat parts of the docstring (especially
the summary and a potentially abbreviated version of the Options
section) when implementing the command-line interface with
:mod:`argparse`.

Options
-------
An |RST_option_list| listing all options with which the script can/must
be called and their meaning.

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
--debug
    Run in :ref:`debug mode <debug-mode-label>`.

Output
------
Optional section containing for example a list of files which are
created by the script.

Outfile1 (-o) : .txt
    A text file containing abc.
Outfile2 (\--dtrj-out): .npy
    A compressed |npz_archive| containing a binary NumPy |npy_file|
    called :file:`dtrj.npy` that holds the discrete trajectory.  The
    discrete trajectory is stored as :class:`numpy.ndarray` of dtype
    :attr:`numpy.uint32` and shape ``(n, f)``, where ``n`` is the number
    of reference compounds and ``f`` is the number of frames.  The
    elements of the discrete trajectory are the states in which a given
    compound resides at a given frame.

See Also
--------
:mod:`scripts.script_template` :
    Script template for scripts that process MD trajectories
:mod:`scripts.script_template_plot` :
    Script template for scripts that create plots
:func:`some_function` :
    A function that is not defined in this script, but which helps
    understanding the script's output or what the script does

Notes
-----
Implementation details and background theory, i.e. a detailed
description of the scientific problem which is solved by the script and
particularly how it is solved. [#]_

References
----------
.. [#] Cited references.

Examples
--------
At least one particular use case of the script, optimally with a graph
demonstrating how the generated data can be visualized.
"""


# TODO: Replace by your name (or add your name if you contribute to an
# already existing script.  Use a comma separated list in this case:
# "Author 1, Author 2, Author 3").
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


# Your function and class definitions go here.  If your function/class
# is very generic and might be used in other contexts as well, consider
# adding it to the MDTools core package instead of putting it here in
# this specific script.


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage.
    # TODO: Implement command line interface.
    parser = argparse.ArgumentParser(
        # The description should only contain the short summary from the
        # docstring and a reference to the documentation.
        description=(
            "Script template for scripts that process discrete trajectories."
            "  For more information, refer to the documentation of this"
            " script."
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
        "--debug",
        dest="DEBUG",
        required=False,
        default=False,
        action="store_true",
        help="Run in debug mode.",
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    # TODO: Check parsed input arguments if necessary.

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
    trans_info_str = mdt.rti.dtrj_trans_info_str(dtrj)
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Reading trajectory...")
    print("Number of compounds:    {:>8d}".format(N_CMPS))
    print("Total number of frames: {:>8d}".format(N_FRAMES_TOT))
    print("Frames to read:         {:>8d}".format(N_FRAMES))
    print("First frame to read:    {:>8d}".format(BEGIN))
    print("Last frame to read:     {:>8d}".format(END - 1))
    print("Read every n-th frame:  {:>8d}".format(EVERY))
    timer = datetime.now()
    dtrj = mdt.rti.ProgressBar(dtrj, unit="compounds")
    for _cmp_trj in dtrj:
        # TODO: Put your computations here (preferably as function).
        # <computations>
        # ProgressBar update.
        dtrj.set_postfix_str(
            "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
        )
    dtrj.close()
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    # TODO: Create your output file(s).
    # When creating text files, use mdtools.file_handler.savetxt or
    # mdtools.file_handler.savetxt_matrix.
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
