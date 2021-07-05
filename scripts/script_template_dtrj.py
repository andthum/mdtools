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


# TODO: Write docstring.
r"""
Script template for scripts that process discrete trajectories stored as
:class:`numpy.ndarray`.

.. deprecated:: 1.6.0
    **Example deprication warning**.  :mod:`scripts.script_template_dtrj`
    will be removed in MDTools 2.0.0.  It is replaced by
    :mod:`scripts.cript_template_dtrj_new`, because the latter has
    additional functionality xyz.

.. todo::
    
    * **Example todo list**.
    * Implement feature xyz.

The following is a guide/template on how to write a docstring for a
MDTools script.  For more information see the
:ref:`developers-guide-label` and the |NumPy_docstring_convention|.

The first part of the docstring should contain the following paragraphs
(all separated by a blank line):

    1. One-sentence summary (preferably one line only).
    2. Potential deprication warning.
    3. Potential todo list.
    4. Extended summary clarifying **functionality**, not implementation
       details or background theory (this goes in the Notes section).

Note that you will have to repeat parts of the docstring (especially
the summary and a potentially abbreviated version of the Options section)
when implementing the command-line interface with :mod:`argparse`.

Options
-------
A |rst_option_list| listing all options with which the script can/must
be called and their meaning.

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
--debug     Run in :ref:`debug mode <debug-mode-label>`.

Output
------
Optional section containing for example a list of files which are
created by the script.

Outfile1 (-o) : .txt
    A text file containing abc.
Outfile2 (\--dtrj-out): .npy
    A binary NumPy :file:`.npy` containing the discrete trajectory as
    :class:`numpy.ndarray` of dtype :attr:`numpy.uint32` and shape
    ``(n, f)``, where ``n`` is the number of reference compounds
    and ``f`` is the number of frames.  The elements of the discrete
    trajectory are the states in which a given compound resides at a
    given frame.

See Also
--------
:mod:`scripts.script_template` :
    Script template for scripts that process MD trajectories
:func:`some_function` :
    A function that is not defined in this script, but which helps
    understanding the script's output or what the script does

Notes
-----
Implementation details and background theory, i.e. a detailed
description of the scientific problem which is solved by the script and
particularly how it is solved.

References
----------
Cited references.

Examples
--------
At least one particular use case of the script, optimally with a graph
demonstrating how the generated data can be visualized.
"""


# TODO: Replace by your name (or add your name if you contribute to an
# alreandy existing script.  Use a comma separated list in this case:
# "Author 1, Author 2, Author 3").
__author__ = "Andreas Thum"


# TODO: Import (only!) the libraries you need
# Standard libraries
import sys
import os
import argparse
from datetime import datetime, timedelta
# Third party libraries
import psutil
# import numpy as np
# Local application/library specific imports
import mdtools as mdt


# TODO: Put your function, class or other object definitions here.
# If your object is very generic and might be used in other contexts as
# well, consider adding it to the MDTools core package instead of
# putting it here in this specific script.


if __name__ == '__main__':
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    # TODO: Implement command line interface.
    parser = argparse.ArgumentParser(
        # The description should only contain the short summary from the
        # docstring and a reference to the documetation.
        description=(
            "Script template for scripts that process discrete"
            " trajectories.  For more information, refer to the"
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
    # TODO: Check parsed input arguments if necessary

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
    dtrj = dtrj[:,BEGIN:END:EVERY]
    trans_info_str = mdt.rti.dtrj_trans_info_str(dtrj)
    print("Elapsed time:         {}".format(datetime.now()-timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20))

    print("\n")
    print("Reading trajectory...")
    print("Number of compounds:    {:>8d}".format(N_CMPS))
    print("Total number of frames: {:>8d}".format(N_FRAMES_TOT))
    print("Frames to read:         {:>8d}".format(N_FRAMES))
    print("First frame to read:    {:>8d}".format(BEGIN))
    print("Last frame to read:     {:>8d}".format(END-1))
    print("Read every n-th frame:  {:>8d}".format(EVERY))
    timer = datetime.now()
    dtrj = mdt.rti.ProgressBar(dtrj, unit="compounds")
    for cmp_trj in dtrj:
        # TODO: Put your computations here (preferably as function)
        # ProgressBar update:
        progress_bar_mem = proc.memory_info().rss / 2**20
        dtrj.set_postfix_str("{:>7.2f}MiB".format(progress_bar_mem),
                             refresh=False)
    dtrj.close()
    print("Elapsed time:         {}".format(datetime.now()-timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20))

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    # TODO: Create your output file(s).  When creating text files, use
    # mdtools.file_handler.savetxt or mdtools.file_handler.savetxt_matrix
    print("Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}".format(datetime.now()-timer))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now()-timer_tot))
    print("CPU time:             {}"
          .format(timedelta(seconds=sum(proc.cpu_times()[:4]))))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20))
