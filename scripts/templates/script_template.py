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
Script template for scripts that process MD trajectories.

.. deprecated:: 1.6.0

    **Example deprecation warning**.  :mod:`scripts.script_template`
    will be removed in MDTools 2.0.0.  It is replaced by
    :mod:`scripts.script_template_new`, because the latter has
    additional functionality xyz.

.. todo::

    * **Example todo list**.
    * Implement feature xyz.

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
    MDAnalysis' |selection_syntax| for possible choices.
--cmp
    {'group', 'segments', 'residues', 'molecules', 'fragments', 'atoms'}

    The compounds of the selection group to use for the analysis.
    Compounds can be 'group' (the entire selection group), 'segments',
    'residues', 'molecules', 'fragments', or 'atoms'.  Refer to the
    MDAnalysis' user guide for an |explanation_of_these_terms|.  Note
    that in any case, even if ``CMP`` is e.g. 'residues', only the atoms
    belonging to the selection group are taken into account, even if the
    compound might comprise additional atoms that are not contained in
    the selection group.  Default: ``'atoms'``.
--center
    {'cog', 'com', 'coc'}

    The center of the compounds to use for the analysis.

        * ``'cog'``: Center of geometry
        * ``'com'``: Center of mass
        * ``'coc'``: Center of charge

    Note that |MDA_always_guesses_atom_masses| from the atom types, even
    if the input file contains the masses.  Default: ``'cog'``.
--debug
    Run in :ref:`debug mode <debug-mode-label>`.

Output
------
Optional section containing for example a list of files which are
created by the script.

Outfile1 (-o) : .txt
    A text file containing abc.
Outfile2 (\--dtrj-out): .npz
    A compressed |npz_archive| containing a binary NumPy |npy_file|
    called :file:`dtrj.npy` that holds the discrete trajectory.  The
    discrete trajectory is stored as :class:`numpy.ndarray` of dtype
    :attr:`numpy.uint32` and shape ``(n, f)``, where ``n`` is the number
    of reference compounds and ``f`` is the number of frames.  The
    elements of the discrete trajectory are the states in which a given
    compound resides at a given frame.

See Also
--------
:mod:`scripts.templates.script_template_dtrj` :
    Script template for scripts that process discrete trajectories
:mod:`scripts.templates.script_template_plot` :
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
            "Script template for scripts that process MD trajectories.  For"
            " more information, refer to the documentation of this script."
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
        required=True,
        help="Selection string.",
    )
    parser.add_argument(
        "--cmp",
        dest="CMP",
        type=str,
        required=False,
        choices=(
            "group",
            "segments",
            "residues",
            "molecules",
            "fragments",
            "atoms",
        ),
        default="atoms",
        help=(
            "The compounds of the selection group to use for the analysis."
            "  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--center",
        dest="CENTER",
        type=str,
        required=False,
        choices=("cog", "com", "coc"),
        default="cog",
        help=(
            "The center of the compounds to use for the analysis.  Default:"
            " %(default)s."
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
    u = mdt.select.universe(top=args.TOPFILE, trj=args.TRJFILE)
    print("\n")
    sel = mdt.select.atoms(ag=u, sel=" ".join(args.SEL))
    print("\n")
    BEGIN, END, EVERY, N_FRAMES = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=u.trajectory.n_frames,
    )
    first_frame_read = u.trajectory[BEGIN].copy()
    last_frame_read = u.trajectory[END - 1].copy()

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
    for _ts in trj:
        # TODO: Put your computations here (preferably as function).
        #
        # Example for calculating different centers of compounds of an
        # MDAnalysis AtomGroup:
        pos = mdt.strc.center(
            ag=sel,
            center=args.CENTER,
            pbc=True,
            cmp=args.CMP,
            make_whole=True,
            debug=args.DEBUG,
        )
        # ProgressBar update.
        trj.set_postfix_str(
            "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
        )
    trj.close()
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
