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
Create a discrete posotion trajectory.

Discretize the positions of compounds of a MDAnalysis
:class:`~MDAnalysis.core.groups.AtomGroup` in a given spatial
direction.

.. todo::

    * Allow to choose between center of mass and center of geometry
      (This feature has to be implemented in
      :func:`mdtools.structure.discrete_pos_trj`).
    * Finish docstring.

Options
-------
-f          Trajectory file.  See |supported_coordinate_formats| of
            MDAnalysis.
-s          Topology file.  See |supported_topology_formats| of
            MDAnalysis.
-o          Output filename for the discrete trajectory.  The discrete
            trajectory is written to a binary :file:`.npy` file of the
            given filename.  The discrete trajectory is stored as
            :class:`numpy.ndarray` of dtype :attr:`numpy.uint32` and
            shape ``(n, f)``, where ``n`` is the number of reference
            compounds and ``f`` is the number of frames.  The elements
            of the discrete trajectory are the states in which a given
            compound resides at a given frame.
--bins-out  Output filename for the bin edges (optional).  If provided,
            the (average) bin edges used for creating the discrete
            trajectory are written to a text file of the given filename.
-b          First frame to read from the trajectory.  Frame numbering
            starts at zero.  Default: ``0``.
-e          Last frame to read from the trajectory.  This is exclusive,
            i.e. the last frame read is actually ``END - 1``.  A value
            of ``-1`` means to read the very last frame.  Default:
            ``-1``.
--every     Read every n-th frame from the trajectory.  Default: ``1``.
--sel       Selection string to select a group of atoms for the analysis.
            See MDAnalysis' |selection_syntax| for possible choices.
--cmp       {'group', 'segments', 'residues', 'fragments', 'atoms'}

            The compounds of the selection group whose center of mass
            positions should be discretized.  Compounds can be 'group'
            (the entire selection group), 'segments', 'residues',
            'fragments', or 'atoms'.  Refer to the MDAnalysis' user
            guide for an |explanation_of_these_terms|.  Compounds are
            made whole before calculating their centers of mass.  The
            centers of mass are wrapped back into the primary unit cell
            before discretizing their positions.  Note that in any case,
            even if ``CMP`` is e.g. 'residues', only the atoms belonging
            to the selection group are taken into account, even if the
            compound might comprise additional atoms that are not
            contained in the selection group.  Default: ``'atoms'``
-d          {'x', 'y', 'z'}

            Direction.  The spatial direction in which to bin the
            positions of the reference compounds.  Default: ``'z'``
--bin-start
            Point (in Angstrom) on the chosen spatial direction to start
            binning.  Note that binning naturally starts at zero (origin
            of the simulation box).  If parsing a start value greater
            than zero, the first bin interval will be ``[0, START)``.
            In this way you can determine the width of the first bin
            independently from the other bins.  Note that ``START`` must
            lie within the simulation box obtained from the first frame
            read and it must be smaller than ``STOP``.  Default: ``0``
--bin-end   Point (in Angstrom) on the chosen spatial direction to stop
            binning.  Note that binning naturally ends at ``lbox + tol``
            (length of the simulation box in the given spatial direction
            plus a small tolerance to account for the right-open bin
            interval).  If parsing a value less than ``lbox``, the last
            bin interval will be ``[STOP, lbox+tol)``.  In this way you
            can determine the width of the last bin independently from
            the other bins.  Note that ``STOP`` must lie within the
            simulation box obtained from the first frame read and it
            must be greater than ``START``.  Default: ``lbox + tol``
--bin-num   Number of equidistant bins (not bin edges!) to use for
            discretizing the given spatial direction between ``START``
            and ``STOP``.  Note that two additional bins, ``[0, START)``
            and ``[STOP, lbox+tol)``, are created if ``START`` is not
            zero and ``STOP`` is not ``lbox``.  Default: ``10``
--bins      Text file containing custom bin edges (in Angstrom).  Bin
            edges are read from the first column, characters following a
            '#' are ignored.  Bins do not need to be equidistant.  All
            bin edges must lie within the simulation box as obtained
            from the first frame read.  If \--bins is given, it takes
            precedence over all other \--bin* flags.
--debug     Run in :ref:`debug mode <debug-mode-label>`.

See Also
--------
:func:`mdtools.structure.discrete_pos_trj` :
    Function that creates a discrete posotion trajectory

Notes
-----
This script simply calls :func:`mdtools.structure.discrete_pos_trj` and
writes its output to disk.

The **simulation box must be orthogonal**, otherwise the discretization
of the center of mass positions of the reference compounds does not work.
For more details about the discretization see the Notes section of
:func:`mdtools.structure.discrete_pos_trj`.

Examples
--------
TODO
"""


__author__ = "Andreas Thum"


# Standard libraries
import sys
import os
from datetime import datetime, timedelta
# Third party libraries
import psutil
import argparse
import numpy as np
# Local application/library specific imports
import mdtools as mdt


if __name__ == "__main__":
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Create a discrete posotion trajectory.  For more"
            " information, refer to the documetation of this script."
        )
    )
    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="Trajectory file."
    )
    parser.add_argument(
        '-s',
        dest='TOPFILE',
        type=str,
        required=True,
        help="Topology file."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename for the discrete trajectory."
    )
    parser.add_argument(
        '--bins-out',
        dest='OUTFILE_BINS',
        type=str,
        required=False,
        default=None,
        help="Output filename for the bin edges (optional)."
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
        '--sel',
        dest='SEL',
        type=str,
        nargs='+',
        required=True,
        help="Selection string."
    )
    parser.add_argument(
        '--cmp',
        dest='CMP',
        type=str,
        required=False,
        choices=('group', 'segments', 'residues', 'fragments', 'atoms'),
        default='atoms',
        help="The compounds of the selection group whose center of mass"
             " positions should be discretized.  Default: %(default)s"
    )
    parser.add_argument(
        '-d',
        dest='DIRECTION',
        type=str,
        required=False,
        choices=('x', 'y', 'z'),
        default='z',
        help="Direction for binning.  Default: %(default)s"
    )
    parser.add_argument(
        '--bin-start',
        dest='START',
        type=float,
        required=False,
        default=0,
        help="Point (in Angstrom) on the chosen spatial direction to"
             " start binning.  Default: %(default)s"
    )
    parser.add_argument(
        '--bin-end',
        dest='STOP',
        type=float,
        required=False,
        default=None,
        help="Point (in Angstrom) on the chosen spatial direction to"
             " stop binning.  Default: lbox+tol"
    )
    parser.add_argument(
        '--bin-num',
        dest='NUM',
        type=int,
        required=False,
        default=10,
        help="Number of equidistant bins (not bin edges!) to use for"
             " discretizing the given spatial direction between START"
             " and STOP.  Default: %(default)s"
    )
    parser.add_argument(
        '--bins',
        dest='BINFILE',
        type=str,
        required=False,
        default=None,
        help="Text file containing custom bin edges (in Angstrom).  If"
             " --bins is given, it takes precedence over all other"
             " --bin* flags."
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
    if args.BINFILE is None:
        bins = None
    else:
        bins = np.loadtxt(args.BINFILE, usecols=0)

    print("\n")
    dtrj, bins, lbox_av, time_step = mdt.strc.discrete_pos_trj(
        sel=' '.join(args.SEL),
        topfile=args.TOPFILE,
        trjfile=args.TRJFILE,
        begin=args.BEGIN,
        end=args.END,
        every=args.EVERY,
        compound=args.CMP,
        direction=args.DIRECTION,
        bin_start=args.START,
        bin_stop=args.STOP,
        bin_num=args.NUM,
        bins=bins,
        return_bins=True,
        return_lbox=True,
        return_dt=True,
        dtype=np.uint32,
        verbose=True,
        debug=args.DEBUG
    )

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    # Discrete
    mdt.fh.backup(args.OUTFILE)
    np.save(args.OUTFILE, dtrj, allow_pickle=False)
    print("Created {}".format(args.OUTFILE))
    # Bin edges
    if args.OUTFILE_BINS is not None:
        header = ("Bin edges in Angstrom\n"
                  "Number of bin edges:                  {:<d}\n"
                  "Number of bins:                       {:<d}\n"
                  "Discretized spatial dimension:        {:<s}\n"
                  "Average box length in this direction: {:<.9e} A\n"
                  "Time step of discrete trajectory:     {:<.3f} ps\n"
                  .format(len(bins),
                          len(bins) - 1,
                          args.DIRECTION,
                          lbox_av,
                          time_step))
        mdt.fh.savetxt(args.OUTFILE_BINS, bins, header=header)
        print("Created {}".format(args.OUTFILE_BINS))
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
