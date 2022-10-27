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
Script template for plotting scripts.

.. deprecated:: 1.6.0

    **Example deprication warning**.
    :mod:`scripts.script_template_plot` will be removed in MDTools
    2.0.0.  It is replaced by :mod:`scripts.script_template_plot_new`,
    because the latter has additional functionality xyz.

.. todo::

    * **Example todo list**.
    * Implement feature xyz.

The following is a guide/template on how to write a docstring for a
MDTools script.  For more information see the |dev_guide| and the
|NumPy_docstring_convention|.

Scripts that create plots from the output of other scripts should have
the same name as the script whose output is plotted, prefixed with
:file:`plot_`.  For instance, a script that plots the output of a script
with the name :file:`msd.py` should have the name :file:`plot_msd.py`.
Suffixes that give further hints on what the script does are allowed,
e.g. :file:`plot_msd_logscale.py`.

The first part of the docstring should contain the following paragraphs
(all separated by a blank line):

    1. One-sentence summary (preferably one line only).
    2. Potential deprication warning.
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
    Input filename.  The input file must be a text file with at least
    two columns of data.
-o
    Output filename.  The image file format is inferred from the
    extension.  The extension .pdf is recommended and will usually
    result in scalable vector graphics.
--cols
    The columns of the input file that should be plotted.  The first
    given column is treated as x data, all other given columns are
    treated as y data.  Column numbering starts at zero.  Default:
    ``None`` (this means read all columns).
--labels
    A label for each set of y data.  The labels will be shown in the
    legend of the plot.  If provided, you must give one label for each
    set of y data.  Labels are assigned to the sets of y data by their
    input order.  If you do not want to label a specific set of y data,
    parse ``None`` for this set.  Default: ``None``.
--xylabel
    x- and y-axis label.  Default: ``[r'$x$', r'$y$']``.
--xlim
    Left and right limit of the x-axis in data coordinates.  Parse
    ``None`` to adjust the limit(s) automatically.  Default:
    ``[None, None]``.
--ylim
    Lower and upper limit of the y-axis in data coordinates.  Parse
    ``None`` to adjust the limit(s) automatically.  Default:
    ``[None, None]``.
--log
    Whether to use a logarithmic scale for the x- and/or y-axis.
    Default:  ``[False, False]``.

Output
------
Optional section containing for example a list of files which are
created by the script.

Outfile (-o) : .pdf
    A plot of y versus x as vector graphic in the portable document
    format (PDF).

See Also
--------
:mod:`scripts.script_template` :
    Script template for scripts that process MD trajectories
:mod:`scripts.script_template_dtrj` :
    Script template for scripts that process discrete trajectories
:func:`some_function` :
    A function that is not defined in this script, but which helps
    understanding the script's output or what the script does

Notes
-----
Implementation details and background theory. [#]_

References
----------
.. [#] Cited references.

Examples
--------
At least one particular use case of the script, optimally with a graph
demonstrating how the plots that are produced by this script look like.
"""


# TODO: Replace by your name (or add your name if you contribute to an
# alreandy existing script.  Use a comma separated list in this case:
# "Author 1, Author 2, Author 3").
__author__ = "Andreas Thum"


# Standard libraries
import argparse
import os
import sys
from datetime import datetime, timedelta

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import psutil

# First-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # noqa: F401; Import MDTools plot style


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
        # docstring and a reference to the documetation.
        description=(
            "Script template for plotting scripts.  For more information,"
            " refer to the documetation of this script."
        )
    )
    parser.add_argument(
        "-f",
        dest="INFILE",
        type=str,
        required=True,
        help="Input filename.",
    )
    parser.add_argument(
        "-o",
        dest="OUTFILE",
        type=str,
        required=True,
        help="Output filename.",
    )
    parser.add_argument(
        "--cols",
        dest="COLS",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help=(
            "The columns of the input file that should be plotted.  Column"
            " numbering starts at zero.  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--labels",
        dest="LABELS",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help="A label for each set of y data.  Default: %(default)s",
    )
    parser.add_argument(
        "--xylabel",
        dest="XYLABEL",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=str),
        nargs=2,
        required=False,
        default=[r"$x$", r"$y$"],
        help="x- and y-axis label.  Default: %(default)s",
    )
    parser.add_argument(
        "--xlim",
        dest="XLIM",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=float),
        nargs=2,
        required=False,
        default=[None, None],
        help=(
            "Left and right limit of the x-axis in data coordinates.  Default:"
            " %(default)s"
        ),
    )
    parser.add_argument(
        "--ylim",
        dest="YLIM",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=float),
        nargs=2,
        required=False,
        default=[None, None],
        help=(
            "Lower and upper limit of the y-axis in data coordinates."
            "  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--log",
        dest="LOG",
        type=mdt.fh.str2bool,
        nargs=2,
        required=False,
        default=[False, False],
        help=(
            "Whether to use a logarithmic scale for the x- and/or y-axis."
            "  Default: %(default)s"
        ),
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    # TODO: Check parsed input arguments if necessary.
    if args.COLS is not None and len(args.COLS) < 2:
        raise ValueError(
            "Invalid input for option --cols: {}.  You must give either none"
            " or at least two columns".format(args.COLS)
        )

    print("\n")
    print("Reading input file...")
    timer = datetime.now()
    data = np.loadtxt(args.INFILE, usecols=args.COLS)
    if data.ndim <= 1:
        raise ValueError("The input file must contain at least two columns")
    xdata = np.ascontiguousarray(data[:, 0])
    ydata = np.ascontiguousarray(data[:, 1:])
    n_ydata_sets = data.shape[1] - 1
    del data
    if args.LABELS is not None and len(args.LABELS) != n_ydata_sets:
        raise ValueError("You must give as many labels as y data sets")
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating plot...")
    timer = datetime.now()
    # When creating multiple plots, use
    # matplotlib.backends.backend_pdf.PdfPages to save all plots to one
    # PDF file.
    fig, ax = plt.subplots(clear=True)
    ax.plot(xdata, ydata, label=args.LABELS)
    if args.LOG[0]:
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    if args.LOG[1]:
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel=args.XYLABEL[0],
        ylabel=args.XYLABEL[1],
        xlim=args.XLIM,
        ylim=args.YLIM,
    )
    if args.LABELS is not None:
        ax.legend()
    mdt.fh.backup(args.OUTFILE)
    plt.savefig(args.OUTFILE)
    plt.close()
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
