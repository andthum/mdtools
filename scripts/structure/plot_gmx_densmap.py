#!/usr/bin/env python3

# This file is part of MDTools.
# Copyright (C) 2021-2023  The MDTools Development Team and all
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
Read up to three matrices from text files and plot them as one RGB
matrix with :meth:`matplotlib.axes.Axes.imshow`.

Each matrix must be stored in a separate text file.  The first column of
the text files must contain the x values and the first row the y values
(note that this is opposed to the standard matrix convention).  The
value in the upper left corner will be ignored.  The remaining elements
of the matrix must contain the z values for each (x,y) pair.  The file
may contain comment lines starting with '#', which will be ignored.

Options
-------
-r
    File containing the matrix that shall be represented as red levels
    in the final RGB matrix.
-g
    File containing the matrix that shall be represented as green levels
    in the final RGB matrix.
-b
    File containing the matrix that shall be represented as blue levels
    in the final RGB matrix.

    Note that at least one of the -r, -g and -b flag must be provided.
    If multiple matrices are given, all matrices must have the same
    shape and the same x and y values.  The input matrices must not
    contain negative values.
-o
    Output filename.
-c
    Eliminate values below a certain cutoff in the final RGB  matrix to
    suppress noise.  The values of each RGB channel are normalized to
    the interval [0, 1] (not [0,255] as usual).  Default: ``0``.
--Otsu
    Use Otsu's binarization [#]_ to automatically calculate a cutoff.
    If \--Otsu is set, -c will be ignored.  This option requires the
    `opencv-python`_ package.
--xylabel
    x- and y-axis label.  Default: ``[r'$x$ / nm', r'$y$ / nm']``.
--xlim
    Left and right limit of the x-axis in data coordinates.  Pass 'None'
    to adjust the limit(s) automatically.  Default: ``[None, None]``.
--ylim
    Lower and upper limit of the y-axis in data coordinates.  Pass
    'None' to adjust the limit(s) automatically.  Default:
    ``[None, None]``.
--xticks-at-yticks
    Set x-ticks at the same positions as y-ticks.

.. _opencv-python: https://pypi.org/project/opencv-python/

See Also
--------
:mod:`scripts.structure.densmap` :
    Compute a 2-dimensional number density map

Notes
-----
This python script is inspired by the work of Hadrian Montes-Campos
[#]_:sup:`,` [#]_.  It was originally designed to read the output file
that is produced by the GROMACS tool `gmx densmap`_ with the '-od' flag
but can equally well read other matrix files (like the one produced by
:mod:`scripts.structure.densmap`) as long as they are provided in the
specified format.

.. _gmx densmap:
    https://manual.gromacs.org/documentation/current/onlinehelp/gmx-densmap.html

References
----------
.. [#] N. Otsu, `"A threshold selection method from gray-level
    histograms" <https://doi.org/10.1109/TSMC.1979.4310076>`_, IEEE
    transactions on systems, man, and cybernetics, 1979, 9, 62-66.
.. [#] H. Montes-Campos, J. M. Otero-Mato, T. Mendez-Morales, O. Cabeza,
    L. J. Gallego, A. Ciach, L. M. Varela, `"Two-dimensional pattern
    formation in ionic liquids confined between graphene walls"
    <https://doi.org/10.1039/C7CP04649A>`_, Physical Chemistry Chemical
    Physics, 2017, 19, 24505-24512.
.. [#] J. M. Otero-Mato, H. Montes-Campos, O. Cabeza, D. Diddens, A.
    Ciach, L. J. Gallego, L. M. Varela, `"3D structure of the electric
    double layer of ionic liquid-alcohol mixtures at the electrochemical
    interface" <https://doi.org/10.1039/C8CP05632C>`_, Physical
    Chemistry Chemical Physics, 2018, 20, 30412-30427.
"""


__author__ = "Andreas Thum"


# Standard libraries
import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import psutil

# First-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt


def read_matrix(fname):
    """
    Read a 2-dimensional matrix from a text file.

    The first column of the text file must contain the x values and the
    first row the y values (note that this is opposed to the standard
    matrix convention).  The value in the upper left corner will be
    ignored.  The remaining elements of the matrix must contain the z
    values for each (x,y) pair.  The file may contain comment lines
    starting with '#', which will be ignored.

    Parameters
    ----------
    fname : str
        Name of the data file.

    Returns
    -------
    x : numpy.ndarray
        1-dimensional array containing the x values.
    y : numpy.ndarray
        1-dimensional array containing the y values.
    z : numpy.ndarray
        2-dimensional array containing the z values for each (x,y) pair.
        The input matrix is transposed and reversed vertically before it
        is returned as `z`.  Vividly speaking, the paper on which the
        matrix is written is turned by 90 degrees anti-clockwise.  This
        is done to get back to the usual matrix representation, where an
        array `z` with shape ``(nrows, ncolumns)`` is plotted with the
        column number as x and the row number as y.  The remaining
        difference to the usual matrix representation is that the
        original origin of the matrix (the value with index [0,0]) is
        now at the lower left corner (i.e. it is now at
        ``[nrows-1,0]``).

    Notes
    -----
    This function was originally designed to read the output file that
    is produced by the GROMACS tool `gmx densmap`_ with the '-od' flag
    and to prepare the matrix for plotting with
    :meth:`matplotlib.axes.Axes.imshow`.

    .. _gmx densmap:
        https://manual.gromacs.org/documentation/current/onlinehelp/gmx-densmap.html
    """
    data = np.loadtxt(fname)
    x = data[1:, 0]
    y = data[0, 1:]
    z = data[1:, 1:]
    z = np.ascontiguousarray(z.T[::-1])
    return x, y, z


if __name__ == "__main__":  # noqa: C901
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage
    parser = argparse.ArgumentParser(
        description=(
            "Read up to three matrices from text files and plot them as one"
            " RGB matrix with matplotlib.axes.Axes.imshow.  For more"
            " information, refer to the documentation of this script."
        )
    )
    parser.add_argument(
        "-r",
        dest="RED",
        type=str,
        required=False,
        default=None,
        help=(
            "File containing the matrix that shall be represented as red"
            " levels in the final RGB matrix."
        ),
    )
    parser.add_argument(
        "-g",
        dest="GREEN",
        type=str,
        required=False,
        default=None,
        help=(
            "File containing the matrix that shall be represented as green"
            " levels in the final RGB matrix."
        ),
    )
    parser.add_argument(
        "-b",
        dest="BLUE",
        type=str,
        required=False,
        default=None,
        help=(
            "File containing the matrix that shall be represented as blue"
            " levels in the final RGB matrix."
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
        "-c",
        dest="CUTOFF",
        type=float,
        required=False,
        default=0,
        help=(
            "Eliminate values below a certain cutoff in the final RGB matrix"
            " to suppress noise.  The values of each RGB channel are"
            " normalized to the interval [0, 1].  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--Otsu",
        dest="OTSU",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Use Otsu's binarization to automatically calculate a cutoff.  If"
            " --Otsu is set, -c will be ignored."
        ),
    )
    parser.add_argument(
        "--xylabel",
        dest="XYLABEL",
        type=lambda val: mdt.fh.str2none_or_type(val, dtype=str),
        nargs=2,
        required=False,
        default=[r"$x$ / nm", r"$y$ / nm"],
        help="x- and y-axis label.  Default: %(default)s.",
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
            " %(default)s."
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
            "  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--xticks-at-yticks",
        dest="XTICKS_AT_YTICKS",
        required=False,
        default=False,
        action="store_true",
        help="Set x-ticks at the same positions as y-ticks.",
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    RGB_ARGS = (args.RED, args.GREEN, args.BLUE)
    RGB_CODE = {0: "red", 1: "green", 2: "blue"}
    if all(rgb_arg is None for rgb_arg in RGB_ARGS):
        raise RuntimeError("Neither -r, nor -g, nor -b is set")
    if args.CUTOFF < 0 or args.CUTOFF > 1:
        raise RuntimeError(
            "-c ({}) must be between 0 and 1".format(args.CUTOFF)
        )
    if args.OTSU and args.CUTOFF > 0:
        warnings.warn(
            "-c ({}) will be ignored, because --Otsu is"
            " set".format(args.CUTOFF),
            RuntimeWarning,
            stacklevel=2,
        )

    print("\n")
    print("Reading input file...")
    timer = datetime.now()
    x, y, z = [], [], []
    rgb_channel_used = np.zeros(3, dtype=bool)
    for i, rgb_arg in enumerate(RGB_ARGS):
        if rgb_arg is not None:
            rgb_channel_used[i] = True
            xtmp, ytmp, ztmp = read_matrix(RGB_ARGS[i])
            x.append(xtmp)
            y.append(ytmp)
            z.append(ztmp)
    for i in range(1, len(x)):
        if x[i].shape != x[0].shape:
            raise ValueError(
                "All input files must contain the same number of x values"
            )
        if not np.allclose(x[i], x[0], rtol=0, equal_nan=True):
            raise ValueError("All input files must contain the same x values")
        if y[i].shape != y[0].shape:
            raise ValueError(
                "All input files must contain the same number of y values"
            )
        if not np.allclose(y[i], y[0], rtol=0, equal_nan=True):
            raise ValueError("All input files must contain the same y values")
        if z[i].shape != z[0].shape:
            raise ValueError("All input matrices must have the same shape")
        if np.any(z[i] < 0):
            raise ValueError(
                "The input matrices must not contain negative values."
            )
    x = np.array(x[0])
    y = np.array(y[0])
    z = np.array(z)
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Combining input matrices to a single RGB matrix...")
    timer = datetime.now()
    rgb = np.zeros(z[0].shape + (3,), dtype=np.float64)
    j = 0
    for i, channel_used in enumerate(rgb_channel_used):
        if channel_used:
            # The three RGB channels can each take a value from 0 to
            # 255, because they are stored as 8-bit unsigned integer.
            # matplotlib.axes.Axes.imshow also accepts a float from 0 to
            # 1, which is easier to accomplish.
            rgb[..., i] = z[j] / np.max(z[j])
            j += 1
    del z
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Applying cutoff or Otsu's binarization...")
    timer = datetime.now()
    if args.OTSU:
        try:
            # Third-party libraries
            import cv2  # noqa: I005
        except ImportError:
            raise ImportError(
                "To use Otsu's binarization, the package cv2 must be installed"
            )
        for i, channel_used in enumerate(rgb_channel_used):
            if channel_used:
                rgb_norm = np.round(rgb[..., i] * 255).astype(np.uint8)
                thresh, rgb[..., i] = cv2.threshold(
                    src=rgb_norm,
                    thresh=0,
                    maxval=255,
                    type=cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
                rgb[..., i] /= np.max(rgb[..., i])
                # print("Histogram:")
                # print(np.bincount(rgb_norm.flatten()))
                print(
                    "Otsu's threshold for {:>5} channel (0 - 255):"
                    " {:>3f}".format(RGB_CODE[i], thresh)
                )
    else:
        rgb[rgb < args.CUTOFF] = 0
    for i, channel_used in enumerate(rgb_channel_used):
        if channel_used:
            surf_cov = np.count_nonzero(rgb[..., i]) / rgb[..., i].size
            print(
                "Amount of surface covered by {:>5} pixels: {:>6.4f}".format(
                    RGB_CODE[i], surf_cov
                )
            )
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating plot...")
    timer = datetime.now()
    fig, ax = plt.subplots(figsize=(5.82677, 5.82677), clear=True)
    mdtplt.imshow_new(
        X=rgb,
        extent=(x.min(), x.max(), y.min(), y.max()),
        ax=ax,
        cbar=False,
    )
    ax.set(
        xlabel=args.XYLABEL[0],
        ylabel=args.XYLABEL[1],
        xlim=args.XLIM,
        ylim=args.YLIM,
    )
    if args.XTICKS_AT_YTICKS:
        yticks = np.asarray(ax.get_yticks())
        mask = (yticks >= ax.get_xlim()[0]) & (yticks <= ax.get_xlim()[1])
        ax.set_xticks(yticks[mask])
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
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage()))
