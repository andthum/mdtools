#!/usr/bin/env python3


# This file is part of MDTools.
# Copyright (C) 2020  Andreas Thum
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


# This python script is inspired by the work of Hadrián Montes-Campos
# * H. Montes-Campos, J.M. Otero-Mato, T. Mendez-Morales, O. Cabeza,
#   L.J. Gallego, A. Ciach, L.M. Varela, PCCP, 2017,19, 24505-24512
# * J.M. Otero-Mato, H. Montes-Campos, O. Cabeza, D. Diddens, A. Ciach,
#   L.J. Gallego, L.M. Varela, PCCP, 2018, 20, 30412-30427




import os
import sys
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mdtools as mdt


plt.rc('lines', linewidth=2)
plt.rc('axes', linewidth=2)
plt.rc('xtick.major', width=2)
plt.rc('xtick.minor', width=2)
plt.rc('ytick.major', width=2)
plt.rc('ytick.minor', width=2)
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preview'] = True
plt.rc('font', **{'family': 'serif', 'serif': 'Times'})




def read_matrix(infile):
    """
    Read data file. The file must have the same format as the output
    produced by GROMACS' function gmx densmap with the -od flag. I.e.,
    the input file must be an ASCII file containing a 2-dimensional
    matrix. The first column must contain the x values and the first row
    the y values. The upper left corner can have any value, but is
    usually set to zero. The remaining elements of the matrix must
    contain the z values for each (x,y) pair. The file may contain
    comment lines starting with #, which will be ignored.

    Parameters
    ----------
    infile : str
        Name of the data file.

    Returns
    -------
    x : numpy.ndarray
        1-dimensional array containing the x values.
    y : numpy.ndarray
        1-dimensional array containing the y values.
    z : numpy.ndarray
        2-dimensional array containing the z values for each (x,y) pair.
    """

    data = np.genfromtxt(infile)
    x = data[1:, 0]
    y = data[0, 1:]
    z = data[1:, 1:]

    # Make the upper left corner of the matrix the lower left corner,
    # i.e. invert the matrix vertically
    z = np.ascontiguousarray(z.T[::-1])

    return x, y, z




def plot_rgb_matrix(ax, z, xmin=None, xmax=None, ymin=None, ymax=None,
                    xlabel=r'$x$ / nm', ylabel=r'$y$ / nm', **kwargs):
    """
    Plot a RGB matrix, i.e. a two dimensional matrix with three channels,
    i.e. a matrix of shape ``(m, n, 3)`` with values between 0 and 255
    or between 0 and 1, with ``matplotlib.axes.Axes``. The matrix can
    have the spahe ``(m, n, 4)`` if a alpha channel, i.e. transperency,
    is included.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    z : array_like
        Array of shape ``(m, n, 3)`` or ``(m, n, 4)`` containing the
        rgb (and alpha) values at each (x,y) point.
    xmin : scalar, optional
        Left limit for plotting on x-axis. The left limit may be greater
        than the right limit, in which case the tick values will show up
        in decreasing order. Default is ``None``, which means set the
        left limit to ``np.min(x)``.
    xmax : float, optional
        Right limit for plotting on x-axis. Default is ``None``, which
        means set the left limit to ``np.max(x)``.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    xlabel : str
        Label for the x-axis. Default is ``r'$x$ / nm'``.
    ylabel : str
        Label for the y-axis. Default is ``r'$y$ / nm'``.
    kwargs : dict
        Keyword arguments to pass to ``ax.imshow()``.

    Returns
    -------
    img : list
        List of artists added to the axis `ax`.
    """

    fontsize_labels = 36
    fontsize_ticks = 32
    tick_length = 10
    tick_pad = 12
    label_pad = 16

    img = ax.imshow(z, **kwargs)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = 20
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(which='major',
                   direction='out',
                   top=False,
                   right=False,
                   length=tick_length,
                   labelsize=fontsize_ticks,
                   pad=tick_pad)
    ax.tick_params(which='minor',
                   direction='out',
                   top=False,
                   right=False,
                   length=0.5 * tick_length,
                   labelsize=0.8 * fontsize_ticks,
                   pad=tick_pad)

    return img








if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Combine up to three matrices to one RGB matrix and"
            " plot this new matrix as heatmap with"
            " matplotlib.pyplot.imshow(). The files containing"
            " the matrices must have the same format as the"
            " output produced by GROMACS' function gmx densmap"
            " with the -od flag. I.e., the input file must be"
            " an ASCII file containing a 2-dimensional matrix."
            " The first column must contain the x values and"
            " the first row the y values. The upper left corner"
            " can have any value, but is usually set to zero."
            " The remaining elements of the matrix must contain"
            " the z values for each (x,y) pair. The file may"
            " contain comment lines starting with #, which will"
            " be ignored."
        )
    )

    parser.add_argument(
        '-r',
        dest='RED',
        type=str,
        required=False,
        default=None,
        help="File containing the matrix that shall be represented as"
             " red levels in the final RGB matrix."
    )
    parser.add_argument(
        '-g',
        dest='GREEN',
        type=str,
        required=False,
        default=None,
        help="File containing the matrix that shall be represented as"
             " green levels in the final RGB matrix."
    )
    parser.add_argument(
        '-b',
        dest='BLUE',
        type=str,
        required=False,
        default=None,
        help="File containing the matrix that shall be represented as"
             " blue levels in the final RGB matrix. Note: At leas one of"
             " the -r, -g and -b flag must be provided. If multiple"
             " matrices are given, all matrices must have the same shape"
             " and the same x and y values."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename. Output is optimized for PDF format with"
             " TeX support."
    )

    parser.add_argument(
        '-c',
        dest='CUTOFF',
        type=float,
        required=False,
        default=0,
        help="Eliminate values below a certain cutoff in the final RGB"
             " matrix to suppress noise. The RGB matrix ranges from 0 to"
             " 1. Default: 0"
    )
    parser.add_argument(
        '--Otsu',
        dest='OTSU',
        required=False,
        default=False,
        action='store_true',
        help="Use Otsu's binarization to automatically calculate a"
             " cutoff. If --Otsu is set, -c will be ignored. This option"
             " requires the opencv-python package."
    )

    parser.add_argument(
        '--xmin',
        dest='XMIN',
        type=float,
        required=False,
        default=None,
        help="Lower limit for plotting on x axis."
    )
    parser.add_argument(
        '--xmax',
        dest='XMAX',
        type=float,
        required=False,
        default=None,
        help="Upper limit for plotting on x axis."
    )
    parser.add_argument(
        '--ymin',
        dest='YMIN',
        type=float,
        required=False,
        default=None,
        help="Lower limit for plotting on y axis."
    )
    parser.add_argument(
        '--ymax',
        dest='YMAX',
        type=float,
        required=False,
        default=None,
        help="Upper limit for plotting on y axis."
    )


    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())


    rgb_args = np.array([args.RED, args.GREEN, args.BLUE])
    rgb_code = {0: 'red', 1: 'green', 2: 'blue'}


    if np.all(rgb_args == None):
        raise RuntimeError("Neither -r, nor -g, nor -b is set")
    if args.CUTOFF < 0 or args.CUTOFF > 1:
        raise RuntimeError("The cutoff must be between 0 and 1, but you"
                           " gave -c {}".format(args.CUTOFF))
    if args.OTSU and args.CUTOFF > 0:
        warnings.warn("-c {} will be ignored, because --Otsu is set"
                      .format(args.CUTOFF), RuntimeWarning)




    x = []
    y = []
    z = []
    rgb_channel_used = np.zeros(3, dtype=bool)
    for i in range(len(rgb_args)):
        if rgb_args[i] is not None:
            rgb_channel_used[i] = True
            xtmp, ytmp, ztmp = read_matrix(rgb_args[i])
            x.append(xtmp)
            y.append(ytmp)
            z.append(ztmp / ztmp.max())
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    for i in range(1, len(x)):
        if np.any(x[i - 1] != x[i]):
            raise ValueError("The x values of the provided data are not"
                             " exactly the same")
        if np.any(y[i - 1] != y[i]):
            raise ValueError("The y values of the provided data are not"
                             " exactly the same")
        if z[i - 1].shape != z[i].shape:
            raise ValueError("The provided matrices have not the same"
                             " shape")
    x = x[0]
    y = y[0]


    # Create RGB matrix from the input matrices. The three RGB channels
    # can each take a value from 0 to 255 because they are stored as
    # 8-bit unsigned integer. But matplotlib.pyplot.imshow() also
    # accepts a float from 0 to 1.
    rgb = np.zeros(z[0].shape + (3,), dtype=np.float64)
    j = 0
    for i in range(len(rgb_channel_used)):
        if rgb_channel_used[i]:
            rgb[..., i] = z[j]
            j += 1


    # Eliminate values below a certain cutoff to suppress noise.
    if args.OTSU:
        try:
            import cv2
        except ImportError:
            raise ImportError("To use Otsu's binarization, the package"
                              " cv2 needs to be installed")
        print("\n\n\n", flush=True)
        for i in range(len(rgb_channel_used)):
            if rgb_channel_used[i]:
                rgb_norm = np.round(rgb[..., i] * 255).astype(np.uint8)
                thresh, rgb[..., i] = cv2.threshold(
                    src=rgb_norm,
                    thresh=0,
                    maxval=255,
                    type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                rgb[..., i] /= rgb[..., i].max()
                #print("Histogram:", flush=True)
                #print(np.bincount(rgb_norm.flatten()), flush=True)
                print("Otsu's threshold for {:>5} channel (0 - 255):"
                      " {:>3f}".format(rgb_code[i], thresh), flush=True)
    else:
        rgb[rgb < args.CUTOFF] = 0


    print("\n\n\n", flush=True)
    for i in range(len(rgb_channel_used)):
        if rgb_channel_used[i]:
            surf_cov = np.count_nonzero(rgb[..., i]) / rgb[..., i].size
            print("Amount of surface covered by {:>5} pixels: {:>6.4f}"
                  .format(rgb_code[i], surf_cov), flush=True)
    print("\n\n", flush=True)


    fig, axis = plt.subplots(figsize=(8.27, 8.27),  # Short side of DIN A4 in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    plot_rgb_matrix(ax=axis,
                    z=rgb,
                    xmin=args.XMIN,
                    xmax=args.XMAX,
                    ymin=args.YMIN,
                    ymax=args.YMAX,
                    extent=(x.min(), x.max(), y.min(), y.max()))
    yticks = np.asarray(axis.get_yticks())
    mask = ((yticks >= axis.get_xlim()[0]) &
            (yticks <= axis.get_xlim()[1]))
    axis.set_xticks(yticks[mask])
    mdt.fh.backup(args.OUTFILE)
    plt.tight_layout()
    plt.savefig(args.OUTFILE)
    plt.close(fig)


    print("\n\n\n", flush=True)
    print("{} done".format(os.path.basename(sys.argv[0])), flush=True)
