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




import os
import sys
from datetime import datetime
import psutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mdtools as mdt




def read_matrix(infile, dtype=np.float32):
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
    dtype : data-type, optional
        Data-type of the resulting arrays.
    
    Returns
    -------
    x : numpy.ndarray
        1-dimensional array containing the x values.
    y : numpy.ndarray
        1-dimensional array containing the y values.
    z : numpy.ndarray
        2-dimensional array containing the z values for each (x,y) pair.
    """
    
    data = np.loadtxt(infile, dtype=dtype)
    x = data[1:,0]
    y = data[0,1:]
    z = data[1:,1:]
    
    # Make the columns corresponding to the x values and the rows to the
    # y values
    z = np.ascontiguousarray(z.T)
    
    return x, y, z




def hex_center_dist(a):
    """
    Get the minimum distance between to hexagon centers in a hexagonal
    lattive with lattice constant `a`. Works only for hexagonal lattices
    whose edges align with the x axis.
    
    Parameters
    ----------
    a : scalar
        The lattice constant of the hexagonal lattice.
    
    Returns
    -------
    dist : tuple
        Minimum x and y distance between two hexagon centers.
    """
    
    xdist = a * np.cos(np.deg2rad(30))
    # = r0 * 3/2, with r0 being the distance between two hexagon
    # vertices (= C-C bond length in graphene)
    ydist = a / 2
    # = r0 * cos(30)
    
    return (xdist, ydist)




def move_hex(pos, n, a):
    """
    Given the position `pos` of a hexagon center and the lattice
    constant `a`, calculate the position of another hexagon center with
    offset `n` from the initial hexagon center. Works only for hexagonal
    lattices whose edges align with the x axis.
    
    Parameters
    ----------
    pos : array_like
        Array of shape ``(2,)`` containing the x and y position of the
        initial hexagon center.
    n : array_like
        Array of the same shape as `pos` indicating how many hexagons to
        move in x and y direction.
    a : scalar
        The lattice constant of the hexagonal lattice.
    
    Returns
    -------
    new_pos : numpy.ndarray
        Array of the same shape as `pos` containing the position of the
        new hexagon center.
    """
    
    pos = np.asarray(pos)
    n = np.asarray(n)
    if pos.shape != (2,):
        raise ValueError("pos must be of shape (2,)")
    if n.shape != pos.shape:
        raise ValueError("n must be of the same shape as pos")
    
    return pos + n * np.asarray(hex_center_dist(a=a))




def is_hex_center(test_pos, hex_pos, a):
    """
    Given the lattice constant of a hexagonal lattice and a known
    position of any hexagon center in that lattice, check whether an
    arbitrary test position is also a hexagon center. Works only for
    hexagonal lattices whose edges align with the x axis.
    
    Parameters
    ----------
    test_pos : array_like
        Array of shape ``(2,)`` containing the x and y position of the
        test position.
    hex_pos : array_like
        Array of shape ``(2,)`` containing the x and y position of any
        hexagon center in the hexagonal lattice.
    a : scalar
        The lattice constant of the hexagonal lattice.
    
    Returns
    -------
    ``True``, if `test_pos` is located at the center of any hexagon of
    the hexagonal lattice.
    ``False`` otherwise.
    """
    
    test_pos = np.asarray(test_pos)
    hex_pos = np.asarray(hex_pos)
    if test_pos.shape != (2,):
        raise ValueError("test_pos must be of shape (2,)")
    if hex_pos.shape != test_pos.shape:
        raise ValueError("hex_pos must be of the same shape as test_pos")
    
    xdist, ydist = hex_center_dist(a=a)
    nx = (test_pos[0] - hex_pos[0]) / xdist
    ny = (test_pos[1] - hex_pos[1]) / ydist
    
    if not nx.is_integer() or not ny.is_integer():
        return False
    elif nx % 2 == 0 and ny % 2 == 0:
        return True
    elif nx % 2 != 0 and ny % 2 != 0:
        return True
    else:
        return False




def g(x, m=1, c=0):
    """Straight line with slope `m` and intercept `c`."""
    return m*x + c




def g_inverse(y, m=1, c=0):
    """
    Inverse function of a straight line with slope `m` and intercept `c`
    """
    return (y - c) / m




def finish_heatmap_plot(axis):
    xsize = abs(axis.get_xlim()[0] - axis.get_xlim()[1])
    ysize = abs(axis.get_ylim()[0] - axis.get_ylim()[1])
    axis.set_aspect(ysize / xsize)
    yticks = np.array(axis.get_yticks())
    mask = ((yticks >= axis.get_xlim()[0]) &
            (yticks <= axis.get_xlim()[1]))
    axis.set_xticks(yticks[mask])
    plt.tight_layout()
    plt.show()
    plt.close()








if __name__ == "__main__":
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Extract cross sections from a 2-dimensional"
                     " density matrix along hexagonal axes (first-"
                     "nearest neighbour and second-nearest neighbour"
                     " axes). The file containing the matrix must have"
                     " the same format as the output produced by"
                     " GROMACS' function gmx densmap with the -od flag."
                     " I.e., the input file must be an ASCII file"
                     " containing a 2-dimensional matrix. The first"
                     " column must contain the x values and the first"
                     " row the y values. The upper left corner can have"
                     " any value, but is usually set to zero. The"
                     " remaining elements of the matrix must contain the"
                     " z values for each (x,y) pair. The file may"
                     " contain comment lines starting with #, which will"
                     " be ignored. Currently, this script is only"
                     " working when the edges of the hexagons align with"
                     " the x axis."
                 )
    )
    
    parser.add_argument(
        '-f',
        dest='INFILE',
        type=str,
        required=False,
        default=None,
        help="File containing the density matrix."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename pattern. There will be created two output"
             " files: <OUTFILE>.txt and <OUTFILE>.pdf"
    )
    parser.add_argument(
        '--show-plots',
        dest='SHOW_PLOTS',
        required=False,
        default=False,
        action='store_true',
        help="Show intermediate plots. The plots are only shown but not"
             " saved, since saving them costs a huge amount of time and"
             " memory. However, you can decide for each individuall plot"
             " that pops up, whether you want to save it or not. The"
             " plots are only intermediate plots illustrating the cross"
             " sectioning procedure."
    )
    
    parser.add_argument(
        '-a',
        dest='LATCONST',
        type=float,
        required=False,
        default=0.246,
        help="Lattice constant of the hexagonal lattice. Default: 0.246,"
             " which is the lattice constant of graphene in nm."
    )
    parser.add_argument(
        '--hex-pos',
        dest='HEXPOS',
        nargs=2,
        type=float,
        required=False,
        default=[0.071, 0.123],
        help="Position of an arbitrary hexagon center. Default: 0.071"
             " 0.123, which is the center of the first graphene hexagon,"
             " when the edges of the hexagons align with the x axis and"
             " a hexagon vertex (C atom) is placed at 0 0."
    )
    parser.add_argument(
        '-w',
        dest='WIDTH',
        type=float,
        required=False,
        default=0.071,
        help="Axis width. The cross section along an axis is averaged"
             " over the width of the axis. Default: 0.071, which is half"
             " the C-C bond length in graphene in nm."
    )
    parser.add_argument(
        '-n',
        dest='NSAMPLES',
        type=int,
        required=False,
        default=10,
        help="How many evenly spaced samples to take for a single cross"
             " section within the given axis width. Default: 10."
    )
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    if args.LATCONST <= 0:
        raise ValueError("-a must be positive")
    if args.WIDTH < 0:
        raise ValueError("-w must not be negative")
    if args.NSAMPLES <= 0:
        raise ValueError("-n must be positive")
    if args.WIDTH == 0 and args.NSAMPLES > 1:
        args.NSAMPLES = 1
        print("\n\n\n", flush=True)
        print("Note: Setting -n to 1, because -w is 0", flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Reading input", flush=True)
    timer = datetime.now()
    
    
    x, y, z = read_matrix(args.INFILE)
    z_shape = z.shape
    xmin, xmax = (np.min(x), np.max(x))
    ymin, ymax = (np.min(y), np.max(y))
    if (args.HEXPOS[0] < xmin or args.HEXPOS[0] > xmax or
        args.HEXPOS[1] < ymin or args.HEXPOS[1] > ymax):
        raise ValueError("--hex-pos is outside the data range")
    
    
    # Minimum distance between hexagon centers in x and y direction.
    # The hexagon edges must align with the x axis.
    xdist, ydist = hex_center_dist(a=args.LATCONST)
    # x/y positions of first hexagon column/row.
    # Rows are defined horizontally (i.e. in x direction) and
    # columns vertically (i.e. in y direction). => The first column
    # contains the hexagon centers with the lowest x values and the
    # first row contains the hexagon centers with the lowest y values
    nx = int((xmin - args.HEXPOS[0]) / xdist)
    ny = int((ymin - args.HEXPOS[1]) / ydist)
    col0, row0 = move_hex(pos=[args.HEXPOS[0], args.HEXPOS[1]],
                          n=[nx, ny],
                          a=args.LATCONST)
    # x/y positions of last hexagon column/row
    nx = int((xmax - args.HEXPOS[0]) / xdist)
    ny = int((ymax - args.HEXPOS[1]) / ydist)
    col1, row1 = move_hex(pos=[args.HEXPOS[0], args.HEXPOS[1]],
                          n=[nx, ny],
                          a=args.LATCONST)
    # Periodic boundary conditions
    if np.isclose(col0, xmin) and np.isclose(col1, xmax):
        col1 -= xdist
    if np.isclose(row0, ymin) and np.isclose(row1, ymax):
        row1 -= ydist
    # Number of hexagon columns/rows.
    # ncols = number of hexagons in x direction
    # nrows = number of hexagons in y direction
    ncols = int((col1 - col0) / xdist)
    nrows = int((row1 - row0) / ydist)
    # Total number of hexagons
    nhex = ncols * nrows//2 + (ncols//2 + ncols%2) * nrows%2
    
    
    if args.SHOW_PLOTS:
        print("  Creating plot", flush=True)
        timer_plot = datetime.now()
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.pcolormesh(ax=axis,
                            x=np.append(x, x[-1]+(x[-1]-x[-2])),
                            y=np.append(y, y[-1]+(y[-1]-y[-2])),
                            z=z,
                            xmin=xmin-0.05*(xmax-xmin),
                            xmax=xmax+0.05*(xmax-xmin),
                            ymin=ymin-0.05*(ymax-ymin),
                            ymax=ymax+0.05*(ymax-ymin))
        axis.axvline(x=col0, color='red', linestyle='--')
        axis.axvline(x=col1, color='red', linestyle='--')
        axis.axhline(y=row0, color='red', linestyle='--')
        axis.axhline(y=row1, color='red', linestyle='--')
        finish_heatmap_plot(axis=axis)
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
    
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    
    
    
    
    print("\n\n\n", flush=True)
    print("Extracting nearest neighbour axes", flush=True)
    timer = datetime.now()
    
    
    # Assume perdiodic boundary conditions.
    # Extend z (and y) in y direction, to simplify the extraction of
    # cross sections that cross the boundary
    unit_cell_slice = slice(z_shape[0], 2*z_shape[0])
    z = np.row_stack([z]*3)
    y_tmp = np.hstack([y, y[-1] + y[-1]-y[-2] + y])
    y = np.hstack([y - y[1]-y[0] - ymax, y_tmp])
    del y_tmp
    
    
    
    
    # Angle between the hexagonal axis and the x axis
    angle = 30
    print(flush=True)
    print("  Axis 1 of 3", flush=True)
    print("  Angle to x axis: {}°".format(angle), flush=True)
    timer_axis = datetime.now()
    # Slope and y-axis intercept of the line along which to take the
    # first sample cross section
    angle = np.deg2rad(angle)
    slope = np.tan(angle)
    if is_hex_center(test_pos=[col0, row0],
                     hex_pos=args.HEXPOS,
                     a=args.LATCONST):
        intercept_min = row0 - slope * col0  # c = y - mx
    else:
        intercept_min = (row0+ydist) - slope * col0
    
    
    if args.SHOW_PLOTS:
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.pcolormesh(ax=axis,
                            x=np.append(x, x[-1]+(x[-1]-x[-2])),
                            y=np.append(y[unit_cell_slice],
                                        (2*y[unit_cell_slice][-1] -
                                           y[unit_cell_slice][-2])),
                            z=z[unit_cell_slice],
                            xmin=xmin-0.05*(xmax-xmin),
                            xmax=xmax+0.05*(xmax-xmin),
                            ymin=ymin-0.05*(ymax-ymin),
                            ymax=ymax+0.05*(ymax-ymin))
        axis.axvline(x=col0, color='red', linestyle='--')
        axis.axvline(x=col1, color='red', linestyle='--')
        axis.axhline(y=row0, color='red', linestyle='--')
        axis.axhline(y=row1, color='red', linestyle='--')
    
    
    # Cross section along axis 1/3
    cs1 = np.zeros(z_shape[1], dtype=np.float32)
    n_hex_axes = nrows//2 + nrows%2
    for i in range(n_hex_axes):
        intercept = intercept_min + 2*i*ydist
        intercept_samples = np.linspace(intercept-args.WIDTH/2,
                                        intercept+args.WIDTH/2,
                                        args.NSAMPLES,
                                        dtype=np.float32)
        
        
        if args.SHOW_PLOTS and args.WIDTH > 0:
            mdt.plot.fill_between(
                ax=axis,
                x=x,
                y1=g(x=x, m=slope, c=intercept_samples[-1]),
                y2=g(x=x, m=slope, c=intercept_samples[0]),
                xmin=axis.get_xlim()[0],
                xmax=axis.get_xlim()[1],
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
                alpha=0.5)
        elif args.SHOW_PLOTS:
            mdt.plot.plot(
                ax=axis,
                x=x,
                y=g(x=x, m=slope, c=intercept_samples[0]),
                xmin=axis.get_xlim()[0],
                xmax=axis.get_xlim()[1],
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
                alpha=0.5)
        
        
        for j, sample in enumerate(intercept_samples):
            hex_ax = [(xmin, g(x=xmin, m=slope, c=sample)),
                      (xmax, g(x=xmax, m=slope, c=sample))]
            cs_tmp, r1 = mdt.nph.cross_section(z=z, x=x, y=y,
                                               line=hex_ax,
                                               num=z_shape[1])
            cs1 += cs_tmp
            if (i > 0 or j > 0) and not np.allclose(r1, r_prev):
                raise ValueError("The sampling points along the"
                                 " different samples of the same axis do"
                                 " not coincide. Try a lower axis width"
                                 " (-w option)")
            r_prev = r1
    cs1 /= n_hex_axes * args.NSAMPLES
    # scipy.ndimage.map_coordinates in mdt.nph.cross_section usually
    # appends a zero to the cross section
    if cs1[-1] == 0:
        cs1 = cs1[:-1]
        r1 = r1[:-1]
    if r1[0] != 0:
        raise ValueError("The first sampling point is not zero")
    print("  Elapsed time:         {}"
          .format(datetime.now()-timer_axis),
          flush=True)
    print("  Current memory usage: {:14.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    if args.SHOW_PLOTS:
        print(flush=True)
        print("  Creating plot of axes", flush=True)
        timer_plot = datetime.now()
        finish_heatmap_plot(axis=axis)
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
        
        
        print(flush=True)
        print("  Creating plot of cross section", flush=True)
        timer_plot = datetime.now()
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(ax=axis,
                      x=r1,
                      y=cs1,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$')
        plt.tight_layout()
        plt.show()
        plt.close()
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
    
    
    
    
    # Angle between the hexagonal axis and the x axis
    angle = 180 - 30
    print(flush=True)
    print("  Axis 2 of 3", flush=True)
    print("  Angle to x axis: {}°".format(angle), flush=True)
    timer_axis = datetime.now()
    # Slope and y-axis intercept of the line along which to take the
    # first sample cross section
    angle = np.deg2rad(angle)
    slope = np.tan(angle)
    if is_hex_center(test_pos=[col0, row0],
                     hex_pos=args.HEXPOS,
                     a=args.LATCONST):
        intercept_min = row0 - slope * col0  # c = y - mx
    else:
        intercept_min = (row0+ydist) - slope * col0
    
    
    if args.SHOW_PLOTS:
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.pcolormesh(ax=axis,
                            x=np.append(x, x[-1]+(x[-1]-x[-2])),
                            y=np.append(y[unit_cell_slice],
                                        (2*y[unit_cell_slice][-1] -
                                           y[unit_cell_slice][-2])),
                            z=z[unit_cell_slice],
                            xmin=xmin-0.05*(xmax-xmin),
                            xmax=xmax+0.05*(xmax-xmin),
                            ymin=ymin-0.05*(ymax-ymin),
                            ymax=ymax+0.05*(ymax-ymin))
        axis.axvline(x=col0, color='red', linestyle='--')
        axis.axvline(x=col1, color='red', linestyle='--')
        axis.axhline(y=row0, color='red', linestyle='--')
        axis.axhline(y=row1, color='red', linestyle='--')
    
    
    # Cross section along axis 2/3
    cs2 = np.zeros(z_shape[1], dtype=np.float32)
    n_hex_axes = nrows//2 + nrows%2
    for i in range(n_hex_axes):
        intercept = intercept_min + 2*i*ydist
        intercept_samples = np.linspace(intercept-args.WIDTH/2,
                                        intercept+args.WIDTH/2,
                                        args.NSAMPLES,
                                        dtype=np.float32)
        
        
        if args.SHOW_PLOTS and args.WIDTH > 0:
            mdt.plot.fill_between(
                ax=axis,
                x=x,
                y1=g(x=x, m=slope, c=intercept_samples[-1]),
                y2=g(x=x, m=slope, c=intercept_samples[0]),
                xmin=axis.get_xlim()[0],
                xmax=axis.get_xlim()[1],
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
                alpha=0.5)
        elif args.SHOW_PLOTS:
            mdt.plot.plot(
                ax=axis,
                x=x,
                y=g(x=x, m=slope, c=intercept_samples[0]),
                xmin=axis.get_xlim()[0],
                xmax=axis.get_xlim()[1],
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
                alpha=0.5)
        
        
        for j, sample in enumerate(intercept_samples):
            hex_ax = [(xmin, g(x=xmin, m=slope, c=sample)),
                      (xmax, g(x=xmax, m=slope, c=sample))]
            cs_tmp, r2 = mdt.nph.cross_section(z=z, x=x, y=y,
                                               line=hex_ax,
                                               num=z_shape[1])
            cs2 += cs_tmp
            if (i > 0 or j > 0) and not np.allclose(r2, r_prev):
                raise ValueError("The sampling points along the"
                                 " different samples of the same axis do"
                                 " not coincide. Try a lower axis width"
                                 " (-w option)")
            r_prev = r2
    cs2 /= n_hex_axes * args.NSAMPLES
    # scipy.ndimage.map_coordinates in mdt.nph.cross_section usually
    # appends a zero to the cross section
    if cs2[-1] == 0:
        cs2 = cs2[:-1]
        r2 = r2[:-1]
    if r2[0] != 0:
        raise ValueError("The first sampling point is not zero")
    print("  Elapsed time:         {}"
          .format(datetime.now()-timer_axis),
          flush=True)
    print("  Current memory usage: {:14.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    if args.SHOW_PLOTS:
        print(flush=True)
        print("  Creating plot of axes", flush=True)
        timer_plot = datetime.now()
        finish_heatmap_plot(axis=axis)
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
        
        
        print(flush=True)
        print("  Creating plot of cross section", flush=True)
        timer_plot = datetime.now()
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(ax=axis,
                      x=r2,
                      y=cs2,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$')
        plt.tight_layout()
        plt.show()
        plt.close()
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
    
    
    
    
    # Angle between the hexagonal axis and the x axis
    angle = 90
    print(flush=True)
    print("  Axis 3 of 3", flush=True)
    print("  Angle to x axis: {}°".format(angle), flush=True)
    timer_axis = datetime.now()
    
    
    if args.SHOW_PLOTS:
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.pcolormesh(ax=axis,
                            x=np.append(x, x[-1]+(x[-1]-x[-2])),
                            y=np.append(y[unit_cell_slice],
                                        (2*y[unit_cell_slice][-1] -
                                           y[unit_cell_slice][-2])),
                            z=z[unit_cell_slice],
                            xmin=xmin-0.05*(xmax-xmin),
                            xmax=xmax+0.05*(xmax-xmin),
                            ymin=ymin-0.05*(ymax-ymin),
                            ymax=ymax+0.05*(ymax-ymin))
        axis.axvline(x=col0, color='red', linestyle='--')
        axis.axvline(x=col1, color='red', linestyle='--')
        axis.axhline(y=row0, color='red', linestyle='--')
        axis.axhline(y=row1, color='red', linestyle='--')
    
    
    # Cross section along axis 3/3
    cs3_1 = np.zeros(z_shape[0], dtype=np.float32)
    cs3_2 = np.zeros(z_shape[0], dtype=np.float32)
    n_hex_axes = ncols + ncols%2
    for i in range(n_hex_axes):
        intercept = col0 + i*xdist  # x-axis (!) intercept
        intercept_samples = np.linspace(intercept-args.WIDTH/2,
                                        intercept+args.WIDTH/2,
                                        args.NSAMPLES,
                                        dtype=np.float32)
        
        
        if args.SHOW_PLOTS and args.WIDTH > 0:
            mdt.plot.fill_betweenx(
                ax=axis,
                y=[ymin, ymax],
                x1=[intercept_samples[-1]] * 2,
                x2=[intercept_samples[0]] * 2,
                xmin=axis.get_xlim()[0],
                xmax=axis.get_xlim()[1],
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
                alpha=0.5)
        elif args.SHOW_PLOTS:
            mdt.plot.plot(
                ax=axis,
                x=[intercept_samples[0]] * 2,
                y=[ymin, ymax],
                xmin=axis.get_xlim()[0],
                xmax=axis.get_xlim()[1],
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
                alpha=0.5)
        
        
        for j, sample in enumerate(intercept_samples):
            hex_ax = [(sample, ymin), (sample, ymax)]
            cs_tmp, r3 = mdt.nph.cross_section(z=z, x=x, y=y,
                                               line=hex_ax,
                                               num=z_shape[0])
            if i%2 == 0:
                cs3_1 += cs_tmp
            else:
                cs3_2 += cs_tmp
            if (i > 0 or j > 0) and not np.allclose(r3, r_prev):
                raise ValueError("The sampling points along the"
                                 " different samples of the same axis do"
                                 " not coincide. Try a lower axis width"
                                 " (-w option)")
            r_prev = r3
    # scipy.ndimage.map_coordinates in mdt.nph.cross_section usually
    # appends a zero to the cross section
    if cs3_1[-1] == 0 and cs3_2[-1] == 0:
        cs3_1 = cs3_1[:-1]
        cs3_2 = cs3_2[:-1]
        r3 = r3[:-1]
    if r3[0] != 0:
        raise ValueError("The first sampling point is not zero")
    # Shift the cross sections in such a way that the first hexagon
    # center resides at r=0
    if is_hex_center(test_pos=[col0, row0],
                     hex_pos=args.HEXPOS,
                     a=args.LATCONST):
        _, shift1 = mdt.nph.find_nearest(r3, row0, return_index=True)
        _, shift2 = mdt.nph.find_nearest(r3,
                                         row0+ydist,
                                         return_index=True)
    else:
        _, shift1 = mdt.nph.find_nearest(r3,
                                         row0+ydist,
                                         return_index=True)
        _, shift2 = mdt.nph.find_nearest(r3, row0, return_index=True)
    print("  Elapsed time:         {}"
          .format(datetime.now()-timer_axis),
          flush=True)
    print("  Current memory usage: {:14.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    if args.SHOW_PLOTS:
        print(flush=True)
        print("  Creating plot of axes", flush=True)
        timer_plot = datetime.now()
        finish_heatmap_plot(axis=axis)
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
        
        
        print(flush=True)
        print("  Creating plot of cross section", flush=True)
        timer_plot = datetime.now()
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(ax=axis,
                      x=r3,
                      y=cs3_1,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 3.1")
        mdt.plot.plot(ax=axis,
                      x=r3,
                      y=cs3_2,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 3.2")
        plt.tight_layout()
        plt.show()
        plt.close()
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
    
    
    cs3_1 = np.roll(cs3_1, shift=-shift1)
    cs3_2 = np.roll(cs3_2, shift=-shift2)
    cs3 = cs3_1 + cs3_2
    cs3 /= n_hex_axes * args.NSAMPLES
    del cs3_1, cs3_2
    
    
    
    
    print(flush=True)
    print("  Averaging over all 3 axes", flush=True)
    timer_average = datetime.now()
    
    if cs1.shape != cs2.shape:
        raise ValueError("The cross sections of axis 1 and 2 do not"
                         " have the same shape")
    if not np.allclose(r1, r2):
        raise ValueError("The cross sections of axis 1 and 2 are not"
                         " sampled at equivalent points ")
    # Shift the cross sections in such a way that the first hexagon
    # center resides at r=0
    shift = col0 / np.cos(np.deg2rad(30))
    _, shift = mdt.nph.find_nearest(r1, shift, return_index=True)
    cs1 = np.roll(cs1, shift=-shift)
    cs2 = np.roll(cs2, shift=-shift)
    
    rmax = min(np.max(r1), np.max(r3))
    rmax_ix = min(np.argmax(r1>=rmax), np.argmax(r3>=rmax)) - 1
    r_1nn = r1[:rmax_ix]
    cs3_interp = np.interp(x=r_1nn, xp=r3, fp=cs3)
    
    # Cross section averaged over all 3 first-nearest neighbour axes
    cs_1nn = (cs1[:rmax_ix] + cs2[:rmax_ix] + cs3_interp[:rmax_ix]) / 3
    
    print("  Elapsed time:         {}"
          .format(datetime.now()-timer_average),
          flush=True)
    print("  Current memory usage: {:14.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    if args.SHOW_PLOTS:
        print(flush=True)
        print("  Creating plot of all three cross sections", flush=True)
        timer_plot = datetime.now()
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(ax=axis,
                      x=r1,
                      y=cs1,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 1")
        mdt.plot.plot(ax=axis,
                      x=r2,
                      y=cs2,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 2")
        mdt.plot.plot(ax=axis,
                      x=r3,
                      y=cs3,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 3")
        mdt.plot.plot(ax=axis,
                      x=r_1nn,
                      y=cs_1nn,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Average")
        plt.tight_layout()
        plt.show()
        plt.close()
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
    
    
    print(flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    
    
    
    
    print("\n\n\n", flush=True)
    print("Extracting second-nearest neighbour axes", flush=True)
    timer = datetime.now()
    
    
    z = z[unit_cell_slice]
    y = y[unit_cell_slice]
    # Assume perdiodic boundary conditions.
    # Extend z (and x) in x direction, to simplify the extraction of
    # cross sections that cross the boundary
    unit_cell_slice = slice(z_shape[1], 2*z_shape[1])
    z = np.tile(z, 3)
    x_tmp = np.hstack([x, x[-1] + x[-1]-x[-2] + x])
    x = np.hstack([x - x[1]-x[0] - xmax, x_tmp])
    del x_tmp
    
    
    
    
    # Angle between the hexagonal axis and the x axis
    angle = 60
    print(flush=True)
    print("  Axis 1 of 3", flush=True)
    print("  Angle to x axis: {}°".format(angle), flush=True)
    timer_axis = datetime.now()
    # Slope and y-axis intercept of the line along which to take the
    # first sample cross section
    angle = np.deg2rad(angle)
    slope = np.tan(angle)
    if is_hex_center(test_pos=[col1, row0],
                     hex_pos=args.HEXPOS,
                     a=args.LATCONST):
        intercept_min = row0 - slope * col1  # c = y - mx
    else:
        intercept_min = row0 - slope * (col1-xdist)
    
    
    if args.SHOW_PLOTS:
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.pcolormesh(ax=axis,
                            x=np.append(x[unit_cell_slice],
                                        (2*x[unit_cell_slice][-1] -
                                           x[unit_cell_slice][-2])),
                            y=np.append(y, y[-1]+(y[-1]-y[-2])),
                            z=z[:,unit_cell_slice],
                            xmin=xmin-0.05*(xmax-xmin),
                            xmax=xmax+0.05*(xmax-xmin),
                            ymin=ymin-0.05*(ymax-ymin),
                            ymax=ymax+0.05*(ymax-ymin))
        axis.axvline(x=col0, color='red', linestyle='--')
        axis.axvline(x=col1, color='red', linestyle='--')
        axis.axhline(y=row0, color='red', linestyle='--')
        axis.axhline(y=row1, color='red', linestyle='--')
    
    
    # Cross section along axis 1/3
    cs1_1 = np.zeros(z_shape[0], dtype=np.float32)
    cs1_2 = np.zeros(z_shape[0], dtype=np.float32)
    cs1_3 = np.zeros(z_shape[0], dtype=np.float32)
    n_hex_axes = ncols + ncols//2 + ncols%2
    for i in range(n_hex_axes):
        intercept = intercept_min + 2*i*ydist
        intercept_samples = np.linspace(intercept-args.WIDTH/2,
                                        intercept+args.WIDTH/2,
                                        args.NSAMPLES,
                                        dtype=np.float32)
        
        
        if args.SHOW_PLOTS and args.WIDTH > 0:
            mask = ((x >= g_inverse(y=ymin, m=slope, c=intercept_samples[0])) &
                    (x <= g_inverse(y=ymax, m=slope, c=intercept_samples[0])))
            mdt.plot.fill_between(
                ax=axis,
                x=x[mask],
                y1=g(x=x[mask], m=slope, c=intercept_samples[-1]),
                y2=g(x=x[mask], m=slope, c=intercept_samples[0]),
                xmin=axis.get_xlim()[0],
                xmax=axis.get_xlim()[1],
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
                alpha=0.5)
        elif args.SHOW_PLOTS:
            mask = ((x >= g_inverse(y=ymin, m=slope, c=intercept_samples[0])) &
                    (x <= g_inverse(y=ymax, m=slope, c=intercept_samples[0])))
            mdt.plot.plot(
                ax=axis,
                x=x[mask],
                y=g(x=x[mask], m=slope, c=intercept_samples[0]),
                xmin=axis.get_xlim()[0],
                xmax=axis.get_xlim()[1],
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
                alpha=0.5)
        
        
        for j, sample in enumerate(intercept_samples):
            hex_ax = [(g_inverse(y=ymin, m=slope, c=sample), ymin),
                      (g_inverse(y=ymax, m=slope, c=sample), ymax)]
            cs_tmp, r1 = mdt.nph.cross_section(z=z, x=x, y=y,
                                               line=hex_ax,
                                               num=z_shape[0])
            if i%3 == 0:
                cs1_1 += cs_tmp
            elif i%3 == 1:
                cs1_2 += cs_tmp
            elif i%3 == 2:
                cs1_3 += cs_tmp
            if (i > 0 or j > 0) and not np.allclose(r1, r_prev):
                raise ValueError("The sampling points along the"
                                 " different samples of the same axis do"
                                 " not coincide. Try a lower axis width"
                                 " (-w option)")
            r_prev = r1
    # scipy.ndimage.map_coordinates in mdt.nph.cross_section usually
    # appends a zero to the cross section
    if cs1_1[-1] == 0 and cs1_2[-1] == 0 and cs1_3[-1] == 0:
        cs1_1 = cs1_1[:-1]
        cs1_2 = cs1_2[:-1]
        cs1_3 = cs1_3[:-1]
        r1 = r1[:-1]
    if r1[0] != 0:
        raise ValueError("The first sampling point is not zero")
    # Shift the cross sections in such a way that the first hexagon
    # center resides at r=0
    shift = row0 / np.sin(np.deg2rad(60))
    _, shift1 = mdt.nph.find_nearest(r1, shift, return_index=True)
    shift = (row0 + 2*ydist) / np.sin(np.deg2rad(60))
    _, shift2 = mdt.nph.find_nearest(r1, shift, return_index=True)
    shift = (row0 + ydist) / np.sin(np.deg2rad(60))
    _, shift3 = mdt.nph.find_nearest(r1, shift, return_index=True)
    print("  Elapsed time:         {}"
          .format(datetime.now()-timer_axis),
          flush=True)
    print("  Current memory usage: {:14.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    if args.SHOW_PLOTS:
        print(flush=True)
        print("  Creating plot of axes", flush=True)
        timer_plot = datetime.now()
        finish_heatmap_plot(axis=axis)
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
        
        
        print(flush=True)
        print("  Creating plot of cross section", flush=True)
        timer_plot = datetime.now()
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(ax=axis,
                      x=r1,
                      y=cs1_1,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 1.1")
        mdt.plot.plot(ax=axis,
                      x=r1,
                      y=cs1_2,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 1.2")
        mdt.plot.plot(ax=axis,
                      x=r1,
                      y=cs1_3,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 1.3")
        plt.tight_layout()
        plt.show()
        plt.close()
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
    
    
    cs1_1 = np.roll(cs1_1, shift=-shift1)
    cs1_2 = np.roll(cs1_2, shift=-shift2)
    cs1_3 = np.roll(cs1_3, shift=-shift3)
    cs1 = cs1_1 + cs1_2 + cs1_3
    cs1 /= n_hex_axes * args.NSAMPLES
    del cs1_1, cs1_2, cs1_3
    
    
    
    
    # Angle between the hexagonal axis and the x axis
    angle = 180 - 60
    print(flush=True)
    print("  Axis 2 of 3", flush=True)
    print("  Angle to x axis: {}°".format(angle), flush=True)
    timer_axis = datetime.now()
    # Slope and y-axis intercept of the line along which to take the
    # first sample cross section
    angle = np.deg2rad(angle)
    slope = np.tan(angle)
    if is_hex_center(test_pos=[col0, row0],
                     hex_pos=args.HEXPOS,
                     a=args.LATCONST):
        intercept_min = row0 - slope * col0  # c = y - mx
    else:
        intercept_min = row0 - slope * (col0+xdist)
    
    
    if args.SHOW_PLOTS:
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.pcolormesh(ax=axis,
                            x=np.append(x[unit_cell_slice],
                                        (2*x[unit_cell_slice][-1] -
                                           x[unit_cell_slice][-2])),
                            y=np.append(y, y[-1]+(y[-1]-y[-2])),
                            z=z[:,unit_cell_slice],
                            xmin=xmin-0.05*(xmax-xmin),
                            xmax=xmax+0.05*(xmax-xmin),
                            ymin=ymin-0.05*(ymax-ymin),
                            ymax=ymax+0.05*(ymax-ymin))
        axis.axvline(x=col0, color='red', linestyle='--')
        axis.axvline(x=col1, color='red', linestyle='--')
        axis.axhline(y=row0, color='red', linestyle='--')
        axis.axhline(y=row1, color='red', linestyle='--')
    
    
    # Cross section along axis 2/3
    cs2_1 = np.zeros(z_shape[0], dtype=np.float32)
    cs2_2 = np.zeros(z_shape[0], dtype=np.float32)
    cs2_3 = np.zeros(z_shape[0], dtype=np.float32)
    n_hex_axes = ncols + ncols//2 + ncols%2
    for i in range(n_hex_axes):
        intercept = intercept_min + 2*i*ydist
        intercept_samples = np.linspace(intercept-args.WIDTH/2,
                                        intercept+args.WIDTH/2,
                                        args.NSAMPLES,
                                        dtype=np.float32)
        
        
        if args.SHOW_PLOTS and args.WIDTH > 0:
            mask = ((x >= g_inverse(y=ymin, m=slope, c=intercept_samples[0])) &
                    (x <= g_inverse(y=ymax, m=slope, c=intercept_samples[0])))
            mdt.plot.fill_between(
                ax=axis,
                x=x[mask],
                y1=g(x=x[mask], m=slope, c=intercept_samples[-1]),
                y2=g(x=x[mask], m=slope, c=intercept_samples[0]),
                xmin=axis.get_xlim()[0],
                xmax=axis.get_xlim()[1],
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
                alpha=0.5)
        elif args.SHOW_PLOTS:
            mask = ((x >= g_inverse(y=ymax, m=slope, c=intercept_samples[0])) &
                    (x <= g_inverse(y=ymin, m=slope, c=intercept_samples[0])))
            mdt.plot.plot(
                ax=axis,
                x=x[mask],
                y=g(x=x[mask], m=slope, c=intercept_samples[0]),
                xmin=axis.get_xlim()[0],
                xmax=axis.get_xlim()[1],
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
                alpha=0.5)
        
        
        for j, sample in enumerate(intercept_samples):
            hex_ax = [(g_inverse(y=ymin, m=slope, c=sample), ymin),
                      (g_inverse(y=ymax, m=slope, c=sample), ymax)]
            cs_tmp, r2 = mdt.nph.cross_section(z=z, x=x, y=y,
                                               line=hex_ax,
                                               num=z_shape[0])
            if i%3 == 0:
                cs2_1 += cs_tmp
            elif i%3 == 1:
                cs2_2 += cs_tmp
            elif i%3 == 2:
                cs2_3 += cs_tmp
            if (i > 0 or j > 0) and not np.allclose(r2, r_prev):
                raise ValueError("The sampling points along the"
                                 " different samples of the same axis do"
                                 " not coincide. Try a lower axis width"
                                 " (-w option)")
            r_prev = r2
    # scipy.ndimage.map_coordinates in mdt.nph.cross_section usually
    # appends a zero to the cross section
    if cs2_1[-1] == 0 and cs2_2[-1] == 0 and cs2_3[-1] == 0:
        cs2_1 = cs2_1[:-1]
        cs2_2 = cs2_2[:-1]
        cs2_3 = cs2_3[:-1]
        r2 = r2[:-1]
    if r2[0] != 0:
        raise ValueError("The first sampling point is not zero")
    # Shift the cross sections in such a way that the first hexagon
    # center resides at r=0
    shift = row0 / np.sin(np.deg2rad(60))
    _, shift1 = mdt.nph.find_nearest(r2, shift, return_index=True)
    shift = (row0 + 2*ydist) / np.sin(np.deg2rad(60))
    _, shift2 = mdt.nph.find_nearest(r2, shift, return_index=True)
    shift = (row0 + ydist) / np.sin(np.deg2rad(60))
    _, shift3 = mdt.nph.find_nearest(r2, shift, return_index=True)
    print("  Elapsed time:         {}"
          .format(datetime.now()-timer_axis),
          flush=True)
    print("  Current memory usage: {:14.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    if args.SHOW_PLOTS:
        print(flush=True)
        print("  Creating plot of axes", flush=True)
        timer_plot = datetime.now()
        finish_heatmap_plot(axis=axis)
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
        
        
        print(flush=True)
        print("  Creating plot of cross section", flush=True)
        timer_plot = datetime.now()
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(ax=axis,
                      x=r2,
                      y=cs2_1,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 2.1")
        mdt.plot.plot(ax=axis,
                      x=r2,
                      y=cs2_2,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 2.2")
        mdt.plot.plot(ax=axis,
                      x=r2,
                      y=cs2_3,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 2.3")
        plt.tight_layout()
        plt.show()
        plt.close()
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
    
    
    cs2_1 = np.roll(cs2_1, shift=-shift1)
    cs2_2 = np.roll(cs2_2, shift=-shift2)
    cs2_3 = np.roll(cs2_3, shift=-shift3)
    cs2 = cs2_1 + cs2_2 + cs2_3
    cs2 /= n_hex_axes * args.NSAMPLES
    del cs2_1, cs2_2, cs2_3
    
    
    
    
    # Angle between the hexagonal axis and the x axis
    angle = 0
    print(flush=True)
    print("  Axis 3 of 3", flush=True)
    print("  Angle to x axis: {}°".format(angle), flush=True)
    timer_axis = datetime.now()
    
    
    if args.SHOW_PLOTS:
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.pcolormesh(ax=axis,
                            x=np.append(x[unit_cell_slice],
                                        (2*x[unit_cell_slice][-1] -
                                           x[unit_cell_slice][-2])),
                            y=np.append(y, y[-1]+(y[-1]-y[-2])),
                            z=z[:,unit_cell_slice],
                            xmin=xmin-0.05*(xmax-xmin),
                            xmax=xmax+0.05*(xmax-xmin),
                            ymin=ymin-0.05*(ymax-ymin),
                            ymax=ymax+0.05*(ymax-ymin))
        axis.axvline(x=col0, color='red', linestyle='--')
        axis.axvline(x=col1, color='red', linestyle='--')
        axis.axhline(y=row0, color='red', linestyle='--')
        axis.axhline(y=row1, color='red', linestyle='--')
    
    
    # Cross section along axis 3/3
    cs3_1 = np.zeros(z_shape[1], dtype=np.float32)
    cs3_2 = np.zeros(z_shape[1], dtype=np.float32)
    n_hex_axes = nrows
    for i in range(n_hex_axes):
        intercept = row0 + i*ydist
        intercept_samples = np.linspace(intercept-args.WIDTH/2,
                                        intercept+args.WIDTH/2,
                                        args.NSAMPLES,
                                        dtype=np.float32)
        
        
        if args.SHOW_PLOTS and args.WIDTH > 0:
            mdt.plot.fill_between(
                ax=axis,
                x=[xmin, xmax],
                y1=[intercept_samples[-1]] * 2,
                y2=[intercept_samples[0]] * 2,
                xmin=axis.get_xlim()[0],
                xmax=axis.get_xlim()[1],
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
                alpha=0.5)
        elif args.SHOW_PLOTS:
            mdt.plot.plot(
                ax=axis,
                x=[xmin, xmax],
                y=[intercept_samples[0]] * 2,
                xmin=axis.get_xlim()[0],
                xmax=axis.get_xlim()[1],
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
                alpha=0.5)
        
        
        for j, sample in enumerate(intercept_samples):
            hex_ax = [(xmin, sample), (xmax, sample)]
            cs_tmp, r3 = mdt.nph.cross_section(z=z, x=x, y=y,
                                               line=hex_ax,
                                               num=z_shape[1])
            if i%2 == 0:
                cs3_1 += cs_tmp
            else:
                cs3_2 += cs_tmp
            if (i > 0 or j > 0) and not np.allclose(r3, r_prev):
                raise ValueError("The sampling points along the"
                                 " different samples of the same axis do"
                                 " not coincide. Try a lower axis width"
                                 " (-w option)")
            r_prev = r3
    # scipy.ndimage.map_coordinates in mdt.nph.cross_section usually
    # appends a zero to the cross section
    if cs3_1[-1] == 0 and cs3_2[-1] == 0:
        cs3_1 = cs3_1[:-1]
        cs3_2 = cs3_2[:-1]
        r3 = r3[:-1]
    if r3[0] != 0:
        raise ValueError("The first sampling point is not zero")
    # Shift the cross sections in such a way that the first hexagon
    # center resides at r=0
    if is_hex_center(test_pos=[col0, row0],
                     hex_pos=args.HEXPOS,
                     a=args.LATCONST):
        _, shift1 = mdt.nph.find_nearest(r3, col0, return_index=True)
        _, shift2 = mdt.nph.find_nearest(r3,
                                         col0+xdist,
                                         return_index=True)
    else:
        _, shift1 = mdt.nph.find_nearest(r3,
                                         col0+xdist,
                                         return_index=True)
        _, shift2 = mdt.nph.find_nearest(r3, col0, return_index=True)
    print("  Elapsed time:         {}"
          .format(datetime.now()-timer_axis),
          flush=True)
    print("  Current memory usage: {:14.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    if args.SHOW_PLOTS:
        print(flush=True)
        print("  Creating plot of axes", flush=True)
        timer_plot = datetime.now()
        finish_heatmap_plot(axis=axis)
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
        
        
        print(flush=True)
        print("  Creating plot of cross section", flush=True)
        timer_plot = datetime.now()
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(ax=axis,
                      x=r3,
                      y=cs3_1,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 3.1")
        mdt.plot.plot(ax=axis,
                      x=r3,
                      y=cs3_2,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 3.2")
        plt.tight_layout()
        plt.show()
        plt.close()
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
    
    
    cs3_1 = np.roll(cs3_1, shift=-shift1)
    cs3_2 = np.roll(cs3_2, shift=-shift2)
    cs3 = cs3_1 + cs3_2
    cs3 /= n_hex_axes * args.NSAMPLES
    del cs3_1, cs3_2
    
    
    
    
    print(flush=True)
    print("  Averaging over all 3 axes", flush=True)
    timer_average = datetime.now()
    
    if cs1.shape != cs2.shape:
        raise ValueError("The cross sections of axis 1 and 2 do not"
                         " have the same shape")
    if not np.allclose(r1, r2):
        raise ValueError("The cross sections of axis 1 and 2 are not"
                         " sampled at equivalent points ")
    rmax = min(np.max(r1), np.max(r3))
    rmax_ix = min(np.argmax(r1>=rmax), np.argmax(r3>=rmax)) - 1
    r_2nn = r1[:rmax_ix]
    cs3_interp = np.interp(x=r_2nn, xp=r3, fp=cs3)
    
    # Cross section averaged over all 3 second-nearest neighbour axes
    cs_2nn = (cs1[:rmax_ix] + cs2[:rmax_ix] + cs3_interp[:rmax_ix]) / 3
    
    print("  Elapsed time:         {}"
          .format(datetime.now()-timer_average),
          flush=True)
    print("  Current memory usage: {:14.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    if args.SHOW_PLOTS:
        print(flush=True)
        print("  Creating plot of all three cross sections", flush=True)
        timer_plot = datetime.now()
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(ax=axis,
                      x=r1,
                      y=cs1,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 1")
        mdt.plot.plot(ax=axis,
                      x=r2,
                      y=cs2,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 2")
        mdt.plot.plot(ax=axis,
                      x=r3,
                      y=cs3,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Axis 3")
        mdt.plot.plot(ax=axis,
                      x=r_2nn,
                      y=cs_2nn,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$',
                      label="Average")
        plt.tight_layout()
        plt.show()
        plt.close()
        print("  Elapsed time:         {}"
              .format(datetime.now()-timer_plot),
              flush=True)
        print("  Current memory usage: {:14.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
    
    
    print(flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()
    
    
    header = (
        "Cross section along first-nearest and second-nearest neighbour\n"
        "axes of a hexagonal density map.\n"
        "\n"
        "All data are in the units of the input file.\n"
        "\n"
        "\n"
        "Lattice constant:                             {}\n"
        "Position of an arbitrary hexagon center:\n"
        "  x: {}\n"
        "  y: {}\n"
        "Cross section width:                          {}\n"
        "Number of samples for a single cross section: {}\n"
        "\n"
        "\n"
        "The columns contain:\n"
        "  1 Sample points for column 2\n"
        "  2 Cross section averaged over all 1st-nearest neighbour axes\n"
        "  3 Sample points for column 4\n"
        "  4 Cross section averaged over all 2nd-nearest neighbour axes\n"
        "\n"
        "Column number:\n"
        "{:>14d} {:>16d} {:>16d} {:>16d}\n"
        .format(args.LATCONST,
                args.HEXPOS[0], args.HEXPOS[1],
                args.WIDTH,
                args.NSAMPLES,
                1, 2, 3, 4)
    )
    
    rmax = min(np.max(r_1nn), np.max(r_2nn))
    rmax_ix = min(np.argmax(r_1nn>=rmax), np.argmax(r_2nn>=rmax)) - 1
    data = np.column_stack([r_1nn[:rmax_ix], cs_1nn[:rmax_ix],
                            r_2nn[:rmax_ix], cs_2nn[:rmax_ix]])
    mdt.fh.savetxt(fname=args.OUTFILE+".txt",
                   data=data,
                   header=header)
    print("  Created {}".format(args.OUTFILE+".txt"), flush=True)
    
    
    
    
    z = z[:,unit_cell_slice]
    x = x[unit_cell_slice]
    mdt.fh.backup(args.OUTFILE+".pdf")
    with PdfPages(args.OUTFILE+".pdf") as pdf:
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.pcolormesh(ax=axis,
                            x=np.append(x, x[-1]+(x[-1]-x[-2])),
                            y=np.append(y, y[-1]+(y[-1]-y[-2])),
                            z=z,
                            xmin=xmin,
                            xmax=xmax,
                            ymin=ymin,
                            ymax=ymax,
                            xlabel=r'$x$ / nm',
                            ylabel=r'$y$ / nm',
                            cbarlabel=r'$\rho(x, y)$ / nm$^{-3}$')
        xsize = abs(axis.get_xlim()[0] - axis.get_xlim()[1])
        ysize = abs(axis.get_ylim()[0] - axis.get_ylim()[1])
        axis.set_aspect(ysize / xsize)
        yticks = np.array(axis.get_yticks())
        mask = ((yticks >= axis.get_xlim()[0]) &
                (yticks <= axis.get_xlim()[1]))
        axis.set_xticks(yticks[mask])
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(ax=axis,
                      x=r_1nn,
                      y=cs_1nn,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        mdt.plot.plot(ax=axis,
                      x=r_2nn,
                      y=cs_2nn,
                      xlabel=r'$r$ / nm',
                      ylabel=r'$\rho_{xy}(r)$ / nm$^{-3}$')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n{} done".format(os.path.basename(sys.argv[0])))
    print("Elapsed time:         {}"
          .format(datetime.now()-timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
