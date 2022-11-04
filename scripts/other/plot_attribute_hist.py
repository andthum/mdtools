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


r"""
Plot histograms of a given attribute of an MDAnalysis
:class:`~MDAnalysis.core.groups.AtomGroup`.

The attribute is selected with \--attribute.

Four histograms are plotted: One for all values of the selected
attribute and three for the three spatial dimensions of the selected
attribute.

Options
-------
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
--updating-sel
    Use an :class:`~MDAnalysis.core.groups.UpdatingAtomGroup` for the
    analysis.  Selection expressions of UpdatingAtomGroups are
    re-evaluated every :attr:`time step
    <MDAnalysis.coordinates.base.Timestep.dt>`.  This is e.g. useful for
    position-based selections like ``'type Li and prop z <= 2.0'``.
--attribute
    {velocities, forces, positions}

    The attribute of the selection group for which to create the
    histograms.  The trajectory must contain information about this
    attribute.  Default: ``'velocities'``.
--density
    If given, the histograms are normed such that the integral over the
    binned region is 1.
--fit
    If given, fit the created histograms with a Gaussian function.  Note
    that not the underlying data but only the final histograms are
    fitted.
--bin-start
    First bin edge in data units (inclusive).  If ``None``, the minimum
    value of the selected attribute is taken.  Default: ``None``.
--bin-stop
    Last bin edge in data units (inclusive).  If ``None``, the maximum
    value of the selected attribute is taken.  Default: ``None``.
--bin-num
    Number of bin edges (the number of bins is the number of bin edges
    minus one).  If ``None``, the number of bins is estimated according
    to the "scott" method as described by
    :func:`numpy.histogram_bin_edges`:

    .. math::

        h = \sigma \sqrt[3]{\frac{24 \sqrt{\pi}}{n}}

    Here, :math:`\sigma` is the standard deviation of the data,
    :math:`n` is the number of data points and :math:`h` is the bin
    width.  The number of bins is calculated by
    ``ceil((start - stop) / bin_width)``.
"""


__author__ = "Andreas Thum"


# Standard libraries
import argparse
import os
import sys
from datetime import datetime, timedelta

# Third-party libraries
import numpy as np
import psutil
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText
from scipy import optimize

# First-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # MDTools plotting style  # noqa: F401


if __name__ == "__main__":  # noqa: C901
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage.
    parser = argparse.ArgumentParser(description=(""))
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
        "--updating-sel",
        dest="UPDATING_SEL",
        required=False,
        default=False,
        action="store_true",
        help="Use an UpdatingAtomGroup for the analysis.",
    )
    parser.add_argument(
        "--attribute",
        dest="ATTR",
        required=False,
        choices=("velocities", "forces", "positions"),
        default="velocities",
        help=(
            "The attribute of the selection group for which to create the"
            " histograms.  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--density",
        dest="DENSITY",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Norm the histograms such that the integral over the binned region"
            " is 1."
        ),
    )
    parser.add_argument(
        "--fit",
        dest="FIT",
        required=False,
        default=False,
        action="store_true",
        help="Fit the histograms with a Gaussian function.",
    )
    parser.add_argument(
        "--bin-start",
        dest="BIN_START",
        type=float,
        required=False,
        default=None,
        help=(
            "First bin edge in data units (inclusive).  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--bin-stop",
        dest="BIN_STOP",
        type=float,
        required=False,
        default=None,
        help="Last bin edge in data units (inclusive).  Default: %(default)s.",
    )
    parser.add_argument(
        "--bin-num",
        dest="BIN_NUM",
        type=int,
        required=False,
        default=None,
        help="Number of bin edges.  Default: %(default)s.",
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    if args.BIN_NUM is not None and args.BIN_NUM <= 0:
        raise ValueError(
            "--bin-num ({}) must be positive".format(args.BIN_NUM)
        )

    # Attribute units.  See
    # https://userguide.mdanalysis.org/stable/units.html?highlight=units
    attr_unit = {
        "velocities": "A ps$^{-1}$",
        "forces": "kJ mol$^{-1}$ A$^{-1}$",
        "positions": "A",
    }
    dimension = {0: "$x$, $y$ and $z$", 1: "$x$", 2: "$y$", 3: "$z$"}

    print("\n")
    u = mdt.select.universe(top=args.TOPFILE, trj=args.TRJFILE)
    print("\n")
    sel = mdt.select.atoms(
        ag=u, sel=" ".join(args.SEL), updating=args.UPDATING_SEL
    )
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
    print("Total number of frames: {:>8d}".format(u.trajectory.n_frames))
    print("Frames to read:         {:>8d}".format(N_FRAMES))
    print("First frame to read:    {:>8d}".format(BEGIN))
    print("Last frame to read:     {:>8d}".format(END - 1))
    print("Read every n-th frame:  {:>8d}".format(EVERY))
    print("Time first frame:       {:>12.3f} ps".format(first_frame_read.time))
    print("Time last frame:        {:>12.3f} ps".format(last_frame_read.time))
    print("Time step first frame:  {:>12.3f} ps".format(first_frame_read.dt))
    print("Time step last frame:   {:>12.3f} ps".format(last_frame_read.dt))

    print("\n")
    print("Inspecting data...")
    timer = datetime.now()
    attr = getattr(sel, args.ATTR).astype(np.float64)
    n_samples = 0
    means = np.zeros(attr.shape[1], dtype=np.float64)
    mean_squares = np.zeros_like(means)
    mins = attr[0]
    maxs = np.copy(mins)
    trj = mdt.rti.ProgressBar(u.trajectory[BEGIN:END:EVERY])
    for _ts in trj:
        attr = getattr(sel, args.ATTR).astype(np.float64)
        n_samples += attr.shape[0]
        means += np.sum(attr, axis=0, dtype=np.float64)
        mean_squares += np.sum(attr**2, axis=0, dtype=np.float64)
        mins = np.min([mins, np.min(attr, axis=0)], axis=0)
        maxs = np.max([maxs, np.max(attr, axis=0)], axis=0)
        # ProgressBar update.
        trj.set_postfix_str(
            "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
        )
    trj.close()
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    n_samples_tot = n_samples * attr.shape[1]
    mean_tot = np.sum(means) / n_samples_tot
    mean_squares_tot = np.sum(mean_squares) / n_samples_tot

    means /= n_samples
    means = np.insert(means, 0, mean_tot)
    mean_squares /= n_samples
    mean_squares = np.insert(mean_squares, 0, mean_squares_tot)
    stds = np.sqrt(mean_squares - means**2)
    n_samples = np.array(
        (n_samples_tot,) + (n_samples,) * attr.shape[1], dtype=np.uint64
    )

    mins = np.insert(mins, 0, np.min(mins))
    maxs = np.insert(maxs, 0, np.max(maxs))

    if args.BIN_NUM is None:
        # Bin width according to the 'scott' method of
        # `numpy.histogram_bin_edges`.
        bin_widths = stds * np.cbrt(24 * np.sqrt(np.pi) / n_samples)
        n_bins = np.ceil((maxs - mins) / bin_widths).astype(int)
    else:
        n_bins = (args.BIN_NUM,) * len(means)
    bins = []
    for i, nb in enumerate(n_bins):
        if args.BIN_START is None:
            start = mins[i]
        else:
            start = args.BIN_START
        if args.BIN_STOP is None:
            stop = maxs[i]
        else:
            stop = args.BIN_STOP
        if stop <= start:
            raise ValueError(
                "The last bin edge ({}) must be greater than the first bin"
                " edge ({})".format(stop, start)
            )
        bins.append(np.linspace(start, stop, nb))

    print("\n")
    print("Creating histograms...")
    timer = datetime.now()
    hists = [np.zeros(len(bns) - 1, dtype=np.uint64) for bns in bins]
    trj = mdt.rti.ProgressBar(u.trajectory[BEGIN:END:EVERY])
    for _ts in trj:
        attr = getattr(sel, args.ATTR)
        hist, bin_edges = np.histogram(
            np.ravel(attr),
            bins=bins[0],
            range=(mins[0], maxs[0]),
            density=False,
        )
        if not np.allclose(bin_edges, bins[0], rtol=0):
            raise ValueError(
                "The bin edges have changed.  This should not have happened"
            )
        hists[0] += hist.astype(np.uint64)
        for i, at in enumerate(attr.T, 1):
            hist, bin_edges = np.histogram(
                at, bins=bins[i], range=(mins[i], maxs[i]), density=False
            )
            hists[i] += hist.astype(np.uint64)
            if not np.allclose(bin_edges, bins[i], rtol=0):
                raise ValueError(
                    "The bin edges have changed.  This should not have"
                    " happened"
                )
        # ProgressBar update.
        trj.set_postfix_str(
            "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
        )
    trj.close()
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    for hist in hists:
        if np.any(hist < 0):
            raise RuntimeError("Overflow encounterd in 'hists'")

    # Convert bin edges to bin widths.
    bin_widths = []
    for bns in bins:
        bin_widths.append(np.diff(bns))
    # Convert bin edges to bin midpoints.
    bin_mids = []
    for i, bns in enumerate(bins):
        bin_mids.append(bns[1:] - bin_widths[i] / 2)
    if args.DENSITY:
        # Norm the histograms such that their integral is 1.
        for i, hist in enumerate(hists):
            integral = np.trapz(y=hist, x=bin_mids[i])
            hists[i] = hist / integral

    print("\n")
    print("Creating output...")
    timer = datetime.now()
    if args.DENSITY:
        ylabel = "Probability"
    else:
        ylabel = "Count"
    mdt.fh.backup(args.OUTFILE)
    with PdfPages(args.OUTFILE) as pdf:
        for i, hist in enumerate(hists):
            fig, ax = plt.subplots(clear=True)
            bars = ax.bar(
                bin_mids[i],
                height=hist,
                width=bin_widths[i],
                color="tab:blue",
                edgecolor="tab:blue",
                rasterized=True,
            )

            if args.FIT:
                try:
                    popt, pcov = optimize.curve_fit(
                        mdt.stats.gaussian,
                        xdata=bin_mids[i],
                        ydata=hist,
                        p0=(means[i], stds[i]),
                    )
                except (ValueError, RuntimeError) as err:
                    print("An error has occurred during fitting:")
                    print("{}".format(err), flush=True)
                    print("Skipping this fit")
                else:
                    lines = ax.plot(
                        bin_mids[i],
                        mdt.stats.gaussian(bin_mids[i], *popt),
                        color="tab:orange",
                    )
                at_data = AnchoredText(
                    (
                        "Gaussian Fit\n"
                        + "Mean: {:.3f}\n".format(popt[0])
                        + "StD: {:.3f}".format(popt[1])
                    ),
                    loc="upper right",
                    prop={
                        "fontsize": "xx-small",
                        "color": lines[0].get_color(),
                    },  # Text properties
                )
                at_data.patch.set(alpha=0.75, edgecolor="lightgrey")
                ax.add_artist(at_data)

            xlabel = args.ATTR.title() + " " + dimension[i]
            xlabel += " / " + attr_unit[args.ATTR]
            ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(0, None))
            at_data = AnchoredText(
                (
                    "Data (n = {})\n".format(n_samples[i])
                    + "Mean: {:.3f}\n".format(means[i])
                    + "StD: {:.3f}\n".format(stds[i])
                    + "Min: {:.3f}\n".format(mins[i])
                    + "Max: {:.3f}".format(maxs[i])
                ),
                loc="upper left",
                prop={
                    "fontsize": "xx-small",
                    "color": bars[0].get_facecolor(),
                },  # Text properties
            )
            at_data.patch.set(alpha=0.75, edgecolor="lightgrey")
            ax.add_artist(at_data)
            pdf.savefig()
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
