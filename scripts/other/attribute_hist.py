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
Compute histograms of a given spatial attribute of an MDAnalysis
:class:`~MDAnalysis.core.groups.AtomGroup`.

The spatial attribute is selected with \--attribute.  It is always
calculated with respect to the center of mass of the selected compound.

Five histograms are plotted:

    * Three histograms showing the distribution of each spatial
      dimension of the selected attribute.
    * One histogram showing the combined distribution of all spatial
      dimensions of the selected attribute.
    * One histogram showing the distribution of Euclidean norms (2-norm)
      of the attribute vectors.

Options
-------
-f
    Trajectory file.  See |supported_coordinate_formats| of MDAnalysis.
-s
    Topology file.  See |supported_topology_formats| of MDAnalysis.
-o
    Five output filenames, one for each histogram.  Alternatively, a
    single filename can be given which will be used as basename for the
    five histogram files.
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
--cmp
    {'group', 'segments', 'residues', 'molecules', 'fragments', 'atoms'}

    The compounds of the selection group to use for the analysis.
    Compounds can be 'group' (the entire selection group), 'segments',
    'residues', 'molecules', 'fragments', or 'atoms'.  Refer to the
    MDAnalysis' user guide for an |explanation_of_these_terms|.  Note
    that in any case, even if ``CMP`` is e.g. 'residues', only the atoms
    belonging to the selection group are taken into account, even if the
    compound might comprise additional atoms that are not contained in
    the selection group.  In all cases, the center of mass of the
    compound is used for the analysis.  Note that
    |MDA_always_guesses_atom_masses| from the atom types, even if the
    input file contains the masses.  Default: ``'atoms'``.
--attribute
    {'velocities', 'forces', 'positions'}

    The attribute of the selection group for which to create the
    histograms.  The trajectory must contain information about this
    attribute.  Default: ``'velocities'``.
--bin-start
    First bin edge of the histograms in data units (inclusive).  If
    ``None``, the minimum value of the selected attribute is taken.
    Default: ``None``.
--bin-stop
    Last bin edge of the histograms in data units (inclusive).  If
    ``None``, the maximum value of the selected attribute is taken.
    Default: ``None``.
--bin-num
    Number of bin edges of the histograms (the number of bins is the
    number of bin edges minus one).  If ``None``, the number of bins is
    estimated according to the "scott" method as described by
    :func:`numpy.histogram_bin_edges`:

    .. math::

        h = \sigma \sqrt[3]{\frac{24 \sqrt{\pi}}{n}}

    Here, :math:`\sigma` is the standard deviation of the data,
    :math:`n` is the number of data points and :math:`h` is the bin
    width.  The number of bins is calculated by
    ``np.ceil((bin_stop - bin_start) / bin_width)``.

Notes
-----
If the selected attribute is 'velocities' or 'forces', all created
histograms, except the last one (Euclidean norm), are fitted by Gaussian
distribution functions:

.. math::

    g(x) = \frac{1}{\sqrt{2\pi\sigma^2}}
    e^{-\frac{(x - \mu)^2}{2\sigma^2}}

If the selected attribute is 'velocities', the last histogram (Euclidean
norm) is fitted by a Maxwell-Boltzmann speed distribution function:

.. math::

    p(v) = 4 \pi (v - u)^2
    \left(\frac{1}{2\pi\sigma^2}\right)^{\frac{3}{2}}
    e^{-\frac{(v - u)^2}{2\sigma^2}}

with :math:`\sigma^2 = \frac{k_B T}{m}`, :math:`k_B` being Boltzmann's
constant, :math:`T` being the temperature and :math:`m` being the
average mass of the selected compounds.  :math:`u` is a drift velocity
that simply shifts the distribution to the right.  :math:`\sigma^2` and
:math:`u` are used as fit parameters.  Afterwards, the temperature is
calculated from :math:`\sigma^2` according to :math:`T = m\sigma^2/k_B`,
where :math:`m` is set to the average mass of all selected compounds.

Usually, it is a bad idea to compute overall histograms of multiple
compounds in the trajectory.  Instead, better compute separate
histograms for each compound.  In other words, only select compounds of
the same type with \--sel and \--cmp (e.g. all water molecules or all
oxygen atoms belonging to water molecules).
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
from scipy import constants, optimize

# First-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # MDTools plotting style  # noqa: F401


def _fit_gaussian(xdata, ydata, p0=None):
    """Fit the given data by a Gaussian function."""
    try:
        popt, pcov = optimize.curve_fit(
            mdt.stats.gaussian,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            bounds=[(-np.inf, 0), (np.inf, np.inf)],
        )
        perr = np.sqrt(np.diag(pcov))
    except (ValueError, RuntimeError) as err:
        print("An error has occurred during fitting:")
        print("{}".format(err), flush=True)
        print("Skipping this fit")
        fit = np.full_like(ydata, np.nan)
        popt = np.full(2, np.nan)
        perr = np.full_like(popt, np.nan)
    else:
        fit = mdt.stats.gaussian(xdata, *popt)
    return fit, popt, perr


def _fit_mb(xdata, ydata):
    """Fit the given data by a Maxwell-Boltzmann speed distribution."""
    func = lambda v, var, drift: mdt.stats.mb_dist(  # noqa: E731
        v=v, var=var, drift=drift
    )
    try:
        # Initial guess for `var` is kT/m with T = 273 K and m = 12 u.
        # The factor 1e-4 comes from the conversion of (m/s)^2 to
        # (A/ps)^2.
        var_guess = constants.k * 273 / (12 * constants.atomic_mass) * 1e-4
        popt, pcov = optimize.curve_fit(
            func,
            xdata=xdata,
            ydata=ydata,
            p0=(var_guess, 0),
            bounds=[(0, 0), (np.inf, np.inf)],
        )
        perr = np.sqrt(np.diag(pcov))
    except (ValueError, RuntimeError) as err:
        print("An error has occurred during fitting:")
        print("{}".format(err), flush=True)
        print("Skipping this fit")
        fit = np.full_like(ydata, np.nan)
        popt = np.full(2, np.nan)
        perr = np.full_like(popt, np.nan)
    else:
        fit = func(xdata, *popt)
    return fit, popt, perr


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
        dest="OUTFILES",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Five output filenames, one for each histogram.  Alternatively, a"
            " single filename can be given which will be used as basename for"
            " the five histogram files."
        ),
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
            "The compounds of the selection group to use for the analysis.  In"
            " all cases, the center of mass of the compound is used for the"
            " analysis.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--attribute",
        dest="ATTR",
        type=str,
        required=False,
        choices=("velocities", "forces", "positions"),
        default="velocities",
        help=(
            "The attribute of the selection group for which to create the"
            " histograms.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--bin-start",
        dest="BIN_START",
        type=float,
        required=False,
        default=None,
        help=(
            "First bin edge in data units (inclusive).  Default: Minimum value"
            " of the selected attribute."
        ),
    )
    parser.add_argument(
        "--bin-stop",
        dest="BIN_STOP",
        type=float,
        required=False,
        default=None,
        help=(
            "Last bin edge in data units (inclusive).  Default: Maximum value"
            " of the selected attribute."
        ),
    )
    parser.add_argument(
        "--bin-num",
        dest="BIN_NUM",
        type=int,
        required=False,
        default=None,
        help=(
            "Number of bin edges.  Default: Determined automatically"
            " according to the 'scott' method."
        ),
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    if len(args.OUTFILES) == 1:
        outfiles = [
            args.OUTFILES[0] + "_x.txt",
            args.OUTFILES[0] + "_y.txt",
            args.OUTFILES[0] + "_z.txt",
            args.OUTFILES[0] + "_xyz.txt",
            args.OUTFILES[0] + "_norm.txt",
        ]
    elif len(args.OUTFILES) != 5:
        raise ValueError(
            "You must give either one or five filenames with -o"
            " ({})".format(args.OUTFILES)
        )
    else:
        outfiles = args.OUTFILES
    if (
        args.BIN_START is not None
        and args.BIN_STOP is not None
        and args.BIN_STOP <= args.BIN_START
    ):
        raise ValueError(
            "--bin-stop ({}) must be greater than --bin-start"
            " ({})".format(args.BIN_STOP, args.BIN_START)
        )
    if args.BIN_NUM is not None and args.BIN_NUM <= 0:
        raise ValueError(
            "--bin-num ({}) must be positive".format(args.BIN_NUM)
        )

    # Attribute units.  See
    # https://userguide.mdanalysis.org/stable/units.html?highlight=units
    ATTR_UNITS = {
        "velocities": "A/ps",
        "forces": "kJ/(mol*A)",
        "positions": "A",
    }
    DIMENSION = {
        0: "x",
        1: "y",
        2: "z",
        3: "x, y and z",
        4: "sqrt(x^2 + y^2 + z^2)",
    }

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

    if args.UPDATING_SEL:
        napc = None
    else:
        napc = mdt.strc.natms_per_cmp(sel, cmp=args.CMP, check_contiguous=True)
    if args.ATTR in ("velocities", "forces"):
        weights = "total"
    elif args.ATTR == "positions":
        weights = "masses"

    print("\n")
    print("Inspecting data...")
    timer = datetime.now()
    attr = mdt.strc.cmp_attr(
        sel, attr=args.ATTR, weights=weights, cmp=args.CMP, natms_per_cmp=napc
    )
    n_samples = 0
    # The meaning of the indices of the following arrays is described by
    # the `DIMENSION` dictionary.
    N_DIMS = attr.shape[1]
    N_EXTRA_HISTS = 2
    N_HISTS = N_DIMS + N_EXTRA_HISTS
    moment1 = np.zeros(N_HISTS, dtype=np.float64)
    moment2 = np.zeros_like(moment1)
    mins = np.append(attr[0], [attr[0][0]] * N_EXTRA_HISTS)
    mins = np.abs(mins, out=mins)
    maxs = -mins
    trj = mdt.rti.ProgressBar(u.trajectory[BEGIN:END:EVERY])
    for _ts in trj:
        attr = mdt.strc.cmp_attr(
            sel,
            attr=args.ATTR,
            weights=weights,
            cmp=args.CMP,
            natms_per_cmp=napc,
        )
        n_samples += attr.shape[0]
        moment1[:N_DIMS] += np.nansum(attr, axis=0)
        moment2[:N_DIMS] += np.nansum(attr**2, axis=0)
        attr_mins = np.nanmin(attr, axis=0)
        attr_maxs = np.nanmax(attr, axis=0)
        mins[:N_DIMS] = np.nanmin([mins[:N_DIMS], attr_mins], axis=0)
        maxs[:N_DIMS] = np.nanmax([maxs[:N_DIMS], attr_maxs], axis=0)
        attr_norms = np.linalg.norm(attr, axis=1)
        moment1[-1] += np.nansum(attr_norms)
        moment2[-1] += np.nansum(attr_norms**2)
        mins[-1] = np.nanmin([mins[-1], np.nanmin(attr_norms)])
        maxs[-1] = np.nanmax([maxs[-1], np.nanmax(attr_norms)])
        # ProgressBar update.
        trj.set_postfix_str(
            "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
        )
    trj.close()
    del attr_mins, attr_maxs, attr_norms
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    n_samples = np.full(N_HISTS, n_samples, dtype=np.uint64)
    n_samples[N_DIMS] = n_samples[N_DIMS] * N_DIMS
    moment1[N_DIMS] = np.sum(moment1[:N_DIMS])
    moment2[N_DIMS] = np.sum(moment2[:N_DIMS])
    moment1 /= n_samples
    moment2 /= n_samples
    stds = np.sqrt(moment2 - moment1**2)
    mins[N_DIMS] = np.min(mins[:N_DIMS])
    maxs[N_DIMS] = np.max(maxs[:N_DIMS])

    bins = []
    for i in range(N_HISTS):
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
        if args.BIN_NUM is None:
            # Bin width according to the 'scott' method of
            # `numpy.histogram_bin_edges`.
            bin_width = stds[i] * np.cbrt(24 * np.sqrt(np.pi) / n_samples[i])
            n_bins = int(np.ceil((stop - start) / bin_width))
        else:
            n_bins = args.BIN_NUM
        bins.append(np.linspace(start, stop, n_bins + 1))
        if i == N_HISTS - 1:
            # Histogram of the Euclidean norms.  Euclidean norms are
            # always positive => bins should be positive, too.
            valid = bins[i] >= 0
            if np.any(valid):
                print("\n")
                print(
                    "Note: Discarding negative bin edges for the histogram of"
                    " Euclidean norms"
                )
                bins[i] = bins[i][valid]
            else:
                bins[i] = np.abs(bins[i], out=bins[i])
                print("\n")
                print(
                    "Note: All bin edges are negative.  Taking absolute bin"
                    " edges for the histogram of Euclidean norms"
                )

    print("\n")
    print("Creating histograms...")
    timer = datetime.now()
    hists = [np.zeros(len(bns) - 1, dtype=np.uint64) for bns in bins]
    trj = mdt.rti.ProgressBar(u.trajectory[BEGIN:END:EVERY])
    for _ts in trj:
        attr = mdt.strc.cmp_attr(
            sel,
            attr=args.ATTR,
            weights=weights,
            cmp=args.CMP,
            natms_per_cmp=napc,
        )
        # One histogram for each spatial dimension.
        for i, at in enumerate(attr.T):
            hist, bin_edges = np.histogram(
                at, bins=bins[i], range=(mins[i], maxs[i]), density=False
            )
            hists[i] += hist.astype(np.uint64)
            if not np.allclose(bin_edges, bins[i], rtol=0):
                raise ValueError(
                    "The bin edges have changed.  This should not have"
                    " happened"
                )
        # Combined histogram of all spatial dimensions.
        hist, bin_edges = np.histogram(
            np.ravel(attr),
            bins=bins[N_DIMS],
            range=(mins[N_DIMS], maxs[N_DIMS]),
            density=False,
        )
        hists[N_DIMS] += hist.astype(np.uint64)
        if not np.allclose(bin_edges, bins[N_DIMS], rtol=0):
            raise ValueError(
                "The bin edges have changed.  This should not have happened"
            )
        # Histogram of Euclidean norms.
        hist, bin_edges = np.histogram(
            np.linalg.norm(attr, axis=1),
            bins=bins[-1],
            range=(mins[-1], maxs[-1]),
            density=False,
        )
        hists[-1] += hist.astype(np.uint64)
        if not np.allclose(bin_edges, bins[-1], rtol=0):
            raise ValueError(
                "The bin edges have changed.  This should not have happened"
            )
        # ProgressBar update.
        trj.set_postfix_str(
            "{:>7.2f}MiB".format(mdt.rti.mem_usage(proc)), refresh=False
        )
    trj.close()
    del attr, at, bin_edges
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Fitting histograms and creating output...")
    timer = datetime.now()
    header_base = (
        "Selection string:  '{:s}'\n".format(" ".join(args.SEL))
        + "Selection compound: '{:s}'\n".format(args.CMP)
        + mdt.rti.ag_info_str(sel)
        + "\n\n\n"
        + "Total number of frames: {:>8d}\n".format(u.trajectory.n_frames)
        + "Frames read:            {:>8d}\n".format(N_FRAMES)
        + "First frame read:       {:>8d}\n".format(BEGIN)
        + "Last frame read:        {:>8d}\n".format(END - 1)
        + "Read every n-th frame:  {:>8d}\n".format(EVERY)
        + "Time first frame:       {:>12.3f} ps\n".format(
            first_frame_read.time
        )
        + "Time last frame:        {:>12.3f} ps\n".format(last_frame_read.time)
        + "\n\n"
        + "Histogram of the center-of-mass {:s}".format(args.ATTR)
        + " of the selected compounds.\n"
        + "\n"
    )
    for i, hist in enumerate(hists):
        header = header_base + (
            "Examined spatial component of the {:s}: {:s}\n".format(
                args.ATTR, DIMENSION[i]
            )
            + "Total number of samples:          {:>16d}\n".format(
                n_samples[i]
            )
            + "Total number of histogram counts: {:>16d}\n".format(
                np.sum(hist)
            )
            + "Number of bins:                   {:>16d}\n".format(
                len(bins[i]) - 1
            )
            + "First bin edge in {:s}: {:>16.9e}\n".format(
                ATTR_UNITS[args.ATTR], bins[i][0]
            )
            + "Last  bin edge in {:s}: {:>16.9e}\n".format(
                ATTR_UNITS[args.ATTR], bins[i][-1]
            )
            + "Bin width in {:s}:      {:>16.9e}\n".format(
                ATTR_UNITS[args.ATTR], bins[i][1] - bins[i][0]
            )
            + "\n"
            + "The columns contain:\n"
            + "  1 Bin midpoints in {:s}\n".format(ATTR_UNITS[args.ATTR])
            + "  2 Counts\n"
            + "  3 Probability density\n"
        )
        bin_mids = bins[i][1:] - np.diff(bins[i]) / 2
        integral = np.trapz(y=hist, x=bin_mids)
        hist_normed = hist / integral
        del integral
        data = np.column_stack([bin_mids, hist, hist_normed])
        if i < N_HISTS - 1 and args.ATTR in ("velocities", "forces"):
            # Fit histogram by a Gaussian function.
            fit, popt, perr = _fit_gaussian(
                xdata=bin_mids, ydata=hist_normed, p0=(moment1[i], stds[i])
            )
            data = np.column_stack([data, fit])
            header += (
                "  4 Gaussian fit of the probability density\n"
                "    p(x) = 1/sqrt(2*pi*sigma^2) * "
                "exp[-(x-mu)^2 / (2*sigma^2)]\n"
                + "\n"
                + "{:>14d} {:>16d} {:>16d} {:>16d}\n".format(1, 2, 3, 4)
                + "{:<14s} {:>16.9e} {:>16.9e} {:>16.9e}\n".format(
                    "Mean:", moment1[i], moment1[i], popt[0]
                )
                + "{:<14s} {:>50.9e}\n".format("Fit param StD:", perr[0])
                + "{:<14s} {:>16.9e} {:>16.9e} {:>16.9e}\n".format(
                    "StD:", stds[i], stds[i], popt[1]
                )
                + "{:<14s} {:>50.9e}\n".format("Fit param StD:", perr[1])
            )
        elif i == N_HISTS - 1 and args.ATTR == "velocities":
            # Fit histogram of the Euclidean norm by a Maxwell-Boltzmann
            # speed distribution.
            # `bin_mids` is given in [A/ps].
            fit, popt, perr = _fit_mb(xdata=bin_mids, ydata=hist_normed)
            data = np.column_stack([data, fit])
            aps2ms = 1e2  # Conversion factor [A/ps] -> [m/s].
            ms2aps = 1 / aps2ms  # Conversion factor [m/s] -> [A/ps].
            sigma2_ms = popt[0] * aps2ms**2  # sigma^2 in [(m/s)^2].
            mass = np.nanmean(sel.masses)  # Mass in [u].
            mass_kg = mass * constants.atomic_mass  # Mass in [kg].
            temp = mass_kg * sigma2_ms / constants.k  # Temperature in [K].
            v_p = np.sqrt(2 * constants.k * temp / mass_kg)
            v_p *= ms2aps  # Most probable speed in [A/ps].
            moment1_mb = np.sqrt(8 * constants.k * temp / (np.pi * mass_kg))
            moment1_mb *= ms2aps  # Mean speed in [A/ps].
            moment2_mb = 3 * constants.k * temp / mass_kg
            moment2_mb *= ms2aps**2  # Mean squared speed in [(A/ps)^2].
            std_mb = np.sqrt(moment2_mb - moment1_mb**2)
            header += (
                "  4 Maxwell-Boltzmann fit of the probability density\n"
                "    p(v) = 4*pi*(v-u)^2 * 1/(2*pi*sigma^2)^(3/2) *"
                " exp[-(v-u)^2 / (2*sigma^2)]\n"
                "    sigma^2 = kT/m\n"
                + "\n"
                + "{:>14d} {:>16d} {:>16d} {:>16d}\n".format(1, 2, 3, 4)
                + "{:<15s} {:<16s} {:<15s} {:>16.9e}\n".format(
                    "sigma^2",
                    "(fit parameter)",
                    "Unit: (A/ps)^2",
                    popt[0],
                )
                + "{:<14s} {:>50.9e}\n".format("Fit param StD:", perr[0])
                + "{:<15s} {:<16s} {:<15s} {:>16.9e}\n".format(
                    "Mass m", "(from topology)", "Unit: u", mass
                )
                + "{:<15s} {:<16s} {:<15s} {:>16.9e}\n".format(
                    "Temperature T", "= m*sigma^2/k", "Unit: K", temp
                )
                + "{:<15s} {:<16s} {:<15s} {:>16.9e}\n".format(
                    "Drift speed u",
                    "(fit parameter)",
                    "Unit: A/ps",
                    popt[1],
                )
                + "{:<14s} {:>50.9e}\n".format("Fit param StD:", perr[1])
                + "{:<15s} {:<16s} {:<15s} {:>16.9e}\n".format(
                    "v_p", "= sqrt(2kT/m)", "Unit: A/ps", v_p
                )
                + "{:<14s} {:>16.9e} {:>16.9e} {:>16.9e}\n".format(
                    "<v^2>:", moment2[i], moment2[i], moment2_mb
                )
                + "{:<14s} {:>16.9e} {:>16.9e} {:>16.9e}\n".format(
                    "Mean (<v>):", moment1[i], moment1[i], moment1_mb
                )
                + "{:<14s} {:>16.9e} {:>16.9e} {:>16.9e}\n".format(
                    "StD:", stds[i], stds[i], std_mb
                )
            )
        else:
            # Column numbers when no fit is created.
            header += "\n{:>14d} {:>16d} {:>16d}\n".format(1, 2, 3)
        # fmt: off
        header += (
            "{:<14s} {:>16.9e} {:>16.9e}\n".format("Min:", mins[i], mins[i])
            + "{:<14s} {:>16.9e} {:>16.9e}\n".format("Max:", maxs[i], maxs[i])
        )
        # fmt: on
        mdt.fh.savetxt(outfiles[i], data, header=header)
        print("Created {}".format(outfiles[i]))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Consistency check...")
    timer = datetime.now()
    for i, hist in enumerate(hists):
        if np.any(hist < 0):
            raise RuntimeError(
                "Component {}: Overflow encountered in"
                " histogram".format(DIMENSION[i])
            )
        if (
            args.BIN_START is None
            and args.BIN_STOP is None
            and np.sum(hist) != n_samples[i]
        ):
            raise ValueError(
                "Component {}: The total number of histogram counts ({}) does"
                " not match the number of samples ({}).  This should not have"
                " happened".format(DIMENSION[i], np.sum(hist), n_samples[i])
            )
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
