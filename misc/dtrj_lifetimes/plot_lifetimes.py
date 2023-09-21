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


"""
Plot the lifetimes generated by
:mod:`misc.dtrj_lifetimes.compare_dtrj_lifetime_methods` for a given set
of lifetime distributions.
"""


__author__ = "Andreas Thum"


# Standard libraries
import glob
import os
import sys

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# First-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401


if __name__ == "__main__":  # noqa: C901
    # Input parameters.
    dist_name = "Exp. Dist."
    sort_by = "n_cmps"

    dist = "_gengamma"  # _gengamma, _burr12
    beta = "_beta_1_1"
    delta = "_delta_1_1"
    tau0 = "_tau0_10_100"
    shape = "_shape_*_10"  # n_cmps, n_frames
    discard = "_discard_1000"
    seed = "_seed_5462_4894_3496_8436"
    infile_pattern = (
        "dtrj"
        + dist
        + beta
        + delta
        + tau0
        + shape
        + discard
        + seed
        + "_compare_dtrj_lifetime_methods.txt.gz"
    )
    outfile = infile_pattern.replace(".txt.gz", ".pdf")
    outfile = outfile.replace("*", "N")

    # Column indices for the input file(s).
    col_states = 0
    col_dist_params = 55  # to 57 (tau0, beta, delta)
    col_fit_end = 49
    # Lifetimes from the different methods.
    col_cnt_cen = 1  # to 8 (mean, std, skew, kurt, median, min, max, nobs)
    col_cnt_unc = 9  # to 16 (as col_cnt_cen)
    col_rate = 17
    # col_e = 18
    col_int = 19  # to 23 (mean, std, skew, kurt, median)
    col_kww = 24  # to 34 (mean, std, skew, kurt, median, fit params)
    col_bur = 35  # to 47 (as col_kww)
    col_dst = 50  # to 54 (as col_int)

    # Read data.
    infiles = glob.glob(infile_pattern)
    if len(infiles) < 1:
        raise ValueError(
            "The glob pattern does not match anything:"
            " {}".format(infile_pattern)
        )
    shapes = []
    data = []
    for infile in infiles:
        data_file = np.loadtxt(infile, ndmin=2)
        # Only keep the row that contains the data for state 1.
        valid = data_file[:, col_states] == 1
        if not np.any(valid):
            continue
        elif np.count_nonzero(valid) != 1:
            raise ValueError(
                "The input file contains multiple rows with state 1:"
                " '{}'".format(infile)
            )
        data.append(np.squeeze(data_file[valid]))
        shape = infile.split("_shape_")[1].split("_")[:2]
        shapes.append([int(s) for s in shape])
    if len(data) < 1:
        raise ValueError("None of the input files contained data for state 1")
    shapes = np.transpose(np.asarray(shapes))
    data = np.transpose(np.asarray(data))

    # Sort data.
    if sort_by == "n_cmps":
        sort_ix = np.argsort(shapes[0])
        xdata = shapes[0][sort_ix]
        xlim = (5e-1, 2e5)
        xlabel = "Number of Particles"
        legend_title = (
            dist_name + r", $\tau_0 = %d$" % data[col_dist_params][0]
        )
    elif sort_by == "n_frames":
        sort_ix = np.argsort(shapes[1])
        xdata = shapes[1][sort_ix]
        xlim = (5e0, 2e5)
        xlabel = "Number of Frames"
        legend_title = (
            dist_name + r", $\tau_0 = %d$" % data[col_dist_params][0]
        )
    elif sort_by == "tau0_true":
        sort_ix = np.argsort(data[col_dist_params])
        xdata = data[col_dist_params][sort_ix]
        xlim = (2e1, 5e2)
        xlabel = r"Scale Parameter $\tau_0$ / Frames"
        legend_title = dist_name
    elif sort_by == "beta_true":
        sort_ix = np.argsort(data[col_dist_params + 1])
        xdata = data[col_dist_params + 1][sort_ix]
        xlabel = r"Shape Parameter $\beta$"
        legend_title = (
            dist_name + r", $\tau_0 = %d$" % data[col_dist_params][0]
        )
        if dist_name == "Log-Logistic Dist.":
            xlim = (1e0, 2e1)
        else:
            xlim = (2e-1, 5e0)
    elif sort_by == "delta_true":
        sort_ix = np.argsort(data[col_dist_params + 2])
        xdata = data[col_dist_params + 2][sort_ix]
        xlabel = r"Shape Parameter $\delta$"
        if dist_name in ("Chi Dist.", "Lomax Dist."):
            xlim = (1e0, 2e1)
        else:
            xlim = (2e-1, 5e0)
        if dist_name == "Chi Dist.":
            legend_title = dist_name
        else:
            legend_title = (
                dist_name + r", $\tau_0 = %d$" % data[col_dist_params][0]
            )
    else:
        raise ValueError("Unknown value for `sort_by`: {}".format(sort_by))
    shapes = shapes[:, sort_ix]
    data = data[:, sort_ix]

    ylims_characs = [(None, None) for i in range(5)]
    ylims_cnt = [(None, None) for i in range(3)]
    ylims_fit_params = [(None, None) for i in range(3)]
    ylims_fit_goodness = [(None, None) for i in range(3)]
    ylims_fit_region = [(None, None)]
    if dist_name == "Exp. Dist." and sort_by == "n_cmps":
        ylims_characs = [
            (2e0, 3e2),  # Mean.
            (1e0, 2e2),  # Standard deviation.
            (2e-1, 3e0),  # Skewness.
            (6e-1, 2e1),  # Excess kurtosis.
            (2e0, 2e2),  # Median.
        ]
    if dist_name == "Exp. Dist." and sort_by == "n_frames":
        ylims_characs = [
            (8e0, 3e2),  # Mean.
            (9e1, 3e2),  # Standard deviation.
            (4e-1, 3e0),  # Skewness.
            (4e-1, 2e1),  # Excess kurtosis.
            (1e0, 1e3),  # Median.
        ]
    if dist_name == "Exp. Dist." and sort_by == "tau0_true":
        ylims_characs = [
            (2e1, 5e2),  # Mean.
            (2e1, 5e2),  # Standard deviation.
            (1e0, 3e0),  # Skewness.
            (5e0, 1e1),  # Excess kurtosis.
            (1e1, 4e2),  # Median.
        ]
    if dist_name == "Gamma Dist." and sort_by == "delta_true":
        ylims_characs = [
            (2e1, 5e2),  # Mean.
            (4e1, 3e2),  # Standard deviation.
            (9e-1, 5e0),  # Skewness.
            (1e0, 3e1),  # Excess kurtosis.
            (3e0, 5e2),  # Median.
        ]
        ylims_cnt = [(9e-1, 2e1), (9e2, 3e3), (1e4, 9e4)]
    if dist_name == "Chi Dist." and sort_by == "delta_true":
        ylims_cnt = [(8e-2, 1.2e1), (4e0, 1e1), (1e6, 4e6)]
    if dist_name == "Log-Logistic Dist." and sort_by == "beta_true":
        ylims_cnt = [(8e-1, 6e1), (1e2, 2e4), (1e4, 5e4)]

    label_true = "True"
    # label_cen = "True Cens."
    # label_unc = "True Uncen."
    label_cnt_cen = "Cens."
    label_cnt_unc = "Uncens."
    label_rate = "Rate"
    # label_e = r"$1/e$"
    label_int = "Area"
    label_kww = "Kohl."
    label_bur = "Burr"

    color_true = "tab:green"
    # color_cen = "tab:olive"
    # color_unc = "darkolivegreen"
    color_cnt_cen = "tab:orange"
    color_cnt_unc = "tab:red"
    color_rate = "tab:brown"
    # color_e = "tab:pink"
    color_int = "tab:purple"
    color_kww = "tab:blue"
    color_bur = "tab:cyan"

    marker_true = "s"
    # marker_cen = "D"
    # marker_unc = "d"
    marker_cnt_cen = "H"
    marker_cnt_unc = "h"
    marker_rate = "p"
    # marker_e = "<"
    marker_int = ">"
    marker_kww = "^"
    marker_bur = "v"

    alpha = 0.75

    mdt.fh.backup(outfile)
    with PdfPages(outfile) as pdf:
        # Plot distribution characteristics.
        ylabels = (
            "Average Lifetime / Frames",
            "Std. Dev. / Frames",
            "Skewness",
            "Excess Kurtosis",
            "Median / Frames",
        )
        for i, ylabel in enumerate(ylabels):
            fig, ax = plt.subplots(clear=True)
            # True lifetimes (from distribution).
            valid = data[col_dst + i] > 0
            if np.any(valid):
                ax.plot(
                    xdata[valid],
                    data[col_dst + i][valid],
                    label=label_true,
                    color=color_true,
                    marker=marker_true,
                    alpha=alpha,
                )
            # Method 1 (censored counting).
            valid = data[col_cnt_cen + i] > 0
            if np.any(valid):
                ax.plot(
                    xdata[valid],
                    data[col_cnt_cen + i][valid],
                    label=label_cnt_cen,
                    color=color_cnt_cen,
                    marker=marker_cnt_cen,
                    alpha=alpha,
                )
            # Method 2 (uncensored counting).
            valid = data[col_cnt_unc + i] > 0
            if np.any(valid):
                ax.plot(
                    xdata[valid],
                    data[col_cnt_unc + i][valid],
                    label=label_cnt_unc,
                    color=color_cnt_unc,
                    marker=marker_cnt_unc,
                    alpha=alpha,
                )
            # Method 3 (inverse transition rate).
            if i == 0:
                valid = data[col_rate] > 0
                if np.any(valid):
                    ax.plot(
                        xdata[valid],
                        data[col_rate][valid],
                        label=label_rate,
                        color=color_rate,
                        marker=marker_rate,
                        alpha=alpha,
                    )
                # # Method 4 (1/e criterion).
                # valid = data[col_e] > 0
                # if np.any(valid):
                #     ax.plot(
                #         xdata[valid],
                #         data[col_e][valid],
                #         label=label_e,
                #         color=color_e,
                #         marker=marker_e,
                #         alpha=alpha,
                #     )
            # Method 5 (direct integral).
            valid = data[col_int + i] > 0
            if np.any(valid):
                ax.plot(
                    xdata[valid],
                    data[col_int + i][valid],
                    label=label_int,
                    color=color_int,
                    marker=marker_int,
                    alpha=alpha,
                )
            # Method 6 (integral of Kohlrausch fit).
            valid = data[col_kww + i] > 0
            if np.any(valid):
                ax.plot(
                    xdata[valid],
                    data[col_kww + i][valid],
                    label=label_kww,
                    color=color_kww,
                    marker=marker_kww,
                    alpha=alpha,
                )
            # Method 7 (integral of Burr fit).
            valid = data[col_bur + i] > 0
            if np.any(valid):
                ax.plot(
                    xdata[valid],
                    data[col_bur + i][valid],
                    label=label_bur,
                    color=color_bur,
                    marker=marker_bur,
                    alpha=alpha,
                )
            ax.set_xscale("log", base=10, subs=np.arange(2, 10))
            ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax.set(
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylims_characs[i],
            )
            legend = ax.legend(
                title=legend_title, ncol=3, **mdtplt.LEGEND_KWARGS_XSMALL
            )
            legend.get_title().set_multialignment("center")
            pdf.savefig()
            plt.close()

        # Plot number of min, max and number of samples for count
        # methods.
        ylabels = (
            "Min. Lifetime / Frames",
            "Max. Lifetime / Frames",
            "No. of Samples",
        )
        for i, ylabel in enumerate(ylabels):
            fig, ax = plt.subplots(clear=True)
            # Method 1 (censored counting).
            valid = data[col_cnt_cen + len(ylims_characs) + i] > 0
            if np.any(valid):
                ax.plot(
                    xdata[valid],
                    data[col_cnt_cen + len(ylims_characs) + i][valid],
                    label=label_cnt_cen,
                    color=color_cnt_cen,
                    marker=marker_cnt_cen,
                    alpha=alpha,
                )
            # Method 2 (uncensored counting).
            valid = data[col_cnt_unc + len(ylims_characs) + i] > 0
            if np.any(valid):
                ax.plot(
                    xdata[valid],
                    data[col_cnt_unc + len(ylims_characs) + i][valid],
                    label=label_cnt_unc,
                    color=color_cnt_unc,
                    marker=marker_cnt_unc,
                    alpha=alpha,
                )
            ax.set_xscale("log", base=10, subs=np.arange(2, 10))
            ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax.set(
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylims_cnt[i],
            )
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(
                title=legend_title,
                ncol=max(len(handles), 1),
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend.get_title().set_multialignment("center")
            pdf.savefig()
            plt.close()

        # Plot fit parameters tau0 and beta.
        ylabels = (
            r"Fit Parameter $\tau_0$ / Frames",
            r"Fit Parameter $\beta$",
        )
        for i, ylabel in enumerate(ylabels):
            fig, ax = plt.subplots(clear=True)
            # True distribution.
            valid = data[col_dist_params + i] > 0
            if np.any(valid):
                ax.plot(
                    xdata[valid],
                    data[col_dist_params + i][valid],
                    label=label_true,
                    color=color_true,
                    marker=marker_true,
                    alpha=alpha,
                )
            # Method 6 (Kohlrausch fit).
            col_kww_i = len(ylims_characs) + 2 * i
            valid = data[col_kww + col_kww_i] > 0
            if np.any(valid):
                ax.errorbar(
                    xdata[valid],
                    data[col_kww + col_kww_i][valid],
                    yerr=data[col_kww + col_kww_i + 1][valid],
                    label=label_kww,
                    color=color_kww,
                    marker=marker_kww,
                    alpha=alpha,
                )
            # Method 7 (Burr fit).
            col_bur_i = len(ylims_characs) + 2 * i
            valid = data[col_bur + col_bur_i] > 0
            if np.any(valid):
                ax.errorbar(
                    xdata[valid],
                    data[col_bur + col_bur_i][valid],
                    yerr=data[col_bur + col_bur_i + 1][valid],
                    label=label_bur,
                    color=color_bur,
                    marker=marker_bur,
                    alpha=alpha,
                )
            ax.set_xscale("log", base=10, subs=np.arange(2, 10))
            ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax.set(
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylims_fit_params[i],
            )
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(
                title=legend_title,
                ncol=max(len(handles), 1),
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend.get_title().set_multialignment("center")
            pdf.savefig()
            plt.close()

        # Plot fit parameter delta.
        fig, ax = plt.subplots(clear=True)
        # True delta (from distribution).
        col_true_i = 2
        valid = data[col_dist_params + col_true_i] > 0
        if np.any(valid):
            ax.plot(
                xdata[valid],
                data[col_dist_params + col_true_i][valid],
                label=label_true,
                color=color_true,
                marker=marker_true,
                alpha=alpha,
            )
        # Method 7 (Burr fit).
        col_bur_i = 9
        valid = data[col_bur + col_bur_i] > 0
        if np.any(valid):
            ax.errorbar(
                xdata[valid],
                data[col_bur + col_bur_i][valid],
                yerr=data[col_bur + col_bur_i + 1][valid],
                label=label_bur,
                color=color_bur,
                marker=marker_bur,
                alpha=alpha,
            )
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set(
            xlabel=xlabel,
            ylabel=r"Fit Parameter $\delta$",
            xlim=xlim,
            ylim=ylims_fit_params[2],
        )
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            title=legend_title,
            ncol=max(len(handles), 1),
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

        # Plot goodness of fit quantities.
        ylabels = (r"Coeff. of Determ. $R^2$", "RMSE")
        for i, ylabel in enumerate(ylabels):
            fig, ax = plt.subplots(clear=True)
            # Remain probability to the true survival function.
            col_true_i = 8 + i
            valid = data[col_dst + col_true_i] > 0
            if np.any(valid):
                ax.plot(
                    xdata[valid],
                    data[col_dst + col_true_i][valid],
                    label=r"$C(t)$",
                    color=color_true,
                    marker=marker_true,
                    alpha=alpha,
                )
            # Method 6 (Kohlrausch fit).
            col_kww_i = 9 + i
            valid = data[col_kww + col_kww_i] > 0
            if np.any(valid):
                ax.plot(
                    xdata[valid],
                    data[col_kww + col_kww_i][valid],
                    label=label_kww,
                    color=color_kww,
                    marker=marker_kww,
                    alpha=alpha,
                )
            # Method 7 (Burr fit).
            col_bur_i = 11 + i
            valid = data[col_bur + col_bur_i] > 0
            if np.any(valid):
                ax.plot(
                    xdata[valid],
                    data[col_bur + col_bur_i][valid],
                    label=label_bur,
                    color=color_bur,
                    marker=marker_bur,
                    alpha=alpha,
                )
            ax.set_xscale("log", base=10, subs=np.arange(2, 10))
            if i > 0:
                ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax.set(
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylims_fit_goodness[i],
            )
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(
                title=legend_title,
                ncol=max(len(handles), 1),
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend.get_title().set_multialignment("center")
            pdf.savefig()
            plt.close()

        # Plot end of fit region.
        fig, ax = plt.subplots(clear=True)
        valid = data[col_fit_end] > 0
        if np.any(valid):
            ax.plot(xdata[valid], data[col_fit_end][valid], marker="v")
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set(
            xlabel=xlabel,
            ylabel="End of Fit Region / Frames",
            xlim=xlim,
            ylim=ylims_fit_region[0],
        )
        legend = ax.legend(title=legend_title, **mdtplt.LEGEND_KWARGS_XSMALL)
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

    print("Created {}".format(outfile))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))