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
Compare state lifetimes calculated from a discrete trajectory using
different methods.

Options
-------
--dtrj
    Name of the file containing the discrete trajectory.  The discrete
    trajectory must be stored as :class:`numpy.ndarray` either in a
    binary NumPy |npy_file| or in a (compressed) NumPy |npz_archive|.
    See :func:`mdtools.file_handler.load_dtrj` for more information
    about the requirements for the input file.
--rp
    Name of the file containing the remain probabilities for each state
    as generated by
    :mod:`scripts.discretization.state_lifetime_discrete`.
--param
    File containing the parameters that were used to generate the
    artificial discrete trajectory as created by
    :mod:`misc.dtrj_lifetimes.generate_dtrj.py` (optional).  If
    provided, the true lifetimes are also plotted.
-o
    Output filename pattern.
--int-thresh
    Only calculate the lifetime by directly integrating the remain
    probability if the remain probability decays below the given
    threshold.  Default: ``0.01``.
--end-fit
    End time for fitting the remain probability (in trajectory steps).
    This is inclusive, i.e. the time given here is still included in the
    fit.  If ``None``, the fit ends at 90% of the lag times.  Default:
    ``None``.
--stop-fit
    Stop fitting the remain probability as soon as it falls below the
    given value.  The fitting is stopped by whatever happens earlier:
    \--end-fit or \--stop-fit.  Default: ``0.01``.


See Also
--------
:mod:`misc.dtrj_lifetimes.generate_dtrj` :
    Generate an artificial discrete trajectory with a given number of
    states with a given lifetime distribution
"""


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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from scipy.special import gamma

# First-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401


# from scipy.stats import gengamma


if __name__ == "__main__":  # noqa: C901
    timer_tot = datetime.now()
    proc = psutil.Process()
    proc.cpu_percent()  # Initiate monitoring of CPU usage.
    parser = argparse.ArgumentParser(
        # The description should only contain the short summary from the
        # docstring and a reference to the documentation.
        description=(
            "Compare state lifetimes calculated from a discrete trajectory"
            " using different methods"
        )
    )
    parser.add_argument(
        "--dtrj",
        dest="INFILE_DTRJ",
        type=str,
        required=True,
        help=(
            "File containing the discrete trajectory stored as numpy.ndarray"
            " in the binary .npy format or as .npz archive."
        ),
    )
    parser.add_argument(
        "--rp",
        dest="INFILE_RP",
        type=str,
        required=True,
        help=(
            "Name of the file containing the remain probabilities for each"
            " state."
        ),
    )
    parser.add_argument(
        "--param",
        dest="INFILE_PARAM",
        type=str,
        required=False,
        default=None,
        help=(
            "File containing the parameters that were used to generate the"
            " artificial discrete trajectory (optional)."
        ),
    )
    parser.add_argument(
        "-o",
        dest="OUTFILE",
        type=str,
        required=True,
        help="Output filename pattern.",
    )
    parser.add_argument(
        "--int-thresh",
        dest="INT_THRESH",
        type=float,
        required=False,
        default=0.01,
        help=(
            "Only calculate the lifetime by directly integrating the remain"
            " probability if the remain probability decays below the given"
            " threshold.  Default:  %(default)s."
        ),
    )
    parser.add_argument(
        "--end-fit",
        dest="ENDFIT",
        type=float,
        required=False,
        default=None,
        help=(
            "End time for fitting the remain probability in trajectory"
            " steps (inclusive).  If None, the fit ends at 90%% of the"
            " lag times.  Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--stop-fit",
        dest="STOPFIT",
        type=float,
        required=False,
        default=0.01,
        help=(
            "Stop fitting the remain probability as soon as it falls below the"
            " given value.  The fitting is stopped by whatever happens"
            " earlier: --end-fit or --stop-fit.  Default: %(default)s"
        ),
    )
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

    # Conversion factor to convert "trajectory steps" to some physical
    # time unit (e.g. ns).
    time_conv = 1

    if args.INFILE_PARAM is not None:
        states_true, delta, beta, tau0 = np.loadtxt(
            args.INFILE_PARAM, unpack=True
        )
        # Moments of the true lifetime distribution assuming a
        # generalized gamma distribution:
        #   <t^n> = tau0^n Gamma[(delta+n)/beta] / Gamma(delta/beta).
        lifetimes_true_mom1 = (
            tau0**1 * gamma((delta + 1) / beta) / gamma(delta / beta)
        )
        lifetimes_true_mom2 = (
            tau0**2 * gamma((delta + 2) / beta) / gamma(delta / beta)
        )
        lifetimes_true_mom3 = (
            tau0**3 * gamma((delta + 3) / beta) / gamma(delta / beta)
        )
        lifetimes_true_mom1 *= time_conv**1
        lifetimes_true_mom2 *= time_conv**2
        lifetimes_true_mom3 *= time_conv**3

    print("\n")
    print("Calculating lifetimes directly from `dtrj` (Methods 1-2)...")
    timer = datetime.now()
    dtrj = mdt.fh.load_dtrj(args.INFILE_DTRJ)
    n_frames = dtrj.shape[1]

    # Method 1: Calculate the average lifetime by counting the number of
    # frames that a given compound stays in a given state.
    lifetimes_cnt, states_cnt = mdt.dtrj.lifetimes_per_state(
        dtrj, return_states=True
    )
    lifetimes_cnt = [lts * time_conv for lts in lifetimes_cnt]
    lifetimes_cnt_mom1 = np.array([np.mean(lts) for lts in lifetimes_cnt])
    lifetimes_cnt_mom2 = np.array([np.mean(lts**2) for lts in lifetimes_cnt])
    lifetimes_cnt_mom3 = np.array([np.mean(lts**3) for lts in lifetimes_cnt])
    del lifetimes_cnt

    if args.INFILE_PARAM is not None and not np.all(
        np.isin(states_cnt, states_true)
    ):
        raise ValueError(
            "`states_cnt` ({}) not in `states_true`"
            " ({})".format(states_cnt, states_true)
        )

    # Method 2: Calculate the transition rate as the number of
    # transitions leading out of a given state divided by the number of
    # frames that compounds have spent in this state.  The average
    # lifetime is calculated as the inverse transition rate.
    rates, states_k = mdt.dtrj.trans_rate_per_state(dtrj, return_states=True)
    lifetimes_k = time_conv / rates
    if not np.array_equal(states_k, states_cnt):
        raise ValueError(
            "`states_k` ({}) != `states_cnt` ({})".format(states_k, states_cnt)
        )
    del dtrj, rates, states_k
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Calculating lifetimes from the remain probability (Methods 3-5)...")
    timer = datetime.now()

    remain_props = np.loadtxt(args.INFILE_RP)
    states = remain_props[0, 1:]  # State indices.
    times = remain_props[1:, 0]  # Lag times in trajectory steps.
    remain_props = remain_props[1:, 1:]  # Remain probability functions.
    if np.any(remain_props < 0) or np.any(remain_props > 1):
        raise ValueError(
            "Some values of the remain probability lie outside the interval"
            " [0, 1]"
        )
    if not np.array_equal(times, np.arange(n_frames)):
        raise ValueError("`times` != `np.arange(n_frames)`")
    times *= time_conv  # Lag times in the given physical time unit.
    if np.any(np.modf(states)[0] != 0):
        raise ValueError(
            "Some state indices are not integers but floats.  states ="
            " {}".format(states)
        )
    if not np.array_equal(states, states_cnt):
        raise ValueError(
            "`states` ({}) != `states_cnt` ({})".format(states, states_cnt)
        )
    del states_cnt
    states = states.astype(np.int32)
    n_states = len(states)

    # Method 3: Set the lifetime to the lag time at which the remain
    # probability crosses 1/e.
    thresh = 1 / np.e
    ix_thresh = np.nanargmax(remain_props <= thresh, axis=0)
    lifetimes_e = np.full(n_states, np.nan, dtype=np.float64)
    for i, rp in enumerate(remain_props.T):
        if rp[ix_thresh[i]] > thresh:
            # The remain probability never falls below the threshold.
            lifetimes_e[i] = np.nan
        elif ix_thresh[i] < 1:
            # The remain probability immediately falls below the
            # threshold.
            lifetimes_e[i] = 0
        elif rp[ix_thresh[i] - 1] < thresh:
            raise ValueError(
                "The threshold ({}) does not lie within the remain probability"
                " interval ([{}, {}]) at the found indices ({}, {}).  This"
                " should not have happened.".format(
                    thresh,
                    rp[ix_thresh[i]],
                    rp[ix_thresh[i] - 1],
                    ix_thresh[i],
                    ix_thresh[i] - 1,
                )
            )
        else:
            slope = rp[ix_thresh[i]] - rp[ix_thresh[i] - 1]
            slope /= times[ix_thresh[i]] - times[ix_thresh[i] - 1]
            intercept = rp[ix_thresh[i] - 1] - slope * times[ix_thresh[i] - 1]
            lifetimes_e[i] = (thresh - intercept) / slope
            if (
                times[ix_thresh[i] - 1] > lifetimes_e[i]
                or times[ix_thresh[i]] < lifetimes_e[i]
            ):
                raise ValueError(
                    "The lifetime ({}) does not lie within the time interval"
                    " ([{}, {}]) at the found indices ({}, {}).  This should"
                    " not have happened.".format(
                        lifetimes_e[i],
                        times[ix_thresh[i] - 1],
                        times[ix_thresh[i]],
                        ix_thresh[i] - 1,
                        ix_thresh[i],
                    )
                )

    # Method 4: Calculate the lifetime as the integral of the remain
    # probability.
    lifetimes_int_mom1 = np.full(n_states, np.nan, dtype=np.float64)
    lifetimes_int_mom2 = np.full(n_states, np.nan, dtype=np.float64)
    lifetimes_int_mom3 = np.full(n_states, np.nan, dtype=np.float64)
    for i, rp in enumerate(remain_props.T):
        valid = ~np.isnan(rp)
        lifetimes_int_mom1[i] = np.trapz(y=rp[valid], x=times[valid])
        lifetimes_int_mom2[i] = np.trapz(
            y=rp[valid] * times[valid], x=times[valid]
        )
        lifetimes_int_mom3[i] = np.trapz(
            y=rp[valid] * (times[valid] ** 2), x=times[valid]
        )
        lifetimes_int_mom3[i] /= 2
    invalid = np.all(remain_props > args.INT_THRESH, axis=0)
    lifetimes_int_mom1[invalid] = np.nan
    lifetimes_int_mom2[invalid] = np.nan
    lifetimes_int_mom3[invalid] = np.nan
    del valid, invalid

    # Method 5: Fit the remain probability with a stretched exponential
    # and calculate the lifetime as the integral of this stretched
    # exponential.
    if args.ENDFIT is None:
        end_fit = int(0.9 * len(times))
    else:
        _, end_fit = mdt.nph.find_nearest(
            times, args.ENDFIT, return_index=True
        )
    end_fit += 1  # Make `end_fit` inclusive.
    fit_start = np.zeros(n_states, dtype=np.uint32)  # Inclusive.
    fit_stop = np.zeros(n_states, dtype=np.uint32)  # Exclusive.

    # Initial guesses for `tau0` and `beta`.
    init_guess = np.column_stack([lifetimes_e, np.ones(n_states)])
    init_guess[np.isnan(init_guess)] = 1.5 * times[-1]

    popt = np.full((n_states, 2), np.nan, dtype=np.float64)
    perr = np.full((n_states, 2), np.nan, dtype=np.float64)
    fit_r2 = np.full(n_states, np.nan, dtype=np.float64)
    fit_mse = np.full(n_states, np.nan, dtype=np.float64)
    for i, rp in enumerate(remain_props.T):
        stop_fit = np.nanargmax(rp < args.STOPFIT)
        if stop_fit == 0 and rp[stop_fit] >= args.STOPFIT:
            stop_fit = len(rp)
        elif stop_fit < 2:
            stop_fit = 2
        fit_stop[i] = min(end_fit, stop_fit)
        times_fit = times[fit_start[i] : fit_stop[i]]
        rp_fit = rp[fit_start[i] : fit_stop[i]]
        popt[i], perr[i] = mdt.func.fit_kww(
            xdata=times_fit, ydata=rp_fit, p0=init_guess[i], method="trf"
        )
        if np.any(np.isnan(popt[i])):
            fit_mse[i] = np.nan
            fit_r2[i] = np.nan
        else:
            fit = mdt.func.kww(times_fit, *popt[i])
            # Residual sum of squares.
            ss_res = np.nansum((rp_fit - fit) ** 2)
            # Mean squared error / mean squared residuals.
            fit_mse[i] = ss_res / len(fit)
            # Total sum of squares
            ss_tot = np.nansum((rp_fit - np.nanmean(rp_fit)) ** 2)
            # (Pseudo) coefficient of determination (R^2).
            # https://www.r-bloggers.com/2021/03/the-r-squared-and-nonlinear-regression-a-difficult-marriage/
            fit_r2[i] = 1 - (ss_res / ss_tot)
    tau0, beta = popt.T
    tau0_sd, beta_sd = perr.T
    lifetimes_exp_mom1 = tau0 / beta * gamma(1 / beta)
    lifetimes_exp_mom2 = tau0**2 / beta * gamma(2 / beta)
    lifetimes_exp_mom3 = tau0**3 / beta * gamma(3 / beta) / 2
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating text output...")
    timer = datetime.now()
    header = (
        "State residence times.\n"
        + "Average time that a given compound stays in a given state\n"
        + "calculated either directly from the discrete trajectory (Method\n"
        + "1-2) or from the corresponding remain probability function\n"
        + "(Method 3-4). \n"
        + "\n"
        + "\n"
        + "Discrete trajectory: {:s}\n".format(args.INFILE_DTRJ)
        + "Remain probability:  {:s}\n".format(args.INFILE_RP)
        + "\n"
        + "\n"
        + "Residence times are calculated using five different methods:\n"
        + "\n"
        + "1) The residence time <tau_cnt> is calculated by counting how\n"
        + "   many frames a given compound stays in a given state.  Note\n"
        + "   that residence times calculated in this way can at maximum be\n"
        + "   as long as the trajectory and are usually biased to lower\n"
        + "   values because of edge effects.\n"
        + "\n"
        + "2) The average transition rate <k> is calculated as the number of\n"
        + "   transitions leading out of a given state divided by the number\n"
        + "   frames that compounds have spent in this state.  The average\n"
        + "   lifetime <tau_k> is calculated as the inverse transition rate:\n"
        + "     <tau_k> = 1 / <k>\n"
        + "\n"
        + "3) The residence time <tau_e> is set to the lag time at which the\n"
        + "   remain probability function p(t) crosses 1/e.  If this never\n"
        + "   happens, <tau_e> is set to NaN.\n"
        + "\n"
        + "4) According to Equations (12) and (14) of Reference [1], the\n"
        + "   n-th moment of the residence time <tau_int^n> is calculated\n"
        + "   as the integral of the remain probability function p(t) times\n"
        + "   t^{n-1}:\n"
        + "     <tau_int^n> = 1/(n-1)! int_0^inf t^{n-1} p(t) dt\n"
        + "   If p(t) does not decay below the given threshold of\n"
        + "   {:.4f}, <tau_int^n> is set to NaN.\n".format(args.INT_THRESH)
        + "\n"
        + "5) The remain probability function p(t) is fitted by a stretched\n"
        + "   exponential function using the 'Trust Region Reflective'\n"
        + "   method of scipy.optimize.curve_fit:\n"
        + "     f(t) = exp[-(t/tau0)^beta]\n"
        + "   Thereby, tau0 is confined to positive values and beta is\n"
        + "   confined to the interval [0, 1].  The remain probability is\n"
        + "   fitted until it decays below a given threshold or until a\n"
        + "   given lag time is reached (whatever happens earlier).  The\n"
        + "   n-th moment of the residence time <tau_exp^n> is calculated\n"
        + "   according to Equations (12) and (14) of Reference [1] and\n"
        + "   Equation (16) of Reference [2] as the integral of f(t) times\n"
        + "    t^{n-1}:\n"
        + "     <tau_exp^n> = 1/(n-1)! int_0^infty t^{n-1} f(t) dt\n"
        + "                 = tau0^n/beta * Gamma(1/beta)/Gamma(n)\n"
        + "   where Gamma(x) is the gamma function.\n"
        + "\n"
        + "Note that the moments calculated by method 4 and 5 are the\n"
        + "moments of an assumed underlying, continuous distribution of time\n"
        + "constants tau.  They are related to the moments of the underlying\n"
        + "distribution of decay times t by\n"
        + "  <t^n> = n! * <tau^{n+1}> / <tau>\n"
        + "Compare Equation (14) of Reference [1].\n"
        + "\n"
        + "Reference [1]:\n"
        + "  M. N. Berberan-Santos, E. N. Bodunov, B. Valeur,\n"
        + "  Mathematical functions for the analysis of luminescence decays\n"
        + "  with underlying distributions 1. Kohlrausch decay function\n"
        + "  (stretched exponential),\n"
        + "  Chemical Physics, 2005, 315, 171-182.\n"
        + "Reference [2]:\n"
        + "  D. C. Johnston,\n"
        + "  Stretched exponential relaxation arising from a continuous sum\n"
        + "  of exponential decays,\n"
        + "  Physical Review B, 2006, 74, 184430.\n"
        + "\n"
        + "int_thresh = {:.4f}\n".format(args.INT_THRESH)
        + "\n"
        + "\n"
        + "The columns contain:\n"
        + "  1 State indices (zero based)\n"
        + "\n"
        + "  Residence times from Method 1 (counting)\n"
        + "  2 1st moment <tau_cnt> / frames\n"
        + "  3 2nd moment <tau_cnt^2> / frames^2\n"
        + "  4 3rd moment <tau_cnt^3> / frames^3\n"
        + "\n"
        + "  Residence times from Method 2 (inverse transition rate)\n"
        + "  5 <tau_k> / frames\n"
        + "\n"
        + "  Residence times from Method 3 (1/e criterion)\n"
        + "  6 <tau_e> / frames\n"
        + "\n"
        + "  Residence times from Method 4 (direct integral)\n"
        + "  7 1st moment <tau_int> / frames\n"
        + "  8 2nd moment <tau_int^2> / frames^2\n"
        + "  9 3rd moment <tau_int^3> / frames^3\n"
        + "\n"
        + "  Residence times from Method 5 (integral of the fit)\n"
        + " 10 1st moment <tau_exp> / frames\n"
        + " 11 2nd moment <tau_exp^2> / frames^2\n"
        + " 12 3rd moment <tau_exp^3> / frames^3\n"
        + " 13 Fit parameter tau0 / frames\n"
        + " 14 Standard deviation of tau0 / frames\n"
        + " 15 Fit parameter beta\n"
        + " 16 Standard deviation of beta\n"
        + " 17 Coefficient of determination of the fit (R^2 value)\n"
        + " 18 Mean squared error of the fit (mean squared residuals) /"
        + " frames^2\n"
        + " 19 Start of fit region (inclusive) / frames\n"
        + " 20 End of fit region (exclusive) / frames\n"
    )
    if args.INFILE_PARAM is not None:
        n_cols = 23
        header += (
            "\n"
            "  True residence times\n"
            " 21 1st moment <tau_true> / frames\n"
            " 22 2nd moment <tau_true^2> = 2 * <tau_true>^2 / frames^2\n"
            " 23 3rd moment <tau_true^3> = 6 * <tau_true>^3 / frames^3\n"
        )
    else:
        n_cols = 20
    header += "\n" + "Column number:\n"
    header += "{:>14d}".format(1)
    for i in range(2, n_cols + 1):
        header += " {:>16d}".format(i)
    data = [
        states,  # 1
        #
        lifetimes_cnt_mom1,  # 2
        lifetimes_cnt_mom2,  # 3
        lifetimes_cnt_mom3,  # 4
        #
        lifetimes_k,  # 5
        #
        lifetimes_e,  # 6
        #
        lifetimes_int_mom1,  # 7
        lifetimes_int_mom2,  # 8
        lifetimes_int_mom3,  # 9
        #
        lifetimes_exp_mom1,  # 10
        lifetimes_exp_mom2,  # 11
        lifetimes_exp_mom3,  # 12
        tau0,  # 13
        tau0_sd,  # 14
        beta,  # 15
        beta_sd,  # 16
        fit_r2,  # 17
        fit_mse,  # 18
        fit_start,  # 19
        fit_stop,  # 20
    ]
    if args.INFILE_PARAM is not None:
        lifetimes_true_mom1 = lifetimes_true_mom1[states]
        lifetimes_true_mom2 = lifetimes_true_mom2[states]
        lifetimes_true_mom3 = lifetimes_true_mom3[states]
        data += [
            lifetimes_true_mom1,  # 21
            lifetimes_true_mom2,  # 22
            lifetimes_true_mom3,  # 23
        ]
    data = np.column_stack(data)
    outfile = args.OUTFILE + ".txt"
    mdt.fh.savetxt(outfile, data, header=header)
    print("Created {}".format(outfile))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("Creating plot(s)...")
    timer = datetime.now()
    xlabel = r"State Index"
    xlim = (np.min(states) - 0.5, np.max(states) + 0.5)
    alpha = 0.75
    lifetime_min = np.nanmin(
        [
            lifetimes_cnt_mom1,
            lifetimes_k,
            lifetimes_e,
            lifetimes_int_mom1,
            lifetimes_exp_mom1,
        ]
    )
    if args.INFILE_PARAM is not None:
        lifetime_min = np.nanmin(
            [lifetime_min, np.nanmin(lifetimes_true_mom1)]
        )
    cmap = plt.get_cmap()
    c_vals = np.arange(n_states)
    c_norm = n_states - 1
    c_vals_normed = c_vals / c_norm
    colors = cmap(c_vals_normed)
    outfile = args.OUTFILE + ".pdf"
    mdt.fh.backup(outfile)
    with PdfPages(outfile) as pdf:
        # Plot residence times vs. state indices ("ensemble average").
        fig, ax = plt.subplots(clear=True)
        # Method 5 (integral of the fit).
        ax.errorbar(
            states,
            lifetimes_exp_mom1,
            yerr=np.sqrt(lifetimes_exp_mom2 - lifetimes_exp_mom1**2),
            label="Fit",
            color="tab:cyan",
            marker="D",
            alpha=alpha,
        )
        # Method 4 (direct integral)
        ax.errorbar(
            states,
            lifetimes_int_mom1,
            yerr=np.sqrt(lifetimes_int_mom2 - lifetimes_int_mom1**2),
            label="Area",
            color="tab:blue",
            marker="d",
            alpha=alpha,
        )
        # Method 3 (1/e criterion).
        ax.plot(
            states,
            lifetimes_e,
            label=r"$1/e$",
            color="tab:purple",
            marker="s",
            alpha=alpha,
        )
        # Method 2 (inverse transition rate).
        ax.plot(
            states,
            lifetimes_k,
            label="Rate",
            color="tab:red",
            marker="h",
            alpha=alpha,
        )
        # Method 1 (counting).
        ax.errorbar(
            states,
            lifetimes_cnt_mom1,
            yerr=np.sqrt(lifetimes_cnt_mom2 - lifetimes_cnt_mom1**2),
            label="Count",
            color="tab:orange",
            marker="H",
            alpha=alpha,
        )
        if args.INFILE_PARAM is not None:
            # True lifetimes.
            ax.errorbar(
                states,
                lifetimes_true_mom1,
                yerr=np.sqrt(lifetimes_true_mom2 - lifetimes_true_mom1**2),
                label="True",
                color="tab:green",
                marker="o",
                alpha=alpha,
            )
        ax.set(
            xlabel=xlabel,
            ylabel=r"Residence Time $\langle \tau \rangle$ / Frames",
            xlim=xlim,
        )
        ylim = ax.get_ylim()
        if ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Sort legend entries.  By default, lines plotted with `ax.plot`
        # come before lines plotted with `ax.errorbar`.
        handles, labels = ax.get_legend_handles_labels()
        legend_order = (2, 3, 0, 1, 4)
        if args.INFILE_PARAM is not None:
            legend_order += (5,)
        handles = [handles[leg_ord] for leg_ord in legend_order]
        labels = [labels[leg_ord] for leg_ord in legend_order]
        legend = ax.legend(
            handles, labels, ncol=2, **mdtplt.LEGEND_KWARGS_XSMALL
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        # Set y axis to log scale
        # (residence times vs. state indices, "ensemble average").
        # Round y limits to next lower and higher power of ten.
        ylim = ax.get_ylim()
        ymin = 10 ** np.floor(np.log10(lifetime_min))
        ymax = 10 ** np.ceil(np.log10(ylim[1]))
        ax.set_ylim(ymin if not np.isnan(ymin) else None, ymax)
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        pdf.savefig()
        plt.close()

        # Plot residence times vs. state indices ("time average").
        fig, ax = plt.subplots(clear=True)
        # Method 5 (integral of the fit).
        ax.errorbar(
            states,
            lifetimes_exp_mom2 / lifetimes_exp_mom1,
            yerr=np.sqrt(2 * lifetimes_exp_mom3 / lifetimes_exp_mom1),
            label="Fit",
            color="tab:cyan",
            marker="D",
            alpha=alpha,
        )
        # Method 4 (direct integral)
        ax.errorbar(
            states,
            lifetimes_int_mom2 / lifetimes_int_mom1,
            yerr=np.sqrt(2 * lifetimes_int_mom3 / lifetimes_int_mom1),
            label="Area",
            color="tab:blue",
            marker="d",
            alpha=alpha,
        )
        # Method 3 (1/e criterion).
        ax.plot(
            states,
            lifetimes_e,
            label=r"$1/e$",
            color="tab:purple",
            marker="s",
            alpha=alpha,
        )
        # Method 2 (inverse transition rate).
        ax.plot(
            states,
            lifetimes_k,
            label="Rate",
            color="tab:red",
            marker="h",
            alpha=alpha,
        )
        # Method 1 (counting).
        ax.errorbar(
            states,
            lifetimes_cnt_mom1,
            yerr=np.sqrt(lifetimes_cnt_mom2 - lifetimes_cnt_mom1**2),
            label="Count",
            color="tab:orange",
            marker="H",
            alpha=alpha,
        )
        if args.INFILE_PARAM is not None:
            # True lifetimes.
            ax.errorbar(
                states,
                lifetimes_true_mom1,
                yerr=np.sqrt(lifetimes_true_mom2 - lifetimes_true_mom1**2),
                label="True",
                color="tab:green",
                marker="o",
                alpha=alpha,
            )
        ax.set(
            xlabel=xlabel,
            ylabel=r"Residence Time $\bar{\tau}$ / Frames",
            xlim=xlim,
        )
        ylim = ax.get_ylim()
        if ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Sort legend entries.  By default, lines plotted with `ax.plot`
        # come before lines plotted with `ax.errorbar`.
        handles, labels = ax.get_legend_handles_labels()
        legend_order = (2, 3, 0, 1, 4)
        if args.INFILE_PARAM is not None:
            legend_order += (5,)
        handles = [handles[leg_ord] for leg_ord in legend_order]
        labels = [labels[leg_ord] for leg_ord in legend_order]
        legend = ax.legend(
            handles, labels, ncol=2, **mdtplt.LEGEND_KWARGS_XSMALL
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        # Set y axis to log scale
        # (residence times vs. state indices, "ensemble average").
        # Round y limits to next lower and higher power of ten.
        ylim = ax.get_ylim()
        ymin = 10 ** np.floor(np.log10(lifetime_min))
        ymax = 10 ** np.ceil(np.log10(ylim[1]))
        ax.set_ylim(ymin if not np.isnan(ymin) else None, ymax)
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        pdf.savefig()
        plt.close()

        # Plot fit parameter tau0.
        fig, ax = plt.subplots(clear=True)
        ax.errorbar(states, tau0, yerr=tau0_sd, marker="o")
        ax.set(
            xlabel=xlabel, ylabel=r"Fit Parameter $\tau_0$ / Frames", xlim=xlim
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pdf.savefig()
        if not np.all(np.isnan(tau0)):
            # Set y axis to log scale (fit parameter tau0).
            # Round y limits to next lower and higher power of ten.
            ylim = ax.get_ylim()
            ymin = 10 ** np.floor(np.log10(np.nanmin(tau0)))
            ymax = 10 ** np.ceil(np.log10(ylim[1]))
            ax.set_ylim(ymin if not np.isnan(ymin) else None, ymax)
            ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            pdf.savefig()
        plt.close()

        # Plot fit parameter beta.
        fig, ax = plt.subplots(clear=True)
        ax.errorbar(states, beta, yerr=beta_sd, marker="o")
        ax.set(
            xlabel=xlabel,
            ylabel=r"Fit Parameter $\beta$",
            xlim=xlim,
            ylim=(0, 1.05),
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pdf.savefig()
        plt.close()

        # Plot R^2 value of the fits.
        fig, ax = plt.subplots(clear=True)
        ax.plot(states, fit_r2, marker="o")
        ax.set(
            xlabel=xlabel,
            ylabel=r"Coeff. of Determ. $R^2$",
            xlim=xlim,
            ylim=(0, 1.05),
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pdf.savefig()
        plt.close()

        # Plot mean squared error.
        fig, ax = plt.subplots(clear=True)
        ax.plot(states, fit_mse, marker="o")
        ax.set(xlabel=xlabel, ylabel=r"MSE / Frames$^2$", xlim=xlim)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pdf.savefig()
        if not np.all(np.isnan(fit_mse)):
            # Set y axis to log scale (mean squared error).
            # Round y limits to next lower and higher power of ten.
            ylim = ax.get_ylim()
            ymin = 10 ** np.floor(np.log10(np.nanmin(fit_mse[fit_mse > 0])))
            ymax = 10 ** np.ceil(np.log10(ylim[1]))
            ax.set_ylim(ymin if not np.isnan(ymin) else None, ymax)
            ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            pdf.savefig()
        plt.close()

        # Plot fitted region.
        fig, ax = plt.subplots(clear=True)
        ax.plot(states, fit_start, label="Start", marker="^")
        ax.plot(states, fit_stop, label="End", marker="v")
        ax.set(xlabel=xlabel, ylabel="Fitted Region / Frames", xlim=xlim)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend = ax.legend(ncol=2)
        pdf.savefig()
        plt.close()

        # Plot remain probabilities and fits for each state.
        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(color=colors)
        for i, rp in enumerate(remain_props.T):
            times_fit = times[fit_start[i] : fit_stop[i]]
            fit = mdt.func.kww(times_fit, *popt[i])
            lines = ax.plot(times, rp, label=r"$%d$" % states[i], linewidth=1)
            ax.plot(
                times_fit, fit, linestyle="dashed", color=lines[0].get_color()
            )
        ax.set(
            xlabel="Lag Time / Frames",
            ylabel="Decay Law",
            xlim=(times[1], times[-1]),
            ylim=(0, 1),
        )
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        legend = ax.legend(
            title="State Index",
            loc="upper right",
            ncol=3,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

        # Plot fit residuals for each state.
        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(color=colors)
        for i, rp in enumerate(remain_props.T):
            times_fit = times[fit_start[i] : fit_stop[i]]
            fit = mdt.func.kww(times_fit, *popt[i])
            res = rp[fit_start[i] : fit_stop[i]] - fit
            ax.plot(times_fit, res, label=r"$%d$" % states[i])
        ax.set(
            xlabel="Lag Time / Frames",
            ylabel="Fit Residuals",
            xlim=(times[1], times[-1]),
        )
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        legend = ax.legend(
            title="State Index",
            loc="lower right",
            ncol=3,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()
    print("Created {}".format(outfile))
    print("Elapsed time:         {}".format(datetime.now() - timer))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

    print("\n")
    print("{} done".format(os.path.basename(sys.argv[0])))
    print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
    _cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
    print("CPU time:             {}".format(_cpu_time))
    print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
    print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
