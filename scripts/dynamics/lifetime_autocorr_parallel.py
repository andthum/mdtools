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




import sys
import os
from datetime import datetime
import psutil
import numpy as np
from scipy import special
import mdtools as mdt
from lifetime_autocorr_serial import (parse_user_input,
                                      autocorr_bound,
                                      autocorr_unbound,
                                      kww,
                                      fit_kww)




if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())


    additional_description = (
        " This script is parallelized. The number of CPUs to use is"
        " specified (in decreasing precedence) by either one of the"
        " environment variables OMP_NUM_THREADS, SLURM_CPUS_PER_TASK,"
        " SLURM_JOB_CPUS_PER_NODE, SLURM_CPUS_ON_NODE or python intern"
        " by os.cpu_count(). Best performance is considered to be"
        " reached, when the number of used CPUs is 1-3 times the number"
        " of blocks for block averaging.")
    args = parse_user_input(add_description=additional_description)
    num_CPUs = mdt.rti.get_num_CPUs()
    print("\n\n\n", flush=True)
    print("Available CPUs: {}".format(num_CPUs), flush=True)




    print("\n\n\n", flush=True)
    u = mdt.select.universe(top=args.TOPFILE,
                            trj=args.TRJFILE,
                            verbose=True)


    print("\n\n\n", flush=True)
    print("Creating selections", flush=True)
    timer = datetime.now()

    ref = u.select_atoms(' '.join(args.REF))
    sel = u.select_atoms(' '.join(args.SEL))
    print("  Reference group: '{}'"
          .format(' '.join(args.REF)),
          flush=True)
    print(mdt.rti.ag_info_str(ag=ref, indent=4))
    print(flush=True)
    print("  Selection group: '{}'"
          .format(' '.join(args.SEL)),
          flush=True)
    print(mdt.rti.ag_info_str(ag=sel, indent=4))

    if ref.n_atoms <= 0:
        raise ValueError("The reference group contains no atoms")
    if sel.n_atoms <= 0:
        raise ValueError("The selection group contains no atoms")

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)




    BEGIN, END, EVERY, n_frames = mdt.check.frame_slicing(
        start=args.BEGIN,
        stop=args.END,
        step=args.EVERY,
        n_frames_tot=u.trajectory.n_frames)
    last_frame = u.trajectory[END - 1].frame
    NBLOCKS, blocksize = mdt.check.block_averaging(n_blocks=args.NBLOCKS,
                                                   n_frames=n_frames,
                                                   check_CPUs=True)
    RESTART, effective_restart = mdt.check.restarts(
        restart_every_nth_frame=args.RESTART,
        read_every_nth_frame=EVERY,
        n_frames=blocksize)
    if args.DEBUG:
        print("\n\n\n", flush=True)
        mdt.check.time_step(trj=u.trajectory[BEGIN:END], verbose=True)
    timestep = u.trajectory[BEGIN].dt




    print("\n\n\n", flush=True)
    print("Reading trajectory", flush=True)
    print("  Total number of frames in trajectory: {:>9d}"
          .format(u.trajectory.n_frames),
          flush=True)
    print("  Time step per frame:                  {:>9} (ps)\n"
          .format(u.trajectory[0].dt),
          flush=True)
    timer = datetime.now()

    nchunks = num_CPUs
    if nchunks > int(n_frames / 10):
        nchunks = int(n_frames / 10)
    pool = mdt.parallel.ProcessPool(nprocs=nchunks)

    chunk_size = int((END - BEGIN) / nchunks)
    chunk_size -= chunk_size % EVERY
    if chunk_size:  # !=0
        nchunks = int((END - BEGIN) / chunk_size)
    else:
        nchunks = 1

    for chunk in range(nchunks):
        pool.submit_task(func=mdt.strc.contact_matrices,
                         args=(args.TOPFILE,
                               args.TRJFILE,
                               args.REF,
                               args.SEL,
                               args.CUTOFF,
                               args.COMPOUND,
                               args.MINCONTACTS,
                               BEGIN + chunk * chunk_size,
                               BEGIN + (chunk + 1) * chunk_size,
                               EVERY,
                               True,
                               args.DEBUG))
    if BEGIN + (chunk + 1) * chunk_size < END:
        chunk += 1
        pool.submit_task(func=mdt.strc.contact_matrices,
                         args=(args.TOPFILE,
                               args.TRJFILE,
                               args.REF,
                               args.SEL,
                               args.CUTOFF,
                               args.COMPOUND,
                               args.MINCONTACTS,
                               BEGIN + chunk * chunk_size,
                               END,
                               EVERY,
                               True,
                               args.DEBUG))
    elif BEGIN + (chunk + 1) * chunk_size > END:
        raise ValueError("I've read more frames than given with -e. This"
                         " should not have happened")

    cms = []
    for result in pool.get_results():
        cms += result
    del result
    pool.close()
    pool.join()

    if len(cms) != n_frames:
        raise ValueError("The number of contact matrices does not equal"
                         " the number of frames to read. This should not"
                         " have happened")

    print(flush=True)
    print("Frames read: {}".format(n_frames), flush=True)
    print("First frame: {:>12d}    Last frame: {:>12d}    "
          "Every Nth frame: {:>12d}"
          .format(u.trajectory[BEGIN].frame, last_frame, EVERY),
          flush=True)
    print("Start time:  {:>12}    End time:   {:>12}    "
          "Every Nth time:  {:>12} (ps)"
          .format(u.trajectory[BEGIN].time,
                  u.trajectory[END - 1].time,
                  u.trajectory[0].dt * EVERY),
          flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)




    if args.INTERMITTENCY > 0:
        print("\n\n\n", flush=True)
        print("Correcting for intermittency", flush=True)
        timer = datetime.now()

        cms = mdt.dyn.correct_intermittency(
            list_of_arrays=cms,
            intermittency=args.INTERMITTENCY,
            verbose=True,
            debug=args.DEBUG)

        print("Elapsed time:         {}"
              .format(datetime.now() - timer),
              flush=True)
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss / 2**20),
              flush=True)




    print("\n\n\n", flush=True)
    print("Calculating autocorrelation function", flush=True)
    timer = datetime.now()

    pool = mdt.parallel.ProcessPool(nprocs=num_CPUs)
    for block in range(NBLOCKS):
        pool.submit_task(func=autocorr_bound,
                         args=(cms[block * blocksize:(block + 1) * blocksize],
                               effective_restart,
                               args.DEBUG))
        pool.submit_task(func=autocorr_unbound,
                         args=(cms[block * blocksize:(block + 1) * blocksize],
                               1,
                               effective_restart,
                               args.DEBUG))
        pool.submit_task(func=autocorr_unbound,
                         args=(cms[block * blocksize:(block + 1) * blocksize],
                               0,
                               effective_restart,
                               args.DEBUG))
    del cms

    acorr_bound = []
    acorr_unbound_ref = []
    acorr_unbound_sel = []
    for i, result in enumerate(pool.get_results()):
        if i % 3 == 0:
            acorr_bound.append(result)
        elif i % 3 == 1:
            acorr_unbound_ref.append(result)
        elif i % 3 == 2:
            acorr_unbound_sel.append(result)
    del result
    pool.close()
    pool.join()

    if len(acorr_bound) != NBLOCKS:
        raise ValueError("The number of autocorrelation functions for"
                         " bound reference-selection complexes does not"
                         " equal the number of blocks for block"
                         " averaging. This should not have happened")
    if len(acorr_unbound_ref) != NBLOCKS:
        raise ValueError("The number of autocorrelation functions for"
                         " unbound reference compounds does not equal"
                         " the number of blocks for block averaging."
                         " This should not have happened")
    if len(acorr_unbound_sel) != NBLOCKS:
        raise ValueError("The number of autocorrelation functions for"
                         " unbound selection compounds does not equal"
                         " the number of blocks for block averaging."
                         " This should not have happened")

    acorr_bound = np.vstack(acorr_bound)
    acorr_unbound_ref = np.vstack(acorr_unbound_ref)
    acorr_unbound_sel = np.vstack(acorr_unbound_sel)

    if NBLOCKS > 1:
        acorr_bound, acorr_bound_sd = mdt.stats.block_average(acorr_bound)
        acorr_bound_sd_fit = np.copy(acorr_bound_sd)
        acorr_bound_sd_fit[acorr_bound_sd_fit == 0] = 1e-20
        acorr_unbound_ref, acorr_unbound_ref_sd = mdt.stats.block_average(acorr_unbound_ref)
        acorr_unbound_ref_sd_fit = np.copy(acorr_unbound_ref_sd)
        acorr_unbound_ref_sd_fit[acorr_unbound_ref_sd_fit == 0] = 1e-20
        acorr_unbound_sel, acorr_unbound_sel_sd = mdt.stats.block_average(acorr_unbound_sel)
        acorr_unbound_sel_sd_fit = np.copy(acorr_unbound_sel_sd)
        acorr_unbound_sel_sd_fit[acorr_unbound_sel_sd_fit == 0] = 1e-20
    else:
        acorr_bound = np.squeeze(acorr_bound)
        acorr_bound_sd_fit = None
        acorr_unbound_ref = np.squeeze(acorr_unbound_ref)
        acorr_unbound_ref_sd_fit = None
        acorr_unbound_sel = np.squeeze(acorr_unbound_sel)
        acorr_unbound_sel_sd_fit = None

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Fitting autocorrelation function", flush=True)
    timer = datetime.now()

    lag_times = np.arange(0,
                          timestep * blocksize * EVERY,
                          timestep * EVERY,
                          dtype=np.float32)
    if args.ENDFIT is None:
        endfit = int(0.9 * len(lag_times))
        args.ENDFIT = lag_times[endfit]
    else:
        _, endfit = mdt.nph.find_nearest(lag_times,
                                         args.ENDFIT,
                                         return_index=True,
                                         debug=args.DEBUG)


    stopfit = np.nonzero(acorr_bound < args.STOPFIT)[0]
    if stopfit.size == 0:
        stopfit = len(acorr_bound)
    else:
        stopfit = stopfit[0]
    valid = np.isfinite(acorr_bound)
    valid[min(endfit, stopfit):] = False
    if acorr_bound_sd_fit is None:
        ysd = None
    else:
        ysd = acorr_bound_sd_fit[valid]
    if np.any(valid):
        fit_bound = fit_kww(xdata=lag_times[valid],
                            ydata=acorr_bound[valid],
                            ysd=ysd)
    else:
        fit_bound = np.full(4, np.nan)
    lifetime_bound = (fit_bound[0] / fit_bound[2] *
                      special.gamma(1 / fit_bound[2]))
    kww_bound = kww(t=lag_times, tau=fit_bound[0], beta=fit_bound[2])
    kww_bound[~valid] = np.nan


    stopfit = np.nonzero(acorr_unbound_ref < args.STOPFIT)[0]
    if stopfit.size == 0:
        stopfit = len(acorr_unbound_ref)
    else:
        stopfit = stopfit[0]
    valid = np.isfinite(acorr_unbound_ref)
    valid[min(endfit, stopfit):] = False
    if acorr_unbound_ref_sd_fit is None:
        ysd = None
    else:
        ysd = acorr_unbound_ref_sd_fit[valid]
    if np.any(valid):
        fit_unbound_ref = fit_kww(xdata=lag_times[valid],
                                  ydata=acorr_unbound_ref[valid],
                                  ysd=ysd)
    else:
        fit_unbound_ref = np.full(4, np.nan)
    lifetime_unbound_ref = (fit_unbound_ref[0] / fit_unbound_ref[2] *
                            special.gamma(1 / fit_unbound_ref[2]))
    kww_unbound_ref = kww(t=lag_times,
                          tau=fit_unbound_ref[0],
                          beta=fit_unbound_ref[2])
    kww_unbound_ref[~valid] = np.nan


    stopfit = np.nonzero(acorr_unbound_sel < args.STOPFIT)[0]
    if stopfit.size == 0:
        stopfit = len(acorr_unbound_sel)
    else:
        stopfit = stopfit[0]
    valid = np.isfinite(acorr_unbound_sel)
    valid[min(endfit, stopfit):] = False
    if acorr_unbound_sel_sd_fit is None:
        ysd = None
    else:
        ysd = acorr_unbound_sel_sd_fit[valid]
    if np.any(valid):
        fit_unbound_sel = fit_kww(xdata=lag_times[valid],
                                  ydata=acorr_unbound_sel[valid],
                                  ysd=ysd)
    else:
        fit_unbound_sel = np.full(4, np.nan)
    lifetime_unbound_sel = (fit_unbound_sel[0] / fit_unbound_sel[2] *
                            special.gamma(1 / fit_unbound_sel[2]))
    kww_unbound_sel = kww(t=lag_times,
                          tau=fit_unbound_sel[0],
                          beta=fit_unbound_sel[2])
    kww_unbound_sel[~valid] = np.nan


    del acorr_bound_sd_fit
    del acorr_unbound_ref_sd_fit
    del acorr_unbound_sel_sd_fit
    del valid
    del ysd

    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()

    header = (
        "Average lifetime of reference-selection complexes and unbound\n"
        "reference and selection compounds\n"
        "\n"
        "\n"
        "Cutoff (Angstrom)     = {}\n"
        "Compound              = {}\n"
        "Minimum contacts      = {}\n"
        "Allowed intermittency = {}\n"
        "End fit at            = {} ps\n"
        "Only fit C(t)        >= {}\n"
        "\n"
        "\n"
        "Reference: '{}'\n"
        "  Segments:               {}\n"
        "    Different segments:   {}\n"
        "    Segment name(s):      '{}'\n"
        "  Residues:               {}\n"
        "    Different residues:   {}\n"
        "    Residue name(s):      '{}'\n"
        "  Atoms:                  {}\n"
        "    Different atom names: {}\n"
        "    Atom name(s):         '{}'\n"
        "    Different atom types: {}\n"
        "    Atom type(s):         '{}'\n"
        "  Fragments:              {}\n"
        "\n"
        "Selection: '{}'\n"
        "  Segments:               {}\n"
        "    Different segments:   {}\n"
        "    Segment name(s):      '{}'\n"
        "  Residues:               {}\n"
        "    Different residues:   {}\n"
        "    Residue name(s):      '{}'\n"
        "  Atoms:                  {}\n"
        "    Different atom names: {}\n"
        "    Atom name(s):         '{}'\n"
        "    Different atom types: {}\n"
        "    Atom type(s):         '{}'\n"
        "  Fragments:              {}\n"
        "\n"
        "\n"
        "The average lifetime tau is estimated from the integral of a\n"
        "stretched exponential fit of the autocorrelation of the\n"
        "existence function\n"
        "\n"
        "Existence function:\n"
        "  S_ij(t) = 1, if at time t a contact exists between compound\n"
        "               j of the selection group and compound i of the\n"
        "               reference group\n"
        "  S_ij(t) = 0, otherwise\n"
        "\n"
        "For calculating the lifetime of unbound compounds, the\n"
        "existence function is redefined as\n"
        "  S_i(t) = 1, if at time t the reference (selection) compound\n"
        "              i is not in contact with any atom of the\n"
        "              selection (reference) group\n"
        "  S_i(t) = 0, otherwise\n"
        "\n"
        "Autocorrelation of the existence function:\n"
        "  C(t) = < S_ij(t0)*S_ij(t0+t) / S_ij(t0)*S_ij(t0) >\n"
        "  <...> = Average over all existing contacts ij and over all\n"
        "          possible starting times t0\n"
        "  You can interprete C(t) as the percentage of contacts that\n"
        "  still exist or exist again after a lag time t\n"
        "\n"
        "The autocorrelation is fitted using a stretched exponential\n"
        "function, also known as Kohlrausch-Williams-Watts (KWW)\n"
        "function:\n"
        "  f(t) = exp[-(t/tau')^beta]\n"
        "  beta is constrained to the intervall [0, 1]\n"
        "  tau' must be positive\n"
        "\n"
        "The average lifetime tau is calculated as the integral of the\n"
        "KWW function from zero to infinity:\n"
        "  tau = integral_0^infty exp[-(t/tau')^beta] dt\n"
        "      = tau'/beta * Gamma(1/beta)\n"
        "  Gamma(x) = Gamma function\n"
        "  If beta=1, tau=tau'\n"
        "\n"
        "\n"
        .format(args.CUTOFF,
                args.COMPOUND,
                args.MINCONTACTS,
                args.INTERMITTENCY,
                args.ENDFIT,
                args.STOPFIT,

                ' '.join(args.REF),
                ref.n_segments,
                len(np.unique(ref.segids)),
                '\' \''.join(i for i in np.unique(ref.segids)),
                ref.n_residues,
                len(np.unique(ref.resnames)),
                '\' \''.join(i for i in np.unique(ref.resnames)),
                ref.n_atoms,
                len(np.unique(ref.names)),
                '\' \''.join(i for i in np.unique(ref.names)),
                len(np.unique(ref.types)),
                '\' \''.join(i for i in np.unique(ref.types)),
                len(ref.fragments),

                ' '.join(args.SEL),
                sel.n_segments,
                len(np.unique(sel.segids)),
                '\' \''.join(i for i in np.unique(sel.segids)),
                sel.n_residues,
                len(np.unique(sel.resnames)),
                '\' \''.join(i for i in np.unique(sel.resnames)),
                sel.n_atoms,
                len(np.unique(sel.names)),
                '\' \''.join(i for i in np.unique(sel.names)),
                len(np.unique(sel.types)),
                '\' \''.join(i for i in np.unique(sel.types)),
                len(sel.fragments),
                )
    )

    if NBLOCKS == 1:
        columns = (
            "The columns contain:\n"
            "  1 Lag time t (ps)\n"
            "  2 Autocorrelation for bound reference-selection complexes\n"
            "  3 Fit of column 2\n"
            "  4 Autocorrelation for unbound reference compounds\n"
            "  5 Fit of column 4\n"
            "  6 Autocorrelation for unbound selection compounds\n"
            "  7 Fit of column 6\n"
            "\n"
            "Column number:\n"
            "{:>14d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d}\n"
            "\n"
            "Fit:\n"
            "Lifetime tau (ps):               {:>15.9e} {:>33.9e} {:>33.9e}\n"
            "Relaxation time tau' (ps):       {:>15.9e} {:>33.9e} {:>33.9e}\n"
            "Standard deviation of tau' (ps): {:>15.9e} {:>33.9e} {:>33.9e}\n"
            "Stretching exponent beta:        {:>15.9e} {:>33.9e} {:>33.9e}\n"
            "Standard deviation of beta:      {:>15.9e} {:>33.9e} {:>33.9e}\n"
            .format(1, 2, 3, 4, 5, 6, 7,

                    lifetime_bound,
                    lifetime_unbound_ref,
                    lifetime_unbound_sel,
                    fit_bound[0], fit_unbound_ref[0], fit_unbound_sel[0],
                    fit_bound[1], fit_unbound_ref[1], fit_unbound_sel[1],
                    fit_bound[2], fit_unbound_ref[2], fit_unbound_sel[2],
                    fit_bound[3], fit_unbound_ref[3], fit_unbound_sel[3]
                    )
        )
        data = np.column_stack([lag_times,

                                acorr_bound,
                                kww_bound,

                                acorr_unbound_ref,
                                kww_unbound_ref,

                                acorr_unbound_sel,
                                kww_unbound_sel])

    else:
        columns = (
            "The columns contain:\n"
            "   1 Lag time t (ps)\n"
            "   2 Autocorrelation for bound reference-selection complexes\n"
            "   3 Standard deviation of column 2\n"
            "   4 Fit of column 2\n"
            "   5 Autocorrelation for unbound reference compounds\n"
            "   6 Standard deviation of column 5\n"
            "   7 Fit of column 5\n"
            "   8 Autocorrelation for unbound selection compounds\n"
            "   9 Standard deviation of column 8\n"
            "  10 Fit of column 8\n"
            "\n"
            "Column number:\n"
            "{:>14d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d} {:>16d}\n"
            "\n"
            "Fit:\n"
            "Lifetime tau (ps):               {:>32.9e} {:>50.9e} {:>50.9e}\n"
            "Relaxation time tau' (ps):       {:>32.9e} {:>50.9e} {:>50.9e}\n"
            "Standard deviation of tau' (ps): {:>32.9e} {:>50.9e} {:>50.9e}\n"
            "Stretching exponent beta:        {:>32.9e} {:>50.9e} {:>50.9e}\n"
            "Standard deviation of beta:      {:>32.9e} {:>50.9e} {:>50.9e}\n"
            .format(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,

                    lifetime_bound,
                    lifetime_unbound_ref,
                    lifetime_unbound_sel,
                    fit_bound[0], fit_unbound_ref[0], fit_unbound_sel[0],
                    fit_bound[1], fit_unbound_ref[1], fit_unbound_sel[1],
                    fit_bound[2], fit_unbound_ref[2], fit_unbound_sel[2],
                    fit_bound[3], fit_unbound_ref[3], fit_unbound_sel[3]
                    )
        )
        data = np.column_stack([lag_times,

                                acorr_bound,
                                acorr_bound_sd,
                                kww_bound,

                                acorr_unbound_ref,
                                acorr_unbound_ref_sd,
                                kww_unbound_ref,

                                acorr_unbound_sel,
                                acorr_unbound_sel_sd,
                                kww_unbound_sel])

    mdt.fh.savetxt(fname=args.OUTFILE,
                   data=data,
                   header=header + columns)

    print("  Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)




    print("\n\n\n", flush=True)
    print("{} done".format(os.path.basename(sys.argv[0])), flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now() - timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)
