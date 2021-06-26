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
import mdtools as mdt
from lifetime_hist_serial import (parse_user_input,
                                  lifetime_hist_bound,
                                  lifetime_hist_unbound)




if __name__ == '__main__':
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    additional_description = (
        "This script is parallelized. The number of CPUs to use is"
        " specified (in decreasing precedence) by either one of the"
        " environment variables OMP_NUM_THREADS, SLURM_CPUS_PER_TASK,"
        " SLURM_JOB_CPUS_PER_NODE, SLURM_CPUS_ON_NODE or python intern"
        " by os.cpu_count(). Best performance is considered to be"
        " reached with 1-3 CPUs.")
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
        raise ValueError("The reference atom group contains no atoms")
    if sel.n_atoms <= 0:
        raise ValueError("The selection atom group contains no atoms")
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    BEGIN, END, EVERY, n_frames = mdt.check.frame_slicing(
                                      start=args.BEGIN,
                                      stop=args.END,
                                      step=args.EVERY,
                                      n_frames_tot=u.trajectory.n_frames)
    last_frame = u.trajectory[END-1].frame
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
    if nchunks > int(n_frames/10):
        nchunks = int(n_frames/10)
    pool = mdt.parallel.ProcessPool(nprocs=nchunks)
    
    chunk_size = int((END-BEGIN) / nchunks)
    chunk_size -= chunk_size % EVERY
    if chunk_size: # !=0
        nchunks = int((END-BEGIN) / chunk_size)
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
                               BEGIN+chunk*chunk_size,
                               BEGIN+(chunk+1)*chunk_size,
                               EVERY,
                               True,
                               args.DEBUG))
    if BEGIN+(chunk+1)*chunk_size < END:
        chunk += 1
        pool.submit_task(func=mdt.strc.contact_matrices,
                         args=(args.TOPFILE,
                               args.TRJFILE,
                               args.REF,
                               args.SEL,
                               args.CUTOFF,
                               args.COMPOUND,
                               args.MINCONTACTS,
                               BEGIN+chunk*chunk_size,
                               END,
                               EVERY,
                               True,
                               args.DEBUG))
    elif BEGIN+(chunk+1)*chunk_size > END:
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
                  u.trajectory[END-1].time,
                  u.trajectory[0].dt * EVERY),
          flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
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
              .format(datetime.now()-timer),
              flush=True)
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Calculating lifetime histograms", flush=True)
    timer = datetime.now()
    
    pool = mdt.parallel.ProcessPool(nprocs=num_CPUs)
    pool.submit_task(func=lifetime_hist_bound, args=(cms, args.DEBUG))
    pool.submit_task(func=lifetime_hist_unbound,
                     args=(cms, 1, args.DEBUG))
    pool.submit_task(func=lifetime_hist_unbound,
                     args=(cms, 0, args.DEBUG))
    del cms
    
    hist_bound, hist_unbound_ref, hist_unbound_sel = pool.get_results()
    pool.close()
    pool.join()
    
    lag_times = np.arange(0, timestep*n_frames*EVERY,
                          timestep*EVERY,
                          dtype=np.float32)
    
    tot_counts_bound = np.sum(hist_bound)
    tot_counts_unbound_ref = np.sum(hist_unbound_ref)
    tot_counts_unbound_sel = np.sum(hist_unbound_sel)
    if tot_counts_bound < 0:
        raise ValueError("The total number of counts in the histogram"
                         " for bound reference-selection complexes is"
                         " less than zero. This should not have happened")
    if tot_counts_unbound_ref < 0:
        raise ValueError("The total number of counts in the histogram"
                         " for unbound reference compounds is less than"
                         " zero. This should not have happened")
    if tot_counts_unbound_sel < 0:
        raise ValueError("The total number of counts in the histogram"
                         " for unbound selection compounds is less than"
                         " zero. This should not have happened")
    
    hist_bound = hist_bound / tot_counts_bound
    hist_unbound_ref = hist_unbound_ref / tot_counts_unbound_ref
    hist_unbound_sel = hist_unbound_sel / tot_counts_unbound_sel
    
    mean_bound = np.sum(lag_times * hist_bound)
    mean_unbound_ref = np.sum(lag_times * hist_unbound_ref)
    mean_unbound_sel = np.sum(lag_times * hist_unbound_sel)
    
    sd_bound = np.sqrt(np.sum((lag_times-mean_bound)**2 * hist_bound))
    sd_unbound_ref = np.sqrt(np.sum((lag_times-mean_unbound_ref)**2 *
                                    hist_unbound_ref))
    sd_unbound_sel = np.sqrt(np.sum((lag_times-mean_unbound_sel)**2 *
                                    hist_unbound_sel))
    
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
        "Average lifetime of reference-selection complexes and unbound\n"
        "reference and selection compounds\n"
        "\n"
        "\n"
        "Cutoff (Angstrom)     = {}\n"
        "Compound              = {}\n"
        "Minimum contacts      = {}\n"
        "Allowed intermittency = {}\n"
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
        "The average lifetime tau is estimated as mean of the lifetime\n"
        "histogram. This estimate is only meaningful, if the fraction\n"
        "of counts for the last two lag times is almost zero, since\n"
        "otherwise the lifetime might be larger than these lag times\n"
        "and you might not have sampled the correct lifetime\n"
        "distribution.\n"
        "\n"
        "\n"
        "The columns contain:\n"
        "   1 Lifetime tau (ps)\n"
        "   2 Lifetime histogram for bound reference-selection complexes\n"
        "   3 Lifetime histogram for unbound reference compounds\n"
        "   4 Lifetime histogram for unbound selection compounds\n"
        "\n"
        "Column number:\n"
        "{:>14d} {:>16d} {:>16d} {:>16d}\n"
        "\n"
        "Mean (ps):     {:>16.9e} {:>16.9e} {:>16.9e}\n"
        "Std. Dev. (ps):{:>16.9e} {:>16.9e} {:>16.9e}\n"
        "Tot. counts:   {:>16d} {:>16d} {:>16d}\n"
        .format(args.CUTOFF,
                args.COMPOUND,
                args.MINCONTACTS,
                args.INTERMITTENCY,
                
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
                
                1, 2, 3, 4,
                
                mean_bound, mean_unbound_ref, mean_unbound_sel,
                sd_bound, sd_unbound_ref, sd_unbound_sel,
                tot_counts_bound,
                tot_counts_unbound_ref,
                tot_counts_unbound_sel
        )
    )
    
    mdt.fh.savetxt(fname=args.OUTFILE,
                   data=np.column_stack([lag_times,
                                         hist_bound,
                                         hist_unbound_ref,
                                         hist_unbound_sel]),
                   header=header)
    
    print("  Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("{} done".format(os.path.basename(sys.argv[0])), flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
