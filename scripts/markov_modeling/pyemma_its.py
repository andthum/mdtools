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
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyemma.msm as msm
import pyemma.plots as mplt
import mdtools as mdt




if __name__ == "__main__":
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    n_cpus = mdt.rti.get_num_CPUs()
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Read a discretized trajectory as e.g. created by"
                     " discrete_pos.py and calculate the implied"
                     " timescales using pyemma.msm.timescales_msm()."
                 )
    )
    
    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="File containing the discretized trajectory stored as"
             " integer numpy.ndarray in .npy format. It is possible to"
             " load multiple trajectories if they are stored in .npy"
             " format as a two dimensional numpy.ndarray. Each"
             " trajectory represents a single particle."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename pattern. There will be created three files:"
             " <OUTFILE>_its.h5 containing the created"
             " pyemma.msm.estimators.implied_timescales.ImpliedTimescales"
             " object;"
             " <OUTFILE>_its.txt containing the implied timescales;"
             " <OUTFILE>_its.pdf containing a plot of the implied"
             " timescales."
    )
    parser.add_argument(
        '--no-plots',
        dest='NOPLOTS',
        required=False,
        default=False,
        action='store_true',
        help="Do not create plots."
    )
    parser.add_argument(
        '--nits',
        dest='NITS',
        type=int,
        required=False,
        default=None,
        help="Number of implied timescales to be computed. Default:"
             " None, i.e. the number of timescales will be automatically"
             " determined. Cannot exceed the number of states."
    )
    parser.add_argument(
        '--errors',
        dest='ERRORS',
        required=False,
        default=False,
        action='store_true',
        help="Estimate the uncertainty of the implied timescales from"
             " Bayesian sampling of the posterior. Note that this will"
             " increase the computational cost and memory consumption"
             " significantly."
    )
    parser.add_argument(
        '--nsamples',
        dest='NSAMPLES',
        type=int,
        required=False,
        default=8,
        help="The number of approximately independent transition matrix"
             " samples generated for each lag time for uncertainty"
             " quantification. Only used if --errors is set. Default: 8"
    )
    
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    
    
    
    print("\n\n\n", flush=True)
    print("Loading discrete trajectories", flush=True)
    timer = datetime.now()
    
    dtrajs = np.load(args.TRJFILE)
    n_frames = dtrajs.shape[-1]
    if n_frames < 2:
        raise ValueError("Trajectories must contain at least two frames"
                         " in order to estimate a Markov model")
    
    if dtrajs.ndim == 1:
        n_trajs = 1
    elif dtrajs.ndim == 2:
        # PyEMMA takes multiple trajectories only as list of
        # numpy.ndarrays, not as 2-dimensional numpy.ndarray
        n_trajs = dtrajs.shape[0]
        dtrajs = [signle_part_traj for signle_part_traj in dtrajs]
        if len(dtrajs) == 1:
            dtrajs = dtrajs[0]
        if len(dtrajs) != n_trajs:
            raise RuntimeError("Unexpected error: len(dtrajs) != n_trajs")
    else:
        raise ValueError("dtrajs has more than two dimensions ({})"
                         .format(dtrajs.ndim))
    
    traj_info = ("  Number of single particle trajectories: {:>9d}\n"
                 "  Number of frames per trajectory:        {:>9d}\n"
                 "  First populated state:                  {:>9d}\n"
                 "  Last populated state:                   {:>9d}"
                 .format(n_trajs,
                         n_frames,
                         np.min(dtrajs),
                         np.max(dtrajs)))
    print("{}".format(traj_info), flush=True)
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Calculating implied timescales", flush=True)
    timer = datetime.now()
    print("  CPUs found: {}".format(n_cpus), flush=True)
    
    if args.ERRORS:
        errors = 'bayes'
        only_timescales = True
    else:
        errors = None
        only_timescales = False
    
    print(flush=True)
    its = msm.timescales_msm(dtrajs=dtrajs,
                             nits=args.NITS,
                             errors=errors,
                             nsamples=args.NSAMPLES,
                             n_jobs=n_cpus,
                             show_progress=True,
                             only_timescales=only_timescales)
    print(flush=True)
    
    del dtrajs
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()
    
    
    mdt.fh.backup(args.OUTFILE+"_its.h5")
    its.save(file_name=args.OUTFILE+"_its.h5")
    print("  Created "+args.OUTFILE+"_its.h5", flush=True)
    
    
    if args.ERRORS:
        if its.sample_mean.shape != its.sample_std.shape:
            raise ValueError("its.sample_mean.shape ({}) !="
                             " its.sample_std.shape ({})"
                             .format(its.sample_mean.shape,
                                     its.sample_std.shape))
        if its.sample_mean.ndim == 2:
            data = np.zeros((its.sample_mean.shape[0],
                             2*its.sample_mean.shape[1]),
                            dtype=np.float64)
            data[:,0::2] = its.sample_mean
            data[:,1::2] = its.sample_std
        else:
            data = np.column_stack((its.sample_mean, its.sample_std))
        header_sd = ("\n"
                     "     Lag time and standard deviation alternating\n"
                     "     Number of transition matrix samples: {}"
                     .format(args.NSAMPLES))
    else:
        data = its.timescales
        header_sd = ""
    data = np.column_stack((its.lags, data))
    
    header=("Markov state model\n"
            "Implied timescales\n\n"
            "Discrete trajectories used for construction:\n"
            + traj_info + "\n\n"
            "The columns contain:\n"
            "  1 Lag time / trajectory steps\n"
            "  2-{} Time scales / trajectory steps\n"
            "    (Descending order{})\n\n"
            .format(data.shape[1], header_sd))
    cols = ' '.join("{:>16d}"
                    .format(i) for i in range(2, data.shape[1]+1))
    cols = "{:>14d} ".format(1) + cols
    header += cols
    
    mdt.fh.savetxt(fname=args.OUTFILE+"_its.txt",
                   data=data,
                   header=header)
    print("  Created "+args.OUTFILE+"_its.txt", flush=True)
    del data, header
    
    
    if not args.NOPLOTS:
        fontsize_labels = 36
        fontsize_ticks = 32
        tick_length = 10
        tick_pad = 12
        label_pad = 16
        
        fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                                 frameon=False,
                                 clear=True,
                                 tight_layout=True)
        axis.set_xlabel(xlabel='Lag time / steps',
                        fontsize=fontsize_labels)
        axis.set_ylabel(ylabel='Timescale / steps',
                        fontsize=fontsize_labels)
        axis.xaxis.labelpad = label_pad
        axis.yaxis.labelpad = label_pad
        axis.xaxis.offsetText.set_fontsize(fontsize_ticks)
        axis.yaxis.offsetText.set_fontsize(fontsize_ticks)
        axis.tick_params(which='major',
                         direction='in',
                         top=True,
                         right=True,
                         length=tick_length,
                         labelsize=fontsize_ticks,
                         pad=tick_pad)
        axis.tick_params(which='minor',
                         direction='in',
                         top=True,
                         right=True,
                         length=0.5*tick_length,
                         labelsize=0.8*fontsize_ticks,
                         pad=tick_pad)
        
        mdt.fh.backup(args.OUTFILE+"_its.pdf")
        mplt.plot_implied_timescales(
            ITS=its,
            ax=axis,
            outfile=args.OUTFILE+"_its.pdf",
            nits=args.NITS if args.NITS is not None else -1)
        print("  Created "+args.OUTFILE+"_its.pdf", flush=True)
    
    
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
