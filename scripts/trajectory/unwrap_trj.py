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
import mdtools as mdt


if __name__ == '__main__':

    timer_tot = datetime.now()
    proc = psutil.Process()

    parser = argparse.ArgumentParser(
        description=(
            "Unwrap a trajectory, i.e. unpack the atoms out of"
            " the pimary unit cell and calculate their"
            " positions in real space. This script uses the"
            " algorithm proposed by von Bülow et al. in"
            " J. Chem. Phys., 2020, 153, 021101. Basically it"
            " calculates the atom displacements of the wrapped"
            " atoms from frame to frame and adds these"
            " displacements to the previous unwrapped atom"
            " positions to build the unwraped trajectory."
        )
    )

    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="Trajectory file [<.trr/.xtc/.gro/.pdb/.xyz/.mol2/...>]."
             " See supported coordinate formats of MDAnalysis."
    )
    parser.add_argument(
        '-s',
        dest='TOPFILE',
        type=str,
        required=True,
        help="Topology file [<.top/.tpr/.gro/.pdb/.xyz/.mol2/...>]. See"
             " supported topology formats of MDAnalysis."
    )
    parser.add_argument(
        '--otrj',
        dest='TRJFILE_OUT',
        type=str,
        required=True,
        help="Output trajectory."
    )
    parser.add_argument(
        '--otop',
        dest='TOPFILE_OUT',
        type=str,
        required=True,
        help="Output topology."
    )

    parser.add_argument(
        "--sel",
        dest="SEL",
        type=str,
        nargs="+",
        required=False,
        default=["all"],
        help="Selection group. Only the atoms of this group will be"
             " unwrapped and written to file. Use 'all' if you want to"
             " unwrap all atoms in the trajectory. See MDAnalysis"
             " selection commands for possible choices (e.g. 'resname"
             " water'). Default: all"
    )
    parser.add_argument(
        "--make-whole",
        dest="MAKEWHOLE",
        required=False,
        default=False,
        action='store_true',
        help="Make the compounds of of the selection group that are"
             " split across the simulation box edges whole again for"
             " each individual frame before unwrapping."
    )
    parser.add_argument(
        "--keep-whole",
        dest="KEEPWHOLE",
        required=False,
        default=False,
        action='store_true',
        help="If the molecules in the input trajectory are already whole"
             " for each frame, it is sufficient to start from a"
             " structure with (at least a part of) the whole molecules"
             " in the primary unit cell and then propagate these whole"
             " molecules in real space instead of making the molecules"
             " whole for each individual frame. Note that --make-whole"
             " takes precedence over --keep-whole."
    )
    parser.add_argument(
        "--compound",
        dest="COMPOUND",
        type=str,
        required=False,
        default='residues',
        help="Which type of component of the selection group to make/keep"
             " whole. Must be either 'group', 'segments', 'residues',"
             " 'molecules' or 'fragments'. Refer to the MDAnalysis user"
             " guide for the meaning of these terms"
             " (https://userguide.mdanalysis.org/1.0.0/groups_of_atoms.html)."
             " Default: residues"
    )
    parser.add_argument(
        "--center",
        dest="CENTER",
        type=str,
        required=False,
        default="com",
        help="How to define the centers of the compounds. Must be either"
             "``'com'`` for center of mass or ``'cog'`` for center of"
             " geometry. The compounds are shifted in such a way that"
             " their centers lie within the primary unit cell. A change"
             " of CENTER might affect the unwrapped trajectory slightly,"
             " because the unwrapping might start from a slightly"
             " different configuration. Default: com"
    )

    parser.add_argument(
        "-e",
        dest="END",
        type=int,
        required=False,
        default=-1,
        help="Last frame to unwrap (exclusive, i.e. the last frame to"
             " unwrap is actually END-1). Default: -1 (means unwrap the"
             " the complete trajectory)"
    )

    parser.add_argument(
        '--debug',
        dest='DEBUG',
        required=False,
        default=False,
        action='store_true',
        help="Run in debug mode."
    )

    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())

    print("\n\n\n", flush=True)
    u = mdt.select.universe(top=args.TOPFILE,
                            trj=args.TRJFILE,
                            verbose=True)

    print("\n\n\n", flush=True)
    sel = mdt.select.atoms(ag=u,
                           sel=' '.join(args.SEL),
                           verbose=True)

    _, END, _, n_frames = mdt.check.frame_slicing(
        start=0,
        stop=args.END,
        step=1,
        n_frames_tot=u.trajectory.n_frames)
    last_frame = u.trajectory[END - 1].frame

    print("\n\n\n", flush=True)
    print("Unwrapping trajectory", flush=True)
    print("  Total number of frames in trajectory: {:>9d}"
          .format(u.trajectory.n_frames),
          flush=True)
    print("  Time step per frame:                  {:>9} (ps)\n"
          .format(u.trajectory[0].dt),
          flush=True)
    timer = datetime.now()

    mdt.box.unwrap_trj(topfile=args.TOPFILE_OUT,
                       trjfile=args.TRJFILE_OUT,
                       universe=u,
                       atm_grp=sel,
                       end=END,
                       make_whole=args.MAKEWHOLE,
                       keep_whole=args.KEEPWHOLE,
                       compound=args.COMPOUND,
                       center=args.CENTER,
                       verbose=True,
                       debug=args.DEBUG)

    print(flush=True)
    print("Frames unwrapped: {}".format(n_frames), flush=True)
    print("First frame: {:>12d}    Last frame: {:>12d}    "
          "Every Nth frame: {:>12d}"
          .format(u.trajectory[0].frame, last_frame, 1),
          flush=True)
    print("Start time:  {:>12}    End time:   {:>12}    "
          "Every Nth time:  {:>12} (ps)"
          .format(u.trajectory[0].time,
                  u.trajectory[END - 1].time,
                  u.trajectory[0].dt),
          flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now() - timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)

    print("\n\n\n{} done".format(os.path.basename(sys.argv[0])))
    print("Elapsed time:         {}"
          .format(datetime.now() - timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss / 2**20),
          flush=True)
