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


# This python script is inspired by the work of Florian Müller-Plathe
# and Wilfred F. van Gunsteren which was adapted by Oleg Borodin and
# Grant D. Smith
# * F. Müller-Plathe, W.F. van Gunsteren, JCP, 1995, 103, 4745-4756
# * O. Borodin, G.D. Smith, Macromolecules, 2006, 39, 1620-1629
# * O. Borodin, G.D. Smith, Macromolecules, 2007, 40, 1252-1258




import sys
import os
from datetime import datetime
import psutil
import argparse
import numpy as np
import mdtools as mdt




if __name__ == '__main__':
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Calculate the 'coordination trajectory' of a"
                     " reference group of atoms (usually a single atom,"
                     " e.g. the first lithium ion) to a selection group"
                     " of atoms (usually all atoms with the same"
                     " atomtype and moleculetype, e.g. all ether oxygens"
                     " of all PEO polymers). This is referred to as"
                     " 'topological map', because it shows to which"
                     " selection atoms the reference atom was coodinated"
                     " at a given time. If the reference group consists"
                     " of more than one atom, the center of mass of all"
                     " reference atoms is taken as reference position."
                     " A selection atom is considered to be coordinated"
                     " to the reference position, if its distance to the"
                     " reference position is within a given cutoff."
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
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename."
    )
    
    parser.add_argument(
        '--ref',
        dest='REF',
        type=str,
        nargs='+',
        required=True,
        help="Reference group. See MDAnalysis selection commands for"
             " possible choices, e.g. 'resid 0'. If the reference group"
             " consists of more than one atom, the center of mass is"
             " taken."
    )
    parser.add_argument(
        "--sel",
        dest="SEL",
        type=str,
        nargs="+",
        required=True,
        help="Selection group. See MDAnalysis selection commands for"
             " possible choices. E.g. 'type OE'"
    )
    parser.add_argument(
        '-c',
        dest='CUTOFF',
        type=float,
        required=True,
        help="Cutoff distance in Angstrom. A selection atom is"
             " considered to be coordinated to the reference position,"
             " if its distance to the reference position is within this"
             " cutoff."
    )
    
    parser.add_argument(
        '-b',
        dest='BEGIN',
        type=int,
        required=False,
        default=0,
        help="First frame to read. Frame numbering starts at zero."
             " Default: 0"
    )
    parser.add_argument(
        "-e",
        dest="END",
        type=int,
        required=False,
        default=-1,
        help="Last frame to read (exclusive, i.e. the last frame read is"
             " actually END-1). Default: -1 (means read the very last"
             " frame of the trajectory)"
    )
    parser.add_argument(
        '--every',
        dest='EVERY',
        type=int,
        required=False,
        default=1,
        help="Read every n-th frame. Default: 1"
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
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    if ref.n_atoms > 1:
        print("\n\n\n", flush=True)
        print("The reference group contain more than one atom. Using\n"
              "its center of mass as reference position")
        print(flush=True)
        mdt.check.masses(ag=ref, flash_test=False)
    
    
    BEGIN, END, EVERY, n_frames = mdt.check.frame_slicing(
                                      start=args.BEGIN,
                                      stop=args.END,
                                      step=args.EVERY,
                                      n_frames_tot=u.trajectory.n_frames)
    last_frame = u.trajectory[END-1].frame
    
    
    seg_types = np.unique(sel.segids)
    res_types = np.unique(sel.resnames)
    atm_types = np.unique(sel.types)
    n_seg_types = len(seg_types)
    n_res_types = len(res_types)
    n_atm_types = len(atm_types)
    max_segs = np.zeros(n_seg_types, dtype=np.uint16)
    max_ress = np.zeros((n_seg_types, n_res_types), dtype=np.uint16)
    max_atms = np.zeros((n_seg_types, n_res_types, n_atm_types),
                        dtype=np.uint16)
    seg_ix = [[[] for j in range(n_seg_types)] for i in range(n_frames)]
    res_ix = [[[] for j in range(n_seg_types)] for i in range(n_frames)]
    atm_ix = [[[] for j in range(n_seg_types)] for i in range(n_frames)]
    pos = np.full((n_frames, 3), np.nan, dtype=np.float32)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Reading trajectory", flush=True)
    print("  Total number of frames in trajectory: {:>9d}"
          .format(u.trajectory.n_frames),
          flush=True)
    print("  Time step per frame:                  {:>9} (ps)\n"
          .format(u.trajectory[0].dt),
          flush=True)
    timer = datetime.now()
    timer_frame = datetime.now()
    
    times = np.array([ts.time for ts in u.trajectory[BEGIN:END:EVERY]],
                     dtype=np.float32)
    
    for i, ts in enumerate(u.trajectory[BEGIN:END:EVERY]):
        if (ts.frame % 10**(len(str(ts.frame))-1) == 0 or
            ts.frame == END-1):
            print("  Frame   {:12d}".format(ts.frame), flush=True)
            print("    Step: {:>12}    Time: {:>12} (ps)"
                  .format(ts.data['step'], ts.data['time']),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now()-timer_frame),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss/2**20),
                  flush=True)
            timer_frame = datetime.now()
        
        mdt.box.wrap(ag=ref, debug=args.DEBUG)
        mdt.box.wrap(ag=sel, debug=args.DEBUG)
        if ref.n_atoms > 1:
            mdt.box.make_whole(ag=ref, debug=args.DEBUG)
            pos[i] = mdt.strc.com(ag=ref,
                                  pbc=True,
                                  debug=args.DEBUG)
        else:
            pos[i] = ref[0].position
        
        if args.DEBUG:
            mdt.check.box(box=ts.dimensions, with_angles=True, dim=1)
            if np.all(ts.dimensions[3:] == 90):
                amin = 0
                amax = ts.dimensions[3:]
            else:
                amin = None
                amax = None
                print("The box is not orthorhombic, minimum and maximum"
                      " coordinates will not be checked.")
            mdt.check.pos_array(pos_array=pos[i],
                                  shape=(3,),
                                  amin=amin,
                                  amax=amax)
        
        sel_near_ref = mdt.select.atoms_around_point(ag=sel,
                                                     point=pos[i],
                                                     cutoff=args.CUTOFF)
        seg_type_ix = np.intersect1d(seg_types,
                                     sel_near_ref.segids,
                                     return_indices=True)[1]
        for j in seg_type_ix:
            seg_atms = sel_near_ref.select_atoms("segid {}"
                                                 .format(seg_types[j]))
            seg_ix[i][j] = seg_atms.segments.segindices + 1
            if max_segs[j] < seg_atms.segments.n_segments:
                max_segs[j] = seg_atms.segments.n_segments
            for s_ix, seg in enumerate(seg_atms.segments):
                res_ix[i][j].append([[] for k in range(n_res_types)])
                atm_ix[i][j].append([[] for k in range(n_res_types)])
                s_atms = seg_atms & seg.atoms
                res_type_ix = np.intersect1d(res_types,
                                             s_atms.resnames,
                                             return_indices=True)[1]
                for k in res_type_ix:
                    res_atms = s_atms.select_atoms("resname {}"
                                                   .format(res_types[k]))
                    res_ix[i][j][s_ix][k] = res_atms.residues.resindices + 1
                    if max_ress[j][k] < res_atms.residues.n_residues:
                        max_ress[j][k] = res_atms.residues.n_residues
                    for r_ix, res in enumerate(res_atms.residues):
                        atm_ix[i][j][s_ix][k].append([[] for l in range(n_atm_types)])
                        r_atms = res_atms & res.atoms
                        atm_type_ix = np.intersect1d(atm_types,
                                                     r_atms.types,
                                                     return_indices=True)[1]
                        for l in atm_type_ix:
                            atms = r_atms.select_atoms("type {}"
                                                       .format(atm_types[l]))
                            atm_ix[i][j][s_ix][k][r_ix][l] = atms.indices + 1
                            if max_atms[j][k][l] < atms.n_atoms:
                                max_atms[j][k][l] = atms.n_atoms
    
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
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()
    
    mdt.fh.write_header(args.OUTFILE)
    with open(args.OUTFILE, 'a') as outfile:
        outfile.write("# Topological map\n")
        outfile.write("# Cutoff (Angstrom): {}\n".format(args.CUTOFF))
        outfile.write("#\n")
        outfile.write("#\n")
        outfile.write("# Reference: '{}'\n".format(' '.join(args.REF)))
        outfile.write("#   Segments:               {}\n".format(ref.n_segments))
        outfile.write("#     Different segments:   {}\n".format(len(np.unique(ref.segids))))
        outfile.write("#     Segment name(s):      '{}'\n".format('\' \''.join(i for i in np.unique(ref.segids))))
        outfile.write("#   Residues:               {}\n".format(ref.n_residues))
        outfile.write("#     Different residues:   {}\n".format(len(np.unique(ref.resnames))))
        outfile.write("#     Residue name(s):      '{}'\n".format('\' \''.join(i for i in np.unique(ref.resnames))))
        outfile.write("#   Atoms:                  {}\n".format(ref.n_atoms))
        outfile.write("#     Different atom names: {}\n".format(len(np.unique(ref.names))))
        outfile.write("#     Atom name(s):         '{}'\n".format('\' \''.join(i for i in np.unique(ref.names))))
        outfile.write("#     Different atom types: {}\n".format(len(np.unique(ref.types))))
        outfile.write("#     Atom type(s):         '{}'\n".format('\' \''.join(i for i in np.unique(ref.types))))
        outfile.write("#   Fragments:              {}\n".format(len(ref.fragments)))
        outfile.write("#\n")
        outfile.write("# Selection: '{}'\n".format(' '.join(args.SEL)))
        outfile.write("#   Segments:               {}\n".format(sel.n_segments))
        outfile.write("#     Different segments:   {}\n".format(len(np.unique(sel.segids))))
        outfile.write("#     Segment name(s):      '{}'\n".format('\' \''.join(i for i in np.unique(sel.segids))))
        outfile.write("#   Residues:               {}\n".format(sel.n_residues))
        outfile.write("#     Different residues:   {}\n".format(len(np.unique(sel.resnames))))
        outfile.write("#     Residue name(s):      '{}'\n".format('\' \''.join(i for i in np.unique(sel.resnames))))
        outfile.write("#   Atoms:                  {}\n".format(sel.n_atoms))
        outfile.write("#     Different atom names: {}\n".format(len(np.unique(sel.names))))
        outfile.write("#     Atom name(s):         '{}'\n".format('\' \''.join(i for i in np.unique(sel.names))))
        outfile.write("#     Different atom types: {}\n".format(len(np.unique(sel.types))))
        outfile.write("#     Atom type(s):         '{}'\n".format('\' \''.join(i for i in np.unique(sel.types))))
        outfile.write("#   Fragments:              {}\n".format(len(sel.fragments)))
        outfile.write("#\n")
        outfile.write("#\n")
        outfile.write("# The columns contain:\n")
        outfile.write("#   1 Time (ps)\n")
        outfile.write("#   2-4 x, y and z coordinate of the reference position (A)\n")
        outfile.write("#   5- Indices of all selection segment/residue/atom types that are within {} A of the reference position\n".format(args.CUTOFF))
        outfile.write("#\n")
        outfile.write("# Indices start at 1\n")
        outfile.write("# Indices of 0 are invalid and just fillers, meaning that no segment/residue/atom of the corresponding type is at the given time within the cutoff of the reference position\n")
        outfile.write("# Residue indices are multiplied by -1, to distinguish them from atom indices (e.g. residue -5 is actually residue 5)\n")
        outfile.write("# Segment indices are numbered alphabetically to distinguish them from atom and residue indices\n")
        outfile.write("#\n")
        outfile.write('# Column number:\n')
        
        # Column numbers
        outfile.write("# {:>12d}   ".format(1))
        counter = 2
        for j in range(len(pos[0])):
            outfile.write(" {:>16d}".format(counter))
            counter += 1
        for j in range(n_seg_types):
            for s_ix in range(max_segs[j]):
                outfile.write("      {:>12d}".format(counter))
                counter += 1
                for k in range(n_res_types):
                    for r_ix in range(max_ress[j][k]):
                        outfile.write("    {:>6d} ".format(counter))
                        counter += 1
                        for l in range(n_atm_types):
                            for atm in range(max_atms[j][k][l]):
                                outfile.write(" {:>6d}".format(counter))
                                counter += 1
        outfile.write("\n")
        
        # Column headers
        outfile.write("# {:>12s}   ".format("Time"))
        for j in ["x", "y", "z"]:
            outfile.write(" {:>16s}".format(j))
        for j in range(n_seg_types):
            for s_ix in range(max_segs[j]):
                outfile.write("      {:>12s}".format(seg_types[j]))
                for k in range(n_res_types):
                    for r_ix in range(max_ress[j][k]):
                        outfile.write("    {:>6s} ".format(res_types[k]))
                        for l in range(n_atm_types):
                            for atm in range(max_atms[j][k][l]):
                                outfile.write(" {:>6s}".format(atm_types[l]))
        outfile.write("\n")
        
        # Data
        for i in range(n_frames):
            outfile.write("  {:>12.3f}   ".format(times[i]))
            for j in range(len(pos[i])):
                outfile.write(" {:16.9e}".format(pos[i][j]))
            for j in range(n_seg_types):
                if len(seg_ix[i][j]) < max_segs[j]:
                    seg_ix[i][j] = np.append(seg_ix[i][j],
                                             [0 for i in range(max_segs[j]-len(seg_ix[i][j]))]).astype(np.uint16)
                if len(res_ix[i][j]) < max_segs[j]:
                    res_ix[i][j] += [[] for i in range(max_segs[j]-len(res_ix[i][j]))]
                if len(atm_ix[i][j]) < max_segs[j]:
                    atm_ix[i][j] += [[] for i in range(max_segs[j]-len(atm_ix[i][j]))]
                for s_ix in range(max_segs[j]):
                    outfile.write("      {:>12s}".format(mdt.nph.excel_colname(seg_ix[i][j][s_ix])))
                    if len(res_ix[i][j][s_ix]) < n_res_types:
                        res_ix[i][j][s_ix] += [[] for i in range(n_res_types-len(res_ix[i][j][s_ix]))]
                    if len(atm_ix[i][j][s_ix]) < n_res_types:
                        atm_ix[i][j][s_ix] += [[] for i in range(n_res_types-len(atm_ix[i][j][s_ix]))]
                    for k in range(n_res_types):
                        if len(res_ix[i][j][s_ix][k]) < max_ress[j][k]:
                            res_ix[i][j][s_ix][k] = np.append(res_ix[i][j][s_ix][k],
                                                              [0 for i in range(max_ress[j][k]-len(res_ix[i][j][s_ix][k]))]).astype(np.int32)
                        if len(atm_ix[i][j][s_ix][k]) < max_ress[j][k]:
                            atm_ix[i][j][s_ix][k] += [[] for i in range(max_ress[j][k]-len(atm_ix[i][j][s_ix][k]))]
                        for r_ix in range(max_ress[j][k]):
                            outfile.write("    {:>6d} ".format(-res_ix[i][j][s_ix][k][r_ix]))
                            if len(atm_ix[i][j][s_ix][k][r_ix]) < n_atm_types:
                                atm_ix[i][j][s_ix][k][r_ix] += [[] for i in range(n_atm_types-len(atm_ix[i][j][s_ix][k][r_ix]))]
                            for l in range(n_atm_types):
                                if len(atm_ix[i][j][s_ix][k][r_ix][l]) < max_atms[j][k][l]:
                                    atm_ix[i][j][s_ix][k][r_ix][l] = np.append(atm_ix[i][j][s_ix][k][r_ix][l],
                                                                               [0 for i in range(max_atms[j][k][l]-len(atm_ix[i][j][s_ix][k][r_ix][l]))]).astype(np.uint32)
                                for atm in range(max_atms[j][k][l]):
                                    outfile.write(" {:>6d}".format(atm_ix[i][j][s_ix][k][r_ix][l][atm]))
            outfile.write("\n")
        
        outfile.flush()
    
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
