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
import argparse
import numpy as np
import MDAnalysis as mda
import mdtools as mdt




def sort_by_resnames(strcfile, outfile, residue_order):
    """
    Sort the entries in a structure file by residue names.
    
    Paramters
    ---------
    strcfile : str
        Name of the structure file.
    outfile : str
        Name of the ouput file.
    residue_order : array_like, optional
        List of residue names in the desired order. If not given,
        residues will be sorted alphabetically.
    """
    
    u = mda.Universe(strcfile)
    
    if residue_order == None:
        residue_order = np.sort(np.unique(u.residues.resnames))
    
    mdt.fh.backup(outfile)
    with mda.Writer(outfile) as W:
        for ts in u.trajectory:
            atomid_counter = 1
            resid_counter = 1
            u_new = 0
            for res in residue_order:
                sel = u.select_atoms("resname " + res)
                sel.atoms.ids = np.arange(atomid_counter,
                                          sel.n_atoms + atomid_counter)
                sel.residues.resids = np.arange(
                                          resid_counter,
                                          sel.n_residues + resid_counter)
                atomid_counter += sel.n_atoms
                resid_counter += sel.n_residues
                u_new += sel
            W.write(u_new)








if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                 description="Sort a structure file by residue names."
    )
    parser.add_argument(
        '-f',
        dest='STRCFILE',
        type=str,
        required=True,
        help="Structure file. Must contain residue names. See supported"
             " coordinate formats of MDAnalysis"
             " [<.trr/.xtc/.gro/.pdb/.xyz/.mol2/...>]."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename."
    )
    parser.add_argument(
        '-r',
        dest='RESIDUE_ORDER',
        nargs='*',
        type=str,
        required=False,
        default=None,
        help="Space separated list of residues. The outfile will contain"
             " the residues in this order. If no order is given,"
             " residues will be sorted alphabetically."
    )
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    sort_by_resnames(args.STRCFILE, args.OUTFILE, args.RESIDUE_ORDER)
    
    print()
    print("{} done".format(os.path.basename(sys.argv[0])), flush=True)
