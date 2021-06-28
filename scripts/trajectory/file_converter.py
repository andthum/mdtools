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
import MDAnalysis as mda
import mdtools as mdt




def conv_strc_file(strcfile, outfile):
    """
    Convert structure files into each other. File conversion relies on
    the file extension.
    
    Paramters
    ---------
    strcfile : str
        Name of the structure file.
    outfile : str
        Name of the ouput file.
    """
    
    u = mda.Universe(strcfile)
    mdt.fh.backup(outfile)
    with mda.Writer(outfile) as W:
        for ts in u.trajectory:
            W.write(ts)








if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                 description=(
                     "Convert structure files into each other. File"
                     " conversion relies on the file extension."
                 )
    )
    parser.add_argument(
        '-f',
        dest='STRCFILE',
        type=str,
        required=True,
        help="Structure file [<.trr/.xtc/.gro/.pdb/.xyz/.mol2/...>]."
             " See supported coordinate formats of MDAnalysis."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename."
    )
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    conv_strc_file(args.STRCFILE, args.OUTFILE)
    
    print()
    print("{} done".format(os.path.basename(sys.argv[0])), flush=True)
