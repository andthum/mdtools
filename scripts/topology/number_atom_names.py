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




def number_atom_names_ignore_residues_and_atom_types(atoms):
    """
    Number atom names in a MDAnalysis atom group consecutively from the
    first to the last atom with no individual numbering for different
    residues or differnt atom types.

    Paramters
    ---------
    atoms : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis atom group whose atom names shall be altered to
        carry a atom number.
    """

    atoms.names = np.char.add(atoms.names.astype(str),
                              np.arange(1, atoms.n_atoms + 1).astype(str))




def number_atom_names_ignore_atom_type(atoms):
    """
    Number atom names in a MDAnalysis atom group consecutively with no
    individual numbering for different atom types. Different residues
    are numbered individually.

    Paramters
    ---------
    atoms : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis atom group whose atom names shall be altered to
        carry a atom number.
    """

    for res in atoms.residues:
        number_atom_names_ignore_residues_and_atom_types(res.atoms)




def number_atom_names_ignore_residues(atoms):
    """
    Number atom names in a MDAnalysis atom group consecutively with no
    individual numbering for different residues. Different atom types
    are numbered individually.

    Paramters
    ---------
    atoms : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis atom group whose atom names shall be altered to
        carry a atom number.
    """

    atom_numbers = np.zeros(len(atoms.names), dtype=int)
    atom_names, atom_counts = np.unique(atoms.names, return_counts=True)

    for i, atom_name in enumerate(atom_names):
        atom_numbers[atoms.names == atom_name] = np.arange(
            1,
            atom_counts[i] + 1)

    atoms.names = np.char.add(atoms.names.astype(str),
                              atom_numbers.astype(str))




def number_atom_names(atoms):
    """
    Number atom names in a MDAnalysis atom group consecutively. Each
    residue and atom type is numbered individually.

    Paramters
    ---------
    atoms : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis atom group whose atom names shall be altered to
        carry a atom number.
    """

    for res in atoms.residues:
        number_atom_names_ignore_residues(res.atoms)




def unnumber_atom_names(atoms):
    """
    Remove all digits from all atom names of a MDAnalysis atom group.

    Paramters
    ---------
    atoms : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis atom group whose atom names shall be relieved
        from any digits.
    """

    for atom in atoms:
        atom.name = ''.join(i for i in atom.name if not i.isdigit())








if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            "Number the atom names in a structure file"
            " consecutively. Numbering starts at 1."
        )
    )

    parser.add_argument(
        "-f",
        dest="STRCFILE",
        type=str,
        required=True,
        help="Structure file [.pdb/.xyz/.gro/...]. See supported"
             " coordinate formats of MDAnalysis. If your file contains"
             " no residue names, you must set --ignore-residues."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename."
    )

    parser.add_argument(
        '--ignore-residues',
        dest='IGNORE_RES',
        required=False,
        default=False,
        action="store_true",
        help="Numbering will not restart at each new residue."
    )
    parser.add_argument(
        '--ignore-atom-type',
        dest='IGNORE_ATOM_TYPE',
        required=False,
        default=False,
        action='store_true',
        help="Numbering will not restart at each new atom type."
    )
    parser.add_argument(
        '--unnumber',
        dest='UNNUMBER',
        required=False,
        default=False,
        action='store_true',
        help="Remove all digits from all atom names."
    )

    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())


    u = mda.Universe(args.STRCFILE)

    if (not args.IGNORE_RES and
        not args.IGNORE_ATOM_TYPE and
            not args.UNNUMBER):
        number_atom_names(u.atoms)
    elif (args.IGNORE_RES and
          args.IGNORE_ATOM_TYPE and
          not args.UNNUMBER):
        number_atom_names_ignore_residues_and_atom_types(u.atoms)
    elif (args.IGNORE_RES and
          not args.IGNORE_ATOM_TYPE and
          not args.UNNUMBER):
        number_atom_names_ignore_residues(u.atoms)
    elif (not args.IGNORE_RES and
          args.IGNORE_ATOM_TYPE and
          not args.UNNUMBER):
        number_atom_names_ignore_atom_type(u.atoms)
    elif (not args.IGNORE_RES and
          not args.IGNORE_ATOM_TYPE and
          args.UNNUMBER):
        unnumber_atom_names(u.atoms)
    else:
        raise RuntimeError("Your parsed arguments did not match any"
                           " condition.")

    mdt.fh.backup(args.OUTFILE)
    with mda.Writer(args.OUTFILE) as W:
        W.write(u)


    print()
    print("{} done".format(os.path.basename(sys.argv[0])), flush=True)
