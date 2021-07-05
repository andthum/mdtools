#!/usr/bin/env python3

# This file is part of MDTools.
# Copyright (C) 2021  The MDTools Development Team and all contributors
# listed in the file AUTHORS.rst
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



# Convert Fourier dihedral coefficients (as used by the OPLS-AA force
# field) to Ryckaert-Bellemans dihedral coefficients (as used internally
# by Gromacs)
#
# For the conversion see
# M.J. Abraham, D. van der Spoel, E. Lindahl, B. Hess, and the GROMACS
# development team, GROMACS User Manual version 2018.8, www.gromacs.org
# (2019)


import argparse
import numpy as np

def calc_rb_coeff(f):
    if len(f) != 4:
        raise ValueError("f must have length 4")

    rb = np.zeros(6)
    rb[0] = f[1] + 0.5 * (f[0] + f[2])
    rb[1] = 0.5 * (-f[0] + 3 * f[2])
    rb[2] = -f[1] + 4 * f[3]
    rb[3] = -2 * f[2]
    rb[4] = -4 * f[3]
    rb[5] = 0

    return rb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Convert Fourier dihedral coefficients"
            "  to Ryckaert-Bellemans dihedral coefficients."
        )
    )
    parser.add_argument(
        '-v',
        dest='V',
        type=float,
        nargs=4,
        required=True,
        help="The four Fourier dihedral coefficients"
    )
    args = parser.parse_args()

    rb = calc_rb_coeff(args.V)

    print("The Ryckaert-Bellemans dihedral coefficients are")
    print("   ".join([str(i) for i in rb]))
