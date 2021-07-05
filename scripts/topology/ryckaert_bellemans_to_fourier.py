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



# Convert Ryckaert-Bellemans dihedral coefficients (as used internally
# by Gromacs) to Fourier dihedral coefficients (as used by the OPLS-AA
# force field)
#
# For the conversion see
# M.J. Abraham, D. van der Spoel, E. Lindahl, B. Hess, and the GROMACS
# development team, GROMACS User Manual version 2018.8, www.gromacs.org
# (2019)


import argparse
import numpy as np

def calc_f_coeff(rb):
    if len(rb) != 6:
        raise ValueError("rb must have length 6")

    f = np.zeros(4)
    f[0] = -2 * rb[1] - 3.0/2 * rb[3]
    f[1] = -rb[2] - rb[4]
    f[2] = -0.5 * rb[3]
    f[3] = -0.25 * rb[4]

    tol = 1e-5
    if not np.isclose(rb[0], f[1]+0.5*(f[0]+f[2]), atol=tol):
        raise ValueError("rb[0] ({}) is not equal to f[1]+0.5*(f[0]+f[2])"
                         " ({})".format(rb[0], f[1]+0.5*(f[0]+f[2])))

    return f

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = (
            "Convert Ryckaert-Bellemans dihedral coefficients"
            "  to Fourier dihedral coefficients."
        )
    )
    parser.add_argument(
        '-c',
        dest='C',
        type=float,
        nargs=6,
        required=True,
        help="The six Ryckaert-Bellemans dihedral coefficients"
    )
    args = parser.parse_args()

    # The six Ryckaert-Bellemans dihedral coefficients from CLI (the last
    # one is actually not needed in the computations and the first one is
    # only needed for consistency checks)
    f = calc_f_coeff(args.C)

    print("The Fourier dihedral coefficients are")
    print("   ".join([str(i) for i in f]))
