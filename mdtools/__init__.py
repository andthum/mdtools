# This file is part of MDTools.
# Copyright (C) 2021, The MDTools Development Team and all contributors
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


"""
MDTools is a collection of Python_ scripts to prepare and analyze
molecular dynamics (MD) simulations.

The actual :mod:`mdtools` package (in the following also called "core
package") is a collection of functions, classes and other Python objects,
which are needed to make the Python scripts shipped along with this
package work.  Of course, you can also import :mod:`mdtools` and use its
functions and objects in your own applications just like you do with any
other Python package.

The original idea behind this package was to gather functions that I
frequently used in my scripts for preparing and analyzing MD simulations.
When writing scripts, I often found myself copying and pasting functions
from one script to the other to have these functions also available in
my new script.  Then, when I have spotted a bug in one of these
functions, I had to change the function in every single script, which
becomes a boring and tedious task even if you have only a few scripts.
So, I outsourced all the functions I used in more than one script in a
separate python file and eventually this ended up in the :mod:`mdtools`
core package.

.. _Python: https://www.python.org/
"""


from mdtools.version import __version__
from mdtools._metadata import (
    __title__,
    __author__,
    __copyright__,
    __copyright_notice__,
)
from mdtools import box
from mdtools import check
from mdtools import dtrj
from mdtools import dynamics as dyn
from mdtools import file_handler as fh
from mdtools import functions as func
from mdtools import numpy_helper_functions as nph
from mdtools import parallel
from mdtools import plot
from mdtools import run_time_info as rti
from mdtools import scipy_helper_functions as sph
from mdtools import select
from mdtools import statistics as stats
from mdtools import structure as strc

__all__ = [
    "__version__",
    "__title__",
    "__author__",
    "__copyright__",
    "__copyright_notice__",
    "box",
    "check",
    "dtrj",
    "dyn",
    "fh",
    "func",
    "nph",
    "parallel",
    "plot",
    "rti",
    "sph",
    "select",
    "stats",
    "strc",
]
