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


# Setup instructions.  For more details about how to create a python
# package take a look at
# https://packaging.python.org/overview/
# and specifically at
# https://packaging.python.org/guides/distributing-packages-using-setuptools/
# and
# https://packaging.python.org/tutorials/packaging-projects/

# Setup instructions were moved to `pyproject.toml`.  This file is now
# basically empty and is only required for editable installs with pip
# versions <21.1.  See
# https://setuptools.pypa.io/en/latest/userguide/quickstart.html#development-mode


"""
Setuptools-based setup script for MDTools.

For a basic installation just type the command::

    python -m pip install .

For more in-depth instructions, see the :ref:`installation-label`
section in MDTools' documentation.
"""

__author__ = "Andreas Thum"


# Third-party libraries
import setuptools


if __name__ == "__main__":
    setuptools.setup()
