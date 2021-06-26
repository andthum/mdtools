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


# This file is read by setup.py.  Do not change its location or the
# variable names!  The first declaration of each variable must be a
# plain string, otherwise setup.py will fail to read its content!


"""
Version information for MDTools.

The version information in :mod:`mdtools.version` indicates the release
of MDTools.  The version number of your installation can be read from
``mdtools.__version___``.

MDTools uses `semantic versioning`_, which is compatible with :pep:`440`.
Given a version number MAJOR.MINOR.PATCH, we increment the

    1. **MAJOR** version when we make **incompatible API changes**,
    2. **MINOR** version when we **add functionality** in a
       **backwards-compatible** manner, and
    3. **PATCH** version when we make backwards-compatible **bug fixes**.

Additionally, pre-release, post-release and developmental release
specifiers can be appended in accordance to :pep:`440`.

.. note::

    As long as the **MAJOR** number is 0 (i.e. the API has not
    stabilized), even **MINOR** increases *may* introduce incompatible
    API changes.

.. _`semantic versioning`: http://semver.org/
"""


#: Release of MDTools as a string, using `semantic versioning`_.
__version__ = "0.0.0.dev0"
