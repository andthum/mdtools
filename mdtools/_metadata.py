# This file is part of MDTools.
# Copyright (C) 2021, 2022  The MDTools Development Team and all
# contributors listed in the file AUTHORS.rst
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
Metadata of MDTools.

Note that the version string :data:`~mdtools.version.__version__` is
contained in :mod:`mdtools.version`.
"""


#: String containing the title of the project.
__title__ = "mdtools"

#: String containig the names of all contributors to MDTools, separated
#: by commas and ordered alphabetically by the contributors' full names.
#: Add your name, if you have contributed to MDTools (either to the core
#: package, the scripts or the documetation).  Additionally, add your
#: name in the authors list in AUTHORS.rst
__author__ = "Andreas Thum, Len Kimms"
# Remove possible dublicates and ensure alphabetical order
author_list = sorted(set(map(str.strip, __author__.split(","))), key=str.lower)
if __author__ != ", ".join(author_list):
    raise ValueError(
        "'__author__' contains dublicates or is not ordered alphabetically"
    )
__author__ = ", ".join(author_list)

#: String containig the name(s) of the maintainer(s) of this project,
#: separated by commas and ordered alphabetically by the maintainers'
#: full names.
__maintainer__ = "Andreas Thum"
# Remove possible dublicates and ensure alphabetical order
maintainer_list = sorted(
    set(map(str.strip, __maintainer__.split(","))), key=str.lower
)
if __maintainer__ != ", ".join(maintainer_list):
    raise ValueError(
        "'__maintainer__' contains dublicates or is not ordered alphabetically"
    )
__maintainer__ = ", ".join(maintainer_list)
del maintainer_list

#: String containing one(!) e-mail address under which the person(s)
#: responsible for the project can be contacted.
__email__ = "andr.thum@gmail.com"

#: Acknowledgments to people that advanced the project without writing
#: code or documentation and acknowledgments to other projects without
#: which this project would not exist.  Sort entries alphabetically!
__credits__ = "matplotlib, MDAnalysis, NumPy, Scipy"
# Remove possible dublicates and ensure alphabetical order
credits_list = sorted(
    set(map(str.strip, __credits__.split(","))), key=str.lower
)
if __credits__ != ", ".join(credits_list):
    raise ValueError(
        "'__credits__' contains dublicates or is not ordered alphabetically"
    )
__credits__ = ", ".join(credits_list)
# Make sure that no authors are mentioned in the credits:
for a in author_list:
    if a in credits_list:
        raise ValueError(
            "{} is mentioned in '__author__' and in '__credits__'".format(a)
        )
del author_list, credits_list

# from datetime import datetime
# now = datetime.now()
# years = "2021-{}, ".format(now)
years = "2021, 2022"
#: Copyright notice.
__copyright__ = "Copyright (C) " + years + " " + __author__

#: Copyright notice to print to standard output when running a program
#: that belongs to this project.
__copyright_notice__ = (
    "This program is part of MDTools\n"
    "Copyright (C) {}, The MDTools Development Team and all\n"
    "contributors listed in the file AUTHORS.rst\n"
    "\n"
    "This program is distributed in the hope that it will be useful,\n"
    "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
    "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
    "GNU General Public License for more details.\n"
    "\n"
    "This program is free software: you can redistribute it and/or\n"
    "modify it under the terms of the GNU General Public License as\n"
    "published by the Free Software Foundation, either version 3 of\n"
    "the License, or (at your option) any later version.".format(years)
)
del years

#: License.
__license__ = "GNU GPLv3+"
