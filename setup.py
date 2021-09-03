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


"""
Setuptools-based setup script for MDTools.

For a basic installation just type the command::

    python setup.py install

For more in-depth instructions, see the :ref:`installation-labe` section
in MDTools' documentation
"""


import setuptools


def get_varstring(fname="mdtools/version.py", varname="__version__"):
    """
    Read the string assigned to a specific variable in a given file.

    Parameters
    ----------
    fname : str, optional
        Path to the file from which to read the string assigned to a
        variable.
    varname : str, optional
        Name of the variable whose value should be read.  The value must
        be a simple string.

    Returns
    -------
    varstring : str
        The string assigned to `varname`.

    Notes
    -----
    If `varname` is assigned multiple times, only the first assignment
    will be considered.
    """
    with open(fname, "r") as f:
        for line in f:
            if line.startswith(varname):
                if '"' in line:
                    delimiter = '"'
                elif "'" in line:
                    delimiter = "'"
                else:
                    raise RuntimeError(
                        "The line starting with '{}' does"
                        " not contain a string".format(varname)
                    )
                return line.split(delimiter)[1]


def get_content(fname="README.rst"):
    """
    Read a text file.

    Parameters
    ----------
    fname : str, optional
        Path to the file to read.

    Returns
    -------
    content : str
        The whole content of the file as one long string.
    """
    with open(fname, "r") as f:
        return f.read()


if __name__ == "__main__":
    setuptools.setup(
        name=get_varstring("mdtools/_metadata.py", "__title__"),
        version=get_varstring("mdtools/version.py", "__version__"),
        author=get_varstring("mdtools/_metadata.py", "__author__"),
        maintainer=get_varstring("mdtools/_metadata.py", "__maintainer__"),
        maintainer_email=get_varstring("mdtools/_metadata.py", "__email__"),
        url="https://github.com/andthum/mdtools",
        download_url="https://github.com/andthum/mdtools",
        project_urls={
            "Source": "https://github.com/andthum/mdtools",
            "Documentation": "https://mdtools.readthedocs.io/en/latest/",
            "Issue Tracker": "https://github.com/andthum/mdtools/issues",
            "Feature Requests": "https://github.com/andthum/mdtools/issues",
            "Q&A": "https://github.com/andthum/mdtools/discussions/categories/q-a",  # noqa: E501
        },
        description=(
            "Python scripts to prepare and analyze molecular dynamics"
            " simulations"
        ),
        long_description=get_content("README.rst"),
        long_description_content_type="text/x-rst",
        license=get_varstring("mdtools/_metadata.py", "__license__"),
        license_files=("LICENSE.txt", "AUTHORS.rst"),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",  # noqa: E501
            "Natural Language :: English",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: Utilities",
        ],
        keywords=(
            "molecular-dynamics computational-science computational-chemistry"
            " materials-science"
        ),
        packages=setuptools.find_packages(include=["mdtools"]),
        include_package_data=True,
        python_requires=">=3.6, <3.9",
        install_requires=get_content("requirements.txt").splitlines(),
    )
