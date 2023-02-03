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

For more in-depth instructions, see the :ref:`installation-label`
section in MDTools' documentation.
"""

__author__ = "Andreas Thum"


# Standard libraries
import glob

# Third-party libraries
import setuptools


def get_varstring(fname, varname):
    """
    Read the string assigned to a specific variable in a given file.

    Parameters
    ----------
    fname : str
        Path to the file from which to read the string assigned to a
        variable.
    varname : str
        Name of the variable whose value should be read.  The value must
        be a simple string enquoted in either 'single' order "double"
        quotes.

    Returns
    -------
    varstring : str
        The string assigned to `varname`.

    Notes
    -----
    If `varname` is assigned multiple times, only the first assignment
    will be considered.
    """
    single_quote = "'"
    double_quote = '"'
    with open(fname, "r") as file:
        for line in file:
            if line.startswith(varname):
                if double_quote in line and single_quote in line:
                    ix_double_quote = line.find(double_quote)
                    ix_single_quote = line.find(single_quote)
                    if ix_single_quote < ix_double_quote:
                        delimiter = single_quote
                    else:
                        delimiter = double_quote
                elif single_quote in line:
                    delimiter = single_quote
                elif double_quote in line:
                    delimiter = double_quote
                else:
                    raise TypeError(
                        "The line starting with '{}' does not contain a"
                        " string".format(varname)
                    )
                if line[::-1].strip()[0] != delimiter:
                    raise TypeError(
                        "The line starting with '{}' does not contain a"
                        " string".format(varname)
                    )
                delimiter_count = line.count(delimiter)
                varstring = line.split(delimiter)[1:delimiter_count]
                return "".join(varstring)


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
    # Directory that contains all packages.
    pkg_dir = "src"
    # Path to the MDTools core package, relative to this file.
    mdt_dir = pkg_dir + "/mdtools"
    # Directory that contains all MDTools scripts.
    script_dir = "scripts"

    # Description of `setuptools.setup` keywords:
    # https://setuptools.pypa.io/en/latest/references/keywords.html
    setuptools.setup(
        name=get_varstring(mdt_dir + "/_metadata.py", "__title__"),
        version=get_varstring(mdt_dir + "/version.py", "__version__"),
        author=get_varstring(mdt_dir + "/_metadata.py", "__author__"),
        maintainer=get_varstring(mdt_dir + "/_metadata.py", "__maintainer__"),
        maintainer_email=get_varstring(mdt_dir + "/_metadata.py", "__email__"),
        url="https://github.com/andthum/mdtools",
        download_url="https://github.com/andthum/mdtools",
        project_urls={
            "Homepage": "https://github.com/andthum/mdtools",
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
        license=get_varstring(mdt_dir + "/_metadata.py", "__license__"),
        license_files=("LICENSE.txt", "AUTHORS.rst"),
        # PyPI classifiers (https://pypi.org/classifiers/).
        classifiers=[
            "Development Status :: 4 - Beta",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",  # noqa: E501
            "Natural Language :: English",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: Utilities",
        ],
        keywords=[
            "Scripts Collection",
            "Python Scripts",
            "Science",
            "Scientific Computing",
            "Computational Science",
            "Computational Biology",
            "Computational Chemistry",
            "Computational Physics",
            "Materials Science",
            "Molecular Simulation",
            "Molecular Modeling",
            "Molecular Mechanics",
            "Molecular Dynamics",
            "Molecular Dynamics Simulation",
            "Trajectory Analysis",
            "MDAnalysis",
            "NumPy",
            "Python3",
            "GPLv3",
        ],
        # Indicate that the relevant code is entirely contained inside
        # the given directory (instead of being directly placed under
        # the project’s root).  All the directories inside the container
        # directory will be copied directly into the final wheel
        # distribution, but the container directory itself will not.
        package_dir={"": pkg_dir},
        # Let setuptools find all packages in the package directory.
        # Use `setuptools.find_namespace_packages` instead of
        # `setuptools.find_packages` to let setuptools find data files
        # in subdirectories that don't contain an `__init__.py` file
        # (https://setuptools.pypa.io/en/latest/userguide/datafiles.html#subdirectory-for-data-files).
        packages=setuptools.find_namespace_packages(where=pkg_dir),
        # Include all data files inside the package directories that are
        # specified in the `MANIFEST.in` file.
        include_package_data=True,
        # A dictionary mapping package names to lists of glob patterns
        # that should be excluded from your distributed package
        # directories.
        exclude_package_data={"": [".git*"]},
        # A list of strings specifying standalone script files to be
        # built and installed.
        scripts=glob.glob(script_dir + "/**/*.py", recursive=True),
        # The Python version(s) supported by this project.
        python_requires=">=3.6, <3.10",
        # Python packages required by this project to run.  Dependencies
        # will be installed by pip when installing this project.
        install_requires=get_content("requirements.txt").splitlines(),
        # Additional groups of dependencies (e.g. development or test
        # dependencies).  Users will be able to install these using
        # pip's "extras" syntax: `pip install mdtools[dev]`
        extras_require={
            "dev": get_content("requirements-dev.txt").splitlines(),
        },
    )
