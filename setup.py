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
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith(varname):
                if '"' in line:
                    delimiter = '"'
                elif "'" in line:
                    delimiter = "'"
                else:
                    raise RuntimeError("The line starting with '{}' does"
                                       " not contain a string"
                                       .format(varname))
                return line.split(delimiter)[1]


def get_content(fname="README.rst"):
    """
    Read a file.
    
    Parameters
    ----------
    fname : str, optional
        Path to the file to read.
    
    Returns
    -------
    content : str
        The whole content of the file as one long string.
    """
    with open(fname, 'r') as f:
        return f.read()


if __name__ == '__main__':
    setuptools.setup(
        name=get_varstring("mdtools/_metadata.py", "__title__"),
        version=get_varstring("mdtools/version.py", "__version__"),
        author="Andreas Thum",
        author_email="andr.thum@gmail.com",
        maintainer=get_varstring("mdtools/_metadata.py", "__maintainer__"),
        maintainer_email=get_varstring("mdtools/_metadata.py", "__email__"),
        url="https://github.com/andthum/mdtools",
        project_urls={
            'Source': "https://github.com/andthum/mdtools",
            'Documentation': "https://mdtools.readthedocs.io/en/latest/",
            'Issue Tracker': "TODO",
            'Feature Requests': "TODO",
        },
        description="Python scripts to prepare and analyze molecular dynamics simulations",
        long_description=get_content("README.rst"),
        long_description_content_type='text/x-rst',
        license=get_varstring("mdtools/_metadata.py", "__license__"),
        classifiers=[
            "Programming Language :: Python :: 3 :: Only",
            "Operating System :: Unix",
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
            "Natural Language :: English",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Topic :: Utilities",
        ],
        keywords="molecular dynamics simulation analysis preparation chemistry physics materials science",
        packages=setuptools.find_packages(include=['mdtools']),
        include_package_data=True,
        python_requires=">=3.6, <4.0",
        install_requires=["psutil >=5.7, <6.0",
                          "numpy >=1.18, <2.0",
                          "scipy >=1.5, <2.0",
                          "matplotlib >=3.2, <3.3",
                          "MDAnalysis >=1.0, <2.0",
                          "pyemma >=2.5, <3.0"]
    )
