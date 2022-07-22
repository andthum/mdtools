.. _scripts-label:

*******
Scripts
*******

The following pages document the **scripts** shipped along with the
MDTools Python package.  If you are interested in using the MDTools
**package**, read the documentation of the :ref:`core-package-label`
instead.

.. todo::

    * Write/finish docstring of all scripts.
    * Revise all scripts that are not already contained in this
      documentation.

.. contents:: Site contents
    :depth: 2
    :local:


Scripts by Application
======================

.. toctree::
    :name: scripts-toc
    :maxdepth: 1
    :titlesonly:
    :glob:

    discretization
    dynamics
    structure
    other


Scripts for Developers
======================

.. currentmodule:: scripts

.. autosummary::
    :toctree: _sphinx_autosummary
    :template: custom-module-template.rst
    :recursive:
    :nosignatures:

    script_template
    script_template_dtrj
    script_template_plot
