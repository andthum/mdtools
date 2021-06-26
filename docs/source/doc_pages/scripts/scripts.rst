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

.. contents::
    :local:
    :depth: 1


Scripts by Application
======================

.. toctree::
    :name: scripts-toc
    :maxdepth: 1
    :titlesonly:
    :glob:
    
    discretization
    markov_modeling
    structure


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


.. _debug-mode-label:

Debug mode
==========

Many scripts and functions have a simple debug mode which triggers
additional checks of input parameters and consistency checks of
output values.  These checks can be computationally expensive, which
is the reason why they are not performed on default.  If you get
weird results or errors from a script, first try to run the script
in debug mode and see if warnings or errors are raised.  These might
help you to identify bad user input, parameter settings or bugs.
