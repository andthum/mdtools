####
TODO
####

A simple todo list.  Should be replaced by GitHub's feature requests and
issue trackers.


Docstrings
==========

* Two spaces between two sentences
* Remove blank lines between the docstring ending and the first line of
  code
* Replace "orthorhombic" by "orthogonal" when talking about simulation
  boxes


Source code
===========

* Where possible (boolean arrays), replace ``numpy.sum`` by
  ``numpy.count_nonzero`` (is faster)
* Check the memory layout of numpy arrays that are used in loops
* Revise normalization codes for quantities that are looped over several
  restarting times ``t0`` and lag times

* Use flake8 to check the code quality

* Use pytest to test the code


Scripts
=======

* Write script docstrings for every script
* In each script give the name of the author of that script in the
  ``__authors__`` dunder
* Unify flags and help texts
* Use :mod:`argpars` `choices` keyword where applicable
* Revise fitting routines in all scripts
* Include re-fitting routines in plot scripts
* Replace ``n_frames`` by ``N_FRAMES`` and ``last_frame`` by
  ``LAST_FRAME`` (-> constant variables)
* Scripts to write:
    - Displacement correlations
    - Van Hove correlation function
    - Radial distribution function
