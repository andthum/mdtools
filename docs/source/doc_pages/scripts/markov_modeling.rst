.. _markov-modeling-label:

Markov Modeling
===============

Scripts to estimate, validate and analyze Markov models.  Most of these
scripts are based on |PyEMMA|.

For constructing Markov models, the intended workflow is as follows (see
also the supplementary information of
`Prinz et al.`_, JCP, 2011, 134, 174105):
    
    1. **Simulation**: Simulate your system with equilibrium MD
       simulation.
    2. **Discretization**: Discretize the MD trajectory (see
       :ref:`discretization-label` for scripts that discretize MD
       trajectories).
    3. **Implied timescales**: Calculate the implied timescales of the
       discretized trajectory as function of lag time with
       :mod:`scripts.markov_modeling.pyemma_its`.  Choose the lag time
       at which the timescales of interest (usually the few largest
       timescales) converge as lag time for the estimation of the Markov
       model.  If the lag time at which these timescales converge is not
       significantly smaller than the timescales themselves, repeat step
       2 with another discretization.
    4. **Markov model**: Estimate the Markov model with
       :mod:`scripts.markov_modeling.pyemma_mm`.
    5. **Chapman-Kolmogorow test**: Check the Markovianity of the
       estimated model by conducting a Chapman-Kolmogorow test with
       :mod:`scripts.markov_modeling.pyemma_cktest`.
    6. **Analysis**: Analyze the Markov model.

.. todo::
    
    Change how the scripts read files containing the bin edges used for
    discretization.  (Background: The output of :mod:`discrete_pos` was
    changed and all the Markov modeling scripts were initially build
    upon the output of :mod:`discrete_pos`)

.. _Prinz et al.: https://doi.org/10.1063/1.3565032


Scripts
-------

.. todo::
    
    Revise and document scripts concerning Markov modeling.

.. todo::
    
    List scripts

.. currentmodule:: scripts/markov_modeling

.. autosummary::
    :toctree: _sphinx_autosummary_markov_modeling
    :template: custom-module-template.rst
    :recursive:
    :nosignatures:
    
    
