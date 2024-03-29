.. _discretization-label:

Discretization
===============

Scripts that either discretize MD trajectories or that work with
discretized trajectories.

.. contents:: Site contents
    :depth: 2
    :local:


Scripts that create discretized trajectories
--------------------------------------------

.. currentmodule:: scripts/discretization

.. autosummary::
    :toctree: _sphinx_autosummary_discretization
    :template: custom-module-template.rst
    :recursive:
    :nosignatures:

    discrete_pos


Scripts that process discretized trajectories
---------------------------------------------

.. currentmodule:: scripts/discretization

.. autosummary::
    :toctree: _sphinx_autosummary_discretization
    :template: custom-module-template.rst
    :recursive:
    :nosignatures:

    back_jump_prob
    back_jump_prob_discrete
    kaplan_meier
    kaplan_meier_discrete
    state_lifetime
    state_lifetime_discrete


Plot scripts
------------

.. currentmodule:: scripts/discretization

.. autosummary::
    :toctree: _sphinx_autosummary_discretization
    :template: custom-module-template.rst
    :recursive:
    :nosignatures:

    plot_discrete_hex_trj
    plot_discrete_pos_trj
    plot_state_lifetime
    plot_state_lifetime_discrete
    plot_trans_per_state_vs_time
