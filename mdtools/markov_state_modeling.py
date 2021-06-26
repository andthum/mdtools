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


"""Markov State Modeling"""

import numpy as np
import pyemma




def merge_states(mm, s, n):
    """
    Merge `n` states of the active set of a Markov model starting from
    state `s`.
    
    Parameters
    ----------
    mm : pyemma.msm.MaximumLikelihoodMSM, pyemma.msm.BayesianMSM,
         pyemma.msm.markov_model
        Markov model whose states should be merged. The active set of
        the model must consist of a continuous sequence of states.
    s : int
        Index of the first state from which to start merging. This state
        will be merged with the following ``n-1`` states. Indexing
        starts at zero.
    n : int
        Number of states to merge.
    
    Returns
    -------
    msm : pyemma.msm.markov_model
        A new pyemma.msm.markov_model object with the new transition
        matrix. If ``n==1``, the input is returned.
    """
    
    if n == 1:
        return mm
    elif n < 1:
        raise ValueError("The number of states to merge must be one or"
                         " greater")
    elif n > mm.nstates - s:
        raise ValueError("The number of states to merge ({}) must be"
                         " smaller than the total number of active"
                         " states ({}) minus the state number from which"
                         " to start merging ({})"
                         .format(n, mm.nstates, s))
    elif n > mm.nstates - 1:
        raise ValueError("The number of states to merge ({}) must be"
                         " smaller than the total number of active"
                         " states ({}) minus one".format(n, mm.nstates))
    
    try:
        mm.active_set
    except AttributeError:
        pass
    else:
        if not np.all(np.diff(mm.active_set[s:s+n]) == 1):
            print(flush=True)
            print("Active states of the input model to merge:",
                  flush=True)
            print(mm.active_set[s:n+1], flush=True)
            raise ValueError("This function works only for Markov models"
                             " where the active set consists of a"
                             " continuous sequence of states")
    
    tm_merged = np.copy(mm.transition_matrix)
    # Sum columns to merge
    tm_merged[:,s] = np.sum(tm_merged[:,s:s+n], axis=1)
    tm_merged[:,s+1:-n+1] = tm_merged[:,s+n:]
    tm_merged = tm_merged[:,:-n+1]
    # Average rows to merge (the transition matrix is row-stochastic)
    tm_merged[s] = np.mean(tm_merged[s:s+n], axis=0)
    tm_merged[s+1:-n+1] = tm_merged[s+n:]
    tm_merged = tm_merged[:-n+1]
    
    if not np.allclose(np.sum(tm_merged, axis=1), 1):
        raise ValueError("Not all rows of the merged transition matrix"
                         " sum up to unity")
    
    return pyemma.msm.markov_model(P=tm_merged, dt_model=mm.dt_model)




def align_active_sets(mm1, mm2):
    """
    Align the active sets of two Markov models that differ only slightly
    (!) in the number of active states. That might e.g. be the case,
    when both models have been created in exactly the same way except
    that another lag time was used. Then a few edge (!) states might
    have been visited in one model but not in the other. This function
    does not work, when the different sizes of the active sets arise
    from not visited states in the middle of the sets.
    
    Parameters
    ----------
    mm1 : pyemma.msm.MaximumLikelihoodMSM, pyemma.msm.BayesianMSM
        Markov model 1. The active set of the model must consist of a
        continuous sequence of states.
    mm2 : pyemma.msm.MaximumLikelihoodMSM, pyemma.msm.BayesianMSM
        Markov model 2. The active set of the model must consist of a
        continuous sequence of states.
    
    Returns
    -------
    mms : list
        The two Markov models whose active sets are now the same.
    
    Note
    ----
    The alignment is done by reducing the active set of the larger model
    to that of the smaller one. The model which was changed (which might
    be both models) is returned as pyemma.msm.markov_model object. If a
    model is not changed, its return type is its input type. If the
    active sets of the two input models were already the same, the input
    models are returned.
    """
    
    mms = [mm1, mm2]
    msms = [mm1, mm2]
    
    if np.all(mm1.active_set == mm2.active_set):
        return mms
    
    for i in range(len(msms)):
        if not np.all(np.diff(msms[i].active_set) == 1):
            print(flush=True)
            print("Active set of states of Model {}:"
                  .format(i),
                  flush=True)
            print(msms[i].active_set, flush=True)
            raise ValueError("This function works only for Markov models"
                             " where the active sets consists of a"
                             " continuous sequence of states")
    
    tol = 0.04 * max(mm1.nstates, mm2.nstates)
    diff = abs(mm2.active_set[0] - mm1.active_set[0])
    if not np.isclose(diff, 0, atol=tol):
        raise ValueError("The differences between the active sets"
                         " of the models are too large")
    diff = abs(mm2.active_set[-1] - mm1.active_set[-1])
    if not np.isclose(diff, 0, atol=tol):
        raise ValueError("The differences between the active sets"
                         " of the models are too large")
    
    if mm1.active_set[0] != mm2.active_set[0]:
        small = np.argmax([mm.active_set[0] for mm in msms])
        large = np.argmin([mm.active_set[0] for mm in msms])
        if (msms[large].active_set[0] > msms[small].active_set[0]):
            raise ValueError("The first state of the larger model is"
                             " higher than the first state of the"
                             " smaller model. This should not have"
                             " happened")
        elif (msms[large].active_set[0] < msms[small].active_set[0]):
            n = msms[small].active_set[0]-msms[large].active_set[0] + 1
            mms[large] = merge_states(mm=msms[large], s=0, n=n)
    
    if mm1.active_set[-1] != mm2.active_set[-1]:
        small = np.argmin([mm.active_set[-1] for mm in msms])
        large = np.argmax([mm.active_set[-1] for mm in msms])
        if (msms[large].active_set[-1] < msms[small].active_set[-1]):
            raise ValueError("The last state of the larger model is"
                             " lower than the last state of the smaller"
                             " model. This should not have happened")
        elif (msms[large].active_set[-1] > msms[small].active_set[-1]):
            n = msms[large].active_set[-1]-msms[small].active_set[-1]+1
            mms[large] = merge_states(mm=mms[large],
                                      s=mms[large].nstates - n,
                                      n=n)
    
    if mms[0].nstates != mms[1].nstates:
        raise ValueError("The models have a different number of active"
                         " states. This should not have happened")
    
    return mms




def coarsen_model(mm, fine2coarse, bounds=None):
    """
    Coarsen a Markov model by consecutively merging `fine2coarse` fine
    states to one coarse state.
    
    Parameters
    ----------
    mm : pyemma.msm.MaximumLikelihoodMSM, pyemma.msm.BayesianMSM
        Markov model which should be coarsened ('the fine model'). The
        active set of the model must consist of a continuous sequence of
        states.
    fine2coarse : int
        Number of fine states that should be merged into one coarse
        state.
    bounds : array_like, optional
        First and last active state of the final coarsened model as list
        `[first, last]`. If the number of active states of the fine
        model is not a mutliple of `fine2coarse`, you can prepend and/or
        append the fine active set by 'virutal' states. If bounds is
        given, the fine active set is prepended till
        ``fine2coarse*bounds[0]`` and appended till
        ``fine2coarse*(bounds[1]+1) - 1``, because these are the states
        of the fine model corresponding to ``bounds[0]`` and
        ``bounds[1]`` of the coarse model, respectively. Note that
        ``fine2coarse*bounds[0]`` must be less than or equal to the first
        state of the fine active set (from here on called `f0`) and
        that ``f0 - fine2coarse*bounds[0]`` must be less than
        `fine2coarse`. The same holds in the other direction for
        ``bounds[1]``.
    
    Returns
    -------
    mm : pyemma.msm.markov_model
        The 'coarse' model. If ``fine2coarse==1``, the input is returned.
    """
    
    if fine2coarse == 1:
        return mm
    elif fine2coarse < 1:
        raise ValueError("The number of fine states to merge to one"
                         " coarse state must be one or greater")
    elif fine2coarse > mm.nstates:
        raise ValueError("The number of fine states to merge to one"
                         " coarse state must be smaller than the total"
                         " number of active states of the fine model")
    
    if not np.all(np.diff(mm.active_set) == 1):
        print(flush=True)
        print("Active set of states of the input model:", flush=True)
        print(mm.active_set, flush=True)
        raise ValueError("This function works only for Markov models"
                         " where the active set consists of a continuous"
                         " sequence of states")
    
    if bounds is None:
        bounds = np.zeros(2, dtype=int)
        bounds[0] = mm.active_set[0]
        bounds[1] = mm.active_set[-1]
    elif bounds[0] == bounds[1]:
        raise ValueError("The lower and upper bound must not be the same")
    elif bounds[0] > bounds[1]:
        upper = bounds[0]
        bounds[0] = bounds[1]
        bounds[1] = upper
    
    first_state = fine2coarse * bounds[0]
    last_state = fine2coarse * (bounds[1] + 1) - 1
    if (first_state > mm.active_set[0] or
        last_state < mm.active_set[-1]):
        raise ValueError("The boundaries of the active set of the coarse"
                         " model you gave lie within the active set of"
                         " the fine model, but must be the same or lie"
                         " outside")
    
    if (last_state-first_state+1) % fine2coarse != 0:
        raise ValueError("The number of fine states is not a multiple of"
                         " fine2coarse")
    n_coarse_states = int((last_state-first_state+1) / fine2coarse)
    
    prepend = mm.active_set[0] - first_state
    append = last_state - mm.active_set[-1]
    if prepend >= fine2coarse or append >= fine2coarse:
        raise ValueError("There are more states to be added to the model"
                         " than fine states to merge to one coarse"
                         " state. It seems that fine2coarse ({}) is"
                         " wrong or that the number of fine states is"
                         " not a multiple of the number of coarse"
                         " states".format(fine2coarse))
    if prepend < 0 or append < 0:
        raise ValueError("The number of states to be added is smaller"
                         " than zero. This should not have happened")
    
    tm = np.pad(mm.transition_matrix, (prepend, append))
    if prepend > 0:
        tm[:prepend] = np.mean(tm[prepend:fine2coarse], axis=0)
    if append > 0:
        tm[-append:] = np.mean(tm[-fine2coarse:-append], axis=0)
    
    tm_coarsened = np.zeros((n_coarse_states, n_coarse_states),
                            dtype=float)
    for k in range(n_coarse_states):
        for l in range(n_coarse_states):
            tm_coarsened[k][l] = np.sum(
                                     tm[k*fine2coarse:(k+1)*fine2coarse,
                                        l*fine2coarse:(l+1)*fine2coarse])
    tm_coarsened /= fine2coarse
    
    if not np.allclose(np.sum(tm_coarsened, axis=1), 1):
        raise ValueError("Not all rows of the coarsened transition"
                         " matrix sum up to unity")
    
    return pyemma.msm.markov_model(P=tm_coarsened, dt_model=mm.dt_model)




def match_active_sets(mm1, mm2, fine2coarse=None):
    """
    Match the active sets of two Markov models that were created by
    either using different lag times or different bin widths.
    
    Parameters
    ----------
    mm1 : pyemma.msm.MaximumLikelihoodMSM, pyemma.msm.BayesianMSM
        Markov model 1. The active set of the model must consist of a
        continuous sequence of states.
    mm2 : pyemma.msm.MaximumLikelihoodMSM, pyemma.msm.BayesianMSM
        Markov model 2. The active set of the model must consist of a
        continuous sequence of states.
    fine2coarse : int, optional
        Number of fine states that should be merged into one coarse
        state.
    
    Returns
    -------
    mms : list
        If the active sets of the two input models are already the same,
        simply return the input models as a list ``[mm1, mm2]``.
        Otherwise match the active sets of both models, i.e. reduce the
        active set of the larger model to the smaller one. The model
        which was changed (might be both models) is returned in the list
        as pyemma.msm.markov_model object.
    
    Note
    ----
    If `fine2coarse` is ``None``
    
    * and the difference of the size of the active sets is minor, this
      function calls :func:`align_active_sets`.
    * and the difference of the size of the active sets is large, this
      function calls :func:`coarsen_model`.
    
    So, if `fine2coarse` is ``None``, this function makes a guess based
    on the size of the active sets, whether the different active sets of
    the models arise from using a different lag time or a different
    discretization. If it arises from a different discretization,
    matching both active sets only makes sense, if for both models
    
    * the total discretized space region is the same,
    * the bins are equidistant and
    * the number of bins used for the 'fine' model is a multiple of the
      number of bins used for the 'coarse' model,
    
    so that there is always a mapping of n fine bins to one coarse bin.
    Simply speaking, the models use different bin widths. So carefully
    check your input and output.
    """
    
    if np.all(mm1.active_set == mm2.active_set):
        return [mm1, mm2]
    
    mms = [mm1, mm2]
    
    for i in range(len(mms)):
        if not np.all(np.diff(mms[i].active_set) == 1):
            print(flush=True)
            print("Active set of states of Model {}:"
                  .format(i),
                  flush=True)
            print(mms[i].active_set, flush=True)
            raise ValueError("This function works only for Markov models"
                             " where the active states consists of a"
                             " continuous sequence of states")
    
    
    tol = 0.04 * max(mm1.nstates, mm2.nstates)
    diff = abs(mm2.nstates - mm1.nstates)
    if np.isclose(diff, 0, atol=tol) and fine2coarse is None:
        mms = align_active_sets(mm1=mm1, mm2=mm2)
        
        
    else:
        if mm1.nstates == mm2.nstates:
            raise ValueError("The models have the same number of active"
                             " states, but the states are not the same."
                             " Trying to match the active sets of both"
                             " models by consecutively merging n states"
                             " of one of the both models makes no sense")
        
        coarse = np.argmin([mm.nstates for mm in mms])
        fine = np.argmax([mm.nstates for mm in mms])
        
        if fine2coarse is None:
            n_states_coarse = (mms[coarse].nstates +
                               mms[coarse].active_set[0])
            n_states_fine = (mms[fine].nstates +
                             mms[fine].active_set[0])
            if n_states_fine % n_states_coarse != 0:
                n_states_fine = (n_states_fine -
                                 n_states_fine % n_states_coarse +
                                 n_states_coarse)
            if n_states_fine >= 1.5*(mms[fine].nstates +
                                     mms[fine].active_set[0]):
                raise ValueError("The number of fine bins seems not to"
                                 " be a multiple of the number of coarse"
                                 " bins")
            fine2coarse = int(n_states_fine / n_states_coarse)
        
        if fine2coarse < 1:
            raise ValueError("The number of fine states to merge to one"
                             " coarse state must be one or greater")
        elif fine2coarse > mms[fine].nstates:
            raise ValueError("The number of fine states to merge to one"
                             " coarse state must be smaller than the"
                             " total number of active states of the fine"
                             " model")
        
        mms[fine] = coarsen_model(mm=mms[fine],
                                  fine2coarse=fine2coarse,
                                  bounds=(mms[coarse].active_set[0],
                                          mms[coarse].active_set[-1]))
    
    
    if mms[0].nstates != mms[1].nstates:
        raise ValueError("The models have a different number of active"
                         " states. This should not have happened")
    
    return mms




def match_lag_time(mms, lags=None):
    """
    Bring two or more Markov models to the same lag time by
    exponentiating their transition matrices by the lowest common
    multiple of the lag times of all models divided by the models own
    lag time.
    
    Parameters
    ----------
    mms : array_like
        One dimensional array of pyemma.msm.MaximumLikelihoodMSM,
        pyemma.msm.BayesianMSM, or pyemma.msm.markov_model objects.
    lags : array_like, optional
        Array of the same shape as `mms` containing the respective lag
        times of the models. Only needed when `mms` contains
        pyemma.msm.markov_model objects.
    
    Returns
    -------
    mms : array_like
        List of pyemma.msm.markov_model objects with transition matrices
        'propagated' to the same lag time.
    """
    
    if lags is None:
        lags = [None,] * len(mms)
    if len(mms) != len(lags):
        raise ValueError("mms and lags must have the same length")
    for i in range(len(mms)):
        try:
            lags[i] = mms[i].lag
        except AttributeError:
            if lags[i] is None:
                raise ValueError("Could not get the lag time of Model {}"
                                 " and this lag time is not given by the"
                                 " lags argument".format(i))
    
    lowest_common_lag = np.lcm.reduce(lags)
    
    for i in range(len(mms)):
        mms[i] = pyemma.msm.markov_model(
                     P=np.linalg.matrix_power(
                         mms[i].transition_matrix,
                         int(lowest_common_lag/lags[i])),
                     dt_model=mms[i].dt_model)
        if lowest_common_lag/lags[i] != int(lowest_common_lag/lags[i]):
            raise ValueError("The lowest common lag time is not a"
                             " multiple of the lag time of Model {}."
                             " This should not have happened".format(i))
        if not np.allclose(np.sum(mms[i].transition_matrix, axis=1), 1):
            raise ValueError("Not all rows of the exponentiated"
                             " transition matrix of Model {} sum up to"
                             " unity".format(i))
    
    return mms
