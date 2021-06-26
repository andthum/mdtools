#!/usr/bin/env python3


# This file is part of MDTools.
# Copyright (C) 2020  Andreas Thum
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




import sys
import os
import warnings
from datetime import datetime
import psutil
import argparse
import numpy as np
import mdtools as mdt
from lifetime_autocorr_serial import unbound_compounds




# This function is also used by: lifetime_hist_parallel.py
def parse_user_input(add_description=""):
    description=("Compute the average lifetime of reference-selection"
                 " complexes, i.e. the mean residence time how long a"
                 " compound of the selection group is within the given"
                 " cutoff of a compound of the reference group."
                 " Additionally the lifetime of unbound reference and"
                 " selection compounds is calculated. This is done by"
                 " calculating the mean of the lifetime histogram. This"
                 " estimate is only meaningful, if the fraction of"
                 " counts for the last two lag times is almost zero,"
                 " since otherwise the lifetime might be larger than"
                 " these lag times and you might not have sampled the"
                 " correct lifetime distribution. However, if the"
                 " fraction of counts at the last lag time is zero and"
                 " the total number of counts is high, this estimate"
                 " should be quite accurate.")
    parser = argparse.ArgumentParser(
                 description=description+add_description
    )
    
    parser.add_argument(
        '-f',
        dest='TRJFILE',
        type=str,
        required=True,
        help="Trajectory file [<.trr/.xtc/.gro/.pdb/.xyz/.mol2/...>]."
             " See supported coordinate formats of MDAnalysis."
    )
    parser.add_argument(
        '-s',
        dest='TOPFILE',
        type=str,
        required=True,
        help="Topology file [<.top/.tpr/.gro/.pdb/.xyz/.mol2/...>]. See"
             " supported topology formats of MDAnalysis."
    )
    parser.add_argument(
        '-o',
        dest='OUTFILE',
        type=str,
        required=True,
        help="Output filename."
    )
    
    parser.add_argument(
        '--ref',
        dest='REF',
        type=str,
        nargs='+',
        required=True,
        help="Reference group. See MDAnalysis selection commands for"
             " possible choices. E.g. 'name Li'"
    )
    parser.add_argument(
        "--sel",
        dest="SEL",
        type=str,
        nargs="+",
        required=True,
        help="Selection group. See MDAnalysis selection commands for"
             " possible choices. E.g. 'type OE'"
    )
    parser.add_argument(
        '-c',
        dest='CUTOFF',
        type=float,
        required=True,
        help="Cutoff distance in Angstrom. A reference and selection"
             " atom are considered to be in contact, if their distance"
             " is less than or equal to this cutoff."
    )
    parser.add_argument(
        '--compound',
        dest='COMPOUND',
        type=str,
        required=False,
        default="atoms",
        help="Contacts between the reference and selection group can be"
             " computed either for individual 'atoms', 'segments',"
             " 'residues', or 'fragments'. Refer to the MDAnalysis user"
             " guide for the meaning of these terms"
             " (https://userguide.mdanalysis.org/1.0.0/groups_of_atoms.html)."
             " Default: 'atoms'"
    )
    parser.add_argument(
        '--min-contacts',
        dest='MINCONTACTS',
        type=int,
        required=False,
        default=1,
        help="Compounds of the reference and selection group are only"
             " considered to be in contact, if there are at least"
             " MINCONTACTS contacts between the atoms of the compounds."
             " --min-contacts is ignored if --compound is set to"
             " 'atoms'. Default: 1"
    )
    parser.add_argument(
        '--intermittency',
        dest='INTERMITTENCY',
        type=int,
        required=False,
        default=0,
        help="Maximum numer of frames a selection atom is allowed to"
             " leave the cutoff range of a reference atom whilst still"
             " being considered to be bound to the reference atom,"
             " provided that it is indeed bound again to the reference"
             " atom after this unbound period. The other way round, a"
             " selection atom is only considered to be bound to a"
             " reference atom, if it has been bound to it for at least"
             " this number of consecutive frames. Default: 0"
    )
    
    parser.add_argument(
        '-b',
        dest='BEGIN',
        type=int,
        required=False,
        default=0,
        help="First frame to read. Frame numbering starts at zero."
             " Default: 0"
    )
    parser.add_argument(
        "-e",
        dest="END",
        type=int,
        required=False,
        default=-1,
        help="Last frame to read (exclusive, i.e. the last frame read is"
             " actually END-1). Default: -1 (means read the very last"
             " frame of the trajectory)"
    )
    parser.add_argument(
        '--every',
        dest='EVERY',
        type=int,
        required=False,
        default=1,
        help="Read every n-th frame. Default: 1"
    )
    
    parser.add_argument(
        '--debug',
        dest='DEBUG',
        required=False,
        default=False,
        action='store_true',
        help="Run in debug mode."
    )
    
    args = parser.parse_args()
    print(mdt.rti.run_time_info_str())
    
    if args.CUTOFF <= 0:
        raise ValueError("-c must be greater than zero, but you gave {}"
                         .format(args.CUTOFF))
    if (args.COMPOUND != 'atoms' and
        args.COMPOUND != 'segments' and
        args.COMPOUND != 'residues' and
        args.COMPOUND != 'fragments'):
        raise ValueError("--compound must be either 'atoms', 'segments',"
                         " 'residues' or 'fragments', but you gave {}"
                         .format(args.COMPOUND))
    if args.MINCONTACTS < 1:
        raise ValueError("--min-contacts must be greater than zero, but"
                         " you gave {}"
                         .format(args.MINCONTACTS))
    if args.MINCONTACTS > 1 and args.COMPOUND == 'atoms':
        args.MINCONTACTS = 1
        print("\n\n\n", flush=True)
        print("Note: Setting --min-contacts to {}, because --compound\n"
              "  is set to 'atoms'".format(args.MINCONTACTS), flush=True)
    if args.INTERMITTENCY < 0:
        raise ValueError("--intermittency must be equal to or greater"
                         " than zero, but you gave {}"
                         .format(args.INTERMITTENCY))
    
    return args




def continuously_bound_at_least_n_frames(cms, debug=False):
    """
    Get all contacts between reference and selection compounds that
    continuously exist in all given contact matrices.
    
    Parameters
    ----------
    cms : array_like
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be either :class:`numpy.ndarray`s or
        scipy sparse matrices.
    debug : bool
        If ``True``, check the input arguments.
    
    Returns
    -------
    continuously_bound : numpy.ndarray or scipy.sparse.spmatrix
        2-dimenstional boolean array of the same shape as the contact
        matrices in `cms`. If reference compound ``i`` is continuously
        bound to selection compound ``j`` in all given contact matrices,
        element ``ij`` evaluates to ``True``.
    """
    
    if debug:
        for i, cm in enumerate(cms):
            if cm.ndim != 2:
                warnings.warn("cms seems not to be a list of contact"
                              " matrices, since its {}-th element has"
                              " not 2 dimensions".format(i),
                              RuntimeWarning)
            if cm.shape != cms[0].shape:
                raise ValueError("All arrays in cms must have the same"
                                 " shape")
            if type(cm) != type(cms[0]):
                raise TypeError("All arrays in cms must be of the same"
                                " type")
    
    if len(cms) == 1:
        return cms[0]
    
    continuously_bound = sum(cm.astype(np.uint8) for cm in cms)
    return continuously_bound == len(cms)




def continuously_bound_since_first_frame(cms, debug=False):
    """
    Get all contacts between reference and selection compounds that
    continuously exist in all given contact matrices except the first
    one, where they must not exist.
    
    Parameters
    ----------
    cms : array_like
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be scipy sparse matrices. `cms` must
        contain at least two contact matrices. Contacts will only be
        counted, when they continuously exist in the last ``len(cms)-1``
        contact matrices but not in the first contact matrix.
    debug : bool
        If ``True``, check the input arguments.
    
    Returns
    -------
    bound_since_n_frames : scipy.sparse.spmatrix
        2-dimenstional boolean array of the same shape as the contact
        matrices in `cms`. Actually the dtype of the array is
        ``numpy.uint8``, but it contains only zeros (``False``) and ones
        (``True``). If reference compound ``i`` is continuously bound to
        selection compound ``j`` in the last ``len(cms)-1`` contact
        matrices but unbound in the first contact matrix, element ``ij``
        evaluates to ``True``.
    """
    
    if debug:
        if len(cms) < 2:
            raise ValueError("cms must contain at least two contact"
                             " matrices")
        for i, cm in enumerate(cms):
            if cm.ndim != 2:
                warnings.warn("cms seems not to be a list of contact"
                              " matrices, since its {}-th element has"
                              " not 2 dimensions".format(i),
                              RuntimeWarning)
            if cm.shape != cms[0].shape:
                raise ValueError("All arrays in cms must have the same"
                                 " shape")
            if type(cm) != type(cms[0]):
                raise TypeError("All arrays in cms must be of the same"
                                " type")
    
    bound = continuously_bound_at_least_n_frames(cms[1:], debug=debug)
    bound_before = bound.multiply(cms[0])
    return bound.astype(np.uint8) - bound_before.astype(np.uint8)




def continuously_bound_till_last_frame(cms, debug=False):
    """
    Get all contacts between reference and selection compounds that
    continuously exist in all given contact matrices except the last
    one, where they must not exist.
    
    Parameters
    ----------
    cms : array_like
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be scipy sparse matrices. `cms` must
        contain at least two contact matrices. Contacts will only be
        counted, when they continuously exist in the first ``len(cms)-1``
        contact matrices but not in the last contact matrix.
    debug : bool
        If ``True``, check the input arguments.
    
    Returns
    -------
    bound_since_n_frames : scipy.sparse.spmatrix
        2-dimenstional boolean array of the same shape as the contact
        matrices in `cms`. Actually the dtype of the array is
        ``numpy.uint8``, but it contains only zeros (``False``) and ones
        (``True``). If reference compound ``i`` is continuously bound to
        selection compound ``j`` in the first ``len(cms)-1`` contact
        matrices but unbound in the last contact matrix, element ``ij``
        evaluates to ``True``.
    """
    
    if debug:
        if len(cms) < 2:
            raise ValueError("cms must contain at least two contact"
                             " matrices")
        for i, cm in enumerate(cms):
            if cm.ndim != 2:
                warnings.warn("cms seems not to be a list of contact"
                              " matrices, since its {}-th element has"
                              " not 2 dimensions".format(i),
                              RuntimeWarning)
            if cm.shape != cms[0].shape:
                raise ValueError("All arrays in cms must have the same"
                                 " shape")
            if type(cm) != type(cms[0]):
                raise TypeError("All arrays in cms must be of the same"
                                " type")
    
    bound = continuously_bound_at_least_n_frames(cms[:-1], debug=debug)
    bound_after = bound.multiply(cms[-1])
    return bound.astype(np.uint8) - bound_after.astype(np.uint8)




# This function is not used
def continuously_bound_exactly_n_frames(cms, debug=False):
    """
    Get all contacts between reference and selection compounds that
    continuously exist in all given contact matrices except the first
    and the last one, where they must not exist.
    
    Parameters
    ----------
    cms : array_like
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be scipy sparse matrices. `cms` must
        contain at least three contact matrices. Contacts will only be
        counted, when they continuously exist in the middle
        ``len(cms)-2`` contact matrices but not in the first and in the
        last contact matrix.
    debug : bool
        If ``True``, check the input arguments.
    
    Returns
    -------
    bound_exactly_n_frames : scipy.sparse.spmatrix
        2-dimenstional boolean array of the same shape as the contact
        matrices in `cms`. Actually the dtype of the array is
        ``numpy.uint8``, but it contains only zeros (``False``) and ones
        (``True``). If reference compound ``i`` is continuously bound to
        selection compound ``j`` in the middle ``len(cms)-2`` contact
        matrices but unbound in the first and in the last contact
        matrix, element ``ij`` evaluates to ``True``.
    """
    
    if debug:
        if len(cms) < 3:
            raise ValueError("cms must contain at least three contact"
                             " matrices")
        for i, cm in enumerate(cms):
            if cm.ndim != 2:
                warnings.warn("cms seems not to be a list of contact"
                              " matrices, since its {}-th element has"
                              " not 2 dimensions".format(i),
                              RuntimeWarning)
            if cm.shape != cms[0].shape:
                raise ValueError("All arrays in cms must have the same"
                                 " shape")
            if type(cm) != type(cms[0]):
                raise TypeError("All arrays in cms must be of the same"
                                " type")
    
    bound = continuously_bound_since_first_frame(cms[:-1], debug=debug)
    bound_after = bound.multiply(cms[-1])
    return bound.astype(np.uint8) - bound_after.astype(np.uint8)




def continuously_unbound_at_least_n_frames(cms, axis, debug=False):
    """
    Get all compounds of a reference (selection) group that are
    continuously not in contact with any atom of a selection (reference)
    group in all given contact matrices.
    
    Parameters
    ----------
    cms : array_like
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be either :class:`numpy.ndarray`s or
        scipy sparse matrices. It is assumed that each row stands for a
        reference compound and each column for a selection compound.
    axis : int, optional
        Must be either ``0`` or ``1``. If ``0``, unbound selection
        compounds will be returned (columns in which all elements are
        ``False``), if ``1`` unbound reference compounds will be
        returned (rows in which all elements are ``False``).
    debug : bool
        If ``True``, check the input arguments.
    
    Returns
    -------
    continuously_unbound : numpy.ndarray
        1-dimensional boolean array containing one element for each
        compound. If compound ``i`` is continuously unbound in all given
        contact matrices, the corresponding array element ``i``
        evaluates to ``True``.
    """
    
    if debug:
        for i, cm in enumerate(cms):
            if cm.ndim != 2:
                warnings.warn("cms seems not to be a list of contact"
                              " matrices, since its {}-th element has"
                              " not 2 dimensions".format(i),
                              RuntimeWarning)
            if axis >= cm.ndim:
                raise np.AxisError("axis {} is out of bounds for array"
                                   " of dimension {}"
                                   .format(axis, cm.ndim))
            if cm.shape[not axis] != cms[0].shape[not axis]:
                raise ValueError("All arrays in cms must have the shape"
                                 " along the axis that is kept")
    
    if len(cms) == 1:
        continuously_unbound = np.asarray(cms[0].sum(axis=axis))
        return np.squeeze(continuously_unbound) == 0
    
    continuously_unbound = sum(np.squeeze(np.asarray(cm.sum(axis=axis)))
                               for cm in cms)
    return continuously_unbound == 0




def continuously_unbound_since_first_frame(cms, axis, debug=False):
    """
    Get all compounds of a reference (selection) group that are
    continuously not in contact with any atom of a selection (reference)
    group in all given contact matrices except the first one, where a
    contact must exist.
    
    Parameters
    ----------
    cms : array_like
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be either :class:`numpy.ndarray`s or
        scipy sparse matrices. `cms` must contain at least two contact
        matrices. It is assumed that each row stands for a reference
        compound and each column for a selection compound. Compounds
        will only be counted, when they are continuously unbound in the
        last ``len(cms)-1`` contact matrices but bound in the first
        contact matrix.
    axis : int, optional
        Must be either ``0`` or ``1``. If ``0``, unbound selection
        compounds will be returned (columns in which all elements are
        ``False``), if ``1`` unbound reference compounds will be
        returned (rows in which all elements are ``False``).
    debug : bool
        If ``True``, check the input arguments.
    
    Returns
    -------
    unbound_since_n_frames : numpy.ndarray
        1-dimensional boolean array containing one element for each
        compound. If compound ``i`` is continuously unbound in the last
        ``len(cms)-1`` contact matrices but bound in the first contact
        matrix, the corresponding array element ``i`` evaluates to
        ``True``.
    """
    
    if debug:
        if len(cms) < 2:
            raise ValueError("cms must contain at least two contact"
                             " matrices")
        for i, cm in enumerate(cms):
            if cm.ndim != 2:
                warnings.warn("cms seems not to be a list of contact"
                              " matrices, since its {}-th element has"
                              " not 2 dimensions".format(i),
                              RuntimeWarning)
            if axis >= cm.ndim:
                raise np.AxisError("axis {} is out of bounds for array"
                                   " of dimension {}"
                                   .format(axis, cm.ndim))
            if cm.shape[not axis] != cms[0].shape[not axis]:
                raise ValueError("All arrays in cms must have the shape"
                                 " along the axis that is kept")
    
    unbound = continuously_unbound_at_least_n_frames(cms=cms[1:],
                                                     axis=axis,
                                                     debug=debug)
    unbound_before = unbound_compounds(contact_matrix=cms[0],
                                       axis=axis,
                                       debug=debug)
    unbound_before *= unbound
    return unbound ^ unbound_before




def continuously_unbound_till_last_frame(cms, axis, debug=False):
    """
    Get all compounds of a reference (selection) group that are
    continuously not in contact with any atom of a selection (reference)
    group in all given contact matrices except the last one, where a
    contact must exist.
    
    Parameters
    ----------
    cms : array_like
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be either :class:`numpy.ndarray`s or
        scipy sparse matrices. `cms` must contain at least two contact
        matrices. It is assumed that each row stands for a reference
        compound and each column for a selection compound. Compounds
        will only be counted, when they are continuously unbound in the
        first ``len(cms)-1`` contact matrices but bound in the last
        contact matrix.
    axis : int, optional
        Must be either ``0`` or ``1``. If ``0``, unbound selection
        compounds will be returned (columns in which all elements are
        ``False``), if ``1`` unbound reference compounds will be
        returned (rows in which all elements are ``False``).
    debug : bool
        If ``True``, check the input arguments.
    
    Returns
    -------
    unbound_since_n_frames : numpy.ndarray
        1-dimensional boolean array containing one element for each
        compound. If compound ``i`` is continuously unbound in the first
        ``len(cms)-1`` contact matrices but bound in the last contact
        matrix, the corresponding array element ``i`` evaluates to
        ``True``.
    """
    
    if debug:
        if len(cms) < 2:
            raise ValueError("cms must contain at least two contact"
                             " matrices")
        for i, cm in enumerate(cms):
            if cm.ndim != 2:
                warnings.warn("cms seems not to be a list of contact"
                              " matrices, since its {}-th element has"
                              " not 2 dimensions".format(i),
                              RuntimeWarning)
            if axis >= cm.ndim:
                raise np.AxisError("axis {} is out of bounds for array"
                                   " of dimension {}"
                                   .format(axis, cm.ndim))
            if cm.shape[not axis] != cms[0].shape[not axis]:
                raise ValueError("All arrays in cms must have the shape"
                                 " along the axis that is kept")
    
    unbound = continuously_unbound_at_least_n_frames(cms=cms[:-1],
                                                     axis=axis,
                                                     debug=debug)
    unbound_after = unbound_compounds(contact_matrix=cms[-1],
                                      axis=axis,
                                      debug=debug)
    unbound_after *= unbound
    return unbound ^ unbound_after




# This function is not used
def continuously_unbound_exactly_n_frames(cms, axis, debug=False):
    """
    Get all compounds of a reference (selection) group that are
    continuously not in contact with any atom of a selection (reference)
    group in all given contact matrices except the first and the last
    one, where a contact must exist.
    
    Parameters
    ----------
    cms : array_like
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be either :class:`numpy.ndarray`s or
        scipy sparse matrices. `cms` must contain at least three contact
        matrices. It is assumed that each row stands for a reference
        compound and each column for a selection compound. Compounds
        will only be counted, when they are continuously unbound in the
        middle ``len(cms)-2`` contact matrices but bound in the first
        and in the last contact matrix.
    axis : int, optional
        Must be either ``0`` or ``1``. If ``0``, unbound selection
        compounds will be returned (columns in which all elements are
        ``False``), if ``1`` unbound reference compounds will be
        returned (rows in which all elements are ``False``).
    debug : bool
        If ``True``, check the input arguments.
    
    Returns
    -------
    unbound_exactly_n_frames : numpy.ndarray
        1-dimensional boolean array containing one element for each
        compound. If compound ``i`` is continuously unbound in the
        middle ``len(cms)-2`` contact matrices but bound in the first
        and in the last contact matrix, the corresponding array element
        ``i`` evaluates to ``True``.
    """
    
    if debug:
        if len(cms) < 2:
            raise ValueError("cms must contain at least two contact"
                             " matrices")
        for i, cm in enumerate(cms):
            if cm.ndim != 2:
                warnings.warn("cms seems not to be a list of contact"
                              " matrices, since its {}-th element has"
                              " not 2 dimensions".format(i),
                              RuntimeWarning)
            if axis >= cm.ndim:
                raise np.AxisError("axis {} is out of bounds for array"
                                   " of dimension {}"
                                   .format(axis, cm.ndim))
            if cm.shape[not axis] != cms[0].shape[not axis]:
                raise ValueError("All arrays in cms must have the shape"
                                 " along the axis that is kept")
    
    unbound = continuously_unbound_since_first_frame(cms=cms[:-1],
                                                     axis=axis,
                                                     debug=debug)
    unbound_after = unbound_compounds(contact_matrix=cms[-1],
                                      axis=axis,
                                      debug=debug)
    unbound_after *= unbound
    return unbound ^ unbound_after




# This function is also used by: lifetime_hist_parallel.py
def lifetime_hist_bound(cms, debug=False):
    """
    Calculate the lifetime histogram for bound reference-selection
    complexes, i.e. count how many complexes continuously exist for a
    given number of frames.
    
    Parameters
    ----------
    cms : array_like
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be given as scipy sparse matrices.
        `cms` must contain at least three contact matrices.
    debug : bool
        If ``True``, check the input arguments.
    
    Returns
    -------
    hist_bound : numpy.ndarray
        Lifetime histogram. The first elements contains the number of
        reference-selection complexes that continuously exist for one
        frame, the second element contains the number of reference-
        selection complexes that continuously exist for two frames and
        so on. The last element contains the number of reference-
        selection complexes that continuously exist at least ``len(cms)``
        frames.
    """
    
    proc = psutil.Process(os.getpid())
    n_frames = len(cms)
    hist_bound = np.zeros(n_frames, dtype=np.uint32)
    
    
    timer = datetime.now()
    t0 = 0
    print("  Restart {:12d} of {:12d}"
          .format(t0, n_frames-1),
          flush=True)
    print("    Elapsed time:             {}"
          .format(datetime.now()-timer),
          flush=True)
    print("    Current memory usage: {:18.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    timer = datetime.now()
    
    bound_since_t0 = cms[t0]
    for lag in range(0, n_frames-1-t0):
        bound_since_t0 = continuously_bound_at_least_n_frames(
                             cms=[bound_since_t0]+[cms[t0+lag]],
                             debug=debug)
        if bound_since_t0.sum() == 0:
            break
        bound_lag_frames = continuously_bound_till_last_frame(
                               cms=[bound_since_t0]+[cms[t0+lag+1]],
                               debug=debug)
        hist_bound[lag] += bound_lag_frames.sum()
    else:
        lag = n_frames-1-t0
        bound_lag_frames = continuously_bound_at_least_n_frames(
                               cms=[bound_since_t0]+[cms[t0+lag]],
                               debug=debug)
        hist_bound[lag] += bound_lag_frames.sum()
    
    
    for t0 in range(1, n_frames):
        if t0 % 10**(len(str(t0))-1) == 0 or t0 == n_frames-1:
            print("  Restart {:12d} of {:12d}"
                  .format(t0, n_frames-1),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now()-timer),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss/2**20),
                  flush=True)
            timer = datetime.now()
        
        bound_since_t0 = cms[t0]
        for lag in range(0, n_frames-1-t0):
            bound_since_t0 = continuously_bound_since_first_frame(
                                 cms=[cms[t0-1]]+[bound_since_t0]+[cms[t0+lag]],
                                 debug=debug)
            if bound_since_t0.sum() == 0:
                break
            bound_lag_frames = continuously_bound_till_last_frame(
                                   cms=[bound_since_t0]+[cms[t0+lag+1]],
                                   debug=debug)
            hist_bound[lag] += bound_lag_frames.sum()
        else:
            lag = n_frames-1-t0
            bound_lag_frames = continuously_bound_since_first_frame(
                                   cms=[cms[t0-1]]+[bound_since_t0]+[cms[t0+lag]],
                                   debug=debug)
            hist_bound[lag] += bound_lag_frames.sum()
    
    
    return hist_bound




# This function is also used by: lifetime_hist_parallel.py
def lifetime_hist_unbound(cms, axis, debug=False):
    """
    Calculate the lifetime histogram for unbound reference (selection)
    compounds, i.e. count how many compounds are continuously unbound
    for a given number of frames.
    
    Parameters
    ----------
    cms : array_like
        List of boolean contact matrices as e.g. generated with
        :func:`mdtools.structure.contact_matrix`, one for each frame.
        The contact matrices must be either :class:`numpy.ndarray`s or
        scipy sparse matrices. `cms` must contain at least three contact
        matrices. It is assumed that each row stands for a reference
        compound and each column for a selection compound.
    axis : int, optional
        Must be either ``0`` or ``1``. If ``0``, unbound selection
        compounds will be returned (columns in which all elements are
        ``False``), if ``1`` unbound reference compounds will be
        returned (rows in which all elements are ``False``).
    debug : bool
        If ``True``, check the input arguments.
    
    Returns
    -------
    hist_unbound : numpy.ndarray
        Lifetime histogram. The first elements contains the number of
        reference (selection) compounds that are continuously unbound
        for one frame, the second element contains the number of
        reference (selection) compounds that are continuously unbound
        for two frames and so on. The last element contains the number
        of reference (selection) compounds that are continuously unbound
        for at least ``len(cms)`` frames.
    """
    
    proc = psutil.Process(os.getpid())
    n_frames = len(cms)
    hist_unbound = np.zeros(n_frames, dtype=np.uint32)
    
    
    timer = datetime.now()
    t0 = 0
    print("  Restart {:12d} of {:12d}"
          .format(t0, n_frames-1),
          flush=True)
    print("    Elapsed time:             {}"
          .format(datetime.now()-timer),
          flush=True)
    print("    Current memory usage: {:18.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    timer = datetime.now()
    
    unbound_since_t0 = unbound_compounds(contact_matrix=cms[t0],
                                         axis=axis,
                                         debug=debug)
    bound_since_t0 = np.invert(unbound_since_t0)
    bound_since_t0 = np.expand_dims(bound_since_t0, axis=axis)
    for lag in range(0, n_frames-1-t0):
        unbound_since_t0 = continuously_unbound_at_least_n_frames(
                               cms=[bound_since_t0]+[cms[t0+lag]],
                               axis=axis,
                               debug=debug)
        if unbound_since_t0.sum() == 0:
            break
        bound_since_t0 = np.invert(unbound_since_t0)
        bound_since_t0 = np.expand_dims(bound_since_t0, axis=axis)
        unbound_lag_frames = continuously_unbound_till_last_frame(
                                 cms=[bound_since_t0]+[cms[t0+lag+1]],
                                 axis=axis,
                                 debug=debug)
        hist_unbound[lag] += unbound_lag_frames.sum()
    else:
        lag = n_frames-1-t0
        unbound_lag_frames = continuously_unbound_at_least_n_frames(
                                 cms=[bound_since_t0]+[cms[t0+lag]],
                                 axis=axis,
                                 debug=debug)
        hist_unbound[lag] += unbound_lag_frames.sum()
    
    
    for t0 in range(1, n_frames):
        if t0 % 10**(len(str(t0))-1) == 0 or t0 == n_frames-1:
            print("  Restart {:12d} of {:12d}"
                  .format(t0, n_frames-1),
                  flush=True)
            print("    Elapsed time:             {}"
                  .format(datetime.now()-timer),
                  flush=True)
            print("    Current memory usage: {:18.2f} MiB"
                  .format(proc.memory_info().rss/2**20),
                  flush=True)
            timer = datetime.now()
        
        unbound_since_t0 = unbound_compounds(contact_matrix=cms[t0],
                                             axis=axis,
                                             debug=debug)
        bound_since_t0 = np.invert(unbound_since_t0)
        bound_since_t0 = np.expand_dims(bound_since_t0, axis=axis)
        for lag in range(0, n_frames-1-t0):
            unbound_since_t0 = continuously_unbound_since_first_frame(
                                   cms=[cms[t0-1]]+[bound_since_t0]+[cms[t0+lag]],
                                   axis=axis,
                                   debug=debug)
            if unbound_since_t0.sum() == 0:
                break
            bound_since_t0 = np.invert(unbound_since_t0)
            bound_since_t0 = np.expand_dims(bound_since_t0, axis=axis)
            unbound_lag_frames = continuously_unbound_till_last_frame(
                                     cms=[bound_since_t0]+[cms[t0+lag+1]],
                                     axis=axis,
                                     debug=debug)
            hist_unbound[lag] += unbound_lag_frames.sum()
        else:
            lag = n_frames-1-t0
            unbound_lag_frames = continuously_unbound_since_first_frame(
                                     cms=[cms[t0-1]]+[bound_since_t0]+[cms[t0+lag]],
                                     axis=axis,
                                     debug=debug)
            hist_unbound[lag] += unbound_lag_frames.sum()
    
    
    return hist_unbound








if __name__ == '__main__':
    
    timer_tot = datetime.now()
    proc = psutil.Process(os.getpid())
    
    
    args = parse_user_input()
    
    
    
    
    print("\n\n\n", flush=True)
    u = mdt.select.universe(top=args.TOPFILE,
                            trj=args.TRJFILE,
                            verbose=True)
    
    
    print("\n\n\n", flush=True)
    print("Creating selections", flush=True)
    timer = datetime.now()
    
    ref = u.select_atoms(' '.join(args.REF))
    sel = u.select_atoms(' '.join(args.SEL))
    print("  Reference group: '{}'"
          .format(' '.join(args.REF)),
          flush=True)
    print(mdt.rti.ag_info_str(ag=ref, indent=4))
    print(flush=True)
    print("  Selection group: '{}'"
          .format(' '.join(args.SEL)),
          flush=True)
    print(mdt.rti.ag_info_str(ag=sel, indent=4))
    
    if ref.n_atoms <= 0:
        raise ValueError("The reference atom group contains no atoms")
    if sel.n_atoms <= 0:
        raise ValueError("The selection atom group contains no atoms")
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    BEGIN, END, EVERY, n_frames = mdt.check.frame_slicing(
                                      start=args.BEGIN,
                                      stop=args.END,
                                      step=args.EVERY,
                                      n_frames_tot=u.trajectory.n_frames)
    last_frame = u.trajectory[END-1].frame
    if args.DEBUG:
        print("\n\n\n", flush=True)
        mdt.check.time_step(trj=u.trajectory[BEGIN:END], verbose=True)
    timestep = u.trajectory[BEGIN].dt
    
    
    
    
    print("\n\n\n", flush=True)
    print("Reading trajectory", flush=True)
    print("  Total number of frames in trajectory: {:>9d}"
          .format(u.trajectory.n_frames),
          flush=True)
    print("  Time step per frame:                  {:>9} (ps)\n"
          .format(u.trajectory[0].dt),
          flush=True)
    timer = datetime.now()
    
    cms = mdt.strc.contact_matrices(topfile=args.TOPFILE,
                                    trjfile=args.TRJFILE,
                                    ref=args.REF,
                                    sel=args.SEL,
                                    cutoff=args.CUTOFF,
                                    compound=args.COMPOUND,
                                    min_contacts=args.MINCONTACTS,
                                    begin=BEGIN,
                                    end=END,
                                    every=EVERY,
                                    verbose=True,
                                    debug=args.DEBUG)
    if len(cms) != n_frames:
        raise ValueError("The number of contact matrices does not equal"
                         " the number of frames to read. This should not"
                         " have happened")
    
    print(flush=True)
    print("Frames read: {}".format(n_frames), flush=True)
    print("First frame: {:>12d}    Last frame: {:>12d}    "
          "Every Nth frame: {:>12d}"
          .format(u.trajectory[BEGIN].frame, last_frame, EVERY),
          flush=True)
    print("Start time:  {:>12}    End time:   {:>12}    "
          "Every Nth time:  {:>12} (ps)"
          .format(u.trajectory[BEGIN].time,
                  u.trajectory[END-1].time,
                  u.trajectory[0].dt * EVERY),
          flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    if args.INTERMITTENCY > 0:
        print("\n\n\n", flush=True)
        print("Correcting for intermittency", flush=True)
        timer = datetime.now()
        
        cms = mdt.dyn.correct_intermittency(
                  list_of_arrays=cms,
                  intermittency=args.INTERMITTENCY,
                  verbose=True,
                  debug=args.DEBUG)
        
        print("Elapsed time:         {}"
              .format(datetime.now()-timer),
              flush=True)
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss/2**20),
              flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Calculating lifetime histograms", flush=True)
    timer = datetime.now()
    
    print(flush=True)
    print("  Bound reference-selection complexes", flush=True)
    hist_bound = lifetime_hist_bound(cms=cms, debug=args.DEBUG)
    print(flush=True)
    print("  Unbound reference compounds", flush=True)
    hist_unbound_ref = lifetime_hist_unbound(cms=cms,
                                             axis=1,
                                             debug=args.DEBUG)
    print(flush=True)
    print("  Unbound selection compounds", flush=True)
    hist_unbound_sel = lifetime_hist_unbound(cms=cms,
                                             axis=0,
                                             debug=args.DEBUG)
    del cms
    
    lag_times = np.arange(0, timestep*n_frames*EVERY,
                          timestep*EVERY,
                          dtype=np.float32)
    
    tot_counts_bound = np.sum(hist_bound)
    tot_counts_unbound_ref = np.sum(hist_unbound_ref)
    tot_counts_unbound_sel = np.sum(hist_unbound_sel)
    if tot_counts_bound < 0:
        raise ValueError("The total number of counts in the histogram"
                         " for bound reference-selection complexes is"
                         " less than zero. This should not have happened")
    if tot_counts_unbound_ref < 0:
        raise ValueError("The total number of counts in the histogram"
                         " for unbound reference compounds is less than"
                         " zero. This should not have happened")
    if tot_counts_unbound_sel < 0:
        raise ValueError("The total number of counts in the histogram"
                         " for unbound selection compounds is less than"
                         " zero. This should not have happened")
    
    hist_bound = hist_bound / tot_counts_bound
    hist_unbound_ref = hist_unbound_ref / tot_counts_unbound_ref
    hist_unbound_sel = hist_unbound_sel / tot_counts_unbound_sel
    
    mean_bound = np.sum(lag_times * hist_bound)
    mean_unbound_ref = np.sum(lag_times * hist_unbound_ref)
    mean_unbound_sel = np.sum(lag_times * hist_unbound_sel)
    
    sd_bound = np.sqrt(np.sum((lag_times-mean_bound)**2 * hist_bound))
    sd_unbound_ref = np.sqrt(np.sum((lag_times-mean_unbound_ref)**2 *
                                    hist_unbound_ref))
    sd_unbound_sel = np.sqrt(np.sum((lag_times-mean_unbound_sel)**2 *
                                    hist_unbound_sel))
    
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("Creating output", flush=True)
    timer = datetime.now()
    
    header = (
        "Average lifetime of reference-selection complexes and unbound\n"
        "reference and selection compounds\n"
        "\n"
        "\n"
        "Cutoff (Angstrom)     = {}\n"
        "Compound              = {}\n"
        "Minimum contacts      = {}\n"
        "Allowed intermittency = {}\n"
        "\n"
        "\n"
        "Reference: '{}'\n"
        "  Segments:               {}\n"
        "    Different segments:   {}\n"
        "    Segment name(s):      '{}'\n"
        "  Residues:               {}\n"
        "    Different residues:   {}\n"
        "    Residue name(s):      '{}'\n"
        "  Atoms:                  {}\n"
        "    Different atom names: {}\n"
        "    Atom name(s):         '{}'\n"
        "    Different atom types: {}\n"
        "    Atom type(s):         '{}'\n"
        "  Fragments:              {}\n"
        "\n"
        "Selection: '{}'\n"
        "  Segments:               {}\n"
        "    Different segments:   {}\n"
        "    Segment name(s):      '{}'\n"
        "  Residues:               {}\n"
        "    Different residues:   {}\n"
        "    Residue name(s):      '{}'\n"
        "  Atoms:                  {}\n"
        "    Different atom names: {}\n"
        "    Atom name(s):         '{}'\n"
        "    Different atom types: {}\n"
        "    Atom type(s):         '{}'\n"
        "  Fragments:              {}\n"
        "\n"
        "\n"
        "The average lifetime tau is estimated as mean of the lifetime\n"
        "histogram. This estimate is only meaningful, if the fraction\n"
        "of counts for the last two lag times is almost zero, since\n"
        "otherwise the lifetime might be larger than these lag times\n"
        "and you might not have sampled the correct lifetime\n"
        "distribution.\n"
        "\n"
        "\n"
        "The columns contain:\n"
        "   1 Lifetime tau (ps)\n"
        "   2 Lifetime histogram for bound reference-selection complexes\n"
        "   3 Lifetime histogram for unbound reference compounds\n"
        "   4 Lifetime histogram for unbound selection compounds\n"
        "\n"
        "Column number:\n"
        "{:>14d} {:>16d} {:>16d} {:>16d}\n"
        "\n"
        "Mean (ps):     {:>16.9e} {:>16.9e} {:>16.9e}\n"
        "Std. Dev. (ps):{:>16.9e} {:>16.9e} {:>16.9e}\n"
        "Tot. counts:   {:>16d} {:>16d} {:>16d}\n"
        .format(args.CUTOFF,
                args.COMPOUND,
                args.MINCONTACTS,
                args.INTERMITTENCY,
                
                ' '.join(args.REF),
                ref.n_segments,
                len(np.unique(ref.segids)),
                '\' \''.join(i for i in np.unique(ref.segids)),
                ref.n_residues,
                len(np.unique(ref.resnames)),
                '\' \''.join(i for i in np.unique(ref.resnames)),
                ref.n_atoms,
                len(np.unique(ref.names)),
                '\' \''.join(i for i in np.unique(ref.names)),
                len(np.unique(ref.types)),
                '\' \''.join(i for i in np.unique(ref.types)),
                len(ref.fragments),
                
                ' '.join(args.SEL),
                sel.n_segments,
                len(np.unique(sel.segids)),
                '\' \''.join(i for i in np.unique(sel.segids)),
                sel.n_residues,
                len(np.unique(sel.resnames)),
                '\' \''.join(i for i in np.unique(sel.resnames)),
                sel.n_atoms,
                len(np.unique(sel.names)),
                '\' \''.join(i for i in np.unique(sel.names)),
                len(np.unique(sel.types)),
                '\' \''.join(i for i in np.unique(sel.types)),
                len(sel.fragments),
                
                1, 2, 3, 4,
                
                mean_bound, mean_unbound_ref, mean_unbound_sel,
                sd_bound, sd_unbound_ref, sd_unbound_sel,
                tot_counts_bound,
                tot_counts_unbound_ref,
                tot_counts_unbound_sel
        )
    )
    
    mdt.fh.savetxt(fname=args.OUTFILE,
                   data=np.column_stack([lag_times,
                                         hist_bound,
                                         hist_unbound_ref,
                                         hist_unbound_sel]),
                   header=header)
    
    print("  Created {}".format(args.OUTFILE))
    print("Elapsed time:         {}"
          .format(datetime.now()-timer),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
    
    
    
    
    print("\n\n\n", flush=True)
    print("{} done".format(os.path.basename(sys.argv[0])), flush=True)
    print("Elapsed time:         {}"
          .format(datetime.now()-timer_tot),
          flush=True)
    print("Current memory usage: {:.2f} MiB"
          .format(proc.memory_info().rss/2**20),
          flush=True)
