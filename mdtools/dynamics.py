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


"""
Functions to calculate dynamic properties.

This module can be called from :mod:`mdtools` via the shortcut ``dyn``::

    import mdtools as mdt
    mdt.dyn  # insetad of mdt.dynamics

"""


# Standard libraries
import os
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
# Third party libraries
import psutil
import numpy as np
from scipy import sparse
# Local application/library specific imports
import mdtools as mdt


def msd(t, D, d=3):
    r"""
    Mean square displacement (MSD) as fuction of time according to the
    Einstein relation:

    .. math::
        \langle \Delta\mathbf{r}^2(\tau) \rangle = 2d \cdot D \cdot \tau

    with
    :math:`\Delta\mathbf{r}(\tau) = |\mathbf{r}(t_0+\tau) - \mathbf{r}(t_0)|`
    being the displacement vector after a diffusion time :math:`\tau`,
    :math:`D` being the diffusion coefficient and :math:`d` being the
    number of spatial dimensions of the diffusion process.  The brackets
    :math:`\langle ... \rangle` denote averaging over all particles of
    the same species and over all possible starting times :math:`t_0`.

    Parameters
    ----------
    t : scalar or numpy.ndarray
        Array of diffusion times :math:`\tau` for which to calculate the
        MSD.
    D : scalar or numpy.ndarray
        Diffustion coefficient.
    d : scalar or numpy.ndarray, optional
        Number of spatial dimensions of the diffusion process.

    Returns
    -------
    msd : scalar or numpy.ndarray
        Mean square displacement MSD(t) = 2d*D*t.

    See Also
    --------
    :func:`mdtools.dynamics.msd_non_diffusive` :
        Calculate a non-diffusive MSD as function of time.

    Notes
    -----
    If more than one input argument is a :class:`numpy.ndarray`, the
    arrays must be broadcastable to a common shape.

    Examples
    --------
    >>> mdt.dyn.msd(t=np.arange(4), D=2, d=3)
    array([ 0, 12, 24, 36])
    """
    return 2 * d * D * t


def msd_non_diffusive(t, a, x):
    r"""
    Non-diffusive Mean Square Displacement (MSD) as fuction of time.

    .. math::
        \langle \Delta\mathbf{r}^2(\tau) \rangle = a \tau^x

    with
    :math:`\Delta\mathbf{r}(\tau) = |\mathbf{r}(t_0+\tau) - \mathbf{r}(t_0)|`
    being the displacement vector after a diffusion time :math:`\tau`.
    :math:`a` and :math:`x` are just parameters.  If :math:`x=1`, the
    equation describes diffusive motion.  In this case, :math:`a` is
    equal to :math:`2dD`, with :math:`D` being the diffusion coefficient
    and :math:`d` being the number of spatial dimensions of the
    diffusion process.  The brackets :math:`\langle ... \rangle` denote
    averaging over all particles of the same species and over all
    possible starting times :math:`t_0`.

    Parameters
    ----------
    t : scalar or numpy.ndarray
        Array of times for which to calculate the (non-diffusive) MSD.
    a : scalar or numpy.ndarray
        Prefactor.
    x : scalar or numpy.ndarray
        Exponent.

    Returns
    -------
    msd : scalar or numpy.ndarray
        (Non-diffusive) mean square displacement MSD(t) = a*t^x.

    See Also
    --------
    :func:`mdtools.dynamics.msd` :
        Calculate the diffusive MSD according to the Einstin relation as
        function of time

    Notes
    -----
    If more than one input argument is a :class:`numpy.ndarray`, the
    arrays must be broadcastable to a common shape.

    Examples
    --------
    >>> mdt.dyn.msd_non_diffusive(t=np.arange(4), a=2, x=3)
    array([ 0,  2, 16, 54])
    """
    return a * t**x


def n_restarts(n_frames, restart):
    """
    Get the number of restarting points for each lag time.

    Useful to normalize quantities that depend on a lag time and should
    be averaged over all possible restarting times.  The most prominent
    example for such a quantity is the Mean Square Displacement (MSD),
    another example is autocorrelation functions.

    Parameters
    ----------
    n_frames : int
        Number of frames in the trajectory.  Must be greater than zero.
    restart : int
        Number of frames between restarting points.  Must be greater
        than zero and should be less than `n_frames`.

    Returns
    -------
    n_restarts : numpy.ndarray
        Array of shape ``(n_frames,)`` and dtype :attr:`numpy.uint32`
        containing the number of restarting points for each lag time.

    Examples
    --------
    >>> n_frames = 5
    >>> for restart in range(1, n_frames+1):
    ...     mdt.dyn.n_restarts(n_frames, restart)
    array([5, 4, 3, 2, 1], dtype=uint32)
    array([3, 2, 2, 1, 1], dtype=uint32)
    array([2, 2, 1, 1, 1], dtype=uint32)
    array([2, 1, 1, 1, 1], dtype=uint32)
    array([1, 1, 1, 1, 1], dtype=uint32)
    >>> #  0  1  2  3  4  Corresponding lag times [in trajectory frames]

    :func:`mdtools.dynamics.n_restarts` is equivalent to but faster than
    the following nested loops

    >>> n_frames, restart = 5, 2
    >>> n_restarts = np.zeros(n_frames, dtype=np.uint32)
    >>> for t0 in range(0, n_frames, restart):
    ...     for lag in range(n_frames-t0):
    ...         n_restarts[lag] += 1
    >>> n_restarts
    array([3, 2, 2, 1, 1], dtype=uint32)
    >>> np.array_equal(n_restarts, mdt.dyn.n_restarts(n_frames, restart))
    True

    The nested loop from above can also be reduced to a single loop.
    But still, :func:`mdtools.dynamics.n_restarts` is faster.

    >>> n_frames, restart = 5, 2
    >>> n_restarts = np.zeros(n_frames, dtype=np.uint32)
    >>> for t0 in range(0, n_frames, restart):
    ...     n_restarts[:n_frames-t0] += 1
    >>> n_restarts
    array([3, 2, 2, 1, 1], dtype=uint32)
    >>> np.array_equal(n_restarts, mdt.dyn.n_restarts(n_frames, restart))
    True
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be greater than zero")
    if restart <= 0:
        raise ValueError("restart must be greater than zero")
    if restart > n_frames:
        warnings.warn("The number of frames between restarting points"
                      " ({}) is equal to or greater than the number of"
                      " frames in the trajectory ({})"
                      .format(restart, n_frames), RuntimeWarning)
    n_restarts = np.arange(n_frames, 0, -1, dtype=np.float32)
    n_restarts /= restart
    np.ceil(n_restarts, out=n_restarts)
    return n_restarts.astype(np.uint32, copy=False)


def correct_intermittency(
        list_of_arrays, intermittency, sparse2array=True, inplace=True,
        verbose=True, debug=False):
    r"""
    Correct for intermittent behavior of discrete variables stored in
    a sequence of arrays.

    Analogue of :func:`MDAnalysis.lib.correlations.correct_intermittency`
    for sequences of arbitrarily shaped arrays (e.g. a list of contact
    matrices as generated by :func:`mdtools.structure.contact_matrices`).

    Fill gaps between same values with a gap size smaller or equal to a
    given maximum gap size with the enclosing values.

    Parameters
    ----------
    list_of_arrays : tuple or list or numpy.ndarray
        Sequence of arrays (usually one array per trajectory frame).
        All arrays in the sequence must have the same shape and must be
        either :class:`NumPy arrays <numpy.ndarray>` or
        :mod:`SciPy sparse matrices <scipy.sparse>`.  A mix of
        :class:`NumPy arrays <numpy.ndarray>` and
        :mod:`SciPy sparse matrices <scipy.sparse>` is not possible.
    intermittency : int
        The maximum allowed gap size.  All gaps with a size smaller or
        equal to the maximum gap size are considered to be negligible
        and are filled with their enclosing values.  If `intermittency`
        is less than one, `list_of_arrays` will be returned without any
        changes.
    sparse2array : bool, optional
        If ``True``, :mod:`SciPy sparse matrices <scipy.sparse>` are
        internally converted on the fly to
        :class:`NumPy arrays <numpy.ndarray>` for the computations and
        back again to :mod:`SciPy sparse matrices <scipy.sparse>` using
        the sliding window method.  The window size is automatically
        determined by the intermittency value.  This conversion can
        increase the performance considerably, but may also lead to a
        higher memory consumption.  `sparse2array` is ignored, if
        `list_of_arrays` is already a sequence of
        :class:`NumPy arrays <numpy.ndarray>`.  The back conversion to
        :mod:`SciPy sparse matrices <scipy.sparse>` will always convert
        back to the format of the first sparse matrix in
        `list_of_arrays`.
    inplace : bool, optional
        If ``True``, modify `list_of_arrays` inplace instead of creating
        a copy of `list_of_arrays`.
    verbose : bool, optional
        If ``True``, print progress information to standard output.
    debug : bool, optional
        If ``True``, run in :ref:`debug mode <debug-mode-label>`.

        .. deprecated:: 0.0.0.dev0
            This argument is without use and will be removed in a future
            release.

    Returns
    -------
    list_of_arrays : tuple or list or numpy.ndarray
        The input sequence of arrays with gaps of size
        ``<=intermittency`` between same values filled with the
        enclosing values.

    See Also
    --------
    :func:`mdtools.dynamics.correct_intermittency_1d` :
        Correct for intermittent behavior of discrete variables stored
        in a 1-dimensional array
    :func:`mdtools.dynamics.replace_short_sequences` :
        Replace consecutive occurrences of the same value that are
        shorter than a given minimum length by a given value
    :func:`mdtools.dynamics.replace_short_sequences_global` : TODO

    Notes
    -----
    Possible use case of this function: Calculating lifetimes from the
    autocorrelation of contact matrices (autocorrelation of the
    existence function).  Lifetimes may be calculated either with a
    strict continuous requirement or a less strict intermittency.  If
    you are for instance calculating the lifetime of a ligand around a
    metal ion, in the former case the ligand must be within a cutoff
    distance of the metal ion at every frame from :math:`t_0` to
    :math:`t_0 + \tau` in order to be considered present at
    :math:`t_0 + \tau`.  In the intermittent case, the ligand is allowed
    to leave the region of interest for up to a specified consecutive
    number of frames whilst still being considered present at
    :math:`t_0 + \tau`.

    For example, if a ligand is absent for a number of frames equal to
    or smaller than the parameter `intermittency`, then this absence
    will be removed and thus the ligand is considered to have not left.
    For instance, T,F,F,T with `intermittency` set to ``2`` will be
    replaced by T,T,T,T, where T=True=present, F=False=absent.  On the
    other hand, F,T,T,F will be replaced by F,F,F,F.  Note the following
    behaviour when the elements of the arrays in `list_of_arrays` can
    take more than two different values:  0,1,1,0,1,1,2 will be replaced
    by 0,0,0,0,1,1,2 with `intermittency=2`, because the last 1,1 block
    is not enclosed by the same value.  If you want 0,1,1,0,1,1,2 to be
    replaced by 0,0,0,0,0,0,0, take a look at
    :func:`mdtools.dynamics.replace_short_sequences`.

    If you want to apply this function on a 1-dimensional array, parse
    the array as ``numpy.expand_dims(array, axis=0).T`` and convert the
    output back to a 1-dimensional array via ``np.squeeze(result.T)``.
    However, we recommend to use
    :func:`mdtools.dynamics.correct_intermittency_1d` instead, which is
    a fast version of :func:`mdtools.dynamics.correct_intermittency` for
    1-dimensional arrays.

    The `intermittency` value must not exceed 65534
    (``np.uint16(-1) - 1``).  If you really need a higher intermittency
    value, change the dtype of ``seen_nframes_ago`` in the source code
    of this function.  Be aware that high intermittency values increase
    the computational cost.  Ask yourself if you could not better
    read-in only every ``intermittency/2``-th frame and set
    `intermittency` to one or even read-in only every `intermittency`-th
    frame and set `intermittency` to zero.

    Examples
    --------
    >>> a = np.array([[0, 0, 0],
    ...               [1, 1, 1]])
    >>> b = np.array([[1, 1, 1],
    ...               [0, 0, 0]])
    >>> c = np.array([[1, 0, 1],
    ...               [0, 1, 0]])
    >>> d = np.array([[0, 1, 0],
    ...               [1, 0, 1]])
    >>> x = (a, b, c, d)
    >>> y = mdt.dyn.correct_intermittency(x,
    ...                                   intermittency=1,
    ...                                   inplace=False,
    ...                                   verbose=False)
    >>> for arr in y:
    ...     print(arr)
    [[0 0 0]
     [1 1 1]]
    [[1 0 1]
     [0 1 0]]
    [[1 0 1]
     [0 1 0]]
    [[0 1 0]
     [1 0 1]]
    >>> y = mdt.dyn.correct_intermittency(x,
    ...                                   intermittency=2,
    ...                                   inplace=False,
    ...                                   verbose=False)
    >>> for arr in y:
    ...     print(arr)
    [[0 0 0]
     [1 1 1]]
    [[0 0 0]
     [1 1 1]]
    [[0 0 0]
     [1 1 1]]
    [[0 1 0]
     [1 0 1]]

    An `intermittency` value larger than ``len(list_of_arrays)-2`` leads
    to the same result as an `intermittency` value of
    ``len(list_of_arrays)-2``.  The first and the last arrays are never
    changed:

    >>> y = mdt.dyn.correct_intermittency(x,
    ...                                   intermittency=len(x)-1,
    ...                                   inplace=False,
    ...                                   verbose=False)
    >>> for arr in y:
    ...     print(arr)
    [[0 0 0]
     [1 1 1]]
    [[0 0 0]
     [1 1 1]]
    [[0 0 0]
     [1 1 1]]
    [[0 1 0]
     [1 0 1]]
    >>> np.array_equal(y[0], a)
    True
    >>> np.array_equal(y[3], d)
    True

    If intermittency is less than one, the input `list_of_arrays` is
    returned without any changes:

    >>> y = mdt.dyn.correct_intermittency(x,
    ...                                   intermittency=0,
    ...                                   inplace=False,
    ...                                   verbose=False)
    >>> all(np.array_equal(x[i], arr) for i, arr in enumerate(y))
    True

    Using :mod:`SciPy sparse matrices <scipy.sparse>`:

    >>> from scipy.sparse import csr_matrix
    >>> x_sparse = [csr_matrix(arr) for arr in x]
    >>> y = mdt.dyn.correct_intermittency(x_sparse,
    ...                                   intermittency=2,
    ...                                   sparse2array=False,
    ...                                   inplace=False,
    ...                                   verbose=False)
    >>> for csr_mat in y:
    ...     print(csr_mat.toarray())
    [[0 0 0]
     [1 1 1]]
    [[0 0 0]
     [1 1 1]]
    [[0 0 0]
     [1 1 1]]
    [[0 1 0]
     [1 0 1]]
    >>> y = mdt.dyn.correct_intermittency(x_sparse,
    ...                                   intermittency=2,
    ...                                   sparse2array=True,
    ...                                   inplace=False,
    ...                                   verbose=False)
    >>> for csr_mat in y:
    ...     print(csr_mat.toarray())
    [[0 0 0]
     [1 1 1]]
    [[0 0 0]
     [1 1 1]]
    [[0 0 0]
     [1 1 1]]
    [[0 1 0]
     [1 0 1]]

    Inplace modification:

    >>> y = mdt.dyn.correct_intermittency(x,
    ...                                   intermittency=2,
    ...                                   inplace=True,
    ...                                   verbose=False)
    >>> x is y
    True
    """
    if verbose:
        timer = datetime.now()
        proc = psutil.Process(os.getpid())
        proc.cpu_percent()  # Initiate monitoring of CPU usage
        print("Running mdtools.dynamics.correct_intermittency()...")
    if intermittency > np.uint16(-1) - 1:
        raise ValueError("intermittency must be less than {}, which is"
                         " np.uint16(-1)-1. If you really need a higher"
                         " intermittency value, change the dtype of"
                         " 'seen_nframes_ago' in the source code of this"
                         " function".format(np.uint16(-1) - 1))
    mdt.check.list_of_cms(list_of_arrays)
    if not inplace:
        list_of_arrays = deepcopy(list_of_arrays)
    if intermittency <= 0:
        if verbose:
            print("mdtools.dynamics.correct_intermittency() done")
            print("Elapsed time: {}".format(datetime.now() - timer))
            print("CPU time:             {}"
                  .format(timedelta(seconds=sum(proc.cpu_times()[:4]))))
            print("CPU usage:            {:.2f} %"
                  .format(proc.cpu_percent()))
            print("Current memory usage: {:.2f} MiB"
                  .format(proc.memory_info().rss / 2**20))
        return list_of_arrays
    if sparse.issparse(list_of_arrays[0]):
        issparse = True
    else:
        issparse = False
        sparse2array = False
    if issparse and sparse2array:
        sparse_format = list_of_arrays[0].format
        stop = min(intermittency + 2, len(list_of_arrays))
        for i in range(0, stop - 1):
            list_of_arrays[i] = list_of_arrays[i].toarray()
    if verbose:
        print("Arrays to process:   {}".format(len(list_of_arrays)))
        print("Shape of the arrays: {}".format(list_of_arrays[0].shape))
        loa = mdt.rti.ProgressBar(list_of_arrays, unit="arrays")
    else:
        loa = list_of_arrays
    for i, a in enumerate(loa):
        ix_max = i + intermittency + 1
        if sparse2array and ix_max < len(list_of_arrays):
            list_of_arrays[ix_max] = list_of_arrays[ix_max].toarray()
        seen_nframes_ago = np.zeros(a.shape, dtype=np.uint16)
        stop = min(intermittency + 2, len(list_of_arrays) - i)
        for j in range(1, stop):
            mask = (list_of_arrays[i + j] != a)
            if issparse and not sparse2array:
                mask = mask.toarray()
            seen_nframes_ago += mask
            seen_nframes_ago[~mask] = 0
        for j in range(1, stop - 1):
            mask = (seen_nframes_ago <= stop - 2 - j)
            list_of_arrays[i + j][mask] = a[mask]
        if sparse2array:
            if sparse_format == 'bsr':
                list_of_arrays[i] = sparse.bsr_matrix(list_of_arrays[i])
            elif sparse_format == 'coo':
                list_of_arrays[i] = sparse.coo_matrix(list_of_arrays[i])
            elif sparse_format == 'csc':
                list_of_arrays[i] = sparse.csc_matrix(list_of_arrays[i])
            elif sparse_format == 'csr':
                list_of_arrays[i] = sparse.csr_matrix(list_of_arrays[i])
            elif sparse_format == 'dia':
                list_of_arrays[i] = sparse.dia_matrix(list_of_arrays[i])
            elif sparse_format == 'dok':
                list_of_arrays[i] = sparse.dok_matrix(list_of_arrays[i])
            elif sparse_format == 'lil':
                list_of_arrays[i] = sparse.lil_matrix(list_of_arrays[i])
            else:
                raise ValueError("Unknown scipy sparse matrix format"
                                 " ({}). Allowed formats: bsr, coo, csc,"
                                 " csr, dia, dok, lil"
                                 .format(sparse_format))
        if verbose:
            progress_bar_mem = proc.memory_info().rss / 2**20
            loa.set_postfix_str("{:>7.2f}MiB".format(progress_bar_mem),
                                refresh=False)
    if verbose:
        loa.close()
        print("mdtools.dynamics.correct_intermittency() done")
        print("Elapsed time: {}".format(datetime.now() - timer))
        print("CPU time:             {}"
              .format(timedelta(seconds=sum(proc.cpu_times()[:4]))))
        print("CPU usage:            {:.2f} %"
              .format(proc.cpu_percent()))
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss / 2**20))
    return list_of_arrays


def correct_intermittency_1d(a, intermittency, inplace=True, debug=False):
    """
    Correct for intermittent behavior of discrete variables stored in a
    1-dimensional array.

    Fill gaps between same values with a gap size smaller or equal to a
    given maximum gap size with the enclosing values.  For instance,
    0,1,1,0,1,1,2 is replaced by 0,0,0,0,1,1,2, when `intermittency` is
    set to ``2``.

    Parameters
    ----------
    a : array_like
        1-dimensional array, in which to fill gaps between same values
        with the enclosing values.  For higher dimensional arrays use
        :func:`mdtools.dynamics.correct_intermittency`.
    intermittency : int
        The maximum allowed gap size.  All gaps with a size smaller or
        equal to the maximum gap size are considered to be negligible
        and are filled with their enclosing values.  If `intermittency`
        is less than one, `a` will be returned without any changes.
    inplace : bool, optional
        If ``True`` (default), modify `a` inplace instead of creating a
        copy of `a`.  Works only, if `a` is a :class:`numpy.ndarray`.
    debug : bool, optional
        If ``True``, check the input arguments.

        .. deprecated:: 0.0.0.dev0
            This argument is without use and will be removed in a future
            release.

    Returns
    -------
    a : numpy.ndarray
        The input array as :class:`numpy.ndarray` with gaps of size
        ``<=intermittency`` between same values filled with the
        enclosing values.

    See Also
    --------
    :func:`mdtools.dynamics.correct_intermittency` :
        Correct for intermittent behavior of discrete variables stored
        in a sequence of arrays
    :func:`mdtools.dynamics.replace_short_sequences` :
        Replace consecutive occurrences of the same value that are
        shorter than a given minimum length by a given value
    :func:`mdtools.dynamics.replace_short_sequences_global` : TODO

    Notes
    -----
    This is a faster version of
    :func:`mdtools.dynamics.correct_intermittency` for 1-dimensional
    arrays.  See there for more details.

    Examples
    --------
    >>> a = np.array([1, 2, 2, 1, 3, 3, 3, 1, 3, 3, 3])
    >>> mdt.dyn.correct_intermittency_1d(a,
    ...                                  intermittency=1,
    ...                                  inplace=False)
    array([1, 2, 2, 1, 3, 3, 3, 3, 3, 3, 3])
    >>> mdt.dyn.correct_intermittency_1d(a,
    ...                                  intermittency=2,
    ...                                  inplace=False)
    array([1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3])
    >>> mdt.dyn.correct_intermittency_1d(a,
    ...                                  intermittency=3,
    ...                                  inplace=False)
    array([1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3])

    An `intermittency` value larger than ``len(a)-2`` leads to the same
    result as an `intermittency` value of ``len(a)-2``.  The first and
    the last elements are never changed:

    >>> mdt.dyn.correct_intermittency_1d(a,
    ...                                  intermittency=len(a)-2,
    ...                                  inplace=False)
    array([1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3])
    >>> mdt.dyn.correct_intermittency_1d(a,
    ...                                  intermittency=len(a)-1,
    ...                                  inplace=False)
    array([1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3])

    If intermittency is less than one, the input array is returned as
    :class:`numpy.ndarray`:

    >>> mdt.dyn.correct_intermittency_1d(a,
    ...                                  intermittency=0,
    ...                                  inplace=False)
    array([1, 2, 2, 1, 3, 3, 3, 1, 3, 3, 3])

    Inplace modification:

    >>> b = mdt.dyn.correct_intermittency_1d(a,
    ...                                      intermittency=3,
    ...                                      inplace=True)
    >>> a is b
    True
    """
    a = np.array(a, copy=(not inplace))
    if a.ndim > 1:
        raise ValueError("The input array must be 1-dimensional")
    if intermittency <= 0:
        return a
    for i, init_val in enumerate(a):
        stop = min(intermittency + 2, len(a) - i)
        for j, final_val in enumerate(a[i + 2:i + stop][::-1]):
            if final_val == init_val:
                a[i + 1:i + stop - 1 - j] = init_val
                break
    return a


def replace_short_sequences(
        list_of_arrays, min_len, val=0, inplace=True, verbose=False,
        debug=False):
    """
    Replace consecutive occurrences of the same value that are shorter
    than a given minimum length by a given value.

    For instance, the sequence 1,3,3,3,2,2,1 is replaced by
    0,3,3,3,0,0,0, when `min_len` is set to ``3`` and `val` is set to
    ``0``.

    Parameters
    ----------
    list_of_arrays : tuple or list or numpy.ndarray
        Sequence of arrays (usually one array per trajectory frame).
        All arrays in the sequence must have the same shape and all
        arrays must be :class:`NumPy arrays <numpy.ndarray>`.
    min_len : int
        All consecutive occurrences **shorter** than `min_len` will be
        replaced by `val`.  If `min_len` is less than two,
        `list_of_arrays` will be returned without any changes.
    val : scalar, optional
        The value used to replace the values in consecutive occurrences
        shorter than `min_len`.  Be aware that the arrays in
        `list_of_arrays` must have a suitable dtype to be able to hold
        `val`.
    inplace : bool, optional
        If ``True``, modify `list_of_arrays` inplace instead of creating
        a copy of `list_of_arrays`.
    verbose : bool, optional
        If ``True``, print progress information to standard output.
    debug : bool, optional
        If ``True``, check the input arguments.

        .. deprecated:: 0.0.0.dev0
            This argument is without use and will be removed in a future
            release.

    Returns
    -------
    list_of_arrays : tuple or list or numpy.ndarray
        List of arrays with consecutive occurrences shorter than
        `min_len` replaced by `val`.

    See Also
    --------
    :func:`mdtools.dynamics.correct_intermittency` :
        Correct for intermittent behavior of discrete variables stored
        in a sequence of arrays
    :func:`mdtools.dynamics.correct_intermittency_1d` :
        Correct for intermittent behavior of discrete variables stored
        in a 1-dimensional array
    :func:`mdtools.dynamics.replace_short_sequences_global` : TODO

    Notes
    -----
    If you want to replace consecutive occurrences in a 1-dimensional
    array, parse the array as ``np.expand_dims(array, axis=0).T`` and
    convert the output back to a 1-dimensional array via
    ``np.squeeze(result.T)``.

    `min_len` must not exceed 65535 (``np.uint16(-1)``). If you really
    need a higher value for `min_len`, change the dtype of
    ``seen_nframes_ago`` and ``seen_nframes_ahead`` in the source code
    of this function.  Be aware that high values of `min_len` increase
    the computational cost.

    Examples
    --------
    >>> a = np.array([[3, 1, 2],
    ...               [1, 2, 3]])
    >>> b = np.array([[3, 2, 2],
    ...               [2, 2, 3]])
    >>> c = np.array([[3, 2, 1],
    ...               [2, 1, 3]])
    >>> x = (a, b, c)
    >>> y = mdt.dyn.replace_short_sequences(x,
    ...                                     min_len=2,
    ...                                     val=0,
    ...                                     inplace=False,
    ...                                     verbose=False)
    >>> for arr in y:
    ...     print(arr)
    [[3 0 2]
     [0 2 3]]
    [[3 2 2]
     [2 2 3]]
    [[3 2 0]
     [2 0 3]]
    >>> y = mdt.dyn.replace_short_sequences(x,
    ...                                     min_len=3,
    ...                                     val=0,
    ...                                     inplace=False,
    ...                                     verbose=False)
    >>> for arr in y:
    ...     print(arr)
    [[3 0 0]
     [0 0 3]]
    [[3 0 0]
     [0 0 3]]
    [[3 0 0]
     [0 0 3]]

    If `min_len` is greater than ``len(list_of_arrays)``, all elements
    of all arrays are replaced by `val`:

    >>> y = mdt.dyn.replace_short_sequences(x,
    ...                                     min_len=len(x)+1,
    ...                                     val=0,
    ...                                     inplace=False,
    ...                                     verbose=False)
    >>> for arr in y:
    ...     print(arr)
    [[0 0 0]
     [0 0 0]]
    [[0 0 0]
     [0 0 0]]
    [[0 0 0]
     [0 0 0]]

    If `min_len` is less than two, the input `list_of_arrays` is
    returned without any changes:

    >>> y = mdt.dyn.replace_short_sequences(x,
    ...                                     min_len=1,
    ...                                     val=0,
    ...                                     inplace=False,
    ...                                     verbose=False)
    >>> all(np.array_equal(x[i], arr) for i, arr in enumerate(y))
    True

    Inplace modification:

    >>> y = mdt.dyn.replace_short_sequences(x,
    ...                                     min_len=3,
    ...                                     val=0,
    ...                                     inplace=True,
    ...                                     verbose=False)
    >>> x is y
    True
    """
    if verbose:
        timer = datetime.now()
        proc = psutil.Process(os.getpid())
        proc.cpu_percent()  # Initiate monitoring of CPU usage
        print("Running mdtools.dynamics.replace_short_sequences()...")
    if min_len > np.uint16(-1):
        raise ValueError("min_len must be less than 65535, which is"
                         " np.uint16(-1). If you really need a higher"
                         " value for 'min_len', change the dtype of"
                         " 'seen_nframes_ago' and 'seen_nframes_ahead'"
                         " in the source code of this function")
    mdt.check.list_of_cms(list_of_arrays)
    if not inplace:
        list_of_arrays = deepcopy(list_of_arrays)
    if min_len < 2:
        if verbose:
            print("mdtools.dynamics.replace_short_sequences() done")
            print("Elapsed time: {}".format(datetime.now() - timer))
            print("CPU time:             {}"
                  .format(timedelta(seconds=sum(proc.cpu_times()[:4]))))
            print("CPU usage:            {:.2f} %"
                  .format(proc.cpu_percent()))
            print("Current memory usage: {:.2f} MiB"
                  .format(proc.memory_info().rss / 2**20))
        return list_of_arrays
    if verbose:
        print("Arrays to process:   {}".format(len(list_of_arrays)))
        print("Shape of the arrays: {}".format(list_of_arrays[0].shape))
    i = 0
    a = list_of_arrays[i]
    seen_nframes_ago = np.ones(a.shape, dtype=np.uint16)
    seen_nframes_ahead = np.zeros(a.shape, dtype=np.uint16)
    seen_before = np.ones(a.shape, dtype=bool)
    for j in range(1, min(min_len, len(list_of_arrays) - i)):
        seen = (list_of_arrays[i + j] == a)
        seen_nframes_ahead += (seen & seen_before)
        seen_before = seen
    mask = ((seen_nframes_ago + seen_nframes_ahead) < min_len)
    a[mask] = val
    seen_nframes_ago += 1
    if verbose:
        loa = mdt.rti.ProgressBar(list_of_arrays[1:],
                                  initial=1,
                                  unit="arrays")
    else:
        loa = list_of_arrays[1:]
    for i, a in enumerate(loa, start=1):
        seen_nframes_ago[a != list_of_arrays[i - 1]] = 1
        seen_nframes_ahead = np.zeros(a.shape, dtype=np.uint16)
        seen_before = np.ones(a.shape, dtype=bool)
        for j in range(1, min(min_len, len(list_of_arrays) - i)):
            seen = (list_of_arrays[i + j] == a)
            seen_nframes_ahead += (seen & seen_before)
            seen_before = seen
        mask = ((seen_nframes_ago + seen_nframes_ahead) < min_len)
        a[mask] = val
        seen_nframes_ago += 1
        if verbose:
            progress_bar_mem = proc.memory_info().rss / 2**20
            loa.set_postfix_str("{:>7.2f}MiB".format(progress_bar_mem),
                                refresh=False)
    if verbose:
        loa.close()
        print("mdtools.dynamics.replace_short_sequences() done")
        print("Elapsed time: {}".format(datetime.now() - timer))
        print("CPU time:             {}"
              .format(timedelta(seconds=sum(proc.cpu_times()[:4]))))
        print("CPU usage:            {:.2f} %"
              .format(proc.cpu_percent()))
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss / 2**20))
    return list_of_arrays


def replace_short_sequences_global(
        list_of_arrays, min_len, val=0, inplace=True, verbose=False,
        debug=False):
    """
    Replace consecutive occurrences of the same value that are shorter
    than a given minimum length by a given value at any location of the
    arrays.

    Parameters
    ----------
    list_of_arrays : tuple or list or numpy.ndarray
        Sequence of arrays (usually one array per trajectory frame).
        All arrays in the sequence must have the same shape and all
        arrays must be :class:`NumPy arrays <numpy.ndarray>`.
    min_len : int
        All consecutive occurrences **shorter** than `min_len` will be
        replaced by `val`.  If `min_len` is less than two,
        `list_of_arrays` will be returned without any changes.
    val : scalar, optional
        The value used to replace the values in consecutive occurrences
        shorter than `min_len`.  Be aware that the arrays in
        `list_of_arrays` must have a suitable dtype to be able to hold
        `val`.
    inplace : bool, optional
        If ``True``, modify `list_of_arrays` inplace instead of creating
        a copy of `list_of_arrays`.
    verbose : bool, optional
        If ``True``, print progress information to standard output.
    debug : bool, optional
        If ``True``, check the input arguments.

        .. deprecated:: 0.0.0.dev0
            This argument is without use and will be removed in a future
            release.

    See Also
    --------
    :func:`mdtools.dynamics.correct_intermittency` :
        Correct for intermittent behavior of discrete variables stored
        in a sequence of arrays
    :func:`mdtools.dynamics.correct_intermittency_1d` :
        Correct for intermittent behavior of discrete variables stored
        in a 1-dimensional array
    :func:`mdtools.dynamics.replace_short_sequences` :
        Replace consecutive occurrences of the same value that are
        shorter than a given minimum length by a given value

    Notes
    -----
    This function is very similar to
    :func:`mdtools.dynamics.replace_short_sequences`, but here
    consecutive occurrences of the same value are not checked between
    elements at the same location of the arrays in `list_of_arrays` but
    all elements of the arrays in `list_of_arrays` are taken into
    account.

    `min_len` must not exceed 65535 (``np.uint16(-1)``). If you really
    need a higher value for `min_len`, change the dtype of
    ``seen_nframes_ago`` and ``seen_nframes_ahead`` in the source code
    of this function.  Be aware that high values of `min_len` increase
    the computational cost.

    Examples
    --------
    >>> a = np.array([[1, 2, 3],
    ...               [1, 2, 3]])
    >>> b = np.array([[2, 2, 3],
    ...               [2, 2, 3]])
    >>> c = np.array([[3, 3, 3],
    ...               [3, 3, 3]])
    >>> x = (a, b, c)
    >>> y = mdt.dyn.replace_short_sequences_global(x,
    ...                                            min_len=2,
    ...                                            val=0,
    ...                                            inplace=False,
    ...                                            verbose=False)
    >>> for arr in y:
    ...     print(arr)
    [[0 2 3]
     [0 2 3]]
    [[2 2 3]
     [2 2 3]]
    [[3 3 3]
     [3 3 3]]
    >>> y = mdt.dyn.replace_short_sequences_global(x,
    ...                                            min_len=3,
    ...                                            val=0,
    ...                                            inplace=False,
    ...                                            verbose=False)
    >>> for arr in y:
    ...     print(arr)
    [[0 0 3]
     [0 0 3]]
    [[0 0 3]
     [0 0 3]]
    [[3 3 3]
     [3 3 3]]

    If `min_len` is greater than ``len(list_of_arrays)``, all elements
    of all arrays are replaced by `val`:

    >>> y = mdt.dyn.replace_short_sequences_global(x,
    ...                                            min_len=len(x)+1,
    ...                                            val=0,
    ...                                            inplace=False,
    ...                                            verbose=False)
    >>> for arr in y:
    ...     print(arr)
    [[0 0 0]
     [0 0 0]]
    [[0 0 0]
     [0 0 0]]
    [[0 0 0]
     [0 0 0]]

    If `min_len` is less than two, the input `list_of_arrays` is
    returned without any changes:

    >>> y = mdt.dyn.replace_short_sequences_global(x,
    ...                                            min_len=1,
    ...                                            val=0,
    ...                                            inplace=False,
    ...                                            verbose=False)
    >>> all(np.array_equal(x[i], arr) for i, arr in enumerate(y))
    True

    Inplace modification:

    >>> y = mdt.dyn.replace_short_sequences_global(x,
    ...                                            min_len=3,
    ...                                            val=0,
    ...                                            inplace=True,
    ...                                            verbose=False)
    >>> x is y
    True
    """
    if verbose:
        timer = datetime.now()
        proc = psutil.Process(os.getpid())
        proc.cpu_percent()  # Initiate monitoring of CPU usage
        print("Running mdtools.dynamics.replace_short_sequences_global()...")
    if min_len > np.uint16(-1):
        raise ValueError("min_len must be less than 65535, which is"
                         " np.uint16(-1). If you really need a higher"
                         " value for 'min_len', change the dtype of"
                         " 'seen_nframes_ago' and 'seen_nframes_ahead'"
                         " in the source code of this function")
    mdt.check.list_of_cms(list_of_arrays)
    if not inplace:
        list_of_arrays = deepcopy(list_of_arrays)
    if min_len < 2:
        if verbose:
            print("mdtools.dynamics.replace_short_sequences_global()"
                  " done")
            print("Elapsed time: {}".format(datetime.now() - timer))
            print("CPU time:             {}"
                  .format(timedelta(seconds=sum(proc.cpu_times()[:4]))))
            print("CPU usage:            {:.2f} %"
                  .format(proc.cpu_percent()))
            print("Current memory usage: {:.2f} MiB"
                  .format(proc.memory_info().rss / 2**20))
        return list_of_arrays
    if verbose:
        print("Arrays to process:   {}".format(len(list_of_arrays)))
        print("Shape of the arrays: {}".format(list_of_arrays[0].shape))
    i = 0
    a = list_of_arrays[i]
    init_values = np.unique(a)
    seen_continuously = init_values
    seen_nframes_ago = np.ones(init_values.shape, dtype=np.uint16)
    seen_nframes_ahead = np.zeros(init_values.shape, dtype=np.uint16)
    for j in range(1, min(min_len, len(list_of_arrays) - i)):
        seen_values = np.unique(list_of_arrays[i + j])
        seen_continuously = np.intersect1d(seen_continuously,
                                           seen_values)
        mask = np.isin(init_values,
                       test_elements=seen_continuously,
                       assume_unique=True)
        seen_nframes_ahead += mask
    mask = ((seen_nframes_ago + seen_nframes_ahead) < min_len)
    a[np.isin(a, init_values[mask])] = val
    seen_nframes_ago += 1
    if verbose:
        loa = mdt.rti.ProgressBar(list_of_arrays[1:],
                                  initial=1,
                                  unit="arrays")
    else:
        loa = list_of_arrays[1:]
    for i, a in enumerate(loa, start=1):
        init_values_prev = init_values
        seen_nframes_ago_prev = seen_nframes_ago
        init_values = np.unique(a)
        seen_continuously = init_values
        seen_nframes_ago = np.ones(init_values.shape, dtype=np.uint16)
        seen_nframes_ahead = np.zeros(init_values.shape, dtype=np.uint16)
        mask_prev = np.isin(init_values_prev,
                            test_elements=init_values,
                            assume_unique=True)
        mask = np.isin(init_values,
                       test_elements=init_values_prev,
                       assume_unique=True)
        seen_nframes_ago[mask] = seen_nframes_ago_prev[mask_prev]
        for j in range(1, min(min_len, len(list_of_arrays) - i)):
            seen_values = np.unique(list_of_arrays[i + j])
            seen_continuously = np.intersect1d(seen_continuously,
                                               seen_values)
            mask = np.isin(init_values,
                           test_elements=seen_continuously,
                           assume_unique=True)
            seen_nframes_ahead += mask
        mask = ((seen_nframes_ago + seen_nframes_ahead) < min_len)
        a[np.isin(a, init_values[mask])] = val
        seen_nframes_ago += 1
        if verbose:
            progress_bar_mem = proc.memory_info().rss / 2**20
            loa.set_postfix_str("{:>7.2f}MiB".format(progress_bar_mem),
                                refresh=False)
    if verbose:
        print("mdtools.dynamics.replace_short_sequences_global()"
              " done")
        print("Elapsed time: {}".format(datetime.now() - timer))
        print("CPU time:             {}"
              .format(timedelta(seconds=sum(proc.cpu_times()[:4]))))
        print("CPU usage:            {:.2f} %"
              .format(proc.cpu_percent()))
        print("Current memory usage: {:.2f} MiB"
              .format(proc.memory_info().rss / 2**20))
    return list_of_arrays
