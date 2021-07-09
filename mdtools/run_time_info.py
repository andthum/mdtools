# This file is part of MDTools.
# Copyright (C) 2021  The MDTools Development Team and all contributors
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
Classes and functions for getting run time information

This module can be called from :mod:`mdtools` via the shortcut ``rti``::

    import mdtools as mdt
    mdt.rti  # insetad of mdt.run_time_info

"""


# Standard libraries
import sys
import os
import warnings
from datetime import datetime
# Third party libraries
import psutil
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.log import ProgressBar
# Local application/library specific imports
import mdtools as mdt


class ProgressBar(ProgressBar):
    """
    Display a visual progress bar and time estimate.

    The :class:`ProgressBar` decorates an iterable object, returning an
    iterator which acts exactly like the original iterable, but prints a
    dynamically updating progressbar every time a value is requested.

    See Also
    --------
    :class:`MDAnalysis.lib.log.ProgressBar` : Parent class
    :class:`tqdm.auto.tqdm` : Grandparent class

    Notes
    -----
    This class is derived from :class:`MDAnalysis.lib.log.ProgressBar`,
    which in turn is derived from :class:`tqdm.auto.tqdm`.  The only
    difference to :class:`tqdm.auto.tqdm` is that some default arguments
    are changed:

        * `ascii` is set to ``True``.
        * `unit` is set to ``'frames'``.
        * `unit_scale` is set to ``True``.
        * The `bar_format` is changed to switch of a possible inversion
          of 'unit/s' to 's/unit' (see also
          https://github.com/tqdm/tqdm/issues/72).
        * `mininterval` is set to ``300`` seconds.
        * `maxinterval` is set to ``3600`` seconds.

    See the MDAnalysis_ and tqdm_ documentations for further information.

    .. _MDAnalysis: https://docs.mdanalysis.org/stable/documentation_pages/lib/log.html#MDAnalysis.lib.log.ProgressBar
    .. _tqdm: https://tqdm.github.io/docs/tqdm/#__init__

    Example
    -------
    .. code-block::

        for ts in mdt.rti.ProgressBar(u.trajectory):
            # Perform analysis

    will produce something similar to::

        25%|#####1           | 25/100 [00:15<00:45,  1.67frames/s]

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the :class:`ProgressBar`.

        Parameters
        ----------
        args : list, optional
            Non-keyword arguments.  See
            :class:`MDAnalysis.lib.log.ProgressBar` for possible choices.
        kwargs : dict, optional
            Keynword arguments.  See
            :class:`MDAnalysis.lib.log.ProgressBar` for possible choices.
        """
        kwargs['ascii'] = kwargs.pop('ascii', True)
        kwargs['unit'] = kwargs.pop('unit', "frames")
        kwargs['mininterval'] = kwargs.pop('mininterval', 300)
        kwargs['maxinterval'] = kwargs.pop('maxinterval', 3600)
        kwargs['unit_scale'] = kwargs.pop('unit_scale', True)
        bar_format = ("{l_bar}{bar}|" +
                      " {n_fmt}/{total_fmt}" +
                      " [{elapsed}<{remaining}," +
                      " {rate_noinv_fmt}" +
                      "{postfix}]")
        kwargs['bar_format'] = kwargs.pop('bar_format', bar_format)
        super().__init__(*args, **kwargs)


def get_num_CPUs():
    """
    Get the number of available CPUs.

    The number of available CPUs is obtained in decreasing precedence
    from the environment variables:

        1. OMP_NUM_THREADS
        2. SLURM_CPUS_PER_TASK
        3. SLURM_JOB_CPUS_PER_NODE
        4. SLURM_CPUS_ON_NODE
        5. or from the Python function :func:`os.cpu_count()`

    """
    if os.environ.get('OMP_NUM_THREADS') is not None:
        return int(os.environ['OMP_NUM_THREADS'])
    elif os.environ.get('SLURM_CPUS_PER_TASK') is not None:
        return int(os.environ['SLURM_CPUS_PER_TASK'])
    elif os.environ.get('SLURM_JOB_CPUS_PER_NODE') is not None:
        return int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    elif os.environ.get('SLURM_CPUS_ON_NODE') is not None:
        return int(os.environ['SLURM_CPUS_ON_NODE'])
    else:
        return os.cpu_count()


def mem_usage(unit='MiB'):
    """
    Get current memory usage.

    Returns the current memory usage of the process that calls this
    function.

    Parameters
    ----------
    unit : {'B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'KB', 'MB', 'GB', 'TB', 'PB'}
        String determining the unit in which the memory usage should be
        returned.  Default is mebibytes (``'MiB'``).

    Returns
    -------
    mem : float
        Memory currently used by the calling process.  Returns ``None``
        if the memory usage cannot be determined.
    """
    try:
        proc = psutil.Process(os.getpid())
    except (psutil.NoSuchProcess, psutil.AccessDenied) as err:
        warnings.warn("Cannot get memory usage of process {} because of"
                      " {}".format(os.getpid(), err), RuntimeWarning)
        return None
    if unit == 'B':
        scaling = 1
    elif unit == 'KiB':
        scaling = 2**10
    elif unit == 'MiB':
        scaling = 2**20
    elif unit == 'GiB':
        scaling = 2**30
    elif unit == 'TiB':
        scaling = 2**40
    elif unit == 'PiB':
        scaling = 2**50
    elif unit == 'KB':
        scaling = 1e3
    elif unit == 'MB':
        scaling = 1e6
    elif unit == 'GB':
        scaling = 1e9
    elif unit == 'TB':
        scaling = 1e12
    elif unit == 'PB':
        scaling = 1e15
    return proc.memory_info().rss / scaling


def run_time_info():
    """
    Generate some run time information.

    Returns
    -------
    script_name : string
        The name of the running script.
    command_line : string
        The command line input used to start the script.
    cwd : string
        The working directory the script was started from.
    exe : string
        The exact call of the executable script.
    mdt_version : string
        The version number of MDTools.
    python_version : string
        The version number of Python.

    See Also
    --------
    :func:`mdtools.run_time_info.run_time_info_str` :
        Create a string containing some run time information
    """
    script_name = str(os.path.basename(sys.argv[0]))
    command_line = script_name + " " + " ".join(sys.argv[1:])
    python_version = (str(sys.version_info[0]) + "." +
                      str(sys.version_info[1]) + "." +
                      str(sys.version_info[2]))
    return (script_name,
            command_line,
            os.getcwd(),
            sys.argv[0],
            mdt.__version__,
            python_version)


def run_time_info_str(indent=0):
    """
    Create a string containing some run time information.

    The string can be printed directly to standard output using
    :func:`print`.

    The information contained in the string is:

        * The date and time the script (actually this function) was
          called.
        * The MDTools copyright notice.
        * The information generated by
          :func:`mdtools.run_time_info.run_time_info`.

    Parameters
    ----------
    indent : int, optional
        Number of spaces to indent the information string.  Negative
        indentation is treated as zero indentation.

    Returns
    -------
    rti : str
        Human readable string containing the above listed content.

    See Also
    --------
    :func:`mdtools.run_time_info.run_time_info` :
        Generate some run time information
    :func:`mdtools.file_handler.header_str` :
        Create a string containing some run time information to be used
        as header for text files
    :func:`mdtools.file_handler.indent` : Indent a text
    """
    timestamp = datetime.now()
    script, command_line, cwd, exe, version, pversion = run_time_info()
    rti = "{}\n".format(script)
    rti += "{}\n".format(timestamp.strftime('%Y/%m/%d %H:%M'))
    rti += "\n"
    rti += mdt.__copyright_notice__ + "\n"
    rti += "\n"
    rti += "\n"
    rti += "Command line input:\n"
    rti += "  {}\n".format(command_line)
    rti += "Working directory:\n"
    rti += "  {}\n".format(cwd)
    rti += "Executable:\n"
    rti += "  {}\n".format(exe)
    rti += "mdtools version:\n"
    rti += "  {}\n".format(version)
    rti += "Python version:\n"
    rti += "  {}".format(pversion)
    if indent > 0:
        rti = mdt.fh.indent(rti, amount=indent, char=" ")
    return rti


def ag_info_str(ag, indent=0):
    """
    Create a string containing information about a MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup`.

    The string can be printed directly to standard output using
    :func:`print`.

    The information contained in the string is:

        * The total number of

            - :class:`Segments <MDAnalysis.core.groups.Segment>`
            - :class:`Residues <MDAnalysis.core.groups.Residue>`
            - :attr:`Fragments <MDAnalysis.core.groups.AtomGroup.fragments>`
            - :class:`Atoms <MDAnalysis.core.groups.Atom>`

        * The number of different

            - :class:`Segments <MDAnalysis.core.groups.Segment>`
            - :class:`Residues <MDAnalysis.core.groups.Residue>`
            - :class:`Atoms <MDAnalysis.core.groups.Atom>`
            - And their respective names.

        * Whether the input :class:`~MDAnalysis.core.groups.AtomGroup` is
          an :class:`~MDAnalysis.core.groups.UpdatingAtomGroup` or not.

    Refer to the MDAnalysis' user guide for an
    |explanation_of_these_terms|.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        The MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup` for
        which to create the information string.
    indent : int, optional
        Number of spaces to indent the information string.  Negative
        indentation is treated as zero indentation.

    Returns
    -------
    ag_info : str
        Human readable string containing the above listed content.

    See Also
    --------
    :func:`mdtools.file_handler.indent` : Indent a text
    """
    unique_segids = np.unique(ag.segids)
    unique_resnames = np.unique(ag.resnames)
    unique_atmnames = np.unique(ag.names)
    unique_atmtypes = np.unique(ag.types)
    try:
        n_fragments = ag.n_fragments
    except mda.exceptions.NoDataError:
        n_fragments = "N/A"
    ag_info = "Segments:               {}\n".format(ag.n_segments)
    ag_info += ("  Different segments:   {}\n"
                .format(len(unique_segids)))
    ag_info += ("  Segment name(s):      '{}'\n"
                .format('\' \''.join(i for i in unique_segids)))
    ag_info += "Residues:               {}\n".format(ag.n_residues)
    ag_info += ("  Different residues:   {}\n"
                .format(len(unique_resnames)))
    ag_info += ("  Residue name(s):      '{}'\n"
                .format('\' \''.join(i for i in unique_resnames)))
    ag_info += "Atoms:                  {}\n".format(ag.n_atoms)
    ag_info += ("  Different atom names: {}\n"
                .format(len(unique_atmnames)))
    # If each atom in a molecule has a different name (frequently the
    # atoms are numbered sequentially), the list of different atom names
    # is too long to be usefully printed on screen.
    # ag_info += ("  Atom name(s):         '{}'\n"
    #             .format('\' \''.join(i for i in unique_atmnames)))
    ag_info += ("  Different atom types: {}\n"
                .format(len(unique_atmtypes)))
    ag_info += ("  Atom type(s):         '{}'\n"
                .format('\' \''.join(i for i in unique_atmtypes)))
    ag_info += "Fragments:              {}\n".format(n_fragments)
    ag_info += ("Updating atom group:    {}"
                .format(isinstance(ag,
                                   mda.core.groups.UpdatingAtomGroup)))
    if indent > 0:
        ag_info = mdt.fh.indent(ag_info, amount=indent, char=" ")
    return ag_info


def dtrj_trans_info(dtrj):
    """
    Generate basic information about the state transitions in a discrete
    trajectory.

    .. deprecated:: 0.0.0.dev0
        :func:`mdtools.run_time_info.dtrj_trans_info` will be replaced
        by :func:`mdtools.run_time_info.dtrj_trans_info_new` in a future
        release.

    Parameters
    ----------
    dtrj : array_like
        The discrete trajectory.  Array of shape ``(n, f)``, where ``n``
        is the number of compounds and ``f`` is the number of frames.
        The elements of `dtrj` are interpreted as the indices of the
        states in which a given compound is at a given frame.

    Returns
    -------
    n_stay : int
        Number of compounds that are in the same state in all frames.
    always_neg : int
        Number of compounds that are in a negative state in all frames.
    never_neg : int
        Number of compounds that are not in a negative state in any
        frame.
    n_frames_neg : int
        Total number of frames with negative states (summed over all
        compounds).
    n_trans : int
        Total number of state transitions.
    pos2pos : int
        Total number of Positive -> Positive transitions (transitions
        from one state with a positive/zero state index to another state
        with a positive/zero state index).
    pos2neg : int
        Number of Positive -> Negative transitions.
    neg2pos : int
        Number of Negative -> Positive transitions.
    neg2neg : int
        Number of Negative -> Negative transitions.

    See Also
    --------
    :func:`mdtools.run_time_info.dtrj_trans_info_str` :
        Create a string containing basic information about the state
        transitions in a discrete trajectory

    Note
    ----
    Positive states are states with a state index equal(!) to or greater
    than zero.  Negative states are states with a state index less than
    zero.
    """
    dtrj = np.asarray(dtrj)
    dtrj = np.asarray(dtrj.T, order='C')
    if dtrj.ndim == 1:
        dtrj = np.expand_dims(dtrj, axis=0)
    elif dtrj.ndim > 2:
        raise ValueError("The discrete trajectory must have one or two"
                         " dimensions")
    if np.any(np.modf(dtrj)[0] != 0):
        warnings.warn("At least one element of the discrete trajectory"
                      " is not an integer", RuntimeWarning)

    n_stay = np.count_nonzero(np.all(dtrj == dtrj[0], axis=0))
    dtrj_neg = (dtrj < 0)
    always_neg = np.count_nonzero(np.all(dtrj_neg, axis=0))
    never_neg = np.count_nonzero(~np.any(dtrj_neg, axis=0))
    n_frames_neg = np.count_nonzero(dtrj_neg)
    del dtrj_neg

    N_CMPS = dtrj.shape[1]
    transitions = (np.diff(dtrj, axis=0) != 0)
    trans_init = np.vstack([transitions, np.zeros(N_CMPS, dtype=bool)])
    trans_final = np.insert(transitions, 0, np.zeros(N_CMPS), axis=0)
    n_trans = np.count_nonzero(transitions)
    if np.count_nonzero(trans_init) != n_trans:
        raise ValueError("The number of transitions in trans_init is not"
                         " the same as in transitions. This should not"
                         " have happened")
    if np.count_nonzero(trans_final) != n_trans:
        raise ValueError("The number of transitions in trans_final is"
                         " not the same as in transitions. This should"
                         " not have happened")
    pos2pos = np.count_nonzero((dtrj[trans_init] >= 0) &
                               (dtrj[trans_final] >= 0))
    pos2neg = np.count_nonzero((dtrj[trans_init] >= 0) &
                               (dtrj[trans_final] < 0))
    neg2pos = np.count_nonzero((dtrj[trans_init] < 0) &
                               (dtrj[trans_final] >= 0))
    neg2neg = np.count_nonzero((dtrj[trans_init] < 0) &
                               (dtrj[trans_final] < 0))
    if pos2pos + pos2neg + neg2pos + neg2neg != n_trans:
        raise ValueError("The sum of Positive <-> Negative transitions"
                         " ({}) is not equal to the total number of"
                         " transitions ({}). This should not have"
                         " happened"
                         .format(pos2pos + pos2neg + neg2pos + neg2neg,
                                 n_trans))

    return (n_stay, always_neg, never_neg, n_frames_neg,
            n_trans, pos2pos, pos2neg, neg2pos, neg2neg)


def dtrj_trans_info_str(dtrj):
    """
    Create a string containing basic information about the state
    transitions in a discrete trajectory.

    .. deprecated:: 0.0.0.dev0
        :func:`mdtools.run_time_info.dtrj_trans_info_str` will be
        replaced by :func:`mdtools.run_time_info.dtrj_trans_info_str_new`
        in a future release.

    The string can be printed directly to standard output using
    :func:`print`.

    The information contained in the string is:

        * The number of frames of the discrete trajectory (per compound).
        * The number of compounds (or in other words, the number of
          single-compound trajectories contained in `dtrj`).
        * The information generated by
          :func:`mdtools.run_time_info.dtrj_trans_info`.

    Parameters
    ----------
    dtrj : array_like
        The discrete trajectory.  Array of shape ``(n, f)``, where ``n``
        is the number of compounds and ``f`` is the number of frames.
        The elements of `dtrj` are interpreted as the indices of the
        states in which a given compound is at a given frame.

    Returns
    -------
    rti : str
        Human readable string containing the above listed content.

    See Also
    --------
    :func:`mdtools.run_time_info.dtrj_trans_info` :
        Generate basic information about the state transitions in a
        discrete trajectory
    :func:`mdtools.file_handler.indent` : Indent a text
    """
    N_CMPS, N_FRAMES = dtrj.shape
    trans_info = dtrj_trans_info(dtrj)
    dti = ("No. of frames (per compound):                         {:>12d}\n"
           .format(N_FRAMES))
    dti += ("No. of compounds:                                     {:>12d}\n"
            .format(N_CMPS))
    dti += ("No. of compounds that never leave their state:        {:>12d}\n"
            .format(trans_info[0]))
    dti += ("No. of compounds that are always in a negative state: {:>12d}\n"
            .format(trans_info[1]))
    dti += ("No. of compounds that are never  in a negative state: {:>12d}\n"
            .format(trans_info[2]))
    dti += ("Total No. of frames with negative states:             {:>12d}\n"
            .format(trans_info[3]))
    dti += "\n"
    dti += ("Total No. of state transitions:          {:>12d}\n"
            .format(trans_info[4]))
    dti += ("No. of Positive -> Positive transitions: {:>12d}  ({:>8.4f} %)\n"
            .format(trans_info[5], 100 * trans_info[5] / trans_info[4]))
    dti += ("No. of Positive -> Negative transitions: {:>12d}  ({:>8.4f} %)\n"
            .format(trans_info[6], 100 * trans_info[6] / trans_info[4]))
    dti += ("No. of Negative -> Positive transitions: {:>12d}  ({:>8.4f} %)\n"
            .format(trans_info[7], 100 * trans_info[7] / trans_info[4]))
    dti += ("No. of Negative -> Negative transitions: {:>12d}  ({:>8.4f} %)\n"
            .format(trans_info[8], 100 * trans_info[8] / trans_info[4]))
    dti += "Positive states are states with a state index >= 0\n"
    dti += "Negative states are states with a state index <  0\n"
    return dti
