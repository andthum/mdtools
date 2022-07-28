# This file is part of MDTools.
# Copyright (C) 2021, 2022  The MDTools Development Team and all
# contributors listed in the file AUTHORS.rst
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
Functions for file input/output handling

This module can be called from :mod:`mdtools` via the shortcut ``fh``::

    import mdtools as mdt
    mdt.fh  # insetad of mdt.file_handler

"""


# Standard libraries
import bz2
import gzip
import lzma
import os
import warnings
from datetime import datetime

# Third-party libraries
import numpy as np

# First-party libraries
import mdtools as mdt


def cd_up(n, path=__file__):
    """
    Move `n` steps upwards in the directory tree.

    Parameters
    ----------
    n : int
        Number of steps to go up in the directory tree.
    path : str or bytes or os.PathLike, optional
        Directory or file to use as start.  Default: Position of the
        file from which this function is called (``__file__``).

    Returns
    -------
    p : str
        The `n`-th parent directory of `path`.
    """
    p = os.path.abspath(os.path.expandvars(path))
    for _ in range(n):
        p = os.path.dirname(p)
    return p


def xopen(fname, mode="rt", fformat=None, rename=True, **kwargs):
    """
    Open a (compressed) file and return a corresponding
    |file-like_object|.

    This function is a replacement for the built-in :func:`open`
    function that can additionally read and write compressed files.
    Supported compression formats:

        * gzip (.gz)
        * bzip2 (.bz2)
        * XZ/LZMA2 (.xz)
        * LZMA (.lzma)

    Parameters
    ----------
    fname : str or bytes or os.PathLike
        Name of the file to open.
    mode : {'r', 'rt', 'rb', 'w', 'wt', 'wb', 'x', 'xt', 'xb', 'a', \
'at', 'ab'}, optional
        Opening mode.  See the built-in :func:`open` function for more
        details.
    fformat : {None, 'gz', 'bz2', 'xz', 'lzma', 'uncompressed'}, \
optional
        Explicitly specify the file format.  If ``None``, the file
        format is guessed from the file name extension if present and
        otherwise from the file signature.  If ``'uncompressed'``, the
        file is treated as uncompressed file.
    rename : bool, optional
        If ``True`` and a file called `fname` already exists and the
        file is opended in writing mode, rename the existing file to
        ``'fname.bak_timestamp'``.  See
        :func:`mdtools.file_handler.backup` for more details.
    kwargs : dict, optional
        Additional keyword arguments to parse to the function that is
        used for opening the file.  See there for possible arguments and
        their description.

    Returns
    -------
    file : file-like object
        The created |file-like_object|.

    See Also
    --------
    :func:`open` :
        Function used to open uncompressed files
    :func:`gzip.open` :
        Function used to open gzip-compressed files
    :func:`bz2.open` :
        Function used to open bzip2-compressed files
    :func:`lzma.open` :
        Function used to open XZ- and LZMA-compressed files

    Notes
    -----
    When writing and `fformat` is ``None``, the compression algorithm is
    chosen based on the extension of the given file:

        * ``'.gz'`` uses gzip compression.
        * ``'.bz2'`` uses bzip2 compression.
        * ``'.xz'`` uses XZ/LZMA2 compression.
        * ``'.lzma'`` uses legacy LZMA compression.
        * otherwise, no compression is done.

    When reading and `fformat` is ``None``, the file format is detected
    from the file name extension if present.  If no extension is present
    or the extension is unknown, the format is detected from the file
    signature, i.e. the first few bytes of the file also known as
    "`magic numbers
    <https://www.garykessler.net/library/file_sigs.html>`__".

    References
    ----------
    Inspired by `xopen <https://github.com/pycompression/xopen>`__ by
    Marcel Martin, Ruben Vorderman et al.
    """
    fname = os.fspath(fname)
    signatures = {
        # https://datatracker.ietf.org/doc/html/rfc1952#page-6
        "gz": b"\x1f\x8b",
        # https://en.wikipedia.org/wiki/List_of_file_signatures
        "bz2": b"\x42\x5a\x68",
        # https://tukaani.org/xz/xz-file-format.txt
        "xz": b"\xfd\x37\x7a\x58\x5a\x00",
        # https://zenhax.com/viewtopic.php?t=27
        "lzma": b"\x5d\x00",
    }

    if fformat not in [None, "uncompressed"] + list(signatures.keys()):
        raise ValueError("Invalid value for 'fformat': {}".format(fformat))

    # Use text mode by default, like the built-in `open` function, also
    # when opening compressed files.
    if mode in ("r", "w", "x", "a"):
        mode += "t"

    # Detect file format from extension.
    if fformat is None:
        for extension in signatures.keys():
            if isinstance(fname, bytes):
                if fname.endswith(b"." + extension.encode()):
                    fformat = extension
            else:
                if fname.endswith("." + extension):
                    fformat = extension

    # Detect file format from file signature.
    if fformat is None and "w" not in mode and "x" not in mode:
        max_len = max(len(signature) for signature in signatures.values())
        try:
            with open(fname, "rb") as fh:
                file_start = fh.read(max_len)
        except OSError:
            # File could not be opened.
            file_start = False
        if file_start:
            for extension, signature in signatures.items():
                if file_start.startswith(signature):
                    fformat = extension
                    break

    if "w" in "mode" and rename:
        mdt.fh.backup(fname)
    if fformat == "gz":
        return gzip.open(fname, mode, **kwargs)
    elif fformat == "bz2":
        return bz2.open(fname, mode, **kwargs)
    elif fformat in ("xz", "lzma"):
        return lzma.open(fname, mode, **kwargs)
    elif fformat == "uncompressed" or fformat is None:
        return open(fname, mode, **kwargs)


def tail(fname, n, **kwargs):
    """
    Read the last n lines from a file.

    Parameters
    ----------
    fname : str or bytes or os.PathLike
        Name of the input file.
    n : int
        The number of lines to read from the end of the input file.
    kwargs : dict, optional
        Additional keyword arguments to parse to
        :func:`mdtools.file_handler.xopen`.  See there for possible
        arguments and their description.  By default, `mode` is set to
        ``'rt'`` (open file for reading in text mode).

    Returns
    -------
    lines : list
        List containing the last `n` lines of the input file.  Each list
        item represents one line of the file.
    """
    lines = []
    if n <= 0:
        return lines
    # Step width by which to move the cursor (the given value was an
    # emprically determined to give best performance and might be
    # further optimized).
    step_width = max(10 * n, 1)
    kwargs.setdefault("mode", "rt")
    with mdt.fh.xopen(fname, **kwargs) as file:
        file.seek(0, 2)  # Set cursor to end of file.
        pos = file.tell()  # Get current cursor position.
        # Move cursor backwards until the n-th last line is reached.
        # Termination criterium must be n+1 to get the entire n-th last
        # line and not just a part of it.
        while len(lines) < n + 1:
            pos -= min(step_width, pos)
            file.seek(pos, 0)
            lines = file.readlines()
            if pos == 0:  # Reached start of file.
                break
    return lines[-n:]


def str2none_or_type(val, dtype, empty_none=False, case_sensitive=True):
    """
    Convert a string to the NoneType ``None`` or to a given type.

    If the input string is ``'None'``, convert it to the NoneType
    ``None``, else convert it to the type given by `dtype`.

    Parameters
    ----------
    val : str_like
        The input value.  Can be anything that can be converted to a
        string.
    dtype : type
        The type to which `val` should be converted if ``str(val)`` is
        not ``'None'``.  An exception will be raised if the conversion
        ``dtype(str(val))`` is not possible.  The exact exception
        depends on `val` and `dtype`.
    empty_none : bool, optional
        If ``True``, also convert `val` to ``None`` if ``str(val)`` is
        the empty string ``''``.
    case_sensitive : bool, optional
        If ``False``, also convert the lower case string ``'none'`` to
        the NoneType ``None``.

    Returns
    -------
    val : None or dtype
        The input string, either converted to ``None`` or to `dtype`.

    See Also
    --------
    :func:`mdtools.file_handler.str2bool` :
        Convert a string to a boolean value

    Notes
    -----
    This function was written to enable passing ``None`` to scripts via
    the command line.  By default, :mod:`argparse` reads command-line
    arguments as simple strings.  This makes it impossible to pass
    ``None`` to a script via the command line, because it will always
    render it as the string ``'None'``.  Pass
    ``lambda val: mdt.fh.str2none_or_type(val, dtype=<dtype>)`` (where
    ``<dtype>`` is e.g. ``float`` or ``str``) to the `type` keyword of
    :meth:`argparse.ArgumentParser.add_argument` to convert the string
    ``'None'`` (and optionally the empty string ``''``) to the NoneType
    ``None``.

    References
    ----------
    This code was adapted from https://stackoverflow.com/a/55063765.

    Examples
    --------
    >>> mdt.fh.str2none_or_type('None', dtype=str)  # Returns None
    >>> mdt.fh.str2none_or_type(None, dtype=str)  # Returns None
    >>> mdt.fh.str2none_or_type('none', dtype=str)
    'none'
    >>> mdt.fh.str2none_or_type('none', dtype=str, \
case_sensitive=False)  # Returns None
    >>> mdt.fh.str2none_or_type('', dtype=str)
    ''
    >>> mdt.fh.str2none_or_type('', dtype=str, empty_none=True)  \
# Returns None
    >>> mdt.fh.str2none_or_type(2, dtype=str)
    '2'

    >>> mdt.fh.str2none_or_type('None', dtype=int)  # Returns None
    >>> mdt.fh.str2none_or_type('2', dtype=int)
    2
    >>> mdt.fh.str2none_or_type(2, dtype=int)
    2

    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument(
    ...     '--spam',
    ...     type=lambda val: mdt.fh.str2none_or_type(val, dtype=str)
    ... )
    _StoreAction(option_strings=['--spam'], dest='spam', ...)
    >>> parser.add_argument('--eggs', type=str)
    _StoreAction(option_strings=['--eggs'], dest='eggs', ...)
    >>> args = parser.parse_args(['--spam', 'None', '--eggs', 'None'])
    >>> args.spam is None
    True
    >>> args.eggs is None
    False
    >>> args.eggs == 'None'
    True
    """
    val = str(val)
    if (
        (case_sensitive and val == "None")
        or (not case_sensitive and val.lower() == "none")
        or (empty_none and val == "")
    ):
        return None
    else:
        return dtype(val)


def str2bool(val, accept_yes_no=True, accept_abbrev=True, accept_01=True):
    """
    Convert a string to a boolean value.

    Convert the strings ``'true'`` and ``'false'`` to their
    corresponding booleans ``True`` and ``False``.  This function is
    case-insensitive!

    Parameters
    ----------
    val : str_like
        The input value.  Can be anything that can be converted to a
        string.
    accept_yes_no : bool, optional
        If ``True``, also convert ``'yes'`` to ``True`` and ``'no'`` to
        ``False``.
    accept_abbrev : bool, optional
        If ``True``, also convert the following abbreviations:

            * ``'t'`` to ``True``.
            * ``'y'`` to ``True``.
            * ``'f'`` to ``False``.
            * ``'n'`` to ``False``.

    accept_01 : bool, optional
        If ``True``, also convert ``'1'`` to ``True`` and ``'0'`` to
        ``False``.

    Returns
    -------
    val : bool
        The input string converted to ``True`` or ``False``.

    Raises
    ------
    ValueError :
        If `val` is an unknown string that cannot be converted to a
        boolean value.

    See Also
    --------
    :func:`mdtools.file_handler.str2none_or_type` :
        Convert a string to the NoneType ``None`` or to a given type

    Notes
    -----
    This function was written to enable passing booleans to scripts via
    the command line.  By default, :mod:`argparse` reads command-line
    arguments as simple strings.  This makes it impossible to pass
    booleans to a script via the command line in an intuitive way,
    because the type conversion to bool will convert all non-empty
    strings to ``True`` (including the string ``'false'``) .  Pass
    ``str2bool(val)`` to the `type` keyword of
    :meth:`argparse.ArgumentParser.add_argument` to convert the strings
    ``'true'`` and ``'false'`` to their corresponding boolean values
    ``True`` or ``False``.

    References
    ----------
    This code was adapted from https://stackoverflow.com/a/43357954.

    Examples
    --------
    >>> mdt.fh.str2bool('True')
    True
    >>> mdt.fh.str2bool('true')
    True
    >>> mdt.fh.str2bool('false')
    False
    >>> mdt.fh.str2bool('False')
    False
    >>> mdt.fh.str2bool('fAlSe')
    False

    >>> mdt.fh.str2bool('yes', accept_yes_no=True)
    True
    >>> mdt.fh.str2bool('no', accept_yes_no=True)
    False

    >>> mdt.fh.str2bool('y', accept_yes_no=True, accept_abbrev=True)
    True
    >>> mdt.fh.str2bool('n', accept_yes_no=True, accept_abbrev=True)
    False
    >>> mdt.fh.str2bool('n', accept_yes_no=True, accept_abbrev=False)
    Traceback (most recent call last):
    ...
    ValueError: ...

    >>> mdt.fh.str2bool(1, accept_01=True)
    True
    >>> mdt.fh.str2bool('1', accept_01=True)
    True
    >>> mdt.fh.str2bool(0, accept_01=True)
    False
    >>> mdt.fh.str2bool('0', accept_01=True)
    False

    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--spam', type=mdt.fh.str2bool)
    _StoreAction(option_strings=['--spam'], ...)
    >>> parser.add_argument('--eggs', type=str)
    _StoreAction(option_strings=['--eggs'], ...)
    >>> args = parser.parse_args(['--spam', 'yes', '--eggs', 'no'])
    >>> args.spam
    True
    >>> args.eggs
    'no'
    """
    val = str(val).lower()
    eval_true = ["true"]
    eval_false = ["false"]
    if accept_yes_no:
        eval_true.append("yes")
        eval_false.append("no")
    if accept_abbrev:
        eval_true += [s[0] for s in eval_true]
        eval_false += [s[0] for s in eval_false]
    if accept_01:
        eval_true.append("1")
        eval_false.append("0")
    if val in eval_true:
        return True
    elif val in eval_false:
        return False
    else:
        raise ValueError(
            "Could not convert 'val' ({}) to bool. 'val' is neither in {} nor"
            " in {}".format(val, eval_true, eval_false)
        )


def backup(fname):
    """
    Backup a file by renaming it.

    Check if a file with name `fname` already exists.  If so, rename it
    to ``'fname.bak_timestamp'``, where ``'timestamp'`` is the time when
    the renaming was done in YYYY-MM-DD_HH-MM-SS format.

    Parameters
    ----------
    fname : str or bytes or os.PathLike
        The name of the file to backup.

    Returns
    -------
    renamed : bool
        Returns ``True`` if a file called `fname` already existed and
        was renamed.  ``False`` if no file called `fname` exists and no
        backup was done.
    """
    if os.path.isfile(fname):
        timestamp = datetime.now()
        backup_name = (
            fname + ".bak_" + str(timestamp.strftime("%Y-%m-%d_%H-%M-%S"))
        )
        os.rename(fname, backup_name)
        print("Backuped {} to {}".format(fname, backup_name))
        return True
    else:
        return False


def indent(text, amount, char=" "):
    r"""
    Indent a text by a given amount.

    Pad every line of `text` with as many instances of `char` as given
    by `amount`.  Lines in `text` are identified by Python's string
    method :meth:`str.splitlines`.

    Parameters
    ----------
    text : str
        String to be indented.
    amount : int
        Pad every line of `text` by this many instances of `char`.
        Negative values are treated as zero.
    char : str, optional
        The string to be used as padding.

    Returns
    -------
    indented_text : str
        The input text with every line padded by the given amount of
        padding characters.

    Examples
    --------
    >>> s = "Hello, World!\n  It's me, Mario!"
    >>> print(s)
    Hello, World!
      It's me, Mario!
    >>> print(mdt.fh.indent(s, amount=4))
        Hello, World!
          It's me, Mario!
    >>> print(mdt.fh.indent(s, amount=1, char="# "))
    # Hello, World!
    #   It's me, Mario!
    """
    padding = amount * char
    return "".join(padding + line for line in text.splitlines(keepends=True))


def header_str():
    """
    Create a standard header string for text files.

    The string can be printed directly to standard output using
    :func:`print`.

    The header string contains:

        * The date and time the text file was created (actually this
          function was called).
        * The MDTools copyright notice.
        * The information generated by
          :func:`mdtools.run_time_info.run_time_info`.

    Returns
    -------
    header : str
        Human readable string containing the above listed content.

    See Also
    --------
    :func:`mdtools.run_time_info.run_time_info` :
        Generate some run time information
    :func:`mdtools.run_time_info.run_time_info_str` :
        Create a string containing some run time information
    """
    timestamp = datetime.now()
    script, command_line, cwd, exe, version, pversion = mdt.rti.run_time_info()
    header = "Created by {} on {}\n".format(
        script, timestamp.strftime("%Y/%m/%d %H:%M:%S")
    )
    header += "\n"
    header += mdt.__copyright_notice__ + "\n"
    header += "\n"
    header += "\n"
    header += "Command line input:\n"
    header += "  {}\n".format(command_line)
    header += "Working directory:\n"
    header += "  {}\n".format(cwd)
    header += "Executable:\n"
    header += "  {}\n".format(exe)
    header += "mdtools version:\n"
    header += "  {}\n".format(version)
    header += "Python version:\n"
    header += "  {}\n".format(pversion)
    return header


def write_header(fname, **kwargs):
    """
    Write the standard MDTools header to file.

    See :func:`mdtools.file_handler.header_str` for further information
    about what is contained in the header.

    Parameters
    ----------
    fname : str or bytes or os.PathLike
        The name of the file to which to write the header.
    kwargs : dict, optional
        Additional keyword arguments to parse to
        :func:`mdtools.file_handler.xopen`.  See there for possible
        arguments and their description.  By default, `mode` is set to
        ``'wt'`` (open file for writing in text mode, truncating the
        file first).

    See Also
    --------
    :func:`mdtools.file_handler.header_str` :
        Create a standard header string for text files
    :func:`mdtools.file_handler.backup` :
        Backup a file by renaming it
    """
    kwargs.setdefault("mode", "wt")
    with mdt.fh.xopen(fname, **kwargs) as outfile:
        outfile.write(mdt.fh.indent(mdt.fh.header_str(), amount=1, char="# "))


def savetxt(fname, data, rename=True, **kwargs):
    """
    Save an array to a text file.

    Parameters
    ----------
    fname : str or os.PathLike
        The name of the file to create.
    data : array_like
        1- or 2-dimensional array of data to be saved.
    rename : bool, optional
        If ``True`` and a file called `fname` already exists, rename it
        to ``'fname.bak_timestamp'``.  See
        :func:`mdtools.file_handler.backup` for more details.
    kwargs : dict, optional
        Additional keyword arguments to parse to :func:`numpy.savetxt`.
        See there for possible arguments and their description.  By
        default, `fmt` is set to ``'%16.9e'``.

    See Also
    --------
    :func:`numpy.savetxt` :
        Save an array to a text file
    :func:`mdtools.file_handler.savetxt_matrix` :
        Save a data matrix to a text file
    :func:`mdtools.file_handler.header_str` :
        Create a standard header string for text files
    :func:`mdtools.file_handler.backup` :
        Backup a file by renaming it

    Notes
    -----
    This function simply calls :func:`numpy.savetxt` and adds a MDTools
    specific header to the output file.  See
    :func:`mdtools.file_handler.header_str` for further information
    about what is included in the header.
    """
    fname = os.fspath(fname)
    kwargs.setdefault("fmt", "%16.9e")
    header = kwargs.pop("header", None)
    if header is None or header.strip() == "":
        header = mdt.fh.header_str()
    else:
        header = mdt.fh.header_str() + "\n\n" + header
    if rename:
        mdt.fh.backup(fname)
    np.savetxt(fname, data, **kwargs)


def savetxt_matrix(
    fname,
    data,
    var1,
    var2,
    init_values1=None,
    init_values2=None,
    upper_left=0,
    **kwargs,
):
    """
    Save a data matrix to a text file.

    Write data that are a function of two independent variables, `var1`
    and `var2`, as a matrix to a text file using
    :func:`mdtools.file_handler.savetxt`.  The dependency of the data
    from `var1` is represented by the rows and the dependency from
    `var2` is represented by the columns.

    Parameters
    ----------
    fname : str or os.PathLike
        The name of the file to create.
    data : array_like
        2-dimensional array of data to be saved.  Must be of shape
        ``(n, m)``, where ``n`` is the number of samples of the first
        independent variable (depicted row wise) and ``m`` is the mumber
        of samples of the second independent variable (depicted column
        wise).
    var1, var2 : array_like
        Array of shape ``(n,)`` (`var1`) or ``(m,)`` (`var2`) containing
        the values of the first or second independent variable at which
        the data were sampled.
    init_values1, init_values2 : array_like, optional
        If supplied, the values stored in this array will be handled as
        special initial data values corresponding to the very first
        value in `var1` or `var2`.  Must be an array of shape ``(m,)``
        (`init_values1`) or ``(n,)`` (`init_values2`).  If given, `data`
        must be of shape ``(n-1, m)`` or ``(n, m-1)`` or ``(n-1, m-1)``
        if both are given.
    upper_left : scalar, optional
        Value to put in the upper left corner of the final data matrix.
        Usually, this value is meaningless and set to zero.
    kwargs : dict, optional
        Additional keyword arguments to parse to
        :func:`mdtools.file_handler.savetxt`.  See there for possible
        arguments and their description.

    See Also
    --------
    :func:`mdtools.file_handler.savetxt` :
        Save an array to a text file
    :func:`mdtools.file_handler.write_matrix_block` :
        Save a data matrix as block to a text file
    :func:`mdtools.file_handler.backup` :
        Backup a file by renaming it

    Notes
    -----
    Internally, this function calls
    :func:`mdtools.file_handler.savetxt` which in turn calls
    :func:`numpy.savetxt`.
    """
    var1 = np.asarray(var1)
    var2 = np.asarray(var2)
    mdt.check.array(var1, dim=1)
    mdt.check.array(var2, dim=1)
    if init_values1 is None and init_values2 is None:
        mdt.check.array(data, shape=(len(var1), len(var2)))
    elif init_values1 is not None and init_values2 is None:
        mdt.check.array(init_values1, shape=var2.shape)
        mdt.check.array(data, shape=(len(var1) - 1, len(var2)))
        data = np.vstack((init_values1, data))
    elif init_values1 is None and init_values2 is not None:
        mdt.check.array(init_values2, shape=var1.shape)
        mdt.check.array(data, shape=(len(var1), len(var2) - 1))
        data = np.column_stack((init_values2, data))
    elif init_values1 is not None and init_values2 is not None:
        mdt.check.array(init_values1, shape=var2.shape)
        mdt.check.array(init_values2, shape=var1.shape)
        mdt.check.array(data, shape=(len(var1) - 1, len(var2) - 1))
        if init_values2[0] != init_values1[0]:
            raise ValueError(
                "init_values2[0] ({}) is not the same as init_values1[0]"
                " ({})".format(init_values2[0], init_values1[0])
            )
        data = np.column_stack((init_values2[1:], data))
        data = np.vstack((init_values1, data))
    data = np.column_stack((var1, data))
    var2 = np.insert(var2, 0, upper_left)
    data = np.vstack((var2, data))
    mdt.fh.savetxt(fname, data, **kwargs)


def write_matrix_block(
    fname,
    data,
    var1,
    var2,
    init_values1=None,
    init_values2=None,
    upper_left=0,
    fmt=">16.9e",
    data_name="z",
    data_unit=None,
    var1_name="x",
    var2_name="y",
    var1_unit=None,
    var2_unit=None,
    block_number=None,
    **kwargs,
):
    """
    Save a data matrix as block to a text file.

    Write data that are a function of two independent variables, `var1`
    and `var2`, as a matrix to a text file.  The dependency of the data
    from `var1` is represented by the rows and the dependency from
    `var2` is represented by the columns.

    Parameters
    ----------
    fname : str or bytes or os.PathLike
        Name of the file to write to.
    data : array_like
        2-dimensional array of data to write to file.  Must be of shape
        ``(n, m)``, where ``n`` is the number of samples of the first
        independent variable (depicted row wise) and ``m`` is the mumber
        of samples of the second independent variable (depicted column
        wise).
    var1, var2 : array_like
        Array of shape ``(n,)`` (`var1`) or ``(m,)`` (`var2`) containing
        the values of the first or second independent variable at which
        the data were sampled.
    init_values1, init_values2 : array_like, optional
        If supplied, the values stored in this array will be handled as
        special initial data values corresponding to the very first
        value in `var1` or `var2`.  Must be an array of shape ``(m,)``
        (`init_values1`) or ``(n,)`` (`init_values2`).  If given, `data`
        must be of shape ``(n-1, m)`` or ``(n, m-1)`` or ``(n-1, m-1)``
        if both are given.
    upper_left : scalar, optional
        Value to put in the upper left corner of the final data matrix.
        Usually, this value is meaningless and set to zero.
    fmt : str, optional
        |format_specifier|.
    data_name : str, optional
        The name of the data.  If supplied, it will be printed in the
        block header.
    data_unit : str, optional
        The unit of the data.  If supplied, will be printed in the
        block header.
    var1_name, var2_name : str, optional
        The names of the independent variables.  If supplied, they will
        be printed in the block header.
    var1_unit, var2_unit : str, optional
        The units of the independent variables.  If supplied, they will
        be printed in the block header.
    block_number : int, optional
        The number of the data block in `fname`.  If supplied, it will
        be printed in the block header.
    kwargs : dict, optional
        Additional keyword arguments to parse to
        :func:`mdtools.file_handler.xopen`.  See there for possible
        arguments and their description.  By default, `mode` is set to
        ``'wt'`` (open file for writing in text mode, truncating the
        file first).

    See Also
    --------
    :func:`mdtools.file_handler.savetxt_matrix` :
        Save a data matrix to a text file
    :func:`mdtools.file_handler.write_header` :
        Create a file and write the standard MDTools header to it
    """
    var1 = np.asarray(var1)
    var2 = np.asarray(var2)
    mdt.check.array(var1, dim=1)
    mdt.check.array(var2, dim=1)
    if init_values1 is None and init_values2 is None:
        mdt.check.array(data, shape=(len(var1), len(var2)))
    elif init_values1 is not None and init_values2 is None:
        mdt.check.array(init_values1, shape=var2.shape)
        mdt.check.array(data, shape=(len(var1) - 1, len(var2)))
    elif init_values1 is None and init_values2 is not None:
        mdt.check.array(init_values2, shape=var1.shape)
        mdt.check.array(data, shape=(len(var1), len(var2) - 1))
    elif init_values1 is not None and init_values2 is not None:
        mdt.check.array(init_values1, shape=var2.shape)
        mdt.check.array(init_values2, shape=var1.shape)
        mdt.check.array(data, shape=(len(var1) - 1, len(var2) - 1))
        if init_values2[0] != init_values1[0]:
            raise ValueError(
                "init_values2[0] ({}) is not the same as init_values1[0]"
                " ({})".format(init_values2[0], init_values1[0])
            )

    kwargs.setdefault("mode", "wt")
    with mdt.fh.xopen(fname, **kwargs) as outfile:
        # Block header
        outfile.write("\n\n\n\n")
        if block_number is not None:
            outfile.write("# Block {}\n".format(block_number))
        if var1_name is not None:
            outfile.write("# First column:    {}".format(var1_name))
            if var1_unit is not None:
                outfile.write(" in {}".format(var1_unit))
            outfile.write("\n")
        if var2_name is not None:
            outfile.write("# First row:       {}".format(var2_name))
            if var2_unit is not None:
                outfile.write(" in {}".format(var2_unit))
            outfile.write("\n")
        if data_name is not None:
            outfile.write("# Matrix elements: {}".format(data_name))
            if data_unit is not None:
                outfile.write(" in {}".format(data_unit))
            outfile.write("\n")
        # Column numbers
        num_cols = len(var2)
        fmt_int = len("{:{fmt}}".format(0, fmt=fmt))
        fmt_int = ">" + str(fmt_int) + "d"
        outfile.write("# Column number:\n")
        outfile.write("# {:{fmt}}".format("1", fmt=fmt_int))
        for col_num in range(2, num_cols + 2):
            outfile.write(" {:{fmt}}".format(col_num, fmt=fmt_int))
        outfile.write("\n")
        # The row after the row with the column numbers contains the
        # values of `var2`.
        outfile.write("  {:{fmt}}".format(upper_left, fmt=fmt))
        for col_num in range(num_cols):
            outfile.write(" {:{fmt}}".format(var2[col_num], fmt=fmt))
        outfile.write("\n")
        # If there are any special initial values for the very first
        # value of `var1`, print them to the next row.
        if init_values1 is not None:
            outfile.write("  {:{fmt}}".format(var1[0], fmt=fmt))
            for col_num in range(num_cols):
                outfile.write(
                    " {:{fmt}}".format(init_values1[col_num], fmt=fmt)
                )
            outfile.write("\n")
            start_row = 1
        else:
            start_row = 0
        # Print remaining rows. The first column always contains the
        # current value of `var1`. The remaining columns contain the
        # data.
        num_rows = len(var1)
        for row_num in range(start_row, num_rows):
            outfile.write("  {:{fmt}}".format(var1[row_num], fmt=fmt))
            # If there are any special initial values for the very first
            # value of `var2`, print them to the second column.
            if init_values2 is not None:
                outfile.write(
                    " {:{fmt}}".format(init_values2[row_num], fmt=fmt)
                )
                start_col = 1
            else:
                start_col = 0
            for col_num in range(start_col, num_cols):
                outfile.write(
                    " {:{fmt}}".format(
                        data[row_num - start_row][col_num - start_col], fmt=fmt
                    )
                )
            outfile.write("\n")


def save_dtrj(fname, dtrj):
    """
    Save a discrete trajectory to file.

    Save a discrete trajectory as :class:`numpy.ndarray` to a compressed
    NumPy |npz_archive|.

    .. warning::

        Creating a gzip-compressed :file:`.npz` archive does not work.
        However bzip2-, xz- and lzma-compressed archives can be created.
        Note that the created :file:`.npz` archive is already compressed
        by :func:`numpy.savez_compressed`.  Therefore, further
        compression is usually not reasonable.

    Parameters
    ----------
    fname : str or bytes or os.PathLike
        Name of the file to write to.
    dtrj : array_like
        The discrete trajectory to save.  Must meet the requirements
        given in :func:`mdtools.check.dtrj`.  Note that the shape of
        `dtrj` is expanded to ``(1, f)`` if it is only of shape
        ``(f,)``.

    See Also
    --------
    :func:`numpy.savez_compressed` :
        Save one or multiple arrays in a compressed :file:`.npz` archive
    :func:`mdtools.file_handler.load_dtrj` :
        Load a discrete trajectory from file

    Notes
    -----
    This function simply checks whether `dtrj` is a suitable discrete
    trajectory and then saves it to file using
    :func:`numpy.savez_compressed`.  Insise the created :file:`.npz`
    archive, the discrete trajectory is stored in the file
    :file:`dtrj.npy`.
    """
    try:
        dtrj = mdt.check.dtrj(dtrj)
    except ValueError as err:
        warnings.warn(
            "The given trajectory is not a suitable discrete trajectory:\n"
            "{}".format(err),
            UserWarning,
        )
    with mdt.fh.xopen(fname, "wb") as fh:
        np.savez_compressed(fh, dtrj=dtrj)


def load_dtrj(fname, **kwargs):
    """
    Load a discrete trajectory from file.

    Load a discrete trajectory stored as :class:`numpy.ndarray` from a
    binary NumPy |npy_file| or from a (compressed) NumPy |npz_archive|.

    Parameters
    ----------
    fname : str or bytes or os.PathLike
        Name of the file containing the discrete trajectory.  The
        discrete trajectory must be stored as :class:`numpy.ndarray`
        either in a binary NumPy |npy_file| or in a (compressed) NumPy
        |npz_archive|.  If loading from an :file:`.npz` archive, it is
        first tried to read the discrete trajectory from the file
        "dtrj.npy".  If this file is not present in the archive, the
        discrete trajectory is read from the first file in the archive.
        The discrete trajectory must be of shape ``(n, f)``, where ``n``
        is the number of compounds and ``f`` is the number of frames.
        The shape can also be ``(f,)``, in which case the array is
        expanded to shape ``(1, f)``.  The array must only contain
        integers or floats whose fractional part is zero, because, the
        elements of a discrete trajectory are interpreted as the indices
        of the states in which a given compound is at a given frame.
    kwargs : dict, optional
        Additional keyword arguments to parse to :func:`numpy.load`.
        See there for posible arguments and their description.  By
        default, `allow_pickle` is set to ``False``.

    Returns
    -------
    dtrj : numpy.ndarray
        The discrete trajectory loaded from the given file.

    See Also
    --------
    :func:`numpy.load` :
        Load arrays or pickled objects from :file:`.npy`, :file:`.npz`
        or pickled files
    :func:`mdtools.file_handler.save_dtrj` :
        Save a discrete trajectory to file

    Notes
    -----
    This function simply calls :func:`numpy.load` and checks whether
    the loaded :class:`numpy.ndarray` is a suitable discrete trajectory.
    """
    fh = mdt.fh.xopen(fname, "rb")
    kwargs.setdefault("allow_pickle", False)
    dtrj_loaded = np.load(fh, **kwargs)
    if isinstance(dtrj_loaded, np.lib.npyio.NpzFile):
        dtrj = dtrj_loaded.get("dtrj", None)
        if dtrj is None:
            dtrj = list(dtrj_loaded.values())[0]
        dtrj_loaded.close()
    elif isinstance(dtrj_loaded, np.ndarray):
        dtrj = dtrj_loaded
    else:
        raise TypeError(
            "Unkown type of the loaded data: {}".format(type(dtrj_loaded))
        )
    fh.close()
    return mdt.check.dtrj(dtrj)
