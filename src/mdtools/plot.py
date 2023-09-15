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
Functions for plotting data.

The functions in this module are simple wrappers around plotting
functions of |matplotlib|.

This module is not directly accessible from mdtools, but must be loaded
explicitly e.g. via::

    import mdtools.plot as mdtplt

This module loads the custom matplotlib style sheet
:file:`src/mdtools/pkg_data/mdtools.mplstyle`.  If you want to keep the
default plotting style, you have to re-load the default style after
importing :mod:`mdtools.plot`::

    import matplotlib.pyplot as plt
    plt.style.use("default")

See `Customizing Matplotlib with style sheets and rcParams`_ for more
information about style sheets.  Among other settings,
:file:`mdtools.mplstyle` sets the default figure size to 8.26772 x
5.82677 inches, which is the size of a DIN A5 paper in landscape format.
The default font size is set to 24 points and also the line widths and
marker sizes are scaled appropriately.  In this way the produced figures
can be used for presentations and posters or they can be scaled down to
the width of a single text column in a scientific journal without
loosing readability.

With the introduction of the :file:`mdtools.mplstyle` style sheet during
preparation of version 0.0.1.0, a lot of functions in this module became
deprecated, because the functions were just wrappers around
corresponding matplotlib functions with hardcoded style settings.  With
the style sheet, the matplotlib functions can be called directly and
create plots with the desired style.

.. todo::

    * Update deprecated plot functions in the plot scripts.
    * Remove the deprecated plot functions in this module.

.. _Customizing Matplotlib with style sheets and rcParams: https://matplotlib.org/stable/tutorials/introductory/customizing.html
"""  # noqa: W505, E501


# Standard libraries
import warnings

# Third-party libraries
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

# The recommended way to read package data at runtime is to use
# `importlib.resources` which is part of the standard library since
# Python version 3.7.  Older Python versions can use the backport
# `importlib_resources`.
# (https://setuptools.pypa.io/en/latest/userguide/datafiles.html#accessing-data-files-at-runtime).
# However, `importlib.resources` requires an `__init__.py` file to be
# present in each data directory (https://bugs.python.org/issue34534,
# https://gitlab.com/python-devs/importlib_resources/-/issues/58).  This
# is undesirable, because this exposes the subdirectory as exportable
# package (https://stackoverflow.com/a/58941536).  The backport
# `importlib_resources` does not require an `__init__.py`.  Because of
# this and because MDTools should support Python versions 3.6 to 3.9, we
# will only rely on the backport `importlib_resources` even if MDTools
# is installed in Python version 3.9.
from importlib_resources import files

# First-party libraries
import mdtools as mdt


# Load custom matplotlib style sheet.
# For customizing Matplotlib with style sheets see
# https://matplotlib.org/stable/tutorials/introductory/customizing.html
# For accessing package data at run time see
# https://setuptools.pypa.io/en/latest/userguide/datafiles.html#accessing-data-files-at-runtime
style_file = files("mdtools.pkg_data").joinpath("mdtools.mplstyle")
try:
    plt.style.use(["default", style_file])
except OSError:
    warnings.warn(
        "Could not find the MDTools plotting style at {}".format(style_file),
        RuntimeWarning,
    )

# Alignment of offsetText of colorbars
CBAR_YAX_HALIGN = "left"
CBAR_YAX_VALIGN = "bottom"

# Settings for legends with fontsize 'x-small' (one times smaller than
# the default fontsize)
LEGEND_KWARGS_XSMALL = {
    "fontsize": "x-small",
    "title_fontsize": "x-small",
    "borderaxespad": 0.8,
    "labelspacing": 0.25,
    "handlelength": 1.35,
    "handletextpad": 0.5,
    "columnspacing": 1.4,
}


class MidpointNormalize(colors.Normalize):
    """
    Class to define your own colorbar normalization.

    .. deprecated:: 0.0.0.dev0
        Use :class:`matplotlib.colors.CenteredNorm` or
        :class:`matplotlib.colors.TwoSlopeNorm` instead.

    Renormalizing a colorbar is for instance useful when you want that a
    diverging colorbar is centered at zero.  Just parse
    ``norm = mdtplt.MidpointNormalize(midpoint=0.0)`` to the plotting
    function.

    See `Colormap Normalization`_ for more information.

    .. _Colormap Normalization:
        https://matplotlib.org/stable/users/explain/colors/colormapnorms.html#custom-normalization-manually-implement-two-linear-ranges
    """  # noqa: W505, E501

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        """
        Initialize the Normalization.

        Parameters
        ----------
        vmin : float, optional
            The data value that defines 0.0 in the normalization.
            Defaults to the min value of the dataset.
        vmax : float, optional
            The data value that defines 1.0 in the normalization.
            Defaults to the the max value of the dataset.
        midpoint : float, optional
            The data value that defines 0.5 in the normalization.
        clip : bool, optional
            This argument is without use!
        """
        warnings.warn(
            "Use 'matplotlib.colors.CenteredNorm' or"
            " 'matplotlib.colors.TwoSlopeNorm' instead",
            DeprecationWarning,
        )
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """
        Map data to the interval [0, 1].

        Parameters
        ----------
        value : array_like
            The data to use for normalizing the colormap.
        clip : bool, optional
            This argument is without use!
        """
        # Including masked values, but ignoring clipping and other edge
        # cases.  See https://stackoverflow.com/a/48598564
        result, is_scalar = self.process_value(value)
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.array(
            np.interp(value, x, y), mask=result.mask, copy=False
        )


def plot(
    ax,
    x,
    y,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    logx=False,
    logy=False,
    xlabel=r"$x$",
    ylabel=r"$y$",
    legend_loc="best",
    **kwargs,
):
    """
    Plot data to a :class:`matplotlib.axes.Axes` object using
    :meth:`matplotlib.axes.Axes.plot`.

    .. deprecated:: 0.0.0.dev2
        Use :meth:`matplotlib.axes.Axes.plot`,
        :meth:`matplotlib.axes.Axes.set` and
        :meth:`matplotlib.axes.Axes.legend` directly.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : array_like
        One dimensional array containing the x data.
    y : array_like
        Array of the same shape as `x` containing the y data.
    xmin : scalar, optional
        Left limit for plotting on x-axis.  The left limit may be
        greater than the right limit, in which case the tick values will
        show up in decreasing order.  Default is ``None``, which means
        set the limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    logx : bool, optional
        Use logarithmic x scale.
    logy : bool, optional
        Same as `logx`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    legend_loc : int or str, optional
        Position of the legend.  See :func:`matplotlib.pyplot.legend`
        for possible arguments.
    kwargs : dict, optional
        Keyword arguments to pass to :meth:`matplotlib.axes.Axes.plot`.
        See there for possible keyword arguments.

    Returns
    -------
    img : list
        A list of :class:`matplotlib.lines.Line2D` objects representing
        the plotted data.
    """
    warnings.warn(
        "Use 'matplotlib.axes.Axes.plot' instead", DeprecationWarning
    )

    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16

    img = ax.plot(x, y, **kwargs)
    if logx:
        ax.set_xscale("log", basex=10, subsx=np.arange(2, 10))
    if logy:
        ax.set_yscale("log", basey=10, subsy=np.arange(2, 10))
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=tick_length,
        labelsize=fontsize_ticks,
        pad=tick_pad,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
        pad=tick_pad,
    )
    if kwargs.get("label") is not None:
        ax.legend(
            loc=legend_loc,
            numpoints=1,
            frameon=False,
            fontsize=fontsize_legend,
        )

    return img


def scatter(
    ax,
    x,
    y,
    s=None,
    c=None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    logx=False,
    logy=False,
    xlabel=r"$x$",
    ylabel=r"$y$",
    legend_loc="best",
    **kwargs,
):
    """
    Add a scatter plot to a :class:`matplotlib.axes.Axes` object using
    :meth:`matplotlib.axes.Axes.scatter`.

    .. deprecated:: 0.0.0.dev2
        Use :meth:`matplotlib.axes.Axes.scatter`,
        :meth:`matplotlib.axes.Axes.set` and
        :meth:`matplotlib.axes.Axes.legend` directly.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : array_like
        One dimensional array containing the x data.
    y : array_like
        Array of the same shape as `x` containing the y data.
    s : scalar or array_like, optional
        The marker size of each individual point.
    c : str or array_like, optional
        The marker color of each individual point.  See the matplotlib
        documentation of :meth:`matplotlib.axes.Axes.scatter` for more
        information.
    xmin : scalar, optional
        Left limit for plotting on x-axis.  The left limit may be
        greater than the right limit, in which case the tick values will
        show up in decreasing order.  Default is ``None``, which means
        set the limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    logx : bool, optional
        Use logarithmic x scale.
    logy : bool, optional
        Same as `logx`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    legend_loc : int or str, optional
        Position of the legend.  See :func:`matplotlib.pyplot.legend`
        for possible arguments.
    kwargs : dict, optional
        Keyword arguments to pass to
        :meth:`matplotlib.axes.Axes.scatter`.  See there for possible
        keyword arguments.

    Returns
    -------
    img : matplotlib.collections.PathCollection
        A :class:`matplotlib.collections.PathCollection`.
    """
    warnings.warn(
        "Use 'matplotlib.axes.Axes.scatter' instead", DeprecationWarning
    )

    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16

    kwargs["cmap"] = kwargs.pop("cmap", "Greys")
    kwargs["vmin"] = kwargs.pop("vmin", np.nanmin(c))
    kwargs["vmax"] = kwargs.pop("vmax", np.nanmax(c))
    img = ax.scatter(x=x, y=y, s=s, c=c, **kwargs)
    if logx:
        ax.set_xscale("log", basex=10, subsx=np.arange(2, 10))
    if logy:
        ax.set_yscale("log", basey=10, subsy=np.arange(2, 10))
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=tick_length,
        labelsize=fontsize_ticks,
        pad=tick_pad,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
        pad=tick_pad,
    )
    if kwargs.get("label") is not None:
        ax.legend(
            loc=legend_loc,
            numpoints=1,
            frameon=False,
            fontsize=fontsize_legend,
        )

    return img


def errorbar(
    ax,
    x,
    y,
    xerr=None,
    yerr=None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    logx=False,
    logy=False,
    xlabel=r"$x$",
    ylabel=r"$y$",
    legend_loc="best",
    **kwargs,
):
    """
    Plot data to a :class:`matplotlib.axes.Axes` object with errorbars
    using :meth:`matplotlib.axes.Axes.errorbar`.

    .. deprecated:: 0.0.0.dev2
        Use :meth:`matplotlib.axes.Axes.errorbar`,
        :meth:`matplotlib.axes.Axes.set` and
        :meth:`matplotlib.axes.Axes.legend` directly.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : array_like
        One dimensional array containing the x data.
    y : array_like
        Array of the same shape as `x` containing the y data.
    xerr, yerr : array_like, optional
        The errorbar sizes.
    xmin : scalar, optional
        Left limit for plotting on x-axis.  The left limit may be
        greater than the right limit, in which case the tick values will
        show up in decreasing order.  Default is ``None``, which means
        set the limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    logx : bool, optional
        Use logarithmic x scale.
    logy : bool, optional
        Same as `logx`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    legend_loc : int or str, optional
        Position of the legend.  See :func:`matplotlib.pyplot.legend`
        for possible arguments.
    kwargs : dict, optional
        Keyword arguments to pass to
        :meth:`matplotlib.axes.Axes.errorbar`.  See there for possible
        keyword arguments.

    Returns
    -------
    img : ErrorbarContainer
        A :class:`matplotlib.container.ErrorbarContainer`.
    """
    warnings.warn(
        "Use 'matplotlib.axes.Axes.errorbar' instead", DeprecationWarning
    )

    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16

    img = ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, **kwargs)
    if logx:
        ax.set_xscale("log", basex=10, subsx=np.arange(2, 10))
    if logy:
        ax.set_yscale("log", basey=10, subsy=np.arange(2, 10))
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=tick_length,
        labelsize=fontsize_ticks,
        pad=tick_pad,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
        pad=tick_pad,
    )
    if kwargs.get("label") is not None:
        ax.legend(
            loc=legend_loc,
            numpoints=1,
            frameon=False,
            fontsize=fontsize_legend,
        )

    return img


def hist(
    ax,
    x,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    xlabel=r"$x$",
    ylabel=r"$y$",
    legend_loc="best",
    **kwargs,
):
    """
    Plot a histogram to a :class:`matplotlib.axes.Axes` object using
    :meth:`matplotlib.axes.Axes.hist`.

    .. deprecated:: 0.0.0.dev2
        Use :meth:`matplotlib.axes.Axes.hist`,
        :meth:`matplotlib.axes.Axes.set` and
        :meth:`matplotlib.axes.Axes.legend` directly.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : array_like
        The data.
    xmin : scalar, optional
        Left limit for plotting on x-axis.  The left limit may be
        greater than the right limit, in which case the tick values will
        show up in decreasing order.  Default is ``None``, which means
        set the limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    legend_loc : int or str, optional
        Position of the legend.  See :func:`matplotlib.pyplot.legend`
        for possible arguments.
    kwargs : dict, optional
        Keyword arguments to pass to :meth:`matplotlib.axes.Axes.hist`.
        See there for possible keyword arguments.

    Returns
    -------
    n : numpy.ndarray or list of numpy.ndarrays
        The values of the histogram bins.
    bins : numpy.ndarray
        The edges of the bins.
    patches : list or list of lists
        Silent list of individual patches used to create the histogram
        or list of such list if multiple input datasets.
    """
    warnings.warn(
        "Use 'matplotlib.axes.Axes.hist' instead", DeprecationWarning
    )

    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16

    img = ax.hist(x=x, **kwargs)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=tick_length,
        labelsize=fontsize_ticks,
        pad=tick_pad,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
        pad=tick_pad,
    )
    if kwargs.get("label") is not None:
        ax.legend(
            loc=legend_loc,
            numpoints=1,
            frameon=False,
            fontsize=fontsize_legend,
        )

    return img


def correlogram(ax, data, lags=None, siglev=0.05, kwargs_acf=None, **kwargs):
    r"""
    Create a correlogram and plot it to the given
    :class:`matplotlib.axes.Axes`.

    Create a correlogram for the given data similar to the one shown in
    the Wikipedia article `Correlogram § Statistical inference with
    correlograms
    <https://en.wikipedia.org/wiki/Correlogram#Statistical_inference_with_correlograms>`_
    by calculating the autocorelation function (ACF) and its confidence
    intervals for the given data and plotting it to the given
    :class:`matplotlib.axes.Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    data : array_like
        1-dimensional array containing the data for which to calculate
        and plot the ACF.
    lags : array_like or scalar or None, optional
        1-dimensional array of lag times with the same shape as `data`
        or the difference between the lag times or None.  If ``None``,
        `lags` is set to ``np.arange(len(data))``.  If a scalar, `lags`
        is set to ``np.arange(0, lags * len(data), lags)``.
    siglev : scalar, optional
        The significance level of the confidence intervals, usually
        denoted as :math:`\alpha`.  See
        :func:`mdtools.statistics.acf_confint` for more details.
    kwargs_acf : dict or None, optional
        Dictionary of keyword arguments to parse to
        :func:`mdtools.statistics.acf_np`.  See there for possible
        keyword arguments.
    kwargs : dict, optional
        Keyword arguments to parse to :meth:`matplotlib.axes.Axes.plot`
        and :meth:`matplotlib.axes.Axes.fill_between` to change the
        appearance of the ACF.  See there for possible keyword
        arguments.

    Returns
    -------
    img : tuple
        A tuple of :class:`matplotlib.lines.Line2D` and
        :class:`matplotlib.collections.PolyCollection` objects
        containing the plotted data.  The first element of `img` is a
        :class:`matplotlib.lines.Line2D` object representing the ACF.
        The second element is a
        :class:`matplotlib.collections.PolyCollection` containing the
        confidence intervals around the ACF.  The third and fourth
        element of `img` are :class:`matplotlib.lines.Line2D` objects
        representing the upper and lower bound of the confidence
        intervals around zero, respectively.

    See Also
    --------
    :func:`mdtools.statistics.acf_np`
        Calculate the autocorrelation function of a 1-dimensional array
    :func:`mdtools.statistics.acf_confint`
        Calculate the confidence intervals of an autocorrelation
        function
    """
    if lags is None:
        lags = np.arange(len(data))
    elif np.ndim(lags) == 0:
        lags = np.arange(0, lags * len(data), lags)
    if kwargs_acf is None:
        kwargs_acf = {}
    acf = mdt.stats.acf_np(data, **kwargs_acf)
    confint = mdt.stats.acf_confint(acf, alpha=siglev)

    confint_label = "{:.1f}% CI".format((1 - siglev) * 100)
    kwargs.setdefault("label", "ACF with " + confint_label)
    lines_acf = ax.plot(lags, acf, **kwargs)

    kwargs.pop("label", None)
    kwargs["color"] = kwargs.pop("color", lines_acf[0].get_color())
    kwargs["alpha"] = kwargs.pop("alpha", 1) * 2 / 3
    polycollection = ax.fill_between(
        lags, acf + confint, acf - confint, **kwargs
    )

    lines_confint_upper = ax.plot(
        lags,
        confint,
        linestyle="--",
        color="red",
        label=confint_label + " around 0",
    )
    lines_confint_lower = ax.plot(
        lags,
        -confint,
        linestyle=lines_confint_upper[0].get_linestyle(),
        color=lines_confint_upper[0].get_color(),
    )

    return lines_acf, polycollection, lines_confint_upper, lines_confint_lower


def hlines(
    ax,
    y,
    start,
    stop,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    legend_loc="best",
    **kwargs,
):
    """
    Plot horizontal lines at each `y` from `start` to `stop` into a
    :class:`matplotlib.axes.Axes` object using
    :meth:`matplotlib.axes.Axes.hlines`.

    .. deprecated:: 0.0.0.dev2
        Use :meth:`matplotlib.axes.Axes.hlines`,
        :meth:`matplotlib.axes.Axes.set` and
        :meth:`matplotlib.axes.Axes.legend` directly.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    y : scalar or array_like
        y-indices where to plot the lines.
    start, stop : scalar or array_like
        Respective beginning and end of each line.  If scalars are
        provided, all lines will have same length.
    xmin : scalar, optional
        Left limit for plotting on x-axis.  The left limit may be
        greater than the right limit, in which case the tick values will
        show up in decreasing order.  Default is ``None``, which means
        set the limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    legend_loc : int or str, optional
        Position of the legend.  See :func:`matplotlib.pyplot.legend`
        for possible arguments.
    kwargs : dict, optional
        Keyword arguments to pass to
        :meth:`matplotlib.axes.Axes.hlines`.  See there for possible
        keyword arguments.

    Returns
    -------
    img : matplotlib.collections.LineCollection
        A :class:`matplotlib.collections.LineCollection`.
    """
    warnings.warn(
        "Use 'matplotlib.axes.Axes.hlines' instead", DeprecationWarning
    )

    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12

    img = ax.hlines(y=y, xmin=start, xmax=stop, **kwargs)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=tick_length,
        labelsize=fontsize_ticks,
        pad=tick_pad,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
        pad=tick_pad,
    )
    if kwargs.get("label") is not None:
        ax.legend(
            loc=legend_loc,
            numpoints=1,
            frameon=False,
            fontsize=fontsize_legend,
        )

    return img


def vlines(
    ax,
    x,
    start,
    stop,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    legend_loc="best",
    **kwargs,
):
    """
    Plot vertical lines at each `x` from `start` to `stop` into a
    :class:`matplotlib.axes.Axes` object using
    :meth:`matplotlib.axes.Axes.vlines`.

    .. deprecated:: 0.0.0.dev2
        Use :meth:`matplotlib.axes.Axes.vlines`,
        :meth:`matplotlib.axes.Axes.set` and
        :meth:`matplotlib.axes.Axes.legend` directly.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : scalar or array_like
        x-indices where to plot the lines.
    start, stop : scalar or array_like
        Respective beginning and end of each line.  If scalars are
        provided, all lines will have same length.
    xmin : scalar, optional
        Left limit for plotting on x-axis.  The left limit may be
        greater than the right limit, in which case the tick values will
        show up in decreasing order.  Default is ``None``, which means
        set the limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    legend_loc : int or str, optional
        Position of the legend.  See :func:`matplotlib.pyplot.legend`
        for possible arguments.
    kwargs : dict, optional
        Keyword arguments to pass to
        :meth:`matplotlib.axes.Axes.vlines`.  See there for possible
        keyword arguments.

    Returns
    -------
    img : matplotlib.collections.LineCollection
        A :class:`matplotlib.collections.LineCollection`.
    """
    warnings.warn(
        "Use 'matplotlib.axes.Axes.vlines' instead", DeprecationWarning
    )

    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12

    img = ax.vlines(x=x, ymin=start, ymax=stop, **kwargs)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=tick_length,
        labelsize=fontsize_ticks,
        pad=tick_pad,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
        pad=tick_pad,
    )
    if kwargs.get("label") is not None:
        ax.legend(
            loc=legend_loc,
            numpoints=1,
            frameon=False,
            fontsize=fontsize_legend,
        )

    return img


def plot_2nd_xaxis(
    ax, x, y, xmin=None, xmax=None, xlabel=r"$x2$", legend_loc="best", **kwargs
):
    """
    Plot data to a second x-axis.

    .. deprecated:: 0.0.0.dev2
        Use :meth:`matplotlib.axes.Axes.twiny`,
        :meth:`matplotlib.axes.Axes.plot`,
        :meth:`matplotlib.axes.Axes.set` and
        :meth:`matplotlib.axes.Axes.legend` directly.

    Create a twin :class:`matplotlib.axes.Axes` of an existing
    :class:`matplotlib.axes.Axes` sharing the same y-axis using
    :meth:`matplotlib.axes.Axes.twiny` and plot data to the twin
    :class:`~matplotlib.axes.Axes` using
    :meth:`matplotlib.axes.Axes.plot`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`matplotlib.axes.Axes` from which to create a twin
        that shares the same y-axis.  Data are plotted to the created
        twin axes.
    x : array_like
        One dimensional array containing the x data.
    y : array_like
        Array of the same shape as `x` containing the y data.
    xmin : scalar, optional
        Left limit for plotting on the secondary x-axis.  The left limit
        may be greater than the right limit, in which case the tick
        values will show up in decreasing order.  Default is ``None``,
        which means set the limit automatically.
    xmax : scalar, optional
        Right limit for plotting on the secondary x-axis.
    xlabel : str, optional
        Label for the secondary x-axis.
    legend_loc : int or str, optional
        Position of the legend.  See :func:`matplotlib.pyplot.legend`
        for possible arguments.
    kwargs : dict, optional
        Keyword arguments to pass to :meth:`matplotlib.axes.Axes.plot`.
        See there for possible keyword arguments.

    Returns
    -------
    img : list
        A list of :class:`matplotlib.lines.Line2D` objects representing
        the plotted data.
    ax2 : matplotlib.axes.Axes
        The twin axis.
    """
    warnings.warn(
        "Use 'matplotlib.axes.Axes.twiny' and 'matplotlib.axes.Axes.plot'"
        " instead",
        DeprecationWarning,
    )

    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    label_pad = 16

    kwargs["color"] = kwargs.pop("color", "black")
    ax2 = ax.twiny()
    img = ax2.plot(x, y, **kwargs)
    ax2.set_xlim(xmin, xmax)
    ax2.set_xlabel(
        xlabel=xlabel, color=kwargs.get("color"), fontsize=fontsize_labels
    )
    ax2.xaxis.labelpad = label_pad
    ax2.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax2.tick_params(
        axis="x",
        which="major",
        direction="in",
        labelcolor=kwargs.get("color"),
        length=tick_length,
        labelsize=fontsize_ticks,
    )
    ax2.tick_params(
        axis="x",
        which="minor",
        direction="in",
        labelcolor=kwargs.get("color"),
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
    )
    if kwargs.get("label") is not None:
        ax2.legend(
            loc=legend_loc,
            numpoints=1,
            frameon=False,
            fontsize=fontsize_legend,
        )

    return img, ax2


def plot_2nd_yaxis(
    ax, x, y, ymin=None, ymax=None, ylabel=r"$y2$", legend_loc="best", **kwargs
):
    """
    Plot data to a second y-axis.

    .. deprecated:: 0.0.0.dev2
        Use :meth:`matplotlib.axes.Axes.twinx`,
        :meth:`matplotlib.axes.Axes.plot`,
        :meth:`matplotlib.axes.Axes.set` and
        :meth:`matplotlib.axes.Axes.legend` directly.

    Create a twin :class:`matplotlib.axes.Axes` of an existing
    :class:`matplotlib.axes.Axes` sharing the same x-axis using
    :meth:`matplotlib.axes.Axes.twinx` and plot data to the twin
    :class:`~matplotlib.axes.Axes` using
    :meth:`matplotlib.axes.Axes.plot`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`matplotlib.axes.Axes` from which to create a twin
        that shares the same x-axis.  Data are plotted to the created
        twin axes.
    x : array_like
        One dimensional array containing the x data.
    y : array_like
        Array of the same shape as `x` containing the y data.
    ymin : scalar, optional
        Left limit for plotting on the secondary y-axis.  The left limit
        may be greater than the right limit, in which case the tick
        values will show up in decreasing order.  Default is ``None``,
        which means set the limit automatically.
    ymax : scalar, optional
        Right limit for plotting on the secondary y-axis.
    ylabel : str, optional
        Label for the secondary y-axis.
    legend_loc : int or str, optional
        Position of the legend.  See :func:`matplotlib.pyplot.legend`
        for possible arguments.
    kwargs : dict, optional
        Keyword arguments to pass to :meth:`matplotlib.axes.Axes.plot`.
        See there for possible keyword arguments.

    Returns
    -------
    img : list
        A list of :class:`matplotlib.lines.Line2D` objects representing
        the plotted data.
    ax2 : matplotlib.axes.Axes
        The twin axis.
    """
    warnings.warn(
        "Use 'matplotlib.axes.Axes.twinx' and 'matplotlib.axes.Axes.plot'"
        " instead",
        DeprecationWarning,
    )

    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16

    kwargs["color"] = kwargs.pop("color", "black")
    ax2 = ax.twinx()
    img = ax2.plot(x, y, **kwargs)
    ax2.set_ylim(ymin, ymax)
    ax2.set_ylabel(
        ylabel=ylabel, color=kwargs.get("color"), fontsize=fontsize_labels
    )
    ax2.yaxis.labelpad = label_pad
    ax2.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax2.tick_params(
        axis="y",
        which="major",
        direction="in",
        labelcolor=kwargs.get("color"),
        length=tick_length,
        labelsize=fontsize_ticks,
        pad=tick_pad,
    )
    ax2.tick_params(
        axis="y",
        which="minor",
        direction="in",
        labelcolor=kwargs.get("color"),
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
        pad=tick_pad,
    )
    if kwargs.get("label") is not None:
        ax2.legend(
            loc=legend_loc,
            numpoints=1,
            frameon=False,
            fontsize=fontsize_legend,
        )

    return img, ax2


def fill_between(
    ax,
    x,
    y1,
    y2=0,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    logx=False,
    logy=False,
    xlabel=r"$x$",
    ylabel=r"$y$",
    legend_loc="best",
    **kwargs,
):
    """
    Fill the area between two curves `y1` and `y2` using
    :meth:`matplotlib.axes.Axes.fill_between`.

    .. deprecated:: 0.0.0.dev2
        Use :meth:`matplotlib.axes.Axes.fill_between`,
        :meth:`matplotlib.axes.Axes.set` and
        :meth:`matplotlib.axes.Axes.legend` directly.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : array_like
        One dimensional array containing the x data.
    y1 : array_like
        Array of the same shape as `x` containing the first y data.
    y2 : array_like
        Array of the same shape as `x` containing the second y data.
        Default: 0
    xmin : scalar, optional
        Left limit for plotting on x-axis.  The left limit may be
        greater than the right limit, in which case the tick values will
        show up in decreasing order.  Default is ``None``, which means
        set the limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    logx : bool, optional
        Use logarithmic x scale.
    logy : bool, optional
        Same as `logx`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    legend_loc : int or str, optional
        Position of the legend.  See :func:`matplotlib.pyplot.legend`
        for possible arguments.
    kwargs : dict, optional
        Keyword arguments to pass to
        :meth:`matplotlib.axes.Axes.fill_between`.  See there for
        possible keyword arguments.

    Returns
    -------
    img : matplotlib.collections.PolyCollection
        A :class:`matplotlib.collections.PolyCollection`.
    """
    warnings.warn(
        "Use 'matplotlib.axes.Axes.fill_between' instead", DeprecationWarning
    )

    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16

    img = ax.fill_between(x=x, y1=y1, y2=y2, **kwargs)
    if logx:
        ax.set_xscale("log", basex=10, subsx=np.arange(2, 10))
    if logy:
        ax.set_yscale("log", basey=10, subsx=np.arange(2, 10))
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=tick_length,
        labelsize=fontsize_ticks,
        pad=tick_pad,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
        pad=tick_pad,
    )
    if kwargs.get("label") is not None:
        ax.legend(
            loc=legend_loc,
            numpoints=1,
            frameon=False,
            fontsize=fontsize_legend,
        )

    return img


def fill_betweenx(
    ax,
    y,
    x1,
    x2=0,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    logx=False,
    logy=False,
    xlabel=r"$x$",
    ylabel=r"$y$",
    legend_loc="best",
    **kwargs,
):
    """
    Fill the area between two vertical curves `x1` and `x2` using
    :meth:`matplotlib.axes.Axes.fill_betweenx`.

    .. deprecated:: 0.0.0.dev2
        Use :meth:`matplotlib.axes.Axes.fill_betweenx`,
        :meth:`matplotlib.axes.Axes.set` and
        :meth:`matplotlib.axes.Axes.legend` directly.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    y : array_like
        One dimensional array containing the y data.
    x1 : array_like
        Array of the same shape as `y` containing the first x data.
    x2 : array_like
        Array of the same shape as `y` containing the second x data.
        Default: 0
    xmin : scalar, optional
        Left limit for plotting on x-axis.  The left limit may be
        greater than the right limit, in which case the tick values will
        show up in decreasing order.  Default is ``None``, which means
        set the limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    logx : bool, optional
        Use logarithmic x scale.
    logy : bool, optional
        Same as `logx`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    legend_loc : int or str, optional
        Position of the legend.  See :func:`matplotlib.pyplot.legend`
        for possible arguments.
    kwargs : dict, optional
        Keyword arguments to pass to
        :meth:`matplotlib.axes.Axes.fill_betweenx`.  See there for
        possible keyword arguments.

    Returns
    -------
    img : matplotlib.collections.PolyCollection
        A :class:`matplotlib.collections.PolyCollection`.
    """
    warnings.warn(
        "Use 'matplotlib.axes.Axes.fill_betweenx' instead", DeprecationWarning
    )

    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16

    img = ax.fill_betweenx(y=y, x1=x1, x2=x2, **kwargs)
    if logx:
        ax.set_xscale("log", basex=10, subsx=np.arange(2, 10))
    if logy:
        ax.set_yscale("log", basey=10, subsx=np.arange(2, 10))
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=tick_length,
        labelsize=fontsize_ticks,
        pad=tick_pad,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
        pad=tick_pad,
    )
    if kwargs.get("label") is not None:
        ax.legend(
            loc=legend_loc,
            numpoints=1,
            frameon=False,
            fontsize=fontsize_legend,
        )

    return img


def pcolormesh(
    ax,
    x,
    y,
    z,
    cax=None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    xlabel=r"$x$",
    ylabel=r"$y$",
    cbarlabel=r"$z$",
    **kwargs,
):
    """
    Plot two dimensional data as heatmap with
    :meth:`matplotlib.axes.Axes.pcolormesh`.

    .. deprecated:: 0.0.0.dev2
        Use :func:`mdtools.plot.pcolormesh_new` instead.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : array_like
        One dimensional array containing the x data.
    y : array_like
        One dimensional array containing the y data.
    z : array_like
        Array of shape ``(len(y)-1, len(x)-1)`` containing the z data.
        Note: The grid orientation follows the standard matrix
        convention, i.e. an array `z` with shape ``(nrows, ncolumns)``
        is plotted with the column number as `x` and the row number as
        `y`.
    cax : matplotlib.axes.Axes, optional
        The axes to put the colorbar in.
    xmin : scalar, optional
        Left limit for plotting on x-axis.  The left limit may be
        greater than the right limit, in which case the tick values will
        show up in decreasing order.  Default is ``None``, which means
        set the left limit to ``np.nanmin(x)``.
    xmax : float, optional
        Right limit for plotting on x-axis. Default is ``None``, which
        means set the left limit to ``np.nanmax(x)``.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    cbarlabel : str, optional
        Label for the colorbar.
    kwargs : dict, optional
        Keyword arguments to pass to
        :meth:`matplotlib.axes.Axes.pcolormesh`.  See there for possible
        keyword arguments.

    Returns
    -------
    heatmap : matplotlib.collections.QuadMesh
        A :class:`matplotlib.collections.QuadMesh`.
    """
    warnings.warn(
        "Use 'mdtools.plot.pcolormesh_new' instead", DeprecationWarning
    )

    fontsize_labels = 36
    fontsize_ticks = 32
    tick_length = 10
    tick_pad = 12
    label_pad = 16

    if xmin is None:
        xmin = np.nanmin(x)
    if xmax is None:
        xmax = np.nanmax(x)
    if ymin is None:
        ymin = np.nanmin(y)
    if ymax is None:
        ymax = np.nanmax(y)
    kwargs["cmap"] = kwargs.pop("cmap", "Greys")
    kwargs["vmin"] = kwargs.pop("vmin", np.nanmin(z))
    kwargs["vmax"] = kwargs.pop("vmax", np.nanmax(z))
    heatmap = ax.pcolormesh(x, y, z, **kwargs)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(
        which="major",
        direction="out",
        top=False,
        right=False,
        length=tick_length,
        labelsize=fontsize_ticks,
        pad=tick_pad,
    )
    ax.tick_params(
        which="minor",
        direction="out",
        top=False,
        right=False,
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
        pad=tick_pad,
    )

    cbar = plt.colorbar(heatmap, cax=cax, ax=ax)
    cbar.set_label(label=cbarlabel, fontsize=fontsize_labels)
    cbar.ax.yaxis.labelpad = label_pad
    cbar.ax.yaxis.offsetText.set(size=fontsize_ticks)
    cbar.ax.tick_params(
        which="major",
        direction="out",
        length=tick_length,
        labelsize=fontsize_ticks,
    )
    cbar.ax.tick_params(
        which="minor",
        direction="out",
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
    )

    return heatmap


def pcolormesh_new(*args, ax, cbar=True, cax=None, **kwargs):
    """
    Plot two dimensional data as colormap with
    :meth:`matplotlib.axes.Axes.pcolormesh`.

    See :meth:`matplotlib.axes.Axes.pcolormesh` and
    :func:`matplotlib.pyplot.colorbar` for more details.

    Parameters
    ----------
    args : iterable, optional
        Positional arguments to parse to
        :meth:`matplotlib.axes.Axes.pcolormesh`.
    ax : matplotlib.axes.Axes
        The :class:`matplotlib.axes.Axes` to which to draw the
        pcolormesh.
    cbar : bool, optional
        If ``True``, create a colorbar and plot it either in the
        :class:`matplotlib.axes.Axes` given by `ax` or `cax`.
    cax : matplotlib.axes.Axes, optional
        If given, the colorbar is placed in this
        :class:`matplotlib.axes.Axes`.  Is ignored if `cbar` is
        ``False``.
    kwargs : dict, optional
        Keyword arguments to parse to
        :meth:`matplotlib.axes.Axes.pcolormesh`.

    Returns
    -------
    pcm : matplotlib.collections.QuadMesh
       The pcolormesh.
    cbar : matplotlib.colorbar.Colorbar
       The colorbar.  Only created and returned if `cbar` is ``True``.

    Notes
    -----
    The following defaults of :meth:`matplotlib.axes.Axes.pcolormesh`
    are changed:

       * `edgecolors` is set to ``'none'``.
       * `rasterized` is set to ``True``.

    In contrast to the default MDTools plotting style, the axis ticks
    are drawn outside of the plot.
    """
    kwargs["edgecolors"] = kwargs.pop("edgecolors", "none")
    kwargs["rasterized"] = kwargs.pop("rasterized", True)
    pcm = ax.pcolormesh(*args, **kwargs)
    ax.tick_params(
        axis="both", which="both", direction="out", top=False, right=False
    )
    if cbar:
        cbar = plt.colorbar(pcm, cax=cax, ax=ax)
        cbar.ax.tick_params(which="both", direction="out")
        cbar.ax.yaxis.offsetText.set_horizontalalignment(CBAR_YAX_HALIGN)
        cbar.ax.yaxis.offsetText.set_verticalalignment(CBAR_YAX_VALIGN)
        return pcm, cbar
    else:
        return pcm


def imshow(
    ax,
    x,
    cax=None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    xlabel=r"$x$",
    ylabel=r"$y$",
    cbarlabel=r"$z$",
    **kwargs,
):
    """
    Plot two dimensional data as heatmap with
    :meth:`matplotlib.axes.Axes.imshow`.

    .. deprecated:: 0.0.0.dev2
        Use :func:`mdtools.plot.imshow_new` instead.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : array_like
        2-dimensional array containing the data to plot.
    cax : matplotlib.axes.Axes, optional
        The axes to put the colorbar in.
    xmin : scalar, optional
        Left limit for plotting on x-axis.  The left limit may be
        greater than the right limit, in which case the tick values will
        show up in decreasing order.  Default is ``None``, which means
        set the limit automatically.
    xmax : float, optional
        Right limit for plotting on x-axis.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    cbarlabel : str, optional
        Label for the colorbar.
    kwargs : dict, optional
        Keyword arguments to pass to
        :meth:`matplotlib.axes.Axes.imshow`.  See there for possible
        keyword arguments.

    Returns
    -------
    heatmap : matplotlib.image.AxesImage
        A :class:`matplotlib.image.AxesImage`.
    """
    warnings.warn("Use 'mdtools.plot.imshow_new' instead", DeprecationWarning)

    fontsize_labels = 36
    fontsize_ticks = 32
    tick_length = 10
    tick_pad = 12
    label_pad = 16

    kwargs["cmap"] = kwargs.pop("cmap", "Greys")
    kwargs["vmin"] = kwargs.pop("vmin", np.nanmin(x))
    kwargs["vmax"] = kwargs.pop("vmax", np.nanmax(x))
    heatmap = ax.imshow(x, **kwargs)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(
        which="major",
        direction="out",
        top=False,
        right=False,
        length=tick_length,
        labelsize=fontsize_ticks,
        pad=tick_pad,
    )
    ax.tick_params(
        which="minor",
        direction="out",
        top=False,
        right=False,
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
        pad=tick_pad,
    )

    cbar = plt.colorbar(heatmap, cax=cax, ax=ax)
    cbar.set_label(label=cbarlabel, fontsize=fontsize_labels)
    cbar.ax.yaxis.labelpad = label_pad
    cbar.ax.yaxis.offsetText.set(size=fontsize_ticks)
    cbar.ax.tick_params(
        which="major",
        direction="out",
        length=tick_length,
        labelsize=fontsize_ticks,
    )
    cbar.ax.tick_params(
        which="minor",
        direction="out",
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
    )

    return heatmap


def imshow_new(*args, ax, cbar=True, cax=None, **kwargs):
    """
    Plot two dimensional data as colormap with
    :meth:`matplotlib.axes.Axes.imshow`.

    See :meth:`matplotlib.axes.Axes.imshow` and
    :func:`matplotlib.pyplot.colorbar` for more details.

    Parameters
    ----------
    args : iterable, optional
        Positional arguments to parse to
        :meth:`matplotlib.axes.Axes.imshow`.
    ax : matplotlib.axes.Axes
        The :class:`matplotlib.axes.Axes` to which to draw the image.
    cbar : bool, optional
        If ``True``, create a colorbar and plot it either in the
        :class:`matplotlib.axes.Axes` given by `ax` or `cax`.
    cax : matplotlib.axes.Axes, optional
        If given, the colorbar is placed in this
        :class:`matplotlib.axes.Axes`.  Is ignored if `cbar` is
        ``False``.
    kwargs : dict, optional
        Keyword arguments to parse to
        :meth:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    img : matplotlib.image.AxesImage
       The image.
    cbar : matplotlib.colorbar.Colorbar
       The colorbar.  Only created and returned if `cbar` is ``True``.

    Notes
    -----
    The following defaults of :meth:`matplotlib.axes.Axes.imshow`
    are changed:

       * `rasterized` is set to ``True``.

    In contrast to the default MDTools plotting style, the axis ticks
    are drawn outside of the plot.
    """
    kwargs["rasterized"] = kwargs.pop("rasterized", True)
    img = ax.imshow(*args, **kwargs)
    ax.tick_params(
        axis="both", which="both", direction="out", top=False, right=False
    )
    if cbar:
        cbar = plt.colorbar(img, cax=cax, ax=ax)
        cbar.ax.tick_params(which="both", direction="out")
        cbar.ax.yaxis.offsetText.set_horizontalalignment(CBAR_YAX_HALIGN)
        cbar.ax.yaxis.offsetText.set_verticalalignment(CBAR_YAX_VALIGN)
        return img, cbar
    else:
        return img


def matshow(
    ax, z, cax=None, xlabel=r"$x$", ylabel=r"$y$", cbarlabel=r"$z$", **kwargs
):
    """
    Plot a two dimensional matrix as heatmap with
    :meth:`matplotlib.axes.Axes.matshow`.

    .. deprecated:: 0.0.0.dev2
        Use :func:`mdtools.plot.matshow_new` instead.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    z : array_like
        2-dimensional array.  Note: The grid orientation follows the
        standard matrix convention, i.e. an array `z` with shape
        ``(nrows, ncolumns)`` is plotted with the column number as `x`
        and the row number as `y`.
    cax : matplotlib.axes.Axes, optional
        The axes to put the colorbar in.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    cbarlabel : str, optional
        Label for the colorbar.
    kwargs : dict, optional
        Keyword arguments to pass to
        :meth:`matplotlib.axes.Axes.matshow`.  See there for possible
        keyword arguments.

    Returns
    -------
    heatmap : matplotlib.image.AxesImage
        A :class:`matplotlib.image.AxesImage`.
    """
    warnings.warn("Use 'mdtools.plot.matshow_new' instead", DeprecationWarning)

    fontsize_labels = 36
    fontsize_ticks = 32
    tick_length = 10
    tick_pad = 12
    label_pad = 16

    kwargs["cmap"] = kwargs.pop("cmap", "Greys")
    kwargs["vmin"] = kwargs.pop("vmin", np.nanmin(z))
    kwargs["vmax"] = kwargs.pop("vmax", np.nanmax(z))
    heatmap = ax.matshow(z, **kwargs)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.set_label_position("top")
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(
        which="major",
        direction="out",
        bottom=False,
        right=False,
        length=tick_length,
        labelsize=fontsize_ticks,
        pad=tick_pad,
    )
    ax.tick_params(
        which="minor",
        direction="out",
        top=False,
        right=False,
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
        pad=tick_pad,
    )

    cbar = plt.colorbar(heatmap, cax=cax, ax=ax)
    cbar.set_label(label=cbarlabel, fontsize=fontsize_labels)
    cbar.ax.yaxis.labelpad = label_pad
    cbar.ax.yaxis.offsetText.set(size=fontsize_ticks)
    cbar.ax.tick_params(
        which="major",
        direction="out",
        length=tick_length,
        labelsize=fontsize_ticks,
    )
    cbar.ax.tick_params(
        which="minor",
        direction="out",
        length=0.5 * tick_length,
        labelsize=0.8 * fontsize_ticks,
    )

    return heatmap


def matshow_new(*args, ax, cbar=True, cax=None, **kwargs):
    """
    Plot two dimensional data as colormap with
    :meth:`matplotlib.axes.Axes.matshow`.

    See :meth:`matplotlib.axes.Axes.matshow` and
    :func:`matplotlib.pyplot.colorbar` for more details.

    Parameters
    ----------
    args : iterable, optional
        Positional arguments to parse to
        :meth:`matplotlib.axes.Axes.matshow`.
    ax : matplotlib.axes.Axes
        The :class:`matplotlib.axes.Axes` to which to draw the image.
    cbar : bool, optional
        If ``True``, create a colorbar and plot it either in the
        :class:`matplotlib.axes.Axes` given by `ax` or `cax`.
    cax : matplotlib.axes.Axes, optional
        If given, the colorbar is placed in this
        :class:`matplotlib.axes.Axes`.  Is ignored if `cbar` is
        ``False``.
    kwargs : dict, optional
        Keyword arguments to parse to
        :meth:`matplotlib.axes.Axes.matshow`.

    Returns
    -------
    img : matplotlib.image.AxesImage
       The image.
    cbar : matplotlib.colorbar.Colorbar
       The colorbar.  Only created and returned if `cbar` is ``True``.

    Notes
    -----
    The following defaults of :meth:`matplotlib.axes.Axes.matshow`
    are changed:

       * `rasterized` is set to ``True``.

    In contrast to the default MDTools plotting style, the axis ticks
    are drawn outside of the plot.
    """
    kwargs["rasterized"] = kwargs.pop("rasterized", True)
    img = ax.matshow(*args, **kwargs)
    ax.tick_params(axis="both", which="both", direction="out")
    if cbar:
        cbar = plt.colorbar(img, cax=cax, ax=ax)
        cbar.ax.tick_params(which="both", direction="out")
        cbar.ax.yaxis.offsetText.set_horizontalalignment(CBAR_YAX_HALIGN)
        cbar.ax.yaxis.offsetText.set_verticalalignment(CBAR_YAX_VALIGN)
        return img, cbar
    else:
        return img


def annotate_heatmap(
    im,
    data=None,
    xpos=None,
    ypos=None,
    fmt="{x:.1f}",
    textcolors=("black", "white"),
    threshold=None,
    **kwargs,
):
    """
    Annotate a heatmap.

    See `Creating annotated heatmaps`_ and
    `how to annotate heatmap with text in matplotlib?`_

    .. _Creating annotated heatmaps:
        https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    .. _how to annotate heatmap with text in matplotlib?:
        https://stackoverflow.com/questions/11917547/how-to-annotate-heatmap-with-text-in-matplotlib

    Parameters
    ----------
    im : matplotlib.image.AxesImage or matplotlib.collections.QuadMesh
        The heatmap to be labeled.
    data : array_like, optional
        Data to annotate.  If None, the heatmap's data is used.
    xpos, ypos : array_like, optional
        Arrays containing the x and y positions of the annotations.  If
        ``None``, the text positions are tried to be inferred from
        `data`.  This will only work for heatmaps whose quadrilaterals
        are located at integer positions.
    fmt : str, optional
        The format of the annotations inside the heatmap.  This should
        either use the string format method, e.g. "$ {x:.2f}", or be a
        :class:`matplotlib.ticker.Formatter`.
    textcolors : array_like, optional
        A list or array of two color specifications.  The first is used
        for data below `threshold`, the second for those above.
    threshold : scalar, optional
        Value in data units according to which the colors from
        `textcolors` are applied.  If ``None`` (the default) uses the
        middle of the colormap as separation.
    kwargs : dict, optional
        Keyword arguments to pass to
        :func:`matplotlib.image.AxesImage.axes.text()`.  See there for
        possible keyword arguments.

    Returns
    -------
    texts : list
        List of :class:`matplotlib.text.Text` objects.
    """  # noqa: W505, E501
    if data is None:
        data = im.get_array()
    if xpos is None:
        xpos = range(data.shape[1])  # Cols = x
    if ypos is None:
        ypos = range(data.shape[0])  # Rows = y

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(np.nanmax(data)) / 2.0

    # Set default alignment to center, but allow it to be overwritten by
    # kwargs.
    kw = {"horizontalalignment": "center", "verticalalignment": "center"}
    kw.update(kwargs)

    # Get the formatter in case a string is supplied
    if isinstance(fmt, str):
        fmt = matplotlib.ticker.StrMethodFormatter(fmt)

    # Loop over the data and create a text for each quadrilateral of the
    # heatmap.
    texts = []
    for i, x in enumerate(xpos):
        for j, y in enumerate(ypos):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(x, y, fmt(data[i, j], None), **kw)
            texts.append(text)
    return texts


def sci_notation_tex(x, dec_places=1):
    r"""
    Convert a number to scientific notation in TeX format.

    Parameters
    ----------
    x : scalar
        The number which to convert to scientific notation in TeX
        format.
    dec_places : int
        Number of decimal places.  Must not be negative.

    Returns
    -------
    x_sci : str
        Scientific notation of `x` as raw string in TeX format.

    Notes
    -----
    This code is based on the following two stackoverflow answers:

        * https://stackoverflow.com/a/31453961
        * https://stackoverflow.com/a/29261252

    Examples
    --------
    >>> import mdtools.plot as mdtplt
    >>> mdtplt.sci_notation_tex(5)
    '5.0 \\times 10^{0}'
    >>> mdtplt.sci_notation_tex(5e-2, dec_places=3)
    '5.000 \\times 10^{-2}'
    """
    if dec_places < 0:
        raise ValueError(
            "'dec_places' ({}) must not be negative".format(dec_places)
        )
    x_sci_string = "{num:.{dp:d}e}".format(num=x, dp=dec_places)
    base, exponent = x_sci_string.split("e")
    return r"{b:s} \times 10^{{{e:d}}}".format(b=base, e=int(exponent))
