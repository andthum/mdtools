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


"""General plotting functions on the basis of :mod:`matplotlib.pyplot`"""

import numpy as np
import matplotlib.pyplot as plt
import mdtools as mdt


plt.rc('axes', linewidth=2)
plt.rc('lines', linewidth=2)
plt.rc('lines', markersize=8)
plt.rc('xtick.major', width=2)
plt.rc('xtick.minor', width=2)
plt.rc('ytick.major', width=2)
plt.rc('ytick.minor', width=2)
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preview'] = True
plt.rc('font',**{'family' : 'serif', 'serif' : 'Times'})




# Deprecated ###########################################################
import warnings
import matplotlib
import matplotlib.colors as colors

class MidpointNormalize(colors.Normalize):
    """
    Deprecated!
    
    Class to define your own colorbar normalization. This is e.g. useful
    when you want that a diverging colorbar is centerd at zero. Just
    parse ``norm = mdt.plot.MidpointNormalize(midpoint=0.0)`` to the
    plotting function.
    
    Needs ``import matplotlib.colors as colors``. See
    `Colormap Normalization`_ for more information.
    
    .. _Colormap Normalization: https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
    """
    
    warnings.warn("This class is deprecated. If your version of"
                  " matplotlib is 3.2, consider using"
                  " matplotlib.colors.TwoSlopeNorm instead. If your"
                  " version is 3.1, consider using"
                  " matplotlib.colors.DivergingNorm instead.",
                  DeprecationWarning)
    
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)
    
    def __call__(self, value, clip=None):
        # Including masked values, but ignoring clipping and other edge
        # cases
        result, is_scalar = self.process_value(value)
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.array(np.interp(value, x, y),
                           mask=result.mask,
                           copy=False)
# Deprecated ###########################################################




def plot(ax, x, y, xmin=None, xmax=None, ymin=None, ymax=None,
        logx=False, logy=False, xlabel=r'$x$', ylabel=r'$y$',
        legend_loc='best', **kwargs):
    """
    Plot data to a :class:`matplotlib.axes.Axes` object using
    :func:`matplotlib.axes.Axes.plot`.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : array_like
        One dimensional array containing the x data.
    y : array_like
        Array of the same shape as `x` containing the y data.
    xmin : scalar, optional
        Left limit for plotting on x-axis. The left limit may be greater
        than the right limit, in which case the tick values will show up
        in decreasing order. Default is ``None``, which means set the
        limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis. Default is ``None``.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    logx : bool, optional
        Use logarithmic x scale. Default is ``False``.
    logy : bool, optional
        Same as `logx`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis. Default is ``r'$x$'``.
    ylabel : str, optional
        Label for the y-axis. Default is ``r'$y$'``.
    legend_loc : int or str, optional
        Postion of the legend. See :func:`matplotlib.pyplot.legend` for
        possible arguments. Default is ``'best'``.
    kwargs : dict, optional
        Keyword arguments to pass to :func:`matplotlib.axes.Axes.plot`.
        Look there for possible keyword arguments.
    
    Returns
    -------
    img : list
        A list of :class:`matplotlib.lines.Line2D` objects representing
        the plotted data.
    """
    
    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16
    
    img = ax.plot(x, y, **kwargs)
    if logx:
        ax.set_xscale('log', basex=10, subsx=np.arange(2, 10))
    if logy:
        ax.set_yscale('log', basey=10, subsy=np.arange(2, 10))
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(which='major',
                   direction='in',
                   top=True,
                   right=True,
                   length=tick_length,
                   labelsize=fontsize_ticks,
                   pad=tick_pad)
    ax.tick_params(which='minor',
                   direction='in',
                   top=True,
                   right=True,
                   length=0.5*tick_length,
                   labelsize=0.8*fontsize_ticks,
                   pad=tick_pad)
    if kwargs.get('label') is not None:
        ax.legend(loc=legend_loc,
                  numpoints=1,
                  frameon=False,
                  fontsize=fontsize_legend)
    
    return img




def scatter(ax, x, y, s=None, c=None, xmin=None, xmax=None, ymin=None,
        ymax=None, logx=False, logy=False, xlabel=r'$x$', ylabel=r'$y$',
        legend_loc='best', **kwargs):
    """
    Add a scatter plot to a :class:`matplotlib.axes.Axes` object using
    :func:`matplotlib.axes.Axes.scatter`.
    
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
        The marker color of each individual point. See the matplotlib
        documentation of :func:`matplotlib.axes.Axes.scatter` for more
        information.
    xmin : scalar, optional
        Left limit for plotting on x-axis. The left limit may be greater
        than the right limit, in which case the tick values will show up
        in decreasing order. Default is ``None``, which means set the
        limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis. Default is ``None``.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    logx : bool, optional
        Use logarithmic x scale. Default is ``False``.
    logy : bool, optional
        Same as `logx`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis. Default is ``r'$x$'``.
    ylabel : str, optional
        Label for the y-axis. Default is ``r'$y$'``.
    legend_loc : int or str, optional
        Postion of the legend. See :func:`matplotlib.pyplot.legend` for
        possible arguments. Default is ``'best'``.
    kwargs : dict, optional
        Keyword arguments to pass to :func:`matplotlib.axes.Axes.scatter`.
        Look there for possible keyword arguments.
    
    Returns
    -------
    img : matplotlib.collections.PathCollection
        A :class:`matplotlib.collections.PathCollection`.
    """
    
    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16
    
    if kwargs.get('cmap') is None:
        kwargs['cmap'] = 'Greys'
    if kwargs.get('vmin') is None:
        kwargs['vmin'] = np.nanmin(c)
    if kwargs.get('vmax') is None:
        kwargs['vmax'] = np.nanmax(c)
    
    img = ax.scatter(x=x, y=y, s=s, c=c, **kwargs)
    if logx:
        ax.set_xscale('log', basex=10, subsx=np.arange(2, 10))
    if logy:
        ax.set_yscale('log', basey=10, subsy=np.arange(2, 10))
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(which='major',
                   direction='in',
                   top=True,
                   right=True,
                   length=tick_length,
                   labelsize=fontsize_ticks,
                   pad=tick_pad)
    ax.tick_params(which='minor',
                   direction='in',
                   top=True,
                   right=True,
                   length=0.5*tick_length,
                   labelsize=0.8*fontsize_ticks,
                   pad=tick_pad)
    if kwargs.get('label') is not None:
        ax.legend(loc=legend_loc,
                  numpoints=1,
                  frameon=False,
                  fontsize=fontsize_legend)
    
    return img




def errorbar(ax, x, y, xerr=None, yerr=None, xmin=None, xmax=None,
        ymin=None, ymax=None, logx=False, logy=False, xlabel=r'$x$',
        ylabel=r'$y$', legend_loc='best', **kwargs):
    """
    Plot data to a :class:`matplotlib.axes.Axes` object with errorbars
    using :func:`matplotlib.axes.Axes.errorbar`.
    
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
        Left limit for plotting on x-axis. The left limit may be greater
        than the right limit, in which case the tick values will show up
        in decreasing order. Default is ``None``, which means set the
        limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis. Default is ``None``.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    logx : bool, optional
        Use logarithmic x scale. Default is ``False``.
    logy : bool, optional
        Same as `logx`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis. Default is ``r'$x$'``.
    ylabel : str, optional
        Label for the y-axis. Default is ``r'$y$'``.
    legend_loc : int or str, optional
        Postion of the legend. See :func:`matplotlib.pyplot.legend` for
        possible arguments. Default is ``'best'``.
    kwargs : dict, optional
        Keyword arguments to pass to
        :func:`matplotlib.axes.Axes.errorbar`. Look there for possible
        keyword arguments.
    
    Returns
    -------
    img : ErrorbarContainer
        A :class:`matplotlib.container.ErrorbarContainer`.
    """
    
    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16
    
    img = ax.errorbar(x=x,
                      y=y,
                      xerr=xerr,
                      yerr=yerr,
                      capsize=6,
                      capthick=2,
                      **kwargs)
    if logx:
        ax.set_xscale('log', basex=10, subsx=np.arange(2, 10))
    if logy:
        ax.set_yscale('log', basey=10, subsy=np.arange(2, 10))
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(which='major',
                   direction='in',
                   top=True,
                   right=True,
                   length=tick_length,
                   labelsize=fontsize_ticks,
                   pad=tick_pad)
    ax.tick_params(which='minor',
                   direction='in',
                   top=True,
                   right=True,
                   length=0.5*tick_length,
                   labelsize=0.8*fontsize_ticks,
                   pad=tick_pad)
    if kwargs.get('label') is not None:
        ax.legend(loc=legend_loc,
                  numpoints=1,
                  frameon=False,
                  fontsize=fontsize_legend)
    
    return img




def hist(ax, x, xmin=None, xmax=None, ymin=None, ymax=None,
         xlabel=r'$x$', ylabel=r'$y$', legend_loc='best', **kwargs):
    """
    Plot a histogram to a :class:`matplotlib.axes.Axes` object using
    :func:`matplotlib.axes.Axes.hist`.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : array_like
        The data.
    xmin : scalar, optional
        Left limit for plotting on x-axis. The left limit may be greater
        than the right limit, in which case the tick values will show up
        in decreasing order. Default is ``None``, which means set the
        limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis. Default is ``None``.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis. Default is ``r'$x$'``.
    ylabel : str, optional
        Label for the y-axis. Default is ``r'$y$'``.
    legend_loc : int or str, optional
        Postion of the legend. See :func:`matplotlib.pyplot.legend` for
        possible arguments. Default is ``'best'``.
    kwargs : dict, optional
        Keyword arguments to pass to :func:`matplotlib.axes.Axes.hist`.
        Look there for possible keyword arguments.
    
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
    ax.tick_params(which='major',
                   direction='in',
                   top=True,
                   right=True,
                   length=tick_length,
                   labelsize=fontsize_ticks,
                   pad=tick_pad)
    ax.tick_params(which='minor',
                   direction='in',
                   top=True,
                   right=True,
                   length=0.5*tick_length,
                   labelsize=0.8*fontsize_ticks,
                   pad=tick_pad)
    if kwargs.get('label') is not None:
        ax.legend(loc=legend_loc,
                  numpoints=1,
                  frameon=False,
                  fontsize=fontsize_legend)
    
    return img




def hlines(ax, y, start, stop, xmin=None, xmax=None, ymin=None,
        ymax=None, legend_loc='best', **kwargs):
    """
    Plot horizontal lines at each `y` from `start` to `stop` into a
    :class:`matplotlib.axes.Axes` object using
    :func:`matplotlib.axes.Axes.hlines`.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    y : scalar or array_like
        y-indices where to plot the lines.
    start, stop : scalar or array_like
        Respective beginning and end of each line. If scalars are
        provided, all lines will have same length.
    xmin : scalar, optional
        Left limit for plotting on x-axis. The left limit may be greater
        than the right limit, in which case the tick values will show up
        in decreasing order. Default is ``None``, which means set the
        limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis. Default is ``None``.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    legend_loc : int or str, optional
        Postion of the legend. See :func:`matplotlib.pyplot.legend` for
        possible arguments. Default is ``'best'``.
    kwargs : dict, optional
        Keyword arguments to pass to :func:`matplotlib.axes.Axes.hlines`.
        Look there for possible keyword arguments.
    
    Returns
    -------
    img : matplotlib.collections.LineCollection
        A :class:`matplotlib.collections.LineCollection`.
    """
    
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    
    img = ax.hlines(y=y, xmin=start, xmax=stop, **kwargs)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(which='major',
                   direction='in',
                   top=True,
                   right=True,
                   length=tick_length,
                   labelsize=fontsize_ticks,
                   pad=tick_pad)
    ax.tick_params(which='minor',
                   direction='in',
                   top=True,
                   right=True,
                   length=0.5*tick_length,
                   labelsize=0.8*fontsize_ticks,
                   pad=tick_pad)
    if kwargs.get('label') is not None:
        ax.legend(loc=legend_loc,
                  numpoints=1,
                  frameon=False,
                  fontsize=fontsize_legend)
    
    return img




def vlines(ax, x, start, stop, xmin=None, xmax=None, ymin=None,
        ymax=None, legend_loc='best', **kwargs):
    """
    Plot vertical lines at each `x` from `start` to `stop` into a
    :class:`matplotlib.axes.Axes` object using
    :func:`matplotlib.axes.Axes.vlines`.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : scalar or array_like
        x-indices where to plot the lines.
    start, stop : scalar or array_like
        Respective beginning and end of each line. If scalars are
        provided, all lines will have same length.
    xmin : scalar, optional
        Left limit for plotting on x-axis. The left limit may be greater
        than the right limit, in which case the tick values will show up
        in decreasing order. Default is ``None``, which means set the
        limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis. Default is ``None``.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    legend_loc : int or str, optional
        Postion of the legend. See :func:`matplotlib.pyplot.legend` for
        possible arguments. Default is ``'best'``.
    kwargs : dict, optional
        Keyword arguments to pass to :func:`matplotlib.axes.Axes.vlines`.
        Look there for possible keyword arguments.
    
    Returns
    -------
    img : matplotlib.collections.LineCollection
        A :class:`matplotlib.collections.LineCollection`.
    """
    
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    
    img = ax.vlines(x=x, ymin=start, ymax=stop, **kwargs)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(which='major',
                   direction='in',
                   top=True,
                   right=True,
                   length=tick_length,
                   labelsize=fontsize_ticks,
                   pad=tick_pad)
    ax.tick_params(which='minor',
                   direction='in',
                   top=True,
                   right=True,
                   length=0.5*tick_length,
                   labelsize=0.8*fontsize_ticks,
                   pad=tick_pad)
    if kwargs.get('label') is not None:
        ax.legend(loc=legend_loc,
                  numpoints=1,
                  frameon=False,
                  fontsize=fontsize_legend)
    
    return img




def plot_2nd_xaxis(ax, x, y, xmin=None, xmax=None, xlabel=r'$x2$',
        legend_loc='best', **kwargs):
    """
    Create a twin :class:`~matplotlib.axes.Axes` of an existing
    :class:`matplotlib.axes.Axes` object sharing the x-axis using
    :func:`matplotlib.axes.Axes.twiny` and plot data to that second
    x-axis using :func:`matplotlib.axes.Axes.plot`.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : array_like
        One dimensional array containing the x data.
    y : array_like
        Array of the same shape as `x` containing the y data.
    xmin : scalar, optional
        Left limit for plotting on the secondary x-axis. The left limit
        may be greater than the right limit, in which case the tick
        values will show up in decreasing order. Default is ``None``,
        which means set the limit automatically.
    xmax : scalar, optional
        Right limit for plotting on the secondary x-axis. Default is
        ``None``.
    xlabel : str, optional
        Label for the secondary x-axis. Default is ``r'$x2$'``.
    legend_loc : int or str, optional
        Postion of the legend. See :func:`matplotlib.pyplot.legend` for
        possible arguments. Default is ``'best'``.
    kwargs : dict, optional
        Keyword arguments to pass to :func:`matplotlib.axes.Axes.plot`.
        Look there for possible keyword arguments.
    
    Returns
    -------
    img : list
        A list of :class:`matplotlib.lines.Line2D` objects representing
        the plotted data.
    ax2 : matplotlib.axes.Axes
        The twin axis.
    """
    
    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    label_pad = 16
    
    if kwargs.get('color') is None:
        kwargs['color'] = 'black'
    
    ax2 = ax.twiny()
    img = ax2.plot(x, y, **kwargs)
    ax2.set_xlim(xmin, xmax)
    ax2.set_xlabel(xlabel=xlabel,
                   color=kwargs.get('color'),
                   fontsize=fontsize_labels)
    ax2.xaxis.labelpad = label_pad
    ax2.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax2.tick_params(axis='x',
                    which='major',
                    direction='in',
                    labelcolor=kwargs.get('color'),
                    length=tick_length,
                    labelsize=fontsize_ticks)
    ax2.tick_params(axis='x',
                    which='minor',
                    direction='in',
                    labelcolor=kwargs.get('color'),
                    length=0.5*tick_length,
                    labelsize=0.8*fontsize_ticks)
    if kwargs.get('label') is not None:
        ax2.legend(loc=legend_loc,
                   numpoints=1,
                   frameon=False,
                   fontsize=fontsize_legend)
    
    return img, ax2




def plot_2nd_yaxis(ax, x, y, ymin=None, ymax=None, ylabel=r'$y2$',
        legend_loc='best', **kwargs):
    """
    Create a twin :class:`~matplotlib.axes.Axes` of an existing
    :class:`matplotlib.axes.Axes` object sharing the x-axis using
    :func:`matplotlib.axes.Axes.twinx` and plot data to that second
    y-axis using :func:`matplotlib.axes.Axes.plot`.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : array_like
        One dimensional array containing the x data.
    y : array_like
        Array of the same shape as `x` containing the y data.
    ymin : scalar, optional
        Left limit for plotting on the secondary y-axis. The left limit
        may be greater than the right limit, in which case the tick
        values will show up in decreasing order. Default is ``None``,
        which means set the limit automatically.
    ymax : scalar, optional
        Right limit for plotting on the secondary y-axis. Default is
        ``None``.
    ylabel : str, optional
        Label for the secondary y-axis. Default is ``r'$y2$'``.
    legend_loc : int or str, optional
        Postion of the legend. See :func:`matplotlib.pyplot.legend` for
        possible arguments. Default is ``'best'``.
    kwargs : dict, optional
        Keyword arguments to pass to :func:`matplotlib.axes.Axes.plot`.
        Look there for possible keyword arguments.
    
    Returns
    -------
    img : list
        A list of :class:`matplotlib.lines.Line2D` objects representing
        the plotted data.
    ax2 : matplotlib.axes.Axes
        The twin axis.
    """
    
    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16
    
    if kwargs.get('color') is None:
        kwargs['color'] = 'black'
    
    ax2 = ax.twinx()
    img = ax2.plot(x, y, **kwargs)
    ax2.set_ylim(ymin, ymax)
    ax2.set_ylabel(ylabel=ylabel,
                   color=kwargs.get('color'),
                   fontsize=fontsize_labels)
    ax2.yaxis.labelpad = label_pad
    ax2.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax2.tick_params(axis='y',
                    which='major',
                    direction='in',
                    labelcolor=kwargs.get('color'),
                    length=tick_length,
                    labelsize=fontsize_ticks,
                    pad=tick_pad)
    ax2.tick_params(axis='y',
                    which='minor',
                    direction='in',
                    labelcolor=kwargs.get('color'),
                    length=0.5*tick_length,
                    labelsize=0.8*fontsize_ticks,
                    pad=tick_pad)
    if kwargs.get('label') is not None:
        ax2.legend(loc=legend_loc,
                   numpoints=1,
                   frameon=False,
                   fontsize=fontsize_legend)
    
    return img, ax2




def fill_between(ax, x, y1, y2=0, xmin=None, xmax=None, ymin=None,
        ymax=None, logx=False, logy=False, xlabel=r'$x$', ylabel=r'$y$',
        legend_loc='best', **kwargs):
    """
    Fill the area between two curves `y1` and `y2` using
    :func:`matplotlib.axes.Axes.fill_between`
    
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
        Left limit for plotting on x-axis. The left limit may be greater
        than the right limit, in which case the tick values will show up
        in decreasing order. Default is ``None``, which means set the
        limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis. Default is ``None``.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    logx : bool, optional
        Use logarithmic x scale. Default is ``False``.
    logy : bool, optional
        Same as `logx`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis. Default is ``r'$x$'``.
    ylabel : str, optional
        Label for the y-axis. Default is ``r'$y$'``.
    legend_loc : int or str, optional
        Postion of the legend. See :func:`matplotlib.pyplot.legend` for
        possible arguments. Default is ``'best'``.
    kwargs : dict, optional
        Keyword arguments to pass to
        :func:`matplotlib.axes.Axes.fill_between`. Look there for
        possible keyword arguments.
    
    Returns
    -------
    img : matplotlib.collections.PolyCollection
        A :class:`matplotlib.collections.PolyCollection`.
    """
    
    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16
    
    img = ax.fill_between(x=x, y1=y1, y2=y2, **kwargs)
    if logx:
        ax.set_xscale('log', basex=10, subsx=np.arange(2, 10))
    if logy:
        ax.set_yscale('log', basey=10, subsx=np.arange(2, 10))
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(which='major',
                   direction='in',
                   top=True,
                   right=True,
                   length=tick_length,
                   labelsize=fontsize_ticks,
                   pad=tick_pad)
    ax.tick_params(which='minor',
                   direction='in',
                   top=True,
                   right=True,
                   length=0.5*tick_length,
                   labelsize=0.8*fontsize_ticks,
                   pad=tick_pad)
    if kwargs.get('label') is not None:
        ax.legend(loc=legend_loc,
                  numpoints=1,
                  frameon=False,
                  fontsize=fontsize_legend)
    
    return img




def fill_betweenx(ax, y, x1, x2=0, xmin=None, xmax=None, ymin=None,
        ymax=None, logx=False, logy=False, xlabel=r'$x$', ylabel=r'$y$',
        legend_loc='best', **kwargs):
    """
    Fill the area between two vertical curves `x1` and `x2` using
    :func:`matplotlib.axes.Axes.fill_betweenx`. The two curves `x1` and
    `x2` are functions of `y`.
    
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
        Left limit for plotting on x-axis. The left limit may be greater
        than the right limit, in which case the tick values will show up
        in decreasing order. Default is ``None``, which means set the
        limit automatically.
    xmax : scalar, optional
        Right limit for plotting on x-axis. Default is ``None``.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    logx : bool, optional
        Use logarithmic x scale. Default is ``False``.
    logy : bool, optional
        Same as `logx`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis. Default is ``r'$x$'``.
    ylabel : str, optional
        Label for the y-axis. Default is ``r'$y$'``.
    legend_loc : int or str, optional
        Postion of the legend. See :func:`matplotlib.pyplot.legend` for
        possible arguments. Default is ``'best'``.
    kwargs : dict, optional
        Keyword arguments to pass to
        :func:`matplotlib.axes.Axes.fill_betweenx`. Look there for
        possible keyword arguments.
    
    Returns
    -------
    img : matplotlib.collections.PolyCollection
        A :class:`matplotlib.collections.PolyCollection`.
    """
    
    fontsize_labels = 36
    fontsize_ticks = 32
    fontsize_legend = 28
    tick_length = 10
    tick_pad = 12
    label_pad = 16
    
    img = ax.fill_betweenx(y=y, x1=x1, x2=x2, **kwargs)
    if logx:
        ax.set_xscale('log', basex=10, subsx=np.arange(2, 10))
    if logy:
        ax.set_yscale('log', basey=10, subsx=np.arange(2, 10))
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(which='major',
                   direction='in',
                   top=True,
                   right=True,
                   length=tick_length,
                   labelsize=fontsize_ticks,
                   pad=tick_pad)
    ax.tick_params(which='minor',
                   direction='in',
                   top=True,
                   right=True,
                   length=0.5*tick_length,
                   labelsize=0.8*fontsize_ticks,
                   pad=tick_pad)
    if kwargs.get('label') is not None:
        ax.legend(loc=legend_loc,
                  numpoints=1,
                  frameon=False,
                  fontsize=fontsize_legend)
    
    return img




def pcolormesh(ax, x, y, z, cax=None, xmin=None, xmax=None, ymin=None,
        ymax=None, xlabel=r'$x$', ylabel=r'$y$', cbarlabel=r'$z$',
        **kwargs):
    """
    Plot two dimensional data as heatmap with
    :func:`matplotlib.axes.Axes.pcolormesh`
    
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
        Left limit for plotting on x-axis. The left limit may be greater
        than the right limit, in which case the tick values will show up
        in decreasing order. Default is ``None``, which means set the
        left limit to ``np.nanmin(x)``.
    xmax : float, optional
        Right limit for plotting on x-axis. Default is ``None``, which
        means set the left limit to ``np.nanmax(x)``.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis. Default is ``r'$x$'``.
    ylabel : str, optional
        Label for the y-axis. Default is ``r'$y$'``.
    cbarlabel : str, optional
        Label for the colorbar. Default is ``r'$z$'``.
    kwargs : dict, optional
        Keyword arguments to pass to
        :func:`matplotlib.axes.Axes.pcolormesh`. See there for possible
        keyword arguments.
    
    Returns
    -------
    heatmap : matplotlib.collections.QuadMesh
        A :class:`matplotlib.collections.QuadMesh`.
    """
    
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
    if kwargs.get('cmap') is None:
        kwargs['cmap'] = 'Greys'
    if kwargs.get('vmin') is None:
        kwargs['vmin'] = np.nanmin(z)
    if kwargs.get('vmax') is None:
        kwargs['vmax'] = np.nanmax(z)
    
    heatmap = ax.pcolormesh(x, y, z, **kwargs)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(which='major',
                   direction='out',
                   top=False,
                   right=False,
                   length=tick_length,
                   labelsize=fontsize_ticks,
                   pad=tick_pad)
    ax.tick_params(which='minor',
                   direction='out',
                   top=False,
                   right=False,
                   length=0.5*tick_length,
                   labelsize=0.8*fontsize_ticks,
                   pad=tick_pad)
    
    cbar = plt.colorbar(heatmap, cax=cax, ax=ax)
    cbar.set_label(label=cbarlabel, fontsize=fontsize_labels)
    cbar.ax.yaxis.labelpad = label_pad
    cbar.ax.yaxis.offsetText.set(size=fontsize_ticks)
    cbar.ax.tick_params(which='major',
                        direction='out',
                        length=tick_length,
                        labelsize=fontsize_ticks)
    cbar.ax.tick_params(which='minor',
                        direction='out',
                        length=0.5*tick_length,
                        labelsize=0.8*fontsize_ticks)
    
    return heatmap




def imshow(ax, x, cax=None, xmin=None, xmax=None, ymin=None, ymax=None,
        xlabel=r'$x$', ylabel=r'$y$', cbarlabel=r'$z$', **kwargs):
    """
    Plot two dimensional data as heatmap with
    :func:`matplotlib.axes.Axes.imshow`
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    x : array_like
        2-dimensional array containing the data to plot.
    cax : matplotlib.axes.Axes, optional
        The axes to put the colorbar in.
    xmin : scalar, optional
        Left limit for plotting on x-axis. The left limit may be greater
        than the right limit, in which case the tick values will show up
        in decreasing order.
    xmax : float, optional
        Right limit for plotting on x-axis.
    ymin, ymax : scalar, optional
        Same as `xmin` and `xmax`, but for the y-axis.
    xlabel : str, optional
        Label for the x-axis. Default is ``r'$x$'``.
    ylabel : str, optional
        Label for the y-axis. Default is ``r'$y$'``.
    cbarlabel : str, optional
        Label for the colorbar. Default is ``r'$z$'``.
    kwargs : dict, optional
        Keyword arguments to pass to
        :func:`matplotlib.axes.Axes.imshow`. See there for possible
        keyword arguments.
    
    Returns
    -------
    heatmap : matplotlib.image.AxesImage
        A :class:`matplotlib.image.AxesImage`.
    """
    
    fontsize_labels = 36
    fontsize_ticks = 32
    tick_length = 10
    tick_pad = 12
    label_pad = 16
    
    if kwargs.get('cmap') is None:
        kwargs['cmap'] = 'Greys'
    if kwargs.get('vmin') is None:
        kwargs['vmin'] = np.nanmin(x)
    if kwargs.get('vmax') is None:
        kwargs['vmax'] = np.nanmax(x)
    
    heatmap = ax.imshow(x, **kwargs)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(which='major',
                   direction='out',
                   top=False,
                   right=False,
                   length=tick_length,
                   labelsize=fontsize_ticks,
                   pad=tick_pad)
    ax.tick_params(which='minor',
                   direction='out',
                   top=False,
                   right=False,
                   length=0.5*tick_length,
                   labelsize=0.8*fontsize_ticks,
                   pad=tick_pad)
    
    cbar = plt.colorbar(heatmap, cax=cax, ax=ax)
    cbar.set_label(label=cbarlabel, fontsize=fontsize_labels)
    cbar.ax.yaxis.labelpad = label_pad
    cbar.ax.yaxis.offsetText.set(size=fontsize_ticks)
    cbar.ax.tick_params(which='major',
                        direction='out',
                        length=tick_length,
                        labelsize=fontsize_ticks)
    cbar.ax.tick_params(which='minor',
                        direction='out',
                        length=0.5*tick_length,
                        labelsize=0.8*fontsize_ticks)
    
    return heatmap




def matshow(ax, z, cax=None, xlabel=r'$x$', ylabel=r'$y$',
        cbarlabel=r'$z$', **kwargs):
    """
    Plot a two dimensional matrix as heatmap with
    :func:`matplotlib.axes.Axes.matshow`
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to.
    z : array_like
        2-dimensional array. Note: The grid orientation follows the
        standard matrix convention, i.e. an array `z` with shape
        ``(nrows, ncolumns)`` is plotted with the column number as `x`
        and the row number as `y`.
    cax : matplotlib.axes.Axes, optional
        The axes to put the colorbar in.
    xlabel : str, optional
        Label for the x-axis. Default is ``r'$x$'``.
    ylabel : str, optional
        Label for the y-axis. Default is ``r'$y$'``.
    cbarlabel : str, optional
        Label for the colorbar. Default is ``r'$z$'``.
    kwargs : dict, optional
        Keyword arguments to pass to
        :func:`matplotlib.axes.Axes.matshow`. See there for possible
        keyword arguments.
    
    Returns
    -------
    heatmap : matplotlib.image.AxesImage
        A :class:`matplotlib.image.AxesImage`.
    """
    
    fontsize_labels = 36
    fontsize_ticks = 32
    tick_length = 10
    tick_pad = 12
    label_pad = 16
    
    if (kwargs.get('cmap') is None):
        kwargs['cmap'] = 'Greys'
    if (kwargs.get('vmin') is None):
        kwargs['vmin'] = np.nanmin(z)
    if (kwargs.get('vmax') is None):
        kwargs['vmax'] = np.nanmax(z)
    
    heatmap = ax.matshow(z, **kwargs)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    ax.xaxis.set_label_position('top')
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.yaxis.offsetText.set_fontsize(fontsize_ticks)
    ax.tick_params(which='major',
                   direction='out',
                   bottom=False,
                   right=False,
                   length=tick_length,
                   labelsize=fontsize_ticks,
                   pad=tick_pad)
    ax.tick_params(which='minor',
                   direction='out',
                   top=False,
                   right=False,
                   length=0.5*tick_length,
                   labelsize=0.8*fontsize_ticks,
                   pad=tick_pad)
    
    cbar = plt.colorbar(heatmap, cax=cax, ax=ax)
    cbar.set_label(label=cbarlabel, fontsize=fontsize_labels)
    cbar.ax.yaxis.labelpad = label_pad
    cbar.ax.yaxis.offsetText.set(size=fontsize_ticks)
    cbar.ax.tick_params(which='major',
                        direction='out',
                        length=tick_length,
                        labelsize=fontsize_ticks)
    cbar.ax.tick_params(which='minor',
                        direction='out',
                        length=0.5*tick_length,
                        labelsize=0.8*fontsize_ticks)
    
    return heatmap




def annotate_heatmap(im, data, xpos, ypos, fmt='{x:.1f}',
        textcolors=["black", "white"], threshold=None, **kwargs):
    """
    A function to annotate a heatmap. See `Creating annotated heatmaps`_
    and `how to annotate heatmap with text in matplotlib?`_
    
    .. _Creating annotated heatmaps: https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    .. _how to annotate heatmap with text in matplotlib?: https://stackoverflow.com/questions/11917547/how-to-annotate-heatmap-with-text-in-matplotlib

    Parameters
    ----------
    im : matplotlib.image.AxesImage or matplotlib.collections.QuadMesh
        The AxesImage to be labeled.
    data : array_like
        Data to annotate.
    xpos : array_like
        Array containing the x positions of the annotations.
    ypos : array_like
        Array containing the y positions of the annotations.
    fmt : str, optional
        The format of the annotations inside the heatmap. This should
        either use the string format method, e.g. "$ {x:.2f}", or be a
        :class:`matplotlib.ticker.Formatter`.
    textcolors : array_like, optional
        A list or array of two color specifications. The first is used
        for values below a threshold, the second for those above.
    threshold : scalar, optional
        Value in data units according to which the colors from
        textcolors are applied. If ``None`` (the default) uses the
        middle of the colormap as separation.
    kwargs : dict, optional
        Keyword arguments to pass to
        :func:`matplotlib.image.AxesImage.axes.text()`. See there for
        possible keyword arguments.
    
    Returns
    -------
    texts : list
        List of :class:`matplotlib.text.Text` objects.
    """
    
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(np.nanmax(data)) / 2.0
    
    # Set default alignment to center, but allow it to be overwritten by
    # kwargs.
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              size=14)
    kw.update(kwargs)
    
    # Get the formatter in case a string is supplied
    if isinstance(fmt, str):
        fmt = matplotlib.ticker.StrMethodFormatter(fmt)
    
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i, x in enumerate(xpos):
        for j, y in enumerate(ypos):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(x, y, fmt(data[i, j], None), **kw)
            texts.append(text)
    
    return texts
