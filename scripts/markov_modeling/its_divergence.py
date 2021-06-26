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




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import mdtools as mdt




# Input parameters
nstates = 4
nlags = 10
show_n_eigvals = nstates
show_n_its = nstates - 1
outfile = "its_divergence.pdf"




# Set up model (row-stochastic transition matrix)
lags = np.geomspace(1, 2**(nlags-1), nlags, dtype=int)

tm = np.random.rand(nstates, nstates)
tm /= tm.sum(axis=1)[:,None]
if not np.all(np.isclose(tm.sum(axis=1), 1)):
    raise ValueError("tm is not row-stochastic")

evals = np.zeros((nstates, nlags), dtype=np.float)
its = np.zeros((nstates-1, nlags), dtype=np.float)




# Propagate model
for i, lag in enumerate(lags):
    evals[:,i], _ = np.linalg.eig(np.linalg.matrix_power(tm, lag))
    ix = np.argsort(np.abs(evals[:,i]))[::-1]
    evals[:,i] = evals[:,i][ix]
    its[:,i] = -lag / np.log(np.abs(evals[:,i][1:]))




# Plot eigenvalues and implied timescales
mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    
    # Transition matrix
    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    mdt.plot.pcolormesh(ax=axis,
                        x=np.arange(1, nstates+2)-0.5,
                        y=np.arange(1, nstates+2)-0.5,
                        z=tm,
                        xmin=0.5,
                        xmax=nstates+0.5,
                        ymin=0.5,
                        ymax=nstates+0.5,
                        xlabel=r'State $j$',
                        ylabel=r'State $i$',
                        cbarlabel=r'Transition probability $T_{ij}$')
    
    axis.invert_yaxis()
    axis.xaxis.set_label_position('top')
    axis.xaxis.labelpad = 22
    axis.xaxis.tick_top()
    axis.tick_params(axis='x', which='both', pad=6)
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    
    
    
    # Eigenvalues linear scale
    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.ticklabel_format(axis='x',
                          style='scientific',
                          scilimits=(0,0),
                          useOffset=False)
    
    for i in range(show_n_eigvals):
        mdt.plot.plot(ax=axis,
                      x=lags,
                      y=evals[i],
                      xlabel=r'Lag time $\tau$ / steps',
                      ylabel=r'Eigenvalue $\lambda_i$',
                      label=r'$\lambda_{}$'.format(i+1),
                      marker='o')
    mdt.plot.plot(
        ax=axis,
        x=lags,
        y=abs(evals[1][0])**lags,
        xlabel=r'Lag time $\tau$ / steps',
        ylabel=r'Eigenvalue $\lambda_i$',
        label=r'${:.2f}^\tau$'.format(abs(evals[1][0])),
        marker='x',
        color='black')
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    
    # Eigenvalues loglog scale
    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.ticklabel_format(axis='both',
                          style='scientific',
                          scilimits=(0,0),
                          useOffset=False)
    
    for i in range(show_n_eigvals):
        mdt.plot.plot(ax=axis,
                      x=lags,
                      y=np.abs(evals[i]),
                      ymin=np.min(np.abs(evals[evals!=0]))/2,
                      ymax=5,
                      logx=True,
                      logy=True,
                      xlabel=r'Lag time $\tau$ / steps',
                      ylabel=r'Eigenvalue $|\lambda_i|$',
                      label=r'$\lambda_{}$'.format(i+1),
                      legend_loc='lower left',
                      marker='o')
    mdt.plot.plot(
        ax=axis,
        x=lags,
        y=abs(evals[1][0])**lags,
        ymin=np.min(np.abs(evals[evals!=0]))/2,
        ymax=5,
        xlabel=r'Lag time $\tau$ / steps',
        ylabel=r'Eigenvalue $|\lambda_i|$',
        label=r'${:.2f}^\tau$'.format(abs(evals[1][0])),
        legend_loc='lower left',
        marker='x',
        color='black')
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    
    
    
    # Timescales linear scale
    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.ticklabel_format(axis='both',
                          style='scientific',
                          scilimits=(0,0),
                          useOffset=False)
    axis.plot([], [])
    
    for i in range(show_n_its):
        mdt.plot.plot(
            ax=axis,
            x=lags,
            y=its[i],
            xlabel=r'Lag time $\tau$ / steps',
            ylabel=r'Timescale $t_i$ / steps',
            label=r'$t_{}$'.format(i+2),
            legend_loc='upper left',
            marker='o')
    mdt.plot.plot(
        ax=axis,
        x=lags,
        y=-lags/np.log(abs(evals[1][0])**lags),
        xlabel=r'Lag time $\tau$ / steps',
        ylabel=r'Timescale $t_i$ / steps',
        label=r'$-\frac{\tau}{\ln({' + str('%.2f' %abs(evals[1][0])) + r'}^\tau)}$',
        legend_loc='upper left',
        linestyle='--',
        marker='x',
        color='black')
    mdt.plot.plot(
        ax=axis,
        x=lags,
        y=0.04*lags,
        xlabel=r'Lag time $\tau$ / steps',
        ylabel=r'Timescale $t_i$ / steps',
        label=r'$0.04\tau$',
        legend_loc='upper left',
        marker='+',
        color='black')
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    
    # Timescales loglog scale
    fig, axis = plt.subplots(figsize=(11.69, 8.27),  # DIN A4 landscape in inches
                             frameon=False,
                             clear=True,
                             tight_layout=True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.ticklabel_format(axis='both',
                          style='scientific',
                          scilimits=(0,0),
                          useOffset=False)
    axis.plot([], [])
    
    for i in range(show_n_its):
        mdt.plot.plot(
            ax=axis,
            x=lags,
            y=its[i],
            logx=True,
            logy=True,
            xlabel=r'Lag time $\tau$ / steps',
            ylabel=r'Timescale $t_i$ / steps',
            label=r'$t_{}$'.format(i+2),
            legend_loc='upper left',
            marker='o')
    mdt.plot.plot(
        ax=axis,
        x=lags,
        y=-lags/np.log(abs(evals[1][0])**lags),
        xlabel=r'Lag time $\tau$ / steps',
        ylabel=r'Timescale $t_i$ / steps',
        label=r'$-\frac{\tau}{\ln({' + str('%.2f' %abs(evals[1][0])) + r'}^\tau)}$',
        legend_loc='upper left',
        linestyle='--',
        marker='x',
        color='black')
    mdt.plot.plot(
        ax=axis,
        x=lags,
        y=0.04*lags,
        xlabel=r'Lag time $\tau$ / steps',
        ylabel=r'Timescale $t_i$ / steps',
        label=r'$0.04\tau$',
        legend_loc='upper left',
        marker='+',
        color='black')
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()
