#!/usr/bin/env python3

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
Test plots to visualize the MDTools plotting style.

Plotting code is taken from
https://matplotlib.org/stable/tutorials/introductory/sample_plots.html
"""


# Third-party libraries
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


np.random.seed(19680801)
print("Matplotlib version:", matplotlib.__version__)
print("Matplotlib backend:", matplotlib.get_backend())
if matplotlib.__version__ < "3.4":
    raise ValueError("Requires at least matplotlib version 3.4")

style_file = "../src/mdtools/pkg_data/mdtools.mplstyle"
for style in ("default", "classic", style_file):
    plt.style.use(["default", style])
    print("Current style:", style)
    if style == "default":
        fname = "default.pdf"
    elif style == "classic":
        fname = "classic.pdf"
    elif style == style_file:
        fname = "mdtools.pdf"
    else:
        fname = style + ".pdf"
    with PdfPages(fname) as pdf:
        # Line plot
        x = np.arange(-1.5, 1.5, 0.01)
        y_sin = 1 + 2 * np.sin(2 * np.pi * (x - 0.5))
        y_cos = 1 + 2 * np.cos(2 * np.pi * (x - 0.5))
        fig, ax = plt.subplots(clear=True)
        ax.plot(x, y_sin, label="sin")
        ax.plot(x, y_cos, label="cos")
        ax.set(
            xlabel=r"x-label $x$",
            ylabel=r"y-label $y$",
            title=r"Ax title: $y = 1 + 2 \sin(2\pi(x-\frac{1}{2}))$",
        )
        ax.legend(title="Legend")
        ax.grid()
        pdf.savefig()
        plt.close()

        # Errorbars
        x = np.arange(-1.5, 1.5, 0.1)
        y = 1 + 2 * np.sin(2 * np.pi * (x - 0.5))
        yerr = 0.1 + 0.2 * np.cos(x)
        fig, ax = plt.subplots(clear=True)
        ax.errorbar(x, y, yerr=yerr, marker="o")
        ax.set(
            xlabel=r"x-label $x$",
            ylabel=r"y-label $y$",
            title="Ax title: Errorbars",
        )
        pdf.savefig()
        plt.close()

        # Scatter plot
        x = np.linspace(0, 2, 25)
        y = np.random.rand(25) * 5
        fig, ax = plt.subplots(clear=True)
        ax.scatter(x, y)
        ax.set(
            xlabel=r"x-label $x$",
            ylabel=r"y-label $y$",
            title="Ax title: Scatter plot",
        )
        pdf.savefig()
        plt.close()

        # Histogram
        mu = 100
        sigma = 15
        x = mu + sigma * np.random.randn(437)
        num_bins = 50
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(x, num_bins, density=True)
        y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            -0.5 * (1 / sigma * (bins - mu)) ** 2
        )
        ax.plot(bins, y, "--")
        ax.set(
            xlabel=r"x-label $x$",
            ylabel=r"y-label $y$",
            title="Ax title: Histogram",
        )
        pdf.savefig()
        plt.close()

        # Contour plot
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-2.0, 2.0, delta)
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-(X**2) - Y**2)
        Z2 = np.exp(-((X - 1) ** 2) - (Y - 1) ** 2)
        Z = (Z1 - Z2) * 2
        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, Z)
        ax.clabel(CS, inline=True, fontsize=10)
        ax.set(
            xlabel=r"x-label $x$",
            ylabel=r"y-label $y$",
            title="Ax title: Contour plot",
        )
        pdf.savefig()
        plt.close()

        # Boxplot
        spread = np.random.rand(50) * 100
        center = np.ones(25) * 50
        flier_high = np.random.rand(10) * 100 + 100
        flier_low = np.random.rand(10) * -100
        data1 = np.concatenate((spread, center, flier_high, flier_low))
        spread = np.random.rand(50) * 100
        center = np.ones(25) * 40
        flier_high = np.random.rand(10) * 100 + 100
        flier_low = np.random.rand(10) * -100
        data2 = np.concatenate((spread, center, flier_high, flier_low))
        data = [data1, data2, data2[::2]]
        fig, ax = plt.subplots()
        ax.boxplot(data)
        ax.set(
            xlabel=r"x-label $x$",
            ylabel=r"y-label $y$",
            title="Ax title: Boxplot",
        )
        pdf.savefig()
        plt.close()

        # Polar plot
        r = np.arange(0, 2, 0.01)
        theta = 2 * np.pi * r
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot(theta, r)
        ax.set_rmax(2)
        ax.grid(True)
        ax.set_title("Ax title: Polar plot")
        pdf.savefig()
        plt.close()

        # pcolormesh
        Z = np.random.rand(6, 10)
        x = np.arange(-0.5, 10, 1)  # len = 11
        y = np.arange(4.5, 11, 1)  # len = 7
        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(x, y, Z)
        ax.set(
            xlabel=r"x-label $x$",
            ylabel=r"y-label $y$",
            title="Ax title: pcolormesh",
        )
        fig.colorbar(pcm, label=r"Colorbar $z$")
        pdf.savefig()
        plt.close()

        # Subplots
        data = np.random.randn(2, 100)
        fig, axs = plt.subplots(2, 2)
        fig.suptitle("Fig title: Subplots")
        axs[0, 0].hist(data[0])
        axs[1, 0].scatter(data[0], data[1])
        axs[0, 1].plot(data[0], data[1])
        axs[1, 1].hist2d(data[0], data[1])
        for ax in axs.flatten():
            ax.set(xlabel=r"x-label $x$", ylabel=r"y-label $y$")
        fig.align_labels()
        pdf.savefig()
        plt.close()

        # Log plots
        t = np.arange(0.01, 20.0, 0.01)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle("Fig title: Log plots")
        # log y axis
        ax1.semilogy(t, np.exp(-t / 5.0))
        ax1.set(title="Ax title: Semilogy")
        ax1.grid()
        # log x axis
        ax2.semilogx(t, np.sin(2 * np.pi * t))
        ax2.set(title="Ax title: Semilogx")
        ax2.grid()
        # log x and y axis
        ax3.loglog(t, 20 * np.exp(-t / 10.0))
        ax3.set_xscale("log", base=2)
        ax3.set(title="Ax title: Loglog base 2 on x")
        ax3.grid()
        # With errorbars: clip non-positive values
        x = 10.0 ** np.linspace(0.0, 2.0, 20)
        y = x**2.0
        ax4.set_xscale("log", nonpositive="clip")
        ax4.set_yscale("log", nonpositive="clip")
        ax4.set(title="Ax title: Errorbars go negative")
        ax4.errorbar(x, y, xerr=0.1 * x, yerr=5.0 + 0.75 * y)
        ax4.set_ylim(bottom=0.1)
        fig.align_labels()
        pdf.savefig()
        plt.close()

        # Hatches and patches
        x = np.arange(1, 5)
        y1 = np.arange(1, 5)
        y2 = np.ones(y1.shape) * 4
        fig = plt.figure()
        fig.suptitle("Fig title: Hatches and patches")
        axs = fig.subplot_mosaic([["bar1", "patches"], ["bar2", "patches"]])
        axs["bar1"].bar(x, y1, edgecolor="black", hatch="/")
        axs["bar1"].bar(x, y2, bottom=y1, edgecolor="black", hatch="//")
        axs["bar2"].bar(x, y1, edgecolor="black", hatch=["--", "+", "x", "\\"])
        axs["bar2"].bar(
            x, y2, bottom=y1, edgecolor="black", hatch=["*", "o", "O", "."]
        )
        x = np.arange(0, 40, 0.2)
        axs["patches"].fill_between(
            x, np.sin(x) * 4 + 30, y2=0, hatch="///", zorder=2, fc="c"
        )
        axs["patches"].add_patch(
            mpatches.Ellipse(
                (4, 50),
                10,
                10,
                fill=True,
                hatch="*",
                edgecolor="black",
                facecolor="y",
            )
        )
        axs["patches"].add_patch(
            mpatches.Polygon(
                [(10, 20), (30, 50), (50, 10)],
                hatch="\\/...",
                edgecolor="black",
                facecolor="g",
            )
        )
        axs["patches"].set_xlim([0, 40])
        axs["patches"].set_ylim([10, 60])
        pdf.savefig()
        plt.close()

        # Patches and paths
        fig, ax = plt.subplots()
        Path = mpath.Path
        path_data = [
            (Path.MOVETO, (1.58, -2.57)),
            (Path.CURVE4, (0.35, -1.1)),
            (Path.CURVE4, (-1.75, 2.0)),
            (Path.CURVE4, (0.375, 2.0)),
            (Path.LINETO, (0.85, 1.15)),
            (Path.CURVE4, (2.2, 3.2)),
            (Path.CURVE4, (3, 0.05)),
            (Path.CURVE4, (2.0, -0.5)),
            (Path.CLOSEPOLY, (1.58, -2.57)),
        ]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor="r", alpha=0.5)
        ax.add_patch(patch)
        x, y = zip(*path.vertices)
        ax.plot(x, y, "go-")
        ax.set(
            xlabel=r"x-label $x$",
            ylabel=r"y-label $y$",
            title=r"Ax title: Patches and paths",
        )
        ax.grid()
        ax.axis("equal")
        pdf.savefig()
        plt.close()

        # 3D plot
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.sin(R)
        surf = ax.plot_surface(X, Y, Z)
        ax.set(
            zlim=(-1.01, 1.01),
            xlabel=r"x-label $x$",
            ylabel=r"y-label $y$",
            title="Ax title: 3D plot",
        )
        fig.colorbar(surf, label=r"Surface $s$")
        pdf.savefig()
        plt.close()

        # Pie chart
        labels = "Frogs", "Hogs", "Dogs", "Logs"
        sizes = [15, 30, 45, 10]
        explode = (0, 0.1, 0, 0)
        fig1, ax = plt.subplots()
        ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.axis("equal")
        ax.set_title("Ax title: Pie chart")
        pdf.savefig()
        plt.close()
