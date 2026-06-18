import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Iterable

try:
    from .setting import plotSet, FIG_SIZE, TITLE, BAR_COLORS
except:
    from setting import plotSet, FIG_SIZE, TITLE, BAR_COLORS

def boxPlot(
    RESULT: pd.DataFrame,
    colName: str | Iterable,
    axs: Axes | None = None,
    ylabel: str = "",
    ylim: tuple[float, float] = (0.0, 0.0),
    figsize: str = "D",
    xticklabel: Iterable = [],
    colorGroup: int = 0, color: int = 0,
    slop: bool = False,
    path: str = "",
    **kwgs
    ) -> None:
    plotSet()
    resultCol = []
    years = list(str(x) for x in range(2015,2026))
    if colName in ["Relative_Accessibility", "M2SFCA_Gini"]:
        for y in years:
            resultCol.append("{}_{}".format(colName, y))
    elif isinstance(colName, str):
        resultCol.append(colName)
    else:
        resultCol += colName

    analysis = RESULT[resultCol]

    if axs is None:
        fig = plt.figure(figsize=getattr(FIG_SIZE, figsize))
        ax = plt.subplot()
    else:
        ax = axs
    
    plot: Axes = analysis.boxplot(
        ax = ax,
        patch_artist = True,
        showmeans=True,
        boxprops = {"color": "gray", "facecolor":BAR_COLORS[colorGroup][color]},
        flierprops = {"marker":'.', "markerfacecolor":"gray", "markeredgecolor": "none"},
        medianprops = {"color": "white"},
        whiskerprops = {"color": "gray"},
        capprops = {"color": "gray"},
        **kwgs
    )
    
    if slop:
        # Add means trend line
        means = analysis.mean()
        xPos = np.arange(1, len(means) + 1)
        _, _, _, p, _ = stats.linregress(xPos, means)
        p = float(p) # type: ignore
        if p < 0.001:
            stars = " $^{\\!\\!\\!***}$"
        elif p < 0.01:
            stars = "$^{**}$"
        elif p < 0.05:
            stars = "$^{*}$"
        else:
            stars = ""
        coeffs = np.polyfit(xPos, means, deg=1)
        polyFunc = np.poly1d(coeffs)
        xSmooth = np.linspace(xPos.min(), xPos.max(), 100)
        ySmooth = polyFunc(xSmooth)

        plot.plot(
            xSmooth, ySmooth,
            marker = None,
            linestyle = "--",
            color = "green",
            label = "Mean Trend"
        )
        # Slop
        ax.text(
            0.68, 0.95,
            f"Slope = {coeffs[0]:.4f}{stars}",
            transform = ax.transAxes,
            ha = "left",
            va = "top",
            bbox = dict(boxstyle="round", facecolor='white', alpha=0.7)
        )

    # Beautify
    if ylim != (0.0, 0.0):
        plot.set_ylim(ylim)
    else:
        plot.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    if isinstance(colName, str):
        ylabel = "{} Index".format(TITLE.get(colName))
    plot.set_ylabel(ylabel)

    if axs is None:
        if xticklabel == []:
            # plot.set_xticks(range(1, len(years) + 1, 2))
            plot.set_xticklabels(["" if int(x) % 2 ==0 else x for x in years])
        else:
            plot.set_xticklabels(xticklabel)
        
        plot.set_xlabel("Year")
        
        plt.tight_layout()
        if path == "":
            plt.show()
        else:
            plt.savefig(os.path.join(path, "{}_boxplot.jpg".format(ylabel.split(' ')[0])))
        plt.close()

    return

# Debug
if __name__ == "__main__":
    from multiFigs import multiFigs
    f = multiFigs(1, 3, figsize="HH", sharex=True, sharey=False)
    meanprops = {"markerfacecolor":"lightgreen"}
    colName = list(range(2015, 2026))

    df = pd.read_excel(r"C:\Users\tengd\OneDrive - The Hong Kong Polytechnic University\Student Assistant\ChinaDynam\data\_AnalysisData\result\Raster_Density_population.xlsx")
    boxPlot(df, colName, f.axs[0], "Pop. Coverage", (0, 0.5), xticklabel=colName, figsize="S", color=0, meanprops=meanprops) # , path=r"paper\\figure\\fig1"
    df = pd.read_excel(r"C:\Users\tengd\OneDrive - The Hong Kong Polytechnic University\Student Assistant\ChinaDynam\data\_AnalysisData\result\Raster_Density_gdp.xlsx")
    boxPlot(df, colName, f.axs[1], "GDP Coverage", (0, 0.5), xticklabel=colName, figsize="S", color=1, meanprops=meanprops) # , path=r"paper\\figure\\fig1"
    df = pd.read_excel(r"C:\Users\tengd\OneDrive - The Hong Kong Polytechnic University\Student Assistant\ChinaDynam\data\_AnalysisData\result\Roads_Density_highway.xlsx")
    boxPlot(df, colName, f.axs[2], "Road Coverage", (0, 16), xticklabel=colName, figsize="S", color=2, meanprops=meanprops) # , path=r"paper\\figure\\fig1"

    f.globalXlabel("Year", [-1])
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker='^', color='lightgreen', 
            linestyle='none', markerfacecolor='lightgreen', markersize=8, label='Mean'),
        Line2D([0], [0], linestyle='--', color='green', linewidth=2, label='Mean Trend')
    ]
    f.legend(
        handles=legend,
        loc="lower center",
        ncol=2
    )
    f.save(r"C:\Users\tengd\OneDrive - The Hong Kong Polytechnic University\Student Assistant\ChinaDynam\_AnalysisData\figure\fig1\basic.jpg")