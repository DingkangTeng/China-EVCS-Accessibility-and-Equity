import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

try:
    from .setting import plotSet, FIG_SIZE, TITLE, BAR_COLORS
except:
    from setting import plotSet, FIG_SIZE, TITLE, BAR_COLORS

from typing import Iterable

def boxPlot(
    RESULT: pd.DataFrame,
    colName: str | Iterable,
    axs: Axes | None = None,
    ylabel: str = "",
    ylim: tuple[float, float] = (0.0, 0.0),
    figsize: str = "D",
    xticklabel: Iterable = [],
    colorGroup: int = 0, color: int = 0,
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
        meanprops = {"color": "gray"},
        whiskerprops = {"color": "gray"},
        capprops = {"color": "gray"},
        **kwgs
    )

    if ylim != (0.0, 0.0):
        plot.set_ylim(ylim)
    else:
        plot.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    if isinstance(colName, str):
        ylabel = "{} Index".format(TITLE.get(colName))
    plot.set_ylabel(ylabel)

    if axs is None:
        if xticklabel == []:
            plot.set_xticklabels(years)
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
    RESULT = pd.read_csv(os.path.join("China_Acc_Results", "Result", "city_efficiency.csv"), encoding="utf-8")
    boxPlot(RESULT.copy(), "Relative_Accessibility", path=r".\\paper\\figure\\fig2")

    # from multiFigs import multiFigs
    # f = multiFigs(1, 3, figsize="D", sharex=True, sharey=False)

    # colName = list(range(2015, 2026))
    # df = pd.read_excel(r"China_Acc_Results\Result\Raster_Density_population.xlsx")
    # boxPlot(df, colName, f.axs[0], "Pop. Coverage", (0, 0.5), xticklabel=colName, figsize="S", color=0) # , path=r"paper\\figure\\fig1"
    # df = pd.read_excel(r"China_Acc_Results\Result\Raster_Density_gdp.xlsx")
    # boxPlot(df, colName, f.axs[1], "GDP Coverage", (0, 0.5), xticklabel=colName, figsize="S", color=1) # , path=r"paper\\figure\\fig1"
    # df = pd.read_excel(r"China_Acc_Results\Result\Roads_Density.xlsx")
    # boxPlot(df, colName, f.axs[2], "Roads Density", (0, 16), xticklabel=colName, figsize="S", color=2) # , path=r"paper\\figure\\fig1"
    # # RESULT = pd.read_csv(os.path.join("China_Acc_Results", "Result", "city_efficiency.csv"), encoding="utf-8")
    # # boxPlot(RESULT.copy(), "Relative_Accessibility", ylim=(0.4, 0.9))

    # f.globalXlabel("Year", [-1])
    # f.save(r"paper\\figure\\fig1\\basic.jpg")