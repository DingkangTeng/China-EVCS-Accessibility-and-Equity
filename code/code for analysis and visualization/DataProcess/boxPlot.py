import os
import pandas as pd
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

    df = pd.read_excel(r"China_Acc_Results\Result\Raster_Density_population.xlsx")
    boxPlot(df, colName, f.axs[0], "Pop. Coverage", (0, 0.5), xticklabel=colName, figsize="S", color=0, meanprops=meanprops) # , path=r"paper\\figure\\fig1"
    df = pd.read_excel(r"China_Acc_Results\Result\Raster_Density_gdp.xlsx")
    boxPlot(df, colName, f.axs[1], "GDP Coverage", (0, 0.5), xticklabel=colName, figsize="S", color=1, meanprops=meanprops) # , path=r"paper\\figure\\fig1"
    df = pd.read_excel(r"China_Acc_Results\Result\Roads_Density_highway.xlsx")
    boxPlot(df, colName, f.axs[2], "Road Coverage", (0, 16), xticklabel=colName, figsize="S", color=2, meanprops=meanprops) # , path=r"paper\\figure\\fig1"

    f.globalXlabel("Year", [-1])
    f.save(r"paper\\figure\\fig1\\basic.jpg")