import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .setting import plotSet, FIG_SIZE, TITLE

def boxPlot(
    RESULT: pd.DataFrame,
    colName: str | list[str],
    ylabel: str = "",
    ylim: tuple[float, float] = (0.0, 0.0),
    xticklabel: list[str] = [],
    path: str = ""
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
    plt.figure(figsize=FIG_SIZE)
    plot: Axes = analysis.boxplot(
        patch_artist = True,
        showmeans=True,
        boxprops = {"color": "gray", "facecolor":"teal"},
        flierprops = {"marker":'.', "markerfacecolor":"gray", "markeredgecolor": "none"},
        medianprops = {"color": "white"},
        meanprops = {"color": "gray"},
        whiskerprops = {"color": "gray"},
        capprops = {"color": "gray"}
    )
    
    if isinstance(colName, str):
        ylabel = "{} Index".format(TITLE.get(colName))
    if xticklabel == []:
        plot.set_xticklabels(years)
    else:
        plot.set_xticklabels(xticklabel)
    plot.set_ylabel(
        ylabel,
        fontweight="bold"
    )
    plot.set_xlabel(
        "Year",
        fontweight="bold"
    )
    if ylim != (0.0, 0.0):
        plt.ylim(ylim)
    else:
        plot.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    
    plt.tight_layout()
    if path == "":
        plt.show()
    else:
        plt.savefig(os.path.join(path, "{}_boxplot.jpg".format(ylabel.split(' ')[0])))
    plt.close()

    return