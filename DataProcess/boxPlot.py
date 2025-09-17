import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

try:
    from .setting import plotSet, FIG_SIZE, TITLE
except:
    from setting import plotSet, FIG_SIZE, TITLE

from typing import Iterable

def boxPlot(
    RESULT: pd.DataFrame,
    colName: str | Iterable,
    ylabel: str = "",
    ylim: tuple[float, float] = (0.0, 0.0),
    xticklabel: Iterable = [],
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
    plt.figure(figsize=FIG_SIZE)
    plot: Axes = analysis.boxplot(
        patch_artist = True,
        showmeans=True,
        boxprops = {"color": "gray", "facecolor":"teal"},
        flierprops = {"marker":'.', "markerfacecolor":"gray", "markeredgecolor": "none"},
        medianprops = {"color": "white"},
        meanprops = {"color": "gray"},
        whiskerprops = {"color": "gray"},
        capprops = {"color": "gray"},
        **kwgs
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
        miny = ylim[0]
        maxy = ylim[1]
        step = (maxy - miny) / 5
        plot.set_yticks([miny + x * step for x in range(0,6)])
    else:
        plot.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    
    plt.tight_layout()
    if path == "":
        plt.show()
    else:
        plt.savefig(os.path.join(path, "{}_boxplot.jpg".format(ylabel.split(' ')[0])))
    plt.close()

    return

# Debug
if __name__ == "__main__":
    # df = pd.read_excel(r"China_Acc_Results\Result\Raster_Density_population.xlsx")
    # df = pd.read_excel(r"China_Acc_Results\Result\Raster_Density_gdp.xlsx")
    df = pd.read_excel(r"China_Acc_Results\Result\Roads_Density.xlsx")
    # df = pd.read_excel(r"China_Acc_Results\Result\Sample_rate.xlsx")
    colName = list(range(2015, 2026))
    # boxPlot(df, colName, "Sample rates", xticklabel=colName)
    boxPlot(df, colName, "Roads network density", (0, 5), xticklabel=colName, showfliers=False)
    # boxPlot(df, list(range(2018, 2026)), "Sample rates(%)", (0, 100), xticklabel=list(range(2018, 2026)))