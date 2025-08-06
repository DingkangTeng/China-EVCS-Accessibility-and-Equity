import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .setting import Y_LABEL_SIZE, plotSet

def boxPlot(RESULT: pd.DataFrame, colName: str) -> None:
    plotSet()
    resultCol = []
    years = list(str(x) for x in range(2015,2026))
    for y in years:
        resultCol.append("{}_{}".format(colName, y))

    analysis = RESULT[resultCol]
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
    plot.set_xticklabels(years)
    plot.set_ylabel(
        colName,
        fontweight="bold",
        fontsize=Y_LABEL_SIZE,
    )
    plot.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    plt.show()