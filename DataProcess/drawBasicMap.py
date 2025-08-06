import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

from .setting import plotSet, INDEX
# Draw Relative Accessibility Efficiency in 2015, 2020, 2025
def draw152535(RESULT: pd.DataFrame, BASE_MAP: gpd.GeoDataFrame) -> None:
    plotSet()
    for y in [2015, 2020, 2025]:
        resultCol = "Relative_Accessibility_{}".format(y)
        yearData = RESULT[[INDEX, resultCol]]
        yearData.loc[:, INDEX] = yearData[INDEX].astype(str)
        yearData.set_index(INDEX, inplace=True)
        result = BASE_MAP.join(yearData, on=INDEX)
        fig, ax = plt.subplots(figsize=(10, 10))
        result.plot(
            resultCol,
            ax = ax,
            missing_kwds={
                "color": "lightgrey",
                "edgecolor": "black",
                "hatch": "////",
                "label": "No Value"
            },
            cmap="Oranges",
            scheme="BoxPlot",
            k=5,
            legend=True,
            legend_kwds={
                'loc': 'lower right',
                'title': 'Relative Accessibiliey',
                'shadow': True
            }
        )
        ax.axis('off')
        plt.tight_layout(pad=4.5) # 调整不同标题之间间距
        plt.show()

def drawClusting(RESULT: pd.DataFrame, BASE_MAP: gpd.GeoDataFrame, reativeClustering:dict = {}, giniClustering: dict = {}) -> None:
    # tag1 = {0: "Stable efficiency", 1: "Decreased efficiency", 2:"Increased efficiency"}
    # tag2 = {0: "Slowly increased equlity", 1: "Quickly increased equlity"}
    # result = []
    # for i in range(3):
    #     for j in range(2):
    #         n = set(reativeClustering[i]) & set(giniClustering[j])
    #         c = f"{tag1[i]} & {tag2[j]}"
    #         print(f"{c} ({len(n)}): {n}")
    #         for x in n:
    #             result.append([x, c])
    # result = pd.DataFrame(result, columns=["name", "clusting"])
    # result = BASE_MAP.join(RESULT.set_index("name").join(result.set_index("name")).fillna("No enough data for clusting").drop(columns='gb'))
    result = BASE_MAP
    result["clusting_num"] = result["clusting"]
    fig, ax = plt.subplots(figsize=(10, 10))
    result.plot(
        "clusting_num",
        ax = ax,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "black",
            "hatch": "////",
            "label": "No Value"
        },
        cmap="Oranges",
        scheme="BoxPlot",
        k=5,
        legend=True,
        legend_kwds={
            'loc': 'lower right',
            'title': 'Relative Accessibiliey',
            'shadow': True
        }
    )
    ax.axis('off')
    plt.tight_layout(pad=4.5) # 调整不同标题之间间距
    plt.show()

    return