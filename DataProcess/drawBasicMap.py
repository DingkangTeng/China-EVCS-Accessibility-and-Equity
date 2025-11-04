import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

try:
    from .setting import plotSet, INDEX, FIG_SIZE, BAR_COLORS
except:
    from setting import plotSet, INDEX, FIG_SIZE, BAR_COLORS

def drawClusting(RESULT: pd.DataFrame, BASE_MAP: gpd.GeoDataFrame, reativeClustering:dict = {}, giniClustering: dict = {}, savePath: str = '') -> None:
    tag1 = {0: "Stable efficiency", 1: "Decreased efficiency", 2:"Increased efficiency"}
    tag2 = {0: "Slowly increased equlity", 1: "Quickly increased equlity"}
    result = []
    for i in range(3):
        for j in range(2):
            n = set(reativeClustering[i]) & set(giniClustering[j])
            ef = tag1[i]
            eq = tag2[j]
            c = f"{ef} & {eq}"
            print(f"{c} ({len(n)}): {n}")
            for x in n:
                result.append([x, ef, eq, c])
    result = pd.DataFrame(result, columns=["name", "clusting_efficiency", "clusting_equity", "clusting"])
    result = RESULT.set_index("name").join(result.set_index("name")).drop(columns='gb')
    if savePath != '':
        result.to_csv(os.path.join(savePath, "city_with_clusting.csv"), encoding='utf-8')
    result = BASE_MAP.join(result, on=INDEX)
    titles = {"clusting_efficiency": "Efficiency", "clusting_equity": "Equity", "clusting": "Efficiency and Equity"}
    for i in ["clusting_efficiency", "clusting_equity", "clusting"]:
        fig, ax = plt.subplots(figsize=(10, 10))
        result.plot(
            i,
            ax = ax,
            missing_kwds={
                "color": "lightgrey",
                "edgecolor": "black",
                "hatch": "////",
                "label": "No Value"
            },
            legend=True,
            legend_kwds={
                "loc": "lower right",
                "title": titles[i],
                "shadow": True
            }
        )
        ax.axis('off')
        plt.tight_layout(pad=4.5)
        plt.show()
        plt.close()

    return

def stackplot(df: pd.DataFrame, name: str, subname: str, figsize: str = "D", rotation: int = 0, path: str = "") -> None:
    if figsize == "N":
        plotSet(1 , 0.99)
    else:
        plotSet()
    LEGEND = {
        "Number of chargers (millions)": "Chargers",
        "Growth rate of chargers (%)": "Growth rate"
    }
    ROW = {
        "Number of chargers (millions)": "sum",
        "Growth rate of chargers (%)": "ratio"
    }
    COLOR = {
        "Number of chargers (millions)": BAR_COLORS[0],
        "Growth rate of chargers (%)": BAR_COLORS[1]
    }
    fig = plt.figure(figsize=getattr(FIG_SIZE, figsize))
    ax = plt.subplot()
    years = list(range(2016, 2026))
    df = df.drop(columns=["region"])

    df.loc[df["name"] == ROW.get(name, name), years].T.plot(
        ax=ax,
        marker='.',
        color=COLOR.get(name, "teal"),
        legend=False
    )

    ax.set_ylabel(name)

    if subname != "":
        ax2 = ax.twinx()
        df.loc[df["name"] == ROW.get(subname, subname), years].T.plot(
            ax=ax2,
            marker="^",
            color=COLOR.get(subname, "teal"),
            legend=False
        )

        lines1, _ = ax.get_legend_handles_labels()
        lines2, _ = ax2.get_legend_handles_labels()

        ax.set_ylim(0, 5)
        ax2.set_yticks(np.linspace(0, 100, 6))
        ax2.set_ylim(0, 100)
        ax2.grid(False)

        ax2.set_ylabel(subname)
        ax.legend(lines1 + lines2, [LEGEND.get(name, name), LEGEND.get(subname, subname)], loc="upper left")
        
    else:
        ax.legend(LEGEND.get(name, name))
        
    ax.set_xlabel("Year")
    ax.set_xticks(range(0,10), years, rotation=rotation) # type: ignore

    plt.tight_layout()
    if path == "":
        plt.show()
    else:
        plt.savefig(os.path.join(path, "{}.jpg".format(name)), dpi=300)
    
    plt.close()

    return

if __name__ == "__main__":
    provinceLevelData = pd.read_excel(os.path.join("China_Acc_Results", "Result", "provinceLevel", "China_EVCS.xlsx"))
    stackplot(provinceLevelData, "Number of chargers (millions)", "Growth rate of chargers (%)", "N", 90, os.path.join(".", "paper", "figure", "fig1")) #