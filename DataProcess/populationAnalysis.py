import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.axes import Axes
from scipy.stats import skew

try:
    from .setting import plotSet, FIG_SIZE, BAR_COLORS, TITLE, TICK_SIZE
except:
    from setting import plotSet, FIG_SIZE, BAR_COLORS, TITLE, TICK_SIZE

LEG_NAME = {
    "M2SFCA_Gini": "Equity",
    "M2SFCA_Accessibility": "Accessibility",
    "Relative_Accessibility": "Eficiency"
}

class populationAnalysis:
    __slots__ = ["classify", "classifyName", "basic", "dfList"]

    def __init__(self, basic: pd.DataFrame, root: str, populations: list[str], classify: str) -> None:
        plotSet()
        self.dfList = []
        self.classify = []
        self.__creatdf(populations, root, classify, self.dfList, self.classify)
        self.classifyName = classify
        self.basic = basic
    
    @property
    def df(self) -> pd.DataFrame:
        return pd.concat(self.dfList)
    
    @staticmethod
    def __creatdf(populations: list, root: str, classify: str, dfList: list, classifyList: list) -> None:
        for i in populations:
            df = pd.read_csv(os.path.join(root, i), encoding="utf-8")
            className = i.split('_')[-1].split('.')[0]
            df[classify] = className
            dfList.append(df)
            classifyList.append(className)

        return
    
    @staticmethod
    def __split(classify: list, dfList: list) -> tuple[str, pd.DataFrame, list[str], list[pd.DataFrame]]:
        return classify[0], dfList[0], classify[1:], dfList[1:]
    
    def __compressdf(
        self,
        baseClass: str, baseDf: pd.DataFrame,
        comparClass: list[str], comparDf: list[pd.DataFrame],
        resultCol: list[str], classifyName: str,
        savePath: tuple[str | None, str] | None = None
    ) -> pd.DataFrame:
        result = []
        for classify, df in zip(comparClass, comparDf):
            # Skip children
            if classify == "children":
                continue

            colName = "{} - {}".format(classify.capitalize(), baseClass.capitalize())
            subResult = pd.DataFrame(columns=resultCol + ["name"])
            for col in resultCol:
                subResult[col] = df[col] - baseDf[col]
            subResult["name"] = df["name"]
            subResult[classifyName] = colName

            if savePath is not None and savePath[0] is not None:
                self.plotHist(
                    subResult[resultCol[-3]],
                    colName,
                    savePath=savePath
                )

            result.append(subResult)

        return pd.concat(result)
    
    def difference(
        self,
        colName: str | tuple[str, str],
        scal: str, adj: int = 0,
        ax: list[Axes] | Axes | None = None,
        savePath: str | None = None
    ) -> None:
        years = list(str(x) for x in range(2015,2026))
        if isinstance(colName, str):
            resultCol1 = ["{}_{}".format(colName, y) for y in years]
            resultCol2 = []
            colName1 = colName
            colName2 = ""
        else:
            resultCol1 = ["{}_{}".format(colName[0], y) for y in years]
            resultCol2 = ["{}_{}".format(colName[1], y) for y in years]
            colName1, colName2 = colName
        
        baseClass, baseDf, comparClass, comparDf = self.__split(self.classify, self.dfList)
        result1 = self.__compressdf(baseClass, baseDf, comparClass, comparDf, resultCol1, self.classifyName, (savePath, colName1))   

        # Show in one
        if isinstance(colName, tuple):
            baseClass2, baseDf2, comparClass2, comparDf2 = self.__split(self.classify, self.dfList)
            result2 = self.__compressdf(baseClass2, baseDf2, comparClass2, comparDf2, resultCol2, self.classifyName, (savePath, colName2))
            self.plotVilon((result1, result2), colName, self.classifyName, scal, ax, adj, savePath)
        else:
            self.plotVilon(result1, colName, self.classifyName, scal, ax, adj, savePath)

        return
    
    @staticmethod
    def plotHist(series: pd.Series, title: str = "", steps: int = 10, savePath: tuple[str | None, str] | None = None) -> None:
        maxs = series.max()
        mins = np.floor(series.min() * 1000) / 1000
        # Set x-axis interval
        i = 0
        step = round((maxs - mins) / steps, 4)
        xticks=[0]
        while i < maxs:
            i += step
            xticks.append(i)
        i = 0
        while i > mins:
            i -= step
            xticks.append(i)
        xticks.sort()

        plt.figure(figsize=FIG_SIZE.D)
        
        # Draw the histogram
        series.plot.hist(bins=xticks, xticks=xticks[::2], color="teal") # type: ignore

        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='x=0')

        plt.tight_layout()
        if savePath is None or savePath[0] is None:
            plt.show()
        else:
            plt.savefig(os.path.join(savePath[0], "{}_{}_hist_2025.jpg".format(TITLE.get(savePath[1]), title)), dpi=300)

        plt.close()

        return

    @staticmethod
    def plotVilon(
        df: pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame],
        colName: str | tuple[str, str], classifyName: str,
        scal: str,
        axs: list[Axes] | Axes | None = None, adj: int = 0,
        savePath: str | None = None
    ) -> list[pd.DataFrame]:
        years = list(str(x) for x in range(2015,2026))
        if isinstance(df, pd.DataFrame) and isinstance(colName, str):
            dfs = [df]
            resultCols = [["{}_{}".format(colName, y) for y in years]]
            colNames = [colName]
        elif isinstance(df, tuple) and isinstance(colName, tuple):
            dfs = list(df)
            resultCols = [
                ["{}_{}".format(colName[0], y) for y in years],
                ["{}_{}".format(colName[1], y) for y in years]
            ]
            colNames = list(colName)
        else:
            raise RuntimeError("df, colName, and classifyName must be tuple when initialized with two csv files.")
        
        classify = dfs[0][classifyName].unique().tolist()
        meltDf: list[pd.DataFrame] = []
        for i, df in enumerate(dfs):
            df = pd.melt(
                df,
                id_vars=[classifyName, "name"], 
                value_vars=resultCols[i],
                var_name="Year", 
                value_name="Value",
            )
            df["Year"] = df["Year"].str.split("_").str[-1]
            df["Order"] = colNames[i]
            df2 = df.copy()
            df2["Order"] = colNames[-(i+1)]
            df2["Value"] = None
            df = pd.concat([df, df2], ignore_index=True)

            meltDf.append(df)
        
        for n, i in enumerate(classify):
            palette = {x: y for x,y in zip(colNames, BAR_COLORS[n + adj])}
            if axs is None:
                fig, ax1 = plt.subplots(figsize=FIG_SIZE.W)
            else:
                ax1 = axs[n] if isinstance(axs, list | np.ndarray) else axs

            if len(meltDf) > 1:
                ax2 = ax1.twinx()
            else:
                ax2 = None

            statsInfo = {x: {} for x in range(len(meltDf))}
            for j, df in enumerate(meltDf):
                subdf: pd.DataFrame = df.loc[df[classifyName] == i]
                ax = ax1 if j == 0 else ax2
                if ax is None:
                    continue

                sns.violinplot(
                    data=subdf,
                    ax = ax,
                    x="Year", y="Value",
                    split=True,
                    hue="Order",
                    hue_order=colNames,
                    palette=palette,
                    alpha=0.8,
                    inner=None
                )
                ax.set_facecolor(BAR_COLORS[n + adj][-j])
                ax.patch.set_alpha(0.05)

                boxwidth = 0.08
                positions = [i + 0.5 - 2 * boxwidth for i in range(len(years))]
                
                sns.boxplot(
                    data=subdf,
                    ax = ax,
                    x="Year", y="Value",
                    width=boxwidth,
                    color="#FFFFFF",
                    showfliers=False,
                    boxprops={"edgecolor": "#000000"},
                    whiskerprops={"color": "#000000"},
                    capprops={"color": "#000000"},
                    medianprops={"color": "#000000"},
                    positions=positions
                )

                # Cal skew and variance in each year
                yearsSorted = sorted(subdf["Year"].unique())
                for idx, year in enumerate(yearsSorted):
                    data: pd.DataFrame = subdf[(subdf["Year"] == year) & (subdf["Order"] == colNames[j])]["Value"]
                    data = data.dropna()
                    
                    if len(data) > 0:
                        skews = skew(data)
                        variance = data.std()
                        statsInfo[j][idx] = {
                            "SK": skews,
                            "SD": variance
                        }

                dif = (df["Value"].max() - df["Value"].min()) / 10
                if dif < 0.1:
                    linthresh = 0.001
                    f = "%.4f"
                else:
                    linthresh = 0.1
                    f = "%.2f"
                
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f))
                ax.set_ylabel("")
                ax.set_yscale("symlog", linthresh=linthresh)

            formatter = ticker.ScalarFormatter()
            formatter.set_scientific(False)
            if ax2 is not None:
                ax2.grid(False)
                ax2.get_legend().remove()
                if scal == "gender":
                    ax2.set_ylim(-10, 10)
                    ax2.set_yticks([-10, -1, 0, 1, 10])
                else:
                    ax2.set_ylim(-1000, 1000)
                    ax2.set_yticks([-1000, -10, -1, 0, 1, 10, 1000])
                ax2.yaxis.set_major_formatter(formatter)
            
            if scal == "gender":
                ax1.set_ylim(-10, 10)
                ax1.set_yticks([-10, -1, 0, 1, 10])
            else:
                ax1.set_ylim(-1000, 1000)
                ax1.set_yticks([-1000, -10, -1, 0, 1, 10, 1000])
            
            ax1.yaxis.set_major_formatter(formatter)
            lines1, labels1 = ax1.get_legend_handles_labels()
            # Legend Name
            ax1.legend(
                lines1,
                [f"{LEG_NAME.get(x, x)} Gap by {scal.capitalize()} ({i})" for x in colNames],
                loc="lower left"
            )
            ax1.set_xticklabels(years)
            ax1.axhline(y=0, color="#FF0000", linestyle="--", linewidth=2)

            # Add skew and variance info on vilon plot
            for j, df in enumerate(meltDf):
                ax = ax1 if j == 0 else ax2
                if ax is None:
                    continue
                    
                _, yMax = ax.get_ylim()
                textYPosition = 2 if yMax == 10 else 100
                
                for x in range(len(years)):
                    if x in statsInfo[j]:
                        stats = statsInfo[j][x]
                        
                        # Add text of skew and variance in specific year
                        ax.text(
                            x, textYPosition, 
                            f"SK:{stats["SK"]:.2f}\nSD:{stats["SD"]:.2f}", 
                            ha="center", va="bottom", 
                            # fontsize=int(TICK_SIZE * 0.9),
                            color="black",
                            # bbox=dict(boxstyle="round,pad=0.3", 
                            #         facecolor='white', 
                            #         alpha=0.8,
                            #         edgecolor=palette[colNames[j]])
                        )

            plt.tight_layout()
            if savePath is None and axs is None:
                plt.show()
                plt.close()
            elif savePath is not None and axs is None:
                plt.savefig(os.path.join(savePath, "{}_{}_vilon.jpg".format(TITLE.get(str(colName)), i)), dpi=300)
                plt.close()

        return meltDf

# Debug
if __name__ == "__main__":
    from multiFigs import multiFigs
    f = multiFigs(1, 3, figsize="H3W", sharex=True, sharey=False)
    x = ["city_optAcc_Female.csv", "city_optAcc_Male.csv"]
    n = "Gender"

    # a = populationAnalysis(pd.DataFrame(), os.path.join("China_Acc_Results", "Result"), (x, x2), (n, n2))
    # a.difference("M2SFCA_Gini")
    # a.difference("Relative_Accessibility", "paper\\figure\\fig4")
    # a.difference("M2SFCA_Accessibility", "paper\\figure\\fig4")
    a = populationAnalysis(pd.DataFrame(), os.path.join("China_Acc_Results", "Result"), x, n)
    a.difference("M2SFCA_Accessibility", "gender", ax=f.axs[0])

    x = ["city_optAcc_All_middle.csv", "city_optAcc_All_old.csv", "city_optAcc_All_young.csv"] # "city_optAcc_All_children.csv" No need for children
    n = "Age"
    b = populationAnalysis(pd.DataFrame(), os.path.join("China_Acc_Results", "Result"), x, n)
    b.difference("M2SFCA_Accessibility", "age", adj=1, ax=f.axs[1:])

    f.globalYlabel("Accessibility Gap", [1])
    f.save(os.path.join(".", "paper", "figure", "fig4", "fig4.jpg"))