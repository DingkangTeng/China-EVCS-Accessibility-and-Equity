import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

try:
    from .setting import plotSet, FIG_SIZE, FIG_SIZE_W, BAR_COLORS, TITLE
except:
    from setting import plotSet, FIG_SIZE, FIG_SIZE_W, BAR_COLORS, TITLE

DIFF_NAME = {
    "M2SFCA_Gini": "Diffenrence in equity",
    "Relative_Accessibility": "Diffenrence in eficiency",
    "M2SFCA_Accessibility": "Diffenrence in accessibility"
}

class populationAnalysis:
    __slots__ = ["classify", "classifyName", "basic", "dfList"]

    def __init__(self, basic: pd.DataFrame, root: str, populations: list[str], classify: str) -> None:
        plotSet()
        self.dfList = []
        self.classify = []
        for i in populations:
            df = pd.read_csv(os.path.join(root, i), encoding="utf-8")
            className = i.split('_')[-1].split('.')[0]
            df[classify] = className
            self.dfList.append(df)
            self.classify.append(className)

        self.classifyName = classify
        self.basic = basic

    @property
    def df(self) -> pd.DataFrame:
        return pd.concat(self.dfList)
    
    def difference(self, colName: str, savePath: str = "") -> None:
        years = list(str(x) for x in range(2015,2026))
        resultCol = ["{}_{}".format(colName, y) for y in years]

        result = []
        baseClass = self.classify[0]
        baseDf = self.dfList[0]
        comparClass = self.classify[1:]
        comparDf = self.dfList[1:]
        for classify, df in zip(comparClass, comparDf):
            subResult = pd.DataFrame(columns=resultCol + ["name"])
            for col in resultCol:
                subResult[col] = df[col] - baseDf[col]
            subResult["name"] = df["name"]
            subResult[self.classifyName] = "{} - {}".format(classify, baseClass)

            self.plotHist(subResult[resultCol[-3]], "{} - {}".format(classify, baseClass), savePath=(savePath, colName))

            result.append(subResult)

        result = pd.concat(result)
        result = self.plotVilon(result, colName, self.classifyName, savePath)

        # result.to_csv("{}.csv".format(colName), encoding="utf-8", index=False)

        return
    
    @staticmethod
    def plotHist(series: pd.Series, title: str = "", steps: int = 10, savePath: tuple[str, str] | None = None) -> None:
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

        plt.figure(figsize=FIG_SIZE)
        
        # Draw the histogram
        series.plot.hist(bins=xticks, xticks=xticks[::2], color="teal") # type: ignore

        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='x=0')
        # plt.title(title)

        plt.tight_layout()
        if savePath is None:
            plt.show()
        else:
            plt.savefig(os.path.join(savePath[0], "{}_{}_hist_2025.jpg".format(TITLE.get(savePath[1]), title)), dpi=300)

        plt.close()

        return

    @staticmethod
    def plotVilon(df: pd.DataFrame, colName: str, classifyName: str, savePath: str = "") -> pd.DataFrame:
        years = list(str(x) for x in range(2015,2026))
        resultCol = ["{}_{}".format(colName, y) for y in years]
        df = pd.melt(
            df,
            id_vars=[classifyName, "name"], 
            value_vars=resultCol,
            var_name="Year", 
            value_name="Value",
        )
        
        classify = df[classifyName].unique().tolist()
        palette = {x: y for x,y in zip(classify, BAR_COLORS)}

        for i in classify:
            plt.figure(figsize=FIG_SIZE_W)

            subdf = df.loc[df[classifyName] == i]
            ax = sns.violinplot(
                data=subdf,
                x="Year", y="Value",
                split=True,
                hue=classifyName,
                palette=palette,
                alpha=0.8,
                inner="box",
                inner_kws={
                    "color": "teal",
                }
            )

            formatter = ticker.ScalarFormatter()
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            dif = (df["Value"].max() - df["Value"].min()) / 10
            if dif < 0.1:
                linthresh = 0.001
                f = "%.4f"
            else:
                linthresh = 0.1
                f = "%.2f"

            ax.set_xticklabels(years)
            plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='y=0')
            plt.yscale("symlog", linthresh=linthresh)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f))
            plt.ylabel(DIFF_NAME.get(colName, colName))

            plt.legend(loc="lower left")
            plt.tight_layout()
            if savePath == "":
                plt.show()
            else:
                plt.savefig(os.path.join(savePath, "{}_{}_vilon.jpg".format(TITLE.get(colName), i)), dpi=300)

            plt.close()

        return df

# Debug
if __name__ == "__main__":
    # RESULT = pd.read_csv(os.path.join("China_Acc_Results", "Result", "city_efficiency.csv"), encoding="utf-8")
    x = ["city_efficiency_Female.csv", "city_efficiency_Male.csv"]
    n = "Gender"
    # x = ["city_efficiency_All_middle.csv", "city_efficiency_All_children.csv", "city_efficiency_All_old.csv", "city_efficiency_All_young.csv"]
    # n = "Age Group"

    a = populationAnalysis(pd.DataFrame(), os.path.join("China_Acc_Results", "Result"), x, n)
    a.difference("M2SFCA_Gini", "paper\\figure\\fig4")
    a.difference("Relative_Accessibility", "paper\\figure\\fig4")
    a.difference("M2SFCA_Accessibility", "paper\\figure\\fig4")