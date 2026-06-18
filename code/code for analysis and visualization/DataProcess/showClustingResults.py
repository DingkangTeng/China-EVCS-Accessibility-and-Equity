import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    from .setting import plotSet, TITLE, BAR_COLORS
except:
    from setting import plotSet, TITLE, BAR_COLORS

class showClustingResults:
    __slots__ = ["clustingResult", "gdp", "path", "_indicator", "_analysisType", "_analysisValue", "_colorGroup"]

    def __init__(self, clustingResult: pd.DataFrame, gdp: pd.DataFrame, indicator: str = "gdp", colorGroup: int = 0, path: str = "") -> None:
        self.clustingResult = clustingResult
        self.gdp = gdp
        self.path = path
        self._indicator = indicator
        self._colorGroup = colorGroup
        self._analysisType: str
        self._analysisValue: str
        plotSet(modxy=False)
        
    def analysisAll(self, andlysisValue: str = "") -> "_AnalysisExecutorImpl":
        self._analysisType = "clusting"
        self._analysisValue = andlysisValue
        return _AnalysisExecutorImpl(self)
    
    def analysisEquity(self) -> "_AnalysisExecutorImpl":
        self._analysisType =  "clusting_equity"
        self._analysisValue = "M2SFCA_Gini"
        
        return _AnalysisExecutorImpl(self)

    def analysisOptAcc(self)  -> "_AnalysisExecutorImpl":
        self._analysisType = "clusting_optAcc"
        self._analysisValue = "Relative_Accessibility"
        return _AnalysisExecutorImpl(self)
    
class _AnalysisExecutorImpl(showClustingResults):
    __slots__ = ["clusterStats", "df", "indicator", "analysisType", "analysisValue", "colorGroup"]

    def __init__(self, builder: showClustingResults) -> None:
        super().__init__(builder.clustingResult, builder.gdp, builder._indicator, builder._colorGroup, builder.path)
        self.analysisType = builder._analysisType
        self.analysisValue = builder._analysisValue
        self.indicator = self._indicator
        self.colorGroup = self._colorGroup
        if self.indicator == "gdp":
            self.df = self.clustingResult.dropna(subset=[self.analysisType]).set_index("name").join(
                self.gdp
            )
            self.clusterStats = self.df.groupby(self.analysisType).agg({
                u"GDP(亿元)": ["mean", "median", "std"],
                u"人均GDP(元)": ["mean", "median"], # Per Capita GDP
                u"第一产业占比(%)": "mean", # Portion of Primary Industry
                u"第二产业占比(%)": "mean", # Portion of Secondary Industry
                u"第三产业占比(%)": "mean" # Portion of Tertiary Industry
            }).reset_index()

            self.clusterStats.columns = [
                "clusting", "avgGDP", "midGDP", "stdGDP",
                "avgPGDP", "midPGDP",
                "PPI", "PSI", "PTI"
            ]

        elif self.indicator == "ev":
            self.df = self.clustingResult.dropna(subset=[self.analysisType]).set_index("name").join(
                self.gdp[[u"城市", u"保有量"]].set_index(u"城市")
            )
            self.clusterStats = self.df.groupby(self.analysisType).agg({
                u"保有量": ["mean", "median", "std"]
            }).reset_index()

            self.clusterStats.columns = ["clusting", "avgEV", "midEV", "stdEV"]
        
        elif self.indicator == "urban":
            def calculateStats(group):
                return pd.Series({
                    "weighted_mean": np.average(group["Urban_Ratio"], weights=group["Shape_Area"]),
                    "median": group['Urban_Ratio'].median(),
                    "std": group['Urban_Ratio'].std(),
                })
            
            self.df = self.clustingResult.dropna(subset=[self.analysisType]).set_index("name").join(
                self.gdp.loc[self.gdp["Urban"] == "Urban", ["name", "Shape_Area", "Urban_Ratio"]].set_index("name")
            ).fillna(0)
            self.clusterStats = self.df.groupby(self.analysisType).apply(calculateStats).reset_index()

            self.clusterStats.columns = ["clusting", "avgRatio", "midRatio", "stdRatio"]

        else:
            raise RuntimeError("Unrecognized indicator.")

        return
    
    def analysis(self) -> None:
        for cluster in self.clusterStats["clusting"].unique():
            data = self.clusterStats[self.clusterStats["clusting"] == cluster].iloc[0]
            if self.indicator == "gdp":
                print(f"\n=== Economic index of {cluster} ===")
                print(f"• Average GDP {data["avgGDP"]:.02f} (Stander Division: {data["stdGDP"]:.02f})")
                print(f"• Average Per Capita GDP {data["avgPGDP"]:.02f}")
                print(f"• Industrial Structue: Primary Industry {data["PPI"]:.02f}% | Secondary Industru {data["PSI"]:.02f}% | Tertiary Industry {data["PTI"]:.02f}%")
            
            elif self.indicator == "ev":
                print(f"\n=== EV stature of {cluster} ===")
                print(f"• Average EV number {data["avgEV"]:.02f} (Stander Division: {data["stdEV"]:.02f})")
                print(f"• Midean EV number {data["midEV"]:.02f}")
            
            elif self.indicator == "urban":
                print(f"\n=== Urbanization stature of {cluster} ===")
                print(f"• Average urbanization ratio {data["avgRatio"]:.02f} (Stander Division: {data["stdRatio"]:.02f})")
                print(f"• Midean urbanization ratio {data["midRatio"]:.02f}")

            else:
                raise RuntimeError("Unrecognized indicator.")

        return
    
    def drawClusting(self, figsize: str = "HH") -> None:
        if self.analysisValue == "":
            print("Please specifice a sub clusting type.")
            return
        
        years = list(range(2015, 2026))
        xPositions = np.arange(len(years))
        data = self.df.copy()
        data = data[["{}_{}".format(self.analysisValue, y) for y in years] + [self.analysisType]]

        clustering: list = data[self.analysisType].unique().tolist()
        clustering.sort()

        colors = BAR_COLORS[self.colorGroup]
        from multiFigs import multiFigs
        fig = multiFigs(1, len(clustering), figsize=figsize, sharex=True)
        axs = fig.axs
        for i, clusterId in enumerate(clustering):
            clusterData = data.loc[data[self.analysisType] == clusterId].drop(columns=self.analysisType)
            
            axs[i].plot(
                xPositions,
                np.nanmedian(clusterData, axis=0),
                label="Median of {}".format(clusterId),
                color=colors[i],
                alpha=0.8
            )
            axs[i].plot(
                xPositions,
                np.nanmax(clusterData, axis=0),
                label="Max of {}".format(clusterId),
                linestyle="--",
                color=colors[i],
                alpha=0.8
            )
            axs[i].plot(
                xPositions,
                np.nanmin(clusterData, axis=0),
                label="Minial of {}".format(clusterId),
                linestyle="dashdot",
                color=colors[i],
                alpha=0.8
            )
            axs[i].fill_between(
                xPositions,
                np.nanmin(clusterData, axis=0),
                np.nanmax(clusterData, axis=0),
                alpha=0.1, color=colors[i]
            )
            
            axs[i].set_xticklabels([None] + [str(x) for x in range(2015, 2027, 2)] + [None]) # type: ignore
            axs[i].set_yticks([x / 10 for x in range(0, 11, 2)], [str(x / 10) for x in range(0, 11, 2)])
            axs[i].set_ylim(0.2, 1)

        fig.globalXlabel("Year", lens=[-1])
        fig.supylabel("{} Index".format(TITLE.get(self.analysisValue)), x=0.05)
        if self.path != "":
            fig.save(os.path.join(self.path, "{}.jpg".format(self.analysisType)), dpi=300)
        else:
            plt.show()

        return
    
if __name__ == "__main__":
    a = pd.read_csv("China_Acc_Results\\Result\\city_with_clusting.csv", encoding="utf-8")
    gdp = pd.read_excel("China_Acc_Results\\Result\\city_gdponly.xlsx").set_index(u"区县")
    ev = pd.read_excel("China_Acc_Results\\Result\\China_2022_EV_ownership.xlsx")
    b = showClustingResults(a, gdp, path=r".\\paper\\figure\\fig2").analysisOptAcc()
    # b.drawRadar()
    # b.showTime()
    b.drawClusting(figsize="SHH")
    b = showClustingResults(a, gdp, colorGroup=1, path=r".\\paper\\figure\\fig3").analysisEquity()
    # b.analysis()
    # b.drawRadar((18,8))
    b.drawClusting(figsize="SD")
    # b.showTime()
    # b = showClustingResults(a, urban, indicator="urban").analysisAll()
    # b.analysis()
    