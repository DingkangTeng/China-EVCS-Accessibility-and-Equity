import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

try:
    from .setting import ECO_COL, plotSet
except:
    from setting import ECO_COL, plotSet

class clustingAnalysis:
    __slots__ = ["clustingResult", "gdp", "_analysisType", "_analysisValue"]

    def __init__(self, clustingResult: pd.DataFrame, gdp: pd.DataFrame) -> None:
        self.clustingResult = clustingResult
        self.gdp = gdp
        self._analysisType: str | None = None
        self._analysisValue: str | None = None
        
    def analysisAll(self) -> "_AnalysisExecutorImpl":
        self._analysisType = "clusting"
        return _AnalysisExecutorImpl(self)
    
    def analysisEquity(self) -> "_AnalysisExecutorImpl":
        self._analysisType =  "clusting_equity"
        self._analysisValue = "M2SFCA_Gini_"
        return _AnalysisExecutorImpl(self)

    def analysisEfficiency(self)  -> "_AnalysisExecutorImpl":
        self._analysisType = "clusting_efficiency"
        self._analysisValue = "Relative_Accessibility_"
        return _AnalysisExecutorImpl(self)
    
    def sankey(self, path: str = "") -> None:
        df = self.clustingResult[["name", "clusting_equity", "clusting_efficiency"]].copy().dropna()
        grouped = df.groupby(["clusting_equity", "clusting_efficiency"]).size().reset_index(name='count')

        # 构建节点列表
        labels = list(pd.unique(df["clusting_equity"])) + list(pd.unique(df["clusting_efficiency"]))
        labelDict = {label: i for i, label in enumerate(labels)}

        # 构建源和目标
        sources = grouped["clusting_equity"].map(labelDict)
        targets = grouped["clusting_efficiency"].map(labelDict)

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=grouped['count']
            ))])

        # fig.show()
        fig.write_html(os.path.join(path, "sankey.html"))

        return

class _AnalysisExecutorImpl(clustingAnalysis):
    __slots__ = ["clusterStats", "df", "analysisType", "analysisValue"]

    def __init__(self, builder: clustingAnalysis) -> None:
        super().__init__(builder.clustingResult, builder.gdp)
        plotSet()
        self.analysisType = builder._analysisType
        self.analysisValue = builder._analysisValue
        self.df = self.clustingResult.dropna(subset=[self.analysisType]).set_index("name").join(self.gdp[[u"区县"] + ECO_COL].set_index(u"区县"))
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

        return
    
    def analysis(self) -> None:
        for cluster in self.clusterStats["clusting"].unique():
            data = self.clusterStats[self.clusterStats["clusting"] == cluster].iloc[0]
            print(f"\n=== Economic index of {cluster} ===")
            print(f"• Average GDP {data["avgGDP"]:.02f} (Stander Division: {data["stdGDP"]:.02f})")
            print(f"• Average Per Capita GDP {data["avgPGDP"]:.02f}")
            print(f"• Industrial Structue: Primary Industry {data["PPI"]:.02f}% | Secondary Industru {data["PSI"]:.02f}% | Tertiary Industry {data["PTI"]:.02f}%")

        return

    def drawRadar(self) -> None:
        categories = ["PPI", "PSI", "PTI"]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = [n / float(len(categories)) * 2 * 3.14 for n in range(len(categories))]
        angles += angles[:1]

        for cluster in self.clusterStats["clusting"]:
            values = self.clusterStats.loc[self.clusterStats["clusting"]==cluster, categories].values.flatten().tolist()
            values += values[:1]  # close graph
            minVal = int(np.floor(min(values) / 10) * 10)
            maxVal = int(np.ceil(max(values) / 10) * 10)
            # Change values to log
            values = np.log10(values)
            percentages = [x for x in range(minVal, maxVal+1, 10)]
            logTicks = np.log10(percentages)
            percentLabels = [f"{tick:d}%" for tick in percentages]
            ax.plot(angles, values, linewidth=2, label=cluster)
            ax.fill(angles, values, alpha=0.1)
            ax.set_yticks(logTicks)
            ax.set_yticklabels(percentLabels)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["Portion of Primary Industry", "Portion of Secondary Industry", "Portion of Tertiary Industry"])
        plt.title("Comparison of Industrial Structure")
        plt.legend()
        plt.show()

        return
    
    def showTime(self) -> None:
        data = self.df.copy()
        data["earlistYear"] = 2025
        for y in range(2015, 2025):
            col = "Relative_Accessibility_{}".format(y)
            data.loc[(~data[col].isna()) & (data["earlistYear"] > y), ["earlistYear"]] = y
        
        allSet = []
        clusting = data[self.analysisType].unique().tolist()
        for c in clusting:
            subdata: pd.DataFrame = data.loc[data[self.analysisType] == c]
            yearSet = subdata.groupby("earlistYear").size().reset_index(name=c).set_index("earlistYear")
            yearSet[c] = yearSet[c] / yearSet[c].sum() * 100
            allSet.append(yearSet)

        data: pd.DataFrame = allSet[0]
        for i in allSet[1:]:
            data = data.join(i, how="outer")
        data = data.reindex(range(2015,2023), fill_value=0)
        data.fillna(0, inplace=True)

        ax = data.plot.bar()

        # for container in ax.containers:
        #     ax.bar_label(container, fmt="%.2f", label_type='edge')

        plt.xlabel("Year")
        plt.ylabel("Ratio of Cities First Deploy Charging Facilities (%)")
        plt.xticks(rotation = 0)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close()

        return
    
    def drawClusting(self) -> None:
        if self.analysisValue is None:
            print("Please specifice a sub clusting type.")
            return
        
        years = list(range(2015, 2026))
        xPositions = np.arange(len(years))
        data = self.df.copy()
        data = data[["{}{}".format(self.analysisValue, y) for y in years] + [self.analysisType]]
        
        for clusterId in data[self.analysisType].unique().tolist():
            clusterData = data.loc[data[self.analysisType] == clusterId].drop(columns=self.analysisType)
            plt.plot(xPositions, np.nanmedian(clusterData, axis=0), label=clusterId)
            plt.fill_between(xPositions, np.nanmin(clusterData, axis=0), np.nanmax(clusterData, axis=0), alpha=0.3)
        
        plt.legend()
        plt.show()

        return
    
if __name__ == "__main__":
    a = pd.read_csv("China_Acc_Results/Result/city_with_clusting.csv", encoding="utf-8")
    gdp = pd.read_excel("China_Acc_Results/Result/city_gdponly.xlsx")
    b = clustingAnalysis(a, gdp).analysisEfficiency()
    # b.analysis()
    # b.drawRadar()
    b.drawClusting()
    # clustingAnalysis(a, gdp).sankey()