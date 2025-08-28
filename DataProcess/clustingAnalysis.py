import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

try:
    from .setting import ECO_COL, plotSet, FIG_SIZE, TITLE
except:
    from setting import ECO_COL, plotSet, FIG_SIZE, TITLE

class clustingAnalysis:
    __slots__ = ["clustingResult", "gdp", "path", "_indicator", "_analysisType", "_analysisValue"]

    def __init__(self, clustingResult: pd.DataFrame, gdp: pd.DataFrame, indicator: str = "gdp", path: str = "") -> None:
        self.clustingResult = clustingResult
        self.gdp = gdp
        self.path = path
        self._indicator = indicator
        self._analysisType: str
        self._analysisValue: str
        plotSet()
        
    def analysisAll(self, andlysisValue: str = "") -> "_AnalysisExecutorImpl":
        self._analysisType = "clusting"
        self._analysisValue = andlysisValue
        return _AnalysisExecutorImpl(self)
    
    def analysisEquity(self) -> "_AnalysisExecutorImpl":
        self._analysisType =  "clusting_equity"
        self._analysisValue = "M2SFCA_Gini"
        return _AnalysisExecutorImpl(self)

    def analysisEfficiency(self)  -> "_AnalysisExecutorImpl":
        self._analysisType = "clusting_efficiency"
        self._analysisValue = "Relative_Accessibility"
        return _AnalysisExecutorImpl(self)
    
    def sankey(self, path: str = "") -> None:
        df = self.clustingResult[["name", "clusting_equity", "clusting_efficiency"]].copy().dropna()
        grouped = df.groupby(["clusting_equity", "clusting_efficiency"]).size().reset_index(name='count')

        # Build node
        labels = list(pd.unique(df["clusting_equity"])) + list(pd.unique(df["clusting_efficiency"]))
        labelDict = {label: i for i, label in enumerate(labels)}

        # Build sources and targets
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
    
    def heat(self) -> None:
        df = self.clustingResult[["name", "clusting_equity", "clusting_efficiency"]].copy().dropna()
        grouped = df.groupby(["clusting_equity", "clusting_efficiency"]).size().reset_index(name='count')
        equity = grouped["clusting_equity"].unique().tolist()
        efficiency = grouped["clusting_efficiency"].unique().tolist()
        matrix = []

        for i in equity:
            col = []
            for j in efficiency:
                col.append(grouped.loc[(grouped["clusting_equity"] == i) & (grouped["clusting_efficiency"] == j), "count"].values[0]) # type: ignore
            matrix.append(col)

        plt.figure(figsize=FIG_SIZE)
        ax = sns.heatmap(
            matrix,
            yticklabels=[' '.join(x.split(' ')[0:2]) for x in equity],
            xticklabels=[x.split(' ')[0] for x in efficiency],
            cmap="Reds",
            linewidths=0.5
        )
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.set_label("Number of Cities", fontweight="bold")
        
        ax.set_ylabel(
            "Equlity Clustering",
            fontweight="bold"
        )
        ax.set_xlabel(
            "Efficiency Clustering",
            fontweight="bold"
        )
        plt.xticks(rotation=0)
        plt.tight_layout()
        if self.path == "":
            plt.show()
        else:
            plt.savefig(os.path.join(self.path, "Efficiency and Equity Heat Map.jpg"))
        plt.close()

        return

class _AnalysisExecutorImpl(clustingAnalysis):
    __slots__ = ["clusterStats", "df", "indicator", "analysisType", "analysisValue"]

    def __init__(self, builder: clustingAnalysis) -> None:
        super().__init__(builder.clustingResult, builder.gdp, builder._indicator, builder.path)
        self.analysisType = builder._analysisType
        self.analysisValue = builder._analysisValue
        self.indicator = self._indicator
        if self.indicator == "gdp":
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
        elif self.indicator == "ev":
            self.df = self.clustingResult.dropna(subset=[self.analysisType]).set_index("name").join(self.gdp[[u"城市", u"保有量"]].set_index(u"城市"))
            self.clusterStats = self.df.groupby(self.analysisType).agg({
                u"保有量": ["mean", "median", "std"]
            }).reset_index()

            self.clusterStats.columns = ["clusting", "avgEV", "midEV", "stdEV"]
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
            else:
                raise RuntimeError("Unrecognized indicator.")

        return
    
    def drawRadar(self, figsize: tuple[int, int] = FIG_SIZE) -> None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(polar=True)
        ax.grid(True)
        plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

        if self.indicator == "gdp":
            categories = ["PPI", "PSI", "PTI"]
            xticklabels = ["Portion of\nprimary\nindustry", "Portion of\nsecondary industry", "Portion of\ntertiary industry"]
        elif self.indicator == "ev":
            categories = xticklabels = self.clusterStats["clusting"]
        else:
            raise RuntimeError("Unrecognized indicator.")
        
        angles = [n / float(len(categories)) * 2 * 3.14 for n in range(len(categories))]
        angles += angles[:1]
        
        if self.indicator == "gdp":
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
        elif self.indicator == "ev":
            values = self.clusterStats[["avgEV"]].values.tolist()
            values += values[:1]  # close graph
            ax.plot(angles, values, linewidth=2, label="Average EV number")
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            xticklabels,
            ha='center',
            va='center'
        )
        # Adjust label position
        for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
            x, y = label.get_position()
            label.set_position((x, y-0.15))
        plt.legend(
            loc="lower right",
            bbox_to_anchor=(0, 0)
        )
        
        plt.tight_layout()
        if self.path != "":
            plt.savefig(os.path.join(self.path, "{}_{}_radar.jpg".format(self.indicator, self.analysisType)), dpi=300)
        else:
            plt.show()
        plt.close()

        return
    
    def showTime(self) -> None:
        data = self.df.copy()
        data["earlistYear"] = 2025
        for y in range(2015, 2025):
            col = "Relative_Accessibility_{}".format(y) # Using Relative_Accessibility to determin the first deployment time
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

        fig = plt.figure(figsize=FIG_SIZE)
        ax = fig.add_subplot()

        data.plot.bar(ax=ax, color=["#BAD540", "#85A2D0", "#FFC339"])

        # for container in ax.containers:
        #     ax.bar_label(container, fmt="%.2f", label_type='edge')

        plt.xlabel(
            "Year",
            fontweight="bold",
        )
        plt.ylabel(
            "Ratio of Cities First Deploy\nCharging Facilities (%)",
            fontweight="bold",
        )
        plt.xticks(rotation = 0)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        if self.path != "":
            plt.savefig(os.path.join(self.path, "{}_timeDistribution.jpg".format(self.analysisType)), dpi=300)
        else:
            plt.show()
        plt.close()

        return
    
    def drawClusting(self) -> None:
        if self.analysisValue == "":
            print("Please specifice a sub clusting type.")
            return
        
        years = list(range(2015, 2026))
        xPositions = np.arange(len(years))
        data = self.df.copy()
        data = data[["{}_{}".format(self.analysisValue, y) for y in years] + [self.analysisType]]
        
        colors = ["#E24A33", "#348ABD", "#988ED5"]
        # plt.figure(figsize=FIG_SIZE)
        
        for i, clusterId in enumerate(data[self.analysisType].unique().tolist()):
            clusterData = data.loc[data[self.analysisType] == clusterId].drop(columns=self.analysisType)
            plt.figure(figsize=FIG_SIZE)
            
            plt.plot(
                xPositions,
                np.nanmedian(clusterData, axis=0),
                label="Median of {}".format(clusterId.lower()),
                color=colors[i]
            )
            plt.plot(
                xPositions,
                np.nanmax(clusterData, axis=0),
                label="Max boundary of {}".format(clusterId.lower()),
                linestyle="--",
                color=colors[i],
                alpha=0.7
            )
            plt.plot(
                xPositions,
                np.nanmin(clusterData, axis=0),
                label="Minial boundary of {}".format(clusterId.lower()),
                linestyle="dashdot",
                color=colors[i],
                alpha=0.7
            )
            plt.fill_between(xPositions, np.nanmin(clusterData, axis=0), np.nanmax(clusterData, axis=0), alpha=0.1, color=colors[i])

            plt.xlabel(
                "Year",
                fontweight="bold",
            )
            plt.ylabel(
                "{} Index".format(TITLE.get(self.analysisValue)),
                fontweight="bold",
            )
            plt.yticks([x / 10 for x in range(0, 11, 2)], [str(x / 10) for x in range(0, 11, 2)])
            plt.gca().set_xticklabels([None] + [str(x) for x in range(2015, 2027, 2)] + [None]) # type: ignore
            plt.legend(loc="lower left")
            plt.tight_layout()

            if self.path != "":
                plt.savefig(os.path.join(self.path, "{}_{}.jpg".format(self.analysisType, clusterId)), dpi=300)
            else:
                plt.show()
            plt.close()

        # plt.yticks([x / 10 for x in range(0, 11, 2)], [str(x / 10) for x in range(0, 11, 2)])
        # plt.gca().set_xticklabels([None] + [str(x) for x in range(2015, 2027, 2)] + [None]) # type: ignore
        # plt.legend()
        # plt.show()

        return
    
if __name__ == "__main__":
    a = pd.read_csv("China_Acc_Results/Result/city_with_clusting.csv", encoding="utf-8")
    gdp = pd.read_excel("China_Acc_Results/Result/city_gdponly.xlsx")
    ev = pd.read_excel("China_Acc_Results/Result/China_2022_EV_ownership.xlsx")
    # b = clustingAnalysis(a, gdp, path=r".\\paper\\figure").analysisEfficiency()
    # b.drawRadar()
    # b.showTime()
    # b.drawClusting()
    # b = clustingAnalysis(a, gdp, path=r".\\paper\\figure").analysisAll()
    # b.analysis()
    # b.drawRadar((18,8))
    # b.drawClusting()
    # b.showTime()
    clustingAnalysis(a, gdp, path=r".\\paper\\figure").heat()