import pandas as pd
import matplotlib.pyplot as plt

try:
    from .setting import ECO_COL
except:
    from setting import ECO_COL

class clustingAnalysis:
    __slots__ = ["clustingResult", "gdp", "_analysisType"]

    def __init__(self, clustingResult: pd.DataFrame, gdp: pd.DataFrame) -> None:
        self.clustingResult = clustingResult
        self.gdp = gdp
        self._analysisType: str | None = None

    def analysisAll(self) -> "_AnalysisExecutorImpl":
        self._analysisType = "clusting"
        return _AnalysisExecutorImpl(self)
    
    def analysisEquity(self) -> "_AnalysisExecutorImpl":
        self._analysisType =  "clusting_equity"
        return _AnalysisExecutorImpl(self)

    def analysisEfficiency(self)  -> "_AnalysisExecutorImpl":
        self._analysisType = "clusting_efficiency"
        return _AnalysisExecutorImpl(self)

class _AnalysisExecutorImpl(clustingAnalysis):
    __slots__ = ["clusterStats"]

    def __init__(self, builder: clustingAnalysis) -> None:
        super().__init__(builder.clustingResult, builder.gdp)
        df = self.clustingResult.dropna(subset=[builder._analysisType]).set_index("name").join(self.gdp[[u"区县"] + ECO_COL].set_index(u"区县"))
        self.clusterStats = df.groupby(builder._analysisType).agg({
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
            ax.plot(angles, values, linewidth=2, label=cluster)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["Portion of Primary Industry", "Portion of Secondary Industry", "Portion of Tertiary Industry"])
        plt.title("Comparison of Industrial Structure")
        plt.legend()
        plt.show()

        return
    
if __name__ == "__main__":
    a = pd.read_csv("China_Acc_Results/Result/city_with_clusting.csv", encoding="utf-8")
    gdp = pd.read_excel("China_Acc_Results/Result/city_gdponly.xlsx")
    b = clustingAnalysis(a, gdp).analysisEquity()
    b.analysis()
    b.drawRadar()