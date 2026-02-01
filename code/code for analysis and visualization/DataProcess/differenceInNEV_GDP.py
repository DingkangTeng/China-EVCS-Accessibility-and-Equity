import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, skew, kurtosis, normaltest

try:
    from .setting import plotSet, FIG_SIZE, TITLE, BAR_COLORS
except:
    from setting import plotSet, FIG_SIZE, TITLE, BAR_COLORS

class differenceInNEV_GDP:
    __slots__ = ["df", "value", "classify"]

    def __init__(self, RESULT:pd.DataFrame, combine: tuple[pd.DataFrame, str] | str, selfClassify: list = []) -> None:
        plotSet()
        if isinstance(combine, tuple):
            self.df = RESULT.join(combine[0][combine[1]], on="name")
            self.value = TITLE.get(combine[1], "") # Group title
            self.df.rename(columns={combine[1]: self.value}, inplace=True)
        else:
            self.df = RESULT
            self.value = combine # Group title

        if selfClassify == []:
            self.classify = ["{} {}".format(x, TITLE.get(combine[1])) for x in ["Low", "Median", "High"]]
            Q1, Q2 = self.analyzeDistribution(self.df[self.value])
            self.df[self.value] = pd.cut(
                self.df[self.value],
                [self.df[self.value].min(), Q1, Q2, self.df[self.value].max()],
                labels=self.classify
            )
        else:
            self.classify = selfClassify

        return
    
    @property
    def comparyClassify(self) -> tuple[pd.DataFrame, str]:
        return self.df[["name", self.value]], self.value
    
    @staticmethod
    def analyzeDistribution(data: pd.Series) -> tuple[float, float]:
        data = data.dropna()
        skewness = skew(data)
        kur = kurtosis(data)

        # Plot data
        fig, axes = plt.subplots(figsize=FIG_SIZE.D)
        sns.kdeplot(pd.DataFrame(data), ax=axes)
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # normal test
        _, p_value = normaltest(data)
        if p_value > 0.05 and abs(skewness) < 0.5:
            if abs(kur) < 1:
                return data.quantile(0.25), data.quantile(0.75) # standar normal
            else:
                return data.quantile(0.25), data.quantile(0.75) # normal distribution but with kurtosis anomaly
        else:
            if skewness > 0:
                return data.quantile(0.50), data.quantile(0.75) # right skewed
            else:
                return data.quantile(0.25), data.quantile(0.50) # left skewed

    def boxPlot(
        self,
        colName: str,
        ylim: tuple[float, float] = (0.0, 0.0),
        figsize: str = "D", legLoc: str = "upper right",
        colorGroup: int = 0,
        savePath: str = ""
    ) -> None:
        years = list(str(x) for x in range(2015,2026))
        palette = {x: y for x,y in zip(self.classify, BAR_COLORS[colorGroup])}

        resultCol = ["{}_{}".format(colName, y) for y in years]
        dfMelted = self.df[resultCol + [self.value]].melt(
            id_vars=self.value, 
            var_name="Year",
            value_name=colName
        )
        
        plt.figure(figsize=getattr(FIG_SIZE, figsize))
        ax = sns.boxplot(
            data=dfMelted,
            x="Year", 
            y=colName, 
            hue=self.value,  # Class color based on different group
            palette=palette,  # Custom color list
            patch_artist = True,
            boxprops = {"edgecolor": "gray"},
            flierprops = {"marker":'.', "markerfacecolor":"gray", "markeredgecolor": "none"},
            medianprops = {"color": "white"},
            whiskerprops = {"color": "gray"},
            capprops = {"color": "gray"}
        )

        ax.set_xticklabels(["" if int(x) % 2 ==0 else x for x in years])
        ax.set_ylabel("{} Index".format(TITLE.get(colName)))

        if ylim != (0.0, 0.0):
            plt.ylim(ylim)
        else:
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])

        plt.legend(loc=legLoc)
        plt.tight_layout()

        if savePath == "":
            plt.show()
            plt.close()
        else:
            plt.savefig(os.path.join(savePath, "{}_{}.jpg".format(self.value, colName)), dpi=300)
            plt.close()

        return

def spearman(class1: differenceInNEV_GDP, class2: differenceInNEV_GDP) -> None:
    a, aCol = class1.comparyClassify
    b, bCol = class2.comparyClassify

    c = a.set_index("name").join(b.set_index("name"))
    c.dropna(inplace=True)
    c[aCol] = c[aCol].str.split(' ').str[0]
    c[bCol] = c[bCol].str.split(' ').str[0]
    rankMapping = {
        "Low": 1, "Median": 2, "High": 3,
        "Tourist": 2, "Not": 1
    }
    c['a_rank'] = c[aCol].map(rankMapping)
    c['b_rank'] = c[bCol].map(rankMapping)

    correlation, p = spearmanr(c['a_rank'], c['b_rank'])

    print(f"Spearman correlation coefficient ρ: {correlation:.4f}")
    print(f"P Value: {p:.4f}")

    return

# Debug
if __name__ == "__main__":
    ROW_DATA_PATH = r"C:\Users\tengd\OneDrive - The Hong Kong Polytechnic University\Student Assistant\ChinaDynam\_RowData"
    FIG_PATH = r"C:\Users\tengd\OneDrive - The Hong Kong Polytechnic University\Student Assistant\ChinaDynam\_AnalysisData\figure"
    AGG_DATA = r"C:\Users\tengd\OneDrive - The Hong Kong Polytechnic University\Student Assistant\ChinaDynam\_AnalysisData\result\AggResult"

    RESULT = pd.read_csv(os.path.join(AGG_DATA, "city_optAcc.csv"), encoding="utf-8")

    nev = pd.read_excel(os.path.join(ROW_DATA_PATH, "China_2022_NEV_ownership.xlsx")).set_index(u"城市")
    gdp = pd.read_excel(os.path.join(ROW_DATA_PATH, "city_gdponly.xlsx")).set_index(u"区县")
    
    a = differenceInNEV_GDP(RESULT, (nev, u"保有量"))
    a.boxPlot("M2SFCA_Gini", (0.5, 1), legLoc="lower left", colorGroup = 1, savePath=os.path.join(FIG_PATH, "fig3"))
    a.boxPlot("Relative_Accessibility", (0.5, 0.8), figsize="SM", savePath=os.path.join(FIG_PATH, "fig2"))
    
    b = differenceInNEV_GDP(RESULT, (gdp, u"人均GDP(元)"))
    b.boxPlot("M2SFCA_Gini", (0.5, 1), legLoc="lower left", colorGroup = 1, savePath=os.path.join(FIG_PATH, "fig3"))
    b.boxPlot("Relative_Accessibility", (0.5, 0.8), figsize="SM", savePath=os.path.join(FIG_PATH, "fig2"))
    
    # # # Spearman's rank correlation coefficient
    # # spearman(a, b)