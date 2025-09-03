import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .setting import plotSet, FIG_SIZE, TITLE, BAR_COLORS
except:
    from setting import plotSet, FIG_SIZE, TITLE, BAR_COLORS

class combinAnalysis:
    __slots__ = ["df", "value", "classify"]

    def __init__(self, RESULT:pd.DataFrame, combine: tuple[pd.DataFrame, str], selfClassify: list = []) -> None:
        plotSet()
        self.df = RESULT.join(combine[0][combine[1]], on="name")
        self.value = "{} Group".format(TITLE.get(combine[1])) # Group title
        self.df.rename(columns={combine[1]: self.value}, inplace=True)
        if selfClassify == []:
            self.classify = ["{} {}".format(x, TITLE.get(combine[1])) for x in ["Low", "Median", "High"]]
            Q2 = self.df[self.value].quantile(0.50)
            Q3 = self.df[self.value].quantile(0.75)
            self.df[self.value] = pd.cut(
                self.df[self.value],
                [self.df[self.value].min(), Q2, Q3, self.df[self.value].max()],
                labels=self.classify
            )
        else:
            self.classify = selfClassify

        return
    
    @property
    def comparyClassify(self) -> tuple[pd.DataFrame, str]:
        return self.df[["name", self.value]], self.value

    def boxPlot(self, colName: str) -> None:
        years = list(str(x) for x in range(2015,2026))
        palette = {x: y for x,y in zip(self.classify, BAR_COLORS)}

        resultCol = ["{}_{}".format(colName, y) for y in years]
        dfMelted = self.df[resultCol + [self.value]].melt(
            id_vars=self.value, 
            var_name="Year",
            value_name=colName
        )
        
        plt.figure(figsize=FIG_SIZE)
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

        ax.set_xticklabels(years)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
        ax.set_ylabel("{} Indeics".format(TITLE.get(colName)))

        plt.show()

        return

def spearman(class1: combinAnalysis, class2: combinAnalysis) -> None:
    from scipy.stats import spearmanr
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

    print(f"斯皮尔曼相关系数 ρ: {correlation:.4f}")
    print(f"P值: {p:.4f}")

    return

# Debug
if __name__ == "__main__":
    import os
    import numpy as np
    RESULT = pd.read_csv(os.path.join("China_Acc_Results", "Result", "city_efficiency.csv"), encoding="utf-8")
    RESULT = RESULT[RESULT["name"] != u"境界线"]
    for y in range(2015, 2026):
        RESULT.loc[RESULT["Relative_Accessibility_{}".format(y)].isna(), "M2SFCA_Gini_{}".format(y)] = np.nan

    ev = pd.read_excel("China_Acc_Results\\Result\\China_2022_EV_ownership.xlsx").set_index(u"城市")
    gdp = pd.read_excel("China_Acc_Results\\Result\\city_gdponly.xlsx").set_index(u"区县")
    tour = pd.read_csv("China_Acc_Results\\Result\\city_tour.csv", encoding="utf-8").set_index("name")
    
    a = combinAnalysis(RESULT, (ev, u"保有量"))
    # # a.boxPlot("M2SFCA_Gini")
    # # a.boxPlot("Relative_Accessibility")
    
    b = combinAnalysis(RESULT, (gdp, u"人均GDP(元)"))
    
    # # Spearman's rank correlation coefficient
    # spearman(a, b)
    
    c = combinAnalysis(RESULT, (tour, "tour"), ["Tourist City", "Not Tourist City"])
    # c.boxPlot("M2SFCA_Gini")
    # c.boxPlot("Relative_Accessibility")
    # spearman(c, a)
    
    df = b.comparyClassify[0].join(c.comparyClassify[0].set_index("name"), on="name")
    df = df.loc[(df["GDP Group"] == "High GDP"), "name"].to_list()
    df = RESULT.loc[RESULT["name"].isin(df)]
    c = combinAnalysis(df, (tour, "tour"), ["Tourist City", "Not Tourist City"])
    c.boxPlot("M2SFCA_Gini")
    c.boxPlot("Relative_Accessibility")