import os
import pandas as pd
import geopandas as gpd
import numpy as np

from .setting import INDEX, COLUMNS, OTHER_COLUMNS

# Merge different method in all years into one table
def mergeData(PATH: str, sub: str = "Full") -> None:
    SFCA = os.path.join(PATH, "2SFCA", "{} d0=1".format(sub))
    M2SFCA = os.path.join(PATH, "M2SFCA", "{} d0=1".format(sub))
    dfLists: list[pd.DataFrame] = []
    for y in range(2015, 2026):
        df1 = (
            pd.read_csv(os.path.join(SFCA, str(y), "{}_city_accessibility_summaries.csv".format(y))).
            set_index(INDEX).rename(columns={"Average_Accessibility": "2SFCA_Accessibility"})
        )
        df2 = (
            pd.read_csv(os.path.join(M2SFCA, str(y), "{}_city_accessibility_summaries.csv".format(y)))[[INDEX] + ["Average_Accessibility"]].
            set_index(INDEX).rename(columns={"Average_Accessibility": "M2SFCA_Accessibility"})
        )
        df = df1.join(df2)
        df["Relative_Accessibility"] = df["M2SFCA_Accessibility"] / df["2SFCA_Accessibility"]
        # df3 = pd.read_csv(os.path.join(SFCA, str(y), "{}_city_gini_summaries.csv".format(y)))[[INDEX] + ["Gini_Coefficient"]].set_index(INDEX).rename(columns={"Gini_Coefficient": "2SFCA_Gini"})
        df4 = (
            pd.read_csv(os.path.join(M2SFCA, str(y), "{}_city_gini_summaries.csv".format(y)))[[INDEX] + ["Gini_Coefficient"]].
            set_index(INDEX).rename(columns={"Gini_Coefficient": "M2SFCA_Gini"})
        )
        # df = df.join(df3)
        df = df.join(df4)
        df.rename(columns={x: "{}_{}".format(x, y) for x in COLUMNS}, inplace=True)
        dfLists.append(df)

    # Merge all dataframes    
    df = dfLists[0]
    for df1 in dfLists[1:]:
        df = df.join(df1.drop(columns=OTHER_COLUMNS))

    # Clean border and gini value
    df = df[df.index != u"境界线"]
    for y in range(2015, 2026):
        df.loc[df["Relative_Accessibility_{}".format(y)].isna(), "M2SFCA_Gini_{}".format(y)] = np.nan

    if sub == "Full":
        savePath = "city_optAcc.csv"
    else:
        savePath = "city_optAcc_{}.csv".format(sub)
    df.to_csv(os.path.join(PATH, "Result", savePath), encoding="utf-8")

    return

def calUrbanRatio(file: str | tuple[str, str], savePath: str = r"China_Acc_Results\\Result\\") -> None:
    if isinstance(file, str):
        a = gpd.read_file(file, encoding="utf-8")
    else:
        a = gpd.read_file(file[0], layer=file[1], encoding="utf-8")
    a["Urban_Ratio"] = 0
    for c in a["name"].unique().tolist():
        area = pd.Series(a.loc[a["name"] == c, "Shape_Area"]).sum()
        a.loc[a["name"] == c, "Urban_Ratio"] = round(a.loc[a["name"] == c, "Shape_Area"] / area * 100, 2)
        sumV = pd.Series(a.loc[a["name"] == c, "Urban_Ratio"]).sum()
        if sumV != 100:
            diff = 100 - sumV
            maxR = pd.Series(a.loc[a["name"] == c, "Urban_Ratio"]).max()
            if diff == 0.01 or diff == -0.01:
                if a.loc[a["name"] == c, "Urban_Ratio"][0] == a.loc[a["name"] == c, "Urban_Ratio"][1]: # type: ignore
                    a.loc[a["name"] == c, "Urban_Ratio"][0] += diff # type: ignore
                else:
                    a.loc[(a["name"] == c) & (a["Urban_Ratio"] == maxR), "Urban_Ratio"] += diff
            else:
                sp1 = round(diff / 2)
                sp2 = diff - sp1
                if diff > 0:
                    ma = max(sp1, sp2)
                    mi = min(sp1, sp2)
                else:
                    ma = min(sp1, sp2)
                    mi = max(sp1, sp2)
                if a.loc[a["name"] == c, "Urban_Ratio"][0] == a.loc[a["name"] == c, "Urban_Ratio"][1]: # type: ignore
                    a.loc[a["name"] == c, "Urban_Ratio"][0] += sp1 # type: ignore
                    a.loc[a["name"] == c, "Urban_Ratio"][1] += sp2 # type: ignore
                else:
                    a.loc[(a["name"] == c) & (a["Urban_Ratio"] == maxR), "Urban_Ratio"] += ma
                    a.loc[(a["name"] == c) & ~(a["Urban_Ratio"] == maxR), "Urban_Ratio"] += mi
    a.drop(columns="geometry").to_csv(os.path.join(savePath, "city_urbanRatio.csv"), encoding="utf-8", index=False)

    return