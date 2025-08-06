import os
import pandas as pd

from .setting import INDEX, COLUMNS, OTHER_COLUMNS

# Merge different method in all years into one table
def mergeData(PATH: str) -> None:
    SFCA = os.path.join(PATH, "2SFCA", "d0=1")
    M2SFCA = os.path.join(PATH, "M2SFCA", "d0=1")
    DF: list[pd.DataFrame] = []
    for y in range(2015, 2026):
        df1 = pd.read_csv(os.path.join(SFCA, str(y), "{}_city_accessibility_summaries.csv".format(y))).set_index(INDEX).rename(columns={"Average_Accessibility": "2SFCA_Accessibility"})
        df2 = pd.read_csv(os.path.join(M2SFCA, str(y), "{}_city_accessibility_summaries.csv".format(y)))[[INDEX] + ["Average_Accessibility"]].set_index(INDEX).rename(columns={"Average_Accessibility": "M2SFCA_Accessibility"})
        df = df1.join(df2)
        df["Relative_Accessibility"] = df["M2SFCA_Accessibility"] / df["2SFCA_Accessibility"]
        df3 = pd.read_csv(os.path.join(SFCA, str(y), "{}_city_gini_summaries.csv".format(y)))[[INDEX] + ["Gini_Coefficient"]].set_index(INDEX).rename(columns={"Gini_Coefficient": "2SFCA_Gini"})
        df4 = pd.read_csv(os.path.join(M2SFCA, str(y), "{}_city_gini_summaries.csv".format(y)))[[INDEX] + ["Gini_Coefficient"]].set_index(INDEX).rename(columns={"Gini_Coefficient": "M2SFCA_Gini"})
        df = df.join(df3)
        df = df.join(df4)
        df.rename(columns={x: "{}_{}".format(x, y) for x in COLUMNS}, inplace=True)
        DF.append(df)
    df = DF[0]
    for df1 in DF[1:]:
        df = df.join(df1.drop(columns=OTHER_COLUMNS))
    df.to_csv(os.path.join(PATH, "Result", "city_accessibility.csv"), encoding="utf-8")

    return