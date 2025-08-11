import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .setting import FONT_SIZE, plotSet
except:
    from setting import FONT_SIZE, plotSet

def cohort(RESULT: pd.DataFrame, colName: str) -> None:
    plotSet()
    TITLE = {"M2SFCA_Gini": "Equity", "Relative_Accessibility":"Efficiency"}
    matrix = []
    for i in range(2015, 2026):
        cohort = [0.0] * (i - 2015)
        # Only filter the rows that the relative accessibility is NA
        data = RESULT[~RESULT["Relative_Accessibility_{}".format(i)].isna()]
        # And the former years are NA
        if i > 2015:
            for j in range(2015, i):
                data = data[data["Relative_Accessibility_{}".format(j)].isna()]
        if data.shape[0] == 0:
            for j in range(i, 2026):
                cohort.append(0)
        else:
            for j in range(i, 2026):
                subdata = data["{}_{}".format(colName, j)]
                cohort.append(subdata.median())
        matrix.append(cohort)

    # Convert to numpy and set 0 as NaN
    matrix = np.array(matrix, dtype=float)
    matrix[matrix == 0] = np.nan

    sns.heatmap(
        matrix,
        cmap="Reds",
        linewidths=.5
    ) 

    plt.title("Pop.-Based {}".format(TITLE[colName]), fontsize=FONT_SIZE)

    plt.show()

    return

if __name__ == "__main__":
    import os
    RESULT = pd.read_csv(os.path.join("China_Acc_Results", "Result", "city_efficiency.csv"), encoding="utf-8")
    RESULT = RESULT[RESULT["name"] != u"境界线"]
    print(RESULT.shape[0])
    cohort(RESULT, "Relative_Accessibility")
    cohort(RESULT, "M2SFCA_Gini")