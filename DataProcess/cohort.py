import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .setting import plotSet, FIG_SIZE, BAR_COLORS
except:
    from setting import plotSet, FIG_SIZE, BAR_COLORS

# def cohort(RESULT: pd.DataFrame, colName: str) -> None:
#     plotSet()
#     matrix = []
#     for i in range(2015, 2026):
#         cohort = [0.0] * (i - 2015)
#         # Only filter the rows that the relative accessibility is NA
#         data = RESULT[~RESULT["Relative_Accessibility_{}".format(i)].isna()]
#         # And the former years are NA
#         if i > 2015:
#             for j in range(2015, i):
#                 data = data[data["Relative_Accessibility_{}".format(j)].isna()]
#         if data.shape[0] == 0:
#             for j in range(i, 2026):
#                 cohort.append(0)
#         else:
#             for j in range(i, 2026):
#                 subdata = data["{}_{}".format(colName, j)]
#                 cohort.append(subdata.median())
#         matrix.append(cohort)

#     # Convert to numpy and set 0 as NaN
#     matrix = np.array(matrix, dtype=float)
#     matrix[matrix == 0] = np.nan

#     plt.figure(figsize=FIG_SIZE.D)
#     sns.heatmap(
#         matrix,
#         cmap="Reds",
#         linewidths=0.5
#     ) 

#     plt.title("Pop.-Based {}".format(TITLE[colName]))

#     plt.show()
#     plt.close()

#     return

def cohort(RESULT: pd.DataFrame, colName: str, colorGroup: list[int], savePath: str = "") -> None:
    plotSet()
    TITLE = {"M2SFCA_Gini": "Equity", "Relative_Accessibility":"Efficiency"}
    years = list(range(2015, 2026))
    matrix = []
    for i in years:
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
    xticks = range(2015, 2026)

    plt.figure(figsize=FIG_SIZE.D)
    colors = []
    for x in colorGroup:
        colors.extend(BAR_COLORS[x])

    skip = 0
    for i, group in enumerate(matrix):
        if np.isnan(group).all():
            skip += 1
            continue
        plt.plot(xticks, group, label="{}".format(xticks[i]), color=colors[i - skip])

    plt.xlabel("Year")
    plt.xticks(years, years) # type: ignore
    plt.ylabel(
        "Median of {} Index".format(TITLE.get(colName))
    )
    plt.legend()

    plt.tight_layout()
    if savePath == "":
        plt.show()
    else:
        plt.savefig(os.path.join(savePath, "chorts.jpg"), dpi=300)

    plt.close()

    return

if __name__ == "__main__":
    import os
    RESULT = pd.read_csv(os.path.join("China_Acc_Results", "Result", "city_efficiency.csv"), encoding="utf-8")
    # cohort(RESULT, "Relative_Accessibility")
    cohort(RESULT, "M2SFCA_Gini", [-1], r".\\paper\\figure\\fig3")