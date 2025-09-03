import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from .setting import plotSet, calSlop, FIG_SIZE, TITLE
except:
    from setting import plotSet, calSlop, FIG_SIZE, TITLE

# Proportional distribution with different levels of improvement in accessibility from 2015 to 2025
def improvement(RESULT: pd.DataFrame, colName: str, path: str = "") -> None:
    plotSet()
    print(RESULT.iloc[112])
    RESULT = calSlop(RESULT, colName)
    RESULT = RESULT.loc[:, ["{}_2025-2015".format(colName)]]
    negativeData = RESULT[RESULT["{}_2025-2015".format(colName)] < 0]["{}_2025-2015".format(colName)]
    positiveData = RESULT[RESULT["{}_2025-2015".format(colName)] >= 0]["{}_2025-2015".format(colName)]
    print(RESULT[RESULT["{}_2025-2015".format(colName)] >= 0])
    step = round((RESULT["{}_2025-2015".format(colName)].max() - RESULT["{}_2025-2015".format(colName)].min()) / 10, 2)
    xp = 0
    binsPositive = [xp]
    while True:
        xp += step
        binsPositive.append(xp)
        if xp > positiveData.max():
            break

    xn = 0
    binsNegative = [xn]
    while True:
        xn -= step
        binsNegative.append(xn)
        if xn < negativeData.min():
            break
    binsNegative.reverse()

    countsPositive, binEdgesPositive = np.histogram(positiveData, bins=binsPositive)
    countsNegative, binEdgesNegative = np.histogram(negativeData, bins=binsNegative)
    countNegative = len(negativeData)
    countPositive = len(positiveData)
    allCounts = np.concatenate([countsNegative, countsPositive])
    total = allCounts.sum()
    percentages = allCounts / total * 100

    # labels = ["<0"]
    # for i in range(len(binsPositive) - 1):
    #     start = binsPositive[i]
    #     end = binsPositive[i + 1]
    #     labels.append(f'{start:.2f}-{end:.2f}')
    labels = []
    boxTicks = set()
    for d in [binsNegative, binsPositive]:
        for i in range(len(d) - 1):
            start = d[i]
            end = d[i + 1]
            boxTicks.add(start)
            boxTicks.add(end)
            labels.append(f'{start:.2f}-{end:.2f}')
    boxTicks = list(boxTicks)
    boxTicks.sort()

    plt.figure(figsize=FIG_SIZE)
    gs = GridSpec(1, 2, width_ratios=[10, 1], wspace=0.05)
    ax = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    xPos = np.arange(len(labels))
    bars = ax.barh(
        xPos,
        percentages, 
        alpha=0.7, 
        color="teal",
        edgecolor=None
    )

    ax.plot(percentages, xPos, "r--", marker='o', linewidth=2, markersize=8)

    ax.set_yticks(xPos, labels, rotation=45)
    ax.set_ylabel(
        "{} Indeics Slops".format(TITLE.get(colName)),
        fontweight="bold"
    )
    ax.set_xlabel(
        "Percentage (%)",
        fontweight="bold"
    )

    ax.legend(["Change Curve", "Percentage"], loc="best")

    ax2.boxplot(
        RESULT.dropna(),
        widths=0.6,
        showmeans=True,
        patch_artist=True,
        boxprops = {"color": "gray", "facecolor":"teal"},
        flierprops = {"marker":'.', "markerfacecolor":"gray", "markeredgecolor": "none"},
        medianprops = {"color": "black"},
        meanprops = {"color": "gray"},
        whiskerprops = {"color": "gray"},
        capprops = {"color": "gray"}
    )

    ax2.set_xticks([])
    ax2.yaxis.tick_right()
    ax2.set_yticks(boxTicks)

    print(f"Total data points: {allCounts}")
    print(f"Negative values: {countNegative} ({countNegative / (countNegative + countPositive) * 100:.02f}%)")

    plt.tight_layout()
    plt.subplots_adjust(left=0.18)
    if path == "":
        plt.show()
    else:
        plt.savefig(os.path.join(path, "{}_slops.jpg".format(TITLE.get(colName))), dpi=300)
    plt.close()

    return

if __name__ == "__main__":
    import os
    RESULT = pd.read_csv(os.path.join(".", "China_Acc_Results", "Result", "city_efficiency.csv"), encoding="utf-8")
    RESULT = RESULT[RESULT["name"] != u"境界线"]

    # Clean Gini Nan
    for y in range(2015, 2026):
        RESULT.loc[RESULT["Relative_Accessibility_{}".format(y)].isna(), "M2SFCA_Gini_{}".format(y)] = np.nan
    
    # improvement(RESULT.copy(), "Relative_Accessibility", r".\\paper\\figure")
    # improvement(RESULT.copy(), "M2SFCA_Gini", r".\\paper\\figure")
    improvement(RESULT.copy(), "M2SFCA_Gini")