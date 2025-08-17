import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from .setting import plotSet, calSlop
except:
    from setting import plotSet, calSlop

# Proportional distribution with different levels of improvement in accessibility from 2015 to 2025
def improvement(RESULT: pd.DataFrame, colName: str) -> None:
    plotSet()
    RESULT = calSlop(RESULT, colName)
    RESULT = RESULT.loc[:, ["{}_2025-2015".format(colName)]]
    negativeData = RESULT[RESULT["{}_2025-2015".format(colName)] < 0]["{}_2025-2015".format(colName)]
    positiveData = RESULT[RESULT["{}_2025-2015".format(colName)] >= 0]["{}_2025-2015".format(colName)]
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
    for d in [binsNegative, binsPositive]:
        for i in range(len(d) - 1):
            start = d[i]
            end = d[i + 1]
            labels.append(f'{start:.2f}-{end:.2f}')

    plt.figure(figsize=(12, 6))
    xPos = np.arange(len(labels))
    bars = plt.bar(xPos, percentages, 
                # width=0.8*(binEdges[1]-binEdges[0]),  # 宽度为区间宽度的80%
                alpha=0.7, 
                color='skyblue',
                edgecolor='black')

    # 绘制变化曲线（虚线）
    plt.plot(xPos, percentages, 'r--', marker='o', linewidth=2, markersize=8)

    # 设置x轴刻度为区间边界
    plt.xticks(xPos, labels, rotation=45, fontsize=10)
    plt.xlabel("Slops", fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # 添加图例和统计信息
    plt.legend(['Change Curve', 'Percentage'], loc='best')
    plt.figtext(0.15, 0.85, f'Total data points: {allCounts}', fontsize=10)
    plt.figtext(0.15, 0.82, f'Negative values: {countNegative} ({countNegative / (countNegative + countPositive) * 100:.02f}%)', fontsize=10)

    # 调整布局
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import os
    RESULT = pd.read_csv(os.path.join(".", "China_Acc_Results", "Result", "city_efficiency.csv"), encoding="utf-8")
    RESULT = RESULT[RESULT["name"] != u"境界线"]

    # Clean Gini Nan
    for y in range(2015, 2026):
        RESULT.loc[RESULT["Relative_Accessibility_{}".format(y)].isna(), "M2SFCA_Gini_{}".format(y)] = np.nan
    improvement(RESULT.copy(), "Relative_Accessibility")
    improvement(RESULT.copy(), "M2SFCA_Gini")