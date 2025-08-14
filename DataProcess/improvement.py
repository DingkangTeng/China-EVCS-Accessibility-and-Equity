import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .setting import plotSet

# Proportional distribution with different levels of improvement in accessibility from 2015 to 2025
def improvement(RESULT: pd.DataFrame, colName: str) -> None:
    plotSet()
    improvement = []
    subset = RESULT[["{}_{}".format(colName, y) for y in range(2015, 2026)]]
    for index, citySeries in zip(subset.index, subset.to_numpy()):
        citySeries = citySeries[~np.isnan(citySeries)] # drop NaN
        if len(citySeries) < 2:
            improvement.append([index, None])
        else:
            improvement.append([index, np.polyfit(range(len(citySeries)), citySeries, 1)[0]])
    improvement = pd.DataFrame(improvement, columns=["index", "{}_2025-2015".format(colName)]).set_index("index")
    RESULT = subset.join(improvement)
    negativeData = RESULT[RESULT["{}_2025-2015".format(colName)] < 0]["{}_2025-2015".format(colName)]
    positiveData = RESULT[RESULT["{}_2025-2015".format(colName)] >= 0]["{}_2025-2015".format(colName)]

    binsPositive = np.linspace(0, positiveData.max(), 10) # 10 groups
    countsPositive, binEdges = np.histogram(positiveData, bins=binsPositive) 
    countNegative = len(negativeData)
    allCounts = np.concatenate([[countNegative], countsPositive])
    total = allCounts.sum()
    percentages = allCounts / total * 100

    labels = ["<0"]
    for i in range(len(binsPositive) - 1):
        start = binsPositive[i]
        end = binsPositive[i + 1]
        labels.append(f'{start:.2f}-{end:.2f}')

    plt.figure(figsize=(12, 6))
    xPos = np.arange(len(labels))
    bars = plt.bar(xPos, percentages, 
                width=0.8*(binEdges[1]-binEdges[0]),  # 宽度为区间宽度的80%
                alpha=0.7, 
                color='skyblue',
                edgecolor='black')

    # 绘制变化曲线（虚线）
    plt.plot(xPos, percentages, 'r--', marker='o', linewidth=2, markersize=8)

    # 添加标题和标签
    plt.title('Data Distribution by Equal-width Bins', fontsize=14)
    plt.xlabel('Value Range', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # 设置x轴刻度为区间边界
    plt.xticks(xPos, labels, rotation=45, fontsize=10)
    plt.title('Data Distribution with Negative Values as Separate Group', fontsize=14)
    plt.xlabel('Value Groups', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # 添加图例和统计信息
    plt.legend(['Change Curve', 'Percentage'], loc='best')
    plt.figtext(0.15, 0.85, f'Total data points: {allCounts}', fontsize=10)
    plt.figtext(0.15, 0.82, f'Negative values: {countNegative} ({percentages[0]:.02f}%)', fontsize=10)

    # 调整布局
    plt.tight_layout()
    plt.show()