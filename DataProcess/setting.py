import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

INDEX = "name"
COLUMNS = ["2SFCA_Accessibility", "M2SFCA_Accessibility", "Relative_Accessibility", "2SFCA_Gini", "M2SFCA_Gini"]
OTHER_COLUMNS = ["gb"]
FONT_SIZE = Y_LABEL_SIZE = 12 # 小四
NULL_CITIES = {u"白杨市", u"莲花山风景林自然保护区", u"昆玉市", u"图木舒克市"}
NO_GDP = {u"中农发山丹马场", u"三沙市"}
#昆玉市 and 图木舒克市 only have two years data and will influence the result of clusting
ECO_COL = [u"GDP(亿元)", u"人均GDP(元)", u"第一产业占比(%)", u"第二产业占比(%)", u"第三产业占比(%)"]

def plotSet() -> None:
    plt.style.use('ggplot')
    plt.rcParams["font.sans-serif"] = "Times New Roman"

    return

def calSlop(RESULT: pd.DataFrame, colName: str) -> pd.DataFrame:
    improvement = []
    subset = RESULT[["{}_{}".format(colName, y) for y in range(2015, 2026)]]
    for index, citySeries in zip(subset.index, subset.to_numpy()):
        citySeries = citySeries[~np.isnan(citySeries)] # drop NaN
        if len(citySeries) < 2:
            improvement.append([index, None])
        else:
            improvement.append([index, np.polyfit(range(len(citySeries)), citySeries, 1)[0]])
    improvement = pd.DataFrame(improvement, columns=["index", "{}_2025-2015".format(colName)]).set_index("index")

    return RESULT.join(improvement).dropna(subset="{}_2025-2015".format(colName))