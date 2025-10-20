import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Constant
INDEX = "name"
COLUMNS = ["2SFCA_Accessibility", "M2SFCA_Accessibility", "Relative_Accessibility", "2SFCA_Gini", "M2SFCA_Gini"]
OTHER_COLUMNS = ["gb"]
NULL_CITIES = {u"白杨市", u"莲花山风景林自然保护区", u"昆玉市", u"图木舒克市"}
NO_GDP = {u"中农发山丹马场", u"三沙市"}
#昆玉市 and 图木舒克市 only have two years data and will influence the result of clusting
ECO_COL = [u"GDP(亿元)", u"人均GDP(元)", u"第一产业占比(%)", u"第二产业占比(%)", u"第三产业占比(%)"]
TITLE = {
    "M2SFCA_Gini": "Equity",
    "Relative_Accessibility": "Opt-Acc",
    "M2SFCA_Accessibility": "Accessibility",
    u"人均GDP(元)": "GDP",
    u"保有量": "EVs",
    "tour": "Tourist City"
}
# https://www.gov.cn/zhengce/zhengceku/2020-12/30/content_5575120.htm
TOUR_CITY_LIST = [
    u"北京市", u"天津市", u"石家庄市", u"唐山市", u"太原市", u"呼和浩特市", u"沈阳市", u"盘锦市", u"吉林市", u"通化市", u"哈尔滨市", u"牡丹江市",
    u"上海市", u"常州市", u"宁波市", u"温州市", u"合肥市", u"芜湖市", u"铜陵市", u"福州市", u"厦门市", u"三明市" ,"南昌市", u"新余市",
    u"烟台市", u"淄博市", u"郑州市", u"开封市", u"宜昌市", u"株洲市", u"郴州市", u"广州市", u"深圳市", u"惠州市", u"佛山市", u"南宁市", u"桂林市",
    u"海口市", u"三亚市", u"重庆市", u"泸州市", u"南充市", u"贵阳市", u"遵义市", u"丽江市", u"大理白族自治州", u"西安市", u"兰州市", u"张掖市",
    u"西宁市", u"黄南藏族自治州", u"银川市", u"乌鲁木齐市", u"石河子市",
    u"廊坊市", u"鄂尔多斯市", u"长春市", u"南京市", u"苏州市", u"杭州市", u"济南市", u"青岛市", u"洛阳市", u"武汉市", u"长沙市", u"成都市", u"昆明市"
]

# Fig setting
LABEL_SIZE = 24
TICK_SIZE = int(LABEL_SIZE * 0.9)
@dataclass
class __FIG_SIZE:
    D: tuple[int, int] = (10,9)     # Default
    R: tuple[int, int] = (9, 10)    # Rotate
    SM: tuple[float, int] = (10 * 2/3,6)     # Smaller
    SD: tuple[float, int] = (10 * 2/3,9)     # Smaller but default high
    SHH: tuple[float, float] = (10 * 2/3, 13.5)     # Smaller but half higher
    H: tuple[int, int] = (10, 18)   # Higher
    HH: tuple[int, float] = (10, 13.5)   # Half Higher
    W: tuple[int, int] = (20, 9)    # Wider
    H3W: tuple[int, int] = (20, 27)  # Higher * 3 and wider
    N: tuple[int, int] = (5, 9)     # Narrower
    S: tuple[int, int] = (10, 3)    # Shorter
FIG_SIZE = __FIG_SIZE()
BAR_COLORS = [
    ["#436C85", "#B73F42", "#DE9960", "#82B29B", "#EEE6CB"],
    # ["#E76727", "#A8C3D1", "#A57E74", "#E9B693", "#EED7C6"],
    ["#DE476A", "#76AEA6", "#D79E8F", "#E5D2C4", "#F0E0D3"],
    # ["#C22525", "#3F3A39", "#6F5E56", "#C3AB8C", "#E1D6C7"],
    ["#7D5A8A", "#DE7294", "#90BBAA", "#E6D2C2", "#F0E0D3"],
    ["#165188", "#BFCF61", "#9FCBC3", "#BFD3BC", "#DDDAB4"],
    ["#DE476A", "#76AEA6", "#D79E8F", "#7D5A8A", "#DE7294", "#90BBAA", "#165188", "#BFCF61", "#F0E0D3", "#000000"],
]
# ["#BAD540", "#85A2D0", "#FFC339", "#000000"]

def plotSet(scal1: float | int = 1, scal2: float | int = 1) -> None:
    labelSize = int(LABEL_SIZE * scal1)
    tickSize = int(TICK_SIZE * scal2)
    plt.style.use("seaborn-v0_8-whitegrid")
    # Font setting
    plt.rcParams["font.sans-serif"] = "Sans Serif Collection"
    plt.rcParams["font.size"] = tickSize
    plt.rcParams["xtick.labelsize"] = tickSize
    plt.rcParams["ytick.labelsize"] = tickSize
    plt.rcParams["axes.labelsize"] = labelSize
    # legend format
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "lightgray"
    plt.rcParams['legend.frameon'] = True
    plt.rcParams["legend.framealpha"] = 1.0
    # axes setting
    plt.rcParams["axes.formatter.use_mathtext"] = False
    plt.rcParams["axes.labelweight"] = "bold"
    # save setting
    plt.rcParams["savefig.dpi"] = 300

    return

# Common function
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