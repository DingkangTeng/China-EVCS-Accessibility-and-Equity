import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    "Relative_Accessibility": "Efficiency",
    u"人均GDP(元)": "GDP",
    u"保有量": "Holding of EVs",
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
FIG_SIZE = (10,8)
FIG_SIZE_H = (8, 10)
BAR_COLORS = ["#BAD540", "#85A2D0", "#FFC339", ]

def plotSet() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    # Font setting
    plt.rcParams["font.sans-serif"] = "Times New Roman"
    plt.rcParams["font.size"] = TICK_SIZE
    plt.rcParams["xtick.labelsize"] = TICK_SIZE
    plt.rcParams["ytick.labelsize"] = TICK_SIZE
    plt.rcParams["axes.labelsize"] = LABEL_SIZE
    # legend format
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "lightgray"
    plt.rcParams['legend.frameon'] = True
    plt.rcParams["legend.framealpha"] = 1.0
    # axes setting
    plt.rcParams["axes.formatter.use_mathtext"] = False
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