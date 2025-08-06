import matplotlib.pyplot as plt
import pandas as pd

INDEX = "name"
COLUMNS = ["2SFCA_Accessibility", "M2SFCA_Accessibility", "Relative_Accessibility", "2SFCA_Gini", "M2SFCA_Gini"]
OTHER_COLUMNS = ["gb"]
Y_LABEL_SIZE = 12 # 小四
NULL_CITIES = {u"白杨市", u"北屯市", u"莲花山风景林自然保护区", u"昆玉市", u"图木舒克市"}
#昆玉市 and 图木舒克市 only have two years data and will influence the result of clusting

def plotSet() -> None:
    plt.style.use('ggplot')
    plt.rcParams["font.sans-serif"] = "Times New Roman"

def calImprovement(RESULT: pd.DataFrame, colName: str):
    RESULT["{}_2025-2015".format(colName)] = RESULT["{}_2025".format(colName)] - RESULT["{}_2015".format(colName)]
    for y in range(2016, 2025):
        index = RESULT["{}_2025-2015".format(colName)].isna().index
        RESULT.loc[index, "{}_2025-2015".format(colName)] = RESULT.loc[index, "{}_2025".format(colName)] - RESULT.loc[index, "{}_{}".format(colName, y)]