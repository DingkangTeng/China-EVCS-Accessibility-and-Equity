import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

try:
    from .setting import plotSet, INDEX, FIG_SIZE, BAR_COLORS
except:
    from setting import plotSet, INDEX, FIG_SIZE, BAR_COLORS
    
# Draw Relative Accessibility Efficiency in 2015, 2020, 2025
def draw152535(RESULT: pd.DataFrame, BASE_MAP: gpd.GeoDataFrame) -> None:
    plotSet()
    for y in [2015, 2020, 2025]:
        resultCol = "Relative_Accessibility_{}".format(y)
        yearData = RESULT[[INDEX, resultCol]]
        yearData.loc[:, INDEX] = yearData[INDEX].astype(str)
        yearData.set_index(INDEX, inplace=True)
        result = BASE_MAP.join(yearData, on=INDEX)
        fig, ax = plt.subplots(figsize=(10, 10))
        result.plot(
            resultCol,
            ax = ax,
            missing_kwds={
                "color": "lightgrey",
                "edgecolor": "black",
                "hatch": "////",
                "label": "No Value"
            },
            cmap="Oranges",
            scheme="BoxPlot",
            k=5,
            legend=True,
            legend_kwds={
                'loc': 'lower right',
                'title': 'Relative Accessibiliey',
                'shadow': True
            }
        )
        ax.axis('off')
        plt.tight_layout(pad=4.5)
        plt.show()
        plt.close()

    return

def drawClusting(RESULT: pd.DataFrame, BASE_MAP: gpd.GeoDataFrame, reativeClustering:dict = {}, giniClustering: dict = {}, savePath: str = '') -> None:
    tag1 = {0: "Stable efficiency", 1: "Decreased efficiency", 2:"Increased efficiency"}
    tag2 = {0: "Slowly increased equlity", 1: "Quickly increased equlity"}
    result = []
    for i in range(3):
        for j in range(2):
            n = set(reativeClustering[i]) & set(giniClustering[j])
            ef = tag1[i]
            eq = tag2[j]
            c = f"{ef} & {eq}"
            print(f"{c} ({len(n)}): {n}")
            for x in n:
                result.append([x, ef, eq, c])
    result = pd.DataFrame(result, columns=["name", "clusting_efficiency", "clusting_equity", "clusting"])
    result = RESULT.set_index("name").join(result.set_index("name")).drop(columns='gb')
    if savePath != '':
        result.to_csv(os.path.join(savePath, "city_with_clusting.csv"), encoding='utf-8')
    result = BASE_MAP.join(result, on=INDEX)
    titles = {"clusting_efficiency": "Efficiency", "clusting_equity": "Equity", "clusting": "Efficiency and Equity"}
    for i in ["clusting_efficiency", "clusting_equity", "clusting"]:
        fig, ax = plt.subplots(figsize=(10, 10))
        result.plot(
            i,
            ax = ax,
            missing_kwds={
                "color": "lightgrey",
                "edgecolor": "black",
                "hatch": "////",
                "label": "No Value"
            },
            legend=True,
            legend_kwds={
                "loc": "lower right",
                "title": titles[i],
                "shadow": True
            }
        )
        ax.axis('off')
        plt.tight_layout(pad=4.5)
        plt.show()
        plt.close()

    return

def stackplot(df: pd.DataFrame, name: str, subname: str, path: str = "") -> None:
    plotSet()
    LEGEND = {
        "Number of chargers (millions)": "Chargers Number",
        "Growth rate of chargers (%)": "Growth rate"
    }
    ROW = {
        "Number of chargers (millions)": "sum",
        "Growth rate of chargers (%)": "ratio"
    }
    COLOR = {
        "Number of chargers (millions)": BAR_COLORS[0],
        "Growth rate of chargers (%)": BAR_COLORS[1]
    }
    fig = plt.figure(figsize=FIG_SIZE.D)
    ax = plt.subplot()
    years = list(range(2016, 2026))
    df = df.drop(columns=["region"])

    df.loc[df["name"] == ROW.get(name, name), years].T.plot(
        ax=ax,
        color=COLOR.get(name, "teal"),
        legend=False
    )

    ax.set_ylabel(name)

    if subname != "":
        ax2 = ax.twinx()
        df.loc[df["name"] == ROW.get(subname, subname), years].T.plot(
            ax=ax2,
            color=COLOR.get(subname, "teal"),
            legend=False
        )

        lines1, _ = ax.get_legend_handles_labels()
        lines2, _ = ax2.get_legend_handles_labels()

        ax.set_ylim(0, 5)
        ax2.set_yticks(np.linspace(0, 100, 6))
        ax2.set_ylim(0, 100)
        ax2.grid(False)

        ax2.set_ylabel(subname)
        ax.legend(lines1 + lines2, [LEGEND.get(name, name), LEGEND.get(subname, subname)], loc="upper left")
        
    else:
        ax.legend(LEGEND.get(name, name))
        
    ax.set_xlabel("Year")
    ax.set_xticks(range(0,10), years) # type: ignore

    plt.tight_layout()
    if path == "":
        plt.show()
    else:
        plt.savefig(os.path.join(path, "{}.jpg".format(name)), dpi=300)
    
    plt.close()

    return

if __name__ == "__main__":
    # BASE_MAP = gpd.read_file("ArcGIS\\ChinaDynam.gdb", layer="CNMap_City", encoding="utf-8")
    # RESULT = pd.read_csv(os.path.join("China_Acc_Results", "Result", "city_efficiency.csv"), encoding="utf-8")
    # RESULT = RESULT[RESULT["name"] != u"境界线"]
    # reativeClustering = {
    #     0: ['鄂州市', '舟山市', '朝阳市', '广安市', '濮阳市', '宣城市', '六盘水市', '深圳市', '三明市', '景德镇市', '楚雄彝族自治州', '大连市', '攀枝花市', '贺州市', '西安市', '淮北市', '邵阳市', '石家庄市', '汕尾市', '荆门市', '衢州市', '迪庆藏族自治州', '葫芦岛市', '济南市', '贵阳市', '珠海市', '长春市', '延边朝鲜族自治州', '萍乡市', '庆阳市', '泉州市', '自贡市', '临沂市', '百色市', '鞍山市', '唐山市', '岳阳市', '河源市', '孝感市', '金华市', '兴安盟', '雅安市', '亳州市', '商洛市', '盐城市', '汕头市', '安顺市', '黔南布依族苗族自治州', '厦门市', '阿克苏地区', '红河哈尼族彝族自治州', '太原市', '常德市', '牡丹江市', '淮南市', '德阳市', '威海市', '绍兴市', '池州市', '淮安市', '黔东南苗族侗族自治州', '莆田市', '南昌市', '佛山市', '昆明市', '张家界市', '泸州市', '沈阳市', '邯郸市', '马鞍山市', '三亚市', '十堰市', '呼和浩特市', '德宏傣族景颇族自治州', '枣庄市', '安阳市', '阳泉市', '保亭黎族苗族自治县', '鹰潭市', '乐东黎族自治县', '龙岩市', '天门市', '潜江市', '邢台市', '仙桃市', '绥化市', '丹东市', '滨州市', '镇江市', '昌江黎族自治县', '周口市', '白沙黎族自治县', '临高县', '澄迈县', '长沙市', '屯昌县', '定安县', '海口市', '乐山市', '凉山彝族自治州', '桂林市', '东营市', '鹤壁市', '锡林郭勒盟', '琼中黎族苗族自治县', '长治市', '通化市', '上海市', '鸡西市', '宁德市', '保定市', '菏泽市', '崇左市', '黑河市', '驻马店市', '哈密市', '锦州市', '肇庆市', '恩施土家族苗族自治州', '宜昌市', '平凉市', '青岛市', '新乡市', '乌海市', '黔西南布依族苗族自治州', '广州市', '吉林市', '九江市', '来宾市', '张家口市', '德州市', '抚顺市', '安庆市', '新星市', '湘潭市', '惠州市', '襄阳市', '南充市', '酒泉市', '淄博市', '焦作市', '石河子市', '赤峰市', '韶关市', '四平市', '新余市', '南平市', '哈尔滨市', '成都市', '河池市', '中卫市', '本溪市', '承德市', '聊城市', '衡阳市', '杭州市', '武威市', '临夏回族自治州', '黄山市', '徐州市', '揭阳市', '那曲市', '郑州市', '通辽市', '贵港市', '安康市', '昌吉回族自治州', '晋中市', '大庆市', '普洱市', '白城市', '荆州市', '沧州市', '三门峡市', '张掖市', '滁州市', '泰安市', '无锡市', '大兴安岭地区', '鄂尔多斯市', '阿拉善盟', '山南市', '开封市', '重庆市', '钦州市', '伊春市', '运城市', '丽江市', '黄冈市', '南阳市', '乌鲁木齐市', '和田地区', '海南藏族自治州', '海东市', '武汉市', '儋州市', '甘孜藏族自治州', '内江市', '烟台市', '南京市', '白银市', '郴州市', '北京市', '洛阳市', '丽水市', '忻州市', '中山市', '咸宁市', '宿迁市', '济源市', '营口市', '商丘市', '潍坊市', '巴彦淖尔市', '平顶山市', '宜春市', '台州市', '东方市', '万宁市', '文昌市', '临沧市', '松原市', '随州市', '资阳市', '银川市', '泰州市', '神农架林区', '阜新市', '琼海市', '五指山市', '信阳市', '湖州市', '抚州市', '清远市', '天津市', '嘉峪关市', '阿里地区', '连云港市', '渭南市', '玉溪市', '江门市', '益阳市', '广元市', '芜湖市', '塔城地区', '嘉兴市', '西宁市', '上饶市', '东莞市', '南通市', '辽阳市', '巴中市', '六安市', '金昌市', '廊坊市', '拉萨市', '喀什地区', '福州市', '咸阳市', '文山壮族苗族自治州', '湛江市', '曲靖市', '蚌埠市', '绵阳市', '温州市', '苏州市', '阜阳市', '林芝市', '许昌市', '宝鸡市', '昭通市', '大理白族自治州', '定西市', '茂名市', '宁波市', '常州市', '宿州市', '昌都市', '漯河市', '陇南市', '保山市', '遂宁市', '合肥市'],
    #     1: ['吕梁市', '盘锦市', '达州市', '怀化市', '辽源市', '齐齐哈尔市', '吐鲁番市', '赣州市', '巴音郭楞蒙古自治州', '株洲市', '可克达拉市', '阿拉尔市', '铁门关市', '五家渠市', '湘西土家族苗族自治州', '济宁市', '石嘴山市', '云浮市', '吴忠市', '呼伦贝尔市', '防城港市', '汉中市', '鹤岗市', '三沙市', '黄石市', '克孜勒苏柯尔克孜自治州', '日喀则市', '阿勒泰地区', '佳木斯市', '兰州市', '潮州市', '海北藏族自治州', '铜川市'],
    #     2: ['梅州市', '宜宾市', '铜陵市', '甘南藏族自治州', '秦皇岛市', '阳江市', '铁岭市', '遵义市', '大同市', '日照市', '梧州市', '陵水黎族自治县', '博尔塔拉蒙古自治州', '娄底市', '包头市', '扬州市', '怒江傈僳族自治州', '柳州市', '眉山市', '晋城市', '漳州市', '固原市', '南宁市', '朔州市', '海西蒙古族藏族自治州', '榆林市', '吉安市', '果洛藏族自治州', '阿坝藏族羌族自治州', '天水市', '永州市', '北海市', '延安市', '双鸭山市', '黄南藏族自治州', '乌兰察布市', '铜仁市', '毕节市', '衡水市', '伊犁哈萨克自治州', '临汾市', '玉林市', '西双版纳傣族自治州']
    # }
    # giniClustering = {
    #     0: ['梅州市', '鄂州市', '吕梁市', '舟山市', '朝阳市', '广安市', '濮阳市', '宣城市', '六盘水市', '三明市', '景德镇市', '楚雄彝族自治州', '大连市', '贺州市', '邵阳市', '荆门市', '衢州市', '迪庆藏族自治州', '宜宾市', '葫芦岛市', '长春市', '延边朝鲜族自治州', '萍乡市', '庆阳市', '自贡市', '临沂市', '百色市', '鞍山市', '铜陵市', '唐山市', '岳阳市', '河源市', '孝感市', '兴安盟', '盘锦市', '雅安市', '亳州市', '商洛市', '盐城市', '安顺市', '黔南布依族苗族自治州', '阿克苏地区', '红河哈尼族彝族自治州', '甘南藏族自治州', '常德市', '牡丹江市', '淮南市', '德阳市', '威海市', '秦皇岛市', '阳江市', '铁岭市', '达州市', '池州市', '淮安市', '遵义市', '黔东南苗族侗族自治州', '莆田市', '大同市', '张家界市', '泸州市', '邯郸市', '马鞍山市', '日照市', '德宏傣族景颇族自治州', '梧州市', '枣庄市', '安阳市', '怀化市', '阳泉市', '保亭黎族苗族自治县', '鹰潭市', '博尔塔拉蒙古自治州', '乐东黎族自治县', '辽源市', '齐齐哈尔市', '龙岩市', '天门市', '潜江市', '邢台市', '仙桃市', '绥化市', '丹东市', '滨州市', '昌江黎族自治县', '周口市', '白沙黎族自治县', '临高县', '吐鲁番市', '澄迈县', '屯昌县', '定安县', '乐山市', '凉山彝族自治州', '桂林市', '东营市', '鹤壁市', '娄底市', '长治市', '赣州市', '通化市', '巴音郭楞蒙古自治州', '鸡西市', '宁德市', '保定市', '菏泽市', '崇左市', '黑河市', '扬州市', '驻马店市', '株洲市', '锦州市', '恩施土家族苗族自治州', '怒江傈僳族自治州', '平凉市', '眉山市', '新乡市', '乌海市', '黔西南布依族苗族自治州', '吉林市', '九江市', '漳州市', '来宾市', '张家口市', '德州市', '固原市', '抚顺市', '安庆市', '新星市', '湘潭市', '可克达拉市', '襄阳市', '南充市', '酒泉市', '焦作市', '石河子市', '赤峰市', '铁门关市', '朔州市', '四平市', '新余市', '南平市', '哈尔滨市', '河池市', '中卫市', '承德市', '聊城市', '衡阳市', '湘西土家族苗族自治州', '武威市', '临夏回族自治州', '黄山市', '徐州市', '济宁市', '那曲市', '通辽市', '贵港市', '安康市', '晋中市', '大庆市', '普洱市', '白城市', '荆州市', '沧州市', '三门峡市', '海西蒙古族藏族自治州', '云浮市', '张掖市', '滁州市', '泰安市', '大兴安岭地区', '山南市', '开封市', '重庆市', '钦州市', '榆林市', '伊春市', '运城市', '丽江市', '黄冈市', '吴忠市', '南阳市', '和田地区', '海南藏族自治州', '海东市', '甘孜藏族自治州', '内江市', '烟台市', '白银市', '郴州市', '呼伦贝尔市', '洛阳市', '汉中市', '吉安市', '丽水市', '忻州市', '鹤岗市', '咸宁市', '宿迁市', '营口市', '商丘市', '果洛藏族自治州', '三沙市', '黄石市', '阿坝藏族羌族自治州', '潍坊市', '天水市', '永州市', '巴彦淖尔市', '北海市', '平顶山市', '宜春市', '台州市', '东方市', '延安市', '万宁市', '文昌市', '双鸭山市', '临沧市', '松原市', '随州市', '资阳市', '泰州市', '神农架林区', '阜新市', '信阳市', '黄南藏族自治州', '抚州市', '清远市', '嘉峪关市', '乌兰察布市', '阿里地区', '连云港市', '克孜勒苏柯尔克孜自治州', '日喀则市', '渭南市', '铜仁市', '玉溪市', '益阳市', '广元市', '塔城地区', '西宁市', '上饶市', '南通市', '辽阳市', '巴中市', '六安市', '金昌市', '廊坊市', '拉萨市', '喀什地区', '阿勒泰地区', '毕节市', '咸阳市', '文山壮族苗族自治州', '湛江市', '曲靖市', '蚌埠市', '绵阳市', '阜阳市', '林芝市', '衡水市', '许昌市', '佳木斯市', '宝鸡市', '昭通市', '大理白族自治州', '定西市', '茂名市', '伊犁哈萨克自治州', '临汾市', '宿州市', '昌都市', '玉林市', '海北藏族自治州', '漯河市', '西双版纳傣族自治州', '陇南市', '保山市', '遂宁市', '铜川市'],
    #     1: ['深圳市', '攀枝花市', '西安市', '淮北市', '石家庄市', '汕尾市', '济南市', '贵阳市', '珠海市', '泉州市', '金华市', '汕头市', '厦门市', '太原市', '绍兴市', '南昌市', '佛山市', '昆明市', '沈阳市', '三亚市', '十堰市', '呼和浩特市', '陵水黎族自治县', '镇江市', '长沙市', '海口市', '包头市', '锡林郭勒盟', '琼中黎族苗族自治县', '上海市', '哈密市', '肇庆市', '宜昌市', '柳州市', '青岛市', '晋城市', '广州市', '惠州市', '南宁市', '淄博市', '阿拉尔市', '五家渠市', '韶关市', '成都市', '本溪市', '杭州市', '揭阳市', '郑州市', '昌吉回族自治州', '石嘴山市', '无锡市', '鄂尔多斯市', '阿拉善盟', '乌鲁木齐市', '武汉市', '儋州市', '南京市', '防城港市', '北京市', '中山市', '济源市', '银川市', '琼海市', '五指山市', '湖州市', '天津市', '江门市', '芜湖市', '嘉兴市', '东莞市', '福州市', '温州市', '苏州市', '宁波市', '兰州市', '常州市', '潮州市', '合肥市']
    # }
    # drawClusting(RESULT, BASE_MAP, reativeClustering, giniClustering, ".\\China_Acc_Results\\Result")
    # draw152535(RESULT, BASE_MAP)

    provinceLevelData = pd.read_excel(os.path.join("China_Acc_Results", "Result", "provinceLevel", "China_EVCS.xlsx"))
    stackplot(provinceLevelData, "Number of chargers (millions)", "Growth rate of chargers (%)", os.path.join(".", "paper", "figure", "fig1"))