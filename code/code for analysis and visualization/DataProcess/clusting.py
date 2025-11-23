import pandas as pd
import geopandas as gpd
import numpy as np
import libpysal as ps  # Spatial weight matrix library
import matplotlib.pyplot as plt
from esda.moran import Moran
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from .setting import INDEX, OTHER_COLUMNS, NULL_CITIES, plotSet
except:
    from setting import INDEX, OTHER_COLUMNS, NULL_CITIES, plotSet

# Clusting analysis
class clusting:
    __slots__ = ["RESULT", "BASE_MAP", "features"]
    accessLabel = []
    giniLabel = []

    def __init__(self, RESULT: pd.DataFrame, BASE_MAP: gpd.GeoDataFrame) -> None:
        self.RESULT = RESULT.set_index(INDEX).drop(columns=OTHER_COLUMNS)
        self.RESULT = self.RESULT[~self.RESULT.index.isin(NULL_CITIES)]
        self.BASE_MAP = BASE_MAP.set_index(INDEX)
        self.features = {}
        plotSet()

        for y in range(2015, 2026):
            self.accessLabel.append("Relative_Accessibility_{}".format(y))
            self.giniLabel.append("M2SFCA_Gini_{}".format(y))
    
    def __getLabel(self, colName: str) -> list:
        if colName == "Relative_Accessibility":
            return self.accessLabel
        else:
            return self.giniLabel
    
    # Function used to evulates the best k
    @staticmethod
    def __findOptimalK(features: dict[str, list], maxK: int=10):
        inertias = []
        silhouettes = []
        data = [x for x in features.values() if x is not None]
        
        for k in range(2, maxK + 1):
            km = KMeans(n_clusters=k).fit(data)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(data, km.labels_)) # type: ignore
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(range(2, maxK + 1), inertias, 'bo-', label='Inertia')
        ax2.plot(range(2, maxK + 1), silhouettes, 'rs-', label='Silhouette') # 轮廓系数	[-1,1] 越大越好
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('Inertia', color='b')
        ax2.set_ylabel('Silhouette Score', color='r')
        plt.show()

    def clusting(self, colName: str, CLUSTER_NUM: int) -> dict[int, list[str]]:
        data: dict[str, list] | None = self.features.get(colName, None)
        if data is None:
            raise RuntimeError("Have not initialize data, run showK() first")
        
        features = [x for x in data.values() if x is not None]
        featuresIndex = [x for x in data.keys() if data[x] is not None]
        X = StandardScaler().fit_transform(features) # type: ignore
        kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=0).fit(X)
        results = pd.DataFrame(kmeans.labels_, index=featuresIndex, columns=["cluster"])
        df = self.RESULT[self.__getLabel(colName)]
        df = df.join(results)

        clusterResult = {}
        for clusterId in range(CLUSTER_NUM):
            clusterData = df[df["cluster"] == clusterId]
            cities = clusterData.index.to_list()
            print(f"Cluster {clusterId}: {cities} ({len(cities)})")
            clusterResult[clusterId] = cities
            plt.plot(np.nanmedian(clusterData.drop(columns="cluster"), axis=0), label=f'Cluster {clusterId}')
        
        plt.legend()
        plt.show()

        return clusterResult

    # Moran I Index
    def calMoranI(self, colName: str, IThres: float) -> None:
        correlation = False
        for i in self.__getLabel(colName):
            subdf = self.RESULT.dropna(subset=i)
            gdf = subdf.join(self.BASE_MAP)
            w = ps.weights.Queen.from_dataframe(gdf, use_index=True, silence_warnings=True)
            moran = Moran(subdf[[i]], w)
            if moran.I >= IThres and moran.p_sim <= 0.05:
                print(f"{i} has spatial correaltion: with Moran's I: {moran.I}, p-value: {moran.p_sim}.")
        if not correlation:
            print(f"{colName} do not have sptial correaltion.")

        return
    
    def showK(self, colName: str, show: bool = True):
        self.features[colName] = {}
        df = self.RESULT.loc[:, self.__getLabel(colName)]
        for index, citySeries in zip(df.index, df.to_numpy()):
            citySeries = citySeries[~np.isnan(citySeries)] # drop NaN
            if len(citySeries) < 2:
                continue # Skip city only have 2025 record
            self.features[colName][index] = [
                np.polyfit(range(len(citySeries)), citySeries, 1)[0], # binomial fitting slop
                np.std(citySeries), # standard deviation
            ]
        if show:
            self.__findOptimalK(self.features[colName])

if __name__ == "__main__":
    import os
    BASE_MAP = gpd.read_file("ArcGIS\\ChinaDynam.gdb", layer="CNMap_City", encoding="utf-8")
    RESULT = pd.read_csv(os.path.join("China_Acc_Results", "Result", "city_optAcc.csv"), encoding="utf-8")
    RESULT = RESULT[RESULT["name"] != u"境界线"]

    # Clean Gini Nan
    for y in range(2015, 2026):
        RESULT.loc[RESULT["Relative_Accessibility_{}".format(y)].isna(), "M2SFCA_Gini_{}".format(y)] = np.nan

    a = clusting(RESULT.copy(), BASE_MAP.copy())
    a.showK("M2SFCA_Gini", False)
    a.showK("Relative_Accessibility",False)
    a.clusting("Relative_Accessibility", 3)
    a.clusting("M2SFCA_Gini", 2)