import os, libpysal
import pandas as pd
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
from rasterio.mask import mask
from shapely.geometry import Polygon, shape
from rasterio.features import shapes
from esda.moran import Moran_Local

try:
    from .setting import calSlop, plotSet
except:
    from setting import calSlop, plotSet
class densityAnalysis:
    __slots__ = ["evcs", "citiesList", "boundarys", "rasterPath", "colName"]

    def __init__(
        self,
        RESULT: pd.DataFrame, colNames: tuple[str, float],
        evcs: gpd.GeoDataFrame,
        gdp: tuple[pd.DataFrame, float],
        bounarys: gpd.GeoDataFrame,
        rasterPath: str
    ) -> None:
        gdpDf, gdpThreshold = gdp
        colName, colThreshold = colNames
        self.colName = "{}_2025-2015".format(colName)
        result = calSlop(RESULT, colName)
        result = result.loc[result[self.colName] <= 0].set_index("name").join(gdpDf)
        result = result.loc[result[u"GDP(亿元)"] >= gdpThreshold]
        self.evcs = evcs.join(result, on="cityname").dropna(subset=self.colName)
        self.citiesList = result.index.tolist()
        self.boundarys = bounarys
        self.rasterPath = rasterPath
        plotSet()

        return
    
    def cal(self) -> None:
        for city in self.citiesList:
            boundary = self.boundarys.loc[self.boundarys["name"] == city]
            for y in [2015, 2018, 2020]:
                raster = os.path.join(self.rasterPath, "chn_ppp_{}_1km_Aggregated_UNadj.tif".format(y if y < 2020 else 2020))
                # Transfore geo
                with rio.open(raster) as src:
                    rasterCrs = src.crs
                    if self.evcs.crs != rasterCrs:
                        self.evcs = self.evcs.to_crs(rasterCrs)
                    if boundary.crs != rasterCrs:
                        boundary = boundary.to_crs(rasterCrs)
                
                boundaryShape = boundary.geometry
                cityEvcs = self.evcs.loc[(self.evcs["cityname"] == city) & (self.evcs["year"] == y), ["geometry"]]
                analysisGrid = self.rasterToGrid(raster, boundaryShape)
                analysisGridJoin = analysisGrid.sjoin(
                    cityEvcs[['geometry']],
                    how='left', 
                    predicate='contains'
                ).dropna(subset="index_right").groupby(level=0).size().reset_index(name="EVCSNum").set_index("index")
                analysisGrid = analysisGrid.join(analysisGridJoin)
                analysisGrid = analysisGrid.loc[~(analysisGrid["raster_value"] == -99999)]
                analysisGrid["EVCSNum"] = analysisGrid["EVCSNum"].fillna(0)

                weights = libpysal.weights.Queen.from_dataframe(analysisGrid, use_index=False)

                # Build Local Moran'I clusting
                y1 = analysisGrid["EVCSNum"].values
                y2 = analysisGrid["raster_value"].values
                localMoranEVCS = (Moran_Local(y1, weights, permutations=999), "EVCS")
                localMoranPop = (Moran_Local(y2, weights, permutations=999), "Pop")
                for localMorans in [localMoranEVCS, localMoranPop]:
                    localMoran, desc = localMorans
                    p = "p{}".format(desc)
                    cluster = "cluster{}".format(desc)
                    clusterType = "clusterType{}".format(desc)
                    analysisGrid["lisa{}".format(desc)] = localMoran.Is
                    analysisGrid[p] = localMoran.p_sim
                    analysisGrid[cluster] = localMoran.q
                    
                    analysisGrid[clusterType] = "Non Significant"
                    analysisGrid.loc[(analysisGrid[p] < 0.05) & (analysisGrid[cluster] == 1), clusterType] = "HH"
                    analysisGrid.loc[(analysisGrid[p] < 0.05) & (analysisGrid[cluster] == 3), clusterType] = "LL"
                    analysisGrid.loc[(analysisGrid[p] < 0.05) & (analysisGrid[cluster] == 2), clusterType] = "LH"
                    analysisGrid.loc[(analysisGrid[p] < 0.05) & (analysisGrid[cluster] == 4), clusterType] = "HL"

                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                ax[0].axis("off")
                ax[1].axis("off")
                colors = {
                    "HH": "red",
                    "LL": "blue",
                    "LH": "lightblue",
                    "HL": "pink",
                    "Non Significant": "lightgrey"
                }

                # Left side: Population
                analysisGrid.plot(ax=ax[0], color=analysisGrid["clusterTypePop"].map(colors),
                                categorical=True, legend=True)
                cityEvcs.plot(ax=ax[0], color='black', markersize=5, alpha=0.5)
                ax[0].set_title("LISA Cluster of Population")

                # Right side: EVCS
                analysisGrid.plot(ax=ax[1], color=analysisGrid["clusterTypeEVCS"].map(colors),
                                categorical=True, legend=True)
                cityEvcs.plot(ax=ax[1], color='black', markersize=5, alpha=0.5)
                ax[1].set_title("LISA Cluster of Charging Stations")

                plt.tight_layout()
                print("{}:{}".format(city, y))
                # plt.savefig("charging_station_hotspots.png", dpi=300)
                plt.show()
                plt.close()

        return
    
    @staticmethod
    def rasterToGrid(rasterPath: str, boundary: Polygon, band: int=1) -> gpd.GeoDataFrame:
        gridPolygons = []
        gridValues = []
        with rio.open(rasterPath) as src:
            clipped, transform = mask(
                dataset=src,
                shapes=boundary,
                crop=True,
                nodata=src.nodata
            )
            
            data = clipped[band - 1]
        
            for geom, val in shapes(data, transform=transform):
                gridPolygons.append(shape(geom))
                gridValues.append(val)
        
        return gpd.GeoDataFrame(
            {
                'geometry': gridPolygons,
                'raster_value': gridValues
            }, crs=src.crs)


if __name__ == "__main__":
    import numpy as np
    RESULT = pd.read_csv(os.path.join("China_Acc_Results", "Result", "city_efficiency.csv"), encoding="utf-8")
    RESULT = RESULT[RESULT["name"] != u"境界线"]
    # Clean Gini Nan
    for y in range(2015, 2026):
        RESULT.loc[RESULT["Relative_Accessibility_{}".format(y)].isna(), "M2SFCA_Gini_{}".format(y)] = np.nan

    evcs = gpd.read_file("ArcGIS\\ChinaDynam.gdb", layer="Merge_Amap_15_25")
    boundarys = gpd.read_file("ArcGIS\\ChinaDynam.gdb", layer="CNMap_City")
    gdp = pd.read_excel("China_Acc_Results/Result/city_gdponly.xlsx").set_index(u"区县")

    densityAnalysis(RESULT, ("Relative_Accessibility", -0.02), evcs, (gdp, 15000), boundarys, r"C:\\0_PolyU\\cn_2015-2025_pop\\").cal()
    