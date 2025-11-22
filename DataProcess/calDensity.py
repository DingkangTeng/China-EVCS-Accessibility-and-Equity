import os, threading, fiona
import rasterio as rio
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from rasterio import mask
from concurrent.futures import as_completed, ThreadPoolExecutor

class calDensity:
    __slots__ = ["cities", "cityname", "evcs", "boundaies", "lock", "maxThread"]

    def __init__(self, boundaries: gpd.GeoDataFrame, evcs: gpd.GeoDataFrame, level: str = "cityname", maxThread: int = 1) -> None:
        self.cities = boundaries["name"].unique().tolist()
        # Exclude SAR
        for x in [u"台湾省", u"香港特别行政区", u"澳门特别行政区"]:
            self.cities.remove(x)
        self.evcs = evcs
        self.boundaies = boundaries
        self.lock = threading.Lock()
        self.maxThread = maxThread
        self.cityname = level

        return

    def calRasterDensity(self, rasterRoot: tuple[str, str], path: str = "") -> None:
        years = list(range(2015, 2026))
        result = pd.DataFrame(index=self.cities, columns=years)
        futures = []
        futuresDict = {}
        bar = tqdm(total=len(self.cities) * 22, desc="Processing rasters")
        with ThreadPoolExecutor(max_workers=self.maxThread) as excutor:
            for city in self.cities:
                for year in years:
                    if rasterRoot[1] == "population":
                        raster = "chn_pd_{}_1km_UNadj.tif".format(year if year < 2021 else 2020) # Population
                    elif rasterRoot[1] == "gdp":
                        raster = "gdp_{}.tif".format(2015 if year < 2020 else 2020)
                    else:
                        raise RuntimeError("Wrong raster defination.")
                    rasterPath = os.path.join(rasterRoot[0], raster)
                    # Projection
                    with rio.open(rasterPath) as src:
                        if self.boundaies.crs != src.crs:
                            self.boundaies = self.boundaies.to_crs(src.crs)
                        if self.evcs.crs != src.crs:
                            self.evcs = self.evcs.to_crs(src.crs)
                    # Process
                    evcsGeo = self.evcs.loc[(self.evcs[self.cityname] == city) & (self.evcs["year"] == year), ["geometry"]]
                    future = excutor.submit(self.calOneYearRaster, rasterPath, evcsGeo, city)
                    futures.append(future)
                    futuresDict[future] = (city, year)
                    bar.update(1)

            for future in as_completed(futures):
                city, year = futuresDict[future]
                try:
                    Rci = future.result()
                except Exception as e:
                    raise RuntimeError("{} in {}: {}".format(city, year, e))
                else:
                    bar.update(1)
                    with self.lock:
                        result.loc[city, year] = Rci

        if path != "":
            result.to_excel(os.path.join(path, "Raster_Density_{}.xlsx".format(rasterRoot[1])))

        bar.close()

        return
    
    def calOneYearRaster(self, rasterPath: str, evcsGeo: gpd.GeoDataFrame, city: str) -> np.floating | None:
        with rio.open(rasterPath) as src:
            boundary = self.boundaies.loc[self.boundaies["name"] == city].geometry
            if len(boundary) == 0 or len(evcsGeo) == 0:
                return None
            
            croppedArray: np.ndarray
            croppedArray, croppedTransform = mask.mask(
                src,
                boundary,
                crop=True,
                all_touched=False,
                nodata=src.nodata
            )
            
            croppedArray = croppedArray[~np.isnan(croppedArray) & (croppedArray != src.nodata)]
            if np.nansum(croppedArray) == 0:
                return None
            DPhigh = np.max(croppedArray)
            DPlow = np.min(croppedArray)
            diff = DPhigh - DPlow
            
            XCoords = evcsGeo.geometry.x.to_numpy().astype(np.float32)
            YCoords = evcsGeo.geometry.y.to_numpy().astype(np.float32)
            coords = np.column_stack((XCoords, YCoords))
            values = np.array([(val[0] - DPlow) / diff if val[0] != src.nodata else 0 for val in src.sample(coords)])

        return np.nanmean(values)
    
    def calRoadsDensity(self, roadsPath: str | tuple[str, str], buffer: float, path: str = "") -> None:
        years = list(range(2015, 2026))
        result = pd.DataFrame(index=self.cities, columns=years)
        Sr = np.pi * (buffer ** 2)
        if isinstance(roadsPath, str):
            src = fiona.open(roadsPath)
        else:
            src = fiona.open(roadsPath[0], layer=roadsPath[1])
        if src.crs is None:
            raise RuntimeError("Roads do not have correct crs.")
        if self.boundaies.crs != src.crs:
            self.boundaies = self.boundaies.to_crs(src.crs)
        src.close()

        futures = []
        futuresDict = {}
        bar = tqdm(total=len(self.cities) * 22, desc="Processing roads")
        with ThreadPoolExecutor(max_workers=self.maxThread) as excutor:
            for city in self.cities:
                for year in years:
                    evcsGeo = self.evcs[(self.evcs[self.cityname] == city) & (self.evcs["year"] == year)].geometry
                    future = excutor.submit(self.calOneYearRoad, roadsPath, evcsGeo, buffer, Sr, city)
                    futures.append(future)
                    futuresDict[future] = (city, year)
                    bar.update(1)
            
            for future in as_completed(futures):
                city, year = futuresDict[future]
                try:
                    Dci = future.result()
                except Exception as e:
                    raise RuntimeError("{} in {}: {}".format(city, year, e))
                else:
                    bar.update(1)
                    with self.lock:
                        result.loc[city, year] = Dci
        
        if path != "":
            result.to_excel(os.path.join(path, "Roads_Density.xlsx"))
        
        return
                
    def calOneYearRoad(
        self,
        roadsPath: str | tuple[str, str],
        evcsGeo: gpd.GeoDataFrame | gpd.GeoSeries,
        buffer: float, Sr: float, city: str
    ) -> float | None:
        boundary = self.boundaies.loc[self.boundaies["name"] == city].geometry
        evcsNum = len(evcsGeo)
        if len(boundary) == 0 or evcsNum == 0:
            return None
        
        if isinstance(roadsPath, str):
            roads = gpd.read_file(roadsPath, mask=boundary).to_crs(epsg=4479) # Projection unit into meter, ESPG4479: CGCS2000 Cartesian
        else:
            roads = gpd.read_file(roadsPath[0], layer=roadsPath[1], mask=boundary).to_crs(epsg=4479) # Projection unit into meter

        roads = roads.geometry
        Dc = roads.length.sum() / boundary.to_crs(epsg=4479).area.sum()
        bufferResult = evcsGeo.geometry.to_crs(epsg=4479).buffer(buffer) # Projection unit into meter

        Dcir = 0.0
        for bufferGeom in bufferResult:
            clipped = roads.intersection(bufferGeom)
            clipped = clipped[~clipped.is_empty]
            if len(clipped) != 0:
                Dcir += clipped.length.sum()
        
        return Dcir / evcsNum / Sr / Dc


# Debug
if __name__ == "__main__":
    evcs = gpd.read_file("ArcGIS\\ChinaDynam.gdb", layer="Merge_Amap_15_25")

    boundarys = gpd.read_file("ArcGIS\\ChinaDynam.gdb", layer="CNMap_City")
    a = calDensity(boundarys, evcs, "cityname", 32)
    a.calRasterDensity((r"C:\\0_PolyU\\cn_2015-2025_popDens", "population"), r"China_Acc_Results\\Result\\")
    # a.calRasterDensity((r"C:\\0_PolyU\\global_gdp", "gdp"), r"China_Acc_Results\\Result\\") # GDP
    # a.calRoadsDensity((r"C:\\0_PolyU\\roadsGraph\\CHN.gpkg", "edges"), 1000, r"China_Acc_Results\\Result\\")
    # a.calRoadsDensity(r"C:\\0_PolyU\\CHN_highwayonly.shp", 1000, r"China_Acc_Results\\Result\\")

    # boundarys = gpd.read_file("ArcGIS\\ChinaDynam.gdb", layer="CNMap_Province")
    # a = calDensity(boundarys, evcs, "pname", 32)
    # a.calRasterDensity((r"C:\\0_PolyU\\cn_2015-2025_pop", "population"), r"China_Acc_Results\\Result\\provinceLevel")
    # a.calRasterDensity((r"C:\\0_PolyU\\global_gdp", "gdp"), r"China_Acc_Results\\Result\\provinceLevel") # GDP
    # a.calRoadsDensity((r"C:\\0_PolyU\\roadsGraph\\CHN.gpkg", "edges"), 1000, r"China_Acc_Results\\Result\\provinceLevel")

    