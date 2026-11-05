import rasterio, os
import numpy as np
import pandas as pd
from pathlib import Path

YEARS = range(2015, 2021)
GROUP_FOLDERS = {
    "Male": "population_Male",
    "Female": "population_Female",
    "Young": "population_All_young",
    "Middle": "population_All_middle",
    "Old": "population_All_elderly",
}

# Find yearly raster
def __findYearTif(folderPath: Path, year: int) -> Path:
    matchedFiles = list(
        folderPath.glob(f"*_{year}_*.tif")
    )

    if len(matchedFiles) == 0:
        raise FileNotFoundError(
            f"No tif found for {year} in:\n"
            f"{folderPath}"
        )

    if len(matchedFiles) > 1:
        raise RuntimeError(
            f"Multiple tif files found for {year} "
            f"in {folderPath}:\n"
            + "\n".join(
                str(filePath)
                for filePath in matchedFiles
            )
        )

    return matchedFiles[0]

# Sum raster population
def __rasterPopulationSum(tifPath: Path) -> float:
    with (
        rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"),
        rasterio.open(tifPath) as src
    ):
        rasterData = src.read(
            1,
            masked=True
        )

    # Mask unexpected negative values
    rasterData = np.ma.masked_less(
        rasterData,
        0
    )

    totalPopulation = np.ma.sum(
        rasterData,
        dtype=np.float64
    )

    return float(totalPopulation)


# Read one demographic group
def __getGroupPopulation(rootPath: Path, folderName: str, year: int) -> float:
    folderPath = rootPath / folderName

    tifPath = __findYearTif(
        folderPath=folderPath,
        year=year
    )

    population = __rasterPopulationSum(tifPath)

    print(
        f"{year} | "
        f"{folderName:<25} | "
        f"{tifPath.name} | "
        f"population = {population:,.0f}"
    )

    return population

# Calculate yearly population ratios
def calculatePopulationRatios(rootPath: Path) -> pd.DataFrame:
    resultRows = []

    for year in YEARS:
        print("\n" + "=" * 80)
        print(f"Processing {year}")
        print("=" * 80)

        groupPopulations = {}

        for groupName, folderName in GROUP_FOLDERS.items():

            groupPopulations[groupName] = (
                __getGroupPopulation(
                    rootPath=rootPath,
                    folderName=folderName,
                    year=year
                )
            )

        malePopulation = groupPopulations["Male"]
        femalePopulation = groupPopulations["Female"]

        youngPopulation = groupPopulations["Young"]
        middlePopulation = groupPopulations["Middle"]
        oldPopulation = groupPopulations["Old"]

        # Gender composition
        genderTotal = malePopulation + femalePopulation
        maleShare = malePopulation / genderTotal
        femaleShare = femalePopulation / genderTotal
        maleFemaleRatio = malePopulation / femalePopulation

        # Age composition
        ageTotal = youngPopulation + middlePopulation + oldPopulation
        youngShare = youngPopulation / ageTotal
        middleShare = middlePopulation / ageTotal
        oldShare = oldPopulation / ageTotal
        youngMiddleRatio = youngPopulation / middlePopulation
        oldMiddleRatio = oldPopulation / middlePopulation

        resultRows.append({
            "Year": year,

            "Male": malePopulation,
            "Female": femalePopulation,
            "Young": youngPopulation,
            "Middle": middlePopulation,
            "Old": oldPopulation,

            "MaleShare": maleShare,
            "FemaleShare": femaleShare,
            "MaleFemaleRatio": maleFemaleRatio,

            "YoungShare": youngShare,
            "MiddleShare": middleShare,
            "OldShare": oldShare,

            "YoungMiddleRatio": youngMiddleRatio,
            "OldMiddleRatio": oldMiddleRatio
        })

    return pd.DataFrame(resultRows)



# Stability summary
def calculateStability(populationRatios: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "MaleFemaleRatio",
        "YoungMiddleRatio",
        "OldMiddleRatio",
        "MaleShare",
        "FemaleShare",
        "YoungShare",
        "MiddleShare",
        "OldShare"
    ]

    resultRows = []
    yearValues = populationRatios["Year"].to_numpy(dtype=float)

    for metric in metrics:
        metricValues = populationRatios[metric].to_numpy(dtype=float)
        meanValue = metricValues.mean()
        stdValue = metricValues.std(ddof=1)

        if meanValue != 0:
            cvPercent = stdValue / meanValue * 100
        else:
            cvPercent = np.nan

        slopePerYear = np.polyfit(
            yearValues,
            metricValues,
            1
        )[0]

        minValue = metricValues.min()
        maxValue = metricValues.max()
        changeValue = metricValues[-1] - metricValues[0]

        if metricValues[0] != 0:
            relativeChangePercent = changeValue / metricValues[0] * 100
        else:
            relativeChangePercent = np.nan

        result = {
            "Metric": metric,

            "Mean": meanValue,
            "Std": stdValue,
            "CVPercent": cvPercent,

            "Min": minValue,
            "Max": maxValue,
            "Range": maxValue - minValue,

            "Value2015": metricValues[0],
            "Value2020": metricValues[-1],

            "Change2015To2020": changeValue,
            "RelativeChangePercent": relativeChangePercent,
            "SlopePerYear": slopePerYear
        }

        if metric.endswith("Share"):
            result["RangePercentagePoints"] = (maxValue - minValue) * 100
            result["Change2015To2020PP"] = (metricValues[-1] - metricValues[0]) * 100

        resultRows.append(result)

    return pd.DataFrame(resultRows)


def populationVerify(path: str, savePath: str) -> None:
    populationRatios = calculatePopulationRatios(Path(path))
    stabilitySummary = calculateStability(populationRatios)

    displayData = populationRatios.copy()

    shareColumns = [
        "MaleShare",
        "FemaleShare",
        "YoungShare",
        "MiddleShare",
        "OldShare"
    ]

    for columnName in shareColumns:
        displayData[columnName] *= 100

    print("\n")
    print("=" * 80)
    print("YEARLY POPULATION RATIOS")
    print("=" * 80)

    print(
        displayData[[
                "Year",

                "MaleShare",
                "FemaleShare",
                "MaleFemaleRatio",

                "YoungShare",
                "MiddleShare",
                "OldShare",

                "YoungMiddleRatio",
                "OldMiddleRatio"
        ]].round(4).to_string(index=False)
    )

    print("\n")
    print("=" * 80)
    print("STABILITY SUMMARY")
    print("=" * 80)

    print(stabilitySummary.round(4).to_string(index=False))

    populationRatios.to_csv(
        os.path.join(savePath, "populationRatios2015To2020.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    stabilitySummary.to_csv(
        os.path.join("populationStability2015To2020.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFinished.")

#Debug
if __name__ == "__main__":
    RESULT_DATA_PATH = r"F:\0_Project_BackupData\0_Finished\China Dynamic\_AnalysisData"
    RESULT_ROOT = os.path.join(RESULT_DATA_PATH, "result")

    populationVerify(r"F:\Population_Related\CHN Charger Pop", RESULT_ROOT)