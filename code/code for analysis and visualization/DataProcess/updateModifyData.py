from pathlib import Path
import pandas as pd

dataDir = Path(__file__).resolve().parent
YEARS = range(2015, 2026)

COLUMN_TO_FILE = {
    "acc_male": "city_optAcc_Male.csv",
    "acc_female": "city_optAcc_Female.csv",
    "acc_children": "city_optAcc_All_children.csv",
    "acc_young": "city_optAcc_All_young.csv",
    "acc_middle": "city_optAcc_All_middle.csv",
    "acc_elderly": "city_optAcc_All_old.csv",
}


def normalize_city_id(series: pd.Series) -> pd.Series:
    """Treat e.g. 156441400 and 156441400.0 as identical city IDs."""
    return series.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)


def updateCity(dataRoot: str | Path, years=YEARS):
    dataDir = Path(dataRoot, "Modified_city_level_acc_pop_groups")
    outDir = Path(dataRoot, "AggResult")
    years = list(years)

    # Load all annual workbooks; acc_total is intentionally never selected.
    annual = {}
    for year in years:
        path = dataDir / f"city_accessibility_{year}.xlsx"
        source = pd.read_excel(path, dtype={"city_id": "string"})

        required = {"city_id", *COLUMN_TO_FILE.keys()}
        missing = required - set(source.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

        source["_city_key"] = normalize_city_id(source["city_id"])
        if source["_city_key"].isna().any() or (source["_city_key"] == "").any():
            raise ValueError(f"{path.name}: missing city_id values")
        if source["_city_key"].duplicated().any():
            raise ValueError(f"{path.name}: duplicate city_id values")

        annual[year] = source.set_index("_city_key")
        print(f"Read {path.name}: {len(source)} cities")

    # Build all six outputs before writing, so input-validation failures
    # do not leave a partially updated set of outputs.
    outputs: dict[str, pd.DataFrame] = {}
    for source_col, csvFile in COLUMN_TO_FILE.items():
        csvPath = outDir / csvFile
        # Read CSV as strings to avoid changing the precision or text of
        # unrelated original columns on read/write.
        target = pd.read_csv(csvPath, dtype=str, keep_default_na=False, encoding="utf-8")
        if "gb" not in target.columns:
            raise ValueError(f"{csvFile}: missing gb column")
        target["_city_key"] = normalize_city_id(target["gb"])
        if target["_city_key"].isna().any() or (target["_city_key"] == "").any():
            raise ValueError(f"{csvFile}: missing gb values")
        if target["_city_key"].duplicated().any():
            raise ValueError(f"{csvFile}: duplicate gb values")

        for year in years:
            targetCol = f"M2SFCA_Accessibility_{year}"
            if targetCol not in target.columns:
                raise ValueError(f"{csvFile}: missing {targetCol}")

            sourceSeries = annual[year][source_col]
            matchedValues = target["_city_key"].map(sourceSeries)
            valid = matchedValues.notna()
            target.loc[valid, targetCol] = matchedValues.loc[valid].astype(str).to_numpy()

            missingCount = (~valid).sum()
            print(f"{csvFile} / {year}: updated {valid.sum()}/{len(target)} cities"
                  + (f"; {missingCount} unmatched/blank source values left unchanged" if missingCount else ""))

        outputs[csvFile] = target.drop(columns="_city_key")

    outDir.mkdir(parents=True, exist_ok=True)
    for csvFile, target in outputs.items():
        outPath = outDir / csvFile
        target.to_csv(outPath, index=False, encoding="utf-8")
        print(f"Saved: {outPath}")


if __name__ == "__main__":
    import os
    RESULT_DATA_PATH = os.path.join(r"F:\0_Project_BackupData\0_Finished\China Dynamic\data\_AnalysisData")
    RESULT_ROOT = os.path.join(RESULT_DATA_PATH, "result")
    updateCity(RESULT_ROOT)