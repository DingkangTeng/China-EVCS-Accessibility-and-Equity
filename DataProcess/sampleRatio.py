import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from .setting import plotSet, FIG_SIZE, BAR_COLORS
except:
    from setting import plotSet, FIG_SIZE, BAR_COLORS

def sampleratio(sample: pd.DataFrame, officoal: pd.DataFrame, path: str ="") -> None:
    plotSet()
    
    fig = plt.figure(figsize=FIG_SIZE.D)
    ax = plt.subplot()
    sampleData = sample.set_index("name").loc["sum"].T
    officialData = officoal.reindex(columns=sample.columns, fill_value=None).set_index("name").loc["sum"].T

    sampleData.plot(
        ax = ax,
        color=BAR_COLORS[0]
    )
    officialData.plot(
        ax = ax,
        color=BAR_COLORS[1]
    )

    # Fill sample gap
    x = range(len(sampleData.index))
    y_sample = sampleData.to_numpy()
    y_official = officialData.to_numpy()
    ax.fill_between(
        x, y_sample, y_official,
        where=(y_official > y_sample).tolist(),
        color=BAR_COLORS[2], alpha=0.8,
        interpolate=True
    )
    
    plt.ylabel("Number of Stations (millions)")
    plt.xlabel("Year")
    plt.legend(["Sample", "Offical", "Sample Gap"])
    
    plt.tight_layout()
    if path == "":
        plt.show()
    else:
        plt.savefig(os.path.join(path, "sampleRate.jpg"), dpi=300)
    
    plt.close()

    return

if __name__ == "__main__":
    evcsStations = pd.read_excel("China_Acc_Results\\Result\\provinceLevel\\China_EVCS.xlsx", sheet_name="stations")
    sample = pd.read_excel("China_Acc_Results\\Result\\provinceLevel\\China_EVCS.xlsx", sheet_name="sample")
    sampleratio(sample, evcsStations)