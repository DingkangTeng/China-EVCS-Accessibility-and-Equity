import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

try:
    from .setting import plotSet, FIG_SIZE, TITLE, BAR_COLORS
except:
    from setting import plotSet, FIG_SIZE, TITLE, BAR_COLORS

def regression(
    xdf: pd.Series | np.ndarray,
    ydf: list[pd.Series | np.ndarray], ylabels: list[str],
    subydf: list[pd.Series | np.ndarray] = [],
    axs: Axes | None = None,
    figsize: str = "D",
    path: str = ""
) -> None:
    
    plotSet()
    if isinstance(xdf, pd.Series):
        xdf = xdf.to_numpy()
    
    if axs is None:
        fig = plt.figure(figsize=getattr(FIG_SIZE, figsize))
        ax = plt.subplot()
    else:
        ax = axs
    
    xReshape = xdf.reshape(-1, 1)
    model = LinearRegression()
    for i, y in enumerate(ydf + subydf):
        if isinstance(y, pd.Series):
            y = y.dropna().to_numpy()
        model.fit(xReshape, y)
        yPred = model.predict(xReshape)
        r2 = r2_score(y, yPred)
        ax.scatter(xdf, y, alpha=0.7, label=ylabels[i], color=BAR_COLORS[0][i])
        ax.plot(xdf, yPred, color=BAR_COLORS[0][i], linewidth=2, label=ylabels[i])

        textstr = f"R² = {r2:.3f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        
        # 调整文本位置避免重叠
        y_pos = 0.95 - i * 0.08  # 每个R²值向下偏移
        ax.text(0.05, y_pos, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
    
    plt.legend()

    if axs is None:
        plt.xlabel("EV points (million)")
        plt.ylabel("Coverage Index")
        
        plt.show()

    return

# Debug
if __name__ == "__main__":
    from multiFigs import multiFigs
    f = multiFigs(2, 1, figsize="W", sharex=False, sharey=True)

    evcsPoints = pd.read_excel("China_Acc_Results\\Result\\provinceLevel\\China_EVCS.xlsx").drop(columns=["region"])
    evcsPoints = evcsPoints[~evcsPoints["name"].isin(["sum", "ratio"])]
    pop = pd.read_excel("China_Acc_Results\\Result\\provinceLevel\\Raster_Density_population.xlsx")
    gdp = pd.read_excel("China_Acc_Results\\Result\\provinceLevel\\Raster_Density_gdp.xlsx")
    road = pd.read_excel("China_Acc_Results\\Result\\provinceLevel\\Roads_Density.xlsx")

    regression(evcsPoints[2016], [pop[2016], gdp[2016]], ["Population", "GDP"], axs=f.axs[0])
    regression(evcsPoints[2025], [pop[2025], gdp[2025]], ["Population", "GDP"], axs=f.axs[1])
    # regression(evcsPoints[2016], [road[2016]], ["Roads"], axs=f.axs[0])
    # regression(evcsPoints[2025], [road[2025]], ["Roads"], axs=f.axs[1])
    
    f.globalXlabel("EV points (million)")
    f.globalYlabel("Coverage Index", [0])
    f.show()

    