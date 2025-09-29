import matplotlib.pyplot as plt
from matplotlib.axes import Axes
try:
    from .setting import plotSet, FIG_SIZE
except:
    from setting import plotSet, FIG_SIZE

class multiFigs:
    __slots__ = ["__axs"]

    def __init__(self, x: int, y: int, figsize: str = "D", sharex: bool = False, sharey: bool = False) -> None:
        plotSet()
        fig, axs = plt.subplots(y, x, figsize=getattr(FIG_SIZE, figsize), sharex=sharex, sharey=sharey)
        self.__axs: list[Axes] = axs.flatten()

    @property
    def axs(self) -> list[Axes]:
        return self.__axs
    
    def globalXlabel(self, label: str, lens: list[int] = []) -> None:
        indexs = range(len(self.__axs)) if lens == [] else lens
        for i in indexs:
            self.__axs[i].set_xlabel(label)

        return
    
    def globalYlabel(self, label: str, lens: list[int] = []) -> None:
        indexs = range(len(self.__axs)) if lens == [] else lens
        for i in indexs:
            self.__axs[i].set_ylabel(label)

        return
    
    def show(self) -> None:
        plt.tight_layout()
        plt.show()
        plt.close()

        return

    def save(self, path) -> None:
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        return