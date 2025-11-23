import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import TYPE_CHECKING, Any

try:
    from .setting import plotSet, FIG_SIZE
except:
    from setting import plotSet, FIG_SIZE

class multiFigs:
    __slots__ = ["__axs", "__fig"]

    def __init__(self, x: int, y: int, figsize: str = "D", sharex: bool = False, sharey: bool = False) -> None:
        plotSet()
        self.__fig, axs = plt.subplots(y, x, figsize=getattr(FIG_SIZE, figsize), sharex=sharex, sharey=sharey)
        self.__axs: list[Axes] = axs.flatten()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__fig, name)
    
    if TYPE_CHECKING:
        def supxlabel(self, xlabel: str, **kwargs: Any) -> Any: ...
        def supylabel(self, ylabel: str, **kwargs: Any) -> Any: ...
        def suptitle(self, title: str, **kwargs: Any) -> Any: ...
        def set_size_inches(self, w: float, h: float, forward: bool = True) -> None: ...
        
    @property
    def axs(self) -> list[Axes]:
        return self.__axs
    
    def globalXlabel(self, label: str | list[str], lens: list[int] = []) -> None:
        indexs = range(len(self.__axs)) if lens == [] else lens
        for j, i in enumerate(indexs):
            if isinstance(label, list):
                if len(label) < len(indexs):
                    raise ValueError("The length of label list is smaller than the length of lens list.")
                self.__axs[i].set_xlabel(label[j])
            else:
                self.__axs[i].set_xlabel(label)

        return
    
    def globalYlabel(self, label: str | list[str], lens: list[int] = []) -> None:
        indexs = range(len(self.__axs)) if lens == [] else lens
        for j, i in enumerate(indexs):
            if isinstance(label, list):
                if len(label) < len(indexs):
                    raise ValueError("The length of label list is smaller than the length of lens list.")
                self.__axs[i].set_ylabel(label[j])
            else:
                self.__axs[i].set_ylabel(label)

        return
    
    def show(self) -> None:
        plt.tight_layout()
        plt.show()
        plt.close()

        return

    def save(self, path, **agrs) -> None:
        plt.tight_layout()
        plt.savefig(path, **agrs)
        plt.close()

        return