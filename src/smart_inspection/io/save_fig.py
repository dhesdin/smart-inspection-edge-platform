from pathlib import Path

from matplotlib import pyplot as plt


def save_figure(fig: plt.Figure, file_path: str | Path) -> None:
    """Save a matplotlib figure as an image file.

    Args:
        fig (plt.Figure): The matplotlib figure to save.
        file_path (str | Path): The path to the image file.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname=file_path, dpi=150, bbox_inches="tight")
    print(f"Figure has been saved to : {file_path}")
    plt.close(fig=fig)
