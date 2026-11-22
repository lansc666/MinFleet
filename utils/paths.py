"""Package inputs and independent result directories."""

from datetime import datetime
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "data"


def make_output(section, task, output=None):
    path = (
        Path(output).resolve()
        if output
        else PACKAGE
        / "results"
        / section
        / (task + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Choose an empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
