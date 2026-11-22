"""Find fonts and PDF rendering tools, and configure figure styles."""

import os
from pathlib import Path
import shutil
import warnings


def publication_font(bold=False):
    variable = "PUBLICATION_FONT_BOLD" if bold else "PUBLICATION_FONT"
    if os.environ.get(variable):
        path = Path(os.environ[variable]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{variable}: {path}")
        return str(path)
    windows = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    if windows.is_file():
        return str(windows)
    from matplotlib.font_manager import FontProperties, findfont

    path = findfont(
        FontProperties(
            family=["Arial", "Liberation Sans", "DejaVu Sans"],
            weight="bold" if bold else "normal",
        )
    )
    warnings.warn(
        "Using an installed sans-serif font. Set PUBLICATION_FONT and "
        "PUBLICATION_FONT_BOLD to use the manuscript Arial files.",
        stacklevel=2,
    )
    return path


def publication_poppler():
    selected = os.environ.get("PDFTOPPM") or shutil.which("pdftoppm")
    if not selected or not Path(selected).is_file():
        raise FileNotFoundError(
            "Install Poppler and put pdftoppm on PATH, or set PDFTOPPM to its executable."
        )
    return Path(selected)


from typing import Optional
import matplotlib as mpl
from matplotlib import font_manager


def configure_arial(
    font_file: Optional[Path] = None,
    allow_font_fallback: bool = False,
    base_size: float = 8.5,
) -> str:
    """Set Matplotlib styles and return the selected font name."""

    if font_file is not None:
        font_file = Path(font_file)
        if not font_file.is_file():
            raise FileNotFoundError(f"Arial font file not found: {font_file}")
        font_manager.fontManager.addfont(str(font_file))

    available = {
        font.name.casefold(): font.name for font in font_manager.fontManager.ttflist
    }
    if "arial" in available:
        selected = available["arial"]
    elif allow_font_fallback:
        selected = available.get("liberation sans", "DejaVu Sans")
        warnings.warn(
            f"Arial is unavailable; using {selected} for a layout preview only.",
            stacklevel=2,
        )
    else:
        raise RuntimeError(
            "Arial is not installed. Supply --font-file /path/to/Arial.ttf, "
            "or use --allow-font-fallback for a layout preview."
        )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected],
            "font.size": base_size,
            "axes.labelsize": base_size,
            "axes.titlesize": base_size,
            "xtick.labelsize": base_size - 0.5,
            "ytick.labelsize": base_size - 0.5,
            "legend.fontsize": base_size - 1,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.unicode_minus": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": selected,
            "mathtext.it": f"{selected}:italic",
            "mathtext.bf": f"{selected}:bold",
            "mathtext.default": "regular",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    return selected
