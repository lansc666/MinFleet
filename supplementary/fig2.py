"""Supplementary Figure 2: sensitivity analysis."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib as mpl

mpl.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plot_style import configure_arial
from utils.paths import DATA, PACKAGE


DEFAULT_DATA_DIR = DATA / "country"
DEFAULT_OUTPUT = PACKAGE / "results" / "supplementary" / "SI_Fig2.png"

CITY_DATA = (
    ("dongguan", "Dongguan", Path("Guangdong/dongguan_pop.shp")),
    ("taiyuan", "Taiyuan", Path("Shanxi/taiyuan_pop.shp")),
    ("qiqihaer", "Qiqihar", Path("Heilongjiang/qiqihaer_pop.shp")),
)
PARAMETER_VALUES = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1])
PANEL_LABELS = ("a", "b", "c", "d", "e", "f")
SERIES = (
    ("country_fr_os.pkl", r"$\alpha_{\mathrm{pop}}$", "#963634"),
    ("country_fr_ds.pkl", r"$\beta_{\mathrm{pop}}$", "#3E68A6"),
    ("country_fr_decay.pkl", r"$\gamma$", "#768696"),
)

PANEL_LABEL_SIZE = 11
CITY_LABEL_SIZE = 9.5
AXIS_LABEL_SIZE = 9
TICK_LABEL_SIZE = 8
LEGEND_SIZE = 7.5
COLORBAR_LABEL_SIZE = 8.5
COLORBAR_TICK_SIZE = 7.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--font-file", type=Path, default=None)
    parser.add_argument("--allow-font-fallback", action="store_true")
    return parser.parse_args()


def load_population_maps(data_dir: Path) -> list[gpd.GeoDataFrame]:
    maps: list[gpd.GeoDataFrame] = []
    for _, city, relative_path in CITY_DATA:
        path = data_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Population shapefile for {city} not found: {path}"
            )
        population = gpd.read_file(path)
        if "value" not in population.columns:
            raise ValueError(f"Population field 'value' not found in {path}")
        population["value"] = pd.to_numeric(
            population["value"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        population = population.dropna(subset=["value", "geometry"])
        population = population[population["value"] > 0].copy()
        if population.empty:
            raise ValueError(
                f"Population shapefile contains no positive values: {path}"
            )
        maps.append(population)
    return maps


def load_pickle(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required sensitivity input not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_sensitivity_data(data_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for filename, _, _ in SERIES:
        raw = load_pickle(data_dir / filename)
        mapping = {
            str(city).strip().casefold(): np.asarray(values, dtype=float).reshape(-1)
            for city, values in raw.items()
        }
        data[filename] = mapping
        for city_key, _, _ in CITY_DATA:
            if city_key not in mapping:
                raise KeyError(f"{filename} does not contain {city_key}")
            if len(mapping[city_key]) != len(PARAMETER_VALUES):
                raise ValueError(
                    f"{filename}[{city_key!r}] has {len(mapping[city_key])} values; "
                    f"expected {len(PARAMETER_VALUES)}"
                )
    return data


def draw_figure(
    population_maps: Sequence[gpd.GeoDataFrame],
    sensitivity_data: Dict[str, Dict[str, np.ndarray]],
    output: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 4.2),
        gridspec_kw={"height_ratios": (1.0, 1.18)},
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.12,
        top=0.94,
        wspace=0.42,
        hspace=0.20,
    )

    for index, (ax, population, (city_key, city, _)) in enumerate(
        zip(axes[0], population_maps, CITY_DATA)
    ):
        existing_axes = set(fig.axes)
        population.plot(
            ax=ax,
            column="value",
            cmap="RdBu_r",
            alpha=0.8,
            legend=True,
            legend_kwds={"shrink": 0.55, "label": "POP", "pad": 0.02},
        )
        colorbar_axes = [item for item in fig.axes if item not in existing_axes]
        if colorbar_axes:
            colorbar_axis = colorbar_axes[-1]
            colorbar_axis.tick_params(labelsize=COLORBAR_TICK_SIZE)
            colorbar_axis.yaxis.label.set_size(COLORBAR_LABEL_SIZE)

        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.text(
            0.50,
            1.02,
            city,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color="0.45",
            fontsize=CITY_LABEL_SIZE,
        )
        ax.text(
            -0.12,
            1.06,
            PANEL_LABELS[index],
            transform=ax.transAxes,
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            va="bottom",
            clip_on=False,
        )

    for index, (ax, (city_key, city, _)) in enumerate(zip(axes[1], CITY_DATA)):
        for filename, label, color in SERIES:
            ax.plot(
                PARAMETER_VALUES,
                sensitivity_data[filename][city_key],
                lw=1.5,
                label=label,
                markersize=3.8,
                marker="o",
                markerfacecolor="white",
                markeredgewidth=0.9,
                color=color,
            )
        ax.set_xlabel("Parameter value", fontsize=AXIS_LABEL_SIZE, labelpad=3)
        ax.set_ylabel(r"$\eta$", fontsize=AXIS_LABEL_SIZE, labelpad=3)
        ax.set_xticks([0, 1, 2])
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
        ax.text(
            0.50,
            1.02,
            city,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color="0.45",
            fontsize=CITY_LABEL_SIZE,
        )
        ax.legend(
            loc="best",
            frameon=False,
            fontsize=LEGEND_SIZE,
            handlelength=1.8,
            borderaxespad=0.2,
        )
        ax.text(
            -0.12,
            1.06,
            PANEL_LABELS[index + 3],
            transform=ax.transAxes,
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            va="bottom",
            clip_on=False,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_arial(args.font_file, args.allow_font_fallback)
    population_maps = load_population_maps(args.data_dir)
    sensitivity_data = load_sensitivity_data(args.data_dir)
    draw_figure(population_maps, sensitivity_data, args.output, args.dpi)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
