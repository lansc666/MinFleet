"""Supplementary Figure 1: sensitivity analysis."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from io import BytesIO
from pathlib import Path
from typing import Dict, Mapping, Sequence, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr

from main_text import fig4
from utils.plot_style import configure_arial
from utils.paths import DATA, PACKAGE


DEFAULT_DATA_DIR = DATA / "country"
DEFAULT_OUTPUT = PACKAGE / "results" / "supplementary" / "SI_Fig1.png"

X_RANGES = (np.arange(5, 36, 5), np.arange(0, 121, 20))
SPEARMAN_COLOR = "#9c5569"
PEARSON_COLOR = "#59ac8c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cities-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--font-file", type=Path, default=None)
    parser.add_argument("--allow-font-fallback", action="store_true")
    return parser.parse_args()


def load_pickle(data_dir: Path, filename: str):
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_metric_mapping(
    data_dir: Path,
    aggregate_filename: Union[str, None],
    per_city_suffix: str,
) -> Dict[str, np.ndarray]:
    """Load an aggregate mapping or rebuild it from the per-city files."""

    aggregate_path = data_dir / aggregate_filename if aggregate_filename else None
    if aggregate_path is not None and aggregate_path.is_file():
        return normalise_mapping(load_pickle(data_dir, aggregate_filename))

    paths = sorted(
        data_dir.glob(f"**/*{per_city_suffix}"),
        key=lambda path: (len(path.relative_to(data_dir).parts), str(path)),
    )
    if not paths:
        raise FileNotFoundError(
            f"No input for {aggregate_filename or per_city_suffix} was found in {data_dir}"
        )

    rebuilt: Dict[str, np.ndarray] = {}
    for path in paths:
        city = path.name[: -len(per_city_suffix)].strip().casefold()
        if city in rebuilt:
            warnings.warn(f"Ignoring duplicate input for {city}: {path}", stacklevel=2)
            continue
        with path.open("rb") as handle:
            raw = np.asarray(pickle.load(handle), dtype=float)
        if raw.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional array in {path}; found {raw.shape}"
            )
        rebuilt[city] = np.mean(raw, axis=1)
    return rebuilt


def find_column(
    frame: pd.DataFrame,
    candidates: Union[str, Sequence[str]],
) -> str:
    if isinstance(candidates, str):
        candidates = (candidates,)
    lookup = {str(column).casefold(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    raise ValueError(f"None of {list(candidates)!r} was found in {list(frame.columns)}")


def normalise_mapping(values: Mapping) -> Dict[str, np.ndarray]:
    return {
        str(key).strip().casefold(): np.asarray(value) for key, value in values.items()
    }


def stack_metric(
    mapping: Mapping[str, np.ndarray],
    cities: Sequence[str],
    name: str,
) -> np.ndarray:
    missing = [city for city in cities if city not in mapping]
    if missing:
        raise KeyError(
            f"{name} is missing {len(missing)} cities, including {missing[:5]}"
        )
    arrays = [np.asarray(mapping[city], dtype=float).reshape(-1) for city in cities]
    lengths = {array.size for array in arrays}
    if len(lengths) != 1:
        raise ValueError(
            f"{name} has inconsistent time-series lengths: {sorted(lengths)}"
        )
    return np.stack(arrays, axis=0)


def correlation_values(
    values: np.ndarray, target: np.ndarray, method: str
) -> list[float]:
    function = spearmanr if method == "spearman" else pearsonr
    result = [function(row, target) for row in values.T]
    return [
        float(item.statistic if hasattr(item, "statistic") else item[0])
        for item in result
    ]


def build_correlations(data_dir: Path, cities_csv: Path, output: Path) -> dict:
    """Compute the two temporal panels and the paired-window comparison."""
    waiting_60 = fig4.load_figure_data(data_dir, expected_city_count=406)[-1]
    waiting_120 = fig4.load_waiting_time_data(data_dir, window_minutes=120)
    vehicle_cities = list(waiting_60)
    waiting_by_window = {60: waiting_60, 120: waiting_120}

    cities = pd.read_csv(cities_csv)
    city_column = find_column(cities, "city")
    population_column = find_column(cities, "POP")
    eta_column = find_column(cities, "eta")
    selected = cities[
        pd.to_numeric(cities[population_column], errors="coerce") >= 500000
    ].copy()
    selected["_city_key"] = selected[city_column].astype(str).str.strip().str.casefold()

    metrics = (
        ("sigma", load_metric_mapping(data_dir, "country_fr_tmp.pkl", "_fr_tmp.pkl")),
        ("mu", load_metric_mapping(data_dir, "country_fr_tmp2.pkl", "_fr_tmp2.pkl")),
    )
    available = set(vehicle_cities).intersection(*(set(mapping) for _, mapping in metrics))
    selected = selected[selected["_city_key"].isin(available)]
    if selected.empty or selected["_city_key"].duplicated().any():
        raise ValueError("The temporal panels require a non-empty sample with unique city keys")

    temporal_cities = selected["_city_key"].tolist()
    target = pd.to_numeric(selected[eta_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError(f"The selected city table contains invalid {eta_column} values")
    temporal_series = [stack_metric(mapping, temporal_cities, name) for name, mapping in metrics]
    temporal = {
        method: [correlation_values(matrix, target, method) for matrix in temporal_series]
        for method in ("spearman", "pearson")
    }

    grid = fig4.GRIDS["waiting"]
    ref = fig4.REFS["waiting"]
    vehicle_ranks = {}
    for window, mapping in waiting_by_window.items():
        matrix = stack_metric(mapping, vehicle_cities, f"vehicle waiting, {window} min")
        vehicle_ranks[window] = fig4.correlations(matrix, ref).tolist()

    record = {
        "temporal_city_count": len(temporal_cities),
        "temporal_cities": temporal_cities,
        "temporal_reference": "city_indicators.csv eta",
        "temporal": temporal,
        "vehicle_city_count": len(vehicle_cities),
        "vehicle_cities": vehicle_cities,
        "vehicle_reference": f"{grid[ref]:g} min within each departure window",
        "vehicle_wait_minutes": grid.tolist(),
        "vehicle_spearman_by_departure_window": vehicle_ranks,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "SI_Fig1_statistics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def draw_figure(statistics: dict, output: Path, dpi: int) -> None:
    temporal = statistics["temporal"]
    window_ranks = statistics["vehicle_spearman_by_departure_window"]
    settings = statistics["vehicle_wait_minutes"]
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.97), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.07, h_pad=0.09)
    for i, (ax, xlabel, ticks) in enumerate(zip(
        axes[:2],
        (r"Departure-time scale, $\sigma$ (min)", r"Departure-time location, $\mu$ (min)"),
        ([5, 15, 25, 35], [0, 40, 80, 120]),
    )):
        for method, color, marker, label in (
            ("spearman", SPEARMAN_COLOR, "o", r"Spearman's $\rho$"),
            ("pearson", PEARSON_COLOR, "s", r"Pearson's $r$"),
        ):
            ax.plot(X_RANGES[i], temporal[method][i], color=color, lw=1.4,
                    marker=marker, ms=3.8, mfc="white", label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Correlation coefficient")
        ax.set_xticks(ticks)
        ax.set_ylim((0.92, 1.002) if i == 0 else (0.935, 1.002))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.legend(loc="lower right", fontsize=7.4, frameon=False)

    wait = axes[2]
    for window, color, marker, linestyle, size in (
        (120, "#268777", "^", "--", 5.9),
        (60, "#3c5488", "o", "-", 3.7),
    ):
        wait.plot(settings, window_ranks[window], color=color, marker=marker, ms=size,
                  mfc="white", lw=1.4, linestyle=linestyle, label=f"{window} min")
    handles, labels = wait.get_legend_handles_labels()
    wait.legend(handles[::-1], labels[::-1], loc="lower right", fontsize=7.4, frameon=False)
    wait.set_xlabel(r"Vehicle waiting time, $w_v$ (min)")
    wait.set_ylabel(r"Spearman correlation, $r_s$")
    wait.set_xticks(settings)
    wait.set_ylim(0.25, 1.04)
    wait.set_yticks([0.4, 0.6, 0.8, 1.0])

    for label, ax in zip("abc", axes):
        ax.tick_params(labelsize=8, length=3, pad=7)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.12, 1.035, label, transform=ax.transAxes, fontweight="bold", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Apply the same vector layout pass as the manuscript artwork before rasterizing.
    with BytesIO() as layout:
        fig.savefig(layout, format="pdf", dpi=dpi)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_arial(args.font_file, args.allow_font_fallback)
    cities_csv = args.cities_csv or args.data_dir.parent / "city_indicators.csv"
    statistics = build_correlations(args.data_dir, cities_csv, args.output.parent)
    draw_figure(statistics, args.output, args.dpi)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
