"""Figure 4: sensitivity curves, rank stability and efficiency ranges."""

from pathlib import Path
from typing import Any, Dict, Tuple
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import DATA
import argparse, csv, hashlib, json, pickle
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import spearmanr, rankdata

HIGHLIGHT_CITIES = ("dongguan", "hefei", "qiqihaer")
GRIDS = {
    "alpha": np.arange(8) * 0.3,
    "sigma": np.arange(5, 36, 5),
    "gamma": np.arange(8) * 0.3,
    "waiting": np.arange(0, 10, 2),
}
REFS = {"alpha": 2, "sigma": 2, "gamma": 4, "waiting": 1}
FILES = {
    "alpha": "country_fr_os.pkl",
    "sigma": "country_fr_tmp.pkl",
    "gamma": "country_fr_decay.pkl",
}
LABELS = {
    "alpha": r"Spatial perturbation, $\alpha$",
    "sigma": r"Departure-time dispersion, $\sigma$ (min)",
    "gamma": r"Scale perturbation, $\gamma$",
    "waiting": r"Vehicle waiting time, $w_v$ (min)",
}
SYMBOLS = {
    "alpha": r"\alpha",
    "sigma": r"\sigma",
    "gamma": r"\gamma",
    "waiting": r"w_v",
}
MEDIAN = "#2c3e50"
RANK = "#3C5488"
REFERENCE = "#777777"
GREEN = "#00A087"


def load_pickle(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def normalize_series_dict(
    raw_data: Dict[str, Any],
    expected_length: int,
    source_name: str,
) -> Dict[str, np.ndarray]:
    if not isinstance(raw_data, dict):
        raise TypeError(f"{source_name} must contain a city-to-series dictionary")

    normalized: Dict[str, np.ndarray] = {}
    for city, values in raw_data.items():
        series = np.asarray(values, dtype=float).squeeze()
        if series.ndim != 1 or series.size != expected_length:
            raise ValueError(
                f"{source_name}: {city!r} has shape {np.asarray(values).shape}; "
                f"expected a one-dimensional series of length {expected_length}"
            )
        normalized[str(city)] = series
    return normalized


def load_waiting_time_data(data_dir: Path, window_minutes: int = 60) -> Dict[str, np.ndarray]:
    metadata = json.loads((data_dir / "waiting_experiments.json").read_text(encoding="utf-8"))
    config = metadata["experiments"]["vehicle_wait"]
    if not np.array_equal(GRIDS["waiting"], config["settings_minutes"]):
        raise ValueError("Vehicle-waiting settings do not match the figure grid")
    suffix = config["result_suffixes"][str(window_minutes)]
    expected = {row["city"]: row["province"] for row in config["cities"]}
    shape = (len(GRIDS["waiting"]), config["replicates"])
    waiting_data: Dict[str, np.ndarray] = {}
    source_paths = sorted(data_dir.glob(f"*/*{suffix}"))
    if not source_paths:
        raise FileNotFoundError(f"No province/*{suffix} files found in {data_dir}")

    for path in source_paths:
        city = path.name[: -len(suffix)]
        if expected.get(city) != path.parent.name or city in waiting_data:
            raise ValueError(f"Unexpected or duplicate vehicle-waiting input: {path}")
        raw = np.asarray(load_pickle(path), dtype=float)
        if raw.shape != shape or not np.isfinite(raw).all():
            raise ValueError(f"{path} must contain finite values in an array of shape {shape}")
        waiting_data[city] = raw.mean(axis=1)

    if set(waiting_data) != set(expected):
        raise ValueError("Vehicle-waiting data do not match the configured city list")
    return waiting_data


def load_figure_data(
    data_dir: Path,
    expected_city_count: int,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    alpha_data = normalize_series_dict(
        load_pickle(data_dir / "country_fr_os.pkl"),
        8,
        "country_fr_os.pkl",
    )
    sigma_data = normalize_series_dict(
        load_pickle(data_dir / "country_fr_tmp.pkl"),
        7,
        "country_fr_tmp.pkl",
    )
    gamma_data = normalize_series_dict(
        load_pickle(data_dir / "country_fr_decay.pkl"),
        8,
        "country_fr_decay.pkl",
    )
    waiting_data = load_waiting_time_data(data_dir)

    common_cities = sorted(
        set(alpha_data) & set(sigma_data) & set(gamma_data) & set(waiting_data)
    )
    if not common_cities:
        raise ValueError("The four Fig. 4 data sets have no cities in common")

    missing_highlights = sorted(set(HIGHLIGHT_CITIES) - set(common_cities))
    if missing_highlights:
        raise ValueError(
            "Representative cities are missing from the common data set: "
            + ", ".join(missing_highlights)
        )

    if expected_city_count > 0 and len(common_cities) != expected_city_count:
        warnings.warn(
            f"Fig. 4 will use {len(common_cities)} common cities, not the expected "
            f"{expected_city_count}.",
            stacklevel=2,
        )

    def select(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {city: data[city] for city in common_cities}

    print(f"Using {len(common_cities)} cities common to all Fig. 4 panels.")
    return (
        select(alpha_data),
        select(sigma_data),
        select(gamma_data),
        select(waiting_data),
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(data_dir, cities_csv):
    a, s, g, w = load_figure_data(data_dir, 406)
    mappings = {"alpha": a, "sigma": s, "gamma": g, "waiting": w}
    cities = list(a)
    matrices = {
        k: np.array([mapping[c] for c in cities], dtype=float)
        for k, mapping in mappings.items()
    }
    for k, m in matrices.items():
        if not np.isfinite(m).all() or m.shape != (len(cities), len(GRIDS[k])):
            raise ValueError(f"Invalid sensitivity input matrix: {k}")
    config_path = data_dir / "waiting_experiments.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))["experiments"]["vehicle_wait"]
    suffix = config["result_suffixes"]["60"]
    paths = {p.name[: -len(suffix)]: p for p in sorted(data_dir.glob("*/*" + suffix))}
    with cities_csv.open(encoding="utf-8-sig", newline="") as f:
        table = {r["city"] for r in csv.DictReader(f)}
    provenance = {
        "selection": "Sorted intersection of the four sensitivity series",
        "n": len(cities),
        "city_order": cities,
        "city_table": str(cities_csv),
        "city_table_sha256": sha(cities_csv),
        "common_cities_outside_manuscript_table": sorted(set(cities) - table),
        "manuscript_cities_without_waiting_input": sorted(table - set(w)),
        "inputs": {
            k: {"path": str(data_dir / n), "sha256": sha(data_dir / n)}
            for k, n in FILES.items()
        },
        "waiting_inputs": [
            {"city": c, "path": str(paths[c]), "sha256": sha(paths[c])} for c in cities
        ],
        "waiting_definition": (
            "Vehicle waiting w_v=0,2,4,6,8 min, punctual pickups and uniform departures in 60 min. "
            "Eight repeats per setting. Every connection advances strictly in scheduled pickup time."
        ),
        "waiting_configuration": {
            "path": str(config_path),
            "sha256": sha(config_path),
        },
        "reference_definitions": {
            k: {"index": REFS[k], "value": float(GRIDS[k][REFS[k]])} for k in matrices
        },
    }
    return cities, matrices, provenance


def correlations(matrix, ref):
    # Spearman correlation between the reference setting and each parameter setting.
    values = np.array(
        [
            spearmanr(matrix[:, ref], matrix[:, i]).statistic
            for i in range(matrix.shape[1])
        ],
        float,
    )
    manual = np.array(
        [
            np.corrcoef(
                rankdata(matrix[:, ref], method="average"),
                rankdata(matrix[:, i], method="average"),
            )[0, 1]
            for i in range(matrix.shape[1])
        ]
    )
    if not np.isfinite(values).all() or not np.allclose(values, manual, atol=1e-12):
        raise ValueError("Spearman/rank verification failed")
    if not np.isclose(values[ref], 1):
        raise ValueError("Reference self-correlation differs from 1")
    return values


def style():
    arial = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    for p in [arial, bold]:
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 24,
            "axes.labelsize": 26,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "axes.linewidth": 1.4,
            "lines.linewidth": 2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.pad": 7,
            "ytick.major.pad": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )


def ribbon(ax, x, m):
    # Draw 15 nested percentile bands around the median.
    cmap = plt.get_cmap("coolwarm")
    for i, p in enumerate(np.linspace(5, 45, 15)):
        ax.fill_between(
            x,
            np.percentile(m, p, axis=0),
            np.percentile(m, 100 - p, axis=0),
            color=cmap(0.1 + i / 15 * 0.8),
            alpha=0.15 + i / 15 * 0.25,
            lw=0,
        )
    ax.plot(x, np.median(m, axis=0), color=MEDIAN, lw=2.6, zorder=4)


def inset_hist(ax, m):
    # Show each city's eta range across parameter settings.
    inset = ax.inset_axes([0.38, 0.23, 0.58, 0.51], zorder=8)
    inset.set_facecolor("white")
    values = np.ptp(m, axis=1)
    counts, _, _ = inset.hist(values, bins=30, color="#768696", alpha=0.35, lw=0.8)
    inset.axvline(values.mean(), color="#768696", ls="--", lw=1.3)
    xmax = max(0.2, np.ceil(values.max() * 10) / 10)
    # Use three evenly spaced integer ticks on the histogram y axis.
    target = counts.max() / 2
    magnitude = 10 ** np.floor(np.log10(max(target, 1)))
    step = next(
        v * magnitude for v in [1, 2, 3, 4, 5, 6, 8, 10] if v * magnitude >= target
    )
    ymax = 2 * step
    inset.set(xlabel=r"$\Delta\eta$", ylabel="Count", xlim=(0, xmax), ylim=(0, ymax))
    inset.set_xticks([0, xmax / 2, xmax])
    inset.set_yticks([0, step, ymax])
    inset.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    inset.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    inset.xaxis.label.set_size(23)
    inset.yaxis.label.set_size(23)
    inset.xaxis.labelpad = 3
    inset.yaxis.labelpad = 3
    inset.tick_params(labelsize=20, pad=5, length=5)
    inset.spines[["top", "right"]].set_visible(False)


def export_stats(cities, matrices, provenance, output):
    records = []
    summary = {}
    np.savez_compressed(
        output / "fig4_inputs_snapshot.npz", cities=np.array(cities), **matrices
    )
    with (output / "city_eta_by_parameter.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["experiment", "city", "parameter", "eta"])
        for k, m in matrices.items():
            for i, c in enumerate(cities):
                for j, x in enumerate(GRIDS[k]):
                    w.writerow([k, c, float(x), float(m[i, j])])
    for k, m in matrices.items():
        scores = correlations(m, REFS[k])
        summary[k] = {
            "n": len(cities),
            "reference_value": float(GRIDS[k][REFS[k]]),
            "rho_min": float(scores.min()),
            "rho_max": float(scores.max()),
            "scores": scores.tolist(),
        }
        for j, x in enumerate(GRIDS[k]):
            records.append(
                [
                    k,
                    float(x),
                    float(GRIDS[k][REFS[k]]),
                    len(cities),
                    float(scores[j]),
                    float(np.median(m[:, j])),
                ]
            )
    with (output / "spearman_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(
            ["experiment", "parameter", "reference", "n", "spearman_rho", "median_eta"]
        )
        w.writerows(records)
    (output / "statistics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8"
    )
    (output / "input_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8"
    )
    return summary


def draw(matrices, n, output, dpi=600):
    style()
    fig, axes = plt.subplots(2, 4, figsize=(25, 13))
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.12, top=0.875, wspace=0.48, hspace=0.58
    )
    slots = [
        ("alpha", axes[0, 0], axes[0, 1], "a", "b"),
        ("sigma", axes[0, 2], axes[0, 3], "c", "d"),
        ("gamma", axes[1, 0], axes[1, 1], "e", "f"),
        ("waiting", axes[1, 2], axes[1, 3], "g", "h"),
    ]
    for k, left, right, a, b in slots:
        x = GRIDS[k]
        m = matrices[k]
        ref = float(x[REFS[k]])
        ribbon(left, x, m)
        left.axvline(ref, color=REFERENCE, ls="-.", lw=1.2)
        left.plot(
            ref,
            0,
            transform=left.get_xaxis_transform(),
            marker="^",
            ms=11,
            mfc="white",
            mec=REFERENCE,
            mew=1.2,
            clip_on=False,
            zorder=7,
        )
        left.set_ylabel(r"Efficiency limit, $\eta$")

        left.set_ylim(
            {
                "alpha": (0.05, 0.35),
                "gamma": (0.05, 0.35),
                "sigma": (0, 1),
                "waiting": (0, 1),
            }[k]
        )
        scores = correlations(m, REFS[k])
        right.axhspan(0.9, 1, color=GREEN, alpha=0.15, zorder=0)
        right.axhline(0.9, color=GREEN, lw=0.8, alpha=0.5, zorder=1)
        # Annotate the comparison value above the curve, away from histogram insets.
        unit = r"\,\mathrm{min}" if k in ["sigma", "waiting"] else ""
        right.text(
            0.5,
            1.025,
            rf"Reference: ${SYMBOLS[k]}={ref:g}{unit}$",
            transform=right.transAxes,
            ha="center",
            va="bottom",
            fontsize=22,
            color=REFERENCE,
        )
        right.plot(x, scores, color=RANK, lw=2.2, marker="o", markersize=4, zorder=4)
        right.plot(
            ref, 1, marker="D", ms=7, mfc="white", mec=REFERENCE, mew=1.4, zorder=5
        )
        right.set_ylim(0, 1.035)
        right.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        right.set_ylabel(r"Spearman correlation, $r_s$")

        inset_hist(right, m)
        for ax, letter in [(left, a), (right, b)]:
            ax.set_xlabel(LABELS[k].replace(", ", chr(10)), labelpad=10)
            ax.set_xlim(x[0] - 0.045 * (x[-1] - x[0]), x[-1] + 0.045 * (x[-1] - x[0]))
            ax.set_xticks(x if k in ["sigma", "waiting"] else [0, 0.5, 1, 1.5, 2])
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(
                -0.22,
                1.16,
                letter,
                transform=ax.transAxes,
                fontsize=30,
                fontweight="bold",
                va="top",
            )
    legend = [
        Line2D([], [], color=MEDIAN, lw=2.6, label="Median across cities"),
        Patch(
            facecolor=plt.get_cmap("coolwarm")(0.2),
            alpha=0.4,
            edgecolor="none",
            label="Cross-city percentile bands (5th-95th)",
        ),
        Line2D(
            [],
            [],
            color=REFERENCE,
            ls="-.",
            marker="^",
            ms=11,
            mfc="white",
            mec=REFERENCE,
            lw=1.2,
            label="Reference setting",
        ),
        Patch(facecolor=GREEN, alpha=0.15, edgecolor="none", label=r"$r_s > 0.9$"),
    ]
    legend = [legend[i] for i in [0, 2, 1, 3]]  # Lines first, then filled regions.
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.973),
        ncol=4,
        frameon=False,
        fontsize=22,
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.savefig(output / "Fig4.pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(
        output / f"Fig4_{dpi}dpi.png", dpi=dpi, bbox_inches="tight", pad_inches=0.16
    )
    fig.savefig(
        output / "Fig4_preview.png", dpi=100, bbox_inches="tight", pad_inches=0.16
    )
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DATA / "country")
    p.add_argument("--cities-csv", type=Path, default=DATA / "city_indicators.csv")
    p.add_argument("--output-dir", type=Path, default=Path("results/main_text/fig4"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cities, matrices, provenance = load_inputs(args.data_dir, args.cities_csv)
    summary = export_stats(cities, matrices, provenance, args.output_dir)
    draw(matrices, len(cities), args.output_dir)
    print(json.dumps(summary, indent=2))
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
