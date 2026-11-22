"""Figure 3: mutual information, UMAP and correlations from nine indicators."""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.paths import DATA, make_output
from main_text import FEATURES
from main_text.mutual_information import sliding

UMAP_PARAMETERS = dict(
    n_components=2,
    n_neighbors=40,
    min_dist=0.3,
    metric="euclidean",
    random_state=42,
    n_jobs=1,
)


def calculate(data_root, output, inputs=None):
    """Read the completed indicator table, then calculate MI and UMAP."""
    output = Path(output)
    inputs = Path(inputs) if inputs else Path(data_root) / "city_indicators.csv"
    frame = pd.read_csv(inputs, float_precision="round_trip")
    missing = sorted(set(FEATURES + ["city", "province", "POP", "eta"]) - set(frame))
    if missing:
        raise ValueError(f"The completed indicator table is missing columns: {missing}")
    if not frame.city.is_unique or frame.city.isna().any():
        raise ValueError("City identifiers must be present and unique.")
    complete = frame[FEATURES].notna().all(axis=1)
    data = (
        frame.loc[complete, ["city", "province", "POP", "eta", *FEATURES]]
        .sort_values("POP", kind="stable")
        .reset_index(drop=True)
    )
    if len(data) != 394:
        raise ValueError("Figure 3 requires 394 cities with all nine indicators.")
    assert np.isfinite(data[FEATURES + ["POP", "eta"]]).all().all()
    data.to_csv(output / "indicators.csv", index=False)
    raw, smooth = sliding(data, FEATURES)
    raw.to_csv(output / "mutual_information_raw.csv", index=False)
    smooth.to_csv(output / "mutual_information.csv", index=False)
    scaled = StandardScaler().fit_transform(data[FEATURES])
    assert scaled.shape == (394, 9) and np.allclose(scaled.mean(axis=0), 0, atol=1e-12)
    pd.DataFrame(scaled, columns=FEATURES).assign(city=data.city).to_csv(
        output / "standardized_indicators.csv", index=False
    )
    print("Fitting UMAP: n_neighbors=40, min_dist=0.3, seed=42.", flush=True)
    embedding = umap.UMAP(**UMAP_PARAMETERS).fit_transform(scaled)
    assert embedding.shape == (394, 2) and np.isfinite(embedding).all()
    coordinates = pd.DataFrame(
        dict(
            city=data.city,
            eta=data.eta,
            UMAP_1=embedding[:, 0],
            UMAP_2=embedding[:, 1],
        )
    )
    coordinates.to_csv(output / "umap_coordinates.csv", index=False)
    summary = dict(
        cities=len(data),
        city_table_cities=len(frame),
        cities_without_complete_indicators=sorted(frame.loc[~complete, "city"]),
        features=FEATURES,
        indicator_table=str(inputs),
        indicator_source="All nine indicators are read from the completed table.",
        MI=dict(
            window_fraction=0.3,
            step_fraction=0.05,
            random_state=42,
            smoothing_sigma=1.5,
            unit="nats",
            row_normalized=False,
        ),
        UMAP=UMAP_PARAMETERS,
        UMAP_standardization="Z-score; ddof=0",
        eta_used_as_UMAP_input=False,
        UMAP_trustworthiness10=float(
            trustworthiness(scaled, embedding, n_neighbors=10)
        ),
    )
    (output / "analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def draw(df, mi, coordinates, output, dpi=600):
    """Draw mutual information, UMAP and indicator correlations."""
    OUT = Path(output)
    F = FEATURES
    assert len(df) == 394 and df.city.is_unique and np.isfinite(df[F]).all().all()
    corr = df[F].corr(method="spearman")
    corr.to_csv(OUT / "spearman.csv")
    mi_max = float(mi[F].to_numpy().max())
    mi_upper = 0.45  # Shared MI colour scale.
    print("MI maximum:", mi_max, "colour upper:", mi_upper, flush=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    eta = df.eta.to_numpy()
    vmin, vmax = np.percentile(eta, [2, 98])
    co = coordinates
    assert co.city.tolist() == df.city.tolist() and np.allclose(co.eta, eta)
    fig = plt.figure(figsize=(9.4, 9.4))
    centers = mi.population_rank_percent.to_numpy()
    step = np.diff(centers).mean()
    wrapped = [
        "Population density (POPDEN)",
        "Population hotspot count\n(C_COUNT)",
        "Population proximity index\n(PPI)",
        "Elongation index (EI)",
        "Landscape shape index (LSI)",
        "Landscape division index\n(LDI)",
        "Distance-normalized travel\ntime (DNTT)",
        "Weighted mean detour ratio\n(WMDR)",
        "Accessibility disparity index\n(ADI)",
    ]
    for group, (title, top) in enumerate(
        [("Population", 0.946), ("Urban form", 0.808), ("Travel time", 0.670)]
    ):
        bottom = top - 0.108
        ax = fig.add_axes([0.275, bottom, 0.625, 0.108])
        im = ax.imshow(
            mi[F[group * 3 : group * 3 + 3]].to_numpy().T,
            aspect="auto",
            interpolation="nearest",
            cmap="magma_r",
            vmin=0,
            vmax=mi_upper,
            extent=[centers[0] - step / 2, centers[-1] + step / 2, 2.5, -0.5],
        )
        ax.set_yticks([])
        fig.text(
            0.055,
            top + 0.009,
            title,
            fontsize=11,
            fontweight="bold",
            color="#444444",
            va="bottom",
        )
        for row in range(3):
            fig.text(
                0.055,
                top - (row + 0.5) * 0.036,
                wrapped[group * 3 + row],
                fontsize=10.5,
                ha="left",
                va="center",
                linespacing=1.05,
            )
        if group == 2:
            ax.set_xticks([20, 40, 60, 80])
            ax.set_xlabel("City population rank percentile", fontsize=11)
        else:
            ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([0.915, 0.562, 0.012, 0.384])
    cb = fig.colorbar(
        im, cax=cax, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.45], format="%.2f"
    )
    cb.set_label("Mutual information", fontsize=11)
    fig.text(0.015, 0.97, "a", fontsize=17, fontweight="bold")
    ax = fig.add_axes([0.075, 0.105, 0.35, 0.35])
    cax = fig.add_axes([0.445, 0.135, 0.012, 0.30])
    emb = co[["UMAP_1", "UMAP_2"]].to_numpy()
    sc = ax.scatter(
        *emb.T,
        c=eta,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        s=14,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.25,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.margins(0.055)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    umap_ax = ax
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    cb = fig.colorbar(sc, cax=cax, extend="both")
    cb.ax.set_title(r"$\eta$", pad=7)
    fig.text(0.025, 0.48, "b", fontsize=17, fontweight="bold")
    ax = fig.add_axes([0.60, 0.105, 0.35, 0.35])
    corr_ax = ax
    mask = np.triu(np.ones((9, 9), bool), k=1)
    vals = np.ma.array(corr.values, mask=mask)
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(9):
        for j in range(i + 1):
            v = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if abs(v) > 0.65 else "#222222",
            )
    ax.set_xticks(range(9), F, rotation=55, ha="right", fontsize=9.5)
    ax.set_yticks(range(9), F, fontsize=9.5)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, 9, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 9, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([0.79, 0.405, 0.16, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.set_xlabel("Spearman's rank\ncorrelation", fontsize=10.5, labelpad=5)
    fig.text(0.55, 0.48, "c", fontsize=17, fontweight="bold")
    stem = "Fig3"
    fig.canvas.draw()
    assert abs(umap_ax.get_position().y0 - corr_ax.get_position().y0) < 1e-8
    assert abs(umap_ax.get_position().y1 - corr_ax.get_position().y1) < 1e-8
    fig.savefig(OUT / f"{stem}_preview.png", dpi=180, facecolor="white")
    fig.savefig(OUT / f"{stem}_{dpi}dpi.png", dpi=dpi, facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", facecolor="white")
    fig.savefig(
        OUT / f"{stem}_180mm_check.png",
        dpi=150 * 180 / (9.4 * 25.4),
        facecolor="white",
    )
    plt.close(fig)
    print(stem, flush=True)


def run(data_root, output, inputs=None, analysis_only=False, dpi=600):
    calculate(data_root, output, inputs)
    if analysis_only:
        return
    # Read the generated CSV representation consistently for plotting.
    data = pd.read_csv(output / "indicators.csv", float_precision="round_trip")
    mi = pd.read_csv(output / "mutual_information.csv", float_precision="round_trip")
    coordinates = pd.read_csv(
        output / "umap_coordinates.csv", float_precision="round_trip"
    )
    draw(data, mi, coordinates, output, dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    output = make_output("main_text", "fig3", args.output)
    run(args.data_root, output, args.inputs, args.analysis_only, args.dpi)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
