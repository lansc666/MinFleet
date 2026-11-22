"""Run a Supplementary Information figure."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import DATA, make_output
from utils.plot_style import configure_arial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=["fig1", "fig2"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA / "country")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--font-file", type=Path)
    parser.add_argument("--allow-font-fallback", action="store_true")
    args = parser.parse_args()
    output = make_output("supplementary", args.figure, args.output)
    configure_arial(args.font_file, args.allow_font_fallback)
    if args.figure == "fig1":
        from supplementary.fig1 import build_correlations, draw_figure

        statistics = build_correlations(
            args.data_dir, args.data_dir.parent / "city_indicators.csv", output
        )
        draw_figure(statistics, output / "SI_Fig1.png", args.dpi)
    else:
        from supplementary.fig2 import (
            load_population_maps,
            load_sensitivity_data,
            draw_figure,
        )

        draw_figure(
            load_population_maps(args.data_dir),
            load_sensitivity_data(args.data_dir),
            output / "SI_Fig2.png",
            args.dpi,
        )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
