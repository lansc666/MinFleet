"""Run the main-text analyses and draw their figures."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import DATA, make_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=["fig2", "fig3", "fig4"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Figure 4: export statistics without rendering.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Figure 3: analyse the completed indicator table without rendering.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.stats_only and args.figure != "fig4":
        parser.error("--stats-only requires fig4")
    if args.analysis_only and args.figure != "fig3":
        parser.error("--analysis-only requires fig3")
    output = make_output("main_text", args.figure, args.output)
    if args.figure == "fig2":
        from main_text.fig2 import run

        run(args.data_root, output, args.dpi)
    elif args.figure == "fig3":
        from main_text.fig3 import run

        run(args.data_root, output, analysis_only=args.analysis_only, dpi=args.dpi)
    else:
        from main_text.fig4 import load_inputs, export_stats, draw

        country = args.data_root / "country"
        cities, matrices, provenance = load_inputs(country, args.data_root / "city_indicators.csv")
        export_stats(cities, matrices, provenance, output)
        if not args.stats_only:
            draw(matrices, len(cities), output, args.dpi)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
