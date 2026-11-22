"""Fit the population gravity model to an aggregate OD table."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_text.gravity import gravity_model_popnum


def main():
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="CSV columns: pop_o,pop_d,t,flow; t is in seconds."
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    columns = ["pop_o", "pop_d", "t"]
    values = frame[columns + ["flow"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Inputs must be finite and non-negative.")
    # Fit the gravity parameters by nonlinear least squares.
    params, cov = curve_fit(gravity_model_popnum, frame[columns], frame.flow)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(zip(["alpha", "beta", "gamma", "K"], map(float, params))), indent=2
        ),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
