"""Read simulation options, copy city inputs and run the selected experiment."""

import argparse
import json
from pathlib import Path
import pickle
import random
import shutil


def run(section, choices):
    import numpy as np
    from . import experiments
    from utils.paths import DATA, make_output

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=choices)
    parser.add_argument(
        "--country-data",
        type=Path,
        default=DATA / "country",
        help="Saved routing matrices and population centres; defaults to data/country/.",
    )
    parser.add_argument("--province")
    parser.add_argument("--city")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the saved waiting seeds; other experiments default to 42.",
    )
    parser.add_argument(
        "--params", type=float, nargs=3, metavar=("ALPHA", "BETA", "GAMMA")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--window-minutes", type=int, choices=(60, 120),
        help="Uniform departure window for vehicle_wait (default: 60).",
    )
    parser.add_argument("--all-cities", action="store_true", help="Run vehicle_wait for its saved city list.")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    waiting = args.experiment == "vehicle_wait"
    if not waiting and (args.all_cities or args.window_minutes is not None):
        parser.error("--all-cities and --window-minutes are available for vehicle_wait")
    if not args.all_cities and (not args.city or not args.province):
        parser.error("--city and --province are required unless --all-cities is used")
    if args.all_cities and (args.city or args.province):
        parser.error("Choose --all-cities or a single city")
    if not args.all_cities and (
        Path(args.city).name != args.city or Path(args.province).name != args.province
        or "_" in args.city
    ):
        parser.error("Use city and province directory keys without path separators or underscores")
    if waiting:
        from .waiting import run as run_waiting

        run_waiting(args, section)
        return
    if "_" in args.city:
        parser.error(
            "City names containing underscores are not supported."
        )
    if Path(args.city).name != args.city or Path(args.province).name != args.province:
        parser.error(
            "City and province must be directory keys, without path separators."
        )
    files = [
        args.country_data / "routes" / f"{args.city}{suffix}"
        for suffix in ["_routes_con.pkl", "_routes_sim.pkl"]
    ]
    shape = args.country_data / args.province / f"{args.city}_center.shp"
    files += [shape.with_suffix(s) for s in [".shp", ".shx", ".dbf", ".prj"]]
    for p in files:
        if not p.is_file():
            raise FileNotFoundError(p)
    if shape.with_suffix(".cpg").exists():
        files.append(shape.with_suffix(".cpg"))
    output = make_output(section, "simulation_" + args.experiment, args.output)
    country = output / "inputs/country"
    for p in files:
        target = country / p.relative_to(args.country_data)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    # Stored fitted vectors are [alpha,beta,gamma,K]; K cancels in normalized OD probabilities.
    params = (
        np.asarray(args.params, dtype=float)
        if args.params
        else np.mean(pickle.loads((DATA / "params_12h.pkl").read_bytes()), axis=0)[:3]
    )
    if params.shape != (3,):
        raise ValueError(f"Expected three pooled parameters, found {params.shape}")
    seed = 42 if args.seed is None else args.seed
    np.random.seed(seed)
    random.seed(seed)
    experiments.country_path = str(country) + "/"
    experiment = args.experiment
    suffix = experiment
    if experiment in ["baseline", "alpha", "beta", "gamma"]:
        settings = [params.tolist()]
        if experiment != "baseline":
            index = {"alpha": 0, "beta": 1, "gamma": 2}[experiment]
            settings = []
            for value in np.arange(8) * 0.3:
                row = params.copy()
                row[index] = value
                settings.append(row.tolist())
        experiments.get_pop_province(args.province, args.city, settings, suffix)
    else:
        function = {
            "mu": experiments.get_province_temporal,
            "sigma": experiments.get_province_temporal_std,
        }[experiment]
        function(args.province, args.city, params)
        suffix = {
            "mu": "tmp_mean",
            "sigma": "tmp",
        }[experiment]
    result = country / args.province / f"{args.city}_fr_{suffix}.pkl"
    if not result.is_file():
        raise RuntimeError(
            "The experiment produced no result. Check the city input files."
        )
    values = np.asarray(pickle.loads(result.read_bytes()))
    np.savetxt(output / "eta_replicates.csv", values, delimiter=",")
    np.savetxt(output / "eta_means.csv", values.mean(axis=1), delimiter=",")
    record = dict(
        city=args.city,
        province=args.province,
        experiment=experiment,
        params=params.tolist(),
        seed=seed,
        replicates=10,
        shape=list(values.shape),
        source="Per-city demand generation and fleet matching",
        baseline_parameters="Pooled day/night alpha, beta and gamma unless supplied explicitly.",
    )
    invalid = (
        not np.isfinite(values).all()
        or bool((values <= 0).any())
        or bool((values > 1).any())
    )
    record["valid_eta_bounds"] = not invalid
    (output / "simulation.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    if invalid:
        raise ValueError(
            f"Simulation returned eta outside (0, 1]. Inspect the trip and matching outputs at {output}."
        )
    print(f"Results: {output}")
