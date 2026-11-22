"""Seeded vehicle-waiting experiments with punctual pickups."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix


def load_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def load_inputs(country, province, city, params, time_window):
    country = Path(country)
    population = gpd.read_file(country / province / f"{city}_center.shp")
    routes = np.asarray(load_pickle(country / "routes" / f"{city}_routes_con.pkl"), dtype=float)
    connector = load_pickle(country / "routes" / f"{city}_routes_sim.pkl")
    pop = population["value"].to_numpy(dtype=float)
    weights = (pop[:, None] ** params[0]) * (pop[None, :] ** params[1]) / ((routes + 1) ** params[2])
    np.fill_diagonal(weights, 0)
    probability = weights.ravel() / weights.sum()
    cdf = probability.cumsum()
    cdf /= cdf[-1]
    lower0 = np.full(routes.shape, 32767, dtype=np.int16)
    upper = np.full(routes.shape, -32768, dtype=np.int16)
    destinations = []
    for origin in range(len(population)):
        keys = np.asarray(list(connector[origin]), dtype=np.int64)
        duration = np.asarray(list(connector[origin].values()), dtype=float) / 60
        lower0[origin, keys] = np.ceil(-duration).astype(np.int16)
        upper[origin, keys] = np.floor(-duration).astype(np.int16)
        destinations.append(keys)
    return {"n": int(pop.sum() // 1000), "grid": population, "routes": routes,
            "cdf": cdf,
            "lower0": lower0, "upper": upper, "destinations": destinations,
            "time_window_minutes": time_window}


@njit(cache=True)
def _hk_dfs(vertex, pointers, neighbors, left_match, right_match, distance):
    sentinel = len(left_match)
    if vertex == sentinel:
        return True
    for k in range(pointers[vertex], pointers[vertex + 1]):
        target = neighbors[k]
        mate = right_match[target]
        if distance[mate] == distance[vertex] + 1:
            if _hk_dfs(mate, pointers, neighbors, left_match, right_match, distance):
                right_match[target] = vertex
                left_match[vertex] = target
                return True
    distance[vertex] = 1 << 60
    return False


@njit(cache=True)
def _hk_ordered(pointers, neighbors, right_count):
    left_count = len(pointers) - 1
    sentinel = left_count
    infinity = 1 << 60
    left_match = np.full(left_count, -1, dtype=np.int64)
    right_match = np.full(right_count, sentinel, dtype=np.int64)
    distance = np.empty(left_count + 1, dtype=np.int64)
    queue = np.empty(left_count + 1, dtype=np.int64)
    while True:
        tail = 0
        for vertex in range(left_count):
            if left_match[vertex] < 0:
                distance[vertex] = 0
                queue[tail] = vertex
                tail += 1
            else:
                distance[vertex] = infinity
        distance[sentinel] = infinity
        head = 0
        while head < tail:
            vertex = queue[head]
            head += 1
            if distance[vertex] < distance[sentinel]:
                for k in range(pointers[vertex], pointers[vertex + 1]):
                    mate = right_match[neighbors[k]]
                    if distance[mate] == infinity:
                        distance[mate] = distance[vertex] + 1
                        queue[tail] = mate
                        tail += 1
        if distance[sentinel] == infinity:
            break
        for vertex in range(left_count):
            if left_match[vertex] < 0:
                _hk_dfs(vertex, pointers, neighbors, left_match, right_match, distance)
    return (left_match, right_match)


def _generate_trip(data, seed):
    rng = np.random.RandomState(seed)
    size = len(data['grid'])
    ids = data['cdf'].searchsorted(rng.random_sample(data['n']), side='right')
    origin, destination = (ids // size, ids % size)
    departure = rng.uniform(0, data['time_window_minutes'], data['n']).astype(np.int64)
    duration = np.floor(data['routes'][origin, destination] / 60 + 0.5).astype(np.int64)
    return np.column_stack((origin, destination, departure, departure + duration))


def _fast_edges(data, trip, wv):
    origin, destination, departure, arrival = trip.T
    nodes = len(data['grid'])

    def grouped(values):
        ordered = np.argsort(values, kind='stable')
        counts = np.bincount(values, minlength=nodes)
        return np.split(ordered, np.cumsum(counts)[:-1])
    ending = grouped(destination)
    starting = grouped(origin)
    rows, columns = ([], [])
    for o, successors in enumerate(starting):
        if successors.size == 0:
            continue
        candidates = [ending[d] for d in data['destinations'][o] if ending[d].size]
        if not candidates:
            continue
        predecessors = np.concatenate(candidates)
        lower = data['lower0'][o, destination[predecessors]] - wv
        upper = data['upper'][o, destination[predecessors]]
        successor_departure = departure[successors][:, None]
        offset = arrival[predecessors][None, :] - successor_departure
        # Strictly increasing pickup times make the trip graph acyclic.
        forward = departure[predecessors][None, :] < successor_departure
        mask = forward & (offset >= lower[None, :]) & (offset <= upper[None, :])
        ii, jj = np.nonzero(mask)
        if ii.size:
            rows.append(predecessors[jj])
            columns.append(successors[ii])
    if not rows:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    return (np.concatenate(rows), np.concatenate(columns))


def _fast_result(data, trip, wv):
    rows, columns = _fast_edges(data, trip, wv)
    n = data["n"]
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, columns)), shape=(n, n))
    matching, _ = _hk_ordered(graph.indptr.astype(np.int64), graph.indices.astype(np.int64), n)
    return (n - int(np.count_nonzero(matching >= 0))) / n


def seed_matrix(experiment, city, settings, replicates, seed):
    return [[int.from_bytes(hashlib.sha256(
        f"minfleet/{experiment}/{city}/{seed}/{setting}/{replicate}".encode()
    ).digest()[:4], "little") for replicate in range(replicates)] for setting in settings]


def simulate_city(country, province, city, params, config, seeds):
    data = load_inputs(country, province, city, params, config["time_window_minutes"])
    if data["n"] == 0:
        raise ValueError(f"{city} has fewer than 1,000 people in its input grid")
    values = np.empty((len(config["settings_minutes"]), config["replicates"]), dtype=float)
    for k, setting in enumerate(config["settings_minutes"]):
        for replicate in range(config["replicates"]):
            trip = _generate_trip(data, seeds[k][replicate])
            values[k, replicate] = _fast_result(data, trip, setting)
    return values


def _run_city(country, output, config, city_record, params, seed):
    city, province = city_record["city"], city_record["province"]
    seeds = (city_record["seeds"] if seed is None else seed_matrix(
        "vehicle_wait", city, config["settings_minutes"], config["replicates"], seed))
    values = simulate_city(country, province, city, params, config, seeds)
    folder = Path(output) / "country" / province
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{city}{config['result_suffix']}").open("wb") as stream:
        pickle.dump(values, stream, protocol=4)
    return {"city": city, "province": province, "seeds": seeds}


def run(args, section):
    from utils.paths import DATA, make_output

    metadata = json.loads((args.country_data / "waiting_experiments.json").read_text(encoding="utf-8"))
    settings = metadata["experiments"]["vehicle_wait"]
    window = args.window_minutes if args.window_minutes is not None else 60
    config = {
        **settings,
        "time_window_minutes": window,
        "result_suffix": settings["result_suffixes"][str(window)],
    }
    city_records = config.pop("cities")
    if not args.all_cities:
        city_records = [r for r in city_records if r["city"] == args.city and r["province"] == args.province]
        if not city_records:
            if args.seed is None:
                raise ValueError("This city has no saved experiment seeds; supply --seed for a new simulation.")
            city_records = [{"city": args.city, "province": args.province}]
    params = (np.asarray(args.params, dtype=float) if args.params else
              np.asarray(load_pickle(DATA / "params_12h.pkl"), dtype=float).mean(axis=0)[:3])
    output = make_output(section, f"simulation_vehicle_wait_{window}min", args.output)
    results = []
    jobs = [(args.country_data, output, config, record, params, args.seed)
            for record in city_records]
    if args.all_cities:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_city, *job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 100 == 0 or len(results) == len(jobs):
                    print(f"Vehicle waiting, {window} min: {len(results)}/{len(jobs)} cities", flush=True)
    else:
        results.append(_run_city(*jobs[0]))
    record = {**config, "experiment": "vehicle_wait", "params": params.tolist(),
              "seed_override": args.seed, "cities": sorted(results, key=lambda r: r["city"])}
    (output / "simulation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if len(city_records) == 1:
        city = city_records[0]
        path = output / "country" / city["province"] / f"{city['city']}{config['result_suffix']}"
        values = np.asarray(load_pickle(path))
        np.savetxt(output / "eta_replicates.csv", values, delimiter=",")
        np.savetxt(output / "eta_means.csv", values.mean(axis=1), delimiter=",")
    print(f"Results: {output}")
