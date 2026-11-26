# Urban mobility resource efficiency

Code for **Quantifying the structural potential of urban mobility resource efficiency** by Sichen Lan, Fan Zhang, Xiaohu Zhang, Shuyi Liu, Lei Dong and Yu Liu.

## Contents

- `main_text/`: main-text figures, structural indicators, geometry, statistical analysis, preprocessing and simulation entry points.
- `supplementary/`: SI figures and sensitivity experiments.
- `shared/`: demand generation, fleet matching and experiment routines shared by the main text and SI.
- `utils/`: common figure styles and input/output locations.
- `data/`: `national_map.png`, `city_indicators.csv`, fitted parameters and saved ORS data. Nationwide inputs are grouped by province under `country/`; matrices and node coordinates are in `country/routes/`.

## Installation

Use Python 3.12:

```bash
conda env create -f environment.yml
conda activate minfleet
```

Alternatively, run `python -m pip install -r requirements.txt` and install [Poppler](https://poppler.freedesktop.org/) for Figure 2 with its `bin` directory on `PATH`. Arial reproduces the figure typography. SI commands accept `--font-file PATH` or `--allow-font-fallback`.

## Reproduce the figures

Run from the repository root:

```bash
python main_text/reproduce.py fig2
python main_text/reproduce.py fig3
python main_text/reproduce.py fig4
python supplementary/reproduce.py fig1
python supplementary/reproduce.py fig2
```

| Figure | Calculation |
|---|---|
| Main Fig. 2 | Rankings and population-group statistics for 407 cities; uses the supplied national map image and four-city aggregate comparison values |
| Main Fig. 3 | Sliding-window mutual information, UMAP and Spearman correlations using the nine completed structural indicators |
| Main Fig. 4 | Sensitivity summaries and rank correlations for 406 cities |
| SI Fig. 1 | Pearson and Spearman correlations with baseline η for departure-time scale and location (a–b, 164 cities), and Spearman correlations for vehicle waiting limits under two departure windows (c, 406 cities) |
| SI Fig. 2 | Population maps and spatial sensitivity for three example cities |

Results go to `results/main_text/` or `results/supplementary/`. `--output PATH` selects an empty output directory; `--dpi` controls image resolution (default: 600). Use `fig4 --stats-only` for statistics or `fig3 --analysis-only` to skip drawing. UMAP uses the nine standardized indicators, excluding eta, with `n_neighbors=40`, `min_dist=0.3` and seed 42.

`data/city_indicators.csv` is the single city table: `city`, `province`, `POP`, `eta` and nine structural indicators. It contains all 407 cities, with missing indicators left blank. Figure 3 selects complete indicator records; Figure 4 uses the intersection of its sensitivity inputs.

## Run simulations and other calculations

### Reference-demand simulation

Simulate baseline η using the supplied travel-time matrices and population centres. The simulations run offline with inputs in `data/country/`. Use `--country-data PATH` to select a different input directory for simulations or indicator calculations.

```bash
python main_text/simulate.py baseline --province Guangdong --city dongguan --seed 42
```

Each city uses `<province>/<city>_center.shp` and its sidecars. The centre layer's `value` field is population. In `routes/`, `_routes_con.pkl` stores dense durations in seconds by origin and destination, and `_routes_sim.pkl` stores sparse connector durations by next origin and previous destination. Fitted gravity vectors store alpha, beta, gamma and K. The first three determine normalized demand probabilities.

Use `--help` to see the preprocessing and gravity-fitting options.

```bash
python main_text/preprocess.py --help
python main_text/fit_gravity.py --help
```

Gravity fitting takes an aggregate origin–destination CSV with `pop_o`, `pop_d`, `t` (seconds) and `flow`. Confidential taxi trajectories and individual observed-trip records are excluded.

### Structural indicators

Recompute the nine indicators from the supplied centre layers, clipped rasters (`<province>/<city>_clip.tif`), node coordinates (`routes/<city>_latlon.pkl`) and dense travel times.

```bash
python main_text/indicators.py
```

The command writes `indicators.csv` for the cities with clipped rasters. To draw Figure 3 from those results, run `python main_text/fig3.py --inputs PATH/indicators.csv`. The standard Figure 3 command uses the supplied indicator table.

### Sensitivity analyses

For gravity-model sensitivity, use `alpha`, `beta` or `gamma` with `main_text/simulate.py`. For departure-time sensitivity, use `mu` or `sigma` with `supplementary/simulate.py`.

```bash
python supplementary/simulate.py sigma --province Guangdong --city dongguan --seed 42
```

The vehicle-waiting experiment tests waiting limits of 0–8 min in 2-min steps. Figure 4g–h uses uniform departures over 60 min. SI Fig. 1c compares 60- and 120-min windows against each window's 2-min waiting reference.

```bash
python main_text/simulate.py vehicle_wait --all-cities
python supplementary/simulate.py vehicle_wait --all-cities --window-minutes 120
```

Both commands use the saved city list and seeds in `data/country/waiting_experiments.json`. For one city, replace `--all-cities` with `--province PROVINCE --city CITY`. Use `--seed` to generate new realizations.

## Sources and license

Please cite the accompanying manuscript and the applicable sources:

- [China Urban Built-Up Area Dataset for 2020](https://doi.org/10.11922/11-6035.csd.2021.0087.zh).
- [UN-adjusted WorldPop population data for 2020](https://doi.org/10.5258/SOTON/WP00671).
- [OpenRouteService](https://openrouteservice.org/) for inter-grid travel times.

Software attribution: **Urban Mobility Resource Efficiency contributors**. Original software is distributed under the [MIT License](LICENSE); third-party software and data retain their applicable terms.
