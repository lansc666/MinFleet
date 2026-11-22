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

Alternatively, run `python -m pip install -r requirements.txt` and install [Poppler](https://poppler.freedesktop.org/) separately. Figure 2 requires `pdftoppm` on `PATH` or its full path in `PDFTOPPM`. Arial reproduces the figure typography; SI commands accept `--font-file PATH` or `--allow-font-fallback`.

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
| Main Fig. 4 | Sensitivity summaries and rank correlations for 406 cities, including vehicle waiting with a 60-min departure window |
| SI Fig. 1 | Departure-time sensitivity correlations for 164 cities (a, b) and a comparison of 60- and 120-min departure windows for vehicle waiting across 406 cities (c) |
| SI Fig. 2 | Population maps and spatial sensitivity for three example cities |

Results go to `results/main_text/` or `results/supplementary/`. `--output PATH` selects an empty output directory; `--dpi` controls image resolution (default: 600). Use `fig4 --stats-only` for statistics or `fig3 --analysis-only` to skip drawing. UMAP uses the nine standardized indicators, excluding eta, with `n_neighbors=40`, `min_dist=0.3` and seed 42.

`data/city_indicators.csv` is the single city table: `city`, `province`, `POP`, `eta` and nine structural indicators. It contains all 407 cities, with missing indicators left blank. Figure 3 selects complete indicator records; Figure 4 uses the intersection of its sensitivity inputs.

## Run simulations and other calculations

Saved matrices and population centres are included for the simulation inputs. No ORS connection or API key is required.

```bash
python main_text/simulate.py baseline --province Guangdong --city dongguan --seed 42
python supplementary/simulate.py sigma --province Guangdong --city dongguan --seed 42
```

Main-text choices: `baseline`, `alpha`, `beta`, `gamma`, `vehicle_wait`. SI choices: `mu`, `sigma`, `vehicle_wait`. Use `--country-data PATH` to override `data/country/`.

The vehicle-waiting experiment uses punctual pickups and tests advance arrival limits of 0, 2, 4, 6 and 8 minutes. A trip can connect only to one with a strictly later scheduled pickup. Figure 4g-h uses uniform departures within 60 minutes. Supplementary Figure 1c compares 60- and 120-minute departure windows using the same 406 cities, with each curve compared to its own 2-minute setting. The two windows share the order counts and eight seeds per setting and city in `data/country/waiting_experiments.json`.

Run the following commands to regenerate the two sets of waiting arrays. The default departure window is 60 minutes. Omit `--all-cities` and provide `--province` and `--city` for one city. Saved seeds are used by default. Use `--seed` to generate a new set of realizations.

```bash
python main_text/simulate.py vehicle_wait --all-cities
python supplementary/simulate.py vehicle_wait --all-cities --window-minutes 120
```

Each city uses `<province>/<city>_center.shp` and its sidecars, plus `routes/<city>_routes_con.pkl`, `_routes_sim.pkl` and `_latlon.pkl`. The centre layer's `value` field is population. Dense durations are in seconds, indexed by origin then destination; sparse connectors use next origin then previous destination. Fitted gravity vectors store alpha, beta, gamma and K; the first three determine normalized demand probabilities.

Additional entry points provide input options through `--help`:

```bash
python main_text/indicators.py --help
python main_text/preprocess.py --help
python main_text/fit_gravity.py --help
```

To recompute the nine indicators, run `python main_text/indicators.py`; `--country-data PATH` overrides the supplied centre layers, `<province>/<city>_clip.tif`, node coordinates and dense durations. It processes cities with supplied clipped rasters and writes `indicators.csv`, which Figure 3 can read with `python main_text/fig3.py --inputs PATH/indicators.csv`. The normal Figure 3 command reads the completed city table without recomputing indicators. Gravity fitting takes an aggregate OD CSV with `pop_o`, `pop_d`, `t` (seconds) and `flow`. Confidential taxi trajectories and individual observed-trip records are excluded.

## Sources and license

Please cite the accompanying manuscript and the applicable sources:

- [China Urban Built-Up Area Dataset for 2020](https://doi.org/10.11922/11-6035.csd.2021.0087.zh).
- [UN-adjusted WorldPop population data for 2020](https://doi.org/10.5258/SOTON/WP00671).
- [OpenRouteService](https://openrouteservice.org/) for inter-grid travel times.

Software attribution: **Urban Mobility Resource Efficiency contributors**. Original software is distributed under the [MIT License](LICENSE); third-party software and data retain their applicable terms.
