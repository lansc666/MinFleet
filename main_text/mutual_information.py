"""Population-ordered sliding-window mutual information."""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.feature_selection import mutual_info_regression

def sliding(data, features):
    data = data.sort_values("POP", kind="stable").reset_index(drop=True)
    n = len(data)
    width = int(n * 0.3)
    step = max(1, int(n * 0.05))
    records = []
    for start in range(0, n - width + 1, step):
        w = data.iloc[start : start + width]
        row = {
            "start_index": start,
            "end_index_exclusive": start + width,
            "n_cities": width,
            "population_rank_percent": 100 * (start + width / 2) / n,
            "window_POP_min": float(w.POP.min()),
            "window_POP_max": float(w.POP.max()),
        }
        for f in features:
            row[f] = float(mutual_info_regression(w[[f]], w.eta, random_state=42)[0])
        records.append(row)
    raw = pd.DataFrame(records)
    smooth = raw.copy()
    for f in features:
        smooth[f] = np.clip(gaussian_filter1d(raw[f].to_numpy(), sigma=1.5), 0, None)
    return raw, smooth
