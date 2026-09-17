"""Dataset loaders. Files are expected in this folder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
YT_FILE = "yellow_tripdata_2024-01.parquet"
CENSUS_UCI_ID = 116      # US Census Data (1990) on the UCI repository

# Default k for the k-means experiment. Covertype has a class variable
# with 7 cover types; Census carries no label, so its k is a choice.
DEFAULT_K = {"covertype": 7, "census": 10}


def load_yt(path: str | Path = DATA_DIR / YT_FILE) -> np.ndarray:
    """NYC taxi trips: all numeric columns, rows with NaNs dropped.
    """
    df = pd.read_parquet(path).select_dtypes(include=np.number).dropna()
    return df.to_numpy(dtype=np.float64)

# ----------------------------------------------------------------------
# Datasets for the k-means experiment
# ----------------------------------------------------------------------
def _subsample(coords: np.ndarray, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points >= len(coords):
        return coords
    idx = np.random.default_rng(0).choice(len(coords), max_points, replace=False)
    return coords[np.sort(idx)]


def load_covertype(max_points: int | None = None) -> np.ndarray:
    path = DATA_DIR / "covertype_coords.npy"
    if not path.exists():
        from sklearn.datasets import fetch_covtype

        np.save(path, np.asarray(fetch_covtype().data, dtype=np.float64))
    return _subsample(np.load(path), max_points)


def _drop_non_feature_columns(df):
    dropped = {"id_like": [], "constant": [], "unique_per_row": []}

    id_names = ("caseid", "case_id", "id", "rownum", "index")
    dropped["id_like"] = [c for c in df.columns if str(c).lower() in id_names]
    df = df.drop(columns=dropped["id_like"])

    counts = df.nunique()
    dropped["constant"] = [str(c) for c in counts[counts <= 1].index]
    df = df.drop(columns=counts[counts <= 1].index)
    if len(df) >= 1000:
        counts = df.nunique()
        unique = counts[counts == len(df)].index
        dropped["unique_per_row"] = [str(c) for c in unique]
        df = df.drop(columns=unique)

    return df, dropped


def load_census(max_points: int | None = None) -> np.ndarray:
    cache = DATA_DIR / "census_coords.npy"
    if not cache.exists():
        from ucimlrepo import fetch_ucirepo

        df, dropped = _drop_non_feature_columns(
            fetch_ucirepo(id=CENSUS_UCI_ID).data.features
        )
        for reason, columns in dropped.items():
            if columns:
                print(f"load_census: dropped {reason}: {columns}")
        np.save(cache, df.to_numpy(dtype=np.float64))
    return _subsample(np.load(cache), max_points)



LOADERS = {"covertype": load_covertype, "census": load_census}
