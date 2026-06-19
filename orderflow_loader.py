"""
Attach trade-level order flow (from tick_orderflow.py) to the kline frame.

Reads the compact per-minute order-flow parquets, derives signed delta and
cumulative volume delta (CVD), and aligns them onto an existing 1-minute
kline DataFrame so the detector can resample them like price. Minutes with
no trades get zero flow; CVD is carried forward (it is cumulative).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

OF_COLS = ["of_vol", "of_buy", "of_ntrades", "of_nbuy",
           "of_maxtrade", "of_buymax", "of_sellmax"]


def load_orderflow(of_dir: str = "data/btc_of") -> pd.DataFrame | None:
    files = sorted(glob.glob(os.path.join(of_dir, "*.parquet")))
    if not files:
        return None
    of = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    of = of.drop_duplicates(subset="ms").sort_values("ms")
    idx = pd.to_datetime(of["ms"].astype("int64"), unit="ms", utc=True)
    of = of.set_index(idx).drop(columns="ms")
    of.index.name = "time"
    return of


def attach_orderflow(df: pd.DataFrame,
                     of_dir: str = "data/btc_of") -> pd.DataFrame:
    """Return df with per-minute order-flow columns + derived delta/CVD.
    No-op (returns df unchanged) if no order-flow parquets are present."""
    of = load_orderflow(of_dir)
    if of is None:
        return df
    of = of.reindex(df.index)
    for c in OF_COLS:
        of[c] = of[c].fillna(0.0)
    of["of_sell"] = of["of_vol"] - of["of_buy"]
    of["of_delta"] = of["of_buy"] - of["of_sell"]          # signed volume
    of["of_cvd"] = of["of_delta"].cumsum()                 # running CVD
    out = df.copy()
    for c in list(OF_COLS) + ["of_sell", "of_delta", "of_cvd"]:
        out[c] = of[c].to_numpy()
    return out
