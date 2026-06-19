"""
Attach futures L2 book depth (from l2_orderbook.py) to the kline frame and
derive resting-liquidity imbalance. Only 2023+ has data; earlier minutes
get NaN (the model tolerates it). Imbalance > 0 => more resting bids
(support); < 0 => more resting asks (resistance).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

L2_RAW = ["l2_bid", "l2_ask", "l2_bid1", "l2_ask1"]


def load_l2(d="data/btc_l2"):
    files = sorted(glob.glob(os.path.join(d, "*.parquet")))
    if not files:
        return None
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df = df.drop_duplicates(subset="ms").sort_values("ms")
    idx = pd.to_datetime(df["ms"].astype("int64"), unit="ms", utc=True)
    return df.set_index(idx).drop(columns="ms")


def attach_l2(df, d="data/btc_l2"):
    l2 = load_l2(d)
    if l2 is None:
        return df
    l2 = l2.reindex(df.index)
    out = df.copy()
    bid, ask = l2["l2_bid"], l2["l2_ask"]
    b1, a1 = l2["l2_bid1"], l2["l2_ask1"]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["l2_imb"] = ((bid - ask) / (bid + ask)).to_numpy()
        out["l2_imb1"] = ((b1 - a1) / (b1 + a1)).to_numpy()
    out["l2_depth"] = (bid + ask).to_numpy()
    return out
