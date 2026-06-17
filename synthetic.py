"""
Synthetic 1-minute NAS100-like bid/ask data.

Used by the test suite and `run_backtest.py --synthetic` to verify the
ENGINE (causality, fills, risk accounting, drawdown guards). A random
walk has no exploitable edge, so P&L on synthetic data says nothing
about the strategy — only about the plumbing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_minutes(n_days: int = 60, seed: int = 0,
                           start_price: float = 15000.0,
                           spread: float = 1.2,
                           drift_per_min: float = 0.0,
                           start_date: str = "2021-01-04",
                           sigma_frac: float = 0.00035,
                           corr: float = 0.9,
                           sp_start: float = 4000.0) -> pd.DataFrame:
    """Random-walk NAS plus a CORRELATED random-walk S&P (corr on the
    per-minute innovations). Used as the honest null for the SMT
    divergence test: the two move together with no predictable structure,
    so divergence happens only by chance and carries no edge."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start_date, periods=n_days, tz="America/New_York")
    frames = []
    price = start_price
    sp_price = sp_start
    for d in days:
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=30),
                            d + pd.Timedelta(hours=16), freq="1min",
                            tz="America/New_York")
        n = len(idx)
        vol = price * sigma_frac * rng.uniform(0.6, 1.6)   # per-min sigma
        z1 = rng.normal(0, 1, n)
        z2 = corr * z1 + np.sqrt(1 - corr ** 2) * rng.normal(0, 1, n)
        steps = drift_per_min + vol * z1
        closes = price + np.cumsum(steps)
        opens = np.concatenate([[price], closes[:-1]])
        wick = np.abs(rng.normal(0, vol * 0.6, n))
        highs = np.maximum(opens, closes) + wick
        lows = np.minimum(opens, closes) - wick
        half = spread / 2.0
        sp_vol = sp_price * sigma_frac * rng.uniform(0.6, 1.6)
        sp_closes = sp_price + np.cumsum(sp_vol * z2)
        sp_opens = np.concatenate([[sp_price], sp_closes[:-1]])
        sp_wick = np.abs(rng.normal(0, sp_vol * 0.6, n))
        sp_h = np.maximum(sp_opens, sp_closes) + sp_wick
        sp_l = np.minimum(sp_opens, sp_closes) - sp_wick
        frames.append(pd.DataFrame({
            "bid_o": opens - half, "bid_h": highs - half,
            "bid_l": lows - half, "bid_c": closes - half,
            "ask_o": opens + half, "ask_h": highs + half,
            "ask_l": lows + half, "ask_c": closes + half,
            "mid_o": opens, "mid_h": highs, "mid_l": lows, "mid_c": closes,
            "sp_h": sp_h, "sp_l": sp_l, "sp_c": sp_closes,
        }, index=idx))
        price = closes[-1] + rng.normal(0, price * 0.002)   # overnight gap
        sp_price = sp_closes[-1] + rng.normal(0, sp_price * 0.002)
    return pd.concat(frames)
