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
                           sp_start: float = 4000.0,
                           crypto: bool = False) -> pd.DataFrame:
    """Random-walk NAS plus a CORRELATED random-walk S&P. Always emits
    random `volume` and `taker_buy` so the noise null carries the same
    volume / order-flow features as real data (as noise) — otherwise the
    null would have fewer features and the comparison would be unfair.
    crypto=True makes it 24/7 (1440 bars/day, no overnight gap) to match
    crypto data structure."""
    rng = np.random.default_rng(seed)
    if crypto:
        days = pd.date_range(start_date, periods=n_days, freq="D", tz="UTC")
    else:
        days = pd.bdate_range(start_date, periods=n_days,
                              tz="America/New_York")
    frames = []
    price = start_price
    sp_price = sp_start
    for d in days:
        if crypto:
            idx = pd.date_range(d, d + pd.Timedelta(hours=23, minutes=59),
                                freq="1min", tz="UTC")
        else:
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
        # random volume + order flow: taker-buy ratio leans with the bar's
        # own move (realistic) but carries no FUTURE signal on a random walk
        volume = rng.lognormal(mean=8.0, sigma=0.5, size=n)
        bar_ret = (closes - opens) / np.maximum(vol, 1e-9)
        ratio = np.clip(0.5 + 0.15 * bar_ret + rng.normal(0, 0.1, n),
                        0.05, 0.95)
        taker_buy = volume * ratio
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
            "volume": volume, "taker_buy": taker_buy,
            "sp_h": sp_h, "sp_l": sp_l, "sp_c": sp_closes,
        }, index=idx))
        if not crypto:
            price = closes[-1] + rng.normal(0, price * 0.002)   # gap
            sp_price = sp_closes[-1] + rng.normal(0, sp_price * 0.002)
        else:
            price, sp_price = closes[-1], sp_closes[-1]          # no gap
    return pd.concat(frames)
