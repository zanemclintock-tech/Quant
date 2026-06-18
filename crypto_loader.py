"""
Load free Binance 1-minute klines into the harness frame.

Binance monthly kline CSVs (free, no API key, from
https://data.binance.vision/) carry information that retail index/FX OHLC
does NOT: real traded volume AND taker-buy volume — i.e. how much of each
bar was aggressive BUYING vs selling. That taker-buy ratio is a free
order-flow imbalance signal, and attacking the data wall with it is the
whole point of moving to crypto.

Kline columns (12, header optional):
  open_time, open, high, low, close, volume, close_time,
  quote_volume, count, taker_buy_base, taker_buy_quote, ignore

Crypto trades 24/7, so there is no session; we index in UTC. klines are a
single price (no quotes), so we synthesise a tight bid/ask from a small
spread for honest fill/cost accounting.
"""
from __future__ import annotations

import glob as _glob

import numpy as np
import pandas as pd

_COLS = ["open_time", "open", "high", "low", "close", "volume",
         "close_time", "quote_volume", "count", "taker_buy_base",
         "taker_buy_quote", "ignore"]


def load_binance_klines(path_glob: str, spread_bps: float = 1.0) -> pd.DataFrame:
    """path_glob: a file, directory, or glob of Binance 1m kline CSVs.
    spread_bps: assumed half-spread*2 in basis points for fill/cost
    (BTC/ETH majors are ~0.5-2 bps; widen for alts)."""
    files = _resolve(path_glob)
    if not files:
        raise FileNotFoundError(f"No kline CSVs match {path_glob!r}")
    frames = []
    for f in files:
        head = pd.read_csv(f, nrows=1, header=None).iloc[0, 0]
        header = 0 if str(head).lower().startswith("open_time") else None
        d = pd.read_csv(f, header=header)
        d.columns = _COLS[:d.shape[1]]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset="open_time").sort_values("open_time")

    ot = df["open_time"].astype("int64")
    # Binance switched open_time from milliseconds to MICROSECONDS in
    # Jan 2025, so a multi-year pull mixes both units in one concat. Detect
    # per-row (ms ~1.7e12, us ~1.7e15) and normalise everything to ms;
    # a single global unit silently throws the ms-era files back to 1970.
    ot = ot.where(ot < 1e14, ot // 1000)
    idx = pd.to_datetime(ot, unit="ms", utc=True)
    out = pd.DataFrame(index=idx)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        out[c] = df[c].to_numpy(dtype=float)

    out = out.rename(columns={"open": "mid_o", "high": "mid_h",
                              "low": "mid_l", "close": "mid_c",
                              "taker_buy_base": "taker_buy"})
    out = out[(out["mid_h"] >= out["mid_l"]) & (out["mid_l"] > 0)]
    half = out["mid_c"] * (spread_bps / 1e4) / 2.0
    for col in ("o", "h", "l", "c"):
        out[f"bid_{col}"] = out[f"mid_{col}"] - half
        out[f"ask_{col}"] = out[f"mid_{col}"] + half
    out.index.name = "time"
    # ORDERFLOW=1 attaches trade-level (tick) order flow if the per-minute
    # parquets from tick_orderflow.py are present.
    import os
    if os.environ.get("ORDERFLOW", "0") not in ("0", "", "false", "no"):
        from orderflow_loader import attach_orderflow
        out = attach_orderflow(out, os.environ.get("OF_DIR", "data/btc_of"))
    return out


def _resolve(path_glob: str) -> list[str]:
    import os
    if os.path.isdir(path_glob):
        return sorted(_glob.glob(os.path.join(path_glob, "*.csv")))
    files = sorted(_glob.glob(path_glob))
    return files if files else ([path_glob] if os.path.exists(path_glob) else [])
