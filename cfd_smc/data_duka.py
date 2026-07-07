"""
Dukascopy CFD data loader.

Downloads tick-level bid/ask data for CFD index instruments directly from
Dukascopy's public datafeed, decodes the .bi5 (LZMA + packed struct) format,
and aggregates to OHLC bars that PRESERVE the real spread so the backtester
can charge honest transaction costs.

Instruments of interest:
    USA500IDXUSD   -> S&P 500 CFD
    USATECHIDXUSD  -> Nasdaq 100 CFD

Notes / gotchas baked in here:
  * Dukascopy URLs use ZERO-BASED months (Jan=00 ... Dec=11).
  * .bi5 = LZMA-compressed array of 20-byte records:
        >IIIff = (ms_since_hour, ask_int, bid_int, ask_vol, bid_vol)
  * Index prices are integers scaled by POINT (0.001 -> divide by 1000).
  * Files are per-UTC-hour. Missing/empty hours (weekends, daily break)
    simply have no data; we skip them.

Everything is cached to parquet so a container recycle can resume.
"""
from __future__ import annotations

import io
import lzma
import struct
import time
import datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

DATAFEED = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"

# Point scale (price integer -> real price). 0.001 for these index CFDs.
POINT = {
    "USA500IDXUSD": 1000.0,
    "USATECHIDXUSD": 1000.0,
}

TICK_STRUCT = struct.Struct(">IIIff")
CACHE = Path(__file__).parent / "data_cache"
CACHE.mkdir(exist_ok=True)

_session = None


def _sess() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (research backtest)"})
        _session = s
    return _session


def _fetch_hour(sym: str, when: dt.datetime, retries: int = 3) -> bytes | None:
    """Download one hourly .bi5 file. Returns raw bytes or None if absent."""
    url = DATAFEED.format(sym=sym, y=when.year, m=when.month - 1, d=when.day, h=when.hour)
    for attempt in range(retries):
        try:
            r = _sess().get(url, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r.content
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def _decode(raw: bytes, sym: str, hour_start: dt.datetime) -> pd.DataFrame | None:
    """Decode one hour of ticks into a DataFrame with mid, spread, volume."""
    if not raw:
        return None
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        return None
    n = len(data) // TICK_STRUCT.size
    if n == 0:
        return None
    scale = POINT[sym]
    ms = np.empty(n, dtype=np.int64)
    ask = np.empty(n, dtype=np.float64)
    bid = np.empty(n, dtype=np.float64)
    avol = np.empty(n, dtype=np.float64)
    bvol = np.empty(n, dtype=np.float64)
    off = 0
    for i in range(n):
        t, a, b, av, bv = TICK_STRUCT.unpack_from(data, off)
        ms[i] = t
        ask[i] = a
        bid[i] = b
        avol[i] = av
        bvol[i] = bv
        off += TICK_STRUCT.size
    ts = pd.to_datetime(hour_start, utc=True) + pd.to_timedelta(ms, unit="ms")
    ask /= scale
    bid /= scale
    df = pd.DataFrame(
        {
            "ask": ask,
            "bid": bid,
            "mid": (ask + bid) / 2.0,
            "spread": ask - bid,          # in price points
            "volume": avol + bvol,        # in millions of units (dukascopy)
        },
        index=ts,
    )
    return df


def _bars_from_ticks(ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate ticks to OHLC bars, preserving mean spread per bar."""
    g = ticks.resample(timeframe, label="left", closed="left")
    bars = pd.DataFrame(
        {
            "open": g["mid"].first(),
            "high": g["mid"].max(),
            "low": g["mid"].min(),
            "close": g["mid"].last(),
            "volume": g["volume"].sum(),
            "spread": g["spread"].mean(),   # avg quoted spread during the bar
            "n_ticks": g["mid"].count(),
        }
    )
    return bars.dropna(subset=["open"])


def load(
    sym: str,
    start: str,
    end: str,
    timeframe: str = "5min",
    max_workers: int = 24,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load bars for `sym` between `start` and `end` (YYYY-MM-DD, UTC).

    Caches per (sym, timeframe, month) parquet so it is resumable and cheap
    to re-run. Returns a single concatenated, de-duplicated, sorted frame.
    """
    start_d = pd.Timestamp(start, tz="UTC")
    end_d = pd.Timestamp(end, tz="UTC")
    months = pd.date_range(start_d.normalize().replace(day=1), end_d, freq="MS", tz="UTC")

    out = []
    for month_start in months:
        cache_f = CACHE / f"{sym}_{timeframe}_{month_start:%Y%m}.parquet"
        if cache_f.exists():
            out.append(pd.read_parquet(cache_f))
            if verbose:
                print(f"  [cache] {cache_f.name}")
            continue

        month_end = (month_start + pd.offsets.MonthBegin(1))
        hours = pd.date_range(month_start, month_end, freq="h", inclusive="left", tz="UTC")
        # index CFDs are closed on weekends; skip Sat and most of Sun/Fri edges is fine
        hours = [h for h in hours if h.weekday() < 5 or (h.weekday() == 6 and h.hour >= 22)]

        raws: dict[dt.datetime, bytes | None] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_fetch_hour, sym, h.to_pydatetime()): h for h in hours}
            for fut in as_completed(futs):
                raws[futs[fut]] = fut.result()

        frames = []
        for h in sorted(raws):
            df = _decode(raws[h], sym, h.to_pydatetime())
            if df is not None:
                frames.append(df)
        if not frames:
            if verbose:
                print(f"  [empty] {month_start:%Y-%m}")
            continue
        ticks = pd.concat(frames).sort_index()
        bars = _bars_from_ticks(ticks, timeframe)
        bars.to_parquet(cache_f)
        out.append(bars)
        if verbose:
            print(f"  [dl] {month_start:%Y-%m}  ticks={len(ticks):>7}  bars={len(bars):>5}")

    if not out:
        return pd.DataFrame()
    full = pd.concat(out).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    full = full.loc[start_d:end_d]
    return full


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "USA500IDXUSD"
    df = load(sym, "2023-06-01", "2023-06-08", timeframe="5min")
    print(df.head(10))
    print(df.tail(5))
    print("rows:", len(df), "avg spread pts:", round(df.spread.mean(), 4))
