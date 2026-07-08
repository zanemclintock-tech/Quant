"""
Download real Dukascopy CFD data for NAS100 + S&P500 at 1-min, session hours,
2021-2025. Caches one parquet per (instrument, month) so it is resumable.

Session window: 12:00-21:00 UTC covers London PM + the full NY cash session
(NY 08:00-16:00 = 13:00-21:00 UTC, incl. the 10:00 NY "order block" = 14:00 UTC).
Downloading only these hours cuts the tick pull ~2.5x vs full 24h.

    python cfd_smc/download_cfd.py
"""
from __future__ import annotations

import sys
import datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data_duka as D

INSTRUMENTS = {"SP500": "USA500IDXUSD", "NAS100": "USATECHIDXUSD"}
SESSION_HOURS = range(12, 21)          # 12:00..20:00 UTC inclusive of 20h file
START, END = "2021-01-01", "2025-12-31"
WORKERS = 6                            # gentle: avoid Dukascopy rate-limit block
CACHE = Path(__file__).parent / "data_cache"
CACHE.mkdir(exist_ok=True)


def _fetch_fast(sym: str, when, timeout: float = 10.0):
    """Short-timeout fetch so a rate-limit block fails fast instead of hanging
    ~100s/file. Months with no data simply aren't cached (resumable)."""
    url = D.DATAFEED.format(sym=sym, y=when.year, m=when.month - 1,
                            d=when.day, h=when.hour)
    try:
        r = D._sess().get(url, timeout=timeout)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None


def month_file(name: str, sym: str, month_start: pd.Timestamp) -> Path:
    return CACHE / f"{name}_1min_{month_start:%Y%m}.parquet"


def download_instrument(name: str, sym: str):
    start_d = pd.Timestamp(START, tz="UTC")
    end_d = pd.Timestamp(END, tz="UTC")
    months = pd.date_range(start_d.normalize().replace(day=1), end_d, freq="MS", tz="UTC")
    for month_start in months:
        f = month_file(name, sym, month_start)
        if f.exists():
            print(f"  [cache] {f.name}", flush=True)
            continue
        month_end = month_start + pd.offsets.MonthBegin(1)
        hours = pd.date_range(month_start, month_end, freq="h", inclusive="left", tz="UTC")
        hours = [h for h in hours if h.weekday() < 5 and h.hour in SESSION_HOURS]
        raws = {}
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(_fetch_fast, sym, h.to_pydatetime()): h for h in hours}
            for fut in as_completed(futs):
                raws[futs[fut]] = fut.result()
        frames = []
        for h in sorted(raws):
            d = D._decode(raws[h], sym, h.to_pydatetime())
            if d is not None:
                frames.append(d)
        if not frames:
            print(f"  [empty] {name} {month_start:%Y-%m}", flush=True)
            continue
        ticks = pd.concat(frames).sort_index()
        bars = D._bars_from_ticks(ticks, "1min")
        bars.to_parquet(f)
        print(f"  [dl] {name} {month_start:%Y-%m}  ticks={len(ticks):>7}  bars={len(bars):>5}", flush=True)


def main():
    for name, sym in INSTRUMENTS.items():
        print(f"=== {name} ({sym}) ===", flush=True)
        download_instrument(name, sym)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
