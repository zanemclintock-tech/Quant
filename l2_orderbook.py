"""
Futures L2 book depth (BTCUSDT, USD-M) -> compact per-minute resting
liquidity. Binance publishes daily bookDepth snapshots (~2/min) of resting
notional at +/-1..5% from mid -- i.e. the LIMIT orders sitting in the book
that aggTrades (taker flow) cannot see. Available 2023-01 onward only.

Per minute we keep bid/ask resting notional (all bands and the near +/-1%
band) so the loader can derive book imbalance -- who is resting more size,
support vs resistance, the other half of "who controls the market".

    python l2_orderbook.py            # 2023-01-01.. -> data/btc_l2/
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess

import polars as pl

BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT"
OUT = "data/btc_l2"
COLS = ["timestamp", "percentage", "depth", "notional"]


def days(start=(2023, 1, 1), end=(2026, 5, 31)):
    d, last = dt.date(*start), dt.date(*end)
    while d <= last:
        yield d.isoformat()
        d += dt.timedelta(days=1)


def aggregate(csv: str) -> pl.DataFrame:
    with open(csv) as f:
        hh = f.readline().startswith("timestamp")
    lf = pl.scan_csv(csv, has_header=hh,
                     new_columns=None if hh else COLS,
                     schema_overrides={"percentage": pl.Float64,
                                       "depth": pl.Float64,
                                       "notional": pl.Float64})
    ts = pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S",
                                          strict=False)
    neg = pl.col("percentage") < 0
    notl = pl.col("notional")
    out = (lf.with_columns(ts.dt.truncate("1m").alias("m"))
           .group_by("m").agg([
               pl.when(neg).then(notl).otherwise(0.0).sum().alias("l2_bid"),
               pl.when(~neg).then(notl).otherwise(0.0).sum().alias("l2_ask"),
               pl.when(pl.col("percentage") == -1).then(notl).otherwise(0.0)
                 .sum().alias("l2_bid1"),
               pl.when(pl.col("percentage") == 1).then(notl).otherwise(0.0)
                 .sum().alias("l2_ask1"),
           ]).sort("m")
           .with_columns(pl.col("m").dt.epoch("ms").alias("ms"))
           .drop("m"))
    return out.collect(engine="streaming")


def run_day(day: str) -> str:
    dst = os.path.join(OUT, f"BTCUSDT-l2-{day}.parquet")
    if os.path.exists(dst):
        return "skip"
    url = f"{BASE}/BTCUSDT-bookDepth-{day}.zip"
    zp = f"/tmp/l2-{day}.zip"
    cp = None
    code = subprocess.run(["curl", "-s", "-o", zp, "-w", "%{http_code}", url],
                          capture_output=True, text=True).stdout.strip()
    if code != "200":
        if os.path.exists(zp):
            os.remove(zp)
        return f"miss({code})"
    try:
        subprocess.run(["unzip", "-o", "-q", zp, "-d", "/tmp"], check=True)
        cp = f"/tmp/BTCUSDT-bookDepth-{day}.csv"
        aggregate(cp).write_parquet(dst)
        return "ok"
    finally:
        for p in (zp, cp):
            if p and os.path.exists(p):
                os.remove(p)


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = miss = 0
    for day in days():
        try:
            s = run_day(day)
        except Exception as e:                       # one bad file != halt
            s = f"err({type(e).__name__})"
        if s in ("ok", "skip"):
            ok += 1
        else:
            miss += 1
            print(f"{day}: {s}", flush=True)
    print(f"DONE ok/skip={ok} miss={miss} files={len(os.listdir(OUT))}")


if __name__ == "__main__":
    main()
