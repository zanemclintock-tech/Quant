"""
Trade-level (tick) order flow from Binance aggTrades -> compact 1-min bars.

Raw BTCUSDT aggTrades are ~0.3-1 GB *zipped* per month (multi-GB raw), so
we never keep them: download a month, stream-aggregate the trades into a
small per-minute order-flow parquet, delete the raw, move on. The result
is information klines do NOT carry -- real signed volume delta (CVD) and
the trade-SIZE distribution (largest aggressive prints = absorption /
big-player footprint).

aggTrades columns (no header):
  aggId, price, qty, firstId, lastId, transactTime, isBuyerMaker, isBest
isBuyerMaker == True  -> the buyer was the maker -> the AGGRESSOR was a
seller. So taker-buy volume is the qty where isBuyerMaker is False.

    python tick_orderflow.py            # all months -> data/btc_of/
"""
from __future__ import annotations

import os
import subprocess
import sys

import polars as pl

BASE = "https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT"
OUT = "data/btc_of"
COLS = ["aggId", "price", "qty", "firstId", "lastId", "ts",
        "isBuyerMaker", "isBest"]


def months(start=(2020, 8), end=(2026, 5)):
    y, m = start
    while (y, m) <= end:
        yield f"{y}-{m:02d}"
        m += 1
        if m > 12:
            y, m = y + 1, 1


def aggregate(csv_path: str) -> pl.DataFrame:
    # detect ms vs microseconds (Binance switched to micros in 2025)
    with open(csv_path, "r") as f:
        first_ts = int(f.readline().split(",")[5])
    per_min = 60_000_000 if first_ts > 1e15 else 60_000
    lf = pl.scan_csv(
        csv_path, has_header=False, new_columns=COLS,
        schema_overrides={"price": pl.Float64, "qty": pl.Float64,
                          "ts": pl.Int64, "isBuyerMaker": pl.Boolean})
    is_buy = ~pl.col("isBuyerMaker")                # aggressor was the buyer
    buy_q = pl.when(is_buy).then(pl.col("qty")).otherwise(0.0)
    sell_q = pl.when(~is_buy).then(pl.col("qty")).otherwise(0.0)
    out = (lf.with_columns((pl.col("ts") // per_min * 60_000).alias("ms"))
           .group_by("ms").agg([
               pl.col("qty").sum().alias("of_vol"),
               buy_q.sum().alias("of_buy"),
               pl.len().alias("of_ntrades"),
               is_buy.sum().alias("of_nbuy"),
               pl.col("qty").max().alias("of_maxtrade"),
               buy_q.max().alias("of_buymax"),
               sell_q.max().alias("of_sellmax"),
           ]).sort("ms")).collect(engine="streaming")
    return out


def run_month(ym: str) -> str:
    dst = os.path.join(OUT, f"BTCUSDT-of-{ym}.parquet")
    if os.path.exists(dst):
        return "skip"
    url = f"{BASE}/BTCUSDT-aggTrades-{ym}.zip"
    zp, cp = f"/tmp/at-{ym}.zip", None
    code = subprocess.run(
        ["curl", "-s", "-o", zp, "-w", "%{http_code}", url],
        capture_output=True, text=True).stdout.strip()
    if code != "200":
        if os.path.exists(zp):
            os.remove(zp)
        return f"miss({code})"
    try:
        subprocess.run(["unzip", "-o", "-q", zp, "-d", "/tmp"], check=True)
        cp = f"/tmp/BTCUSDT-aggTrades-{ym}.csv"
        df = aggregate(cp)
        df.write_parquet(dst)
        return f"ok {df.height} min"
    finally:
        for p in (zp, cp):
            if p and os.path.exists(p):
                os.remove(p)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ok = miss = 0
    for ym in months():
        status = run_month(ym)
        print(f"{ym}: {status}", flush=True)
        if status.startswith("ok") or status == "skip":
            ok += 1
        else:
            miss += 1
    print(f"DONE ok/skip={ok} miss={miss}")


if __name__ == "__main__":
    main()
