"""
Generate LIVE-outcome labels for every setup, so the model can be trained on
what actually happens live rather than the idealized backtest fill.

For each setup we rest a limit at the order block and replay the REAL fill
engine (live_stream.on_quote: far->retrace gate, real-price fills, BE-lock,
2R TP, 8h max-hold, spread cost) over its window -- recording whether it FILLED
on a genuine retrace within the TTL and, if so, the real R the trade made.
Independent of portfolio (fresh book per setup) so R is clean.

    CRYPTO=1 python label_live.py     -> data/trades/<asset>_live.parquet

Rerun this whenever the fill engine, costs, or detector features change; then
retrain with train_live_models.py. ~10 min for BTC/ETH/SOL.
"""
from __future__ import annotations

import os
os.environ.setdefault("CRYPTO", "1")
os.environ.setdefault("ORDERFLOW", "0")
os.environ.setdefault("BASE_TF", "15min")

import numpy as np
import pandas as pd

import backtest_ftmo as B
import goat_test as G
import live_stream as L
from smc_detector import detect_setups
from crypto_loader import load_binance_klines

ASSETS = {"BTC": "data/btc", "ETH": "data/eth", "SOL": "data/sol"}
TTL_MIN = L.LIMIT_TTL_MIN
MAXH = B.MAX_HOLD_MIN


def label_asset(df, btc, name, cost):
    if df.index.tz is not None:
        df = df.tz_localize(None)
    ts = df.index.astype("datetime64[ns]").astype("int64")
    o = df["mid_o"].values; h = df["mid_h"].values
    lo = df["mid_l"].values; c = df["mid_c"].values
    half = (cost[name][0] + cost[name][1]) / 2.0 / 1e4     # symmetric half-spread
    rows = []
    for s in detect_setups(df, xref=btc):
        risk = abs(s.entry_price - s.stop)
        if risk <= 0:
            continue
        d = s.direction
        arm = pd.Timestamp(s.entry_time)
        lim = dict(entry=float(s.entry_price), stop=float(s.stop),
                   tp=float(s.entry_price + d * 2.0 * risk),
                   side="buy" if d > 0 else "sell", risk=float(risk), prob=0.7,
                   expire=arm.timestamp() + TTL_MIN * 60)
        st = {"limits": [lim], "pos": None, "quote": None}
        port = {"open_notional": 0.0, "equity": L.INIT, "day": None,
                "day_start": L.INIT, "open_worst": 0.0, "peak_lev": 0.0}
        i0 = int(np.searchsorted(ts, arm.value, "left"))
        iend = int(np.searchsorted(ts, arm.value
                                   + int((TTL_MIN + MAXH + 5) * 60e9), "left"))
        filled = False; live_R = np.nan; outcome = "nofill"
        for i in range(i0, min(iend, len(ts))):
            tsec = ts[i] // 1_000_000_000
            if st["pos"] is None and not st["limits"]:
                break                                  # expired unfilled
            if st["pos"] is None:
                sd = st["limits"][0]["side"]
                first, second = (h[i], lo[i]) if sd == "buy" else (lo[i], h[i])
            else:
                dd = 1 if st["pos"]["side"] == "buy" else -1
                first, second = (lo[i], h[i]) if dd == 1 else (h[i], lo[i])
            done = False
            for px in (o[i], first, second, c[i]):
                px = float(px)
                for e in L.on_quote(name, px - px * half, px + px * half,
                                    st, tsec, port):
                    if e["type"] == "FILL":
                        filled = True
                    elif e["type"] == "EXIT":
                        live_R = e["R"]; outcome = e["outcome"]; done = True
                if done:
                    break
            if done:
                break
        rows.append({**s.features, "entry_time": arm, "year": int(arm.year),
                     "ob_entry": float(s.entry_price), "ob_stop": float(s.stop),
                     "ob_risk": float(risk), "ob_side": "buy" if d > 0 else "sell",
                     "live_fill": int(filled), "live_R": live_R, "outcome": outcome,
                     "live_win": int(filled and np.isfinite(live_R) and live_R > 0)})
    out = pd.DataFrame(rows)
    out.to_parquet(f"data/trades/{name}_live.parquet")
    fill = out.live_fill.mean() * 100
    wf = out[out.live_fill == 1].live_win.mean() * 100 if out.live_fill.any() else 0
    print(f"{name}: {len(out)} setups | fill {fill:.0f}% | "
          f"win-of-filled {wf:.0f}% -> data/trades/{name}_live.parquet", flush=True)


def main():
    btc = load_binance_klines("data/btc")
    if btc.index.tz is not None:
        btc = btc.tz_localize(None)
    for name, folder in ASSETS.items():
        df = load_binance_klines(folder)
        label_asset(df, btc, name, G.GOAT_TYPICAL)
    print("DONE")


if __name__ == "__main__":
    main()
