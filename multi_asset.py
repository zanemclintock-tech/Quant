"""
Multi-asset test: run the Stage-4 strategy per instrument, then pool the
trades into ONE FTMO account (the hedge-fund move -- many simultaneous
bets share the drawdown budget).

Portfolio rules: no two open positions in the SAME asset (no stacking /
no same-asset hedge), up to MAX_CONCURRENT positions across DIFFERENT
assets at once, 0.25% risk each, judged on pooled equity against the 6%
trailing / 3% daily limits.

    ORDERFLOW=1 CRYPTO=1 python multi_asset.py
"""
from __future__ import annotations

import heapq
import os

import numpy as np
import pandas as pd

import backtest_ftmo as B
from crypto_loader import load_binance_klines
from smc_detector import detect_setups

ASSETS = [("BTC", "data/btc", True), ("ETH", "data/eth", False),
          ("SOL", "data/sol", False), ("BNB", "data/bnb", False)]
MONTHS = 29.0
MAX_CONCURRENT = 4


def prep(folder, tf, orderflow):
    os.environ["ORDERFLOW"] = "1" if orderflow else "0"
    os.environ["BASE_TF"] = tf
    df = load_binance_klines(folder)
    data = B.gen_trades(df, detect_setups(df)).dropna(subset=["realized_R"])
    feats = [c for c in data.columns
             if c not in B.NON_FEATURES and data[c].notna().any()]
    is_ = data[data["year"] <= B.IS_END]
    oos = data[data["year"] >= B.OOS_START]
    if len(is_) < 200 or len(oos) < 60 or is_["win"].nunique() < 2:
        return None
    model, kept, thr = B.select_and_train(data, feats)
    p = model.predict_proba(oos[kept])[:, 1]
    sel = oos[p >= thr].sort_values("entry_time").copy()
    return sel[["entry_time", "exit_time", "realized_R", "entry_px",
               "exit_px", "risk_px", "outcome"]]


def portfolio(trades, risk_frac, spec, max_concurrent=MAX_CONCURRENT):
    """Event-driven pooled account. trades: one frame, 'asset' column,
    sorted by entry_time."""
    eq = peak = B.INIT
    floor = B.INIT - B.MAX_DD * B.INIT
    day, day_open = None, B.INIT
    open_heap = []                       # (exit_time, pnl, asset)
    open_assets = set()
    curve, taken, dviol, worst_daily, blown = [], 0, 0, 0.0, False

    def realize(ex_time, pnl, asset):
        nonlocal eq, peak, floor, day, day_open, dviol, worst_daily, blown
        d = ex_time.normalize()
        if day != d:
            day, day_open = d, eq
        eq += pnl
        peak = max(peak, eq)
        floor = min(peak - B.MAX_DD * B.INIT, B.INIT)
        dd = (day_open - eq) / day_open
        worst_daily = max(worst_daily, dd)
        if dd > B.DAILY_DD:
            dviol += 1
        if eq <= floor:
            blown = True
        curve.append((ex_time, eq))

    for _, t in trades.iterrows():
        while open_heap and open_heap[0][0] <= t["entry_time"]:
            ex, pnl, asset = heapq.heappop(open_heap)
            open_assets.discard(asset)
            realize(ex, pnl, asset)
            if blown:
                break
        if blown:
            break
        if t["asset"] in open_assets or len(open_heap) >= max_concurrent:
            continue                      # no same-asset overlap / cap
        cost_r = B._cost_r(t, spec)
        pnl = (t["realized_R"] - cost_r) * risk_frac * B.INIT
        heapq.heappush(open_heap, (t["exit_time"], pnl, t["asset"]))
        open_assets.add(t["asset"])
        taken += 1
    while open_heap and not blown:
        ex, pnl, asset = heapq.heappop(open_heap)
        realize(ex, pnl, asset)
    cur = pd.Series({c[0]: c[1] for c in curve})
    maxdd = ((cur.cummax() - cur) / cur.cummax()).max() if len(cur) else 0.0
    ret = eq / B.INIT - 1
    mo = ((1 + ret) ** (1 / MONTHS) - 1) if ret > -1 else float("nan")
    return {"ret": ret, "mo": mo, "maxdd": float(maxdd), "n": taken,
            "dviol": dviol, "blown": blown, "curve": cur,
            "passed": (not blown) and dviol == 0 and ret > 0}


def main():
    spec = B.EXEC["realistic"]
    for tf in ("15min", "60min"):
        print(f"\n================  {tf}  ================")
        sels = {}
        print(f"{'asset':>5} {'n_sel':>6} {'standalone 0.25% (ret/mo/DD/pass)':>40}")
        for name, folder, of in ASSETS:
            s = prep(folder, tf, of)
            if s is None or len(s) < 30:
                print(f"{name:>5}  (insufficient)"); continue
            s["asset"] = name
            sels[name] = s
            B.RISK_FRAC = 0.0025
            r = B.simulate(s.sort_values("entry_time"), spec)
            mo = ((1 + r["ret"]) ** (1 / MONTHS) - 1) * 100
            print(f"{name:>5} {len(s):>6}   {r['ret']*100:>+7.1f}% "
                  f"{mo:>+5.2f}%/mo  DD{r['maxdd']*100:>5.2f}%  "
                  f"{'PASS' if r['passed'] else 'no'}")
        if len(sels) < 2:
            continue
        allt = pd.concat(sels.values()).sort_values("entry_time")
        print(f"\n  PORTFOLIO ({'+'.join(sels)}, <= {MAX_CONCURRENT} concurrent):")
        print(f"  {'risk%':>6} {'ret%':>7} {'monthly%':>8} {'maxDD%':>7} "
              f"{'trades':>6} {'dViol':>5} {'PASS':>4}")
        for rf in (0.0025, 0.005, 0.0075):
            r = portfolio(allt, rf, spec)
            print(f"  {rf*100:>6.2f} {r['ret']*100:>+7.1f} {r['mo']*100:>+8.2f} "
                  f"{r['maxdd']*100:>7.2f} {r['n']:>6} {r['dviol']:>5} "
                  f"{'YES' if r['passed'] else 'no':>4}")


if __name__ == "__main__":
    main()
