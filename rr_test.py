"""
Test a 3:1 reward:risk target (TP at +3R) against the current 2:1, with and
without a late break-even. Same machinery as be_test: re-walk every setup
bar-by-bar, classify win/scratch/loss, report raw stats, then save trades so
be_sequence-style selection + sizing gives the bottom-line verdict.

Configs (label -> target_rr, be_trigger or None):
  fixed2   2R, no BE     (current system)
  fixed3   3R, no BE
  3_be1.5  3R, BE@+1.5R
  3_be2.0  3R, BE@+2.0R
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import backtest_ftmo as B
from smc_detector import detect_setups
from crypto_loader import load_binance_klines

os.environ["CRYPTO"] = "1"
os.environ["BASE_TF"] = "15min"
ASSETS = [("BTC", "data/btc", True), ("ETH", "data/eth", False),
          ("SOL", "data/sol", False), ("BNB", "data/bnb", False),
          ("LTC", "data/ltc", False)]
BUF = 1e-4
CONFIGS = {"fixed2": (2.0, None), "fixed3": (3.0, None),
           "3_be1.5": (3.0, 1.5), "3_be2.0": (3.0, 2.0)}


def walk(df, setups, target_rr, be_trig, min_hold=B.MIN_HOLD_MIN):
    a = B._arrays(df)
    rows = []
    for s in setups:
        d = s.direction
        fi = B._fill_index(a, s, True)
        if fi is None:
            continue
        fill, entry = fi
        risk = abs(entry - s.stop)
        if risk <= 0:
            continue
        tp = entry + d * target_rr * risk
        hi = B._same_day_end(a, fill, B.MAX_HOLD_MIN, True)
        stop = s.stop
        best = entry
        exit_px = outcome = None
        exit_i = hi - 1
        for k in range(fill + 1 + min_hold, hi):
            hk, lk = a["h"][k], a["l"][k]
            best = max(best, hk) if d == 1 else min(best, lk)
            prog = (best - entry) * d / risk
            if be_trig is not None and prog >= be_trig and \
                    ((d == 1 and stop < entry) or (d == -1 and stop > entry)):
                stop = entry + d * BUF * entry
            if d == 1:
                if lk <= stop:
                    exit_px, outcome, exit_i = stop, "stop", k; break
                if hk >= tp:
                    exit_px, outcome, exit_i = tp, "tp", k; break
            else:
                if hk >= stop:
                    exit_px, outcome, exit_i = stop, "stop", k; break
                if lk <= tp:
                    exit_px, outcome, exit_i = tp, "tp", k; break
        if outcome is None:
            exit_px, outcome = float(a["c"][hi - 1]), "maxhold"
        r = (exit_px - entry) * d / risk
        cls = "scratch" if abs(r) < 0.02 else ("win" if r > 0 else "loss")
        rows.append({**s.features, "entry_time": df.index[fill],
                     "exit_time": df.index[exit_i], "dir_": d,
                     "entry_px": entry, "exit_px": float(exit_px),
                     "risk_px": risk, "year": int(df.index[fill].year),
                     "realized_R": r, "win": int(r > 0), "cls": cls,
                     "outcome": "tp" if outcome == "tp"
                     else ("maxhold" if outcome == "maxhold" else "sl")})
    return pd.DataFrame(rows)


def summarise(df, label):
    w = (df["cls"] == "win").mean() * 100
    sc = (df["cls"] == "scratch").mean() * 100
    ls = (df["cls"] == "loss").mean() * 100
    avgw = df[df["realized_R"] > 0]["realized_R"].mean()
    avgl = df[df["realized_R"] < 0]["realized_R"].mean()
    exp = df["realized_R"].mean()
    tp = (df["outcome"] == "tp").mean() * 100
    print(f"{label:>9} | win {w:4.1f}% scratch {sc:4.1f}% loss {ls:4.1f}% "
          f"| TPhit {tp:4.1f}% | avgW {avgw:+.2f} avgL {avgl:+.2f} "
          f"| exp {exp:+.3f}R")


def main():
    os.makedirs("data/be", exist_ok=True)
    per = {c: [] for c in CONFIGS}
    for name, folder, of in ASSETS:
        os.environ["ORDERFLOW"] = "1" if of else "0"
        df = load_binance_klines(folder)
        setups = detect_setups(df)
        for c, (rr, bt) in CONFIGS.items():
            t = walk(df, setups, rr, bt).dropna(subset=["realized_R"])
            t["asset"] = name
            per[c].append(t)
        print(f"  {name}: {len(setups)} setups", flush=True)
    print("\n=== RR comparison (all detected setups) ===")
    for c in CONFIGS:
        allc = pd.concat(per[c])
        summarise(allc, c)
        allc.to_parquet(f"data/be/{c}.parquet")


if __name__ == "__main__":
    main()
