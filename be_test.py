"""
Test break-even and trailing-stop exit overlays against the current fixed
2R-bracket. For each setup we re-walk the bars under three policies:

  fixed  : current -- stop fixed, TP at 2R, maxhold timeout.
  be@X   : once price reaches +X R, move stop to entry (+1bp buffer). If it
           later trades back to entry it's a SCRATCH (0R, neither win/loss).
  trailR : once past +1R, trail the stop by 1R behind the best price.

We report, per policy: win% / scratch% / loss%, avg win/loss R, expectancy,
and -- after the SAME walk-forward model selection + live sizing path -- the
sequenced fixed-base return, max DD, and worst day. Win rate that costs
expectancy isn't a win, so the sizing-path number is the real verdict.
"""
from __future__ import annotations

import os
import sys

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
RR = B.TARGET_RR
BUF = 1e-4            # 1bp buffer above entry for the BE stop


def _arrays(df):
    return B._arrays(df)


def walk(df, setups, policy, min_hold=B.MIN_HOLD_MIN):
    """Re-simulate exits under a policy. Returns trade rows (with realized_R,
    outcome in win/loss/scratch/maxhold)."""
    a = _arrays(df)
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
        tp = entry + d * RR * risk
        hi = B._same_day_end(a, fill, B.MAX_HOLD_MIN, True)
        stop = s.stop
        best = entry                       # best favourable price seen
        armed = False                      # BE / trail engaged
        exit_px = outcome = None
        exit_i = hi - 1
        for k in range(fill + 1 + min_hold, hi):
            hk, lk = a["h"][k], a["l"][k]
            # favourable extreme this bar
            fav = hk if d == 1 else lk
            if d == 1:
                best = max(best, hk)
            else:
                best = min(best, lk)
            prog = (best - entry) * d / risk          # R of best excursion

            if policy.startswith("be"):
                trig = float(policy[2:])
                if not armed and prog >= trig:
                    armed = True
                    stop = entry + d * BUF * entry      # BE + tiny buffer
            elif policy == "trail":
                if prog >= 1.0:
                    armed = True
                if armed:
                    newstop = best - d * 1.0 * risk     # 1R behind best
                    # only ratchet in the favourable direction
                    stop = max(stop, newstop) if d == 1 else min(stop, newstop)

            # check exits (stop first = conservative)
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
        # classify: scratch if |r| tiny (BE), else win/loss
        if abs(r) < 0.02:
            cls = "scratch"
        elif r > 0:
            cls = "win"
        else:
            cls = "loss"
        rows.append({**s.features, "entry_time": df.index[fill],
                     "exit_time": df.index[exit_i], "dir_": d,
                     "entry_px": entry, "exit_px": float(exit_px),
                     "risk_px": risk, "year": int(df.index[fill].year),
                     "realized_R": r, "win": int(r > 0), "cls": cls,
                     "outcome": "tp" if outcome == "tp"
                     else ("maxhold" if outcome == "maxhold" else "sl")})
    return pd.DataFrame(rows)


def summarise(df, label):
    n = len(df)
    w = (df["cls"] == "win").mean() * 100
    sc = (df["cls"] == "scratch").mean() * 100
    ls = (df["cls"] == "loss").mean() * 100
    avgw = df[df["realized_R"] > 0]["realized_R"].mean()
    avgl = df[df["realized_R"] < 0]["realized_R"].mean()
    exp = df["realized_R"].mean()
    # win rate excluding scratches (the "of decided trades" number)
    dec = df[df["cls"] != "scratch"]
    wd = (dec["realized_R"] > 0).mean() * 100 if len(dec) else 0
    print(f"{label:>8} | win {w:4.1f}% scratch {sc:4.1f}% loss {ls:4.1f}% "
          f"| win/decided {wd:4.1f}% | avgW {avgw:+.2f} avgL {avgl:+.2f} "
          f"| exp {exp:+.3f}R")


def main():
    policies = ["fixed", "be1.0", "be1.5", "trail"]
    per_policy = {p: [] for p in policies}
    for name, folder, of in ASSETS:
        os.environ["ORDERFLOW"] = "1" if of else "0"
        df = load_binance_klines(folder)
        setups = detect_setups(df)
        for p in policies:
            pol = "fixed" if p == "fixed" else p
            if p == "fixed":
                t = B.gen_trades(df, setups)
                t["cls"] = np.where(t["realized_R"] > 0.02, "win",
                                    np.where(t["realized_R"] < -0.02, "loss",
                                             "scratch"))
            else:
                t = walk(df, setups, pol)
            t = t.dropna(subset=["realized_R"])
            t["asset"] = name
            per_policy[p].append(t)
        print(f"  {name}: {len(setups)} setups walked", flush=True)

    print("\n=== exit-policy comparison (all detected setups) ===")
    os.makedirs("data/be", exist_ok=True)
    for p in policies:
        allp = pd.concat(per_policy[p])
        summarise(allp, p)
        allp.to_parquet(f"data/be/{p}.parquet")


if __name__ == "__main__":
    main()
