"""
Scale-out exit test: take 50% off at +1R, move the stop to break-even, run the
other 50% to the 2R target. Outcomes become:
  * -1.0R  : stopped before +1R (full loss)
  * +0.5R  : hit +1R (booked half), runner gave back to break-even
  * +1.5R  : hit +1R, runner reached the 2R target
Re-walks every setup (causal: fixed price levels, stop checked before target,
runner checked from the NEXT bar so no intrabar look-ahead), then runs the
SAME sized walk-forward at Goat's real spreads to compare head-to-head with
the current 2R + break-even engine.
"""
from __future__ import annotations

import heapq
import os

import numpy as np
import pandas as pd

import backtest_ftmo as B
import sizing as SZ
import goat_test as G
from live_runner import INIT
from smc_detector import detect_setups
from crypto_loader import load_binance_klines

os.environ["CRYPTO"] = "1"
os.environ["BASE_TF"] = "15min"
FOLDERS = {"BTC": ("data/btc", True), "ETH": ("data/eth", False),
           "SOL": ("data/sol", False)}
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
BUF = 1e-4


def walk(df, setups, min_hold=B.MIN_HOLD_MIN):
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
        t1 = entry + d * 1.0 * risk            # partial target (+1R)
        tp2 = entry + d * 2.0 * risk           # runner target (+2R)
        be = entry + d * BUF * entry
        hi = B._same_day_end(a, fill, B.MAX_HOLD_MIN, True)
        partial = False
        R = None
        outc = None
        for k in range(fill + 1 + min_hold, hi):
            hk, lk = a["h"][k], a["l"][k]
            if not partial:
                if d == 1:
                    if lk <= s.stop:
                        R, outc = -1.0, "sl"; break
                    if hk >= t1:
                        partial = True            # runner checked next bar
                else:
                    if hk >= s.stop:
                        R, outc = -1.0, "sl"; break
                    if lk <= t1:
                        partial = True
            else:
                if d == 1:
                    if lk <= be:
                        R, outc = 0.5, "partial"; break
                    if hk >= tp2:
                        R, outc = 1.5, "tp"; break
                else:
                    if hk >= be:
                        R, outc = 0.5, "partial"; break
                    if lk <= tp2:
                        R, outc = 1.5, "tp"; break
        if R is None:                              # timed out
            run = (a["c"][hi - 1] - entry) * d / risk
            R = 0.5 * 1.0 + 0.5 * max(run, 0.0) if partial else run
            outc = "maxhold"
        rows.append({**s.features, "entry_time": df.index[fill],
                     "exit_time": df.index[min(k, hi - 1)], "dir_": d,
                     "entry_px": entry, "exit_px": entry + d * R * risk,
                     "risk_px": risk, "year": int(df.index[fill].year),
                     "realized_R": R, "win": int(R > 0), "outcome": outc})
    return pd.DataFrame(rows)


def wf_picks(d, a):
    d = d.dropna(subset=["realized_R"]).copy()
    feats = [c for c in d.columns if c not in B.NON_FEATURES and c != "asset"
             and d[c].dtype.kind in "fiu" and d[c].notna().any()]
    out = []
    for Y in YEARS:
        tr, te = d[d["year"] < Y], d[d["year"] == Y]
        if len(tr) < 200 or len(te) < 20 or tr["win"].nunique() < 2:
            continue
        m = B._model().fit(tr[feats], tr["win"])
        p_tr = m.predict_proba(tr[feats])[:, 1]
        p_te = m.predict_proba(te[feats])[:, 1]
        thr, best = float(np.median(p_tr)), -9.9
        for q in np.quantile(p_tr, np.linspace(0.3, 0.9, 25)):
            sb = tr[p_tr >= q]
            if len(sb) >= 0.15 * len(tr) and sb["realized_R"].mean() > best:
                best, thr = sb["realized_R"].mean(), q
        sel = te[p_te >= thr].copy()
        sel["prob"] = p_te[p_te >= thr]
        sel["asset"] = a
        out.append(sel)
    return pd.concat(out) if out else None


def main():
    picks = []
    raw = []
    for a, (folder, of) in FOLDERS.items():
        os.environ["ORDERFLOW"] = "1" if of else "0"
        df = load_binance_klines(folder)
        t = walk(df, detect_setups(df))
        t["asset"] = a
        raw.append(t)
        picks.append(wf_picks(t, a))
        print(f"  {a}: {len(t)} setups walked", flush=True)
    allt = pd.concat([p for p in picks if p is not None]
                     ).sort_values("entry_time").reset_index(drop=True)

    # selected-trade stats (your convention: scratch excluded)
    r = allt["realized_R"]
    win = (r > 0.1).sum(); loss = (r < -0.1).sum()
    aw, al = r[r > 0.1].mean(), r[r < -0.1].mean()
    print(f"\nSCALE-OUT selected: {len(allt)} trades")
    print(f"  win rate (scratch excl): {win/(win+loss)*100:.1f}%   "
          f"avg win {aw:+.2f}R avg loss {al:+.2f}R  RR {aw/abs(al):.2f}")
    print(f"  outcome mix: {allt['outcome'].value_counts(normalize=True).round(3).to_dict()}")

    led = G.sequence(allt, G.GOAT_TYPICAL)
    led["d"] = pd.to_datetime(led["exit_time"]).dt.normalize()
    ret = led["pnl"].sum() / INIT * 100
    eqc = led.set_index("exit_time")["eq"]
    dd = ((eqc.cummax() - eqc) / eqc.cummax()).max() * 100
    monthly = led.groupby(pd.to_datetime(led["exit_time"]).dt.strftime("%Y-%m"))["pnl"].sum()/INIT*100
    gp = led.loc[led.pnl > 0, "pnl"].sum(); gl = -led.loc[led.pnl < 0, "pnl"].sum()
    print(f"\n  SIZED @ Goat typical: ret {ret:+.1f}% | maxDD {dd:.2f}% | "
          f"monthly mean {monthly.mean():+.2f}% | PF {gp/gl:.2f} | n {len(led)}")
    print("\n  vs CURRENT 2R+BE @ Goat typical: ret +1169% | maxDD 4.27% | "
          "monthly +17.7% | PF 1.82 | winrate 63%")


if __name__ == "__main__":
    main()
