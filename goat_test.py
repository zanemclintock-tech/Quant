"""
Re-run the walk-forward at Goat Funded Trader's REAL per-coin spreads (from
their published ranges) to get an honest expected result on that venue, and
test the impact of dropping LTC (whose ~38bps spread is 3x what we modelled).

Goat round-trip cost per coin = entry (half spread) + stop exit (full spread
+ ~3bps slippage), at typical and high spread levels.
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

import backtest_ftmo as B
import sizing as SZ
from live_runner import INIT

ASSETS = ["BTC", "ETH", "SOL", "BNB", "LTC"]
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

# (entry_bps, stop_exit_bps) from Goat's published spreads
GOAT_TYPICAL = {"BTC": (1.25, 5.5), "ETH": (2.0, 7.0), "SOL": (5.0, 13.0),
                "BNB": (5.5, 14.0), "LTC": (12.5, 28.0)}
GOAT_HIGH = {"BTC": (2.0, 7.0), "ETH": (3.0, 9.0), "SOL": (8.0, 19.0),
             "BNB": (9.0, 21.0), "LTC": (20.0, 43.0)}


def picks(a):
    d = pd.read_parquet(f"data/trades/{a}_15min.parquet").dropna(subset=["realized_R"])
    feats = [c for c in d.columns if c not in B.NON_FEATURES and c not in
             ("asset", "cls") and d[c].dtype.kind in "fiu" and d[c].notna().any()]
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
            s = tr[p_tr >= q]
            if len(s) >= 0.15 * len(tr) and s["realized_R"].mean() > best:
                best, thr = s["realized_R"].mean(), q
        sel = te[p_te >= thr].copy()
        sel["prob"] = p_te[p_te >= thr]
        sel["asset"] = a
        out.append(sel)
    return pd.concat(out) if out else None


def sequence(allt, cost):
    eq = peak = INIT
    heap, open_a, open_n, open_w = [], set(), 0.0, 0.0
    day, day_start, mlev = None, INIT, 0.0
    rows = []
    for _, t in allt.iterrows():
        dd = t["entry_time"].normalize()
        if dd != day:
            day, day_start = dd, eq
        while heap and heap[0][0] <= t["entry_time"]:
            ex, pnl, asset, ntl, wd, idx = heapq.heappop(heap)
            cdd = ex.normalize()
            if cdd != day:
                day, day_start = cdd, eq
            open_a.discard(asset); open_n -= ntl; open_w -= wd
            eq += pnl; peak = max(peak, eq); rows[idx]["eq"] = eq
        if t["asset"] in open_a:
            continue
        sf = t["risk_px"] / t["entry_px"]
        e, s = cost[t["asset"]]
        wf_ = sf + (e + s) / 1e4
        rem = SZ.DAY_BUDGET * INIT - max(0.0, day_start - eq) - open_w
        ntl = SZ.size_notional(t["prob"], sf, open_n, INIT, budget_notional=rem / wf_)
        if ntl <= 0.02 * INIT:
            continue
        exit_bps = e if t["outcome"] == "tp" else s
        cost_r = (e + exit_bps) / 1e4 * t["entry_px"] / t["risk_px"]
        pnl = (t["realized_R"] - cost_r) * ntl * sf
        rows.append({"exit_time": t["exit_time"], "asset": t["asset"], "pnl": pnl,
                     "win": int(t["realized_R"] - cost_r > 0), "eq": np.nan})
        idx = len(rows) - 1; wd = ntl * wf_
        open_a.add(t["asset"]); open_n += ntl; open_w += wd
        heapq.heappush(heap, (t["exit_time"], pnl, t["asset"], ntl, wd, idx))
    while heap:
        ex, pnl, asset, ntl, wd, idx = heapq.heappop(heap)
        eq += pnl; peak = max(peak, eq); rows[idx]["eq"] = eq
    return pd.DataFrame(rows)


def report(led, label):
    led = led.copy(); led["d"] = pd.to_datetime(led["exit_time"]).dt.normalize()
    ret = led["pnl"].sum() / INIT * 100
    eqc = led.set_index("exit_time")["eq"]
    dd = ((eqc.cummax() - eqc) / eqc.cummax()).max() * 100
    worst = (led.groupby("d")["pnl"].sum() / INIT * 100).min()
    print(f"{label:>26} | ret {ret:>+8.1f}% | maxDD {dd:5.2f}% | "
          f"worstDay {worst:+.2f}% | win {led['win'].mean()*100:4.1f}% | n {len(led)}")


def main():
    allt = pd.concat([p for p in (picks(a) for a in ASSETS) if p is not None]
                     ).sort_values("entry_time").reset_index(drop=True)
    print("== Goat TYPICAL spreads ==")
    report(sequence(allt, GOAT_TYPICAL), "all 5 coins")
    report(sequence(allt[allt.asset != "LTC"], GOAT_TYPICAL), "drop LTC")
    print("\n== Goat HIGH spreads (stress) ==")
    report(sequence(allt, GOAT_HIGH), "all 5 coins")
    report(sequence(allt[allt.asset != "LTC"], GOAT_HIGH), "drop LTC")

    print("\n== per-coin net contribution (Goat typical) ==")
    led = sequence(allt, GOAT_TYPICAL)
    for a in ASSETS:
        sub = led[led["asset"] == a]
        print(f"  {a}: {sub['pnl'].sum()/INIT*100:+8.1f}%  ({len(sub)} trades, "
              f"win {sub['win'].mean()*100:.0f}%)")


if __name__ == "__main__":
    main()
