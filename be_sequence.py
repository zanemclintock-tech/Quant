"""
Bottom-line verdict for each exit policy: walk-forward model selection +
the live sizing path (adaptive, 1:2 shared cap, cost-inclusive daily budget),
sequenced into one account. Reports fixed-base return, max DD, worst day,
and win rate. This is the number that decides, not raw win rate.
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

import backtest_ftmo as B
import sizing as SZ
from live_runner import COST, INIT

POLICIES = ["fixed", "be1.0", "be1.5", "trail"]
TEST_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]


def wf_picks(df, a):
    df = df.dropna(subset=["realized_R"]).copy()
    feats = [c for c in df.columns
             if c not in B.NON_FEATURES and c != "asset" and c != "cls"
             and df[c].dtype.kind in "fiu" and df[c].notna().any()]
    out = []
    for Y in TEST_YEARS:
        tr, te = df[df["year"] < Y], df[df["year"] == Y]
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
        out.append(sel)
    return pd.concat(out) if out else None


def sequence(allt):
    eq = peak = INIT
    open_heap, open_assets = [], set()
    open_notional = open_worst = 0.0
    day, day_start, max_lev = None, INIT, 0.0
    rows = []
    for _, t in allt.iterrows():
        d = t["entry_time"].normalize()
        if d != day:
            day, day_start = d, eq
        while open_heap and open_heap[0][0] <= t["entry_time"]:
            ex, pnl, asset, ntl, wdl, idx = heapq.heappop(open_heap)
            cd = ex.normalize()
            if cd != day:
                day, day_start = cd, eq
            open_assets.discard(asset)
            open_notional -= ntl; open_worst -= wdl
            eq += pnl; peak = max(peak, eq)
            rows[idx]["equity_after"] = eq
        if t["asset"] in open_assets:
            continue
        stop_frac = t["risk_px"] / t["entry_px"]
        e_bps, s_bps = COST[t["asset"]]
        worst_frac = stop_frac + (e_bps + s_bps) / 1e4
        remaining = SZ.DAY_BUDGET * INIT - max(0.0, day_start - eq) - open_worst
        notional = SZ.size_notional(t["prob"], stop_frac, open_notional, INIT,
                                    budget_notional=remaining / worst_frac)
        if notional <= 0.02 * INIT:
            continue
        exit_bps = e_bps if t["outcome"] == "tp" else s_bps
        cost_r = (e_bps + exit_bps) / 1e4 * t["entry_px"] / t["risk_px"]
        net_r = t["realized_R"] - cost_r
        pnl = net_r * notional * stop_frac
        rows.append({"entry_time": t["entry_time"], "exit_time": t["exit_time"],
                     "asset": t["asset"], "pnl": pnl,
                     "win": int(t["realized_R"] > 0.02), "equity_after": np.nan})
        idx = len(rows) - 1
        wd = notional * worst_frac
        open_assets.add(t["asset"]); open_notional += notional; open_worst += wd
        max_lev = max(max_lev, open_notional / INIT)
        heapq.heappush(open_heap, (t["exit_time"], pnl, t["asset"], notional, wd, idx))
    while open_heap:
        ex, pnl, asset, ntl, wdl, idx = heapq.heappop(open_heap)
        eq += pnl; peak = max(peak, eq)
        rows[idx]["equity_after"] = eq
    return pd.DataFrame(rows), max_lev


def main():
    print(f"{'policy':>8} {'trades':>7} {'ret%':>9} {'maxDD%':>7} "
          f"{'win%':>6} {'worstDay%':>10} {'peakLev':>8}")
    for p in POLICIES:
        df = pd.read_parquet(f"data/be/{p}.parquet")
        picks = []
        for a, g in df.groupby("asset"):
            sel = wf_picks(g, a)
            if sel is not None:
                sel["asset"] = a
                picks.append(sel)
        allt = pd.concat(picks).sort_values("entry_time").reset_index(drop=True)
        led, max_lev = sequence(allt)
        led["day"] = pd.to_datetime(led["exit_time"]).dt.normalize()
        ret = led["pnl"].sum() / INIT * 100
        eqc = led.set_index("exit_time")["equity_after"]
        dd = ((eqc.cummax() - eqc) / eqc.cummax()).max() * 100
        win = led["win"].mean() * 100
        worst = (led.groupby("day")["pnl"].sum() / INIT * 100).min()
        print(f"{p:>8} {len(led):>7} {ret:>+9.1f} {dd:>7.2f} "
              f"{win:>6.1f} {worst:>+10.2f} {max_lev:>7.2f}x")


if __name__ == "__main__":
    main()
