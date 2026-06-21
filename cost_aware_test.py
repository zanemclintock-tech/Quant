"""
Cost-aware model selection vs cost-naive, across spread regimes.

The fix for cost sensitivity: train/select the model on NET-of-cost profit at
an ASSUMED cost level, not gross win. At wider spreads the net-positive label
flips many marginal trades to losers, so the model keeps only the fat setups
that still clear the higher cost -- fewer trades, but expectancy stays > 0 and
the drawdown stays controlled instead of bleeding.

For each spread regime we compare, sequenced AT that regime's real cost:
  naive  : current models (label = gross win, select on gross R)
  aware  : label = (net R at this cost) > 0, select on net R at this cost
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

import backtest_ftmo as B
import sizing as SZ
from live_runner import COST, INIT

ASSETS = ["BTC", "ETH", "SOL", "BNB", "LTC"]
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
# (label, cost multiplier, extra exit slippage bps)
REGIMES = [("tight (prop)", 1.0, 0.0), ("medium 2x+5", 2.0, 5.0),
           ("wide 3x+10", 3.0, 10.0)]


def cost_r(row, mult, extra):
    """Realised cost in R -- uses the actual exit (TP=maker, SL=taker). Only
    valid for PnL accounting AFTER the trade, never for the entry gate."""
    e, s = COST[row["asset"]]
    e *= mult
    exit_bps = (e if row["outcome"] == "tp" else s * mult + extra)
    return (e + exit_bps) / 1e4 * row["entry_px"] / row["risk_px"]


def gate_cost_r(row, mult, extra):
    """Entry-time cost estimate in R -- OUTCOME-BLIND: assumes the worst-case
    taker exit (a stop crossing the spread). Knowable live, no lookahead."""
    e, s = COST[row["asset"]]
    e *= mult
    s = s * mult + extra
    return (e + s) / 1e4 * row["entry_px"] / row["risk_px"]


def load(a):
    d = pd.read_parquet(f"data/trades/{a}_15min.parquet").dropna(subset=["realized_R"])
    d["asset"] = a
    return d


def wf(d, mult, extra, aware):
    feats = [c for c in d.columns if c not in B.NON_FEATURES and c not in
             ("asset", "cls", "net_r") and d[c].dtype.kind in "fiu"
             and d[c].notna().any()]
    d = d.copy()
    d["net_r"] = d.apply(lambda r: r["realized_R"] - cost_r(r, mult, extra), axis=1)
    # both train on gross win (the durable signal); the difference is SELECTION:
    #  naive -> pick threshold maximizing gross R (cost-blind)
    #  aware -> net-edge GATE: raise the bar until the kept trades clear cost
    #           with margin; if none clear, trade NOTHING (don't bleed)
    MARGIN = 0.05
    out = []
    for Y in YEARS:
        tr, te = d[d["year"] < Y], d[d["year"] == Y]
        if len(tr) < 200 or len(te) < 20 or tr["win"].nunique() < 2:
            continue
        m = B._model().fit(tr[feats], tr["win"])
        p_tr = m.predict_proba(tr[feats])[:, 1]
        p_te = m.predict_proba(te[feats])[:, 1]
        if not aware:
            thr, best = float(np.median(p_tr)), -9.9
            for q in np.quantile(p_tr, np.linspace(0.3, 0.9, 25)):
                s = tr[p_tr >= q]
                if len(s) >= 0.15 * len(tr) and s["realized_R"].mean() > best:
                    best, thr = s["realized_R"].mean(), q
        else:
            thr = np.inf            # default: trade nothing this cost regime
            for q in np.quantile(p_tr, np.linspace(0.2, 0.97, 40)):
                s = tr[p_tr >= q]
                if len(s) >= 0.02 * len(tr) and s["net_r"].mean() > MARGIN:
                    thr = q         # lowest bar whose kept trades clear cost
                    break
        sel = te[p_te >= thr].copy()
        sel["prob"] = p_te[p_te >= thr]
        out.append(sel)
    return pd.concat(out) if out else None


def sequence(allt, mult, extra, gate=np.inf):
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
        # economic circuit-breaker: skip if the live round-trip cost (in R)
        # eats more than `gate` of the trade's risk. Uses only entry/risk/cost,
        # all known live -> at tight spreads it never triggers; at wide spreads
        # it blocks the now-unprofitable trades instead of bleeding.
        if gate_cost_r(t, mult, extra) > gate:    # entry-time, outcome-blind
            continue
        sf = t["risk_px"] / t["entry_px"]
        e, s = COST[t["asset"]]; e *= mult; s = s * mult + extra
        wf_ = sf + (e + s) / 1e4
        rem = SZ.DAY_BUDGET * INIT - max(0.0, day_start - eq) - open_w
        ntl = SZ.size_notional(t["prob"], sf, open_n, INIT, budget_notional=rem / wf_)
        if ntl <= 0.02 * INIT:
            continue
        net = t["realized_R"] - cost_r(t, mult, extra)
        pnl = net * ntl * sf
        rows.append({"exit_time": t["exit_time"], "asset": t["asset"],
                     "pnl": pnl, "win": int(net > 0), "eq": np.nan})
        idx = len(rows) - 1; wd = ntl * wf_
        open_a.add(t["asset"]); open_n += ntl; open_w += wd
        mlev = max(mlev, open_n / INIT)
        heapq.heappush(heap, (t["exit_time"], pnl, t["asset"], ntl, wd, idx))
    while heap:
        ex, pnl, asset, ntl, wd, idx = heapq.heappop(heap)
        eq += pnl; peak = max(peak, eq); rows[idx]["eq"] = eq
    return pd.DataFrame(rows)


def stats(led):
    if not len(led):
        return "no trades"
    led = led.copy(); led["d"] = pd.to_datetime(led["exit_time"]).dt.normalize()
    ret = led["pnl"].sum() / INIT * 100
    eqc = led.set_index("exit_time")["eq"]
    dd = ((eqc.cummax() - eqc) / eqc.cummax()).max() * 100
    worst = (led.groupby("d")["pnl"].sum() / INIT * 100).min()
    return (f"ret {ret:>+8.1f}% | maxDD {dd:5.2f}% | worstDay {worst:+.2f}% | "
            f"win {led['win'].mean()*100:4.1f}% | n {len(led)}")


def main():
    data = {a: load(a) for a in ASSETS}
    # selection is cost-blind (the proven model) -> compute picks once
    picks = [wf(data[a], 1.0, 0.0, False) for a in ASSETS]
    allt = pd.concat([p for p in picks if p is not None]
                     ).sort_values("entry_time").reset_index(drop=True)
    for label, mult, extra in REGIMES:
        print(f"\n=== {label} ===")
        for gate, gname in [(np.inf, "no gate"), (0.70, "gate .70R"),
                            (0.60, "gate .60R")]:
            led = sequence(allt, mult, extra, gate=gate)
            print(f"  {gname:>10}: {stats(led)}")


if __name__ == "__main__":
    main()
