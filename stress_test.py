"""
Adversarial stress tests (answering the audit):
  * average holding time + total OOS trades
  * a brutal 50% "haircut": DOUBLE every spread/cost AND add mandatory
    slippage to every entry and exit, then re-sequence -- does it still hold?
  * funding-cost sensitivity given the real holding time

Walk-forward selection (train < Y, score year Y), pooled into one account,
sequenced with the live rules (adaptive sizing, 1:2 shared cap, daily budget).
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


def wf_picks(a):
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


def sequence(allt, cost_mult=1.0, extra_exit_bps=0.0):
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
            open_assets.discard(asset); open_notional -= ntl; open_worst -= wdl
            eq += pnl; peak = max(peak, eq); rows[idx]["equity_after"] = eq
        if t["asset"] in open_assets:
            continue
        stop_frac = t["risk_px"] / t["entry_px"]
        e_bps, s_bps = COST[t["asset"]]
        e_bps *= cost_mult; s_bps = s_bps * cost_mult + extra_exit_bps
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
        rows.append({"exit_time": t["exit_time"], "asset": t["asset"], "pnl": pnl,
                     "win": int(net_r > 0), "equity_after": np.nan,
                     "hold_h": (t["exit_time"] - t["entry_time"]).total_seconds()/3600})
        idx = len(rows) - 1
        wd = notional * worst_frac
        open_assets.add(t["asset"]); open_notional += notional; open_worst += wd
        max_lev = max(max_lev, open_notional / INIT)
        heapq.heappush(open_heap, (t["exit_time"], pnl, t["asset"], notional, wd, idx))
    while open_heap:
        ex, pnl, asset, ntl, wdl, idx = heapq.heappop(open_heap)
        eq += pnl; peak = max(peak, eq); rows[idx]["equity_after"] = eq
    return pd.DataFrame(rows)


def report(led, label):
    led = led.copy()
    led["day"] = pd.to_datetime(led["exit_time"]).dt.normalize()
    ret = led["pnl"].sum() / INIT * 100
    eqc = led.set_index("exit_time")["equity_after"]
    dd = ((eqc.cummax() - eqc) / eqc.cummax()).max() * 100
    worst = (led.groupby("day")["pnl"].sum() / INIT * 100).min()
    win = led["win"].mean() * 100
    print(f"{label:>26} | ret {ret:>+8.1f}% | maxDD {dd:5.2f}% | "
          f"worstDay {worst:+.2f}% | win {win:.1f}% | n {len(led)}")
    return led


def main():
    allt = pd.concat([p for p in (wf_picks(a) for a in ASSETS) if p is not None]
                     ).sort_values("entry_time").reset_index(drop=True)
    base = report(sequence(allt), "baseline")
    report(sequence(allt, cost_mult=2.0, extra_exit_bps=5.0),
           "50% haircut (2x cost+slip)")
    report(sequence(allt, cost_mult=3.0, extra_exit_bps=10.0),
           "brutal (3x cost+10bps slip)")

    # holding time + funding sensitivity
    hh = base["hold_h"]
    print("\nHolding time: mean {:.1f}h  median {:.1f}h  "
          "(<2min exits: {})".format(hh.mean(), hh.median(), int((hh < 1/30).sum())))
    oos = base[pd.to_datetime(base["exit_time"]).dt.year >= 2024]
    print(f"Total trades 2021-2026: {len(base)}   (OOS 2024+: {len(oos)})")
    # funding: typical perp ~0.01%/8h on notional; our risk_dollar = notional*stop_frac
    # so funding-as-%-of-base per trade = 0.0001 * (hold_h/8) * lev_per_trade
    avg_lev = 0.5   # ~0.5x per position on average (2x cap across ~4 positions)
    fund_pct = (0.0001 * (hh.mean()/8) * avg_lev) * 100
    print(f"Funding drag est: ~{fund_pct:.3f}% of base per trade "
          f"-> ~{fund_pct*len(base)/6:.2f}%/yr if every position were a perp")


if __name__ == "__main__":
    main()
