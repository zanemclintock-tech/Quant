"""
The honest validation: expanding walk-forward, multi-asset, one continuous
FTMO account across 2022..2026. For each test year Y the model trains ONLY
on years < Y (per asset), selects that year's trades, and they are pooled
across assets into a single account whose equity carries forward through
the whole period under the 6% trailing / 3% daily rules.

This kills the single-window luck the earlier OOS split could hide, and it
is the test that decides whether the multi-asset edge is real and tradeable.

Costs: demo (prop reality -- tiny spread, no real-market slippage) and
realistic (live markets). Focus timeframes settable via TFS.

    CRYPTO=1 TFS=1min,5min python walk_forward_portfolio.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import backtest_ftmo as B
import multi_asset as M

ASSETS = ["BTC", "ETH", "SOL", "BNB"]
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
COLS = ["entry_time", "exit_time", "realized_R", "entry_px", "exit_px",
        "risk_px", "outcome", "asset"]
# demo = prop firm reality (fills at price, minimal spread, no slippage);
# realistic = live markets (stops cross the spread + slip).
DEMO = (1.0, {"tp": 1.0, "sl": 3.0, "maxhold": 3.0})
REAL = B.EXEC["realistic"]


def wf_select(data: pd.DataFrame) -> pd.DataFrame | None:
    """Expanding walk-forward selection for one asset: train < Y, pick the
    IS threshold, keep year-Y trades above it."""
    feats = [c for c in data.columns
             if c not in B.NON_FEATURES and c != "asset"
             and data[c].dtype.kind in "fiu" and data[c].notna().any()]
    picked = []
    for Y in TEST_YEARS:
        tr, te = data[data["year"] < Y], data[data["year"] == Y]
        if len(tr) < 200 or len(te) < 30 or tr["win"].nunique() < 2:
            continue
        m = B._model().fit(tr[feats], tr["win"])
        p_tr = m.predict_proba(tr[feats])[:, 1]
        p_te = m.predict_proba(te[feats])[:, 1]
        thr, best = float(np.median(p_tr)), -9.9
        for q in np.quantile(p_tr, np.linspace(0.3, 0.9, 25)):
            s = tr[p_tr >= q]
            if len(s) >= 0.15 * len(tr) and s["realized_R"].mean() > best:
                best, thr = s["realized_R"].mean(), q
        picked.append(te[p_te >= thr])
    return pd.concat(picked) if picked else None


def main():
    tfs = os.environ.get("TFS", "1min,5min,15min,60min").split(",")
    for tf in tfs:
        sels = []
        for a in ASSETS:
            f = f"data/trades/{a}_{tf}.parquet"
            if not os.path.exists(f):
                continue
            data = pd.read_parquet(f)
            s = wf_select(data)
            if s is not None and len(s):
                s = s.copy(); s["asset"] = a
                sels.append(s[COLS])
        if not sels:
            print(f"{tf}: no selectable trades"); continue
        allt = pd.concat(sels).sort_values("entry_time")
        span_months = ((allt["entry_time"].iloc[-1]
                        - allt["entry_time"].iloc[0]).days) / 30.44
        print(f"\n===== {tf}  (walk-forward 2022+, {len(allt)} pooled "
              f"trades, ~{span_months:.0f} months) =====")
        print(f" {'cost':>5} {'risk%':>6} {'ret%':>8} {'monthly%':>8} "
              f"{'maxDD%':>7} {'dViol':>5} {'blown':>5} {'PASS':>4}   per-year%")
        for cname, spec in (("demo", DEMO), ("real", REAL)):
            for rf in (0.0025, 0.005):
                r = M.portfolio(allt, rf, spec)
                cur = r["curve"]
                mo = ((1 + r["ret"]) ** (1 / span_months) - 1) * 100 \
                    if r["ret"] > -1 else float("nan")
                yr = cur.groupby(cur.index.year).last()
                yret = (yr / yr.shift(1) - 1).fillna(yr.iloc[0] / B.INIT - 1)
                ys = " ".join(f"{int(y)}:{v*100:+.0f}" for y, v in yret.items())
                print(f" {cname:>5} {rf*100:>6.2f} {r['ret']*100:>+8.1f} "
                      f"{mo:>+8.2f} {r['maxdd']*100:>7.2f} {r['dviol']:>5} "
                      f"{str(r['blown']):>5} {'YES' if r['passed'] else 'no':>4}"
                      f"   {ys}")


if __name__ == "__main__":
    main()
