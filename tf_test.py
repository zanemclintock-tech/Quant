"""
RESEARCH ONLY (not used by the live engine): does the strategy + its volume
profile work better at a different base timeframe? Re-detects setups and
re-walks every trade at each TF (5min..1H), so the volume profile / VWAP /
structure are all computed on that TF's bars, then runs the SAME pipeline as
live -- walk-forward, strict top-12% selection, sized at Goat's real spreads
-- and reports OOS AUC, win%, monthly%, drawdown and profit factor per TF.
"""
from __future__ import annotations

import os
os.environ.setdefault("CRYPTO", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import backtest_ftmo as B
import goat_test as G
from smc_detector import detect_setups
from crypto_loader import load_binance_klines
from live_runner import INIT

ASSETS = {"BTC": "data/btc", "ETH": "data/eth", "SOL": "data/sol"}
TFS = ["5min", "10min", "15min", "30min", "60min"]
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]


def trades_for(folder, tf):
    os.environ["ORDERFLOW"] = "0"          # kline-only, like the live models
    os.environ["BASE_TF"] = tf
    df = load_binance_klines(folder)
    return B.gen_trades(df, detect_setups(df)).dropna(subset=["realized_R"])


def feats_of(d):
    return [c for c in d.columns if c not in B.NON_FEATURES and c != "asset"
            and d[c].dtype.kind in "fiu" and d[c].notna().any()]


def run_tf(tf):
    per = {}
    for a, folder in ASSETS.items():
        d = trades_for(folder, tf)
        d["asset"] = a
        per[a] = d
    feats = sorted(set.intersection(*[set(feats_of(d)) for d in per.values()]))
    picks, aucs = [], []
    for a, d in per.items():
        for Y in YEARS:
            tr, te = d[d.year < Y], d[d.year == Y]
            if len(tr) < 200 or len(te) < 20 or tr.win.nunique() < 2:
                continue
            m = B._model().fit(tr[feats], tr.win)
            ptr = m.predict_proba(tr[feats])[:, 1]
            pte = m.predict_proba(te[feats])[:, 1]
            if te.win.nunique() > 1:
                aucs.append(roc_auc_score(te.win, pte))
            thr = np.quantile(ptr, 0.88)        # strict top-12%, matching live
            s = te[pte >= thr].copy()
            s["prob"] = pte[pte >= thr]
            s["asset"] = a
            picks.append(s)
    allt = pd.concat(picks).sort_values("entry_time").reset_index(drop=True)
    led = G.sequence(allt, G.GOAT_TYPICAL)
    if not len(led):
        return dict(tf=tf, auc=np.mean(aucs), win=0, monthly=0, dd=0, pf=0,
                    n=0, hold=0)
    led["ym"] = pd.to_datetime(led.exit_time).dt.strftime("%Y-%m")
    led["d"] = pd.to_datetime(led.exit_time).dt.normalize()
    eqc = led.set_index("exit_time")["eq"]
    dd = ((eqc.cummax() - eqc) / eqc.cummax()).max() * 100
    mo = (led.groupby("ym").pnl.sum() / INIT * 100).mean()
    gp = led.loc[led.pnl > 0, "pnl"].sum()
    gl = -led.loc[led.pnl < 0, "pnl"].sum()
    r = allt.realized_R
    w = (r > 0.1).sum(); l = (r < -0.1).sum()
    hold = (pd.to_datetime(allt.exit_time) - pd.to_datetime(allt.entry_time)
            ).dt.total_seconds().mean() / 3600
    return dict(tf=tf, auc=np.mean(aucs), win=w / (w + l) * 100, monthly=mo,
                dd=dd, pf=gp / gl if gl else float("inf"), n=len(led), hold=hold)


def main():
    rows = []
    for tf in TFS:
        rows.append(run_tf(tf))
        print(f"  done {tf}", flush=True)
    print("\n=== volume profile / strategy by base timeframe "
          "(BTC/ETH/SOL, Goat spreads, top-12%) ===")
    print(f"{'TF':>6} {'OOS AUC':>8} {'win%':>6} {'monthly%':>9} {'maxDD%':>7} "
          f"{'PF':>6} {'trades/yr':>10} {'avg hold(h)':>11}")
    for x in rows:
        print(f"{x['tf']:>6} {x['auc']:>8.3f} {x['win']:>6.1f} "
              f"{x['monthly']:>8.2f}% {x['dd']:>6.2f}% {x['pf']:>6.2f} "
              f"{x['n']//6:>10} {x['hold']:>11.1f}")


if __name__ == "__main__":
    main()
