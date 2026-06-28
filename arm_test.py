"""
RESEARCH: does the edge survive deciding EARLY (at the order-block bar) so a
pending limit can fill on first touch -- backtest-aligned execution?

Same setups, same fills, same realized R. The ONLY change: the model selects
on the arm-time feature copy (arm_*) instead of the trigger-time features. If
the sized OOS edge holds, the pending/first-touch live engine is viable.
"""
from __future__ import annotations

import os
os.environ.setdefault("CRYPTO", "1")
os.environ["ARM_FEATS"] = "1"     # detect_setups gates the arm_* copy on this

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import backtest_ftmo as B
import goat_test as G
from smc_detector import detect_setups
from crypto_loader import load_binance_klines
from live_runner import INIT

ASSETS = {"BTC": "data/btc", "ETH": "data/eth", "SOL": "data/sol"}
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]


def trades(folder):
    os.environ["ORDERFLOW"] = "0"; os.environ["BASE_TF"] = "15min"
    df = load_binance_klines(folder)
    return B.gen_trades(df, detect_setups(df)).dropna(subset=["realized_R"])


def run(per, feat_kind):
    # feat_kind: "trigger" -> the live 19; "arm" -> the arm_* copy
    sample = next(iter(per.values()))
    if feat_kind == "trigger":
        feats = [c for c in sample.columns if c not in B.NON_FEATURES
                 and c != "asset" and not c.startswith("arm_")
                 and sample[c].dtype.kind in "fiu" and sample[c].notna().any()]
    else:
        feats = [c for c in sample.columns if c.startswith("arm_")
                 and sample[c].dtype.kind in "fiu" and sample[c].notna().any()]
    picks, aucs = [], []
    for a, d in per.items():
        for Y in YEARS:
            tr, te = d[d.year < Y], d[d.year == Y]
            if len(tr) < 200 or len(te) < 20 or tr.win.nunique() < 2:
                continue
            mm = B._model().fit(tr[feats], tr.win)
            ptr = mm.predict_proba(tr[feats])[:, 1]; pte = mm.predict_proba(te[feats])[:, 1]
            if te.win.nunique() > 1:
                aucs.append(roc_auc_score(te.win, pte))
            thr = np.quantile(ptr, 0.88)
            s = te[pte >= thr].copy(); s["prob"] = pte[pte >= thr]; s["asset"] = a
            picks.append(s)
    allt = pd.concat(picks).sort_values("entry_time").reset_index(drop=True)
    led = G.sequence(allt, G.GOAT_TYPICAL)
    led["ym"] = pd.to_datetime(led.exit_time).dt.strftime("%Y-%m")
    eqc = led.set_index("exit_time")["eq"]
    dd = ((eqc.cummax() - eqc) / eqc.cummax()).max() * 100
    mo = (led.groupby("ym").pnl.sum() / INIT * 100).mean()
    gp = led.loc[led.pnl > 0, "pnl"].sum(); gl = -led.loc[led.pnl < 0, "pnl"].sum()
    r = allt.realized_R; w = (r > 0.1).sum(); ll = (r < -0.1).sum()
    return dict(auc=np.mean(aucs), win=w / (w + ll) * 100, monthly=mo, dd=dd,
                pf=gp / gl if gl else 0, n=len(led))


def main():
    per = {}
    for a, f in ASSETS.items():
        per[a] = trades(f); per[a]["asset"] = a
        print(f"  {a}: {len(per[a])} trades", flush=True)
    print("\n=== decision timing (same fills; BTC/ETH/SOL, Goat, top-12%) ===")
    print(f"{'decide on':>26} {'OOS AUC':>8} {'win%':>6} {'monthly%':>9} {'maxDD%':>7} {'PF':>6}")
    for kind, label in [("trigger", "TRIGGER (current backtest)"),
                        ("arm", "ARM-TIME (pending/live-real)")]:
        x = run(per, kind)
        print(f"{label:>26} {x['auc']:>8.3f} {x['win']:>6.1f} {x['monthly']:>8.2f}% "
              f"{x['dd']:>6.2f}% {x['pf']:>6.2f}")


if __name__ == "__main__":
    main()
