"""
Diagnose the edge decay and test fixes. Same walk-forward portfolio, but
vary HOW the model is trained each year:

  expanding  -- train on ALL prior years (current approach)
  rolling12  -- train only on the last 12 months (adapt to current regime)
  rolling18  -- last 18 months
  weighted   -- all prior years, exponentially down-weighting old trades

First a diagnosis table (per-year AUC vs base vs selected expectancy) to
see whether the SIGNAL decays or the market REWARD decays.

    CRYPTO=1 python decay_fix.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import backtest_ftmo as B
import multi_asset as M

ASSETS = ["BTC", "ETH", "SOL", "BNB"]
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
COLS = ["entry_time", "exit_time", "realized_R", "entry_px", "exit_px",
        "risk_px", "outcome", "asset"]
TF = os.environ.get("TF", "15min")


def feats_of(data):
    return [c for c in data.columns if c not in B.NON_FEATURES
            and c != "asset" and data[c].dtype.kind in "fiu"
            and data[c].notna().any()]


def train_set(data, Y, mode):
    jan1 = pd.Timestamp(year=Y, month=1, day=1, tz="UTC")
    if mode == "expanding" or mode == "weighted":
        return data[data["entry_time"] < jan1]
    win = {"rolling12": 365, "rolling18": 548}[mode]
    lo = jan1 - pd.Timedelta(days=win)
    return data[(data["entry_time"] < jan1) & (data["entry_time"] >= lo)]


def wf_select(data, mode):
    feats = feats_of(data)
    picked, diag = [], []
    for Y in TEST_YEARS:
        tr = train_set(data, Y, mode)
        te = data[data["year"] == Y]
        if len(tr) < 200 or len(te) < 30 or tr["win"].nunique() < 2:
            continue
        m = B._model()
        if mode == "weighted":
            jan1 = pd.Timestamp(year=Y, month=1, day=1, tz="UTC")
            age = (jan1 - tr["entry_time"]).dt.days.to_numpy()
            w = 0.5 ** (age / 365.0)                  # 1-year half-life
            m.fit(tr[feats], tr["win"], sample_weight=w)
        else:
            m.fit(tr[feats], tr["win"])
        p_tr = m.predict_proba(tr[feats])[:, 1]
        p_te = m.predict_proba(te[feats])[:, 1]
        thr, best = float(np.median(p_tr)), -9.9
        for q in np.quantile(p_tr, np.linspace(0.3, 0.9, 25)):
            s = tr[p_tr >= q]
            if len(s) >= 0.15 * len(tr) and s["realized_R"].mean() > best:
                best, thr = s["realized_R"].mean(), q
        sel = te[p_te >= thr]
        picked.append(sel)
        diag.append((Y, roc_auc_score(te["win"], p_te),
                     te["realized_R"].mean(), sel["realized_R"].mean()
                     if len(sel) else np.nan, len(sel)))
    return (pd.concat(picked) if picked else None), diag


def main():
    data = {a: pd.read_parquet(f"data/trades/{a}_{TF}.parquet")
            for a in ASSETS}

    print(f"=== DIAGNOSIS ({TF}, BTC, expanding) — does signal or reward "
          f"decay? ===")
    _, diag = wf_select(data["BTC"], "expanding")
    print(f"  {'year':>5} {'OOS_AUC':>8} {'base_exp':>9} {'sel_exp':>9} "
          f"{'n_sel':>6}")
    for Y, auc, base, sel, n in diag:
        print(f"  {Y:>5} {auc:>8.3f} {base:>+9.3f} {sel:>+9.3f} {n:>6}")

    print(f"\n=== FIX TEST ({TF} 4-asset portfolio, realistic cost, 0.5% "
          f"risk) ===")
    print(f"  {'mode':>10} {'total%':>8} {'2024%':>7} {'2025%':>7} "
          f"{'2026%':>7} {'maxDD%':>7} {'dViol':>5} {'PASS':>4}")
    for mode in ("expanding", "rolling12", "rolling18", "weighted"):
        sels = []
        for a in ASSETS:
            s, _ = wf_select(data[a], mode)
            if s is not None and len(s):
                s = s.copy(); s["asset"] = a
                sels.append(s[COLS])
        allt = pd.concat(sels).sort_values("entry_time")
        B.RISK_FRAC = 0.005
        r = M.portfolio(allt, 0.005, B.EXEC["realistic"])
        cur = r["curve"]; yr = cur.groupby(cur.index.year).last()
        yret = (yr / yr.shift(1) - 1)
        g = lambda y: yret.get(y, np.nan) * 100
        print(f"  {mode:>10} {r['ret']*100:>+8.0f} {g(2024):>+7.0f} "
              f"{g(2025):>+7.0f} {g(2026):>+7.0f} {r['maxdd']*100:>7.2f} "
              f"{r['dviol']:>5} {'YES' if r['passed'] else 'no':>4}")


if __name__ == "__main__":
    main()
