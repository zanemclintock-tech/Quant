"""
Add every feature, then let the model KEEP ONLY THE ONES THAT HELP.

Single pre-registered config (no sweep): 60-min structure, 60-min
directional horizon. Methodology, leak-free:

  1. Split: IS = years <= 2023 (train), OOS = years >= 2024 (untouched).
  2. Selection happens INSIDE the IS only -- fit on IS-train (<=2022),
     rank every feature by permutation importance on an IS-validation
     fold (2023), and DROP features whose importance is <= 0 (they hurt
     or do nothing). The OOS is never seen during selection.
  3. Refit on the full IS with the kept features, score ONCE on OOS.
  4. Compare full vs selected, and judge both against a matched 24/7
     noise null (a control, not a sweep) for an honest p-value, net of a
     realistic round-trip cost.

    ORDERFLOW=1 CRYPTO=1 CRYPTO_GLOB=data/btc BASE_TF=60min HORIZON=60 \
        NULLS=10 python feature_select.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from labeler import label_setups_directional
from smc_detector import detect_setups
from synthetic import make_synthetic_minutes

warnings.filterwarnings("ignore")

NON_FEATURES = {"entry_time", "year", "realized_R", "win", "outcome",
                "entry_px", "risk_px"}
IS_END_YEAR = 2023
VAL_YEAR = 2023            # IS-internal validation fold for selection
OOS_START_YEAR = 2024
HORIZON = int(os.environ.get("HORIZON", "60"))
NULLS = int(os.environ.get("NULLS", "10"))
COST_BPS = float(os.environ.get("COST_BPS", "10"))


def _model():
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=250, learning_rate=0.04, l2_regularization=1.0,
        min_samples_leaf=40, early_stopping=True, validation_fraction=0.2,
        random_state=0)


def _cost_r(d: pd.DataFrame, bps: float) -> pd.Series:
    return (bps / 1e4) * d["entry_px"] / d["risk_px"]


def select_features(data: pd.DataFrame, feats: list, verbose=False) -> list:
    """Keep only features with > 0 permutation importance on the IS
    validation fold. Pure-IS, so the OOS stays untouched."""
    tr = data[data["year"] < VAL_YEAR]
    val = data[data["year"] == VAL_YEAR]
    if len(tr) < 150 or len(val) < 40 or tr["win"].nunique() < 2 \
            or val["win"].nunique() < 2:
        return feats                                  # too small to select
    m = _model().fit(tr[feats], tr["win"])
    r = permutation_importance(m, val[feats], val["win"], n_repeats=20,
                               random_state=0, scoring="roc_auc")
    kept = [f for f, imp in zip(feats, r.importances_mean) if imp > 0]
    if verbose:
        dropped = [(f, imp) for f, imp in zip(feats, r.importances_mean)
                   if imp <= 0]
        print(f"  selection: kept {len(kept)}/{len(feats)} features")
        for f, imp in sorted(zip(feats, r.importances_mean),
                             key=lambda x: -x[1]):
            mark = "keep" if imp > 0 else "DROP"
            print(f"    [{mark}] {f:>22} {imp:>+.4f}")
    return kept or feats


def evaluate(data: pd.DataFrame, feats: list) -> dict | None:
    data = data.dropna(subset=["realized_R"])
    is_ = data[data["year"] <= IS_END_YEAR]
    oos = data[data["year"] >= OOS_START_YEAR]
    if len(is_) < 150 or len(oos) < 60 or is_["win"].nunique() < 2 \
            or oos["win"].nunique() < 2:
        return None
    m = _model().fit(is_[feats], is_["win"])
    p_is = m.predict_proba(is_[feats])[:, 1]
    p_oos = m.predict_proba(oos[feats])[:, 1]
    thr, best = float(np.median(p_is)), -9.9
    for q in np.quantile(p_is, np.linspace(0.3, 0.9, 25)):
        s = is_[p_is >= q]
        if len(s) >= 0.15 * len(is_) and s["realized_R"].mean() > best:
            best, thr = s["realized_R"].mean(), q
    sel = oos[p_oos >= thr]
    base = oos["realized_R"].mean()
    sexp = sel["realized_R"].mean() if len(sel) else np.nan
    net = ((sel["realized_R"] - _cost_r(sel, COST_BPS)).mean()
           if len(sel) else np.nan)
    return {"auc": roc_auc_score(oos["win"], p_oos), "edge": sexp - base,
            "sel_exp": sexp, "net": net, "kept": len(sel) / len(oos),
            "sel_wr": sel["win"].mean() if len(sel) else np.nan,
            "n_oos": len(oos), "n_sel": len(sel)}


def _features(data: pd.DataFrame) -> list:
    return [c for c in data.columns
            if c not in NON_FEATURES and data[c].notna().any()]


def main() -> None:
    crypto = os.environ.get("CRYPTO", "0") not in ("0", "", "false", "no")
    if not crypto:
        print("Set CRYPTO=1 / CRYPTO_GLOB / ORDERFLOW=1."); return
    from crypto_loader import load_binance_klines
    df = load_binance_klines(os.environ.get("CRYPTO_GLOB", "data/btc"))
    tf = os.environ.get("BASE_TF", "30min")
    print(f"timeframe {tf} | horizon {HORIZON}m | order-flow "
          f"{'on' if 'of_cvd' in df.columns else 'OFF'} | cost {COST_BPS}bps")
    data = label_setups_directional(df, detect_setups(df), HORIZON,
                                    continuous=True).dropna(
                                        subset=["realized_R"])
    feats = _features(data)
    print(f"setups {len(data)} | all features {len(feats)}\n")

    print("FEATURE SELECTION (IS-only, drop importance<=0):")
    kept = select_features(data, feats, verbose=True)

    full = evaluate(data, feats)
    selr = evaluate(data, kept)
    print(f"\nOOS (2024+):  AUC      edge      sel_exp   net@{int(COST_BPS)}bps"
          f"   kept   win")
    for name, r in (("all-feats", full), ("selected", selr)):
        if r:
            print(f"  {name:>9}: {r['auc']:.3f}   {r['edge']:+.3f}    "
                  f"{r['sel_exp']:+.3f}     {r['net']:+.3f}     "
                  f"{r['kept']:.0%}   {r['sel_wr']:.0%}")

    # matched 24/7 noise null on the SELECTED set (control, not a sweep)
    sigma = float(df["mid_c"].pct_change().std())
    start = df.index[0].strftime("%Y-%m-%d")
    span = (df.index[-1] - df.index[0]).days + 1
    print(f"\nNoise null ({NULLS} matched 24/7 runs)...")
    ne, na = [], []
    for k in range(NULLS):
        sd = make_synthetic_minutes(n_days=span, seed=9000 + k,
                                    start_date=start, sigma_frac=sigma,
                                    crypto=True)
        nd = label_setups_directional(sd, detect_setups(sd), HORIZON,
                                      continuous=True).dropna(
                                          subset=["realized_R"])
        nf = _features(nd)
        nk = select_features(nd, nf)
        r = evaluate(nd, nk)
        if r:
            ne.append(r["edge"]); na.append(r["auc"])
        print(f"  null {k+1}/{NULLS}: "
              + (f"AUC {r['auc']:.3f} edge {r['edge']:+.3f}" if r else "skip"))

    if selr and ne:
        p_auc = float(np.mean([a >= selr["auc"] for a in na]))
        p_edge = float(np.mean([e >= selr["edge"] for e in ne]))
        print("\n" + "=" * 64)
        print(" SELECTED-MODEL VERDICT (single 60m config, net of cost)")
        print("=" * 64)
        print(f" OOS AUC {selr['auc']:.3f}  (noise95 {np.quantile(na,.95):.3f}"
              f", p={p_auc:.2f})")
        print(f" OOS edge {selr['edge']:+.3f}R (noise95 "
              f"{np.quantile(ne,.95):+.3f}, p={p_edge:.2f})")
        print(f" net selected expectancy @ {int(COST_BPS)}bps: "
              f"{selr['net']:+.3f}R on {selr['n_sel']} trades")
        good = p_auc < 0.05 and p_edge < 0.05 and selr["net"] > 0
        print("-" * 64)
        print(" VERDICT: real, net-positive edge beating noise."
              if good else
              " VERDICT: no clear net edge beyond noise on this config.")
        print("=" * 64)


if __name__ == "__main__":
    main()
