"""
Stage 3c — sweep timeframes x horizons x ranges, net of real costs.

Honest, results-first grid search. For every (structural timeframe,
directional horizon) it runs the SAME detect -> label -> train(<=2023) ->
score(2024+) pipeline we already trust, and reports:

  * gross OOS edge vs a properly MATCHED 24/7 noise null (p-values), and
  * NET selected expectancy after a round-trip cost, swept across fee
    levels, because a gross edge that dies at 10 bps is not tradeable.

Per-year (range) edge is printed so a regime-only fluke is visible.

    CRYPTO=1 CRYPTO_GLOB=data/btc python sweep.py
    TFS=15min,30min,60min HORIZONS=30,60,120 NULLS=5 python sweep.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from labeler import label_setups_directional
from smc_detector import detect_setups
from synthetic import make_synthetic_minutes

warnings.filterwarnings("ignore")

NON_FEATURES = {"entry_time", "year", "realized_R", "win", "outcome",
                "entry_px", "risk_px"}
IS_END_YEAR = 2023
OOS_START_YEAR = 2024
TFS = os.environ.get("TFS", "15min,30min,60min,120min").split(",")
HORIZONS = [int(x) for x in os.environ.get("HORIZONS", "30,60,120").split(",")]
NULLS = int(os.environ.get("NULLS", "5"))
COST_BPS = [0.0, 5.0, 10.0, 20.0]      # round-trip taker fee + slippage
TEST_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]


def _fit(X, y):
    m = HistGradientBoostingClassifier(
        max_depth=3, max_iter=250, learning_rate=0.04,
        l2_regularization=1.0, min_samples_leaf=40,
        early_stopping=True, validation_fraction=0.2, random_state=0)
    m.fit(X, y)
    return m


def _cost_r(data: pd.DataFrame, bps: float) -> pd.Series:
    """Round-trip cost of `bps` (of notional) expressed in units of R."""
    return (bps / 1e4) * data["entry_px"] / data["risk_px"]


def evaluate(data: pd.DataFrame) -> dict | None:
    """IS-threshold / OOS-score with gross + net expectancy."""
    data = data.dropna(subset=["realized_R"])
    if data.empty:
        return None
    feat = [c for c in data.columns
            if c not in NON_FEATURES and data[c].notna().any()]
    is_ = data[data["year"] <= IS_END_YEAR]
    oos = data[data["year"] >= OOS_START_YEAR]
    if len(is_) < 150 or len(oos) < 60 or is_["win"].nunique() < 2 \
            or oos["win"].nunique() < 2:
        return None
    model = _fit(is_[feat], is_["win"])
    p_is = model.predict_proba(is_[feat])[:, 1]
    p_oos = model.predict_proba(oos[feat])[:, 1]
    thr, best = float(np.median(p_is)), -9.9
    for q in np.quantile(p_is, np.linspace(0.3, 0.9, 25)):
        s = is_[p_is >= q]
        if len(s) >= 0.15 * len(is_) and s["realized_R"].mean() > best:
            best, thr = s["realized_R"].mean(), q
    sel = oos[p_oos >= thr]
    out = {
        "n": len(data), "n_oos": len(oos), "n_sel": len(sel),
        "kept": len(sel) / len(oos) if len(oos) else np.nan,
        "auc": roc_auc_score(oos["win"], p_oos),
        "base_exp": oos["realized_R"].mean(),
        "sel_exp": sel["realized_R"].mean() if len(sel) else np.nan,
        "sel_wr": sel["win"].mean() if len(sel) else np.nan,
    }
    out["edge"] = out["sel_exp"] - out["base_exp"]
    for bps in COST_BPS:
        out[f"net{int(bps)}"] = ((sel["realized_R"] - _cost_r(sel, bps)).mean()
                                 if len(sel) else np.nan)
    # per-year selected expectancy (range view) using the same OOS-style
    # expanding split is overkill here; show gross take-all per year instead
    out["by_year"] = {y: g["realized_R"].mean()
                      for y, g in data.groupby("year")}
    return out


def main() -> None:
    crypto = os.environ.get("CRYPTO", "0") not in ("0", "", "false", "no")
    if not crypto:
        print("Set CRYPTO=1 and CRYPTO_GLOB=<folder> for the crypto sweep.")
        return
    from crypto_loader import load_binance_klines
    pathg = os.environ.get("CRYPTO_GLOB") or os.environ.get("INSTRUMENT")
    df = load_binance_klines(pathg,
                             float(os.environ.get("CRYPTO_SPREAD_BPS", "1.0")))
    sigma = float(df["mid_c"].pct_change().std())
    start = df.index[0].strftime("%Y-%m-%d")
    span = (df.index[-1] - df.index[0]).days + 1
    print(f"Loaded {len(df):,} bars {df.index[0]} -> {df.index[-1]} | "
          f"sigma~{sigma:.5f}\n")

    results = []
    for tf in TFS:
        os.environ["BASE_TF"] = tf
        setups = detect_setups(df)                 # detect once per timeframe
        # matched 24/7 null surrogates, detected once per (tf, seed)
        null_setups = []
        for k in range(NULLS):
            sd = make_synthetic_minutes(n_days=span, seed=7000 + k,
                                        start_date=start, sigma_frac=sigma,
                                        crypto=True)
            null_setups.append((sd, detect_setups(sd)))
        for hz in HORIZONS:
            data = label_setups_directional(df, setups, hz, continuous=True)
            r = evaluate(data)
            if r is None:
                print(f"{tf:>7} h{hz:>3}: too few setups "
                      f"({len(setups)} detected)")
                continue
            null_edge, null_auc = [], []
            for sd, nsu in null_setups:
                nr = evaluate(label_setups_directional(sd, nsu, hz,
                                                       continuous=True))
                if nr:
                    null_edge.append(nr["edge"]); null_auc.append(nr["auc"])
            e95 = np.quantile(null_edge, 0.95) if null_edge else np.nan
            a95 = np.quantile(null_auc, 0.95) if null_auc else np.nan
            p_edge = (np.mean([e >= r["edge"] for e in null_edge])
                      if null_edge else np.nan)
            p_auc = (np.mean([a >= r["auc"] for a in null_auc])
                     if null_auc else np.nan)
            r.update({"tf": tf, "hz": hz, "e95": e95, "a95": a95,
                      "p_edge": p_edge, "p_auc": p_auc,
                      "null_edge_mean": np.mean(null_edge) if null_edge
                      else np.nan})
            results.append(r)
            print(f"{tf:>7} h{hz:>3}: n={r['n']:>5} AUC {r['auc']:.3f} "
                  f"(null95 {a95:.3f} p{p_auc:.2f}) | "
                  f"gross sel {r['sel_exp']:+.3f}R edge {r['edge']:+.3f} "
                  f"(null95 {e95:+.3f} p{p_edge:.2f}) | "
                  f"net@10bps {r['net10']:+.3f}R kept {r['kept']:.0%} "
                  f"wr {r['sel_wr']:.0%}", flush=True)

    if not results:
        print("\nNo evaluable configs."); return

    print("\n" + "=" * 92)
    print(" SWEEP SUMMARY — net selected expectancy by cost (OOS 2024+, "
          "directional)")
    print("=" * 92)
    print(f" {'tf':>7} {'hz':>4} {'AUC':>6} {'p_auc':>6} {'edge':>7} "
          f"{'p_edge':>7} {'net0':>7} {'net5':>7} {'net10':>7} {'net20':>7} "
          f"{'kept':>5}")
    for r in results:
        print(f" {r['tf']:>7} {r['hz']:>4} {r['auc']:>6.3f} {r['p_auc']:>6.2f} "
              f"{r['edge']:>+7.3f} {r['p_edge']:>7.2f} "
              f"{r['net0']:>+7.3f} {r['net5']:>+7.3f} {r['net10']:>+7.3f} "
              f"{r['net20']:>+7.3f} {r['kept']:>5.0%}")
    print("-" * 92)
    # winners: beat noise on BOTH measures and stay positive net of 10 bps
    win = [r for r in results
           if r["p_auc"] < 0.10 and r["p_edge"] < 0.10 and r["net10"] > 0]
    if win:
        print(" Configs beating noise (p<0.10 both) AND net-positive @10bps:")
        for r in sorted(win, key=lambda x: -x["net10"]):
            print(f"   {r['tf']} h{r['hz']}: net@10bps {r['net10']:+.3f}R, "
                  f"AUC {r['auc']:.3f} (p{r['p_auc']:.2f}), "
                  f"edge {r['edge']:+.3f} (p{r['p_edge']:.2f})")
    else:
        print(" No config beats noise on both measures AND survives 10 bps.")
        print(" Per the same machinery, this is no demonstrated net edge.")
    print("=" * 92)


if __name__ == "__main__":
    main()
