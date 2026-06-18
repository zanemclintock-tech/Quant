"""
Stage 3b — is the edge CONSISTENT across years, or a one-window fluke?

The single 2024+ split hinted at an expectancy edge but it could be luck.
This runs an expanding walk-forward: for each test year Y, train only on
the years before Y and score year Y once. A real edge shows up in most
years; a fluke shows up in one. Each year is compared to the same
machinery on random-walk noise (which, as we learned, manufactures a
positive 'edge' on its own) so we judge real vs that noise per year.

    python walk_forward.py            (NULL_RUNS env var, default 6)
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from data_loader import load_bid_ask, attach_secondary
from labeler import label_setups, label_setups_directional, TARGET_RR
from run_backtest import autodetect_csvs, autodetect_sp_csvs
from smc_detector import detect_setups
from synthetic import make_synthetic_minutes

warnings.filterwarnings("ignore")

NON_FEATURES = {"entry_time", "year", "realized_R", "win", "outcome",
                "entry_px", "risk_px"}
TEST_YEARS = [2022, 2023, 2024, 2025]
NULL_RUNS = int(os.environ.get("NULL_RUNS", "6"))
LABEL_MODE = os.environ.get("LABEL", "directional")
HORIZON = int(os.environ.get("HORIZON", "60"))
# Must match how model_train builds its null: a 24/7 crypto surrogate has
# 1440 bars/day and no overnight gap, a session surrogate has ~390 with
# gaps. Comparing the 24/7 BTC strategy to a session null is apples to
# oranges and manufactures a fake "edge", so thread the flag through here.
CRYPTO = os.environ.get("CRYPTO", "0") not in ("0", "", "false", "no")


def labelled(df: pd.DataFrame):
    setups = detect_setups(df)
    data = (label_setups_directional(df, setups, HORIZON, continuous=CRYPTO)
            if LABEL_MODE == "directional"
            else label_setups(df, setups, continuous=CRYPTO)
            ).dropna(subset=["realized_R"])
    feat = [c for c in data.columns
            if c not in NON_FEATURES and data[c].notna().any()]
    return data, feat


def wf_edges(data: pd.DataFrame, feat: list) -> dict:
    """Per test year: expanding train on prior years, score that year
    once. Returns {year: (edge, sel_exp, base_exp, sel_win, n_oos)}."""
    out = {}
    for ty in TEST_YEARS:
        tr = data[data["year"] < ty]
        te = data[data["year"] == ty]
        if len(tr) < 200 or len(te) < 40 or tr["win"].nunique() < 2:
            continue
        m = HistGradientBoostingClassifier(
            max_depth=3, max_iter=250, learning_rate=0.04,
            l2_regularization=1.0, min_samples_leaf=40,
            early_stopping=True, validation_fraction=0.2, random_state=0)
        m.fit(tr[feat], tr["win"])
        p_tr = m.predict_proba(tr[feat])[:, 1]
        p_te = m.predict_proba(te[feat])[:, 1]
        thr, best = float(np.median(p_tr)), -9.9
        for q in np.quantile(p_tr, np.linspace(0.3, 0.9, 25)):
            s = tr[p_tr >= q]
            if len(s) >= 0.15 * len(tr) and s["realized_R"].mean() > best:
                best, thr = s["realized_R"].mean(), q
        sel = te[p_te >= thr]
        base = te["realized_R"].mean()
        se = sel["realized_R"].mean() if len(sel) else np.nan
        out[ty] = (se - base, se, base,
                   sel["win"].mean() if len(sel) else np.nan, len(sel))
    return out


def main() -> None:
    from model_train import load_primary_with_secondary
    df = load_primary_with_secondary()
    if df is None:
        return

    lab = f"directional {HORIZON}m" if LABEL_MODE == "directional" else f"{TARGET_RR:.0f}R"
    print(f"\nReal walk-forward (label {lab}, expanding train)...")
    data, feat = labelled(df)
    real = wf_edges(data, feat)

    sigma = float(df["mid_c"].pct_change().std())
    # surrogate must span the same calendar range as the real data so the
    # null covers every TEST_YEAR (1400 *calendar* days only reaches 2024).
    start = df.index[0].strftime("%Y-%m-%d")
    span = (df.index[-1] - df.index[0]).days + 1 if CRYPTO else 1400
    print(f"\nNoise null walk-forward ({NULL_RUNS} {'24/7 ' if CRYPTO else ''}"
          f"runs of {span} days, sigma~{sigma:.5f})...")
    null_by_year = {ty: [] for ty in TEST_YEARS}
    null_means = []
    for k in range(NULL_RUNS):
        sd = make_synthetic_minutes(n_days=span, seed=2000 + k,
                                    start_date=start, sigma_frac=sigma,
                                    crypto=CRYPTO)
        e = wf_edges(*labelled(sd))
        for ty, vals in e.items():
            null_by_year[ty].append(vals[0])
        if e:
            null_means.append(np.mean([v[0] for v in e.values()]))
        print(f"  null {k + 1}/{NULL_RUNS}: "
              + " ".join(f"{ty}:{e[ty][0]:+.2f}" for ty in e))

    print("\n" + "=" * 70)
    print(" PER-YEAR EDGE vs NOISE  (edge = selected expectancy - take-all)")
    print("=" * 70)
    print(f" {'year':>5} {'edge(R)':>9} {'sel_exp':>9} {'base':>8} "
          f"{'sel_win':>8} {'n':>5} {'noise95':>9} {'beats?':>7}")
    beats = 0
    for ty in TEST_YEARS:
        if ty not in real:
            continue
        edge, se, base, win, n = real[ty]
        n95 = (np.quantile(null_by_year[ty], 0.95)
               if null_by_year[ty] else np.nan)
        ok = np.isfinite(n95) and edge > n95
        beats += ok
        print(f" {ty:>5} {edge:>+9.3f} {se:>+9.3f} {base:>+8.3f} "
              f"{win:>8.1%} {n:>5} {n95:>+9.3f} {'YES' if ok else 'no':>7}")

    real_mean = np.mean([real[ty][0] for ty in real]) if real else np.nan
    p_mean = (np.mean([m >= real_mean for m in null_means])
              if null_means else np.nan)
    print("-" * 70)
    print(f" Real mean edge across years: {real_mean:+.3f}R   "
          f"beats noise in {beats}/{len(real)} years")
    print(f" p(noise mean-edge >= real mean-edge) = {p_mean:.3f}")
    print("-" * 70)
    if beats >= max(3, len(real) - 1) and p_mean < 0.05:
        print(" VERDICT: edge is CONSISTENT across years and beats noise.")
        print(" This is real, repeatable signal. Proceed to Stage 4:")
        print(" a full costed backtest of just the model-selected trades")
        print(" under the 6%/3%/0.5% risk rules.")
    else:
        print(" VERDICT: the edge is NOT consistently above noise across")
        print(" years. The single-window hint did not replicate. Honestly:")
        print(" not a dependable edge to trade.")
    print("=" * 70)


if __name__ == "__main__":
    main()
