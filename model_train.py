"""
Stage 3 — does the model learn a REAL edge, beyond backtest artifacts?

Hard lesson baked into this test: conservative fill/label rules
(stop-checked-before-target, end-of-day exit, dropping unresolved
trades) all correlate with the causal features, so an ML model extracts
positive-looking "edge" from them — and it GENERALIZES out-of-sample,
because the artifact is mechanical, not market-driven. We proved this:
the same pipeline shows OOS AUC ~0.6 and positive selected expectancy on
pure random-walk data that contains no signal whatsoever.

So an absolute number (AUC, expectancy) means nothing on its own. The
only honest question is: does the real data beat what this exact
machinery produces on NOISE?

Method:
  * run the full pipeline (detect -> label with 1-min fills -> train on
    2020-2023 -> score once on 2024+) on the REAL data.
  * run it on N random-walk surrogates (no signal, same machinery) to
    build the NULL distribution of OOS AUC and OOS expectancy edge.
  * p-value = fraction of surrogates that match or beat the real result.
    The model is judged to have learned a real edge only if it beats the
    noise null at p < 0.05 on BOTH AUC and expectancy.

Run:  python model_train.py            (NULL_RUNS env var, default 10)
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
from run_backtest import autodetect_csvs, autodetect_sp_csvs, find_bid_ask
from smc_detector import detect_setups
from synthetic import make_synthetic_minutes

warnings.filterwarnings("ignore")

IS_END_YEAR = 2023
OOS_START_YEAR = 2024
NON_FEATURES = {"entry_time", "year", "realized_R", "win", "outcome"}
NULL_RUNS = int(os.environ.get("NULL_RUNS", "10"))

# LABEL=directional uses the clean directional target (recommended for the
# learning test); LABEL=bracket uses the realistic 2R/stop/EOD outcome.
LABEL_MODE = os.environ.get("LABEL", "directional")
HORIZON = int(os.environ.get("HORIZON", "60"))
# PRIMARY=sp tests the S&P 500 as the traded instrument (NAS becomes the
# correlated secondary for SMT); default is NAS100.
PRIMARY = os.environ.get("PRIMARY", "nas").lower()
# CRYPTO=1 loads free Binance klines (real volume + order flow) from
# CRYPTO_GLOB (a folder or glob of 1m kline CSVs); the null goes 24/7.
CRYPTO = os.environ.get("CRYPTO", "0") not in ("0", "", "false", "no")


def load_primary_with_secondary():
    """Load the primary instrument's 1-min bid/ask and attach the other
    index as the correlated secondary (for SMT divergence). PRIMARY env
    swaps which one is traded; INSTRUMENT=<name> tests any instrument
    whose CSVs contain <name> (e.g. INSTRUMENT=chfjpy), with optional
    SECONDARY=<name> for SMT."""
    if CRYPTO:
        from crypto_loader import load_binance_klines
        pathg = os.environ.get("CRYPTO_GLOB") or os.environ.get("INSTRUMENT")
        if not pathg:
            print("Set CRYPTO_GLOB to your Binance 1m kline folder/glob.")
            return None
        df = load_binance_klines(
            pathg, float(os.environ.get("CRYPTO_SPREAD_BPS", "1.0")))
        print(f"Crypto klines: {pathg} | {len(df):,} bars "
              f"{df.index[0]} -> {df.index[-1]}")
        print(f"volume: {'volume' in df.columns} | "
              f"order-flow: {'taker_buy' in df.columns}")
        return df
    inst = os.environ.get("INSTRUMENT")
    if inst:
        pbid, pask = find_bid_ask(inst)
        if not (pbid and pask):
            print(f"Could not find bid/ask CSVs matching '{inst}'.")
            return None
        print(f"Primary = {inst}: {pbid} / {pask}")
        df = load_bid_ask(pbid, pask)
        sec = os.environ.get("SECONDARY")
        if sec:
            sbid, sask = find_bid_ask(sec)
            if sbid and sask:
                df = attach_secondary(df, sbid, sask, "sp")
                print(f"Attached {sec} as secondary for SMT.")
        print(f"{len(df):,} bars | volume: {'volume' in df.columns} | "
              f"secondary: {'sp_c' in df.columns}")
        return df
    nas, sp = autodetect_csvs(), autodetect_sp_csvs()
    if PRIMARY in ("sp", "sp500", "spx"):
        (pbid, pask), (sbid, sask) = sp, nas
        pname, sname = "S&P 500", "NAS100"
    else:
        (pbid, pask), (sbid, sask) = nas, sp
        pname, sname = "NAS100", "S&P 500"
    if not (pbid and pask):
        print(f"Could not find {pname} bid/ask CSVs in this folder.")
        return None
    print(f"Primary = {pname}: {pbid} / {pask}")
    df = load_bid_ask(pbid, pask)
    if sbid and sask:
        df = attach_secondary(df, sbid, sask, "sp")
        print(f"Attached {sname} for SMT divergence features.")
    print(f"{len(df):,} bars | volume: {'volume' in df.columns} | "
          f"secondary: {'sp_c' in df.columns}")
    return df


def _label(df):
    if LABEL_MODE == "directional":
        return label_setups_directional(df, detect_setups(df), HORIZON,
                                        continuous=CRYPTO)
    return label_setups(df, detect_setups(df))


def _fit(X, y):
    m = HistGradientBoostingClassifier(
        max_depth=3, max_iter=250, learning_rate=0.04,
        l2_regularization=1.0, min_samples_leaf=40,
        early_stopping=True, validation_fraction=0.2, random_state=0)
    m.fit(X, y)
    return m


def run_pipeline(df: pd.DataFrame, verbose: bool = False) -> dict | None:
    """One full detect -> label -> train(IS) -> score(OOS) pass.
    Returns OOS AUC and the model's OOS expectancy edge over taking all."""
    if verbose:
        setups = detect_setups(df)
        print(f"  detected {len(setups)} setups, labelling...", flush=True)
        data = (label_setups_directional(df, setups, HORIZON)
                if LABEL_MODE == "directional"
                else label_setups(df, setups)).dropna(subset=["realized_R"])
        print(f"  labelled {len(data)} filled setups, training...", flush=True)
    else:
        data = _label(df).dropna(subset=["realized_R"])
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

    # threshold chosen on IN-SAMPLE only (max IS expectancy, >=15% kept)
    best_thr, best = float(np.median(p_is)), -9.9
    for thr in np.quantile(p_is, np.linspace(0.3, 0.9, 25)):
        sel = is_[p_is >= thr]
        if len(sel) >= 0.15 * len(is_) and sel["realized_R"].mean() > best:
            best, best_thr = sel["realized_R"].mean(), thr

    sel_oos = oos[p_oos >= best_thr]
    base_exp = oos["realized_R"].mean()
    sel_exp = sel_oos["realized_R"].mean() if len(sel_oos) else np.nan
    return {
        "auc": roc_auc_score(oos["win"], p_oos),
        "edge": sel_exp - base_exp,
        "base_exp": base_exp, "sel_exp": sel_exp,
        "n_oos": len(oos), "n_sel": len(sel_oos),
        "kept": len(sel_oos) / len(oos), "thr": best_thr,
        "feat": feat, "model": model,
        "base_wr": oos["win"].mean(),
        "sel_wr": sel_oos["win"].mean() if len(sel_oos) else np.nan,
    }


def build_null(span_days: int, sigma_frac: float, n: int,
               crypto: bool = False) -> pd.DataFrame:
    """Run the identical pipeline on n random-walk surrogates."""
    out = []
    for k in range(n):
        df = make_synthetic_minutes(
            n_days=span_days, seed=1000 + k, start_date="2020-09-01",
            sigma_frac=sigma_frac, crypto=crypto)
        r = run_pipeline(df)
        if r:
            out.append({"auc": r["auc"], "edge": r["edge"],
                        "sel_exp": r["sel_exp"], "base_exp": r["base_exp"]})
        print(f"  null {k + 1}/{n}: "
              + (f"AUC {r['auc']:.3f} base {r['base_exp']:+.3f} "
                 f"edge {r['edge']:+.3f}"
                 if r else "skipped"))
    return pd.DataFrame(out)


def main() -> None:
    df = load_primary_with_secondary()
    if df is None:
        return

    lab = (f"directional {HORIZON}m" if LABEL_MODE == "directional"
           else f"bracket {TARGET_RR:.0f}R")
    print(f"\nReal data: detect -> label ({lab}) -> train(<= "
          f"{IS_END_YEAR}) -> score({OOS_START_YEAR}+)...")
    real = run_pipeline(df, verbose=True)
    if real is None:
        print("Not enough setups to evaluate."); return

    # match the null's volatility scale to the real data
    sigma = float(df["mid_c"].pct_change().std())
    if CRYPTO:                     # 24/7 surrogates spanning the real range
        surrogate_days = min(int((df.index[-1] - df.index[0]).days) + 1, 2000)
    else:
        surrogate_days = 1400      # ~5.5y of business days
    print(f"\nBuilding NOISE null ({NULL_RUNS} {'24/7 ' if CRYPTO else ''}"
          f"random-walk runs of {surrogate_days} days, sigma~{sigma:.5f})...")
    null = build_null(surrogate_days, sigma, NULL_RUNS, crypto=CRYPTO)

    def pval(col, val):
        if null.empty:
            return float("nan")
        return float((null[col] >= val).mean())

    p_auc = pval("auc", real["auc"])
    p_edge = pval("edge", real["edge"])
    p_base = pval("base_exp", real["base_exp"])

    print("\n" + "=" * 68)
    print(" DOES THE MODEL LEARN A REAL EDGE?  (real vs noise null)")
    print("=" * 68)
    print(f" REAL   OOS AUC {real['auc']:.3f} | selected expectancy "
          f"{real['sel_exp']:+.3f}R | edge {real['edge']:+.3f}R "
          f"(took {real['kept']:.0%}, win {real['sel_wr']:.1%})")
    print(f" baseline take-all: expectancy {real['base_exp']:+.3f}R, "
          f"win {real['base_wr']:.1%}")
    if not null.empty:
        print(f"\n NOISE null ({len(null)} runs):")
        print(f"   AUC  mean {null['auc'].mean():.3f}  "
              f"95th pct {null['auc'].quantile(.95):.3f}")
        print(f"   baseline take-all mean {null['base_exp'].mean():+.3f}R  "
              f"95th pct {null['base_exp'].quantile(.95):+.3f}R")
        print(f"   edge mean {null['edge'].mean():+.3f}R  "
              f"95th pct {null['edge'].quantile(.95):+.3f}R")
        print(f"\n  >>> The 'profit' is GROSS (no costs) and noise makes the")
        print(f"      same: p(noise baseline >= real baseline) = {p_base:.3f}.")
        print(f"      If this is ~0.3+, your positive expectancy is the")
        print(f"      measurement artifact, not edge.")
        print(f"\n p(AUC from noise >= real)  = {p_auc:.3f}")
        print(f" p(edge from noise >= real) = {p_edge:.3f}")
    learned = (not null.empty and p_auc < 0.05 and p_edge < 0.05
               and real["sel_exp"] > 0)
    print("-" * 68)
    if learned:
        print(" VERDICT: real edge BEATS the noise null on both measures.")
        print(" The model is learning something the market actually")
        print(" contains, not a backtest artifact. Next: walk-forward")
        print(" across years, then Stage-4 costed backtest of the trades.")
    else:
        print(" VERDICT: the real result does NOT clearly beat noise.")
        print(" Whatever the model 'found' is within the range this same")
        print(" machinery produces on data with zero signal. Honestly:")
        print(" no demonstrated edge. This is the answer the old in-sample")
        print(" pipeline could never give you.")
    print("=" * 68)


if __name__ == "__main__":
    main()
