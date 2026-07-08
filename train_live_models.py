"""
Train and persist the POOLED "one brain" model the live runner loads, trained
on LIVE outcomes. Kline-only features (no tick/L2), so the same model works on
any Binance pair without a real-time trade-stream aggregator.

Two design choices:
  * POOLED -- one model on BTC+ETH+SOL combined, not three siloed models. It
    sees 3x the data and learns cross-pair structure.
  * LIVE-LABEL -- trained on `live_win` (did the setup actually FILL on a
    retrace and win at the real price), from label_live.py.

    CRYPTO=1 python label_live.py            # (re)build the live labels first
    CRYPTO=1 python train_live_models.py     -> models/pooled_15min.joblib

Falls back to the idealized `_15min.parquet` win labels if the `_live.parquet`
files are absent, so an old checkout still trains something.

HONESTY NOTE (see CRYPTO_AUDIT.md). Earlier this docstring claimed "~2
trades/day, ~18%/mo, ~2.9% max DD, PF ~2.6 -- funded-safe." That number was
measured IN-SAMPLE: the code below fits on every row and then reads the metric
off the SAME rows (no train/test split) -- the exact failure ANALYSIS.md calls
"fatal." Two independent checks refute the edge:
  * take-all expectancy on the live labels is NEGATIVE (BTC -0.066R, ETH
    -0.058R, SOL -0.033R -- the raw strategy loses after costs);
  * the project's own noise-null (model_train.py) does NOT reject noise
    (bracket label: p(AUC>=real)=0.63, p(edge>=real)=0.50).
So this model is a DEMO/research artifact, not a funded-safe edge. To keep it
honest, main() now also prints a held-out OOS number and a NOT-BLESSED verdict
instead of an in-sample headline. Do not fund on the strength of it.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

import backtest_ftmo as B

# default training/trading universe -- the validated 3-coin pool.
POOL = [c.strip().upper() for c in
        os.environ.get("COINS", "BTC,ETH,SOL").split(",") if c.strip()]
# features that need tick (aggTrades) or L2 streams -- excluded so the live
# runner only needs ordinary klines.
STREAM = {"of_vol", "of_buy", "of_ntrades", "of_nbuy", "of_maxtrade",
          "of_buymax", "of_sellmax", "of_sell", "of_delta", "of_cvd",
          "tofi_sweep", "tofi_entry", "tofi_leg", "cvd_slope_leg",
          "maxtrade_z_sweep", "bigprint_imb_sweep", "l2_imb_entry",
          "l2_imb_sweep", "l2_imb1_entry", "l2_depth_z_entry",
          "ofi_sweep", "ofi_entry", "ofi_leg"}
# live-label bookkeeping columns (never features)
LABEL_COLS = {"live_fill", "live_R", "live_win", "outcome",
              "ob_entry", "ob_stop", "ob_risk", "ob_side"}


def kline_feats(data):
    # arm_* (pending-time research) and kf_* (leaky Kalman research) are also
    # excluded so the live model stays exactly as validated.
    return [c for c in data.columns
            if c not in B.NON_FEATURES and c not in STREAM and c not in LABEL_COLS
            and c != "asset" and not c.startswith("arm_")
            and not c.startswith("kf_")
            and data[c].dtype.kind in "fiu" and data[c].notna().any()]


def _load(assets):
    """Prefer live-outcome labels; fall back to idealized win labels."""
    live = all(os.path.exists(f"data/trades/{a}_live.parquet") for a in assets)
    label = "live_win" if live else "win"
    frames = []
    for a in assets:
        path = (f"data/trades/{a}_live.parquet" if live
                else f"data/trades/{a}_15min.parquet")
        d = pd.read_parquet(path)
        if not live:
            d = d.dropna(subset=["realized_R"])
        d["asset"] = a
        frames.append(d)
    return frames, label, live


def main():
    os.makedirs("models", exist_ok=True)
    assets = [a for a in POOL
              if os.path.exists(f"data/trades/{a}_live.parquet")
              or os.path.exists(f"data/trades/{a}_15min.parquet")]
    if not assets:
        raise SystemExit(f"no trade parquets for pool {POOL}")
    frames, label, live = _load(assets)
    common = None
    for d in frames:
        s = set(kline_feats(d))
        common = s if common is None else (common & s)
    feats = sorted(common)
    pooled = pd.concat(frames, ignore_index=True)
    kind = "LIVE-outcome" if live else "idealized (no live labels found)"
    print(f"POOLED {kind} training on {assets}: {len(pooled)} rows, "
          f"base win {pooled[label].mean():.0%}")
    print(f"kline-only feature set ({len(feats)}): {feats}\n")

    m = B._model().fit(pooled[feats], pooled[label])
    p = m.predict_proba(pooled[feats])[:, 1]
    # strict selection: top (1-STRICT_Q). 0.84 (top 16%) is the live-label
    # sweet spot -- ~2 trades/day at the lowest live drawdown. (Ideal-label
    # fallback keeps 0.88.)
    strict_q = float(os.environ.get("STRICT_Q", "0.84" if live else "0.88"))
    # RECALIBRATE the threshold on the most recent 12 months, not all history:
    # predicted prob-levels drift as the market changes, so an all-history
    # threshold silently arms fewer than intended (measured live: ~12% of
    # recent setups vs the 16% target). Setting it on recent data holds the
    # designed arming rate (~2/day) at the SAME monthly and SAME drawdown --
    # validated on the live engine. No effect on money/DD, restores trade count.
    pr = p
    if live and "entry_time" in pooled.columns:
        et = pd.to_datetime(pooled["entry_time"])
        recent = (et >= (et.max() - pd.Timedelta(days=365))).values
        if recent.sum() > 1000:
            pr = p[recent]
    thr = float(np.quantile(pr, strict_q))
    kept = (p >= thr).mean()
    # SIZING CALIBRATION: the confidence->risk dial must span THIS model's
    # selected-trade probability range, or it goes stale. Each retrain/recal
    # shifts the prob scale (base rate, drift), and a fixed dial then floors
    # most trades at the minimum risk -- measured: 66% floored, mean risk
    # 0.43% vs 1.0% intended, ~halving return. Save the selected-prob 10th/90th
    # pct so sizing tracks the model automatically. (This dial only reallocates
    # risk across the selected trades; it cannot create an edge that isn't
    # there -- see the OOS verdict printed below and CRYPTO_AUDIT.md.)
    sel = pr[pr >= thr]
    size_plo = float(np.quantile(sel, 0.10)) if len(sel) > 50 else 0.41
    size_phi = float(np.quantile(sel, 0.90)) if len(sel) > 50 else 0.63
    bundle = {"model": m, "threshold": thr, "feats": feats,
              "coins": assets, "pooled": True, "live_label": live,
              "timeframe": "15min", "target_rr": 2.0,
              "size_plo": size_plo, "size_phi": size_phi}
    joblib.dump(bundle, "models/pooled_15min.joblib")
    print(f"pooled ({kind}): threshold {thr:.3f} (keeps {kept:.0%}) | "
          f"sizing dial {size_plo:.3f}/{size_phi:.3f} "
          f"-> models/pooled_15min.joblib")
    # The training threshold is calibrated to FULL-HISTORY features; live scores
    # features from a rolling window and sits ~0.15 lower, so this threshold
    # arms ~nothing live. Recalibrate the live threshold to the serving
    # distribution or live will barely trade.
    print("  NEXT: run  python calibrate_live_threshold.py  to set the LIVE "
          "arming threshold (else live under-arms ~8x)")

    _honest_oos_verdict(pooled, feats, label, strict_q, live)


def _honest_oos_verdict(pooled, feats, label, strict_q, live):
    """Print a held-out OOS number and an explicit NOT-BLESSED verdict, so this
    script can never again imply an in-sample headline is a real edge. The
    threshold is taken from the TRAIN split only and applied to the untouched
    test split. Positive OOS bracket expectancy is NOT proof -- the bracket
    selection is positive on noise too; see model_train.py's noise-null."""
    if "entry_time" not in pooled.columns:
        return
    yr = pd.to_datetime(pooled["entry_time"]).dt.year
    cut = int(os.environ.get("OOS_CUT_YEAR", "2023"))
    tr, te = pooled[yr <= cut], pooled[yr > cut]
    if len(tr) < 500 or len(te) < 200 or tr[label].nunique() < 2:
        print("\n  (OOS verdict skipped: not enough held-out data)")
        return
    mo = B._model().fit(tr[feats], tr[label])
    ptr = mo.predict_proba(tr[feats])[:, 1]
    pte = mo.predict_proba(te[feats])[:, 1]
    thr = float(np.quantile(ptr, strict_q))
    sel = te[pte >= thr]
    rcol = "live_R" if "live_R" in te.columns else (
        "realized_R" if "realized_R" in te.columns else None)
    base_r = te[rcol].mean() if rcol else float("nan")
    sel_r = sel[rcol].mean() if (rcol and len(sel)) else float("nan")
    print("\n" + "=" * 68)
    print(" HONEST OUT-OF-SAMPLE VERDICT  (train <= %d, test > %d)" % (cut, cut))
    print("=" * 68)
    print(f"  test setups: {len(te)} | selected (top {100*(1-strict_q):.0f}%): "
          f"{len(sel)} | win {sel[label].mean():.1%}")
    if rcol:
        print(f"  take-all expectancy {base_r:+.3f}R | selected {sel_r:+.3f}R")
    print("  NOTE: positive selected expectancy here is NOT proof of edge --")
    print("  the same bracket selection is positive on pure noise. The only")
    print("  valid gate is model_train.py's noise-null, which this strategy")
    print("  does NOT pass. VERDICT: model is a DEMO artifact, NOT funded-safe.")
    print("=" * 68)


if __name__ == "__main__":
    main()
