"""
Train and persist the POOLED "one brain" model the live runner loads. Kline-
only features (no tick/L2), so the same model works on any Binance pair without
a real-time trade-stream aggregator.

One model is trained on BTC+ETH+SOL combined rather than three siloed models:
it sees 3x the data and learns cross-pair structure, which lifts OOS AUC 0.646
-> 0.672 (the alts gain most: SOL +0.047). Trained on ALL available history --
the walk-forward already proved the edge out-of-sample, so for deployment we
use every bar.

    CRYPTO=1 python train_live_models.py           -> models/pooled_15min.joblib
    CRYPTO=1 COINS=BTC,ETH,SOL,BNB ... to change the training pool.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

import backtest_ftmo as B

# default training/trading universe -- the validated 3-coin pool. Extra coins
# add only marginal AUC (+0.008 for 6 more) and the user opted to keep 3.
POOL = [c.strip().upper() for c in
        os.environ.get("COINS", "BTC,ETH,SOL").split(",") if c.strip()]
# features that need tick (aggTrades) or L2 streams -- excluded so the live
# runner only needs ordinary klines.
STREAM = {"of_vol", "of_buy", "of_ntrades", "of_nbuy", "of_maxtrade",
          "of_buymax", "of_sellmax", "of_sell", "of_delta", "of_cvd",
          "tofi_sweep", "tofi_entry", "tofi_leg", "cvd_slope_leg",
          "maxtrade_z_sweep", "bigprint_imb_sweep", "l2_imb_entry",
          "l2_imb_sweep", "l2_imb1_entry", "l2_depth_z_entry",
          # taker-buy split: not in standard ccxt OHLCV, so drop it too so
          # the live model runs on plain candles from ANY exchange.
          "ofi_sweep", "ofi_entry", "ofi_leg"}


def kline_feats(data):
    # arm_* (pending-time research copy) and kf_* (leaky Kalman research
    # features) are excluded so the live model stays exactly as validated.
    return [c for c in data.columns
            if c not in B.NON_FEATURES and c not in STREAM and c != "asset"
            and not c.startswith("arm_") and not c.startswith("kf_")
            and data[c].dtype.kind in "fiu" and data[c].notna().any()]


def main():
    os.makedirs("models", exist_ok=True)
    assets = [a for a in POOL
              if os.path.exists(f"data/trades/{a}_15min.parquet")]
    if not assets:
        raise SystemExit(f"no trade parquets for pool {POOL}")
    # identical feature set across every pooled coin (intersection)
    common = None
    frames = []
    for a in assets:
        d = pd.read_parquet(f"data/trades/{a}_15min.parquet").dropna(
            subset=["realized_R"])
        d["asset"] = a
        s = set(kline_feats(d))
        common = s if common is None else (common & s)
        frames.append(d)
    feats = sorted(common)
    pooled = pd.concat(frames, ignore_index=True)
    print(f"POOLED training on {assets}: {len(pooled)} trades")
    print(f"kline-only feature set ({len(feats)}): {feats}\n")

    m = B._model().fit(pooled[feats], pooled["win"])
    p = m.predict_proba(pooled[feats])[:, 1]
    # strict selection: keep only the top (1-STRICT_Q) highest-conviction
    # signals. 0.88 -> top 12%, the prop-optimised setting (pooled: OOS AUC
    # ~0.67, win ~80%, ~2.8% max DD, every month green at typical spreads).
    strict_q = float(os.environ.get("STRICT_Q", "0.88"))
    thr = float(np.quantile(p, strict_q))
    kept = (p >= thr).mean()
    bundle = {"model": m, "threshold": thr, "feats": feats,
              "coins": assets, "pooled": True, "timeframe": "15min",
              "target_rr": 2.0}
    joblib.dump(bundle, "models/pooled_15min.joblib")
    print(f"pooled: threshold {thr:.3f} (keeps {kept:.0%}) "
          f"-> models/pooled_15min.joblib")


if __name__ == "__main__":
    main()
