"""
Train and persist the per-asset models the live runner loads. Kline-only
features (no tick/L2), so the same model works on any Binance pair without
a real-time trade-stream aggregator. Trained on ALL available history --
the walk-forward already proved the edge out-of-sample, so for deployment
we use every bar.

    CRYPTO=1 python train_live_models.py     -> models/<asset>_15min.joblib
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

import backtest_ftmo as B

ASSETS = ["BTC", "ETH", "SOL", "BNB", "LTC"]
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
    return [c for c in data.columns
            if c not in B.NON_FEATURES and c not in STREAM and c != "asset"
            and data[c].dtype.kind in "fiu" and data[c].notna().any()]


def main():
    os.makedirs("models", exist_ok=True)
    # train whatever trade data is present (parquets ship in the repo so the
    # models can be rebuilt locally -- avoids cross-version pickle errors).
    assets = [a for a in ASSETS
              if os.path.exists(f"data/trades/{a}_15min.parquet")]
    common = None
    cols = {}
    for a in assets:
        d = pd.read_parquet(f"data/trades/{a}_15min.parquet")
        cols[a] = set(kline_feats(d))
        common = cols[a] if common is None else (common & cols[a])
    feats = sorted(common)                      # identical feature set for all
    print(f"kline-only feature set ({len(feats)}): {feats}\n")
    for a in assets:
        d = pd.read_parquet(f"data/trades/{a}_15min.parquet").dropna(
            subset=["realized_R"])
        m = B._model().fit(d[feats], d["win"])
        p = m.predict_proba(d[feats])[:, 1]
        # strict selection: keep only the top (1-STRICT_Q) highest-conviction
        # signals. 0.88 -> top 12%, the prop-optimised setting (PF ~1.9, win
        # ~65%, ~3% max DD) that scales cleanly to large funded allocations.
        strict_q = float(os.environ.get("STRICT_Q", "0.88"))
        thr = float(np.quantile(p, strict_q))
        kept = (p >= thr).mean()
        joblib.dump({"model": m, "threshold": thr, "feats": feats,
                     "asset": a, "timeframe": "15min", "target_rr": 2.0},
                    f"models/{a}_15min.joblib")
        print(f"{a}: trained on {len(d)} trades | threshold {thr:.3f} "
              f"(keeps {kept:.0%}) -> models/{a}_15min.joblib")


if __name__ == "__main__":
    main()
