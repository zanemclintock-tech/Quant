"""
Cache full bracket-trade sets (features + entry/exit/prices) per asset for
one timeframe, so walk-forward and portfolio sims read them instantly
instead of re-detecting/labelling the 3M-bar history every run.

    CRYPTO=1 TF=5min python cache_trades.py     -> data/trades/<asset>_5min.parquet
"""
from __future__ import annotations

import os

import backtest_ftmo as B
from crypto_loader import load_binance_klines
from smc_detector import detect_setups

ASSETS = [("BTC", "data/btc", True), ("ETH", "data/eth", False),
          ("SOL", "data/sol", False), ("BNB", "data/bnb", False),
          ("LTC", "data/ltc", False)]


def main():
    tf = os.environ.get("TF", "5min")
    os.makedirs("data/trades", exist_ok=True)
    # BTC is the cross-asset reference for the xa_* lead-lag features (it
    # references itself: rel/corr go degenerate-constant, the model ignores
    # them). Only mid_c is read from the reference, so ORDERFLOW=0 is fine.
    os.environ["ORDERFLOW"] = "0"
    os.environ["BASE_TF"] = tf
    btc = load_binance_klines("data/btc")
    for name, folder, of in ASSETS:
        os.environ["ORDERFLOW"] = "1" if of else "0"
        os.environ["BASE_TF"] = tf
        df = load_binance_klines(folder)
        data = B.gen_trades(df, detect_setups(df, xref=btc)).dropna(
            subset=["realized_R"])
        data["asset"] = name
        # entry/exit times must survive parquet round-trip as tz-aware
        out = f"data/trades/{name}_{tf}.parquet"
        data.to_parquet(out)
        oos = (data["year"] >= B.OOS_START).sum()
        print(f"{name} {tf}: {len(data)} trades ({oos} OOS) -> {out}",
              flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
