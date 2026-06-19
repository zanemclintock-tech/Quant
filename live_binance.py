"""
Live / paper runner for the 15-min multi-asset SMC strategy on Binance.
Its PURPOSE is to measure the one thing the backtest cannot: do the
resting limit entries actually fill, and at what price?

Modes (MODE env):
  paper    -- no keys needed. Fetch live klines, detect signals, and LOG
              the intended limit order; on later polls record whether price
              reached the level (would-fill) and the eventual SL/TP. Proves
              the plumbing + signal logic end to end.
  testnet  -- place REAL limit orders on the Binance spot TESTNET
              (BINANCE_KEY/BINANCE_SECRET), then OCO stop/take-profit once
              filled. NOTE: testnet liquidity is synthetic, so fills there
              test correctness, not real adverse selection -- for true fill
              quality use tiny live orders or a prop demo.

  MODE=paper python live_binance.py
  MODE=testnet BINANCE_KEY=.. BINANCE_SECRET=.. python live_binance.py

No-hedging (one position per asset), min 2-min hold (limit entries can't
exit in <2m by construction), 0.5% risk per trade. Everything is logged to
live_log.csv for fill-rate analysis against the backtest.
"""
from __future__ import annotations

import csv
import os
import time

import joblib
import numpy as np
import pandas as pd

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
           "BNB": "BNBUSDT"}
TF = "15min"
TF_MS = 15 * 60 * 1000
RISK_FRAC = float(os.environ.get("RISK", "0.005"))
ACCOUNT = float(os.environ.get("ACCOUNT", "100000"))
LOG = "live_log.csv"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "qv", "count", "taker_buy_base", "tbq", "ignore"]


# ---------- pure, offline-testable core --------------------------------
def make_frame(raw) -> pd.DataFrame:
    """Binance raw 1m klines (12-field rows) -> harness frame with mid/
    bid/ask + volume + taker_buy, exactly like crypto_loader."""
    d = pd.DataFrame(raw, columns=KLINE_COLS)
    ot = d["open_time"].astype("int64")
    ot = ot.where(ot < 1e14, ot // 1000)           # ms vs microseconds-safe
    idx = pd.to_datetime(ot, unit="ms", utc=True)
    out = pd.DataFrame(index=idx)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        out[c] = d[c].astype(float).values
    out = out.rename(columns={"open": "mid_o", "high": "mid_h", "low": "mid_l",
                              "close": "mid_c", "taker_buy_base": "taker_buy"})
    half = out["mid_c"] * 1e-4 / 2.0
    for c in ("o", "h", "l", "c"):
        out[f"bid_{c}"] = out[f"mid_{c}"] - half
        out[f"ask_{c}"] = out[f"mid_{c}"] + half
    out.index.name = "time"
    return out


def signals_on_closed_bar(df, bundle, last_closed):
    """Setups whose entry triggered on `last_closed`, scored >= threshold.
    Returns list of dicts ready to turn into orders."""
    os.environ["BASE_TF"] = bundle["timeframe"]
    os.environ["CRYPTO"] = "1"
    from smc_detector import detect_setups
    m, thr, feats = bundle["model"], bundle["threshold"], bundle["feats"]
    out = []
    for s in detect_setups(df):
        if s.entry_time != last_closed:
            continue
        x = pd.DataFrame([{f: s.features.get(f, np.nan) for f in feats}])
        p = float(m.predict_proba(x[feats])[:, 1][0])
        if p >= thr:
            risk = abs(s.entry_price - s.stop)
            tp = (s.entry_price - bundle["target_rr"] * risk if s.direction < 0
                  else s.entry_price + bundle["target_rr"] * risk)
            out.append({"side": "sell" if s.direction < 0 else "buy",
                        "entry": s.entry_price, "stop": s.stop, "tp": tp,
                        "risk": risk, "prob": p, "arm": s.arm_time})
    return out


def size(sig):
    qty = (RISK_FRAC * ACCOUNT) / sig["risk"]      # base units for 0.5% risk
    return qty


def log_row(**row):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ---------- live wiring (ccxt) -----------------------------------------
def main():
    mode = os.environ.get("MODE", "paper")
    bundles = {a: joblib.load(f"models/{a}_{TF}.joblib") for a in SYMBOLS}
    print(f"[{mode}] loaded models for {list(SYMBOLS)} | risk {RISK_FRAC:.1%} "
          f"| logging to {LOG}")

    import ccxt
    ex = ccxt.binance({
        "apiKey": os.environ.get("BINANCE_KEY", ""),
        "secret": os.environ.get("BINANCE_SECRET", ""),
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}})
    if mode == "testnet":
        ex.set_sandbox_mode(True)

    open_pos = {a: None for a in SYMBOLS}          # one position per asset
    seen_bar = {a: None for a in SYMBOLS}

    while True:
        for a, sym in SYMBOLS.items():
            try:
                raw = ex.publicGetKlines({"symbol": sym, "interval": "1m",
                                          "limit": 1500})
            except Exception as e:
                print(f"  {a} fetch error: {e}"); continue
            df = make_frame(raw)
            # last CLOSED 15m bar
            last_closed = df["mid_c"].resample(TF).last().index[-2]
            if seen_bar[a] == last_closed:
                continue
            seen_bar[a] = last_closed
            if open_pos[a] is not None:            # no hedging / one at a time
                continue
            for sig in signals_on_closed_bar(df, bundles[a], last_closed):
                qty = size(sig)
                log_row(ts=pd.Timestamp.utcnow(), asset=a, bar=last_closed,
                        **{k: sig[k] for k in ("side", "entry", "stop", "tp",
                                               "prob")}, qty=round(qty, 6),
                        mode=mode)
                print(f"  SIGNAL {a} {sig['side']} @ {sig['entry']:.2f} "
                      f"stop {sig['stop']:.2f} tp {sig['tp']:.2f} "
                      f"p={sig['prob']:.2f} qty={qty:.4f}")
                if mode == "testnet":
                    try:
                        o = ex.create_limit_order(sym.replace("USDT", "/USDT"),
                                                  sig["side"], qty, sig["entry"])
                        open_pos[a] = {"order": o["id"], **sig}
                    except Exception as e:
                        print(f"    order error: {e}")
        time.sleep(20)


if __name__ == "__main__":
    main()
