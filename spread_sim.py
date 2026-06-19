"""
Estimate the bid-ask spread we ACTUALLY face, per coin, at the times the
strategy trades -- then charge that data-driven spread instead of a flat
assumption. No quotes exist in the data, so we use the Corwin-Schultz
(2012) high-low spread estimator: it separates spread from volatility
using two consecutive bars' highs/lows. Reported in bps, sampled at every
trade's entry time, and fed back into the FTMO P&L.

    CRYPTO=1 python spread_sim.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest_ftmo as B
import multi_asset as M
import decay_fix as D
from crypto_loader import load_binance_klines

FOLDERS = {"BTC": "data/btc", "ETH": "data/eth", "SOL": "data/sol",
           "BNB": "data/bnb"}
DEN = 3 - 2 * np.sqrt(2)


def corwin_schultz(h, l, smooth=96):
    """Proportional spread per bar from consecutive high/low. Negative
    estimates clipped to 0, then smoothed over `smooth` bars (~1 day at
    15m) for a stable ambient spread. Returns spread in bps."""
    h, l = np.asarray(h, float), np.asarray(l, float)
    lh = np.log(h / l) ** 2                          # bar range^2
    beta = lh[:-1] + lh[1:]                           # two-bar beta
    h2 = np.maximum(h[:-1], h[1:]); l2 = np.minimum(l[:-1], l[1:])
    gamma = np.log(h2 / l2) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / DEN - np.sqrt(gamma / DEN)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = np.clip(s, 0, None)
    s = np.concatenate([[np.nan], s])                 # align to bar t+1
    return pd.Series(s).rolling(smooth, min_periods=8).mean().to_numpy() * 1e4


def spread_series(folder, tf="15min"):
    df = load_binance_klines(folder)
    m = pd.DataFrame({"h": df["mid_h"].resample(tf).max(),
                      "l": df["mid_l"].resample(tf).min()}).dropna()
    m["spread_bps"] = corwin_schultz(m["h"].values, m["l"].values)
    return m["spread_bps"]


def main():
    print("Estimating spread (Corwin-Schultz) at the trades' own entry "
          "times...\n")
    spr = {a: spread_series(f) for a, f in FOLDERS.items()}
    trades = {a: pd.read_parquet(f"data/trades/{a}_15min.parquet")
              for a in FOLDERS}

    print(f"  {'coin':>5} {'median bps':>11} {'mean bps':>9} {'90th bps':>9} "
          f"{'at trade times (entries)':>26}")
    per_trade = {}
    for a in FOLDERS:
        s = spr[a]
        t = trades[a].copy()
        key = t["entry_time"].dt.floor("15min")
        sp = key.map(s).astype(float)                 # spread at each entry
        per_trade[a] = sp
        print(f"  {a:>5} {sp.median():>11.1f} {sp.mean():>9.1f} "
              f"{sp.quantile(0.9):>9.1f}")

    # spread by UTC hour at trade times (when does it trade, how wide then?)
    print("\n  spread by UTC hour (BTC, at entries) -- liquidity clock:")
    t = trades["BTC"].copy()
    t["hr"] = t["entry_time"].dt.hour
    t["sp"] = per_trade["BTC"].values
    by = t.groupby("hr")["sp"].mean()
    line = "  ".join(f"{h:02d}:{by.get(h, np.nan):.0f}" for h in range(0, 24, 3))
    print(f"    {line}")

    # feed the per-trade estimated spread back into the FTMO P&L: limit
    # entry/TP pay ~1bp maker; market stop crosses the estimated spread.
    print("\n=== Net P&L with DATA-DRIVEN per-trade spreads (15min "
          "portfolio, walk-forward) ===")
    sels = []
    for a in FOLDERS:
        s, _ = D.wf_select(trades[a], "expanding")
        if s is None:
            continue
        s = s.copy(); s["asset"] = a
        key = s["entry_time"].dt.floor("15min")
        s["sp_bps"] = key.map(spr[a]).astype(float).fillna(spr[a].median())
        sels.append(s[D.COLS + ["sp_bps"]])
    allt = pd.concat(sels).sort_values("entry_time")
    maker, taker = 1.0, 1.0
    entry_c = maker
    exit_c = np.where(allt["outcome"].values == "tp", maker,
                      allt["sp_bps"].values + taker)
    cost_r = (entry_c + exit_c) / 1e4 * allt["entry_px"].values \
        / allt["risk_px"].values
    allt = allt.copy(); allt["realized_R"] = allt["realized_R"].values - cost_r
    for rf in (0.0025, 0.005):
        B.RISK_FRAC = rf
        r = M.portfolio(allt, rf, (0.0, {"tp": 0.0, "sl": 0.0, "maxhold": 0.0}))
        mo = r["ret"] / 53 * 100
        print(f"  risk {rf*100:.2f}%: total {r['ret']*100:+.0f}% "
              f"(~{mo:+.1f}%/mo fixed base) | maxDD {r['maxdd']*100:.2f}% | "
              f"dViol {r['dviol']} | {'PASS' if r['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
