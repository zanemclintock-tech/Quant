"""
Stage 1 inspection — see the 30-min sweep / order-block setups firing.

    python inspect_smc.py

Prints how many setups the detector finds, the long/short split, setups
per year, and a FIRST-LOOK raw win rate using a coarse 30-min
triple-barrier (target = RR x risk, stop = the setup's stop, intraday).
It also splits each feature into winners vs losers, so we can already
see which inputs (std-dev pullback, equilibrium distance, volume, ...)
lean toward predicting success BEFORE we train anything.

This raw win rate is NOT the strategy result — fills are approximated on
30-min bars and every setup is taken. It is a sanity check that the
setup exists and is worth modelling. Realistic 1-min fills and the model
come in the next stages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import load_bid_ask
from run_backtest import autodetect_csvs
from smc_detector import detect_setups, to_m30

RR_TO_CHECK = (1.0, 1.5, 2.0, 3.0)


def coarse_outcome(df: pd.DataFrame, setups, rr: float) -> pd.Series:
    """Label each setup win(1)/loss(0) on 30-min bars: from the bar AFTER
    entry, did price reach entry +/- rr*risk (TP) before the stop? Stop
    checked before target within a bar; same-day only (no overnight)."""
    m = to_m30(df)
    pos = {t: i for i, t in enumerate(m.index)}
    h, l = m["h"].values, m["l"].values
    dates = np.array([t.date() for t in m.index])
    out = []
    for s in setups:
        i = pos.get(s.entry_time)
        if i is None:
            out.append(np.nan); continue
        risk = abs(s.entry_price - s.stop)
        tp = (s.entry_price - rr * risk if s.direction == -1
              else s.entry_price + rr * risk)
        res = np.nan
        for k in range(i + 1, len(m)):
            if dates[k] != dates[i]:
                break
            if s.direction == -1:                 # short
                if h[k] >= s.stop:   res = 0; break
                if l[k] <= tp:       res = 1; break
            else:                                  # long
                if l[k] <= s.stop:   res = 0; break
                if h[k] >= tp:       res = 1; break
        out.append(res)
    return pd.Series(out)


def main() -> None:
    bid, ask = autodetect_csvs()
    if not (bid and ask):
        raise SystemExit("Put the Dukascopy Bid/Ask CSVs in this folder.")
    print(f"Loading {bid} / {ask} ...")
    df = load_bid_ask(bid, ask)
    has_vol = "volume" in df.columns
    print(f"{len(df):,} 1-min bars {df.index[0].date()} -> "
          f"{df.index[-1].date()} | volume column: "
          f"{'YES (tick volume)' if has_vol else 'no'}\n")

    setups = detect_setups(df)
    if not setups:
        print("No setups detected — check the data loaded correctly.")
        return
    n = len(setups)
    longs = sum(1 for s in setups if s.direction == 1)
    print(f"Detected {n} setups  ({longs} long / {n - longs} short)  "
          f"= {n / max((df.index[-1] - df.index[0]).days / 365.25, 1):.0f}/yr")

    yrs = pd.Series([s.entry_time.year for s in setups]).value_counts()
    print("\nSetups per year:")
    print(yrs.sort_index().to_string())

    print("\nFirst-look raw win rate (coarse 30-min fills, every setup taken):")
    print(f"{'RR':>5} {'win%':>7} {'expectancy(R)':>15}")
    feats = pd.DataFrame([s.features for s in setups])
    for rr in RR_TO_CHECK:
        y = coarse_outcome(df, setups, rr).dropna()
        if y.empty:
            continue
        wr = y.mean()
        exp_r = wr * rr - (1 - wr) * 1.0      # R won vs 1R risked
        print(f"{rr:>5.1f} {wr:>6.1%} {exp_r:>+15.2f}")

    # which features separate winners from losers at RR=2 (a first hint)
    y2 = coarse_outcome(df, setups, 2.0)
    mask = y2.notna()
    fy = feats[mask.values].copy()
    fy["win"] = y2[mask].values
    print("\nFeature means — winners vs losers at RR=2 "
          "(bigger gap = more predictive):")
    cols = [c for c in fy.columns if c not in ("win", "dir")]
    summary = fy.groupby("win")[cols].mean().T
    summary.columns = ["loser", "winner"]
    summary["gap"] = (summary["winner"] - summary["loser"]).abs()
    print(summary.sort_values("gap", ascending=False).round(3).to_string())
    print("\nNote: this is the pre-model sanity check. If some features show "
          "a real gap, the model has something to learn. Next stage: 1-min "
          "fills + train/test the model and measure if it beats taking all.")


if __name__ == "__main__":
    main()
