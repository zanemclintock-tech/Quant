"""
RESEARCH ONLY (no live file touched): test the std-dev / fib-extension "band"
confluence as a feature. For each 15m setup we draw the extension grid from the
recent swing (high->low and reversed) at a chosen swing timeframe, using the
user's levels, and measure how close the entry sits to the nearest band
(ATR-normalised). That feature is added to the model and run through the same
walk-forward (strict top-12%, Goat spreads); we compare OOS AUC and sized PF
to the baseline, at swing timeframes 1m / 5m / 15m.
"""
from __future__ import annotations

import os
os.environ.setdefault("CRYPTO", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import backtest_ftmo as B
import goat_test as G
from smc_detector import to_m30, _atr, _tf_minutes
from crypto_loader import load_binance_klines
from live_runner import INIT

ASSETS = {"BTC": "data/btc", "ETH": "data/eth", "SOL": "data/sol"}
SWING_TFS = ["1min", "5min", "15min"]
LB = 40                     # swing lookback (bars at the swing TF)
RATIOS = [1, 0, -1, -1.5, -2, -2.33, -2.5, -3, -4]   # the user's tool levels
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]


def add_fib(klines, trades, swing_tf):
    if klines.index.tz is not None:
        klines = klines.copy()
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)
    m = to_m30(klines, freq=swing_tf)
    hi, lo = m["h"].values, m["l"].values
    times = m.index
    m15 = to_m30(klines, freq="15min")
    atr15 = pd.Series(_atr(m15).values, index=m15.index)

    out = trades.copy()
    et = pd.DatetimeIndex(pd.to_datetime(out["entry_time"], utc=True).dt.tz_localize(None))
    cut = (et - pd.Timedelta(minutes=_tf_minutes(swing_tf))).values   # causal
    eb = times.searchsorted(cut, side="right") - 1
    atr_at = atr15.reindex(et, method="ffill").values
    entry = out["entry_px"].values

    dist = np.full(len(out), np.nan)
    for k in range(len(out)):
        e = int(eb[k]); a = atr_at[k]
        if e < LB or not np.isfinite(a) or a <= 0:
            continue
        sh = float(np.nanmax(hi[e - LB + 1:e + 1]))
        sl = float(np.nanmin(lo[e - LB + 1:e + 1]))
        rng = sh - sl
        if rng <= 0:
            continue
        levels = ([sl + v * rng for v in RATIOS]          # high->low draw
                  + [sh - v * rng for v in RATIOS])        # reversed draw
        dist[k] = min(abs(entry[k] - lv) for lv in levels) / a
    out["fib_dist_atr"] = dist
    out["fib_near"] = (dist < 0.20).astype(float)         # tight band touch
    return out


def wf_metrics(per):
    feats = sorted(set.intersection(*[
        {c for c in d.columns if c not in B.NON_FEATURES and c not in ("asset", "cls")
         and d[c].dtype.kind in "fiu" and d[c].notna().any()} for d in per.values()]))
    picks, aucs = [], []
    for a, d in per.items():
        d = d.dropna(subset=["realized_R"])
        for Y in YEARS:
            tr, te = d[d.year < Y], d[d.year == Y]
            if len(tr) < 200 or len(te) < 20 or tr.win.nunique() < 2:
                continue
            mm = B._model().fit(tr[feats], tr.win)
            ptr = mm.predict_proba(tr[feats])[:, 1]
            pte = mm.predict_proba(te[feats])[:, 1]
            if te.win.nunique() > 1:
                aucs.append(roc_auc_score(te.win, pte))
            thr = np.quantile(ptr, 0.88)
            s = te[pte >= thr].copy(); s["prob"] = pte[pte >= thr]; s["asset"] = a
            picks.append(s)
    allt = pd.concat(picks).sort_values("entry_time").reset_index(drop=True)
    led = G.sequence(allt, G.GOAT_TYPICAL)
    led["ym"] = pd.to_datetime(led.exit_time).dt.strftime("%Y-%m")
    eqc = led.set_index("exit_time")["eq"]
    dd = ((eqc.cummax() - eqc) / eqc.cummax()).max() * 100
    mo = (led.groupby("ym").pnl.sum() / INIT * 100).mean()
    gp = led.loc[led.pnl > 0, "pnl"].sum(); gl = -led.loc[led.pnl < 0, "pnl"].sum()
    r = allt.realized_R; w = (r > 0.1).sum(); ll = (r < -0.1).sum()
    return dict(auc=np.mean(aucs), win=w / (w + ll) * 100, monthly=mo,
                dd=dd, pf=gp / gl if gl else float("inf"), n=len(led))


def main():
    base = {a: pd.read_parquet(f"data/trades/{a}_15min.parquet") for a in ASSETS}
    klines = {a: load_binance_klines(f) for a, f in ASSETS.items()}
    rows = [("baseline (no fib)", wf_metrics(base))]
    print("  baseline done", flush=True)
    for stf in SWING_TFS:
        per = {a: add_fib(klines[a], base[a], stf) for a in ASSETS}
        cov = np.mean([p["fib_near"].mean() for p in per.values()]) * 100
        rows.append((f"+fib swing={stf}", wf_metrics(per)))
        print(f"  fib {stf} done ({cov:.0f}% near a band)", flush=True)

    print("\n=== std-dev/fib band confluence as a feature "
          "(15m setups; BTC/ETH/SOL, Goat, top-12%) ===")
    print(f"{'config':>20} {'OOS AUC':>8} {'win%':>6} {'monthly%':>9} "
          f"{'maxDD%':>7} {'PF':>6} {'trades/yr':>10}")
    for name, x in rows:
        print(f"{name:>20} {x['auc']:>8.3f} {x['win']:>6.1f} {x['monthly']:>8.2f}% "
              f"{x['dd']:>6.2f}% {x['pf']:>6.2f} {x['n']//6:>10}")


if __name__ == "__main__":
    main()
