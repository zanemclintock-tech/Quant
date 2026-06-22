"""
RESEARCH ONLY (touches NO live file): keep the 15-min setups/structure/entries
exactly as they are, and recompute ONLY the volume-profile features
(vwap_dist_atr, poc_dist_atr, in_value_area, va_width_atr) on a different
timeframe's bars over the same wall-clock window. Then run the identical
pipeline (walk-forward, strict top-12%, Goat spreads) and compare.

  baseline = the live 15-min volume profile (cached trades, untouched)
  5m/15m/30m/60m = the same features, volume profile computed on that TF
"""
from __future__ import annotations

import os
os.environ.setdefault("CRYPTO", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import backtest_ftmo as B
import goat_test as G
from smc_detector import to_m30, _atr, _vol_profile, _tf_minutes
from crypto_loader import load_binance_klines
from live_runner import INIT

ASSETS = {"BTC": "data/btc", "ETH": "data/eth", "SOL": "data/sol"}
VP_TFS = ["5min", "15min", "30min", "60min"]
WIN_H = 6.0                       # volume-profile window, same hours at every TF
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]


def recompute_vp(klines, trades, tf):
    """Return a copy of `trades` with the 4 volume-profile features recomputed
    on `tf` bars over a WIN_H-hour window ending at each trade's entry."""
    if klines.index.tz is not None:
        klines = klines.copy()
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)
    m = to_m30(klines, freq=tf)
    h, l, c = m["h"].values, m["l"].values, m["c"].values
    v = m["v"].values if "v" in m.columns else np.ones(len(m))
    vv = np.nan_to_num(v, nan=0.0)
    tp = (h + l + c) / 3.0
    cum_v, cum_tpv = np.cumsum(vv), np.cumsum(tp * vv)
    times = m.index
    m15 = to_m30(klines, freq="15min")
    atr15 = pd.Series(_atr(m15).values, index=m15.index)

    out = trades.copy()
    et = pd.DatetimeIndex(pd.to_datetime(out["entry_time"], utc=True).dt.tz_localize(None))
    eb_arr = times.searchsorted(et.values, side="right") - 1
    atr_at = atr15.reindex(et, method="ffill").values
    d = (out["dir_"] if "dir_" in out else out["dir"]).values
    entry = out["entry_px"].values
    wbars = max(2, int(WIN_H * 60 / _tf_minutes(tf)))

    vw = np.full(len(out), np.nan)
    pocd, inva, vaw = vw.copy(), vw.copy(), vw.copy()
    for k in range(len(out)):
        eb = int(eb_arr[k]); lo = max(0, eb - wbars + 1)
        a = atr_at[k]
        if eb <= lo or not np.isfinite(a) or a <= 0:
            continue
        sv = cum_v[eb] - (cum_v[lo - 1] if lo > 0 else 0.0)
        if sv > 0:
            vwap = (cum_tpv[eb] - (cum_tpv[lo - 1] if lo > 0 else 0.0)) / sv
            vw[k] = (entry[k] - vwap) * d[k] / a
        prof = _vol_profile(l, h, vv, lo, eb)
        if prof is not None:
            poc, vah, val = prof
            pocd[k] = (entry[k] - poc) * d[k] / a
            inva[k] = float(val <= entry[k] <= vah)
            vaw[k] = (vah - val) / a
    out["vwap_dist_atr"], out["poc_dist_atr"] = vw, pocd
    out["in_value_area"], out["va_width_atr"] = inva, vaw
    return out


def wf_and_metrics(per):
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
    rows = [("baseline(live 15m)", wf_and_metrics(base))]
    print("  baseline done", flush=True)
    for tf in VP_TFS:
        per = {a: recompute_vp(klines[a], base[a], tf) for a in ASSETS}
        rows.append((f"VP@{tf}", wf_and_metrics(per)))
        print(f"  VP@{tf} done", flush=True)

    print("\n=== volume-profile timeframe (15m setups unchanged; "
          "BTC/ETH/SOL, Goat spreads, top-12%) ===")
    print(f"{'config':>18} {'OOS AUC':>8} {'win%':>6} {'monthly%':>9} "
          f"{'maxDD%':>7} {'PF':>6} {'trades/yr':>10}")
    for name, x in rows:
        print(f"{name:>18} {x['auc']:>8.3f} {x['win']:>6.1f} {x['monthly']:>8.2f}% "
              f"{x['dd']:>6.2f}% {x['pf']:>6.2f} {x['n']//6:>10}")


if __name__ == "__main__":
    main()
