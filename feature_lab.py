"""
Feature lab: does any new feature beat the current model OOS?

Baseline = the live feature set on BTC/ETH/SOL. For each candidate we add it,
re-run the expanding walk-forward (train < Y, score Y), and measure OOS AUC vs
baseline. Candidates that help are then stacked and run through the SAME sized
sequence at Goat spreads to confirm they improve PF / return / win, not just
AUC. Discipline: keep ONLY what improves OOS robustly; reject the rest.

Candidates (all causal -- computed on closed 15m bars, shifted 1):
  ema_dist, ema_slope   : base-TF trend (close vs EMA50, in ATR)
  ret_4h, rsi14         : momentum
  vol_z                 : volatility regime
  frac_up/dn_dist       : Williams-fractal (swing) distance, in ATR
  btc_ema_dist, btc_ret_4h : cross-asset BTC context (alts are BTC beta)
  *_al                  : direction-aligned versions (x * dir)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import backtest_ftmo as B
import goat_test as G
from live_runner import INIT
from crypto_loader import load_binance_klines

os.environ["CRYPTO"] = "1"; os.environ["BASE_TF"] = "15min"
FOLDERS = {"BTC": ("data/btc", True), "ETH": ("data/eth", False), "SOL": ("data/sol", False)}
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
CANDS = ["ema_dist", "ema_slope", "ret_4h", "rsi14", "vol_z",
         "frac_up_dist", "frac_dn_dist", "btc_ema_dist", "btc_ret_4h",
         "ema_dist_al", "btc_ema_dist_al", "ret_4h_al"]


def feats_15m(folder, of):
    os.environ["ORDERFLOW"] = "1" if of else "0"
    df = load_binance_klines(folder)
    m = df["mid_c"].resample("15min").last().dropna()
    hi = df["mid_h"].resample("15min").max().reindex(m.index)
    lo = df["mid_l"].resample("15min").min().reindex(m.index)
    tr = pd.concat([(hi - lo), (hi - m.shift()).abs(), (lo - m.shift()).abs()], axis=1).max(1)
    atr = tr.rolling(14).mean()
    ema = m.ewm(span=50).mean()
    ret = m.pct_change()
    out = pd.DataFrame(index=m.index)
    out["ema_dist"] = (m - ema) / atr
    out["ema_slope"] = (ema - ema.shift(5)) / atr
    out["ret_4h"] = m.pct_change(16)
    d = m.diff()
    up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    out["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    out["vol_z"] = (ret.rolling(20).std() - ret.rolling(200).std().mean()) / ret.rolling(200).std().std()
    # Williams 5-bar fractals -> distance to nearest swing in ATR
    fh = ((hi.shift(2) < hi) & (hi.shift(1) < hi) & (hi.shift(-1) < hi) & (hi.shift(-2) < hi))
    fl = ((lo.shift(2) > lo) & (lo.shift(1) > lo) & (lo.shift(-1) > lo) & (lo.shift(-2) > lo))
    last_fh = hi.where(fh).shift(2).ffill()     # shift 2 so it's confirmed/causal
    last_fl = lo.where(fl).shift(2).ffill()
    out["frac_up_dist"] = (last_fh - m) / atr
    out["frac_dn_dist"] = (m - last_fl) / atr
    return out.shift(1)                          # use only CLOSED prior bar


def main():
    # base features per asset from cache + candidate features joined by time
    btc = feats_15m(*FOLDERS["BTC"])[["ema_dist", "ret_4h"]].rename(
        columns={"ema_dist": "btc_ema_dist", "ret_4h": "btc_ret_4h"})
    data = {}
    for a, (folder, of) in FOLDERS.items():
        d = pd.read_parquet(f"data/trades/{a}_15min.parquet").dropna(subset=["realized_R"])
        f = feats_15m(folder, of)
        f = f.join(btc)                          # cross-asset BTC context
        d = d.merge(f, left_on="entry_time", right_index=True, how="left")
        for c in ("ema_dist", "btc_ema_dist", "ret_4h"):
            d[c + "_al"] = d[c] * d["dir"]        # direction-aligned
        data[a] = d
    base = list(__import__("joblib").load("models/BTC_15min.joblib")["feats"])

    def wf_auc(feats):
        aucs = []
        for a, d in data.items():
            for Y in YEARS:
                tr, te = d[d.year < Y], d[d.year == Y]
                if len(tr) < 200 or len(te) < 40 or tr.win.nunique() < 2 or te.win.nunique() < 2:
                    continue
                m = B._model().fit(tr[feats], tr.win)
                aucs.append(roc_auc_score(te.win, m.predict_proba(te[feats])[:, 1]))
        return np.mean(aucs)

    base_auc = wf_auc(base)
    print(f"baseline OOS AUC (19 feats): {base_auc:.4f}\n  each candidate added:")
    helps = []
    for c in CANDS:
        a = wf_auc(base + [c])
        flag = "  <- helps" if a > base_auc + 0.0005 else ""
        if a > base_auc + 0.0005:
            helps.append((c, a))
        print(f"   +{c:16} AUC {a:.4f}  ({a-base_auc:+.4f}){flag}")
    helps.sort(key=lambda x: -x[1])
    keep = [c for c, _ in helps]
    print(f"\n  stack of all that help {keep}: AUC {wf_auc(base+keep):.4f}" if keep
          else "\n  no single candidate helps")
    return data, base, keep


if __name__ == "__main__":
    main()
