"""
RESEARCH ONLY (no live file touched): honest MARKET-execution backtest.

The cached numbers assume a LIMIT fill back at the order-block level. Live
can't reliably hit that -- it only knows a setup fired after the 15m bar
closes. This models the realistic alternative: enter at MARKET at the open of
the next 15m bar (after the trigger closes), paying the TAKER spread on entry.
Same BE/2R management, same walk-forward + strict top-12% + Goat spreads.
Compares market-exec to the (optimistic) limit baseline.
"""
from __future__ import annotations

import os
os.environ.setdefault("CRYPTO", "1")

import heapq
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import backtest_ftmo as B
import goat_test as G
import sizing as SZ
from backtest_ftmo import BE_TRIGGER, BE_BUF, TARGET_RR, MIN_HOLD_MIN, MAX_HOLD_MIN
from smc_detector import detect_setups, _tf_minutes
from crypto_loader import load_binance_klines
from live_runner import INIT

ASSETS = {"BTC": "data/btc", "ETH": "data/eth", "SOL": "data/sol"}
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]


def gen_trades_market(df, setups):
    """Enter at the open of the bar AFTER the trigger bar closes (what live can
    actually do), then manage with BE@1.5R / 2R like the limit backtest."""
    a = B._arrays(df)
    tf = _tf_minutes()
    rows = []
    for s in setups:
        d = s.direction
        # market entry: first 1m bar at/after the trigger bar's close
        t_close = np.datetime64(pd.Timestamp(s.entry_time).tz_localize(None)
                                + pd.Timedelta(minutes=tf))
        fill = int(np.searchsorted(a["ts"], t_close, "left"))
        if fill >= a["n"]:
            continue
        entry = float(a["o"][fill])
        risk = abs(entry - s.stop)
        if risk <= 0:
            continue
        tp = entry + d * TARGET_RR * risk
        hi = B._same_day_end(a, fill, MAX_HOLD_MIN, True)
        exit_px, outcome, exit_i = np.nan, None, hi - 1
        stop, best = s.stop, entry
        for k in range(fill + 1 + MIN_HOLD_MIN, hi):
            hk, lk = a["h"][k], a["l"][k]
            if BE_TRIGGER is not None and \
                    ((d == 1 and stop < entry) or (d == -1 and stop > entry)):
                if (best - entry) * d / risk >= BE_TRIGGER:
                    stop = entry + d * BE_BUF * entry
            if d == -1:
                if hk >= stop: exit_px, outcome, exit_i = stop, "sl", k; break
                if lk <= tp:   exit_px, outcome, exit_i = tp, "tp", k; break
            else:
                if lk <= stop: exit_px, outcome, exit_i = stop, "sl", k; break
                if hk >= tp:   exit_px, outcome, exit_i = tp, "tp", k; break
            best = max(best, hk) if d == 1 else min(best, lk)
        if outcome is None:
            exit_px, outcome = float(a["c"][hi - 1]), "maxhold"
        gr = (exit_px - entry) * d / risk
        rows.append({**s.features, "entry_time": df.index[fill],
                     "exit_time": df.index[exit_i], "dir_": d,
                     "entry_px": entry, "exit_px": float(exit_px),
                     "risk_px": risk, "year": int(df.index[fill].year),
                     "realized_R": gr, "win": int(gr > 0), "outcome": outcome})
    return pd.DataFrame(rows)


def sequence_market(allt, cost):
    """Like goat_test.sequence but entry is a TAKER (crosses the spread)."""
    eq = peak = INIT
    heap, open_a, open_n, open_w = [], set(), 0.0, 0.0
    day, day_start = None, INIT
    rows = []
    for _, t in allt.iterrows():
        dd = t.entry_time.normalize()
        if dd != day:
            day, day_start = dd, eq
        while heap and heap[0][0] <= t.entry_time:
            ex, pnl, asset, ntl, wd, idx = heapq.heappop(heap)
            cdd = ex.normalize()
            if cdd != day:
                day, day_start = cdd, eq
            open_a.discard(asset); open_n -= ntl; open_w -= wd
            eq += pnl; peak = max(peak, eq); rows[idx]["eq"] = eq
        if t.asset in open_a:
            continue
        sf = t.risk_px / t.entry_px
        e, s = cost[t.asset]
        wf_ = sf + (s + s) / 1e4               # taker entry + taker stop
        rem = SZ.DAY_BUDGET * INIT - max(0.0, day_start - eq) - open_w
        ntl = SZ.size_notional(t.prob, sf, open_n, INIT, budget_notional=rem / wf_)
        if ntl <= 0.02 * INIT:
            continue
        exit_bps = e if t.outcome == "tp" else s      # TP maker, SL taker
        cost_r = (s + exit_bps) / 1e4 * t.entry_px / t.risk_px   # entry = taker
        pnl = (t.realized_R - cost_r) * ntl * sf
        rows.append({"exit_time": t.exit_time, "asset": t.asset, "pnl": pnl,
                     "win": int(t.realized_R - cost_r > 0), "eq": np.nan})
        idx = len(rows) - 1; wd = ntl * wf_
        open_a.add(t.asset); open_n += ntl; open_w += wd
        heapq.heappush(heap, (t.exit_time, pnl, t.asset, ntl, wd, idx))
    while heap:
        ex, pnl, asset, ntl, wd, idx = heapq.heappop(heap)
        eq += pnl; peak = max(peak, eq); rows[idx]["eq"] = eq
    return pd.DataFrame(rows)


def wf(per, seqfn, cost):
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
            ptr = mm.predict_proba(tr[feats])[:, 1]; pte = mm.predict_proba(te[feats])[:, 1]
            if te.win.nunique() > 1:
                aucs.append(roc_auc_score(te.win, pte))
            thr = np.quantile(ptr, 0.88)
            sb = te[pte >= thr].copy(); sb["prob"] = pte[pte >= thr]; sb["asset"] = a
            picks.append(sb)
    allt = pd.concat(picks).sort_values("entry_time").reset_index(drop=True)
    led = seqfn(allt, cost)
    led["ym"] = pd.to_datetime(led.exit_time).dt.strftime("%Y-%m")
    eqc = led.set_index("exit_time")["eq"]
    dd = ((eqc.cummax() - eqc) / eqc.cummax()).max() * 100
    mo = (led.groupby("ym").pnl.sum() / INIT * 100).mean()
    gp = led.loc[led.pnl > 0, "pnl"].sum(); gl = -led.loc[led.pnl < 0, "pnl"].sum()
    r = allt.realized_R; w = (r > 0.1).sum(); ll = (r < -0.1).sum()
    return dict(auc=np.mean(aucs), win=w / (w + ll) * 100, monthly=mo, dd=dd,
                pf=gp / gl if gl else 0, n=len(led))


def main():
    limit = {a: pd.read_parquet(f"data/trades/{a}_15min.parquet") for a in ASSETS}
    mkt = {}
    for a, folder in ASSETS.items():
        os.environ["ORDERFLOW"] = "0"; os.environ["BASE_TF"] = "15min"
        df = load_binance_klines(folder)
        t = gen_trades_market(df, detect_setups(df)); t["asset"] = a
        mkt[a] = t
        print(f"  {a} market trades: {len(t)}", flush=True)
    rL = wf(limit, G.sequence, G.GOAT_TYPICAL)
    rM = wf(mkt, sequence_market, G.GOAT_TYPICAL)
    print("\n=== execution model (BTC/ETH/SOL, Goat spreads, top-12%) ===")
    print(f"{'model':>22} {'OOS AUC':>8} {'win%':>6} {'monthly%':>9} {'maxDD%':>7} {'PF':>6}")
    for name, x in [("LIMIT (fill at OB)", rL), ("MARKET (next-bar, taker)", rM)]:
        print(f"{name:>22} {x['auc']:>8.3f} {x['win']:>6.1f} {x['monthly']:>8.2f}% "
              f"{x['dd']:>6.2f}% {x['pf']:>6.2f}")


if __name__ == "__main__":
    main()
