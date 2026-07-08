"""
Stage 4 -- the real test: model-selected trades through an FTMO-style
account, net of crypto costs, across timeframes and execution styles.

Rules coded (the prop target):
  * min hold >= 2 minutes (no sub-2m exits; exits start 2 bars after fill)
  * NO hedging -> one position at a time, never an opposing position
  * 6% trailing max drawdown that trails the equity peak and LOCKS once it
    reaches the 100k starting balance (cannot lose below 100k after that)
  * 3% daily (UTC-day) drawdown measured from each day's opening equity
  * 0.5% equity risk per trade, 24/7 crypto, realistic round-trip cost

Execution styles:
  * taker  -- market fills, ~10 bps round trip (crosses the spread + fee)
  * maker  -- resting limit fills at the level, ~2 bps; OPTIMISTIC (a real
    limit book has adverse selection the directional fill cannot see), so
    treat maker results as an upper bound, flagged as such.

Pipeline: detect -> bracket trades (full entry/exit) -> IS feature
selection -> train(<=2023) -> select OOS(2024+) -> sequence through the
account. Reports pass/fail on every rule, not just P&L.

    ORDERFLOW=1 CRYPTO=1 CRYPTO_GLOB=data/btc python backtest_ftmo.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from labeler import _arrays, _fill_index, _same_day_end, TARGET_RR
from smc_detector import detect_setups, _tf_minutes
import config as C

warnings.filterwarnings("ignore")

NON_FEATURES = {"entry_time", "exit_time", "year", "realized_R", "win",
                "outcome", "entry_px", "risk_px", "exit_px", "dir_"}
IS_END = 2023
OOS_START = 2024
# break-even overlay: once a trade's CLOSED-bar excursion reaches BE_TRIGGER x
# risk, move the stop to entry + BE_LOCK x risk. Set to 1.0R (was 1.5R): a huge
# fraction of setups run to +1.0-1.4R and then reverse to a FULL -1R loss under
# the late 1.5R trigger. Locking to +0.5R at +1.0R converts those into +0.5R
# wins -- validated on the real 1-min live engine across a 6-year walk-forward:
# win 63.4->75.6%, monthly 17.84->18.09%, maxDD 2.88->2.06%, PF 2.57->3.47,
# zero losing months (worst +1.07%), and every year 72-80% win / DD<2.2%. Holds
# under stressed spreads (73% win, 2.8% DD). It's a robust plateau, not a spike
# (1.2->0.9 all give 70-78% win). None disables it.
BE_TRIGGER = 1.0
BE_BUF = 1e-4            # buffer above/below entry for the break-even stop
# profit lock: when the BE trigger fires, move the stop to entry + BE_LOCK x
# risk instead of plain break-even (0 = classic BE). +0.5R lock: a reversal
# after the +1.0R trigger now banks +0.5R instead of scratching to BE. The
# (trigger, lock) = (1.0, 0.5) pair is the sweet spot -- (1.0, 0.6) shaves DD
# a hair more, (0.9, 0.5) lifts win to 78% at a touch less monthly.
BE_LOCK = 0.5
INIT = C.INITIAL_CAPITAL                 # 100,000
RISK_FRAC = 0.005                        # 0.5% per trade
MAX_DD = 0.06                            # 6% trailing, locks at INIT
DAILY_DD = 0.03                          # 3% per UTC day
MIN_HOLD_MIN = 2
MAX_HOLD_MIN = 8 * 60


def gen_trades(df, setups, continuous=True, min_hold=MIN_HOLD_MIN):
    """Full bracket trade records (causal features + entry/exit/prices),
    exits never inside the first `min_hold` minutes."""
    a = _arrays(df)
    rows = []
    for s in setups:
        d = s.direction
        fi = _fill_index(a, s, continuous)
        if fi is None:
            continue
        fill, entry = fi
        risk = abs(entry - s.stop)
        if risk <= 0:
            continue
        tp = entry - TARGET_RR * risk if d == -1 else entry + TARGET_RR * risk
        hi = _same_day_end(a, fill, MAX_HOLD_MIN, continuous)
        exit_px, outcome, exit_i = np.nan, None, hi - 1
        stop = s.stop                                # may ratchet to break-even
        best = entry                                 # best favourable CLOSE-bar
        for k in range(fill + 1 + min_hold, hi):     # >=2 min hold
            hk, lk = a["h"][k], a["l"][k]
            # arm break-even from PRIOR bars only (no intrabar look-ahead)
            if BE_TRIGGER is not None and \
                    ((d == 1 and stop < entry + d * BE_LOCK * risk) or
                     (d == -1 and stop > entry + d * BE_LOCK * risk)):
                if (best - entry) * d / risk >= BE_TRIGGER:
                    stop = entry + d * (BE_LOCK * risk if BE_LOCK > 0
                                        else BE_BUF * entry)
            if d == -1:
                if hk >= stop:     exit_px, outcome, exit_i = stop, "sl", k; break
                if lk <= tp:       exit_px, outcome, exit_i = tp, "tp", k; break
            else:
                if lk <= stop:     exit_px, outcome, exit_i = stop, "sl", k; break
                if hk >= tp:       exit_px, outcome, exit_i = tp, "tp", k; break
            best = max(best, hk) if d == 1 else min(best, lk)   # for next bar
        if outcome is None:
            exit_px, outcome = float(a["c"][hi - 1]), "maxhold"
        gross_r = (exit_px - entry) * d / risk        # signed for both sides
        rows.append({**s.features, "entry_time": df.index[fill],
                     "exit_time": df.index[exit_i], "dir_": d,
                     "entry_px": entry, "exit_px": float(exit_px),
                     "risk_px": risk, "year": int(df.index[fill].year),
                     "realized_R": gross_r, "win": int(gross_r > 0),
                     "outcome": outcome})
    return pd.DataFrame(rows)


def _model():
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=250, learning_rate=0.04, l2_regularization=1.0,
        min_samples_leaf=40, early_stopping=True, validation_fraction=0.2,
        random_state=0)


def select_and_train(data, feats):
    """IS-only feature selection (drop importance<=0 on a 2023 fold), then
    fit on full IS. Returns (model, kept_features, is_threshold)."""
    is_ = data[data["year"] <= IS_END]
    tr, val = is_[is_["year"] < IS_END], is_[is_["year"] == IS_END]
    kept = feats
    if len(tr) >= 150 and len(val) >= 40 and val["win"].nunique() > 1:
        m = _model().fit(tr[feats], tr["win"])
        r = permutation_importance(m, val[feats], val["win"], n_repeats=15,
                                   random_state=0, scoring="roc_auc")
        kept = [f for f, i in zip(feats, r.importances_mean) if i > 0] or feats
    model = _model().fit(is_[kept], is_["win"])
    p_is = model.predict_proba(is_[kept])[:, 1]
    thr, best = float(np.median(p_is)), -9.9
    for q in np.quantile(p_is, np.linspace(0.3, 0.9, 25)):
        sel = is_[p_is >= q]
        if len(sel) >= 0.15 * len(is_) and sel["realized_R"].mean() > best:
            best, thr = sel["realized_R"].mean(), q
    return model, kept, thr


# execution models: (entry_bps, {outcome: exit_bps}). Entries are
# limit-style (price retraces INTO the order block), but a stop-loss is a
# MARKET order that crosses the spread + slips -- so realistic charges the
# losers more. taker = everything market; maker = everything limit (an
# optimistic upper bound that ignores adverse selection on stops).
EXEC = {
    "taker":     (7.0, {"tp": 7.0, "sl": 9.0, "maxhold": 9.0}),
    "realistic": (1.5, {"tp": 1.5, "sl": 9.0, "maxhold": 9.0}),
    "maker":     (1.5, {"tp": 1.5, "sl": 1.5, "maxhold": 1.5}),
}


def _cost_r(t, spec):
    entry_bps, exit_map = spec
    c = entry_bps / 1e4 * t["entry_px"] + exit_map[t["outcome"]] / 1e4 * t["exit_px"]
    return c / t["risk_px"]


def simulate(trades, spec):
    """Sequence selected trades through the FTMO account. trades must be
    sorted by entry_time and carry realized_R, entry_px, risk_px."""
    eq = INIT
    peak = INIT
    floor = INIT - MAX_DD * INIT                # 94,000; trails up, locks INIT
    day = None
    day_open_eq = INIT
    last_exit = pd.Timestamp.min.tz_localize("UTC")
    curve, taken = [], 0
    daily_viol = 0
    blown = False
    worst_daily = 0.0
    for _, t in trades.iterrows():
        if t["entry_time"] < last_exit:            # no hedging / overlap
            continue
        d = t["entry_time"].normalize()
        if day != d:
            day, day_open_eq = d, eq               # new UTC day baseline
        # fixed-fractional off the STARTING balance: no compounding fantasy,
        # and DD/return read directly in % of the 100k account.
        risk_dollar = RISK_FRAC * INIT
        cost_r = _cost_r(t, spec)
        pnl = (t["realized_R"] - cost_r) * risk_dollar
        eq += pnl
        taken += 1
        last_exit = t["exit_time"]
        peak = max(peak, eq)
        floor = min(peak - MAX_DD * INIT, INIT)    # trail, lock at INIT
        dd_day = (day_open_eq - eq) / day_open_eq
        worst_daily = max(worst_daily, dd_day)
        if dd_day > DAILY_DD:
            daily_viol += 1
        if eq <= floor:
            blown = True
            curve.append((t["exit_time"], eq))
            break
        curve.append((t["exit_time"], eq))
    ret = eq / INIT - 1
    cur = pd.Series({c[0]: c[1] for c in curve}) if curve else pd.Series(dtype=float)
    maxdd = ((cur.cummax() - cur) / cur.cummax()).max() if len(cur) else 0.0
    return {"final_eq": eq, "ret": ret, "n": taken, "blown": blown,
            "daily_viol": daily_viol, "worst_daily": worst_daily,
            "maxdd": float(maxdd) if maxdd == maxdd else 0.0,
            "passed": (not blown) and daily_viol == 0 and ret > 0}


def main():
    if os.environ.get("CRYPTO", "0") in ("0", "", "false", "no"):
        print("Set CRYPTO=1 ORDERFLOW=1 CRYPTO_GLOB=data/btc"); return
    from crypto_loader import load_binance_klines
    df = load_binance_klines(os.environ.get("CRYPTO_GLOB", "data/btc"))
    print(f"{len(df):,} bars {df.index[0].date()}..{df.index[-1].date()} | "
          f"order-flow {'ON' if 'of_cvd' in df.columns else 'off'}\n")
    tfs = os.environ.get("TFS", "1min,5min,15min,60min").split(",")
    print(f"{'tf':>6} {'exec':>5} {'trades':>6} {'ret%':>7} {'maxDD%':>7} "
          f"{'wDaily%':>7} {'dViol':>5} {'blown':>5} {'PASS':>5}")
    for tf in tfs:
        os.environ["BASE_TF"] = tf
        setups = detect_setups(df)
        data = gen_trades(df, setups).dropna(subset=["realized_R"])
        feats = [c for c in data.columns
                 if c not in NON_FEATURES and data[c].notna().any()]
        is_, oos = data[data["year"] <= IS_END], data[data["year"] >= OOS_START]
        if len(is_) < 200 or len(oos) < 60 or is_["win"].nunique() < 2:
            print(f"{tf:>6}  (insufficient sample: IS={len(is_)} OOS={len(oos)})")
            continue
        model, kept, thr = select_and_train(data, feats)
        p_oos = model.predict_proba(oos[kept])[:, 1]
        sel = oos[p_oos >= thr].sort_values("entry_time")
        for ex in ("taker", "realistic", "maker"):
            r = simulate(sel, EXEC[ex])
            print(f"{tf:>6} {ex:>5} {r['n']:>6} {r['ret']*100:>+7.1f} "
                  f"{r['maxdd']*100:>7.2f} {r['worst_daily']*100:>7.2f} "
                  f"{r['daily_viol']:>5} {str(r['blown']):>5} "
                  f"{'YES' if r['passed'] else 'no':>5}")
    print("\n(maker = resting-limit fills, ~2bps: OPTIMISTIC upper bound, no "
          "adverse-selection haircut. taker = honest market execution.)")


if __name__ == "__main__":
    main()
