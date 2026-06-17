"""
Out-of-sample research — does the 'fast trend filter' edge survive?

PROTOCOL (committed before looking at any result):
  1. Split the history into IN-SAMPLE (trades up to IS_END) and
     OUT-OF-SAMPLE (trades from OOS_START). Indicators warm up on the
     full history in both runs, but P&L is counted only inside the
     window — see engine.run_backtest(trade_start, trade_end).
  2. Search the parameter grid on IN-SAMPLE ONLY. Keep only configs
     that satisfy the hard constraints in-sample (max DD < 6%, daily
     DD < 3%, profitable, enough trades to mean something). Among those,
     pick the one with the best in-sample Sharpe — Sharpe, not raw
     return, because we want a steady edge, not one lucky cell.
  3. Lock that single config. Run it ONCE on out-of-sample.
  4. Report the OOS result honestly, pass or fail. No second look, no
     re-pick. If nothing clears in-sample, say so and stop.

Why this is not what the old system did: the old pipeline fitted on all
the data and evaluated on the same data. Here the OOS years are never
used to choose anything — they are only ever scored once, at the end.

A single split is the honest minimum, not the last word; if a config
clears it, the next step is walk-forward across several splits before
trusting real money. Run:  python research_oos.py
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import config as C
from data_loader import load_bid_ask
from engine import run_backtest
from run_backtest import autodetect_csvs, intraday_max_drawdown

IS_END = "2023-12-31"
OOS_START = "2024-01-01"

# Grid. SMA includes the slow values too, so the IS search must discover
# the fast-filter preference on its own rather than us hard-coding it.
GRID = {
    "OR_MINUTES": [10, 15, 30],
    "TP_R": [3.0, 4.0, 5.0],
    "TREND_SMA_DAYS": [20, 50, 100],
    "RISK_PER_TRADE": [0.003, 0.005],
}
MIN_IS_TRADES = 120        # ~35/yr over the in-sample span — enough to judge


def metrics(res) -> dict:
    eq = res.equity_daily
    if eq is None or len(eq) < 2 or not res.trades:
        return {"ret": -1, "mdd": 1, "daily": 1, "sharpe": -9, "n": 0}
    ret = res.final_equity / C.INITIAL_CAPITAL - 1
    dr = eq.pct_change(fill_method=None).dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    return {
        "ret": ret,
        "mdd": intraday_max_drawdown(res),
        "daily": max((d.daily_dd for d in res.days), default=0.0),
        "sharpe": sharpe,
        "n": len(res.trades),
    }


def set_cfg(combo: dict) -> None:
    for k, v in combo.items():
        setattr(C, k, v)


def compliant(m: dict) -> bool:
    return (m["mdd"] < C.MAX_TOTAL_DD and m["daily"] < C.MAX_DAILY_DD
            and m["ret"] > 0 and m["n"] >= MIN_IS_TRADES)


def main() -> None:
    bid, ask = autodetect_csvs()
    if not (bid and ask):
        raise SystemExit("Put the Dukascopy Bid/Ask CSVs in this folder.")
    print(f"Loading {bid} / {ask} ...")
    df = load_bid_ask(bid, ask)
    print(f"{len(df):,} bars {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"In-sample: trades <= {IS_END}   "
          f"Out-of-sample: trades >= {OOS_START}\n")

    base = {k: getattr(C, k) for k in GRID}
    combos = [dict(zip(GRID, vals))
              for vals in itertools.product(*GRID.values())]

    # ── 1-2. search IN-SAMPLE only ──────────────────────────────────
    print(f"Searching {len(combos)} configs on in-sample only "
          f"(a few minutes)...\n")
    rows = []
    for combo in combos:
        set_cfg(combo)
        m = metrics(run_backtest(df, trade_end=IS_END))
        m.update(combo)
        m["ok"] = compliant(m)
        rows.append(m)

    tab = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    show = ["OR_MINUTES", "TP_R", "TREND_SMA_DAYS", "RISK_PER_TRADE",
            "ret", "mdd", "daily", "sharpe", "n", "ok"]
    pd.set_option("display.width", 140)
    print("In-sample results (sorted by Sharpe):")
    print(tab[show].head(12).to_string(
        index=False,
        formatters={"ret": "{:+.2%}".format, "mdd": "{:.2%}".format,
                    "daily": "{:.2%}".format, "sharpe": "{:.2f}".format,
                    "RISK_PER_TRADE": "{:.1%}".format}))

    winners = tab[tab["ok"]]
    if winners.empty:
        set_cfg(base)
        print("\n" + "=" * 64)
        print(" VERDICT: no config met the constraints IN-SAMPLE.")
        print(" The fast-filter effect is not strong/clean enough to build")
        print(" on under a 6% drawdown cap. Honest answer: don't trade it.")
        print("=" * 64)
        return

    pick = winners.iloc[0]
    chosen = {k: pick[k] for k in GRID}
    print(f"\nChosen on IN-SAMPLE (best compliant Sharpe): {chosen}")
    print(f"  in-sample: ret {pick['ret']:+.2%}, maxDD {pick['mdd']:.2%}, "
          f"dailyDD {pick['daily']:.2%}, Sharpe {pick['sharpe']:.2f}, "
          f"{int(pick['n'])} trades")

    # ── 3. ONE out-of-sample run with the locked config ─────────────
    set_cfg(chosen)
    oos = metrics(run_backtest(df, trade_start=OOS_START))
    full = metrics(run_backtest(df))
    set_cfg(base)

    print("\n" + "=" * 64)
    print(" OUT-OF-SAMPLE VERDICT (2024-01-01 -> end, scored once)")
    print("=" * 64)
    print(f" Return        : {oos['ret']:+.2%}")
    print(f" Max DD        : {oos['mdd']:.2%}   (limit 6%)")
    print(f" Max daily DD  : {oos['daily']:.2%}   (limit 3%)")
    print(f" Sharpe        : {oos['sharpe']:.2f}")
    print(f" Trades        : {oos['n']}")
    passed = (oos["ret"] > 0 and oos["mdd"] < C.MAX_TOTAL_DD
              and oos["daily"] < C.MAX_DAILY_DD)
    print(f"\n {'PASS' if passed else 'FAIL'} — the edge "
          f"{'held' if passed else 'did NOT hold'} out-of-sample.")
    if passed:
        print(" Next step before real money: walk-forward across multiple")
        print(" splits to confirm it is not one lucky 2-year stretch.")
    else:
        print(" In-sample success did not transfer. This is the honest")
        print(" result the old system never showed you. Do not trade it.")
    print(f"\n For context only — same config on the FULL period: "
          f"{full['ret']:+.2%} / DD {full['mdd']:.2%}")
    print("=" * 64)


if __name__ == "__main__":
    main()
