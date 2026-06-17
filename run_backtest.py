"""
Run the full backtest and print a prop-firm constraint report.

On the machine that holds the Dukascopy CSVs:

    python run_backtest.py \
        --bid "USATECHIDXUSD_1 Min_Bid_2020.08.30_2025.09.05.csv" \
        --ask "USATECHIDXUSD_1 Min_Ask_2020.08.30_2025.09.05.csv"

Add --sensitivity to also print a parameter-robustness grid (this is
evidence of robustness around the fixed parameters — do NOT use it to
pick new parameters, that would overfit).

Smoke-test without data:   python run_backtest.py --synthetic
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import config as C
from data_loader import load_bid_ask
from engine import run_backtest, BacktestResult


def intraday_max_drawdown(res: BacktestResult) -> float:
    """Max drawdown using intraday marked-to-market troughs, not just
    day closes — the number a prop firm would measure."""
    peak, mdd = -np.inf, 0.0
    for d in res.days:
        peak = max(peak, d.start_equity)
        mdd = max(mdd, (peak - d.min_equity) / peak)
        peak = max(peak, d.end_equity)
    return mdd


def print_report(res: BacktestResult, label: str = "") -> bool:
    t = pd.DataFrame([vars(x) for x in res.trades])
    eq = res.equity_daily
    if t.empty or eq is None or eq.empty:
        print("No trades generated.")
        return False

    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total_ret = res.final_equity / C.INITIAL_CAPITAL - 1.0
    cagr = (res.final_equity / C.INITIAL_CAPITAL) ** (1 / years) - 1.0
    daily_ret = eq.pct_change(fill_method=None).dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)
    mdd = intraday_max_drawdown(res)
    max_daily = max((d.daily_dd for d in res.days), default=0.0)
    wins = t[t.pnl > 0]
    losses = t[t.pnl <= 0]
    pf = (wins.pnl.sum() / abs(losses.pnl.sum())
          if len(losses) and losses.pnl.sum() < 0 else float("inf"))
    short_trades = t[t.duration_min < C.MIN_TRADE_MINUTES]

    print(f"\n{'=' * 64}")
    print(f" BACKTEST REPORT {label}")
    print(f" {eq.index[0].date()} -> {eq.index[-1].date()}  "
          f"({years:.2f} years, {len(eq)} trading days)")
    print(f"{'=' * 64}")
    print(f" Final equity        : ${res.final_equity:,.2f}")
    print(f" Total return        : {total_ret:+.2%}   CAGR: {cagr:+.2%}")
    print(f" Sharpe (daily, ann.): {sharpe:.2f}")
    print(f" Trades              : {len(t)}  "
          f"({len(t) / years:.0f}/year)")
    print(f" Win rate            : {(t.pnl > 0).mean():.1%}   "
          f"Profit factor: {pf:.2f}")
    print(f" Avg win / avg loss  : "
          f"${wins.pnl.mean() if len(wins) else 0:,.0f} / "
          f"${losses.pnl.mean() if len(losses) else 0:,.0f}")
    print(f" Exit breakdown      : {t.exit_reason.value_counts().to_dict()}")
    print(f" Trade duration min/med/max: {t.duration_min.min()} / "
          f"{int(t.duration_min.median())} / {t.duration_min.max()} bars(min)")

    checks = [
        ("Max total drawdown  < 6%",
         mdd < C.MAX_TOTAL_DD, f"{mdd:.2%}"),
        ("Max daily drawdown  < 3%",
         max_daily < C.MAX_DAILY_DD, f"{max_daily:.2%}"),
        (f"Trades >= {C.MIN_TRADE_MINUTES} min",
         len(short_trades) == 0,
         f"{len(short_trades)} of {len(t)} under "
         f"{C.MIN_TRADE_MINUTES} min"),
        ("Risk per trade <= 0.5% (+costs)",
         bool((t.pnl >= -(t.risk_dollars * 1.15)).all()),
         f"worst loss {t.pnl.min():,.0f} vs budget"),
        ("Profitable over period", total_ret > 0, f"{total_ret:+.2%}"),
    ]
    print(f"{'-' * 64}")
    ok = True
    for name, passed, detail in checks:
        ok &= passed
        print(f" [{'PASS' if passed else 'FAIL'}] {name:<32} {detail}")
    print(f"{'=' * 64}\n")

    # Monthly returns table (pandas >= 2.2 renamed 'M' -> 'ME')
    try:
        pd.tseries.frequencies.to_offset("ME")
        _mefreq = "ME"
    except (ValueError, AttributeError):
        _mefreq = "M"
    monthly = eq.resample(_mefreq).last().pct_change(fill_method=None).dropna()
    if len(monthly):
        tab = monthly.to_frame("ret")
        tab["Y"], tab["M"] = tab.index.year, tab.index.month
        pivot = tab.pivot_table(index="Y", columns="M", values="ret")
        print(" Monthly returns (%):")
        print((pivot * 100).round(2).fillna("").to_string())
    return ok


def save_outputs(res: BacktestResult) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("output", exist_ok=True)
    t = pd.DataFrame([vars(x) for x in res.trades])
    t.to_csv("output/trades.csv", index=False)
    eq = res.equity_daily
    eq.to_csv("output/equity_daily.csv", header=["equity"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(eq.index, eq.values, lw=1.2)
    ax1.set_title("NAS100 CFD ORB — equity (day close)")
    ax1.grid(alpha=0.3)
    dd = (eq.cummax() - eq) / eq.cummax()
    ax2.fill_between(eq.index, -dd * 100, 0, alpha=0.5, color="tab:red")
    ax2.axhline(-6, color="k", ls="--", lw=0.8, label="6% limit")
    ax2.set_ylabel("DD %")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig("output/equity_curve.png", dpi=130)
    print("Saved output/trades.csv, output/equity_daily.csv, "
          "output/equity_curve.png")


def sensitivity_grid(df: pd.DataFrame) -> None:
    """Robustness evidence: nearby parameter values should give a similar
    result. A strategy that only works at exactly one setting is curve-fit."""
    base = (C.OR_MINUTES, C.TP_R, C.TREND_SMA_DAYS)
    print("\nParameter sensitivity (total return / max intraday DD):")
    for or_min in (10, 15, 30):
        for tp_r in (3.0, 4.0, 5.0):
            for sma in (20, 50, 100):
                C.OR_MINUTES, C.TP_R, C.TREND_SMA_DAYS = or_min, tp_r, sma
                r = run_backtest(df)
                ret = r.final_equity / C.INITIAL_CAPITAL - 1
                mdd = intraday_max_drawdown(r)
                star = " <- base" if (or_min, tp_r, sma) == base else ""
                print(f"  OR={or_min:>2}  TP={tp_r:.0f}R  SMA={sma:>3}  "
                      f"-> {ret:+7.2%} / {mdd:5.2%}{star}")
    C.OR_MINUTES, C.TP_R, C.TREND_SMA_DAYS = base


def autodetect_csvs() -> tuple[str | None, str | None]:
    """Find the Dukascopy bid/ask CSVs in the current folder so the user
    never has to type the long filenames. Looks for *Bid*.csv / *Ask*.csv
    (case-insensitive), preferring the NAS100 (USATECHIDX) files."""
    import glob
    bids = [f for f in glob.glob("*.csv") if "bid" in f.lower()]
    asks = [f for f in glob.glob("*.csv") if "ask" in f.lower()]

    def prefer(files: list[str]) -> str | None:
        if not files:
            return None
        nas = [f for f in files if "usatechidx" in f.lower()
               or "nas" in f.lower()]
        return (nas or files)[0]

    return prefer(bids), prefer(asks)


def autodetect_sp_csvs() -> tuple[str | None, str | None]:
    """Find the S&P 500 (USA500IDX / SPX) bid/ask CSVs for SMT features."""
    import glob
    def pick(side: str) -> str | None:
        files = [f for f in glob.glob("*.csv")
                 if side in f.lower()
                 and ("usa500idx" in f.lower() or "spx" in f.lower()
                      or "us500" in f.lower() or "sp500" in f.lower())]
        return files[0] if files else None
    return pick("bid"), pick("ask")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bid", help="Dukascopy 1-min BID csv (USATECHIDXUSD)")
    p.add_argument("--ask", help="Dukascopy 1-min ASK csv (USATECHIDXUSD)")
    p.add_argument("--synthetic", action="store_true",
                   help="run on synthetic data (engine smoke test only)")
    p.add_argument("--sensitivity", action="store_true")
    args = p.parse_args()

    if args.synthetic:
        from synthetic import make_synthetic_minutes
        df = make_synthetic_minutes(n_days=400, seed=7)
        label = "(SYNTHETIC DATA — engine demo, not evidence of an edge)"
    else:
        bid, ask = args.bid, args.ask
        if not bid or not ask:
            bid, ask = autodetect_csvs()
            if bid and ask:
                print(f"Auto-detected data files:\n  BID: {bid}\n  ASK: {ask}")
        if not bid or not ask:
            p.error(
                "Could not find the data CSVs. Put the Dukascopy bid/ask "
                "files in this folder (their names contain 'Bid'/'Ask'), "
                "or pass them with --bid and --ask. To smoke-test the "
                "engine without data, use --synthetic.")
            return
        print("Loading Dukascopy CSVs (this can take a minute)...")
        df = load_bid_ask(bid, ask)
        label = "(NAS100 CFD, Dukascopy 1-min bid/ask)"
        print(f"Loaded {len(df):,} 1-min bars "
              f"{df.index[0]} -> {df.index[-1]}")

    res = run_backtest(df)
    save_outputs(res)            # save first so a report hiccup can't lose them
    ok = print_report(res, label)
    if args.sensitivity:
        sensitivity_grid(df)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
