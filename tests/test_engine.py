"""
Engine correctness tests.

The lookahead tests work by contradiction: if any decision at time t
used data after t, then truncating or mutating the future would change
trades that closed before the truncation point. We assert it never does.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C
from engine import run_backtest, causal_atr15, daily_trend
from synthetic import make_synthetic_minutes


def trades_key(res):
    return [(t.entry_time, t.exit_time, round(t.entry_price, 6),
             round(t.exit_price, 6), round(t.pnl, 6), t.direction)
            for t in res.trades]


@pytest.fixture(scope="module")
def df():
    return make_synthetic_minutes(n_days=140, seed=42)


@pytest.fixture(scope="module")
def res(df):
    return run_backtest(df)


# ── No-lookahead ────────────────────────────────────────────────────────
def test_truncating_future_days_does_not_change_past_trades(df, res):
    cut = df.index[int(len(df) * 0.7)].normalize()
    res_trunc = run_backtest(df[df.index < cut])
    full = [t for t in trades_key(res) if t[1] < cut]
    assert trades_key(res_trunc) == full


def test_mutating_future_prices_does_not_change_past_trades(df, res):
    cut = df.index[int(len(df) * 0.7)].normalize()
    df2 = df.copy()
    df2.loc[df2.index >= cut, :] = df2.loc[df2.index >= cut, :] * 1.5
    res_mut = run_backtest(df2)
    past_full = [t for t in trades_key(res) if t[1] < cut]
    past_mut = [t for t in trades_key(res_mut) if t[1] < cut]
    assert past_mut == past_full


def test_atr15_is_causal(df):
    """ATR at minute t must be identical whether or not the future
    (anything from t onward) exists in the input."""
    full = causal_atr15(df)
    for frac in (0.31, 0.555, 0.83):
        t = df.index[int(len(df) * frac)]
        trunc = causal_atr15(df[df.index < t])
        last = trunc.index[-1]
        assert np.isclose(trunc.loc[last], full.loc[last], equal_nan=True)


def test_trend_filter_is_causal(df):
    full = daily_trend(df)
    cut_date = df.index[int(len(df) * 0.6)].date()
    trunc = daily_trend(df[df.index.date < cut_date])
    for d in trunc.index:
        if d in full.index:
            a, b = trunc.loc[d], full.loc[d]
            assert (np.isnan(a) and np.isnan(b)) or a == b


# ── Accounting & risk ───────────────────────────────────────────────────
def test_equity_accounting_identity(res):
    pnl = sum(t.pnl for t in res.trades)
    assert np.isclose(res.final_equity, C.INITIAL_CAPITAL + pnl)


def test_no_trade_loses_more_than_risk_budget(res):
    """Loss is bounded by stop distance + slippage (+ rare 1-min gap
    through the stop, absent in this synthetic set)."""
    for t in res.trades:
        max_loss = t.units * (t.risk_points + 2 * C.SLIPPAGE_POINTS + 1.3)
        assert t.pnl >= -max_loss - 1e-6, vars(t)


def test_risk_fraction_respected(res, df):
    """Dollar risk at entry <= 0.5% of equity at the time (tiered down,
    never up)."""
    eq = {d.date: d.start_equity for d in res.days}
    for t in res.trades:
        assert t.risk_dollars <= eq[t.date] * C.RISK_PER_TRADE + 1e-6


def test_exits_after_entries_and_positive_duration(res):
    for t in res.trades:
        assert t.exit_time > t.entry_time or (
            t.exit_time == t.entry_time and t.duration_min == 1)
        assert t.duration_min >= 1


def test_all_trades_closed_same_day(res):
    for t in res.trades:
        assert t.entry_time.date() == t.exit_time.date()
        assert t.exit_time.time() <= pd.Timestamp("16:00").time()


def test_max_one_trade_per_day(res):
    days = pd.Series([t.date for t in res.trades]).value_counts()
    assert (days <= C.MAX_TRADES_PER_DAY).all()


def test_entry_pays_spread_and_slippage(res, df):
    for t in res.trades[:50]:
        bar = df.loc[t.entry_time]
        if t.direction == 1:
            assert np.isclose(t.entry_price,
                              bar["ask_o"] + C.SLIPPAGE_POINTS)
        else:
            assert np.isclose(t.entry_price,
                              bar["bid_o"] - C.SLIPPAGE_POINTS)


# ── Daily loss stop (crafted adversarial day) ───────────────────────────
def _crafted_day(closes_by_day):
    frames = []
    for day, closes in closes_by_day:
        closes = np.asarray(closes, dtype=float)
        idx = pd.date_range(f"{day} 09:30", periods=len(closes),
                            freq="1min", tz="America/New_York")
        opens = np.concatenate([[closes[0]], closes[:-1]])
        highs = np.maximum(opens, closes) + 2.0
        lows = np.minimum(opens, closes) - 2.0
        half = 0.6
        frames.append(pd.DataFrame({
            "bid_o": opens - half, "bid_h": highs - half,
            "bid_l": lows - half, "bid_c": closes - half,
            "ask_o": opens + half, "ask_h": highs + half,
            "ask_l": lows + half, "ask_c": closes + half,
            "mid_h": highs, "mid_l": lows, "mid_c": closes,
        }, index=idx))
    return pd.concat(frames)


def test_daily_loss_stop_blocks_reentry(monkeypatch):
    monkeypatch.setattr(C, "TREND_SMA_DAYS", 5)
    monkeypatch.setattr(C, "MAX_TRADES_PER_DAY", 10)
    monkeypatch.setattr(C, "RISK_PER_TRADE", 0.02)  # force a >1% day loss

    days = []
    base = 15000.0
    for i in range(8):  # rising days -> long trend
        d = pd.bdate_range("2021-02-01", periods=9)[i].date()
        days.append((str(d), np.linspace(base + i * 40,
                                         base + i * 40 + 30, 390)))
    # whipsaw day: flat OR, breakout up, crash through stop, then a
    # second breakout that MUST be blocked by the daily halt
    flat = [base + 320.0] * 15
    breakout = list(np.linspace(base + 320, base + 340, 5))
    crash = list(np.linspace(base + 340, base + 250, 10))
    rebreak = list(np.linspace(base + 250, base + 360, 30))
    rest = [base + 360.0] * (390 - len(flat + breakout + crash + rebreak))
    d9 = str(pd.bdate_range("2021-02-01", periods=9)[8].date())
    days.append((d9, flat + breakout + crash + rebreak + rest))

    res = run_backtest(_crafted_day(days))
    last_day_trades = [t for t in res.trades if str(t.date) == d9]
    # Trade 1 exits at breakeven during the crash (no halt: day is
    # positive), trade 2 takes the full stop loss (> 1% of equity) and
    # trips the daily halt. The rebreak crosses the OR high again at
    # ~10:06 — that third entry MUST be blocked.
    assert len(last_day_trades) == 2
    assert last_day_trades[-1].exit_reason == "stop"
    assert all(t.entry_time.time() < pd.Timestamp("10:00").time()
               for t in last_day_trades)
    last = res.days[-1]
    assert last.daily_dd > C.DAILY_LOSS_STOP   # halt actually engaged
    assert last.daily_dd < 0.03                # and contained the day
