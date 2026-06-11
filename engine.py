"""
Event-driven 1-minute backtest engine for the NAS100 CFD ORB strategy.

Causality rules (no lookahead):
  * A signal is evaluated on the CLOSE of bar t and filled at the OPEN of
    bar t+1 (long fills at ask_o + slippage, short at bid_o - slippage).
  * ATR15 available at minute t is computed only from 15-min bars that
    COMPLETED strictly before t (shift(1) on the resampled series).
  * The trend filter for day d uses daily closes of days < d only.
  * The breakeven stop move is TRIGGERED on a bar close and becomes
    EFFECTIVE from the next bar — never applied to the bar that
    triggered it.
  * The drawdown governor scales risk off the high-water mark of
    prior-day closing equity, never the current day's outcome.

Conservatism rules:
  * Stop-loss is checked before take-profit inside every bar; if both
    levels are touched in the same bar the trade is booked as a loss.
  * Longs exit on the BID, shorts on the ASK; slippage is charged on
    every fill on top of the quoted spread.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as C


@dataclass
class Trade:
    date: object
    direction: int          # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    units: float
    stop: float
    tp: float
    risk_points: float
    risk_dollars: float
    exit_time: pd.Timestamp | None = None
    exit_price: float = np.nan
    pnl: float = np.nan
    exit_reason: str = ""
    duration_min: int = 0
    be_moved: bool = False


@dataclass
class DayResult:
    date: object
    start_equity: float
    end_equity: float
    min_equity: float       # intraday marked-to-market trough
    daily_dd: float         # (start - min) / start


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    days: list[DayResult] = field(default_factory=list)
    equity_daily: pd.Series | None = None   # day-close equity
    final_equity: float = C.INITIAL_CAPITAL


# ────────────────────────────────────────────────────────────────────────
# Causal feature preparation
# ────────────────────────────────────────────────────────────────────────
def causal_atr15(df: pd.DataFrame) -> pd.Series:
    """ATR(14) on completed 15-min bars, mapped to each 1-min stamp.

    The value at minute t comes from the last 15-min bar that completed
    at or before t's 15-min boundary — never from the bar in progress.
    """
    m15 = pd.DataFrame({
        "h": df["mid_h"].resample("15min").max(),
        "l": df["mid_l"].resample("15min").min(),
        "c": df["mid_c"].resample("15min").last(),
    }).dropna()
    prev_c = m15["c"].shift(1)
    tr = pd.concat([
        m15["h"] - m15["l"],
        (m15["h"] - prev_c).abs(),
        (m15["l"] - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / C.ATR15_PERIOD, adjust=False,
                 min_periods=C.ATR15_PERIOD).mean()
    # Bar labelled L spans [L, L+15) and completes at L+15. shift(1)
    # puts the ATR of the bar completed at L onto label L, so a forward
    # fill onto minute stamps within [L, L+15) only ever sees completed
    # bars.
    return atr.shift(1).reindex(df.index, method="ffill")


def daily_trend(df: pd.DataFrame) -> pd.Series:
    """Per-date trend: +1 if the PREVIOUS day's close is above the SMA of
    the TREND_SMA_DAYS daily closes ending on that previous day, else -1.
    Indexed by date. NaN until enough history exists."""
    closes = df["mid_c"].resample("1D").last().dropna()
    sma = closes.rolling(C.TREND_SMA_DAYS).mean()
    trend_prev = (closes > sma).astype(float).map({1.0: 1.0, 0.0: -1.0})
    trend_prev[sma.isna()] = np.nan
    # value computed from day d's own close becomes usable on day d+1
    out = trend_prev.shift(1)
    out.index = out.index.date
    return out


# ────────────────────────────────────────────────────────────────────────
# Engine
# ────────────────────────────────────────────────────────────────────────
def _risk_multiplier(drawdown: float) -> float:
    mult = 1.0
    for thresh, m in sorted(C.DD_TIERS):
        if drawdown > thresh:
            mult = m
    return mult


def run_backtest(df: pd.DataFrame,
                 initial_capital: float = C.INITIAL_CAPITAL) -> BacktestResult:
    """df: 1-min bid/ask/mid frame from data_loader.load_bid_ask
    (America/New_York index)."""
    atr_map = causal_atr15(df)
    trend_map = daily_trend(df)

    open_t = pd.to_datetime(C.SESSION_OPEN).time()
    or_end = (pd.Timestamp("2000-01-01 " + C.SESSION_OPEN)
              + pd.Timedelta(minutes=C.OR_MINUTES)).time()
    cutoff_t = pd.to_datetime(C.ENTRY_CUTOFF).time()
    eod_t = pd.to_datetime(C.EOD_EXIT).time()

    res = BacktestResult()
    equity = initial_capital
    hwm = initial_capital            # prior-day close equity high-water mark
    eq_dates, eq_vals = [], []

    session = df.between_time(C.SESSION_OPEN, "16:00")
    for date, day in session.groupby(session.index.date):
        if len(day) < C.OR_MINUTES + 5:
            continue
        trend = trend_map.get(date, np.nan)
        if not np.isfinite(trend):
            continue

        or_bars = day[(day.index.time >= open_t) & (day.index.time < or_end)]
        if len(or_bars) < C.OR_MINUTES - 3:     # tolerate a few missing bars
            continue
        or_high = float(or_bars["mid_h"].max())
        or_low = float(or_bars["mid_l"].min())
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        dd = max(0.0, (hwm - equity) / hwm)
        risk_frac = C.RISK_PER_TRADE * _risk_multiplier(dd)

        day_start_eq = equity
        day_min_eq = equity
        trades_today = 0
        daily_halt = False
        pos: Trade | None = None
        pending: dict | None = None      # signal awaiting next-bar fill
        pending_be: bool = False         # BE move awaiting next bar

        bars = list(day.itertuples())
        for i, bar in enumerate(bars):
            t = bar.Index

            # ── 1. fill a pending entry at this bar's open ────────────
            if pending is not None and pos is None and not daily_halt:
                direction = pending["dir"]
                entry = (bar.ask_o + C.SLIPPAGE_POINTS if direction == 1
                         else bar.bid_o - C.SLIPPAGE_POINTS)
                dist = pending["dist"]
                risk_dollars = equity * risk_frac
                units = np.floor(risk_dollars / dist / C.UNIT_STEP) * C.UNIT_STEP
                if units >= C.MIN_UNITS:
                    spread = max(bar.ask_o - bar.bid_o, 0.0)
                    be_off = spread + 2 * C.SLIPPAGE_POINTS
                    pos = Trade(
                        date=date, direction=direction, entry_time=t,
                        entry_price=entry, units=units,
                        stop=entry - direction * dist,
                        tp=entry + direction * dist * C.TP_R,
                        risk_points=dist, risk_dollars=units * dist,
                    )
                    pos._be_price = entry + direction * be_off  # type: ignore
                    pos._entry_idx = i                          # type: ignore
                    trades_today += 1
                pending = None

            # ── 2. apply a BE move triggered on a PREVIOUS bar ────────
            if pos is not None and pending_be and not pos.be_moved:
                new_stop = pos._be_price                        # type: ignore
                if (pos.direction == 1 and new_stop > pos.stop) or \
                   (pos.direction == -1 and new_stop < pos.stop):
                    pos.stop = new_stop
                pos.be_moved = True
                pending_be = False

            # ── 3. manage the open position on this bar ───────────────
            if pos is not None:
                exit_price, reason = np.nan, ""
                # Stop fills honor gaps: if the bar OPENS beyond the
                # stop, fill at the open (worse), not at the stop level.
                if pos.direction == 1:
                    if bar.bid_l <= pos.stop:
                        exit_price = min(pos.stop, bar.bid_o) \
                            - C.SLIPPAGE_POINTS
                        reason = "be" if pos.be_moved else "stop"
                    elif bar.bid_h >= pos.tp:
                        exit_price, reason = pos.tp - C.SLIPPAGE_POINTS, "tp"
                else:
                    if bar.ask_h >= pos.stop:
                        exit_price = max(pos.stop, bar.ask_o) \
                            + C.SLIPPAGE_POINTS
                        reason = "be" if pos.be_moved else "stop"
                    elif bar.ask_l <= pos.tp:
                        exit_price, reason = pos.tp + C.SLIPPAGE_POINTS, "tp"

                if not reason and t.time() >= eod_t:
                    exit_price = (bar.bid_c - C.SLIPPAGE_POINTS
                                  if pos.direction == 1
                                  else bar.ask_c + C.SLIPPAGE_POINTS)
                    reason = "eod"

                if not reason and daily_halt:
                    exit_price = (bar.bid_c - C.SLIPPAGE_POINTS
                                  if pos.direction == 1
                                  else bar.ask_c + C.SLIPPAGE_POINTS)
                    reason = "daily_stop"

                if reason:
                    pos.exit_time = t
                    pos.exit_price = float(exit_price)
                    pos.pnl = (pos.exit_price - pos.entry_price) \
                        * pos.direction * pos.units
                    pos.exit_reason = reason
                    pos.duration_min = i - pos._entry_idx + 1   # type: ignore
                    equity += pos.pnl
                    res.trades.append(pos)
                    pos = None
                    pending_be = False
                else:
                    # BE trigger on bar close -> effective next bar
                    if not pos.be_moved and not pending_be:
                        fav = bar.bid_h if pos.direction == 1 else bar.ask_l
                        if (fav - pos.entry_price) * pos.direction \
                                >= C.BE_TRIGGER_R * pos.risk_points:
                            pending_be = True

            # ── 4. marked-to-market equity / daily loss stop ──────────
            marked = equity
            if pos is not None:
                liq = bar.bid_c if pos.direction == 1 else bar.ask_c
                marked += (liq - pos.entry_price) * pos.direction * pos.units
            day_min_eq = min(day_min_eq, marked)
            if not daily_halt and \
                    marked <= day_start_eq * (1.0 - C.DAILY_LOSS_STOP):
                daily_halt = True       # position closes on the next bar
                pending = None

            # ── 5. signal on bar close (filled next bar) ──────────────
            if (pos is None and pending is None and not daily_halt
                    and trades_today < C.MAX_TRADES_PER_DAY
                    and t.time() >= or_end and t.time() <= cutoff_t):
                atr = atr_map.get(t, np.nan)
                if np.isfinite(atr) and atr > 0:
                    dist = float(np.clip(or_range,
                                         C.STOP_FLOOR_ATR * atr,
                                         C.STOP_CAP_ATR * atr))
                    dist = max(dist, C.MIN_STOP_POINTS)
                    if trend == 1 and bar.mid_c > or_high:
                        pending = {"dir": 1, "dist": dist}
                    elif trend == -1 and bar.mid_c < or_low:
                        pending = {"dir": -1, "dist": dist}

        # ── force-flat safety net (data ends before EOD stamp) ────────
        if pos is not None:
            last = bars[-1]
            pos.exit_time = last.Index
            pos.exit_price = float(last.bid_c - C.SLIPPAGE_POINTS
                                   if pos.direction == 1
                                   else last.ask_c + C.SLIPPAGE_POINTS)
            pos.pnl = (pos.exit_price - pos.entry_price) \
                * pos.direction * pos.units
            pos.exit_reason = "data_end"
            pos.duration_min = len(bars) - pos._entry_idx       # type: ignore
            equity += pos.pnl
            res.trades.append(pos)
            pos = None

        day_min_eq = min(day_min_eq, equity)
        res.days.append(DayResult(
            date=date, start_equity=day_start_eq, end_equity=equity,
            min_equity=day_min_eq,
            daily_dd=(day_start_eq - day_min_eq) / day_start_eq,
        ))
        eq_dates.append(date)
        eq_vals.append(equity)
        hwm = max(hwm, equity)          # prior-day close HWM for governor

    res.equity_daily = pd.Series(eq_vals, index=pd.to_datetime(eq_dates))
    res.final_equity = equity
    return res
