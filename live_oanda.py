"""
Live runner — NAS100 CFD ORB on OANDA (NAS100_USD).

Mirrors engine.py decision-for-decision:
  * Opening range from completed M1 bars 09:30-09:44 New York.
  * One breakout trade per day, in the direction of the 50-day SMA
    trend filter, signal on M1 close, market order with SL/TP attached.
  * Stop distance = clip(OR range, 0.25*ATR15, 2*ATR15), >= 15 points.
  * Breakeven move after +1R on a COMPLETED bar (next-bar effective,
    same as the backtest).
  * Force-flat at 15:55 NY. Hard daily halt at -1.0% of day-start NAV.
  * Drawdown governor on prior-day closing NAV high-water mark
    (persisted in state.json): DD>2% -> half risk, DD>3.5% -> quarter.

Run on a demo account first and compare its decisions against the
backtest day by day before risking anything.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades_ep

from zoneinfo import ZoneInfo

import config as C

NY = ZoneInfo("America/New_York")

OANDA_ENV = os.environ.get("OANDA_ENV", "practice")
OANDA_TOKEN = os.environ.get("OANDA_TOKEN", "")
OANDA_ACCOUNT = os.environ.get("OANDA_ACCOUNT", "")
INSTRUMENT = "NAS100_USD"
ACCOUNT_CURRENCY = os.environ.get("ACCOUNT_CURRENCY", "USD")
STATE_FILE = "state.json"


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"hwm_nav": None}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


class Bot:
    def __init__(self) -> None:
        if not OANDA_TOKEN or not OANDA_ACCOUNT:
            sys.exit("Set OANDA_TOKEN and OANDA_ACCOUNT env vars "
                     "(never hardcode credentials).")
        self.api = API(access_token=OANDA_TOKEN, environment=OANDA_ENV)
        self.state = load_state()

    # ── data ────────────────────────────────────────────────────────
    def candles(self, granularity: str, count: int) -> pd.DataFrame:
        r = instruments.InstrumentsCandles(
            instrument=INSTRUMENT,
            params={"count": count, "granularity": granularity,
                    "price": "M"})
        self.api.request(r)
        rows = [(pd.to_datetime(c["time"]), float(c["mid"]["o"]),
                 float(c["mid"]["h"]), float(c["mid"]["l"]),
                 float(c["mid"]["c"]))
                for c in r.response["candles"] if c["complete"]]
        df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c"]) \
            .set_index("t")
        df.index = df.index.tz_convert(NY)
        return df

    def nav(self) -> float:
        r = accounts.AccountSummary(OANDA_ACCOUNT)
        self.api.request(r)
        return float(r.response["account"]["NAV"])

    def open_trade(self):
        r = trades_ep.OpenTrades(accountID=OANDA_ACCOUNT)
        self.api.request(r)
        for t in r.response.get("trades", []):
            if t["instrument"] == INSTRUMENT:
                return t
        return None

    def fx_to_usd(self) -> float:
        if ACCOUNT_CURRENCY == "USD":
            return 1.0
        pair = {"GBP": "GBP_USD", "EUR": "EUR_USD"}.get(ACCOUNT_CURRENCY)
        if pair is None:
            return 1.0
        r = pricing.PricingInfo(OANDA_ACCOUNT,
                                params={"instruments": pair})
        self.api.request(r)
        p = r.response["prices"][0]
        return (float(p["bids"][0]["price"])
                + float(p["asks"][0]["price"])) / 2.0

    # ── strategy inputs ─────────────────────────────────────────────
    def trend(self) -> float:
        d = self.candles("D", C.TREND_SMA_DAYS + 10)
        closes = d["c"]
        # last completed daily candle only (prior days)
        sma = closes.rolling(C.TREND_SMA_DAYS).mean()
        if np.isnan(sma.iloc[-1]):
            return np.nan
        return 1.0 if closes.iloc[-1] > sma.iloc[-1] else -1.0

    def atr15(self) -> float:
        m = self.candles("M15", C.ATR15_PERIOD * 4)
        prev_c = m["c"].shift(1)
        tr = pd.concat([m["h"] - m["l"], (m["h"] - prev_c).abs(),
                        (m["l"] - prev_c).abs()], axis=1).max(axis=1)
        return float(tr.ewm(alpha=1 / C.ATR15_PERIOD,
                            adjust=False).mean().iloc[-1])

    def risk_multiplier(self, nav: float) -> float:
        hwm = self.state.get("hwm_nav") or nav
        dd = max(0.0, (hwm - nav) / hwm)
        mult = 1.0
        for thresh, m in sorted(C.DD_TIERS):
            if dd > thresh:
                mult = m
        if mult < 1.0:
            log(f"DD governor: drawdown {dd:.2%} -> risk x{mult}")
        return mult

    # ── orders ──────────────────────────────────────────────────────
    def market_order(self, direction: int, dist: float,
                     risk_frac: float) -> None:
        nav = self.nav()
        r = pricing.PricingInfo(OANDA_ACCOUNT,
                                params={"instruments": INSTRUMENT})
        self.api.request(r)
        p = r.response["prices"][0]
        entry = (float(p["asks"][0]["price"]) if direction == 1
                 else float(p["bids"][0]["price"]))
        sl = round(entry - direction * dist, 1)
        tp = round(entry + direction * dist * C.TP_R, 1)
        units = int(np.floor(nav * risk_frac * self.fx_to_usd() / dist))
        if units < C.MIN_UNITS:
            log("Position size below minimum — skipping")
            return
        req = {"order": {
            "units": str(units * direction),
            "instrument": INSTRUMENT, "type": "MARKET",
            "timeInForce": "FOK",
            "stopLossOnFill": {"price": f"{sl:.1f}"},
            "takeProfitOnFill": {"price": f"{tp:.1f}"},
        }}
        self.api.request(orders.OrderCreate(OANDA_ACCOUNT, data=req))
        log(f"{'BUY' if direction == 1 else 'SELL'} {units} @ ~{entry} "
            f"SL {sl} TP {tp} (risk {risk_frac:.3%} of NAV)")

    def close_position(self, trade) -> None:
        r = trades_ep.TradeClose(OANDA_ACCOUNT, tradeID=trade["id"])
        self.api.request(r)
        log("Position closed")

    def move_to_breakeven(self, trade) -> None:
        entry = float(trade["price"])
        direction = 1 if float(trade["initialUnits"]) > 0 else -1
        sl_price = float(trade["stopLossOrder"]["price"])
        risk_pts = abs(entry - sl_price)
        if risk_pts < 1e-6 or (entry - sl_price) * direction <= 0:
            return  # already at/through breakeven
        bar = self.candles("M1", 2).iloc[-1]
        fav = bar["h"] if direction == 1 else bar["l"]
        if (fav - entry) * direction >= C.BE_TRIGGER_R * risk_pts:
            be = round(entry + direction * 2.0, 1)  # entry + ~spread
            r = trades_ep.TradeCRCDO(
                OANDA_ACCOUNT, tradeID=trade["id"],
                data={"stopLoss": {"price": f"{be:.1f}"}})
            self.api.request(r)
            log(f"Stop moved to breakeven {be}")

    # ── main loop ───────────────────────────────────────────────────
    def run(self) -> None:
        day = None
        day_start_nav = None
        or_high = or_low = None
        traded_today = 0
        halted = False
        trend = np.nan

        log(f"ORB live runner started ({OANDA_ENV}, {INSTRUMENT})")
        while True:
            try:
                now = datetime.now(NY)
                if day != now.date():
                    # persist yesterday's close into the HWM before reset
                    if day_start_nav is not None:
                        nav = self.nav()
                        self.state["hwm_nav"] = max(
                            self.state.get("hwm_nav") or nav, nav)
                        save_state(self.state)
                    day = now.date()
                    day_start_nav = self.nav()
                    or_high = or_low = None
                    traded_today, halted = 0, False
                    trend = self.trend()
                    log(f"New day {day} | NAV {day_start_nav:,.2f} | "
                        f"trend {'LONG' if trend == 1 else 'SHORT' if trend == -1 else 'N/A'}")

                t = now.time()
                open_t = pd.Timestamp(C.SESSION_OPEN).time()
                eod_t = pd.Timestamp(C.EOD_EXIT).time()
                cutoff_t = pd.Timestamp(C.ENTRY_CUTOFF).time()

                if t < open_t or now.weekday() >= 5 or not np.isfinite(trend):
                    time.sleep(30)
                    continue

                trade = self.open_trade()

                # EOD force-flat
                if t >= eod_t:
                    if trade:
                        self.close_position(trade)
                    time.sleep(60)
                    continue

                # daily loss halt (NAV basis, includes open P&L)
                nav = self.nav()
                if not halted and \
                        nav <= day_start_nav * (1 - C.DAILY_LOSS_STOP):
                    halted = True
                    log(f"DAILY HALT: NAV {nav:,.2f} <= "
                        f"-{C.DAILY_LOSS_STOP:.1%} of day start")
                    if trade:
                        self.close_position(trade)
                if halted:
                    time.sleep(60)
                    continue

                if trade:
                    self.move_to_breakeven(trade)
                    time.sleep(15)
                    continue

                # build opening range once M1 bars 09:30-09:44 are sealed
                or_done = (now.hour * 60 + now.minute) >= \
                    (open_t.hour * 60 + open_t.minute + C.OR_MINUTES + 1)
                if or_high is None and or_done:
                    m1 = self.candles("M1", 120)
                    onday = m1[m1.index.date == day]
                    orb = onday.between_time(
                        C.SESSION_OPEN,
                        (pd.Timestamp(C.SESSION_OPEN)
                         + pd.Timedelta(minutes=C.OR_MINUTES - 1)).time())
                    if len(orb) >= C.OR_MINUTES - 3:
                        or_high = float(orb["h"].max())
                        or_low = float(orb["l"].min())
                        log(f"Opening range {or_low:.1f} – {or_high:.1f}")

                # breakout check on the latest COMPLETED M1 bar
                if (or_high is not None and traded_today <
                        C.MAX_TRADES_PER_DAY and t <= cutoff_t):
                    last = self.candles("M1", 2).iloc[-1]
                    atr = self.atr15()
                    dist = float(np.clip(or_high - or_low,
                                         C.STOP_FLOOR_ATR * atr,
                                         C.STOP_CAP_ATR * atr))
                    dist = max(dist, C.MIN_STOP_POINTS)
                    risk = C.RISK_PER_TRADE * self.risk_multiplier(nav)
                    if trend == 1 and last["c"] > or_high:
                        self.market_order(1, dist, risk)
                        traded_today += 1
                    elif trend == -1 and last["c"] < or_low:
                        self.market_order(-1, dist, risk)
                        traded_today += 1

                time.sleep(20)

            except KeyboardInterrupt:
                log("Stopped by user")
                break
            except Exception as e:  # noqa: BLE001
                log(f"Loop error: {e}")
                time.sleep(60)


if __name__ == "__main__":
    Bot().run()
