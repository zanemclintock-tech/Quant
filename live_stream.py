"""
Real bid/ask streaming fill engine -- the prop-close upgrade.

Instead of filling at candle mid, this streams live best bid/ask (public
WebSocket via ccxt.pro -- no account) and fills HONESTLY:

  * a resting limit fills only when the market actually trades through it
    (buy: ask <= limit; sell: bid >= limit) -- at the limit price (maker);
  * a stop exits at the real opposite quote (crossing the spread, with
    whatever slippage the live book gives);
  * a take-profit fills as a limit at its level;
  * min 2-minute hold, one position per asset (no hedging).

Model-approved 15-min signals are detected from candles (REST, each bar)
and registered as resting limits with an expiry; the live quote stream
decides if/where they fill. Real fills -> ledger.csv + Telegram pings, so
the same dashboard now shows REAL-quote results, not mid approximations.

    EXCHANGE=binance python live_stream.py        # public WS, no keys
"""
from __future__ import annotations

import asyncio
import csv
import os
import time

import joblib
import numpy as np
import pandas as pd

from live_runner import ASSETS, TF, fetch, notify
from smc_detector import detect_setups

INIT = 100_000.0
RISK = float(os.environ.get("RISK", "0.005"))
LIMIT_TTL_MIN = int(os.environ.get("LIMIT_TTL_MIN", "30"))   # resting window
MIN_HOLD_S = 120
LEDGER = "ledger.csv"


# ---------- pure fill engine (unit-tested) ------------------------------
def approved_entries(df, bundle, last_closed):
    """Model-approved setups that triggered on the last closed bar."""
    m, thr, feats = bundle["model"], bundle["threshold"], bundle["feats"]
    out = []
    for s in detect_setups(df):
        if s.entry_time != last_closed:
            continue
        x = pd.DataFrame([{f: s.features.get(f, np.nan) for f in feats}])
        pr = float(m.predict_proba(x[feats])[:, 1][0])
        if pr < thr:
            continue
        risk = abs(s.entry_price - s.stop)
        if risk <= 0:
            continue
        tp = (s.entry_price - bundle["target_rr"] * risk if s.direction < 0
              else s.entry_price + bundle["target_rr"] * risk)
        out.append({"side": "sell" if s.direction < 0 else "buy",
                    "entry": s.entry_price, "stop": s.stop, "tp": tp,
                    "risk": risk, "prob": pr})
    return out


def on_quote(asset, bid, ask, state, now, port):
    """Process one bid/ask tick. Mutates state + shared portfolio `port`
    ({"open_notional"}); returns event dicts. Position size is adaptive by
    confidence and hard-capped so total exposure stays <= 1:2."""
    import sizing as SZ
    ev = []
    # daily-loss circuit breaker: reset on each UTC day, halt new fills once
    # the day is down DAILY_STOP (keeps realised daily loss under the 2% rule)
    utcday = pd.Timestamp(now, unit="s", tz="UTC").normalize()
    if port.get("day") != utcday:
        port["day"] = utcday
        port["day_start"] = port["equity"]
        port["day_stopped"] = False
    pos = state["pos"]
    if pos is None:
        for lim in list(state["limits"]):
            if now >= lim["expire"]:
                state["limits"].remove(lim)
                ev.append({"type": "EXPIRE", "asset": asset, **lim})
                continue
            if port.get("day_stopped"):              # daily stop hit -> no new
                continue
            hit = (ask <= lim["entry"] if lim["side"] == "buy"
                   else bid >= lim["entry"])
            if hit:
                sf = lim["risk"] / lim["entry"]
                notl = SZ.size_notional(lim["prob"], sf,
                                        port["open_notional"], INIT)
                if notl <= 0.02 * INIT:           # no 1:2 budget -> skip
                    state["limits"].remove(lim)
                    continue
                state["pos"] = {"side": lim["side"], "entry": lim["entry"],
                                "stop": lim["stop"], "tp": lim["tp"],
                                "risk": lim["risk"], "t0": now,
                                "notional": notl, "qty": notl / lim["entry"]}
                port["open_notional"] += notl
                state["limits"].clear()           # one position per asset
                ev.append({"type": "FILL", "asset": asset,
                           "price": lim["entry"], "qty": notl / lim["entry"]})
                break
    else:
        if now - pos["t0"] < MIN_HOLD_S:           # honour 2-min min hold
            return ev
        px = out = None
        if pos["side"] == "buy":
            if bid <= pos["stop"]:
                px, out = bid, "sl"                # market exit at real bid
            elif bid >= pos["tp"]:
                px, out = pos["tp"], "tp"
        else:
            if ask >= pos["stop"]:
                px, out = ask, "sl"
            elif ask <= pos["tp"]:
                px, out = pos["tp"], "tp"
        if out:
            d = -1 if pos["side"] == "sell" else 1
            r = (px - pos["entry"]) * d / pos["risk"]
            risk_dollar = pos["notional"] * pos["risk"] / pos["entry"]
            port["open_notional"] -= pos["notional"]
            port["equity"] += r * risk_dollar
            if (port["day_start"] - port["equity"]) / port["day_start"] \
                    >= SZ.DAILY_STOP:
                port["day_stopped"] = True
            ev.append({"type": "EXIT", "asset": asset, "outcome": out,
                       "price": px, "R": r, "pnl": r * risk_dollar,
                       "entry": pos["entry"], "side": pos["side"],
                       "t0": pos["t0"]})
            state["pos"] = None
    return ev


# ---------- live wiring (ccxt.pro) --------------------------------------
def log_trade(e, equity):
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["exit_time", "asset", "side", "outcome", "R_net",
                        "pnl", "equity_after", "open"])
        w.writerow([pd.Timestamp.utcnow(), e["asset"], e["side"],
                    e["outcome"], round(e["R"], 3), round(e["pnl"], 2),
                    round(equity, 2), False])


def write_status():
    """Rebuild status.json from the real-fill ledger so the dashboard
    (use_local) shows live monthly %, equity, win rate, drawdown."""
    if not os.path.exists(LEDGER):
        return
    import json
    led = pd.read_csv(LEDGER, parse_dates=["exit_time"])
    if not len(led):
        return
    eq = led["equity_after"].iloc[-1]
    led["m"] = led["exit_time"].dt.strftime("%Y-%m")
    monthly = (led.groupby("m")["pnl"].sum() / INIT * 100).round(2)
    eqc = led.set_index("exit_time")["equity_after"]
    mdd = ((eqc.cummax() - eqc) / eqc.cummax()).max()
    status = {
        "equity": round(float(eq), 2),
        "total_ret_pct": round((eq / INIT - 1) * 100, 2),
        "this_month_pct": float(monthly.iloc[-1]) if len(monthly) else 0.0,
        "win_rate": round((led["R_net"] > 0).mean() * 100, 1),
        "trades": int(len(led)), "open_positions": 0,
        "maxdd_pct": round(float(mdd) * 100, 2),
        "monthly": monthly.to_dict(), "pending": [],
        "updated": str(pd.Timestamp.utcnow())}
    json.dump(status, open("status.json", "w"), indent=2, default=str)


async def candle_loop(ex_rest, state, bundles):
    while True:
        try:
            for a, sym in ASSETS.items():
                df = fetch(ex_rest, sym, days=7)
                last_closed = df["mid_c"].resample(TF).last().index[-2]
                if state[a]["pos"] is not None:
                    continue
                have = {round(l["entry"], 2) for l in state[a]["limits"]}
                for e in approved_entries(df, bundles[a], last_closed):
                    if round(e["entry"], 2) in have:
                        continue
                    e["expire"] = time.time() + LIMIT_TTL_MIN * 60
                    state[a]["limits"].append(e)
                    notify(f"🔔 {a} {e['side']} limit @ {e['entry']:.2f} "
                           f"(stop {e['stop']:.2f}, tp {e['tp']:.2f})")
        except Exception as ex:
            print(f"  candle error: {ex}")
        await asyncio.sleep(60)


async def quote_loop(ex_ws, a, sym, state, port):
    while True:
        try:
            ob = await ex_ws.watch_order_book(sym, limit=5)
            bid, ask = ob["bids"][0][0], ob["asks"][0][0]
            for e in on_quote(a, bid, ask, state[a], time.time(), port):
                if e["type"] == "FILL":
                    notify(f"➡️ FILLED {a} {e['side']} @ {e['price']:.2f} "
                           f"(lev now {port['open_notional']/INIT:.2f}x)")
                elif e["type"] == "EXIT":
                    log_trade(e, port["equity"])     # equity updated in on_quote
                    write_status()
                    ico = "✅" if e["R"] > 0 else "❌"
                    tag = " · DAILY STOP HIT" if port["day_stopped"] else ""
                    notify(f"{ico} {a} {e['side']} {e['outcome'].upper()} "
                           f"{e['R']:+.2f}R ({e['pnl']:+,.0f}) · "
                           f"equity ${port['equity']:,.0f}{tag}")
        except Exception as ex:
            print(f"  {a} quote error: {ex}")
            await asyncio.sleep(2)


async def main():
    import ccxt
    import ccxt.pro as ccxtpro
    ex_id = os.environ.get("EXCHANGE", "binance")
    bundles = {a: joblib.load(f"models/{a}_{TF}.joblib") for a in ASSETS}
    state = {a: {"limits": [], "pos": None} for a in ASSETS}
    # shared portfolio: 1:2 exposure budget + daily-loss circuit breaker
    port = {"open_notional": 0.0, "equity": INIT, "day": None,
            "day_start": INIT, "day_stopped": False}
    ex_rest = getattr(ccxt, ex_id)({"enableRateLimit": True})
    ex_ws = getattr(ccxtpro, ex_id)({"enableRateLimit": True})
    print(f"[{ex_id}] streaming bid/ask fills | {list(ASSETS)} @ {TF}")
    notify(f"▶️ Streaming runner (real bid/ask, adaptive 1:2) on {ex_id}")
    try:
        await asyncio.gather(
            candle_loop(ex_rest, state, bundles),
            *[quote_loop(ex_ws, a, sym, state, port)
              for a, sym in ASSETS.items()])
    finally:
        await ex_ws.close()


if __name__ == "__main__":
    asyncio.run(main())
