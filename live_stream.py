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

import cloud_sync
import sizing as SZ
from backtest_ftmo import BE_TRIGGER, BE_BUF
from live_runner import ASSETS, COST, TF, fetch, notify, resolve_pairs
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
    # daily risk budget: reset the day-start baseline each UTC day; new fills
    # are sized so today's realised loss + all open risk <= 1.5% (holds the
    # daily loss under the 2% rule even if every stop fills at once).
    utcday = pd.Timestamp(now, unit="s", tz="UTC").normalize()
    if port.get("day") != utcday:
        port["day"] = utcday
        port["day_start"] = port["equity"]
    pos = state["pos"]
    if pos is None:
        for lim in list(state["limits"]):
            if now >= lim["expire"]:
                state["limits"].remove(lim)
                ev.append({"type": "EXPIRE", "asset": asset, **lim})
                continue
            hit = (ask <= lim["entry"] if lim["side"] == "buy"
                   else bid >= lim["entry"])
            if hit:
                sf = lim["risk"] / lim["entry"]
                e_bps, s_bps = COST.get(asset, (2.0, 10.0))
                worst_frac = sf + (e_bps + s_bps) / 1e4   # risk + exit cost
                remaining = (SZ.DAY_BUDGET * INIT
                             - max(0.0, port["day_start"] - port["equity"])
                             - port["open_worst"])
                notl = SZ.size_notional(lim["prob"], sf, port["open_notional"],
                                        INIT,
                                        budget_notional=remaining / worst_frac)
                if notl <= 0.02 * INIT:           # no 1:2 / budget -> skip
                    state["limits"].remove(lim)
                    continue
                wd = notl * worst_frac
                state["pos"] = {"side": lim["side"], "entry": lim["entry"],
                                "stop": lim["stop"], "tp": lim["tp"],
                                "risk": lim["risk"], "t0": now,
                                "notional": notl, "worst_d": wd,
                                "qty": notl / lim["entry"]}
                port["open_notional"] += notl
                port["open_worst"] += wd
                port["peak_lev"] = max(port.get("peak_lev", 0.0),
                                       port["open_notional"] / INIT)
                state["limits"].clear()           # one position per asset
                ev.append({"type": "FILL", "asset": asset,
                           "price": lim["entry"], "qty": notl / lim["entry"]})
                break
    else:
        if now - pos["t0"] < MIN_HOLD_S:           # honour 2-min min hold
            return ev
        # break-even: once the realisable exit price has run BE_TRIGGER x risk
        # in our favour, ratchet the stop to entry (+tiny buffer).
        if BE_TRIGGER is not None:
            d = 1 if pos["side"] == "buy" else -1
            fav = bid if pos["side"] == "buy" else ask
            if (fav - pos["entry"]) * d / pos["risk"] >= BE_TRIGGER:
                be = pos["entry"] + d * BE_BUF * pos["entry"]
                pos["stop"] = max(pos["stop"], be) if d == 1 \
                    else min(pos["stop"], be)
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
            port["open_worst"] -= pos["worst_d"]
            port["equity"] += r * risk_dollar
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


def write_status(state=None, port=None):
    """Rebuild status.json from the real-fill ledger (+ live open positions
    and resting limits from `state`) so the dashboard shows live monthly %,
    equity, win rate, drawdown, open trades and active limit orders."""
    import json
    led = (pd.read_csv(LEDGER, parse_dates=["exit_time"])
           if os.path.exists(LEDGER) else pd.DataFrame())
    if len(led):
        eq = float(led["equity_after"].iloc[-1])
        led["m"] = led["exit_time"].dt.strftime("%Y-%m")
        monthly = (led.groupby("m")["pnl"].sum() / INIT * 100).round(2)
        eqc = led.set_index("exit_time")["equity_after"]
        mdd = float(((eqc.cummax() - eqc) / eqc.cummax()).max())
        win = round((led["R_net"] > 0).mean() * 100, 1)
        ntr = int(len(led))
    else:
        eq, monthly, mdd, win, ntr = INIT, pd.Series(dtype=float), 0.0, 0.0, 0
    if port is not None:
        eq = port.get("equity", eq)

    # live open positions + resting limit orders straight from state
    pending, open_n = [], 0
    if state is not None:
        now = pd.Timestamp.utcnow()
        for a, st in state.items():
            if st["pos"] is not None:
                open_n += 1
            for lim in st["limits"]:
                mins = (lim["expire"] - now.timestamp()) / 60
                pending.append({
                    "asset": a, "side": lim["side"],
                    "limit": round(lim["entry"], 2), "price": round(lim["entry"], 2),
                    "away_%": 0.0, "stop": round(lim["stop"], 2),
                    "target": round(lim["tp"], 2),
                    "expires_in_min": int(max(mins, 0))})

    status = {
        "equity": round(eq, 2),
        "total_ret_pct": round((eq / INIT - 1) * 100, 2),
        "this_month_pct": float(monthly.iloc[-1]) if len(monthly) else 0.0,
        "win_rate": win, "trades": ntr, "open_positions": open_n,
        "maxdd_pct": round(mdd * 100, 2),
        "peak_leverage": round(port.get("peak_lev", 0.0), 2) if port else 0.0,
        "monthly": monthly.to_dict(),
        "pending": sorted(pending, key=lambda r: r["expires_in_min"]),
        "updated": str(pd.Timestamp.utcnow())}
    json.dump(status, open("status.json", "w"), indent=2, default=str)


async def candle_loop(ex_rest, pairs, state, bundles):
    while True:
        try:
            for a, sym in pairs.items():
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
                    write_status(state, port)
                    cloud_sync.push(min_interval=0)  # instant on a closed trade
                    ico = "✅" if e["R"] > 0 else "❌"
                    rem = (SZ.DAY_BUDGET * INIT
                           - max(0.0, port["day_start"] - port["equity"])
                           - port["open_worst"])
                    tag = " · DAILY BUDGET USED" if rem <= 0.02 * INIT else ""
                    notify(f"{ico} {a} {e['side']} {e['outcome'].upper()} "
                           f"{e['R']:+.2f}R ({e['pnl']:+,.0f}) · "
                           f"equity ${port['equity']:,.0f}{tag}")
        except Exception as ex:
            print(f"  {a} quote error: {ex}")
            await asyncio.sleep(2)


async def sync_loop(state, port, every=20):
    """Heartbeat: refresh status.json (open trades, resting limits, equity)
    and push it + the ledger to the cloud gist so the phone view stays live
    even when nothing is trading."""
    while True:
        try:
            write_status(state, port)
            cloud_sync.push()
        except Exception as ex:
            print(f"  sync error: {ex}")
        await asyncio.sleep(every)


async def main():
    import ccxt
    import ccxt.pro as ccxtpro
    ex_id = os.environ.get("EXCHANGE", "kucoin")
    ex_rest = getattr(ccxt, ex_id)({"enableRateLimit": True})
    ex_ws = getattr(ccxtpro, ex_id)({"enableRateLimit": True})
    # only trade the coins THIS exchange actually lists (e.g. Kraken/Coinbase
    # have no BNB) -- skip the rest instead of crashing.
    pairs = resolve_pairs(ex_rest, [s.split("/")[0] for s in ASSETS.values()])
    missing = [a for a in ASSETS if a not in pairs]
    bundles = {a: joblib.load(f"models/{a}_{TF}.joblib") for a in pairs}
    state = {a: {"limits": [], "pos": None} for a in pairs}
    # shared portfolio: 1:2 exposure budget + daily-loss circuit breaker
    port = {"open_notional": 0.0, "equity": INIT, "day": None,
            "day_start": INIT, "open_worst": 0.0, "peak_lev": 0.0}
    cloud = " + cloud sync" if os.environ.get("GIST_ID") else ""
    print(f"[{ex_id}] streaming bid/ask fills | "
          f"{[f'{a}={s}' for a, s in pairs.items()]} @ {TF}{cloud}")
    if missing:
        print(f"  (not listed on {ex_id}, skipped: {missing})")
    notify(f"▶️ Streaming runner (real bid/ask, adaptive 1:2) on {ex_id} — "
           f"{', '.join(pairs)}" + (f" (no {','.join(missing)})" if missing else ""))
    try:
        await asyncio.gather(
            candle_loop(ex_rest, pairs, state, bundles),
            sync_loop(state, port),
            *[quote_loop(ex_ws, a, sym, state, port)
              for a, sym in pairs.items()])
    finally:
        await ex_ws.close()


if __name__ == "__main__":
    asyncio.run(main())
