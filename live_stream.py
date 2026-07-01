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
from backtest_ftmo import BE_TRIGGER, BE_BUF, BE_LOCK, MAX_HOLD_MIN
from live_runner import (ASSETS, COST, TF, fetch, notify, resolve_pairs,
                         rows_to_frame)
# CRITICAL: pin the detector's structural timeframe to the model's TF (15min)
# BEFORE importing/using detect_setups. _base_tf() defaults to 30min, so without
# this the streaming engine would detect on 30-min bars while loading 15-min
# models -- wrong features and a setup grid that never matches the 15-min
# last_closed, so NO limit ever arms (silent: heartbeats, no error, no trades).
os.environ["BASE_TF"] = TF
os.environ["CRYPTO"] = "1"
from smc_detector import detect_setups, _base_tf

INIT = 100_000.0
RISK = float(os.environ.get("RISK", "0.005"))
LIMIT_TTL_MIN = int(os.environ.get("LIMIT_TTL_MIN", "30"))   # resting window
MIN_HOLD_S = 120
# spread circuit-breaker: skip an entry if the live round-trip spread would
# cost more than this fraction of the trade's risk (R). 0.5 ignores normal
# spreads (Goat baseline ~3-5bps is <0.05R) but blocks news/weekend spikes
# ($150-300 on BTC) that would otherwise burn the daily limit on a stop.
SPREAD_GATE_R = float(os.environ.get("SPREAD_GATE_R", "0.5"))
# days of 1-min history to pull each detect cycle. The detector's longest
# lookback is ~50 bars (15m), so 5 days is ample for correct features on the
# latest bar while keeping the fetch small (one detect per 15-min close).
DETECT_DAYS = int(os.environ.get("DETECT_DAYS", "3"))
CHECK_S = int(os.environ.get("CHECK_S", "120"))   # re-check for setups cadence
LEDGER = "ledger.csv"
LEDGER_COLS = ["entry_time", "exit_time", "hold_min", "asset", "side",
               "outcome", "entry_px", "stop", "tp", "exit_px", "risk_pct",
               "lev", "R_net", "pnl", "equity_after", "open"]

# live health/heartbeat -- surfaced in status.json so the phone dashboard shows
# whether the engine is actually detecting and streaming, not just "running".
# detect_tf vs model_tf catches the exact silent misconfig that stops all
# arming (detector on 30min while models are 15min).
HEALTH = {
    "model_tf": TF, "detect_tf": _base_tf(), "started": None,
    "last_detect": None, "last_closed_bar": None,
    "setups_on_last_bar": 0, "passed_gate": 0, "armed_total": 0, "resting": 0,
    "candle_updated": {}, "candle_src": {}, "ticks": {}, "last_tick": {},
    "last_error": None, "last_error_t": None}


def _hnow():
    return str(pd.Timestamp.utcnow())


def _herr(where, ex):
    HEALTH["last_error"] = f"{where}: {ex}"[:300]
    HEALTH["last_error_t"] = _hnow()


# ---------- restart persistence ------------------------------------------
STATE_FILE = "state.json"


def save_state(state, port):
    """Persist open positions, resting limits and the portfolio (atomically)
    so a crash/restart can't reset equity to $100k, drop an open position, or
    forget today's used risk budget."""
    import json
    doc = {"port": {**port, "day": str(port["day"]) if port.get("day") is not None
                    else None},
           "state": state, "saved": _hnow()}
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, default=str)
    os.replace(tmp, STATE_FILE)


def load_state(state, port):
    """Restore a prior run's positions/limits/portfolio into the fresh dicts.
    Open exposure is recomputed from the restored positions (not trusted from
    disk) and stale limits are left for on_quote's expiry check to clear."""
    import json
    if not os.path.exists(STATE_FILE):
        # older runs had no state file -- at least don't reset equity if the
        # ledger already shows a different balance.
        if os.path.exists(LEDGER):
            led = pd.read_csv(LEDGER)
            if len(led):
                port["equity"] = float(led["equity_after"].iloc[-1])
        return False
    doc = json.load(open(STATE_FILE))
    p = doc.get("port", {})
    for k in ("equity", "day_start", "peak_lev"):
        if k in p:
            port[k] = float(p[k])
    if p.get("day"):
        port["day"] = pd.Timestamp(p["day"])
    port["open_notional"] = port["open_worst"] = 0.0
    for a, st in doc.get("state", {}).items():
        if a not in state:
            continue
        state[a]["limits"] = st.get("limits", [])
        state[a]["pos"] = st.get("pos")
        if state[a]["pos"]:
            port["open_notional"] += state[a]["pos"]["notional"]
            port["open_worst"] += state[a]["pos"]["worst_d"]
    return True


def ensure_ledger():
    """Create an empty (header-only) ledger at startup if none exists, so the
    cloud gist gets a clean slate instead of keeping a stale prior run."""
    if not os.path.exists(LEDGER):
        with open(LEDGER, "w", newline="") as f:
            csv.writer(f).writerow(LEDGER_COLS)


# ---------- pure fill engine (unit-tested) ------------------------------
def approved_entries(df, bundle, last_closed, diag=None, xref=None):
    """Model-approved setups that triggered on the last closed bar. If `diag`
    is given, record how many setups triggered on that bar (pre-gate) and how
    many passed the model -- so the health panel can tell 'no setups' from
    'setups but all rejected' from 'timeframe mismatch'. xref = BTC candle
    frame for the cross-asset lead-lag features."""
    m, thr, feats = bundle["model"], bundle["threshold"], bundle["feats"]
    out = []
    on_bar = passed = 0
    for s in detect_setups(df, xref=xref):
        if s.entry_time != last_closed:
            continue
        on_bar += 1
        x = pd.DataFrame([{f: s.features.get(f, np.nan) for f in feats}])
        pr = float(m.predict_proba(x[feats])[:, 1][0])
        if pr < thr:
            continue
        passed += 1
        risk = abs(s.entry_price - s.stop)
        if risk <= 0:
            continue
        tp = (s.entry_price - bundle["target_rr"] * risk if s.direction < 0
              else s.entry_price + bundle["target_rr"] * risk)
        out.append({"side": "sell" if s.direction < 0 else "buy",
                    "entry": s.entry_price, "stop": s.stop, "tp": tp,
                    "risk": risk, "prob": pr})
    if diag is not None:
        diag["on_bar"] = on_bar
        diag["passed"] = passed
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
            # A resting limit may ONLY fill on a genuine retrace INTO it: price
            # must first be seen on its far side (above a buy / below a sell),
            # then come back to the level. Without this, a setup armed when the
            # market has ALREADY passed its entry fills instantly at a phantom
            # price and stops out immediately -- a big loss on a market that
            # never moved. (This is what the backtest's next-bar retrace fill
            # models; the live engine was missing it.)
            far = (ask > lim["entry"]) if lim["side"] == "buy" \
                else (bid < lim["entry"])
            if far:
                lim["ready"] = True
            if not lim.get("ready"):
                continue
            hit = (ask <= lim["entry"] if lim["side"] == "buy"
                   else bid >= lim["entry"])
            if hit:
                # SPREAD CIRCUIT-BREAKER: refuse to enter when the live spread
                # has spiked (news / thin-hour / weekend widening).
                spread = max(ask - bid, 0.0)
                if lim["risk"] > 0 and spread / lim["risk"] > SPREAD_GATE_R:
                    continue
                # fill at the REAL market, never worse than our limit (a buy
                # fills at the ask if it has dipped below the limit, a sell at
                # the bid) -- so the recorded entry can't diverge from reality.
                entry_px = (min(lim["entry"], ask) if lim["side"] == "buy"
                            else max(lim["entry"], bid))
                risk = abs(entry_px - lim["stop"])
                if risk <= 0:                     # entry already at/through stop
                    state["limits"].remove(lim)
                    continue
                sf = risk / entry_px
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
                state["pos"] = {"side": lim["side"], "entry": entry_px,
                                "stop": lim["stop"], "tp": lim["tp"],
                                "risk": risk, "t0": now,
                                "notional": notl, "worst_d": wd,
                                "qty": notl / entry_px}
                port["open_notional"] += notl
                port["open_worst"] += wd
                port["peak_lev"] = max(port.get("peak_lev", 0.0),
                                       port["open_notional"] / INIT)
                state["limits"].clear()           # one position per asset
                ev.append({"type": "FILL", "asset": asset, "price": entry_px,
                           "qty": notl / entry_px, "stop": lim["stop"],
                           "tp": lim["tp"], "t0": now})
                break
    else:
        if now - pos["t0"] < MIN_HOLD_S:           # honour 2-min min hold
            return ev
        # profit lock: once the realisable exit price has run BE_TRIGGER x risk
        # in our favour, ratchet the stop to entry + BE_LOCK x risk (locks in
        # +0.5R even if the move reverses after 1.5R; BE_LOCK=0 = classic BE).
        if BE_TRIGGER is not None:
            d = 1 if pos["side"] == "buy" else -1
            fav = bid if pos["side"] == "buy" else ask
            if (fav - pos["entry"]) * d / pos["risk"] >= BE_TRIGGER:
                be = pos["entry"] + d * (BE_LOCK * pos["risk"] if BE_LOCK > 0
                                         else BE_BUF * pos["entry"])
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
        # time exit: the backtest closes at MAX_HOLD_MIN (8h) and its stats
        # price that in (taker exit). Without this, a live trade that never
        # reaches stop or TP blocks its asset indefinitely and carries open
        # risk across days the budget never planned for.
        if out is None and now - pos["t0"] >= MAX_HOLD_MIN * 60:
            px = bid if pos["side"] == "buy" else ask
            out = "maxhold"
        if out:
            d = -1 if pos["side"] == "sell" else 1
            r = (px - pos["entry"]) * d / pos["risk"]
            risk_dollar = pos["notional"] * pos["risk"] / pos["entry"]
            port["open_notional"] -= pos["notional"]
            port["open_worst"] -= pos["worst_d"]
            port["equity"] += r * risk_dollar
            ev.append({"type": "EXIT", "asset": asset, "outcome": out,
                       "price": px, "R": r, "pnl": r * risk_dollar,
                       "entry": pos["entry"], "stop": pos["stop"],
                       "tp": pos["tp"], "risk": pos["risk"],
                       "notional": pos["notional"], "side": pos["side"],
                       "t0": pos["t0"]})
            state["pos"] = None
    return ev


# ---------- live wiring (ccxt.pro) --------------------------------------
def log_trade(e, equity, now):
    """Full trade record so every fill is auditable: entry/exit time, hold,
    fill prices, stop, TP, risk %, leverage, R and P&L."""
    new = not os.path.exists(LEDGER)
    t0 = pd.Timestamp(e["t0"], unit="s", tz="UTC")
    t1 = pd.Timestamp(now, unit="s", tz="UTC")
    with open(LEDGER, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LEDGER_COLS)
        w.writerow([t0, t1, round((now - e["t0"]) / 60, 2), e["asset"],
                    e["side"], e["outcome"], round(e["entry"], 6),
                    round(e["stop"], 6), round(e["tp"], 6), round(e["price"], 6),
                    round(e["risk"] / e["entry"] * 100, 4),
                    round(e["notional"] / INIT, 3), round(e["R"], 3),
                    round(e["pnl"], 2), round(equity, 2), False])


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
        # seed the $100k starting base so drawdown is measured from the start,
        # not the first trade's equity (else a single losing trade shows 0% DD)
        seed = pd.Series([INIT], index=[eqc.index.min() - pd.Timedelta(seconds=1)])
        eqc = pd.concat([seed, eqc])
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
        "health": HEALTH,
        "updated": str(pd.Timestamp.utcnow())}
    json.dump(status, open("status.json", "w"), indent=2, default=str)


async def _fetch_retry(ex, sym, tries=4, **kw):
    """fetch() with backoff on transient rate limits, so a flaky public pool
    (KuCoin 'system-level' 429s) is ridden through within the cycle instead of
    waiting 15 min for the next one."""
    for i in range(tries):
        try:
            return fetch(ex, sym, **kw)
        except Exception as e:
            transient = "429" in str(e) or "rate limit" in str(e).lower()
            if i < tries - 1 and transient:
                await asyncio.sleep(5 * (i + 1))     # 5, 10, 15s
                continue
            raise


def _merge_bars(cache, a, new):
    df = pd.concat([cache[a], new]) if a in cache else new
    df = df[~df.index.duplicated(keep="last")].sort_index()
    cache[a] = df.iloc[-DETECT_DAYS * 1440:]           # trim history


async def ohlcv_loop(ex_ws, a, sym, cache, ws_ok):
    """Push-based 1m candles over the WebSocket (watch_ohlcv). Once running,
    the REST poll for this asset stops entirely -- the rate-limit exposure
    that caused the KuCoin 429s drops to just the one-off history seed. If the
    venue doesn't support it, flags ws_ok[a]=False and the detect loop keeps
    the old incremental REST poll as fallback."""
    if not getattr(ex_ws, "has", {}).get("watchOHLCV"):
        ws_ok[a] = False
        print(f"  {a}: no WS candles on this venue -- REST fallback")
        return
    while True:
        try:
            rows = await ex_ws.watch_ohlcv(sym, "1m")
            if a not in cache or not rows:
                continue                     # wait for the REST history seed
            _merge_bars(cache, a, rows_to_frame(rows))
            ws_ok[a] = True
            HEALTH["candle_updated"][a] = _hnow()
            HEALTH["candle_src"][a] = "ws"
        except Exception as ex:
            if "notsupported" in type(ex).__name__.lower():
                ws_ok[a] = False
                print(f"  {a}: WS candles unsupported -- REST fallback")
                return
            print(f"  {a} candle WS error: {ex}")
            _herr(f"ohlcv {a}", ex)
            ws_ok[a] = None                  # unknown -> detect loop tops up
            await asyncio.sleep(2)


async def candle_loop(ex_rest, pairs, state, bundles, cache, ws_ok, port):
    # Detect on CLOSED 15-min bars, re-checking every CHECK_S seconds. Candles
    # arrive by WS push (ohlcv_loop); REST is only the one-off history seed
    # plus a fallback top-up if the WS stream is unsupported or goes stale.
    while True:
        armed_total = 0
        on_bar_total = passed_total = 0
        for a, sym in pairs.items():
            try:
                if a not in cache:
                    _merge_bars(cache, a,
                                await _fetch_retry(ex_rest, sym, days=DETECT_DAYS))
                    HEALTH["candle_updated"][a] = _hnow()
                    HEALTH["candle_src"][a] = "rest-seed"
                else:
                    age = (pd.Timestamp.now(tz="UTC")
                           - cache[a].index[-1]).total_seconds()
                    if ws_ok.get(a) is not True or age > 180:
                        last_ms = int(cache[a].index[-1].timestamp() * 1000) - 120_000
                        _merge_bars(cache, a,
                                    await _fetch_retry(ex_rest, sym, since_ms=last_ms))
                        HEALTH["candle_updated"][a] = _hnow()
                        HEALTH["candle_src"][a] = "rest"
                df = cache[a]
                # last FULLY-CLOSED 15m bar by wall clock -- robust whether or
                # not the current forming bar's data has arrived yet.
                res = df["mid_c"].resample(TF).last()
                nowu = pd.Timestamp.now(tz="UTC")
                done = res.index[res.index + pd.Timedelta(TF) <= nowu]
                if len(done) == 0:
                    continue
                last_closed = done[-1]
                HEALTH["last_closed_bar"] = str(last_closed)
                if state[a]["pos"] is not None:
                    continue
                have = {round(l["entry"], 2) for l in state[a]["limits"]}
                diag = {}
                for e in approved_entries(df, bundles[a], last_closed, diag,
                                          xref=cache.get("BTC")):
                    if round(e["entry"], 2) in have:
                        continue
                    e["expire"] = time.time() + LIMIT_TTL_MIN * 60
                    state[a]["limits"].append(e)
                    armed_total += 1
                    save_state(state, port)
                    notify(f"🔔 {a} {e['side']} limit @ {e['entry']:.2f} "
                           f"(stop {e['stop']:.2f}, tp {e['tp']:.2f})")
                on_bar_total += diag.get("on_bar", 0)
                passed_total += diag.get("passed", 0)
            except Exception as ex:
                print(f"  candle error {a}: {ex}")
                _herr(f"candle {a}", ex)
        resting = sum(len(state[a]["limits"]) for a in pairs)
        openp = sum(state[a]["pos"] is not None for a in pairs)
        HEALTH["last_detect"] = _hnow()
        HEALTH["setups_on_last_bar"] = on_bar_total
        HEALTH["passed_gate"] = passed_total
        HEALTH["armed_total"] += armed_total
        HEALTH["resting"] = resting
        print(f"  {pd.Timestamp.utcnow():%H:%M:%S} detect: {resting} resting "
              f"limit(s), {openp} open, {armed_total} new "
              f"[{on_bar_total} setups on last bar, {passed_total} passed gate]")
        await asyncio.sleep(CHECK_S)


async def quote_loop(ex_ws, a, sym, state, port):
    while True:
        try:
            # best bid/ask via the WS ticker feed -- pure push, NO REST order
            # book snapshots (watch_order_book re-syncs over REST and exhausts
            # KuCoin's small keyless rate-limit pool).
            t = await ex_ws.watch_ticker(sym)
            bid, ask = t.get("bid"), t.get("ask")
            if bid is None or ask is None:
                continue
            now = time.time()
            HEALTH["ticks"][a] = HEALTH["ticks"].get(a, 0) + 1
            HEALTH["last_tick"][a] = _hnow()
            for e in on_quote(a, bid, ask, state[a], now, port):
                if e["type"] == "FILL":
                    save_state(state, port)
                    notify(f"➡️ FILLED {a} {e['side']} @ {e['price']:.2f} "
                           f"(lev now {port['open_notional']/INIT:.2f}x)")
                elif e["type"] == "EXIT":
                    save_state(state, port)
                    log_trade(e, port["equity"], now)  # equity updated in on_quote
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
            _herr(f"quote {a}", ex)
            await asyncio.sleep(2)


async def sync_loop(state, port, every=60):
    """Heartbeat: refresh status.json (open trades, resting limits, equity)
    and push it + the ledger to the cloud gist so the phone view stays live
    even when nothing is trading. 60s keeps well clear of GitHub's write
    limits; closed trades still push instantly via quote_loop."""
    while True:
        try:
            save_state(state, port)     # catches BE ratchets + expiries too
            write_status(state, port)
            cloud_sync.push(min_interval=55)
        except Exception as ex:
            print(f"  sync error: {ex}")
        await asyncio.sleep(every)


def _creds():
    """ccxt config, with API keys if present (authenticated requests get a far
    larger rate-limit pool). For KuCoin, API_PASSWORD is the passphrase."""
    c = {"enableRateLimit": True}
    if os.environ.get("API_KEY"):
        c["apiKey"] = os.environ["API_KEY"]
        c["secret"] = os.environ.get("API_SECRET", "")
        if os.environ.get("API_PASSWORD"):
            c["password"] = os.environ["API_PASSWORD"]
    return c


async def main():
    import ccxt
    import ccxt.pro as ccxtpro
    ex_id = os.environ.get("EXCHANGE", "kucoin")
    ensure_ledger()                              # clean slate -> clears stale gist
    ex_rest = getattr(ccxt, ex_id)(_creds())
    ex_ws = getattr(ccxtpro, ex_id)(_creds())
    auth = " (authenticated)" if os.environ.get("API_KEY") else " (public)"
    # COINS picks the lineup. Default BTC,ETH,SOL -- the set validated against
    # Goat's REAL per-coin spreads (LTC's ~38bps spread makes it a net loser;
    # BNB is marginal and adds drawdown under stressed spreads). Override with
    # COINS=BTC,ETH,SOL,BNB,LTC to trade all five (e.g. on a tight exchange).
    want = [c.strip().upper() for c in
            os.environ.get("COINS", "BTC,ETH,SOL").split(",") if c.strip()]
    # only trade coins THIS exchange actually lists (e.g. Kraken has no BNB).
    try:
        pairs = resolve_pairs(ex_rest, want)
    except ccxt.AuthenticationError as e:
        # bad/incomplete API key (e.g. wrong KuCoin passphrase) -> don't crash;
        # fall back to public access. With the WS ticker feed + incremental
        # candles, public REST usage is tiny and usually fine on its own.
        print(f"  API key rejected ({str(e)[:80]}); using PUBLIC access.")
        await ex_ws.close()
        ex_rest = getattr(ccxt, ex_id)({"enableRateLimit": True})
        ex_ws = getattr(ccxtpro, ex_id)({"enableRateLimit": True})
        auth = " (public — key rejected, fix API_PASSWORD to authenticate)"
        pairs = resolve_pairs(ex_rest, want)
    missing = [a for a in want if a not in pairs]
    bundles = {a: joblib.load(f"models/{a}_{TF}.joblib") for a in pairs}
    state = {a: {"limits": [], "pos": None} for a in pairs}
    # shared portfolio: 1:2 exposure budget + daily-loss circuit breaker
    port = {"open_notional": 0.0, "equity": INIT, "day": None,
            "day_start": INIT, "open_worst": 0.0, "peak_lev": 0.0}
    # restore a prior run: equity, today's used budget, open positions and
    # resting limits all survive a restart (expiry clears anything stale).
    if load_state(state, port):
        npos = sum(state[a]["pos"] is not None for a in state)
        nlim = sum(len(state[a]["limits"]) for a in state)
        print(f"  restored state: equity ${port['equity']:,.0f}, "
              f"{npos} open, {nlim} resting limit(s)")
    HEALTH["started"] = _hnow()
    HEALTH["detect_tf"] = _base_tf()        # record the ACTUAL detector TF
    cloud = " + cloud sync" if os.environ.get("GIST_ID") else ""
    print(f"[{ex_id}]{auth} streaming bid/ask fills | "
          f"{[f'{a}={s}' for a, s in pairs.items()]} @ {TF}{cloud}")
    if HEALTH["detect_tf"] != TF:
        print(f"  WARNING: detector TF {HEALTH['detect_tf']} != model TF {TF} "
              f"-- nothing will arm. (BASE_TF must equal {TF}.)")
    if missing:
        print(f"  (not listed on {ex_id}, skipped: {missing})")
    notify(f"▶️ Streaming runner (real bid/ask, adaptive 1:2) on {ex_id} — "
           f"{', '.join(pairs)}" + (f" (no {','.join(missing)})" if missing else ""))
    cache, ws_ok = {}, {}
    try:
        await asyncio.gather(
            candle_loop(ex_rest, pairs, state, bundles, cache, ws_ok, port),
            sync_loop(state, port),
            *[ohlcv_loop(ex_ws, a, sym, cache, ws_ok)
              for a, sym in pairs.items()],
            *[quote_loop(ex_ws, a, sym, state, port)
              for a, sym in pairs.items()])
    finally:
        await ex_ws.close()


if __name__ == "__main__":
    asyncio.run(main())
