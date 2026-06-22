"""
Trading 212 news/catalyst swing bot — main runner.

Loop (default every SCAN_INTERVAL_MIN minutes):

  1. Pull account + open positions from Trading 212.
  2. MANAGE — for every open position, check the bot-managed exits
     (stop-loss, take-profit, trailing stop, time stop) against the live
     price and submit a market SELL when one trips. Live T212 has no
     native stops, so the bot is the stop.
  3. SCAN — fetch fresh news for the whole universe, score catalyst +
     sentiment (+ momentum), rank.
  4. TRADE — once per day at TRADE_HOUR_UTC, open up to the free slots
     with market BUYs sized by risk.

Everything defaults to the DEMO account and DRY_RUN=on: it logs the
orders it *would* place without sending them. Flip T212_DRY_RUN=0 (and,
to use real money, T212_ENV=live) only once you trust it.

    T212_API_KEY=... python t212_bot.py            # demo, dry-run
    T212_API_KEY=... T212_DRY_RUN=0 python t212_bot.py   # demo, live orders
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time

import t212_config as C
from t212_client import T212Client, T212Error
from t212_news import scan_universe
from t212_signals import rank_signals, select_buys, size_position


# ── notifications (same Telegram pattern as live_runner) ─────────────────
def notify(msg: str) -> None:
    """Telegram push if TELEGRAM_TOKEN/TELEGRAM_CHAT are set; else no-op."""
    import urllib.parse
    import urllib.request
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT")
    if not (tok and chat):
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": msg,
                                   "parse_mode": "HTML"}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://api.telegram.org/bot{tok}/sendMessage", data=data),
            timeout=10)
    except Exception as e:
        print(f"  notify error: {e}")


# ── state ────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if os.path.exists(C.STATE_FILE):
        with open(C.STATE_FILE) as f:
            return json.load(f)
    return {"positions": {}, "equity_hwm": 0.0}


def save_state(state: dict) -> None:
    with open(C.STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def append_ledger(row: dict) -> None:
    import csv
    new = not os.path.exists(C.LEDGER_FILE)
    with open(C.LEDGER_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# ── pure exit logic (unit-tested) ────────────────────────────────────────
def decide_exit(pos: dict, current_price: float,
                now: dt.datetime) -> tuple[bool, str]:
    """Should this position be closed? Returns (close?, reason).

    ``pos`` is our stored state: entry_price, stop, target, peak, opened_at.
    Trailing (if TRAIL_PCT>0) ratchets the stop up under the running peak.
    """
    entry = pos["entry_price"]
    if entry <= 0 or current_price <= 0:
        return False, ""
    if current_price <= pos["stop"]:
        return True, "stop_loss"
    if current_price >= pos["target"]:
        return True, "take_profit"
    if C.TRAIL_PCT > 0:
        peak = max(pos.get("peak", entry), current_price)
        if current_price <= peak * (1 - C.TRAIL_PCT):
            return True, "trailing_stop"
    opened = pos.get("opened_at")
    if opened:
        opened_dt = (dt.datetime.fromisoformat(opened)
                     if isinstance(opened, str) else opened)
        if (now - opened_dt).total_seconds() / 86400 >= C.MAX_HOLD_DAYS:
            return True, "time_stop"
    return False, ""


# ── price helper for sizing new entries ──────────────────────────────────
def last_price(symbol: str) -> float | None:
    """Last close for sizing a new buy (yfinance if available, else None)."""
    try:
        import yfinance as yf
        closes = yf.Ticker(symbol).history(period="5d")["Close"].dropna()
        return float(closes.iloc[-1]) if len(closes) else None
    except Exception:
        return None


# ── cycle steps ──────────────────────────────────────────────────────────
def manage_positions(cli: T212Client, state: dict, now: dt.datetime) -> None:
    """Check exits on every held position and submit sells when tripped."""
    try:
        portfolio = {p["ticker"]: p for p in cli.portfolio()}
    except T212Error as e:
        print(f"  portfolio fetch failed: {e}")
        return
    for ticker, pos in list(state["positions"].items()):
        live = portfolio.get(ticker)
        if live is None:
            # Closed outside the bot (or filled/manual). Forget it.
            print(f"  {ticker}: no longer held, dropping from state")
            state["positions"].pop(ticker, None)
            continue
        price = float(live.get("currentPrice") or 0.0)
        if C.TRAIL_PCT > 0:
            pos["peak"] = max(pos.get("peak", pos["entry_price"]), price)
        close, reason = decide_exit(pos, price, now)
        if not close:
            continue
        qty = float(live.get("quantity") or pos["qty"])
        pnl_pct = (price / pos["entry_price"] - 1) * 100
        print(f"  EXIT {ticker} {reason} @ {price:.2f} "
              f"({pnl_pct:+.1f}%)  dry_run={C.DRY_RUN}")
        if not C.DRY_RUN:
            try:
                cli.place_market_order(ticker, -abs(qty))
            except T212Error as e:
                print(f"    sell failed: {e}")
                continue
        append_ledger({
            "symbol": pos.get("symbol", ticker), "ticker": ticker,
            "side": "SELL", "reason": reason,
            "entry_time": pos.get("opened_at"), "exit_time": now.isoformat(),
            "entry_px": round(pos["entry_price"], 4), "exit_px": round(price, 4),
            "qty": round(qty, 4), "pnl_pct": round(pnl_pct, 2),
            "dry_run": int(C.DRY_RUN)})
        notify(f"{'✅' if pnl_pct >= 0 else '❌'} SELL {pos.get('symbol', ticker)} "
               f"{reason} {pnl_pct:+.1f}% @ {price:.2f}"
               + (" [dry]" if C.DRY_RUN else ""))
        state["positions"].pop(ticker, None)


def open_positions(cli: T212Client, state: dict, signals, equity: float,
                   free_cash: float, now: dt.datetime) -> None:
    """Open new buys for the best signals into the free slots."""
    held = set(state["positions"].keys())
    held_syms = {p.get("symbol") for p in state["positions"].values()}
    slots = C.MAX_POSITIONS - len(state["positions"])
    if slots <= 0:
        print("  no free slots")
        return
    picks = select_buys([s for s in signals if s.symbol not in held_syms],
                        held_syms, slots)
    if not picks:
        print("  no signals clear the threshold")
        return
    for s in picks:
        ticker = cli.resolve(s.symbol)
        if not ticker:
            print(f"  {s.symbol}: not tradable on T212, skipping")
            continue
        if ticker in held:
            continue
        price = last_price(s.symbol)
        if price is None:
            print(f"  {s.symbol}: no price for sizing, skipping")
            continue
        qty = size_position(equity, free_cash, price)
        value = qty * price
        if qty <= 0:
            print(f"  {s.symbol}: position too small, skipping")
            continue
        print(f"  BUY {s.symbol} ({ticker}) qty {qty} @~{price:.2f} "
              f"= {value:.2f}  score {s.score:+.2f}  dry_run={C.DRY_RUN}")
        if not C.DRY_RUN:
            try:
                cli.place_market_order(ticker, qty)
            except T212Error as e:
                print(f"    buy failed: {e}")
                continue
        state["positions"][ticker] = {
            "symbol": s.symbol, "entry_price": price, "qty": qty,
            "stop": round(price * (1 - C.STOP_LOSS_PCT), 4),
            "target": round(price * (1 + C.TAKE_PROFIT_PCT), 4),
            "peak": price, "opened_at": now.isoformat(),
            "headline": s.headline[:160], "score": round(s.score, 3)}
        free_cash -= value
        append_ledger({
            "symbol": s.symbol, "ticker": ticker, "side": "BUY",
            "reason": "signal", "entry_time": now.isoformat(),
            "exit_time": "", "entry_px": round(price, 4), "exit_px": "",
            "qty": round(qty, 4), "pnl_pct": "", "dry_run": int(C.DRY_RUN)})
        notify(f"🟢 BUY {s.symbol} {qty}@~{price:.2f} score {s.score:+.2f} "
               f"· {s.headline[:80]}" + (" [dry]" if C.DRY_RUN else ""))


def write_status(state: dict, signals, equity: float, free_cash: float,
                 now: dt.datetime) -> None:
    status = {
        "updated": now.isoformat(), "env": C.ENV, "dry_run": C.DRY_RUN,
        "equity": round(equity, 2), "free_cash": round(free_cash, 2),
        "equity_hwm": round(state.get("equity_hwm", 0.0), 2),
        "open_positions": len(state["positions"]),
        "positions": state["positions"],
        "top_signals": [s.as_row() for s in signals[:8]],
    }
    with open(C.STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2, default=str)


def is_trade_time(now: dt.datetime, last_trade_day) -> bool:
    if C.TRADE_WEEKDAYS_ONLY and now.weekday() >= 5:
        return False
    if now.hour < C.TRADE_HOUR_UTC:
        return False
    return now.date() != last_trade_day


def run_cycle(cli: T212Client, state: dict, last_trade_day) -> object:
    now = dt.datetime.now(dt.timezone.utc)
    print(f"\n=== cycle {now.isoformat()} | {C.summary()} ===")

    # account snapshot
    try:
        cash = cli.account_cash()
        equity = float(cash.get("total") or cash.get("free") or 0.0)
        free_cash = float(cash.get("free") or 0.0) * C.MAX_INVESTED_FRAC
    except T212Error as e:
        print(f"  account fetch failed: {e}; using last-known")
        equity = state.get("equity_hwm", 0.0) or 1000.0
        free_cash = equity * C.MAX_INVESTED_FRAC
    state["equity_hwm"] = max(state.get("equity_hwm", 0.0), equity)

    # 1+2: manage open positions every cycle
    manage_positions(cli, state, now)

    # 3: scan the whole universe for catalysts
    print(f"  scanning {len(C.UNIVERSE)} tickers...")
    news = scan_universe(C.UNIVERSE)
    signals = rank_signals(news)
    print("  top:", ", ".join(f"{s.symbol}{s.score:+.2f}"
                              for s in signals[:5]))

    # 4: trade once per day at the trade hour
    if is_trade_time(now, last_trade_day):
        print("  trade window open")
        open_positions(cli, state, signals, equity, free_cash, now)
        last_trade_day = now.date()
    else:
        print(f"  outside trade window (hour {now.hour} vs {C.TRADE_HOUR_UTC})")

    write_status(state, signals, equity, free_cash, now)
    save_state(state)
    return last_trade_day


def main() -> None:
    if not C.API_KEY:
        print("WARNING: T212_API_KEY not set — account/order calls will fail. "
              "Scanning still works for a dry preview.")
    print(f"Trading 212 catalyst bot | {C.summary()}")
    notify(f"▶️ T212 catalyst bot up | {C.summary()}")
    cli = T212Client()
    state = load_state()
    last_trade_day = None
    while True:
        try:
            last_trade_day = run_cycle(cli, state, last_trade_day)
        except Exception as e:
            print(f"  cycle error: {e}")
        time.sleep(C.SCAN_INTERVAL_MIN * 60)


if __name__ == "__main__":
    main()
