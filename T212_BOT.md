# Trading 212 — News/Catalyst Swing Bot

An automated swing-trading bot for a [Trading 212](https://www.trading212.com)
Invest account. It continuously scans the web for catalysts (earnings beats,
FDA approvals, M&A, upgrades, guidance raises, contract wins...) across a
universe of liquid US stocks, scores each name, and on a daily cadence opens
market-order positions in the strongest fresh stories — then manages the exits
itself (stop-loss / take-profit / trailing / time stop), because the live
Trading 212 API only accepts market orders and has no native stops.

**Defaults to the DEMO (paper) account and dry-run mode.** It will not touch
real money or place a single order until you explicitly opt in.

## Why this is possible

Trading 212 ships an official (beta) REST API: account & portfolio data, Pie
management, and equity orders. On a live account only **market orders** are
supported, so the bot is the stop-loss. Docs: <https://docs.trading212.com/api>.

## Files

| File | Role |
|------|------|
| `t212_config.py` | All knobs (env-var driven). Universe, risk, schedule, thresholds. |
| `t212_client.py` | Rate-limit-aware REST client (stdlib only). Account, portfolio, orders, pies. |
| `t212_news.py`   | Web news scanner + financial sentiment & catalyst lexicons. |
| `t212_signals.py`| Combines catalyst + sentiment + momentum → ranked buys + sizing. |
| `t212_bot.py`    | Runner: manage exits → scan → trade. Telegram alerts, CSV ledger, JSON status. |
| `tests/test_t212.py` | Offline unit tests (no network/key needed). |

## Setup

1. In the Trading 212 app: **Settings → API** → generate a key. Use the
   **practice/demo** key first.
2. Export config:

   ```bash
   export T212_API_KEY="your-demo-key"
   export T212_ENV=demo          # demo (default) or live
   # optional richer news + price momentum:
   export FINNHUB_KEY="..."      # free tier at finnhub.io
   pip install yfinance          # enables the momentum filter
   # optional phone alerts:
   export TELEGRAM_TOKEN="..."; export TELEGRAM_CHAT="..."
   ```

3. Run a **dry-run preview** (scans + scores + logs intended orders, sends
   nothing):

   ```bash
   python t212_bot.py
   ```

4. When you trust it, let it place **demo** orders:

   ```bash
   T212_DRY_RUN=0 python t212_bot.py
   ```

5. Only after proving it on demo, repeat with `T212_ENV=live` and a live key.

## How it decides

```
score = 0.5·catalyst + 0.3·sentiment + 0.2·momentum     # each in [-1, 1]
```

A name is bought when `score ≥ MIN_SIGNAL_SCORE` (0.35), the catalyst is not
negative, and a slot is free. Catalysts only count from articles that actually
name the ticker, so an unrelated company's "earnings miss" can't poison a score.

Positions are sized to risk a fixed fraction of equity given the stop distance
(`risk·equity / stop%`), capped by free cash.

## Key knobs (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `T212_ENV` | `demo` | `demo` or `live` |
| `T212_DRY_RUN` | `1` | `0` to actually send orders |
| `T212_UNIVERSE` | holdings + watchlist | comma-separated tickers to scan |
| `T212_RISK` | `0.02` | fraction of equity risked per trade |
| `T212_MAX_POS` | `5` | max concurrent positions |
| `T212_STOP_PCT` / `T212_TP_PCT` | `0.08` / `0.20` | stop / take-profit |
| `T212_MAX_HOLD_DAYS` | `10` | time stop |
| `T212_TRAIL_PCT` | `0` | trailing stop (0 = off) |
| `T212_MIN_SCORE` | `0.35` | buy threshold |
| `T212_SCAN_MIN` | `60` | minutes between scans |
| `T212_TRADE_HOUR` | `14` | UTC hour to open trades (once/day) |

## Outputs

* `t212_status.json` — live snapshot (equity, positions, top signals).
* `t212_ledger.csv` — one row per order (buys + closes with P&L%).
* `t212_state.json` — open positions with their bot-managed stops/targets.
* Telegram push on every buy/sell if configured.

## Honest limitations

* **News sentiment is lexicon-based**, not an LLM — fast and free, but it
  misreads sarcasm/nuance. Treat scores as a coarse filter.
* **"Constantly scans" = polls on an interval** (default hourly). True realtime
  would need a streaming news provider and would blow past the API rate limits.
* **Market orders only on live**, so exits suffer slippage and gaps can jump the
  stop. The bot polls; it cannot guarantee a fill price.
* **No backtest of this exact strategy is included** — catalyst/news data is hard
  to get historically for free. Prove it on the demo account before risking cash.
* This is a tool, not financial advice. You are responsible for what it trades.

## Tests

```bash
python -m pytest tests/test_t212.py -q
```
