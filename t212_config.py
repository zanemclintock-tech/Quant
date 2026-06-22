"""
Configuration for the Trading 212 news/catalyst swing bot.

Everything here is read once at import. Secrets and per-run knobs come
from environment variables so nothing sensitive lands in git; the
defaults are tuned for a small (sub-£1k) Invest account trading liquid
US large-caps on a daily/weekly horizon.

The whole thing defaults to the DEMO (paper) environment. You have to
deliberately set T212_ENV=live to touch real money.
"""
from __future__ import annotations

import os

# ── API environment ──────────────────────────────────────────────────────
#   demo -> free paper account, live -> real money. Default is demo.
ENV = os.environ.get("T212_ENV", "demo").lower()
BASE_URLS = {
    "demo": "https://demo.trading212.com/api/v0",
    "live": "https://live.trading212.com/api/v0",
}
BASE_URL = BASE_URLS.get(ENV, BASE_URLS["demo"])
API_KEY = os.environ.get("T212_API_KEY", "")

# DRY_RUN never sends orders — it scans, scores and logs what it *would*
# do. Stays on unless you explicitly set T212_DRY_RUN=0.
DRY_RUN = os.environ.get("T212_DRY_RUN", "1") != "0"

# ── Universe ─────────────────────────────────────────────────────────────
# The pool of tickers the bot scans for catalysts. Defaults to the user's
# current holdings plus a liquid US growth/large-cap watchlist (the kind
# of names that actually spike on news). Override with T212_UNIVERSE as a
# comma-separated list, e.g. "AAPL,NVDA,TSLA".
_DEFAULT_UNIVERSE = [
    # current holdings
    "MU", "AMD", "GOOGL", "AAPL", "IBKR", "NVDA", "VUAG.L",
    # liquid momentum / high-beta names that move on catalysts
    "TSLA", "META", "AMZN", "MSFT", "NFLX", "AVGO", "PLTR", "SMCI",
    "COIN", "ARM", "MRVL", "CRWD", "SHOP", "UBER", "QCOM", "DELL",
]
UNIVERSE = [
    t.strip().upper()
    for t in os.environ.get("T212_UNIVERSE", ",".join(_DEFAULT_UNIVERSE)).split(",")
    if t.strip()
]

# ── Signal thresholds ────────────────────────────────────────────────────
# A candidate must clear this combined score (catalyst + sentiment +
# momentum, each in roughly [-1, 1]) to be considered for a buy.
MIN_SIGNAL_SCORE = float(os.environ.get("T212_MIN_SCORE", "0.35"))
# Weighting of the three signal components (need not sum to 1).
W_CATALYST = float(os.environ.get("T212_W_CATALYST", "0.5"))
W_SENTIMENT = float(os.environ.get("T212_W_SENTIMENT", "0.3"))
W_MOMENTUM = float(os.environ.get("T212_W_MOMENTUM", "0.2"))
# News older than this many hours is ignored — a catalyst is only a
# catalyst while it's fresh.
NEWS_LOOKBACK_HOURS = float(os.environ.get("T212_NEWS_HOURS", "48"))

# ── Risk / sizing ────────────────────────────────────────────────────────
RISK_PER_TRADE = float(os.environ.get("T212_RISK", "0.02"))   # 2% of equity
MAX_POSITIONS = int(os.environ.get("T212_MAX_POS", "5"))      # concurrent
MAX_INVESTED_FRAC = float(os.environ.get("T212_MAX_INVESTED", "0.95"))
MIN_ORDER_VALUE = float(os.environ.get("T212_MIN_ORDER", "5"))  # account ccy

# Exit rules (bot-managed, because live T212 only takes market orders —
# there are no native stops). Checked every cycle by polling price.
STOP_LOSS_PCT = float(os.environ.get("T212_STOP_PCT", "0.08"))    # -8%
TAKE_PROFIT_PCT = float(os.environ.get("T212_TP_PCT", "0.20"))    # +20%
MAX_HOLD_DAYS = float(os.environ.get("T212_MAX_HOLD_DAYS", "10")) # time stop
TRAIL_PCT = float(os.environ.get("T212_TRAIL_PCT", "0"))         # 0 = off

# ── Schedule ─────────────────────────────────────────────────────────────
# How often the loop wakes to scan news + manage open positions. Trading
# decisions are gated to TRADE_HOUR so we act once a day, not on every
# headline (this is a swing bot, not an HFT).
SCAN_INTERVAL_MIN = float(os.environ.get("T212_SCAN_MIN", "60"))
TRADE_HOUR_UTC = int(os.environ.get("T212_TRADE_HOUR", "14"))  # ~US pre-open
TRADE_WEEKDAYS_ONLY = os.environ.get("T212_WEEKDAYS_ONLY", "1") != "0"

# ── Optional external news providers (used if the key is set) ────────────
FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")

# ── Files ────────────────────────────────────────────────────────────────
STATE_FILE = os.environ.get("T212_STATE", "t212_state.json")
LEDGER_FILE = os.environ.get("T212_LEDGER", "t212_ledger.csv")
STATUS_FILE = os.environ.get("T212_STATUS", "t212_status.json")


def summary() -> str:
    """One-line human summary of the active config (no secrets)."""
    return (f"env={ENV} dry_run={DRY_RUN} universe={len(UNIVERSE)} "
            f"risk={RISK_PER_TRADE:.1%} max_pos={MAX_POSITIONS} "
            f"stop={STOP_LOSS_PCT:.0%} tp={TAKE_PROFIT_PCT:.0%} "
            f"min_score={MIN_SIGNAL_SCORE}")
