"""
Strategy configuration — NAS100 CFD Opening Range Breakout (ORB).

Instrument: NASDAQ-100 cash index CFD only (no futures).
  - Backtest data: Dukascopy USATECHIDXUSD 1-min bid/ask CSVs
  - Live broker:   OANDA NAS100_USD

IMPORTANT: every parameter below is fixed a priori (from published ORB
research and prop-firm risk rules), NOT optimized against the backtest
data. That makes the entire 5-year backtest out-of-sample by
construction — there is no in-sample/out-of-sample split to leak across
because nothing was fitted. Do not tune these against the backtest and
then re-run; that reintroduces the overfitting the old ML system had.
"""

# ── Session (America/New_York) ───────────────────────────────────────────
SESSION_OPEN   = "09:30"   # US cash open
OR_MINUTES     = 15        # opening range = first 15 one-minute bars
ENTRY_CUTOFF   = "11:30"   # no new entries after this time
EOD_EXIT       = "15:55"   # force-flat before the cash close (no overnight)

# ── Signal ───────────────────────────────────────────────────────────────
TREND_SMA_DAYS = 50        # daily-close SMA; prev close above -> longs only,
                           # below -> shorts only (uses prior days only)
MAX_TRADES_PER_DAY = 1     # first valid breakout only

# ── Stops / targets (all distances in NAS100 index points) ───────────────
ATR15_PERIOD     = 14      # ATR on completed 15-min bars (causal)
STOP_FLOOR_ATR   = 0.25    # stop distance >= 0.25 x ATR15
STOP_CAP_ATR     = 2.00    # stop distance <= 2.00 x ATR15
MIN_STOP_POINTS  = 15.0    # hard floor so a stop-out inside the first
                           # 2 minutes is practically impossible
TP_R             = 4.0     # take profit at 4R
BE_TRIGGER_R     = 1.0     # at +1R (bar close), stop -> true breakeven
                           # (entry + spread + slippage), effective next bar

# ── Risk (prop-firm constraints) ─────────────────────────────────────────
RISK_PER_TRADE   = 0.005   # 0.5% of equity per trade
DAILY_LOSS_STOP  = 0.010   # halt the day at -1.0% equity (buffer vs 3% rule)
# Overall drawdown governor: risk multiplier by current equity drawdown,
# measured against the high-water mark of *prior-day closing equity*
# (causal — never uses the current day's outcome).
DD_TIERS = [               # (drawdown threshold, risk multiplier)
    (0.020, 0.50),         # DD >  2.0%  -> risk 0.25%
    (0.035, 0.25),         # DD >  3.5%  -> risk 0.125%
]
MAX_TOTAL_DD     = 0.06    # 6%  — backtest PASS/FAIL check
MAX_DAILY_DD     = 0.03    # 3%  — backtest PASS/FAIL check
MIN_TRADE_MINUTES = 2      # prop rule — measured and reported

# ── Execution / costs ────────────────────────────────────────────────────
SLIPPAGE_POINTS  = 0.5     # per side, on top of the bid/ask spread
UNIT_STEP        = 1.0     # CFD position size granularity (OANDA: 1 unit)
MIN_UNITS        = 1.0

INITIAL_CAPITAL  = 100_000.0
