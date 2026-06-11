# Why the old system (trainer2.py / validator.py / nas100.py) wasn't working

I reviewed the three uploaded files line by line. The model itself is not
the main problem — the **evaluation pipeline is broken in ways that make
the backtest results unrelated to live behavior**. In rough order of
severity:

## 1. The validator backtests in-sample (fatal)
`trainer2.py` runs `GridSearchCV` (144 combinations) over **all of
2020–2025** and refits the best model on **all of it**. `validator.py`
then "validates" on **the same 2020–2025 data**. The equity curve is a
read-out of memorization, not of an edge. `TimeSeriesSplit` inside the
grid search does not fix this — the final refit still sees every bar the
validator later trades.
*Any* positive result from this pipeline is expected, and none of it
transfers live. This alone explains "backtest looks fine, live doesn't".

## 2. The model was trained on a different session than it trades (fatal)
The Dukascopy CSVs are stamped **EET**. The trainer filters
`between_time("08:00", "12:00")` on those raw EET stamps — that is
**01:00–05:00 New York**, the overnight session. The live bot
(`nas100.py`) converts to `America/New_York` and trades **08:00–11:30
NY**. The model learned overnight microstructure and is being asked to
predict the NY morning. Feature distributions (`minute_of_hour`,
`range_size`, `atr_norm`, …) are from a different market regime.

## 3. The validator and the live bot trade different systems (fatal)
`validator.py` uses `model.predict(...)` — a 0.5 probability threshold —
and takes a position on **every** opening-hour bar it is flat.
`nas100.py` requires `predict_proba >= 0.85` and trades rarely. The
validated equity curve describes a strategy the live bot never runs.

## 4. Validator fill simulation uses bar CLOSES as highs/lows
In `numba_validation_loop`, `prices[i, 4] / prices[i, 5]` are the
**resampled close** ask/bid (`.last()`), but the code treats them as
`high_ask` / `low_bid`. Consequences:
- intra-bar stop hits are missed entirely (drawdown understated),
- long take-profits are checked against the **ask** (a long exits at the
  bid — optimistic by the full spread),
- the 0.75R trailing logic reads the same bar's "favorable price" and
  then checks the same bar's exit — an intra-bar sequence lookahead.

## 5. The live bot's daily-loss guard is dead code
`state["daily_limit"]` is initialized `False` each day and **never set
anywhere**. Live, nothing stops a losing day. The validator meanwhile
enforces −2%. For a prop account with a 3% daily rule this is the single
most dangerous line in the system.

## 6. Risk mismatch
Validator risks **0.9%** per trade; live risks **0.5%**. Neither the
drawdown profile nor the equity curve transfers.

## 7. Smaller train/live mismatches
- GARCH regimes come from a precomputed `*_LOOSENED.csv` offline file;
  live recomputes a 200-day GARCH daily with a different tolerance. If
  the offline thresholds were derived from full-sample statistics, 2020
  bars were classified using 2025 information.
- Trainer min-stop is relative (`close * 0.0002` ≈ 3–4 points); live is
  absolute (5.0 points).
- Kalman state on live's rolling 250-bar window differs from the
  full-history filter values the model was trained on (warm-up drift).
- The API token committed in `nas100.py` should be revoked and moved to
  an environment variable.

## What the replacement does differently
- **No fitted parameters at all.** The ORB strategy's parameters are
  fixed a priori in `config.py` from published opening-range-breakout
  research; the full 5-year backtest is out-of-sample by construction.
  `--sensitivity` shows the result is not a one-point fluke.
- **One engine, one rule set** shared by backtest and live; identical
  session, risk, stop, breakeven, and halt logic.
- **Causality is tested, not asserted**: `tests/test_engine.py` proves
  truncating or mutating future data never changes past trades, and that
  ATR/trend features are identical with and without the future present.
- **Conservative fills**: signals fill on the *next* bar at ask/bid plus
  slippage; stops honor gaps (fill at the worse of stop or open); when
  stop and target are both touched within a bar, the loss is booked.
- **Risk rules are live code, with tests**: 0.5% per trade, hard −1%
  daily halt (buffer vs the 3% rule), drawdown governor that quarters
  risk before 6% is ever in sight, and a forced flat at 15:55 NY so an
  overnight gap can never blow the daily limit.

## A note on honesty
A random-walk input produces a *negative* result in this engine (spread
and slippage drag) — that is the expected behavior of a clean backtest
and a useful smoke test: lookahead bugs typically manufacture profits
even on random data. The strategy's actual 5-year numbers must come from
running `run_backtest.py` on the real Dukascopy CSVs, which only exist
on your machine. If any constraint check prints FAIL there, the answer
is "don't trade it", not "tune until it passes".
