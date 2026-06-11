# NAS100 CFD — Opening Range Breakout (prop-firm constrained)

A complete, lookahead-free intraday strategy for the **NASDAQ-100 cash
index CFD** (Dukascopy `USATECHIDXUSD` for backtesting, OANDA
`NAS100_USD` for live). No futures, no overnight exposure, no HFT.

## The strategy in one paragraph

At the 09:30 New York cash open, record the high/low of the first 15
one-minute bars (the opening range). Take **at most one trade per day**:
the first 1-minute close beyond the range, but only in the direction of
the 50-day trend (previous close vs 50-day SMA — longs above, shorts
below). Stop = the opening-range width, clamped to [0.25×ATR15,
2×ATR15] and never under 15 points; target = 4R; stop moves to true
breakeven (entry + spread + slippage) after +1R on a completed bar; any
open position is force-flattened at 15:55 NY. Opening-range breakout on
NASDAQ is one of the few intraday effects with published academic
support (Zarattini & Aziz, 2023); everything else here is risk control.

## How each of your constraints is met

| Constraint | Mechanism |
|---|---|
| Risk = 0.5% per trade | Position sized as `equity × 0.5% / stop_distance`; tested (`test_risk_fraction_respected`) |
| Daily drawdown < 3% | Max 1 trade/day at 0.5% ⇒ worst normal day ≈ −0.6%; hard −1.0% equity halt closes everything and blocks re-entry; no overnight gap risk (flat by 15:55) |
| Max drawdown < 6% | DD governor: risk ×0.5 beyond 2% DD, ×0.25 beyond 3.5% (measured off prior-day close HWM — causal); backtest report FAILs if 6% intraday DD is ever touched |
| Trades ≥ 2 min (no HFT) | 15-point minimum stop + breakout entries make sub-2-min exits practically impossible; the report counts violations explicitly |
| No lookahead | Signals on bar close fill at the NEXT bar's ask/bid + slippage; features proven causal by tests that mutate/truncate the future; stop-before-target inside every bar; gap-through stops fill at the open |
| 5-year proof | `run_backtest.py` runs the full 2020–2025 Dukascopy file and prints PASS/FAIL per constraint. Parameters were **not fitted to the data**, so the whole period is out-of-sample |

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 12 checks: causality, risk, accounting

# the real 5-year backtest (CSVs are on your machine):
python run_backtest.py \
  --bid "USATECHIDXUSD_1 Min_Bid_2020.08.30_2025.09.05.csv" \
  --ask "USATECHIDXUSD_1 Min_Ask_2020.08.30_2025.09.05.csv" \
  --sensitivity

# engine smoke test without data:
python run_backtest.py --synthetic
```

Outputs: constraint report (stdout), `output/trades.csv`,
`output/equity_daily.csv`, `output/equity_curve.png`.

The exit code is 0 only when **every** constraint passes, so you can
gate deployment on it.

## Going live

```bash
export OANDA_TOKEN=...      # never hardcode (revoke the token that was
export OANDA_ACCOUNT=...    #  committed in the old nas100.py!)
export ACCOUNT_CURRENCY=GBP # or USD for prop accounts
python live_oanda.py
```

`live_oanda.py` mirrors the backtest engine decision-for-decision.
Forward-run it on a **demo account for at least a month** and diff its
daily decisions against the backtest on the same dates before funding.

## Rules of the road

1. If a constraint FAILs on the real data, the strategy is not deployable.
   Do **not** tune parameters until it passes — that is curve fitting,
   the exact failure mode of the previous ML system (see `ANALYSIS.md`).
2. `--sensitivity` exists to show robustness *around* the fixed
   parameters, not to pick better ones.
3. The spread on NAS100 CFDs widens outside US cash hours and around
   news; the backtest uses the real per-bar bid/ask from Dukascopy, but
   live OANDA spreads can differ — demo-forward comparison is the check.

## Files

| File | Purpose |
|---|---|
| `config.py` | Every parameter, fixed a priori, documented |
| `data_loader.py` | Dukascopy CSV → NY-timezone bid/ask frame (fixes the EET bug, see ANALYSIS) |
| `engine.py` | Event-driven 1-min backtest engine + risk layer |
| `run_backtest.py` | 5-year run, constraint PASS/FAIL report, plots |
| `live_oanda.py` | Live runner with the same rules |
| `synthetic.py` | Random-walk data for engine verification |
| `tests/test_engine.py` | No-lookahead proofs, risk/accounting checks |
| `ANALYSIS.md` | Post-mortem of the previous ML system |
