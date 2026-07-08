# Intraday NAS100 / S&P500 CFD — honest strategy pipeline

A complete, no-lookahead pipeline to code the ICT/SMC intraday playbook (Powell
"10am order block", PB Blake/Patty, AhmedyFX, Trading Pool), backtest it truly
net of CFD spread + slippage, **gate it against a hostile null + hold-out**, and
decide — via a prop-firm Monte-Carlo — whether it's worth a paid evaluation.

The pipeline is finished and validated. The only thing this cloud sandbox could
not do is fetch 5 years of intraday index data: Dukascopy IP-banned us after a
parallel-download burst, and every no-API-key fallback (HistData, Yahoo, Stooq)
is either JS-gated or blocked here. **Run the two commands below on your own
machine** (normal IP) and you'll get the real 5-year gated result in minutes.

## Run it

```bash
pip install pandas numpy pyarrow requests

# 1) pull 5y of real Dukascopy bid/ask for both indices (resumable, ~15-30 min)
python cfd_smc/download_cfd.py

# 2) honest backtest + gate (per variant: train / 2024 / 2025 hold-out,
#    paired direction-flip permutation null, BH correction)
python cfd_smc/run_cfd_backtest.py --name SP500
python cfd_smc/run_cfd_backtest.py --name NAS100

# 3) for anything that CLEARS the 2025 gate, feed its per-trade R to the
#    prop-firm simulator to get a BUY / DON'T-PAY decision
python cfd_smc/prop_sim.py        # demo; wire your strategy's r_net + firm rules
```

Already have data? Drop 1-minute OHLC CSVs (or parquet with open/high/low/close/
spread) into `cfd_smc/data_cache/` named `SP500_1min_YYYYMM.parquet` (or point
`run_cfd_backtest.py --file`) and it runs immediately.

## What the gate means (why this one is different)

- **Paired direction-flip null:** for every setup, we also compute the R it
  would have made with the long/short call reversed — same timing, risk, R:R,
  session, cost. If your real direction can't beat a coin-flip on the *same*
  setups (`p<0.05`, BH-corrected), the "edge" is R:R mechanics, not signal.
- **Nested hold-out:** discover on ≤2023, validate on 2024, judge ONCE on 2025.
- **Real costs always on:** spread + slippage charged on entry and exit.
- A variant is believed ONLY if 2025 is net-positive AND beats the null.

## Honest expectation

This is built to tell the truth, which includes the real chance that these
strategies don't clear the gate — exactly as the crypto SMC system didn't
(see `../CRYPTO_AUDIT.md`, `../EDGE_SEARCH_RESULTS.md`). The manual-vs-coded gap
is real: a discretionary "70% at 2R" usually falls toward ~40-50% coded
objectively. If a component *does* survive, `prop_sim.py` tells you whether it
clears an evaluation profitably — and refuses ("DON'T PAY") when there's no edge.

## Files
| File | Purpose |
|---|---|
| `data_duka.py` | Dukascopy tick → bar loader (real bid/ask spread) |
| `download_cfd.py` | Gentle, resumable 5y session-hours downloader |
| `strategies.py` | No-lookahead intraday SMC engine (toggleable components) |
| `run_cfd_backtest.py` | Nested hold-out + paired-null gate + BH report |
| `prop_sim.py` | Prop-firm evaluation Monte-Carlo (edge-gated decision) |
