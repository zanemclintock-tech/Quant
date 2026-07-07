# Crypto bot audit — is it fake?

You asked me to scan the crypto bot and find out what is fake, then fix it.
I read the whole pipeline, verified the data against source, ran the tests,
and ran the project's own edge test on **5 full years of real BTC data**.
Here is the honest result.

## Short version

- **The data is real.** Not fabricated, not synthetic-passed-off-as-real.
- **The engine is honest.** No lookahead, costs are charged, fills are
  next-bar. The causality tests pass.
- **But the "edge" is not real.** On 5 years of real data the strategy does
  **not beat a random walk**. Its apparent profitability is a measurement
  artifact that pure noise reproduces just as well.
- **What was misleading** is the *dashboard*: it presented a rolling,
  short-window re-backtest as if it were a live, profitable account. Fixed.
- **One real bug** (test isolation / overnight carry) fixed.

So: not a fraud, but also **not something with a demonstrated edge, and it
was being displayed in a way that implied one.** Don't fund it.

---

## What I verified is REAL

| Claim | Check | Result |
|---|---|---|
| Order-flow data is real Binance data | `of_vol` in `data/btc_of/*.parquet` vs the kline `volume` column from `data.binance.vision` | **Exact match** (e.g. 2024‑06‑01 00:00 = `11.67408` in both) |
| Data is complete & not fabricated | 70 monthly files 2020‑08 → 2026‑05, contiguous, no duplicate file sizes, 40,160–44,640 rows/month | **Real, contiguous** |
| No lookahead in the crypto detector | `tests/test_crypto_causality.py` truncates the future and requires byte‑identical past setups + features | **Passes** |
| Costs are charged | `labeler.py` / `backtest_ftmo.py` charge spread + slippage per round trip, stop checked before target, next‑bar fills | **Honest** |
| The trainer is anti‑overfit | `model_train.py` compares real results to a random‑walk **noise null**, not to zero | **Honest by design** |

The author clearly knew the failure modes of the old ML system (see
`ANALYSIS.md`) and built genuinely rigorous tooling. Credit where due.

## What is actually FAKE — the "edge"

The whole point of `model_train.py` is that conservative fill/label rules
*manufacture* apparent edge, and that same machinery produces a positive
result on data with **zero signal**. So the only honest question is: does the
real data beat what the machinery produces on noise?

I ran it on 5 years of real BTC (2.63M 1‑min bars, real order flow attached),
train ≤2023 / score 2024+, 8 noise surrogates:

```
REAL   OOS AUC 0.540 | selected expectancy +0.352R | edge +0.162R (win 63.2%)
       baseline take-all: +0.191R, win 58.8%

NOISE null (8 runs):
   AUC  mean 0.538   95th pct 0.565      <- real 0.540 is INSIDE the noise band
   baseline take-all mean +0.242R        <- noise is BETTER than the real +0.191R
   edge mean +0.134R  95th pct +0.206R   <- real +0.162R is INSIDE the noise band

 p(noise baseline >= real baseline) = 1.000
 p(AUC  from noise >= real)         = 0.250   (need < 0.05)
 p(edge from noise >= real)         = 0.125   (need < 0.05)

 VERDICT: the real result does NOT clearly beat noise. No demonstrated edge.
```

Full log: `edge_test_5yr_BTC.log`. Reproduce:

```bash
ORDERFLOW=1 CRYPTO=1 CRYPTO_GLOB=data/btc_klines \
  LABEL=directional HORIZON=60 NULL_RUNS=8 python model_train.py
```

**Plain English:** a 63% win rate looks great, but a coin-flip strategy run
through this exact machinery lands in the same place. There is no evidence the
model has learned anything the market actually contains. The dashboard's nice
win rate is small-sample luck on top of a no-edge strategy.

## What I fixed

1. **Dashboard honesty** (`dashboard.py`). It showed `Equity`,
   `Total return`, `Win rate`, and a green equity curve computed from a
   rolling 14–90‑day re-backtest — framed like a live, accumulating account.
   Added a prominent banner stating it is a research backtest, not a live
   track record, that 5-year testing shows no edge, and a small-sample
   warning when the window has < 100 trades.

2. **Test isolation / overnight-carry bug**
   (`tests/test_crypto_causality.py`, `tests/test_smc_detector.py`).
   `test_crypto_causality.py` set `os.environ["CRYPTO"]=1` at *import* time
   with no teardown. Because pytest imports every test module before running,
   that global leaked into the session-market detector tests and put the
   *index* detector into 24/7 mode — which let a setup armed at 15:45 trigger
   an entry the *next morning* (an overnight carry the strategy claims never
   to take). `test_entry_after_arm` was failing because of this. Fixed by
   scoping the env with autouse fixtures that restore it, so mode no longer
   leaks across modules. Suite is green in both orderings.

## What I did NOT change

The strategy logic, the models, and the live runner are left as-is. The point
is not that the code is broken — it mostly isn't. The point is that **the
thing it trades has no demonstrated edge**, and it was being *presented* as if
it did. The fix for that is disclosure and a rigorous verdict, both now in the
repo, not more parameter tuning (that would just be curve-fitting — the exact
mistake `ANALYSIS.md` warns about).

## Bottom line

Trading this live is trading noise with real costs — negative expectancy after
fees. Keep it as a research harness (it's a good one). Do not fund it, and do
not trust the dashboard's win rate as evidence of edge.
