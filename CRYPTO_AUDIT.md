# Crypto bot audit — is the edge real?

Audit of the **live** system on this branch: `live_stream.py` + `live_dashboard.py`,
pooled model `models/pooled_15min.joblib`, trained by `train_live_models.py` /
`label_live.py` / `calibrate_live_threshold.py`. Question asked: are the headline
numbers (~18%/mo, ~2.9% max DD, ~75% win) real?

## Verdict
**No.** They are an **in-sample + selection artifact** on a signal that does **not
beat a random walk**. The raw strategy has **negative expectancy after costs**. The
model is a demo/research artifact, not a funded-safe edge. Do not fund it.

## Evidence (all reproducible on this repo)

### 1. The "18%" is measured in-sample (overfitting)
`train_live_models.py` fit on every row, then read the metric off the same rows:
```
m = B._model().fit(pooled[feats], pooled[label])   # fit on ALL rows
p = m.predict_proba(pooled[feats])[:, 1]           # score the SAME rows
thr = np.quantile(pr, 0.84)                          # threshold from in-sample scores
```
No train/test split. `ANALYSIS.md` in this repo calls exactly that pattern "fatal."
The claims lived in `train_live_models.py` ("~18%/mo, ~2.9% max DD, PF ~2.6"),
`sizing.py` ("+16.8%/mo at 5.5% max DD") and an inline note ("18.6→21.9%").

### 2. Raw expectancy is NEGATIVE (the system's own live labels)
Take-all filled setups from `data/trades/{BTC,ETH,SOL}_live.parquet`:
BTC 42.9% win / **−0.066R**, ETH 42.7% / **−0.058R**, SOL 42.0% / **−0.033R**.
The 2R bracket with costs loses on average; "edge" appears only after the model
cherry-picks the top 16% by (in-sample) probability.

### 3. It does not beat noise (the project's own test)
`model_train.py` compares the real result to the same machinery on random-walk noise,
on 5 years of real BTC (2.63M 1-min bars + real order flow):
- Directional label: `p(AUC≥real)=0.25`, `p(edge≥real)=0.13`.
- Bracket label (the deployed method): `p(AUC≥real)=0.63`, `p(edge≥real)=0.50` — the
  real result is **worse than the median noise run**. On pure noise this same bracket
  selection manufactures +0.26…+0.36R of "edge."
Reproduce: `ORDERFLOW=1 CRYPTO=1 CRYPTO_GLOB=<klines> LABEL=bracket NULL_RUNS=8
python model_train.py` → "no demonstrated edge."

### 4. In-sample vs honest OOS
`train_live_models.py` now prints a held-out verdict itself. On the pooled live labels,
train ≤2023 / test 2024+: selected win 55.3%, selected +0.547R, **take-all −0.086R**.
The selected OOS number looks alive — but #3 shows that positive bracket expectancy is
exactly what noise produces, so it is the artifact, not skill.

### 5. Detection live-faithfulness — tested, and it is NOT the problem
Your #1 suspect was detection lookahead: setups armed in the backtest that the live
rolling-`DETECT_DAYS`-window path can't arm in time. `audit_live_faithfulness.py`
re-detects both ways over a held-out window and labels them identically:
```
window 2024-01-19..2024-03-01 (6w)  DETECT_DAYS=3
FULL-HISTORY : 286 | win 36.0% | mean R -0.111
LIVE-FAITHFUL: 344 | win 27.6% | mean R -0.135
recovered live: 81% of full-history setups
FULL-ONLY (live missed): 53 | win 22.6% | mean R -1.198   ← the missed ones are junk
```
**81%** of backtest setups are armable live, and the ones live misses are the **worst**
(−1.2R). So detection timing is not inflating the backtest — both populations simply
lose at the raw level. (Absolute win rates here use a simplified next-bar bracket, not
the live retrace-gate fill, so compare full-vs-live within the table, not to the live
dashboard.) The known live/train gap is a DIFFERENT issue: `calibrate_live_threshold.py`
documents live scores sitting ~0.15 below training (~8× under-arming), patched by
lowering the live threshold — which means live trades a lower-scoring set than the
in-sample "18%" ever described.

## What is genuinely real (fair to the author)
- Data is real Binance data (verified: `of_vol` == source kline `volume` to the
  decimal; 70 contiguous months, no fabrication).
- Fills/costs are honest (next-bar, stop-before-target, spread charged).
- The frame-level no-lookahead test (`tests/test_crypto_causality.py`) passes.
- `live_dashboard.py` shows a genuinely accumulating demo ledger (not a re-backtest).
- `model_train.py` is a rigorous edge test — and it is the tool that proves no edge.

## Fixes applied on this branch
- `train_live_models.py`: removed the in-sample performance claims from the docstring;
  `main()` now prints a held-out OOS number and an explicit **NOT-funded-safe** verdict
  (`_honest_oos_verdict`).
- `sizing.py`: replaced "+16.8%/mo…" with the risk-invariant it can actually guarantee
  plus an honesty note.
- `live_dashboard.py`: added a "no demonstrated edge / demo research" banner and a
  small-sample warning.
- `audit_live_faithfulness.py` + `tests/test_live_faithfulness.py`: the detection-
  faithfulness audit, committed and testable.

## Bottom line
Negative expectancy after costs; the 18% was in-sample; two independent noise tests say
no edge; and live trades a different, lower-scoring set than the backtest that produced
the headline. Keep it as a research harness. Do not fund it.
