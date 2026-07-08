# Edge search — honest conclusion

**Verdict: no tradeable edge survived.** A pre-registered search (price momentum,
mean-reversion, range, volatility, taker imbalance, and 5 order-flow signals; 3
assets; 15min + 1h; horizons 1/4/12) with a circular-shift permutation null,
Benjamini-Hochberg correction, and a nested train/2024/2025 hold-out found:

1. **A real but tiny effect: short-horizon MEAN REVERSION.** Momentum IC is
   consistently negative (~-0.04 to -0.07, permutation p=0.0000). Crypto
   intraday reverses. This is genuine and statistically strong.
2. **It does not beat costs.** Gross reversion capture is ~1 bp/trade; even a
   generous 3 bps round-trip cost makes nearly every signal net-NEGATIVE.
   Realistic retail crypto fees (10-20 bps round trip) bury it completely.
3. **The order-flow signals add nothing** beyond being weaker proxies of the
   same reversion — no independent edge in CVD / taker-imbalance / big prints.
4. **The one train+2024 survivor died on the 2025 hold-out** (p 0.006 -> 0.567,
   net -0.82 -> -4.91 bps). Exactly the in-sample-shimmer-that-evaporates
   pattern that made every prior "fix" look real and then fail live.

This is WHY the last 11 months went the way they did: there is a faint,
real statistical signal that any backtest without honest costs or a true
hold-out will "discover" and mistake for edge. It is below the cost floor.

**What is real to build on (per the plan's Phase 2/3):** the only thing with a
statistical basis is intraday mean-reversion, and it is not tradeable at retail
cost. So the honest system is capital preservation: do not trade this signal
set with real money; it has negative expectancy after fees. Chasing 18%/mo
against it is what produced the losses. The search harness (edge_search.py) is
reusable: any NEW idea must clear this same permutation-null + hold-out gate,
net of cost, before it is believed.

Cost note: the 3-8 bps assumed here is GENEROUS. At real Binance retail fees the
picture is strictly worse, so this negative conclusion is robust.

---

# Edge search results

## Timeframe 15min
Discovery: 81 tests on train (<= 2023-12-31). Survivors (BH q<0.05 AND net>0): 0.
Top 12 by permutation p (train):
| asset | signal | h | IC | perm p | BH q | net bps |
|---|---|---|---|---|---|---|
| BTC | mom_1 | 1 | -0.0701 | 0.0000 | 0.000 | -1.91 |
| BTC | mom_4 | 1 | -0.0568 | 0.0000 | 0.000 | -2.75 |
| BTC | mom_12 | 1 | -0.0433 | 0.0000 | 0.000 | -2.90 |
| BTC | revert_4 | 1 | +0.0568 | 0.0000 | 0.000 | -2.75 |
| BTC | range_pos | 1 | -0.0682 | 0.0000 | 0.000 | -1.68 |
| BTC | taker_imb | 1 | -0.0367 | 0.0000 | 0.000 | -2.64 |
| BTC | ofi | 1 | -0.0367 | 0.0000 | 0.000 | -2.64 |
| BTC | ofi_ma4 | 1 | -0.0300 | 0.0000 | 0.000 | -2.78 |
| BTC | mom_12 | 4 | -0.0534 | 0.0000 | 0.000 | -2.86 |
| BTC | cvd_slope12 | 1 | -0.0232 | 0.0000 | 0.000 | -2.78 |
| BTC | mom_4 | 4 | -0.0409 | 0.0000 | 0.000 | -3.09 |
| BTC | mom_1 | 4 | -0.0408 | 0.0000 | 0.000 | -2.16 |

## Timeframe 1h
Discovery: 81 tests on train (<= 2023-12-31). Survivors (BH q<0.05 AND net>0): 2.
Top 12 by permutation p (train):
| asset | signal | h | IC | perm p | BH q | net bps |
|---|---|---|---|---|---|---|
| BTC | mom_1 | 1 | -0.0463 | 0.0000 | 0.000 | -3.05 |
| BTC | mom_4 | 1 | -0.0692 | 0.0000 | 0.000 | -1.75 |
| BTC | mom_12 | 1 | -0.0420 | 0.0000 | 0.000 | -3.11 |
| BTC | revert_4 | 1 | +0.0692 | 0.0000 | 0.000 | -1.75 |
| BTC | range_pos | 1 | -0.0579 | 0.0000 | 0.000 | -2.38 |
| BTC | taker_imb | 1 | -0.0457 | 0.0000 | 0.000 | -2.00 |
| BTC | ofi | 1 | -0.0456 | 0.0000 | 0.000 | -2.01 |
| BTC | ofi_ma4 | 1 | -0.0416 | 0.0000 | 0.000 | -2.96 |
| BTC | mom_4 | 4 | -0.0610 | 0.0000 | 0.000 | +0.03 |
| BTC | cvd_slope12 | 1 | -0.0265 | 0.0000 | 0.000 | -3.31 |
| BTC | mom_1 | 4 | -0.0349 | 0.0000 | 0.000 | -1.57 |
| BTC | revert_4 | 4 | +0.0610 | 0.0000 | 0.000 | +0.03 |
- SURVIVOR BTC mom_4 h=4 → 2024 validation: IC -0.0457, perm p 0.006, net -0.82 bps (sign fixed from train)
- SURVIVOR BTC mom_4 h=4 → 2025 HOLDOUT: IC -0.0101, perm p 0.567, net -4.91 bps (sign fixed from train)
- SURVIVOR BTC revert_4 h=4 → 2024 validation: IC +0.0457, perm p 0.006, net -0.82 bps (sign fixed from train)
- SURVIVOR BTC revert_4 h=4 → 2025 HOLDOUT: IC +0.0101, perm p 0.567, net -4.91 bps (sign fixed from train)
