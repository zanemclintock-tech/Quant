"""
Adaptive, 1:2-leverage-capped position sizing (funded-account compliant).

Risk per trade scales with model confidence (0.3% at low conviction -> 1.0%
at high), but position size is HARD-CAPPED so neither a single position nor
total open exposure ever exceeds MAX_LEV x the account. Sized off a fixed
base (you withdraw profits), not compounding.

What is actually verifiable here is the RISK INVARIANT: neither a single
position nor total open exposure ever exceeds MAX_LEV x equity (the caps
below). That property is real and worth keeping.

HONESTY NOTE (see CRYPTO_AUDIT.md): earlier docstrings quoted "+16.8%/mo at
5.5% max DD" from a walk-forward. Do NOT trust that number. It came from an
in-sample pipeline (train_live_models.py fit and scored the SAME rows) on a
strategy that a 5-year noise-null test (model_train.py) shows does NOT beat a
random walk. Confidence-scaled sizing on a no-edge signal amplifies variance,
not return -- so any monthly/drawdown figure attached to it is not evidence of
edge. Sizing controls risk; it cannot manufacture an edge that isn't there.
"""
from __future__ import annotations

INIT = 100_000.0
MAX_LEV = 2.0
DAY_BUDGET = 0.018    # cap the worst-case NET daily loss (today's realised
                      # loss + every open position's risk PLUS its exit cost,
                      # if they all stop at once) at 1.8% of the account, so
                      # the day can't breach the 2% limit. Budgeting gross
                      # risk alone wasn't enough -- a 2x alt position's spread
                      # cost is ~0.3% of the account, and several stopping
                      # together pushed days to 2.7%. This holds it under 2%.
# selected-trade prob 10th/90th pct -- the endpoints of the confidence->risk
# ramp. These MUST match the deployed model's probability scale; live_stream
# overrides them at startup from the model bundle's saved size_plo/size_phi so
# they never go stale on a retrain. The defaults below are the live-label
# pooled model's calibration (a safe fallback if the bundle lacks them).
P_LO, P_HI = 0.41, 0.63
R_MIN, R_MAX = 0.003, 0.010      # risk fraction at low / high confidence


def conf_risk(prob: float) -> float:
    """Confidence -> risk fraction, clamped to [R_MIN, R_MAX]."""
    f = (prob - P_LO) / (P_HI - P_LO)
    return max(R_MIN, min(R_MAX, R_MIN + (R_MAX - R_MIN) * f))


def size_notional(prob: float, stop_frac: float, open_notional: float,
                  equity: float = INIT, max_lev: float = MAX_LEV,
                  budget_notional: float | None = None) -> float:
    """Position notional ($): adaptive by confidence, capped so (a) this
    trade and total open exposure each stay <= max_lev x equity, and (b) it
    fits the remaining daily risk budget (budget_notional). 0 = skip."""
    if stop_frac <= 0:
        return 0.0
    caps = [conf_risk(prob) * equity / stop_frac, max_lev * equity,
            max_lev * equity - open_notional]
    if budget_notional is not None:
        caps.append(max(0.0, budget_notional))
    return max(0.0, min(caps))
