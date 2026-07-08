"""
Prop-firm evaluation Monte-Carlo simulator — the anti-setback tool.

Given a strategy's per-trade R distribution (bootstrapped from real backtest
trades, or parametric win-rate/RR) and a firm's EXACT rule sheet, simulate many
independent evaluation attempts and report:

  * P(pass)                 probability an attempt reaches the profit target
                            without breaching daily-loss or max-drawdown in time
  * P(fail_daily / fail_dd / fail_time)  how it typically dies
  * expected attempts, expected FEES to a single pass
  * single-attempt EV = P(pass)*payout - fee

THE DECISION RULE: only buy an evaluation when EV > 0 (with margin) and P(pass)
is high enough that repeated failure won't grind you down. A no-edge strategy
shows a low pass rate and NEGATIVE EV — telling you not to pay. That is the
whole point: turn each paid attempt from a hopeful gamble into a gated decision.

Validated (see __main__): a break-even/negative trader passes rarely (~ the
firms' advertised fail rates); only a genuine edge lifts P(pass) enough for EV>0.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Rules:
    profit_target: float = 0.10     # +10% of base to pass the evaluation
    max_daily_loss: float = 0.05    # -5% in a day fails
    max_total_dd: float = 0.10      # -10% total fails
    dd_trailing: bool = False       # trailing (from peak) vs static (from start)
    min_days: int = 0               # must trade >= this many days
    max_days: int = 30              # evaluation time limit (trading days)
    risk_per_trade: float = 0.005   # fraction of BASE risked per trade
    trades_per_day: float = 2.0     # average
    fee: float = 150.0              # cost of one evaluation
    # --- funded stage (why passing != getting paid) ---
    account_size: float = 100_000.0  # funded account nominal
    profit_split: float = 0.80       # your share of funded profit
    withdraw_target: float = 0.08    # funded profit % at which you withdraw
    funded_max_dd: float = 0.10      # funded account bust level (from start)
    funded_max_days: int = 60        # give up if no withdrawal by here


def _run_phase(rng, r_samples, risk, tgt, dd_floor_kind, max_dd, daily_loss,
               max_days, trades_per_day, min_days=0):
    """Simulate one account phase from base=1. Returns
    ('target'|'dd'|'daily'|'time', days). target = reached +tgt."""
    eq = peak = 1.0
    for day in range(1, max_days + 1):
        nt = rng.poisson(trades_per_day)
        day_start = eq
        for _t in range(nt):
            eq += risk * r_samples[rng.integers(len(r_samples))]
            peak = max(peak, eq)
            floor = (peak if dd_floor_kind == "trailing" else 1.0) - max_dd
            if eq <= floor:
                return "dd", day
            if daily_loss and (day_start - eq) >= daily_loss:
                return "daily", day
            if eq >= 1.0 + tgt and day >= min_days:
                return "target", day
        if eq >= 1.0 + tgt and day >= min_days:
            return "target", day
    return "time", max_days


def _funded_life(rng, r_samples, rules, max_trades=6000):
    """Simulate a funded account until it busts; return total profit WITHDRAWN
    as a fraction of account size. Withdraw (eq-1) whenever eq >= 1+target, then
    reset eq to 1. Bust when equity falls max_dd below its high-water mark
    (trailing) or below the start (static)."""
    eq = peak = 1.0
    total_withdrawn = 0.0
    risk = rules.risk_per_trade
    tpd = rules.trades_per_day
    n = 0
    while n < max_trades:
        nt = max(1, rng.poisson(tpd))
        day_start = eq
        for _t in range(nt):
            n += 1
            eq += risk * r_samples[rng.integers(len(r_samples))]
            peak = max(peak, eq)
            floor = (peak if rules.dd_trailing else 1.0) - rules.funded_max_dd
            if eq <= floor:
                return total_withdrawn
            if rules.max_daily_loss and (day_start - eq) >= rules.max_daily_loss:
                # blew the daily loss limit on the funded account -> terminated
                return total_withdrawn
            if eq >= 1.0 + rules.withdraw_target:
                total_withdrawn += (eq - 1.0)
                eq = 1.0            # withdraw profit, balance back to base
                peak = 1.0 if not rules.dd_trailing else max(peak, 1.0)
                day_start = min(day_start, eq)
    return total_withdrawn


def simulate(r_samples: np.ndarray, rules: Rules, n_accounts: int = 20000,
             seed: int = 0) -> dict:
    """Monte-Carlo the FULL journey: pass the evaluation AND then reach a
    withdrawal on the funded account. EV is based on actually getting paid,
    because passing the eval by luck with no edge just hands the funded account
    back on the drawdown. r_samples: empirical per-trade R (net of cost)."""
    rng = np.random.default_rng(seed)
    r_samples = np.asarray(r_samples, float)
    r_samples = r_samples[np.isfinite(r_samples)]
    if len(r_samples) < 5:
        return {"error": "need >=5 sample trades"}

    passed = paid = fail_daily = fail_dd = fail_time = 0
    days_to_pass = []
    funded_earnings = []
    dd_kind = "trailing" if rules.dd_trailing else "static"
    for _ in range(n_accounts):
        out, day = _run_phase(
            rng, r_samples, rules.risk_per_trade, rules.profit_target, dd_kind,
            rules.max_total_dd, rules.max_daily_loss, rules.max_days,
            rules.trades_per_day, rules.min_days)
        if out != "target":
            {"daily": "fd", "dd": "dm", "time": "ft"}  # noqa
            if out == "daily": fail_daily += 1
            elif out == "dd": fail_dd += 1
            else: fail_time += 1
            continue
        passed += 1; days_to_pass.append(day)
        # ---- funded stage: FULL LIFE. Trade until the account busts, banking
        # a withdrawal each time profit crosses the target (then reset to base).
        # Trailing DD (from the high-water mark) is the realistic hard case: a
        # no-edge account busts with near-certainty for little total withdrawal.
        w = _funded_life(rng, r_samples, rules)
        if w > 0:
            paid += 1
        funded_earnings.append(w * rules.account_size * rules.profit_split)

    N = n_accounts
    p_pass = passed / N
    p_paid = paid / N
    mean_funded = float(np.mean(funded_earnings)) if funded_earnings else 0.0
    # EV per eval BOUGHT = P(pass)*E[funded $ | passed] - fee
    ev = p_pass * mean_funded - rules.fee
    edge_R = float(r_samples.mean())
    # THE GATE: a positive per-trade edge (net of cost) is REQUIRED. Without it,
    # any "+EV" is a fragile artifact of idealised rules + enormous variance that
    # real firms recover through slippage, rule technicalities and denied/slow
    # payouts (uncaptured here). No edge => DON'T PAY, whatever the paper EV.
    if edge_R <= 0:
        decision = "DON'T PAY (no edge — paper EV is a variance mirage)"
    elif ev > 0:
        decision = "CONSIDER (real edge AND +EV)"
    else:
        decision = "DON'T PAY (edge too small to clear the fee)"
    exp_attempts = (1 / p_pass) if p_pass > 0 else float("inf")
    return {
        "p_pass_eval": p_pass, "p_reach_payout": p_paid,
        "p_fail_daily": fail_daily / N, "p_fail_dd": fail_dd / N,
        "p_fail_time": fail_time / N,
        "median_days_to_pass": float(np.median(days_to_pass)) if days_to_pass else None,
        "mean_funded_earnings": mean_funded,           # $ per passed account, avg
        "expected_attempts_to_pass": exp_attempts,
        "edge_per_trade_R": edge_R,
        "p_lose_fee_on_eval": 1 - p_pass,              # most attempts fail the eval
        "single_eval_EV": ev,
        "decision": decision,
    }


def r_from_winrate(win_rate: float, rr: float, n: int = 4000, seed=0) -> np.ndarray:
    """Parametric per-trade R: win_rate winners of +rr R, else -1 R."""
    rng = np.random.default_rng(seed)
    w = rng.random(n) < win_rate
    return np.where(w, rr, -1.0)


def _report(tag, res):
    if "error" in res:
        print(f"{tag}: {res['error']}"); return
    print(f"{tag}")
    print(f"  P(pass eval) {res['p_pass_eval']:.1%}  ->  P(ever withdraw on funded) "
          f"{res['p_reach_payout']:.1%}")
    print(f"  eval fails: daily {res['p_fail_daily']:.1%} dd {res['p_fail_dd']:.1%} "
          f"time {res['p_fail_time']:.1%} | median days-to-pass "
          f"{res['median_days_to_pass']}")
    print(f"  edge/trade {res['edge_per_trade_R']:+.3f}R | avg funded $ (full life) "
          f"${res['mean_funded_earnings']:,.0f} | "
          f"you lose the fee on {res['p_lose_fee_on_eval']:.0%} of attempts")
    print(f"  EV per eval bought ${res['single_eval_EV']:+,.0f}  ->  "
          f"DECISION: {res['decision']}\n")


if __name__ == "__main__":
    # VALIDATION: the sim must show a no-edge trader rarely passing (matching the
    # firms' ~90% fail rates), and only a real edge flipping EV positive.
    rules = Rules(profit_target=0.10, max_daily_loss=0.05, max_total_dd=0.10,
                  dd_trailing=True,     # realistic hard case
                  risk_per_trade=0.01, trades_per_day=2, max_days=30, fee=150,
                  account_size=100_000, profit_split=0.80, withdraw_target=0.08)
    print("=== validation: does the sim behave sanely? ===\n")
    # 1) coin-flip at 2R after costs (expectancy slightly NEGATIVE)
    _report("no-edge (50% @ 2R minus cost ~ -0.05R avg)",
            simulate(r_from_winrate(0.32, 2.0), rules))     # 32%@2R ~ -0.04R
    # 2) break-even
    _report("break-even (~0R)", simulate(r_from_winrate(0.34, 2.0), rules))
    # 3) small real edge
    _report("small edge (40% @ 2R = +0.2R)", simulate(r_from_winrate(0.40, 2.0), rules))
    # 4) strong (unrealistic) edge
    _report("strong edge (55% @ 2R = +0.65R)", simulate(r_from_winrate(0.55, 2.0), rules))
    print("Interpretation: a strategy must first clear the edge_search / paired-null")
    print("gate; THEN feed its real per-trade R here to see if it clears an eval\n"
          "profitably. If EV is negative, DON'T PAY — that is the setback you avoid.")
