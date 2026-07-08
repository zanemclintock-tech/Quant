"""
Honest backtest + gate for the intraday SMC strategies on NAS100 / S&P500 CFD.

For each strategy variant (and ablations that turn components off), reports the
TRUE net-of-cost result split into a nested hold-out (train <=2023 / val 2024 /
test 2025), and applies the DIRECTIONAL-SKILL gate:

  paired permutation test on (r_net, r_flip) per trade. r_flip is the R the
  SAME setup would have made with the long/short call reversed (timing, risk,
  R:R, session, cost all identical). Under H0 "the SMC direction has no skill",
  real and flipped are exchangeable. p = P(permuted mean >= real mean).

A variant is only believed if, on the UNTOUCHED 2025 test, it is net-positive
AND beats the paired null at BH-corrected p<0.05. Everything else is discarded,
not tuned. This is deliberately hostile to wishful thinking.

    python cfd_smc/run_cfd_backtest.py --name SP500
    python cfd_smc/run_cfd_backtest.py --file <parquet>   # ad-hoc single file
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import strategies as S

CACHE = Path(__file__).parent / "data_cache"
RISK_FRAC = 0.005          # 0.5% risk/trade, for the equity/DD read-out


def load_index(name: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(CACHE / f"{name}_1min_*.parquet")))
    if not fs:
        raise FileNotFoundError(f"no cached {name}_1min_*.parquet in {CACHE}")
    df = pd.concat([pd.read_parquet(f) for f in fs]).sort_index()
    return df[~df.index.duplicated(keep="first")]


def variants() -> dict[str, S.Cfg]:
    """The named playbook + ablations to see which components carry weight."""
    return {
        "10am_OB_sweep+mss":   S.Cfg(trade_from="10:00", trade_to="11:00", ref_level="or"),
        "morning_OB_sweep+mss": S.Cfg(trade_from="09:30", trade_to="11:30", ref_level="or"),
        "pdhl_sweep+mss":      S.Cfg(ref_level="pdhl", trade_from="09:30", trade_to="11:30"),
        "sweep_only(no mss)":  S.Cfg(require_mss=False, trade_from="09:30", trade_to="11:30"),
        "mss_only(no sweep)":  S.Cfg(require_sweep=False, trade_from="09:30", trade_to="11:30"),
        "fvg_entry":           S.Cfg(entry_mode="fvg", trade_from="09:30", trade_to="11:30"),
        "rr3":                 S.Cfg(rr=3.0, trade_from="09:30", trade_to="11:30"),
        "no_filters(baseline)": S.Cfg(require_sweep=False, require_mss=False,
                                      entry_mode="market", trade_from="09:30", trade_to="11:30"),
    }


def paired_perm_p(r_real: np.ndarray, r_flip: np.ndarray, n: int = 5000, seed=0):
    if len(r_real) < 10:
        return float(r_real.mean()) if len(r_real) else np.nan, np.nan
    rng = np.random.default_rng(seed)
    obs = r_real.mean()
    flip_mask = rng.random((n, len(r_real))) < 0.5
    perms = np.where(flip_mask, r_flip, r_real).mean(axis=1)
    return float(obs), float((perms >= obs).mean())


def bh(pvals):
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    pv = p[idx]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    qv = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv); out[order] = np.clip(qv, 0, 1)
    q[idx] = out
    return q


def split_stats(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0}
    r = df["r_net"].to_numpy()
    eq = 100_000 * np.cumprod(1 + RISK_FRAC * r)
    dd = (np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)
    gains = r[r > 0].sum(); losses = -r[r < 0].sum()
    return {"n": len(df), "win": df["win"].mean(), "meanR": r.mean(),
            "pf": gains / losses if losses > 0 else np.inf,
            "maxdd_pct": dd.max() * 100,
            "ret_pct": (eq[-1] / 100_000 - 1) * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="SP500")
    ap.add_argument("--file", default=None, help="ad-hoc single parquet (overrides --name)")
    args = ap.parse_args()

    if args.file:
        df = pd.read_parquet(args.file).sort_index()
    else:
        df = load_index(args.name)
    span = f"{df.index[0]} -> {df.index[-1]} ({len(df):,} 1m bars)"
    print(f"data: {span}\n")

    rows = []
    for vname, cfg in variants().items():
        trades = S.backtest(df, cfg)
        t = S.trades_to_df(trades)
        if len(t) == 0:
            print(f"{vname:24} no trades"); continue
        full = split_stats(t)
        te = t[t.year >= 2025]
        _, p_full = paired_perm_p(t.r_net.to_numpy(), t.r_flip.to_numpy())
        _, p_test = paired_perm_p(te.r_net.to_numpy(), te.r_flip.to_numpy()) if len(te) >= 10 else (np.nan, np.nan)
        rows.append({"variant": vname, "n": full["n"], "win%": full["win"] * 100,
                     "meanR": full["meanR"], "pf": full["pf"],
                     "maxDD%": full["maxdd_pct"], "p_full": p_full,
                     "n25": len(te), "p_test25": p_test,
                     "meanR25": te.r_net.mean() if len(te) else np.nan})

    res = pd.DataFrame(rows)
    if len(res):
        res["q_test25"] = bh(res["p_test25"].values)
        res["VERDICT"] = np.where(
            (res["meanR25"] > 0) & (res["q_test25"] < 0.05), "REAL?", "no edge")
        pd.set_option("display.width", 200, "display.max_columns", 20)
        print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nGate: a variant is believed ONLY if 2025 test is net-positive AND")
        print("beats the paired direction-flip null at BH q<0.05. Everything else")
        print("is the manual illusion / RR mechanics, not directional edge.")
    else:
        print("no variant produced trades (need more data).")


if __name__ == "__main__":
    main()
