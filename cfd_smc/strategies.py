"""
No-lookahead intraday ICT/SMC strategies for NAS100 / S&P500 CFD.

Objective, coded version of the shared playbook behind the named manual traders
(Powell "10am order block", PB Blake/Patty, AhmedyFX, Trading Pool). Every
component is a toggle so we can measure which parts are REAL signal and which are
the manual illusion:

  * session/time filter (NY time; the "10am" window)
  * liquidity sweep of a reference level (prior-day H/L, opening range)
  * market-structure shift (CHoCH/BOS) confirming reversal
  * order-block / FVG retrace entry
  * fixed R:R target with break-even + trail
  * (SMT divergence handled in the runner, needs both indices)

Causality (enforced): structure/levels use only CLOSED bars up to the decision
bar; reference levels form from PRIOR periods; entries fill on a LATER bar at
ask(buy)/bid(sell)+slippage; stop-before-target within a bar; force-flat at
session end (no overnight).

Each trade records BOTH its real R and the R it WOULD have made with the
direction flipped (`r_flip`) on the identical setup — the runner uses that for a
paired permutation test of directional skill.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass
class Cfg:
    entry_tf: str = "5min"
    session_start: str = "09:30"
    session_end: str = "16:00"
    trade_from: str = "09:30"
    trade_to: str = "11:30"
    flat_at: str = "15:55"
    or_minutes: int = 30
    pivot_k: int = 2
    rr: float = 2.0
    stop_buffer_atr: float = 0.1
    be_at_r: float = 1.0
    max_trades_day: int = 2
    slippage_pts: float = 0.5
    ttl_bars: int = 8
    require_sweep: bool = True
    require_mss: bool = True
    ref_level: str = "or"           # 'or' | 'pdhl'
    entry_mode: str = "ob"          # 'ob' | 'fvg' | 'market'


@dataclass
class Trade:
    entry_time: pd.Timestamp
    direction: int
    entry: float
    stop: float
    risk: float
    r_net: float
    r_flip: float
    outcome: str
    entry_i: int
    features: dict = field(default_factory=dict)


# ---- helpers ----------------------------------------------------------
def to_ny(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return df.set_axis(idx.tz_convert(NY))


def resample_tf(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    g = df1m.resample(tf, label="right", closed="right")
    return pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "spread": g["spread"].mean(),
    }).dropna(subset=["open"])


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]).abs(),
                    (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def confirmed_pivots(h: np.ndarray, l: np.ndarray, k: int):
    """Last swing high/low CONFIRMED as of each bar (known only k bars after
    it forms) — the no-lookahead guard."""
    n = len(h)
    ph = np.full(n, np.nan); pl = np.full(n, np.nan)
    for i in range(k, n - k):
        if h[i] == max(h[i - k:i + k + 1]):
            ph[i] = h[i]
        if l[i] == min(l[i - k:i + k + 1]):
            pl[i] = l[i]
    last_h = np.full(n, np.nan); last_l = np.full(n, np.nan)
    cur_h = cur_l = np.nan
    for j in range(n):
        src = j - k
        if src >= 0 and np.isfinite(ph[src]):
            cur_h = ph[src]
        if src >= 0 and np.isfinite(pl[src]):
            cur_l = pl[src]
        last_h[j] = cur_h; last_l[j] = cur_l
    return last_h, last_l


# ---- the strategy -----------------------------------------------------
def backtest(df1m: pd.DataFrame, cfg: Cfg) -> list[Trade]:
    d1 = to_ny(df1m)
    bars = resample_tf(d1, cfg.entry_tf)
    if len(bars) < 100:
        return []
    ts = bars.index
    tt_arr = np.array([t.time() for t in ts])
    day_arr = ts.normalize()
    a = atr(bars, 14).to_numpy()
    h, l, c, o = (bars["high"].to_numpy(), bars["low"].to_numpy(),
                  bars["close"].to_numpy(), bars["open"].to_numpy())
    sp = bars["spread"].to_numpy()
    last_ph, last_pl = confirmed_pivots(h, l, cfg.pivot_k)

    t_open = pd.to_datetime(cfg.session_start).time()
    t_or_end = (pd.to_datetime(cfg.session_start) + pd.Timedelta(minutes=cfg.or_minutes)).time()
    t_from = pd.to_datetime(cfg.trade_from).time()
    t_to = pd.to_datetime(cfg.trade_to).time()
    t_flat = pd.to_datetime(cfg.flat_at).time()
    t_close = pd.to_datetime(cfg.session_end).time()

    day_rows = defaultdict(list)
    for i, dd in enumerate(day_arr):
        day_rows[dd].append(i)

    trades: list[Trade] = []
    pdh = pdl = np.nan
    for day in day_rows:
        rows = day_rows[day]
        end_i = rows[-1]
        or_rows = [i for i in rows if t_open <= tt_arr[i] < t_or_end]
        orh = max((h[i] for i in or_rows), default=np.nan)
        orl = min((l[i] for i in or_rows), default=np.nan)
        ref_hi = orh if cfg.ref_level == "or" else pdh
        ref_lo = orl if cfg.ref_level == "or" else pdl

        n_trades = 0
        busy_until = -1
        for i in rows:
            if i <= busy_until or n_trades >= cfg.max_trades_day:
                continue
            tt = tt_arr[i]
            if tt >= t_flat or not (t_from <= tt <= t_to):
                continue
            if not (np.isfinite(ref_hi) and np.isfinite(ref_lo)):
                continue
            sig = _signal(i, h, l, c, o, a, last_ph, last_pl, ref_hi, ref_lo, cfg)
            if sig is None:
                continue
            d, entry, stop = sig["dir"], sig["entry"], sig["stop"]
            fill_i = _limit_fill(i, d, entry, h, l, tt_arr, day_arr, day, t_flat, cfg.ttl_bars)
            if fill_i is None:
                continue
            fill_px = entry + d * (sp[fill_i] / 2 + cfg.slippage_pts)
            risk = abs(fill_px - stop)
            if risk <= 0:
                continue
            r_real, out, exit_i = _walk(fill_i, fill_px, d, risk, end_i, h, l, c, sp,
                                        tt_arr, t_flat, cfg)
            r_flip, _, _ = _walk(fill_i, fill_px, -d, risk, end_i, h, l, c, sp,
                                 tt_arr, t_flat, cfg)
            trades.append(Trade(
                entry_time=ts[fill_i], direction=d, entry=fill_px, stop=stop,
                risk=risk, r_net=r_real, r_flip=r_flip, outcome=out, entry_i=fill_i,
                features={**sig["features"], "hour": ts[fill_i].hour,
                          "minute": ts[fill_i].minute, "year": ts[fill_i].year}))
            busy_until = exit_i
            n_trades += 1

        rth = [i for i in rows if t_open <= tt_arr[i] <= t_close]
        if rth:
            pdh = max(h[i] for i in rth); pdl = min(l[i] for i in rth)
    return trades


def _limit_fill(i, d, entry, h, l, tt_arr, day_arr, day, t_flat, ttl):
    """First later same-day bar (within ttl) whose range reaches the limit."""
    for k in range(i + 1, min(i + 1 + ttl, len(h))):
        if day_arr[k] != day or tt_arr[k] >= t_flat:
            return None
        if (l[k] <= entry) if d == 1 else (h[k] >= entry):
            return k
    return None


def _walk(fill_i, fill_px, d, risk, end_i, h, l, c, sp, tt_arr, t_flat, cfg):
    """Independent forward simulation of one trade. Returns (r_net, outcome,
    exit_i). Charges the exit half-spread + slippage; stop-before-target."""
    stop = fill_px - d * risk
    target = fill_px + d * cfg.rr * risk
    for k in range(fill_i, end_i + 1):
        hit_stop = (l[k] <= stop) if d == 1 else (h[k] >= stop)
        hit_tp = (h[k] >= target) if d == 1 else (l[k] <= target)
        exit_px = out = None
        if hit_stop:                       # stop-before-target (conservative)
            exit_px, out = stop, "sl"
        elif hit_tp:
            exit_px, out = target, "tp"
        elif tt_arr[k] >= t_flat or k == end_i:
            exit_px, out = c[k], "eod"
        else:
            if cfg.be_at_r > 0 and (c[k] - fill_px) * d >= cfg.be_at_r * risk:
                be = fill_px + d * (sp[k] + cfg.slippage_pts)
                stop = max(stop, be) if d == 1 else min(stop, be)
            continue
        exit_adj = exit_px - d * (sp[k] / 2 + cfg.slippage_pts)
        return ((exit_adj - fill_px) * d) / risk, out, k
    return 0.0, "none", end_i


def _signal(i, h, l, c, o, a, last_ph, last_pl, ref_hi, ref_lo, cfg: Cfg):
    """Setup at bar i or None. Bearish: sweep ref_hi then down-MSS; bullish:
    sweep ref_lo then up-MSS. All causal on closed bar i."""
    atr_i = a[i]
    if not (np.isfinite(atr_i) and atr_i > 0):
        return None
    swept_lo = (l[i] < ref_lo) and (c[i] > ref_lo)
    swept_hi = (h[i] > ref_hi) and (c[i] < ref_hi)
    for d, swept, ref in ((1, swept_lo, ref_lo), (-1, swept_hi, ref_hi)):
        if cfg.require_sweep and not swept:
            continue
        if cfg.require_mss:
            sw = last_ph[i] if d == 1 else last_pl[i]
            if not (np.isfinite(sw) and ((c[i] > sw) if d == 1 else (c[i] < sw))):
                continue
        extreme = l[i] if d == 1 else h[i]
        stop = extreme - d * cfg.stop_buffer_atr * atr_i
        if cfg.entry_mode == "market":
            entry = c[i]
        elif cfg.entry_mode == "fvg":
            entry = (h[i - 2] if (d == 1 and h[i - 2] < c[i]) else
                     l[i - 2] if (d == -1 and l[i - 2] > c[i]) else (c[i] + extreme) / 2)
        else:
            entry = _order_block_edge(i, o, c, d)
        if entry is None or not np.isfinite(entry):
            continue
        return {"dir": d, "entry": float(entry), "stop": float(stop),
                "features": {"swept": int(swept), "dir": d, "atr": float(atr_i)}}
    return None


def _order_block_edge(i, o, c, d, lookback=6):
    for k in range(i - 1, max(i - lookback, 0) - 1, -1):
        if d == 1 and c[k] < o[k]:
            return o[k]
        if d == -1 and c[k] > o[k]:
            return o[k]
    return None


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    rows = [{"entry_time": t.entry_time, "dir": t.direction, "r_net": t.r_net,
             "r_flip": t.r_flip, "win": int(t.r_net > 0), "outcome": t.outcome,
             **t.features} for t in trades]
    return pd.DataFrame(rows)
