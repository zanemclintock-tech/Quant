"""
Smart-money entry detector (Stage 1) — causal, 30-minute structure.

The setup (short example; long is the mirror):
  1. A 30-min swing HIGH forms. It is a PIVOT, confirmed only after
     PIVOT_K bars have closed beyond it — so we never "know" about a
     pivot until PIVOT_K bars later. This is the key no-lookahead guard.
  2. Price later TAKES that high: a 30-min bar wicks above the level but
     CLOSES back below it (liquidity sweep / failed breakout).
  3. That leaves a bearish ORDER BLOCK — the last up-candle of the leg
     into the high. Its zone is a supply area.
  4. Price RETRACES up into the order block, in PREMIUM (above the
     EQUILIBRIUM = 50% of the swing low -> swing high leg). Entry short.
  5. Stop above the sweep extreme; target a multiple of risk.

Everything is computed from bars at or before the decision bar. The
test suite proves that truncating the future never changes a setup that
was already emitted.

This module only DETECTS setups and attaches causal features. It does
not decide whether to take them — that is the model's job (Stage 3),
and the features here are what it will learn from. "Std-dev pullback"
is provided in BOTH senses (returns-sigma depth and leg-fraction) so the
data can decide which matters, per the design discussion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import os

import numpy as np
import pandas as pd

# SMC_RELAXED=1 drops the hard premium/discount filter so the model sees
# the full population of sweep setups and decides for itself.
RELAXED = os.environ.get("SMC_RELAXED", "0") not in ("0", "", "false", "no")


def _base_tf() -> str:
    """Structural timeframe the detector runs on (pandas offset alias).
    BASE_TF lets us test 15min/30min/60min/... without touching code."""
    return os.environ.get("BASE_TF", "30min")


def _tf_minutes(tf: str | None = None) -> int:
    tf = (tf or _base_tf()).strip().lower()
    if tf.endswith("min"):
        return int(tf[:-3])
    if tf.endswith("h"):
        return int(float(tf[:-1]) * 60)
    if tf.endswith("m"):
        return int(tf[:-1])
    return int(tf)


def _continuous() -> bool:
    """24/7 markets (crypto) have no calendar-day session boundary, so a
    setup must not expire at UTC midnight and a position must not be
    force-flattened at a fake cash close."""
    return os.environ.get("CRYPTO", "0") not in ("0", "", "false", "no")

# ── fixed structural parameters (documented, not fitted) ────────────────
PIVOT_K = 2            # bars each side to confirm a 30-min pivot
OB_LOOKBACK = 6        # bars back from the sweep to find the order block
RETRACE_WINDOW = 16    # 30-min bars (~8h) to trigger the entry after a sweep
STOP_BUFFER_ATR = 0.10 # stop placed this far beyond the sweep extreme
ATR_PERIOD = 14        # ATR on 30-min bars
RET_STD_WINDOW = 20    # window for returns-sigma (the statistical pullback)


@dataclass
class Setup:
    direction: int                 # -1 short (swept high), +1 long (swept low)
    arm_time: pd.Timestamp         # sweep bar close — when the setup is known
    entry_time: pd.Timestamp       # bar whose close triggers entry
    entry_price: float
    stop: float
    swept_level: float
    sweep_extreme: float
    ob_low: float
    ob_high: float
    equilibrium: float
    leg_low: float
    leg_high: float
    features: dict = field(default_factory=dict)


def to_m30(df: pd.DataFrame, freq: str | None = None) -> pd.DataFrame:
    """1-min mid (+ optional volume) -> structural OHLC bars. The freq
    defaults to BASE_TF (30min) but any pandas offset works (15min, 1h)."""
    freq = freq or _base_tf()
    o = df["mid_c"].resample(freq).first()
    m = pd.DataFrame({
        "o": o,
        "h": df["mid_h"].resample(freq).max(),
        "l": df["mid_l"].resample(freq).min(),
        "c": df["mid_c"].resample(freq).last(),
    })
    if "volume" in df.columns:
        m["v"] = df["volume"].resample(freq).sum()
        if "taker_buy" in df.columns:            # order-flow imbalance
            m["tbuy"] = df["taker_buy"].resample(freq).sum()
    if "sp_c" in df.columns:                 # correlated secondary (S&P)
        m["sp_h"] = df["sp_h"].resample(freq).max()
        m["sp_l"] = df["sp_l"].resample(freq).min()
        m["sp_c"] = df["sp_c"].resample(freq).last()
    return m.dropna(subset=["o", "h", "l", "c"])


def _atr(m: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_c = m["c"].shift(1)
    tr = pd.concat([m["h"] - m["l"], (m["h"] - prev_c).abs(),
                    (m["l"] - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False,
                  min_periods=period).mean()


def _confirmed_pivots(m: pd.DataFrame, k: int):
    """Return two arrays: for each bar index, the price of the most recent
    swing high / low that is ALREADY CONFIRMED as of that bar (NaN until
    one exists). A pivot at i is confirmed at i+k."""
    h, l = m["h"].values, m["l"].values
    n = len(m)
    sh_at = np.full(n, np.nan)   # swing-high price available at this index
    sl_at = np.full(n, np.nan)
    sh_idx = np.full(n, -1)      # bar index of that swing high
    sl_idx = np.full(n, -1)
    last_h = np.nan; last_h_i = -1
    last_l = np.nan; last_l_i = -1
    for c in range(n):           # c = bar at which we may learn a pivot
        p = c - k                # candidate pivot bar
        if p - k >= 0:
            win_h = h[p - k:p + k + 1]
            win_l = l[p - k:p + k + 1]
            if h[p] == win_h.max() and (h[p] > np.delete(win_h, k)).all():
                last_h, last_h_i = h[p], p
            if l[p] == win_l.min() and (l[p] < np.delete(win_l, k)).all():
                last_l, last_l_i = l[p], p
        sh_at[c], sh_idx[c] = last_h, last_h_i
        sl_at[c], sl_idx[c] = last_l, last_l_i
    return sh_at, sl_at, sh_idx, sl_idx


def detect_setups(df: pd.DataFrame) -> list[Setup]:
    m = to_m30(df)
    if len(m) < ATR_PERIOD + PIVOT_K + 5:
        return []
    atr = _atr(m).values
    ret_std = m["c"].pct_change().rolling(RET_STD_WINDOW).std().values
    sh_at, sl_at, sh_idx, sl_idx = _confirmed_pivots(m, PIVOT_K)

    o, h, l, c = (m[x].values for x in "ohlc")
    v = m["v"].values if "v" in m.columns else None
    vol_z = None
    if v is not None:
        vs = pd.Series(v)
        vol_z = ((vs - vs.rolling(50).mean())
                 / vs.rolling(50).std()).values
    # order-flow imbalance per 30m bar: +1 all aggressive buying, -1 selling
    ofi = None
    if "tbuy" in m.columns:
        with np.errstate(invalid="ignore", divide="ignore"):
            ofi = (2.0 * m["tbuy"].values - v) / v
    # cross-asset (S&P) arrays for SMT divergence, if present
    sp_h = sp_l = sp_atr = nas_sp_corr = None
    if "sp_c" in m.columns:
        sp_h, sp_l, sp_c = m["sp_h"].values, m["sp_l"].values, m["sp_c"].values
        sp_atr = _atr(pd.DataFrame({"h": sp_h, "l": sp_l, "c": sp_c})).values
        nas_sp_corr = pd.Series(c).pct_change().rolling(20).corr(
            pd.Series(sp_c).pct_change()).values
    idx = m.index
    dates = np.array([t.date() for t in idx])
    continuous = _continuous()   # 24/7: no UTC-midnight session boundary

    setups: list[Setup] = []
    armed: list[dict] = []       # setups waiting for a retracement trigger

    for j in range(PIVOT_K + 1, len(m)):
        # ---- 1. try to ARM a new setup on this bar's close (causal) ----
        # short: sweep of the confirmed swing high known as of bar j-1
        shp, shi = sh_at[j - 1], sh_idx[j - 1]
        if np.isfinite(shp) and shi >= 0 and h[j] > shp and c[j] <= shp:
            slp = sl_at[j - 1]                       # leg starts at swing low
            if np.isfinite(slp) and shp > slp:
                ob_lo, ob_hi = _order_block(o, c, l, h, j, +1)  # bullish OB
                armed.append({
                    "dir": -1, "j": j, "level": shp, "ext": h[j],
                    "ob_low": ob_lo, "ob_high": ob_hi,
                    "leg_low": slp, "leg_high": shp, "pivot": shi,
                    "eq": (slp + shp) / 2.0, "expires": j + RETRACE_WINDOW,
                })
        # long: sweep of the confirmed swing low
        slp, sli = sl_at[j - 1], sl_idx[j - 1]
        if np.isfinite(slp) and sli >= 0 and l[j] < slp and c[j] >= slp:
            shp2 = sh_at[j - 1]
            if np.isfinite(shp2) and shp2 > slp:
                ob_lo, ob_hi = _order_block(o, c, l, h, j, -1)  # bearish OB
                armed.append({
                    "dir": 1, "j": j, "level": slp, "ext": l[j],
                    "ob_low": ob_lo, "ob_high": ob_hi,
                    "leg_low": slp, "leg_high": shp2, "pivot": sli,
                    "eq": (slp + shp2) / 2.0, "expires": j + RETRACE_WINDOW,
                })

        # ---- 2. check armed setups for a retracement trigger on bar j ----
        still: list[dict] = []
        for a in armed:
            same_session = continuous or dates[j] == dates[a["j"]]
            if j <= a["j"] or j > a["expires"] or not same_session:
                if j <= a["expires"] and same_session:
                    still.append(a)              # keep waiting (same session)
                continue
            # invalidation: price ran past the swept extreme -> setup void
            if (a["dir"] == -1 and h[j] > a["ext"]) or \
               (a["dir"] == 1 and l[j] < a["ext"]):
                continue                         # drop, do not re-arm
            # entry at the order-block edge facing the retracement, in
            # premium (short) / discount (long), and inside the swept range.
            # SMC_RELAXED=1 drops the premium/discount HARD FILTER, so the
            # model sees every retracement setup and decides via the
            # eq_distance feature instead of us gating on it.
            triggered = False
            if a["dir"] == -1:                   # short: sell into supply
                entry = a["ob_low"]
                if h[j] >= entry and entry < a["ext"] \
                        and (RELAXED or entry >= a["eq"]):
                    triggered = True
            else:                                # long: buy into demand
                entry = a["ob_high"]
                if l[j] <= entry and entry > a["ext"] \
                        and (RELAXED or entry <= a["eq"]):
                    triggered = True
            if not triggered:
                still.append(a); continue

            d = a["dir"]
            stop = (a["ext"] + STOP_BUFFER_ATR * atr[j] if d == -1
                    else a["ext"] - STOP_BUFFER_ATR * atr[j])
            risk = abs(entry - stop)
            if risk <= 0 or not np.isfinite(atr[j]):
                continue
            leg = a["leg_high"] - a["leg_low"]
            impulse = (a["ext"] - a["leg_low"] if d == -1
                       else a["leg_high"] - a["ext"])
            feats = {
                "dir": d,
                "sweep_depth_atr": abs(a["ext"] - a["level"]) / atr[j],
                "pullback_sigma": (abs(entry - a["ext"]) / c[j])
                / ret_std[j] if np.isfinite(ret_std[j]) and ret_std[j] > 0
                else np.nan,                      # statistical std-dev pullback
                "retrace_frac_of_leg": abs(entry - a["ext"]) / leg
                if leg > 0 else np.nan,           # ICT leg-fraction pullback
                "eq_distance_atr": (entry - a["eq"]) * d / atr[j],
                "ob_size_atr": (a["ob_high"] - a["ob_low"]) / atr[j],
                "impulse_atr": impulse / atr[j],
                "risk_atr": risk / atr[j],
                "bars_to_trigger": j - a["j"],
                "minute_of_day": idx[j].hour * 60 + idx[j].minute,
                "atr_regime": atr[j] / np.nanmean(atr[max(0, j - 50):j + 1]),
            }
            if vol_z is not None:
                feats["vol_z_sweep"] = vol_z[a["j"]]
                feats["vol_z_entry"] = vol_z[j]
            if ofi is not None:
                # strictly-pre-trigger bars only: the trigger bar j and
                # the fill sit inside the forward label window, so using
                # ofi[j] would leak future order flow (it correlates with
                # that bar's own move). Use the last completed bar, j-1.
                eb = j - 1
                if eb >= 0:
                    feats["ofi_sweep"] = ofi[min(a["j"], eb)]
                    feats["ofi_entry"] = ofi[eb]
                    seg = ofi[max(0, a["pivot"]):eb + 1]
                    feats["ofi_leg"] = (float(np.nanmean(seg))
                                        if len(seg) else np.nan)
            if sp_h is not None:
                jj, pv = a["j"], a["pivot"]      # sweep bar, swing pivot bar
                sa = sp_atr[jj]
                if np.isfinite(sa) and sa > 0 and pv >= 0:
                    # SMT: did the S&P confirm the NAS sweep or diverge?
                    # short -> S&P failing to make a new high favours us;
                    # long  -> S&P failing to make a new low favours us.
                    if d == -1:
                        favor = (sp_h[pv] - sp_h[jj]) / sa
                    else:
                        favor = (sp_l[jj] - sp_l[pv]) / sa
                    feats["smt_div_favor"] = favor      # >0 = divergence for us
                    feats["smt_diverged"] = float(favor > 0)
                    feats["nas_sp_corr"] = (nas_sp_corr[jj]
                                            if np.isfinite(nas_sp_corr[jj])
                                            else np.nan)
            setups.append(Setup(
                direction=d, arm_time=idx[a["j"]], entry_time=idx[j],
                entry_price=float(entry), stop=float(stop),
                swept_level=float(a["level"]), sweep_extreme=float(a["ext"]),
                ob_low=float(a["ob_low"]), ob_high=float(a["ob_high"]),
                equilibrium=float(a["eq"]), leg_low=float(a["leg_low"]),
                leg_high=float(a["leg_high"]), features=feats))
        armed = still
    return setups


def _order_block(o, c, l, h, j, color):
    """Zone of the last candle of the given color within OB_LOOKBACK bars
    ending at the sweep bar j. color=+1 wants a bullish candle (bearish OB
    for a short), color=-1 wants a bearish candle (bullish OB for a long).
    Returns (low, high) of that candle's range; falls back to bar j."""
    for b in range(j, max(j - OB_LOOKBACK, 0) - 1, -1):
        bull = c[b] > o[b]
        if (color == 1 and bull) or (color == -1 and not bull):
            return float(l[b]), float(h[b])
    return float(l[j]), float(h[j])
