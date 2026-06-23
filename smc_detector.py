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
    if "of_delta" in df.columns:             # trade-level order flow (ticks)
        m["of_vol"] = df["of_vol"].resample(freq).sum()
        m["of_delta"] = df["of_delta"].resample(freq).sum()
        m["of_cvd"] = df["of_cvd"].resample(freq).last()      # cumulative
        m["of_maxtrade"] = df["of_maxtrade"].resample(freq).max()
        m["of_buymax"] = df["of_buymax"].resample(freq).max()
        m["of_sellmax"] = df["of_sellmax"].resample(freq).max()
    if "l2_imb" in df.columns:               # L2 resting-liquidity imbalance
        m["l2_imb"] = df["l2_imb"].resample(freq).mean()
        m["l2_imb1"] = df["l2_imb1"].resample(freq).mean()
        m["l2_depth"] = df["l2_depth"].resample(freq).mean()
    return m.dropna(subset=["o", "h", "l", "c"])


def _atr(m: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_c = m["c"].shift(1)
    tr = pd.concat([m["h"] - m["l"], (m["h"] - prev_c).abs(),
                    (m["l"] - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False,
                  min_periods=period).mean()


def _value_area(centers, hist, frac: float = 0.70):
    """Point of control + 70% value area, expanding from the POC bin to
    whichever neighbour holds more volume (standard Market-Profile rule)."""
    total = float(hist.sum())
    if total <= 0:
        return None
    poc_i = int(hist.argmax())
    lo = hi = poc_i
    acc = float(hist[poc_i])
    n = len(hist)
    while acc < frac * total and (lo > 0 or hi < n - 1):
        left = hist[lo - 1] if lo > 0 else -1.0
        right = hist[hi + 1] if hi < n - 1 else -1.0
        if right >= left:
            hi += 1; acc += float(hist[hi])
        else:
            lo -= 1; acc += float(hist[lo])
    return float(centers[poc_i]), float(centers[hi]), float(centers[lo])


def _vol_profile(l, h, v, lo: int, hi: int, nbins: int = 40):
    """Volume-at-price over bars [lo, hi] (causal), each bar's volume
    spread across the price bins its range spans. Returns (poc, vah, val)."""
    sl, sh, sv = l[lo:hi + 1], h[lo:hi + 1], v[lo:hi + 1]
    ok = (np.isfinite(sl) & np.isfinite(sh) & np.isfinite(sv)
          & (sh > sl) & (sv > 0))
    if int(ok.sum()) < 3:
        return None
    sl, sh, sv = sl[ok], sh[ok], sv[ok]
    pmin, pmax = float(sl.min()), float(sh.max())
    if pmax <= pmin:
        return None
    bw = (pmax - pmin) / nbins
    edges = np.linspace(pmin, pmax, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist = np.zeros(nbins)
    for bl, bh, bv in zip(sl, sh, sv):
        i0 = min(nbins - 1, max(0, int((bl - pmin) / bw)))
        i1 = min(nbins - 1, max(0, int((bh - pmin) / bw)))
        hist[i0:i1 + 1] += bv / (i1 - i0 + 1)
    return _value_area(centers, hist)


def _kline_feats(ref, a, entry, d, leg, impulse, risk, atr, idx, price_z,
                 ret_std, c, vol_z, cum_v, cum_tpv, l, h, vv):
    """The kline feature set evaluated at bar `ref` (=trigger j normally, or
    the sweep/OB bar a['j'] for the pending/arm-time copy). With ref=j this is
    byte-identical to the original inline block (the causality test guards it)."""
    f = {
        "dir": d,
        "sweep_depth_atr": abs(a["ext"] - a["level"]) / atr[ref],
        "pullback_sigma": (abs(entry - a["ext"]) / c[ref]) / ret_std[ref]
        if np.isfinite(ret_std[ref]) and ret_std[ref] > 0 else np.nan,
        "retrace_frac_of_leg": abs(entry - a["ext"]) / leg if leg > 0 else np.nan,
        "eq_distance_atr": (entry - a["eq"]) * d / atr[ref],
        "ob_size_atr": (a["ob_high"] - a["ob_low"]) / atr[ref],
        "impulse_atr": impulse / atr[ref],
        "risk_atr": risk / atr[ref],
        "bars_to_trigger": ref - a["j"],
        "minute_of_day": idx[ref].hour * 60 + idx[ref].minute,
        "atr_regime": atr[ref] / np.nanmean(atr[max(0, ref - 50):ref + 1]),
        "price_z": price_z[ref - 1] if ref - 1 >= 0 else np.nan,
    }
    if vol_z is not None:
        f["vol_z_sweep"] = vol_z[a["j"]]
        f["vol_z_entry"] = vol_z[ref]
    if cum_v is not None:
        lo, eb = max(0, a["pivot"]), ref - 1
        if eb > lo and np.isfinite(atr[ref]) and atr[ref] > 0:
            sv = cum_v[eb] - (cum_v[lo - 1] if lo > 0 else 0.0)
            if sv > 0:
                vwap = (cum_tpv[eb] - (cum_tpv[lo - 1] if lo > 0 else 0.0)) / sv
                f["vwap_dist_atr"] = (entry - vwap) * d / atr[ref]
            prof = _vol_profile(l, h, vv, lo, eb)
            if prof is not None:
                poc, vah, val = prof
                f["poc_dist_atr"] = (entry - poc) * d / atr[ref]
                f["in_value_area"] = float(val <= entry <= vah)
                f["va_width_atr"] = (vah - val) / atr[ref]
                f["sweep_vs_poc_atr"] = (a["level"] - poc) * d / atr[ref]
    return f


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


def detect_setups(df: pd.DataFrame, return_pending: bool = False):
    m = to_m30(df)
    if len(m) < ATR_PERIOD + PIVOT_K + 5:
        return []
    atr = _atr(m).values
    ret_std = m["c"].pct_change().rolling(RET_STD_WINDOW).std().values
    # price z-score: how many std devs the close sits from its 20-bar mean
    # (mean-reversion / stretch signal), read causally at the last bar.
    _cz = m["c"]
    price_z = ((_cz - _cz.rolling(20).mean()) / _cz.rolling(20).std()).values
    sh_at, sl_at, sh_idx, sl_idx = _confirmed_pivots(m, PIVOT_K)

    o, h, l, c = (m[x].values for x in "ohlc")
    v = m["v"].values if "v" in m.columns else None
    vol_z = None
    cum_v = cum_tpv = None
    if v is not None:
        vs = pd.Series(v)
        vol_z = ((vs - vs.rolling(50).mean())
                 / vs.rolling(50).std()).values
        # anchored-VWAP support: cumulative typical-price*volume so a VWAP
        # over any causal window [lo, hi] is an O(1) difference.
        vv = np.nan_to_num(v, nan=0.0)
        tp = (h + l + c) / 3.0
        cum_v = np.cumsum(vv)
        cum_tpv = np.cumsum(tp * vv)
    # order-flow imbalance per 30m bar: +1 all aggressive buying, -1 selling
    ofi = None
    if "tbuy" in m.columns:
        with np.errstate(invalid="ignore", divide="ignore"):
            ofi = (2.0 * m["tbuy"].values - v) / v
    # trade-level (tick) order flow: real signed delta, CVD, and the size
    # of the largest aggressive prints (absorption / big-player footprint).
    tdelta = tcvd = tmax_z = tbig_imb = None
    if "of_delta" in m.columns:
        ofv = m["of_vol"].values
        with np.errstate(invalid="ignore", divide="ignore"):
            tdelta = np.where(ofv > 0, m["of_delta"].values / ofv, np.nan)
        tcvd = m["of_cvd"].values
        omax = m["of_maxtrade"].values
        oms = pd.Series(omax)
        tmax_z = ((oms - oms.rolling(50).mean())
                  / oms.rolling(50).std()).values
        bmax, smax = m["of_buymax"].values, m["of_sellmax"].values
        denom = bmax + smax
        with np.errstate(invalid="ignore", divide="ignore"):
            tbig_imb = np.where(denom > 0, (bmax - smax) / denom, np.nan)
    # L2 resting-liquidity imbalance per structural bar (2023+ only)
    l2_imb = l2_imb1 = l2_depth_z = None
    if "l2_imb" in m.columns:
        l2_imb, l2_imb1 = m["l2_imb"].values, m["l2_imb1"].values
        ds = pd.Series(m["l2_depth"].values)
        l2_depth_z = ((ds - ds.rolling(50).mean())
                      / ds.rolling(50).std()).values
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
            feats = _kline_feats(j, a, entry, d, leg, impulse, risk, atr, idx,
                                 price_z, ret_std, c, vol_z, cum_v, cum_tpv,
                                 l, h, vv)
            # arm-time copy: the SAME features evaluated at the sweep/OB bar
            # (a["j"]) rather than the trigger. A pending limit must be scored
            # before price retraces, so this is what a first-touch (backtest-
            # aligned) live fill would decide on. Prefixed -> the live trigger
            # model is byte-identical (guarded by the causality test).
            for _k, _v in _kline_feats(a["j"], a, entry, d, leg, impulse, risk,
                                       atr, idx, price_z, ret_std, c, vol_z,
                                       cum_v, cum_tpv, l, h, vv).items():
                feats["arm_" + _k] = _v
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
            if tdelta is not None:
                # all strictly causal: sweep bar and pivot already closed,
                # entry/CVD read at the last completed bar j-1.
                eb, sj, pv = j - 1, a["j"], a["pivot"]
                if eb >= 0:
                    feats["tofi_sweep"] = tdelta[min(sj, eb)]
                    feats["tofi_entry"] = tdelta[eb]
                    seg = tdelta[max(0, pv):eb + 1]
                    feats["tofi_leg"] = (float(np.nanmean(seg))
                                         if len(seg) else np.nan)
                    # net CVD over the leg, normalised by leg volume in
                    # [-1, 1]: did real aggressive flow back the move or
                    # diverge (signed so >0 favours the trade direction)?
                    lo = max(0, pv)
                    vsum = float(np.nansum(m["of_vol"].values[lo:eb + 1]))
                    if vsum > 0:
                        feats["cvd_slope_leg"] = ((tcvd[eb] - tcvd[lo])
                                                  / vsum) * d
                    feats["maxtrade_z_sweep"] = tmax_z[min(sj, eb)]
                    feats["bigprint_imb_sweep"] = tbig_imb[min(sj, eb)] * d
            if l2_imb is not None:
                # resting book imbalance read at the last completed bar,
                # signed so >0 means the book backs the trade direction
                # (bids stacked under a long, asks stacked over a short).
                eb, sj = j - 1, a["j"]
                if eb >= 0:
                    feats["l2_imb_entry"] = l2_imb[eb] * d
                    feats["l2_imb_sweep"] = l2_imb[min(sj, eb)] * d
                    feats["l2_imb1_entry"] = l2_imb1[eb] * d
                    feats["l2_depth_z_entry"] = l2_depth_z[eb]
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
    if not return_pending:
        return setups
    # still-armed setups as of the last bar = live resting limit orders:
    # a limit at the order-block edge waiting for price to retrace, expiring
    # RETRACE_WINDOW bars after the sweep.
    last_i = len(m) - 1
    tf = pd.Timedelta(minutes=_tf_minutes())
    pending = []
    for a in armed:
        if a["expires"] < last_i:
            continue
        d = a["dir"]
        entry = a["ob_low"] if d == -1 else a["ob_high"]
        atr_l = atr[last_i]
        if not np.isfinite(atr_l):
            continue
        stop = (a["ext"] + STOP_BUFFER_ATR * atr_l if d == -1
                else a["ext"] - STOP_BUFFER_ATR * atr_l)
        exp = a["expires"]
        exp_time = idx[exp] if exp < len(m) else idx[-1] + (exp - last_i) * tf
        pending.append({"direction": d, "entry": float(entry),
                        "stop": float(stop), "level": float(a["level"]),
                        "arm_time": idx[a["j"]], "expire_time": exp_time})
    return setups, pending


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
