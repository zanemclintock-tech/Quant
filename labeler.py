"""
Stage 2 — turn each detected setup into a realistic, costed outcome.

Entry is an honest NEXT-BAR market fill: when price first touches the
order-block edge inside the 30-min trigger window, we enter at the
following 1-minute bar's open (never the favourable extreme — that fake
edge shows up on random-walk noise; see model_train.py).

Two labels:
  * label_setups            — realistic bracket trade: TP at TARGET_RR x
    risk vs stop, stop checked before target, force-flat at the cash
    close, round-trip cost charged. This is the TRADING outcome.
  * label_setups_directional — gross directional return over a fixed
    horizon, no barriers/exits/costs. This is the clean LEARNING probe:
    on noise it is unpredictable, so any edge is real directional skill.

Labels are allowed to know the future (that is what a target is). The
model only ever sees the causal features attached at entry.

Both functions use numpy arrays + searchsorted so labelling 1.7M bars /
thousands of setups runs in seconds, not minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from smc_detector import _tf_minutes

TARGET_RR = 2.0          # reward:risk we label/trade at
MAX_HOLD_MIN = 8 * 60    # give up a setup that neither hits TP/SL same day
EOD = "15:55"


def _arrays(df: pd.DataFrame):
    # tz-naive New-York wall-clock so positions line up with setup times
    naive = df.index.tz_localize(None) if df.index.tz is not None else df.index
    return {
        "ts": naive.values.astype("datetime64[ns]"),
        "day": naive.normalize().values.astype("datetime64[ns]"),
        "o": df["mid_o"].values, "h": df["mid_h"].values,
        "l": df["mid_l"].values, "c": df["mid_c"].values,
        "spread": (df["ask_c"].values - df["bid_c"].values).clip(min=0.0),
        "n": len(df),
    }


def _fill_index(a, s, continuous: bool = False) -> tuple[int, float] | None:
    """Position of the next-bar fill and the entry price, or None."""
    t0 = np.datetime64(s.entry_time.tz_localize(None))
    t1 = t0 + np.timedelta64(_tf_minutes(), "m")   # one structural bar
    lo = int(np.searchsorted(a["ts"], t0, "left"))
    hi = int(np.searchsorted(a["ts"], t1, "left"))
    if hi <= lo:
        return None
    if s.direction == -1:
        touch = np.nonzero(a["h"][lo:hi] >= s.entry_price)[0]
    else:
        touch = np.nonzero(a["l"][lo:hi] <= s.entry_price)[0]
    if len(touch) == 0:
        return None
    trig = lo + int(touch[0])
    fill = trig + 1
    if fill >= a["n"]:
        return None
    if not continuous and a["day"][fill] != a["day"][trig]:
        return None                      # no overnight fills for session mkts
    return fill, float(a["o"][fill])


def _same_day_end(a, fill: int, horizon_min: int,
                  continuous: bool = False) -> int:
    """Last array position within `horizon_min` of the fill. Session
    markets cap at the same calendar day; continuous (24/7 crypto) runs
    the horizon straight through midnight."""
    t_end = a["ts"][fill] + np.timedelta64(horizon_min, "m")
    hi = int(np.searchsorted(a["ts"], t_end, "right"))
    hi = min(hi, a["n"])
    if continuous:
        return hi
    fill_day = a["day"][fill]
    seg_day = a["day"][fill + 1:hi]
    diff = np.nonzero(seg_day != fill_day)[0]
    if len(diff):
        hi = fill + 1 + int(diff[0])
    return hi


def label_setups_directional(df: pd.DataFrame, setups: list,
                             horizon_min: int = 60,
                             continuous: bool = False) -> pd.DataFrame:
    a = _arrays(df)
    rows = []
    for s in setups:
        risk = abs(s.entry_price - s.stop)
        if risk <= 0:
            continue
        fi = _fill_index(a, s, continuous)
        if fi is None:
            continue
        fill, entry = fi
        hi = _same_day_end(a, fill, horizon_min, continuous)
        if hi <= fill + 1:
            continue
        exit_px = float(a["c"][hi - 1])
        dir_r = s.direction * (exit_px - entry) / risk
        rows.append({**s.features, "entry_time": df.index[fill],
                     "year": int(df.index[fill].year),
                     "realized_R": dir_r, "win": int(dir_r > 0),
                     "outcome": "dir",
                     # carried for net-of-cost analysis only (kept out of
                     # the feature set via NON_FEATURES) — a fixed bps cost
                     # converts to R as cost_frac * entry / risk.
                     "entry_px": entry, "risk_px": risk})
    return pd.DataFrame(rows)


def label_setups(df: pd.DataFrame, setups: list,
                 continuous: bool = False) -> pd.DataFrame:
    a = _arrays(df)
    eod = pd.to_datetime(EOD).time()
    rows = []
    for s in setups:
        d = s.direction
        fi = _fill_index(a, s, continuous)
        if fi is None:
            continue
        fill, entry = fi
        risk = abs(entry - s.stop)
        if risk <= 0:
            continue
        cost = float(a["spread"][fill]) + 2 * C.SLIPPAGE_POINTS
        tp = entry - TARGET_RR * risk if d == -1 else entry + TARGET_RR * risk
        hi = _same_day_end(a, fill, MAX_HOLD_MIN, continuous)
        exit_px, outcome = np.nan, None
        for k in range(fill + 1, hi):
            hk, lk = a["h"][k], a["l"][k]
            if d == -1:
                if hk >= s.stop:   exit_px, outcome = s.stop, "sl"; break
                if lk <= tp:       exit_px, outcome = tp, "tp"; break
            else:
                if lk <= s.stop:   exit_px, outcome = s.stop, "sl"; break
                if hk >= tp:       exit_px, outcome = tp, "tp"; break
            # session markets force-flat at the cash close; 24/7 markets do
            # not — they ride to the TP/SL/MAX_HOLD horizon instead.
            if not continuous and pd.Timestamp(a["ts"][k]).time() >= eod:
                exit_px, outcome = a["c"][k], "eod"; break
        if outcome is None:
            exit_px, outcome = float(a["c"][hi - 1]), "eod"
        net = (entry - exit_px) * d - cost
        realized_r = net / risk
        rows.append({**s.features, "entry_time": df.index[fill],
                     "year": int(df.index[fill].year),
                     "realized_R": realized_r, "win": int(realized_r > 0),
                     "outcome": outcome})
    return pd.DataFrame(rows)
