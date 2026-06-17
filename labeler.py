"""
Stage 2 — turn each detected setup into a realistic, costed outcome.

For every setup the limit entry sits at the order-block edge. We fill it
on 1-minute bars (when price reaches that edge inside the setup's 30-min
window), then walk forward minute by minute to a take-profit at
TARGET_RR x risk or the stop, whichever comes first. Stop is checked
before target within a bar (conservative). Position is force-flat at the
cash close; round-trip cost (entry spread + slippage both sides) is
charged in points and converted to R.

The realized R and win/loss this produces are the TRAINING TARGET. They
depend on what happened after entry — that is correct: a label is
allowed to know the outcome. The model never sees the outcome at predict
time; it only sees the causal features attached at entry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

TARGET_RR = 2.0          # reward:risk we label/trade at
MAX_HOLD_MIN = 8 * 60    # give up a setup that neither hits TP/SL same day
EOD = "15:55"


def label_setups_directional(df: pd.DataFrame, setups: list,
                             horizon_min: int = 60) -> pd.DataFrame:
    """Clean directional label for the LEARNING test.

    For each setup, fill the limit at the order-block edge (same as the
    bracket labeler), then measure the GROSS directional return over a
    fixed horizon: direction * (price[t+H] - entry) / risk. No stop, no
    target, no end-of-day truncation logic beyond the same-day cap, no
    costs. On a random walk this has mean zero and its sign is
    independent of every feature known at entry, so the noise null
    collapses to ~chance — which is the whole point: a real directional
    edge is no longer hidden under the bracket/exit artifact.
    """
    idx = df.index
    rows = []
    for s in setups:
        win_end = s.entry_time + pd.Timedelta(minutes=30)
        seg = df.loc[(idx >= s.entry_time) & (idx < win_end)]
        d = s.direction
        risk = abs(s.entry_price - s.stop)
        if seg.empty or risk <= 0:
            continue
        # trigger: price first touches the order-block edge
        if d == -1:
            hit = seg.index[seg["mid_h"].values >= s.entry_price]
        else:
            hit = seg.index[seg["mid_l"].values <= s.entry_price]
        if len(hit) == 0:
            continue
        # fill at the NEXT bar's open (honest market fill — never assume
        # execution at the favorable extreme; that fake edge shows up on
        # noise, see model_train.py)
        nxt = df.loc[idx > hit[0]]
        if nxt.empty or nxt.index[0].date() != hit[0].date():
            continue
        fill_t = nxt.index[0]
        entry = float(nxt.iloc[0]["mid_o"])
        h_end = fill_t + pd.Timedelta(minutes=horizon_min)
        same_day = df.loc[(idx > fill_t) & (idx <= h_end)
                          & (df.index.date == fill_t.date())]
        if same_day.empty:
            continue
        exit_px = float(same_day["mid_c"].iloc[-1])
        dir_r = d * (exit_px - entry) / risk
        rows.append({
            **s.features,
            "entry_time": fill_t,
            "year": fill_t.year,
            "realized_R": dir_r,        # gross directional return in R
            "win": int(dir_r > 0),
            "outcome": "dir",
        })
    return pd.DataFrame(rows)


def label_setups(df: pd.DataFrame, setups: list) -> pd.DataFrame:
    mid_h = df["mid_h"]; mid_l = df["mid_l"]; mid_c = df["mid_c"]
    spread = (df["ask_c"] - df["bid_c"]).clip(lower=0.0)
    idx = df.index
    eod_t = pd.to_datetime(EOD).time()

    rows = []
    for s in setups:
        win_end = s.entry_time + pd.Timedelta(minutes=30)
        # 1-min bars within the 30-min trigger window
        seg = df.loc[(idx >= s.entry_time) & (idx < win_end)]
        d = s.direction
        if seg.empty:
            continue
        # trigger: price first touches the order-block edge
        if d == -1:
            hit = seg.index[seg["mid_h"].values >= s.entry_price]
        else:
            hit = seg.index[seg["mid_l"].values <= s.entry_price]
        if len(hit) == 0:
            continue
        # honest next-bar market fill (never the favorable extreme)
        nxt = df.loc[idx > hit[0]]
        if nxt.empty or nxt.index[0].date() != hit[0].date():
            continue
        fill_t = nxt.index[0]
        entry = float(nxt.iloc[0]["mid_o"])
        risk = abs(entry - s.stop)
        if risk <= 0:
            continue
        cost_pts = float(spread.get(fill_t, 0.0)) + 2 * C.SLIPPAGE_POINTS
        tp = (entry - TARGET_RR * risk if d == -1
              else entry + TARGET_RR * risk)

        # walk forward from the minute AFTER the fill, same day only
        fwd = df.loc[(idx > fill_t)
                     & (idx <= fill_t + pd.Timedelta(minutes=MAX_HOLD_MIN))]
        fwd = fwd[fwd.index.date == fill_t.date()]
        exit_px, outcome = np.nan, None
        for t, hi, lo in zip(fwd.index, fwd["mid_h"].values,
                             fwd["mid_l"].values):
            if d == -1:                              # short
                if hi >= s.stop:   exit_px, outcome = s.stop, "sl"; break
                if lo <= tp:       exit_px, outcome = tp, "tp"; break
            else:                                    # long
                if lo <= s.stop:   exit_px, outcome = s.stop, "sl"; break
                if hi >= tp:       exit_px, outcome = tp, "tp"; break
            if t.time() >= eod_t:
                exit_px, outcome = mid_c.get(t, np.nan), "eod"; break
        if outcome is None:
            last = fwd.index[-1] if len(fwd) else fill_t
            exit_px, outcome = float(mid_c.get(last, entry)), "eod"

        gross_pts = (entry - exit_px) * d
        net_pts = gross_pts - cost_pts
        realized_r = net_pts / risk
        rows.append({
            **s.features,
            "entry_time": fill_t,
            "year": fill_t.year,
            "realized_R": realized_r,
            "win": int(realized_r > 0),
            "outcome": outcome,
        })
    return pd.DataFrame(rows)
