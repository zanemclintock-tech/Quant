"""
Load Dukascopy 1-minute bid/ask CSV exports into a single OHLC frame.

Expected CSV format (the same files trainer2.py / validator.py used):
    Time (EET),Open,High,Low,Close,Volume

Dukascopy "EET" timestamps follow Eastern European time *with* European
DST (UTC+2 winter / UTC+3 summer), i.e. the Europe/Athens zone. We
localize there and convert to America/New_York so the session logic
(09:30 cash open) is correct year-round, including the few weeks each
year when US and EU DST transitions are out of sync.

NOTE: the old trainer/validator never did this conversion — they ran
between_time("08:00","12:00") on raw EET stamps, which is 01:00–05:00
New York. The model was trained on the overnight session while the live
bot traded the NY morning. See ANALYSIS.md.
"""
from __future__ import annotations

import pandas as pd

OHLC = ["Open", "High", "Low", "Close"]


def _read_dukascopy_csv(path: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Time (EET)"] + OHLC)
    df["Time (EET)"] = pd.to_datetime(
        df["Time (EET)"], errors="coerce", dayfirst=False
    )
    df = df.dropna(subset=["Time (EET)"]).set_index("Time (EET)").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index = df.index.tz_localize(
        "Europe/Athens", ambiguous="NaT", nonexistent="NaT"
    )
    df = df[df.index.notna()]
    df.index = df.index.tz_convert("America/New_York")
    df.columns = [f"{prefix}_{c[0].lower()}" for c in OHLC]
    return df


def load_bid_ask(bid_csv: str, ask_csv: str) -> pd.DataFrame:
    """Return a 1-min frame with bid_o/h/l/c, ask_o/h/l/c and mid_h/l/c,
    indexed in America/New_York."""
    bid = _read_dukascopy_csv(bid_csv, "bid")
    ask = _read_dukascopy_csv(ask_csv, "ask")
    df = bid.join(ask, how="inner")
    df = df.dropna()
    # Guard against corrupt rows (crossed or absurd quotes)
    df = df[(df["ask_c"] >= df["bid_c"]) & (df["bid_l"] > 0)]
    df["mid_h"] = (df["bid_h"] + df["ask_h"]) / 2.0
    df["mid_l"] = (df["bid_l"] + df["ask_l"]) / 2.0
    df["mid_c"] = (df["bid_c"] + df["ask_c"]) / 2.0
    return df
