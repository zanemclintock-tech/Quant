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
    # read Volume too if the export includes it (Dukascopy usually does);
    # it is tick volume — a usable proxy for CFD activity. Match the
    # column robustly (ignore case / surrounding spaces).
    cols = pd.read_csv(path, nrows=0).columns
    vol_col = next((c for c in cols if c.strip().lower() == "volume"), None)
    want = ["Time (EET)"] + OHLC + ([vol_col] if vol_col else [])
    df = pd.read_csv(path, usecols=want)
    if vol_col:
        df = df.rename(columns={vol_col: "Volume"})
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
    rename = {c: f"{prefix}_{c[0].lower()}" for c in OHLC}
    if "Volume" in df.columns:
        rename["Volume"] = f"{prefix}_v"
    return df.rename(columns=rename)


def load_bid_ask(bid_csv: str, ask_csv: str) -> pd.DataFrame:
    """Return a 1-min frame with bid_o/h/l/c, ask_o/h/l/c and mid_h/l/c,
    indexed in America/New_York."""
    bid = _read_dukascopy_csv(bid_csv, "bid")
    ask = _read_dukascopy_csv(ask_csv, "ask")
    df = bid.join(ask, how="inner")
    df = df.dropna()
    # Guard against corrupt rows (crossed or absurd quotes)
    df = df[(df["ask_c"] >= df["bid_c"]) & (df["bid_l"] > 0)]
    df["mid_o"] = (df["bid_o"] + df["ask_o"]) / 2.0
    df["mid_h"] = (df["bid_h"] + df["ask_h"]) / 2.0
    df["mid_l"] = (df["bid_l"] + df["ask_l"]) / 2.0
    df["mid_c"] = (df["bid_c"] + df["ask_c"]) / 2.0
    if "bid_v" in df.columns and "ask_v" in df.columns:
        df["volume"] = df["bid_v"] + df["ask_v"]
    elif "bid_v" in df.columns:
        df["volume"] = df["bid_v"]
    return df


def attach_secondary(df: pd.DataFrame, bid_csv: str, ask_csv: str,
                     prefix: str = "sp") -> pd.DataFrame:
    """Join a second instrument's mid OHLC (e.g. S&P 500) onto the NAS
    frame as {prefix}_h/{prefix}_l/{prefix}_c, aligned by timestamp.
    Enables the cross-asset SMT-divergence features."""
    bid = _read_dukascopy_csv(bid_csv, "b")
    ask = _read_dukascopy_csv(ask_csv, "a")
    sec = bid.join(ask, how="inner").dropna()
    out = pd.DataFrame(index=sec.index)
    out[f"{prefix}_h"] = (sec["b_h"] + sec["a_h"]) / 2.0
    out[f"{prefix}_l"] = (sec["b_l"] + sec["a_l"]) / 2.0
    out[f"{prefix}_c"] = (sec["b_c"] + sec["a_c"]) / 2.0
    return df.join(out, how="left")
