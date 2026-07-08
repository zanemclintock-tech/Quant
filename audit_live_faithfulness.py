"""
Live-faithfulness audit: does the backtest arm setups the LIVE runner cannot?

FULL  : detect_setups over full history (what the trainer/backtest used).
LIVE  : for each 15-min bar b, detect_setups on ONLY the rolling DETECT_DAYS
        window ending at b (exactly what live_stream can see), keeping setups
        whose entry == b.

Both are labeled with the SAME 2R bracket (stop-before-target, round-trip cost)
on real forward klines, so any win-rate/expectancy gap is purely a
detection-timing effect, not a labeling difference.

    python audit_live_faithfulness.py                 # default 6-week window
    WINDOW_WEEKS=2 python audit_live_faithfulness.py   # faster, for the test
"""
import os
os.environ.setdefault("CRYPTO", "1"); os.environ["BASE_TF"] = "15min"
os.environ.setdefault("ORDERFLOW", "0")
import numpy as np, pandas as pd
from smc_detector import detect_setups
from crypto_loader import load_binance_klines

KL = os.environ.get("KL", "/home/user/Quant/data/btc_klines")
DETECT_DAYS = int(os.environ.get("DETECT_DAYS", "3"))
WEEKS = int(os.environ.get("WINDOW_WEEKS", "6"))
END = pd.Timestamp(os.environ.get("WIN_END", "2024-03-01"))
START = END - pd.Timedelta(weeks=WEEKS)
COST_BPS = float(os.environ.get("COST_BPS", "5.0"))


def load():
    btc = load_binance_klines(KL)
    return btc.tz_localize(None) if btc.index.tz is not None else btc


def bracket_labels(btc, setups):
    """2R bracket on real forward prices; entry on the next 1-min bar open."""
    ts = btc.index.values.astype("datetime64[ns]").astype("int64")
    o, h, l, c = (btc["mid_o"].values, btc["mid_h"].values,
                  btc["mid_l"].values, btc["mid_c"].values)
    half = COST_BPS / 1e4 / 2.0
    rows = []
    for s in setups:
        d = s.direction; risk = abs(s.entry_price - s.stop)
        if risk <= 0:
            continue
        i0 = int(np.searchsorted(ts, pd.Timestamp(s.entry_time).value, "left")) + 1
        if i0 >= len(ts):
            continue
        entry = float(o[i0]); tp = entry + d * 2 * risk; stop = float(s.stop)
        cost = 2 * half * entry
        exit_px = None
        for k in range(i0, min(i0 + 8 * 60, len(ts))):     # up to 8h hold
            if d == 1:
                if l[k] <= stop: exit_px = stop; break
                if h[k] >= tp:   exit_px = tp; break
            else:
                if h[k] >= stop: exit_px = stop; break
                if l[k] <= tp:   exit_px = tp; break
        if exit_px is None:
            exit_px = float(c[min(i0 + 8 * 60, len(ts) - 1)])
        net = (exit_px - entry) * d - cost
        rows.append({"entry_time": pd.Timestamp(s.entry_time), "dir": int(d),
                     "R": net / risk, "win": int(net > 0)})
    return pd.DataFrame(rows)


def main():
    btc = load()
    full = [s for s in detect_setups(btc.loc[:END], xref=btc.loc[:END])
            if START <= pd.Timestamp(s.entry_time) < END]
    res = btc["mid_c"].resample("15min").last()
    bars = [b for b in res.index if START <= b < END]
    live = []
    for b in bars:
        w = btc.loc[b - pd.Timedelta(days=DETECT_DAYS):b]
        if len(w) < 200:
            continue
        for s in detect_setups(w, xref=w):
            if pd.Timestamp(s.entry_time) == b:
                live.append(s)

    lf, ll = bracket_labels(btc, full), bracket_labels(btc, live)
    fk = {(r.entry_time, r.dir) for r in lf.itertuples()}
    lk = {(r.entry_time, r.dir) for r in ll.itertuples()}
    rec = fk & lk
    cov = len(rec) / max(len(fk), 1) * 100

    print(f"window {START.date()}..{END.date()} ({WEEKS}w)  DETECT_DAYS={DETECT_DAYS}")
    print(f"FULL-HISTORY setups : {len(lf)}")
    print(f"LIVE-FAITHFUL setups: {len(ll)}")
    print(f"recovered live      : {len(rec)} = {cov:.0f}% of full-history setups\n")

    def summ(name, df):
        if len(df) == 0:
            print(f"{name:24}: n=0"); return
        print(f"{name:24}: n={len(df):4d} | win {df.win.mean()*100:5.1f}% | "
              f"mean R {df.R.mean():+.3f} | total R {df.R.sum():+6.1f}")
    summ("FULL backtest", lf)
    summ("LIVE-faithful", ll)
    only_full = lf[[(r.entry_time, r.dir) not in lk for r in lf.itertuples()]]
    summ("FULL-ONLY (live missed)", only_full)
    return cov, lf, ll


if __name__ == "__main__":
    main()
