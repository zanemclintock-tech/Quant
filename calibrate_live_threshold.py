"""
Calibrate the LIVE arming threshold to the SERVING score distribution and bake
it into models/pooled_15min.joblib as `live_threshold`.

Why this exists: the model is trained on features from FULL-HISTORY detection
(label_live.py runs detect_setups over the whole frame). The live runner scores
features from a rolling DETECT_DAYS window instead, and detect_setups yields
systematically lower-scoring features on a slice (vol_z / cvd / xa / fib all
drift down). So live scores sit ~0.15 below training, and the training threshold
(the top-16% cut on full-history scores, ~0.40) arms ~nothing live -- the root
cause of chronic under-arming (measured: 0/40 live setups pass vs ~15% on the
training features for the same bars).

Fix: replay the exact live path (windowed detect -> score) over a recent
validation window, and set live_threshold to the SAME top-(1-STRICT_Q) cut on
THAT distribution. Restores the designed ~2 arms/day. Re-run after each retrain.

    CRYPTO=1 python train_live_models.py
    CRYPTO=1 python calibrate_live_threshold.py     # -> updates the bundle
"""
import os
os.environ.setdefault("CRYPTO", "1")
os.environ["BASE_TF"] = "15min"
os.environ.setdefault("ORDERFLOW", "0")
import numpy as np
import pandas as pd
import joblib
from smc_detector import detect_setups
from crypto_loader import load_binance_klines

TF = "15min"
DETECT_DAYS = int(os.environ.get("DETECT_DAYS", "3"))     # must match live_stream
CAL_DAYS = int(os.environ.get("CAL_DAYS", "21"))          # validation window
STRICT_Q = float(os.environ.get("STRICT_Q", "0.84"))      # same top-% as training
POOL = [c.strip().upper() for c in
        os.environ.get("COINS", "BTC,ETH,SOL").split(",") if c.strip()]


def main():
    bundle = joblib.load(f"models/pooled_{TF}.joblib")
    m, thr, feats = bundle["model"], bundle["threshold"], bundle["feats"]
    data = {a: load_binance_klines(f"data/{a.lower()}").tz_localize(None)
            for a in POOL}
    btc = data.get("BTC")
    end = min(d.index.max() for d in data.values())
    res = btc["mid_c"].resample(TF).last()
    bars = [b for b in res.index if b >= end - pd.Timedelta(days=CAL_DAYS)]
    print(f"calibrating over {len(bars)} bars x {len(POOL)} assets "
          f"(DETECT_DAYS={DETECT_DAYS}, target top {100*(1-STRICT_Q):.0f}%)")

    scores = []
    for a in POOL:
        df = data[a]
        for b in bars:
            w = df.loc[b - pd.Timedelta(days=DETECT_DAYS):b]
            xr = btc.loc[b - pd.Timedelta(days=DETECT_DAYS):b]
            if len(w) < 200:
                continue
            for s in detect_setups(w, xref=xr):
                if pd.Timestamp(s.entry_time) != b:
                    continue
                x = pd.DataFrame([{f: s.features.get(f, np.nan) for f in feats}])
                scores.append(float(m.predict_proba(x[feats])[:, 1][0]))
    sc = np.array(scores)
    if len(sc) < 100:
        print(f"  only {len(sc)} live setups -- too few to calibrate, aborting")
        return
    live_thr = float(np.quantile(sc, STRICT_Q))
    days = len(bars) * len(POOL) / len(POOL)          # calendar days
    cal_days = (bars[-1] - bars[0]).total_seconds() / 86400
    armed_day = (sc >= live_thr).sum() / max(cal_days, 1)
    print(f"  live setups scored: {len(sc)} ({len(sc)/max(cal_days,1):.1f}/day)")
    print(f"  serving score dist: p50 {np.median(sc):.3f}  p{100*STRICT_Q:.0f} "
          f"{live_thr:.3f}  max {sc.max():.3f}")
    print(f"  training threshold {thr:.3f} would arm "
          f"{(sc>=thr).sum()/max(cal_days,1):.2f}/day (the bug)")
    print(f"  -> live_threshold {live_thr:.3f} arms {armed_day:.2f}/day")
    bundle["live_threshold"] = live_thr
    joblib.dump(bundle, f"models/pooled_{TF}.joblib")
    print(f"  saved live_threshold={live_thr:.3f} to models/pooled_{TF}.joblib")


if __name__ == "__main__":
    main()
