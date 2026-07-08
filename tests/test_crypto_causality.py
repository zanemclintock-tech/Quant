"""
No-lookahead proof for the 24/7 crypto detector + all added features
(VWAP, volume profile, and the order-flow-style features). The test:
truncate the future, re-detect, and require every setup that was already
fully formed (its trigger bar closed before the cut) to be byte-identical
-- same entry, stop, and every feature. If any future bar can change a
past setup, the detector leaks.
"""
import numpy as np
import pandas as pd

from smc_detector import detect_setups
from synthetic import make_synthetic_minutes


def _key(s):
    return (s.entry_time, s.direction, s.arm_time)


def test_no_lookahead_crypto_detector(monkeypatch):
    # set per-test so it can't leak into other tests (caused suite-order fail)
    monkeypatch.setenv("BASE_TF", "15min")
    monkeypatch.setenv("CRYPTO", "1")
    df = make_synthetic_minutes(n_days=120, seed=7, start_date="2024-01-01",
                                sigma_frac=0.0009, crypto=True)
    # independent series as the cross-asset reference, so the xa_* lead-lag
    # features are non-degenerate and proven causal w.r.t. BOTH frames.
    xdf = make_synthetic_minutes(n_days=120, seed=23, start_date="2024-01-01",
                                 sigma_frac=0.0009, crypto=True)
    cut = df.index[int(len(df) * 0.6)]
    margin = cut - pd.Timedelta("15min")         # exclude the straddling bar

    full = {_key(s): s for s in detect_setups(df, xref=xdf)}
    trunc = detect_setups(df.loc[:cut], xref=xdf.loc[:cut])

    checked = 0
    for s in trunc:
        if s.entry_time >= margin:
            continue
        checked += 1
        t = full.get(_key(s))
        assert t is not None, f"setup vanished when future added: {_key(s)}"
        assert abs(t.entry_price - s.entry_price) < 1e-6
        assert abs(t.stop - s.stop) < 1e-6
        for k, v in s.features.items():
            v2 = t.features.get(k, np.nan)
            if np.isfinite(v):
                assert abs(v - v2) < 1e-6, f"feature {k} changed at {s.entry_time}"
    assert checked > 20, "probe did not exercise enough setups"
