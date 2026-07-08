"""
Smoke test for the live-faithfulness audit (audit_live_faithfulness.py).

Skips automatically when 1-min kline data isn't available (the klines are not
committed -- only the order-flow parquets are), so CI stays green without data.
When data IS present (KL env var or a data/btc folder of Binance 1m klines), it
runs a tiny 1-week window and asserts the audit produces a sane coverage figure
and labels both the full-history and live-faithful setup populations.

The audit's PURPOSE is to catch detection-lookahead: if the backtest armed many
setups the live rolling-window path could not, coverage would be low. Measured
on a real window it is ~81% and the missed setups are the worst ones -- i.e.
detection timing is not what inflates the backtest (see CRYPTO_AUDIT.md).
"""
import os
import importlib

import pytest


def _have_klines():
    kl = os.environ.get("KL", "/home/user/Quant/data/btc_klines")
    import glob
    return bool(glob.glob(os.path.join(kl, "*.csv"))) or os.path.isdir("data/btc")


@pytest.mark.skipif(not _have_klines(),
                    reason="1-min klines not present (not committed); "
                           "set KL=<folder> to run this audit")
def test_live_faithfulness_smoke(monkeypatch):
    monkeypatch.setenv("WINDOW_WEEKS", "1")
    monkeypatch.setenv("WIN_END", "2024-02-15")
    mod = importlib.import_module("audit_live_faithfulness")
    importlib.reload(mod)
    cov, lf, ll = mod.main()
    # both populations should be labeled, and coverage is a valid percentage
    assert 0.0 <= cov <= 100.0
    assert len(lf) > 0 and len(ll) > 0
    # every labeled row carries the shared bracket outcome fields
    for df in (lf, ll):
        assert {"entry_time", "dir", "R", "win"}.issubset(df.columns)
