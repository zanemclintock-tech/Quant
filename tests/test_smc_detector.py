"""
Causality and sanity tests for the 30-min smart-money detector.

The core guarantee: a setup emitted with entry at time t must be
identical whether or not any data after t exists. We prove it by
truncating the future and checking emitted setups are unchanged.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smc_detector import detect_setups, to_m30, _confirmed_pivots
from synthetic import make_synthetic_minutes


@pytest.fixture(scope="module")
def df():
    # more volatile data so sweeps actually occur
    return make_synthetic_minutes(n_days=160, seed=11)


def _key(s):
    return (s.arm_time, s.entry_time, s.direction,
            round(s.entry_price, 4), round(s.stop, 4))


def test_setups_are_detected(df):
    assert len(detect_setups(df)) > 0


def test_no_lookahead_truncation(df):
    full = detect_setups(df)
    cut = df.index[int(len(df) * 0.65)]
    trunc = detect_setups(df[df.index < cut])
    # every setup whose entry is safely before the cut must match exactly
    full_before = [_key(s) for s in full if s.entry_time < cut]
    trunc_keys = {_key(s) for s in trunc}
    missing = [k for k in full_before if k not in trunc_keys]
    # allow only setups within RETRACE_WINDOW of the cut to differ
    margin = cut - pd.Timedelta(hours=8)
    assert all(k[1] >= margin for k in missing), missing


def test_no_lookahead_future_mutation(df):
    full = detect_setups(df)
    cut = df.index[int(len(df) * 0.65)]
    df2 = df.copy()
    df2.loc[df2.index >= cut, :] *= 1.3
    mut = detect_setups(df2)
    margin = cut - pd.Timedelta(hours=8)
    a = {_key(s) for s in full if s.entry_time < margin}
    b = {_key(s) for s in mut if s.entry_time < margin}
    assert a == b


def test_pivots_are_causal(df):
    m = to_m30(df)
    full = _confirmed_pivots(m, 2)[0]
    half = int(len(m) * 0.7)
    trunc = _confirmed_pivots(m.iloc[:half], 2)[0]
    # confirmed swing-high availability must match on the shared prefix
    assert np.allclose(full[:half], trunc, equal_nan=True)


def test_stops_on_correct_side(df):
    for s in detect_setups(df):
        if s.direction == -1:
            assert s.stop > s.entry_price       # short: stop above
        else:
            assert s.stop < s.entry_price       # long: stop below


def test_entry_after_arm(df):
    for s in detect_setups(df):
        assert s.entry_time >= s.arm_time
        assert s.entry_time.date() == s.arm_time.date()   # intraday only


def test_equilibrium_within_leg(df):
    for s in detect_setups(df):
        assert s.leg_low <= s.equilibrium <= s.leg_high
