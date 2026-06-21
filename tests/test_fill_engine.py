"""
Regression tests for the live fill engine (live_stream.on_quote), guarding the
phantom-fill bug: a limit armed when the market has ALREADY passed its entry
must NOT fill instantly at the limit price and stop out -- that produced a
multi-R loss on a market that never moved. A limit may only fill on a genuine
retrace INTO it (price first on the far side, then back to the level).
"""
import live_stream as L


def _port():
    return {"open_notional": 0.0, "equity": 100_000.0, "day": None,
            "day_start": 100_000.0, "open_worst": 0.0, "peak_lev": 0.0}


def _buy_limit(now):
    return {"side": "buy", "entry": 100.0, "stop": 99.0, "tp": 102.0,
            "risk": 1.0, "prob": 0.7, "expire": now + 9999}


def test_no_phantom_fill_when_price_already_below_entry():
    # market is already BELOW a buy limit (stale setup) -> must NOT fill,
    # so equity is untouched (this was the -2.6% phantom-loss bug).
    now = 1_000_000.0
    state = {"limits": [_buy_limit(now)], "pos": None}
    port = _port()
    for _ in range(5):                       # several ticks, price stuck at 98
        L.on_quote("BTC", 97.99, 98.0, state, now, port)
        now += 1
    assert state["pos"] is None, "stale limit phantom-filled"
    assert port["equity"] == 100_000.0, "equity moved with no real trade"


def test_genuine_retrace_fills_at_real_price():
    now = 1_000_000.0
    state = {"limits": [_buy_limit(now)], "pos": None}
    port = _port()
    # price is above the limit first (arms the resting order, no fill yet)
    L.on_quote("BTC", 100.99, 101.0, state, now, port)
    assert state["pos"] is None
    # then retraces down to the limit -> fills at ~the entry, not a phantom
    now += 5
    L.on_quote("BTC", 99.99, 100.0, state, now, port)
    assert state["pos"] is not None, "genuine retrace did not fill"
    assert abs(state["pos"]["entry"] - 100.0) < 0.05


def test_stale_buy_does_not_lose_money():
    # full path: arm stale, feed quotes, confirm no catastrophic loss booked
    now = 1_000_000.0
    state = {"limits": [_buy_limit(now)], "pos": None}
    port = _port()
    evs = []
    for _ in range(10):
        evs += L.on_quote("BTC", 97.9, 98.0, state, now, port)
        now += 30
    assert not any(e["type"] == "EXIT" for e in evs)
    assert port["equity"] == 100_000.0


def test_fill_price_never_worse_than_market_on_gap():
    # if price gaps THROUGH a ready buy limit, fill at the (better) ask,
    # never at the stale limit price
    now = 1_000_000.0
    state = {"limits": [_buy_limit(now)], "pos": None}
    port = _port()
    L.on_quote("BTC", 100.99, 101.0, state, now, port)        # arm (far side)
    now += 5
    L.on_quote("BTC", 99.49, 99.5, state, now, port)          # gap down to 99.5
    assert state["pos"] is not None
    assert state["pos"]["entry"] <= 100.0 + 1e-9              # not phantom 100+
