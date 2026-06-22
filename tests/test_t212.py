"""
Offline tests for the Trading 212 catalyst bot. No network, no API key:
sentiment/catalyst lexicons, RSS parsing, signal ranking + selection,
position sizing, and the bot-managed exit logic.
"""
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import t212_config as C
import t212_news as N
import t212_signals as S
import t212_bot as B


# ── sentiment / catalyst scoring ─────────────────────────────────────────
def test_sentiment_direction():
    assert N.score_sentiment("shares surge on strong record profit") > 0.2
    assert N.score_sentiment("stock plunges on weak guidance and losses") < -0.2
    assert abs(N.score_sentiment("the company held a meeting today")) < 0.2


def test_sentiment_negation():
    pos = N.score_sentiment("earnings beat expectations")
    neg = N.score_sentiment("earnings did not beat expectations")
    assert pos > 0 and neg < pos


def test_catalyst_picks_strongest():
    val, phrase = N.score_catalyst(
        "biotech wins fda approval after minor delay")
    assert val > 0.8 and phrase == "fda approval"

    val2, _ = N.score_catalyst("company faces sec investigation and lawsuit")
    assert val2 <= -0.9


def test_catalyst_none():
    val, phrase = N.score_catalyst("analysts discuss the broader sector")
    assert val == 0.0 and phrase == ""


# ── RSS parsing ──────────────────────────────────────────────────────────
SAMPLE_RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>NVDA soars after earnings beat</title>
    <description>Record revenue tops estimates</description>
    <link>http://example.com/1</link>
    <pubDate>Mon, 22 Jun 2026 12:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Analysts react</title>
    <description>price target raised to new high</description>
    <link>http://example.com/2</link>
    <pubDate>Mon, 22 Jun 2026 09:00:00 +0000</pubDate>
  </item>
</channel></rss>"""

SAMPLE_ATOM = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Company downgraded to sell</title>
    <summary>price target cut on weak demand</summary>
    <link href="http://example.com/a"/>
    <updated>2026-06-22T10:00:00Z</updated>
  </entry>
</feed>"""


def test_parse_rss():
    arts = N.parse_rss(SAMPLE_RSS, "test")
    assert len(arts) == 2
    assert arts[0].title == "NVDA soars after earnings beat"
    assert arts[0].published is not None
    assert arts[0].published.tzinfo is not None


def test_parse_atom():
    arts = N.parse_rss(SAMPLE_ATOM, "test")
    assert len(arts) == 1
    assert arts[0].link == "http://example.com/a"
    assert arts[0].published is not None


def test_parse_garbage_is_empty():
    assert N.parse_rss(b"not xml at all", "test") == []


def test_aggregate_uses_fresh_strong_catalyst():
    now = dt.datetime(2026, 6, 22, 13, 0, tzinfo=dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=C.NEWS_LOOKBACK_HOURS)
    tn = N.TickerNews("NVDA", articles=N.parse_rss(SAMPLE_RSS, "yahoo"))
    out = N.aggregate(tn, cutoff)
    assert out.catalyst > 0.8           # "earnings beat"
    assert out.sentiment > 0
    assert out.n_recent == 2


def test_relevance_filters_unrelated_search_hits():
    now = dt.datetime(2026, 6, 22, 13, 0, tzinfo=dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=C.NEWS_LOOKBACK_HOURS)
    # a Google-News-style hit that mentions a *different* company's miss
    unrelated = N.Article("Some Energy Stock Falls After Earnings Miss",
                          "unrelated company misses estimates",
                          "http://x", now, "google")
    tn = N.TickerNews("NVDA", articles=[unrelated])
    out = N.aggregate(tn, cutoff)
    assert out.catalyst == 0.0 and out.n_recent == 0   # filtered out

    # same headline but it actually names the ticker -> counts
    related = N.Article("NVDA Falls After Earnings Miss", "nvda misses",
                        "http://y", now, "google")
    tn2 = N.TickerNews("NVDA", articles=[related])
    out2 = N.aggregate(tn2, cutoff)
    assert out2.catalyst < 0 and out2.n_recent == 1


def test_aggregate_drops_stale():
    now = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)   # far future
    cutoff = now - dt.timedelta(hours=C.NEWS_LOOKBACK_HOURS)
    tn = N.TickerNews("NVDA", articles=N.parse_rss(SAMPLE_RSS, "yahoo"))
    out = N.aggregate(tn, cutoff)
    assert out.n_recent == 0 and out.catalyst == 0.0


# ── signals ──────────────────────────────────────────────────────────────
def _news(symbol, catalyst, sentiment):
    tn = N.TickerNews(symbol)
    tn.catalyst, tn.sentiment, tn.n_recent = catalyst, sentiment, 3
    tn.top_headline = f"{symbol} headline"
    return tn


def test_rank_and_select(monkeypatch):
    monkeypatch.setattr(S, "momentum_score", lambda s: 0.0)
    news = {
        "AAA": _news("AAA", 0.9, 0.5),     # strong buy
        "BBB": _news("BBB", -0.9, -0.5),   # bad news -> excluded
        "CCC": _news("CCC", 0.1, 0.05),    # too weak
    }
    sigs = S.rank_signals(news)
    assert sigs[0].symbol == "AAA"
    buys = S.select_buys(sigs, held=set(), slots=5)
    assert [b.symbol for b in buys] == ["AAA"]


def test_select_respects_held_and_slots(monkeypatch):
    monkeypatch.setattr(S, "momentum_score", lambda s: 0.0)
    news = {x: _news(x, 0.9, 0.6) for x in ("AAA", "BBB", "CCC")}
    sigs = S.rank_signals(news)
    buys = S.select_buys(sigs, held={"AAA"}, slots=1)
    assert len(buys) == 1 and buys[0].symbol != "AAA"


def test_size_position():
    # equity 1000, stop 8%, risk 2% -> budget 250, price 50 -> 5 shares
    qty = S.size_position(equity=1000, free_cash=1000, price=50)
    assert abs(qty - 5.0) < 1e-6
    # capped by cash
    assert S.size_position(equity=1000, free_cash=30, price=50) == 0.6
    # below min order value -> 0
    assert S.size_position(equity=1000, free_cash=1, price=50) == 0.0


# ── exit logic ───────────────────────────────────────────────────────────
def _pos(entry=100.0, opened="2026-06-20T12:00:00+00:00"):
    return {"entry_price": entry, "qty": 1.0,
            "stop": entry * (1 - C.STOP_LOSS_PCT),
            "target": entry * (1 + C.TAKE_PROFIT_PCT),
            "peak": entry, "opened_at": opened}


def test_exit_stop_and_target():
    now = dt.datetime(2026, 6, 22, tzinfo=dt.timezone.utc)
    close, reason = B.decide_exit(_pos(), 100 * (1 - C.STOP_LOSS_PCT) - 1, now)
    assert close and reason == "stop_loss"
    close, reason = B.decide_exit(_pos(), 100 * (1 + C.TAKE_PROFIT_PCT) + 1, now)
    assert close and reason == "take_profit"
    close, _ = B.decide_exit(_pos(), 101.0, now)
    assert not close


def test_exit_time_stop():
    pos = _pos(opened="2026-06-01T00:00:00+00:00")
    now = dt.datetime(2026, 6, 22, tzinfo=dt.timezone.utc)   # >10 days
    close, reason = B.decide_exit(pos, 101.0, now)
    assert close and reason == "time_stop"


def test_trade_time_gating():
    monday_early = dt.datetime(2026, 6, 22, 9, tzinfo=dt.timezone.utc)
    monday_open = dt.datetime(2026, 6, 22, 15, tzinfo=dt.timezone.utc)
    saturday = dt.datetime(2026, 6, 20, 15, tzinfo=dt.timezone.utc)
    assert not B.is_trade_time(monday_early, None)
    assert B.is_trade_time(monday_open, None)
    assert not B.is_trade_time(monday_open, monday_open.date())   # already today
    if C.TRADE_WEEKDAYS_ONLY:
        assert not B.is_trade_time(saturday, None)
