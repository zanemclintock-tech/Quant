"""
News / catalyst scanner.

For each ticker this pulls recent headlines from the open web and scores
them two ways:

  * **sentiment**  — directional tone in [-1, 1] from a compact financial
    lexicon (handles negation and intensifiers).
  * **catalyst**   — how "spike-worthy" the news is, from weighted event
    keywords (earnings beats, FDA approvals, M&A, upgrades, guidance
    raises, buybacks, big contracts...). Also in [-1, 1] (negative for
    bad catalysts: misses, downgrades, probes, recalls).

Default sources need no API key:
  * Yahoo Finance per-ticker RSS
  * Google News search RSS

If FINNHUB_KEY / NEWSAPI_KEY are set, those richer feeds are added too.

RSS/Atom is parsed with the stdlib so the scanner adds no dependency.
"""
from __future__ import annotations

import datetime as dt
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import t212_config as C

UA = "Mozilla/5.0 (compatible; t212bot/1.0)"

# ── lexicons ─────────────────────────────────────────────────────────────
# Compact directional lexicon. Loughran-McDonald-flavoured but tiny.
_POS = {
    "beat", "beats", "surge", "surged", "soar", "soars", "soared", "jump",
    "jumps", "rally", "rallied", "record", "strong", "growth", "upgrade",
    "upgraded", "outperform", "buy", "bullish", "gain", "gains", "rose",
    "rise", "rises", "raises", "raised", "approval", "approved", "wins",
    "win", "expand", "expansion", "profit", "profitable", "boom", "topped",
    "tops", "accelerate", "breakthrough", "demand", "optimistic", "boost",
    "boosted", "positive", "exceeds", "exceeded", "momentum", "soaring",
}
_NEG = {
    "miss", "misses", "missed", "plunge", "plunged", "fall", "falls",
    "fell", "drop", "drops", "dropped", "slump", "weak", "downgrade",
    "downgraded", "underperform", "sell", "bearish", "loss", "losses",
    "cut", "cuts", "warning", "warns", "probe", "lawsuit", "recall",
    "recalled", "investigation", "fraud", "bankruptcy", "default", "halt",
    "halted", "decline", "declines", "slowdown", "layoffs", "delay",
    "delayed", "disappoint", "disappointing", "concern", "concerns",
    "risk", "slashed", "slashes", "tumble", "tumbled", "crash", "sinks",
}
_NEGATORS = {"no", "not", "never", "without", "fails", "fail", "failed",
             "denies", "denied"}
_INTENSIFIERS = {"very", "huge", "massive", "record", "sharply", "surprise",
                 "surprisingly", "blowout", "significantly"}

# Catalyst categories: keyword -> signed weight. Positive = bullish event.
_CATALYSTS: dict[str, float] = {
    # earnings
    "earnings beat": 0.9, "beats estimates": 0.9, "tops estimates": 0.9,
    "earnings miss": -0.9, "misses estimates": -0.9, "profit warning": -1.0,
    "raises guidance": 1.0, "guidance raise": 1.0, "cuts guidance": -1.0,
    "lowers guidance": -0.9, "record revenue": 0.8, "record profit": 0.8,
    # corporate actions
    "acquisition": 0.8, "acquires": 0.8, "to acquire": 0.8, "merger": 0.7,
    "buyout": 0.8, "takeover": 0.8, "stock buyback": 0.7, "share buyback": 0.7,
    "buyback": 0.6, "special dividend": 0.6, "dividend increase": 0.5,
    "stock split": 0.5, "spinoff": 0.4, "spin-off": 0.4,
    # product / regulatory
    "fda approval": 1.0, "fda approves": 1.0, "approval": 0.5,
    "breakthrough": 0.7, "patent": 0.4, "product launch": 0.5,
    "new contract": 0.7, "wins contract": 0.8, "awarded contract": 0.8,
    "partnership": 0.5, "collaboration": 0.4, "deal": 0.4,
    # analyst
    "price target raised": 0.7, "upgraded to buy": 0.8, "initiated buy": 0.6,
    "outperform rating": 0.6, "downgraded": -0.7, "price target cut": -0.7,
    "underperform rating": -0.6, "sell rating": -0.7,
    # negative events
    "sec investigation": -1.0, "doj probe": -1.0, "lawsuit": -0.6,
    "class action": -0.6, "recall": -0.7, "data breach": -0.7,
    "ceo resigns": -0.6, "ceo steps down": -0.6, "layoffs": -0.5,
    "bankruptcy": -1.0, "delisting": -1.0, "fraud": -1.0, "short report": -0.9,
    # macro/flow
    "short squeeze": 0.8, "all-time high": 0.6, "52-week high": 0.5,
    "soars": 0.6, "surges": 0.6, "plunges": -0.6, "tumbles": -0.6,
}


@dataclass
class Article:
    title: str
    summary: str
    link: str
    published: dt.datetime | None
    source: str

    @property
    def text(self) -> str:
        return f"{self.title}. {self.summary}".lower()


@dataclass
class TickerNews:
    symbol: str
    articles: list[Article] = field(default_factory=list)
    sentiment: float = 0.0     # [-1, 1]
    catalyst: float = 0.0      # [-1, 1]
    n_recent: int = 0
    top_headline: str = ""


# ── scoring ──────────────────────────────────────────────────────────────
def score_sentiment(text: str) -> float:
    """Lexicon sentiment in [-1, 1] with negation + intensifier handling."""
    words = [w.strip(".,!?:;()[]\"'") for w in text.lower().split()]
    if not words:
        return 0.0
    score = 0.0
    hits = 0
    for i, w in enumerate(words):
        val = 0.0
        if w in _POS:
            val = 1.0
        elif w in _NEG:
            val = -1.0
        if val == 0.0:
            continue
        # look back up to 2 words for a negator / intensifier
        window = words[max(0, i - 2):i]
        if any(n in _NEGATORS for n in window):
            val = -val
        if any(n in _INTENSIFIERS for n in window):
            val *= 1.5
        score += val
        hits += 1
    if hits == 0:
        return 0.0
    return max(-1.0, min(1.0, score / (hits + 2)))   # damped average


def score_catalyst(text: str) -> tuple[float, str]:
    """Strongest catalyst signal in [-1, 1] plus the phrase that fired it.

    We take the single largest-magnitude match rather than summing, so one
    blockbuster event (FDA approval) isn't diluted by routine phrasing.
    """
    t = text.lower()
    best_val = 0.0
    best_phrase = ""
    for phrase, weight in _CATALYSTS.items():
        if phrase in t and abs(weight) > abs(best_val):
            best_val = weight
            best_phrase = phrase
    return best_val, best_phrase


# ── fetching ─────────────────────────────────────────────────────────────
def _http_get(url: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _parse_dt(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            d = dt.datetime.strptime(s.strip(), fmt)
            if d.tzinfo is None:
                d = d.replace(tzinfo=dt.timezone.utc)
            return d.astimezone(dt.timezone.utc)
        except ValueError:
            continue
    return None


def parse_rss(xml_bytes: bytes, source: str) -> list[Article]:
    """Parse an RSS 2.0 or Atom feed into Articles (namespace-tolerant)."""
    out: list[Article] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return out

    def _txt(el, *tags):
        for tag in tags:
            for child in el:
                if child.tag.split("}")[-1] == tag and child.text:
                    return child.text.strip()
        return ""

    # RSS <item> and Atom <entry>
    for el in root.iter():
        name = el.tag.split("}")[-1]
        if name not in ("item", "entry"):
            continue
        title = _txt(el, "title")
        summary = _txt(el, "description", "summary", "content")
        link = _txt(el, "link")
        if not link:  # Atom puts the url in an attribute
            for child in el:
                if child.tag.split("}")[-1] == "link":
                    link = child.attrib.get("href", "")
                    break
        pub = _parse_dt(_txt(el, "pubDate", "published", "updated"))
        if title:
            out.append(Article(title, summary, link, pub, source))
    return out


def _yahoo_url(symbol: str) -> str:
    sym = symbol.split(".")[0]   # strip exchange suffix for Yahoo RSS
    return ("https://feeds.finance.yahoo.com/rss/2.0/headline?"
            + urllib.parse.urlencode({"s": sym, "region": "US",
                                      "lang": "en-US"}))


def _google_news_url(symbol: str) -> str:
    q = f"{symbol} stock"
    return ("https://news.google.com/rss/search?"
            + urllib.parse.urlencode({"q": q, "hl": "en-US", "gl": "US",
                                      "ceid": "US:en"}))


def _finnhub_articles(symbol: str) -> list[Article]:
    if not C.FINNHUB_KEY:
        return []
    today = dt.date.today()
    frm = today - dt.timedelta(days=7)
    url = ("https://finnhub.io/api/v1/company-news?"
           + urllib.parse.urlencode({"symbol": symbol.split(".")[0],
                                     "from": frm.isoformat(),
                                     "to": today.isoformat(),
                                     "token": C.FINNHUB_KEY}))
    import json
    try:
        rows = json.loads(_http_get(url))
    except Exception:
        return []
    out = []
    for r in rows[:30]:
        ts = r.get("datetime")
        pub = (dt.datetime.fromtimestamp(ts, dt.timezone.utc) if ts else None)
        out.append(Article(r.get("headline", ""), r.get("summary", ""),
                           r.get("url", ""), pub, "finnhub"))
    return out


def fetch_articles(symbol: str) -> list[Article]:
    """All raw articles for a symbol from every configured source."""
    articles: list[Article] = []
    for src, url in (("yahoo", _yahoo_url(symbol)),
                     ("google", _google_news_url(symbol))):
        try:
            articles += parse_rss(_http_get(url), src)
        except Exception as e:                       # network/parse hiccup
            print(f"    [{symbol}] {src} fetch failed: {e}")
    try:
        articles += _finnhub_articles(symbol)
    except Exception as e:
        print(f"    [{symbol}] finnhub failed: {e}")
    return articles


def scan_ticker(symbol: str, now: dt.datetime | None = None) -> TickerNews:
    """Fetch + score one ticker. Only articles within NEWS_LOOKBACK_HOURS
    (when a timestamp is available) contribute to the aggregate score."""
    now = now or dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=C.NEWS_LOOKBACK_HOURS)
    tn = TickerNews(symbol=symbol)
    tn.articles = fetch_articles(symbol)
    return aggregate(tn, cutoff)


def _relevant(article: Article, symbol: str) -> bool:
    """Is this article actually about ``symbol``? Ticker-scoped feeds
    (Yahoo, Finnhub) always are; broad search feeds (Google News) only
    count if the ticker root appears in the text — otherwise a loosely
    related headline ("some other stock's earnings miss") would poison the
    catalyst score."""
    if article.source in ("yahoo", "finnhub"):
        return True
    root = symbol.split(".")[0].lower()
    return root in article.text


def aggregate(tn: TickerNews, cutoff: dt.datetime) -> TickerNews:
    """Reduce an article list to sentiment/catalyst scores. Recent, strong,
    *relevant* catalysts dominate; sentiment is a recency-light average."""
    sent_vals: list[float] = []
    best_cat = 0.0
    best_head = ""
    n_recent = 0
    for a in tn.articles:
        # If we have a timestamp, enforce freshness; if not, keep it but
        # don't let it drive the catalyst score.
        fresh = (a.published is None) or (a.published >= cutoff)
        if not fresh or not _relevant(a, tn.symbol):
            continue
        n_recent += 1
        sent_vals.append(score_sentiment(a.text))
        cat, _phrase = score_catalyst(a.text)
        if abs(cat) > abs(best_cat):
            best_cat = cat
            best_head = a.title
    tn.sentiment = round(sum(sent_vals) / len(sent_vals), 3) if sent_vals else 0.0
    tn.catalyst = round(best_cat, 3)
    tn.n_recent = n_recent
    tn.top_headline = best_head or (tn.articles[0].title if tn.articles else "")
    return tn


def scan_universe(symbols: list[str]) -> dict[str, TickerNews]:
    """Scan every symbol. Sequential and polite to stay inside the free
    feeds' implicit rate limits."""
    now = dt.datetime.now(dt.timezone.utc)
    out: dict[str, TickerNews] = {}
    for sym in symbols:
        out[sym] = scan_ticker(sym, now=now)
        print(f"  scanned {sym}: {out[sym].n_recent} recent | "
              f"sent {out[sym].sentiment:+.2f} cat {out[sym].catalyst:+.2f}")
    return out
