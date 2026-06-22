"""
Signal construction + position sizing.

Turns the news scan into a ranked list of buy candidates. The combined
score blends three components, each in roughly [-1, 1]:

    score = W_CATALYST * catalyst + W_SENTIMENT * sentiment
            + W_MOMENTUM * momentum

Momentum is a price-trend confirmation (don't buy a "good news" name that
is technically falling apart). It uses yfinance if it's installed; with no
price source the momentum term is simply 0 and the bot trades on news
alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import t212_config as C
from t212_news import TickerNews


@dataclass
class Signal:
    symbol: str
    score: float
    catalyst: float
    sentiment: float
    momentum: float
    headline: str
    n_recent: int

    def as_row(self) -> dict:
        return {"symbol": self.symbol, "score": round(self.score, 3),
                "catalyst": self.catalyst, "sentiment": self.sentiment,
                "momentum": round(self.momentum, 3),
                "n_recent": self.n_recent, "headline": self.headline[:120]}


def momentum_score(symbol: str) -> float:
    """20-day price trend mapped to [-1, 1] (0 if no price source).

    Uses (last / SMA20 - 1) scaled so ±10% from the average saturates.
    Returns 0.0 on any error so the pipeline degrades gracefully.
    """
    try:
        import yfinance as yf
    except Exception:
        return 0.0
    try:
        hist = yf.Ticker(symbol).history(period="2mo", interval="1d")
        closes = hist["Close"].dropna()
        if len(closes) < 20:
            return 0.0
        sma20 = closes.iloc[-20:].mean()
        if sma20 <= 0:
            return 0.0
        dev = float(closes.iloc[-1] / sma20 - 1.0)
        return max(-1.0, min(1.0, dev / 0.10))
    except Exception:
        return 0.0


def build_signal(tn: TickerNews, use_momentum: bool = True) -> Signal:
    mom = momentum_score(tn.symbol) if use_momentum else 0.0
    score = (C.W_CATALYST * tn.catalyst
             + C.W_SENTIMENT * tn.sentiment
             + C.W_MOMENTUM * mom)
    return Signal(symbol=tn.symbol, score=score, catalyst=tn.catalyst,
                  sentiment=tn.sentiment, momentum=mom,
                  headline=tn.top_headline, n_recent=tn.n_recent)


def rank_signals(news: dict[str, TickerNews],
                 use_momentum: bool = True) -> list[Signal]:
    """All symbols scored, sorted best-first."""
    sigs = [build_signal(tn, use_momentum) for tn in news.values()]
    sigs.sort(key=lambda s: s.score, reverse=True)
    return sigs


def select_buys(signals: list[Signal], held: set[str],
                slots: int) -> list[Signal]:
    """Pick up to ``slots`` new positions: above the score threshold, not
    already held, positive catalyst (never buy purely on a bad catalyst)."""
    out = []
    for s in signals:
        if len(out) >= slots:
            break
        if s.symbol in held:
            continue
        if s.score < C.MIN_SIGNAL_SCORE:
            continue
        if s.catalyst < 0:           # bad-news name even if other terms lift it
            continue
        out.append(s)
    return out


def size_position(equity: float, free_cash: float, price: float) -> float:
    """Fractional share quantity for one new position.

    Budget per trade = RISK_PER_TRADE * equity / STOP_LOSS_PCT, i.e. risk a
    fixed fraction of equity given the bot-managed stop distance. Capped by
    available cash and the min order value.
    """
    if price <= 0:
        return 0.0
    risk_budget = C.RISK_PER_TRADE * equity / max(C.STOP_LOSS_PCT, 1e-6)
    budget = min(risk_budget, free_cash)
    if budget < C.MIN_ORDER_VALUE:
        return 0.0
    qty = budget / price
    return round(qty, 4)
