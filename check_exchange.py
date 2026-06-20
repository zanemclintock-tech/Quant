"""
Run this ON YOUR MAC to find which exchanges actually reach you (some
geo-block the UK), and which of our coins each one lists. It also pulls a
live best bid/ask so you can confirm the real-quote stream will work.

    python check_exchange.py

Pick the exchange that (a) connects and (b) lists the most of
BTC/ETH/SOL/BNB/LTC, then set EXCHANGE=<that> in your .env. Note: Kraken and
Coinbase are UK-legal but don't list BNB; KuCoin usually lists all five and
is reachable for public data, but isn't FCA-registered (fine for a data-only
stream, no account).
"""
from __future__ import annotations

import ccxt

BASES = ["BTC", "ETH", "SOL", "BNB", "LTC"]
QUOTES = ("USDT", "USD", "USDC")
# ordered best-first for a UK user wanting real bid/ask
CANDIDATES = ["kraken", "coinbase", "kucoin", "bybit", "okx", "binance"]


def resolve(markets, base):
    for q in QUOTES:
        if f"{base}/{q}" in markets:
            return f"{base}/{q}"
    return None


def main():
    print(f"Testing {len(CANDIDATES)} exchanges for connectivity + coins…\n")
    best = None
    for ex_id in CANDIDATES:
        try:
            ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
            markets = ex.load_markets()
        except Exception as e:
            print(f"✗ {ex_id:9} unreachable — {str(e)[:60]}")
            continue
        syms = {b: resolve(markets, b) for b in BASES}
        have = [b for b, s in syms.items() if s]
        # sample a live quote on BTC to prove the stream works
        quote = ""
        try:
            t = ex.fetch_ticker(syms["BTC"])
            quote = f"  BTC bid/ask {t.get('bid')}/{t.get('ask')}"
        except Exception:
            quote = "  (ticker fetch failed)"
        print(f"✓ {ex_id:9} lists {len(have)}/5: {have}"
              f"{' — missing ' + str([b for b in BASES if b not in have]) if len(have) < 5 else ''}"
              f"{quote}")
        if best is None or len(have) > best[1]:
            best = (ex_id, len(have))
    if best:
        print(f"\n→ Recommended: EXCHANGE={best[0]} "
              f"(connects, lists {best[1]}/5 coins). Put it in your .env.")
    else:
        print("\nNo exchange reachable — check your internet / VPN.")


if __name__ == "__main__":
    main()
