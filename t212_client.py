"""
Thin client for the Trading 212 public REST API (beta).

Docs: https://docs.trading212.com/api

Design notes
------------
* Auth is a single API key in the ``Authorization`` header (no Bearer
  prefix), generated in the app under Settings -> API. Invest / Stocks
  ISA accounts only.
* Two environments share this code: demo (paper) and live. The base URL
  comes from :mod:`t212_config`.
* The API is rate-limited per endpoint and the limits are tight (the
  instrument list especially). We throttle per path and cache the
  instrument map, which is the only heavy call.
* Live accounts accept **market orders only** — there is no native stop
  or limit, so exits are managed by the bot, not the broker.

Pure stdlib (urllib) so it adds no dependency to the repo.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import t212_config as C


class T212Error(RuntimeError):
    """Any non-2xx response from the API (carries status + body)."""

    def __init__(self, status: int, body: str, path: str):
        super().__init__(f"{status} on {path}: {body[:300]}")
        self.status = status
        self.body = body
        self.path = path


class T212Client:
    """Minimal, rate-limit-aware Trading 212 API wrapper."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 20.0):
        self.api_key = api_key if api_key is not None else C.API_KEY
        self.base_url = (base_url or C.BASE_URL).rstrip("/")
        self.timeout = timeout
        self._last_call: dict[str, float] = {}     # path -> monotonic ts
        self._instruments: dict[str, str] | None = None  # symbol -> t212 ticker

    # ── low-level ─────────────────────────────────────────────────────────
    def _throttle(self, path: str, min_gap: float) -> None:
        last = self._last_call.get(path, 0.0)
        wait = min_gap - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        self._last_call[path] = time.monotonic()

    def _request(self, method: str, path: str, *, body: dict | None = None,
                 params: dict | None = None, min_gap: float = 1.0,
                 retries: int = 3) -> Any:
        if not self.api_key:
            raise T212Error(0, "no API key set (T212_API_KEY)", path)
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": self.api_key, "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"

        for attempt in range(retries):
            self._throttle(path, min_gap)
            req = urllib.request.Request(url, data=data, method=method,
                                         headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                bodytxt = e.read().decode(errors="replace")
                # 429 = rate limited: back off and retry.
                if e.code == 429 and attempt < retries - 1:
                    time.sleep(2 ** attempt * min_gap + 1)
                    continue
                raise T212Error(e.code, bodytxt, path) from None
            except urllib.error.URLError as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise T212Error(0, str(e.reason), path) from None
        raise T212Error(0, "exhausted retries", path)

    # ── account ───────────────────────────────────────────────────────────
    def account_cash(self) -> dict:
        """Free / total / invested cash for the account."""
        return self._request("GET", "/equity/account/cash", min_gap=2.0)

    def account_info(self) -> dict:
        """Account metadata (currencyCode, id)."""
        return self._request("GET", "/equity/account/info", min_gap=2.0)

    def portfolio(self) -> list[dict]:
        """All open positions."""
        return self._request("GET", "/equity/portfolio", min_gap=5.0) or []

    def position(self, ticker: str) -> dict | None:
        try:
            return self._request("GET", f"/equity/portfolio/{ticker}",
                                  min_gap=1.0)
        except T212Error as e:
            if e.status == 404:
                return None
            raise

    # ── instruments ───────────────────────────────────────────────────────
    def instruments(self, force: bool = False) -> dict[str, str]:
        """Map of friendly symbol -> Trading 212 ticker (e.g. AAPL ->
        AAPL_US_EQ). Cached; the underlying endpoint is heavily throttled.
        """
        if self._instruments is not None and not force:
            return self._instruments
        rows = self._request("GET", "/equity/metadata/instruments",
                             min_gap=50.0) or []
        mapping: dict[str, str] = {}
        for r in rows:
            t212 = r.get("ticker")
            short = r.get("shortName") or ""
            if not t212:
                continue
            # Prefer the human short name (AAPL); fall back to the part of
            # the t212 ticker before the first underscore.
            for key in (short.upper(), t212.split("_")[0].upper()):
                if key and key not in mapping:
                    mapping[key] = t212
        self._instruments = mapping
        return mapping

    def resolve(self, symbol: str) -> str | None:
        """Friendly symbol -> T212 ticker, or None if not tradable here."""
        sym = symbol.upper()
        m = self.instruments()
        if sym in m:
            return m[sym]
        # Allow callers to pass an already-resolved t212 ticker through.
        if "_" in symbol:
            return symbol
        return None

    # ── orders ────────────────────────────────────────────────────────────
    def place_market_order(self, ticker: str, quantity: float) -> dict:
        """Market order. Positive quantity buys, negative sells. Fractional
        quantities are allowed. ``ticker`` must be a resolved T212 ticker.
        """
        return self._request("POST", "/equity/orders/market", min_gap=2.0,
                             body={"ticker": ticker, "quantity": quantity})

    def open_orders(self) -> list[dict]:
        return self._request("GET", "/equity/orders", min_gap=1.0) or []

    def cancel_order(self, order_id: int | str) -> Any:
        return self._request("DELETE", f"/equity/orders/{order_id}",
                             min_gap=1.0)

    def order_history(self, limit: int = 50) -> dict:
        return self._request("GET", "/equity/history/orders", min_gap=6.0,
                             params={"limit": limit})

    # ── pies ──────────────────────────────────────────────────────────────
    def pies(self) -> list[dict]:
        return self._request("GET", "/equity/pies", min_gap=5.0) or []

    def pie(self, pie_id: int | str) -> dict:
        return self._request("GET", f"/equity/pies/{pie_id}", min_gap=5.0)
