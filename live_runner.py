"""
Exchange-agnostic paper/live runner. Every cycle it pulls recent candles
for each asset, regenerates the strategy's full trade ledger (detect ->
model-select -> sequence through the FTMO account), and writes:

    ledger.csv    one row per trade (entry/exit, side, outcome, R, $P&L,
                  running equity) -- closed AND currently-open
    status.json   headline stats the dashboard shows (monthly %, win rate,
                  equity, drawdown, open positions)

Venue is set by EXCHANGE (any ccxt id: bybit, kraken, okx, coinbase...),
using PUBLIC candle data only -- no account, UK-accessible. Run the
dashboard (dashboard.py) alongside to watch it.

    EXCHANGE=bybit python live_runner.py
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

import backtest_ftmo as B
from smc_detector import detect_setups

ASSETS = {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT",
          "BNB": "BNB/USDT"}
TF = "15min"
INIT = 100_000.0
RISK = float(os.environ.get("RISK", "0.005"))
MAX_CONC = 4
# per-asset round-trip cost: entry/TP limit (maker) + stop market-cross,
# from the Corwin-Schultz spread study (BTC tight, alts wider), bps.
COST = {"BTC": (1.5, 7.0), "ETH": (1.5, 9.0), "SOL": (3.0, 16.0),
        "BNB": (2.5, 11.0)}


def _bundle_feats(df, feats):
    return df.reindex(columns=feats)


def build_ledger(klines: dict, bundles: dict):
    """Pure: candles per asset -> (ledger DataFrame, status dict)."""
    os.environ["BASE_TF"] = TF
    os.environ["CRYPTO"] = "1"
    picks = []
    pending = []
    now = max(df.index[-1] for df in klines.values())
    for a, df in klines.items():
        b = bundles[a]
        setups, pend = detect_setups(df, return_pending=True)
        cur = float(df["mid_c"].iloc[-1])
        for p in pend:                              # resting limit orders
            risk = abs(p["entry"] - p["stop"])
            if risk <= 0:
                continue
            tp = (p["entry"] - 2 * risk if p["direction"] < 0
                  else p["entry"] + 2 * risk)
            mins = (p["expire_time"] - now).total_seconds() / 60
            pending.append({
                "asset": a, "side": "sell" if p["direction"] < 0 else "buy",
                "limit": round(p["entry"], 2), "price": round(cur, 2),
                "away_%": round((p["entry"] - cur) / cur * 100, 2),
                "stop": round(p["stop"], 2), "target": round(tp, 2),
                "expires_in_min": int(max(mins, 0))})
        data = B.gen_trades(df, setups).dropna(subset=["realized_R"])
        if data.empty:
            continue
        p = b["model"].predict_proba(_bundle_feats(data, b["feats"]))[:, 1]
        sel = data[p >= b["threshold"]].copy()
        sel["asset"] = a
        sel["side"] = np.where(sel["dir_"] < 0, "sell", "buy")
        picks.append(sel[["entry_time", "exit_time", "asset", "side",
                          "outcome", "realized_R", "entry_px", "risk_px"]])
    pending = sorted(pending, key=lambda r: r["expires_in_min"])
    if not picks:
        return pd.DataFrame(), {"trades": 0, "pending": pending}
    allt = pd.concat(picks).sort_values("entry_time").reset_index(drop=True)

    # sequence through the account: no same-asset overlap, <=4 concurrent
    import heapq
    eq = peak = INIT
    floor = INIT - B.MAX_DD * INIT
    open_heap, open_assets = [], set()
    rows = []
    for _, t in allt.iterrows():
        while open_heap and open_heap[0][0] <= t["entry_time"]:
            ex, pnl, asset, idx = heapq.heappop(open_heap)
            open_assets.discard(asset)
            eq += pnl; peak = max(peak, eq)
            floor = min(peak - B.MAX_DD * INIT, INIT)
            rows[idx]["equity_after"] = eq
        if t["asset"] in open_assets or len(open_heap) >= MAX_CONC:
            continue
        e_bps, s_bps = COST[t["asset"]]
        exit_bps = e_bps if t["outcome"] == "tp" else s_bps
        cost_r = (e_bps + exit_bps) / 1e4 * t["entry_px"] / t["risk_px"]
        net_r = t["realized_R"] - cost_r
        pnl = net_r * RISK * INIT
        is_open = t["exit_time"] > now
        row = {"entry_time": t["entry_time"], "exit_time": t["exit_time"],
               "asset": t["asset"], "side": t["side"],
               "outcome": "OPEN" if is_open else t["outcome"],
               "R_net": round(net_r, 3), "pnl": round(pnl, 2),
               "win": int(net_r > 0), "equity_after": np.nan,
               "open": is_open}
        rows.append(row)
        idx = len(rows) - 1
        open_assets.add(t["asset"])
        heapq.heappush(open_heap, (t["exit_time"], pnl, t["asset"], idx))
    while open_heap:
        ex, pnl, asset, idx = heapq.heappop(open_heap)
        eq += pnl; peak = max(peak, eq)
        rows[idx]["equity_after"] = eq

    led = pd.DataFrame(rows)
    closed = led[~led["open"]]
    eqc = pd.Series(closed["equity_after"].values,
                    index=pd.to_datetime(closed["exit_time"]))
    maxdd = ((eqc.cummax() - eqc) / eqc.cummax()).max() if len(eqc) else 0.0
    bymon = closed.copy()
    bymon["m"] = pd.to_datetime(bymon["exit_time"]).dt.strftime("%Y-%m")
    monthly = (bymon.groupby("m")["pnl"].sum() / INIT * 100).round(2)
    status = {
        "equity": round(eq, 2), "total_ret_pct": round((eq / INIT - 1) * 100, 2),
        "this_month_pct": float(monthly.iloc[-1]) if len(monthly) else 0.0,
        "win_rate": round(closed["win"].mean() * 100, 1) if len(closed) else 0,
        "trades": int(len(closed)), "open_positions": int(led["open"].sum()),
        "maxdd_pct": round(float(maxdd) * 100, 2),
        "monthly": monthly.to_dict(),
        "pending": pending,
        "updated": str(pd.Timestamp.utcnow()),
    }
    return led, status


def fetch(ex, symbol, days=14):
    """Recent 1m candles -> harness frame (mid + volume), paginated."""
    import datetime as dt
    since = ex.milliseconds() - days * 86400 * 1000
    rows = []
    while since < ex.milliseconds():
        batch = ex.fetch_ohlcv(symbol, "1m", since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 60_000
        if len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000)
    d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).drop_duplicates("t")
    idx = pd.to_datetime(d["t"].astype("int64"), unit="ms", utc=True)
    out = pd.DataFrame(index=idx)
    out["mid_o"], out["mid_h"] = d["o"].values, d["h"].values
    out["mid_l"], out["mid_c"], out["volume"] = d["l"].values, d["c"].values, d["v"].values
    half = out["mid_c"] * 1e-4 / 2.0              # synth bid/ask the detector needs
    for c in ("o", "h", "l", "c"):
        out[f"bid_{c}"] = out[f"mid_{c}"] - half
        out[f"ask_{c}"] = out[f"mid_{c}"] + half
    out.index.name = "time"
    return out


def notify(msg: str):
    """Send a Telegram push if TELEGRAM_TOKEN/TELEGRAM_CHAT are set (free,
    instant phone notifications). No-op otherwise."""
    import urllib.parse
    import urllib.request
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT")
    if not (tok and chat):
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": msg,
                                   "parse_mode": "HTML"}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://api.telegram.org/bot{tok}/sendMessage", data=data),
            timeout=10)
    except Exception as e:
        print(f"  notify error: {e}")


def main():
    ex_id = os.environ.get("EXCHANGE", "bybit")
    bundles = {a: joblib.load(f"models/{a}_{TF}.joblib") for a in ASSETS}
    import ccxt
    ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
    print(f"[{ex_id}] paper runner | {list(ASSETS)} | risk {RISK:.1%}")
    notify(f"▶️ Runner started on {ex_id} — {', '.join(ASSETS)} @ {TF}")
    seen_closed, seen_pending, first = set(), set(), True
    while True:
        try:
            klines = {a: fetch(ex, sym) for a, sym in ASSETS.items()}
            led, status = build_ledger(klines, bundles)
            led.to_csv("ledger.csv", index=False)
            json.dump(status, open("status.json", "w"), indent=2)
            print(f"  {status['updated']}: equity {status.get('equity', 0):,.0f} "
                  f"| this month {status.get('this_month_pct', 0):+.2f}% "
                  f"| open {status.get('open_positions', 0)}")

            # push: newly CLOSED trades and newly-RESTING limit orders
            closed = led[~led["open"]] if len(led) else led
            for _, r in closed.iterrows():
                k = (r["asset"], str(r["exit_time"]))
                if k in seen_closed:
                    continue
                seen_closed.add(k)
                if not first:
                    ico = "✅" if r["win"] else "❌"
                    notify(f"{ico} {r['asset']} {r['side']} {r['outcome'].upper()} "
                           f"{r['R_net']:+.2f}R ({r['pnl']:+,.0f}) · "
                           f"equity ${r['equity_after']:,.0f} · "
                           f"month {status.get('this_month_pct', 0):+.2f}%")
            for p in status.get("pending", []):
                k = (p["asset"], p["side"], p["limit"])
                if k in seen_pending:
                    continue
                seen_pending.add(k)
                if not first:
                    notify(f"🔔 {p['asset']} {p['side']} limit @ {p['limit']} "
                           f"(stop {p['stop']}, target {p['target']}, "
                           f"expires {p['expires_in_min']}m)")
            first = False
        except Exception as e:
            print(f"  cycle error: {e}")
        time.sleep(300)


if __name__ == "__main__":
    main()
