# Go live on demo — real markets, phone dashboard, Telegram

This is the prop-close setup you asked for:

- **Real markets.** `live_stream.py` streams live best **bid/ask** (public
  WebSocket, no account, no money) and fills honestly — a resting limit only
  fills when the market actually trades through it, stops exit at the real
  opposite quote with real slippage, 2-minute min hold, one position per
  asset, adaptive sizing hard-capped at 1:2 **shared** leverage, and the
  cost-inclusive daily budget that keeps any day under 2%. These are the
  results that carry over to a funded account.
- **Phone dashboard, from anywhere, instantly.** Your Mac pushes its live
  ledger to a GitHub gist every few seconds; a free hosted dashboard reads
  it. One stable link, works on cellular, survives a wifi blip.
- **Telegram pings** on every new limit, fill and closed trade.

You run your Mac 24/7; everything else is automatic.

---

## 1. One-time Mac setup

```bash
git clone https://github.com/zanemclintock-tech/quant.git Quant
cd Quant
git checkout claude/retry-session-uf5pfk
pip3 install -r requirements.txt
```

## 2. Pick an exchange that reaches you (30 sec)

Bybit and OKX **geo-block the UK** — don't use them there. Find what works
from your line:
```bash
python3 check_exchange.py
```
It prints which exchanges connect and how many of BTC/ETH/SOL/BNB/LTC each
lists, then recommends one. Notes:
- **Kraken** — UK-legal, reliable, but **no BNB** (the engine just trades the
  other four; BNB is added later on the funded account).
- **KuCoin** — usually lists all five and is reachable for public data (it's
  a data-only stream, no account, so FCA registration doesn't apply).

Put the winner in `.env` as `EXCHANGE=...` (step 4). The engine auto-skips
any coin the exchange doesn't list, so nothing breaks either way.

## 3. Telegram notifications (2 min)

1. In Telegram, message **@BotFather** → `/newbot` → copy the **token**.
2. Message your new bot once (say "hi"), then open in a browser:
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy the
   **chat id** (the `"chat":{"id":<number>}`).

## 4. Cloud sync — so the dashboard sees your Mac from anywhere (3 min)

1. Create a GitHub token with **only the `gist` scope**:
   github.com/settings/tokens → *Generate new token (classic)* → tick
   **gist** → generate → copy it.
2. Create the gist the Mac will write to:
   ```bash
   GITHUB_TOKEN=<your-token> python3 cloud_sync.py
   ```
   It prints `GIST_ID=<id>`. Copy that id.

## 5. Put your secrets in `.env`

```bash
cp .env.example .env
```
Edit `.env` and fill in `EXCHANGE` (from step 2, default `kucoin` — lists all
5 coins and mirrors Goat's set), `TELEGRAM_TOKEN`, `TELEGRAM_CHAT`,
`GITHUB_TOKEN`, `GIST_ID`. (`.env` is gitignored.)

## 6. Run it 24/7

```bash
chmod +x run_live.sh
./run_live.sh
```
That keeps the Mac awake (`caffeinate`) and **auto-restarts** the engine if
it ever drops. Leave the terminal open. You'll get a Telegram
"▶️ Streaming runner…" message immediately.

**Want it to survive reboots / closing the terminal?** Use the launchd
service (auto-start at login, auto-restart on crash):
```bash
mkdir -p ~/Library/LaunchAgents
sed "s#__DIR__#$(pwd)#g" com.quant.live.plist.template \
    > ~/Library/LaunchAgents/com.quant.live.plist
launchctl load ~/Library/LaunchAgents/com.quant.live.plist
```
Logs go to `live.out`/`live.err` in the repo. Stop it with
`launchctl unload ~/Library/LaunchAgents/com.quant.live.plist`.

## 7. Deploy the phone dashboard (free, 5 min)

1. Go to **https://share.streamlit.io**, sign in with GitHub.
2. **Create app → from GitHub**:
   - Repository: `zanemclintock-tech/quant`
   - Branch: `claude/retry-session-uf5pfk`
   - Main file: `dashboard.py`
3. **Advanced settings → Secrets**, paste:
   ```toml
   GIST_ID = "<the id from step 4>"
   ```
4. **Deploy**. You get `https://<name>.streamlit.app`. Open it on your phone →
   **Share → Add to Home Screen** so it behaves like an app.

The dashboard auto-detects the gist and shows **"live · real bid/ask
fills"** — your Mac's actual fills, monthly %, win/loss, equity, drawdown,
open trades and active limit orders. It refreshes itself every 60s.

---

## What you'll see

- Telegram: `🔔 SOL buy limit @ 81.67 (stop 80.9, tp 83.1)` when a setup
  arms, `➡️ FILLED …`, then `✅ BTC buy TP +1.80R (+1,240) · equity
  $101,240` on close. A `· DAILY BUDGET USED` tag appears if the day's risk
  budget is spent (no more entries until the next UTC day).
- Dashboard: headline **This month %**, total return, equity curve, monthly
  bars, win rate, max drawdown (vs the 6% limit), open positions, and the
  resting limit-order table.

## Notes

- **Exchange:** default is `kucoin` (lists all 5 coins, mirrors Goat's set, a
  good proxy for the funded feed). `kraken` is a UK-legal fallback but has no
  BNB. Run `python3 check_exchange.py` to confirm what connects from your
  line. **Bybit/OKX geo-block the UK** — that's the "banned in your country"
  error; avoid them there. The engine auto-skips any coin the chosen venue
  doesn't list, and prices are near-identical across majors, so the choice
  only affects which coins trade.
- **Honesty boundary:** fills, spread-crossing and stop slippage are now
  **real**. The one thing a paper stream still can't prove is *queue
  position* (whether your specific order fills given size resting ahead) —
  only a real (prop demo) order does that. Everything else mirrors a funded
  account.
- Profits are taken off a fixed $100k base (no compounding), matching how
  you'll withdraw — so the monthly % you watch is the real, repeatable rate.
