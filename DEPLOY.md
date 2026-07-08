# Going to demo — free dashboard on your phone

The dashboard runs the strategy on **real live prices** in the cloud and
shows monthly %, win/loss, equity and active limit orders. No account, no
keys, no money — a forward paper-test you can watch from anywhere.

## Deploy to Streamlit Community Cloud (free, ~5 min)

1. Go to **https://share.streamlit.io** and sign in with your **GitHub**
   account (the one this repo lives under).
2. Click **Create app → Deploy a public/ private app from GitHub**.
3. Fill in:
   - **Repository:** `zanemclintock-tech/quant`
   - **Branch:** `claude/retry-session-uf5pfk`  *(or `main` once merged)*
   - **Main file path:** `live_dashboard.py`
4. Click **Deploy**. It installs `requirements.txt` and launches (first
   build ~2-3 min).
5. You get a URL like `https://<name>.streamlit.app`. Open it on your
   phone → **Share → Add to Home Screen** so it behaves like an app.

## First time it loads

- It fetches ~30 days of candles for BTC/ETH/SOL/BNB and scores every
  setup — the **first load takes ~30-60s**, then it's cached and snappy.
- Use the **sidebar** to change exchange, history length, or risk %.

## If the charts are empty / a fetch error shows

Streamlit's servers must be able to reach the exchange's **public** API.
If the default (`bybit`) is blocked from their region, just switch the
**Exchange** dropdown in the sidebar to `kraken`, `coinbase`, or `okx` —
prices are effectively identical across majors.

## What this is (and isn't)

- **Is:** the strategy running on real prices, leak-free, with realistic
  per-coin costs — an honest forward simulation of monthly %.
- **Isn't:** real fills. No orders are placed, so it assumes the limit
  entries fill. That last bit is only proven by real resting orders
  (tiny live / prop demo) — the next step after you've watched this.

## Phone notifications (Telegram, free)

Pushes a message on every new signal and closed trade. These come from
the **runner** (`live_runner.py`), so it needs to be running somewhere
always-on (a small VPS, or your Mac while testing) — the passive cloud
dashboard can't push.

1. In Telegram, message **@BotFather** → `/newbot` → copy the **token**.
2. Message your new bot once, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy your
   **chat id** (the `"chat":{"id":...}` number).
3. Run the runner with:
   ```bash
   TELEGRAM_TOKEN=... TELEGRAM_CHAT=... EXCHANGE=bybit python live_runner.py
   ```
You'll get pings like
`✅ BTC sell TP +1.62R (+810) · equity $105,230 · month +3.21%` and
`🔔 SOL buy limit @ 81.67 (expires 12m)`.

## Tracking the month

The cloud view shows a rolling ~30-day window. To keep a permanent record,
just note the **"This month"** figure at month-end (as you planned). For a
full multi-year ledger, run `live_runner.py` on an always-on box (a small
VPS) so it writes `ledger.csv`, and point the dashboard at that.

## Prop-close mode: real bid/ask streaming (`live_stream.py`)

`live_stream.py` is the upgrade from candle-mid to **real quotes**. It
streams live best bid/ask (public WebSocket, ccxt.pro, no account) and
fills only when the market actually trades through your limit; stops exit
at the real opposite quote (honest slippage). Writes the same `ledger.csv`
the dashboard reads, and sends the same Telegram pings.

```bash
EXCHANGE=binance TELEGRAM_TOKEN=.. TELEGRAM_CHAT=.. python live_stream.py
```
Run it on an always-on machine (Mac while testing, or a small VPS) with
WebSocket access. If a venue's stream is blocked UK-side, switch EXCHANGE
to bybit / kraken / okx / coinbase (all ccxt.pro-supported).

Note: this still can't prove **queue position** (whether *your* order fills
given size resting ahead of you) — only a real order does that. But fills,
spread-crossing and stop slippage are now real, not modelled.

## Real fills + phone-anywhere (the full live setup)

This is the prop-close setup: real bid/ask fills, adaptive 1:2 sizing, and
the dashboard on your phone from anywhere — all off your always-on Mac.

On the Mac (one-time):
```bash
git clone <your repo>; cd Quant
git checkout claude/retry-session-uf5pfk      # or whichever branch is latest
pip install -r requirements.txt
```
Then run three things (e.g. three terminal tabs):
```bash
# 1) the real-fill engine (writes ledger.csv + status.json, sends pings)
EXCHANGE=binance TELEGRAM_TOKEN=.. TELEGRAM_CHAT=.. python live_stream.py
# 2) the dashboard, reading that live ledger
streamlit run live_dashboard.py
# 3) make it reachable from your phone anywhere (pick one):
#    Tailscale (private, recommended): install the app on Mac + phone, then
#    on your phone open  http://<mac-tailscale-ip>:8501
#    or a public link:   cloudflared tunnel --url http://localhost:8501
```
The dashboard auto-detects `ledger.csv` and switches to "local runner"
mode, so the phone view shows your REAL-quote fills, monthly %, win/loss,
drawdown and live leverage — not the candle simulation.

Sizing is adaptive-by-confidence and hard-capped at 1:2 leverage
(funded-account compliant) everywhere — dashboard, ledger, and live_stream.
