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
   - **Main file path:** `dashboard.py`
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

## Tracking the month

The cloud view shows a rolling ~30-day window. To keep a permanent record,
just note the **"This month"** figure at month-end (as you planned). For a
full multi-year ledger, run `live_runner.py` on an always-on box (a small
VPS) so it writes `ledger.csv`, and point the dashboard at that.
