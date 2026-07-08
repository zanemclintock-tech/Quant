# Start here — the simple version

You do **3 things, once**. Then it runs itself.

## Step 1 — set it up (answer a few questions)
In Terminal, inside the `Quant` folder:
```bash
python3 setup.py
```
It asks for an exchange (just press Enter for KuCoin), and optionally a
GitHub token (for the phone dashboard) and a Telegram bot (for alerts). You
can press Enter to skip those and add them later.

## Step 2 — start trading
```bash
./run_live.sh
```
Leave this Terminal window open. That's the engine running on real markets,
24/7. It keeps your Mac awake and restarts itself if anything hiccups.

## Step 3 — watch it
- **On your Mac:** open a *second* Terminal window and run
  `streamlit run live_dashboard.py` → a dashboard opens in your browser.
- **On your phone, anywhere:** do the 1-minute Streamlit step in `GO_LIVE.md`
  (paste the `GIST_ID` that `setup.py` printed). Then open the link on your
  phone and *Add to Home Screen*.

---

### That's it.
The engine finds trades, fills them at real bid/ask, manages them (2-min
hold, break-even, 1:2 cap, daily loss kept under 2%), and shows results on
the dashboard. If you set up Telegram, you'll also get a ping on every trade.

**Nothing to babysit.** Profits are tracked off a fixed $100k base, so the
monthly % you see is the real rate you'd repeat on a funded account.

If a command says "command not found":
- `git` → run `xcode-select --install`
- `python3` / `pip3` → install from python.org or `brew install python`
