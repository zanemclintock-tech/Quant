"""
Dark, self-contained monitor for the crypto SMC strategy.

Two ways it gets data:
  * if ledger.csv exists (live_runner.py is running locally) it shows that
    accumulating ledger;
  * otherwise it fetches recent public candles itself and rebuilds the
    ledger on the fly -- so it runs standalone, including on Streamlit
    Community Cloud (free) for phone access, with no separate runner.

    pip install streamlit
    streamlit run dashboard.py
"""
from __future__ import annotations

import json
import os

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crypto dashboard", layout="wide",
                   page_icon="📈")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 1rem; max-width: 1300px;}
  html, body, [class*="css"] {font-weight: 600;}
  [data-testid="stMetric"] {background: #151a23; border: 1px solid #222b38;
     border-radius: 12px; padding: 14px 16px;}
  [data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 800;}
  [data-testid="stMetricLabel"] {color: #8b97a7; font-weight: 700;}
  [data-testid="stDataFrame"] {font-weight: 700;}
  h1, h2, h3 {font-weight: 800 !important; letter-spacing: -.5px;}
  #MainMenu, footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)


def _ledger_from_text(text: str) -> pd.DataFrame:
    """Tolerant CSV loader: live_stream's ledger has no entry_time, the local
    runner's does -- parse whatever's present."""
    import io
    led = pd.read_csv(io.StringIO(text))
    for c in ("entry_time", "exit_time"):
        if c in led.columns:
            led[c] = pd.to_datetime(led[c], errors="coerce", utc=True)
    if "open" in led.columns:
        led["open"] = led["open"].astype(str).isin(["True", "true", "1"])
    else:
        led["open"] = False
    return led


@st.cache_data(ttl=30, show_spinner="Loading live data…")
def read_gist(gist_id: str):
    """Read status.json + ledger.csv the Mac pushed to a public gist."""
    import urllib.request
    req = urllib.request.Request(f"https://api.github.com/gists/{gist_id}",
                                 headers={"Accept": "application/vnd.github+json"})
    meta = json.load(urllib.request.urlopen(req, timeout=15))
    f = meta["files"]
    status = json.loads(f["status.json"]["content"]) if "status.json" in f else {}
    if "ledger.csv" in f:
        fc = f["ledger.csv"]
        text = (fc["content"] if not fc.get("truncated")
                else urllib.request.urlopen(fc["raw_url"], timeout=15)
                .read().decode())
        led = _ledger_from_text(text)
    else:
        led = pd.DataFrame(columns=["exit_time", "open"])
    return led, status


# ---- data source -------------------------------------------------------
# Only ever shows YOUR real data: the local ledger if the engine runs on this
# same machine, otherwise the cloud gist your Mac pushes to. No exchange is
# ever contacted here, so there's nothing to be geo-blocked.
def _gist_id() -> str:
    if os.environ.get("GIST_ID"):
        return os.environ["GIST_ID"]
    try:
        return st.secrets.get("GIST_ID", "")     # Streamlit Cloud secret
    except Exception:
        return ""


GIST_ID = _gist_id()

with st.sidebar:
    st.header("⚙︎ Settings")
    st.caption("Sizing: adaptive by confidence, capped at 1:2 leverage.")
    if st.button("↻ Refresh now"):
        st.cache_data.clear()

has_local = os.path.exists("ledger.csv")
if has_local:
    led = _ledger_from_text(open("ledger.csv").read())
    status = json.load(open("status.json")) if os.path.exists("status.json") else {}
    src = "this Mac · real fills"
elif GIST_ID:
    try:
        led, status = read_gist(GIST_ID)
        src = "live · real bid/ask fills"
    except Exception:
        led, status, src = pd.DataFrame(columns=["open"]), {}, "live (connecting…)"
else:
    # nothing wired up yet -> friendly instructions, not an error
    st.title("Crypto dashboard")
    st.info(
        "**Waiting for data.**\n\n"
        "This screen shows your live demo results once the engine is running.\n\n"
        "- **On your Mac**, after `./run_live.sh` is going, you'll see fills here.\n"
        "- **For phone-anywhere**, add your `GIST_ID` in the app's "
        "*Settings → Secrets* (see GO_LIVE.md). The dashboard then reads the "
        "data your Mac uploads.")
    st.stop()

closed = led[~led["open"]].copy() if len(led) else led

# ---- header ------------------------------------------------------------
st.title("Crypto dashboard")
st.caption(f"source: {src} · BTC · ETH · SOL · BNB · 15-min · adaptive 1:2 "
           f"(peak {status.get('peak_leverage', 0):.2f}x) · updated "
           f"{str(status.get('updated', '—'))[:19]}")

m = st.columns(6)
tm = status.get("this_month_pct", 0)
m[0].metric("This month", f"{tm:+.2f}%")
m[1].metric("Total return", f"{status.get('total_ret_pct', 0):+.1f}%")
m[2].metric("Equity", f"${status.get('equity', 0):,.0f}")
m[3].metric("Win rate", f"{status.get('win_rate', 0):.0f}%")
m[4].metric("Max drawdown", f"{status.get('maxdd_pct', 0):.2f}%",
            help="FTMO trailing limit = 6%")
m[5].metric("Open now", status.get("open_positions", 0))

# ---- active limit orders (resting, waiting to fill) --------------------
pend = pd.DataFrame(status.get("pending", []))
st.subheader(f"Active limit orders ({len(pend)})")
if len(pend):
    pend = pend.sort_values("expires_in_min")
    pend.columns = ["asset", "side", "limit", "price", "away %", "stop",
                    "target", "expires (min)"]
    st.dataframe(
        pend.style
        .map(lambda v: f"color: {'#10b981' if v == 'buy' else '#ef4444'}",
             subset=["side"])
        .bar(subset=["expires (min)"], color="#3b4250", vmin=0)
        .format({"limit": "{:,.2f}", "price": "{:,.2f}", "away %": "{:+.2f}",
                 "stop": "{:,.2f}", "target": "{:,.2f}"}),
        hide_index=True, use_container_width=True)
    st.caption("Limits expire when the retrace window closes — if price "
               "drifts too far and only returns later, the setup is stale "
               "(usually a loss), so it's cancelled instead.")
else:
    st.caption("No resting limit orders right now.")

if not len(closed):
    st.info("No closed trades in this window yet."); st.stop()

# ---- charts ------------------------------------------------------------
GRID = alt.Axis(grid=True, gridColor="#1d2530", domain=False, tickColor="#1d2530")
left, right = st.columns([3, 2])

with left:
    st.subheader("Equity")
    eq = closed[["exit_time", "equity_after"]].dropna()
    area = alt.Chart(eq).mark_area(
        line={"color": "#10b981"}, color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color="#0b0e14", offset=0),
                   alt.GradientStop(color="#10b981", offset=1)],
            x1=1, x2=1, y1=1, y2=0)).encode(
        x=alt.X("exit_time:T", title=None, axis=GRID),
        y=alt.Y("equity_after:Q", title=None, scale=alt.Scale(zero=False),
                axis=GRID)).properties(height=300)
    st.altair_chart(area, use_container_width=True)

with right:
    st.subheader("Monthly return %")
    mon = (pd.Series(status.get("monthly", {})).rename_axis("month")
           .reset_index(name="pct"))
    bars = alt.Chart(mon).mark_bar(cornerRadius=3).encode(
        x=alt.X("month:N", title=None, axis=alt.Axis(labelAngle=-45,
                grid=False, domain=False)),
        y=alt.Y("pct:Q", title=None, axis=GRID),
        color=alt.condition(alt.datum.pct >= 0, alt.value("#10b981"),
                            alt.value("#ef4444"))).properties(height=300)
    st.altair_chart(bars, use_container_width=True)

# ---- trades ------------------------------------------------------------
st.subheader("Recent trades")
recent = closed.sort_values("exit_time", ascending=False).head(25)[
    ["exit_time", "asset", "side", "outcome", "R_net", "pnl", "equity_after"]]
recent.columns = ["exit", "asset", "side", "outcome", "R", "P&L $", "equity"]
st.dataframe(
    recent.style
    .map(lambda v: f"color: {'#10b981' if v > 0 else '#ef4444'}",
         subset=["R", "P&L $"])
    .format({"R": "{:+.2f}", "P&L $": "{:+,.0f}", "equity": "{:,.0f}"}),
    hide_index=True, use_container_width=True, height=420)

st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)
