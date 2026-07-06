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


@st.cache_data(ttl=60, show_spinner="Loading live data…")
def read_gist(gist_id: str, user: str):
    """Read status.json + ledger.csv from the gist's RAW CDN
    (gist.githubusercontent.com) -- NOT the GitHub API. The unauthenticated
    API is capped at 60 req/hour and the dashboard refresh exhausts it; the
    raw CDN is effectively unlimited (just cached ~1 min)."""
    import time
    import urllib.error
    import urllib.request
    base = f"https://gist.githubusercontent.com/{user}/{gist_id}/raw"
    cb = int(time.time() // 30)                  # nudge past intermediary caches

    def get(fn):
        try:
            return urllib.request.urlopen(f"{base}/{fn}?_={cb}",
                                          timeout=15).read().decode()
        except urllib.error.HTTPError:
            return ""
    s = get("status.json")
    status = json.loads(s) if s.strip() else {}
    lt = get("ledger.csv")
    led = (_ledger_from_text(lt) if lt.strip()
           else pd.DataFrame(columns=["exit_time", "open"]))
    return led, status


# ---- data source -------------------------------------------------------
# Only ever shows YOUR real data: the local ledger if the engine runs on this
# same machine, otherwise the cloud gist your Mac pushes to. No exchange is
# ever contacted here, so there's nothing to be geo-blocked.
def _secret(name: str, default: str = "") -> str:
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default)     # Streamlit Cloud secret
    except Exception:
        return default


GIST_ID = _secret("GIST_ID")
GIST_USER = _secret("GIST_USER", "zanemclintock-tech")

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
        led, status = read_gist(GIST_ID, GIST_USER)
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
st.caption(f"source: {src} · BTC · ETH · SOL · 15-min · adaptive 1:2 "
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

# ---- engine health -----------------------------------------------------
# Surfaces whether the engine is actually detecting + streaming, so a silent
# stall (stale candles, dropped WS, or a detect-TF/model-TF mismatch that arms
# nothing) is visible at a glance instead of looking "fine but no trades".
h = status.get("health", {})
if h:
    now = pd.Timestamp.now(tz="UTC")

    def _age(ts):
        try:
            return (now - pd.Timestamp(ts)).total_seconds()
        except Exception:
            return None

    def _ago(ts):
        a = _age(ts)
        if a is None:
            return "never"
        if a < 90:
            return f"{int(a)}s ago"
        if a < 5400:
            return f"{int(a / 60)}m ago"
        return f"{a / 3600:.1f}h ago"

    det_age = _age(h.get("last_detect"))
    tf_ok = h.get("detect_tf") == h.get("model_tf")
    alive = det_age is not None and det_age < 360       # ~3 detect cycles
    if not tf_ok:
        st.error(f"🔴 **Timeframe mismatch** — detector on `{h.get('detect_tf')}` "
                 f"but models are `{h.get('model_tf')}`. **Nothing will arm.** "
                 f"Restart the engine on the latest code (BASE_TF must equal "
                 f"{h.get('model_tf')}).")
    elif not alive:
        st.error(f"🔴 **Engine stalled** — last detect {_ago(h.get('last_detect'))}. "
                 f"Detection isn't running. Check the Mac / terminal."
                 + (f" Last error: {h['last_error']}" if h.get("last_error") else ""))
    else:
        st.success(f"🟢 **Engine live** — last check {_ago(h.get('last_detect'))} · "
                   f"detect/model TF {h.get('detect_tf')} · "
                   f"{h.get('setups_on_last_bar', 0)} setups on last bar, "
                   f"{h.get('passed_gate', 0)} passed the model · "
                   f"{h.get('armed_total', 0)} limits armed since start")

    with st.expander("Engine health detail", expanded=not (tf_ok and alive)):
        cu, tk, ltk = (h.get("candle_updated", {}), h.get("ticks", {}),
                       h.get("last_tick", {}))
        src = h.get("candle_src", {})
        assets = sorted(set(cu) | set(tk) | set(ltk))
        rows = [{"asset": a, "candles updated": _ago(cu.get(a)),
                 "feed": src.get(a, "—"),
                 "ticks recv": tk.get(a, 0), "last tick": _ago(ltk.get(a))}
                for a in assets]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
        # ---- fill funnel: why aren't more trades happening? ----
        armed = h.get("armed_total", 0)
        filled = h.get("filled_total", 0)
        exp_nr = h.get("expired_noretrace", 0)
        exp_sp = h.get("expired_spread", 0)
        resolved = filled + exp_nr + exp_sp
        if resolved:
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Limits armed", armed)
            f2.metric("Filled", filled,
                      help=f"{filled/resolved*100:.0f}% of resolved limits")
            f3.metric("Expired: no retrace", exp_nr)
            f4.metric("Blocked: spread", exp_sp,
                      help="Spread too wide vs risk when price touched — "
                           "raise SPREAD_GATE_R if this is high (wide demo feed)")
            if exp_sp > max(3, 0.25 * resolved):
                st.warning(f"⚠︎ {exp_sp/resolved*100:.0f}% of limits were "
                           "blocked by the spread gate — your live spreads are "
                           "wider than the model assumes and it's costing trades. "
                           "Set `SPREAD_GATE_R=1.0` (or higher) and restart.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Last closed bar", str(h.get("last_closed_bar", "—"))[:16])
        c2.metric("Started", _ago(h.get("started")))
        c3.metric("Resting limits", h.get("resting", 0))
        if h.get("last_error"):
            st.caption(f"⚠︎ last error ({_ago(h.get('last_error_t'))}): "
                       f"{h['last_error']}")
        st.caption("Healthy = green within ~3 min, detect TF = model TF, and "
                   "ticks climbing on each asset. ‘setups on last bar’ shows the "
                   "detector is finding structure even when nothing passes the "
                   "model gate.")
else:
    st.caption("Engine health: not reported yet (update the engine to the latest "
               "code to see the live health panel).")

# ---- live open positions tracker ---------------------------------------
opn = pd.DataFrame(status.get("open", []))
st.subheader(f"Open positions ({len(opn)})")
if len(opn):
    def _prog(r):
        # how far to TP vs SL, as a 0-100% bar toward target
        span = abs(r["tp"] - r["stop"])
        return max(0.0, min(100.0, (r["price"] - r["stop"]) / span * 100
                            if r["side"] == "buy"
                            else (r["stop"] - r["price"]) / span * 100)) \
            if span else 0.0
    opn["→ TP/SL %"] = opn.apply(_prog, axis=1)
    view = opn[["asset", "side", "entry", "price", "unreal_R", "risk_pct", "rr",
                "to_tp", "to_tp_pct", "to_sl", "to_sl_pct", "lev", "hold_min",
                "→ TP/SL %"]].copy()
    view.columns = ["asset", "side", "entry", "price", "live R", "risk %", "RR",
                    "pts→TP", "→TP %", "pts→SL", "→SL %", "lev", "held (min)",
                    "→ TP/SL %"]
    st.dataframe(
        view.style
        .map(lambda v: f"color: {'#10b981' if v == 'buy' else '#ef4444'}",
             subset=["side"])
        .map(lambda v: f"color: {'#10b981' if v >= 0 else '#ef4444'}",
             subset=["live R"])
        .bar(subset=["→ TP/SL %"], color="#1f6f4f", vmin=0, vmax=100)
        .format({"entry": "{:,.2f}", "price": "{:,.2f}", "live R": "{:+.2f}R",
                 "risk %": "{:.2f}%", "RR": "{:.1f}", "pts→TP": "{:,.2f}",
                 "→TP %": "{:+.2f}%", "pts→SL": "{:,.2f}", "→SL %": "{:+.2f}%",
                 "lev": "{:.2f}x", "held (min)": "{:.0f}", "→ TP/SL %": "{:.0f}%"}),
        hide_index=True, use_container_width=True)
    st.caption("live R = unrealised reward in R multiples · pts→TP / pts→SL = "
               "price distance still to run to the target / stop · the bar "
               "shows progress from stop (0%) to target (100%).")
else:
    st.caption("No open positions right now.")

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
    # seed the $100k starting point so the curve draws from the base (and
    # isn't an empty/one-dot chart after the very first trade)
    if len(eq):
        start = pd.DataFrame({"exit_time": [eq["exit_time"].min()
                              - pd.Timedelta(minutes=1)], "equity_after": [100_000.0]})
        eq = pd.concat([start, eq], ignore_index=True)
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
