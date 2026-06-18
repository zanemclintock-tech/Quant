"""
Local dashboard for the paper/live runner. Reads ledger.csv + status.json
(written by live_runner.py) and shows equity, MONTHLY %, win/loss trades,
and drawdown vs the 6% limit. Auto-refreshes.

    pip install streamlit
    streamlit run dashboard.py
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Strategy Monitor", layout="wide")
st.title("📈 Crypto SMC — live monitor")

if not os.path.exists("ledger.csv"):
    st.warning("No ledger yet. Start the runner:  EXCHANGE=bybit python live_runner.py")
    st.stop()

led = pd.read_csv("ledger.csv", parse_dates=["entry_time", "exit_time"])
status = json.load(open("status.json")) if os.path.exists("status.json") else {}
closed = led[~led["open"]].copy()

c = st.columns(6)
c[0].metric("Equity", f"${status.get('equity', 0):,.0f}")
c[1].metric("Total return", f"{status.get('total_ret_pct', 0):+.1f}%")
c[2].metric("This month", f"{status.get('this_month_pct', 0):+.2f}%")
c[3].metric("Win rate", f"{status.get('win_rate', 0):.0f}%")
c[4].metric("Max drawdown", f"{status.get('maxdd_pct', 0):.2f}%",
            help="FTMO trailing limit is 6%")
c[5].metric("Open now", status.get("open_positions", 0))

left, right = st.columns([2, 1])
with left:
    st.subheader("Equity curve")
    if len(closed):
        eq = closed.set_index("exit_time")["equity_after"]
        st.line_chart(eq)
    st.subheader("Monthly return %")
    mon = pd.Series(status.get("monthly", {}))
    if len(mon):
        st.bar_chart(mon)
with right:
    st.subheader("Open positions")
    op = led[led["open"]][["asset", "side", "entry_time"]]
    st.dataframe(op, hide_index=True, use_container_width=True)
    st.subheader("Win / loss")
    if len(closed):
        wl = closed["win"].value_counts().rename({1: "win", 0: "loss"})
        st.bar_chart(wl)

st.subheader("Recent trades")
recent = closed.sort_values("exit_time", ascending=False).head(40)[
    ["exit_time", "asset", "side", "outcome", "R_net", "pnl", "equity_after"]]


def _color(v):
    return f"color: {'#16a34a' if v > 0 else '#dc2626'}"


st.dataframe(recent.style.map(_color, subset=["R_net", "pnl"]),
             hide_index=True, use_container_width=True)
st.caption(f"Updated {status.get('updated', '—')} · auto-refresh every 60s")

st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)
