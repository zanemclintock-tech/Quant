#!/usr/bin/env bash
#
# One-step backtest runner for the NAS100 CFD ORB strategy.
#
# Usage:  put this folder's files together with your two Dukascopy CSVs
#         (the ones with "Bid" and "Ask" in the name), then run:
#
#             bash run.sh
#
# It installs the Python dependencies, runs the engine tests, and runs
# the full backtest, auto-finding your CSV files. No arguments needed.

set -e
cd "$(dirname "$0")"

echo "=================================================="
echo " NAS100 CFD ORB — one-step backtest"
echo "=================================================="

# Pick a Python (prefer python3)
PY="python3"
command -v python3 >/dev/null 2>&1 || PY="python"

echo
echo "[1/3] Installing Python dependencies..."
$PY -m pip install -q -r requirements.txt

echo
echo "[2/3] Running engine self-tests (causality, risk, accounting)..."
$PY -m pytest tests/ -q

echo
echo "[3/3] Running the 5-year backtest..."
$PY run_backtest.py --sensitivity

echo
echo "Done. See the report above, plus output/equity_curve.png and"
echo "output/trades.csv. Paste the report back to Claude for review."
