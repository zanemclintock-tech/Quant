#!/usr/bin/env bash
# One command to run the real-bid/ask demo engine on your Mac 24/7.
# Loads secrets from .env (copy .env.example -> .env first), keeps the Mac
# awake while it runs, and restarts the engine automatically if it ever dies.
#
#   chmod +x run_live.sh
#   ./run_live.sh
set -u
cd "$(dirname "$0")"

[ -f .env ] && set -a && . ./.env && set +a

: "${EXCHANGE:=bybit}"          # real market for the live bid/ask stream
export EXCHANGE

echo "[run_live] exchange=$EXCHANGE  cloud=$([ -n "${GIST_ID:-}" ] && echo on || echo off)  telegram=$([ -n "${TELEGRAM_TOKEN:-}" ] && echo on || echo off)"

# caffeinate keeps macOS from sleeping; the loop auto-restarts on any crash.
while true; do
  caffeinate -is python live_stream.py
  echo "[run_live] engine exited ($(date)); restarting in 10s…"
  sleep 10
done
