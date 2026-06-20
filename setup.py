"""
One-time guided setup. Run it once on your Mac:

    python3 setup.py

It asks a few questions, creates the cloud link for your phone dashboard,
and writes your .env file. After this, you only ever run ./run_live.sh.
Nothing here is sent anywhere except: (a) creating your own GitHub gist,
(b) an optional Telegram test message to you.
"""
from __future__ import annotations

import os
import urllib.parse
import urllib.request

import cloud_sync


def ask(prompt, default=""):
    d = f" [{default}]" if default else ""
    v = input(f"{prompt}{d}: ").strip()
    return v or default


def tg_test(token, chat):
    data = urllib.parse.urlencode({
        "chat_id": chat, "text": "✅ Setup works — you'll get trade alerts here."
    }).encode()
    urllib.request.urlopen(urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage", data=data), timeout=10)


def main():
    print("\n=== Crypto demo setup ===\n")
    env = {"EXCHANGE": "kucoin"}

    # 1) exchange
    print("1) Market to stream (KuCoin lists all 5 coins; press Enter to keep).")
    env["EXCHANGE"] = ask("   EXCHANGE", "kucoin")

    # 2) phone dashboard link (GitHub gist)
    print("\n2) Phone dashboard link.")
    print("   Make a GitHub token with ONLY the 'gist' box ticked:")
    print("   https://github.com/settings/tokens  (Generate new token, classic)")
    token = ask("   Paste GitHub token (or Enter to skip phone access)")
    if token:
        try:
            gid = cloud_sync.create_gist(token)
            env["GITHUB_TOKEN"] = token
            env["GIST_ID"] = gid
            print(f"   ✓ Created your dashboard link. GIST_ID = {gid}")
            print("   >>> Paste this into Streamlit later:  GIST_ID = \"%s\"" % gid)
        except Exception as e:
            print(f"   ✗ Couldn't create gist ({e}). Skipping phone access.")

    # 3) Telegram alerts
    print("\n3) Phone notifications (optional).")
    print("   In Telegram: message @BotFather -> /newbot -> copy the token.")
    ttok = ask("   Paste Telegram bot token (or Enter to skip)")
    if ttok:
        print("   Now message your new bot once, then open:")
        print(f"   https://api.telegram.org/bot{ttok}/getUpdates")
        chat = ask("   Paste the chat id (the number after \"id\":)")
        if chat:
            try:
                tg_test(ttok, chat)
                env["TELEGRAM_TOKEN"], env["TELEGRAM_CHAT"] = ttok, chat
                print("   ✓ Sent you a test message — check Telegram.")
            except Exception as e:
                print(f"   ✗ Telegram test failed ({e}). Skipping.")

    # write .env
    with open(".env", "w") as f:
        for k in ("EXCHANGE", "TELEGRAM_TOKEN", "TELEGRAM_CHAT",
                  "GITHUB_TOKEN", "GIST_ID"):
            f.write(f"{k}={env.get(k, '')}\n")
    print("\nSaved .env. You're done here.\n")
    print("Start trading any time with:   ./run_live.sh")
    if env.get("GIST_ID"):
        print("For the phone dashboard, finish the 1-minute Streamlit step in "
              "GO_LIVE.md using the GIST_ID printed above.")


if __name__ == "__main__":
    main()
