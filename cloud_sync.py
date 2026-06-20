"""
Tiny bridge that pushes the live runner's output up to a GitHub Gist so a
hosted dashboard (Streamlit Community Cloud) can show it from anywhere --
no tunnel back to the Mac, and the last snapshot survives a wifi blip.

The Mac (which runs live_stream.py) WRITES the gist with a personal token;
the dashboard READS it from the public raw URL (no token needed in the cloud
app, so nothing secret is deployed). Paper-trading P&L isn't sensitive, so a
public gist is fine and keeps the cloud side credential-free.

Env:
  GIST_ID         the gist to update (create once -- see GO_LIVE.md)
  GITHUB_TOKEN    a token with the `gist` scope (Mac side only)

All functions are no-ops if the env isn't set, so local-only runs are
unaffected.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

API = "https://api.github.com/gists"
FILES = ("status.json", "ledger.csv")
_last = {"t": 0.0}


def _request(url, token, method="GET", payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def create_gist(token: str, desc: str = "crypto dashboard live data") -> str:
    """One-time helper: create a PUBLIC gist seeded with empty files and
    return its id. Run `python cloud_sync.py` to use it interactively."""
    body = {"description": desc, "public": True,
            "files": {f: {"content": "{}" if f.endswith(".json") else "x"}
                      for f in FILES}}
    return _request(API, token, "POST", body)["id"]


def push(min_interval: float = 15.0) -> bool:
    """PATCH the gist with the current status.json + ledger.csv. Throttled to
    at most once per `min_interval` s. Returns True if it uploaded."""
    gid, token = os.environ.get("GIST_ID"), os.environ.get("GITHUB_TOKEN")
    if not (gid and token):
        return False
    if time.time() - _last["t"] < min_interval:
        return False
    files = {}
    for f in FILES:
        if os.path.exists(f):
            with open(f) as fh:
                files[f] = {"content": fh.read() or " "}
    if not files:
        return False
    try:
        _request(f"{API}/{gid}", token, "PATCH", {"files": files})
        _last["t"] = time.time()
        return True
    except Exception as e:
        print(f"  cloud_sync error: {e}")
        return False


def raw_url(filename: str, gid: str | None = None) -> str | None:
    """Public raw URL for a gist file (latest revision), cache-busted."""
    gid = gid or os.environ.get("GIST_ID")
    if not gid:
        return None
    # the API gives us the per-file raw_url incl. the owner login
    try:
        meta = _request(f"{API}/{gid}", os.environ.get("GITHUB_TOKEN"))
        return meta["files"][filename]["raw_url"]
    except Exception:
        return None


if __name__ == "__main__":      # interactive: create a gist, print its id
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        raise SystemExit("set GITHUB_TOKEN (with 'gist' scope) first")
    print("GIST_ID=" + create_gist(tok))
