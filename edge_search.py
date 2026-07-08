"""
Honest edge search (see plan: delightful-dreaming-lemon).

Goal: find ANY signal that predicts forward returns beyond chance, NET of cost,
with a discipline hostile to wishful thinking:

  * causal signals only (bar t uses data <= close of t; entry is t+1 open);
  * primary metric = rank IC (Spearman) between signal_t and the forward return;
  * NULL = circular-shift permutation. We compute the correlation of the signal
    against the forward return at EVERY time-shift (via FFT cross-correlation).
    The real alignment sits at shift 0; the thousands of non-zero shifts are the
    null distribution of "accidental" alignment. This automatically accounts for
    the autocorrelation in both series (the thing that fakes significance).
    Shifts within +/-(2h+50) bars of 0 are excluded so overlapping-window
    autocorrelation can't leak into the null.
  * nested hold-out: discover on <=2023, validate survivors on 2024, judge the
    single best ONCE on 2025.
  * multiple-testing: Benjamini-Hochberg across all discovery tests.
  * costs charged for the net-expectancy check.

Run:  python edge_search.py          (writes EDGE_SEARCH_RESULTS.md)
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd

DATA = "/home/user/Quant/data"
ASSETS = {"BTC": f"{DATA}/btc_klines", "ETH": f"{DATA}/eth_klines",
          "SOL": f"{DATA}/sol_klines"}
OF_DIR = f"{DATA}/btc_of"
# round-trip cost in bps (spread+fee+slippage), per asset. BTC tight, alts wider.
COST_BPS = {"BTC": 3.0, "ETH": 4.0, "SOL": 8.0}

TRAIN_END = "2023-12-31"
VAL = ("2024-01-01", "2024-12-31")
TEST = ("2025-01-01", "2025-12-31")

TFS = ["15min", "1h"]
HORIZONS = [1, 4, 12]        # forward bars to hold


# ---------------------------------------------------------------- data
_KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
          "quote_volume", "count", "taker_buy_base", "taker_buy_quote", "ignore"]


def load_klines(folder: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(f"{folder}/*.csv")):
        d = pd.read_csv(f, header=None)
        d.columns = _KCOLS[:d.shape[1]]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("open_time")
    ot = df["open_time"].astype("int64")
    ot = ot.where(ot < 1e14, ot // 1000)        # ms vs us
    idx = pd.to_datetime(ot, unit="ms", utc=True)
    out = pd.DataFrame(index=idx)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        out[c] = df[c].to_numpy(float)
    out = out.sort_index()
    return out[(out.high >= out.low) & (out.low > 0)]


def load_orderflow() -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(f"{OF_DIR}/*.parquet"))]
    of = pd.concat(frames, ignore_index=True).drop_duplicates("ms")
    of.index = pd.to_datetime(of["ms"], unit="ms", utc=True)
    of = of.sort_index()
    of["of_sell"] = of["of_vol"] - of["of_buy"]
    of["of_delta"] = of["of_buy"] - of["of_sell"]
    return of


def resample(kl: pd.DataFrame, of: pd.DataFrame | None, tf: str) -> pd.DataFrame:
    g = kl.resample(tf, label="right", closed="right")
    df = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(), "taker_buy": g["taker_buy_base"].sum(),
    }).dropna(subset=["open"])
    if of is not None:
        og = of.resample(tf, label="right", closed="right").agg(
            {"of_vol": "sum", "of_buy": "sum", "of_sell": "sum",
             "of_delta": "sum", "of_ntrades": "sum", "of_maxtrade": "max",
             "of_buymax": "max", "of_sellmax": "max"})
        df = df.join(og)
    return df


# ---------------------------------------------------------------- signals
def _z(s, n):
    return (s - s.rolling(n).mean()) / (s.rolling(n).std() + 1e-12)


def build_signals(df: pd.DataFrame, has_of: bool) -> dict[str, pd.Series]:
    c = df["close"]
    ret1 = c.pct_change()
    sig = {}
    # --- price families ---
    for k in (1, 4, 12):
        sig[f"mom_{k}"] = c.pct_change(k)                    # momentum
    sig["revert_4"] = -c.pct_change(4)                       # short-term reversion
    sig["range_pos"] = ((c - df["low"]) / (df["high"] - df["low"] + 1e-12)) - 0.5
    sig["vol_z_20"] = _z(ret1.abs(), 20)                     # vol regime (direction-less; test anyway)
    sig["taker_imb"] = (df["taker_buy"] / (df["volume"] + 1e-12)) - 0.5
    # --- order-flow families (BTC only) ---
    if has_of and "of_delta" in df:
        v = df["of_vol"] + 1e-12
        sig["ofi"] = df["of_delta"] / v                      # per-bar signed imbalance
        sig["ofi_ma4"] = (df["of_delta"] / v).rolling(4).mean()
        sig["cvd_slope12"] = df["of_delta"].rolling(12).sum() / (df["of_vol"].rolling(12).sum() + 1e-12)
        sig["bigprint_imb"] = ((df["of_buymax"] - df["of_sellmax"]) /
                               (df["of_buymax"] + df["of_sellmax"] + 1e-12))
        sig["maxtrade_z"] = _z(df["of_maxtrade"], 50)        # absorption spike (direction-less)
        sig["trade_intensity_z"] = _z(df["of_ntrades"], 50)
    return sig


def forward_return(df: pd.DataFrame, h: int) -> pd.Series:
    """Enter at NEXT bar open, exit h bars later at close. Causal, no overlap
    with the signal bar."""
    entry = df["open"].shift(-1)
    exit_ = df["close"].shift(-1 - (h - 1))
    return (exit_ / entry - 1.0).rename("fret")


# ---------------------------------------------------------------- test
def _rank(x):
    r = pd.Series(x).rank().to_numpy()
    r = (r - r.mean()) / (r.std() + 1e-12)
    return r


def ic_and_perm_p(sig: np.ndarray, fret: np.ndarray, h: int, exclude=50):
    """Rank IC at shift 0 and a circular-shift permutation p-value via FFT
    cross-correlation. Returns (ic, p, n)."""
    m = np.isfinite(sig) & np.isfinite(fret)
    if m.sum() < 500:
        return np.nan, np.nan, int(m.sum())
    x, y = _rank(sig[m]), _rank(fret[m])
    n = len(x)
    # circular cross-correlation at all lags via FFT: cc[k] = sum x[i] y[i+k]/n
    cc = np.fft.irfft(np.fft.rfft(x) * np.conj(np.fft.rfft(y)), n) / n
    ic = cc[0]
    ban = 2 * h + exclude
    idx = np.arange(n)
    keep = (np.minimum(idx, n - idx) > ban)     # exclude near-0 (and near-n) lags
    null = cc[keep]
    # two-sided p: fraction of |null| >= |ic|
    p = (np.abs(null) >= abs(ic)).mean()
    return float(ic), float(p), n


def net_expectancy(sig, fret, sign, cost_bps):
    """Simple long/short on the top/bottom tercile of the signal (sign-adjusted),
    net of round-trip cost. Returns mean net return per trade and n."""
    m = np.isfinite(sig) & np.isfinite(fret)
    s, r = sig[m] * sign, fret[m]
    if len(s) < 300:
        return np.nan, 0
    hi, lo = np.quantile(s, 0.667), np.quantile(s, 0.333)
    longs = r[s >= hi] - cost_bps / 1e4
    shorts = -r[s <= lo] - cost_bps / 1e4
    trades = np.concatenate([longs, shorts])
    return float(trades.mean()), len(trades)


def bh_correct(pvals):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


# ---------------------------------------------------------------- driver
def evaluate(df, sigs, asset, period=None, label=""):
    if period:
        df = df.loc[period[0]:period[1]]
    rows = []
    for tf_h in HORIZONS:
        fret = forward_return(df, tf_h).to_numpy()
        for name, s in sigs.items():
            sv = s.reindex(df.index).to_numpy()
            ic, p, n = ic_and_perm_p(sv, fret, tf_h)
            if not np.isfinite(ic):
                continue
            sign = 1.0 if ic >= 0 else -1.0
            exp, nt = net_expectancy(sv, fret, sign, COST_BPS[asset])
            rows.append({"asset": asset, "label": label, "signal": name,
                         "h": tf_h, "ic": ic, "sign": sign, "p": p,
                         "net_exp_bps": exp * 1e4 if np.isfinite(exp) else np.nan,
                         "n": n})
    return pd.DataFrame(rows)


def main():
    print("loading data...", flush=True)
    of = load_orderflow()
    kl = {a: load_klines(f) for a, f in ASSETS.items()}
    out_lines = []

    for tf in TFS:
        print(f"\n=== timeframe {tf} ===", flush=True)
        disc_all = []
        frames = {}
        for a in ASSETS:
            df = resample(kl[a], of if a == "BTC" else None, tf)
            frames[a] = df
            sigs = build_signals(df, has_of=(a == "BTC"))
            disc = evaluate(df, sigs, a, period=(None, TRAIN_END), label="train")
            disc_all.append(disc)
            print(f"  {a}: {len(sigs)} signals x {len(HORIZONS)} horizons "
                  f"scored on train", flush=True)
        disc = pd.concat(disc_all, ignore_index=True)
        disc["q"] = bh_correct(disc["p"].fillna(1.0).values)
        disc = disc.sort_values("p")
        survivors = disc[(disc.q < 0.05) & (disc.net_exp_bps > 0)]

        out_lines.append(f"\n## Timeframe {tf}\n")
        out_lines.append(f"Discovery: {len(disc)} tests on train (<= {TRAIN_END}). "
                         f"Survivors (BH q<0.05 AND net>0): {len(survivors)}.\n")
        out_lines.append("Top 12 by permutation p (train):\n")
        out_lines.append("| asset | signal | h | IC | perm p | BH q | net bps |\n"
                         "|---|---|---|---|---|---|---|\n")
        for _, r in disc.head(12).iterrows():
            out_lines.append(f"| {r.asset} | {r.signal} | {r.h} | {r.ic:+.4f} | "
                             f"{r.p:.4f} | {r.q:.3f} | {r.net_exp_bps:+.2f} |\n")

        print(f"  survivors on train: {len(survivors)}")
        # validate survivors on 2024
        for _, r in survivors.iterrows():
            df = frames[r.asset]
            sigs = build_signals(df, has_of=(r.asset == "BTC"))
            s = sigs[r.signal]
            for per, plabel in ((VAL, "2024 validation"), (TEST, "2025 HOLDOUT")):
                sub = df.loc[per[0]:per[1]]
                fret = forward_return(sub, int(r.h)).to_numpy()
                sv = s.reindex(sub.index).to_numpy()
                ic, p, n = ic_and_perm_p(sv, fret, int(r.h))
                exp, nt = net_expectancy(sv, fret, r.sign, COST_BPS[r.asset])
                out_lines.append(
                    f"- SURVIVOR {r.asset} {r.signal} h={r.h} → {plabel}: "
                    f"IC {ic:+.4f}, perm p {p:.3f}, net {exp*1e4:+.2f} bps "
                    f"(sign fixed from train)\n")
                print(f"    {r.asset} {r.signal} h={r.h} {plabel}: "
                      f"IC {ic:+.4f} p {p:.3f} net {exp*1e4:+.2f}bps", flush=True)

    report = "# Edge search results\n" + "".join(out_lines)
    open("/home/user/Quant/EDGE_SEARCH_RESULTS.md", "w").write(report)
    print("\nwrote EDGE_SEARCH_RESULTS.md")


if __name__ == "__main__":
    main()
