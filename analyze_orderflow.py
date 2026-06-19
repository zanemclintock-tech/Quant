"""
Does trade-level order flow actually matter? Permutation importance of
every feature on the OOS set for one config, tick features flagged.

    ORDERFLOW=1 CRYPTO=1 CRYPTO_GLOB=data/btc BASE_TF=60min HORIZON=60 \
        python analyze_orderflow.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from crypto_loader import load_binance_klines
from labeler import label_setups_directional
from smc_detector import detect_setups

NON_FEATURES = {"entry_time", "year", "realized_R", "win", "outcome",
                "entry_px", "risk_px"}
TICK = {"tofi_sweep", "tofi_entry", "tofi_leg", "cvd_slope_leg",
        "maxtrade_z_sweep", "bigprint_imb_sweep"}
HORIZON = int(os.environ.get("HORIZON", "60"))


def main() -> None:
    df = load_binance_klines(os.environ.get("CRYPTO_GLOB", "data/btc"))
    print(f"order-flow attached: {'of_cvd' in df.columns} | "
          f"timeframe {os.environ.get('BASE_TF', '30min')}")
    data = label_setups_directional(df, detect_setups(df), HORIZON,
                                    continuous=True).dropna(
                                        subset=["realized_R"])
    feat = [c for c in data.columns
            if c not in NON_FEATURES and data[c].notna().any()]
    is_ = data[data["year"] <= 2023]
    oos = data[data["year"] >= 2024]
    print(f"features {len(feat)} | IS {len(is_)} OOS {len(oos)}")
    m = HistGradientBoostingClassifier(
        max_depth=3, max_iter=250, learning_rate=0.04, l2_regularization=1.0,
        min_samples_leaf=40, early_stopping=True, validation_fraction=0.2,
        random_state=0).fit(is_[feat], is_["win"])
    auc = roc_auc_score(oos["win"], m.predict_proba(oos[feat])[:, 1])
    print(f"OOS AUC {auc:.3f}\n")
    r = permutation_importance(m, oos[feat], oos["win"], n_repeats=20,
                               random_state=0, scoring="roc_auc")
    order = np.argsort(r.importances_mean)[::-1]
    print(f" {'feature':>22} {'imp(AUC)':>9} {'std':>7}   tick?")
    for i in order:
        tag = "  <== TICK" if feat[i] in TICK else ""
        print(f" {feat[i]:>22} {r.importances_mean[i]:>+9.4f} "
              f"{r.importances_std[i]:>7.4f}{tag}")
    tick_present = [f for f in feat if f in TICK]
    tick_imp = sum(r.importances_mean[feat.index(f)] for f in tick_present)
    print(f"\n tick features present: {len(tick_present)} | "
          f"summed importance {tick_imp:+.4f} "
          f"({'helps' if tick_imp > 0 else 'no net help'})")


if __name__ == "__main__":
    main()
