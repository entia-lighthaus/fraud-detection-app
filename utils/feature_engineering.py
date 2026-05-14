# utils/feature_engineering.py
# ─────────────────────────────────────────────────────────────
# Replicates the feature engineering pipeline from training.
# Called by the Streamlit app to transform raw transaction
# inputs into the 35-feature vector the model expects.
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


def build_features(transaction: dict) -> pd.DataFrame:
    """
    Takes a single transaction as a dictionary and returns
    a one-row DataFrame with all 35 engineered features
    (anomaly_score is added separately in app.py).

    Parameters
    ----------
    transaction : dict
        Keys expected:
          amount, hour, dayofweek, cross_border,
          ip_country_match, card_present, is_emulator,
          is_unknown_browser, is_new_device,
          device_shared_users, user_device_count,
          txn_count_1h, txn_count_24h, sum_24h,
          avg_amount_7d, time_since_last_txn,
          amount_change_pct, is_rapid_succession,
          is_first_txn, merchant_category,
          ip_user_count, merchant_txn_density_24h,
          user_merchant_diversity,
          device_fraud_neighbourhood

    Returns
    -------
    pd.DataFrame : one row, 35 columns
    """

    t = transaction  # shorthand

    # ── Amount features ───────────────────────────────────────
    amount      = float(t["amount"])
    amount_log  = np.log1p(amount)

    # ── Time features ─────────────────────────────────────────
    hour        = int(t["hour"])
    dayofweek   = int(t["dayofweek"])
    is_weekend  = int(dayofweek >= 5)
    is_night    = int(hour >= 22 or hour <= 5)

    # ── Identity features ─────────────────────────────────────
    cross_border     = int(t["cross_border"])
    ip_country_match = int(t["ip_country_match"])
    card_not_present = int(not bool(t["card_present"]))

    # ── Device fingerprinting ─────────────────────────────────
    is_emulator            = int(t["is_emulator"])
    is_unknown_browser     = int(t["is_unknown_browser"])
    is_new_device          = int(t["is_new_device"])
    device_shared_users    = int(t["device_shared_users"])
    user_device_count      = int(t["user_device_count"])

    # ── Velocity features ─────────────────────────────────────
    txn_count_1h  = int(t["txn_count_1h"])
    txn_count_24h = int(t["txn_count_24h"])
    sum_24h       = float(t["sum_24h"])
    avg_amount_7d = float(t["avg_amount_7d"])

    # ── Sequence / behavioural ────────────────────────────────
    time_since_last_txn = float(t.get("time_since_last_txn", 86400))
    amount_change_pct   = float(t.get("amount_change_pct", 0))
    is_rapid_succession = int(t.get("is_rapid_succession", 0))
    is_first_txn        = int(t.get("is_first_txn", 0))
    sudden_large_spend  = int(amount_change_pct > 5 and not is_first_txn)

    # ── Graph features ────────────────────────────────────────
    ip_user_count               = int(t["ip_user_count"])
    merchant_txn_density_24h    = float(t["merchant_txn_density_24h"])
    user_merchant_diversity     = int(t["user_merchant_diversity"])
    device_fraud_neighbourhood  = float(t["device_fraud_neighbourhood"])

    # ── Ratio features ────────────────────────────────────────
    amount_vs_avg7d = amount / (avg_amount_7d + 1)

    # ── Merchant category one-hot ─────────────────────────────
    all_categories = ["atm", "dining", "gas", "grocery",
                      "online", "retail", "travel"]
    merchant       = t.get("merchant_category", "online")
    cat_dummies    = {f"cat_{c}": int(c == merchant)
                     for c in all_categories}

    # ── Assemble into ordered dict ────────────────────────────
    features = {
        "amount":                    amount,
        "amount_log":                amount_log,
        "hour":                      hour,
        "dayofweek":                 dayofweek,
        "is_weekend":                is_weekend,
        "is_night":                  is_night,
        "cross_border":              cross_border,
        "ip_country_match":          ip_country_match,
        "card_not_present":          card_not_present,
        "is_emulator":               is_emulator,
        "is_unknown_browser":        is_unknown_browser,
        "is_new_device":             is_new_device,
        "device_shared_users":       device_shared_users,
        "user_device_count":         user_device_count,
        "txn_count_1h":              txn_count_1h,
        "txn_count_24h":             txn_count_24h,
        "sum_24h":                   sum_24h,
        "avg_amount_7d":             avg_amount_7d,
        "time_since_last_txn":       time_since_last_txn,
        "amount_change_pct":         amount_change_pct,
        "is_rapid_succession":       is_rapid_succession,
        "is_first_txn":              is_first_txn,
        "sudden_large_spend":        sudden_large_spend,
        "ip_user_count":             ip_user_count,
        "merchant_txn_density_24h":  merchant_txn_density_24h,
        "user_merchant_diversity":   user_merchant_diversity,
        "device_fraud_neighbourhood":device_fraud_neighbourhood,
        "amount_vs_avg7d":           amount_vs_avg7d,
        **cat_dummies,
    }

    return pd.DataFrame([features])