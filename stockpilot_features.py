"""
StockPilot - feature engineering bersama (Colab + Streamlit).

Modul ini WAJIB tersedia di sys.path saat memuat models/model.pkl, karena
pipeline menyimpan referensi ke fungsi `engineer_features` dan `signed_log1p`.
"""
import numpy as np
import pandas as pd

TARGET = "went_on_backorder"
SPLIT_COLUMN = "Validation"      # flag split bawaan dataset, BUKAN fitur
ID_COLUMNS = ["sku"]             # identifier, BUKAN fitur
PERF_SENTINEL = -99              # nilai -99 pada perf_* diperlakukan sebagai "tidak ada data"
CAP_COVER_MONTHS = 60.0          # batas atas cakupan stok bila tidak ada demand

RAW_FEATURES = [
    "national_inv", "lead_time", "in_transit_qty",
    "forecast_3_month", "forecast_6_month", "forecast_9_month",
    "sales_1_month", "sales_3_month", "sales_6_month", "sales_9_month",
    "min_bank", "potential_issue", "pieces_past_due",
    "perf_6_month_avg", "perf_12_month_avg", "local_bo_qty",
    "deck_risk", "oe_constraint", "ppap_risk", "stop_auto_buy", "rev_stop",
]
RAW_BINARY = ["potential_issue", "deck_risk", "oe_constraint", "ppap_risk",
              "stop_auto_buy", "rev_stop"]

ENGINEERED_CONTINUOUS = [
    "national_inv", "lead_time", "in_transit_qty",
    "forecast_3_month", "forecast_6_month", "forecast_9_month",
    "sales_1_month", "sales_3_month", "sales_6_month", "sales_9_month",
    "min_bank", "pieces_past_due", "perf_6_month_avg", "perf_12_month_avg",
    "local_bo_qty",
    # turunan
    "inv_position", "demand_month", "inv_cover_months", "sales_trend_1v3",
    "forecast_minus_sales_3m", "inv_position_minus_forecast_3m",
    "transit_to_forecast_ratio",
]
ENGINEERED_BINARY = RAW_BINARY + [
    "lead_time_missing", "perf_6_missing", "perf_12_missing",
    "negative_inv", "below_min_bank", "no_demand",
]
ENGINEERED_ALL = ENGINEERED_CONTINUOUS + ENGINEERED_BINARY


def engineer_features(X):
    """Terima DataFrame berisi RAW_FEATURES (urutan kolom bebas), kembalikan fitur turunan."""
    df = pd.DataFrame(X).copy()
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    df = df[RAW_FEATURES].apply(pd.to_numeric, errors="coerce").astype("float64")

    # sentinel -99 -> NaN + indikator missing
    for c, flag in (("perf_6_month_avg", "perf_6_missing"),
                    ("perf_12_month_avg", "perf_12_missing")):
        is_sent = df[c].eq(PERF_SENTINEL) | df[c].isna()
        df[flag] = is_sent.astype(float)
        df.loc[is_sent, c] = np.nan
    df["lead_time_missing"] = df["lead_time"].isna().astype(float)

    # demand bulanan: forecast 3 bulan; bila nol gunakan penjualan 3 bulan
    dm_f = df["forecast_3_month"] / 3.0
    dm_s = df["sales_3_month"] / 3.0
    df["demand_month"] = np.where(dm_f > 0, dm_f, dm_s)
    df["no_demand"] = (df["demand_month"] <= 0).astype(float)

    df["inv_position"] = df["national_inv"] + df["in_transit_qty"] - df["local_bo_qty"]
    cover = np.where(df["demand_month"] > 0,
                     df["national_inv"] / df["demand_month"].where(df["demand_month"] > 0, 1.0),
                     CAP_COVER_MONTHS)
    df["inv_cover_months"] = np.clip(cover, -12.0, CAP_COVER_MONTHS)
    df["sales_trend_1v3"] = df["sales_1_month"] - df["sales_3_month"] / 3.0
    df["forecast_minus_sales_3m"] = df["forecast_3_month"] - df["sales_3_month"]
    df["inv_position_minus_forecast_3m"] = df["inv_position"] - df["forecast_3_month"]
    df["transit_to_forecast_ratio"] = df["in_transit_qty"] / (df["forecast_3_month"] + 1.0)
    df["negative_inv"] = (df["national_inv"] < 0).astype(float)
    df["below_min_bank"] = (df["national_inv"] < df["min_bank"]).astype(float)

    return df[ENGINEERED_ALL]


def signed_log1p(X):
    """Transformasi log yang menjaga tanda (untuk variabel berekor panjang & bernilai negatif)."""
    X = np.asarray(X, dtype="float64")
    return np.sign(X) * np.log1p(np.abs(X))
