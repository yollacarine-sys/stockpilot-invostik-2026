"""StockPilot - analitik risiko, metrik persediaan, dan rekomendasi replenishment (berbasis aturan)."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from stockpilot_features import RAW_FEATURES, TARGET

LEAD_TIME_FACTORS = {"hari": 1 / 30.0, "minggu": 12 / 52.0, "bulan": 1.0}  # -> bulan
RISK_ORDER = ["Rendah", "Sedang", "Tinggi"]


# ----------------------------------------------------------------- artefak model
class ArtifactError(Exception):
    """Artefak model tidak ditemukan / tidak dapat dimuat / tidak cocok."""


def load_artifacts(models_dir: str = "models") -> dict:
    import joblib
    paths = {k: os.path.join(models_dir, v) for k, v in {
        "model": "model.pkl", "features": "feature_columns.pkl", "meta": "model_metadata.json"}.items()}
    missing = [os.path.basename(p) for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise ArtifactError(
            f"Artefak tidak ditemukan di '{models_dir}/': {', '.join(missing)}. "
            "Jalankan notebook StockPilot_Analytics.ipynb di Colab lalu salin isi ZIP deployment ke folder models/.")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:      # agar pickle dapat mengimpor stockpilot_features
        sys.path.insert(0, here)
    try:
        model = joblib.load(paths["model"])
        features = joblib.load(paths["features"])
        with open(paths["meta"], encoding="utf-8") as fh:
            meta = json.load(fh)
    except ModuleNotFoundError as exc:
        raise ArtifactError(f"Modul pendukung model tidak ditemukan ({exc}). "
                            "Pastikan stockpilot_features.py berada di folder utama aplikasi.")
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(f"Gagal memuat artefak model: {exc}")
    warnings = []
    try:
        import sklearn
        v_train = meta.get("environment", {}).get("scikit_learn")
        if v_train and v_train.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            warnings.append(f"Versi scikit-learn berbeda (training {v_train}, aplikasi {sklearn.__version__}). "
                            "Hasil dapat berbeda atau gagal; samakan versi di requirements.txt.")
    except Exception:  # noqa: BLE001
        pass
    return {"model": model, "features": list(features), "meta": meta, "warnings": warnings}


# ----------------------------------------------------------------- prediksi & kategori
def assign_risk_category(prob, high_thr: float, med_thr: float):
    p = np.asarray(prob, dtype=float)
    return np.where(p >= high_thr, "Tinggi", np.where(p >= med_thr, "Sedang", "Rendah"))


def predict_risk(artifacts: dict, df: pd.DataFrame, high_thr: float | None = None,
                 med_thr: float | None = None) -> pd.DataFrame:
    """Kembalikan df asli + kolom: risk_probability, predicted_class, risk_score, risk_category."""
    feats = artifacts["features"]
    miss = [c for c in feats if c not in df.columns]
    if miss:
        raise ArtifactError(f"Dataset tidak memiliki kolom yang dibutuhkan model: {miss}")
    meta = artifacts["meta"]
    tiers = meta["risk_tiers"]
    high_thr = tiers["high_threshold"] if high_thr is None else high_thr
    med_thr = tiers["medium_threshold"] if med_thr is None else med_thr
    X = df[feats]
    try:
        prob = artifacts["model"].predict_proba(X)[:, 1]
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(f"Prediksi gagal: {exc}")
    out = df.copy()
    out["risk_probability"] = prob                                   # keluaran mentah model (tidak terkalibrasi)
    out["predicted_class"] = (prob >= meta["operating_threshold"]).astype(int)   # kelas pada threshold operasi
    out["risk_score"] = pd.Series(prob, index=out.index).rank(pct=True, method="average") * 100  # peringkat relatif batch
    out["risk_category"] = assign_risk_category(prob, high_thr, med_thr)
    return out


# ----------------------------------------------------------------- metrik persediaan
def inventory_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Metrik turunan sederhana untuk Inventory Overview (semua dihitung dari kolom yang ada)."""
    out = df.copy()
    f3 = pd.to_numeric(out.get("forecast_3_month"), errors="coerce") if "forecast_3_month" in out else None
    s3 = pd.to_numeric(out.get("sales_3_month"), errors="coerce") if "sales_3_month" in out else None
    if f3 is not None or s3 is not None:
        f = (f3 / 3.0) if f3 is not None else pd.Series(0.0, index=out.index)
        s = (s3 / 3.0) if s3 is not None else pd.Series(0.0, index=out.index)
        out["demand_month"] = np.where(f > 0, f, s)
        if "national_inv" in out:
            dm = out["demand_month"]
            out["days_of_cover"] = np.where(dm > 0, out["national_inv"] / dm * 30.0, np.nan)
    if {"national_inv", "in_transit_qty"} <= set(out.columns):
        lb = out["local_bo_qty"] if "local_bo_qty" in out else 0
        out["inv_position"] = out["national_inv"] + out["in_transit_qty"] - lb
    if {"national_inv", "min_bank"} <= set(out.columns):
        out["below_min_bank"] = out["national_inv"] < out["min_bank"]
    return out


# ----------------------------------------------------------------- replenishment (aturan)
@dataclass
class ReplenParams:
    lead_time_unit: str = "minggu"        # ASUMSI: satuan lead_time tidak terdokumentasi di file
    fallback_lead_time: float = 8.0       # dipakai bila lead_time kosong (dalam satuan di atas)
    safety_pct: float = 0.25              # safety stock = persen x demand selama lead time (ASUMSI)
    use_min_bank: bool = True             # safety stock minimal = min_bank
    review_horizon_months: float = 1.0    # cakupan tambahan setelah lead time
    moq: float = 0.0                      # 0 = tidak dipakai
    round_to: float = 1.0
    demand_multiplier: float = 1.0        # untuk skenario what-if
    lead_time_multiplier: float = 1.0


def replenishment_plan(df: pd.DataFrame, p: ReplenParams | None = None) -> pd.DataFrame:
    """Rekomendasi pemeriksaan/replenishment BERBASIS ATURAN (bukan kuantitas optimal)."""
    p = p or ReplenParams()
    need = ["national_inv", "in_transit_qty", "local_bo_qty", "forecast_3_month", "sales_3_month", "lead_time", "min_bank"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Kolom untuk replenishment tidak lengkap: {miss}")
    if p.lead_time_unit not in LEAD_TIME_FACTORS:
        raise ValueError(f"lead_time_unit harus salah satu dari {list(LEAD_TIME_FACTORS)}")
    d = df.copy()
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    f3, s3 = num("forecast_3_month") / 3.0, num("sales_3_month") / 3.0
    demand_m = np.where(f3 > 0, f3, s3) * p.demand_multiplier
    demand_m = np.nan_to_num(demand_m, nan=0.0)
    lt_raw = num("lead_time")
    lt_assumed = lt_raw.isna()
    lt = lt_raw.fillna(p.fallback_lead_time) * p.lead_time_multiplier
    lt_months = lt * LEAD_TIME_FACTORS[p.lead_time_unit]
    ltd = demand_m * lt_months
    ss = p.safety_pct * ltd
    if p.use_min_bank:
        ss = np.maximum(ss, num("min_bank").fillna(0).values)
    rop = ltd + ss
    position = num("national_inv").fillna(0) + num("in_transit_qty").fillna(0) - num("local_bo_qty").fillna(0)
    target = rop + demand_m * p.review_horizon_months
    below = position < rop
    raw_qty = np.where(below, np.maximum(target - position, 0), 0.0)
    qty = raw_qty.copy()
    if p.moq > 0:
        qty = np.where(qty > 0, np.maximum(qty, p.moq), 0)
    if p.round_to and p.round_to > 1:
        qty = np.ceil(qty / p.round_to) * p.round_to
    else:
        qty = np.ceil(qty)
    d["demand_month_est"] = demand_m
    d["lead_time_used"] = lt
    d["lead_time_assumed"] = lt_assumed.values
    d["lead_time_demand"] = ltd
    d["safety_stock"] = np.asarray(ss, dtype=float)
    d["reorder_point"] = np.asarray(rop, dtype=float)
    d["inventory_position"] = position.values
    d["below_reorder_point"] = np.asarray(below)
    d["shortfall_to_rop"] = np.maximum(np.asarray(rop) - position.values, 0)
    d["suggested_order_qty"] = qty

    has_risk = "risk_category" in d.columns
    cat = d["risk_category"] if has_risk else pd.Series("N/A", index=d.index)
    actions, reasons = [], []
    for i in range(len(d)):
        c, b = cat.iloc[i], bool(below[i] if not hasattr(below, "iloc") else below.iloc[i])
        dm = demand_m[i]
        if c == "Tinggi" and b:
            a = "P1 - Periksa & pesan segera"
        elif c == "Tinggi":
            a = "P2 - Periksa pemasok/data (risiko model tinggi, stok di atas ROP)"
        elif b and dm > 0:
            a = "P3 - Pertimbangkan replenishment"
        elif c == "Sedang":
            a = "P4 - Pantau"
        else:
            a = "Tidak ada tindakan"
        actions.append(a)
        parts = [f"posisi stok {position.iloc[i]:,.0f} {'<' if b else '>='} ROP {rop[i]:,.0f}"]
        if has_risk:
            parts.append(f"risiko model {c} (p={d['risk_probability'].iloc[i]:.2f})" if "risk_probability" in d else f"risiko {c}")
        if dm <= 0:
            parts.append("tidak ada demand terdeteksi")
        if lt_assumed.iloc[i]:
            parts.append(f"lead time diasumsikan {p.fallback_lead_time:g} {p.lead_time_unit}")
        reasons.append("; ".join(parts))
    d["action"] = actions
    d["reason"] = reasons
    return d


REPLEN_FORMULA = (
    "demand_bulanan = forecast_3_month/3 (jika 0, pakai sales_3_month/3)\n"
    "lead_time_demand = demand_bulanan x lead_time (dikonversi ke bulan)\n"
    "safety_stock = max(safety_pct x lead_time_demand, min_bank)\n"
    "ROP = lead_time_demand + safety_stock\n"
    "posisi_stok = national_inv + in_transit_qty - local_bo_qty\n"
    "jika posisi_stok < ROP: qty_indikatif = ROP + demand_bulanan x horizon - posisi_stok (dibulatkan/MOQ)"
)


# ----------------------------------------------------------------- evaluasi & anggaran pemeriksaan
def threshold_table(y_true, prob, thresholds=None) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"threshold": float(t), "flagged": int(pred.sum()), "flagged_%": 100 * pred.mean(),
                     "precision": prec, "recall": rec, "f1": f1, "TP": tp, "FP": fp, "FN": fn, "TN": tn})
    return pd.DataFrame(rows)


def capture_curve(y_true, score, n_points: int = 20) -> pd.DataFrame:
    """Berapa % backorder aktual tertangkap jika hanya X% SKU teratas (berdasarkan skor) diperiksa."""
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    ys = y[order]
    total_pos = max(int(y.sum()), 1)
    rows = []
    for frac in np.linspace(0.01, 1.0, 100):
        k = max(int(round(frac * len(y))), 1)
        cap = ys[:k].sum() / total_pos
        rows.append({"review_fraction": frac, "review_%": 100 * frac, "captured_%": 100 * cap,
                     "random_baseline_%": 100 * frac, "lift": cap / frac if frac else np.nan})
    return pd.DataFrame(rows)
