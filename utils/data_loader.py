"""StockPilot - pemuatan, standardisasi, validasi, dan audit kualitas data."""
from __future__ import annotations

import io
import os
import re

import numpy as np
import pandas as pd

from stockpilot_features import (
    PERF_SENTINEL, RAW_BINARY, RAW_FEATURES, SPLIT_COLUMN, TARGET,
)

SUPPORTED_EXT = (".csv", ".xlsx", ".xls")
MAX_FILE_MB = 200

# Kolom wajib per modul (dipakai pada halaman Data Quality & pesan keterbatasan)
MODULE_REQUIREMENTS = {
    "Inventory Overview": ["national_inv", "forecast_3_month", "lead_time", "in_transit_qty"],
    "Backorder Risk Monitor": RAW_FEATURES,
    "Replenishment Recommendation": ["national_inv", "in_transit_qty", "local_bo_qty",
                                     "forecast_3_month", "sales_3_month", "lead_time", "min_bank"],
    "What-if & Financial Simulator": ["national_inv", "in_transit_qty", "local_bo_qty",
                                      "forecast_3_month", "sales_3_month", "lead_time", "min_bank"],
    "Smart Slotting Simulator": ["(file terpisah) sku, current_location, required_volume, pick_frequency; "
                                 "locations: location_id, distance_to_dock/x,y, capacity_volume"],
}

_YES = {"yes", "y", "true", "t", "1", "ya", "1.0"}
_NO = {"no", "n", "false", "f", "0", "tidak", "0.0"}


class DataLoadError(Exception):
    """Error pemuatan data dengan pesan yang ramah pengguna."""


def _canon(name: str) -> str:
    return re.sub(r"[\s\-\.]+", "_", str(name).strip()).lower()


def read_table(source, filename: str | None = None, sheet_name=0) -> pd.DataFrame:
    """Baca CSV/XLSX/XLS dari path, file-like (Streamlit upload), atau bytes."""
    name = filename or getattr(source, "name", None) or (source if isinstance(source, str) else "")
    ext = os.path.splitext(str(name))[1].lower()
    if ext not in SUPPORTED_EXT:
        raise DataLoadError(f"Format '{ext or 'tanpa ekstensi'}' tidak didukung. Gunakan: {', '.join(SUPPORTED_EXT)}.")
    if hasattr(source, "getvalue"):
        raw = source.getvalue()
    elif hasattr(source, "read"):
        raw = source.read()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        if not os.path.exists(source):
            raise DataLoadError(f"File tidak ditemukan: {source}")
        with open(source, "rb") as fh:
            raw = fh.read()
    if len(raw) == 0:
        raise DataLoadError("File kosong (0 byte).")
    if len(raw) > MAX_FILE_MB * 1024 * 1024:
        raise DataLoadError(f"File lebih besar dari {MAX_FILE_MB} MB.")
    try:
        if ext == ".csv":
            df = None
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                    if df.shape[1] == 1:   # kemungkinan delimiter bukan koma
                        df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise DataLoadError("Encoding file tidak dapat dibaca (coba simpan ulang sebagai UTF-8).")
        else:
            df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name)
    except pd.errors.EmptyDataError:
        raise DataLoadError("File tidak berisi data yang dapat dibaca.")
    except DataLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Gagal membaca file: {exc}")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise DataLoadError("Dataset kosong: tidak ada baris atau kolom.")
    return df


def excel_sheet_names(source) -> list[str]:
    raw = source.getvalue() if hasattr(source, "getvalue") else source.read()
    return pd.ExcelFile(io.BytesIO(raw)).sheet_names


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Samakan nama kolom (spasi/kapital) dengan nama kanonik StockPilot."""
    canon_map = {_canon(c): c for c in RAW_FEATURES + [TARGET, SPLIT_COLUMN, "sku"]}
    rename = {}
    for c in df.columns:
        k = _canon(c)
        if k in canon_map and c != canon_map[k]:
            rename[c] = canon_map[k]
    out = df.rename(columns=rename).copy()
    if out.columns.duplicated().any():
        raise DataLoadError("Terdapat nama kolom ganda setelah standardisasi: "
                            f"{out.columns[out.columns.duplicated()].tolist()}")
    return out


def standardize_binary(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Ubah Yes/No/True/False pada kolom flag & target menjadi 0/1. Return (df, catatan)."""
    out = df.copy()
    notes = []
    for c in RAW_BINARY + [TARGET]:
        if c in out.columns and (out[c].dtype == object or str(out[c].dtype).startswith(("str", "string"))):
            s = out[c].astype(str).str.strip().str.lower()
            mapped = s.map(lambda v: 1 if v in _YES else (0 if v in _NO else np.nan))
            unknown = int(mapped.isna().sum() - out[c].isna().sum())
            out[c] = mapped
            notes.append(f"Kolom '{c}' dikonversi dari teks Yes/No ke 0/1"
                         + (f" ({unknown} nilai tidak dikenali -> kosong)" if unknown > 0 else ""))
    return out, notes


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Paksa kolom numerik; kembalikan jumlah nilai non-numerik yang menjadi NaN."""
    out = df.copy()
    bad = {}
    for c in cols:
        if c in out.columns:
            before = out[c].isna().sum()
            out[c] = pd.to_numeric(out[c], errors="coerce")
            n_bad = int(out[c].isna().sum() - before)
            if n_bad > 0:
                bad[c] = n_bad
    return out, bad


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pipa standar: nama kolom -> biner -> numerik. Return (df_siap, catatan)."""
    notes = []
    d = standardize_columns(df)
    d, n1 = standardize_binary(d)
    notes += n1
    num_cols = [c for c in RAW_FEATURES if c in d.columns] + [c for c in (TARGET,) if c in d.columns]
    d, bad = coerce_numeric(d, num_cols)
    for c, n in bad.items():
        notes.append(f"Kolom '{c}': {n} nilai non-numerik diubah menjadi kosong.")
    return d, notes


def classify_dataset(df: pd.DataFrame) -> str:
    """'labeled' | 'unlabeled' | 'submission_format' | 'insufficient'."""
    cols = set(df.columns)
    has_feats = len(set(RAW_FEATURES) & cols)
    if cols <= {"sku", TARGET} and TARGET in cols:
        return "submission_format"
    if has_feats < 3:
        return "insufficient"
    missing = [c for c in RAW_FEATURES if c not in cols]
    if TARGET in cols and not missing:
        return "labeled"
    if not missing:
        return "unlabeled"
    return "insufficient"


def validate_for_model(df: pd.DataFrame) -> dict:
    cols = set(df.columns)
    missing = [c for c in RAW_FEATURES if c not in cols]
    ignored = [c for c in df.columns if c not in RAW_FEATURES and c not in ("sku", TARGET, SPLIT_COLUMN)]
    return {"ok": not missing, "missing_columns": missing, "unused_columns": ignored,
            "has_target": TARGET in cols, "has_sku": "sku" in cols}


def module_availability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = set(df.columns)
    for mod, req in MODULE_REQUIREMENTS.items():
        if mod.startswith("Smart Slotting"):
            rows.append({"Modul": mod, "Kolom wajib": "Data lokasi terpisah (lihat halaman Slotting)",
                         "Kolom hilang": "-", "Status": "Butuh file terpisah / simulasi sintetis"})
            continue
        miss = [c for c in req if c not in cols]
        rows.append({"Modul": mod, "Kolom wajib": ", ".join(req),
                     "Kolom hilang": ", ".join(miss) if miss else "-",
                     "Status": "Siap" if not miss else "Tidak lengkap"})
    return pd.DataFrame(rows)


def data_quality_report(df: pd.DataFrame) -> dict:
    """Audit kualitas data. Semua angka dihitung dari data yang diberikan."""
    n = len(df)
    col_rows = []
    for c in df.columns:
        s = df[c]
        n_missing = int(s.isna().sum())
        col_rows.append({
            "kolom": c, "tipe": str(s.dtype), "missing": n_missing,
            "missing_%": round(100 * n_missing / max(n, 1), 3),
            "n_unik": int(s.nunique(dropna=True)),
            "konstan": bool(s.nunique(dropna=True) <= 1),
        })
    col_table = pd.DataFrame(col_rows)

    issues = []

    def add(check, count, severity, note):
        issues.append({"pemeriksaan": check, "jumlah": int(count), "tingkat": severity, "catatan": note})

    add("Baris duplikat penuh", df.duplicated().sum(), "warning" if df.duplicated().sum() else "ok",
        "Baris identik di semua kolom.")
    if "sku" in df.columns:
        add("SKU duplikat", df["sku"].duplicated().sum(), "warning" if df["sku"].duplicated().sum() else "ok",
            "Satu SKU muncul lebih dari sekali; pastikan 1 baris = 1 SKU pada snapshot yang sama.")
    const_cols = col_table.loc[col_table["konstan"], "kolom"].tolist()
    add("Kolom konstan", len(const_cols), "warning" if const_cols else "ok",
        ", ".join(const_cols) if const_cols else "Tidak ada.")
    n_missing_cells = int(df.isna().sum().sum())
    add("Total sel kosong", n_missing_cells, "warning" if n_missing_cells else "ok",
        "Rincian per kolom pada tabel kolom.")

    for c in ("perf_6_month_avg", "perf_12_month_avg"):
        if c in df.columns:
            k = int((pd.to_numeric(df[c], errors="coerce") == PERF_SENTINEL).sum())
            add(f"Sentinel {PERF_SENTINEL} pada {c}", k, "info" if k else "ok",
                "Nilai -99 diperlakukan sebagai 'tidak ada data' (bukan kinerja sebenarnya).")
            v = pd.to_numeric(df[c], errors="coerce")
            k2 = int(((v < 0) | (v > 1)).sum() - (v == PERF_SENTINEL).sum())
            add(f"{c} di luar rentang 0-1 (selain sentinel)", k2, "warning" if k2 else "ok",
                "Asumsi: skor kinerja berskala 0-1 (definisi kolom belum didokumentasikan).")
    if "national_inv" in df.columns:
        k = int((pd.to_numeric(df["national_inv"], errors="coerce") < 0).sum())
        add("Stok nasional negatif", k, "info" if k else "ok",
            "Dapat berarti stok kurang/backlog atau kesalahan pencatatan; perlu konfirmasi.")
    nonneg = ["lead_time", "in_transit_qty", "forecast_3_month", "forecast_6_month", "forecast_9_month",
              "sales_1_month", "sales_3_month", "sales_6_month", "sales_9_month", "min_bank",
              "pieces_past_due", "local_bo_qty"]
    for c in nonneg:
        if c in df.columns:
            k = int((pd.to_numeric(df[c], errors="coerce") < 0).sum())
            if k:
                add(f"Nilai negatif pada {c}", k, "warning", "Kolom ini diharapkan tidak negatif.")
    if {"sales_1_month", "sales_3_month"} <= set(df.columns):
        k = int((df["sales_1_month"] > df["sales_3_month"]).sum())
        add("sales_1_month > sales_3_month", k, "info" if k else "ok",
            "Jika kolom bersifat kumulatif, nilai ini janggal; definisi jendela waktu belum terdokumentasi.")
    for c in ("national_inv", "forecast_3_month", "sales_1_month", "in_transit_qty"):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v) > 10:
                q1, q3 = v.quantile([0.25, 0.75])
                iqr = q3 - q1
                k = int((v > q3 + 10 * iqr).sum()) if iqr > 0 else int((v > v.quantile(0.999)).sum())
                add(f"Nilai ekstrem tinggi pada {c}", k, "info" if k else "ok",
                    "Melebihi Q3 + 10xIQR; periksa apakah SKU bervolume sangat besar atau kesalahan entri.")
    return {"n_rows": n, "n_cols": df.shape[1], "column_table": col_table,
            "issues": pd.DataFrame(issues),
            "n_duplicates": int(df.duplicated().sum()),
            "missing_cells": n_missing_cells,
            "missing_pct": round(100 * n_missing_cells / max(df.size, 1), 3)}


def quality_report_to_csv(rep: dict) -> bytes:
    """Gabungkan ringkasan, isu, dan tabel kolom ke satu CSV."""
    buf = io.StringIO()
    buf.write(f"# StockPilot Data Quality Report\n# baris,{rep['n_rows']}\n# kolom,{rep['n_cols']}\n")
    buf.write(f"# sel_kosong,{rep['missing_cells']}\n# duplikat,{rep['n_duplicates']}\n\n")
    buf.write("## PEMERIKSAAN\n")
    rep["issues"].to_csv(buf, index=False)
    buf.write("\n## KOLOM\n")
    rep["column_table"].to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
