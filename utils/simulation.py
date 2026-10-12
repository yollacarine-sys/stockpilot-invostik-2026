"""StockPilot - Smart slotting (heuristik COI) dan simulasi what-if replenishment.

PENTING: hasil di modul ini adalah SIMULASI. Data sintetis selalu diberi kolom
`data_source = 'SYNTHETIC'`. Slotting memakai heuristik greedy yang dapat dijelaskan,
BUKAN optimasi matematis global.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.analytics import ReplenParams, replenishment_plan

SLOT_SKU_COLS = ["sku", "current_location", "required_volume", "pick_frequency"]
SLOT_LOC_COLS = ["location_id", "capacity_volume"]   # + distance_to_dock ATAU (x, y)
OPTIONAL_SKU = ["storage_class"]                     # STANDARD | HEAVY
OPTIONAL_LOC = ["allowed_storage_class", "level"]    # ANY | STANDARD ; level 1 = lantai


# ----------------------------------------------------------------- data sintetis
def generate_synthetic_slotting(n_skus: int = 200, n_aisles: int = 8, bays_per_aisle: int = 15,
                                levels: int = 3, seed: int = 42, sku_ids=None):
    """Hasilkan (skus_df, locations_df) SINTETIS. Kondisi awal = penempatan acak yang layak."""
    rng = np.random.default_rng(seed)
    locs = []
    for a in range(1, n_aisles + 1):
        for b in range(1, bays_per_aisle + 1):
            for lv in range(1, levels + 1):
                x, y = a * 3.0, b * 1.2
                locs.append({
                    "location_id": f"A{a:02d}-B{b:02d}-L{lv}", "aisle": a, "bay": b, "level": lv,
                    "x": x, "y": y,
                    "distance_to_dock": round(x + y + (lv - 1) * 1.5, 2),   # Manhattan + penalti level
                    "capacity_volume": float(rng.choice([0.4, 0.8, 1.2, 1.6], p=[0.25, 0.35, 0.25, 0.15])),
                    "allowed_storage_class": "ANY" if lv == 1 else "STANDARD",
                    "data_source": "SYNTHETIC",
                })
    loc_df = pd.DataFrame(locs)
    n_skus = min(n_skus, int(len(loc_df) * 0.85))
    ids = list(sku_ids)[:n_skus] if sku_ids is not None else [f"SYN-{i:04d}" for i in range(1, n_skus + 1)]
    n_skus = len(ids)
    freq = np.round(rng.pareto(1.2, n_skus) * 20 + 1).astype(int)          # beberapa SKU sangat sering diambil
    vol = np.round(rng.uniform(0.05, 1.2, n_skus), 2)
    cls = rng.choice(["STANDARD", "HEAVY"], n_skus, p=[0.85, 0.15])
    sku_df = pd.DataFrame({"sku": ids, "required_volume": vol, "pick_frequency": freq,
                           "storage_class": cls, "data_source": "SYNTHETIC"})
    # baseline: penempatan acak layak
    free = loc_df.sample(frac=1, random_state=seed).to_dict("records")
    used, cur = set(), []
    for _, r in sku_df.sample(frac=1, random_state=seed + 1).iterrows():
        for L in free:
            if L["location_id"] in used:
                continue
            if L["capacity_volume"] >= r["required_volume"] and (
                    r["storage_class"] == "STANDARD" or L["allowed_storage_class"] == "ANY"):
                used.add(L["location_id"]); cur.append((r["sku"], L["location_id"])); break
    cur_map = dict(cur)
    sku_df["current_location"] = sku_df["sku"].map(cur_map)
    sku_df = sku_df[["sku", "current_location", "required_volume", "pick_frequency", "storage_class", "data_source"]]
    return sku_df, loc_df


# ----------------------------------------------------------------- validasi
def validate_slotting_inputs(sku_df: pd.DataFrame, loc_df: pd.DataFrame) -> dict:
    errors, warnings = [], []
    miss_s = [c for c in SLOT_SKU_COLS if c not in sku_df.columns]
    miss_l = [c for c in SLOT_LOC_COLS if c not in loc_df.columns]
    if miss_s:
        errors.append(f"File SKU kehilangan kolom wajib: {miss_s}")
    if miss_l:
        errors.append(f"File lokasi kehilangan kolom wajib: {miss_l}")
    if "distance_to_dock" not in loc_df.columns and not {"x", "y"} <= set(loc_df.columns):
        errors.append("File lokasi harus memiliki 'distance_to_dock' atau pasangan koordinat 'x' dan 'y'.")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}
    if sku_df["sku"].duplicated().any():
        errors.append("Terdapat SKU duplikat pada file SKU.")
    if loc_df["location_id"].duplicated().any():
        errors.append("Terdapat location_id duplikat pada file lokasi.")
    for c in ("required_volume", "pick_frequency"):
        if pd.to_numeric(sku_df[c], errors="coerce").isna().any():
            errors.append(f"Kolom {c} berisi nilai kosong/non-numerik.")
    if pd.to_numeric(loc_df["capacity_volume"], errors="coerce").isna().any():
        errors.append("Kolom capacity_volume berisi nilai kosong/non-numerik.")
    unknown = set(sku_df["current_location"].dropna()) - set(loc_df["location_id"])
    if unknown:
        warnings.append(f"{len(unknown)} lokasi awal tidak ada di file lokasi (SKU tsb dianggap belum ditempatkan).")
    if len(sku_df) > len(loc_df):
        errors.append("Jumlah SKU melebihi jumlah lokasi (asumsi 1 SKU per lokasi).")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def _distances(loc_df: pd.DataFrame) -> pd.Series:
    if "distance_to_dock" in loc_df.columns:
        d = pd.to_numeric(loc_df["distance_to_dock"], errors="coerce")
    else:
        d = pd.to_numeric(loc_df["x"], errors="coerce").abs() + pd.to_numeric(loc_df["y"], errors="coerce").abs()
    return pd.Series(d.values, index=loc_df["location_id"].values)


def _compatible(sku_class: str, loc_allowed: str) -> bool:
    return str(sku_class).upper() != "HEAVY" or str(loc_allowed).upper() == "ANY"


# ----------------------------------------------------------------- heuristik slotting
def recommend_slotting(sku_df: pd.DataFrame, loc_df: pd.DataFrame) -> dict:
    """Heuristik Cube-per-Order Index (COI): SKU dengan volume/frekuensi terkecil -> lokasi terdekat dock
    yang memenuhi kapasitas & kompatibilitas. 1 SKU per lokasi. Bukan optimasi global."""
    chk = validate_slotting_inputs(sku_df, loc_df)
    if not chk["ok"]:
        raise ValueError(" | ".join(chk["errors"]))
    s = sku_df.copy()
    s["required_volume"] = pd.to_numeric(s["required_volume"])
    s["pick_frequency"] = pd.to_numeric(s["pick_frequency"])
    if "storage_class" not in s.columns:
        s["storage_class"] = "STANDARD"
    L = loc_df.copy()
    L["capacity_volume"] = pd.to_numeric(L["capacity_volume"])
    if "allowed_storage_class" not in L.columns:
        L["allowed_storage_class"] = "ANY"
    dist = _distances(L)
    L["distance_to_dock"] = L["location_id"].map(dist)
    L = L.sort_values("distance_to_dock").reset_index(drop=True)

    s["coi"] = s["required_volume"] / s["pick_frequency"].clip(lower=1e-9)
    order = s.sort_values(["coi", "pick_frequency"], ascending=[True, False])
    taken, assign, unassigned = set(), {}, []
    Ld = L.to_dict("records")
    for _, r in order.iterrows():
        placed = False
        for loc in Ld:
            if loc["location_id"] in taken:
                continue
            if loc["capacity_volume"] >= r["required_volume"] and _compatible(r["storage_class"], loc["allowed_storage_class"]):
                assign[r["sku"]] = loc["location_id"]; taken.add(loc["location_id"]); placed = True; break
        if not placed:
            unassigned.append(r["sku"])
    s["recommended_location"] = s["sku"].map(assign)
    s["current_distance"] = s["current_location"].map(dist)
    s["recommended_distance"] = s["recommended_location"].map(dist)
    s["moved"] = s["current_location"] != s["recommended_location"]
    s["distance_saved_per_pick"] = s["current_distance"] - s["recommended_distance"]
    s["weighted_distance_current"] = s["pick_frequency"] * s["current_distance"]
    s["weighted_distance_recommended"] = s["pick_frequency"] * s["recommended_distance"]

    ok = s["recommended_location"].notna() & s["current_distance"].notna()
    base = float(s.loc[ok, "weighted_distance_current"].sum())
    rec = float(s.loc[ok, "weighted_distance_recommended"].sum())
    # kelayakan kondisi awal
    cap = L.set_index("location_id")["capacity_volume"]
    over = s[s["current_location"].map(cap) < s["required_volume"]]
    dup_cur = s["current_location"].dropna().duplicated().sum()
    # ABC berdasarkan frekuensi kumulatif
    ss = s.sort_values("pick_frequency", ascending=False)
    cum = ss["pick_frequency"].cumsum() / ss["pick_frequency"].sum()
    s["abc_class"] = pd.Series(np.where(cum <= 0.8, "A", np.where(cum <= 0.95, "B", "C")), index=ss.index)
    summary = {
        "n_skus": int(len(s)), "n_unassigned": len(unassigned), "n_moves": int(s["moved"].sum()),
        "baseline_weighted_distance": base, "recommended_weighted_distance": rec,
        "abs_change": rec - base, "pct_change": (100 * (rec - base) / base) if base else np.nan,
        "baseline_overcapacity_skus": int(len(over)), "baseline_shared_locations": int(dup_cur),
        "synthetic": bool("data_source" in s.columns and (s["data_source"] == "SYNTHETIC").any()),
        "method": "Heuristik greedy COI (volume/frekuensi) -> lokasi terdekat dock yang layak; 1 SKU per lokasi",
    }
    return {"table": s.sort_values("weighted_distance_current", ascending=False), "summary": summary,
            "unassigned": unassigned, "warnings": chk["warnings"]}


SLOTTING_ASSUMPTIONS = [
    "Metrik = frekuensi picking x jarak lokasi ke dock (satu arah, satuan layout). Bukan waktu atau biaya nyata.",
    "Satu SKU per lokasi; kapasitas dibandingkan dengan required_volume tanpa mempertimbangkan beban/berat.",
    "Heuristik greedy (COI) dapat dijelaskan namun tidak menjamin optimum global.",
    "Pada data sintetis, kondisi awal adalah penempatan ACAK sehingga penurunan jarak cenderung berlebihan; "
    "hasil hanya ilustrasi mekanisme, bukan estimasi penghematan gudang nyata.",
    "Tidak memodelkan pemindahan (biaya relokasi), kongesti lorong, ataupun aturan FEFO/FIFO.",
]


# ----------------------------------------------------------------- what-if replenishment
def _kpis(plan: pd.DataFrame, costs: dict) -> dict:
    below = plan["below_reorder_point"]
    high = plan["risk_category"].eq("Tinggi") if "risk_category" in plan else pd.Series(False, index=plan.index)
    units = float(plan["suggested_order_qty"].sum())
    lines = int((plan["suggested_order_qty"] > 0).sum())
    out = {
        "SKU di bawah ROP": int(below.sum()),
        "SKU berisiko Tinggi & di bawah ROP": int((below & high).sum()),
        "Total shortfall ke ROP (unit)": float(plan["shortfall_to_rop"].sum()),
        "Total kuantitas order indikatif (unit)": units,
        "Jumlah baris order": lines,
        "Total reorder point (unit)": float(plan["reorder_point"].sum()),
    }
    uc, oc, hc, bc = (costs.get(k, 0.0) for k in ("unit_cost", "order_cost", "holding_cost_per_unit_month", "backorder_event_cost"))
    out["Nilai order indikatif (ILUSTRATIF)"] = units * uc
    out["Biaya pemesanan (ILUSTRATIF)"] = lines * oc
    out["Biaya simpan horizon (ILUSTRATIF)"] = units * hc * costs.get("horizon_months", 1.0)
    out["Paparan backorder berisiko tinggi (ILUSTRATIF)"] = float((below & high).sum()) * bc
    return out


def whatif_replenishment(df: pd.DataFrame, base: ReplenParams, scenario: ReplenParams, costs: dict) -> pd.DataFrame:
    """Bandingkan baseline vs skenario pada aturan replenishment yang sama. Mengembalikan tabel perbandingan."""
    pb = replenishment_plan(df, base)
    ps = replenishment_plan(df, scenario)
    kb, ks = _kpis(pb, costs), _kpis(ps, costs)
    rows = []
    for k in kb:
        b, s_ = kb[k], ks[k]
        rows.append({"Indikator": k, "Baseline": b, "Skenario": s_, "Perubahan absolut": s_ - b,
                     "Perubahan %": (100 * (s_ - b) / b) if b else np.nan,
                     "Jenis": "Hitungan dari data" if "ILUSTRATIF" not in k else "Asumsi biaya (simulasi)"})
    return pd.DataFrame(rows)


def params_diff(base: ReplenParams, scenario: ReplenParams) -> pd.DataFrame:
    from dataclasses import asdict
    a, b = asdict(base), asdict(scenario)
    return pd.DataFrame([{"Parameter": k, "Baseline": a[k], "Skenario": b[k]} for k in a if a[k] != b[k]])
