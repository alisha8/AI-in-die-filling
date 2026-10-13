"""
Merge powder characterization data (sheet "Powder characterization") with the
stress/speed-dependent curves (sheet "Full Datasets powder charact.") into a
single one-row-per-powder table ready for Part A / Part B correlation analysis.

Sheet 1 supplies the scalar properties that have no curve in sheet 2:
  bulk density, tapped density, Hausner ratio, true density,
  static angle of repose.
(Its fixed-point values for ffc/dynamic angle/cohesion/volume flow/
permeability are dropped here since they duplicate one slice of sheet 2's
curves -- verified earlier to match exactly.)

Sheet 2 supplies the full curve for five properties, each kept as ONE COLUMN
PER MEASURED LEVEL (no summarization), so no shape information is lost:
  permeability   vs compression stress    (5 levels: 1,5,10,15,20 kPa)
  ffc            vs pre-shear normal stress (4 levels: 1,2,4,8 kPa)
  dynamic_angle  vs rotational speed       (10 levels: 2..60 rpm)
  cohesion_index vs rotational speed       (10 levels: 2..60 rpm)
  volume_flow    vs orifice diameter       (6 levels: 6..25 mm)
"no flow" entries in volume_flow are recoded as 0.0.
"""

import pandas as pd
import numpy as np

INPUT_PATH = "Data/raw/Powder characterization and paddle wheel geometry_new.xlsx"

# ---------- sheet 1: fixed-point properties with no curve ----------
s1 = pd.read_excel(INPUT_PATH, sheet_name="Powder characterization", header=None)

s1_header_row = 4
s1_data_rows  = range(6, 16)   # 10 powders

s1_cols = {
    1: "powder_full_name",
    2: "bulk_density_g_cm3",
    4: "tapped_density_g_cm3",
    6: "hausner_ratio",
    11: "static_angle_of_repose_deg",
    # sheet 1 columns 6, 8, 9, 10, 11 (ffc/dynamic angle/cohesion/volume
    # flow/permeability fixed points) are intentionally skipped -- they
    # duplicate one slice of sheet 2's curves, kept below in full.
    # col 5 - true_density_g_cm3) dropped : missing 5/10 powders.
}

sheet1 = s1.loc[list(s1_data_rows), list(s1_cols.keys())].copy()
sheet1.columns = list(s1_cols.values())
sheet1 = sheet1.set_index("powder_full_name")

# map sheet 1's long material names -> sheet 2 / tablet-dataset short names
name_map = {
    "Dicalciumphosphate (DCP)": "DCP",
    "Tricalciumcitrate (TCC)": "TCC",
    "Lactose-Anhydrate (ST21AN)": "Lactose-Anhydrate",
    "MCC coarse": "MCC coarse",
    "MCC  fine": "MCC fine",   # note: double space in source sheet
    "Powdered Cellulose (Arbocel P290)": "Powdered Cellulose (Arbocel P290)",
    "MCC finest (Vivapur 120)": "MCC finest (Vivapur 120)",
    #"MCC finest + Pellets (10%Cellets700 10%Vivapur101 80%Vivapur102)":
    #    "MCC finest + Pellets (10%Cellets700 10%Vivapur101 80%Vivapur102)",
    "HPMC (60% Tylopur 90SH 40% Vivapur120)": "HPMC (60% Tylopur 90SH 40% Vivapur120)",
    "CaCO3(50% MagGranCC 39% TCC 10% MCC Vivapur 120 1% MgSt)":
        "CaCO3(50% MagGranCC 39% TCC 10% MCC Vivapur 120 1% MgSt)",
}
sheet1.index = sheet1.index.str.strip().map(lambda n: name_map.get(n, n))
sheet1.index.name = "powder"
sheet1 = sheet1[~sheet1.index.str.contains("Pellets")]

# ---------- sheet 2: curves -> per-powder summary features ----------
raw = pd.read_excel(INPUT_PATH, sheet_name="Full Datasets powder charact.", header=None)
title_row, header_row = raw.iloc[2], raw.iloc[6]

BLOCK_PREFIX = {
    "Permeability [m2]": "permeability",
    "ffc-values [-]": "ffc",
    "Dynamic angle of repose [°]": "dynamic_angle",
    "Cohesion index [-]": "cohesion_index",
    "Volume flow through orifice [cm3/s]": "volume_flow",
}

def extract_block(title_col):
    c = title_col
    while pd.isna(header_row[c]):
        c += 1
    cols = []
    while c < raw.shape[1] and pd.notna(header_row[c]):
        cols.append(c)
        c += 1
    indep_col, powder_cols = cols[0], cols[1:]
    powder_names = [header_row[pc] for pc in powder_cols]
    data = raw.iloc[7:, [indep_col] + powder_cols].copy()
    data.columns = ["x"] + powder_names
    data = data.loc[:, [c for c in data.columns if "Pellets" not in str(c)]]
    data = data.dropna(subset=["x"]).reset_index(drop=True)
    return data

def fmt_level(x):
    """Format an independent-variable value for a column name, e.g. 11.3 -> '11p3'."""
    s = f"{x:g}"
    return s.replace(".", "p").replace("-", "m")

curve_features = {}
curve_long = []  # also keep a tidy long-format version, for plotting
imputed_log =[]
deviation_blocks = {}
for tc in raw.columns:
    if pd.notna(title_row[tc]) and title_row[tc] in BLOCK_PREFIX:
        prefix = BLOCK_PREFIX[title_row[tc]]
        assert prefix not in curve_features, f"duplicate prefix {prefix}"
        block = extract_block(tc)

        dev_cols = [c for c in block.columns if "deviation" in str(c).lower()]
        if dev_cols:
            deviation_blocks[prefix] = block[["x"] + dev_cols].copy()
        block = block.drop(columns=dev_cols)

        # recode "no flow" (text) as 0.0 -- a real physical outcome, not missing
        block["x"] = pd.to_numeric(block["x"], errors="coerce")
        block = block.dropna(subset=["x"]).sort_values("x").reset_index(drop=True)

        for col in block.columns[1:]:
            no_flow = block[col].astype(str).str.strip().str.lower().eq("no flow")
            block[col] = pd.to_numeric(block[col], errors="coerce")
            block.loc[no_flow, col] = 0.0       # real outcome, not missing
            
        was_missing = block.iloc[:, 1:].isna()  # remember what gets imputed

        for col in block.columns[1:]:
            # interior gaps: interpolate on the real x values
            s = block.set_index("x")[col].interpolate(method="index", limit_area="inside")
            block[col] = s.values

            # edge gaps (extrapolate from the 2 nearest measured points
            for idx in block.index[block[col].isna()]:
                    known = block[["x", col]].dropna()
                    if len(known) < 2:
                        continue    # not enough points to extrapolate from

                    x_t = block.loc[idx, "x"]
                    nearest = known.iloc[(known["x"] - x_t).abs().argsort()[:2]]
                    slope, intercept = np.polyfit(nearest["x"], nearest[col], 1)
                    block.loc[idx, col] = slope * x_t + intercept

        n_filled = int(was_missing.sum().sum() - block.iloc[:, 1:].isna().sum().sum())
        print(f"[{prefix}] imputed {n_filled} of {int(was_missing.sum().sum())} missing values")
        imputed_log.append((prefix, was_missing.assign(x=block["x"])))      

        # wide format: one column per measured level, no summarization
        wide = block.set_index("x")[block.columns[1:]].T  # rows = powders
        wide.columns = [f"{prefix}_{fmt_level(x)}" for x in wide.columns]
        curve_features[prefix] = wide

        # long format for plotting later
        tidy = block.melt(id_vars="x", var_name="powder", value_name="value")
        tidy["property"] = prefix
        curve_long.append(tidy)

for name, dev in deviation_blocks.items():
    dev.to_csv(f"Data/processed/powder_{name}_deviation.csv", index=False)

sheet2_features = pd.concat(curve_features.values(), axis=1)
curve_long_df = pd.concat(curve_long, ignore_index=True)
curve_long_df.to_csv("Data/processed/powder_curves_long.csv", index=False)

# ---------- merge ----------
merged = sheet1.join(sheet2_features, how="outer")
merged.index.name = "powder"

merged.to_csv("Data/processed/powder_properties_merged.csv", index_label="powder")
print(f"Saved merged table (full precision, use this for analysis): "
      f"{merged.shape[0]} powders x {merged.shape[1]} features -> Data/processed/powder_properties_merged.csv")
print(f"Saved tidy long-format curves (for plotting)        -> Data/processed/powder_curves_long.csv")

remaining_nan = merged.isna().sum()
remaining_nan = remaining_nan[remaining_nan > 0]
if len(remaining_nan):
    print("\nStill missing after interpolation (gap at the edge of a curve, or no "
          "data at all for that powder -- interpolation can't invent them):")
    print(remaining_nan.to_string())


# ---------- human-readable display version (rounded, for reports/meetings) ----------
# NOTE: this file is for reading, not for the correlation analysis -- use
# powder_properties_merged.csv (full precision) for Part A / Part B.
display_df = merged.copy()
for col in display_df.columns:
    # force numeric regardless of current dtype -- defends against any
    # column that ended up as object dtype (e.g. pandas-version differences
    # in how NaN-filled or interpolated columns get typed)
    numeric_col = pd.to_numeric(display_df[col], errors="coerce")
    if col.startswith("permeability_"):
        # tiny values (~1e-12 to 1e-13): fixed-decimal rounding would show
        # 0.000, so use scientific notation with 3 significant figures
        display_df[col] = numeric_col.map(lambda v: f"{v:.2e}" if pd.notna(v) else "")
    else:
        display_df[col] = numeric_col.round(2)
 
display_df.to_csv("Data/processed/powder_properties_display.csv", index_label="powder")
print(f"Saved rounded, aligned display version              -> Data/processed/powder_properties_display.csv")
 
print("\nColumns:", list(merged.columns))


# ------------------------- Verification -----------------------------
print("\n====Verification====")

# 1. Shape and powders
print("Shape:", merged.shape, "(expect 9 x 39)")
print(merged.index.tolist())
assert merged.index.is_unique
assert not any("Pellets" in p for p in merged.index), "Pellets still present"
assert merged.shape == (9, 39)

# 2. Column groups
expected = {"bulk_density": 1, "tapped_density": 1, "hausner": 1, "static_angle": 1, "permeability_": 5,
            "ffc_": 4, "dynamic_angle_": 10, "cohesion_index_": 10, "volume_flow_": 6}
for pref, n in expected.items():
    got = sum(c.startswith(pref) for c in merged.columns)
    print(f"{pref:<18} {got:>2} columns (expect {n})", "" if got == n else " <---- MISMATCH")

# 3. Missing values after imputation
nan_counts = merged.isna().sum()
print("\nNaNs per column (only nonzero): ")
print(nan_counts[nan_counts > 0].to_string() if nan_counts.any() else "none")

# 4. Imputation log: exactly which cells were filled
print("\nImputed cells:")
for prefix, mask in imputed_log:
    xs = mask["x"]
    for powder in [c for c in mask.columns if c != "x"]:
        if powder not in merged.index:      # e.g. dropped Pellets
            continue
        for i in mask.index[mask[powder]]:
            col = f"{prefix}_{fmt_level(xs[i])}"
            print(f"  {col:<22} {powder:<45} {merged.loc[powder, col]}")

# 5. 'no flow' -> 0.0 check
vf = merged.filter(like="volume_flow_")
print("\nvolume_flow zeros (recoded 'no flow'):", int((vf == 0).sum().sum()))
print("negative values anywhere:", int((merged.select_dtypes("number") < 0).sum().sum()))

# 6. Fixed points in sheet 1 must appear in the sheet 2 curves
fixed_map = {10: "ffc", 13: "dynamic_angle", 14: "cohesion_index",
             15: "volume_flow", 17: "permeability"}
s1_fixed = s1.loc[list(s1_data_rows), [1] + list(fixed_map)].copy()
s1_fixed[1] = s1_fixed[1].astype(str).str.strip().map(lambda n: name_map.get(n, n))
s1_fixed = s1_fixed.set_index(1)
print("\nFixed-point cross-check (sheet 1 value found among the sheet 2 curve levels):")
for col, prefix in fixed_map.items():
    curve = merged.filter(regex=f"^{prefix}_")
    bad = []
    for p in merged.index:
        v = pd.to_numeric(s1_fixed.loc[p, col], errors="coerce")
        if pd.isna(v):
            continue
        if not np.isclose(curve.loc[p].values.astype(float), v, rtol=1e-3, atol=0).any():
            bad.append(p)
    print(f"  {prefix:<15}", "OK" if not bad else f"NOT FOUND for: {bad}")

# 7. Random spot-check against the workbook, compare by hand
print("\nSpot-check sample (compare with Excel):")
rng = np.random.default_rng(0)
for _ in range(10):
    p = rng.choice(merged.index); c = rng.choice(merged.columns)
    print(f"  {p:<45} {c:<22} {merged.loc[p, c]}")

chk = merged[["bulk_density_g_cm3", "tapped_density_g_cm3", "hausner_ratio"]].copy()
chk["tapped/bulk"] = chk["tapped_density_g_cm3"] / chk["bulk_density_g_cm3"]
print(chk.round(4).to_string())    

p = "DCP"
curve = merged.filter(like="volume_flow_").astype(float)
for p in merged.index:
    v = pd.to_numeric(s1_fixed.loc[p, 10], errors="coerce")
    rel = (curve.loc[p] - v).abs() / max(abs(v), 1e-12)
    print(f"{p[:30]:<32} sheet1={v:<10.4f} nearest={rel.idxmin():<18} rel.diff={rel.min():.3%}")