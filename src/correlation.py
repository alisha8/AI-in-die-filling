import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import kruskal
from sklearn.feature_selection import mutual_info_regression

INPUT_PATH = "Data/raw/Powder characterization and paddle wheel geometry_new.xlsx"

model = pd.read_csv("Data/processed/modeling_table.csv")
# print(model.iloc[:, :9].dtypes)          # the 8 summary columns + powder
# print(model.iloc[:, :9].nunique())

TARGET   = "rsd_pct"                                   # your RSD column
PROCESS  = ["nt", "np"]                         # numeric process settings
CATEG    = ["geometry"]                                # categorical settings (e.g. paddle geometry), if any
prop_cols = list(pd.read_csv("Data/processed/powder_properties_merged.csv").columns[1:])

g = pd.read_excel(INPUT_PATH, sheet_name="Paddle wheel geometries", header=None)
hdr = g.index[g.iloc[:, 1].astype(str).str.contains("Paddle wheel geometry name", na=False)][0]

geo = g.iloc[hdr + 1:, 1:7].copy()
geo.columns = ["geometry", "hub_height_mm", "hub_diameter_mm",
               "n_spokes", "spoke_height_mm", "spoke_thickness_mm"]
geo = geo.dropna(subset=["hub_height_mm"])            # drops the blank rows between entries
geo["geometry"] = (geo["geometry"].astype(str).str.strip().replace({
    "Rectangular cranked 10 mm":  "Rectangular cranked 10mm",
        "Rectangular cranked 5,5 mm": "Rectangular cranked 5.5mm",
        "Round cranked 10 mm":        "Round cranked 10mm",
        "Round cranked 5,5 mm":       "Round cranked 5.5mm",
}))
num_cols = ["hub_height_mm", "hub_diameter_mm", "n_spokes", "spoke_height_mm", "spoke_thickness_mm"]
geo[num_cols] = geo[num_cols].apply(pd.to_numeric)
print(geo.dtypes)                      # all five should be float64 or int64

print(sorted(set(geo["geometry"]) ^ set(model["geometry"])))   # expect []

model = model.merge(geo, on="geometry", how="left", validate="m:1")
assert len(model) == 648 and model[geo.columns[1:]].notna().all().all()
GEO_NUM = list(geo.columns[1:])

# 1. Row-level Spearman and MI
y = model[TARGET]
rows = []
for c in PROCESS + prop_cols:
    x = model[c]
    if x.nunique() < 2:
        continue
    rho, p = spearmanr(x, y)
    rows.append({"feature": c, "type": "process" if c in PROCESS else "property",
                 "n_distinct": x.nunique(), "rho": rho})
rowlevel = pd.DataFrame(rows)

X = model[PROCESS + prop_cols]
disc = [c in PROCESS and model[c].nunique() <= 10 for c in X.columns]
rowlevel["MI"] = pd.Series(
    mutual_info_regression(X, y, discrete_features=disc, n_neighbors=5, random_state=0),
    index=X.columns).reindex(rowlevel["feature"]).values

print(rowlevel.assign(abs_rho=lambda d: d.rho.abs()).sort_values("abs_rho", ascending=False)
      .drop(columns="abs_rho").round(3).to_string(index=False))

# 2. Process setting, powder effect removed
within = model.copy()
within["_y"] = within.groupby("powder")[TARGET].rank(pct=True)
rows = []
for c in PROCESS:
    r = within.groupby("powder")[c].rank(pct=True)
    rows.append({"feature": c, "rho_within": spearmanr(r, within["_y"])[0]})
    # per-powder consistency: same sign in how many of the 9?
    per = [spearmanr(g[c], g[TARGET])[0] for _, g in model.groupby("powder")]
    rows[-1]["n_pos"] = sum(v > 0 for v in per); rows[-1]["n_neg"] = sum(v < 0 for v in per)
print(pd.DataFrame(rows).round(3).to_string(index=False))

# 3. Powder level properties
pw = model.groupby("powder").agg({TARGET: "median", **{c: "first" for c in prop_cols}})
out = []
for c in prop_cols:
    if pw[c].nunique() < 3:
        continue
    rho, p = spearmanr(pw[c], pw[TARGET])
    out.append({"property": c, "n_distinct": pw[c].nunique(), "rho": rho, "p": p})
res = pd.DataFrame(out).sort_values("rho", key=abs, ascending=False)
print(res.round(3).to_string(index=False))

top = ["cohesion_index_50", "cohesion_index_40", "cohesion_index_30",
       "dynamic_angle_30", "hausner_ratio", "bulk_density_g_cm3"]

# 1. leave-one-powder-out
for c in top:
    r = [spearmanr(pw.drop(p)[c], pw.drop(p)[TARGET])[0] for p in pw.index]
    print(f"{c:<20} full={spearmanr(pw[c], pw[TARGET])[0]:.2f}  LOO min={min(r):.2f} max={max(r):.2f}")

# 2. does the powder ranking hold at each nt level?
for lvl, d in model.groupby("nt"):
    pwl = d.groupby("powder").agg({TARGET: "median", **{c: "first" for c in top}})
    print(lvl, {c: round(spearmanr(pwl[c], pwl[TARGET])[0], 2) for c in top})
    print(model.groupby(["powder", "nt", "np"]).size().value_counts())

model["rsd_rank"] = model.groupby("powder")[TARGET].rank(pct=True)
H, p = kruskal(*[d["rsd_rank"].values for _, d in model.groupby("geometry")])
print(model.groupby("geometry")["rsd_rank"].median().sort_values().round(3))
print(f"H={H:.1f}  eps2={H/(len(model)-1):.3f}")

# is the geometry ordering the same in every powder?
print(model.pivot_table(index="powder", columns="geometry", values=TARGET, aggfunc="median")
      .rank(axis=1).astype(int))
print(model.groupby(["powder", "nt", "np"]).size().value_counts())