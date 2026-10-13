import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

model = pd.read_csv("Data/processed/modeling_table.csv")
print(model.iloc[:, :9].dtypes)          # the 8 summary columns + powder
print(model.iloc[:, :9].nunique())

TARGET   = "rsd_pct"                                   # your RSD column
PROCESS  = ["...", "..."]                         # numeric process settings
CATEG    = ["..."]                                # categorical settings (e.g. paddle geometry), if any
prop_cols = list(pd.read_csv("Data/processed/powder_properties_merged.csv").columns[1:])

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