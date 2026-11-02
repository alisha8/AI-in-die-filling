import numpy as np
import pandas as pd
import seaborn as sns
from itertools import permutations
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

TARGET = "rsd_pct"
PROCESS = ["nt", "np"]

model = pd.read_csv("Data/processed/modeling_table.csv")
props_cols = list(pd.read_csv("Data/processed/powder_properties_merged.csv").columns[1:])
clusters = pd.read_csv("Data/reports/property_cluster_labels.csv")

# ------------- helpers -------------
def powder_table(d, props):
    """ 9 rows: median RSD + property values per powder """
    return d.groupby("powder").agg({TARGET: "median", **{c: "first" for c in props}})

def rho_series(pw, props):
    out = {}
    for c in props:
        if pw[c].nunique() >= 3:
            out[c] = spearmanr(pw[c], pw[TARGET])[0]
    return pd.Series(out)

# exact permutation test (9! = 362,880 perms, fast when vectorised)
def exact_perm_p(x, y):
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    rx = (rx - rx.mean()) / np.linalg.norm(rx - rx.mean())
    ry = ry - ry.mean()
    ry = ry / np.linalg.norm(ry)
    P = np.array(list(permutations(ry)))        # 362880 x 9
    null = P @ rx
    obs = rx @ ry
    return (np.abs(null) >= abs(obs) - 1e-12).mean()


# ------------- strata -------------
strata = {"ALL": model}
for k, d in model.groupby("geometry"): strata[f"geo:{k}"] = d
for k, d in model.groupby("nt"):       strata[f"nt:{k}"] = d
for (a, b), d in model.groupby(["nt", "np"]): strata[f"nt{a}_np{b}"] = d    # optional

heat = pd.DataFrame({name: rho_series(powder_table(d, props_cols), props_cols) for name, d in strata.items()})


# ------------- overall significance (screening only) -------------
pw = powder_table(model, props_cols)
summary = pd.DataFrame({"rho_all": heat["ALL"]})
summary["p_perm"] = [exact_perm_p(pw[c], pw[TARGET]) for c in summary.index]


# ------------- consistency across strata -------------
sub = heat.drop(columns="ALL")
geo_cols = [c for c in sub.columns if c.startswith("geo:")]
nt_cols = [c for c in sub.columns if c.startswith("nt:")]
summary["n_same_sign_geo"] = (np.sign(heat[geo_cols]).eq(np.sign(summary["rho_all"]), axis=0)).sum(axis=1)
summary["n_same_sign_nt"] = (np.sign(heat[nt_cols]).eq(np.sign(summary["rho_all"]), axis=0)).sum(axis=1)
summary["median_abs_rho_strata"] = sub.abs().median(axis=1)


# ------------- leave-one-powder-out -------------
loo = pd.DataFrame({p: rho_series(pw.drop(p), props_cols) for p in pw.index})
summary["loo_min"], summary["loo_max"] = loo.min(axis=1), loo.max(axis=1)
summary["loo_sign_flip"] = (loo.min(axis=1) * loo.max(axis=1)) < 0


# ------------- join Redundancy and correlation -------------
final = (summary.reset_index(names="property").merge(clusters, on="property", how="left", validate="1:1"))
final["abs_rho"] = final["rho_all"].abs()
final = final.sort_values(["cluster", "abs_rho"], ascending=[True, False])
print(final.round(3).to_string(index=False))


# cluster-level view: is the *clusters* relevant, not just it's representative?
print(final.groupby("cluster")["rho_all"].agg(["count", "min", "max", "median"]).round(2))


# heatmap
order = final["property"]
plt.figure(figsize=(14, 9))
sns.heatmap(heat.loc[order], center=0, cmap="RdBu_r", vmin=-1, vmax=1, annot=False)
plt.tight_layout()
plt.savefig("Data/reports/relevance_heatmap.png", dpi=200)
final.to_csv("Data/reports/property_relevance_summary.csv", index=False)
