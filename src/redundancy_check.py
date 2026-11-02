import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ------------------ 1. Load ------------------
props = pd.read_csv("Data/processed/powder_properties_merged.csv")
props = props.set_index(props.columns[0])   # powder name
props = props.select_dtypes(include="number")  # keeping numeric properties only
print(props.shape)  # expected shape (9,39)


# ------------------ 2. Clean: drop columns that can't be correlated ------------------
const = props.columns[props.nunique(dropna=True) <= 1]
print("Constant columns (dropped): ", list(const))
props = props.drop(columns=const)
print("Missing values per column: \n", props.isna().sum()[lambda s: s > 0])


# ------------------ 3. Spearman correlation -> distance ------------------
rho = props.corr(method="spearman")     # pairwise, ignores NaN per pair
dist = 1 - rho.abs()
dist = dist.fillna(1.0)     # undefined pair -> treated as unrelated
np.fill_diagonal(dist.values, 0.0)
dist = (dist + dist.T) / 2      # force exact symmetry

print (rho.iloc[:5, :5])

# ------------------ 4. Hierarchical clustering ------------------
Z = linkage(squareform(dist.values, checks=False), method="average")

def cut(Z, threshold):
    return fcluster(Z, t=threshold, criterion='distance')

# sensitivity: how many clusters at different cut heights?
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    print(f"d <= {t: .2f} (|rho| >= {1-t: .2f}): {cut(Z, t).max()} clusters")

THRESH = 0.10
labels = pd.Series(cut(Z, THRESH), index=props.columns, name="cluster")


# ------------------ 5.Plots ------------------
out = Path("Data/reports/figures"); out.mkdir(parents=True, exist_ok=True)

# 5a. Dendrogram
fig, ax = plt.subplots(figsize=(8,10))
dendrogram(Z, labels=props.columns.tolist(), orientation="left", color_threshold=THRESH, ax=ax)
ax.axvline(THRESH, ls="--", c="k")
ax.set_xlabel("1 - |Spearman rho|")
fig.tight_layout()
fig.savefig(out / "dendrogram.png", dpi=200, bbox_inches="tight")       # save before showing
plt.show()

# 5b. Heatmap in dendrogram order
order = leaves_list(Z)      # heatmap in cluster order
names = props.columns[order]
fig, ax = plt.subplots(figsize=(11,10))
im = ax.imshow(rho.values[np.ix_(order, order)], cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(order))); ax.set_xticklabels(names, rotation=90, fontsize=7)
ax.set_yticks(range(len(order))); ax.set_xticklabels(names, fontsize=7)
fig.colorbar(im, label="Spearman rho")
fig.tight_layout()
fig.savefig(out / "heatmap.png", dpi=200, bbox_inches="tight")      # bbox+inches fixes clipped labels
plt.show()


# ------------------ 6. Representatives ------------------
# Optional/ set later: own measurement-cost ranking (1 = cheap/easy). missing -> default 3
cost = {
    # "bulk_density": 1,
    # "d50": 2,
    # "shear_cell_ffc": 5,
    }

def pick_representatives(members, rho, cost):
    if len(members) == 1:
        return members[0]
    # most central member: highest mean |rho| to the others in the cluster
    centrality = rho.loc[members, members].abs().mean()
    df = pd.DataFrame({"cost": [cost.get(m, 3) for m in members],
                       "centrality": centrality.values}, index=members)
    # cheapest first, ties broken by centrality
    return df.sort_values(["cost", "centrality"], ascending=[True, False]).index[0]

reps = {c: pick_representatives(list(g.index), rho, cost)
        for c, g in labels.groupby(labels)}

summary = (labels.reset_index().rename(columns={"index": "property"})
           .groupby("cluster")["property"].apply(list).to_frame("members"))
summary["representative"] = pd.Series(reps)
summary["size"] = summary["members"].str.len()
print(summary.to_string())

summary_out = summary.copy()
summary_out["members"] = summary_out["members"].apply(lambda m: "; ".join(m))
summary_out.to_csv(out.parent / "redundancy_clusters.csv")
labels.rename_axis("property").to_csv(out.parent / "property_cluster_labels.csv")       # one row per property

rep_cols = list(reps.values())
print(f"\n{len(rep_cols)} representaties out of {props.shape[1]} properties")


# ------------------ 7. VIF on the representatives ------------------
X = props[rep_cols].dropna()
X = (X - X.mean()) / X.std()        # standardize
print(f"VIF computed on n={len(X)} powders, p={X.shape[1]} properties")
if X.shape[1] >= len(X) - 1:
    print("WARNING: p >= n-1, VIF is not meaningful. Use a higher threshold/ fewer representatives.")
else:
    Xc = X.copy()
    Xc.insert(0, "const", 1.0)
    vif = pd.Series([variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])],
                    index=X.columns, name="VIF").sort_values(ascending=False)
    print(vif.round(2))     # rule of thumb: >5 concerning, >10 severe
