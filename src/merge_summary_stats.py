import pandas as pd

props = pd.read_csv("Data/processed/powder_properties_merged.csv")
summary = pd.read_csv("Data/processed/summary_stats.csv")

# 1. Rename the properties file's powders to the summary file's short names
props_to_summary = {
    "CaCO3(50% MagGranCC 39% TCC 10% MCC Vivapur 120 1% MgSt)": "CaCO3(50%MagGranCC)",
    "HPMC (60% Tylopur 90SH 40% Vivapur120)": "HPMC(60%Tylopur40%V120)",
    "Lactose-Anhydrate": "Lactose Anhydrate",
    "Powdered Cellulose (Arbocel P290)": "Powdered cellulose(ArbocelP290)",
}
props["powder"] = props["powder"].replace(props_to_summary)

# 2. MCC finest: treat as Vivapur 120
summary["powder"] = summary["powder"].replace(
    {"MCC finest (Vivapur102)": "MCC finest (Vivapur 120)"}
)

# 3. Names must now match exactly
print(sorted(set(props["powder"]) ^ set(summary["powder"])))

# 4. Merge, check, save
model = summary.merge(props, on="powder", how="left", validate="m:1")
assert len(model) == 648
prop_cols = [c for c in props.columns if c != "powder"]
assert model[prop_cols].notna().all().all()
model.to_csv("Data/processed/modeling_table.csv", index=False)
print(model.shape, model["powder"].nunique())   # expect (648, ...), 9