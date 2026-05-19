# ============================================================
# Step 10 — Export cleaned tables for the Power BI dashboard
# ============================================================
#
# The dissertation proposal specifies a 5-page Power BI dashboard.
# This script builds one clean CSV per page, optimised for Power BI
# (no index columns, human-readable labels, all values rounded).
#
# Dashboard pages:
#   Page 1 — Overview: model performance and NLP corpus summary
#   Page 2 — NLP findings: topic keywords and theme-variable mapping
#   Page 3 — Empirical results: Model F coefficients + bootstrap CIs
#   Page 4 — Country map: cereal availability by country (choropleth)
#   Page 5 — Robustness: 7 specs and Cook's Distance flagged countries
#
# Output folder: outputs/powerbi/
# ============================================================

import os
import numpy as np
import pandas as pd

os.makedirs("outputs/powerbi", exist_ok=True)

print("Step 10 — Building Power BI data exports...")
print("=" * 60)


# ============================================================
# Page 1 — Overview
# ============================================================
# Two tables:
#   1a. Model comparison (OLS R², Adj R², CV R², N)
#   1b. Corpus summary (source, count, strict vs broad)
# ============================================================

print("\n[Page 1] Overview tables...")

# ── 1a. Model performance ──────────────────────────────────────────────────
models_raw = pd.read_csv("outputs/tables/model_comparison.csv")

page1_models = models_raw[[
    "Model", "N (countries)", "Predictors used",
    "OLS R²", "OLS Adj R²", "OLS F-stat p",
    "RF 5-fold CV R²", "XGB 5-fold CV R²",
]].copy()

# Clean model name (remove long dashes for Power BI labels)
page1_models["Model Label"] = page1_models["Model"].str.replace(
    " — ", ": ", regex=False
).str.replace("—", "", regex=False).str.strip()


# I define the significance function for the F-stat column
def sig(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return "n.s."


page1_models["Overall F sig"] = page1_models["OLS F-stat p"].apply(sig)
page1_models["Delta R² vs A*"] = np.nan

# Compute delta vs Model A* for Model F
r2_a_star = page1_models.loc[
    page1_models["Model"].str.contains("NLP sample", regex=False), "OLS R²"
]
r2_f = page1_models.loc[
    page1_models["Model"].str.contains("Model F", regex=False), "OLS R²"
]
if len(r2_a_star) > 0 and len(r2_f) > 0:
    idx_f = page1_models[page1_models["Model"].str.contains("Model F", regex=False)].index[0]
    page1_models.loc[idx_f, "Delta R² vs A*"] = round(
        float(r2_f.iloc[0]) - float(r2_a_star.iloc[0]), 3
    )

page1_models.to_csv("outputs/powerbi/page1_model_performance.csv", index=False)
print("  Saved: page1_model_performance.csv")

# ── 1b. Corpus summary ────────────────────────────────────────────────────
summary_raw = pd.read_csv("outputs/tables/literature_alignment_summary.csv")
page1_corpus = summary_raw.rename(columns={
    "alignment_level": "Alignment Level",
    "source_db":       "Source",
    "papers":          "Paper Count",
})
page1_corpus["Alignment Level"] = page1_corpus["Alignment Level"].str.capitalize()
page1_corpus.to_csv("outputs/powerbi/page1_corpus_summary.csv", index=False)
print("  Saved: page1_corpus_summary.csv")


# ============================================================
# Page 2 — NLP findings
# ============================================================
# Two tables:
#   2a. NMF topic keywords (topic, top keywords, dominant docs)
#   2b. Theme → variable mapping
# ============================================================

print("\n[Page 2] NLP topic tables...")

# ── 2a. NMF topics ────────────────────────────────────────────────────────
if os.path.exists("data/processed/nmf_availability_topics.csv"):
    nmf_raw = pd.read_csv("data/processed/nmf_availability_topics.csv")
    page2_nmf = nmf_raw[["topic_id", "top_keywords", "n_dominant_docs"]].copy()
    page2_nmf.columns = ["Topic ID", "Top Keywords", "Dominant Papers"]
    page2_nmf["Topic Label"] = [
        "Land / Soil / Water",
        "Household Determinants",
        "Post-Harvest Loss & Storage",
        "Climate Change Adaptation",
        "Grain Variety & Temperature",
        "Technology & Storage Adoption",
        "Africa: Value Chain & Investment",
    ][:len(page2_nmf)]
    page2_nmf.to_csv("outputs/powerbi/page2_nmf_topics.csv", index=False)
    print("  Saved: page2_nmf_topics.csv")

# ── 2b. Theme → variable mapping ──────────────────────────────────────────
if os.path.exists("data/processed/step3_theme_variable_mapping.csv"):
    map_raw = pd.read_csv("data/processed/step3_theme_variable_mapping.csv")
    map_raw = map_raw.rename(columns={
        "theme_label":    "NLP Theme",
        "proxy_variable": "Proxy Variable (Model F)",
        "dataset_source": "Data Source",
        "topic_id":       "Topic ID",
        "top_words":      "Top Words",
    })
    map_raw.to_csv("outputs/powerbi/page2_theme_variable_map.csv", index=False)
    print("  Saved: page2_theme_variable_map.csv")

# ── 2c. TF-IDF top keywords (for word cloud) ──────────────────────────────
if os.path.exists("data/processed/tfidf_top_keywords.csv"):
    tfidf_raw = pd.read_csv("data/processed/tfidf_top_keywords.csv")
    page2_tfidf = (
        tfidf_raw[tfidf_raw["scope"] == "corpus-wide"]
        .sort_values("rank")
        .head(30)
        [["rank", "keyword", "mean_tfidf"]]
        .rename(columns={"rank": "Rank", "keyword": "Keyword", "mean_tfidf": "TF-IDF Score"})
    )
    page2_tfidf.to_csv("outputs/powerbi/page2_tfidf_keywords.csv", index=False)
    print("  Saved: page2_tfidf_keywords.csv")


# ============================================================
# Page 3 — Empirical results
# ============================================================
# Three tables:
#   3a. NLP synthesis: theme, variable, coef, p, significance
#   3b. Bootstrap CIs for NLP predictors
#   3c. Nested F-test summary (one row)
# ============================================================

print("\n[Page 3] Empirical results tables...")

# ── 3a. NLP synthesis ─────────────────────────────────────────────────────
synth_raw = pd.read_csv("outputs/tables/nlp_empirical_synthesis.csv")
page3_synth = synth_raw[[
    "theme", "proxy_variable", "coefficient", "p_value", "significance"
]].copy()
page3_synth.columns = [
    "NLP Theme", "Model F Variable", "OLS Coefficient", "p-value", "Significance"
]


# I define a named function to replace the lambda that was here
def get_direction(x):
    if x > 0:
        return "Positive"
    elif x < 0:
        return "Negative"
    else:
        return "—"


page3_synth["Direction"] = page3_synth["OLS Coefficient"].apply(get_direction)
page3_synth.to_csv("outputs/powerbi/page3_nlp_synthesis.csv", index=False)
print("  Saved: page3_nlp_synthesis.csv")

# ── 3b. Bootstrap CIs ─────────────────────────────────────────────────────
boot_raw = pd.read_csv("outputs/tables/bootstrap_confidence_intervals.csv")
nlp_vars = [
    "cereal_loss_pct", "trade_pct_gdp",
    "rural_electricity_access_pct", "fertiliser_efficiency",
    "food_price_inflation_pct",
]
boot_f = boot_raw[
    (boot_raw["model"] == "Model F") & (boot_raw["variable"].isin(nlp_vars))
].copy()
boot_f["CI Excludes Zero"] = (boot_f["ci_lower_95"] * boot_f["ci_upper_95"] > 0).map(
    {True: "Yes ✓", False: "No"}
)
boot_f = boot_f.rename(columns={
    "variable":    "Variable",
    "boot_mean":   "Bootstrap Mean",
    "ci_lower_95": "95% CI Lower",
    "ci_upper_95": "95% CI Upper",
    "n_valid":     "Bootstrap Iterations",
    "model":       "Model",
})
boot_f.to_csv("outputs/powerbi/page3_bootstrap_cis.csv", index=False)
print("  Saved: page3_bootstrap_cis.csv")

# ── 3c. F-test summary ────────────────────────────────────────────────────
from scipy.stats import f as f_dist

models_csv = pd.read_csv("outputs/tables/model_comparison.csv")
row_a_star = models_csv[models_csv["Model"].str.contains("NLP sample", regex=False)].iloc[0]
row_f      = models_csv[models_csv["Model"].str.contains("Model F", regex=False)].iloc[0]

r2_full  = float(row_f["OLS R²"])
r2_restr = float(row_a_star["OLS R²"])
n        = int(row_f["N (countries)"])
k_full   = int(row_f["Predictors used"])
k_restr  = int(row_a_star["Predictors used"])
q        = k_full - k_restr
df2      = n - k_full - 1
delta_r2 = r2_full - r2_restr
f_stat   = (delta_r2 / q) / ((1 - r2_full) / df2)
p_value  = float(1 - f_dist.cdf(f_stat, q, df2))
partial  = delta_r2 / (1 - r2_restr)


def sig_label(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return "n.s."


ftest_df = pd.DataFrame([{
    "Test":          "Nested F-test: Model A* vs Model F",
    "Sample N":      n,
    "df1 (extra vars)": q,
    "df2":           df2,
    "F statistic":   round(f_stat, 3),
    "p-value":       round(p_value, 4),
    "Significance":  sig_label(p_value),
    "Delta R²":      round(delta_r2, 3),
    "Partial R²":    round(partial, 3),
    "Adj R² Model A*": round(float(row_a_star["OLS Adj R²"]), 3),
    "Adj R² Model F":  round(float(row_f["OLS Adj R²"]), 3),
}])
ftest_df.to_csv("outputs/powerbi/page3_ftest_summary.csv", index=False)
print("  Saved: page3_ftest_summary.csv")


# ============================================================
# Page 4 — Country map (choropleth)
# ============================================================
# One table: country ISO3, name, cereal availability,
#            region, income group, and key predictors
# ============================================================

print("\n[Page 4] Country map data...")

master_dv = pd.read_csv("data/processed/master_dataset_with_dv.csv")

MAP_COLS = [
    "country_code", "country_name",
    "cereal_availability_kg_pc",
    "cereal_yield_kg_per_ha",
    "gdp_per_capita_usd",
    "rural_electricity_access_pct",
    "cereal_loss_pct",
    "trade_pct_gdp",
    "fertiliser_efficiency",
    "rural_population_pct",
]

# I build the list of existing map columns with an explicit for loop
existing_map = []
for c in MAP_COLS:
    if c in master_dv.columns:
        existing_map.append(c)

page4 = master_dv[existing_map].dropna(subset=["cereal_availability_kg_pc"]).copy()

# Human-readable column names
page4 = page4.rename(columns={
    "country_code":               "ISO3 Code",
    "country_name":               "Country",
    "cereal_availability_kg_pc":  "Cereal Availability (kg/person/yr)",
    "cereal_yield_kg_per_ha":     "Cereal Yield (kg/ha)",
    "gdp_per_capita_usd":         "GDP per Capita (USD)",
    "rural_electricity_access_pct": "Rural Electricity Access (%)",
    "cereal_loss_pct":            "Post-Harvest Loss (%)",
    "trade_pct_gdp":              "Trade % GDP",
    "fertiliser_efficiency":      "Fertiliser Efficiency",
    "rural_population_pct":       "Rural Population (%)",
})

# Quintile bands for the choropleth colour scale
dv_col = "Cereal Availability (kg/person/yr)"
if dv_col in page4.columns:
    page4["Availability Band"] = pd.qcut(
        page4[dv_col],
        q=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    )

page4.to_csv("outputs/powerbi/page4_country_map.csv", index=False)
print(f"  Saved: page4_country_map.csv ({len(page4)} countries)")


# ============================================================
# Page 5 — Robustness
# ============================================================
# Two tables:
#   5a. Robustness spec comparison (7 specs, key coefficients)
#   5b. Cook's Distance flagged countries
# ============================================================

print("\n[Page 5] Robustness tables...")

# ── 5a. Robustness specs (reshaped for Power BI) ──────────────────────────
rob_raw = pd.read_csv("outputs/tables/robustness_specifications.csv")

# Keep only the summary columns + the three most important predictors
KEY_PREDICTORS = [
    "gdp_per_capita_usd",
    "livestock_production_index",
    "fertiliser_kg_per_ha",
    "agri_employment_pct",
]
page5_cols = ["Specification", "N", "R²", "Adj R²"]
for pred in KEY_PREDICTORS:
    coef_col = pred + "_coef"
    sig_col  = pred + "_sig"
    if coef_col in rob_raw.columns:
        page5_cols += [coef_col, sig_col]

# I build the final column list with an explicit for loop
page5_cols_existing = []
for c in page5_cols:
    if c in rob_raw.columns:
        page5_cols_existing.append(c)

page5_rob = rob_raw[page5_cols_existing].copy()

# Rename for readability
rename_map = {}
for pred in KEY_PREDICTORS:
    label = pred.replace("_", " ").title()
    if pred + "_coef" in page5_rob.columns:
        rename_map[pred + "_coef"] = label + " Coef"
    if pred + "_sig" in page5_rob.columns:
        rename_map[pred + "_sig"]  = label + " Sig"
page5_rob = page5_rob.rename(columns=rename_map)
page5_rob.to_csv("outputs/powerbi/page5_robustness_specs.csv", index=False)
print("  Saved: page5_robustness_specs.csv")

# ── 5b. Cook's outliers flagged countries ─────────────────────────────────
# Load master and re-compute Cook's Distance to get the flagged list
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

LOG_COLS_E = [
    "cereal_availability_kg_pc", "gdp_per_capita_usd",
    "cereal_yield_kg_per_ha", "fertiliser_kg_per_ha",
]
MODEL_A_VARS = [
    "cereal_yield_kg_per_ha", "fertiliser_kg_per_ha", "arable_land_pct",
    "gdp_per_capita_usd", "rural_population_pct",
    "agri_employment_pct", "livestock_production_index",
]
DV = "cereal_availability_kg_pc"

master_e = master_dv.copy()
for col in LOG_COLS_E:
    if col in master_e.columns:
        master_e[col] = np.log1p(master_e[col].clip(lower=0))

needed = MODEL_A_VARS + [DV, "country_name", "country_code"]

# I build the needed columns list with an explicit for loop
needed_existing = []
for c in needed:
    if c in master_e.columns:
        needed_existing.append(c)

working_e = master_e[needed_existing].dropna().reset_index(drop=True)
X_e = working_e[MODEL_A_VARS]
y_e = working_e[DV]
fit_e = sm.OLS(y_e, sm.add_constant(X_e)).fit()
cooks_d, _ = OLSInfluence(fit_e).cooks_distance
threshold = 4 / len(working_e)

cooks_df = working_e[["country_code", "country_name"]].copy()
cooks_df["Cooks D"] = cooks_d.round(4)
cooks_df["Threshold"] = round(threshold, 4)
cooks_df["Flagged"] = (cooks_d > threshold).map({True: "Yes", False: "No"})
cooks_df = cooks_df.rename(columns={
    "country_code": "ISO3 Code",
    "country_name": "Country",
})
cooks_df = cooks_df.sort_values("Cooks D", ascending=False)
cooks_df.to_csv("outputs/powerbi/page5_cooks_distance.csv", index=False)
print(f"  Saved: page5_cooks_distance.csv ({(cooks_d > threshold).sum()} flagged countries)")

# ── Model F robustness across specs ───────────────────────────────────────
if os.path.exists("outputs/tables/robustness_model_f.csv"):
    rob_f_raw = pd.read_csv("outputs/tables/robustness_model_f.csv")
    rob_f_cols = ["Specification", "N", "R²", "Adj R²"]
    for pred in ["rural_electricity_access_pct", "cereal_loss_pct", "trade_pct_gdp"]:
        for suffix in ["_coef", "_sig"]:
            if pred + suffix in rob_f_raw.columns:
                rob_f_cols.append(pred + suffix)

    # I build the existing columns list with an explicit for loop
    rob_f_cols_existing = []
    for c in rob_f_cols:
        if c in rob_f_raw.columns:
            rob_f_cols_existing.append(c)

    page5_f = rob_f_raw[rob_f_cols_existing].copy()
    page5_f = page5_f.rename(columns={
        "rural_electricity_access_pct_coef": "Rural Electricity Coef",
        "rural_electricity_access_pct_sig":  "Rural Electricity Sig",
        "cereal_loss_pct_coef":              "Post-Harvest Loss Coef",
        "cereal_loss_pct_sig":               "Post-Harvest Loss Sig",
        "trade_pct_gdp_coef":                "Trade % GDP Coef",
        "trade_pct_gdp_sig":                 "Trade % GDP Sig",
    })
    page5_f.to_csv("outputs/powerbi/page5_model_f_robustness.csv", index=False)
    print("  Saved: page5_model_f_robustness.csv")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("STEP 10 COMPLETE — Power BI data exports")
print("=" * 60)
print("""
All files saved to outputs/powerbi/:

PAGE 1 — Overview
  page1_model_performance.csv   Models A/B/C/F/A* with R², Adj R², CV R²
  page1_corpus_summary.csv      Paper counts by source and alignment level

PAGE 2 — NLP Findings
  page2_nmf_topics.csv          7 NMF topics with keywords and dominant paper count
  page2_theme_variable_map.csv  NLP theme -> proxy variable -> data source
  page2_tfidf_keywords.csv      Top 30 TF-IDF keywords (for word cloud visual)

PAGE 3 — Empirical Results
  page3_nlp_synthesis.csv       Model F: NLP themes with coefficients, p-values, direction
  page3_bootstrap_cis.csv       Bootstrap 95% CIs for all 5 NLP predictors
  page3_ftest_summary.csv       Nested F-test: F(5,147)=3.649, p=0.004 (***)

PAGE 4 — Country Map
  page4_country_map.csv         160 countries: cereal availability + key indicators
                                (use ISO3 Code for Power BI map visual)

PAGE 5 — Robustness
  page5_robustness_specs.csv    7-spec table: N, R², Adj R², key coefficients
  page5_cooks_distance.csv      All countries with Cook's D (flagged = Yes/No)
  page5_model_f_robustness.csv  Model F: NLP variables across 4 robustness specs

TO BUILD IN POWER BI:
  1. Open Power BI Desktop
  2. Get Data > Text/CSV > select all files in outputs/powerbi/
  3. Use ISO3 Code in page4_country_map.csv for the choropleth map
  4. Bar chart: page1_model_performance.csv — OLS R² and Adj R² by model
  5. Bar chart with error bars: page3_bootstrap_cis.csv — CI Lower/Upper
  6. Table visual: page3_nlp_synthesis.csv — conditional formatting on p-value
  7. Word cloud (custom visual): page2_tfidf_keywords.csv
""")
