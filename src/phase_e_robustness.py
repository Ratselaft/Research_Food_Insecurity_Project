# ============================================================
# PHASE E — Outlier detection and robustness checks
# ============================================================
#
# What this file does:
#   Phase D fitted our three models and found significant results.
#   But we need to check: are those results reliable, or are
#   they being driven by one or two unusual countries?
#
#   We do two types of checks:
#
#   1. OUTLIER DETECTION
#      a) Cook's Distance — finds countries whose values pull
#         the OLS regression line far from where it would be
#         without them. Think of it as: which country, if
#         removed, would change our results the most?
#      b) Isolation Forest — a machine learning method that
#         finds countries that are unusual across ALL variables
#         at the same time (multivariate outliers).
#
#   2. ROBUSTNESS SPECIFICATIONS (5 total)
#      We re-run Model A five different ways. If the main
#      findings hold up in all five, we can trust them.
#
#      Spec 1 — Baseline (exactly as in Phase D)
#      Spec 2 — Add climate variable (average precipitation)
#      Spec 3 — Log-transform the dependent variable
#      Spec 4 — Drop Cook's Distance outliers, re-run
#      Spec 5 — Drop Isolation Forest outliers, re-run
#
#   A finding is called "robust" if the sign and significance
#   of the key predictors (internet_users_pct, cereal_loss_pct)
#   stay the same across all five specifications.
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import requests, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import statsmodels.api        as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.ensemble         import IsolationForest
from sklearn.preprocessing    import StandardScaler

os.makedirs("outputs/tables",  exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

RANDOM_SEED = 42

print("Starting Phase E — outlier detection and robustness checks...")
print("=" * 60)


# ============================================================
# STEP 1: Load the master dataset (with dependent variable)
# ============================================================

print("\n[1] Loading data...")

master = pd.read_csv("data/processed/master_dataset_with_dv.csv")
print(f"  Countries loaded: {len(master)}")

# These are the predictors we use in Model A
MODEL_A_VARS = [
    "cereal_yield_kg_per_ha",
    "fertiliser_kg_per_ha",
    "arable_land_pct",
    "gdp_per_capita_usd",
    "internet_users_pct",
]
DV = "undernourishment_pct"

# Log-transform the skewed predictors (same as Phase D)
LOG_COLS = ["gdp_per_capita_usd", "cereal_yield_kg_per_ha", "fertiliser_kg_per_ha"]
for col in LOG_COLS:
    if col in master.columns:
        master[col] = np.log1p(master[col].clip(lower=0))


# ============================================================
# STEP 2: Download climate variable — average precipitation
# ============================================================
# Average precipitation (mm per year) is a proxy for climate
# conditions. Countries with less rainfall tend to face more
# food insecurity, especially in dryland cereal farming regions.

print("\n[2] Downloading climate variable (average precipitation)...")

REGIONAL_CODES = {
    'AFE','AFW','ARB','CEB','CSS','EAP','EAR','EAS','ECA','ECS',
    'EMU','EUU','FCS','HIC','HPC','IBD','IBT','IDA','IDB','IDX',
    'LAC','LCN','LDC','LIC','LMC','LMY','LTE','MEA','MIC','MNA',
    'NAC','OED','OSS','PRE','PSS','PST','SAS','SSA','SSF','SST',
    'TEA','TEC','TLA','TMN','TSA','TSS','UMC','WLD','XZN'
}

try:
    response = requests.get(
        "https://api.worldbank.org/v2/country/all/indicator/AG.LND.PRCP.MM",
        params={"mrv": 1, "format": "json", "per_page": 300},
        timeout=30,
    )
    data = response.json()
    rows = []
    for entry in (data[1] or []):
        code = entry.get("countryiso3code", "")
        val  = entry.get("value")
        if val is not None and code and code not in REGIONAL_CODES:
            rows.append({"country_code": code, "avg_precipitation_mm": val})
    precip_df = pd.DataFrame(rows)
    print(f"  Precipitation data: {len(precip_df)} countries")

    # Merge into master
    master = master.merge(precip_df, on="country_code", how="left")
    print(f"  Countries with precipitation: {master['avg_precipitation_mm'].notna().sum()}")

except Exception as e:
    print(f"  Could not download — {e}")
    master["avg_precipitation_mm"] = np.nan


# ============================================================
# STEP 3: Prepare working dataset (complete cases for Model A)
# ============================================================

# Keep only countries where ALL Model A variables + DV are present
needed_cols = MODEL_A_VARS + [DV, "country_name", "country_code"]
working = master[needed_cols].dropna().copy().reset_index(drop=True)

print(f"\n[3] Working dataset for Model A: {len(working)} countries")

X_base = working[MODEL_A_VARS]
y      = working[DV]


# ============================================================
# STEP 4: Cook's Distance — find influential countries
# ============================================================
# Cook's Distance measures how much the regression results
# would change if we removed a single country.
#
# Rule of thumb: any country with Cook's D > 4/N is "influential"
# (N = number of countries in the regression).
# We flag these and later re-run the model without them.

print("\n[4] Computing Cook's Distance...")

X_const = sm.add_constant(X_base)
ols_fit = sm.OLS(y, X_const).fit()
influence   = OLSInfluence(ols_fit)
cooks_d, _  = influence.cooks_distance

threshold    = 4 / len(working)          # standard rule: 4 / N
outlier_mask = cooks_d > threshold
n_outliers   = outlier_mask.sum()

print(f"  Cook's D threshold (4/N = 4/{len(working)}): {threshold:.4f}")
print(f"  Countries above threshold: {n_outliers}")

# Show which countries are influential
if n_outliers > 0:
    outlier_countries = working[outlier_mask][["country_name", DV]].copy()
    outlier_countries["cooks_d"] = cooks_d[outlier_mask]
    outlier_countries = outlier_countries.sort_values("cooks_d", ascending=False)
    print("\n  Influential countries (Cook's D > threshold):")
    print(outlier_countries.to_string(index=False))

# Save a Cook's Distance plot
plt.figure(figsize=(10, 4))
plt.stem(range(len(working)), cooks_d, markerfmt="o", linefmt="grey", basefmt="k-")
plt.axhline(threshold, color="red", linestyle="--",
            label=f"Threshold (4/N = {threshold:.3f})")
# Label the flagged countries
for i, (name, d) in enumerate(zip(working["country_name"], cooks_d)):
    if d > threshold:
        plt.text(i, d + 0.002, name[:10], fontsize=7, ha="center", color="red")
plt.xlabel("Country index")
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance — influential countries in Model A")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/cooks_distance.png", dpi=150)
plt.close()
print("\n  Cook's Distance chart saved → outputs/figures/cooks_distance.png")


# ============================================================
# STEP 5: Isolation Forest — multivariate outlier detection
# ============================================================
# Isolation Forest is a machine learning algorithm.
# Instead of looking at one variable at a time, it looks at
# ALL variables together. A country is an outlier if it is
# unusual across many dimensions at once.
#
# It works by randomly splitting the data and seeing which
# countries are easiest to isolate — unusual countries are
# isolated in fewer steps.

print("\n[5] Running Isolation Forest...")

# Standardise all predictors so no single variable dominates
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_base)

iso_forest = IsolationForest(
    n_estimators  = 200,
    contamination = 0.10,    # we expect about 10% of countries to be outliers
    random_state  = RANDOM_SEED,
)
iso_labels = iso_forest.fit_predict(X_scaled)
# IsolationForest returns -1 for outliers and +1 for normal observations
iso_outlier_mask = iso_labels == -1
n_iso_outliers   = iso_outlier_mask.sum()

print(f"  Isolation Forest found {n_iso_outliers} multivariate outliers")

iso_outlier_countries = working[iso_outlier_mask]["country_name"].tolist()
print(f"  Countries flagged: {iso_outlier_countries}")


# ============================================================
# STEP 6: Five robustness specifications
# ============================================================
# We run Model A five different ways. If the key predictors
# stay significant across all five, our findings are robust.

print(f"\n[6] Running 5 robustness specifications...")
print("=" * 60)

def fit_ols_spec(X, y, spec_name):
    """Fit OLS and return a one-row summary dictionary."""
    X_c = sm.add_constant(X)
    m   = sm.OLS(y, X_c).fit()
    row = {
        "Specification": spec_name,
        "N":             int(m.nobs),
        "R²":            round(m.rsquared, 3),
        "Adj R²":        round(m.rsquared_adj, 3),
    }
    # Collect key coefficients and their significance
    for var in X.columns:
        coef = m.params.get(var, np.nan)
        pval = m.pvalues.get(var, np.nan)
        sig  = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        row[f"{var}_coef"] = round(coef, 3)
        row[f"{var}_sig"]  = sig
    return row, m

spec_results = []

# ── Spec 1: Baseline (same as Phase D Model A) ────────────────
print("\nSpec 1 — Baseline (full sample, Model A predictors)")
row1, m1 = fit_ols_spec(X_base, y, "Spec 1 — Baseline")
spec_results.append(row1)
print(f"  N={row1['N']}  R²={row1['R²']}  Adj R²={row1['Adj R²']}")

# ── Spec 2: Add climate variable (precipitation) ──────────────
print("\nSpec 2 — Add average precipitation (climate proxy)")
precip_col = "avg_precipitation_mm"
working_precip = master[MODEL_A_VARS + [DV, precip_col, "country_name"]].dropna().copy()
if len(working_precip) >= 30:
    X2 = working_precip[MODEL_A_VARS + [precip_col]]
    # Log precipitation as it is also skewed
    X2 = X2.copy()
    X2[precip_col] = np.log1p(X2[precip_col].clip(lower=0))
    y2 = working_precip[DV]
    row2, m2 = fit_ols_spec(X2, y2, "Spec 2 — +Precipitation")
    spec_results.append(row2)
    precip_coef = m2.params.get(precip_col, np.nan)
    precip_pval = m2.pvalues.get(precip_col, np.nan)
    print(f"  N={row2['N']}  R²={row2['R²']}  Adj R²={row2['Adj R²']}")
    print(f"  precipitation coef={precip_coef:.3f}  p={precip_pval:.4f}")
else:
    print(f"  Skipped — only {len(working_precip)} complete rows")

# ── Spec 3: Log-transform the dependent variable ──────────────
print("\nSpec 3 — Log-transform undernourishment (DV)")
# Some researchers prefer log(DV) when the outcome is a percentage
# that is very skewed. This checks whether our results change.
y3 = np.log1p(y.clip(lower=0))
row3, m3 = fit_ols_spec(X_base, y3, "Spec 3 — Log DV")
spec_results.append(row3)
print(f"  N={row3['N']}  R²={row3['R²']}  Adj R²={row3['Adj R²']}")
print(f"  Note: coefficients now interpreted as % changes in log(undernourishment)")

# ── Spec 4: Drop Cook's Distance outliers ─────────────────────
print(f"\nSpec 4 — Drop Cook's Distance outliers ({n_outliers} countries removed)")
X4 = X_base[~outlier_mask]
y4 = y[~outlier_mask]
if len(X4) >= 30:
    row4, m4 = fit_ols_spec(X4, y4, f"Spec 4 — No Cook outliers (N-{n_outliers})")
    spec_results.append(row4)
    print(f"  N={row4['N']}  R²={row4['R²']}  Adj R²={row4['Adj R²']}")
    if n_outliers > 0:
        print(f"  Removed: {', '.join(outlier_countries['country_name'].tolist())}")
else:
    print(f"  Skipped — too few countries remaining after removal")

# ── Spec 5: Drop Isolation Forest outliers ────────────────────
print(f"\nSpec 5 — Drop Isolation Forest outliers ({n_iso_outliers} countries removed)")
X5 = X_base[~iso_outlier_mask]
y5 = y[~iso_outlier_mask]
if len(X5) >= 30:
    row5, m5 = fit_ols_spec(X5, y5, f"Spec 5 — No ISO outliers (N-{n_iso_outliers})")
    spec_results.append(row5)
    print(f"  N={row5['N']}  R²={row5['R²']}  Adj R²={row5['Adj R²']}")
else:
    print(f"  Skipped — too few countries remaining after removal")


# ============================================================
# STEP 7: Build the robustness summary table
# ============================================================
# This table shows each specification as a column and each
# predictor as a row. You can see at a glance whether the
# same variables are significant across all five specs.

print(f"\n[7] Building robustness summary table...")

# The variables we want to track across specs
TRACK_VARS = MODEL_A_VARS + (
    ["avg_precipitation_mm"] if len(spec_results) > 1 else []
)

# Print a readable summary
print(f"\n{'Variable':<30}", end="")
for r in spec_results:
    label = r['Specification'].split("—")[0].strip() + "\n" + r['Specification'].split("—")[1].strip() if "—" in r['Specification'] else r['Specification']
    short = r['Specification'][:18]
    print(f" {short:>18}", end="")
print()
print("-" * (30 + 19 * len(spec_results)))

for var in TRACK_VARS:
    coef_key = f"{var}_coef"
    sig_key  = f"{var}_sig"
    print(f"{var:<30}", end="")
    for r in spec_results:
        coef = r.get(coef_key, "")
        sig  = r.get(sig_key, "")
        cell = f"{coef}{sig}" if coef != "" else "—"
        print(f" {cell:>18}", end="")
    print()

print()
print(f"{'N':<30}", end="")
for r in spec_results:
    print(f" {r['N']:>18}", end="")
print()

print(f"{'R²':<30}", end="")
for r in spec_results:
    print(f" {r['R²']:>18}", end="")
print()

# Save full robustness table as CSV
rob_df = pd.DataFrame(spec_results)
rob_df.to_csv("outputs/tables/robustness_specifications.csv", index=False)
print(f"\nFull robustness table saved → outputs/tables/robustness_specifications.csv")


# ============================================================
# STEP 8: Robustness coefficient plot
# ============================================================
# A visual way to show that the key coefficients stay stable.
# We plot the coefficient of internet_users_pct across all specs.
# If the bars all point the same direction and stay similar,
# the result is robust.

print("\n[8] Saving robustness coefficient plot...")

key_vars_to_plot = [v for v in ["internet_users_pct", "gdp_per_capita_usd",
                                  "cereal_yield_kg_per_ha"]
                    if any(f"{v}_coef" in r for r in spec_results)]

fig, axes = plt.subplots(1, len(key_vars_to_plot), figsize=(5 * len(key_vars_to_plot), 5))
if len(key_vars_to_plot) == 1:
    axes = [axes]

colors = ["#4472C4", "#ED7D31", "#70AD47", "#FF0000", "#7030A0"]

for ax, var in zip(axes, key_vars_to_plot):
    spec_labels = [r["Specification"].replace("—", "\n") for r in spec_results
                   if f"{var}_coef" in r]
    coefs       = [r[f"{var}_coef"] for r in spec_results if f"{var}_coef" in r]
    sigs        = [r.get(f"{var}_sig", "") for r in spec_results if f"{var}_coef" in r]

    bars = ax.bar(range(len(coefs)), coefs,
                  color=[colors[i] for i in range(len(coefs))],
                  edgecolor="black", linewidth=0.5)

    # Add significance stars on top of each bar
    for i, (bar, sig) in enumerate(zip(bars, sigs)):
        h = bar.get_height()
        offset = 0.01 if h >= 0 else -0.04
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + offset, sig, ha="center", va="bottom", fontsize=11)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Coefficient of\n{var}", fontsize=10)
    ax.set_xticks(range(len(spec_labels)))
    ax.set_xticklabels([f"S{i+1}" for i in range(len(spec_labels))], fontsize=9)
    ax.set_ylabel("OLS coefficient")

plt.suptitle("Robustness check: key coefficients across 5 specifications\n"
             "(S1=Baseline, S2=+Precip, S3=Log DV, S4=No Cook, S5=No ISO)",
             fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig("outputs/figures/robustness_coefficients.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Coefficient stability chart saved → outputs/figures/robustness_coefficients.png")


# ============================================================
# STEP 9: Plain English interpretation
# ============================================================

print(f"\n{'='*60}")
print("PHASE E COMPLETE — Key Findings")
print(f"{'='*60}")
print("""
What we checked:
  1. Cook's Distance — identified countries pulling results
  2. Isolation Forest — found multivariate unusual countries
  3. Five robustness specifications tested

How to interpret this for your dissertation:
  - If internet_users_pct stays negative and significant
    across most specs → it is a robust predictor
  - If a coefficient flips sign or loses significance in
    Spec 4 or Spec 5 → the result is sensitive to outliers
    and you should mention this as a limitation
  - If R² stays similar across specs → model is stable

Outputs saved:
  outputs/figures/cooks_distance.png
  outputs/figures/robustness_coefficients.png
  outputs/tables/robustness_specifications.csv

Note on governance variable (Rule of Law):
  The World Governance Indicators are not available through
  the standard World Bank REST API. To add them as Spec 6:
  Download manually from: https://info.worldbank.org/governance/wgi/
  Save as data/raw/wgi_rule_of_law.csv and merge on country_code.
""")
