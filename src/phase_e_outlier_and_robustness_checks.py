# ============================================================
# I'm running outlier detection and robustness checks
# ============================================================
#
# What I'm doing here:
#   Phase D gave me results I'm happy with. But I need to check
#   whether those results are reliable, or whether they're being
#   driven by just one or two unusual countries.
#
#   I do two types of checks:
#
#   1. OUTLIER DETECTION
#      a) Cook's Distance — finds countries whose data pulls the
#         regression line further than normal. If I remove them
#         and my results don't change, I can trust the findings.
#      b) Isolation Forest — a machine learning method that finds
#         countries that are unusual across ALL variables at once.
#
#   2. FIVE ROBUSTNESS SPECIFICATIONS
#      I re-run Model A five different ways. If the main findings
#      hold up across all five, I can call them robust.
#
#      Spec 1 — Baseline (exactly as in Phase D)
#      Spec 2 — Add climate variable (average precipitation)
#      Spec 3 — Log-transform the dependent variable
#      Spec 4 — Drop Cook's Distance outliers and re-run
#      Spec 5 — Drop Isolation Forest outliers and re-run
# ============================================================

# I suppress warning messages to keep the output tidy
import warnings
warnings.filterwarnings('ignore')

# I need requests to download data from the internet
import requests

# I need os to create folders
import os

# I need numpy for mathematical operations
import numpy as np

# I need pandas to work with tables
import pandas as pd

# I need matplotlib to draw charts
import matplotlib
matplotlib.use('Agg')   # I save charts to files, so I don't need a screen
import matplotlib.pyplot as plt

# I need statsmodels for OLS regression
import statsmodels.api as sm

# OLSInfluence lets me compute Cook's Distance
from statsmodels.stats.outliers_influence import OLSInfluence

# IsolationForest is the machine learning outlier detector
from sklearn.ensemble import IsolationForest

# StandardScaler normalises all my variables to the same scale
from sklearn.preprocessing import StandardScaler

# I make sure my output folders exist before I try to save anything
os.makedirs("outputs/tables",  exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# I set my random seed so results are the same every time I run this
RANDOM_SEED = 42

# I let the user know I'm starting
print("Starting Phase E — outlier detection and robustness checks...")
print("=" * 60)


# ============================================================
# Step 1: I'm loading the data
# ============================================================

print("\n[1] Loading data...")

# I load the master dataset with the dependent variable included
master = pd.read_csv("data/processed/master_dataset_with_dv.csv")

# I let the user know how many countries I've loaded
print(f"  Countries loaded: {len(master)}")

# I list the predictors I use in Model A
MODEL_A_VARS = [
    "cereal_yield_kg_per_ha",
    "fertiliser_kg_per_ha",
    "arable_land_pct",
    "gdp_per_capita_usd",
    "internet_users_pct",
]

# This is the variable I'm trying to predict
DV = "undernourishment_pct"

# I log-transform the skewed predictors (same as I did in Phase D)
LOG_COLS = ["gdp_per_capita_usd", "cereal_yield_kg_per_ha", "fertiliser_kg_per_ha"]

# I go through each column that needs log-transforming
for col in LOG_COLS:
    # I check that this column exists in my data
    if col in master.columns:
        # I apply log1p — this adds 1 before taking the log so 0 values don't cause an error
        master[col] = np.log1p(master[col].clip(lower=0))


# ============================================================
# Step 2: I'm downloading the climate variable
# ============================================================
# Average precipitation (mm per year) is my climate proxy.
# Countries with less rainfall tend to face more food insecurity.

print("\n[2] Downloading climate variable (average precipitation)...")

# I define the set of World Bank regional codes that are not real countries
REGIONAL_CODES = set()
regional_list = ['AFE','AFW','ARB','CEB','CSS','EAP','EAR','EAS','ECA','ECS',
                 'EMU','EUU','FCS','HIC','HPC','IBD','IBT','IDA','IDB','IDX',
                 'LAC','LCN','LDC','LIC','LMC','LMY','LTE','MEA','MIC','MNA',
                 'NAC','OED','OSS','PRE','PSS','PST','SAS','SSA','SSF','SST',
                 'TEA','TEC','TLA','TMN','TSA','TSS','UMC','WLD','XZN']

# I add each code to my set one by one
for code in regional_list:
    REGIONAL_CODES.add(code)

# I try to download the precipitation data
try:
    # I send the request to the World Bank API
    response = requests.get(
        "https://api.worldbank.org/v2/country/all/indicator/AG.LND.PRCP.MM",
        params={"mrv": 1, "format": "json", "per_page": 300},
        timeout=30,
    )

    # I convert the response from JSON into Python data
    data = response.json()

    # I'll collect one row per real country here
    rows = []

    # I go through each entry in the data
    for entry in (data[1] or []):
        # I get the country code
        code = entry.get("countryiso3code", "")

        # I get the precipitation value
        val = entry.get("value")

        # I only keep real countries with an actual value
        if val is not None and code and code not in REGIONAL_CODES:
            one_row = {}
            one_row["country_code"]        = code
            one_row["avg_precipitation_mm"] = val
            rows.append(one_row)

    # I turn my rows into a table
    precip_df = pd.DataFrame(rows)

    # I print how many countries I got
    print(f"  Precipitation data: {len(precip_df)} countries")

    # I merge the precipitation data into my master table
    master = master.merge(precip_df, on="country_code", how="left")

    # I print how many of my countries now have precipitation data
    print(f"  Countries with precipitation: {master['avg_precipitation_mm'].notna().sum()}")

except Exception as e:
    # If the download fails, I just set all values to blank
    print(f"  Could not download — {e}")
    master["avg_precipitation_mm"] = np.nan


# ============================================================
# Step 3: I'm preparing the working dataset for Model A
# ============================================================

# I figure out which columns I need to have complete data for
needed_cols = MODEL_A_VARS + [DV, "country_name", "country_code"]

# I keep only columns that actually exist in my table
cols_that_exist = []
for col in needed_cols:
    if col in master.columns:
        cols_that_exist.append(col)

# I drop any rows where any of these columns has a blank value
working = master[cols_that_exist].dropna().copy()

# I reset the row index so it goes 0, 1, 2, 3...
working = working.reset_index(drop=True)

# I print how many countries made it into my working dataset
print(f"\n[3] Working dataset for Model A: {len(working)} countries")

# I pull out just the predictor columns
X_base = working[MODEL_A_VARS]

# I pull out just the outcome column
y = working[DV]


# ============================================================
# Step 4: I'm computing Cook's Distance
# ============================================================
# Cook's Distance measures how much the regression results would
# change if I removed a single country from the analysis.
# The standard rule: if Cook's D > 4/N, the country is "influential".

print("\n[4] Computing Cook's Distance...")

# I add an intercept column (OLS needs this)
X_const = sm.add_constant(X_base)

# I fit the OLS model on the full working dataset
ols_fit = sm.OLS(y, X_const).fit()

# I create an influence object — this lets me compute Cook's Distance
influence = OLSInfluence(ols_fit)

# I compute Cook's Distance for every country
# cooks_distance returns (distances, p_values) — I only want the distances
cooks_d, _ = influence.cooks_distance

# I calculate the threshold: 4 divided by the number of countries
threshold = 4 / len(working)

# I create a True/False list: True if this country is above the threshold
outlier_mask = cooks_d > threshold

# I count how many countries are above the threshold
n_outliers = outlier_mask.sum()

# I print the threshold and how many countries exceeded it
print(f"  Cook's D threshold (4/N = 4/{len(working)}): {threshold:.4f}")
print(f"  Countries above threshold: {n_outliers}")

# I print which countries are flagged as influential
if n_outliers > 0:
    # I build a table of the outlier countries and their Cook's D values
    outlier_countries = working[outlier_mask][["country_name", DV]].copy()
    outlier_countries["cooks_d"] = cooks_d[outlier_mask]

    # I sort by Cook's D so the most influential is at the top
    outlier_countries = outlier_countries.sort_values("cooks_d", ascending=False)

    print("\n  Influential countries (Cook's D > threshold):")
    print(outlier_countries.to_string(index=False))

# I draw and save the Cook's Distance chart
plt.figure(figsize=(10, 4))

# I draw a stem chart — each country gets a vertical line up to its Cook's D value
plt.stem(range(len(working)), cooks_d,
         markerfmt="o", linefmt="grey", basefmt="k-")

# I draw a red dashed line at the threshold
plt.axhline(threshold, color="red", linestyle="--",
            label=f"Threshold (4/N = {threshold:.3f})")

# I label any countries that are above the threshold on the chart
for i in range(len(working)):
    if cooks_d[i] > threshold:
        country_label = working["country_name"].iloc[i][:10]
        plt.text(i, cooks_d[i] + 0.002, country_label,
                 fontsize=7, ha="center", color="red")

# I label the axes
plt.xlabel("Country index")
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance — influential countries in Model A")
plt.legend()
plt.tight_layout()

# I save the chart
plt.savefig("outputs/figures/cooks_distance.png", dpi=150)
plt.close()

print("\n  Cook's Distance chart saved → outputs/figures/cooks_distance.png")


# ============================================================
# Step 5: I'm running the Isolation Forest
# ============================================================
# The Isolation Forest looks at ALL variables together to find
# countries that are unusual across many dimensions at once.
# It returns -1 for outliers and +1 for normal countries.

print("\n[5] Running Isolation Forest...")

# I standardise all my predictors so no single variable dominates
# StandardScaler centres each variable on 0 and scales it to unit variance
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_base)

# I create the Isolation Forest
iso_forest = IsolationForest(
    n_estimators  = 200,    # I use 200 trees in the forest
    contamination = 0.10,   # I expect about 10% of countries to be outliers
    random_state  = RANDOM_SEED,
)

# I fit the forest and get a label for each country (-1 or +1)
iso_labels = iso_forest.fit_predict(X_scaled)

# I create a True/False mask: True where the label is -1 (outlier)
iso_outlier_mask = iso_labels == -1

# I count the outliers
n_iso_outliers = iso_outlier_mask.sum()

# I print the results
print(f"  Isolation Forest found {n_iso_outliers} multivariate outliers")

# I collect the names of the outlier countries
iso_outlier_names = []
for i in range(len(working)):
    if iso_outlier_mask[i]:
        iso_outlier_names.append(working["country_name"].iloc[i])

print(f"  Countries flagged: {iso_outlier_names}")


# ============================================================
# Step 6: I'm running the five robustness specifications
# ============================================================

print(f"\n[6] Running 5 robustness specifications...")
print("=" * 60)


def fit_ols_spec(X, y, spec_name):
    # I add an intercept column
    X_c = sm.add_constant(X)

    # I fit the OLS model
    m = sm.OLS(y, X_c).fit()

    # I build a summary dictionary for this specification
    row = {}
    row["Specification"] = spec_name
    row["N"]             = int(m.nobs)
    row["R²"]            = round(m.rsquared, 3)
    row["Adj R²"]        = round(m.rsquared_adj, 3)

    # I go through each predictor and record its coefficient and significance
    for var in X.columns:
        # I get the coefficient for this variable
        coef = m.params.get(var, np.nan)

        # I get the p-value for this variable
        pval = m.pvalues.get(var, np.nan)

        # I work out the significance stars
        if pval < 0.01:
            sig = "***"
        elif pval < 0.05:
            sig = "**"
        elif pval < 0.10:
            sig = "*"
        else:
            sig = ""

        # I save the coefficient and significance for this variable
        row[var + "_coef"] = round(coef, 3)
        row[var + "_sig"]  = sig

    # I return the summary row and the fitted model
    return row, m


# I'll collect the results from all five specs in this list
spec_results = []

# ── Spec 1: I'm running the baseline (same as Phase D Model A) ─────────────
print("\nSpec 1 — Baseline (full sample, Model A predictors)")
row1, m1 = fit_ols_spec(X_base, y, "Spec 1 — Baseline")
spec_results.append(row1)
print(f"  N={row1['N']}  R²={row1['R²']}  Adj R²={row1['Adj R²']}")


# ── Spec 2: I'm adding the climate variable (average precipitation) ─────────
print("\nSpec 2 — Add average precipitation (climate proxy)")

# I get the name of the precipitation column
precip_col = "avg_precipitation_mm"

# I need all Model A vars plus precipitation plus the DV and country name
cols_for_spec2 = MODEL_A_VARS + [DV, precip_col, "country_name"]

# I keep only countries that have all of these columns filled in
working_precip = master[cols_for_spec2].dropna().copy()

# I only run this if I have at least 30 countries
if len(working_precip) >= 30:
    # I make a copy of the predictor columns including precipitation
    X2 = working_precip[MODEL_A_VARS + [precip_col]].copy()

    # I log-transform precipitation too since it's also skewed
    X2[precip_col] = np.log1p(X2[precip_col].clip(lower=0))

    # I get the outcome column
    y2 = working_precip[DV]

    # I fit the OLS model for Spec 2
    row2, m2 = fit_ols_spec(X2, y2, "Spec 2 — +Precipitation")
    spec_results.append(row2)

    # I get the precipitation coefficient and p-value to highlight
    precip_coef = m2.params.get(precip_col, np.nan)
    precip_pval = m2.pvalues.get(precip_col, np.nan)

    # I print the results
    print(f"  N={row2['N']}  R²={row2['R²']}  Adj R²={row2['Adj R²']}")
    print(f"  precipitation coef={precip_coef:.3f}  p={precip_pval:.4f}")
else:
    print(f"  Skipped — only {len(working_precip)} complete rows")


# ── Spec 3: I'm log-transforming the dependent variable ─────────────────────
print("\nSpec 3 — Log-transform undernourishment (DV)")

# I log-transform the outcome variable
# Some researchers prefer this when the outcome is a skewed percentage
y3 = np.log1p(y.clip(lower=0))

# I fit the OLS with the same predictors but the log-transformed outcome
row3, m3 = fit_ols_spec(X_base, y3, "Spec 3 — Log DV")
spec_results.append(row3)

print(f"  N={row3['N']}  R²={row3['R²']}  Adj R²={row3['Adj R²']}")
print(f"  Note: coefficients now mean % changes in log(undernourishment)")


# ── Spec 4: I'm dropping Cook's Distance outliers ────────────────────────────
print(f"\nSpec 4 — Drop Cook's Distance outliers ({n_outliers} countries removed)")

# I remove the countries that were flagged by Cook's Distance
X4 = X_base[~outlier_mask]
y4 = y[~outlier_mask]

# I only run if I have at least 30 countries left
if len(X4) >= 30:
    row4, m4 = fit_ols_spec(X4, y4, f"Spec 4 — No Cook outliers (N-{n_outliers})")
    spec_results.append(row4)
    print(f"  N={row4['N']}  R²={row4['R²']}  Adj R²={row4['Adj R²']}")

    # I print the names of the removed countries
    if n_outliers > 0:
        removed_names = outlier_countries["country_name"].tolist()
        print(f"  Removed: {', '.join(removed_names)}")
else:
    print(f"  Skipped — too few countries remaining after removal")


# ── Spec 5: I'm dropping Isolation Forest outliers ───────────────────────────
print(f"\nSpec 5 — Drop Isolation Forest outliers ({n_iso_outliers} countries removed)")

# I remove the countries flagged by the Isolation Forest
X5 = X_base[~iso_outlier_mask]
y5 = y[~iso_outlier_mask]

# I only run if I have at least 30 countries left
if len(X5) >= 30:
    row5, m5 = fit_ols_spec(X5, y5, f"Spec 5 — No ISO outliers (N-{n_iso_outliers})")
    spec_results.append(row5)
    print(f"  N={row5['N']}  R²={row5['R²']}  Adj R²={row5['Adj R²']}")
else:
    print(f"  Skipped — too few countries remaining")


# ============================================================
# Step 7: I'm building the robustness summary table
# ============================================================

print(f"\n[7] Building robustness summary table...")

# I list the variables I want to track across all five specs
TRACK_VARS = MODEL_A_VARS + ["avg_precipitation_mm"]

# I print a header row
print(f"\n{'Variable':<30}", end="")
for r in spec_results:
    short = r["Specification"][:18]
    print(f" {short:>18}", end="")
print()

# I print a dividing line
print("-" * (30 + 19 * len(spec_results)))

# I go through each variable and print its coefficient across all specs
for var in TRACK_VARS:
    coef_key = var + "_coef"
    sig_key  = var + "_sig"

    # I print the variable name
    print(f"{var:<30}", end="")

    # I go through each specification
    for r in spec_results:
        coef = r.get(coef_key, "")
        sig  = r.get(sig_key, "")

        # I build the cell content
        if coef != "":
            cell = str(coef) + str(sig)
        else:
            cell = "—"

        print(f" {cell:>18}", end="")
    print()

# I print N and R² rows at the bottom
print()
print(f"{'N':<30}", end="")
for r in spec_results:
    print(f" {r['N']:>18}", end="")
print()

print(f"{'R²':<30}", end="")
for r in spec_results:
    print(f" {r['R²']:>18}", end="")
print()

# I save the full robustness table as a CSV file
rob_df = pd.DataFrame(spec_results)
rob_df.to_csv("outputs/tables/robustness_specifications.csv", index=False)
print(f"\nFull robustness table saved → outputs/tables/robustness_specifications.csv")


# ============================================================
# Step 8: I'm drawing the coefficient stability chart
# ============================================================

print("\n[8] Saving robustness coefficient plot...")

# I select the key variables I want to show stability for
key_vars_to_plot = []
for v in ["internet_users_pct", "gdp_per_capita_usd", "cereal_yield_kg_per_ha"]:
    # I only include it if it appears in at least one spec result
    if any(v + "_coef" in r for r in spec_results):
        key_vars_to_plot.append(v)

# I set up one subplot per variable
n_subplots = len(key_vars_to_plot)
fig, axes  = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 5))

# If there's only one variable, axes is not a list — I wrap it in a list
if n_subplots == 1:
    axes = [axes]

# I set the bar colours for the five specs
colours = ["#4472C4", "#ED7D31", "#70AD47", "#FF0000", "#7030A0"]

# I go through each variable and draw its bar chart
for ax_index in range(len(key_vars_to_plot)):
    var = key_vars_to_plot[ax_index]
    ax  = axes[ax_index]

    # I collect the coefficient and significance for each spec that has this variable
    spec_labels = []
    coefs       = []
    sigs        = []

    for r in spec_results:
        if var + "_coef" in r:
            spec_labels.append(r["Specification"])
            coefs.append(r[var + "_coef"])
            sigs.append(r.get(var + "_sig", ""))

    # I draw the bars
    bars = ax.bar(range(len(coefs)), coefs,
                  color=[colours[i] for i in range(len(coefs))],
                  edgecolor="black", linewidth=0.5)

    # I add significance stars on top of each bar
    for i in range(len(bars)):
        bar = bars[i]
        sig = sigs[i]
        h   = bar.get_height()

        # I position the star just above the top of the bar
        offset = 0.01 if h >= 0 else -0.04
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + offset, sig, ha="center", va="bottom", fontsize=11)

    # I draw a horizontal line at 0
    ax.axhline(0, color="black", linewidth=0.8)

    # I add a title showing which variable this subplot is for
    ax.set_title(f"Coefficient of\n{var}", fontsize=10)

    # I label the x ticks as S1, S2, etc.
    ax.set_xticks(range(len(spec_labels)))
    ax.set_xticklabels(["S" + str(i + 1) for i in range(len(spec_labels))], fontsize=9)

    # I label the y axis
    ax.set_ylabel("OLS coefficient")

# I add a main title for the whole figure
plt.suptitle("Robustness check: key coefficients across 5 specifications\n"
             "(S1=Baseline, S2=+Precip, S3=Log DV, S4=No Cook, S5=No ISO)",
             fontsize=10, y=1.02)

# I tidy up the layout
plt.tight_layout()

# I save the chart
plt.savefig("outputs/figures/robustness_coefficients.png", dpi=150, bbox_inches="tight")
plt.close()

print("  Coefficient stability chart saved → outputs/figures/robustness_coefficients.png")


# ============================================================
# Step 9: I'm printing the plain English interpretation
# ============================================================

print(f"\n{'='*60}")
print("PHASE E COMPLETE — Key Findings")
print(f"{'='*60}")
print("""
What I checked:
  1. Cook's Distance — identified countries pulling results
  2. Isolation Forest — found multivariate unusual countries
  3. Five robustness specifications tested

How to use this in my dissertation:
  - If internet_users_pct stays negative and significant
    across most specs → it is a robust predictor
  - If a coefficient flips sign or loses significance in
    Spec 4 or Spec 5 → the result is sensitive to outliers
    and I should mention this as a limitation
  - If R² stays similar across specs → model is stable

Outputs saved:
  outputs/figures/cooks_distance.png
  outputs/figures/robustness_coefficients.png
  outputs/tables/robustness_specifications.csv

Note on governance variable (Rule of Law):
  The World Governance Indicators are not available through
  the standard World Bank REST API. To add them as Spec 6,
  I can download manually from:
  https://info.worldbank.org/governance/wgi/
  Then save as data/raw/wgi_rule_of_law.csv and merge on country_code.
""")
