# ============================================================
# PHASE D — Fit Models A, B, C and measure performance
# ============================================================
#
# What this file does:
#   We have a master dataset with one row per country and many
#   variables. Now we fit three regression models:
#
#   Model A — Baseline (production factors only)
#     Predictors: cereal yield, fertiliser use, arable land,
#                 GDP per capita, internet users
#     This is the "what do we know without PHL or finance data?"
#
#   Model B — Add Post-Harvest Loss (PHL) block
#     Everything in Model A, plus: cereal_loss_pct
#     This tests whether PHL explains food insecurity
#     on top of the baseline.
#
#   Model C — Add Finance block
#     Everything in Model B, plus: account_ownership_pct,
#                                   bank_branches_per_100k
#     This tests whether financial access explains more.
#
#   For each model we use THREE methods:
#     1. OLS regression (gives coefficients, p-values, R²)
#     2. Random Forest (handles non-linear patterns)
#     3. XGBoost (boosted trees — often the most accurate)
#
#   We also do 5-fold cross-validation so we know how well
#   each model predicts countries it has not seen before.
#
#   Finally we compute SHAP values — these tell us WHICH
#   variables matter most inside the Random Forest model.
#
# Dependent variable (what we are trying to predict):
#   Prevalence of undernourishment (% of population) — 2021
#   Source: World Bank / FAO
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import requests, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble         import RandomForestRegressor
from sklearn.model_selection  import cross_val_score, KFold
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import r2_score, mean_squared_error
from sklearn.impute           import SimpleImputer
import statsmodels.api        as sm
import xgboost                as xgb
import shap
import os

# Make sure output folders exist
os.makedirs("outputs/tables",  exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

RANDOM_SEED = 42

print("Starting Phase D — modelling...")
print("=" * 55)


# ============================================================
# STEP 1: Download and add the dependent variable
# ============================================================
# The thing we are trying to predict is:
#   Prevalence of undernourishment (% of population)
#   World Bank code: SN.ITK.DEFC.ZS
#
# This comes from the FAO and tells us what share of each
# country's population does not have enough food to eat.
# Higher = worse food insecurity.

print("\n[1] Downloading dependent variable — undernourishment...")

# These World Bank codes are regional aggregates, not real countries
# We remove them so we only keep individual countries
REGIONAL_CODES = {
    'AFE','AFW','ARB','CEB','CSS','EAP','EAR','EAS','ECA','ECS',
    'EMU','EUU','FCS','HIC','HPC','IBD','IBT','IDA','IDB','IDX',
    'LAC','LCN','LDC','LIC','LMC','LMY','LTE','MEA','MIC','MNA',
    'NAC','OED','OSS','PRE','PSS','PST','SAS','SSA','SSF','SST',
    'TEA','TEC','TLA','TMN','TSA','TSS','UMC','WLD','XZN'
}

try:
    response = requests.get(
        "https://api.worldbank.org/v2/country/all/indicator/SN.ITK.DEFC.ZS",
        params={"date": 2021, "format": "json", "per_page": 300},
        timeout=30,
    )
    data = response.json()
    rows = []
    for entry in data[1]:
        code = entry.get("countryiso3code", "")
        val  = entry.get("value")
        if val is not None and code and code not in REGIONAL_CODES:
            rows.append({"country_code": code, "undernourishment_pct": val})
    dv_df = pd.DataFrame(rows)
    print(f"  Downloaded undernourishment data for {len(dv_df)} real countries")
    dv_df.to_csv("data/raw/undernourishment_2021.csv", index=False)
except Exception as e:
    print(f"  Could not download — {e}")
    dv_df = pd.DataFrame(columns=["country_code", "undernourishment_pct"])


# ============================================================
# STEP 2: Load master dataset and merge with dependent variable
# ============================================================

print("\n[2] Loading master dataset and merging with DV...")

master = pd.read_csv("data/processed/master_dataset_clean.csv")
master = master.merge(dv_df, on="country_code", how="left")
print(f"  Master: {len(master)} countries")
print(f"  Countries with undernourishment data: {master['undernourishment_pct'].notna().sum()}")

# Save the updated master with the DV
master.to_csv("data/processed/master_dataset_with_dv.csv", index=False)


# ============================================================
# STEP 3: Define the three model specifications
# ============================================================
# Each model builds on the previous one by adding more variables.
# This lets us test: does adding PHL data improve predictions?
#                    does adding Finance data improve further?

# The outcome we are predicting
DV = "undernourishment_pct"

# Model A: production and country factors only
MODEL_A_VARS = [
    "cereal_yield_kg_per_ha",    # how much cereal the country grows
    "fertiliser_kg_per_ha",      # how much fertiliser it uses
    "arable_land_pct",           # how much land is available for farming
    "gdp_per_capita_usd",        # income per person
    "internet_users_pct",        # infrastructure / development proxy
]

# Model B: add post-harvest loss
MODEL_B_VARS = MODEL_A_VARS + [
    "cereal_loss_pct",           # % of harvested cereal lost before it reaches people
]

# Model C: add financial access
MODEL_C_VARS = MODEL_B_VARS + [
    "account_ownership_pct",     # % adults with a bank or mobile money account
    "bank_branches_per_100k",    # bank infrastructure per 100,000 adults
]

MODELS = {
    "Model A — Baseline":          MODEL_A_VARS,
    "Model B — +PHL":              MODEL_B_VARS,
    "Model C — +Finance":          MODEL_C_VARS,
}


# ============================================================
# STEP 4: Helper functions
# ============================================================

def prepare_data(df, predictor_cols, outcome_col):
    """
    Takes the master dataset and returns clean X (predictors)
    and y (outcome) ready for modelling.

    Steps:
    - Drops any row where the outcome is missing
    - Drops any row where a predictor is missing
    - Log-transforms skewed variables (GDP, yield, population)
    - Returns X, y, and the list of final column names
    """
    working = df.copy()

    # Log-transform variables that are very skewed
    # (OLS works better when values are not spread across huge ranges)
    LOG_COLS = ["gdp_per_capita_usd", "cereal_yield_kg_per_ha",
                "population_total", "fertiliser_kg_per_ha"]
    for col in LOG_COLS:
        if col in predictor_cols and col in working.columns:
            working[col] = np.log1p(working[col].clip(lower=0))

    # Keep only the rows where all needed columns have values
    needed = [outcome_col] + predictor_cols
    needed = [c for c in needed if c in working.columns]
    working = working.dropna(subset=needed)

    X = working[predictor_cols].copy()
    y = working[outcome_col].copy()

    return X, y


def run_ols(X, y, model_name):
    """
    Fits an OLS regression (ordinary least squares).
    This is the standard method economists use.
    Returns the fitted model and prints a summary.
    """
    # Add a constant (intercept) — OLS needs this
    X_const = sm.add_constant(X)
    model   = sm.OLS(y, X_const).fit()

    print(f"\n  OLS result for {model_name}:")
    print(f"    N = {int(model.nobs)}  |  R² = {model.rsquared:.3f}  |  "
          f"Adj R² = {model.rsquared_adj:.3f}  |  "
          f"F-stat p = {model.f_pvalue:.4f}")

    # Save the full OLS table to a text file
    safe_name = model_name.replace(" ", "_").replace("+", "").replace("—", "").strip("_")
    table_path = f"outputs/tables/ols_{safe_name}.txt"
    with open(table_path, "w") as f:
        f.write(model.summary().as_text())
    print(f"    Full table saved → {table_path}")

    return model


def run_ml(X, y, model_name, kfold):
    """
    Fits a Random Forest and an XGBoost model.
    Uses 5-fold cross-validation to get a reliable R² score.
    Returns both fitted models and their CV scores.
    """
    # ── Random Forest ─────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators  = 200,   # 200 decision trees
        max_depth     = 4,     # keep trees shallow to avoid overfitting
        random_state  = RANDOM_SEED,
        n_jobs        = -1,    # use all CPU cores
    )
    # 5-fold CV: train on 4 folds, test on 1, repeat 5 times
    rf_cv_scores = cross_val_score(rf, X, y, cv=kfold, scoring="r2")
    rf.fit(X, y)                       # fit on ALL data for SHAP later
    rf_r2_cv = rf_cv_scores.mean()

    # ── XGBoost ───────────────────────────────────────────────
    xgb_model = xgb.XGBRegressor(
        n_estimators      = 200,
        max_depth         = 3,
        learning_rate     = 0.05,
        subsample         = 0.8,       # use 80% of data per tree
        colsample_bytree  = 0.8,       # use 80% of features per tree
        random_state      = RANDOM_SEED,
        verbosity         = 0,
    )
    xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=kfold, scoring="r2")
    xgb_model.fit(X, y)
    xgb_r2_cv = xgb_cv_scores.mean()

    print(f"  Random Forest  5-fold CV R² = {rf_r2_cv:.3f}  "
          f"(std ± {rf_cv_scores.std():.3f})")
    print(f"  XGBoost        5-fold CV R² = {xgb_r2_cv:.3f}  "
          f"(std ± {xgb_cv_scores.std():.3f})")

    return rf, xgb_model, rf_r2_cv, xgb_r2_cv


def make_shap_plot(rf_model, X, model_name):
    """
    Uses SHAP values to show which variables matter most.
    SHAP = SHapley Additive exPlanations.
    A high SHAP value for a variable means it has a big effect
    on the prediction — either pushing it up or pulling it down.
    """
    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)

    # Bar chart: mean absolute SHAP value per feature
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"SHAP feature importance — {model_name}", fontsize=12)
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("+", "").replace("—", "").strip("_")
    fig_path = f"outputs/figures/shap_{safe_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    SHAP chart saved → {fig_path}")


# ============================================================
# STEP 5: Run all three models
# ============================================================

print("\n[3] Fitting Models A, B, C...")

# We use the same 5-fold split for all models so results are comparable
kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# This table will collect the key numbers from all models
results = []

for model_name, predictor_cols in MODELS.items():
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"  Predictors: {predictor_cols}")
    print(f"{'='*55}")

    # Prepare data for this model
    X, y = prepare_data(master, predictor_cols, DV)
    print(f"  Sample size (countries with complete data): {len(X)}")

    if len(X) < 30:
        print("  Skipping — too few countries for reliable results")
        continue

    # ── OLS regression ────────────────────────────────────────
    ols_model = run_ols(X, y, model_name)

    # ── Random Forest + XGBoost with 5-fold CV ────────────────
    print(f"\n  Machine learning results for {model_name}:")
    rf_model, xgb_model, rf_cv_r2, xgb_cv_r2 = run_ml(X, y, model_name, kfold)

    # ── SHAP importance chart for Random Forest ───────────────
    print(f"  Computing SHAP values...")
    make_shap_plot(rf_model, X, model_name)

    # Collect results for the comparison table
    results.append({
        "Model":                model_name,
        "N (countries)":        len(X),
        "OLS R²":               round(ols_model.rsquared, 3),
        "OLS Adj R²":           round(ols_model.rsquared_adj, 3),
        "OLS F-stat p-value":   round(ols_model.f_pvalue, 4),
        "RF 5-fold CV R²":      round(rf_cv_r2, 3),
        "XGB 5-fold CV R²":     round(xgb_cv_r2, 3),
    })


# ============================================================
# STEP 6: Save the comparison table
# ============================================================

print(f"\n{'='*55}")
print("MODEL COMPARISON TABLE")
print(f"{'='*55}")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

results_df.to_csv("outputs/tables/model_comparison.csv", index=False)
print("\nComparison table saved → outputs/tables/model_comparison.csv")


# ============================================================
# STEP 7: Plot R² comparison across models and methods
# ============================================================

print("\nSaving R² comparison chart...")

fig, ax = plt.subplots(figsize=(9, 5))

x      = np.arange(len(results_df))
width  = 0.25

bars1 = ax.bar(x - width, results_df["OLS R²"],        width, label="OLS R²",          color="#4472C4")
bars2 = ax.bar(x,          results_df["RF 5-fold CV R²"], width, label="Random Forest CV R²", color="#ED7D31")
bars3 = ax.bar(x + width,  results_df["XGB 5-fold CV R²"],width, label="XGBoost CV R²",    color="#70AD47")

# Add value labels on top of each bar
for bar_group in [bars1, bars2, bars3]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xlabel("Model specification")
ax.set_ylabel("R² (higher is better)")
ax.set_title("Model performance: OLS vs Random Forest vs XGBoost\n"
             "(Dependent variable: Prevalence of undernourishment %)")
ax.set_xticks(x)
ax.set_xticklabels(
    [n.split("—")[0].strip() + "\n" + n.split("—")[1].strip()
     if "—" in n else n for n in results_df["Model"]],
    fontsize=9
)
ax.legend()
ax.set_ylim(0, 1)
ax.axhline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig("outputs/figures/model_r2_comparison.png", dpi=150)
plt.close()
print("R² chart saved → outputs/figures/model_r2_comparison.png")


# ============================================================
# STEP 8: Print OLS coefficient table for each model
# ============================================================
# This shows which predictors are statistically significant
# (p < 0.05 means we can be 95% confident the effect is real)

print(f"\n{'='*55}")
print("KEY OLS COEFFICIENTS SUMMARY")
print("(stars: *** p<0.01  ** p<0.05  * p<0.10)")
print(f"{'='*55}")

for model_name, predictor_cols in MODELS.items():
    X, y = prepare_data(master, predictor_cols, DV)
    if len(X) < 30:
        continue
    X_const = sm.add_constant(X)
    fitted  = sm.OLS(y, X_const).fit()

    print(f"\n{model_name}  (N={int(fitted.nobs)}, R²={fitted.rsquared:.3f})")
    print(f"  {'Variable':<30} {'Coef':>8} {'p-value':>9} {'Sig':>5}")
    print(f"  {'-'*55}")
    for var in fitted.params.index:
        coef = fitted.params[var]
        pval = fitted.pvalues[var]
        if   pval < 0.01: sig = "***"
        elif pval < 0.05: sig = "**"
        elif pval < 0.10: sig = "*"
        else:             sig = ""
        print(f"  {var:<30} {coef:>8.3f} {pval:>9.4f} {sig:>5}")


print(f"\n{'='*55}")
print("PHASE D COMPLETE")
print("Outputs saved in:")
print("  outputs/tables/   — OLS tables and comparison CSV")
print("  outputs/figures/  — SHAP charts and R² comparison")
print("Next step: Phase E — outlier checks and robustness tests")
print(f"{'='*55}")
