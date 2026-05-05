# ============================================================
# I'm fitting Models A, B, C and measuring their performance
# ============================================================
#
# What I'm doing here:
#   I have a master dataset with one row per country. I fit
#   three regression models, each building on the last:
#
#   Model A — Baseline (production factors only)
#     Predictors: cereal yield, fertiliser use, arable land,
#                 GDP per capita, internet users
#
#   Model B — Add Post-Harvest Loss (PHL) block
#     Everything in Model A, plus: cereal_loss_pct
#
#   Model C — Add Finance block
#     Everything in Model B, plus: account_ownership_pct,
#                                   bank_branches_per_100k
#
#   For each model I use THREE methods:
#     1. OLS regression   (gives p-values, R², coefficients)
#     2. Random Forest    (handles non-linear patterns)
#     3. XGBoost          (boosted trees — often most accurate)
#
#   I also do 5-fold cross-validation to check each model
#   generalises to countries it hasn't seen.
#
#   Finally I compute SHAP values to show which variables
#   matter most inside the Random Forest model.
#
# Dependent variable (what I'm predicting):
#   Prevalence of undernourishment (% of population) — 2021
# ============================================================

# I suppress any warning messages so the output stays clean
import warnings
warnings.filterwarnings('ignore')

# I need requests to download data from the internet
import requests

# I need time to pause between downloads
import time

# I need numpy for mathematical operations
import numpy as np

# I need pandas to work with tables
import pandas as pd

# I need matplotlib to draw charts
import matplotlib
matplotlib.use('Agg')   # I use Agg so charts save to files without needing a screen
import matplotlib.pyplot as plt

# I need these from sklearn for my machine learning models
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics         import r2_score
from sklearn.metrics         import mean_squared_error

# I need statsmodels for the OLS regression (gives proper p-values)
import statsmodels.api as sm

# I need xgboost for the boosted tree model
import xgboost as xgb

# I need shap to explain which variables drive the predictions
import shap

# I need os to create folders
import os

# I make sure my output folders exist before I try to save anything
os.makedirs("outputs/tables",  exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# I set a fixed random seed so my results are the same every time I run this
RANDOM_SEED = 42

# I let the user know I'm starting
print("Starting Phase D — modelling...")
print("=" * 55)


# ============================================================
# Step 1: I'm downloading the dependent variable
# ============================================================
# The thing I'm predicting is:
#   Prevalence of undernourishment (% of population)
#   World Bank code: SN.ITK.DEFC.ZS
# Higher values mean worse food insecurity.

print("\n[1] Downloading dependent variable — undernourishment...")

# I define the set of World Bank regional codes that are NOT real countries
REGIONAL_CODES = set()
REGIONAL_CODES.add('AFE'); REGIONAL_CODES.add('AFW'); REGIONAL_CODES.add('ARB')
REGIONAL_CODES.add('CEB'); REGIONAL_CODES.add('CSS'); REGIONAL_CODES.add('EAP')
REGIONAL_CODES.add('EAR'); REGIONAL_CODES.add('EAS'); REGIONAL_CODES.add('ECA')
REGIONAL_CODES.add('ECS'); REGIONAL_CODES.add('EMU'); REGIONAL_CODES.add('EUU')
REGIONAL_CODES.add('FCS'); REGIONAL_CODES.add('HIC'); REGIONAL_CODES.add('HPC')
REGIONAL_CODES.add('IBD'); REGIONAL_CODES.add('IBT'); REGIONAL_CODES.add('IDA')
REGIONAL_CODES.add('IDB'); REGIONAL_CODES.add('IDX'); REGIONAL_CODES.add('LAC')
REGIONAL_CODES.add('LCN'); REGIONAL_CODES.add('LDC'); REGIONAL_CODES.add('LIC')
REGIONAL_CODES.add('LMC'); REGIONAL_CODES.add('LMY'); REGIONAL_CODES.add('LTE')
REGIONAL_CODES.add('MEA'); REGIONAL_CODES.add('MIC'); REGIONAL_CODES.add('MNA')
REGIONAL_CODES.add('NAC'); REGIONAL_CODES.add('OED'); REGIONAL_CODES.add('OSS')
REGIONAL_CODES.add('PRE'); REGIONAL_CODES.add('PSS'); REGIONAL_CODES.add('PST')
REGIONAL_CODES.add('SAS'); REGIONAL_CODES.add('SSA'); REGIONAL_CODES.add('SSF')
REGIONAL_CODES.add('SST'); REGIONAL_CODES.add('TEA'); REGIONAL_CODES.add('TEC')
REGIONAL_CODES.add('TLA'); REGIONAL_CODES.add('TMN'); REGIONAL_CODES.add('TSA')
REGIONAL_CODES.add('TSS'); REGIONAL_CODES.add('UMC'); REGIONAL_CODES.add('WLD')
REGIONAL_CODES.add('XZN')

# I try to download the undernourishment data from the World Bank API
try:
    # I send the request for the undernourishment indicator
    response = requests.get(
        "https://api.worldbank.org/v2/country/all/indicator/SN.ITK.DEFC.ZS",
        params={"date": 2021, "format": "json", "per_page": 300},
        timeout=30,
    )

    # I convert the response from JSON into Python data
    data = response.json()

    # I'll collect one row per real country in this list
    rows = []

    # I go through each entry in the data
    for entry in data[1]:
        # I get the country code for this entry
        code = entry.get("countryiso3code", "")

        # I get the undernourishment value
        val = entry.get("value")

        # I only keep real countries with an actual value
        if val is not None and code and code not in REGIONAL_CODES:
            one_row = {}
            one_row["country_code"]         = code
            one_row["undernourishment_pct"] = val
            rows.append(one_row)

    # I turn my list into a table
    dv_df = pd.DataFrame(rows)

    # I tell the user how many countries I got data for
    print(f"  Downloaded undernourishment data for {len(dv_df)} real countries")

    # I save this to a file in case I need it again
    dv_df.to_csv("data/raw/undernourishment_2021.csv", index=False)

except Exception as e:
    print(f"  Could not download — {e}")
    # I create an empty table so the rest of the script doesn't crash
    dv_df = pd.DataFrame(columns=["country_code", "undernourishment_pct"])


# ============================================================
# Step 2: I'm loading my master dataset and adding the DV
# ============================================================

print("\n[2] Loading master dataset and merging with DV...")

# I load the master dataset I created in Phase C
master = pd.read_csv("data/processed/master_dataset_clean.csv")

# I merge the undernourishment data onto the master table
master = master.merge(dv_df, on="country_code", how="left")

# I print a quick summary
print(f"  Master: {len(master)} countries")
print(f"  Countries with undernourishment data: {master['undernourishment_pct'].notna().sum()}")

# I save the updated master with the dependent variable included
master.to_csv("data/processed/master_dataset_with_dv.csv", index=False)


# ============================================================
# Step 3: I'm defining the three model specifications
# ============================================================

# This is the column I'm trying to predict
DV = "undernourishment_pct"

# Model A uses only production and development variables
MODEL_A_VARS = [
    "cereal_yield_kg_per_ha",   # how much cereal the country grows
    "fertiliser_kg_per_ha",     # how much fertiliser it uses
    "arable_land_pct",          # how much land is farmable
    "gdp_per_capita_usd",       # income per person
    "internet_users_pct",       # infrastructure and development proxy
]

# Model B adds post-harvest loss on top of Model A
MODEL_B_VARS = MODEL_A_VARS + [
    "cereal_loss_pct",          # % of harvested cereal lost before reaching people
]

# Model C adds financial access on top of Model B
MODEL_C_VARS = MODEL_B_VARS + [
    "account_ownership_pct",    # % adults with a bank or mobile money account
    "bank_branches_per_100k",   # bank infrastructure per 100,000 adults
]

# I put all three model definitions into one dictionary
MODELS = {}
MODELS["Model A — Baseline"] = MODEL_A_VARS
MODELS["Model B — +PHL"]     = MODEL_B_VARS
MODELS["Model C — +Finance"] = MODEL_C_VARS


# ============================================================
# Step 4: I'm writing helper functions
# ============================================================

def prepare_data(df, predictor_cols, outcome_col):
    # I make a copy of the data so I don't accidentally change the original
    working = df.copy()

    # I log-transform variables that are heavily skewed
    # Log transformation makes the regression work better on skewed data
    LOG_COLS = ["gdp_per_capita_usd", "cereal_yield_kg_per_ha",
                "population_total", "fertiliser_kg_per_ha"]

    # I go through each column in my log list
    for col in LOG_COLS:
        # I only transform it if it's both in my predictor list AND in the table
        if col in predictor_cols and col in working.columns:
            # clip(lower=0) makes sure I don't take the log of a negative number
            # log1p adds 1 before taking the log so log(0) doesn't cause an error
            working[col] = np.log1p(working[col].clip(lower=0))

    # I build a list of all the columns I need
    needed = [outcome_col] + predictor_cols

    # I only keep columns that actually exist in the table
    needed_existing = []
    for col in needed:
        if col in working.columns:
            needed_existing.append(col)

    # I remove any rows where any of my needed columns is blank
    working = working.dropna(subset=needed_existing)

    # I pull out just the predictor columns
    X = working[predictor_cols].copy()

    # I pull out just the outcome column
    y = working[outcome_col].copy()

    # I return both
    return X, y


def run_ols(X, y, model_name):
    # I add a constant column — OLS regression needs this for the intercept
    X_const = sm.add_constant(X)

    # I fit the OLS model
    model = sm.OLS(y, X_const).fit()

    # I print a one-line summary
    print(f"\n  OLS result for {model_name}:")
    print(f"    N = {int(model.nobs)}  |  R² = {model.rsquared:.3f}  |  "
          f"Adj R² = {model.rsquared_adj:.3f}  |  "
          f"F-stat p = {model.f_pvalue:.4f}")

    # I build a safe filename by removing special characters
    safe_name = model_name.replace(" ", "_")
    safe_name = safe_name.replace("+", "")
    safe_name = safe_name.replace("—", "")
    safe_name = safe_name.strip("_")

    # I save the full OLS table to a text file
    table_path = "outputs/tables/ols_" + safe_name + ".txt"
    with open(table_path, "w") as f:
        f.write(model.summary().as_text())
    print(f"    Full table saved → {table_path}")

    # I return the fitted model so I can use it elsewhere
    return model


def run_ml(X, y, model_name, kfold):
    # ── I'm fitting the Random Forest model ───────────────────────────────────
    # A Random Forest builds many decision trees and averages their predictions
    rf = RandomForestRegressor(
        n_estimators = 200,    # I want 200 trees in my forest
        max_depth    = 4,      # I keep trees shallow to avoid overfitting
        random_state = RANDOM_SEED,
        n_jobs       = -1,     # I use all available CPU cores
    )

    # I use 5-fold cross-validation to measure how well my RF model predicts
    # It trains on 4 folds and tests on 1, rotating through all 5 combinations
    rf_cv_scores = cross_val_score(rf, X, y, cv=kfold, scoring="r2")

    # I now fit the RF on ALL my data (so I can compute SHAP values later)
    rf.fit(X, y)

    # I calculate the average CV score across all 5 folds
    rf_r2_cv = rf_cv_scores.mean()

    # ── I'm fitting the XGBoost model ─────────────────────────────────────────
    # XGBoost builds trees one at a time, each correcting the errors of the last
    xgb_model = xgb.XGBRegressor(
        n_estimators     = 200,
        max_depth        = 3,
        learning_rate    = 0.05,   # small steps = more careful learning
        subsample        = 0.8,    # I use 80% of data per tree to reduce overfitting
        colsample_bytree = 0.8,    # I use 80% of features per tree
        random_state     = RANDOM_SEED,
        verbosity        = 0,      # I silence XGBoost's own printout
    )

    # I do 5-fold CV for XGBoost too
    xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=kfold, scoring="r2")

    # I fit XGBoost on all my data
    xgb_model.fit(X, y)

    # I calculate the average XGBoost CV score
    xgb_r2_cv = xgb_cv_scores.mean()

    # I print the results for both models
    print(f"  Random Forest  5-fold CV R² = {rf_r2_cv:.3f}  "
          f"(std ± {rf_cv_scores.std():.3f})")
    print(f"  XGBoost        5-fold CV R² = {xgb_r2_cv:.3f}  "
          f"(std ± {xgb_cv_scores.std():.3f})")

    # I return both models and their scores
    return rf, xgb_model, rf_r2_cv, xgb_r2_cv


def make_shap_plot(rf_model, X, model_name):
    # I create a SHAP explainer for my Random Forest model
    # SHAP tells me how much each variable pushed each prediction up or down
    explainer = shap.TreeExplainer(rf_model)

    # I compute the SHAP values for all my data
    shap_values = explainer.shap_values(X)

    # I create a new chart
    plt.figure(figsize=(8, 5))

    # I draw a SHAP bar chart showing mean absolute importance per feature
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)

    # I add a title to the chart
    plt.title(f"SHAP feature importance — {model_name}", fontsize=12)

    # I make the layout tidy
    plt.tight_layout()

    # I build a safe filename
    safe_name = model_name.replace(" ", "_")
    safe_name = safe_name.replace("+", "")
    safe_name = safe_name.replace("—", "")
    safe_name = safe_name.strip("_")

    # I save the chart to a file
    fig_path = "outputs/figures/shap_" + safe_name + ".png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")

    # I close the chart to free up memory
    plt.close()

    # I tell the user where I saved it
    print(f"    SHAP chart saved → {fig_path}")


# ============================================================
# Step 5: I'm running all three models
# ============================================================

print("\n[3] Fitting Models A, B, C...")

# I set up the 5-fold cross-validation split
# shuffle=True means I randomly mix the data before splitting
kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# I'll collect the key numbers from all models in this list
results = []

# I go through each model one by one
for model_name in MODELS:
    # I get the list of predictor columns for this model
    predictor_cols = MODELS[model_name]

    # I print a section header
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"  Predictors: {predictor_cols}")
    print(f"{'='*55}")

    # I prepare the data for this model
    X, y = prepare_data(master, predictor_cols, DV)

    # I tell the user how many countries have complete data
    print(f"  Sample size (countries with complete data): {len(X)}")

    # If I have fewer than 30 countries, the model won't be reliable
    if len(X) < 30:
        print("  Skipping — too few countries for reliable results")
        continue

    # I run the OLS regression
    ols_model = run_ols(X, y, model_name)

    # I run the machine learning models
    print(f"\n  Machine learning results for {model_name}:")
    rf_model, xgb_model, rf_cv_r2, xgb_cv_r2 = run_ml(X, y, model_name, kfold)

    # I compute and save the SHAP importance chart
    print(f"  Computing SHAP values...")
    make_shap_plot(rf_model, X, model_name)

    # I collect the key numbers into a result row
    result_row = {}
    result_row["Model"]              = model_name
    result_row["N (countries)"]      = len(X)
    result_row["OLS R²"]             = round(ols_model.rsquared, 3)
    result_row["OLS Adj R²"]         = round(ols_model.rsquared_adj, 3)
    result_row["OLS F-stat p-value"] = round(ols_model.f_pvalue, 4)
    result_row["RF 5-fold CV R²"]    = round(rf_cv_r2, 3)
    result_row["XGB 5-fold CV R²"]   = round(xgb_cv_r2, 3)

    # I add this row to my results list
    results.append(result_row)


# ============================================================
# Step 6: I'm saving the comparison table
# ============================================================

print(f"\n{'='*55}")
print("MODEL COMPARISON TABLE")
print(f"{'='*55}")

# I turn my list of result rows into a table
results_df = pd.DataFrame(results)

# I print the table
print(results_df.to_string(index=False))

# I save the table to a CSV file
results_df.to_csv("outputs/tables/model_comparison.csv", index=False)
print("\nComparison table saved → outputs/tables/model_comparison.csv")


# ============================================================
# Step 7: I'm drawing the R² comparison chart
# ============================================================

print("\nSaving R² comparison chart...")

# I create a figure with a reasonable size
fig, ax = plt.subplots(figsize=(9, 5))

# I set the x positions for my three groups of bars
x = np.arange(len(results_df))

# I set how wide each bar should be
width = 0.25

# I draw the OLS R² bars (one per model)
bars1 = ax.bar(x - width, results_df["OLS R²"], width,
               label="OLS R²", color="#4472C4")

# I draw the Random Forest CV R² bars
bars2 = ax.bar(x, results_df["RF 5-fold CV R²"], width,
               label="Random Forest CV R²", color="#ED7D31")

# I draw the XGBoost CV R² bars
bars3 = ax.bar(x + width, results_df["XGB 5-fold CV R²"], width,
               label="XGBoost CV R²", color="#70AD47")

# I add value labels on top of each bar
for bar_group in [bars1, bars2, bars3]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + 0.005, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)

# I label the axes
ax.set_xlabel("Model specification")
ax.set_ylabel("R² (higher is better)")
ax.set_title("Model performance: OLS vs Random Forest vs XGBoost\n"
             "(Dependent variable: Prevalence of undernourishment %)")

# I set the x tick labels to the model names
ax.set_xticks(x)
ax.set_xticklabels(results_df["Model"], fontsize=9)

# I add a legend
ax.legend()

# I set the y axis to go from 0 to 1
ax.set_ylim(0, 1)

# I draw a black line at y=0
ax.axhline(0, color="black", linewidth=0.5)

# I make the layout tidy
plt.tight_layout()

# I save the chart
plt.savefig("outputs/figures/model_r2_comparison.png", dpi=150)

# I close the chart
plt.close()

# I tell the user where I saved it
print("R² chart saved → outputs/figures/model_r2_comparison.png")


# ============================================================
# Step 8: I'm printing the OLS coefficient table
# ============================================================
# This shows which predictors are statistically significant
# Stars: *** means p<0.01  ** means p<0.05  * means p<0.10

print(f"\n{'='*55}")
print("KEY OLS COEFFICIENTS SUMMARY")
print("(stars: *** p<0.01  ** p<0.05  * p<0.10)")
print(f"{'='*55}")

# I go through each model again to print its coefficients
for model_name in MODELS:
    predictor_cols = MODELS[model_name]

    # I prepare the data again for this model
    X, y = prepare_data(master, predictor_cols, DV)

    # I skip models with too few countries
    if len(X) < 30:
        continue

    # I add the intercept column and fit the OLS model
    X_const = sm.add_constant(X)
    fitted  = sm.OLS(y, X_const).fit()

    # I print the header for this model
    print(f"\n{model_name}  (N={int(fitted.nobs)}, R²={fitted.rsquared:.3f})")
    print(f"  {'Variable':<30} {'Coef':>8} {'p-value':>9} {'Sig':>5}")
    print(f"  {'-'*55}")

    # I go through each variable and print its coefficient and significance
    for var in fitted.params.index:
        # I get the coefficient and p-value for this variable
        coef = fitted.params[var]
        pval = fitted.pvalues[var]

        # I work out the significance stars
        if pval < 0.01:
            sig = "***"
        elif pval < 0.05:
            sig = "**"
        elif pval < 0.10:
            sig = "*"
        else:
            sig = ""

        # I print the row for this variable
        print(f"  {var:<30} {coef:>8.3f} {pval:>9.4f} {sig:>5}")


print(f"\n{'='*55}")
print("PHASE D COMPLETE")
print("Outputs saved in:")
print("  outputs/tables/   — OLS tables and comparison CSV")
print("  outputs/figures/  — SHAP charts and R² comparison")
print("Next step: Phase E — outlier checks and robustness tests")
print(f"{'='*55}")
