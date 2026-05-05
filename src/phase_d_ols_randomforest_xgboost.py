# ============================================================
# I'm fitting Models A, B, C, D and measuring their performance
# ============================================================
#
# What I'm doing here:
#   I have a master dataset with one row per country.
#   I fit FOUR regression models, each one adding a new block
#   of variables to test whether it improves prediction.
#
#   Model A — Baseline (production factors + income)
#     Tests: does basic agricultural capacity explain food insecurity?
#     Predictors: cereal yield, fertiliser use, arable land,
#                 GDP per capita, rural population,
#                 agricultural employment, livestock production
#
#   Model B — Add Post-Harvest Loss block
#     Tests: does food LOSS between farm and market matter
#            beyond production capacity?
#     Add: cereal_loss_pct
#
#   Model C — Add National Financial Access block
#     Tests: does NATIONAL financial inclusion (the old way of
#            measuring it) improve predictions?
#     Add: account_ownership_pct, bank_branches_per_100k,
#          private_credit_pct_gdp
#
#   Model D — Value Chain Financial Access block
#     THIS IS THE KEY CONTRIBUTION OF MY DISSERTATION.
#     Tests: does financial access measured SPECIFICALLY along
#            the food value chain (rural people, women farmers,
#            poorest 40%, agricultural payments, borrowing)
#            explain food insecurity BETTER than national averages?
#     Add: account_ownership_rural_pct (instead of national)
#          account_ownership_female_pct (women grow 60-80% of food)
#          account_ownership_poorest40_pct (poorest smallholders)
#          agri_payments_digital_pct (farmers in digital economy)
#          borrowed_from_bank_pct (credit actually reaching people)
#          value_chain_finance_score (my composite indicator)
#
#   Model E — Full Governance Model
#     Tests: once I control for governance quality, do value chain
#            finance variables still matter?
#     Add: wgi_political_stability, wgi_control_of_corruption,
#          wgi_government_effectiveness
#
#   For every model I use THREE methods:
#     1. OLS regression   (gives p-values, R², coefficients)
#     2. Random Forest    (handles non-linear relationships)
#     3. XGBoost          (boosted trees — often the most accurate)
#
#   I also do 5-fold cross-validation (CV) to check whether each
#   model generalises to countries it has not seen before.
#   CV R² is more trustworthy than in-sample R².
#
#   Finally I compute SHAP values for the Random Forest model
#   to show exactly which variables matter most.
#
# Dependent variable:
#   Prevalence of undernourishment (% of population) — 2021
#   Higher values = worse food insecurity.
# ============================================================

# I suppress warning messages to keep the output readable
import warnings

warnings.filterwarnings("ignore")

# I need time to pause between downloads
import time

# I need matplotlib to draw charts
import matplotlib
# I need numpy for mathematical operations
import numpy as np
# I need pandas for working with tables
import pandas as pd
# I need requests to download the dependent variable from the internet
import requests

matplotlib.use("Agg")   # I save charts to files rather than displaying on screen
# I need os to create output folders
import os

import matplotlib.pyplot as plt
# I need shap to explain which variables drive the predictions
import shap
# I need statsmodels for OLS regression (it gives me proper p-values)
import statsmodels.api as sm
# I need xgboost for the boosted tree model
import xgboost as xgb
# I need these from sklearn for machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# I make sure my output folders exist
os.makedirs("outputs/tables",  exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# I fix the random seed for reproducibility
RANDOM_SEED = 42

print("Starting Phase D — modelling...")
print("=" * 60)


# ============================================================
# Step 1: I'm downloading the dependent variable
# ============================================================
# What I am predicting:
#   Prevalence of undernourishment (% of population) — 2021
#   World Bank indicator code: SN.ITK.DEFC.ZS
#   Higher number = worse food insecurity

print("\n[1] Downloading dependent variable — undernourishment...")

# I define the set of World Bank regional/aggregate codes
# These are NOT real countries and must be excluded
REGIONAL_CODES = set()
REGIONAL_CODES.add("AFE"); REGIONAL_CODES.add("AFW"); REGIONAL_CODES.add("ARB")
REGIONAL_CODES.add("CEB"); REGIONAL_CODES.add("CSS"); REGIONAL_CODES.add("EAP")
REGIONAL_CODES.add("EAR"); REGIONAL_CODES.add("EAS"); REGIONAL_CODES.add("ECA")
REGIONAL_CODES.add("ECS"); REGIONAL_CODES.add("EMU"); REGIONAL_CODES.add("EUU")
REGIONAL_CODES.add("FCS"); REGIONAL_CODES.add("HIC"); REGIONAL_CODES.add("HPC")
REGIONAL_CODES.add("IBD"); REGIONAL_CODES.add("IBT"); REGIONAL_CODES.add("IDA")
REGIONAL_CODES.add("IDB"); REGIONAL_CODES.add("IDX"); REGIONAL_CODES.add("LAC")
REGIONAL_CODES.add("LCN"); REGIONAL_CODES.add("LDC"); REGIONAL_CODES.add("LIC")
REGIONAL_CODES.add("LMC"); REGIONAL_CODES.add("LMY"); REGIONAL_CODES.add("LTE")
REGIONAL_CODES.add("MEA"); REGIONAL_CODES.add("MIC"); REGIONAL_CODES.add("MNA")
REGIONAL_CODES.add("NAC"); REGIONAL_CODES.add("OED"); REGIONAL_CODES.add("OSS")
REGIONAL_CODES.add("PRE"); REGIONAL_CODES.add("PSS"); REGIONAL_CODES.add("PST")
REGIONAL_CODES.add("SAS"); REGIONAL_CODES.add("SSA"); REGIONAL_CODES.add("SSF")
REGIONAL_CODES.add("SST"); REGIONAL_CODES.add("TEA"); REGIONAL_CODES.add("TEC")
REGIONAL_CODES.add("TLA"); REGIONAL_CODES.add("TMN"); REGIONAL_CODES.add("TSA")
REGIONAL_CODES.add("TSS"); REGIONAL_CODES.add("UMC"); REGIONAL_CODES.add("WLD")
REGIONAL_CODES.add("XZN")

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
            one_row = {}
            one_row["country_code"]         = code
            one_row["undernourishment_pct"] = val
            rows.append(one_row)

    dv_df = pd.DataFrame(rows)
    dv_df.to_csv("data/raw/undernourishment_2021.csv", index=False)
    print("  Downloaded undernourishment data for", len(dv_df), "real countries")

except Exception as e:
    print("  Could not download —", e)
    dv_df = pd.DataFrame(columns=["country_code", "undernourishment_pct"])


# ============================================================
# Step 2: I'm loading the master dataset and adding the DV
# ============================================================

print("\n[2] Loading master dataset and merging dependent variable...")

master = pd.read_csv("data/processed/master_dataset_clean.csv")

# I merge the undernourishment data onto the master table
master = master.merge(dv_df, on="country_code", how="left")

print("  Master size:", len(master), "countries")
print("  Countries with undernourishment data:", master["undernourishment_pct"].notna().sum())
print("  All columns in master dataset:")
for col in master.columns:
    n_valid = master[col].notna().sum()
    print(f"    {col:<45} ({n_valid} countries have data)")

# I save the updated master with the dependent variable included
master.to_csv("data/processed/master_dataset_with_dv.csv", index=False)


# ============================================================
# Step 3: I'm defining ALL model specifications
# ============================================================
# Each model builds on the previous one by adding a new block of variables.
# This lets me see exactly how much each block improves prediction.

# This is the variable I am trying to predict in every model
DV = "undernourishment_pct"

# ── Model A — Baseline production + income ────────────────────────────────────
# I now use a richer baseline than before.
# I include rural population and agricultural employment because they tell me
# WHO is in the food system, not just how productive it is.
MODEL_A_VARS = [
    "cereal_yield_kg_per_ha",    # Production capacity — tonnes per hectare
    "fertiliser_kg_per_ha",      # Input intensity — investment in yield
    "arable_land_pct",           # Land availability for farming
    "gdp_per_capita_usd",        # Income — can people afford to buy food?
    "rural_population_pct",      # What share lives in the farming sector?
    "agri_employment_pct",       # What share WORKS in agriculture?
    "livestock_production_index",# Animal-source food production
]

# ── Model B — Add Post-Harvest Loss ────────────────────────────────────────────
# Post-harvest loss is the percentage of cereal that is lost between the
# farm and the consumer. If 30% of food is lost, the country needs to produce
# 30% more just to maintain the same level of food availability.
MODEL_B_VARS = MODEL_A_VARS + [
    "cereal_loss_pct",           # % of harvested cereal lost before reaching people
]

# ── Model C — Add NATIONAL financial access ────────────────────────────────────
# The traditional way to measure financial inclusion in food security research
# is to use the NATIONAL average. I include it here as a comparison point.
# If Model D (value chain finance) adds MORE than Model C, that supports
# my dissertation argument.
MODEL_C_VARS = MODEL_B_VARS + [
    "account_ownership_pct",     # National average: % adults with an account
    "bank_branches_per_100k",    # Physical bank presence in the country
    "private_credit_pct_gdp",    # Does the financial system actually lend? (% GDP)
]

# ── Model D — Value Chain Financial Access ─────────────────────────────────────
# This is the CORE CONTRIBUTION of my dissertation.
# Instead of national averages, I use financial access indicators that
# specifically target the people in the food value chain:
#   - Rural adults (where farmers live)
#   - Women (who grow 60-80% of food in low-income countries)
#   - Poorest 40% (who face the most food insecurity)
#   - Agricultural digital payments (farmers in the digital economy)
#   - Borrowing (credit actually reaching people)
#   - My composite value chain score
#
# My hypothesis: if the disaggregated indicators explain food insecurity
# BETTER than the national averages in Model C, it means that the
# distribution of financial access — not just its average level —
# is what matters for food security outcomes.
MODEL_D_VARS = MODEL_B_VARS + [
    "account_ownership_rural_pct",   # Finance where farmers live
    "account_ownership_female_pct",  # Finance for women who grow the food
    "account_ownership_poorest40_pct",# Finance for the most food-insecure
    "agri_payments_digital_pct",     # Farmers receiving digital payments
    "borrowed_from_bank_pct",        # Credit actually reaching people
    "value_chain_finance_score",     # My composite value chain indicator
]

# ── Model E — Full Governance Control ─────────────────────────────────────────
# Governance quality could be a confounding variable: poor countries may have
# BOTH bad governance AND low financial access. If financial access loses
# significance once I control for governance, it means governance is the
# true driver. If it stays significant, financial access has an independent effect.
MODEL_E_VARS = MODEL_D_VARS + [
    "wgi_political_stability",       # Conflict destroys food supply chains
    "wgi_control_of_corruption",     # Corruption diverts food aid from farmers
    "wgi_government_effectiveness",  # Can the government deliver services to farmers?
    "food_price_inflation_pct",      # Rising prices reduce food purchasing power
]

# I put all models in one dictionary so I can loop through them
MODELS = {}
MODELS["Model A — Baseline"]               = MODEL_A_VARS
MODELS["Model B — +Post-Harvest Loss"]     = MODEL_B_VARS
MODELS["Model C — +National Finance"]      = MODEL_C_VARS
MODELS["Model D — +Value Chain Finance"]   = MODEL_D_VARS
MODELS["Model E — +Governance Controls"]   = MODEL_E_VARS


# ============================================================
# Step 4: I'm writing helper functions
# ============================================================

def prepare_data(df, predictor_cols, outcome_col):
    # I make a copy so I don't accidentally change the original
    working = df.copy()

    # I log-transform variables that are heavily skewed
    # Log transformation makes OLS regression more reliable
    LOG_COLS = [
        "gdp_per_capita_usd",
        "cereal_yield_kg_per_ha",
        "population_total",
        "fertiliser_kg_per_ha",
        "bank_branches_per_100k",
        "atm_per_100k",
        "private_credit_pct_gdp",
    ]

    for col in LOG_COLS:
        # I only transform columns that are both in my predictor list AND in the table
        if col in predictor_cols and col in working.columns:
            # clip(lower=0) ensures I don't take the log of a negative number
            # log1p adds 1 before taking the log so log(0) becomes 0, not infinity
            working[col] = np.log1p(working[col].clip(lower=0))

    # I build a list of all columns I need
    needed = [outcome_col] + predictor_cols

    # I only keep columns that actually exist in the table
    # (some variables may not be in the master if Phase B couldn't download them)
    needed_existing = []
    for col in needed:
        if col in working.columns:
            needed_existing.append(col)

    # I keep only columns that actually exist as predictors
    existing_predictors = []
    for col in predictor_cols:
        if col in working.columns:
            existing_predictors.append(col)

    # I remove any rows where any needed column is blank
    working = working.dropna(subset=needed_existing)

    # I pull out the predictor values (X) and outcome values (y)
    X = working[existing_predictors].copy()
    y = working[outcome_col].copy()

    return X, y, existing_predictors


def run_ols(X, y, model_name, predictor_list):
    # I add a constant column — OLS needs this for the intercept term
    X_const = sm.add_constant(X)

    # I fit the OLS model
    model = sm.OLS(y, X_const).fit()

    # I print a one-line summary
    print("\n  OLS result for", model_name, ":")
    print("    N =", int(model.nobs), " | R² =", round(model.rsquared, 3),
          " | Adj R² =", round(model.rsquared_adj, 3),
          " | F p-value =", round(model.f_pvalue, 4))

    # I build a safe filename (no spaces or special characters)
    safe_name = model_name.replace(" ", "_")
    safe_name = safe_name.replace("+", "Plus")
    safe_name = safe_name.replace("—", "")
    safe_name = safe_name.strip("_")

    # I save the full OLS results table to a text file
    table_path = "outputs/tables/ols_" + safe_name + ".txt"
    with open(table_path, "w") as f:
        f.write(model.summary().as_text())
    print("    Full OLS table saved →", table_path)

    return model


def run_ml(X, y, model_name, kfold):
    # ── Random Forest ─────────────────────────────────────────────────────────
    # A Random Forest builds 200 decision trees and averages their predictions.
    # It handles non-linear relationships and interactions between variables.
    rf = RandomForestRegressor(
        n_estimators = 200,      # I build 200 trees
        max_depth    = 4,        # I keep trees shallow to avoid overfitting
        min_samples_leaf = 3,    # Each leaf node needs at least 3 countries
        random_state = RANDOM_SEED,
        n_jobs       = -1,       # I use all available CPU cores
    )

    # I use 5-fold cross-validation to measure how well RF predicts
    # unseen countries. This is more honest than in-sample performance.
    rf_cv_scores = cross_val_score(rf, X, y, cv=kfold, scoring="r2")

    # I fit RF on all data (so I can compute SHAP values later)
    rf.fit(X, y)

    rf_r2_cv = rf_cv_scores.mean()

    # ── XGBoost ───────────────────────────────────────────────────────────────
    # XGBoost builds trees one at a time. Each new tree corrects the mistakes
    # of all previous trees. It is often the most accurate tree-based method.
    xgb_model = xgb.XGBRegressor(
        n_estimators     = 200,
        max_depth        = 3,
        learning_rate    = 0.05,   # Small steps = more careful, less overfitting
        subsample        = 0.8,    # I use 80% of the data per tree
        colsample_bytree = 0.8,    # I use 80% of features per tree
        random_state     = RANDOM_SEED,
        verbosity        = 0,      # I silence XGBoost's own output
    )

    xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=kfold, scoring="r2")
    xgb_model.fit(X, y)
    xgb_r2_cv = xgb_cv_scores.mean()

    print("  Random Forest  5-fold CV R² =", round(rf_r2_cv, 3),
          "(std ±", round(rf_cv_scores.std(), 3), ")")
    print("  XGBoost        5-fold CV R² =", round(xgb_r2_cv, 3),
          "(std ±", round(xgb_cv_scores.std(), 3), ")")

    return rf, xgb_model, rf_r2_cv, xgb_r2_cv


def make_shap_plot(rf_model, X, model_name):
    # SHAP tells me how much each variable pushed each prediction up or down.
    # This is much more informative than just saying "cereal yield is important".
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP feature importance — " + model_name, fontsize=11)
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_")
    safe_name = safe_name.replace("+", "Plus")
    safe_name = safe_name.replace("—", "")
    safe_name = safe_name.strip("_")

    fig_path = "outputs/figures/shap_" + safe_name + ".png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("    SHAP chart saved →", fig_path)


# ============================================================
# Step 5: I'm running all five models
# ============================================================

print("\n[3] Fitting Models A, B, C, D, E...")

# I set up the 5-fold cross-validation scheme
kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# I'll collect performance numbers from each model here
results = []

# I loop through each model specification
for model_name in MODELS:
    predictor_cols = MODELS[model_name]

    print("\n" + "=" * 60)
    print(" ", model_name)
    print("=" * 60)

    # I prepare the data for this model
    X, y, used_predictors = prepare_data(master, predictor_cols, DV)

    # I report which predictors are actually available
    missing_vars = [c for c in predictor_cols if c not in X.columns]
    if missing_vars:
        print("  NOTE: These variables were not available and are excluded:")
        for v in missing_vars:
            print("        -", v, "(download or merge failed — check Phase B / Phase C)")

    print("  Predictors being used:", used_predictors)
    print("  Countries with complete data:", len(X))

    # I need at least 30 countries for any meaningful result
    if len(X) < 30:
        print("  Skipping — fewer than 30 countries with complete data")
        continue

    # I run OLS regression
    ols_model = run_ols(X, y, model_name, used_predictors)

    # I run Random Forest and XGBoost
    print("\n  Machine learning results for", model_name, ":")
    rf_model, xgb_model, rf_cv_r2, xgb_cv_r2 = run_ml(X, y, model_name, kfold)

    # I compute and save the SHAP importance chart
    print("  Computing SHAP values...")
    make_shap_plot(rf_model, X, model_name)

    # I collect all the key numbers into one row
    result_row = {}
    result_row["Model"]              = model_name
    result_row["N (countries)"]      = len(X)
    result_row["Predictors used"]    = len(used_predictors)
    result_row["OLS R²"]             = round(ols_model.rsquared, 3)
    result_row["OLS Adj R²"]         = round(ols_model.rsquared_adj, 3)
    result_row["OLS F-stat p"]       = round(ols_model.f_pvalue, 4)
    result_row["RF 5-fold CV R²"]    = round(rf_cv_r2, 3)
    result_row["XGB 5-fold CV R²"]   = round(xgb_cv_r2, 3)
    results.append(result_row)


# ============================================================
# Step 6: I'm saving the comparison table
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON TABLE")
print("=" * 60)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

results_df.to_csv("outputs/tables/model_comparison.csv", index=False)
print("\nComparison table saved → outputs/tables/model_comparison.csv")


# ============================================================
# Step 7: I'm drawing the R² progression chart
# ============================================================
# This chart shows how each new block of variables improves
# the model's ability to predict food insecurity.
# A big jump from Model C to Model D would support my argument
# that value chain financial access matters more than national averages.

print("\nSaving R² progression chart...")

fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(results_df))
width = 0.25

# I draw three sets of bars: OLS, Random Forest, XGBoost
bars1 = ax.bar(x - width, results_df["OLS R²"],          width, label="OLS R²",            color="#4472C4")
bars2 = ax.bar(x,          results_df["RF 5-fold CV R²"], width, label="Random Forest CV R²", color="#ED7D31")
bars3 = ax.bar(x + width,  results_df["XGB 5-fold CV R²"],width, label="XGBoost CV R²",      color="#70AD47")

# I add value labels on top of every bar
for bar_group in [bars1, bars2, bars3]:
    for bar in bar_group:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.008, str(round(h, 2)),
                    ha="center", va="bottom", fontsize=7)

ax.set_xlabel("Model specification (each adds a new block of variables)")
ax.set_ylabel("R² (higher = better prediction of undernourishment %)")
ax.set_title(
    "Model Performance Progression: A → B → C → D → E\n"
    "Testing whether Value Chain Financial Access explains more than National Averages\n"
    "(Dependent variable: Prevalence of undernourishment, % of population — 2021)"
)

# I label the x axis with model names, split across two lines for readability
x_labels = []
for name in results_df["Model"]:
    parts = name.split(" — ")
    if len(parts) == 2:
        x_labels.append(parts[0] + "\n" + parts[1])
    else:
        x_labels.append(name)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=8)
ax.legend()
ax.set_ylim(0, 1.05)
ax.axhline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig("outputs/figures/model_r2_comparison.png", dpi=150)
plt.close()
print("R² progression chart saved → outputs/figures/model_r2_comparison.png")


# ============================================================
# Step 8: I'm printing the OLS coefficient table
# ============================================================
# This shows me which variables are statistically significant
# and in which direction they affect undernourishment.
# A negative coefficient means the variable REDUCES food insecurity.
# Significance levels: *** p<0.01  ** p<0.05  * p<0.10

print("\n" + "=" * 60)
print("OLS COEFFICIENT SUMMARY — ALL MODELS")
print("Stars: *** p<0.01  ** p<0.05  * p<0.10")
print("Negative coefficient = reduces undernourishment (good)")
print("=" * 60)

for model_name in MODELS:
    predictor_cols = MODELS[model_name]

    X, y, used_predictors = prepare_data(master, predictor_cols, DV)

    if len(X) < 30:
        continue

    X_const = sm.add_constant(X)
    fitted  = sm.OLS(y, X_const).fit()

    print("\n" + model_name, " (N=" + str(int(fitted.nobs)) + ",  R²=" + str(round(fitted.rsquared, 3)) + ")")
    print("  " + "-" * 58)
    print(f"  {'Variable':<38} {'Coef':>7} {'p-value':>9} {'Sig':>5}")
    print("  " + "-" * 58)

    for var in fitted.params.index:
        coef = fitted.params[var]
        pval = fitted.pvalues[var]

        if pval < 0.01:
            sig = "***"
        elif pval < 0.05:
            sig = "**"
        elif pval < 0.10:
            sig = "*"
        else:
            sig = ""

        print(f"  {var:<38} {coef:>7.3f} {pval:>9.4f} {sig:>5}")


# ============================================================
# Step 9: I'm drawing a coefficient summary chart for Model D
# ============================================================
# I plot the value chain financial access coefficients from Model D
# to show visually which channels of financial access matter most.

print("\nSaving value chain finance coefficient chart...")

X_d, y_d, used_d = prepare_data(master, MODEL_D_VARS, DV)

if len(X_d) >= 30:
    X_d_const = sm.add_constant(X_d)
    model_d   = sm.OLS(y_d, X_d_const).fit()

    # I pull out coefficients and confidence intervals for all variables
    coefs = model_d.params.drop("const")
    conf  = model_d.conf_int().drop("const")
    pvals = model_d.pvalues.drop("const")

    # I sort by coefficient magnitude so the chart is easy to read
    order = coefs.abs().sort_values(ascending=True).index
    coefs_sorted = coefs[order]
    conf_sorted  = conf.loc[order]
    pvals_sorted = pvals[order]

    fig, ax = plt.subplots(figsize=(10, max(5, len(coefs_sorted) * 0.45)))

    # I colour bars by significance
    colours = []
    for var in coefs_sorted.index:
        p = pvals_sorted[var]
        if p < 0.01:
            colours.append("#C00000")  # Dark red = highly significant
        elif p < 0.05:
            colours.append("#FF6600")  # Orange = significant
        elif p < 0.10:
            colours.append("#FFC000")  # Yellow = marginally significant
        else:
            colours.append("#AAAAAA")  # Grey = not significant

    # I draw horizontal bars (easier to read long variable names)
    y_pos = np.arange(len(coefs_sorted))
    ax.barh(y_pos, coefs_sorted.values, color=colours, alpha=0.85)

    # I draw horizontal confidence interval lines
    for i in range(len(coefs_sorted)):
        var_name = coefs_sorted.index[i]
        lo = conf_sorted.loc[var_name, 0]
        hi = conf_sorted.loc[var_name, 1]
        ax.plot([lo, hi], [i, i], color="black", linewidth=1.5)

    # I draw a vertical line at zero (no effect)
    ax.axvline(0, color="black", linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(coefs_sorted.index, fontsize=8)
    ax.set_xlabel("OLS Coefficient (effect on undernourishment %)")
    ax.set_title(
        "Model D — Value Chain Financial Access Coefficients\n"
        "(Colours: red=p<0.01, orange=p<0.05, yellow=p<0.10, grey=not significant)\n"
        "Negative = reduces undernourishment (beneficial)"
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/model_d_valuechain_coefficients.png", dpi=150)
    plt.close()
    print("Coefficient chart saved → outputs/figures/model_d_valuechain_coefficients.png")
else:
    print("  Skipping Model D coefficient chart — too few countries with complete data")


print("\n" + "=" * 60)
print("PHASE D COMPLETE")
print("Outputs saved:")
print("  outputs/tables/   — OLS tables (one per model) + comparison CSV")
print("  outputs/figures/  — SHAP charts + R² progression + Model D coefficients")
print("Next step: Phase E — outlier checks and robustness specifications")
print("=" * 60)
