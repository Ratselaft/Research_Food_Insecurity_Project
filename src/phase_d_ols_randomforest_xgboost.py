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

from matplotlib_setup import use_project_matplotlib_config

use_project_matplotlib_config()
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
# I need this to check multicollinearity between predictors
from statsmodels.stats.outliers_influence import variance_inflation_factor
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

# ML cross-validation is unreliable below this sample size.
# With 5-fold CV each fold trains on ~80% of N — if N is 53,
# that is ~42 training observations for 11+ predictors, which leads
# to severe overfitting and negative out-of-sample R².
MIN_ML_N = 80

print("Starting Phase D — modelling...")
print("=" * 60)


# ============================================================
# Step 1: Downloading the dependent variable — cereal availability
# ============================================================
# What I am predicting:
#   Cereal production per capita (kg / person / year) — 2021
#   Derived from World Bank indicators:
#     AG.PRD.CREL.MT — Total cereal production (metric tons)
#     SP.POP.TOTL    — Total population
#
# WHY this DV instead of undernourishment:
#   The dissertation title says "Cereal Food Availability."
#   Availability = how much food is PRODUCED / accessible per person,
#   not whether people can afford to buy it (that is the ACCESS dimension).
#   Undernourishment mixes both availability AND access signals; it cannot
#   isolate which availability-side factors (yield, PHL, logistics) matter.
#   Cereal production per capita is a clean availability-side measure:
#   higher kg/person = more cereal food available in that country.
#
#   I use production per capita rather than net supply (which would
#   subtract exports and add imports) because FAO Food Balance Sheet
#   net supply data was unavailable via API at the time of analysis.
#   Countries where production ≈ 0 (e.g., city-states, islands that
#   import all cereals) are excluded with a 5 kg/person minimum filter,
#   since they represent a structurally different food system.

print("\n[1] Downloading dependent variable — cereal production per capita...")

# I use the World Bank country metadata to identify real countries
# (as opposed to regional and income-group aggregates)
try:
    meta_resp = requests.get(
        "https://api.worldbank.org/v2/country",
        params={"format": "json", "per_page": 400},
        timeout=30,
    )
    meta_json    = meta_resp.json()
    REAL_ISO3    = {c["id"] for c in meta_json[1]
                    if c.get("region", {}).get("id", "") != "NA"}
    print(f"  Real countries in World Bank metadata: {len(REAL_ISO3)}")
except Exception as e:
    print(f"  Warning: could not fetch WB metadata ({e}) — using hardcoded exclusion list")
    REAL_ISO3 = None   # will fall back to exclusion logic below

# Download cereal production (metric tons)
try:
    prod_resp = requests.get(
        "https://api.worldbank.org/v2/country/all/indicator/AG.PRD.CREL.MT",
        params={"date": 2021, "format": "json", "per_page": 300},
        timeout=30,
    )
    prod_data = prod_resp.json()[1]
    prod = {}
    for e in prod_data:
        iso = e.get("countryiso3code", "")
        val = e.get("value")
        if val is None or not iso:
            continue
        if REAL_ISO3 is not None and iso not in REAL_ISO3:
            continue
        prod[iso] = val        # metric tons
    print(f"  Cereal production data: {len(prod)} real countries")
except Exception as ex:
    print(f"  Could not download cereal production: {ex}")
    prod = {}

# Download population (total)
try:
    pop_resp = requests.get(
        "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL",
        params={"date": 2021, "format": "json", "per_page": 300},
        timeout=30,
    )
    pop_data = pop_resp.json()[1]
    pop = {}
    for e in pop_data:
        iso = e.get("countryiso3code", "")
        val = e.get("value")
        if val and iso:
            pop[iso] = val
except Exception as ex:
    print(f"  Could not download population: {ex}")
    pop = {}

# Compute cereal production per capita (kg / person)
#   1 metric ton = 1,000 kg  →  (tons × 1000) / population
MIN_KG_PC = 5   # exclude city-states / islands that produce no cereals

rows = []
for iso in prod:
    if iso not in pop or pop[iso] <= 0:
        continue
    kg_pc = (prod[iso] * 1000) / pop[iso]
    if kg_pc < MIN_KG_PC:
        continue  # country does not produce cereals — import-dependent
    rows.append({"country_code": iso, "cereal_availability_kg_pc": round(kg_pc, 2)})

dv_df = pd.DataFrame(rows)
dv_df.to_csv("data/raw/cereal_availability_2021.csv", index=False)
print(f"  Cereal availability computed for {len(dv_df)} countries (≥ {MIN_KG_PC} kg/capita)")
print(f"  Range: {dv_df['cereal_availability_kg_pc'].min():.1f} – "
      f"{dv_df['cereal_availability_kg_pc'].max():.1f} kg/person/year")


# ============================================================
# Step 2: I'm loading the master dataset and adding the DV
# ============================================================

print("\n[2] Loading master dataset and merging dependent variable...")

master = pd.read_csv("data/processed/master_dataset_clean.csv")

# Merge cereal availability onto the master table
master = master.merge(dv_df, on="country_code", how="left")

print("  Master size:", len(master), "countries")
print("  Countries with cereal availability data:",
      master["cereal_availability_kg_pc"].notna().sum())
print("  All columns in master dataset:")
for col in master.columns:
    n_valid = master[col].notna().sum()
    print(f"    {col:<45} ({n_valid} countries have data)")

master.to_csv("data/processed/master_dataset_with_dv.csv", index=False)


# ============================================================
# Step 3: I'm defining ALL model specifications
# ============================================================
# Each model builds on the previous one by adding a new block of variables.
# This lets me see exactly how much each block improves prediction.

# This is the variable I am trying to predict in every model
DV = "cereal_availability_kg_pc"

# ── Model A — Production baseline (availability-side) ─────────────────────────
# What physical and economic inputs drive how much cereal a country produces
# per person? These are the core AVAILABILITY-SIDE determinants: they determine
# how much food is grown, not whether people can afford to buy it.
MODEL_A_VARS = [
    "cereal_yield_kg_per_ha",     # Productivity: output per unit of land
    "fertiliser_kg_per_ha",       # Input intensity: investment in soil fertility
    "arable_land_pct",            # Resource base: share of land suitable for farming
    "gdp_per_capita_usd",         # Economic capacity for agricultural investment
    "rural_population_pct",       # Labour supply in the farming sector
    "agri_employment_pct",        # Actual workforce in agriculture
    "livestock_production_index", # Non-cereal food production capacity
]

# ── Model B — +Post-Harvest Loss ───────────────────────────────────────────────
# Food lost between the farm and the consumer is food that was PRODUCED but is
# NOT AVAILABLE. If a country loses 25% of its cereal harvest post-harvest,
# its effective cereal availability is 25% lower than its gross production.
# This is the most direct link between production and actual food availability.
# DISSERTATION HYPOTHESIS: PHL has a negative, significant effect on cereal
# availability beyond production capacity — literature attention (Topic 7 in LDA,
# strongest distinct topic) is empirically justified.
MODEL_B_VARS = MODEL_A_VARS + [
    "cereal_loss_pct",            # % of harvested cereal lost before reaching people
]

# ── Model C — +Logistics and Infrastructure ─────────────────────────────────
# Even if food is produced and not lost, it must MOVE from farms to consumers.
# Logistics quality (LPI) measures how efficiently goods traverse a country:
# roads, cold chains, customs, warehousing. Rural electricity access enables
# storage, milling, and processing — turning raw grain into available food.
# Both are availability-side: they determine whether produced food reaches people.
# NOTE: LPI data covers only ~90 countries, so N drops here from Model B.
MODEL_C_VARS = MODEL_B_VARS + [
    "lpi_overall",                   # Logistics Performance Index (World Bank)
    "rural_electricity_access_pct",  # Infrastructure for storage and processing
]

# ── Model F — NLP-Discovered Availability Themes ──────────────────────────────
# This model tests the themes DISCOVERED by NLP topic modelling of 328 strictly
# aligned food insecurity papers. Only AVAILABILITY-SIDE themes are included.
#
# LDA K=9 topics on 328 focused papers (coherence=0.384) identified:
#
#   Topic 7 (clearest signal): post-harvest loss, storage, grain_storage
#     → cereal_loss_pct  (APHLIS + FAO FBS, 100% country coverage)
#
#   Topic 3: agricultural investment, logistics, financing frameworks
#     → lpi_overall  (logistics quality: goods move from farms to markets)
#
#   Topic 2 & 4: climate-smart agriculture, adaptation infrastructure
#     → rural_electricity_access_pct  (storage, processing, cold chain)
#
#   Topic 6: land productivity, production systems, crop efficiency
#     → fertiliser_efficiency  (yield per kg of fertiliser — input productivity)
#
#   Topic 1: climate projections, availability disruption, market signals
#     → food_price_inflation_pct  (price inflation = market signal of
#       supply-side scarcity in cereals)
#
# Access-side themes (smallholder poverty, financial access, gender) are
# deliberately excluded: they explain who can AFFORD food, not whether food
# is produced and available — which is this dissertation's research question.
MODEL_F_VARS = MODEL_A_VARS + [
    "cereal_loss_pct",               # Topic 7: post-harvest loss (NLP priority theme)
    "lpi_overall",                   # Topic 3: logistics and market investment
    "rural_electricity_access_pct",  # Topics 2/4: infrastructure for food systems
    "fertiliser_efficiency",         # Topic 6: land productivity / input efficiency
    "food_price_inflation_pct",      # Topic 1: market signal of availability disruption
]

# Three models (as per research proposal) + Model F (NLP contribution)
MODELS = {}
MODELS["Model A — Baseline Production"]       = MODEL_A_VARS
MODELS["Model B — +Post-Harvest Loss"]        = MODEL_B_VARS
MODELS["Model C — +Logistics Infrastructure"] = MODEL_C_VARS
MODELS["Model F — NLP-Discovered Themes"]     = MODEL_F_VARS


# ============================================================
# Step 4: I'm writing helper functions
# ============================================================

def prepare_data(df, predictor_cols, outcome_col):
    # I make a copy so I don't accidentally change the original
    working = df.copy()

    # I log-transform variables that are heavily right-skewed.
    # Log transformation stabilises variance and gives elasticity interpretations.
    # For the DV (cereal_availability_kg_pc), log-transforming means coefficients
    # represent the % change in cereal availability per unit change in the predictor.
    LOG_COLS = [
        "cereal_availability_kg_pc",  # DV: right-skewed (range: ~5 to 2,100 kg/person)
        "gdp_per_capita_usd",
        "cereal_yield_kg_per_ha",
        "population_total",
        "fertiliser_kg_per_ha",
        "fertiliser_efficiency",      # Ratio: can be very right-skewed
        "bank_branches_per_100k",
        "atm_per_100k",
        "private_credit_pct_gdp",
    ]

    DV_LOG_COL = "cereal_availability_kg_pc"
    if outcome_col == DV_LOG_COL and DV_LOG_COL in working.columns:
        working[DV_LOG_COL] = np.log1p(working[DV_LOG_COL].clip(lower=0))

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

    # I fit OLS with HC3 heteroskedasticity-consistent standard errors.
    # HC3 is the most conservative sandwich estimator and is appropriate when
    # residuals are non-normal (which Jarque-Bera tests reveal in Model F).
    # HC3 does not change coefficients or R² — only standard errors and p-values.
    model = sm.OLS(y, X_const).fit(cov_type="HC3")

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
    print("    Full OLS table (HC3 SEs) saved →", table_path)

    return model


def check_vif(X):
    """Print VIF for each predictor. VIF > 10 signals severe multicollinearity."""
    try:
        vif_vals = [
            variance_inflation_factor(X.values.astype(float), i)
            for i in range(X.shape[1])
        ]
        vif_df = pd.DataFrame({"Variable": X.columns, "VIF": vif_vals})
        high = vif_df[vif_df["VIF"] > 10]
        if len(high) > 0:
            pairs = {row["Variable"]: round(row["VIF"], 1) for _, row in high.iterrows()}
            print("  Multicollinearity warning — VIF > 10:", pairs)
        else:
            max_vif = round(vif_df["VIF"].max(), 1)
            print(f"  VIF: all predictors below 10 (max = {max_vif}) — no severe collinearity")
    except Exception as e:
        print(f"  VIF check skipped ({e})")


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
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)

    safe_name = model_name.replace(" ", "_")
    safe_name = safe_name.replace("+", "Plus")
    safe_name = safe_name.replace("—", "")
    safe_name = safe_name.strip("_")

    # I save mean absolute SHAP value per variable as a CSV for downstream analysis
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_csv = pd.DataFrame({
        "variable":       X.columns.tolist(),
        "mean_abs_shap":  mean_abs_shap.round(4),
    }).sort_values("mean_abs_shap", ascending=False)
    csv_path = "outputs/tables/shap_" + safe_name + ".csv"
    shap_csv.to_csv(csv_path, index=False)
    print("    SHAP importance CSV saved →", csv_path)

    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP feature importance — " + model_name, fontsize=11)
    plt.tight_layout()

    fig_path = "outputs/figures/shap_" + safe_name + ".png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("    SHAP chart saved →", fig_path)


# ============================================================
# Step 5: I'm running all five models
# ============================================================

print("\n[3] Fitting Models A, B, C, D, E, F...")

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

    # I check for multicollinearity between predictors
    check_vif(X)

    # I run Random Forest and XGBoost only when there are enough countries.
    # Below MIN_ML_N the 5-fold CV folds are too small to be meaningful,
    # producing negative out-of-sample R² due to overfitting.
    print("\n  Machine learning results for", model_name, ":")
    if len(X) >= MIN_ML_N:
        rf_model, xgb_model, rf_cv_r2, xgb_cv_r2 = run_ml(X, y, model_name, kfold)
    else:
        print(f"  Sample N={len(X)} < {MIN_ML_N} — skipping CV (too few countries for "
              f"reliable cross-validation). Fitting RF in-sample only for SHAP.")
        rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=3,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
        rf_model.fit(X, y)
        rf_cv_r2  = np.nan
        xgb_cv_r2 = np.nan

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
# Step 5b: Model A★ — same N=80 sample as Model F (honest comparison)
# ============================================================
# Comparing Model F R²=0.721 (N=80) with Model A R²=0.553 (N=158) is
# misleading because the samples differ. Countries with LPI data tend to be
# better-governed, wealthier, and easier to predict. Running Model A on the
# IDENTICAL 80 countries gives the honest incremental R² attributable to the
# NLP-discovered variables, not sample composition.

print("\n" + "=" * 60)
print("  Model A★ — Restricted to Model F sample (fair comparison)")
print("=" * 60)

X_f_temp, y_f_temp, _ = prepare_data(master, MODEL_F_VARS, DV)
master_nlp_sample = master.loc[X_f_temp.index].copy()

X_a_star, y_a_star, used_a_star = prepare_data(master_nlp_sample, MODEL_A_VARS, DV)

if len(X_a_star) >= 30:
    ols_a_star = run_ols(X_a_star, y_a_star, "Model A★ — NLP sample", used_a_star)
    check_vif(X_a_star)

    if len(X_a_star) >= MIN_ML_N:
        rf_a_star, _, rf_cv_a_star, xgb_cv_a_star = run_ml(
            X_a_star, y_a_star, "Model A★", kfold
        )
    else:
        rf_a_star = RandomForestRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=3,
            random_state=RANDOM_SEED, n_jobs=-1,
        ).fit(X_a_star, y_a_star)
        rf_cv_a_star = np.nan
        xgb_cv_a_star = np.nan

    print("  Computing SHAP values for Model A★...")
    make_shap_plot(rf_a_star, X_a_star, "Model A★ — NLP sample")

    # Get Model F R² from the results list (last Model F entry)
    r2_f_live = next(
        (r["OLS R²"] for r in reversed(results) if "Model F" in r["Model"]),
        None
    )
    if r2_f_live is not None:
        incr_r2 = round(r2_f_live - ols_a_star.rsquared, 3)
        print(f"\n  HONEST COMPARISON (same N={len(X_a_star)} sample):")
        print(f"    Model A★  R² = {ols_a_star.rsquared:.3f}")
        print(f"    Model F   R² = {r2_f_live:.3f}")
        print(f"    NLP variables add ΔR² = {incr_r2:.3f} on the same countries")
    else:
        print("  (Could not locate Model F R² for honest comparison)")

    results.append({
        "Model":            "Model A★ — NLP sample",
        "N (countries)":    len(X_a_star),
        "Predictors used":  len(used_a_star),
        "OLS R²":           round(ols_a_star.rsquared, 3),
        "OLS Adj R²":       round(ols_a_star.rsquared_adj, 3),
        "OLS F-stat p":     round(ols_a_star.f_pvalue, 4),
        "RF 5-fold CV R²":  round(rf_cv_a_star, 3) if not np.isnan(rf_cv_a_star) else np.nan,
        "XGB 5-fold CV R²": round(xgb_cv_a_star, 3) if not np.isnan(xgb_cv_a_star) else np.nan,
    })
else:
    print("  Skipped Model A★ — too few countries with complete data")


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
# Step 6b: 1000-iteration bootstrap confidence intervals
# ============================================================
# The research proposal specifies bootstrap CIs for key predictors.
# Bootstrap sampling (with replacement, N iterations) produces an
# empirical distribution of each coefficient. The 2.5th and 97.5th
# percentiles give the 95% CI without assuming normality.
# This matters most for Model F where residuals are non-normal (JB p<0.001).

print("\n[3b] Computing 1000-iteration bootstrap CIs...")

N_BOOT = 1000

def bootstrap_coefs(df, predictor_cols, outcome_col, n_iter=N_BOOT):
    X_all, y_all, used = prepare_data(df, predictor_cols, outcome_col)
    if len(X_all) < 30:
        return pd.DataFrame()
    store = {c: [] for c in used}
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(n_iter):
        idx = rng.choice(len(X_all), len(X_all), replace=True)
        Xb, yb = X_all.iloc[idx], y_all.iloc[idx]
        try:
            m = sm.OLS(yb, sm.add_constant(Xb)).fit()
            for c in used:
                if c in m.params:
                    store[c].append(m.params[c])
        except Exception:
            pass
    rows = []
    for c, vals in store.items():
        if len(vals) >= 50:
            rows.append({
                "variable":    c,
                "boot_mean":   round(float(np.mean(vals)),              4),
                "ci_lower_95": round(float(np.percentile(vals,  2.5)),  4),
                "ci_upper_95": round(float(np.percentile(vals, 97.5)),  4),
                "n_valid":     len(vals),
            })
    return pd.DataFrame(rows)

print("  Bootstrapping Model A (1000 iterations)...")
boot_a = bootstrap_coefs(master, MODEL_A_VARS, DV)
boot_a["model"] = "Model A"

print("  Bootstrapping Model F (1000 iterations)...")
boot_f = bootstrap_coefs(master, MODEL_F_VARS, DV)
boot_f["model"] = "Model F"

boot_all = pd.concat([boot_a, boot_f], ignore_index=True)
boot_all.to_csv("outputs/tables/bootstrap_confidence_intervals.csv", index=False)
print("  Bootstrap CIs saved → outputs/tables/bootstrap_confidence_intervals.csv")

# I print CIs for the NLP-discovered availability-side predictors
nlp_vars = ["cereal_loss_pct", "lpi_overall", "rural_electricity_access_pct",
            "fertiliser_efficiency", "food_price_inflation_pct"]
print("\n  Bootstrap 95% CIs — NLP-discovered availability predictors (Model F):")
print(f"  {'Variable':<38} {'Mean':>8} {'Lower 95%':>10} {'Upper 95%':>10}")
print("  " + "-" * 68)
for _, row in boot_f.iterrows():
    if row["variable"] in nlp_vars:
        cross_zero = "(CI crosses zero)" if row["ci_lower_95"] * row["ci_upper_95"] < 0 else "(CI excludes zero ✓)"
        print(f"  {row['variable']:<38} {row['boot_mean']:>8.4f} "
              f"{row['ci_lower_95']:>10.4f} {row['ci_upper_95']:>10.4f}  {cross_zero}")


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

# Replace NaN with 0 for bar heights (NaN bars show as invisible gaps)
rf_heights  = results_df["RF 5-fold CV R²"].fillna(0)
xgb_heights = results_df["XGB 5-fold CV R²"].fillna(0)

# I draw three sets of bars: OLS, Random Forest, XGBoost
bars1 = ax.bar(x - width, results_df["OLS R²"], width, label="OLS R²",              color="#4472C4")
bars2 = ax.bar(x,          rf_heights,           width, label="Random Forest CV R²", color="#ED7D31")
bars3 = ax.bar(x + width,  xgb_heights,          width, label="XGBoost CV R²",       color="#70AD47")

# I add value labels on top of every bar; NaN models get an "N/A" note
for bar_idx, bar_group in enumerate([(bars1, results_df["OLS R²"]),
                                      (bars2, results_df["RF 5-fold CV R²"]),
                                      (bars3, results_df["XGB 5-fold CV R²"])]):
    bars, values = bar_group
    for i, bar in enumerate(bars):
        val = values.iloc[i]
        h   = bar.get_height()
        if pd.isna(val):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.015,
                    "N/A", ha="center", va="bottom", fontsize=6, color="#888888")
        elif h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    str(round(h, 2)), ha="center", va="bottom", fontsize=7)

ax.set_xlabel("Model specification (each adds a new block of variables)")
ax.set_ylabel("R² (higher = better prediction of cereal availability)")
ax.set_title(
    "Model Performance Progression: A → B → C → D → E → F\n"
    "Does NLP-Discovered Evidence Match Empirical Signal? (Model F = NLP themes)\n"
    "(Dependent variable: Cereal production per capita, kg/person/year — 2021, log-transformed)"
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
print("Positive coefficient = increases cereal availability (good)")
print("=" * 60)

for model_name in MODELS:
    predictor_cols = MODELS[model_name]

    X, y, used_predictors = prepare_data(master, predictor_cols, DV)

    if len(X) < 30:
        continue

    X_const = sm.add_constant(X)
    fitted  = sm.OLS(y, X_const).fit(cov_type="HC3")

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
# Step 9: NLP-discovered availability coefficients chart (Model F)
# ============================================================
# I highlight the NLP-discovered availability-side predictors in Model F
# to show visually whether they add explanatory power beyond the baseline.

print("\nSaving NLP availability coefficient chart (Model F)...")

X_f_chart, y_f_chart, used_f_chart = prepare_data(master, MODEL_F_VARS, DV)

if len(X_f_chart) >= 30:
    X_f_const = sm.add_constant(X_f_chart)
    model_f_chart = sm.OLS(y_f_chart, X_f_const).fit(cov_type="HC3")

    coefs = model_f_chart.params.drop("const")
    conf  = model_f_chart.conf_int().drop("const")
    pvals = model_f_chart.pvalues.drop("const")

    order        = coefs.abs().sort_values(ascending=True).index
    coefs_sorted = coefs[order]
    conf_sorted  = conf.loc[order]
    pvals_sorted = pvals[order]

    # NLP-discovered availability variables (orange border)
    nlp_highlight = set(MODEL_F_VARS) - set(MODEL_A_VARS)

    fig, ax = plt.subplots(figsize=(10, max(5, len(coefs_sorted) * 0.45)))

    colours = []
    for var in coefs_sorted.index:
        p = pvals_sorted[var]
        if p < 0.01:
            colours.append("#C00000")
        elif p < 0.05:
            colours.append("#FF6600")
        elif p < 0.10:
            colours.append("#FFC000")
        else:
            colours.append("#AAAAAA")

    y_pos = np.arange(len(coefs_sorted))
    bars  = ax.barh(y_pos, coefs_sorted.values, color=colours, alpha=0.85)

    # Thicker border for NLP-discovered variables
    for i, var in enumerate(coefs_sorted.index):
        if var in nlp_highlight:
            bars[i].set_edgecolor("#1565C0")
            bars[i].set_linewidth(2)

    for i in range(len(coefs_sorted)):
        var_name = coefs_sorted.index[i]
        lo = conf_sorted.loc[var_name, 0]
        hi = conf_sorted.loc[var_name, 1]
        ax.plot([lo, hi], [i, i], color="black", linewidth=1.5)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coefs_sorted.index, fontsize=8)
    ax.set_xlabel("OLS Coefficient HC3 (effect on log cereal availability per capita)")
    ax.set_title(
        "Model F — NLP-Discovered Availability Themes (HC3 SEs)\n"
        "(Blue border = NLP-discovered; colours: red=p<0.01, orange=p<0.05, "
        "yellow=p<0.10, grey=n.s.)\n"
        "Positive = increases cereal availability"
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/model_f_nlp_coefficients.png", dpi=150)
    plt.close()
    print("Coefficient chart saved → outputs/figures/model_f_nlp_coefficients.png")
else:
    print("  Skipping Model F coefficient chart — too few countries with complete data")


print("\n" + "=" * 60)
print("PHASE D COMPLETE")
print("Outputs saved:")
print("  outputs/tables/   — OLS tables (one per model) + comparison CSV")
print("  outputs/figures/  — SHAP charts + R² progression + Model D coefficients")
print("  Model F tests NLP-discovered themes vs baseline — check its R² vs Model A")
print("Next step: Phase E — outlier checks and robustness specifications")
print("=" * 60)
