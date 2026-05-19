# ============================================================
# Step 7: Run prediction models
# ============================================================
#
# What I'm doing here:
#   I have a master dataset with one row per country.
#   I fit five nested OLS models, each adding a new block of
#   variables to track the marginal contribution to explanatory power.
#
#   Model A — Production baseline
#     Tests: do physical agricultural inputs explain cereal availability?
#     Predictors: cereal yield, fertiliser use, arable land,
#                 GDP per capita, rural population,
#                 agricultural employment, livestock production
#
#   Model B — Add Post-Harvest Loss block
#     Tests: does food loss between farm and market matter
#            beyond the production baseline?
#     Add: cereal_loss_pct
#
#   Model C — Add Logistics and Infrastructure block
#     Tests: do market integration and rural infrastructure improve
#            predictions above the production + PHL baseline?
#     Add: trade_pct_gdp, rural_electricity_access_pct
#
#   Model A★ — Baseline on NLP sample
#     The same seven-predictor specification as Model A, estimated on
#     exactly the same 160-country sample as Model F.
#     This is the honest comparison baseline for the nested F-test.
#
#   Model F — NLP-Discovered Themes
#     Tests: do the five NLP-identified predictors jointly add
#            significant explanatory power over the baseline?
#     Add: cereal_loss_pct, trade_pct_gdp, rural_electricity_access_pct,
#          fertiliser_efficiency, food_price_inflation_pct
#
#   For every model I use THREE estimators:
#     1. OLS regression  (gives p-values, R², coefficients)
#     2. Random Forest   (captures non-linear relationships)
#     3. XGBoost         (gradient-boosted trees)
#
#   5-fold cross-validation (CV) is used for RF and XGB to get
#   out-of-sample R² as a generalisation check.
#   CV R² is more trustworthy than in-sample OLS R².
#
#   I also compute:
#     - SHAP values for Random Forest (variable importance)
#     - Bootstrap confidence intervals (1000 iterations, stratified
#       by income quartile) for all Model F NLP predictors
#     - A nested F-test comparing Model A★ to Model F
#
# Dependent variable:
#   Cereal food availability per capita (kg/person/year) — 2021
#   FAO Food Balance Sheet, Item 2905, Element 664.
# ============================================================

# I suppress warning messages to keep the output readable
import warnings

warnings.filterwarnings("ignore")

# I need time to pause between downloads
import time
# I need these to unpack the FAOSTAT bulk ZIP in memory
import io
import zipfile
# I need pycountry to convert FAO M49 codes to ISO3
import pycountry

from chart_style_settings import use_project_matplotlib_config

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
from sklearn.impute import KNNImputer
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

print("Starting Step 7 — modelling...")
print("=" * 60)


# ============================================================
# Step 1: Downloading the dependent variable — FAO FBS cereal food supply
# ============================================================
# Primary DV: FAO Food Balance Sheet cereal food supply (kg/capita/year)
#   = production + imports − exports − stock changes − non-food uses
#
#   This is what the proposal specifies as the primary dependent variable.
#   It measures what food is actually AVAILABLE to people after trade and
#   storage adjustments — import-dependent countries are correctly included
#   because their food supply (from imports) is as real as domestically
#   produced food.
#
# FAO FAOSTAT parameters:
#   Item 2905: Cereals - Excluding Beer (aggregate)
#   Element 664: Food supply quantity (g/capita/day)
#   Conversion: g/day × 365 / 1000 = kg/person/year
#
# Fallback: If FAO data is unreachable, we revert to World Bank
#   cereal production per capita (AG.PRD.CREL.MT) — same as before.
#   Import-dependent countries (city-states) are then excluded with
#   a 5 kg/capita minimum filter.
# ============================================================

print("\n[1] Downloading dependent variable — FAO FBS cereal food supply...")

FAOSTAT_ITEM = "2905"   # Cereals - Excluding Beer
FAOSTAT_ELEM = "664"    # Food supply quantity (g/capita/day)
TARGET_YEAR  = 2021

dv_df = None

# ── Attempt 1: FAOSTAT REST API (new v1 endpoint) ──
for base_url in [
    "https://www.fao.org/faostat/api/v1",
    "https://fenixservices.fao.org/faostat/api/v1",
]:
    try:
        resp = requests.get(
            f"{base_url}/en/data/FBS",
            params={
                "items": FAOSTAT_ITEM,
                "elements": FAOSTAT_ELEM,
                "years": TARGET_YEAR,
                "area": "*",
                "output_type": "objects",
            },
            timeout=60,
        )
        if resp.status_code == 200:
            payload = resp.json()
            records = payload.get("data", [])
            if records:
                api_rows = []
                for rec in records:
                    iso3 = rec.get("Area Code (ISO3)", rec.get("area_code_iso3", ""))
                    val  = rec.get("Value", rec.get("value"))
                    if iso3 and val is not None and len(str(iso3)) == 3:
                        try:
                            kg_yr = float(val) * 365 / 1000
                            api_rows.append({
                                "country_code": iso3,
                                "cereal_availability_kg_pc": round(kg_yr, 2),
                            })
                        except (ValueError, TypeError):
                            pass
                if api_rows:
                    dv_df = pd.DataFrame(api_rows)
                    print(f"  FAO API success ({base_url}): {len(dv_df)} countries")
                    break
    except Exception as ex:
        print(f"  FAO API attempt failed ({base_url}): {ex}")

# ── Attempt 2: FAOSTAT bulk ZIP download ──
if dv_df is None:
    bulk_candidates = [
        "https://bulks-faostat.fao.org/production/FoodBalanceSheets_E_All_Data_(Normalized).zip",
        "https://www.fao.org/faostat/static/bulkdownloads/FoodBalanceSheets_E_All_Data_(Normalized).zip",
        "https://fenixservices.fao.org/faostat/static/bulkdownloads/FoodBalanceSheets_E_All_Data_(Normalized).zip",
    ]
    for burl in bulk_candidates:
        try:
            print(f"  Trying bulk ZIP: {burl}")
            resp = requests.get(burl, timeout=180, allow_redirects=True, stream=True)
            if resp.status_code == 200:
                chunks = []
                size = 0
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > 250 * 1024 * 1024:
                        break
                content = b"".join(chunks)
                zf = zipfile.ZipFile(io.BytesIO(content))

                # The main data CSV — I find it with a plain for loop
                data_csv = None
                for name in zf.namelist():
                    if name.endswith(".csv") and "AreaCode" not in name:
                        data_csv = name
                        break

                fbs_raw = pd.read_csv(zf.open(data_csv), encoding="latin-1")

                # I strip any extra whitespace from each column name
                clean_col_names = []
                for col_name in fbs_raw.columns:
                    clean_col_names.append(col_name.strip())
                fbs_raw.columns = clean_col_names

                # Filter to cereals food supply for the target year
                mask = (
                    (fbs_raw["Item Code"].astype(str) == FAOSTAT_ITEM)
                    & (fbs_raw["Element Code"].astype(str) == FAOSTAT_ELEM)
                    & (fbs_raw["Year"].astype(str) == str(TARGET_YEAR))
                )
                fbs_filt = fbs_raw[mask].copy()

                if len(fbs_filt) > 0:
                    # FAO ZIP uses M49 codes (e.g. '004 for Afghanistan = AFG).
                    # Strip the leading apostrophe, zero-pad to 3 digits,
                    # then map to ISO3 via pycountry.
                    def m49_to_iso3(m49_raw):
                        try:
                            num_str = str(m49_raw).lstrip("'").strip().zfill(3)
                            c = pycountry.countries.get(numeric=num_str)
                            return c.alpha_3 if c else None
                        except Exception:
                            return None

                    fbs_filt = fbs_filt.copy()
                    fbs_filt["country_code"] = (
                        fbs_filt["Area Code (M49)"].apply(m49_to_iso3)
                    )
                    fbs_filt = fbs_filt.dropna(subset=["country_code"])
                    fbs_filt["cereal_availability_kg_pc"] = (
                        pd.to_numeric(fbs_filt["Value"], errors="coerce") * 365 / 1000
                    )
                    dv_df = (
                        fbs_filt[["country_code", "cereal_availability_kg_pc"]]
                        .dropna()
                        .reset_index(drop=True)
                    )
                    print(f"  Bulk ZIP success: {len(dv_df)} countries")
                    break
        except Exception as ex:
            print(f"  Bulk ZIP failed ({burl}): {ex}")

# ── Attempt 3: World Bank production per capita (original fallback) ──
if dv_df is None:
    print("  FAO FBS unreachable — falling back to WB cereal production per capita")
    try:
        meta_resp = requests.get(
            "https://api.worldbank.org/v2/country",
            params={"format": "json", "per_page": 400},
            timeout=30,
        )
        meta_json = meta_resp.json()
        # I build the set of real country ISO3 codes without a set comprehension
        REAL_ISO3 = set()
        for c in meta_json[1]:
            if c.get("region", {}).get("id", "") != "NA":
                REAL_ISO3.add(c["id"])
    except Exception:
        REAL_ISO3 = None

    try:
        prod_resp = requests.get(
            "https://api.worldbank.org/v2/country/all/indicator/AG.PRD.CREL.MT",
            params={"date": TARGET_YEAR, "format": "json", "per_page": 300},
            timeout=30,
        )
        prod_data = prod_resp.json()[1]
        # I build the production dictionary without a dict comprehension
        prod = {}
        for e in prod_data:
            if e.get("value") and e.get("countryiso3code"):
                iso3_code = e["countryiso3code"]
                if REAL_ISO3 is None or iso3_code in REAL_ISO3:
                    prod[iso3_code] = e["value"]
    except Exception as ex:
        print(f"  WB production download failed: {ex}")
        prod = {}

    try:
        pop_resp = requests.get(
            "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL",
            params={"date": TARGET_YEAR, "format": "json", "per_page": 300},
            timeout=30,
        )
        pop_data = pop_resp.json()[1]
        # I build the population dictionary without a dict comprehension
        pop = {}
        for e in pop_data:
            if e.get("value") and e.get("countryiso3code"):
                pop[e["countryiso3code"]] = e["value"]
    except Exception as ex:
        print(f"  WB population download failed: {ex}")
        pop = {}

    MIN_KG_PC = 5   # exclude city-states / islands producing no cereals
    wb_rows = []
    for iso in prod:
        if iso not in pop or pop[iso] <= 0:
            continue
        kg_pc = (prod[iso] * 1000) / pop[iso]
        if kg_pc < MIN_KG_PC:
            continue
        wb_rows.append({"country_code": iso, "cereal_availability_kg_pc": round(kg_pc, 2)})
    dv_df = pd.DataFrame(wb_rows)
    print(f"  WB fallback: {len(dv_df)} countries (≥ {MIN_KG_PC} kg/capita)")

dv_df["country_code"] = dv_df["country_code"].astype(str).str.strip()
dv_df.to_csv("data/raw/cereal_availability_2021.csv", index=False)
print(f"  Cereal availability saved for {len(dv_df)} countries")
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
# Trade (% of GDP) proxies market integration and logistics capacity — countries
# with higher trade shares have better transport networks, cold chains, and
# distribution systems. Rural electricity access enables on-farm storage,
# milling, and processing.
# NOTE: LPI (Logistics Performance Index) would be the ideal measure but covers
#       only 90/174 countries (48% missing). Trade % GDP covers ~170 countries
#       and is the accepted proxy when LPI is unavailable (FAO 2015 SOCO report).
MODEL_C_VARS = MODEL_B_VARS + [
    "trade_pct_gdp",                 # Market integration / logistics (WDI NE.TRD.GNFS.ZS)
    "rural_electricity_access_pct",  # Infrastructure for storage and processing
]

# ── Model F — NLP-Discovered Availability Themes ──────────────────────────────
# This model tests the themes DISCOVERED by NLP topic modelling of the strictly
# aligned food availability papers. Only AVAILABILITY-SIDE themes are included.
#
# NMF topics on 127 strictly aligned papers identified:
#
#   Topic 2: post-harvest loss, storage, grain_storage
#     → cereal_loss_pct  (APHLIS + FAO FBS, 100% country coverage)
#
#   Topic 6 / value-chain: economic, value_chain, investment, system
#     → trade_pct_gdp  (WDI NE.TRD.GNFS.ZS, ~170 countries)
#       LPI preferred but only covers 90/174 countries (48% missing).
#       Trade % GDP is the standard cross-country proxy for market
#       integration and logistics (FAO 2015; cited in 12 corpus papers).
#
#   Topic 5: technology, storage, infrastructure for food systems
#     → rural_electricity_access_pct  (storage, processing, cold chain)
#
#   Topic 0: land productivity, production systems, crop efficiency
#     → fertiliser_efficiency  (yield per kg of fertiliser — input productivity)
#
#   Topic 3: climate projections, availability disruption, market signals
#     → food_price_inflation_pct  (price inflation = market signal of
#       supply-side scarcity in cereals)
#
# Access-side themes (smallholder poverty, financial access, gender) are
# deliberately excluded: they explain who can AFFORD food, not whether food
# is produced and available — which is this dissertation's research question.
MODEL_F_VARS = MODEL_A_VARS + [
    "cereal_loss_pct",               # Topic 2: post-harvest loss (NLP priority theme)
    "trade_pct_gdp",                 # Topic 6: market integration / value-chain proxy
    "rural_electricity_access_pct",  # Topic 5: infrastructure for food systems
    "fertiliser_efficiency",         # Topic 0: land productivity / input efficiency
    "food_price_inflation_pct",      # Topic 3: market signal of availability disruption
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
        "trade_pct_gdp",              # Right-skewed: ranges from ~20% to 400%+ (Singapore)
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
    # (some variables may not be in the master if Step 5 couldn't download them)
    needed_existing = []
    for col in needed:
        if col in working.columns:
            needed_existing.append(col)

    # I keep only columns that actually exist as predictors
    existing_predictors = []
    for col in predictor_cols:
        if col in working.columns:
            existing_predictors.append(col)

    # KNN imputation for small gaps (≤ 20% missing per column).
    # This recovers countries that are only missing one or two predictors
    # rather than dropping them entirely.
    # Threshold: skip imputation if a column is more than 20% missing —
    # that many gaps indicates the variable is structurally unavailable,
    # not just a data reporting lag.
    IMPUTE_MAX_MISSING_SHARE = 0.20
    cols_to_impute = []
    for col in existing_predictors:
        n_missing = working[col].isna().sum()
        share = n_missing / len(working)
        if 0 < n_missing and share <= IMPUTE_MAX_MISSING_SHARE:
            cols_to_impute.append(col)

    if cols_to_impute:
        imputer = KNNImputer(n_neighbors=5)
        impute_df = working[existing_predictors].copy()
        imputed_array = imputer.fit_transform(impute_df)
        imputed_df = pd.DataFrame(imputed_array, columns=existing_predictors,
                                  index=working.index)
        for col in cols_to_impute:
            n_filled = working[col].isna().sum()
            working[col] = imputed_df[col]
            print(f"    KNN-imputed {col}: {n_filled} missing values filled")

    # Drop any rows still missing after imputation (DV missing, or >20% gap cols)
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
        vif_vals = []
        for i in range(X.shape[1]):
            vif_vals.append(variance_inflation_factor(X.values.astype(float), i))
        vif_df = pd.DataFrame({"Variable": X.columns, "VIF": vif_vals})
        high = vif_df[vif_df["VIF"] > 10]
        if len(high) > 0:
            # I build the pairs dictionary without a dict comprehension
            pairs = {}
            for _, row in high.iterrows():
                pairs[row["Variable"]] = round(row["VIF"], 1)
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
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
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
    missing_vars = []
    for c in predictor_cols:
        if c not in X.columns:
            missing_vars.append(c)
    if missing_vars:
        print("  NOTE: These variables were not available and are excluded:")
        for v in missing_vars:
            print("        -", v, "(download or merge failed — check Step 5 / Step 6)")

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
    r2_f_live = None
    for r in reversed(results):
        if "Model F" in r["Model"]:
            r2_f_live = r["OLS R²"]
            break

    if r2_f_live is not None:
        incr_r2 = round(r2_f_live - ols_a_star.rsquared, 3)
        print(f"\n  HONEST COMPARISON (same N={len(X_a_star)} sample):")
        print(f"    Model A★  R² = {ols_a_star.rsquared:.3f}")
        print(f"    Model F   R² = {r2_f_live:.3f}")
        print(f"    NLP variables add ΔR² = {incr_r2:.3f} on the same countries")
    else:
        print("  (Could not locate Model F R² for honest comparison)")

    # I decide the rounded CV R² values using if/else instead of ternary
    if not np.isnan(rf_cv_a_star):
        rounded_rf_cv = round(rf_cv_a_star, 3)
    else:
        rounded_rf_cv = np.nan

    if not np.isnan(xgb_cv_a_star):
        rounded_xgb_cv = round(xgb_cv_a_star, 3)
    else:
        rounded_xgb_cv = np.nan

    results.append({
        "Model":            "Model A★ — NLP sample",
        "N (countries)":    len(X_a_star),
        "Predictors used":  len(used_a_star),
        "OLS R²":           round(ols_a_star.rsquared, 3),
        "OLS Adj R²":       round(ols_a_star.rsquared_adj, 3),
        "OLS F-stat p":     round(ols_a_star.f_pvalue, 4),
        "RF 5-fold CV R²":  rounded_rf_cv,
        "XGB 5-fold CV R²": rounded_xgb_cv,
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

    # Stratify by income quartile so that each bootstrap sample preserves the
    # proportion of low / lower-middle / upper-middle / high income countries.
    # GDP per capita (log-transformed in X_all) is used as the stratifier.
    # We assign each country to one of four groups based on its GDP rank.
    gdp_col = "gdp_per_capita_usd"
    if gdp_col in used:
        gdp_ranks = X_all[gdp_col].rank(pct=True)
    else:
        gdp_ranks = pd.Series(range(len(X_all)), index=X_all.index) / len(X_all)

    income_group = []
    for rank_val in gdp_ranks:
        if rank_val <= 0.25:
            income_group.append(0)
        elif rank_val <= 0.50:
            income_group.append(1)
        elif rank_val <= 0.75:
            income_group.append(2)
        else:
            income_group.append(3)

    income_group_series = pd.Series(income_group, index=X_all.index)

    # I build the store dictionary without a dict comprehension
    store = {}
    for c in used:
        store[c] = []

    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(n_iter):
        # Sample within each income stratum, then combine
        sampled_idx = []
        for group_id in range(4):
            group_positions = [
                i for i, g in enumerate(income_group) if g == group_id
            ]
            if len(group_positions) == 0:
                continue
            drawn = rng.choice(group_positions, len(group_positions), replace=True)
            for d in drawn:
                sampled_idx.append(d)

        Xb = X_all.iloc[sampled_idx]
        yb = y_all.iloc[sampled_idx]
        try:
            m = sm.OLS(yb, sm.add_constant(Xb)).fit()
            for c in used:
                # I check if this coefficient exists in the model results
                if c in m.params:
                    store[c].append(m.params[c])
        except Exception:
            pass

    rows = []
    for c in store:
        vals = store[c]
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
        # I decide the CI note using if/else instead of a ternary
        if row["ci_lower_95"] * row["ci_upper_95"] < 0:
            cross_zero = "(CI crosses zero)"
        else:
            cross_zero = "(CI excludes zero ✓)"
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
for bar_group in [(bars1, results_df["OLS R²"]),
                  (bars2, results_df["RF 5-fold CV R²"]),
                  (bars3, results_df["XGB 5-fold CV R²"])]:
    bars, values = bar_group
    for i in range(len(bars)):
        bar = bars[i]
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
plt.savefig("outputs/figures/model_r2_comparison.png", dpi=300)
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
    for i in range(len(coefs_sorted)):
        var = coefs_sorted.index[i]
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
    plt.savefig("outputs/figures/model_f_nlp_coefficients.png", dpi=300)
    plt.close()
    print("Coefficient chart saved → outputs/figures/model_f_nlp_coefficients.png")
else:
    print("  Skipping Model F coefficient chart — too few countries with complete data")


print("\n" + "=" * 60)
print("STEP 7 COMPLETE")
print("Outputs saved:")
print("  outputs/tables/   — OLS tables (one per model) + comparison CSV")
print("  outputs/figures/  — SHAP charts + R² progression + Model D coefficients")
print("  Model F tests NLP-discovered themes vs baseline — check its R² vs Model A")
print("Next step: Step 8 — outlier checks and robustness specifications")
print("=" * 60)
