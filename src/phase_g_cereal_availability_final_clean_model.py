# ============================================================
# PHASE G — Final cereal availability cleaning and rigorous modelling
# ============================================================
#
# Why I am adding this phase:
#   I am keeping the dissertation focused on CEREAL FOOD AVAILABILITY.
#   I am therefore moving the final modelling away from undernourishment
#   and towards cereal availability as the dependent variable.
#
# What this script fixes:
#   1. I remove noisy papers from the NLP corpus so irrelevant agriculture,
#      biomedical, detection-only, and marketing-only papers do not shape my themes.
#   2. I build a cereal availability dependent variable, preferring FAOSTAT
#      Food Balance Sheet cereal food supply when available.
#   3. I restrict Model F to availability-side variables only.
#   4. I report RMSE as well as R², because the proposal promised both.
#   5. I use HC3 robust standard errors for OLS.
#   6. I save VIF diagnostics and Ridge results to manage multicollinearity.
#   7. I compare models on the same sample, so R² gains are not caused by
#      changing country coverage.
#   8. I bootstrap block-level ΔR² so the dissertation can report uncertainty.
#
# Output files:
#   data/processed/availability_aligned_papers.csv
#   data/processed/corpus_noise_audit.csv
#   data/processed/master_dataset_cereal_availability_final.csv
#   outputs/tables/final_cereal_availability_model_comparison.csv
#   outputs/tables/final_cereal_availability_vif.csv
#   outputs/tables/final_cereal_availability_ridge.csv
#   outputs/tables/final_block_delta_r2_bootstrap.csv
#   outputs/tables/final_same_sample_comparison.csv
# ============================================================

import os
import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

print("PHASE G — cereal availability final cleaning and modelling")
print("=" * 70)


# ============================================================
# 1. I clean the literature corpus so noisy papers do not dominate NLP
# ============================================================

AVAILABILITY_TERMS = [
    "cereal", "grain", "maize", "wheat", "rice", "sorghum", "millet",
    "food availability", "food supply", "food balance", "production",
    "yield", "post-harvest", "postharvest", "storage", "loss", "waste",
    "logistics", "infrastructure", "supply chain", "value chain", "market access",
    "fertiliser", "fertilizer", "irrigation", "arable", "smallholder",
]

FOOD_SECURITY_TERMS = [
    "food security", "food insecurity", "availability", "undernourishment",
    "hunger", "nutrition", "dietary energy", "food system",
]

EXCLUSION_TERMS = [
    "aflatoxin", "immunoassay", "lateral flow", "biosensor", "leaf disease",
    "plant disease detection", "image classification", "spectroscopy", "microbial",
    "pathogen", "mycotoxin detection", "gene expression", "soil bacteria",
    "potato marketing", "consumer preference", "sensory evaluation", "restaurant",
    "aquaculture", "fish", "livestock disease", "veterinary", "clinical trial",
]


def normalise_text(value):
    # I standardise text so that keyword checks are consistent.
    if pd.isna(value):
        return ""
    return str(value).lower()


def count_keyword_hits(text, terms):
    # I count how many concept terms appear in the title and abstract.
    hits = 0
    for term in terms:
        if term in text:
            hits = hits + 1
    return hits


def score_literature_row(row):
    # I combine title and abstract because titles are precise and abstracts add context.
    title = normalise_text(row.get("title", ""))
    abstract = normalise_text(row.get("abstract", ""))
    combined = title + " " + abstract

    availability_hits = count_keyword_hits(combined, AVAILABILITY_TERMS)
    food_security_hits = count_keyword_hits(combined, FOOD_SECURITY_TERMS)
    exclusion_hits = count_keyword_hits(combined, EXCLUSION_TERMS)

    # I give title hits extra weight because the paper's title usually states its core topic.
    title_availability_hits = count_keyword_hits(title, AVAILABILITY_TERMS)
    title_food_security_hits = count_keyword_hits(title, FOOD_SECURITY_TERMS)

    alignment_score = availability_hits + food_security_hits
    alignment_score = alignment_score + title_availability_hits
    alignment_score = alignment_score + title_food_security_hits
    alignment_score = alignment_score - (2 * exclusion_hits)

    # I keep a paper only when it is truly aligned with cereal/food availability.
    # I reject papers where exclusion terms dominate the evidence of relevance.
    keep = False
    if alignment_score >= 4 and exclusion_hits <= availability_hits:
        keep = True
    if food_security_hits >= 1 and availability_hits >= 2 and exclusion_hits <= 1:
        keep = True
    if exclusion_hits >= 2 and food_security_hits == 0:
        keep = False

    return pd.Series({
        "availability_hits": availability_hits,
        "food_security_hits": food_security_hits,
        "exclusion_hits": exclusion_hits,
        "alignment_score": alignment_score,
        "keep_for_availability_nlp": keep,
    })


def clean_literature_corpus():
    # I read the broad corpus if it exists, then create a stricter availability corpus.
    corpus_path = "data/raw/corpus_metadata.csv"
    if not os.path.exists(corpus_path):
        print("No corpus_metadata.csv found — skipping corpus cleaning.")
        return None

    corpus = pd.read_csv(corpus_path)
    if "title" not in corpus.columns:
        print("Corpus has no title column — skipping corpus cleaning.")
        return None
    if "abstract" not in corpus.columns:
        corpus["abstract"] = ""

    scores = corpus.apply(score_literature_row, axis=1)
    audited = pd.concat([corpus, scores], axis=1)

    # I deduplicate by DOI first, then by normalised title when DOI is missing.
    if "doi" in audited.columns:
        audited["doi_clean"] = audited["doi"].fillna("").astype(str).str.lower().str.strip()
        with_doi = audited[audited["doi_clean"] != ""].copy()
        without_doi = audited[audited["doi_clean"] == ""].copy()
        with_doi = with_doi.sort_values("alignment_score", ascending=False)
        with_doi = with_doi.drop_duplicates(subset=["doi_clean"], keep="first")
        audited = pd.concat([with_doi, without_doi], ignore_index=True)

    audited["title_clean"] = audited["title"].fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    audited = audited.sort_values("alignment_score", ascending=False)
    audited = audited.drop_duplicates(subset=["title_clean"], keep="first")

    strict = audited[audited["keep_for_availability_nlp"] == True].copy()
    strict = strict.sort_values("alignment_score", ascending=False)

    audited.to_csv("data/processed/corpus_noise_audit.csv", index=False)
    strict.to_csv("data/processed/availability_aligned_papers.csv", index=False)

    print("\n[1] Corpus cleaning")
    print("  Broad corpus papers:", len(corpus))
    print("  After deduplication:", len(audited))
    print("  Strict availability-aligned papers:", len(strict))
    print("  Saved: data/processed/availability_aligned_papers.csv")
    print("  Saved: data/processed/corpus_noise_audit.csv")

    return strict


availability_corpus = clean_literature_corpus()


# ============================================================
# 2. I build a cereal food availability dependent variable
# ============================================================

def get_real_country_codes_from_world_bank():
    # I use World Bank metadata to remove regional aggregates.
    try:
        response = requests.get(
            "https://api.worldbank.org/v2/country",
            params={"format": "json", "per_page": 400},
            timeout=30,
        )
        data = response.json()
        real_codes = set()
        for row in data[1]:
            region_id = row.get("region", {}).get("id", "")
            iso3 = row.get("id", "")
            if region_id != "NA" and iso3 != "":
                real_codes.add(iso3)
        return real_codes
    except Exception:
        return None


def find_column(df, candidates):
    # I find the first likely column name in a messy downloaded dataset.
    lower_map = {}
    for col in df.columns:
        lower_map[col.lower().strip()] = col
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None


def build_dv_from_faostat_fbs():
    # I prefer FAOSTAT Food Balance Sheets because they measure food supply/availability.
    # I search for cereal food supply quantity in kg/capita/year.
    path = "data/raw/faostat_food_supply.csv"
    if not os.path.exists(path):
        return None, "FAOSTAT FBS file not found"

    try:
        fbs = pd.read_csv(path, low_memory=False)
    except Exception as ex:
        return None, "Could not read FAOSTAT FBS: " + str(ex)

    area_col = find_column(fbs, ["Area Code (ISO3)", "Area Code", "area_code", "country_code"])
    area_name_col = find_column(fbs, ["Area", "Area Name", "country_name"])
    item_col = find_column(fbs, ["Item", "Item Name", "item"])
    element_col = find_column(fbs, ["Element", "Element Name", "element"])
    value_col = find_column(fbs, ["Value", "value"])
    year_col = find_column(fbs, ["Year", "year"])

    required = [area_col, item_col, element_col, value_col]
    if any(col is None for col in required):
        return None, "FAOSTAT FBS columns not recognised"

    working = fbs.copy()
    working["_item"] = working[item_col].fillna("").astype(str).str.lower()
    working["_element"] = working[element_col].fillna("").astype(str).str.lower()

    cereal_mask = working["_item"].str.contains("cereal", na=False)
    kg_mask = working["_element"].str.contains("food supply quantity", na=False)
    kg_mask = kg_mask & working["_element"].str.contains("kg", na=False)

    filtered = working[cereal_mask & kg_mask].copy()
    if len(filtered) == 0:
        return None, "No cereal food supply kg/capita/year rows found in FBS"

    if year_col is not None:
        filtered["_year_numeric"] = pd.to_numeric(filtered[year_col], errors="coerce")
        filtered = filtered[filtered["_year_numeric"] <= 2021]
        filtered = filtered.sort_values("_year_numeric")
        filtered = filtered.groupby(area_col, as_index=False).tail(1)

    filtered["cereal_availability_kg_pc"] = pd.to_numeric(filtered[value_col], errors="coerce")
    filtered = filtered.dropna(subset=["cereal_availability_kg_pc"])

    out = pd.DataFrame()
    out["country_code"] = filtered[area_col].astype(str)
    if area_name_col is not None:
        out["country_name_from_dv"] = filtered[area_name_col].astype(str)
    out["cereal_availability_kg_pc"] = filtered["cereal_availability_kg_pc"]
    out["cereal_availability_source"] = "FAOSTAT FBS cereal food supply quantity kg/capita/year"

    return out, "Built DV from FAOSTAT FBS cereal food supply quantity"


def build_dv_from_world_bank_production_proxy():
    # I use this only as a fallback when FBS food supply is unavailable.
    # This is production per capita, not net supply, so the dissertation must label it as a proxy.
    real_codes = get_real_country_codes_from_world_bank()

    try:
        prod_response = requests.get(
            "https://api.worldbank.org/v2/country/all/indicator/AG.PRD.CREL.MT",
            params={"date": 2021, "format": "json", "per_page": 300},
            timeout=30,
        )
        pop_response = requests.get(
            "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL",
            params={"date": 2021, "format": "json", "per_page": 300},
            timeout=30,
        )
        prod_data = prod_response.json()[1]
        pop_data = pop_response.json()[1]
    except Exception as ex:
        return None, "Could not download World Bank production proxy: " + str(ex)

    prod = {}
    pop = {}

    for entry in prod_data:
        iso = entry.get("countryiso3code", "")
        val = entry.get("value")
        if iso == "" or val is None:
            continue
        if real_codes is not None and iso not in real_codes:
            continue
        prod[iso] = val

    for entry in pop_data:
        iso = entry.get("countryiso3code", "")
        val = entry.get("value")
        if iso == "" or val is None:
            continue
        pop[iso] = val

    rows = []
    for iso in prod:
        if iso not in pop:
            continue
        if pop[iso] <= 0:
            continue
        kg_pc = (prod[iso] * 1000.0) / pop[iso]
        if kg_pc < 5:
            continue
        rows.append({
            "country_code": iso,
            "cereal_availability_kg_pc": round(kg_pc, 3),
            "cereal_availability_source": "World Bank cereal production per capita proxy",
        })

    return pd.DataFrame(rows), "Built fallback DV from World Bank cereal production per capita"


def build_final_master_dataset():
    # I merge the chosen cereal availability DV onto the cleaned master dataset.
    master_path = "data/processed/master_dataset_clean.csv"
    if not os.path.exists(master_path):
        raise FileNotFoundError("data/processed/master_dataset_clean.csv is required")

    master = pd.read_csv(master_path)

    dv, message = build_dv_from_faostat_fbs()
    if dv is None or len(dv) < 60:
        print("\n[2] FAOSTAT FBS DV not available enough:", message)
        dv, message = build_dv_from_world_bank_production_proxy()

    if dv is None or len(dv) == 0:
        raise RuntimeError("Could not build any cereal availability dependent variable")

    print("\n[2] Dependent variable")
    print(" ", message)
    print("  Countries with cereal availability DV:", len(dv))

    if "cereal_availability_kg_pc" in master.columns:
        master = master.drop(columns=["cereal_availability_kg_pc"])
    if "cereal_availability_source" in master.columns:
        master = master.drop(columns=["cereal_availability_source"])

    final = master.merge(dv, on="country_code", how="left")
    final = final[final["cereal_availability_kg_pc"].notna()].copy()

    # I remove physically impossible or extreme bad records.
    final = final[final["cereal_availability_kg_pc"] > 0].copy()
    upper_cap = final["cereal_availability_kg_pc"].quantile(0.995)
    final = final[final["cereal_availability_kg_pc"] <= upper_cap].copy()

    final.to_csv("data/processed/master_dataset_cereal_availability_final.csv", index=False)
    print("  Final modelling countries:", len(final))
    print("  Saved: data/processed/master_dataset_cereal_availability_final.csv")

    return final


master = build_final_master_dataset()


# ============================================================
# 3. I define availability-side model blocks only
# ============================================================

DV = "cereal_availability_kg_pc"

MODEL_A_VARS = [
    "cereal_yield_kg_per_ha",
    "fertiliser_kg_per_ha",
    "arable_land_pct",
    "gdp_per_capita_usd",
    "rural_population_pct",
    "agri_employment_pct",
]

MODEL_B_VARS = MODEL_A_VARS + [
    "cereal_loss_pct",
]

MODEL_C_VARS = MODEL_B_VARS + [
    "lpi_overall",
    "rural_electricity_access_pct",
]

MODEL_F_VARS = MODEL_A_VARS + [
    "cereal_loss_pct",
    "lpi_overall",
    "rural_electricity_access_pct",
    "fertiliser_efficiency",
    "food_price_inflation_pct",
]

MODELS = {
    "Model A — Production Baseline": MODEL_A_VARS,
    "Model B — Baseline + PHL": MODEL_B_VARS,
    "Model C — Baseline + PHL + Logistics": MODEL_C_VARS,
    "Model F — NLP Availability Themes": MODEL_F_VARS,
}

LOG_TRANSFORM_COLUMNS = [
    "cereal_availability_kg_pc",
    "cereal_yield_kg_per_ha",
    "fertiliser_kg_per_ha",
    "fertiliser_efficiency",
    "gdp_per_capita_usd",
]


def prepare_model_frame(df, predictors, outcome):
    # I keep only available predictors and remove rows with missing model inputs.
    working = df.copy()
    used_predictors = []
    missing_predictors = []

    for predictor in predictors:
        if predictor in working.columns:
            used_predictors.append(predictor)
        else:
            missing_predictors.append(predictor)

    for column in LOG_TRANSFORM_COLUMNS:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
            working[column] = np.log1p(working[column].clip(lower=0))

    needed = [outcome] + used_predictors
    working = working.dropna(subset=needed).copy()

    X = working[used_predictors].copy()
    y = working[outcome].copy()

    return X, y, used_predictors, missing_predictors, working


def calculate_rmse_cv(model, X, y, kfold):
    # I use negative RMSE scoring because sklearn returns higher-is-better scores.
    scores = cross_val_score(model, X, y, cv=kfold, scoring="neg_root_mean_squared_error")
    rmse_scores = -scores
    return rmse_scores.mean(), rmse_scores.std()


def calculate_r2_cv(model, X, y, kfold):
    scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
    return scores.mean(), scores.std()


def fit_ols_hc3(X, y):
    # I fit OLS with HC3 robust standard errors.
    X_const = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X_const).fit(cov_type="HC3")


def calculate_vif_table(X, model_name):
    # I calculate VIF after median imputation so missingness does not break diagnostics.
    if X.shape[1] < 2:
        return pd.DataFrame()

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    rows = []
    for index in range(X_imp.shape[1]):
        try:
            vif_value = variance_inflation_factor(X_imp.values, index)
        except Exception:
            vif_value = np.nan
        rows.append({
            "Model": model_name,
            "Variable": X_imp.columns[index],
            "VIF": vif_value,
            "VIF_flag": "severe" if pd.notna(vif_value) and vif_value > 10 else "acceptable",
        })
    return pd.DataFrame(rows)


def fit_ridge_cv(X, y, kfold):
    # I use Ridge as the beginner-friendly fix when VIF shows collinearity.
    alphas = np.logspace(-3, 3, 25)
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas, cv=kfold)),
    ])
    ridge.fit(X, y)

    r2_mean, r2_std = calculate_r2_cv(ridge, X, y, kfold)
    rmse_mean, rmse_std = calculate_rmse_cv(ridge, X, y, kfold)
    alpha = ridge.named_steps["ridge"].alpha_

    return ridge, alpha, r2_mean, r2_std, rmse_mean, rmse_std


def model_factory(model_type):
    # I create the requested ML model inside a pipeline so missing values are handled safely.
    if model_type == "rf":
        regressor = RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    elif model_type == "xgb" and XGBOOST_AVAILABLE:
        regressor = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbosity=0,
        )
    else:
        return None

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", regressor),
    ])


def run_model_suite(df):
    # I run all models and save RMSE, R², HC3, VIF, and Ridge outputs.
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    model_rows = []
    vif_frames = []
    ridge_rows = []
    ols_models = {}
    model_frames = {}

    print("\n[3] Final cereal availability models")

    for model_name in MODELS:
        predictors = MODELS[model_name]
        X, y, used, missing, model_frame = prepare_model_frame(df, predictors, DV)

        print("\n", model_name)
        print("  Countries:", len(X))
        print("  Predictors:", used)
        if len(missing) > 0:
            print("  Missing predictors excluded:", missing)

        if len(X) < 30:
            print("  Skipped: fewer than 30 complete countries")
            continue

        ols = fit_ols_hc3(X, y)
        ols_models[model_name] = ols
        model_frames[model_name] = model_frame

        vif_df = calculate_vif_table(X, model_name)
        if len(vif_df) > 0:
            vif_frames.append(vif_df)

        ridge, alpha, ridge_r2, ridge_r2_std, ridge_rmse, ridge_rmse_std = fit_ridge_cv(X, y, kfold)
        ridge_rows.append({
            "Model": model_name,
            "N": len(X),
            "Ridge alpha": alpha,
            "Ridge CV R2 mean": ridge_r2,
            "Ridge CV R2 std": ridge_r2_std,
            "Ridge CV RMSE mean": ridge_rmse,
            "Ridge CV RMSE std": ridge_rmse_std,
        })

        rf = model_factory("rf")
        rf_r2, rf_r2_std = calculate_r2_cv(rf, X, y, kfold)
        rf_rmse, rf_rmse_std = calculate_rmse_cv(rf, X, y, kfold)

        xgb_r2 = np.nan
        xgb_r2_std = np.nan
        xgb_rmse = np.nan
        xgb_rmse_std = np.nan
        xgb_model = model_factory("xgb")
        if xgb_model is not None:
            xgb_r2, xgb_r2_std = calculate_r2_cv(xgb_model, X, y, kfold)
            xgb_rmse, xgb_rmse_std = calculate_rmse_cv(xgb_model, X, y, kfold)

        model_rows.append({
            "Model": model_name,
            "N": len(X),
            "Predictors": len(used),
            "OLS R2": ols.rsquared,
            "OLS Adj R2": ols.rsquared_adj,
            "OLS F p-value HC3": ols.f_pvalue,
            "RF CV R2 mean": rf_r2,
            "RF CV R2 std": rf_r2_std,
            "RF CV RMSE mean": rf_rmse,
            "RF CV RMSE std": rf_rmse_std,
            "XGB CV R2 mean": xgb_r2,
            "XGB CV R2 std": xgb_r2_std,
            "XGB CV RMSE mean": xgb_rmse,
            "XGB CV RMSE std": xgb_rmse_std,
            "Ridge CV R2 mean": ridge_r2,
            "Ridge CV RMSE mean": ridge_rmse,
        })

        safe = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
        with open("outputs/tables/final_ols_hc3_" + safe + ".txt", "w") as handle:
            handle.write(ols.summary().as_text())

        print("  OLS HC3 R²:", round(ols.rsquared, 3), "Adj R²:", round(ols.rsquared_adj, 3))
        print("  RF CV R²:", round(rf_r2, 3), "RMSE:", round(rf_rmse, 3))
        print("  Ridge CV R²:", round(ridge_r2, 3), "RMSE:", round(ridge_rmse, 3))

    model_results = pd.DataFrame(model_rows)
    model_results.to_csv("outputs/tables/final_cereal_availability_model_comparison.csv", index=False)

    if len(vif_frames) > 0:
        vif_all = pd.concat(vif_frames, ignore_index=True)
        vif_all.to_csv("outputs/tables/final_cereal_availability_vif.csv", index=False)
    else:
        vif_all = pd.DataFrame()

    ridge_results = pd.DataFrame(ridge_rows)
    ridge_results.to_csv("outputs/tables/final_cereal_availability_ridge.csv", index=False)

    print("\n  Saved: outputs/tables/final_cereal_availability_model_comparison.csv")
    print("  Saved: outputs/tables/final_cereal_availability_vif.csv")
    print("  Saved: outputs/tables/final_cereal_availability_ridge.csv")

    return model_results, vif_all, ridge_results, ols_models, model_frames


model_results, vif_all, ridge_results, ols_models, model_frames = run_model_suite(master)


# ============================================================
# 4. I run same-sample comparisons and bootstrap ΔR²
# ============================================================

def fit_ols_r2_for_predictors(df, predictors):
    X, y, used, missing, frame = prepare_model_frame(df, predictors, DV)
    if len(X) < 30:
        return np.nan, len(X)
    model = fit_ols_hc3(X, y)
    return model.rsquared, len(X)


def same_sample_comparison(df, base_predictors, extended_predictors, label):
    # I restrict both models to the exact same countries by requiring all extended predictors.
    X_ext, y_ext, used_ext, missing_ext, frame = prepare_model_frame(df, extended_predictors, DV)
    if len(X_ext) < 30:
        return None

    common = frame.copy()
    X_base, y_base, used_base, missing_base, base_frame = prepare_model_frame(common, base_predictors, DV)
    X_extended, y_extended, used_extended, missing_extended, ext_frame = prepare_model_frame(common, extended_predictors, DV)

    base_model = fit_ols_hc3(X_base, y_base)
    ext_model = fit_ols_hc3(X_extended, y_extended)

    return {
        "Comparison": label,
        "N same sample": len(X_extended),
        "Base R2": base_model.rsquared,
        "Extended R2": ext_model.rsquared,
        "Delta R2": ext_model.rsquared - base_model.rsquared,
        "Base predictors": len(used_base),
        "Extended predictors": len(used_extended),
    }


def bootstrap_delta_r2(df, base_predictors, extended_predictors, label, n_boot=1000):
    # I bootstrap the improvement in R² from adding a block of variables.
    X_ext, y_ext, used_ext, missing_ext, frame = prepare_model_frame(df, extended_predictors, DV)
    if len(frame) < 40:
        return None

    deltas = []
    rng = np.random.default_rng(RANDOM_SEED)

    for _ in range(n_boot):
        sample_indices = rng.integers(0, len(frame), len(frame))
        sample = frame.iloc[sample_indices].copy()

        try:
            base_r2, base_n = fit_ols_r2_for_predictors(sample, base_predictors)
            ext_r2, ext_n = fit_ols_r2_for_predictors(sample, extended_predictors)
            if pd.notna(base_r2) and pd.notna(ext_r2):
                deltas.append(ext_r2 - base_r2)
        except Exception:
            continue

    if len(deltas) == 0:
        return None

    delta_array = np.array(deltas)
    return {
        "Comparison": label,
        "Bootstrap iterations used": len(delta_array),
        "Delta R2 mean": delta_array.mean(),
        "Delta R2 median": np.median(delta_array),
        "Delta R2 2.5%": np.percentile(delta_array, 2.5),
        "Delta R2 97.5%": np.percentile(delta_array, 97.5),
    }


print("\n[4] Same-sample and bootstrap ΔR² checks")

same_rows = []
same_specs = [
    (MODEL_A_VARS, MODEL_B_VARS, "Model B vs Model A — PHL gain"),
    (MODEL_B_VARS, MODEL_C_VARS, "Model C vs Model B — logistics gain"),
    (MODEL_A_VARS, MODEL_F_VARS, "Model F vs Model A — NLP availability gain"),
]

for base_vars, ext_vars, label in same_specs:
    row = same_sample_comparison(master, base_vars, ext_vars, label)
    if row is not None:
        same_rows.append(row)

same_df = pd.DataFrame(same_rows)
same_df.to_csv("outputs/tables/final_same_sample_comparison.csv", index=False)
print("  Saved: outputs/tables/final_same_sample_comparison.csv")

boot_rows = []
for base_vars, ext_vars, label in same_specs:
    row = bootstrap_delta_r2(master, base_vars, ext_vars, label, n_boot=1000)
    if row is not None:
        boot_rows.append(row)

boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv("outputs/tables/final_block_delta_r2_bootstrap.csv", index=False)
print("  Saved: outputs/tables/final_block_delta_r2_bootstrap.csv")

print("\nPHASE G COMPLETE")
print("=" * 70)
print("Use the Phase G outputs as the final dissertation evidence for cereal food availability.")
print("Keep older undernourishment outputs only as superseded exploratory analysis.")
