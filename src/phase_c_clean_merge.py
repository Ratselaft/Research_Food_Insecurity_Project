# ============================================================
# PHASE C — Clean and merge all datasets into one master file
# ============================================================
#
# What this file does:
#   We downloaded 5 datasets in Phase B. Now we need to clean
#   each one and join them all together into a single table —
#   one row per country — so we can use it in Phase D modelling.
#
# The master dataset will have these columns:
#   - country_code, country_name
#   - cereal_yield_kg_per_ha        (from World Bank WDI)
#   - fertiliser_kg_per_ha          (from World Bank WDI)
#   - arable_land_pct               (from World Bank WDI)
#   - gdp_per_capita_usd            (from World Bank WDI)
#   - population_total              (from World Bank WDI)
#   - internet_users_pct            (from World Bank WDI)
#   - stunting_pct_children         (from World Bank WDI)
#   - irrigated_land_pct            (from World Bank WDI)
#   - account_ownership_pct         (from Findex 2021)
#   - bank_branches_per_100k        (from IMF FAS)
#   - atm_per_100k                  (from IMF FAS)
#   - cereal_loss_pct               (from FAO FLW — cleaned)
#   - fertiliser_efficiency         (engineered: yield / fertiliser use)
#
# We will keep only countries that have ALL the core variables.
# We aim for at least 80 countries for the regression to work.
# ============================================================

import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

print("Starting Phase C — cleaning and merging datasets...")
print("=" * 55)


# ============================================================
# STEP 1: Clean the FAO Food Loss & Waste data
# ============================================================
# The raw FLW file has 21,000+ rows covering many commodities
# and many food supply stages. We need to:
#   a) Keep only cereal crops (wheat, maize, rice, sorghum etc.)
#   b) Keep only rows where loss_percentage is a real number
#   c) Average across commodities and years per country
#   d) Produce one number per country: average % cereal loss

print("\n[1/5] Cleaning FAO Food Loss & Waste data...")

# The file has a double extension from the download — we read it directly
FLW_FILE = "data/raw/fao_flw_losses.csv.csv"
if not os.path.exists(FLW_FILE):
    FLW_FILE = "data/raw/fao_flw_losses.csv"

flw_raw = pd.read_csv(FLW_FILE)
print(f"  Raw FLW file: {len(flw_raw)} rows, {flw_raw['country'].nunique()} countries")

# Keep only cereal commodities
# These are the crops we care about for our research
CEREAL_WORDS = ["wheat", "maize", "corn", "rice", "sorghum", "barley",
                "millet", "oat", "rye", "cereal", "teff", "fonio"]

cereal_mask = flw_raw["commodity"].str.lower().str.contains(
    "|".join(CEREAL_WORDS), na=False
)
flw_cereal = flw_raw[cereal_mask].copy()
print(f"  After keeping only cereals: {len(flw_cereal)} rows")

# Keep only rows where loss_percentage is a real number (not blank)
flw_cereal = flw_cereal.dropna(subset=["loss_percentage"])
flw_cereal["loss_percentage"] = pd.to_numeric(flw_cereal["loss_percentage"], errors="coerce")
flw_cereal = flw_cereal.dropna(subset=["loss_percentage"])

# Remove impossible values (loss can't be below 0 or above 100)
flw_cereal = flw_cereal[
    (flw_cereal["loss_percentage"] >= 0) &
    (flw_cereal["loss_percentage"] <= 100)
]
print(f"  After removing missing/impossible values: {len(flw_cereal)} rows")

# We want a 5-year window centred on 2021 (2018–2022) to maximise coverage
# This is the same year as the other datasets
flw_recent = flw_cereal[
    (flw_cereal["year"] >= 2018) & (flw_cereal["year"] <= 2022)
].copy()

if len(flw_recent) < 100:
    # If 2018–2022 is too sparse, widen to 2015–2023
    print("  Warning: few rows in 2018–2022 — widening window to 2015–2023")
    flw_recent = flw_cereal[
        (flw_cereal["year"] >= 2015) & (flw_cereal["year"] <= 2023)
    ].copy()

print(f"  Rows in selected year window: {len(flw_recent)}")

# Average the loss percentage across all commodities and stages per country
flw_country = (
    flw_recent
    .groupby("country")["loss_percentage"]
    .mean()
    .reset_index()
    .rename(columns={"loss_percentage": "cereal_loss_pct"})
)
flw_country["cereal_loss_pct"] = flw_country["cereal_loss_pct"].round(2)
print(f"  Countries with cereal loss data: {len(flw_country)}")

# We need a country CODE (ISO3) to merge with the other tables.
# The FAO FLW file uses country names — we will match them to
# the World Bank table which has both names and ISO3 codes.


# ============================================================
# STEP 2: Load and inspect all other datasets
# ============================================================

print("\n[2/5] Loading World Bank WDI, Findex, and IMF data...")

# World Bank API returns regional/income-group aggregates alongside real countries.
# These codes are NOT ISO3 country codes — we remove them to keep only real countries.
REGIONAL_CODES = {
    'AFE','AFW','ARB','CEB','CSS','EAP','EAR','EAS','ECA','ECS',
    'EMU','EUU','FCS','HIC','HPC','IBD','IBT','IDA','IDB','IDX',
    'LAC','LCN','LDC','LIC','LMC','LMY','LTE','MEA','MIC','MNA',
    'NAC','OED','OSS','PRE','PSS','PST','SAS','SSA','SSF','SST',
    'TEA','TEC','TLA','TMN','TSA','TSS','UMC','WLD','XZN'
}

wdi_raw = pd.read_csv("data/raw/worldbank_wdi_2021.csv")
wdi = (
    wdi_raw[
        wdi_raw["country_code"].notna() &
        ~wdi_raw["country_code"].isin(REGIONAL_CODES)
    ]
    .drop_duplicates(subset="country_code", keep="first")
    .reset_index(drop=True)
)

findex_raw = pd.read_csv("data/raw/findex_2021.csv")
findex = (
    findex_raw.dropna(subset=["country_code"])
    .drop_duplicates(subset="country_code", keep="first")
    .reset_index(drop=True)
)

imf_raw = pd.read_csv("data/raw/imf_financial_access.csv")
imf = (
    imf_raw.dropna(subset=["country_code"])
    .drop_duplicates(subset="country_code", keep="first")
    .reset_index(drop=True)
)

print(f"  WDI:    {len(wdi)} real countries")
print(f"  Findex: {len(findex)} countries")
print(f"  IMF:    {len(imf)} countries")


# ============================================================
# STEP 3: Match FAO country names to ISO3 codes
# ============================================================
# FAO uses country names like "Ethiopia", "Côte d'Ivoire" etc.
# We need to match these to the ISO3 codes in the WDI table
# so we can merge everything on country_code.

print("\n[3/5] Matching FAO country names to ISO3 codes...")

# Build a name→code lookup from the WDI table
name_to_code = dict(zip(
    wdi["country_name"].str.lower().str.strip(),
    wdi["country_code"]
))

# Some FAO names are spelled differently from World Bank names.
# We list the most common differences here so they match.
FAO_NAME_FIXES = {
    "bolivia (plurinational state of)":        "BOL",
    "china, mainland":                         "CHN",
    "china, hong kong sar":                    "HKG",
    "china, macao sar":                        "MAC",
    "congo":                                   "COG",
    "democratic republic of the congo":        "COD",
    "egypt":                                   "EGY",
    "iran (islamic republic of)":              "IRN",
    "côte d'ivoire":                           "CIV",
    "cote d'ivoire":                           "CIV",
    "korea, republic of":                      "KOR",
    "korea, dem. people's rep.":               "PRK",
    "lao people's democratic republic":        "LAO",
    "libyan arab jamahiriya":                  "LBY",
    "libya":                                   "LBY",
    "moldova, republic of":                    "MDA",
    "russian federation":                      "RUS",
    "syrian arab republic":                    "SYR",
    "tanzania, united republic of":            "TZA",
    "united republic of tanzania":             "TZA",
    "united states of america":                "USA",
    "venezuela (bolivarian republic of)":      "VEN",
    "viet nam":                                "VNM",
    "vietnam":                                 "VNM",
    "türkiye":                                 "TUR",
    "turkey":                                  "TUR",
    "eswatini":                                "SWZ",
    "swaziland":                               "SWZ",
    "north macedonia":                         "MKD",
    "gambia":                                  "GMB",
    "slovakia":                                "SVK",
    "slovak republic":                         "SVK",
    "united kingdom of great britain and northern ireland": "GBR",
    "united kingdom":                          "GBR",
    "somalia":                                 "SOM",
}

NOT_COUNTRIES = {"world", "asia", "africa", "europe", "americas", "oceania"}

def get_iso3(country_name):
    """Try to find the ISO3 code for a country name. Returns None for aggregates."""
    name_lower = str(country_name).lower().strip()
    if name_lower in NOT_COUNTRIES:
        return None
    # Try exact match first
    if name_lower in name_to_code:
        return name_to_code[name_lower]
    # Try the manual fixes
    if name_lower in FAO_NAME_FIXES:
        return FAO_NAME_FIXES[name_lower]
    return None

flw_country["country_code"] = flw_country["country"].apply(get_iso3)

# Check how many matched
matched   = flw_country["country_code"].notna().sum()
unmatched = flw_country[flw_country["country_code"].isna()]["country"].tolist()
print(f"  Matched {matched} out of {len(flw_country)} FAO countries to ISO3 codes")
if unmatched:
    print(f"  Could not match: {unmatched[:10]}")

# Keep only rows where we found a code
flw_country = flw_country.dropna(subset=["country_code"])


# ============================================================
# STEP 4: Merge all datasets on country_code
# ============================================================

print("\n[4/5] Merging all datasets into one master table...")

# Start with WDI as the base (it has the most countries)
master = wdi.copy()

# Add Findex financial access data
findex_slim = findex[["country_code", "account_ownership_pct"]].copy()
master = master.merge(findex_slim, on="country_code", how="left")

# Add IMF banking infrastructure data
imf_slim = imf[["country_code", "bank_branches_per_100k", "atm_per_100k"]].copy()
master = master.merge(imf_slim, on="country_code", how="left")

# Add FAO FLW cereal loss data
flw_slim = flw_country[["country_code", "cereal_loss_pct"]].copy()
master = master.merge(flw_slim, on="country_code", how="left")

print(f"  Master table: {len(master)} countries, {master.shape[1]} columns")
print(f"  Columns: {list(master.columns)}")


# ============================================================
# STEP 5: Engineer new features
# ============================================================

print("\n[5/5] Engineering new features...")

# Fertiliser efficiency = how many kg of cereal we get per kg of fertiliser
# This tells us how well each country converts fertiliser into food
# We only compute this where both values are available and fertiliser > 0
mask = (master["fertiliser_kg_per_ha"] > 0) & master["cereal_yield_kg_per_ha"].notna()
master.loc[mask, "fertiliser_efficiency"] = (
    master.loc[mask, "cereal_yield_kg_per_ha"] /
    master.loc[mask, "fertiliser_kg_per_ha"]
).round(2)

print("  fertiliser_efficiency = cereal_yield / fertiliser_kg_per_ha")


# ============================================================
# STEP 6: Check coverage and decide which countries to keep
# ============================================================

# List which columns are the CORE predictors for Model A
CORE_COLS = [
    "cereal_yield_kg_per_ha",
    "gdp_per_capita_usd",
    "arable_land_pct",
]

# Finance block columns (Model C)
FINANCE_COLS = ["account_ownership_pct"]

# PHL block columns (Model B)
PHL_COLS = ["cereal_loss_pct"]

print("\n--- Coverage check (how many countries have each variable) ---")
for col in master.columns:
    if col not in ["country_code", "country_name"]:
        n = master[col].notna().sum()
        print(f"  {col:<35} {n} / {len(master)} countries")

# How complete are countries on core variables?
master["core_complete"] = master[CORE_COLS].notna().all(axis=1)
complete_count = master["core_complete"].sum()
print(f"\n  Countries with ALL core variables: {complete_count}")

# We want at least 80 countries for reliable regression results
if complete_count >= 80:
    print(f"  Good — {complete_count} countries meets the ≥80 threshold")
else:
    print(f"  Warning — only {complete_count} countries. Regression may be less reliable.")


# ============================================================
# STEP 7: Save the master dataset
# ============================================================

# Save the FULL master (all countries, including those with missing values)
# Phase D will decide which to drop depending on which model is being run
master = master.drop(columns=["core_complete"])
out_full = "data/processed/master_dataset.csv"
master.to_csv(out_full, index=False)
print(f"\nFull master dataset saved → {out_full}  ({len(master)} countries)")

# Also save a clean subset: only countries with all core variables
master_clean = master[master[CORE_COLS].notna().all(axis=1)].copy()
out_clean = "data/processed/master_dataset_clean.csv"
master_clean.to_csv(out_clean, index=False)
print(f"Clean subset saved → {out_clean}  ({len(master_clean)} countries)")

# Show the first few rows so we can check it looks right
print("\n--- Preview of clean master dataset ---")
preview_cols = ["country_name", "cereal_yield_kg_per_ha", "gdp_per_capita_usd",
                "account_ownership_pct", "cereal_loss_pct", "fertiliser_efficiency"]
print(master_clean[preview_cols].head(10).to_string(index=False))

print("\n" + "=" * 55)
print("PHASE C COMPLETE")
print("Next step: Phase D — fit Models A, B, C")
print("=" * 55)
