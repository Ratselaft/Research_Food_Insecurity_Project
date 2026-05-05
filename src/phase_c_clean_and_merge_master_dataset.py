# ============================================================
# I'm cleaning and merging ALL datasets into one master file
# ============================================================
#
# What I'm doing here:
#   Phase B downloaded 9 different dataset files. Now I need to
#   clean each one and join them all into a single table —
#   one row per country — ready for modelling in Phase D.
#
# What has changed in this improved version:
#   - I now merge ALL the new WDI indicators (rural population,
#     agricultural employment, female agriculture, food price
#     inflation, livestock production, credit rights)
#   - I now merge ALL disaggregated Findex indicators (rural,
#     female, poorest 40%, agricultural payments, borrowing)
#   - I now merge ALL 6 WGI governance indicators
#     (or skip gracefully if the manual file is not ready)
#   - I now try to extract supply chain stage from the FAO FLW
#     data (so I know WHERE in the chain losses happen)
#   - I now compute a VALUE CHAIN FINANCIAL ACCESS score —
#     a single composite number that captures whether finance
#     is reaching people in the food production chain
#
# The final master dataset has one row per country, with every
# indicator as a separate column. Missing values are allowed —
# Phase D handles them by only using countries with complete
# data for each model specification.
# ============================================================

# I need os to create folders and check whether files exist
import os

# I need numpy for maths (log, divide, etc.)
import numpy as np
# I need pandas to work with tables of data
import pandas as pd

# I make sure my output folder exists
os.makedirs("data/processed", exist_ok=True)

# I let the user know I'm starting
print("Starting Phase C — cleaning and merging datasets...")
print("=" * 60)


# ============================================================
# I define the set of World Bank regional/aggregate codes
# ============================================================
# These are NOT real countries — they are World Bank groupings
# like "High income", "Sub-Saharan Africa", "World".
# I must remove them before any analysis.

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


# ============================================================
# Step 1: I'm cleaning the FAO Food Loss and Waste data
# ============================================================
# The raw FLW file has thousands of rows covering many commodities
# and (if available) multiple supply chain stages.
#
# I need to:
#   a) Keep only cereal crops
#   b) Try to capture supply chain stage information
#   c) Calculate average loss per country (and per stage if possible)
#   d) Produce one number per country: average % cereal loss

print("\n[1/7] Cleaning FAO Food Loss and Waste data...")

# I set the path — I try both possible filename versions
FLW_FILE = "data/raw/fao_flw_losses.csv.csv"
if not os.path.exists(FLW_FILE):
    FLW_FILE = "data/raw/fao_flw_losses.csv"

# I read the file
flw_raw = pd.read_csv(FLW_FILE)

# I check if the file has real data or is just a placeholder
if len(flw_raw) == 0:
    print("  FLW file is a placeholder — no real data yet")
    print("  Download the file manually from:")
    print("  https://www.fao.org/platform-food-loss-waste/flw-data/en/")
    # I create empty tables that Phase D can handle gracefully
    flw_country = pd.DataFrame(columns=["country_code", "cereal_loss_pct"])
    flw_by_stage = pd.DataFrame(columns=["country_code", "farm_loss_pct",
                                         "storage_loss_pct", "market_loss_pct"])
else:
    print("  Raw FLW rows:", len(flw_raw))
    print("  Countries in file:", flw_raw["country"].nunique() if "country" in flw_raw.columns else "unknown")

    # I look for a supply chain stage column in the file
    stage_col = None
    for col in flw_raw.columns:
        if "stage" in col.lower() or "chain" in col.lower():
            stage_col = col
            print("  Found supply chain stage column:", stage_col)
            break

    # I keep only cereal crops
    CEREAL_WORDS = ["wheat", "maize", "corn", "rice", "sorghum", "barley",
                    "millet", "oat", "rye", "cereal", "teff", "fonio"]
    cereal_pattern = "|".join(CEREAL_WORDS)

    # I find the commodity column
    commodity_col = None
    for col in flw_raw.columns:
        if "commodity" in col.lower() or "item" in col.lower() or "crop" in col.lower():
            commodity_col = col
            break

    if commodity_col is not None:
        cereal_mask = flw_raw[commodity_col].str.lower().str.contains(cereal_pattern, na=False)
        flw_cereal = flw_raw[cereal_mask].copy()
    else:
        # If I cannot find a commodity column, I keep all rows
        flw_cereal = flw_raw.copy()

    # I find the loss percentage column
    loss_col = None
    for col in flw_raw.columns:
        if "loss" in col.lower() and "percent" in col.lower():
            loss_col = col
            break
    if loss_col is None:
        for col in flw_raw.columns:
            if "loss" in col.lower():
                loss_col = col
                break

    if loss_col is None:
        print("  Cannot find loss percentage column — check FLW file columns:")
        print("  ", list(flw_raw.columns))
        flw_country = pd.DataFrame(columns=["country_code", "cereal_loss_pct"])
        flw_by_stage = pd.DataFrame(columns=["country_code", "farm_loss_pct",
                                             "storage_loss_pct", "market_loss_pct"])
    else:
        # I convert the loss column to numbers
        flw_cereal[loss_col] = pd.to_numeric(flw_cereal[loss_col], errors="coerce")

        # I remove impossible values
        flw_cereal = flw_cereal.dropna(subset=[loss_col])
        flw_cereal = flw_cereal[flw_cereal[loss_col] >= 0]
        flw_cereal = flw_cereal[flw_cereal[loss_col] <= 100]

        # I find the year column
        year_col = None
        for col in flw_raw.columns:
            if "year" in col.lower():
                year_col = col
                break

        if year_col is not None:
            flw_cereal[year_col] = pd.to_numeric(flw_cereal[year_col], errors="coerce")
            flw_recent = flw_cereal[flw_cereal[year_col] >= 2015].copy()
            flw_recent = flw_recent[flw_recent[year_col] <= 2023].copy()
        else:
            flw_recent = flw_cereal.copy()

        print("  Rows after filtering to cereals and recent years:", len(flw_recent))

        # I find the country column
        country_col = None
        for col in flw_raw.columns:
            if "country" in col.lower() or "area" in col.lower():
                country_col = col
                break

        if country_col is None:
            country_col = flw_raw.columns[0]

        # I compute average cereal loss per country
        flw_country = flw_recent.groupby(country_col)[loss_col].mean().reset_index()
        flw_country.columns = ["country", "cereal_loss_pct"]
        flw_country["cereal_loss_pct"] = flw_country["cereal_loss_pct"].round(2)

        # I try to compute loss by supply chain stage
        flw_by_stage = pd.DataFrame(columns=["country_code", "farm_loss_pct",
                                             "storage_loss_pct", "market_loss_pct"])

        if stage_col is not None:
            # I check what stage labels exist
            stage_values = flw_recent[stage_col].dropna().unique()
            print("  Supply chain stages found:", list(stage_values[:8]))

            # I try to identify farm-stage, storage-stage, and market-stage rows
            stage_tables = []

            for stage_value in stage_values:
                stage_label = str(stage_value).lower()
                stage_df = flw_recent[flw_recent[stage_col] == stage_value]

                # I work out which column name to give this stage
                if any(word in stage_label for word in ["farm", "harvest", "field", "production"]):
                    col_label = "farm_loss_pct"
                elif any(word in stage_label for word in ["storage", "warehouse", "silo", "store"]):
                    col_label = "storage_loss_pct"
                elif any(word in stage_label for word in ["market", "retail", "trade", "wholesale"]):
                    col_label = "market_loss_pct"
                elif any(word in stage_label for word in ["transport", "logistics", "distribution"]):
                    col_label = "transport_loss_pct"
                elif any(word in stage_label for word in ["process", "handling"]):
                    col_label = "processing_loss_pct"
                else:
                    continue

                # I compute average loss at this stage per country
                stage_country = stage_df.groupby(country_col)[loss_col].mean().reset_index()
                stage_country.columns = ["country", col_label]
                stage_country[col_label] = stage_country[col_label].round(2)
                stage_tables.append(stage_country)

            if len(stage_tables) > 0:
                # I merge all stage tables
                flw_by_stage = stage_tables[0]
                for t in stage_tables[1:]:
                    flw_by_stage = flw_by_stage.merge(t, on="country", how="outer")
                print("  Stage-level loss columns created:", list(flw_by_stage.columns))

        print("  Countries with cereal loss data:", len(flw_country))


# ============================================================
# Step 2: I'm matching FAO country names to ISO3 codes
# ============================================================
# FAO uses country names. I need ISO3 codes to merge with WDI.

print("\n[2/7] Matching FAO names to ISO3 country codes...")

# I will build the lookup table AFTER loading WDI (Step 3 below)
# so I define the fix dictionary here and use it after WDI is loaded

FAO_NAME_FIXES = {}
FAO_NAME_FIXES["bolivia (plurinational state of)"]        = "BOL"
FAO_NAME_FIXES["china, mainland"]                         = "CHN"
FAO_NAME_FIXES["china, hong kong sar"]                    = "HKG"
FAO_NAME_FIXES["china, macao sar"]                        = "MAC"
FAO_NAME_FIXES["congo"]                                   = "COG"
FAO_NAME_FIXES["democratic republic of the congo"]        = "COD"
FAO_NAME_FIXES["egypt"]                                   = "EGY"
FAO_NAME_FIXES["iran (islamic republic of)"]              = "IRN"
FAO_NAME_FIXES["côte d'ivoire"]                           = "CIV"
FAO_NAME_FIXES["cote d'ivoire"]                           = "CIV"
FAO_NAME_FIXES["korea, republic of"]                      = "KOR"
FAO_NAME_FIXES["korea, dem. people's rep."]               = "PRK"
FAO_NAME_FIXES["lao people's democratic republic"]        = "LAO"
FAO_NAME_FIXES["libyan arab jamahiriya"]                  = "LBY"
FAO_NAME_FIXES["libya"]                                   = "LBY"
FAO_NAME_FIXES["moldova, republic of"]                    = "MDA"
FAO_NAME_FIXES["russian federation"]                      = "RUS"
FAO_NAME_FIXES["syrian arab republic"]                    = "SYR"
FAO_NAME_FIXES["tanzania, united republic of"]            = "TZA"
FAO_NAME_FIXES["united republic of tanzania"]             = "TZA"
FAO_NAME_FIXES["united states of america"]                = "USA"
FAO_NAME_FIXES["venezuela (bolivarian republic of)"]      = "VEN"
FAO_NAME_FIXES["viet nam"]                                = "VNM"
FAO_NAME_FIXES["vietnam"]                                 = "VNM"
FAO_NAME_FIXES["türkiye"]                                 = "TUR"
FAO_NAME_FIXES["turkey"]                                  = "TUR"
FAO_NAME_FIXES["eswatini"]                                = "SWZ"
FAO_NAME_FIXES["swaziland"]                               = "SWZ"
FAO_NAME_FIXES["north macedonia"]                         = "MKD"
FAO_NAME_FIXES["gambia"]                                  = "GMB"
FAO_NAME_FIXES["slovakia"]                                = "SVK"
FAO_NAME_FIXES["slovak republic"]                         = "SVK"
FAO_NAME_FIXES["united kingdom"]                          = "GBR"
FAO_NAME_FIXES["united kingdom of great britain and northern ireland"] = "GBR"
FAO_NAME_FIXES["somalia"]                                 = "SOM"

# These are aggregate region names that should be excluded
NOT_COUNTRIES = set()
NOT_COUNTRIES.add("world")
NOT_COUNTRIES.add("asia")
NOT_COUNTRIES.add("africa")
NOT_COUNTRIES.add("europe")
NOT_COUNTRIES.add("americas")
NOT_COUNTRIES.add("oceania")


# ============================================================
# Step 3: I'm loading and cleaning the World Bank WDI data
# ============================================================

print("\n[3/7] Loading World Bank WDI data...")

wdi_raw = pd.read_csv("data/raw/worldbank_wdi_2021.csv")

# I remove rows with no country code
wdi_no_nulls = wdi_raw[wdi_raw["country_code"].notna()].copy()

# I remove rows where the code is a regional aggregate, not a real country
wdi_real = wdi_no_nulls[~wdi_no_nulls["country_code"].isin(REGIONAL_CODES)].copy()

# I remove any duplicate country codes
wdi = wdi_real.drop_duplicates(subset="country_code", keep="first").reset_index(drop=True)

print("  WDI countries (after removing aggregates):", len(wdi))
print("  WDI columns:", list(wdi.columns))

# Now that I have WDI loaded, I can build the name-to-code lookup for FAO matching
name_to_code = {}
for i in range(len(wdi)):
    name_lower = str(wdi["country_name"].iloc[i]).lower().strip()
    code = wdi["country_code"].iloc[i]
    name_to_code[name_lower] = code


def get_iso3(country_name):
    # I convert the name to lowercase for matching
    name_lower = str(country_name).lower().strip()
    # I skip aggregate region names
    if name_lower in NOT_COUNTRIES:
        return None
    # I try an exact match in the WDI name lookup
    if name_lower in name_to_code:
        return name_to_code[name_lower]
    # I try my manual corrections dictionary
    if name_lower in FAO_NAME_FIXES:
        return FAO_NAME_FIXES[name_lower]
    # If nothing matched, I return None
    return None


# I apply the lookup to my FLW country table
if "country" in flw_country.columns:
    flw_country["country_code"] = flw_country["country"].apply(get_iso3)
    matched = flw_country["country_code"].notna().sum()
    print("  FAO FLW matched:", matched, "out of", len(flw_country), "countries")
    flw_country = flw_country.dropna(subset=["country_code"])
elif "country_code" in flw_country.columns:
    print("  FLW already has country_code column")

# I apply the same lookup to the stage-level table if it exists
if "country" in flw_by_stage.columns:
    flw_by_stage["country_code"] = flw_by_stage["country"].apply(get_iso3)
    flw_by_stage = flw_by_stage.dropna(subset=["country_code"])


# ============================================================
# Step 4: I'm loading Findex, IMF, and WGI data
# ============================================================

print("\n[4/7] Loading Findex, IMF, and WGI data...")

# I load the Findex disaggregated financial access data
findex_raw = pd.read_csv("data/raw/findex_2021.csv")
findex = findex_raw.dropna(subset=["country_code"]).drop_duplicates(
    subset="country_code", keep="first"
).reset_index(drop=True)
print("  Findex countries:", len(findex))
print("  Findex columns:", list(findex.columns))

# I load the IMF banking infrastructure data
imf_raw = pd.read_csv("data/raw/imf_financial_access.csv")
imf = imf_raw.dropna(subset=["country_code"]).drop_duplicates(
    subset="country_code", keep="first"
).reset_index(drop=True)
print("  IMF/Banking countries:", len(imf))
print("  IMF columns:", list(imf.columns))

# I try to load the WGI governance data
# This might be a placeholder if the manual download has not been done yet
wgi = None
WGI_FILE = "data/raw/wgi_governance_2021.csv"
if os.path.exists(WGI_FILE):
    wgi_raw = pd.read_csv(WGI_FILE)
    if len(wgi_raw) > 0:
        # I check it has a country_code column
        if "country_code" in wgi_raw.columns:
            wgi = wgi_raw.dropna(subset=["country_code"]).drop_duplicates(
                subset="country_code", keep="first"
            ).reset_index(drop=True)
            print("  WGI countries:", len(wgi))
            print("  WGI columns:", list(wgi.columns))
        else:
            print("  WGI file exists but has no country_code column — check the column names")
            print("  WGI file columns:", list(wgi_raw.columns))
    else:
        print("  WGI file is empty (placeholder) — download manually from:")
        print("  https://info.worldbank.org/governance/wgi/")
else:
    print("  WGI file not found — run Phase B first")


# ============================================================
# Step 5: I'm merging everything into one master table
# ============================================================

print("\n[5/7] Merging all datasets into one master table...")

# I start with WDI as my base — it has the most countries
master = wdi.copy()

# I identify ALL columns from Findex to bring in
# (I exclude country_name and year if present — I already have those from WDI)
findex_merge_cols = ["country_code"]
for col in findex.columns:
    if col not in ["country_code", "country_name", "year"]:
        findex_merge_cols.append(col)
findex_slim = findex[findex_merge_cols].copy()

# I merge the Findex disaggregated data
master = master.merge(findex_slim, on="country_code", how="left")
print("  After merging Findex:", master.shape)

# I identify all IMF columns to bring in
imf_merge_cols = ["country_code"]
for col in imf.columns:
    if col not in ["country_code", "country_name", "year"]:
        imf_merge_cols.append(col)
imf_slim = imf[imf_merge_cols].copy()

# I merge the IMF banking data
master = master.merge(imf_slim, on="country_code", how="left")
print("  After merging IMF:", master.shape)

# I merge the FAO Food Loss and Waste data (country-level average)
if "country_code" in flw_country.columns:
    flw_slim = flw_country[["country_code", "cereal_loss_pct"]].copy()
    master = master.merge(flw_slim, on="country_code", how="left")
    print("  After merging FLW:", master.shape)

# I merge the supply chain stage loss data if it exists and has content
if "country_code" in flw_by_stage.columns and len(flw_by_stage) > 0:
    stage_cols = ["country_code"]
    for col in flw_by_stage.columns:
        if "loss_pct" in col and col != "cereal_loss_pct":
            stage_cols.append(col)
    if len(stage_cols) > 1:
        flw_stage_slim = flw_by_stage[stage_cols].copy()
        master = master.merge(flw_stage_slim, on="country_code", how="left")
        print("  After merging supply chain stage losses:", master.shape)

# I merge the WGI governance data if it was available
if wgi is not None:
    wgi_merge_cols = ["country_code"]
    for col in wgi.columns:
        if col not in ["country_code", "country_name", "year"]:
            wgi_merge_cols.append(col)
    wgi_slim = wgi[wgi_merge_cols].copy()
    master = master.merge(wgi_slim, on="country_code", how="left")
    print("  After merging WGI governance:", master.shape)
else:
    print("  WGI not merged — no data available yet")

print("  Final master table:", len(master), "countries,", master.shape[1], "columns")


# ============================================================
# Step 6: I'm engineering new features
# ============================================================
# I now compute two engineered variables:
#
#   1. fertiliser_efficiency
#      = cereal yield / fertiliser use
#      Tells me how efficiently each kg of fertiliser is being
#      converted into food. High efficiency = better farming practice.
#
#   2. value_chain_finance_score
#      This is a NEW composite indicator I am creating.
#      It combines the financial access indicators that are most
#      specifically relevant to the food VALUE CHAIN:
#        - rural account ownership (finance where farmers live)
#        - agricultural payments received digitally (farmers in digital economy)
#        - borrowing from banks (actually using credit, not just having an account)
#        - bank branches per 100k (physical access to finance)
#      I normalise each to 0–100 scale and average them.
#      This composite is my key value chain financial access variable
#      and is the central predictor in Model D.

print("\n[6/7] Engineering new features...")

# ── Feature 1: Fertiliser efficiency ─────────────────────────────────────────
for i in range(len(master)):
    yield_val = master["cereal_yield_kg_per_ha"].iloc[i] if "cereal_yield_kg_per_ha" in master.columns else None
    fert_val  = master["fertiliser_kg_per_ha"].iloc[i] if "fertiliser_kg_per_ha" in master.columns else None

    if pd.notna(yield_val) and pd.notna(fert_val) and fert_val > 0:
        efficiency = round(yield_val / fert_val, 2)
        master.loc[master.index[i], "fertiliser_efficiency"] = efficiency

print("  fertiliser_efficiency created (cereal_yield / fertiliser_kg_per_ha)")

# ── Feature 2: Value chain financial access score ─────────────────────────────
# I combine multiple financial access indicators into one composite score.
# Each sub-indicator already is expressed as a percentage (0–100),
# so I can simply average the ones that are available for each country.
#
# The variables I combine (if they exist in the master table):
#   - account_ownership_rural_pct (most important: finance where farmers are)
#   - agri_payments_digital_pct   (farmers in the digital payment system)
#   - borrowed_from_bank_pct      (credit actually flowing to people)
#   - bank_branches_per_100k      (scaled: physical access)

# I list the sub-components of my value chain score
VCHAIN_COMPONENTS = []
if "account_ownership_rural_pct" in master.columns:
    VCHAIN_COMPONENTS.append("account_ownership_rural_pct")
if "agri_payments_digital_pct" in master.columns:
    VCHAIN_COMPONENTS.append("agri_payments_digital_pct")
if "borrowed_from_bank_pct" in master.columns:
    VCHAIN_COMPONENTS.append("borrowed_from_bank_pct")

# Bank branches need special scaling — they are not a percentage
# I scale bank branches to a 0–100 range using the observed maximum
if "bank_branches_per_100k" in master.columns:
    max_branches = master["bank_branches_per_100k"].max()
    if pd.notna(max_branches) and max_branches > 0:
        master["bank_branches_scaled"] = (master["bank_branches_per_100k"] / max_branches) * 100
        VCHAIN_COMPONENTS.append("bank_branches_scaled")

if len(VCHAIN_COMPONENTS) > 0:
    print("  Computing value_chain_finance_score from:", VCHAIN_COMPONENTS)

    # I compute the row-wise average across all available components
    # .mean(axis=1) averages across columns for each row, skipping blanks
    component_data = master[VCHAIN_COMPONENTS].copy()
    master["value_chain_finance_score"] = component_data.mean(axis=1)
    master["value_chain_finance_score"] = master["value_chain_finance_score"].round(2)

    non_null = master["value_chain_finance_score"].notna().sum()
    print("  value_chain_finance_score computed for", non_null, "countries")
else:
    print("  No disaggregated Findex data available for value chain score")
    print("  Check that Phase B downloaded findex_2021.csv with rural columns")


# ============================================================
# Step 7: I'm checking coverage and saving the master dataset
# ============================================================

print("\n[7/7] Coverage check and saving...")

# I print how many countries have each variable
print("\n--- Coverage (countries with data for each variable) ---")
for col in master.columns:
    if col not in ["country_code", "country_name"]:
        n = master[col].notna().sum()
        total = len(master)
        pct = round(100 * n / total)
        print(f"  {col:<45} {n:>3}/{total} ({pct}%)")

# I define the core variables needed for Model A
CORE_COLS = ["cereal_yield_kg_per_ha", "gdp_per_capita_usd", "arable_land_pct"]

# I count how many countries have ALL core variables
countries_with_core = 0
for i in range(len(master)):
    all_present = True
    for col in CORE_COLS:
        if col not in master.columns or pd.isna(master[col].iloc[i]):
            all_present = False
            break
    if all_present:
        countries_with_core = countries_with_core + 1

print(f"\n  Countries with ALL core variables: {countries_with_core}")

if countries_with_core >= 80:
    print(f"  Good — {countries_with_core} countries meets the 80-country minimum threshold")
else:
    print(f"  Warning — only {countries_with_core} countries. Results may be less reliable.")

# I save the FULL master dataset (all countries, including those with missing values)
out_full = "data/processed/master_dataset.csv"
master.to_csv(out_full, index=False)
print(f"\nFull master saved → {out_full} ({len(master)} countries, {master.shape[1]} columns)")

# I save the CLEAN subset — only countries with all core variables present
core_mask = master[CORE_COLS[0]].notna()
for col in CORE_COLS[1:]:
    core_mask = core_mask & master[col].notna()

master_clean = master[core_mask].copy()
out_clean = "data/processed/master_dataset_clean.csv"
master_clean.to_csv(out_clean, index=False)
print(f"Clean subset saved → {out_clean} ({len(master_clean)} countries)")

# I print a quick preview of the clean dataset
print("\n--- Preview of clean master dataset (key columns) ---")
preview_cols = ["country_name", "cereal_yield_kg_per_ha", "gdp_per_capita_usd"]
if "account_ownership_pct" in master_clean.columns:
    preview_cols.append("account_ownership_pct")
if "account_ownership_rural_pct" in master_clean.columns:
    preview_cols.append("account_ownership_rural_pct")
if "value_chain_finance_score" in master_clean.columns:
    preview_cols.append("value_chain_finance_score")
if "wgi_political_stability" in master_clean.columns:
    preview_cols.append("wgi_political_stability")
if "cereal_loss_pct" in master_clean.columns:
    preview_cols.append("cereal_loss_pct")

# I only show columns that actually exist
available_preview = [c for c in preview_cols if c in master_clean.columns]
print(master_clean[available_preview].head(10).to_string(index=False))

print("\n" + "=" * 60)
print("PHASE C COMPLETE")
print("Next step: Phase D — fit Models A, B, C, D")
print("=" * 60)
