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
#   - I now use the APHLIS + FAO FBS combined PHL dataset
#     (real country-level post-harvest loss for 159 countries)
#     instead of the old FAO FLW platform data (regional proxies)
#   - I now merge ALL the new WDI indicators (rural population,
#     agricultural employment, female agriculture, food price
#     inflation, livestock production, credit rights)
#   - I now merge ALL disaggregated Findex indicators (rural,
#     female, poorest 40%, agricultural payments, borrowing)
#   - I now merge ALL 6 WGI governance indicators
#     (or skip gracefully if the manual file is not ready)
#   - I now merge the Logistics Performance Index (LPI) —
#     an NLP-discovered variable for value chain / market access
#   - I now merge rural poverty headcount (NLP smallholder theme)
#   - I now merge mobile / internet access (NLP financial access
#     in rural context)
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
# Step 1: I'm loading the Post-Harvest Loss combined dataset
# ============================================================
# scripts/mine_additional_data.py built phl_combined.csv by layering
# three data sources in priority order:
#
#   Layer A1 (highest quality): 39 African countries from APHLIS.
#     APHLIS uses validated crop-system models — not accounting residuals.
#
#   Layer A2: FAO Food Balance Sheets (real country-level data).
#     Element Code 5123 (Losses) / Element Code 5301 (Domestic supply)
#     gives a real cereal loss percentage for 120+ additional countries.
#
#   Layer C: sub-regional proxies for the remaining countries
#     (9 unique values rather than the original 4 continental averages).
#
# The phl_combined.csv already has country_code and cereal_loss_pct
# columns — no name matching or cleaning is needed here.

print("\n[1/7] Loading Post-Harvest Loss data (APHLIS + FAO Food Balance Sheets)...")

PHL_FILE = "data/raw/phl_combined.csv"
if os.path.exists(PHL_FILE):
    phl_raw = pd.read_csv(PHL_FILE)
    flw_country = phl_raw[["country_code", "cereal_loss_pct"]].dropna(
        subset=["country_code", "cereal_loss_pct"]
    ).copy()
    if "phl_quality" in phl_raw.columns:
        n_real = phl_raw["phl_quality"].str.startswith("A").sum()
        print(f"  Loaded {len(flw_country)} countries with PHL data")
        print(f"  Real country-level data (Quality A): {n_real} countries")
    else:
        print(f"  Loaded {len(flw_country)} countries with PHL data")
else:
    print("  phl_combined.csv not found — run scripts/mine_additional_data.py first")
    flw_country = pd.DataFrame(columns=["country_code", "cereal_loss_pct"])

# phl_combined has no supply chain stage breakdown
flw_by_stage = pd.DataFrame(columns=["country_code"])


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

# I merge the Post-Harvest Loss data (cereal_loss_pct)
# phl_combined.csv already uses country_code so no name matching is needed
if "country_code" in flw_country.columns:
    flw_slim = flw_country[["country_code", "cereal_loss_pct"]].copy()
    master = master.merge(flw_slim, on="country_code", how="left")
    print("  After merging PHL (cereal_loss_pct):", master.shape)

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

# ── LPI retained for robustness checks only (~90/174 countries) ────────────────
LPI_FILE = "data/raw/lpi.csv"
if os.path.exists(LPI_FILE):
    lpi_raw = pd.read_csv(LPI_FILE)
    lpi_slim = lpi_raw[["country_code", "lpi_overall"]].dropna(subset=["country_code"]).copy()
    master = master.merge(lpi_slim, on="country_code", how="left")
    print(f"  After merging LPI (robustness only): {master.shape}")

# ── Trade % GDP — primary logistics/market-integration proxy ───────────────────
# NE.TRD.GNFS.ZS covers ~175 countries and proxies the value-chain / market
# integration theme from NMF Topic 6. Used in Models C and F instead of LPI.
TRADE_FILE = "data/raw/trade_pct_gdp.csv"
if os.path.exists(TRADE_FILE):
    trade_raw = pd.read_csv(TRADE_FILE)
    trade_slim = trade_raw[["country_code", "trade_pct_gdp"]].dropna(
        subset=["country_code"]
    ).copy()
    master = master.merge(trade_slim, on="country_code", how="left")
    n_trade = master["trade_pct_gdp"].notna().sum()
    print(f"  After merging trade % GDP: {master.shape} — {n_trade} countries have data")
else:
    print("  trade_pct_gdp.csv not found — run scripts/download_missing_data.py first")

# ── NEW: Rural poverty headcount (NLP theme: smallholder / poverty) ────────────
# The $2.15/day poverty headcount directly captures the resource constraints
# facing smallholder farmers — the group the literature most discusses.
POVERTY_FILE = "data/raw/rural_poverty.csv"
if os.path.exists(POVERTY_FILE):
    pov_raw = pd.read_csv(POVERTY_FILE)
    pov_slim = pov_raw[["country_code", "poverty_headcount_pct_215"]].dropna(
        subset=["country_code"]
    ).copy()
    master = master.merge(pov_slim, on="country_code", how="left")
    print("  After merging rural poverty:", master.shape)
else:
    print("  Rural poverty file not found — run scripts/download_new_data.py first")

# ── NEW: Mobile & internet access (NLP theme: financial access in rural areas) ──
# Mobile subscriptions and internet usage proxy the reach of digital financial
# services in rural areas — the channel the literature highlights for
# smallholder inclusion beyond formal bank branches.
MOBILE_FILE = "data/raw/mobile_financial_access.csv"
if os.path.exists(MOBILE_FILE):
    mob_raw = pd.read_csv(MOBILE_FILE)
    mob_cols = ["country_code"]
    for col in ["mobile_subscriptions_per_100"]:
        # internet_users_pct is already in master from WDI — skip it here
        if col in mob_raw.columns:
            mob_cols.append(col)
    mob_slim = mob_raw[mob_cols].dropna(subset=["country_code"]).copy()
    mob_slim = mob_slim.drop_duplicates(subset="country_code", keep="first")
    master = master.merge(mob_slim, on="country_code", how="left")
    print("  After merging mobile/digital access:", master.shape)
else:
    print("  Mobile access file not found — run scripts/download_new_data.py first")

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
if "cereal_yield_kg_per_ha" in master.columns and "fertiliser_kg_per_ha" in master.columns:
    fert_nonzero = master["fertiliser_kg_per_ha"].replace(0, np.nan)
    master["fertiliser_efficiency"] = (
        master["cereal_yield_kg_per_ha"] / fert_nonzero
    ).round(2)

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

# ── Feature 2: Value chain financial access score ─────────────────────────────
# Priority order for components (most to least value-chain-specific):
#   Tier 1 — directly value-chain-specific (from Findex disaggregated):
#     account_ownership_rural_pct, agri_payments_digital_pct, borrowed_from_bank_pct
#   Tier 2 — partially value-chain-specific (available disaggregated Findex):
#     account_ownership_female_pct, account_ownership_poorest40_pct
#   Tier 3 — broad financial access (used only if Tier 1/2 are absent):
#     bank_branches_per_100k, atm_per_100k, private_credit_pct_gdp
#
# All non-percentage variables are scaled to 0–100 using the sample maximum.
# Components are averaged row-wise, so countries with partial data still get a score.

VCHAIN_COMPONENTS = []

# Tier 1 — preferred (value-chain-specific Findex variables)
for col in ["account_ownership_rural_pct", "agri_payments_digital_pct",
            "borrowed_from_bank_pct"]:
    if col in master.columns and master[col].notna().sum() >= 50:
        VCHAIN_COMPONENTS.append(col)

# Tier 2 — disaggregated Findex (available for ~117 countries)
for col in ["account_ownership_female_pct", "account_ownership_poorest40_pct"]:
    if col in master.columns and master[col].notna().sum() >= 50:
        VCHAIN_COMPONENTS.append(col)

# Tier 3 — broad financial infrastructure (scaled to 0–100)
for col, raw_col in [
    ("bank_branches_scaled", "bank_branches_per_100k"),
    ("atm_scaled",           "atm_per_100k"),
    ("credit_scaled",        "private_credit_pct_gdp"),
]:
    if raw_col in master.columns:
        col_max = master[raw_col].max()
        if pd.notna(col_max) and col_max > 0:
            master[col] = (master[raw_col] / col_max * 100).round(2)
            VCHAIN_COMPONENTS.append(col)

if len(VCHAIN_COMPONENTS) > 0:
    print("  Computing value_chain_finance_score from:", VCHAIN_COMPONENTS)
    master["value_chain_finance_score"] = (
        master[VCHAIN_COMPONENTS].mean(axis=1).round(2)
    )
    non_null = master["value_chain_finance_score"].notna().sum()
    print("  value_chain_finance_score computed for", non_null, "countries")
else:
    print("  No financial access data available for value chain score")
    print("  Check that Phase B downloaded findex_2021.csv and imf_financial_access.csv")


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
if "lpi_overall" in master_clean.columns:
    preview_cols.append("lpi_overall")
if "poverty_headcount_pct_215" in master_clean.columns:
    preview_cols.append("poverty_headcount_pct_215")

# I only show columns that actually exist
available_preview = [c for c in preview_cols if c in master_clean.columns]
print(master_clean[available_preview].head(10).to_string(index=False))

print("\n" + "=" * 60)
print("PHASE C COMPLETE")
print("Next step: Phase D — fit Models A, B, C, D, E, F")
print("=" * 60)
