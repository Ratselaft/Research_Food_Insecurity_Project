# ============================================================
# I'm cleaning and merging all my datasets into one master file
# ============================================================
#
# What I'm doing here:
#   I downloaded 5 datasets in Phase B. Now I need to clean
#   each one and join them all together into a single table —
#   one row per country — so I can use it in Phase D modelling.
#
# The master dataset will have these columns:
#   country_code, country_name
#   cereal_yield_kg_per_ha    (World Bank WDI)
#   fertiliser_kg_per_ha      (World Bank WDI)
#   arable_land_pct           (World Bank WDI)
#   gdp_per_capita_usd        (World Bank WDI)
#   population_total          (World Bank WDI)
#   internet_users_pct        (World Bank WDI)
#   stunting_pct_children     (World Bank WDI)
#   irrigated_land_pct        (World Bank WDI)
#   account_ownership_pct     (Findex 2021)
#   bank_branches_per_100k    (IMF FAS)
#   atm_per_100k              (IMF FAS)
#   cereal_loss_pct           (FAO FLW — cleaned)
#   fertiliser_efficiency     (engineered from yield / fertiliser)
# ============================================================

# I need pandas to work with tables of data
import pandas as pd

# I need numpy for mathematical operations
import numpy as np

# I need os to create folders and check file paths
import os

# I make sure my output folder exists before I try to save anything
os.makedirs("data/processed", exist_ok=True)

# I let the user know I'm starting
print("Starting Phase C — cleaning and merging datasets...")
print("=" * 55)


# ============================================================
# Step 1: I'm cleaning the FAO Food Loss and Waste data
# ============================================================
# The raw FLW file has 21,000+ rows covering many commodities.
# I need to:
#   a) Keep only cereal crops (wheat, maize, rice, sorghum etc.)
#   b) Keep only rows where loss_percentage is a real number
#   c) Average across commodities and years per country
#   d) Produce one number per country: average % cereal loss

print("\n[1/5] Cleaning FAO Food Loss and Waste data...")

# I set the path to the FLW file
# It has a double extension from the download, so I try both
FLW_FILE = "data/raw/fao_flw_losses.csv.csv"

# If that file doesn't exist, I try the single-extension version
if not os.path.exists(FLW_FILE):
    FLW_FILE = "data/raw/fao_flw_losses.csv"

# I read the raw FLW file into a table
flw_raw = pd.read_csv(FLW_FILE)

# I print a quick summary to check it loaded correctly
print(f"  Raw FLW file: {len(flw_raw)} rows, {flw_raw['country'].nunique()} countries")

# I write a list of words that appear in cereal commodity names
# I'll use these to filter only the cereal rows
CEREAL_WORDS = ["wheat", "maize", "corn", "rice", "sorghum", "barley",
                "millet", "oat", "rye", "cereal", "teff", "fonio"]

# I build a single search pattern by joining my words with "|" (means "or")
cereal_pattern = "|".join(CEREAL_WORDS)

# I create a True/False mask — True for rows where the commodity is a cereal
cereal_mask = flw_raw["commodity"].str.lower().str.contains(cereal_pattern, na=False)

# I keep only the cereal rows
flw_cereal = flw_raw[cereal_mask].copy()

# I print how many rows remain after filtering
print(f"  After keeping only cereals: {len(flw_cereal)} rows")

# I remove any rows where loss_percentage is blank
flw_cereal = flw_cereal.dropna(subset=["loss_percentage"])

# I convert the loss_percentage column to a proper number
# "errors='coerce'" turns any non-numeric values into NaN (blank)
flw_cereal["loss_percentage"] = pd.to_numeric(flw_cereal["loss_percentage"], errors="coerce")

# I remove any rows that became blank after the conversion
flw_cereal = flw_cereal.dropna(subset=["loss_percentage"])

# I remove any rows with impossible values (loss can't be below 0 or above 100)
flw_cereal = flw_cereal[flw_cereal["loss_percentage"] >= 0]
flw_cereal = flw_cereal[flw_cereal["loss_percentage"] <= 100]

# I print how many rows are left after cleaning
print(f"  After removing missing/impossible values: {len(flw_cereal)} rows")

# I want a 5-year window centred on 2021 (2018–2022) to match my other data
flw_recent = flw_cereal[flw_cereal["year"] >= 2018]
flw_recent = flw_recent[flw_recent["year"] <= 2022].copy()

# If this window gives me too few rows, I widen it
if len(flw_recent) < 100:
    print("  Warning: few rows in 2018–2022 — widening window to 2015–2023")
    flw_recent = flw_cereal[flw_cereal["year"] >= 2015]
    flw_recent = flw_recent[flw_recent["year"] <= 2023].copy()

# I print how many rows fell into my chosen year window
print(f"  Rows in selected year window: {len(flw_recent)}")

# I now average the loss percentage across all commodities and years per country
flw_country = flw_recent.groupby("country")["loss_percentage"].mean()

# I convert the result back into a regular table (not a grouped object)
flw_country = flw_country.reset_index()

# I rename the column to my preferred name
flw_country = flw_country.rename(columns={"loss_percentage": "cereal_loss_pct"})

# I round the loss percentage to 2 decimal places
flw_country["cereal_loss_pct"] = flw_country["cereal_loss_pct"].round(2)

# I print how many countries I ended up with
print(f"  Countries with cereal loss data: {len(flw_country)}")


# ============================================================
# Step 2: I'm loading the other datasets
# ============================================================

print("\n[2/5] Loading World Bank WDI, Findex, and IMF data...")

# I define a set of World Bank codes that are NOT real countries
# These are regional aggregates like "High income", "East Asia", etc.
# I need to remove them so my dataset only has real countries
REGIONAL_CODES = set()
REGIONAL_CODES.add('AFE')
REGIONAL_CODES.add('AFW')
REGIONAL_CODES.add('ARB')
REGIONAL_CODES.add('CEB')
REGIONAL_CODES.add('CSS')
REGIONAL_CODES.add('EAP')
REGIONAL_CODES.add('EAR')
REGIONAL_CODES.add('EAS')
REGIONAL_CODES.add('ECA')
REGIONAL_CODES.add('ECS')
REGIONAL_CODES.add('EMU')
REGIONAL_CODES.add('EUU')
REGIONAL_CODES.add('FCS')
REGIONAL_CODES.add('HIC')
REGIONAL_CODES.add('HPC')
REGIONAL_CODES.add('IBD')
REGIONAL_CODES.add('IBT')
REGIONAL_CODES.add('IDA')
REGIONAL_CODES.add('IDB')
REGIONAL_CODES.add('IDX')
REGIONAL_CODES.add('LAC')
REGIONAL_CODES.add('LCN')
REGIONAL_CODES.add('LDC')
REGIONAL_CODES.add('LIC')
REGIONAL_CODES.add('LMC')
REGIONAL_CODES.add('LMY')
REGIONAL_CODES.add('LTE')
REGIONAL_CODES.add('MEA')
REGIONAL_CODES.add('MIC')
REGIONAL_CODES.add('MNA')
REGIONAL_CODES.add('NAC')
REGIONAL_CODES.add('OED')
REGIONAL_CODES.add('OSS')
REGIONAL_CODES.add('PRE')
REGIONAL_CODES.add('PSS')
REGIONAL_CODES.add('PST')
REGIONAL_CODES.add('SAS')
REGIONAL_CODES.add('SSA')
REGIONAL_CODES.add('SSF')
REGIONAL_CODES.add('SST')
REGIONAL_CODES.add('TEA')
REGIONAL_CODES.add('TEC')
REGIONAL_CODES.add('TLA')
REGIONAL_CODES.add('TMN')
REGIONAL_CODES.add('TSA')
REGIONAL_CODES.add('TSS')
REGIONAL_CODES.add('UMC')
REGIONAL_CODES.add('WLD')
REGIONAL_CODES.add('XZN')

# I read the World Bank WDI file
wdi_raw = pd.read_csv("data/raw/worldbank_wdi_2021.csv")

# I remove rows with no country code
wdi_no_nulls = wdi_raw[wdi_raw["country_code"].notna()]

# I remove rows where the country code is a regional aggregate
wdi_real = wdi_no_nulls[~wdi_no_nulls["country_code"].isin(REGIONAL_CODES)]

# I remove any duplicate country codes (keeping the first one)
wdi = wdi_real.drop_duplicates(subset="country_code", keep="first")

# I reset the row numbers
wdi = wdi.reset_index(drop=True)

# I read the Findex file
findex_raw = pd.read_csv("data/raw/findex_2021.csv")

# I remove rows with no country code
findex_no_nulls = findex_raw.dropna(subset=["country_code"])

# I remove duplicate country codes
findex = findex_no_nulls.drop_duplicates(subset="country_code", keep="first")

# I reset the row numbers
findex = findex.reset_index(drop=True)

# I read the IMF file
imf_raw = pd.read_csv("data/raw/imf_financial_access.csv")

# I remove rows with no country code
imf_no_nulls = imf_raw.dropna(subset=["country_code"])

# I remove duplicate country codes
imf = imf_no_nulls.drop_duplicates(subset="country_code", keep="first")

# I reset the row numbers
imf = imf.reset_index(drop=True)

# I print a summary of what I've loaded
print(f"  WDI:    {len(wdi)} real countries")
print(f"  Findex: {len(findex)} countries")
print(f"  IMF:    {len(imf)} countries")


# ============================================================
# Step 3: I'm matching FAO country names to ISO3 codes
# ============================================================
# FAO uses country names like "Ethiopia", "Côte d'Ivoire".
# I need to match these to the ISO3 codes in the WDI table.

print("\n[3/5] Matching FAO country names to ISO3 codes...")

# I build a lookup dictionary: lowercase country name → ISO3 code
# I get this from the WDI table which has both names and codes
name_to_code = {}
for i in range(len(wdi)):
    name_lower = wdi["country_name"].iloc[i].lower().strip()
    code = wdi["country_code"].iloc[i]
    name_to_code[name_lower] = code

# I write a manual correction dictionary for names that don't match exactly
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
FAO_NAME_FIXES["united kingdom of great britain and northern ireland"] = "GBR"
FAO_NAME_FIXES["united kingdom"]                          = "GBR"
FAO_NAME_FIXES["somalia"]                                 = "SOM"

# I write a set of names that are aggregates, not real countries
NOT_COUNTRIES = set()
NOT_COUNTRIES.add("world")
NOT_COUNTRIES.add("asia")
NOT_COUNTRIES.add("africa")
NOT_COUNTRIES.add("europe")
NOT_COUNTRIES.add("americas")
NOT_COUNTRIES.add("oceania")


def get_iso3(country_name):
    # I convert the country name to lowercase for matching
    name_lower = str(country_name).lower().strip()

    # I skip any names that are aggregates, not real countries
    if name_lower in NOT_COUNTRIES:
        return None

    # I try to find an exact match in my WDI name lookup
    if name_lower in name_to_code:
        return name_to_code[name_lower]

    # If that didn't work, I try my manual fixes dictionary
    if name_lower in FAO_NAME_FIXES:
        return FAO_NAME_FIXES[name_lower]

    # If nothing matched, I return None (no code found)
    return None


# I apply my lookup function to every country in the FLW table
flw_country["country_code"] = flw_country["country"].apply(get_iso3)

# I count how many matched successfully
matched_count = flw_country["country_code"].notna().sum()

# I find the ones that didn't match so I can report them
unmatched = flw_country[flw_country["country_code"].isna()]["country"].tolist()

# I print the results
print(f"  Matched {matched_count} out of {len(flw_country)} FAO countries to ISO3 codes")

if unmatched:
    print(f"  Could not match: {unmatched[:10]}")

# I keep only the rows where I found a code
flw_country = flw_country.dropna(subset=["country_code"])


# ============================================================
# Step 4: I'm merging all datasets into one master table
# ============================================================

print("\n[4/5] Merging all datasets into one master table...")

# I start with the WDI table as my base (it has the most countries)
master = wdi.copy()

# I add the Findex financial access data
# I only need the country code and account ownership column from Findex
findex_slim = findex[["country_code", "account_ownership_pct"]].copy()

# I merge Findex onto my master table using country_code as the link
# "left" merge means I keep all WDI countries even if Findex has no data for them
master = master.merge(findex_slim, on="country_code", how="left")

# I add the IMF banking infrastructure data
# I only need the country code and the two banking columns
imf_slim = imf[["country_code", "bank_branches_per_100k", "atm_per_100k"]].copy()

# I merge IMF data onto my master table
master = master.merge(imf_slim, on="country_code", how="left")

# I add the FAO FLW cereal loss data
# I only need the country code and cereal loss column
flw_slim = flw_country[["country_code", "cereal_loss_pct"]].copy()

# I merge FLW data onto my master table
master = master.merge(flw_slim, on="country_code", how="left")

# I print the size of the merged table
print(f"  Master table: {len(master)} countries, {master.shape[1]} columns")
print(f"  Columns: {list(master.columns)}")


# ============================================================
# Step 5: I'm creating new engineered features
# ============================================================

print("\n[5/5] Engineering new features...")

# I create a fertiliser efficiency column
# This tells me how many kg of cereal each kg of fertiliser produces
# A higher number means the fertiliser is being used more efficiently

# I only calculate this where both values exist and fertiliser use is above zero
for i in range(len(master)):
    # I get the yield and fertiliser values for this country
    yield_val = master["cereal_yield_kg_per_ha"].iloc[i]
    fert_val  = master["fertiliser_kg_per_ha"].iloc[i]

    # I check that both values are numbers (not blank) and fertiliser is positive
    if pd.notna(yield_val) and pd.notna(fert_val) and fert_val > 0:
        # I calculate efficiency and round to 2 decimal places
        efficiency = round(yield_val / fert_val, 2)
        master.loc[master.index[i], "fertiliser_efficiency"] = efficiency

print("  fertiliser_efficiency = cereal_yield / fertiliser_kg_per_ha")


# ============================================================
# Step 6: I'm checking coverage and deciding which countries to keep
# ============================================================

# I list the columns that are essential for Model A (core predictors)
CORE_COLS = ["cereal_yield_kg_per_ha", "gdp_per_capita_usd", "arable_land_pct"]

print("\n--- Coverage check (how many countries have each variable) ---")

# I go through each column and count how many countries have a value
for col in master.columns:
    # I skip the identifier columns
    if col not in ["country_code", "country_name"]:
        n_with_data = master[col].notna().sum()
        print(f"  {col:<35} {n_with_data} / {len(master)} countries")

# I check how many countries have ALL three core variables
countries_with_core = 0
for i in range(len(master)):
    all_present = True
    for col in CORE_COLS:
        if pd.isna(master[col].iloc[i]):
            all_present = False
            break
    if all_present:
        countries_with_core += 1

print(f"\n  Countries with ALL core variables: {countries_with_core}")

# I need at least 80 countries for reliable regression results
if countries_with_core >= 80:
    print(f"  Good — {countries_with_core} countries meets the minimum threshold of 80")
else:
    print(f"  Warning — only {countries_with_core} countries. Regression may be less reliable.")


# ============================================================
# Step 7: I'm saving the master dataset
# ============================================================

# I save the FULL master dataset (all countries, including those with missing values)
out_full = "data/processed/master_dataset.csv"
master.to_csv(out_full, index=False)
print(f"\nFull master dataset saved → {out_full}  ({len(master)} countries)")

# I also save a clean subset: only countries with all three core variables present
all_core_present = master[CORE_COLS[0]].notna()
for col in CORE_COLS[1:]:
    all_core_present = all_core_present & master[col].notna()

master_clean = master[all_core_present].copy()
out_clean = "data/processed/master_dataset_clean.csv"
master_clean.to_csv(out_clean, index=False)
print(f"Clean subset saved → {out_clean}  ({len(master_clean)} countries)")

# I print the first few rows as a quick check
print("\n--- Preview of clean master dataset ---")
preview_cols = ["country_name", "cereal_yield_kg_per_ha", "gdp_per_capita_usd",
                "account_ownership_pct", "cereal_loss_pct", "fertiliser_efficiency"]
print(master_clean[preview_cols].head(10).to_string(index=False))

print("\n" + "=" * 55)
print("PHASE C COMPLETE")
print("Next step: Phase D — fit Models A, B, C")
print("=" * 55)
