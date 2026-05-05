# ============================================================
# I'm downloading ALL the datasets I need for Phase B
# ============================================================
#
# What I'm doing here:
#   Phase A found 9 topics in my papers. Each topic maps to
#   a real, measurable country-level number. This script
#   downloads all those numbers from free public databases.
#
# What has changed in this improved version:
#   - I now download MORE World Bank WDI indicators, including
#     rural population, female agriculture employment, food
#     price inflation, and agricultural value added
#   - I now download DISAGGREGATED Findex indicators so I can
#     see whether financial access reaches rural people,
#     women, and the poorest 40% — not just the national average
#   - I now try to download ALL 6 World Bank Governance
#     indicators (WGI), not just Rule of Law
#   - I now download FAOSTAT producer prices so I can see
#     what farmers actually earn per tonne
#   - I now download more IMF Financial Access data including
#     private sector credit (a proxy for agricultural lending)
#
# Why this matters for my dissertation:
#   My argument is that financial access along the value chain
#   matters more than national account ownership averages.
#   The disaggregated Findex indicators let me test this.
#
# Data focus year: 2021
#   - Global Findex 2021 only covers this year
#   - World Bank data is best populated for 2021
#   - FAO Food Loss data is most complete around 2021
# ============================================================

# I need requests to download files from the internet
import requests

# I need pandas to work with tables of data
import pandas as pd

# I need os to check folders and file paths
import os

# I need time to pause between downloads (polite to the servers!)
import time

# I make sure my output folders exist before saving anything
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# I let the user know I'm starting
print("Starting Phase B — downloading all datasets...")
print("=" * 60)


# ============================================================
# SHARED HELPER FUNCTION — World Bank API downloader
# ============================================================
# I will reuse this function many times throughout this script.
# It downloads one indicator for all countries and returns a table.

def download_world_bank(indicator_code, column_name, year=2021, source=None):
    # I build the URL for this specific World Bank indicator
    url = "https://api.worldbank.org/v2/country/all/indicator/" + indicator_code

    # I set up the request parameters
    params = {}
    params["date"]     = year    # I only want data for this year
    params["format"]   = "json"  # I want the response as JSON
    params["per_page"] = 350     # I want up to 350 entries in one request

    # Some World Bank datasets need a special 'source' parameter
    # For example: source=3 is needed for WGI governance indicators
    if source is not None:
        params["source"] = source

    # I try to download — if anything goes wrong I catch the error
    try:
        # I send the request and wait up to 30 seconds
        response = requests.get(url, params=params, timeout=30)

        # I convert the JSON response into Python data
        data = response.json()

        # The World Bank API returns a list with two items:
        # data[0] = header information (I don't need this)
        # data[1] = the actual country data (this is what I want)
        if len(data) < 2 or not data[1]:
            # If there is no data, I return nothing
            return None

        # I'll collect one row per country here
        rows = []

        # I go through each country entry one at a time
        for entry in data[1]:
            # I only keep entries that have an actual numeric value
            if entry.get("value") is not None:
                # I build a small dictionary for this country
                one_row = {}
                one_row["country_code"] = entry.get("countryiso3code", "")
                one_row["country_name"] = entry.get("country", {}).get("value", "")
                one_row[column_name]    = entry["value"]
                one_row["year"]         = year
                # I add this row to my list
                rows.append(one_row)

        # If I collected any rows, I turn them into a pandas table
        if len(rows) == 0:
            return None

        return pd.DataFrame(rows)

    except Exception as e:
        print("    Error downloading", column_name, ":", e)
        return None


# ============================================================
# Dataset 1: FAOSTAT — Crop yield, fertiliser, food supply,
#            and PRODUCER PRICES
# ============================================================
# FAOSTAT is the UN Food and Agriculture Organisation's database.
# Producer prices tell me what farmers actually receive per tonne.
# This is critical — a farmer who grows food but earns very little
# may still face food insecurity because they cannot afford to buy
# food on the market.

print("\n[1/6] Downloading FAOSTAT data...")

# This is the base web address for the FAOSTAT API
FAOSTAT_BASE = "https://fenixservices.fao.org/faostat/api/v1/en/data/"


def download_faostat(dataset_code, filename, year=2021):
    # I build the full web address for this FAOSTAT dataset
    url = FAOSTAT_BASE + dataset_code

    # I set the parameters — I want ISO3 country codes and CSV format
    params = {}
    params["area_cs"]     = "ISO3"   # I want country codes like NGA, KEN, IND
    params["year"]        = year     # I only want data for my chosen year
    params["output_type"] = "csv"    # I want a CSV file back

    # I tell the user what I'm fetching
    print("  Fetching", filename, "...")

    # I try to download and save the file
    try:
        response = requests.get(url, params=params, timeout=60)

        if response.status_code == 200:
            save_path = "data/raw/" + filename

            # I open the file for writing in binary mode
            with open(save_path, "wb") as f:
                f.write(response.content)

            # I try to read it back to check it worked properly
            try:
                df = pd.read_csv(save_path)
                print("  Saved", filename, "—", len(df), "rows,", df.shape[1], "columns")
                return df
            except Exception:
                print("  Saved", filename, "(could not preview)")
                return None
        else:
            print("  Could not download", filename, "(server status:", response.status_code, ")")
            return None

    except Exception as e:
        print("  Error downloading", filename, ":", e)
        return None


# I download cereal yield data — kg of cereal produced per hectare
yield_df = download_faostat("QCL", "faostat_cereal_yield.csv")
time.sleep(1)

# I download fertiliser use data — kg of fertiliser applied per hectare
fert_df = download_faostat("RFN", "faostat_fertiliser.csv")
time.sleep(1)

# I download food balance sheets — kcal available per person per day
food_supply_df = download_faostat("FBS", "faostat_food_supply.csv")
time.sleep(1)

# NEW: I download producer prices — what farmers are paid per tonne
# This tells me whether farmers are receiving fair economic value
# for their crops, which directly affects their food security
print("  Fetching producer prices (what farmers earn per tonne)...")
producer_prices_df = download_faostat("PP", "faostat_producer_prices.csv")
time.sleep(1)


# ============================================================
# Dataset 2: World Bank WDI — expanded indicator set
# ============================================================
# I've significantly expanded this from the original 9 indicators.
# I now also download:
#   - Rural population percentage (tells me how agricultural the country is)
#   - Agricultural employment % (how many people work in farming)
#   - Female agricultural employment % (critical for value chain analysis)
#   - Agriculture's share of GDP (how important farming is to the economy)
#   - Food price inflation (do price rises hurt food access?)
#   - Rural electricity access (can farmers use modern equipment?)
#   - Livestock production index (not just crops — animals matter too)
#   - Credit rights index (how legally protected is borrowing?)

print("\n[2/6] Downloading World Bank WDI indicators (expanded)...")

# I build my dictionary of World Bank indicator codes.
# The left side (key) is the code the World Bank uses.
# The right side (value) is the friendly column name I want.
WB_INDICATORS = {}

# ── Production and land variables ────────────────────────────────────────────
WB_INDICATORS["AG.YLD.CREL.KG"]    = "cereal_yield_kg_per_ha"
# How many kg of cereal does each hectare of farmland produce?

WB_INDICATORS["AG.CON.FERT.ZS"]    = "fertiliser_kg_per_ha"
# How many kg of fertiliser are applied per hectare?

WB_INDICATORS["AG.LND.ARBL.ZS"]    = "arable_land_pct"
# What percentage of land is suitable for crops?

WB_INDICATORS["AG.LND.IRIG.AG.ZS"] = "irrigated_land_pct"
# What share of agricultural land has irrigation?

WB_INDICATORS["AG.PRD.LVSK.XD"]    = "livestock_production_index"
# An index showing how much livestock production has changed

# ── Income and development variables ─────────────────────────────────────────
WB_INDICATORS["NY.GDP.PCAP.KD"]    = "gdp_per_capita_usd"
# Average income per person, adjusted for inflation

WB_INDICATORS["NV.AGR.TOTL.ZS"]    = "agri_value_added_pct_gdp"
# How much does farming contribute to the national economy? (% of GDP)
# Countries where agriculture is a large % of GDP tend to have more
# smallholder farmers who are vulnerable to food insecurity

WB_INDICATORS["FP.CPI.TOTL.ZG"]    = "food_price_inflation_pct"
# How fast are food prices rising? High food price inflation
# directly reduces what a fixed income can buy

# ── Population and demographics ───────────────────────────────────────────────
WB_INDICATORS["SP.POP.TOTL"]        = "population_total"
# Total number of people in the country

WB_INDICATORS["SP.RUR.TOTL.ZS"]    = "rural_population_pct"
# What share of the population lives in rural areas?
# High rural population often means more people depend on farming

# ── Agricultural employment — WHO is in the food value chain? ─────────────────
WB_INDICATORS["SL.AGR.EMPL.ZS"]    = "agri_employment_pct"
# What share of all workers are in agriculture?
# This tells me how many people's livelihoods depend on food production

WB_INDICATORS["SL.AGR.EMPL.FE.ZS"] = "female_agri_employment_pct"
# What share of female workers are in agriculture?
# Women in Sub-Saharan Africa do 60–80% of food production work
# but often have less access to finance — this is a key variable

# ── Infrastructure ─────────────────────────────────────────────────────────────
WB_INDICATORS["IT.NET.USER.ZS"]    = "internet_users_pct"
# Internet access is a proxy for digital financial inclusion

WB_INDICATORS["EG.ELC.ACCS.RU.ZS"] = "rural_electricity_access_pct"
# Can rural/farming communities access electricity?
# Without power, cold storage is impossible — food losses increase

# ── Health and nutrition ───────────────────────────────────────────────────────
WB_INDICATORS["SH.STA.STNT.ZS"]    = "stunting_pct_children"
# % of children under 5 who are stunted (a long-run malnutrition indicator)

# ── Credit and financial system ────────────────────────────────────────────────
WB_INDICATORS["IC.LGL.CRED.XQ"]    = "credit_rights_index"
# How strong are the legal rights that protect lenders and borrowers?
# A higher score means it is easier and safer to get a loan

WB_INDICATORS["EN.ATM.CO2E.KT"]    = "co2_emissions_kt"
# CO2 emissions as a climate change proxy


# I'll collect each indicator's table in this list
wdi_tables = []

# I go through each indicator one by one
for code in WB_INDICATORS:
    # I get the friendly column name for this indicator
    name = WB_INDICATORS[code]

    # I print progress
    print("  Fetching", name, "...")

    # I download this indicator for all countries
    df = download_world_bank(code, name, year=2021)

    # If I got real data back
    if df is not None and len(df) > 0:
        # I only keep the three columns I need: code, name, value
        slim_df = df[["country_code", "country_name", name]].copy()
        wdi_tables.append(slim_df)
        print("    Got data for", len(slim_df), "countries")

    # I pause briefly between requests to be polite to the server
    time.sleep(0.4)

# Now I merge all individual indicator tables into one big table
if len(wdi_tables) > 0:
    # I start with the first table as my base
    wdi_merged = wdi_tables[0]

    # I add each subsequent table by merging on country_code and country_name
    for table in wdi_tables[1:]:
        wdi_merged = wdi_merged.merge(
            table,
            on=["country_code", "country_name"],
            how="outer"  # 'outer' means I keep all countries even if some data is missing
        )

    # I save the combined table
    wdi_merged.to_csv("data/raw/worldbank_wdi_2021.csv", index=False)
    print("  Saved worldbank_wdi_2021.csv —", len(wdi_merged), "countries,",
          wdi_merged.shape[1], "columns")
else:
    print("  Could not download any World Bank data")


# ============================================================
# Dataset 3: Global Findex 2021 — DISAGGREGATED financial access
# ============================================================
# This is the most important improvement in Phase B.
#
# Previously I only downloaded the NATIONAL average account ownership.
# But my dissertation argument is about whether financial access reaches
# PEOPLE IN THE FOOD VALUE CHAIN — farmers, rural traders, women.
#
# So now I download SEPARATE indicators for:
#   - Rural adults (do people in farming areas have accounts?)
#   - Women (do female farmers — who grow 60–80% of food in Africa — have access?)
#   - Poorest 40% (does finance reach smallholders who earn the least?)
#   - Agricultural payments received digitally (are farmers in the digital economy?)
#   - Borrowing from a financial institution (not just having an account — using it for credit)
#
# If rural and agricultural-specific financial access explains food insecurity
# BETTER than the national average, that supports my dissertation argument.

print("\n[3/6] Downloading Findex 2021 — disaggregated financial access indicators...")

# I define ALL the Findex indicators I want to download.
# Each one is a separate API call.
FINDEX_INDICATORS = {}

# Overall/national account ownership (I keep this for comparison)
FINDEX_INDICATORS["FX.OWN.TOTL.ZS"]    = "account_ownership_pct"

# RURAL account ownership — this is the key variable for value chain analysis
# If rural people don't have accounts, farmers can't receive payments digitally
FINDEX_INDICATORS["FX.OWN.TOTL.RU.ZS"] = "account_ownership_rural_pct"

# FEMALE account ownership — women grow most of the food in low-income countries
# If they cannot access finance, they cannot invest in better farming
FINDEX_INDICATORS["FX.OWN.TOTL.FE.ZS"] = "account_ownership_female_pct"

# POOREST 40% account ownership — the poorest households face the most food insecurity
# Does finance reach the people who need it most?
FINDEX_INDICATORS["FX.OWN.TOTL.PL.ZS"] = "account_ownership_poorest40_pct"

# BORROWED from a financial institution (not just having an account — actually getting credit)
# This shows whether credit actually flows to people, not just whether accounts exist
FINDEX_INDICATORS["FX.TRN.FINM.ZS"]    = "borrowed_from_bank_pct"

# AGRICULTURAL payments received digitally
# This directly measures whether farmers are connected to the formal financial system
# A farmer receiving payment in cash cannot save easily or access credit
FINDEX_INDICATORS["FX.TRN.AGRI.ZS"]    = "agri_payments_digital_pct"

# I'll collect each Findex indicator table here
findex_tables = []

# I go through each Findex indicator one at a time
for code in FINDEX_INDICATORS:
    col_name = FINDEX_INDICATORS[code]
    print("  Fetching", col_name, "...")

    # I download this Findex indicator for all countries
    df = download_world_bank(code, col_name, year=2021)

    if df is not None and len(df) > 0:
        # I only keep the country code and the value column
        slim_df = df[["country_code", col_name]].copy()
        findex_tables.append(slim_df)
        print("    Got data for", len(slim_df), "countries")
    else:
        print("    No data for", col_name, "(may not be in WB API — check Findex website)")

    time.sleep(0.5)

# I merge all Findex tables into one
if len(findex_tables) > 0:
    findex_merged = findex_tables[0]
    for table in findex_tables[1:]:
        findex_merged = findex_merged.merge(table, on="country_code", how="outer")

    # I save the combined Findex table
    findex_merged.to_csv("data/raw/findex_2021.csv", index=False)
    print("  Saved findex_2021.csv —", len(findex_merged), "countries,",
          findex_merged.shape[1], "columns")
else:
    print("  Could not get any Findex data")
    pd.DataFrame(columns=["country_code", "account_ownership_pct"]).to_csv(
        "data/raw/findex_2021.csv", index=False
    )

time.sleep(1)


# ============================================================
# Dataset 4: FAO Food Loss and Waste — post-harvest loss rates
# ============================================================
# I try the FAOSTAT API for Food Loss data.
# The improved version also captures supply chain stage information
# (farm, storage, transport, processing, retail) where available.
# This tells me WHERE in the value chain the losses happen.

print("\n[4/6] Downloading FAO Food Loss and Waste data...")
print("  Trying FAO FLW API...")

try:
    flw_api = "https://fenixservices.fao.org/faostat/api/v1/en/data/FLWSTAT"

    # I request ALL years so I have more coverage
    # I will filter to recent years in Phase C
    params = {}
    params["output_type"] = "csv"
    params["area_cs"]     = "ISO3"

    response = requests.get(flw_api, params=params, timeout=60)

    if response.status_code == 200 and len(response.content) > 500:
        # I save the raw file
        with open("data/raw/fao_flw_losses.csv", "wb") as f:
            f.write(response.content)

        # I check what columns are available — I want to find supply chain stage
        try:
            flw_df = pd.read_csv("data/raw/fao_flw_losses.csv")
            print("  Saved fao_flw_losses.csv —", len(flw_df), "rows")
            print("  Columns available:", list(flw_df.columns))

            # I check if there is a supply chain stage column
            for col in flw_df.columns:
                if "stage" in col.lower() or "chain" in col.lower() or "supply" in col.lower():
                    print("  Value chain stage column found:", col)
                    print("  Unique stages:", flw_df[col].unique()[:10])

        except Exception:
            print("  File saved (could not preview columns)")
    else:
        # If the API didn't work, I create a placeholder and show download instructions
        print("  FAO FLW API returned no data.")
        print("  IMPORTANT — I need to download this manually:")
        print("  1. Go to: https://www.fao.org/platform-food-loss-waste/flw-data/en/")
        print("  2. Click 'Download data'")
        print("  3. Select 'All regions', 'All commodities', 'All supply chain stages'")
        print("  4. Save the file as: data/raw/fao_flw_losses.csv")
        print("  Creating placeholder file so the rest of the pipeline does not crash...")

        placeholder_cols = ["country", "country_code", "commodity", "year",
                            "loss_percentage", "supply_chain_stage"]
        pd.DataFrame(columns=placeholder_cols).to_csv("data/raw/fao_flw_losses.csv", index=False)

except Exception as e:
    print("  Error:", e)
    placeholder_cols = ["country", "country_code", "commodity", "year",
                        "loss_percentage", "supply_chain_stage"]
    pd.DataFrame(columns=placeholder_cols).to_csv("data/raw/fao_flw_losses.csv", index=False)
    print("  Placeholder created.")

time.sleep(1)


# ============================================================
# Dataset 5: IMF Financial Access Survey — expanded
# ============================================================
# I now download more banking infrastructure indicators:
#   - Bank branches per 100,000 adults (physical access)
#   - ATMs per 100,000 adults (cash access)
#   - Domestic credit to private sector (% GDP) — proxy for whether
#     banks actually LEND to businesses and farms, not just hold deposits
#
# The credit to private sector indicator is important because a country
# can have many bank branches but still not lend to smallholders.
# Low private credit % signals that banks are not serving the productive economy.

print("\n[5/6] Downloading IMF Financial Access Survey data...")

# I'll collect each IMF/banking indicator here
imf_tables = []

# Indicator 1: Bank branches per 100,000 adults
print("  Fetching bank branches per 100,000 adults...")
branches_df = download_world_bank("FB.CBK.BRCH.P5", "bank_branches_per_100k", 2021)
if branches_df is not None and len(branches_df) > 0:
    imf_tables.append(branches_df[["country_code", "bank_branches_per_100k"]])
    print("    Got data for", len(branches_df), "countries")
time.sleep(0.5)

# Indicator 2: ATMs per 100,000 adults
print("  Fetching ATMs per 100,000 adults...")
atm_df = download_world_bank("FB.ATM.TOTL.P5", "atm_per_100k", 2021)
if atm_df is not None and len(atm_df) > 0:
    imf_tables.append(atm_df[["country_code", "atm_per_100k"]])
    print("    Got data for", len(atm_df), "countries")
time.sleep(0.5)

# Indicator 3: Domestic credit to private sector (% of GDP)
# This is my proxy for whether the financial system is ACTIVELY LENDING
# to businesses and farms. A country where banks barely lend (low %)
# means smallholder farmers cannot get the loans they need to invest
# in better inputs, storage, or post-harvest handling.
print("  Fetching domestic credit to private sector (% GDP)...")
credit_df = download_world_bank("FS.AST.PRVT.GD.ZS", "private_credit_pct_gdp", 2021)
if credit_df is not None and len(credit_df) > 0:
    imf_tables.append(credit_df[["country_code", "private_credit_pct_gdp"]])
    print("    Got data for", len(credit_df), "countries")
time.sleep(0.5)

# Indicator 4: Commercial bank deposits (% of GDP)
# Where bank branches exist, do people actually USE them to save?
print("  Fetching bank deposits as % of GDP...")
deposits_df = download_world_bank("FD.AST.PRVT.GD.ZS", "bank_deposits_pct_gdp", 2021)
if deposits_df is not None and len(deposits_df) > 0:
    imf_tables.append(deposits_df[["country_code", "bank_deposits_pct_gdp"]])
    print("    Got data for", len(deposits_df), "countries")
time.sleep(0.5)

# I merge all IMF/banking tables together
if len(imf_tables) > 0:
    imf_merged = imf_tables[0]
    for table in imf_tables[1:]:
        imf_merged = imf_merged.merge(table, on="country_code", how="outer")

    imf_merged.to_csv("data/raw/imf_financial_access.csv", index=False)
    print("  Saved imf_financial_access.csv —", len(imf_merged), "countries,",
          imf_merged.shape[1], "columns")
else:
    print("  Could not get any IMF data — placeholder created")
    pd.DataFrame(columns=["country_code", "bank_branches_per_100k"]).to_csv(
        "data/raw/imf_financial_access.csv", index=False
    )

time.sleep(1)


# ============================================================
# Dataset 6: World Bank WGI — ALL SIX governance dimensions
# ============================================================
# Previously I only tried Rule of Law and got nothing.
# Now I try ALL SIX governance indicators using the correct
# World Bank API source parameter (source=3 = WGI dataset).
#
# The six WGI dimensions and why each matters for food security:
#
#   1. Control of Corruption (CC.EST)
#      Corruption diverts food aid, subsidies, and agricultural grants
#      away from smallholders. High corruption = finance never reaches farmers.
#
#   2. Government Effectiveness (GE.EST)
#      Can the government actually deliver extension services, build roads,
#      run storage facilities? Poor effectiveness = policy failures.
#
#   3. Political Stability (PV.EST)
#      Conflict and instability destroy food supply chains directly.
#      This is one of the strongest predictors of famine.
#
#   4. Regulatory Quality (RQ.EST)
#      Can the government make sensible food, agriculture, and finance policies?
#
#   5. Rule of Law (RL.EST)
#      Land rights, contract enforcement — farmers who don't own their land
#      cannot use it as collateral for a loan.
#
#   6. Voice and Accountability (VA.EST)
#      Can citizens and farmers hold governments to account for food policy?

print("\n[6/6] Downloading World Bank WGI — Governance Indicators...")
print("  Trying WGI API (source=3)...")

# I define all six WGI indicator codes and their names
WGI_INDICATORS = {}
WGI_INDICATORS["CC.EST"] = "wgi_control_of_corruption"
WGI_INDICATORS["GE.EST"] = "wgi_government_effectiveness"
WGI_INDICATORS["PV.EST"] = "wgi_political_stability"
WGI_INDICATORS["RQ.EST"] = "wgi_regulatory_quality"
WGI_INDICATORS["RL.EST"] = "wgi_rule_of_law"
WGI_INDICATORS["VA.EST"] = "wgi_voice_accountability"

# I'll collect each WGI indicator table here
wgi_tables = []

# I try each WGI indicator one at a time
for code in WGI_INDICATORS:
    col_name = WGI_INDICATORS[code]
    print("  Fetching", col_name, "...")

    # I try with source=3 (WGI database source)
    df = download_world_bank(code, col_name, year=2021, source=3)

    if df is not None and len(df) > 0:
        wgi_tables.append(df[["country_code", col_name]])
        print("    Got data for", len(df), "countries")
    else:
        # The WGI API often doesn't work via the standard endpoint
        # I try without the source parameter as a backup
        df = download_world_bank(code, col_name, year=2021)
        if df is not None and len(df) > 0:
            wgi_tables.append(df[["country_code", col_name]])
            print("    Got data via fallback for", len(df), "countries")
        else:
            print("    No data for", col_name, "(API unavailable)")

    time.sleep(0.5)

# I merge whatever WGI data I got
if len(wgi_tables) > 0:
    wgi_merged = wgi_tables[0]
    for table in wgi_tables[1:]:
        wgi_merged = wgi_merged.merge(table, on="country_code", how="outer")

    wgi_merged.to_csv("data/raw/wgi_governance_2021.csv", index=False)
    print("  Saved wgi_governance_2021.csv —", len(wgi_merged), "countries,",
          wgi_merged.shape[1], "columns")
else:
    # The WGI API returned nothing — I create a placeholder and show clear instructions
    print()
    print("  WGI API returned no data. I need to download this manually.")
    print("  Here is exactly what to do:")
    print("  1. Go to: https://info.worldbank.org/governance/wgi/")
    print("  2. Click 'Download the data'")
    print("  3. Select: All Indicators, All Countries, Year = 2021")
    print("  4. Download as CSV or Excel")
    print("  5. Rename/save as: data/raw/wgi_governance_2021.csv")
    print("  6. Make sure the file has these column names:")
    print("       country_code, wgi_control_of_corruption,")
    print("       wgi_government_effectiveness, wgi_political_stability,")
    print("       wgi_regulatory_quality, wgi_rule_of_law, wgi_voice_accountability")
    print()
    print("  Creating a placeholder file so the pipeline does not crash...")

    # I create a placeholder so Phase C does not crash when looking for this file
    wgi_placeholder_cols = [
        "country_code",
        "wgi_control_of_corruption",
        "wgi_government_effectiveness",
        "wgi_political_stability",
        "wgi_regulatory_quality",
        "wgi_rule_of_law",
        "wgi_voice_accountability"
    ]
    pd.DataFrame(columns=wgi_placeholder_cols).to_csv(
        "data/raw/wgi_governance_2021.csv", index=False
    )
    print("  Placeholder saved as data/raw/wgi_governance_2021.csv")


# ============================================================
# Final summary — I print what I have and what is missing
# ============================================================

print("\n" + "=" * 60)
print("PHASE B COMPLETE — File Summary")
print("=" * 60)

# I check each expected file and report its status
files_to_check = {}
files_to_check["data/raw/faostat_cereal_yield.csv"]    = "FAOSTAT — Cereal yield"
files_to_check["data/raw/faostat_fertiliser.csv"]      = "FAOSTAT — Fertiliser use"
files_to_check["data/raw/faostat_food_supply.csv"]     = "FAOSTAT — Food balance sheets"
files_to_check["data/raw/faostat_producer_prices.csv"] = "FAOSTAT — Producer prices (what farmers earn)"
files_to_check["data/raw/worldbank_wdi_2021.csv"]      = "World Bank WDI — all indicators"
files_to_check["data/raw/findex_2021.csv"]             = "Findex — disaggregated financial access"
files_to_check["data/raw/fao_flw_losses.csv"]          = "FAO — Food loss and waste"
files_to_check["data/raw/imf_financial_access.csv"]    = "IMF/WB — Banking infrastructure"
files_to_check["data/raw/wgi_governance_2021.csv"]     = "WGI — Governance indicators"

for filepath in files_to_check:
    label = files_to_check[filepath]
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            if len(df) > 0:
                status = "OK — " + str(len(df)) + " rows"
            else:
                status = "EMPTY — fill manually (see instructions above)"
        except Exception:
            status = "Saved (unreadable)"
    else:
        status = "MISSING"

    if "OK" in status:
        print("  OK  ", label, ":", status)
    elif "EMPTY" in status:
        print("  FILL", label, ":", status)
    else:
        print("  MISS", label, ":", status)

print()
print("Files marked FILL need manual download — see instructions printed above.")
print("Next step: Phase C — clean and merge all files into one master dataset.")
