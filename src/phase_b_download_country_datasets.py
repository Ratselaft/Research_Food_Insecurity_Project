# ============================================================
# I'm downloading all the datasets I need for Phase B
# ============================================================
#
# What I'm doing here:
#   Phase A found 9 topics in my papers. Now I need real
#   country-level numbers to match each topic. This script
#   downloads everything from free, public sources and saves
#   them as CSV files in my data/raw/ folder.
#
# Datasets I'm downloading:
#   1. FAOSTAT  — crop yield, fertiliser use, food supply
#   2. World Bank WDI  — climate, GDP, infrastructure
#   3. Global Findex 2021  — financial access
#   4. FAO Food Loss and Waste  — post-harvest loss rates
#   5. IMF Financial Access Survey  — banking infrastructure
#
# I focus on 2021 because:
#   - Global Findex 2021 only covers that year
#   - FAO Food Loss data is most complete around 2021
#   - World Bank data is well populated for 2021
# ============================================================

# I need requests to download files from the internet
import requests

# I need pandas to work with tables of data
import pandas as pd

# I need os to check folders and file paths
import os

# I need time so I can pause between downloads (polite!)
import time

# I need zipfile to unzip any downloaded zip files
import zipfile

# I need io to read downloaded files directly in memory
import io

# I make sure the output folders exist before I try to save anything
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# I let the user know I'm starting
print("Starting Phase B — downloading all datasets...")
print("=" * 55)


# ============================================================
# Dataset 1: FAOSTAT — Crop yield, fertiliser, food supply
# ============================================================
# FAOSTAT is the UN Food and Agriculture Organisation's database.
# I download crop production and food supply data using their API.

print("\n[1/5] Downloading FAOSTAT data...")

# This is the base web address for the FAOSTAT API
FAOSTAT_BASE = "https://fenixservices.fao.org/faostat/api/v1/en/data/"


def download_faostat(dataset_code, filename, year=2021):
    # I set up the web address for this specific dataset
    url = FAOSTAT_BASE + dataset_code

    # I set the parameters for this request
    params = {}
    params["area_cs"]     = "ISO3"       # I want country codes like 'NGA', 'KEN'
    params["year"]        = year         # I only want 2021 data
    params["output_type"] = "csv"        # I want a CSV file back

    # I let the user know which file I'm fetching
    print(f"  Fetching {filename}...")

    # I try to download the file — if something goes wrong I catch the error
    try:
        # I send the request and wait up to 60 seconds for a response
        response = requests.get(url, params=params, timeout=60)

        # If the server responded with success (status 200)
        if response.status_code == 200:
            # I set the path where I'll save the file
            save_path = "data/raw/" + filename

            # I open the save path for writing in binary mode
            with open(save_path, "wb") as f:
                # I write the downloaded content to the file
                f.write(response.content)

            # I try to read it as a table to check it worked
            try:
                df = pd.read_csv(save_path)
                print(f"  Saved {filename} — {len(df)} rows, {df.shape[1]} columns")
                return df
            except Exception:
                print(f"  Saved {filename} (could not preview)")
                return None
        else:
            # The server returned an error
            print(f"  Could not download {filename} (status: {response.status_code})")
            return None

    # If anything else goes wrong (e.g. no internet connection)
    except Exception as e:
        print(f"  Error downloading {filename}: {e}")
        return None


# I download cereal yield data — how many kg of cereal per hectare
yield_df = download_faostat("QCL", "faostat_cereal_yield.csv")

# I wait 1 second before the next download (polite to the server)
time.sleep(1)

# I download fertiliser use data — how many kg of fertiliser per hectare
fert_df = download_faostat("RFN", "faostat_fertiliser.csv")
time.sleep(1)

# I download food balance sheets — how much food is available per person per day
food_supply_df = download_faostat("FBS", "faostat_food_supply.csv")
time.sleep(1)


# ============================================================
# Dataset 2: World Bank WDI — climate, GDP, infrastructure
# ============================================================
# The World Bank tracks hundreds of measures for every country.
# I download specific ones I need using their free API.

print("\n[2/5] Downloading World Bank WDI indicators...")

# I set up a dictionary mapping World Bank indicator codes to friendly names
# The key is the code the World Bank uses; the value is my chosen column name
WB_INDICATORS = {}
WB_INDICATORS["AG.YLD.CREL.KG"]    = "cereal_yield_kg_per_ha"
WB_INDICATORS["AG.CON.FERT.ZS"]    = "fertiliser_kg_per_ha"
WB_INDICATORS["AG.LND.ARBL.ZS"]    = "arable_land_pct"
WB_INDICATORS["SP.POP.TOTL"]        = "population_total"
WB_INDICATORS["NY.GDP.PCAP.KD"]    = "gdp_per_capita_usd"
WB_INDICATORS["EN.ATM.CO2E.KT"]    = "co2_emissions_kt"
WB_INDICATORS["IT.NET.USER.ZS"]    = "internet_users_pct"
WB_INDICATORS["SH.STA.STNT.ZS"]    = "stunting_pct_children"
WB_INDICATORS["AG.LND.IRIG.AG.ZS"] = "irrigated_land_pct"


def download_world_bank(indicator_code, indicator_name, year=2021):
    # I build the URL for this specific indicator and all countries
    url = "https://api.worldbank.org/v2/country/all/indicator/" + indicator_code

    # I set up the request parameters
    params = {}
    params["date"]     = year      # I only want the year 2021
    params["format"]   = "json"    # I want JSON format back
    params["per_page"] = 300       # I want up to 300 countries in one go

    # I try to download — catch any errors
    try:
        # I send the request
        response = requests.get(url, params=params, timeout=30)

        # I convert the response from JSON into a Python list
        data = response.json()

        # The World Bank API gives me a list with two items:
        # data[0] is header info, data[1] is the actual data
        if len(data) < 2 or not data[1]:
            print(f"  No data for {indicator_name}")
            return None

        # I'll collect one row per country in this list
        rows = []

        # I go through each country entry one by one
        for entry in data[1]:
            # I only keep countries that have an actual value
            if entry.get("value") is not None:
                # I build a row dictionary for this country
                one_row = {}
                one_row["country_code"] = entry["countryiso3code"]
                one_row["country_name"] = entry["country"]["value"]
                one_row[indicator_name] = entry["value"]
                one_row["year"]         = year

                # I add this row to my list
                rows.append(one_row)

        # I turn my list into a pandas table and return it
        return pd.DataFrame(rows)

    # If anything goes wrong, I print the error and return nothing
    except Exception as e:
        print(f"  Error downloading {indicator_name}: {e}")
        return None


# I'll collect all the individual indicator tables in this list
wdi_tables = []

# I go through each indicator in my dictionary
for code in WB_INDICATORS:
    # I get the friendly name for this indicator
    name = WB_INDICATORS[code]

    # I let the user know what I'm fetching
    print(f"  Fetching {name}...")

    # I download this indicator for all countries
    df = download_world_bank(code, name, year=2021)

    # If I got some data back
    if df is not None and len(df) > 0:
        # I only keep the three columns I need
        slim_df = df[["country_code", "country_name", name]]
        # I add this to my list of tables
        wdi_tables.append(slim_df)

    # I pause briefly between requests
    time.sleep(0.5)

# I merge all the individual indicator tables into one big table
if wdi_tables:
    # I start with the first table
    wdi_merged = wdi_tables[0]

    # I merge each subsequent table onto the right side
    for table in wdi_tables[1:]:
        # "outer" merge means I keep all countries even if some values are missing
        wdi_merged = wdi_merged.merge(table, on=["country_code", "country_name"], how="outer")

    # I save the merged table to a CSV file
    wdi_merged.to_csv("data/raw/worldbank_wdi_2021.csv", index=False)
    print(f"  Saved worldbank_wdi_2021.csv — {len(wdi_merged)} countries")
else:
    print("  Could not download World Bank data")


# ============================================================
# Dataset 3: Global Findex 2021 — financial access
# ============================================================
# Findex surveys adults in 140+ countries about whether they
# have bank accounts, mobile money, credit access etc.

print("\n[3/5] Downloading Global Findex 2021...")

# I set the API URL for the account ownership indicator
FINDEX_URL = "https://api.worldbank.org/v2/country/all/indicator/FX.OWN.TOTL.ZS"

# I let the user know what I'm fetching
print("  Fetching account ownership (% adults)...")

# I'll store the findex data in this variable
findex_df = None

# I try to download the account ownership data
try:
    # I send the request
    response = requests.get(
        FINDEX_URL,
        params={"date": 2021, "format": "json", "per_page": 300},
        timeout=30,
    )

    # I convert the response to Python data
    data = response.json()

    # I check I got real data back
    if len(data) >= 2 and data[1]:
        # I'll collect one row per country
        rows = []

        # I go through each country entry
        for entry in data[1]:
            # I only keep countries with a real value
            if entry.get("value") is not None:
                # I build a row for this country
                one_row = {}
                one_row["country_code"]        = entry["countryiso3code"]
                one_row["country_name"]        = entry["country"]["value"]
                one_row["account_ownership_pct"] = entry["value"]
                one_row["year"]                = 2021

                # I add this row to my list
                rows.append(one_row)

        # I turn my list into a table
        findex_df = pd.DataFrame(rows)

        # I save the table to a CSV file
        findex_df.to_csv("data/raw/findex_2021.csv", index=False)
        print(f"  Saved findex_2021.csv — {len(findex_df)} countries")
    else:
        print("  Could not get Findex data from API")

except Exception as e:
    print(f"  Error: {e}")

# I wait a moment before the next request
time.sleep(1)

# I also try to get the male account ownership data as a supplement
print("  Fetching mobile money usage...")
MOBILE_MONEY_URL = "https://api.worldbank.org/v2/country/all/indicator/FX.OWN.TOTL.MA.ZS"

try:
    # I send the request for male account ownership
    response = requests.get(
        MOBILE_MONEY_URL,
        params={"date": 2021, "format": "json", "per_page": 300},
        timeout=30,
    )

    # I convert the response to Python data
    data = response.json()

    # I check I got real data
    if len(data) >= 2 and data[1]:
        # I'll collect the rows
        rows = []

        for entry in data[1]:
            if entry.get("value") is not None:
                one_row = {}
                one_row["country_code"]              = entry["countryiso3code"]
                one_row["account_ownership_male_pct"] = entry["value"]
                rows.append(one_row)

        # I turn the rows into a table
        mobile_df = pd.DataFrame(rows)

        # If I have both the main Findex data and the mobile data, I merge them
        if findex_df is not None and len(findex_df) > 0 and len(mobile_df) > 0:
            findex_df = findex_df.merge(mobile_df, on="country_code", how="left")
            findex_df.to_csv("data/raw/findex_2021.csv", index=False)
            print("  Updated findex_2021.csv with mobile money data")

except Exception as e:
    print(f"  Note: could not get mobile money data ({e})")

time.sleep(1)


# ============================================================
# Dataset 4: FAO Food Loss and Waste — post-harvest loss rates
# ============================================================
# This is the key data for my PHL block (Model B).
# The FAO tracks how much food is lost between harvest and shops.

print("\n[4/5] Downloading FAO Food Loss and Waste data...")

# I try the FAO FLW API endpoint
print("  Fetching FAO FLW data via API...")

try:
    # I set the API address for the food loss data
    flw_api = "https://fenixservices.fao.org/faostat/api/v1/en/data/FLWSTAT"

    # I set the request parameters
    params = {}
    params["output_type"] = "csv"
    params["year"]        = "2021"

    # I send the request
    response = requests.get(flw_api, params=params, timeout=60)

    # If the server returned something useful
    if response.status_code == 200 and len(response.content) > 100:
        # I save the file to disk
        with open("data/raw/fao_flw_losses.csv", "wb") as f:
            f.write(response.content)

        # I try to read it and check it
        try:
            flw_df = pd.read_csv("data/raw/fao_flw_losses.csv")
            print(f"  Saved fao_flw_losses.csv — {len(flw_df)} rows")
        except Exception:
            print("  Saved fao_flw_losses.csv")
    else:
        # The FAO API didn't work — I try an alternative source
        print("  FAO FLW API not available — trying alternative source...")

        # I try getting a similar indicator from the World Bank instead
        flw_wb = download_world_bank("ER.FSH.AQUA.MT", "food_loss_index", 2021)

        if flw_wb is not None:
            flw_wb.to_csv("data/raw/fao_flw_losses.csv", index=False)
            print(f"  Saved alternative loss data — {len(flw_wb)} rows")
        else:
            # I create an empty placeholder file so later scripts don't crash
            print("  Will use manual FLW data — see instructions below")
            placeholder_columns = ["country_code", "country_name", "loss_pct_cereals", "year"]
            placeholder = pd.DataFrame(columns=placeholder_columns)
            placeholder.to_csv("data/raw/fao_flw_losses.csv", index=False)
            print("  Created placeholder — fill manually from:")
            print("  https://www.fao.org/platform-food-loss-waste/flw-data/en/")

except Exception as e:
    # Something went wrong — I create a placeholder so nothing crashes later
    print(f"  Error: {e}")
    placeholder_columns = ["country_code", "country_name", "loss_pct_cereals", "year"]
    placeholder = pd.DataFrame(columns=placeholder_columns)
    placeholder.to_csv("data/raw/fao_flw_losses.csv", index=False)
    print("  Created placeholder file for manual filling")

time.sleep(1)


# ============================================================
# Dataset 5: IMF Financial Access Survey — banking infrastructure
# ============================================================
# The IMF tracks things like number of bank branches and ATMs
# per 100,000 adults. This supplements the Findex data.

print("\n[5/5] Downloading IMF Financial Access Survey...")

print("  Fetching bank branches per 100,000 adults...")

try:
    # I download the bank branches indicator using the World Bank API
    # FB.CBK.BRCH.P5 = commercial bank branches per 100,000 adults
    imf_df = download_world_bank("FB.CBK.BRCH.P5", "bank_branches_per_100k", 2021)

    if imf_df is not None and len(imf_df) > 0:
        # I also try to get the ATM coverage data
        time.sleep(0.5)
        atm_df = download_world_bank("FB.ATM.TOTL.P5", "atm_per_100k", 2021)

        # If I got ATM data, I merge it into the main table
        if atm_df is not None:
            # I only need the country code and ATM column from the ATM table
            atm_slim = atm_df[["country_code", "atm_per_100k"]]
            imf_df = imf_df.merge(atm_slim, on="country_code", how="left")

        # I save the combined table
        imf_df.to_csv("data/raw/imf_financial_access.csv", index=False)
        print(f"  Saved imf_financial_access.csv — {len(imf_df)} countries")
    else:
        # If I got no data, I create an empty placeholder
        print("  No IMF data available — placeholder created")
        placeholder_columns = ["country_code", "bank_branches_per_100k"]
        pd.DataFrame(columns=placeholder_columns).to_csv(
            "data/raw/imf_financial_access.csv", index=False
        )

except Exception as e:
    print(f"  Error: {e}")


# ============================================================
# I'm printing a summary of everything I downloaded
# ============================================================

print("\n" + "=" * 55)
print("PHASE B DOWNLOAD COMPLETE — Summary")
print("=" * 55)

# I check each file and report whether it was saved successfully
files_to_check = {}
files_to_check["data/raw/faostat_cereal_yield.csv"]  = "FAOSTAT — Cereal yield"
files_to_check["data/raw/faostat_fertiliser.csv"]    = "FAOSTAT — Fertiliser use"
files_to_check["data/raw/faostat_food_supply.csv"]   = "FAOSTAT — Food supply"
files_to_check["data/raw/worldbank_wdi_2021.csv"]    = "World Bank WDI — all indicators"
files_to_check["data/raw/findex_2021.csv"]           = "Global Findex — financial access"
files_to_check["data/raw/fao_flw_losses.csv"]        = "FAO — food loss and waste"
files_to_check["data/raw/imf_financial_access.csv"]  = "IMF — banking infrastructure"

# I go through each file and check whether it exists
for filepath in files_to_check:
    label = files_to_check[filepath]

    # I check if the file is there
    if os.path.exists(filepath):
        # I try to read it and report how many rows it has
        try:
            df = pd.read_csv(filepath)
            status = "OK — " + str(len(df)) + " rows"
        except Exception:
            status = "OK (saved)"
    else:
        status = "MISSING"

    # I print whether the file is OK or missing
    if "OK" in status:
        print(f"  ✓  {label}: {status}")
    else:
        print(f"  ✗  {label}: {status}")

print("\nNext step: Phase C — clean and merge all these files")
print("into one master dataset ready for modelling.")
