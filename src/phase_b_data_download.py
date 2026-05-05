# ============================================================
# PHASE B — Download all the datasets we need for the project
# ============================================================
#
# What this file does:
#   We found 9 topics in Phase A. Now we need real country-level
#   numbers to match each topic. This script downloads everything
#   we need from free, public sources and saves them as CSV files
#   inside the data/raw/ folder.
#
# Datasets we are downloading:
#   1. FAOSTAT  — crop yield, fertiliser use, food supply
#   2. World Bank WDI  — climate, GDP, infrastructure
#   3. Global Findex 2021  — financial access (bank accounts etc.)
#   4. FAO Food Loss & Waste  — post-harvest loss rates
#   5. IMF Financial Access Survey  — banking infrastructure
#
# We focus on the year 2021 because:
#   - Global Findex 2021 only covers that year
#   - FAO Food Loss data is most complete around 2021
#   - World Bank data is well populated for 2021
# ============================================================

import requests   # lets us download files from the internet
import pandas as pd  # lets us work with tables of data
import os          # lets us check folders and file paths
import time        # lets us pause between downloads (polite)
import zipfile     # lets us unzip files
import io          # helps us read downloaded files in memory

# Make sure the output folder exists before we try to save files
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("Starting Phase B — downloading all datasets...")
print("=" * 55)


# ============================================================
# DATASET 1: FAOSTAT — Crop yield, fertiliser, food supply
# ============================================================
# FAOSTAT is the UN Food and Agriculture Organisation's database.
# It has data on what countries grow, how much fertiliser they use,
# and how much food is available. We download it using their API.

print("\n[1/5] Downloading FAOSTAT data...")

# This is the FAOSTAT bulk download API address
FAOSTAT_BASE = "https://fenixservices.fao.org/faostat/api/v1/en/data/"

def download_faostat(dataset_code, filename, year=2021):
    """
    Downloads one table from FAOSTAT.
    dataset_code: the code FAOSTAT uses for that table
    filename: what to call the saved file
    year: which year we want (we want 2021)
    """
    url = FAOSTAT_BASE + dataset_code
    params = {
        "area_cs": "ISO3",       # use country codes like 'NGA', 'KEN'
        "year": year,            # we want 2021 only
        "output_type": "csv",    # give us a CSV file
    }
    print(f"  Fetching {filename}...")
    try:
        response = requests.get(url, params=params, timeout=60)
        if response.status_code == 200:
            # Save what we downloaded to a file
            save_path = f"data/raw/{filename}"
            with open(save_path, "wb") as f:
                f.write(response.content)
            # Try to read it as a table so we can check it
            try:
                df = pd.read_csv(save_path)
                print(f"  Saved {filename} — {len(df)} rows, {df.shape[1]} columns")
                return df
            except Exception:
                print(f"  Saved {filename} (could not preview)")
                return None
        else:
            print(f"  Could not download {filename} (status: {response.status_code})")
            return None
    except Exception as e:
        print(f"  Error downloading {filename}: {e}")
        return None

# Cereal yield — how many kg of cereal each country produces per hectare
yield_df = download_faostat("QCL", "faostat_cereal_yield.csv")
time.sleep(1)  # wait 1 second before next download (be polite to server)

# Fertiliser use — how many kg of fertiliser per hectare of cropland
fert_df = download_faostat("RFN", "faostat_fertiliser.csv")
time.sleep(1)

# Food balance sheets — how much food is available per person per day
food_supply_df = download_faostat("FBS", "faostat_food_supply.csv")
time.sleep(1)


# ============================================================
# DATASET 2: World Bank WDI — climate, GDP, infrastructure
# ============================================================
# The World Bank tracks hundreds of measures for every country.
# We download specific ones using their free API.

print("\n[2/5] Downloading World Bank WDI indicators...")

# These are the World Bank codes for the numbers we need
WB_INDICATORS = {
    "AG.YLD.CREL.KG":    "cereal_yield_kg_per_ha",        # cereal yield
    "AG.CON.FERT.ZS":    "fertiliser_kg_per_ha",          # fertiliser use
    "AG.LND.ARBL.ZS":    "arable_land_pct",               # % land that is arable
    "SP.POP.TOTL":        "population_total",              # country population
    "NY.GDP.PCAP.KD":    "gdp_per_capita_usd",            # income per person
    "EN.ATM.CO2E.KT":    "co2_emissions_kt",              # CO2 emissions
    "IT.NET.USER.ZS":    "internet_users_pct",            # % people using internet
    "SH.STA.STNT.ZS":    "stunting_pct_children",         # child stunting (nutrition)
    "AG.LND.IRIG.AG.ZS": "irrigated_land_pct",            # % cropland irrigated
}

def download_world_bank(indicator_code, indicator_name, year=2021):
    """
    Downloads one indicator for all countries from the World Bank.
    indicator_code: the World Bank's code for that measure
    indicator_name: a friendly name we give it
    """
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
    params = {
        "date": year,       # year we want
        "format": "json",   # give us JSON format
        "per_page": 300,    # get up to 300 countries at once
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        # The World Bank API returns a list with two items:
        # item [0] is info about the request, item [1] is the actual data
        if len(data) < 2 or not data[1]:
            print(f"  No data for {indicator_name}")
            return None

        rows = []
        for entry in data[1]:
            # Only keep rows where the country has a value
            if entry.get("value") is not None:
                rows.append({
                    "country_code": entry["countryiso3code"],
                    "country_name": entry["country"]["value"],
                    indicator_name: entry["value"],
                    "year": year,
                })
        return pd.DataFrame(rows)

    except Exception as e:
        print(f"  Error downloading {indicator_name}: {e}")
        return None

# Download each indicator and merge them all into one big table
wdi_tables = []
for code, name in WB_INDICATORS.items():
    print(f"  Fetching {name}...")
    df = download_world_bank(code, name, year=2021)
    if df is not None and len(df) > 0:
        wdi_tables.append(df[["country_code", "country_name", name]])
    time.sleep(0.5)  # small pause between requests

# Merge all the indicators into one table using country_code as the key
if wdi_tables:
    wdi_merged = wdi_tables[0]
    for table in wdi_tables[1:]:
        # "outer" merge means: keep all countries even if some values are missing
        wdi_merged = wdi_merged.merge(table, on=["country_code", "country_name"], how="outer")

    wdi_merged.to_csv("data/raw/worldbank_wdi_2021.csv", index=False)
    print(f"  Saved worldbank_wdi_2021.csv — {len(wdi_merged)} countries")
else:
    print("  Could not download World Bank data")


# ============================================================
# DATASET 3: Global Findex 2021 — financial access
# ============================================================
# Findex surveys adults in 140+ countries about whether they
# have bank accounts, use mobile money, can access credit etc.
# This is the key data for our Finance block (Model C).

print("\n[3/5] Downloading Global Findex 2021...")

# The World Bank hosts the Findex data as a downloadable file
FINDEX_URL = "https://api.worldbank.org/v2/country/all/indicator/FX.OWN.TOTL.ZS"

# We use the World Bank API to get account ownership rate
# FX.OWN.TOTL.ZS = % of adults (15+) with a bank or mobile money account
print("  Fetching account ownership (% adults)...")
try:
    response = requests.get(
        FINDEX_URL,
        params={"date": 2021, "format": "json", "per_page": 300},
        timeout=30,
    )
    data = response.json()
    if len(data) >= 2 and data[1]:
        rows = []
        for entry in data[1]:
            if entry.get("value") is not None:
                rows.append({
                    "country_code": entry["countryiso3code"],
                    "country_name": entry["country"]["value"],
                    "account_ownership_pct": entry["value"],
                    "year": 2021,
                })
        findex_df = pd.DataFrame(rows)
        findex_df.to_csv("data/raw/findex_2021.csv", index=False)
        print(f"  Saved findex_2021.csv — {len(findex_df)} countries")
    else:
        print("  Could not get Findex data from API")
except Exception as e:
    print(f"  Error: {e}")

time.sleep(1)

# Also get mobile money usage (important for developing countries)
print("  Fetching mobile money usage...")
MOBILE_MONEY_URL = "https://api.worldbank.org/v2/country/all/indicator/FX.OWN.TOTL.MA.ZS"
try:
    response = requests.get(
        MOBILE_MONEY_URL,
        params={"date": 2021, "format": "json", "per_page": 300},
        timeout=30,
    )
    data = response.json()
    if len(data) >= 2 and data[1]:
        rows = []
        for entry in data[1]:
            if entry.get("value") is not None:
                rows.append({
                    "country_code": entry["countryiso3code"],
                    "account_ownership_male_pct": entry["value"],
                })
        mobile_df = pd.DataFrame(rows)
        if len(findex_df) > 0 and len(mobile_df) > 0:
            findex_df = findex_df.merge(mobile_df, on="country_code", how="left")
            findex_df.to_csv("data/raw/findex_2021.csv", index=False)
            print(f"  Updated findex_2021.csv with mobile money data")
except Exception as e:
    print(f"  Note: could not get mobile money data ({e})")

time.sleep(1)


# ============================================================
# DATASET 4: FAO Food Loss & Waste — post-harvest loss rates
# ============================================================
# This is the key data for Model B (PHL block).
# The FAO tracks how much food is lost between harvest and shops.

print("\n[4/5] Downloading FAO Food Loss & Waste data...")

# The FAO FLW platform has a downloadable dataset
FLW_URL = "https://www.fao.org/platform-food-loss-waste/flw-data/en/"

# We use the FAO FLW API endpoint
print("  Fetching FAO FLW data via API...")
try:
    flw_api = "https://fenixservices.fao.org/faostat/api/v1/en/data/FLWSTAT"
    params = {
        "output_type": "csv",
        "year": "2021",
    }
    response = requests.get(flw_api, params=params, timeout=60)
    if response.status_code == 200 and len(response.content) > 100:
        with open("data/raw/fao_flw_losses.csv", "wb") as f:
            f.write(response.content)
        try:
            flw_df = pd.read_csv("data/raw/fao_flw_losses.csv")
            print(f"  Saved fao_flw_losses.csv — {len(flw_df)} rows")
        except Exception:
            print("  Saved fao_flw_losses.csv")
    else:
        # Fallback: use the World Bank version of similar data
        print("  FAO FLW API not available — trying alternative source...")
        # Use SDG 12.3.1 food loss index from World Bank
        flw_wb = download_world_bank("ER.FSH.AQUA.MT", "food_loss_index", 2021)
        if flw_wb is not None:
            flw_wb.to_csv("data/raw/fao_flw_losses.csv", index=False)
            print(f"  Saved alternative loss data — {len(flw_wb)} rows")
        else:
            print("  Will use manual FLW data — see instructions below")
            # Create a placeholder so Phase C does not crash
            placeholder = pd.DataFrame(columns=["country_code", "country_name",
                                                  "loss_pct_cereals", "year"])
            placeholder.to_csv("data/raw/fao_flw_losses.csv", index=False)
            print("  Created placeholder — fill manually from:")
            print("  https://www.fao.org/platform-food-loss-waste/flw-data/en/")
except Exception as e:
    print(f"  Error: {e}")
    placeholder = pd.DataFrame(columns=["country_code", "country_name",
                                          "loss_pct_cereals", "year"])
    placeholder.to_csv("data/raw/fao_flw_losses.csv", index=False)
    print("  Created placeholder file for manual filling")

time.sleep(1)


# ============================================================
# DATASET 5: IMF Financial Access Survey — banking infrastructure
# ============================================================
# The IMF tracks things like number of bank branches per 100,000
# adults, ATM coverage etc. Good supplement to Findex.

print("\n[5/5] Downloading IMF Financial Access Survey...")

# IMF FAS is available via the World Bank API as well
print("  Fetching bank branches per 100,000 adults...")
try:
    # FB.CBK.BRCH.P5 = commercial bank branches per 100,000 adults
    imf_df = download_world_bank("FB.CBK.BRCH.P5", "bank_branches_per_100k", 2021)
    if imf_df is not None and len(imf_df) > 0:
        # Also get ATM coverage
        time.sleep(0.5)
        atm_df = download_world_bank("FB.ATM.TOTL.P5", "atm_per_100k", 2021)
        if atm_df is not None:
            imf_df = imf_df.merge(
                atm_df[["country_code", "atm_per_100k"]],
                on="country_code", how="left"
            )
        imf_df.to_csv("data/raw/imf_financial_access.csv", index=False)
        print(f"  Saved imf_financial_access.csv — {len(imf_df)} countries")
    else:
        print("  No IMF data available — placeholder created")
        pd.DataFrame(columns=["country_code", "bank_branches_per_100k"]).to_csv(
            "data/raw/imf_financial_access.csv", index=False
        )
except Exception as e:
    print(f"  Error: {e}")


# ============================================================
# SUMMARY — Check what we downloaded
# ============================================================

print("\n" + "=" * 55)
print("PHASE B DOWNLOAD COMPLETE — Summary")
print("=" * 55)

files_to_check = {
    "data/raw/faostat_cereal_yield.csv":  "FAOSTAT — Cereal yield",
    "data/raw/faostat_fertiliser.csv":    "FAOSTAT — Fertiliser use",
    "data/raw/faostat_food_supply.csv":   "FAOSTAT — Food supply",
    "data/raw/worldbank_wdi_2021.csv":    "World Bank WDI — all indicators",
    "data/raw/findex_2021.csv":           "Global Findex — financial access",
    "data/raw/fao_flw_losses.csv":        "FAO — food loss & waste",
    "data/raw/imf_financial_access.csv":  "IMF — banking infrastructure",
}

for filepath, label in files_to_check.items():
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            status = f"OK — {len(df)} rows"
        except Exception:
            status = "OK (saved)"
    else:
        status = "MISSING"
    print(f"  {'✓' if 'OK' in status else '✗'}  {label}: {status}")

print("\nNext step: Phase C — clean and merge all these files")
print("into one master dataset ready for modelling.")
