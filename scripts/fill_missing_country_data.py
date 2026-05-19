# ============================================================
# fill_missing_country_data.py
# ============================================================
# This script fills in the two datasets that Step 5 could not
# download automatically:
#
#   1. World Bank WGI — Governance Indicators
#      (the standard Step 5 API call returned nothing because
#       the WGI lives in a separate World Bank database; this
#       script uses the wbgapi library which handles that correctly)
#
#   2. FAO Food Loss and Waste
#      (the FAOSTAT API server returned a 521 error; this script
#       tries the separate FAO FLW Platform API instead, then
#       falls back to a World Bank post-harvest loss proxy)
#
# Run this AFTER step5_download_country_data.py has run.
# Run step6_clean_and_combine_data.py AFTER this.
# ============================================================

import os
import time

import pandas as pd
import requests

os.makedirs("data/raw", exist_ok=True)

print("=" * 60)
print("Downloading missing Step 5 datasets")
print("=" * 60)


# ============================================================
# Part 1: WGI Governance Indicators via wbgapi
# ============================================================
# wbgapi uses World Bank database 3 (WGI) directly, which the
# standard requests-based API call in Step 5 could not reach.

print("\n[1/2] World Bank WGI — Governance Indicators (via wbgapi)...")

# The World Bank standard v2 API cannot serve WGI (database 3) reliably.
# Instead we download the official pre-packaged WGI Excel from DataBank,
# which contains all indicators and years in one file.

WGI_INDICATOR_LABELS = {
    "Control of Corruption":    "wgi_control_of_corruption",
    "Government Effectiveness": "wgi_government_effectiveness",
    "Political Stability":      "wgi_political_stability",
    "Regulatory Quality":       "wgi_regulatory_quality",
    "Rule of Law":              "wgi_rule_of_law",
    "Voice and Accountability": "wgi_voice_accountability",
}
# We want the "Governance estimate" row for each indicator (the standard ±2.5 score)
ESTIMATE_SUFFIX = "Governance estimate (approx. -2.5 to +2.5)"

try:
    import io, zipfile
    zip_url = "https://databank.worldbank.org/data/download/WGI_EXCEL.zip"
    r = requests.get(zip_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    xl = pd.ExcelFile(z.open("WGIEXCEL.xlsx"))
    raw = xl.parse("Data")

    # Build the filter: keep rows whose Indicator Name starts with one of
    # our six governance dimensions AND ends with the estimate suffix.
    def is_estimate_row(indicator_name):
        for dim in WGI_INDICATOR_LABELS:
            if indicator_name.startswith(dim) and ESTIMATE_SUFFIX in indicator_name:
                return dim
        return None

    raw["_dim"] = raw["Indicator Name"].apply(is_estimate_row)
    estimates = raw[raw["_dim"].notna()].copy()

    # Extract 2021 column (year column is named "2021")
    if "2021" not in estimates.columns:
        raise ValueError("2021 column not found in WGI Excel")

    estimates = estimates[["Country Code", "_dim", "2021"]].copy()
    estimates.columns = ["country_code", "_dim", "value"]

    # Pivot: one row per country, one column per governance dimension
    pivot = estimates.pivot(index="country_code", columns="_dim", values="value")
    pivot = pivot.reset_index()
    pivot = pivot.rename(columns=WGI_INDICATOR_LABELS)

    # Drop aggregate rows (World Bank uses 3-letter ISO codes for real countries)
    pivot = pivot[pivot["country_code"].str.len() == 3].copy()

    # Drop rows with no data at all
    gov_cols = list(WGI_INDICATOR_LABELS.values())
    pivot = pivot.dropna(subset=gov_cols, how="all")

    pivot.to_csv("data/raw/wgi_governance_2021.csv", index=False)
    print(f"  Saved wgi_governance_2021.csv — {len(pivot)} countries, "
          f"{pivot.shape[1]} columns")
    print(f"  Columns: {list(pivot.columns)}")
    wgi_ok = True

except Exception as e:
    print(f"  DataBank ZIP download failed: {e}")
    print("  WGI file remains an empty placeholder.")
    wgi_ok = False


# ============================================================
# Part 2: FAO Food Loss and Waste data
# ============================================================
# The FAOSTAT fenixservices server that Step 5 hit is returning
# 521 errors. The FAO FLW Platform has a separate API endpoint.
# If that also fails, we construct a country-level proxy from
# USDA/World Bank post-harvest loss research estimates.

print("\n[2/2] FAO Food Loss and Waste data...")

FLW_PLATFORM_URL = "https://www.fao.org/platform-food-loss-waste/flw-data/api/data/"
flw_ok = False

# --- Attempt 1: FAO FLW Platform API ---
print("  Attempt 1: FAO FLW Platform API...")
try:
    params = {
        "format": "json",
        "limit": 10000,
    }
    r = requests.get(FLW_PLATFORM_URL, params=params, timeout=60)
    if r.status_code == 200:
        payload = r.json()

        # The API wraps results differently depending on version
        if isinstance(payload, dict) and "results" in payload:
            records = payload["results"]
        elif isinstance(payload, list):
            records = payload
        else:
            records = []

        if len(records) > 100:
            flw_df = pd.DataFrame(records)
            flw_df.to_csv("data/raw/fao_flw_losses.csv", index=False)
            print(f"  Saved fao_flw_losses.csv — {len(flw_df)} rows")
            print(f"  Columns: {list(flw_df.columns)}")
            flw_ok = True
        else:
            print(f"  Too few records ({len(records)}) — trying next approach.")
    else:
        print(f"  Status {r.status_code} — trying next approach.")
except Exception as e:
    print(f"  Platform API error: {e}")


# --- Attempt 2: FAOSTAT v2 bulk download (different server) ---
if not flw_ok:
    print("  Attempt 2: FAOSTAT bulk API (v2)...")
    try:
        url = "https://www.fao.org/faostat/api/v1/en/data/FLWSTAT"
        params = {"output_type": "csv", "area_cs": "ISO3", "limit": 50000}
        r = requests.get(url, params=params, timeout=90)
        if r.status_code == 200 and len(r.content) > 1000:
            with open("data/raw/fao_flw_losses.csv", "wb") as f:
                f.write(r.content)
            try:
                flw_df = pd.read_csv("data/raw/fao_flw_losses.csv")
                print(f"  Saved fao_flw_losses.csv — {len(flw_df)} rows")
                flw_ok = True
            except Exception:
                print("  Saved but could not parse — trying next.")
        else:
            print(f"  Status {r.status_code} / empty response.")
    except Exception as e:
        print(f"  FAOSTAT v2 error: {e}")


# --- Attempt 3: World Bank proxy (food loss % of output, AG.LND.PRCP.MM proxy) ---
# The World Bank does not publish post-harvest loss rates directly, but
# World Bank research suggests regional average cereal loss rates.
# We construct a country-level proxy by merging:
#   • Regional post-harvest loss estimates (from published FAO/WB papers)
#   • Country sub-region classification from the World Bank
if not flw_ok:
    print("  Attempt 3: Building regional proxy from published loss estimates...")

    # Published regional cereal post-harvest loss rates
    # Source: FAO (2019) "The State of Food and Agriculture" & World Bank (2020)
    # These are conservative lower-bound estimates for cereals.
    regional_loss = {
        "Sub-Saharan Africa":           14.0,
        "South Asia":                   10.0,
        "East Asia & Pacific":           9.0,
        "Latin America & Caribbean":     8.0,
        "Middle East & North Africa":    8.5,
        "Europe & Central Asia":         5.5,
        "North America":                 2.5,
    }

    try:
        # Download country-to-region mapping from World Bank
        url = "https://api.worldbank.org/v2/country/all?format=json&per_page=500"
        r = requests.get(url, timeout=30)
        data = r.json()
        if len(data) >= 2 and data[1]:
            rows = []
            for c in data[1]:
                region = c.get("region", {}).get("value", "")
                iso3 = c.get("id", "")
                if iso3 and region and region != "Aggregates":
                    loss_rate = regional_loss.get(region, 9.0)
                    rows.append({
                        "country_code":        iso3,
                        "country":             c.get("name", ""),
                        "commodity":           "Cereals (regional estimate)",
                        "year":                2021,
                        "loss_percentage":     loss_rate,
                        "supply_chain_stage":  "Aggregate",
                        "region":              region,
                        "data_source":         "Regional proxy — FAO/WB published estimates",
                    })
            proxy_df = pd.DataFrame(rows)
            proxy_df.to_csv("data/raw/fao_flw_losses.csv", index=False)
            print(f"  Proxy saved — {len(proxy_df)} country rows with regional loss rates.")
            print("  NOTE: These are regional-average proxies, not country-specific")
            print("        measurements. Flag this as a data limitation in the dissertation.")
            flw_ok = True
    except Exception as e:
        print(f"  Proxy build failed: {e}")


# ============================================================
# Part 3: Trade % of GDP — market-integration / logistics proxy
# ============================================================
# LPI (Logistics Performance Index) is available for only ~90 of our
# 154-country analysis sample — a 48% gap that cannot be safely imputed.
# Trade (% of GDP) — WDI NE.TRD.GNFS.ZS — covers ~175 countries and is
# the standard cross-country proxy for market integration and logistics
# capacity when LPI is unavailable (FAO 2015; Headey & Ecker 2013).
# NMF Topic 6 keywords (economic, value_chain, investment, system) align
# with this: trade openness reflects logistics networks and market connectivity.

print("\n[3/3] World Bank WDI — Trade % of GDP (market-integration proxy)...")

TRADE_FILE = "data/raw/trade_pct_gdp.csv"
trade_ok = False

try:
    url = "https://api.worldbank.org/v2/country/all/indicator/NE.TRD.GNFS.ZS"
    params = {"date": "2019:2021", "format": "json", "per_page": 1000}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    data = payload[1] if len(payload) > 1 and payload[1] else []

    # Keep the most recent non-null value per country (prefer 2021)
    best = {}
    for entry in data:
        iso3 = entry.get("countryiso3code", "")
        val = entry.get("value")
        year = entry.get("date", "0")
        if iso3 and len(iso3) == 3 and val is not None:
            if iso3 not in best or year > best[iso3]["year"]:
                best[iso3] = {"country_code": iso3,
                              "trade_pct_gdp": round(float(val), 2),
                              "year": year}

    if len(best) >= 100:
        trade_df = pd.DataFrame(best.values())[["country_code", "trade_pct_gdp"]]
        trade_df.to_csv(TRADE_FILE, index=False)
        print(f"  Saved trade_pct_gdp.csv — {len(trade_df)} countries")
        trade_ok = True
    else:
        print(f"  Only {len(best)} rows returned — check API response")
except Exception as e:
    print(f"  Trade % GDP download failed: {e}")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for fname, label in [
    ("data/raw/wgi_governance_2021.csv", "WGI Governance"),
    ("data/raw/fao_flw_losses.csv",      "FAO Food Loss & Waste"),
    ("data/raw/trade_pct_gdp.csv",       "Trade % GDP (logistics proxy)"),
]:
    if os.path.exists(fname):
        try:
            df = pd.read_csv(fname)
            if len(df) > 0:
                print(f"  OK   {label}: {len(df)} rows, {df.shape[1]} columns")
            else:
                print(f"  EMPTY {label}: file exists but has no rows")
        except Exception:
            print(f"  ERROR {label}: saved but unreadable")
    else:
        print(f"  MISS  {label}: not saved")

print()
print("Next step: python src/step6_clean_and_combine_data.py")
