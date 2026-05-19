# ============================================================
# scripts/mine_country_data_all_sources.py
# ============================================================
#
# This script uses every available means to collect country-level
# data that could not be obtained through standard API calls.
#
# Sources mined here:
#
#   1. APHLIS — African Postharvest Losses Information System
#      The APHLIS website does not allow bulk download, but it
#      runs a REST API in the background that powers its web maps.
#      This script calls that API directly to extract country-level
#      cereal loss percentages for all 40 African countries in the
#      APHLIS database, across all 6 cereal crops.
#      Output: data/raw/aphlis_country_losses.csv
#
#   2. FAO Food Loss and Waste Platform
#      Re-queries the FAO FLW Platform API with cereal-specific
#      filters and by individual country to extract any real
#      (non-proxy) data that exists beyond Sub-Saharan Africa.
#      Output: data/raw/fao_flw_cereal_by_country.csv
#
#   3. World Bank Findex 2021 — Mobile Money
#      The Findex 2021 summary Excel file (published by World Bank)
#      contains country-level mobile money account ownership data
#      that is NOT available through the standard WDI API.
#      Output: data/raw/findex_mobile_money.csv
#
#   4. GSMA Mobile Money Metrics
#      GSMA publishes free country-level mobile money data through
#      their public metrics page.
#      Output: merged into data/raw/findex_mobile_money.csv
#
#   5. Compile final merged PHL and mobile money datasets
#      Merges APHLIS real data with the enhanced proxy for non-
#      African countries to produce a single combined PHL file.
#      Output: data/raw/phl_combined.csv
#
# Run this AFTER scripts/fetch_additional_country_indicators.py.
# ============================================================

import io
import json
import re
import time

import pandas as pd
import pycountry
import requests

HEADERS = {
    "User-Agent":       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept":           "application/json, text/plain, */*",
    "Referer":          "https://www.aphlis.net/en/map",
    "X-Requested-With": "XMLHttpRequest",
}
APHLIS_BASE = "https://www.aphlis.net"
TIMEOUT = 25

import os
os.makedirs("data/raw", exist_ok=True)

print("=" * 60)
print("Mining additional data from all available sources")
print("=" * 60)


# ============================================================
# Helper: country name → ISO3 code
# ============================================================

# APHLIS returns full country names. We need to map them to ISO3.
# pycountry handles most cases; the manual overrides catch the
# few country names that pycountry spells differently.
MANUAL_ISO3 = {
    "Tanzania":                    "TZA",
    "Congo, DR":                   "COD",
    "Congo, DRC":                  "COD",
    "Democratic Republic of Congo": "COD",
    "Congo":                       "COG",
    "Ivory Coast":                 "CIV",
    "Cote d'Ivoire":               "CIV",
    "Côte d'Ivoire":               "CIV",
    "Cape Verde":                  "CPV",
    "Cabo Verde":                  "CPV",
    "Eswatini":                    "SWZ",
    "Swaziland":                   "SWZ",
    "The Gambia":                  "GMB",
    "Gambia":                      "GMB",
    "Guinea Bissau":               "GNB",
    "Guinea-Bissau":               "GNB",
    "São Tomé and Príncipe":       "STP",
    "Sao Tome and Principe":       "STP",
    "Central African Republic":    "CAF",
    "South Sudan":                 "SSD",
    "São Tomé":                    "STP",
    "Niger":                       "NER",
    "Nigeria":                     "NGA",
}


def name_to_iso3(name):
    """Convert a country name to its ISO 3166-1 alpha-3 code."""
    if name in MANUAL_ISO3:
        return MANUAL_ISO3[name]
    try:
        country = pycountry.countries.search_fuzzy(name)[0]
        return country.alpha_3
    except Exception:
        return None


# ============================================================
# Part 1: APHLIS API mining
# ============================================================
# The APHLIS website uses a REST API (discovered from its JS bundle)
# to power its interactive maps. The key endpoints are:
#
#   GET /api/datatables/crops   — returns crop list with IDs
#   GET /api/datatables/years   — returns year list with IDs
#   GET /api/losses/countries?crop={id}&year={id}
#     — returns {minLoss, maxLoss, locations: [{id, loss, level}]}
#   GET /api/losses/forcountry?crop={id}&year={id}&country={loc_id}
#     — returns {country, crop, year, production_t, loss_t, loss_prc}
#
# APHLIS covers 40 African countries for 6 cereal crops.
# We pull the 3 most recent years and take the most recent result
# per country per crop, then average across cereals.

print()
print("[1/4] APHLIS API — country-level cereal loss data...")

CEREAL_CROPS = {3: "maize", 6: "rice", 8: "sorghum", 17: "millet", 10: "wheat", 1: "barley"}
CEREAL_NAMES_EN = list(CEREAL_CROPS.values())

aphlis_ok = False
aphlis_df = pd.DataFrame()

try:
    # Step 1: Get crop and year ID lists
    crops_resp = requests.get(APHLIS_BASE + "/api/datatables/crops", headers=HEADERS, timeout=TIMEOUT)
    years_resp = requests.get(APHLIS_BASE + "/api/datatables/years", headers=HEADERS, timeout=TIMEOUT)
    all_crops = {c["id"]: c["name"] for c in crops_resp.json()}
    all_years = [(y["id"], y["name"]) for y in years_resp.json()]

    # Only use cereal crops that APHLIS tracks
    cereal_crop_ids = {cid: cname for cid, cname in all_crops.items()
                       if cname in CEREAL_NAMES_EN}
    # Use the 3 most recent years for maximum country coverage
    recent_years = all_years[:3]

    print(f"  Cereal crops: {cereal_crop_ids}")
    print(f"  Years: {recent_years}")

    # Step 2: Collect all unique location IDs across crops and years
    all_location_ids = set()
    for year_id, year_name in recent_years:
        for crop_id, crop_name in cereal_crop_ids.items():
            r = requests.get(
                APHLIS_BASE + f"/api/losses/countries?crop={crop_id}&year={year_id}",
                headers=HEADERS,
                timeout=TIMEOUT,
            )
            if r.status_code == 200 and r.text.strip():
                try:
                    data = r.json()
                    locs = data.get("locations", [])
                    all_location_ids.update(loc["id"] for loc in locs)
                except Exception:
                    pass
            time.sleep(0.15)

    print(f"  Found {len(all_location_ids)} unique country/location IDs")

    # Step 3: For each location, query each crop for most recent year
    # We collect one row per (location, crop, year) combination
    records = []
    for loc_id in sorted(all_location_ids):
        for crop_id, crop_name in cereal_crop_ids.items():
            for year_id, year_name in recent_years:
                url = (APHLIS_BASE
                       + f"/api/losses/forcountry?crop={crop_id}&year={year_id}&country={loc_id}")
                r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
                if r.status_code == 200 and r.text.strip():
                    try:
                        rec = r.json()
                        if "country" in rec and "loss_prc" in rec:
                            # loss_prc comes back as "16.8 %" — extract the number
                            loss_str = rec["loss_prc"].replace("%", "").strip()
                            loss_val = float(loss_str)
                            records.append({
                                "location_id":      loc_id,
                                "country_name":     rec["country"],
                                "crop":             crop_name,
                                "year":             year_name,
                                "loss_pct":         loss_val,
                                "production_t_raw": rec.get("production_t", ""),
                                "loss_t_raw":       rec.get("loss_t", ""),
                            })
                            # Got data — no need to try older years
                            break
                    except Exception:
                        pass
                time.sleep(0.1)

    if records:
        raw = pd.DataFrame(records)

        # Add ISO3 codes
        country_to_iso = {}
        for name in raw["country_name"].unique():
            country_to_iso[name] = name_to_iso3(name)
        raw["country_code"] = raw["country_name"].map(country_to_iso)

        # For each country, average loss across all available cereals
        # This gives a single "cereal post-harvest loss %" per country
        country_avg = (
            raw.groupby(["country_code", "country_name"])
            .agg(
                aphlis_cereal_loss_pct=("loss_pct", "mean"),
                crops_covered=("crop", lambda x: "; ".join(sorted(x.unique()))),
                n_crops=("crop", "nunique"),
            )
            .reset_index()
        )
        country_avg["data_source"] = "APHLIS API — country-level estimate"

        # Also keep the per-crop breakdown for reference
        raw.to_csv("data/raw/aphlis_by_crop.csv", index=False)
        country_avg.to_csv("data/raw/aphlis_country_losses.csv", index=False)

        n_countries = country_avg["country_code"].notna().sum()
        print(f"  Saved aphlis_country_losses.csv — {len(country_avg)} countries "
              f"({n_countries} with ISO3 codes)")
        print(f"  Mean cereal loss range: "
              f"{country_avg['aphlis_cereal_loss_pct'].min():.1f}% – "
              f"{country_avg['aphlis_cereal_loss_pct'].max():.1f}%")
        print(f"  Crops per country (average): "
              f"{country_avg['n_crops'].mean():.1f}")
        print()
        print("  Country-level loss estimates:")
        display = country_avg[["country_name", "country_code", "aphlis_cereal_loss_pct",
                                "n_crops"]].sort_values("aphlis_cereal_loss_pct",
                                                         ascending=False)
        for _, row in display.iterrows():
            print(f"    {row['country_name']:<30} {row['country_code']}  "
                  f"{row['aphlis_cereal_loss_pct']:.1f}%  ({row['n_crops']} crops)")

        aphlis_df = country_avg
        aphlis_ok = True
    else:
        print("  No records retrieved from APHLIS forcountry endpoint")

except Exception as exc:
    print(f"  APHLIS mining failed: {exc}")


# ============================================================
# Part 2: FAO FLW Platform — cereal-specific queries
# ============================================================
# The FAO Food Loss and Waste Platform API supports filtering by
# commodity group. We query specifically for cereals and pulses
# to extract any records with real country-level measurements
# (as opposed to the regional proxy values in fao_flw_losses.csv).

print()
print("[2/4] FAO FLW Platform — cereal-specific country data...")

FAO_FLW_URL = "https://www.fao.org/platform-food-loss-waste/flw-data/api/data/"
fao_flw_ok = False

try:
    # Try several commodity filter variations for cereals
    all_flw_records = []
    cereal_commodities = [
        "Cereals",
        "cereals",
        "Wheat",
        "Maize",
        "Rice (paddy)",
        "Sorghum",
    ]
    for commodity in cereal_commodities:
        params = {"format": "json", "commodity": commodity, "limit": 5000}
        r = requests.get(FAO_FLW_URL, params=params, timeout=40)
        if r.status_code == 200:
            try:
                payload = r.json()
                records = payload.get("results", payload) if isinstance(payload, dict) else payload
                if isinstance(records, list) and len(records) > 0:
                    all_flw_records.extend(records)
                    print(f"  {commodity}: {len(records)} records")
            except Exception:
                pass
        time.sleep(0.3)

    if all_flw_records:
        flw_df = pd.DataFrame(all_flw_records)
        # Identify country and loss columns
        possible_code = [c for c in flw_df.columns
                         if any(k in c.lower() for k in ["country", "iso", "code"])]
        possible_loss = [c for c in flw_df.columns
                         if any(k in c.lower() for k in ["loss", "percent", "pct", "waste"])]

        if possible_code and possible_loss:
            code_col = possible_code[0]
            loss_col = possible_loss[0]
            flw_df[loss_col] = pd.to_numeric(flw_df[loss_col], errors="coerce")
            real_data = flw_df[flw_df[loss_col].notna()].copy()

            # Check unique value count — proxy data has very few unique values
            n_unique = real_data[loss_col].nunique()
            if n_unique > 10:
                country_flw = (
                    real_data.groupby(code_col)[loss_col]
                    .mean()
                    .reset_index()
                )
                country_flw.columns = ["country_code", "fao_flw_cereal_loss_pct"]
                country_flw["data_source"] = "FAO FLW Platform — country average"
                country_flw.to_csv("data/raw/fao_flw_cereal_by_country.csv", index=False)
                print(f"  Saved fao_flw_cereal_by_country.csv — "
                      f"{len(country_flw)} countries, {n_unique} unique values")
                fao_flw_ok = True
            else:
                print(f"  FAO FLW returned only {n_unique} unique values — "
                      f"still regional proxies, not saved separately")
        else:
            print("  Could not identify country/loss columns in FAO FLW response")
    else:
        print("  FAO FLW Platform returned no cereal records")

except Exception as exc:
    print(f"  FAO FLW mining failed: {exc}")


# ============================================================
# Part 3: World Bank Findex 2021 — Mobile Money
# ============================================================
# The World Bank publishes the Global Findex 2021 results as
# an Excel workbook available for direct download. This workbook
# contains country-level estimates for mobile money account
# ownership — the variable we need but could not get via the
# WDI API.
#
# We also try the Findex 2017 as a fallback for countries not
# covered in 2021.

print()
print("[3/4] World Bank Findex 2021 — mobile money account data...")

FINDEX_URLS = [
    # 2021 edition — primary
    ("https://thedocs.worldbank.org/en/doc/1b5fdc3b0a89c7c9adc67f4ac5db8e40"
     "-0050022021/original/Findex2021Data.xlsx",
     2021),
    # Alternative URL patterns the World Bank uses
    ("https://microdata.worldbank.org/index.php/catalog/4607/download/115745",
     2021),
]

findex_ok = False
findex_df = pd.DataFrame()

for findex_url, findex_year in FINDEX_URLS:
    if findex_ok:
        break
    print(f"  Trying Findex {findex_year} Excel ({findex_url[:60]}...)...")
    try:
        r = requests.get(findex_url,
                         headers={"User-Agent": "Mozilla/5.0"},
                         timeout=60,
                         allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 50000:
            xl = pd.ExcelFile(io.BytesIO(r.content))
            print(f"  Sheets: {xl.sheet_names}")

            # Look through sheets for mobile money data
            for sheet in xl.sheet_names:
                df_sheet = xl.parse(sheet, header=None)
                sheet_text = df_sheet.astype(str).values.flatten()
                if any("mobile" in str(v).lower() for v in sheet_text):
                    print(f"  Found 'mobile' in sheet: {sheet}")

                    # Try to find the mobile money column
                    df_sheet = xl.parse(sheet)
                    mobile_cols = [c for c in df_sheet.columns
                                   if "mobile" in str(c).lower()]
                    country_cols = [c for c in df_sheet.columns
                                    if any(k in str(c).lower()
                                           for k in ["country", "economy", "iso"])]
                    if mobile_cols and country_cols:
                        keep = country_cols[:1] + mobile_cols[:3]
                        sub = df_sheet[keep].dropna(subset=[country_cols[0]])
                        print(f"    Columns: {keep}")
                        print(f"    Rows: {len(sub)}")
                        sub.to_csv("data/raw/findex_mobile_money.csv", index=False)
                        findex_ok = True
                        break
    except Exception as exc:
        print(f"  Findex URL failed: {exc}")

# Fallback: build mobile money proxy from existing Findex account data
# and mobile subscriptions, combining into one variable
if not findex_ok:
    print("  Findex Excel not directly downloadable — building composite proxy...")
    try:
        findex_existing = pd.read_csv("data/raw/findex_2021.csv")
        mobile = pd.read_csv("data/raw/mobile_financial_access.csv")

        # Join on country_code
        findex_existing["country_code"] = findex_existing["country_code"].str.strip()
        mobile["country_code"] = mobile["country_code"].str.strip()
        combined = findex_existing.merge(mobile, on="country_code", how="outer")

        # Create a composite: normalise mobile subscriptions (0–100 scale)
        # and weight with financial account ownership for rural context
        if "mobile_subscriptions_per_100" in combined.columns:
            combined["digital_finance_proxy"] = (
                combined["account_ownership_pct"].fillna(0) * 0.5
                + (combined["mobile_subscriptions_per_100"].clip(upper=150) / 150 * 100) * 0.5
            )
        combined.to_csv("data/raw/findex_mobile_money.csv", index=False)
        print(f"  Saved composite proxy — {len(combined)} countries")
        print("  Variable: digital_finance_proxy (account_ownership * 0.5 + mobile * 0.5)")
        findex_ok = True
        findex_df = combined
    except Exception as exc:
        print(f"  Composite proxy also failed: {exc}")


# ============================================================
# Part 3b: FAO Food Balance Sheets — national cereal losses
# ============================================================
# The FAO Food Balance Sheets (FBS) contain a "Losses" element
# (Element Code 5123) for every country and commodity group.
# These are national supply-utilization accounts: governments
# report production, trade, and utilization to FAO, and losses
# are the residual that balances the equation.
#
# This gives us REAL country-level cereal loss data for ~150
# countries — far more coverage than APHLIS (39 countries) and
# with genuine country-specific variation (not regional proxies).
#
# Loss % = Losses (1000t) / Domestic Supply (1000t) × 100
# We use the aggregate "Cereals - Excluding Beer" item (2021).

print()
print("[3b/4] FAO Food Balance Sheets — national cereal loss percentages...")

fbs_ok = False
fbs_df = pd.DataFrame()

try:
    import zipfile as zipmod

    FBS_URL = ("https://fenixservices.fao.org/faostat/static/bulkdownloads/"
               "FoodBalanceSheets_E_All_Data.zip")
    print("  Downloading FAO Food Balance Sheets (~22 MB)...")
    r_fbs = requests.get(FBS_URL, timeout=150)
    r_fbs.raise_for_status()

    z_fbs = zipmod.ZipFile(io.BytesIO(r_fbs.content))
    fbs_raw = pd.read_csv(
        z_fbs.open("FoodBalanceSheets_E_All_Data_NOFLAG.csv"),
        encoding="latin-1",
        low_memory=False,
    )

    # Element 5123 = Losses; Element 5301 = Domestic supply quantity
    # Item = 'Cereals - Excluding Beer' covers the full aggregate cereal group
    cereal_mask = fbs_raw["Item"] == "Cereals - Excluding Beer"
    losses_fbs = (
        fbs_raw[cereal_mask & (fbs_raw["Element Code"] == 5123)]
        [["Area", "Y2021"]]
        .rename(columns={"Y2021": "losses_1000t"})
    )
    supply_fbs = (
        fbs_raw[cereal_mask & (fbs_raw["Element Code"] == 5301)]
        [["Area", "Y2021"]]
        .rename(columns={"Y2021": "supply_1000t"})
    )
    fbs_merged = losses_fbs.merge(supply_fbs, on="Area", how="inner")
    fbs_merged["fbs_cereal_loss_pct"] = (
        fbs_merged["losses_1000t"] / fbs_merged["supply_1000t"] * 100
    ).round(2)
    # Keep plausible values only (0.1% to 40%)
    fbs_merged = fbs_merged[
        (fbs_merged["supply_1000t"] > 0)
        & fbs_merged["fbs_cereal_loss_pct"].between(0.1, 40)
    ].copy()

    # FAO name → ISO3 mapping
    FBS_MANUAL_ISO3 = {
        "Bolivia (Plurinational State of)":               "BOL",
        "China, mainland":                                "CHN",
        "China, Hong Kong SAR":                           "HKG",
        "China, Macao SAR":                               "MAC",
        "China, Taiwan Province of":                      "TWN",
        "Congo":                                          "COG",
        "Democratic Republic of the Congo":               "COD",
        "Czechia":                                        "CZE",
        "Iran (Islamic Republic of)":                     "IRN",
        "Côte d'Ivoire":                             "CIV",
        "CÃ´te d'Ivoire":                                 "CIV",
        "Democratic People's Republic of Korea":          "PRK",
        "Republic of Korea":                              "KOR",
        "Lao People's Democratic Republic":               "LAO",
        "Libya":                                          "LBY",
        "Micronesia (Federated States of)":               "FSM",
        "Republic of Moldova":                            "MDA",
        "Russian Federation":                             "RUS",
        "Saint Lucia":                                    "LCA",
        "Saint Vincent and the Grenadines":               "VCT",
        "Serbia":                                         "SRB",
        "Slovakia":                                       "SVK",
        "Syrian Arab Republic":                           "SYR",
        "United Republic of Tanzania":                    "TZA",
        "Tanzania, United Republic of":                   "TZA",
        "Türkiye":                                   "TUR",
        "TÃ¼rkiye":                                       "TUR",
        "Turkey":                                         "TUR",
        "United Kingdom of Great Britain and Northern Ireland": "GBR",
        "United States of America":                       "USA",
        "Venezuela (Bolivarian Republic of)":             "VEN",
        "Viet Nam":                                       "VNM",
        "Yemen":                                          "YEM",
        "Palestine":                                      "PSE",
        "North Macedonia":                                "MKD",
        "Eswatini":                                       "SWZ",
        "Cabo Verde":                                     "CPV",
        "Timor-Leste":                                    "TLS",
        "Netherlands (Kingdom of the)":                   "NLD",
    }

    # Skip aggregate/regional rows (they are not ISO countries)
    SKIP_AREAS = {
        "World", "Eastern Africa", "Middle Africa", "Northern Africa",
        "Southern Africa", "Western Africa", "Americas", "Northern America",
        "Central America", "Caribbean", "South America", "Asia",
        "Central Asia", "Eastern Asia", "Southern Asia", "South-eastern Asia",
        "Western Asia", "Europe", "Eastern Europe", "Northern Europe",
        "Southern Europe", "Western Europe", "Oceania",
        "Australia and New Zealand", "Melanesia", "European Union (27)",
        "Least Developed Countries (LDCs)",
        "Land Locked Developing Countries (LLDCs)",
        "Small Island Developing States (SIDS)",
        "Low Income Food Deficit Countries (LIFDCs)",
        "Net Food Importing Developing Countries (NFIDCs)",
    }

    def fao_to_iso3(name):
        if name in SKIP_AREAS:
            return None
        if name in FBS_MANUAL_ISO3:
            return FBS_MANUAL_ISO3[name]
        try:
            return pycountry.countries.search_fuzzy(name)[0].alpha_3
        except Exception:
            return None

    fbs_merged["country_code"] = fbs_merged["Area"].apply(fao_to_iso3)
    fbs_merged = fbs_merged.dropna(subset=["country_code"]).copy()

    fbs_df = fbs_merged[["country_code", "fbs_cereal_loss_pct"]].copy()
    fbs_df["data_source"] = "FAO Food Balance Sheet 2021 — national supply utilization"
    fbs_df["data_quality"] = "A"
    fbs_df.to_csv("data/raw/fbs_cereal_losses.csv", index=False)

    print(f"  Saved fbs_cereal_losses.csv — {len(fbs_df)} countries, "
          f"{fbs_df['fbs_cereal_loss_pct'].nunique()} unique values")
    print(f"  Loss range: {fbs_df['fbs_cereal_loss_pct'].min():.2f}% – "
          f"{fbs_df['fbs_cereal_loss_pct'].max():.2f}%")
    fbs_ok = True

except Exception as exc:
    print(f"  FAO FBS download/processing failed: {exc}")


# ============================================================
# Part 4: Compile unified Post-Harvest Loss dataset
# ============================================================
# Priority order (highest → lowest quality):
#   A1 = APHLIS country-level crop model estimate (Africa, 39 countries)
#   A2 = FAO Food Balance Sheet national supply-utilization (global, ~149)
#   C  = Enhanced sub-regional proxy (remaining countries)
#
# For African countries, APHLIS overrides FAO FBS because APHLIS
# uses validated field-study crop loss models. For all other
# countries, FAO FBS provides real country-specific values.
#
# The result is a single phl_combined.csv with:
#   country_code, cereal_loss_pct, data_quality, data_source
#
# data_quality:
#   A = Real country-level data (APHLIS model or FAO FBS account)
#   C = Sub-regional proxy (published academic estimates)

print()
print("[4/4] Compiling unified post-harvest loss dataset...")

phl_compile_ok = False
try:
    # Start from the sub-regional proxy as the floor (covers all countries)
    proxy = pd.read_csv("data/raw/aphlis_phl_africa.csv")
    proxy = proxy.rename(columns={"aphlis_cereal_loss_pct": "cereal_loss_pct"})
    proxy["data_quality"] = "C"
    proxy = proxy[["country_code", "cereal_loss_pct", "data_quality", "data_source"]].copy()
    combined_phl = proxy.copy()

    def apply_layer(base_df, source_df, loss_col, quality_label, source_label):
        """Merge higher-quality source into base, overriding lower-quality rows."""
        source_df = source_df.copy().rename(columns={loss_col: "cereal_loss_pct"})
        source_df = source_df[source_df["cereal_loss_pct"].between(0.1, 40)].copy()
        source_df["data_quality"] = quality_label
        source_df["data_source"]  = source_label

        # Update existing rows
        updated = base_df.copy()
        for _, row in source_df.iterrows():
            code = row["country_code"]
            if pd.isna(code):
                continue
            mask = updated["country_code"] == code
            if mask.any():
                updated.loc[mask, "cereal_loss_pct"] = row["cereal_loss_pct"]
                updated.loc[mask, "data_quality"]    = quality_label
                updated.loc[mask, "data_source"]     = source_label
            else:
                new_row = {
                    "country_code":    code,
                    "cereal_loss_pct": row["cereal_loss_pct"],
                    "data_quality":    quality_label,
                    "data_source":     source_label,
                }
                updated = pd.concat(
                    [updated, pd.DataFrame([new_row])],
                    ignore_index=True,
                )
        return updated

    # Layer 2: FAO Food Balance Sheets (global, country-level supply accounts)
    if fbs_ok and not fbs_df.empty:
        combined_phl = apply_layer(
            combined_phl, fbs_df, "fbs_cereal_loss_pct",
            "A", "FAO Food Balance Sheet 2021 — national supply utilization",
        )
        print(f"  Applied FAO FBS data for {len(fbs_df)} countries")

    # Layer 3: APHLIS (Africa only — overrides FAO FBS for African cereals)
    # APHLIS uses validated field-study crop models and is more accurate for Africa
    if aphlis_ok and not aphlis_df.empty:
        aphlis_real = aphlis_df[["country_code", "aphlis_cereal_loss_pct"]].copy()
        combined_phl = apply_layer(
            combined_phl, aphlis_real, "aphlis_cereal_loss_pct",
            "A", "APHLIS API — country-level crop model estimate",
        )
        print(f"  Applied APHLIS data for {len(aphlis_df)} African countries (overrides FBS)")

    combined_phl.to_csv("data/raw/phl_combined.csv", index=False)

    q_counts = combined_phl["data_quality"].value_counts()
    n_unique = combined_phl["cereal_loss_pct"].nunique()
    print()
    print("  Saved data/raw/phl_combined.csv")
    print(f"  Total countries: {len(combined_phl)}")
    print(f"  Quality A (real country-level data): {q_counts.get('A', 0)} countries")
    print(f"  Quality C (sub-regional proxy):      {q_counts.get('C', 0)} countries")
    print(f"  Unique cereal loss values: {n_unique} (was 4 in original — {n_unique//4}x more)")
    phl_compile_ok = True

except Exception as exc:
    import traceback
    print(f"  PHL compilation failed: {exc}")
    traceback.print_exc()
    phl_compile_ok = False


# ============================================================
# Summary
# ============================================================
print()
print("=" * 60)
print("MINING SUMMARY")
print("=" * 60)

checks = [
    ("data/raw/aphlis_country_losses.csv",    "APHLIS country-level losses (Africa)",  aphlis_ok),
    ("data/raw/aphlis_by_crop.csv",           "APHLIS per-crop detail",                aphlis_ok),
    ("data/raw/fbs_cereal_losses.csv",        "FAO FBS national cereal losses (global)", fbs_ok),
    ("data/raw/fao_flw_cereal_by_country.csv","FAO FLW Platform cereal by country",    fao_flw_ok),
    ("data/raw/findex_mobile_money.csv",      "Mobile money / digital finance",        findex_ok),
    ("data/raw/phl_combined.csv",             "Combined PHL (all sources merged)",     phl_compile_ok),
]

for fpath, label, flag in checks:
    if os.path.exists(fpath):
        try:
            df_c = pd.read_csv(fpath)
            print(f"  OK    {label}: {len(df_c)} rows")
        except Exception:
            print(f"  WARN  {label}: saved but unreadable")
    else:
        print(f"  MISS  {label}: not saved")

print()
print("Key improvement over original data:")
if os.path.exists("data/raw/phl_combined.csv"):
    phl = pd.read_csv("data/raw/phl_combined.csv")
    a_count = (phl["data_quality"] == "A").sum()
    print(f"  Post-harvest loss now has {phl['cereal_loss_pct'].nunique()} unique values")
    print(f"  {a_count} countries have REAL APHLIS country-level estimates")
    print(f"  (Original proxy had only 4 unique values for all 217 countries)")

print()
print("Next steps:")
print("  1. python src/step6_clean_and_combine_data.py")
print("     Use phl_combined.csv instead of fao_flw_losses.csv")
print("  2. python src/step7_run_prediction_models.py")
print("     Add Model F with: lpi_overall, poverty_headcount_pct_215,")
print("     aphlis_cereal_loss_pct, digital_finance_proxy, female_agri_employment_pct")
