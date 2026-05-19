# parse_findex_manual_download.py
#
# PURPOSE:
#   Three Findex 2021 variables cannot be downloaded from the World Bank API:
#     - account_ownership_rural_pct   (% adults in rural areas with a financial account)
#     - agri_payments_digital_pct     (% adults who received agricultural payments digitally)
#     - borrowed_from_bank_pct        (% adults who borrowed from a financial institution)
#
#   This script reads the manually downloaded Global Findex 2021 Excel file,
#   extracts those three columns, and adds them to data/raw/findex_2021.csv.
#
# HOW TO USE:
#   1. Go to: https://www.worldbank.org/en/publication/globalfindex
#   2. Download the "Findex 2021 Data" Excel file (usually named
#      "Global Findex Database 2021 - Data.xlsx" or similar)
#   3. Save it to: data/raw/findex_2021_manual.xlsx
#   4. Run this script: python scripts/parse_findex_manual_download.py
#   5. It will update data/raw/findex_2021.csv with the three new columns
#
# WHAT THE SCRIPT DOES:
#   - Opens the Excel file and looks for the indicators by their Findex indicator codes
#   - Maps country names to ISO3 codes using pycountry
#   - Merges the new columns into the existing findex_2021.csv
#

import os
import sys
import pandas as pd
import pycountry

# ── File paths ────────────────────────────────────────────────────────
EXCEL_FILE  = os.path.join('data', 'raw', 'findex_2021_manual.xlsx')
FINDEX_CSV  = os.path.join('data', 'raw', 'findex_2021.csv')

# ── Findex indicator codes for the three missing variables ────────────
# These are the standard Findex 2021 indicator IDs used in the Excel file
INDICATOR_MAP = {
    'account_ownership_rural_pct':  'FX.OWN.TOTL.RU.ZS',
    'agri_payments_digital_pct':    'FX.TRN.AGRI.ZS',
    'borrowed_from_bank_pct':       'FX.TRN.BORR.ZS',
}

# ── Helper: convert country name to ISO3 code ─────────────────────────
def name_to_iso3(country_name):
    """Try several strategies to find an ISO3 code for a country name."""
    if pd.isna(country_name):
        return None

    name_str = str(country_name).strip()

    # Strategy 1: exact match
    try:
        result = pycountry.countries.lookup(name_str)
        return result.alpha_3
    except LookupError:
        pass

    # Strategy 2: search by common name (handles "Iran, Islamic Rep." etc.)
    search_results = pycountry.countries.search_fuzzy(name_str)
    if search_results:
        return search_results[0].alpha_3

    return None


# ── Check input files exist ───────────────────────────────────────────
if not os.path.exists(EXCEL_FILE):
    print('ERROR: Excel file not found at:', EXCEL_FILE)
    print('')
    print('Please download the Global Findex 2021 data from:')
    print('  https://www.worldbank.org/en/publication/globalfindex')
    print('Then save it as:', EXCEL_FILE)
    sys.exit(1)

if not os.path.exists(FINDEX_CSV):
    print('ERROR: findex_2021.csv not found at:', FINDEX_CSV)
    print('Please run step5_download_country_data.py first.')
    sys.exit(1)

# ── Load the Excel file ───────────────────────────────────────────────
print('Loading Excel file:', EXCEL_FILE)

# Try reading the first sheet; adjust sheet_name if needed
try:
    excel_df = pd.read_excel(EXCEL_FILE, sheet_name=0)
    print('  Loaded sheet 0 —', excel_df.shape[0], 'rows,', excel_df.shape[1], 'columns')
except Exception as e:
    print('ERROR reading Excel file:', e)
    sys.exit(1)

# Print the first few column names so you can check the structure
print('  First 10 column names:', list(excel_df.columns[:10]))

# ── Find the indicator column and country column ──────────────────────
# The Findex Excel typically has columns: Country, CountryCode, Indicator, IndicatorCode, Value
# We need to find the right column names

country_col   = None
iso_col       = None
indicator_col = None
value_col     = None

# Look for columns by common names
for col in excel_df.columns:
    col_lower = str(col).lower().strip()
    if 'country' in col_lower and 'code' in col_lower and iso_col is None:
        iso_col = col
    elif 'country' in col_lower and country_col is None:
        country_col = col
    elif 'indicator' in col_lower and 'code' in col_lower and indicator_col is None:
        indicator_col = col
    elif col_lower in ('value', '2021', 'data') and value_col is None:
        value_col = col

print('  Detected columns:')
print('    country_col:  ', country_col)
print('    iso_col:      ', iso_col)
print('    indicator_col:', indicator_col)
print('    value_col:    ', value_col)

if indicator_col is None or value_col is None:
    print('')
    print('WARNING: Could not automatically detect column names.')
    print('Please check the Excel file structure and update this script if needed.')
    print('All column names in the file are:')
    for col in excel_df.columns:
        print(' ', col)
    sys.exit(1)

# ── Extract each of the three target indicators ───────────────────────
new_columns = {}  # will hold: column_name → dict of {iso3: value}

for output_col, indicator_code in INDICATOR_MAP.items():
    print('')
    print('Extracting:', output_col, '(indicator code:', indicator_code + ')')

    # Filter rows for this indicator
    mask = excel_df[indicator_col].astype(str).str.strip() == indicator_code
    filtered = excel_df[mask].copy()

    if len(filtered) == 0:
        print('  WARNING: No rows found for indicator code:', indicator_code)
        print('  Skipping this variable.')
        continue

    print('  Found', len(filtered), 'rows')

    # Build a dict of iso3 → value
    iso_value_map = {}

    for index, row in filtered.iterrows():
        # Get ISO3 code
        if iso_col is not None and pd.notna(row[iso_col]):
            iso3 = str(row[iso_col]).strip().upper()
            # Only keep valid 3-letter codes
            if len(iso3) == 3:
                pass  # iso3 is good
            else:
                iso3 = None
        else:
            iso3 = None

        # If no ISO3 from the iso_col, try converting from country name
        if iso3 is None and country_col is not None:
            iso3 = name_to_iso3(row[country_col])

        if iso3 is None:
            continue

        # Get the numeric value
        raw_value = row[value_col]
        try:
            numeric_value = float(raw_value)
        except (ValueError, TypeError):
            continue

        iso_value_map[iso3] = round(numeric_value, 2)

    print('  Successfully mapped', len(iso_value_map), 'countries')
    new_columns[output_col] = iso_value_map


# ── Merge into existing findex_2021.csv ──────────────────────────────
print('')
print('Loading existing Findex CSV:', FINDEX_CSV)
findex_df = pd.read_csv(FINDEX_CSV)
print('  Existing shape:', findex_df.shape)

for col_name, iso_value_map in new_columns.items():
    values_series = []
    matched_count = 0

    for iso3 in findex_df['country_code']:
        if iso3 in iso_value_map:
            values_series.append(iso_value_map[iso3])
            matched_count += 1
        else:
            values_series.append(None)

    findex_df[col_name] = values_series
    print(f'  Added column "{col_name}": {matched_count} countries matched')

# ── Save updated file ─────────────────────────────────────────────────
findex_df.to_csv(FINDEX_CSV, index=False)
print('')
print('Saved updated findex_2021.csv — shape:', findex_df.shape)
print('Columns now in file:')
for col in findex_df.columns:
    n_filled = findex_df[col].notna().sum()
    print(f'  {col:<45} {n_filled} countries')
