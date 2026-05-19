# generate_choropleth_map.py
#
# PURPOSE:
#   Create a publication-quality choropleth map showing cereal food availability
#   (kg per person per year) across all countries in the analysis sample.
#
#   The map uses the country-level data from the dashboard export file and a
#   Natural Earth world boundary GeoJSON to draw country polygons coloured by
#   availability band (Very Low → Very High).
#
# HOW TO RUN:
#   python scripts/generate_choropleth_map.py
#
# OUTPUT:
#   outputs/figures/choropleth_cereal_availability.png   (300 DPI, print quality)
#
# REQUIREMENTS:
#   geopandas, matplotlib, pandas
#   pip install geopandas
#

import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

# ── Add src/ so chart_style_settings can be imported ────────────────────────
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

from chart_style_settings import use_project_matplotlib_config
use_project_matplotlib_config()

# ── File paths ───────────────────────────────────────────────────────────────
DATA_FILE    = os.path.join(project_root, "outputs", "powerbi", "page4_country_map.csv")
GEOJSON_FILE = os.path.join(project_root, "data", "raw", "world_countries.geojson")
OUTPUT_FILE  = os.path.join(project_root, "outputs", "figures", "choropleth_cereal_availability.png")

# ── Check input files ────────────────────────────────────────────────────────
if not os.path.exists(DATA_FILE):
    print("ERROR: country data file not found:", DATA_FILE)
    print("Please run step10_export_for_dashboard.py first.")
    sys.exit(1)

if not os.path.exists(GEOJSON_FILE):
    print("ERROR: world GeoJSON not found:", GEOJSON_FILE)
    print("Please download it by running step7 or manually from:")
    print("  https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson")
    sys.exit(1)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading country availability data...")
data_df = pd.read_csv(DATA_FILE)
print("  Rows:", len(data_df))
print("  Columns:", list(data_df.columns))

# Rename so the ISO3 column is simply "iso3" for easy merging
data_df = data_df.rename(columns={"ISO3 Code": "iso3"})

# Compute actual quintile cut-points so the map annotation is accurate
dv_col = "Cereal Availability (kg/person/yr)"
q20 = data_df[dv_col].quantile(0.20)
q40 = data_df[dv_col].quantile(0.40)
q60 = data_df[dv_col].quantile(0.60)
q80 = data_df[dv_col].quantile(0.80)
print(f"  Quintile boundaries: <{q20:.0f} | {q20:.0f}–{q40:.0f} | {q40:.0f}–{q60:.0f} | {q60:.0f}–{q80:.0f} | >{q80:.0f}")

print("\nLoading world map boundaries...")
world_gdf = gpd.read_file(GEOJSON_FILE)
print("  Country polygons:", len(world_gdf))

# The GeoJSON uses 'ISO3166-1-Alpha-3' as the ISO3 column
# Rename it to 'iso3' for the merge
world_gdf = world_gdf.rename(columns={"ISO3166-1-Alpha-3": "iso3"})

# ── Merge availability data onto world map ───────────────────────────────────
print("\nMerging availability data onto map...")
merged_gdf = world_gdf.merge(data_df, on="iso3", how="left")

# Count how many countries have data
n_with_data = merged_gdf["Cereal Availability (kg/person/yr)"].notna().sum()
n_total     = len(merged_gdf)
print(f"  Countries with availability data: {n_with_data} / {n_total}")

# ── Define the five availability bands and their colours ─────────────────────
# I use a sequential green palette from light to dark:
#   Very Low  → pale yellow (food insecure)
#   Low       → light orange
#   Medium    → mid green
#   High      → darker green
#   Very High → deep green (food abundant)
# Grey is used for countries with no data.

BAND_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"]

BAND_COLOURS = {
    "Very Low":  "#d7191c",   # red    (severe scarcity)
    "Low":       "#fdae61",   # orange
    "Medium":    "#ffffbf",   # pale yellow
    "High":      "#a6d96a",   # light green
    "Very High": "#1a9641",   # dark green (abundance)
    "No data":   "#d3d3d3",   # light grey
}

# Map each row to a colour
def get_colour(band_value):
    if pd.isna(band_value):
        return BAND_COLOURS["No data"]
    band_str = str(band_value).strip()
    if band_str in BAND_COLOURS:
        return BAND_COLOURS[band_str]
    return BAND_COLOURS["No data"]

colour_list = []
for band_value in merged_gdf["Availability Band"]:
    colour_list.append(get_colour(band_value))

merged_gdf["plot_colour"] = colour_list

# ── Draw the map ─────────────────────────────────────────────────────────────
print("\nDrawing choropleth map...")

fig, ax = plt.subplots(figsize=(18, 9))

# Use Robinson projection for a visually balanced world map
# geopandas can reproject on the fly
try:
    merged_proj = merged_gdf.to_crs("+proj=robin")
except Exception as e:
    print("  Note: could not reproject to Robinson —", e)
    print("  Falling back to Plate Carrée (WGS84)")
    merged_proj = merged_gdf

# Draw country polygons
merged_proj.plot(
    ax=ax,
    color=merged_proj["plot_colour"],
    linewidth=0.25,
    edgecolor="#555555",
)

# ── Build legend patches ─────────────────────────────────────────────────────
legend_handles = []

for band_name in BAND_ORDER:
    colour = BAND_COLOURS[band_name]
    patch  = mpatches.Patch(facecolor=colour, edgecolor="#555555", linewidth=0.5, label=band_name)
    legend_handles.append(patch)

# Add a "No data" patch
no_data_patch = mpatches.Patch(
    facecolor=BAND_COLOURS["No data"],
    edgecolor="#555555",
    linewidth=0.5,
    label="No data"
)
legend_handles.append(no_data_patch)

ax.legend(
    handles=legend_handles,
    title="Cereal Availability\n(kg/person/year)",
    loc="lower left",
    frameon=True,
    framealpha=0.9,
    fontsize=9,
    title_fontsize=9,
)

# ── Titles and labels ────────────────────────────────────────────────────────
ax.set_title(
    "Cereal Food Availability by Country, 2021\n"
    "(kg of cereal per person per year — FAO Food Balance Sheet)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)

ax.axis("off")

# Add a small annotation showing how many countries are in the sample
ax.annotate(
    f"Sample: {n_with_data} countries  |  "
    f"Quintile bands (kg/person/yr): Very Low <{q20:.0f} | Low {q20:.0f}–{q40:.0f} | "
    f"Medium {q40:.0f}–{q60:.0f} | High {q60:.0f}–{q80:.0f} | Very High >{q80:.0f}",
    xy=(0.5, 0.01),
    xycoords="figure fraction",
    ha="center",
    va="bottom",
    fontsize=7.5,
    color="#444444",
)

plt.tight_layout(pad=1.0)

# ── Save figure ──────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_FILE}")
plt.close()

print("\nDone. Choropleth map complete.")
