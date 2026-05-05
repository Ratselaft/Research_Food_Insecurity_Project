# ============================================================
# I'm tying the whole dissertation together in Phase F
# ============================================================
#
# What I'm doing here:
#   Phase A used NLP (topic modelling) on 217 papers to find
#   which themes the academic literature says matter for food
#   insecurity.
#
#   Phases B–E collected real country data and ran regression
#   models to test whether those variables actually predict
#   undernourishment in practice.
#
#   Phase F puts those two things side by side:
#     — What did the literature say?   (NLP output)
#     — What did the data confirm?     (empirical models)
#     — Where do they agree?
#     — Where is the gap?
#
#   Outputs I'm saving:
#     1. Synthesis comparison table (CSV)
#     2. NLP-to-evidence heatmap figure (colour-coded)
#     3. R² progression chart (Model A → B → C)
#     4. Coefficient summary bar chart
#     5. Dissertation narrative text file
# ============================================================

# I need os to create folders and work with file paths
import os

# I need numpy for mathematical operations like array of x positions
import numpy as np

# I need pandas to work with tables of data
import pandas as pd

# I need matplotlib to draw all my charts
import matplotlib
matplotlib.use('Agg')   # I save charts to files — no screen needed
import matplotlib.pyplot as plt

# I need patches to draw my legend boxes on the heatmap
import matplotlib.patches as mpatches

# I make sure my output folders exist
os.makedirs("outputs/tables",    exist_ok=True)
os.makedirs("outputs/figures",   exist_ok=True)
os.makedirs("outputs/narrative", exist_ok=True)

# I let the user know I'm starting
print("Starting Phase F — NLP vs Empirical synthesis...")
print("=" * 60)


# ============================================================
# Step 1: I'm building the synthesis comparison table
# ============================================================
# This table has one row per LDA topic (9 rows total).
# For each topic I record:
#   - What the NLP found (theme name, proxy variable)
#   - What the empirical models found (coefficient, p-value)
#   - Whether the NLP prediction was confirmed by the data

print("\n[1] Building synthesis comparison table...")

# I build the table as a list of dictionaries, one per topic.
# All coefficient and p-value numbers come from my Phase D results.
# I use "confirmed", "partial", "not significant", or "not tested"
# to describe the outcome.

synthesis_rows = []

# ── Topic 0 ───────────────────────────────────────────────────
row0 = {}
row0["topic_id"]       = 0
row0["nlp_theme"]      = "Global food system risk / SDG indicators"
row0["proxy_variable"] = "Prevalence of undernourishment (%)"
row0["role_in_model"]  = "Dependent variable"
row0["model"]          = "A, B, C (DV)"
row0["ols_coef"]       = "—"
row0["p_value"]        = "—"
row0["significance"]   = "—"
row0["confirmed"]      = "Confirmed"
row0["note"]           = ("Used directly as the outcome I predict. "
                          "Validates the NLP choice of undernourishment as the focal measure.")
synthesis_rows.append(row0)

# ── Topic 1 ───────────────────────────────────────────────────
row1 = {}
row1["topic_id"]       = 1
row1["nlp_theme"]      = "Land use & climate mitigation"
row1["proxy_variable"] = "arable_land_pct"
row1["role_in_model"]  = "Predictor — Model A"
row1["model"]          = "A"
row1["ols_coef"]       = -0.074
row1["p_value"]        = 0.045
row1["significance"]   = "**"
row1["confirmed"]      = "Confirmed"
row1["note"]           = ("Negative and significant (p=0.045). More arable land → "
                          "lower undernourishment, consistent with land-availability theory.")
synthesis_rows.append(row1)

# ── Topic 2 ───────────────────────────────────────────────────
row2 = {}
row2["topic_id"]       = 2
row2["nlp_theme"]      = "Sustainable agricultural policy & governance"
row2["proxy_variable"] = "Rule of Law index (WGI)"
row2["role_in_model"]  = "Not tested"
row2["model"]          = "—"
row2["ols_coef"]       = "—"
row2["p_value"]        = "—"
row2["significance"]   = "—"
row2["confirmed"]      = "Not tested"
row2["note"]           = ("WGI data not available via standard API. "
                          "Identified as an extension for future research.")
synthesis_rows.append(row2)

# ── Topic 3 ───────────────────────────────────────────────────
row3 = {}
row3["topic_id"]       = 3
row3["nlp_theme"]      = "Climate change & adaptation"
row3["proxy_variable"] = "avg_precipitation_mm"
row3["role_in_model"]  = "Robustness Spec 2"
row3["model"]          = "E — Spec 2"
row3["ols_coef"]       = 0.267
row3["p_value"]        = 0.662
row3["significance"]   = "n.s."
row3["confirmed"]      = "Not significant"
row3["note"]           = ("Average precipitation adds no explanatory power (p=0.66) "
                          "once yield and GDP are controlled for.")
synthesis_rows.append(row3)

# ── Topic 4 ───────────────────────────────────────────────────
row4 = {}
row4["topic_id"]       = 4
row4["nlp_theme"]      = "Soil degradation & post-harvest loss"
row4["proxy_variable"] = "cereal_loss_pct"
row4["role_in_model"]  = "Predictor — Model B (PHL block)"
row4["model"]          = "B"
row4["ols_coef"]       = 0.505
row4["p_value"]        = 0.032
row4["significance"]   = "**"
row4["confirmed"]      = "Confirmed"
row4["note"]           = ("Positive and significant (p=0.032). Higher cereal loss → "
                          "higher undernourishment. Core finding of the dissertation.")
synthesis_rows.append(row4)

# ── Topic 5 ───────────────────────────────────────────────────
row5 = {}
row5["topic_id"]       = 5
row5["nlp_theme"]      = "Household livelihoods & financial access"
row5["proxy_variable"] = "account_ownership_pct, bank_branches_per_100k"
row5["role_in_model"]  = "Predictor — Model C (Finance block)"
row5["model"]          = "C"
row5["ols_coef"]       = 0.127
row5["p_value"]        = 0.144
row5["significance"]   = "n.s."
row5["confirmed"]      = "Partial"
row5["note"]           = ("Not individually significant (p=0.14) but Model C R² rises "
                          "to 0.727. Small sample (N=45) limits power.")
synthesis_rows.append(row5)

# ── Topic 6 ───────────────────────────────────────────────────
row6 = {}
row6["topic_id"]       = 6
row6["nlp_theme"]      = "Agricultural waste / yield gaps & CSA practices"
row6["proxy_variable"] = "fertiliser_efficiency (cereal_yield / fertiliser_kg)"
row6["role_in_model"]  = "Engineered feature — not directly modelled"
row6["model"]          = "—"
row6["ols_coef"]       = "—"
row6["p_value"]        = "—"
row6["significance"]   = "—"
row6["confirmed"]      = "Not tested"
row6["note"]           = ("Computed in Phase C but not entered as a predictor. "
                          "Recommended for a future robustness specification.")
synthesis_rows.append(row6)

# ── Topic 7 ───────────────────────────────────────────────────
row7 = {}
row7["topic_id"]       = 7
row7["nlp_theme"]      = "Crop yield & production systems — baseline"
row7["proxy_variable"] = "cereal_yield_kg_per_ha, fertiliser_kg_per_ha"
row7["role_in_model"]  = "Predictor — Models A, B, C"
row7["model"]          = "A/B/C"
row7["ols_coef"]       = -1.302
row7["p_value"]        = 0.143
row7["significance"]   = "* (after outlier removal)"
row7["confirmed"]      = "Partial"
row7["note"]           = ("Not significant in full sample (p=0.143) but becomes "
                          "significant (*) after Cook's D removal and ** in Model C (N=45).")
synthesis_rows.append(row7)

# ── Topic 8 ───────────────────────────────────────────────────
row8 = {}
row8["topic_id"]       = 8
row8["nlp_theme"]      = "ML & technology for crop monitoring"
row8["proxy_variable"] = "internet_users_pct"
row8["role_in_model"]  = "Predictor — Models A, B, C"
row8["model"]          = "A/B/C"
row8["ols_coef"]       = -0.270
row8["p_value"]        = 0.000
row8["significance"]   = "***"
row8["confirmed"]      = "Confirmed"
row8["note"]           = ("Strongest single predictor. Negative and *** across all "
                          "five robustness specifications.")
synthesis_rows.append(row8)

# I turn my list of row dictionaries into a proper pandas table
synthesis_df = pd.DataFrame(synthesis_rows)

# I save the synthesis table to a CSV file
synthesis_df.to_csv("outputs/tables/nlp_empirical_synthesis.csv", index=False)
print("  Synthesis table saved → outputs/tables/nlp_empirical_synthesis.csv")

# I print a readable summary of the table
print("\n  Summary:")
print(f"  {'Topic':<5} {'NLP Theme':<42} {'Confirmed?':<20} {'Sig'}")
print("  " + "-" * 75)

for i in range(len(synthesis_df)):
    row = synthesis_df.iloc[i]
    print(f"  T{row['topic_id']:<4} {row['nlp_theme'][:41]:<42} "
          f"{row['confirmed']:<20} {row['significance']}")


# ============================================================
# Step 2: I'm drawing the NLP-to-evidence heatmap
# ============================================================
# I want a colour-coded chart showing all 9 topics and whether
# they were confirmed by the data.
# Green = confirmed, Amber = partial, Red = not significant, Grey = not tested

print("\n[2] Creating NLP-to-evidence heatmap...")

# I map each "confirmed" category to a number I can use for colouring
COLOUR_MAP = {}
COLOUR_MAP["Confirmed"]        = 3
COLOUR_MAP["Partial"]          = 2
COLOUR_MAP["Not significant"]  = 1
COLOUR_MAP["Not tested"]       = 0

# I map each number to a label and a colour
COLOUR_LABELS = {}
COLOUR_LABELS[3] = ("Confirmed",        "#2E7D32")  # dark green
COLOUR_LABELS[2] = ("Partial",          "#F9A825")  # amber
COLOUR_LABELS[1] = ("Not significant",  "#C62828")  # dark red
COLOUR_LABELS[0] = ("Not tested",       "#9E9E9E")  # grey

# I write the short theme labels I'll use on the chart
short_themes = [
    "SDG / Undernourishment",
    "Land use & climate",
    "Governance & policy",
    "Climate & rainfall",
    "Post-harvest loss",
    "Financial access",
    "Yield gaps & CSA",
    "Crop production",
    "ICT & technology",
]

# I build my list of colour numbers, one per topic
values = []
for row in synthesis_rows:
    colour_number = COLOUR_MAP[row["confirmed"]]
    values.append(colour_number)

# I create the figure
fig, ax = plt.subplots(figsize=(9, 5))

# I draw one coloured bar per topic
for i in range(len(short_themes)):
    theme = short_themes[i]
    val   = values[i]

    # I look up the colour for this confirmation status
    colour = COLOUR_LABELS[val][1]

    # I draw a horizontal bar for this topic
    ax.barh(i, 1, color=colour, edgecolor="white", linewidth=1.5)

    # I add a text label inside the bar showing the status
    status = COLOUR_LABELS[val][0]
    ax.text(0.5, i, status, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")

# I set the y tick labels to the topic numbers and short names
y_tick_labels = []
for i in range(len(short_themes)):
    y_tick_labels.append("T" + str(i) + ": " + short_themes[i])

ax.set_yticks(range(len(short_themes)))
ax.set_yticklabels(y_tick_labels, fontsize=9)

# I remove the x axis ticks since they have no meaning here
ax.set_xticks([])
ax.set_xlim(0, 1)

# I put the most recent topic at the top
ax.invert_yaxis()

# I add a title
ax.set_title("NLP-identified themes vs empirical confirmation\n"
             "(LDA topics → proxy variables → regression results)",
             fontsize=11, fontweight="bold")

# I build the legend patches manually
legend_patches = []
for num in COLOUR_LABELS:
    label  = COLOUR_LABELS[num][0]
    colour = COLOUR_LABELS[num][1]
    patch  = mpatches.Patch(color=colour, label=label)
    legend_patches.append(patch)

# I add the legend to the chart
ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

# I tidy up the layout
plt.tight_layout()

# I save the heatmap
plt.savefig("outputs/figures/nlp_empirical_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Heatmap saved → outputs/figures/nlp_empirical_heatmap.png")


# ============================================================
# Step 3: I'm drawing the R² progression chart
# ============================================================
# This shows how much each new block of variables adds to R².
# Model A → B → C is the key story of the dissertation.

print("\n[3] Creating R² progression chart...")

# I write the labels, OLS R², RF CV R², and XGB CV R² for each model
model_labels = ["Model A\nBaseline\n(N=151)", "Model B\n+PHL block\n(N=64)", "Model C\n+Finance\n(N=45)"]
ols_r2       = [0.644, 0.613, 0.727]
rf_cv_r2     = [0.630, 0.387, 0.222]
xgb_cv_r2    = [0.610, 0.302, 0.191]

# I set the x positions for the three groups of bars
x = np.arange(len(model_labels))

# I set the width of each bar
width = 0.25

# I create the figure
fig, ax = plt.subplots(figsize=(9, 5))

# I draw the OLS R² bars
b1 = ax.bar(x - width, ols_r2,    width, label="OLS R²",              color="#1565C0", edgecolor="white")

# I draw the Random Forest CV R² bars
b2 = ax.bar(x,         rf_cv_r2,  width, label="Random Forest CV R²", color="#EF6C00", edgecolor="white")

# I draw the XGBoost CV R² bars
b3 = ax.bar(x + width, xgb_cv_r2, width, label="XGBoost CV R²",       color="#2E7D32", edgecolor="white")

# I add value labels on top of every bar
for bar_group in [b1, b2, b3]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)

# I add a note explaining the sample size drop in Model B
ax.annotate("PHL data available\nfor 64 countries only",
            xy=(1, 0.613), xytext=(1.2, 0.75),
            arrowprops=dict(arrowstyle="->", color="grey"),
            fontsize=8, color="grey")

# I add a note explaining the sample size drop in Model C
ax.annotate("Finance data\n45 countries",
            xy=(2, 0.727), xytext=(1.6, 0.85),
            arrowprops=dict(arrowstyle="->", color="grey"),
            fontsize=8, color="grey")

# I label the axes
ax.set_xlabel("Model specification")
ax.set_ylabel("R²")
ax.set_title("Incremental R² across three model specifications\n"
             "Dependent variable: Prevalence of undernourishment (%)",
             fontsize=11)

# I set the x tick labels
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9)

# I set the y axis limit
ax.set_ylim(0, 1.0)

# I add a legend
ax.legend(fontsize=9)

# I draw a line at y=0
ax.axhline(0, color="black", linewidth=0.5)

# I tidy up the layout
plt.tight_layout()

# I save the chart
plt.savefig("outputs/figures/r2_progression.png", dpi=150)
plt.close()
print("  R² progression chart saved → outputs/figures/r2_progression.png")


# ============================================================
# Step 4: I'm drawing the coefficient summary chart
# ============================================================
# One chart showing all significant predictors across models.
# Blue = reduces undernourishment, Red = increases it.

print("\n[4] Creating coefficient summary chart...")

# I write the data for this chart manually (from my Phase D results)
# Each entry: (variable label, OLS coefficient, significance, source model)
predictor_data = [
    ("internet_users_pct",              -0.270, "***",  "Model A (N=151)"),
    ("arable_land_pct",                 -0.074, "**",   "Model A (N=151)"),
    ("cereal_loss_pct",                  0.505, "**",   "Model B (N=64)"),
    ("cereal_yield_kg_per_ha (Mdl C)",  -5.281, "**",   "Model C (N=45)"),
    ("account_ownership_pct",            0.127, "n.s.", "Model C (N=45)"),
]

# I pull out just the labels, coefficients, significance, and sources
labels  = []
coefs   = []
sigs    = []
sources = []

for item in predictor_data:
    labels.append(item[0])
    coefs.append(item[1])
    sigs.append(item[2])
    sources.append(item[3])

# I colour bars blue if the coefficient is negative (reduces undernourishment)
# and red if it's positive (increases undernourishment)
colours = []
for c in coefs:
    if c < 0:
        colours.append("#1565C0")   # blue for negative
    else:
        colours.append("#C62828")   # red for positive

# I set the transparency — full opacity for significant, faded for not significant
alphas = []
for s in sigs:
    if s == "n.s.":
        alphas.append(0.4)
    else:
        alphas.append(1.0)

# I create the figure
fig, ax = plt.subplots(figsize=(9, 5))

# I draw the horizontal bars
for i in range(len(labels)):
    ax.barh(i, coefs[i], color=colours[i],
            edgecolor="white", linewidth=0.8, alpha=alphas[i])

# I add significance labels and source text to each bar
for i in range(len(labels)):
    x_pos  = coefs[i]
    sig    = sigs[i]
    source = sources[i]

    # I position the label to the right if positive, to the left if negative
    if x_pos >= 0:
        offset = 0.02
        align  = "left"
    else:
        offset = -0.02
        align  = "right"

    ax.text(x_pos + offset, i,
            sig + "  (" + source + ")",
            va="center", ha=align, fontsize=8, color="black")

# I set the y tick labels
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)

# I draw a vertical line at 0
ax.axvline(0, color="black", linewidth=0.8)

# I label the x axis
ax.set_xlabel("OLS coefficient (log-scaled predictors)")

# I add a title
ax.set_title("Key OLS coefficients across Models A, B, C\n"
             "Blue = reduces undernourishment   Red = increases undernourishment",
             fontsize=10)

# I put the most important variable at the top
ax.invert_yaxis()

# I tidy up the layout
plt.tight_layout()

# I save the chart
plt.savefig("outputs/figures/coefficient_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Coefficient summary chart saved → outputs/figures/coefficient_summary.png")


# ============================================================
# Step 5: I'm writing the dissertation narrative
# ============================================================
# This is the text I can paste directly into my dissertation
# discussion chapter. I've written it in academic language.

print("\n[5] Writing dissertation narrative...")

# I count how many topics fell into each confirmation category
n_confirmed   = 0
n_partial     = 0
n_not_sig     = 0
n_not_tested  = 0

for row in synthesis_rows:
    if row["confirmed"] == "Confirmed":
        n_confirmed += 1
    elif row["confirmed"] == "Partial":
        n_partial += 1
    elif row["confirmed"] == "Not significant":
        n_not_sig += 1
    elif row["confirmed"] == "Not tested":
        n_not_tested += 1

# I write the full narrative text
narrative = """
================================================================
DISSERTATION SYNTHESIS — PHASE F
NLP-Identified Themes vs Empirical Model Results
================================================================
Generated: 2026-05-05
Project: From Literature to Evidence: An NLP-Driven Predictive
         Analysis of Food Insecurity Factors

----------------------------------------------------------------
4.1  Overview
----------------------------------------------------------------
The NLP analysis in Phase A applied Latent Dirichlet Allocation
(K=9, c_v coherence = 0.368) to a corpus of 217 peer-reviewed
papers, identifying nine thematic clusters that the academic
literature associates with food insecurity. Phases B through E
operationalised these themes as quantitative proxy variables and
tested them in cross-country OLS regressions and machine
learning models, using prevalence of undernourishment (%) as
the dependent variable (World Bank / FAO, 2021).

Of the nine NLP topics:
  • 4 were empirically confirmed (statistically significant)
  • 2 showed partial confirmation (correct sign, limited power)
  • 1 were not statistically significant
  • 2 could not be tested due to data availability

----------------------------------------------------------------
4.2  Confirmed Findings
----------------------------------------------------------------

Topic 7 — Crop production baseline (cereal yield, fertiliser):
  Model A establishes the production baseline across 151
  countries (R² = 0.644, F p < 0.001). Cereal yield becomes
  significant (** p=0.048) in Model C once post-harvest loss
  and financial variables are controlled for, consistent with
  Tittonell & Giller (2013) on yield-gap theory.

Topic 1 — Land use & climate mitigation (arable_land_pct):
  Arable land share enters negatively and significantly
  (b = -0.074, p = 0.045) in Model A. Countries with more
  farmable land sustain lower undernourishment rates, aligning
  with land-availability arguments in Godfray et al. (2010).

Topic 4 — Post-harvest loss (cereal_loss_pct):
  The PHL block (Model B) introduces cereal loss rate as a
  significant positive predictor (b = 0.505, p = 0.032,
  N = 64). A one-percentage-point rise in cereal losses is
  associated with 0.5 percentage points more undernourishment,
  after controlling for production and development variables.
  This directly supports Affognon et al. (2015) and the
  dissertation's central hypothesis that PHL is an independent
  driver of food insecurity.

Topic 8 — ICT & technology (internet_users_pct):
  Internet penetration is the most robust predictor across
  the entire study (b = -0.270, p < 0.001 in Models A, B, C
  and all five robustness specifications). This likely captures
  both market access and broader development, consistent with
  the ML-for-agriculture literature identified in Topic 8.

Topic 0 — SDG / Undernourishment prevalence:
  The NLP correctly identified undernourishment as the focal
  outcome measure. Its use as the dependent variable validates
  the topic-modelling framework as a literature-driven
  variable-selection tool.

----------------------------------------------------------------
4.3  Partial Confirmations
----------------------------------------------------------------

Topic 7 — Cereal yield (Model A):
  Yield is negative but not significant in the full sample
  (b = -1.302, p = 0.143), likely because outlier countries
  distort the relationship. After Cook's Distance removal
  (Spec 4) the coefficient becomes significant (* p < 0.10),
  and in Model C it reaches ** (p = 0.048). This suggests the
  effect is real but sensitive to extreme observations.

Topic 5 — Financial access (account_ownership_pct):
  Account ownership enters positively (b = 0.127) but not
  significantly (p = 0.144) in Model C. The small sample
  (N = 45) reduces statistical power substantially. The
  direction is unexpected — possible endogeneity: countries
  with higher food insecurity may have expanded mobile money
  programmes specifically to address poverty. This warrants
  further investigation.

----------------------------------------------------------------
4.4  Non-Confirmations and Data Gaps
----------------------------------------------------------------

Topic 3 — Climate & rainfall (avg_precipitation_mm):
  Average precipitation is not significant (b = 0.267,
  p = 0.662) in Robustness Spec 2. This does not contradict
  climate-food linkages in the literature; rather, it suggests
  that annual average precipitation is too crude a measure.
  Rainfall variability (coefficient of variation) would be
  more appropriate, as argued by Mbow et al. (2019 IPCC).

Topic 2 — Governance & policy (Rule of Law):
  The World Governance Indicators are not accessible via the
  World Bank REST API and require manual download. This
  represents a data gap rather than a theoretical gap.

Topic 6 — Yield gaps & CSA (fertiliser_efficiency):
  Fertiliser efficiency was engineered in Phase C but not
  entered directly as a predictor. This is an opportunity for
  future modelling to test the marginal return to fertiliser
  use conditional on yield.

----------------------------------------------------------------
4.5  Robustness
----------------------------------------------------------------
Five robustness specifications were tested in Phase E:
  Spec 1 — Baseline:             R² = 0.644 (N=151)
  Spec 2 — +Precipitation:       R² = 0.640 (N=147)
  Spec 3 — Log DV:               R² = 0.746 (N=151)
  Spec 4 — Drop Cook outliers:   R² = 0.711 (N=143)
  Spec 5 — Drop ISO outliers:    R² = 0.664 (N=136)

  internet_users_pct is negative and *** significant in all
  five specifications. arable_land_pct is significant in four
  of five. These are the two most robust predictors in the study.

  Cook's Distance identified 8 influential countries (Somalia,
  Haiti, Cabo Verde, Niger, Madagascar, Rwanda, Zambia, Liberia).
  Results do not change qualitatively when these are excluded,
  confirming the main findings are not outlier-driven.

----------------------------------------------------------------
4.6  Summary Table
----------------------------------------------------------------
Topic  Theme                           Variable              Result
----------------------------------------------------------------------
T0     SDG / Undernourishment          undernourishment_pct  DV confirmed
T1     Land use & climate              arable_land_pct       ** confirmed
T2     Governance & policy             Rule of Law (WGI)     Not tested
T3     Climate & rainfall              avg_precipitation_mm  n.s.
T4     Post-harvest loss               cereal_loss_pct       ** confirmed
T5     Financial access                account_ownership_pct n.s. (partial)
T6     Yield gaps & CSA                fertiliser_efficiency  Not tested
T7     Crop production baseline        cereal_yield_kg_per_ha * partial
T8     ICT & technology                internet_users_pct    *** confirmed

Key: *** p<0.01   ** p<0.05   * p<0.10   n.s. = not significant

----------------------------------------------------------------
4.7  Power BI Dashboard — recommended content
----------------------------------------------------------------
Build five dashboard pages:

  Page 1 — World map
    Choropleth: undernourishment_pct by country
    Filters: year, region

  Page 2 — Model performance
    Bar chart: OLS R² vs RF CV R² vs XGB CV R² for A, B, C
    Table: N, R², Adj R², F-stat for each model

  Page 3 — Key predictors
    Horizontal bar: OLS coefficients with significance stars
    SHAP importance chart (import PNG from outputs/figures/)

  Page 4 — NLP synthesis
    Import nlp_empirical_heatmap.png
    Table: 9 topics, proxy variable, confirmed status

  Page 5 — Robustness
    Import robustness_coefficients.png
    Table: 5 specs, R², N, internet_users_pct coefficient

================================================================
"""

# I set the path where I'll save the narrative
out_path = "outputs/narrative/phase_f_synthesis.txt"

# I open the file for writing
with open(out_path, "w") as f:
    # I write the full narrative text to the file
    f.write(narrative)

# I also print it to the screen so the user can read it now
print(narrative)

# I confirm where I saved it
print(f"  Narrative saved → {out_path}")


# ============================================================
# Step 6: I'm printing the final summary
# ============================================================

print("=" * 60)
print("PHASE F COMPLETE")
print("=" * 60)
print("""
All outputs saved:
  outputs/tables/nlp_empirical_synthesis.csv
  outputs/figures/nlp_empirical_heatmap.png
  outputs/figures/r2_progression.png
  outputs/figures/coefficient_summary.png
  outputs/narrative/phase_f_synthesis.txt

Pipeline status:
  Phase A — NLP corpus + LDA              DONE
  Phase B — Data download                 DONE
  Phase C — Clean + merge datasets        DONE
  Phase D — Models A, B, C               DONE
  Phase E — Outliers + robustness         DONE
  Phase F — Synthesis                     DONE
  Phase G — Final write-up               NEXT

Deadline: 12 May 2026  (7 days remaining)
""")
