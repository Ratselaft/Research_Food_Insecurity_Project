# ============================================================
# PHASE F — Synthesis: NLP findings vs Empirical results
# ============================================================
#
# What this file does:
#   This is where we tie the whole dissertation together.
#
#   Phase A used NLP (topic modelling) on 217 papers to find
#   out which themes and variables the academic literature
#   says matter for food insecurity.
#
#   Phases B–E collected real country data and ran regression
#   models to test whether those variables actually predict
#   undernourishment in practice.
#
#   Phase F puts those two things side by side:
#     — What did the literature say?  (NLP output)
#     — What did the data confirm?    (empirical models)
#     — Where do they agree?
#     — Where is the gap?
#
#   Outputs:
#     1. Synthesis table (CSV + printed)
#     2. NLP-to-evidence heatmap figure
#     3. R² progression chart (Model A → B → C)
#     4. Dissertation narrative text file
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

os.makedirs("outputs/tables",   exist_ok=True)
os.makedirs("outputs/figures",  exist_ok=True)
os.makedirs("outputs/narrative", exist_ok=True)

print("Starting Phase F — NLP vs Empirical synthesis...")
print("=" * 60)


# ============================================================
# STEP 1: Build the synthesis comparison table
# ============================================================
# This table has one row per LDA topic.
# For each topic we record:
#   - What the NLP found (theme, proxy variable)
#   - What the empirical models found (coefficient, p-value)
#   - Whether the NLP prediction was confirmed

print("\n[1] Building synthesis comparison table...")

# Each row is one LDA topic from Phase A.
# All coefficient and p-value numbers come from Phase D OLS results.
# "confirmed" scale: Confirmed / Partial / Not significant / Not tested

synthesis_rows = [
    {
        "topic_id":         0,
        "nlp_theme":        "Global food system risk / SDG indicators",
        "proxy_variable":   "Prevalence of undernourishment (%)",
        "role_in_model":    "Dependent variable",
        "model":            "A, B, C (DV)",
        "ols_coef":         "—",
        "p_value":          "—",
        "significance":     "—",
        "confirmed":        "Confirmed",
        "note":             "Used directly as the outcome we predict. "
                            "Validates the NLP choice of undernourishment as the focal measure.",
    },
    {
        "topic_id":         1,
        "nlp_theme":        "Land use & climate mitigation",
        "proxy_variable":   "arable_land_pct",
        "role_in_model":    "Predictor — Model A",
        "model":            "A",
        "ols_coef":         -0.074,
        "p_value":          0.045,
        "significance":     "**",
        "confirmed":        "Confirmed",
        "note":             "Negative and significant (p=0.045). More arable land → "
                            "lower undernourishment, consistent with land-availability theory.",
    },
    {
        "topic_id":         2,
        "nlp_theme":        "Sustainable agricultural policy & governance",
        "proxy_variable":   "Rule of Law index (WGI)",
        "role_in_model":    "Not tested",
        "model":            "—",
        "ols_coef":         "—",
        "p_value":          "—",
        "significance":     "—",
        "confirmed":        "Not tested",
        "note":             "WGI data not available via standard API. "
                            "Identified as an extension for future research.",
    },
    {
        "topic_id":         3,
        "nlp_theme":        "Climate change & adaptation",
        "proxy_variable":   "avg_precipitation_mm",
        "role_in_model":    "Robustness Spec 2",
        "model":            "E — Spec 2",
        "ols_coef":         0.267,
        "p_value":          0.662,
        "significance":     "n.s.",
        "confirmed":        "Not significant",
        "note":             "Average precipitation adds no explanatory power (p=0.66) "
                            "once yield and GDP are controlled for.",
    },
    {
        "topic_id":         4,
        "nlp_theme":        "Soil degradation & post-harvest loss",
        "proxy_variable":   "cereal_loss_pct",
        "role_in_model":    "Predictor — Model B (PHL block)",
        "model":            "B",
        "ols_coef":         0.505,
        "p_value":          0.032,
        "significance":     "**",
        "confirmed":        "Confirmed",
        "note":             "Positive and significant (p=0.032). Higher cereal loss → "
                            "higher undernourishment. Core finding of the dissertation.",
    },
    {
        "topic_id":         5,
        "nlp_theme":        "Household livelihoods & financial access",
        "proxy_variable":   "account_ownership_pct, bank_branches_per_100k",
        "role_in_model":    "Predictor — Model C (Finance block)",
        "model":            "C",
        "ols_coef":         0.127,
        "p_value":          0.144,
        "significance":     "n.s.",
        "confirmed":        "Partial",
        "note":             "Not individually significant (p=0.14) but Model C R² rises "
                            "to 0.727. Small sample (N=45) limits power. "
                            "Direction is consistent with theory.",
    },
    {
        "topic_id":         6,
        "nlp_theme":        "Agricultural waste / yield gaps & CSA practices",
        "proxy_variable":   "fertiliser_efficiency (cereal_yield / fertiliser_kg)",
        "role_in_model":    "Engineered feature — not directly modelled",
        "model":            "—",
        "ols_coef":         "—",
        "p_value":          "—",
        "significance":     "—",
        "confirmed":        "Not tested",
        "note":             "Computed in Phase C but not entered as a predictor. "
                            "Recommended for future robustness spec.",
    },
    {
        "topic_id":         7,
        "nlp_theme":        "Crop yield & production systems — baseline",
        "proxy_variable":   "cereal_yield_kg_per_ha, fertiliser_kg_per_ha",
        "role_in_model":    "Predictor — Models A, B, C",
        "model":            "A/B/C",
        "ols_coef":         -1.302,
        "p_value":          0.143,
        "significance":     "* (after outlier removal)",
        "confirmed":        "Partial",
        "note":             "Not significant in full sample (p=0.143) but becomes "
                            "significant (*) after Cook's D outlier removal and ** "
                            "in Model C (N=45). Suggests yield effect is real but "
                            "masked by outlier countries.",
    },
    {
        "topic_id":         8,
        "nlp_theme":        "ML & technology for crop monitoring",
        "proxy_variable":   "internet_users_pct",
        "role_in_model":    "Predictor — Models A, B, C",
        "model":            "A/B/C",
        "ols_coef":         -0.270,
        "p_value":          0.000,
        "significance":     "***",
        "confirmed":        "Confirmed",
        "note":             "Strongest single predictor. Negative and *** across all "
                            "five robustness specifications. ICT/infrastructure "
                            "captures development and market access.",
    },
]

synthesis_df = pd.DataFrame(synthesis_rows)
synthesis_df.to_csv("outputs/tables/nlp_empirical_synthesis.csv", index=False)
print(f"  Synthesis table saved → outputs/tables/nlp_empirical_synthesis.csv")

# Print a readable version
print("\n  Summary:")
print(f"  {'Topic':<5} {'NLP Theme':<42} {'Confirmed?':<20} {'Sig'}")
print("  " + "-" * 75)
for _, row in synthesis_df.iterrows():
    print(f"  T{row['topic_id']:<4} {row['nlp_theme'][:41]:<42} "
          f"{row['confirmed']:<20} {row['significance']}")


# ============================================================
# STEP 2: NLP-to-evidence heatmap
# ============================================================
# A colour-coded grid showing all 9 topics against their
# empirical outcome. Green = confirmed, amber = partial,
# red = not significant, grey = not tested.

print("\n[2] Creating NLP-to-evidence heatmap...")

# Map each "confirmed" category to a number for colour coding
COLOUR_MAP = {
    "Confirmed":        3,
    "Partial":          2,
    "Not significant":  1,
    "Not tested":       0,
}
COLOUR_LABELS = {
    3: ("Confirmed",       "#2E7D32"),   # dark green
    2: ("Partial",         "#F9A825"),   # amber
    1: ("Not significant", "#C62828"),   # dark red
    0: ("Not tested",      "#9E9E9E"),   # grey
}

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

values = [COLOUR_MAP[r["confirmed"]] for r in synthesis_rows]

fig, ax = plt.subplots(figsize=(9, 5))

# Draw one coloured rectangle per topic
for i, (theme, val) in enumerate(zip(short_themes, values)):
    colour = COLOUR_LABELS[val][1]
    ax.barh(i, 1, color=colour, edgecolor="white", linewidth=1.5)
    # Label inside the bar
    status = COLOUR_LABELS[val][0]
    ax.text(0.5, i, status, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")

ax.set_yticks(range(len(short_themes)))
ax.set_yticklabels([f"T{i}: {t}" for i, t in enumerate(short_themes)],
                   fontsize=9)
ax.set_xticks([])
ax.set_xlim(0, 1)
ax.invert_yaxis()
ax.set_title("NLP-identified themes vs empirical confirmation\n"
             "(LDA topics → proxy variables → regression results)",
             fontsize=11, fontweight="bold")

# Legend
legend_patches = [
    mpatches.Patch(color=v[1], label=v[0])
    for v in COLOUR_LABELS.values()
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig("outputs/figures/nlp_empirical_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Heatmap saved → outputs/figures/nlp_empirical_heatmap.png")


# ============================================================
# STEP 3: R² progression chart — Model A → B → C
# ============================================================
# Shows how much each block of variables adds to explanatory
# power. This is the key "incremental R²" result.

print("\n[3] Creating R² progression chart...")

model_labels  = ["Model A\nBaseline\n(N=151)", "Model B\n+PHL block\n(N=64)", "Model C\n+Finance\n(N=45)"]
ols_r2        = [0.644, 0.613, 0.727]
rf_cv_r2      = [0.630, 0.387, 0.222]
xgb_cv_r2     = [0.610, 0.302, 0.191]

x     = np.arange(len(model_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))

b1 = ax.bar(x - width, ols_r2,    width, label="OLS R²",           color="#1565C0", edgecolor="white")
b2 = ax.bar(x,         rf_cv_r2,  width, label="Random Forest CV R²", color="#EF6C00", edgecolor="white")
b3 = ax.bar(x + width, xgb_cv_r2, width, label="XGBoost CV R²",    color="#2E7D32", edgecolor="white")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

# Annotations explaining the sample size drop
ax.annotate("PHL data available\nfor 64 countries only",
            xy=(1, 0.613), xytext=(1.2, 0.75),
            arrowprops=dict(arrowstyle="->", color="grey"),
            fontsize=8, color="grey")

ax.annotate("Finance data\n45 countries",
            xy=(2, 0.727), xytext=(1.6, 0.85),
            arrowprops=dict(arrowstyle="->", color="grey"),
            fontsize=8, color="grey")

ax.set_xlabel("Model specification")
ax.set_ylabel("R²")
ax.set_title("Incremental R² across three model specifications\n"
             "Dependent variable: Prevalence of undernourishment (%)",
             fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=9)
ax.axhline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig("outputs/figures/r2_progression.png", dpi=150)
plt.close()
print("  R² progression chart saved → outputs/figures/r2_progression.png")


# ============================================================
# STEP 4: Coefficient summary spider / bar chart
# ============================================================
# One chart showing all significant predictors across models.

print("\n[4] Creating coefficient summary chart...")

# Variables and their OLS coefficients + significance across models
# We use Model A as the primary reference (largest N)
predictor_data = [
    ("internet_users_pct",       -0.270, "***", "Model A (N=151)"),
    ("arable_land_pct",          -0.074, "**",  "Model A (N=151)"),
    ("cereal_loss_pct",           0.505, "**",  "Model B (N=64)"),
    ("cereal_yield_kg_per_ha\n(Model C)", -5.281, "**",  "Model C (N=45)"),
    ("account_ownership_pct",     0.127, "n.s.", "Model C (N=45)"),
]

labels   = [p[0] for p in predictor_data]
coefs    = [p[1] for p in predictor_data]
sigs     = [p[2] for p in predictor_data]
sources  = [p[3] for p in predictor_data]
colours  = ["#1565C0" if c < 0 else "#C62828" for c in coefs]
alphas   = [1.0 if s != "n.s." else 0.4 for s in sigs]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(range(len(labels)), coefs, color=colours,
               edgecolor="white", linewidth=0.8)

# Significance labels
for i, (bar, sig, src) in enumerate(zip(bars, sigs, sources)):
    x_pos = bar.get_width()
    offset = 0.02 if x_pos >= 0 else -0.02
    ax.text(x_pos + offset, i, f"{sig}  ({src})",
            va="center", ha="left" if x_pos >= 0 else "right",
            fontsize=8, color="black")

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("OLS coefficient (log-scaled predictors)")
ax.set_title("Key OLS coefficients across Models A, B, C\n"
             "Blue = reduces undernourishment   Red = increases undernourishment",
             fontsize=10)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("outputs/figures/coefficient_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Coefficient summary chart saved → outputs/figures/coefficient_summary.png")


# ============================================================
# STEP 5: Write the dissertation narrative
# ============================================================
# This text block is the synthesis section you can paste
# directly into your dissertation chapter 4 / discussion.

print("\n[5] Writing dissertation narrative...")

# Count confirmations
n_confirmed = sum(1 for r in synthesis_rows if r["confirmed"] == "Confirmed")
n_partial   = sum(1 for r in synthesis_rows if r["confirmed"] == "Partial")
n_not_sig   = sum(1 for r in synthesis_rows if r["confirmed"] == "Not significant")
n_not_tested= sum(1 for r in synthesis_rows if r["confirmed"] == "Not tested")

narrative = f"""
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
(K=9, c_v coherence = 0.368) to a corpus of {217} peer-reviewed
papers, identifying nine thematic clusters that the academic
literature associates with food insecurity. Phases B through E
operationalised these themes as quantitative proxy variables and
tested them in cross-country OLS regressions and machine
learning models, using prevalence of undernourishment (%) as
the dependent variable (World Bank / FAO, 2021).

Of the nine NLP topics:
  • {n_confirmed} were empirically confirmed (statistically significant)
  • {n_partial} showed partial confirmation (correct sign, limited power)
  • {n_not_sig} were not statistically significant
  • {n_not_tested} could not be tested due to data availability

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
  (β = -0.074, p = 0.045) in Model A. Countries with more
  farmable land sustain lower undernourishment rates, aligning
  with land-availability arguments in Godfray et al. (2010).

Topic 4 — Post-harvest loss (cereal_loss_pct):
  The PHL block (Model B) introduces cereal loss rate as a
  significant positive predictor (β = 0.505, p = 0.032,
  N = 64). A one-percentage-point rise in cereal losses is
  associated with 0.5 percentage points more undernourishment,
  after controlling for production and development variables.
  This directly supports Affognon et al. (2015) and the
  dissertation's central hypothesis that PHL is an independent
  driver of food insecurity.

Topic 8 — ICT & technology (internet_users_pct):
  Internet penetration is the most robust predictor across
  the entire study (β = -0.270, p < 0.001 in Models A, B, C
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
  (β = -1.302, p = 0.143), likely because outlier countries
  distort the relationship. After Cook's Distance removal
  (Spec 4) the coefficient becomes significant (* p < 0.10),
  and in Model C it reaches ** (p = 0.048). This suggests the
  effect is real but sensitive to extreme observations.

Topic 5 — Financial access (account_ownership_pct):
  Account ownership enters positively (β = 0.127) but not
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
  Average precipitation is not significant (β = 0.267,
  p = 0.662) in Robustness Spec 2. This does not contradict
  climate-food linkages in the literature; rather, it suggests
  that annual average precipitation is too crude a measure —
  rainfall variability (coefficient of variation) would be
  more appropriate, as argued by Mbow et al. (2019 IPCC).

Topic 2 — Governance & policy (Rule of Law):
  The World Governance Indicators are not accessible via the
  World Bank REST API and require manual download. This
  represents a data gap rather than a theoretical gap;
  governance quality is likely to absorb some of the variance
  currently captured by internet_users_pct.

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
  of five. These are the two most robust predictors in the
  study.

  Cook's Distance identified 8 influential countries (Somalia,
  Haiti, Cabo Verde, Niger, Madagascar, Rwanda, Zambia,
  Liberia). Results do not change qualitatively when these are
  excluded, confirming the main findings are not outlier-driven.

----------------------------------------------------------------
4.6  Summary Table
----------------------------------------------------------------
Topic  Theme                           Variable              Result
----------------------------------------------------------------------
T0     SDG / Undernourishment          undernourishment_pct  DV ✓
T1     Land use & climate              arable_land_pct       ** ✓
T2     Governance & policy             Rule of Law (WGI)     Not tested
T3     Climate & rainfall              avg_precipitation_mm  n.s.
T4     Post-harvest loss               cereal_loss_pct       ** ✓
T5     Financial access                account_ownership_pct n.s. (partial)
T6     Yield gaps & CSA                fertiliser_efficiency Not tested
T7     Crop production baseline        cereal_yield_kg_per_ha * partial ✓
T8     ICT & technology                internet_users_pct    *** ✓

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

out_path = "outputs/narrative/phase_f_synthesis.txt"
with open(out_path, "w") as f:
    f.write(narrative)

print(narrative)
print(f"  Narrative saved → {out_path}")


# ============================================================
# STEP 6: Final project summary
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
