# CLAUDE.md — Food Insecurity Research Pipeline

## What this project does

This is a dissertation pipeline for Sheffield Hallam University (SHU) that answers:

> *Do themes identified by NLP analysis of food-security literature predict cross-country
> cereal food availability, and does adding them to an empirical model improve explanatory power?*

The pipeline has 10 sequential steps. Each step feeds the next. Run all steps with:

```bash
bash run_pipeline.sh
```

Or run a single step directly:

```bash
python src/step3_find_topics_in_papers.py
```

---

## File layout

```
src/
  step1_collect_research_papers.py   ← OpenAlex + Scopus API, builds corpus
  step2_read_text_from_pdfs.py       ← extracts text from local PDFs
  step3_find_topics_in_papers.py     ← TF-IDF, LDA coherence sweep, NMF topics
  step4_score_and_filter_papers.py   ← scores papers; keeps "strict" alignment only
  step5_download_country_data.py     ← World Bank WDI, Findex, IMF, WGI, PHL
  step6_clean_and_combine_data.py    ← merges all sources → master_dataset_clean.csv
  step7_run_prediction_models.py     ← OLS/RF/XGBoost models A→F, SHAP, bootstrap CIs
  step8_check_results_are_reliable.py← VIF, nested F-test, robustness checks
  step9_write_the_findings_report.py ← plain-text narrative → step9_synthesis.txt
  step10_export_for_dashboard.py     ← JSON/CSV exports for Streamlit dashboard
  chart_style_settings.py            ← shared matplotlib style (imported by all steps)

pipeline_notebook.ipynb  ← same pipeline as a runnable Colab / Jupyter notebook
run_pipeline.sh           ← runs all 10 steps in order; stops on first failure

data/raw/                 ← downloaded source files (not committed if large)
data/processed/           ← cleaned/merged files used by models
outputs/tables/           ← CSV and TXT model results
outputs/figures/          ← PNG charts (dpi=300 for print quality)
outputs/narrative/        ← step9_synthesis.txt narrative report
outputs/dashboard/        ← JSON/CSV exports for the dashboard
```

---

## Model structure (step7)

The empirical models are nested — each adds variables to the previous:

| Model | Extra variables | Purpose |
|-------|----------------|---------|
| A (baseline) | Production: yield, fertiliser, arable land, GDP/pc, rural pop, agri employment, livestock | How much do physical inputs explain cereal availability? |
| B | + cereal_loss_pct (post-harvest loss) | Does farm-to-market loss reduce availability? |
| C | + trade_pct_gdp, rural_electricity_access_pct | Does market integration / infrastructure matter? |
| F (NLP model) | Model A + 5 NLP-discovered variables | Do literature-identified themes predict outcomes? |
| A★ | Model A on the same sample as F | Honest incremental comparison (same N countries) |

**Dependent variable:** `cereal_availability_kg_pc` — kg of cereal per person per year from the
FAO Food Balance Sheet (Element 664, Item 2905). This is food *available* after
production + imports − exports ± stocks, not just domestic production.

**NLP-discovered variables in Model F:**
- `cereal_loss_pct` — post-harvest loss (Topic: grain storage / PHL)
- `trade_pct_gdp` — market integration proxy (Topic: value chain)
- `rural_electricity_access_pct` — storage/processing infrastructure (Topic: technology)
- `fertiliser_efficiency` — cereal yield / fertiliser kg/ha (Topic: land productivity)
- `food_price_inflation_pct` — price volatility signal (Topic: climate disruption)

---

## Key design decisions

- **RANDOM_SEED = 42** everywhere — set in step7 and step8.
- **Log transformations** applied to skewed variables (GDP, yield, trade, etc.) before OLS.
- **KNN imputation** (k=5) for columns with ≤ 20% missing before model fitting.
- **HC3 heteroskedasticity-robust standard errors** on all OLS models.
- **Bootstrap CIs**: 1000-iteration bootstrap for 95% CIs on NLP predictors.
- **Nested F-test** (step8): tests whether the NLP-discovered block (q=5 extra vars) adds
  genuine predictive power — F(5, n−k−1), compared against Model A★ on the same sample.
- **dpi=300** on all saved figures (print quality for dissertation).

---

## Known data gap — Findex Tier-1 variables

Three variables intended for `value_chain_finance_score` **cannot be downloaded from the
World Bank API**:
- `account_ownership_rural_pct` (FX.OWN.TOTL.RU.ZS)
- `borrowed_from_bank_pct` (FX.TRN.BORR.ZS)
- `agri_payments_digital_pct` (FX.TRN.AGRI.ZS)

The API returns no data for these regardless of source parameter (tried source=14, 71, 89).
They must be downloaded manually from https://www.worldbank.org/en/publication/globalfindex

A helper script is provided: `scripts/parse_findex_manual_download.py`
Once you download the Excel file, run that script to extract and merge these columns.

Currently `value_chain_finance_score` is computed from Tier-2/3 variables only
(female account ownership, poorest-40% ownership, bank branches, ATMs, private credit).
The dissertation should note this limitation.

---

## Running in Google Colab

Open `pipeline_notebook.ipynb` in Colab and run all cells top-to-bottom.
Mount Google Drive first and set `BASE_DIR` to wherever you uploaded the project folder.
Expected runtime: 15–25 minutes (most time in step5 downloads and step7 bootstrap).

---

## Dependencies

All packages are version-pinned in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

Key versions: pandas==2.2.3, numpy==1.26.4, gensim==4.3.3, xgboost==2.1.3, shap==0.46.0,
statsmodels==0.14.4, scikit-learn==1.5.2, pycountry==24.6.1.
