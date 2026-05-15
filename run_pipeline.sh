#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh — executes all seven phases in order
#
# Each phase feeds the next. The script stops immediately if any phase fails
# so you never silently run downstream steps on broken inputs.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "========================================================"
echo "  Food Insecurity NLP/ML Pipeline"
echo "========================================================"

echo ""
echo "── Phase A1: Fetch papers from OpenAlex ──────────────"
python src/phase_a1_fetch_papers_from_openalex.py

echo ""
echo "── Phase A2: Extract text from PDFs ──────────────────"
python src/phase_a2_extract_text_from_pdfs.py

echo ""
echo "── Phase A3: LDA topic modelling ─────────────────────"
python src/phase_a3_lda_topic_modelling.py

echo ""
echo "── Phase A4: Score aligned literature ────────────────"
python src/phase_a4_score_aligned_literature.py

echo ""
echo "── Phase B: Download country datasets ────────────────"
python src/phase_b_download_country_datasets.py

echo ""
echo "── Phase C: Clean and merge master dataset ───────────"
python src/phase_c_clean_and_merge_master_dataset.py

echo ""
echo "── Phase D: OLS, Random Forest, XGBoost ─────────────"
python src/phase_d_ols_randomforest_xgboost.py

echo ""
echo "── Phase E: Outlier and robustness checks ────────────"
python src/phase_e_outlier_and_robustness_checks.py

echo ""
echo "── Phase F: NLP-to-empirical synthesis ───────────────"
python src/phase_f_nlp_to_empirical_synthesis.py

echo ""
echo "── Phase G: Power BI export ──────────────────────────"
python src/phase_g_powerbi_export.py

echo ""
echo "========================================================"
echo "  Pipeline complete. Outputs written to outputs/"
echo "========================================================"
