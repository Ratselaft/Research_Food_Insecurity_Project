#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh — executes all ten steps in order
#
# Each step feeds the next. The script stops immediately if any step fails
# so you never silently run downstream steps on broken inputs.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "========================================================"
echo "  Food Insecurity NLP/ML Pipeline"
echo "========================================================"

echo ""
echo "── Step 1: Collect research papers ──────────────────"
python src/step1_collect_research_papers.py

echo ""
echo "── Step 2: Read text from PDFs ──────────────────────"
python src/step2_read_text_from_pdfs.py

echo ""
echo "── Step 3: Find topics in papers ────────────────────"
python src/step3_find_topics_in_papers.py

echo ""
echo "── Step 4: Score and filter papers ──────────────────"
python src/step4_score_and_filter_papers.py

echo ""
echo "── Step 5: Download country data ────────────────────"
python src/step5_download_country_data.py

echo ""
echo "── Step 6: Clean and combine data ───────────────────"
python src/step6_clean_and_combine_data.py

echo ""
echo "── Step 7: Run prediction models ────────────────────"
python src/step7_run_prediction_models.py

echo ""
echo "── Step 8: Check results are reliable ───────────────"
python src/step8_check_results_are_reliable.py

echo ""
echo "── Step 9: Write the findings report ────────────────"
python src/step9_write_the_findings_report.py

echo ""
echo "── Step 10: Export for dashboard ────────────────────"
python src/step10_export_for_dashboard.py

echo ""
echo "========================================================"
echo "  Pipeline complete. Outputs written to outputs/"
echo "========================================================"
