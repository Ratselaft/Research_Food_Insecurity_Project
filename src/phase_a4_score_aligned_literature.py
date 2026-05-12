# ============================================================
# I'm scoring the literature corpus for dissertation alignment
# ============================================================
#
# What I'm doing here:
#   Phase A1 gives me a large corpus from OpenAlex and Scopus.
#   Not every paper in that corpus is equally useful for the
#   dissertation. This script scores each paper against the
#   project themes:
#
#     - food security / food insecurity / undernourishment
#     - post-harvest loss
#     - value chains and market access
#     - financial access / credit / inclusion
#     - smallholder agriculture
#     - production, yield, cereals
#     - climate, governance, gender, empirical methods
#
# Outputs:
#   data/processed/scored_literature_alignment.csv
#   data/processed/strictly_aligned_papers.csv
#   outputs/tables/literature_alignment_summary.csv
# ============================================================

import os
import re

import pandas as pd


CORPUS_FILE = "data/raw/corpus_metadata.csv"
SCORED_FILE = "data/processed/scored_literature_alignment.csv"
STRICT_FILE = "data/processed/strictly_aligned_papers.csv"
SUMMARY_FILE = "outputs/tables/literature_alignment_summary.csv"


CORE_TERMS = {
    "food security": 4,
    "food insecurity": 4,
    "undernourishment": 4,
    "undernourished": 4,
    "hunger": 3,
    "malnutrition": 3,
    "food availability": 3,
    "food access": 3,
    "nutrition security": 3,
}


THEMES = {
    "post_harvest_loss": [
        "post-harvest",
        "postharvest",
        "food loss",
        "food losses",
        "food waste",
        "storage loss",
        "grain storage",
        "stored products",
        "cold chain",
    ],
    "value_chain_market_access": [
        "value chain",
        "supply chain",
        "market access",
        "marketing channel",
        "short supply chain",
        "distribution",
        "logistical",
        "logistics",
    ],
    "financial_access": [
        "financial access",
        "financial inclusion",
        "credit access",
        "agricultural credit",
        "rural finance",
        "microfinance",
        "microcredit",
        "loan",
        "loans",
        "savings",
        "mobile money",
        "digital payments",
        "fintech",
    ],
    "smallholder_agriculture": [
        "smallholder",
        "smallholders",
        "small-scale farm",
        "small farming",
        "farm household",
        "rural household",
        "farmer",
        "farmers",
    ],
    "production_yield_cereals": [
        "cereal",
        "cereals",
        "maize",
        "wheat",
        "rice",
        "yield",
        "yields",
        "productivity",
        "fertilizer",
        "fertiliser",
        "irrigation",
    ],
    "climate_environment": [
        "climate change",
        "climate variability",
        "rainfall",
        "drought",
        "water scarcity",
        "land degradation",
        "resilience",
    ],
    "governance_institutions": [
        "governance",
        "institutional",
        "institutions",
        "political stability",
        "corruption",
        "government effectiveness",
        "policy",
    ],
    "gender_poverty_inclusion": [
        "women",
        "female",
        "gender",
        "poorest",
        "poverty",
        "inclusive",
        "inclusion",
    ],
    "empirical_methods": [
        "regression",
        "cross-country",
        "random forest",
        "machine learning",
        "xgboost",
        "panel data",
        "household survey",
        "survey",
    ],
}


PROJECT_DRIVER_THEMES = [
    "post_harvest_loss",
    "value_chain_market_access",
    "financial_access",
    "smallholder_agriculture",
    "climate_environment",      # climate shocks are a recognised structural driver of food insecurity
    "governance_institutions",  # governance quality shapes food system access and stability
]


SKIP_TERMS = [
    "genome",
    "transcriptome",
    "rna splicing",
    "virulence",
    "fungal stress",
    "protein source",
    "meat analogue",
    "robotic",
    "grasping",
    "culinary applications",
    "lateral flow immunoassay",
    "aflatoxin b1 detection",
    "mineral content",
    "insect-mineral",
    "soil spectral library",
    "environmental kuznets",
    "bioenergy expansion",
    "organic agriculture and climate change",
    "urea application",
    "legumes for agriculture sustainability",
    "plant and fungal use",
    "spectral library",
    "food upcycling",
    "volunteering",
    "water savings potentials",
    "military-private partnerships",
    "military-state building",
    "halal food online purchasing",
    "customer behavior",
]


def clean_text(value):
    """Convert a value to lowercase searchable text."""
    if pd.isna(value):
        return ""

    return str(value).lower()


def contains_phrase(text, phrase):
    """Check for a phrase using simple word boundaries."""
    escaped = re.escape(phrase.lower())
    pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def match_terms(text, terms):
    """Return every term that appears in the text."""
    matches = []

    for term in terms:
        if contains_phrase(text, term):
            matches.append(term)

    return matches


def score_one_paper(row):
    """Score one paper against the dissertation themes."""
    title = clean_text(row.get("title", ""))
    abstract = clean_text(row.get("abstract", ""))
    full_text = title + " " + abstract

    skip_matches = match_terms(full_text, SKIP_TERMS)

    score = 0
    core_matches = []
    title_core_matches = []

    for term, weight in CORE_TERMS.items():
        if contains_phrase(title, term):
            score = score + weight + 2
            core_matches.append(term)
            title_core_matches.append(term)
        elif contains_phrase(abstract, term):
            score = score + weight
            core_matches.append(term)

    theme_matches = []

    for theme_name, terms in THEMES.items():
        matched_terms = match_terms(full_text, terms)

        if len(matched_terms) > 0:
            theme_matches.append(theme_name)
            score = score + 2

        title_matches = match_terms(title, terms)
        score = score + len(title_matches)

    if clean_text(row.get("source_db", "")) == "scopus":
        score = score + 1

    abstract_length = len(abstract)
    if abstract_length >= 500:
        score = score + 1

    year = row.get("year", "")
    try:
        if int(float(year)) >= 2015:
            score = score + 1
    except Exception:
        pass

    if len(skip_matches) > 0:
        score = score - 5

    has_core = len(core_matches) > 0
    theme_count = len(theme_matches)
    project_driver_count = 0

    for theme_name in theme_matches:
        if theme_name in PROJECT_DRIVER_THEMES:
            project_driver_count = project_driver_count + 1

    is_pdf = clean_text(row.get("source_db", "")) == "pdf"

    if is_pdf:
        # PDFs are manually curated by the researcher, so they are pre-vetted
        # for topic relevance. Two paths to strict are used:
        #   • Core-term path: score ≥ 7 and a food-security core term matched.
        #   • Driver path: score ≥ 6 and two project-driver themes matched
        #     (handles post-harvest / value-chain papers that discuss food
        #     security implications without using the exact core phrase).
        core_path = score >= 7 and has_core and len(skip_matches) == 0
        driver_path = score >= 6 and project_driver_count >= 2 and len(skip_matches) == 0
        if core_path or driver_path:
            level = "strict"
        elif score >= 4 and len(skip_matches) == 0 and (has_core or theme_count >= 1):
            level = "moderate"
        else:
            level = "weak"
    else:
        if (
            score >= 12
            and has_core
            and theme_count >= 2
            and project_driver_count >= 1
            and len(skip_matches) == 0
            and (len(title_core_matches) > 0 or project_driver_count >= 2)
        ):
            level = "strict"
        elif score >= 7 and has_core and theme_count >= 1 and len(skip_matches) == 0:
            level = "moderate"
        else:
            level = "weak"

    return pd.Series(
        {
            "alignment_score": score,
            "alignment_level": level,
            "matched_core_terms": "; ".join(sorted(set(core_matches))),
            "matched_title_core_terms": "; ".join(sorted(set(title_core_matches))),
            "matched_themes": "; ".join(sorted(set(theme_matches))),
            "theme_count": theme_count,
            "project_driver_theme_count": project_driver_count,
            "skip_terms": "; ".join(sorted(set(skip_matches))),
        }
    )


def main():
    """Run the literature scoring workflow."""
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    print("Starting Phase A4 - scoring literature alignment...")
    print("=" * 60)

    corpus = pd.read_csv(CORPUS_FILE)
    print("Papers in corpus:", len(corpus))

    scores = corpus.apply(score_one_paper, axis=1)
    scored = pd.concat([corpus, scores], axis=1)

    scored = scored.sort_values(
        by=["alignment_level", "alignment_score", "year"],
        ascending=[True, False, False],
    )

    level_order = {"strict": 1, "moderate": 2, "weak": 3}
    scored["level_order"] = scored["alignment_level"].map(level_order)
    scored = scored.sort_values(
        by=["level_order", "alignment_score", "year"],
        ascending=[True, False, False],
    )
    scored = scored.drop(columns=["level_order"])

    strict = scored[scored["alignment_level"] == "strict"].copy()

    summary = (
        scored.groupby(["alignment_level", "source_db"])
        .size()
        .reset_index(name="papers")
        .sort_values(["alignment_level", "source_db"])
    )

    scored.to_csv(SCORED_FILE, index=False)
    strict.to_csv(STRICT_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print()
    print("Alignment summary:")
    print(summary.to_string(index=False))

    print()
    print("Strictly aligned papers:", len(strict))
    if len(strict) > 0:
        print("Strict source breakdown:")
        print(strict["source_db"].value_counts().to_string())

    print()
    print("Saved scored file:", SCORED_FILE)
    print("Saved strict shortlist:", STRICT_FILE)
    print("Saved summary:", SUMMARY_FILE)
    print("=" * 60)
    print("PHASE A4 COMPLETE")


if __name__ == "__main__":
    main()
