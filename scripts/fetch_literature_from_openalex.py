# ============================================================
# Download country-linked food availability literature
# ============================================================
#
# Purpose:
#   Strengthen the Step 1 literature corpus for the dissertation's
#   actual focus: food availability, not broad undernourishment.
#
# Source:
#   OpenAlex Works API. OpenAlex is a reproducible academic metadata
#   index used in bibliometric research. Google Scholar is not used
#   here because it does not provide a stable, permitted bulk API.
#
# Method:
#   1. Read the countries included in the project dataset.
#   2. Search OpenAlex for availability-side phrases plus country names.
#   3. Keep only papers whose title/abstract explicitly links to a
#      project country.
#   4. Save a raw supplemental CSV and append deduplicated records into
#      data/raw/corpus_metadata.csv for Step 4 scoring.
# ============================================================

import os
import re
import time

import pandas as pd
import requests


BASE_URL = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "mailto:odekunlejj@gmail.com"}

MASTER_FILE = "data/processed/master_dataset_clean.csv"
CORPUS_FILE = "data/raw/corpus_metadata.csv"
SUPPLEMENT_FILE = "data/raw/openalex_country_food_availability.csv"

os.makedirs("data/raw", exist_ok=True)


AVAILABILITY_QUERY_TEMPLATES = [
    '"food availability" "{country}"',
    '"household food availability" "{country}"',
    '"food supply" "{country}" "food security"',
    '"cereal availability" "{country}"',
    '"dietary energy supply" "{country}"',
    '"cereal production" "{country}" "food security"',
    '"crop production" "{country}" "food security"',
    '"post-harvest loss" "{country}" "food availability"',
    '"postharvest loss" "{country}" "food security"',
    '"grain storage" "{country}" "food security"',
]


COUNTRY_ALIASES = {
    "Bahamas, The": ["Bahamas"],
    "Bolivia": ["Bolivia", "Plurinational State of Bolivia"],
    "Brunei Darussalam": ["Brunei", "Brunei Darussalam"],
    "Congo, Dem. Rep.": ["Democratic Republic of Congo", "DRC", "Congo"],
    "Congo, Rep.": ["Republic of Congo", "Congo"],
    "Egypt, Arab Rep.": ["Egypt"],
    "Gambia, The": ["Gambia"],
    "Iran, Islamic Rep.": ["Iran"],
    "Kyrgyz Republic": ["Kyrgyzstan", "Kyrgyz Republic"],
    "Lao PDR": ["Laos", "Lao PDR", "Lao People's Democratic Republic"],
    "Micronesia, Fed. Sts.": ["Micronesia"],
    "Russian Federation": ["Russia", "Russian Federation"],
    "Slovak Republic": ["Slovakia", "Slovak Republic"],
    "St. Vincent and the Grenadines": ["Saint Vincent", "St. Vincent"],
    "Syrian Arab Republic": ["Syria", "Syrian Arab Republic"],
    "Tanzania": ["Tanzania", "United Republic of Tanzania"],
    "Turkiye": ["Turkey", "Turkiye"],
    "United States": ["United States", "USA", "United States of America"],
    "Venezuela, RB": ["Venezuela"],
    "Viet Nam": ["Vietnam", "Viet Nam"],
    "Yemen, Rep.": ["Yemen"],
}


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[pos] for pos in sorted(positions))


def normalise_key(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def country_aliases(country_name):
    aliases = COUNTRY_ALIASES.get(country_name, [country_name])
    if country_name not in aliases:
        aliases.append(country_name)
    return aliases


def contains_country(text, aliases):
    text_norm = normalise_key(text)
    for alias in aliases:
        alias_norm = normalise_key(alias)
        if len(alias_norm) >= 3 and re.search(rf"\b{re.escape(alias_norm)}\b", text_norm):
            return alias
    return ""


def search_openalex(query, per_query=8):
    params = {
        "search": query,
        "filter": "language:en,type:article,from_publication_date:2010-01-01",
        "per-page": per_query,
        "select": "id,doi,title,abstract_inverted_index,authorships,primary_location,publication_year",
        "sort": "cited_by_count:desc",
    }
    response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("results", [])


def record_from_work(work, country_code, country_name, matched_alias, query):
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    authorships = work.get("authorships") or []
    authors = []
    for entry in authorships[:3]:
        name = entry.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)

    doi = (work.get("doi") or "").replace("https://doi.org/", "")

    return {
        "title": work.get("title", ""),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index") or {}),
        "authors": "; ".join(authors),
        "journal": src.get("display_name", ""),
        "year": str(work.get("publication_year", "")),
        "doi": doi,
        "openalex_id": work.get("id", ""),
        "source_db": "OpenAlex-CountryAvailability",
        "matched_project_country_code": country_code,
        "matched_project_country": country_name,
        "matched_country_alias": matched_alias,
        "availability_query": query,
    }


def deduplicate(df):
    work = df.copy()
    for col in ["title", "abstract", "doi", "openalex_id"]:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str)

    work["doi_key"] = work["doi"].str.lower().str.strip()
    work["openalex_key"] = work["openalex_id"].str.lower().str.strip()
    work["title_key"] = (
        work["title"]
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
        .str[:120]
    )

    has_openalex = work["openalex_key"].str.len() > 5
    with_openalex = work[has_openalex].drop_duplicates("openalex_key", keep="first")
    without_openalex = work[~has_openalex]
    work = pd.concat([with_openalex, without_openalex], ignore_index=True)

    has_doi = work["doi_key"].str.len() > 3
    with_doi = work[has_doi].drop_duplicates("doi_key", keep="first")
    without_doi = work[~has_doi]
    work = pd.concat([with_doi, without_doi], ignore_index=True)

    work = work.drop_duplicates("title_key", keep="first")
    return work.drop(columns=["doi_key", "openalex_key", "title_key"], errors="ignore")


def main():
    print("=" * 60)
    print("Downloading country-linked food availability literature")
    print("=" * 60)

    countries = pd.read_csv(MASTER_FILE)[["country_code", "country_name"]].dropna()
    countries = countries.drop_duplicates("country_code").reset_index(drop=True)
    print(f"Project countries loaded: {len(countries)}")

    rows = []
    seen_work_ids = set()

    for idx, country in countries.iterrows():
        code = country["country_code"]
        name = country["country_name"]
        aliases = country_aliases(name)
        search_name = aliases[0]

        print(f"[{idx + 1:>3}/{len(countries)}] {name}")
        country_hits = 0

        for template in AVAILABILITY_QUERY_TEMPLATES:
            query = template.format(country=search_name)
            try:
                works = search_openalex(query)
            except Exception as exc:
                print(f"    query failed: {query} ({exc})")
                time.sleep(1)
                continue

            for work in works:
                work_id = work.get("id", "")
                if work_id and work_id in seen_work_ids:
                    continue

                title = work.get("title", "") or ""
                abstract = reconstruct_abstract(work.get("abstract_inverted_index") or {})
                matched_alias = contains_country(title + " " + abstract, aliases)
                if not matched_alias:
                    continue

                rows.append(record_from_work(work, code, name, matched_alias, query))
                if work_id:
                    seen_work_ids.add(work_id)
                country_hits += 1

            time.sleep(0.15)

        print(f"    kept {country_hits} country-linked records")

    if not rows:
        raise RuntimeError("No country-linked availability literature found.")

    supplement = deduplicate(pd.DataFrame(rows))
    supplement.to_csv(SUPPLEMENT_FILE, index=False)
    print(f"\nSupplement saved: {SUPPLEMENT_FILE}")
    print(f"Supplement records after deduplication: {len(supplement)}")

    if os.path.exists(CORPUS_FILE):
        corpus = pd.read_csv(CORPUS_FILE)
        combined = pd.concat([corpus, supplement], ignore_index=True, sort=False)
        combined = deduplicate(combined)
        combined.to_csv(CORPUS_FILE, index=False)
        print(f"Updated corpus saved: {CORPUS_FILE}")
        print(f"Updated corpus records: {len(combined)}")
    else:
        supplement.to_csv(CORPUS_FILE, index=False)
        print(f"Corpus did not exist; created: {CORPUS_FILE}")

    print("=" * 60)
    print("COUNTRY AVAILABILITY LITERATURE DOWNLOAD COMPLETE")


if __name__ == "__main__":
    main()
