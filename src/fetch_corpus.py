"""
Fetches food insecurity corpus from OpenAlex API.
OpenAlex is free, open, and returns full abstracts — no institutional network needed.
Runs themed searches, deduplicates by DOI, saves to data/raw/corpus_metadata.csv.
"""

import time, requests, pandas as pd

BASE_URL = "https://api.openalex.org/works"
# Polite pool: include email so OpenAlex gives faster access
HEADERS = {"User-Agent": "mailto:odekunlejj@gmail.com"}

def reconstruct_abstract(inverted_index: dict) -> str:
    """OpenAlex stores abstracts as {word: [position, ...]}. Reconstruct to string."""
    if not inverted_index:
        return ""
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))

SEARCHES = [
    # Theme 1 — Post-harvest loss
    '"post-harvest loss" "food security"',
    '"postharvest loss" "cereal" "food"',
    '"post-harvest loss" "sub-Saharan Africa"',
    # Theme 2 — Financial access
    '"financial inclusion" "food security"',
    '"credit access" "food security" "smallholder"',
    '"microfinance" "food security"',
    # Theme 3 — Production / climate
    '"fertiliser efficiency" "food security"',
    '"fertilizer" "yield gap" "food security"',
    '"climate variability" "cereal yield"',
    '"rainfall" "food availability" "cross-country"',
    # Theme 4 — NLP / ML
    '"machine learning" "food security" prediction',
    '"text mining" "food insecurity"',
    '"topic model" "food security"',
    # Theme 5 — Cross-country food availability
    '"cereal food availability" "cross-country"',
    '"food balance sheet" "country-level"',
    # Theme 6 — Infrastructure / governance
    '"infrastructure" "food security" "developing countries"',
    '"governance" "food insecurity" "cross-country"',
]

def search_openalex(query: str, max_results: int = 15) -> list[dict]:
    records = []
    cursor = "*"
    while len(records) < max_results:
        params = {
            "search": query,
            "filter": "language:en,type:article,from_publication_date:2010-01-01",
            "per-page": min(25, max_results - len(records)),
            "cursor": cursor,
            "select": "id,doi,title,abstract_inverted_index,authorships,primary_location,publication_year",
            "sort": "cited_by_count:desc",
        }
        resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"    Error {resp.status_code}: {resp.text[:200]}")
            break
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        for r in results:
            abstract = reconstruct_abstract(r.get("abstract_inverted_index") or {})
            journal = ""
            loc = r.get("primary_location") or {}
            src = loc.get("source") or {}
            journal = src.get("display_name", "")
            authors = "; ".join(
                a.get("author", {}).get("display_name", "")
                for a in (r.get("authorships") or [])[:3]
            )
            doi = (r.get("doi") or "").replace("https://doi.org/", "")
            records.append({
                "title":    r.get("title", ""),
                "abstract": abstract,
                "authors":  authors,
                "journal":  journal,
                "year":     str(r.get("publication_year", "")),
                "doi":      doi,
                "openalex_id": r.get("id", ""),
            })
        cursor = data.get("meta", {}).get("next_cursor", None)
        if not cursor:
            break
        time.sleep(0.2)
    return records

all_records = []
for i, query in enumerate(SEARCHES, 1):
    print(f"[{i}/{len(SEARCHES)}] {query[:70]}...")
    results = search_openalex(query, max_results=15)
    print(f"  → {len(results)} records")
    all_records.extend(results)
    time.sleep(0.5)

df = pd.DataFrame(all_records)
df["abstract"] = df["abstract"].fillna("").astype(str)
df["title"]    = df["title"].fillna("").astype(str)
df["doi"]      = df["doi"].fillna("").astype(str)

before = len(df)
df = df[df["title"].str.len() > 0]
df = df.drop_duplicates(subset="doi").reset_index(drop=True)

has_abstract = (df["abstract"].str.len() > 50).sum()
print(f"\nTotal fetched: {before} | After dedup: {len(df)} | With abstracts: {has_abstract}")

out = "data/raw/corpus_metadata.csv"
df.to_csv(out, index=False)
print(f"Saved to {out}\n")

print(df[["title", "year", "journal"]].head(10).to_string())
print("\nSample abstract:")
print(df[df["abstract"].str.len() > 50]["abstract"].iloc[0][:300] if has_abstract > 0 else "No abstracts found")
