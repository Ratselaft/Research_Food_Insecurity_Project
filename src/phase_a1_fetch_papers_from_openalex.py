# ============================================================
# I'm building my academic paper corpus from multiple sources
# ============================================================
#
# What I'm doing here:
#   I collect papers about food insecurity from THREE sources:
#
#   Source 1 — OpenAlex API (automatic)
#     OpenAlex is a free, public database of academic papers.
#     No university login is needed. I search it automatically.
#
#   Source 2 — Scopus / Web of Science exports from SHU library
#     My Sheffield Hallam University login gives me access to
#     Scopus and Web of Science, which are much larger and more
#     comprehensive than OpenAlex. I export search results from
#     both databases as CSV files, place them in data/raw/, and
#     this script reads and merges them automatically.
#     THIS IS NOW THE PRIMARY SOURCE of high-quality papers.
#
#   Source 3 — Manually downloaded PDFs
#     Any PDF papers I download manually are handled by the
#     next script: phase_a2_extract_text_from_pdfs.py
#
# Why all three?
#   OpenAlex alone misses many journals and conference papers.
#   Scopus has 5x more coverage. Books, reports, and grey
#   literature (e.g. FAO reports, IFAD studies) must come
#   from PDFs since no API covers them.
#
# Output:
#   data/raw/corpus_metadata.csv
#   One row per paper. Columns: title, abstract, authors,
#   journal, year, doi, source (which database it came from)
# ============================================================

# I need time to pause between API requests (polite to the server)
import time

# I need requests to send web requests to the OpenAlex API
import requests

# I need pandas to work with tables of data
import pandas as pd

# I need os to check whether files exist
import os


# ============================================================
# Part 1: I'm fetching papers from OpenAlex (automatic)
# ============================================================

# This is the web address of the OpenAlex API
BASE_URL = "https://api.openalex.org/works"

# OpenAlex asks me to include my email so they can give me faster access
# This is called the "polite pool" — they prioritise requests with an email
HEADERS = {"User-Agent": "mailto:odekunlejj@gmail.com"}


# ============================================================
# Helper function: I rebuild abstracts from OpenAlex's format
# ============================================================
# OpenAlex stores abstracts in an "inverted index" format.
# Instead of the full text, it gives me a dictionary where each
# word maps to a list of positions.
# For example: {"food": [0, 5], "insecurity": [1]}
# This means "food" appears at positions 0 and 5, "insecurity" at 1.
# I put the words back in the right order.

def reconstruct_abstract(inverted_index):
    # If there is no abstract data, I return an empty string
    if not inverted_index:
        return ""

    # I build a dictionary: position number → word
    positions = {}
    for word in inverted_index:
        pos_list = inverted_index[word]
        for pos in pos_list:
            positions[pos] = word

    # I sort the positions and put the words back in order
    sorted_positions = sorted(positions)
    words_in_order = []
    for pos in sorted_positions:
        words_in_order.append(positions[pos])

    # I join all words into a sentence
    return " ".join(words_in_order)


# ============================================================
# My search queries for OpenAlex
# ============================================================
# I've grouped the searches by the themes from my LDA analysis.
# This ensures every topic has papers supporting it.

SEARCHES = [
    # Post-harvest loss
    '"post-harvest loss" "food security"',
    '"postharvest loss" "cereal" "food"',
    '"post-harvest loss" "sub-Saharan Africa"',
    '"food loss" "value chain" "smallholder"',
    '"storage loss" "cereal" "developing countries"',

    # Financial access and food security
    '"financial inclusion" "food security"',
    '"credit access" "food security" "smallholder"',
    '"microfinance" "food security"',
    '"digital payments" "agriculture" "food"',
    '"mobile money" "food security"',
    '"rural finance" "food insecurity"',
    '"agricultural credit" "food security"',

    # Value chain and supply chain
    '"value chain" "food insecurity" "smallholder"',
    '"supply chain" "food loss" "developing"',
    '"market access" "food security" "rural"',
    '"food value chain" "financial access"',

    # Agricultural production and inputs
    '"fertiliser efficiency" "food security"',
    '"fertilizer" "yield gap" "food security"',
    '"cereal yield" "food security" "cross-country"',
    '"irrigation" "food security" "smallholder"',

    # Climate and environment
    '"climate variability" "cereal yield"',
    '"rainfall" "food availability" "cross-country"',
    '"drought" "food insecurity" "Africa"',

    # Governance and institutions
    '"governance" "food insecurity" "cross-country"',
    '"political stability" "food security"',
    '"corruption" "food security" "agriculture"',

    # Gender in agriculture
    '"women" "food security" "agriculture" "finance"',
    '"female" "food insecurity" "smallholder"',
    '"gender" "agricultural finance" "food"',

    # Rural poverty and development
    '"rural poverty" "food insecurity" "cross-country"',
    '"gdp" "undernourishment" "regression"',
    '"infrastructure" "food security" "developing countries"',

    # Machine learning and quantitative methods
    '"machine learning" "food security" prediction',
    '"topic model" "food security"',
    '"cross-country" "undernourishment" "determinants"',
]


# ============================================================
# Function: I search OpenAlex for one query
# ============================================================

def search_openalex(query, max_results):
    # I'll collect all the results in this list
    records = []

    # OpenAlex uses a "cursor" to page through results
    # A cursor is like a bookmark — it tells the server where I left off
    cursor = "*"

    # I keep fetching pages of results until I have enough
    while len(records) < max_results:

        # I set up the parameters for this request
        params = {}
        params["search"]   = query
        params["filter"]   = "language:en,type:article,from_publication_date:2010-01-01"
        params["per-page"] = min(25, max_results - len(records))
        params["cursor"]   = cursor
        params["select"]   = "id,doi,title,abstract_inverted_index,authorships,primary_location,publication_year"
        params["sort"]     = "cited_by_count:desc"   # I want the most-cited papers first

        # I send the request and wait up to 30 seconds for a reply
        resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)

        # If the request failed, I print the error and stop
        if resp.status_code != 200:
            print("    Error", resp.status_code, ":", resp.text[:200])
            break

        # I convert the JSON response into a Python dictionary
        data = resp.json()

        # "results" is the list of papers OpenAlex returned
        results = data.get("results", [])

        # If there are no results, I stop
        if not results:
            break

        # I go through each paper one by one
        for r in results:
            # I rebuild the abstract from the inverted index
            abstract_text = reconstruct_abstract(r.get("abstract_inverted_index") or {})

            # I get the journal name — it is buried inside primary_location → source
            loc    = r.get("primary_location") or {}
            src    = loc.get("source") or {}
            journal = src.get("display_name", "")

            # I get the first three author names
            author_list  = r.get("authorships") or []
            author_names = []
            for author_entry in author_list[:3]:
                name = author_entry.get("author", {}).get("display_name", "")
                author_names.append(name)
            authors = "; ".join(author_names)

            # I clean up the DOI (remove the prefix if it is there)
            doi_raw = r.get("doi") or ""
            doi = doi_raw.replace("https://doi.org/", "")

            # I build one row for this paper
            one_paper = {}
            one_paper["title"]       = r.get("title", "")
            one_paper["abstract"]    = abstract_text
            one_paper["authors"]     = authors
            one_paper["journal"]     = journal
            one_paper["year"]        = str(r.get("publication_year", ""))
            one_paper["doi"]         = doi
            one_paper["openalex_id"] = r.get("id", "")
            one_paper["source_db"]   = "OpenAlex"   # I record which database this came from

            records.append(one_paper)

        # I check for a next-page cursor
        meta = data.get("meta", {})
        cursor = meta.get("next_cursor", None)

        # If there is no next cursor, I've reached the last page
        if not cursor:
            break

        # I wait briefly before the next request
        time.sleep(0.2)

    return records


# ============================================================
# Run the OpenAlex searches
# ============================================================

print("=" * 60)
print("PHASE A1 — Building the academic paper corpus")
print("=" * 60)
print()
print("SOURCE 1: OpenAlex API searches")
print("-" * 40)

all_records = []

for i in range(len(SEARCHES)):
    query = SEARCHES[i]
    print("[" + str(i + 1) + "/" + str(len(SEARCHES)) + "] " + query[:70] + "...")

    results = search_openalex(query, max_results=20)
    print("  Got", len(results), "papers")
    all_records.extend(results)

    time.sleep(0.5)

print("\nOpenAlex total (before dedup):", len(all_records))


# ============================================================
# Part 2: I'm loading Scopus CSV exports from SHU library
# ============================================================
# This is NOW the most important part of corpus building.
#
# How to export from Scopus (SHU library):
#   1. Log in at https://www.scopus.com using your SHU credentials
#   2. Search for: ("food insecurity" OR "food security") AND
#                  ("financial access" OR "post-harvest" OR
#                   "value chain" OR "smallholder" OR "governance")
#   3. Filter: Document type = Article, Language = English, Year >= 2010
#   4. Select all results → Export → CSV
#   5. Include: Title, Abstract, Authors, Source title, Year, DOI
#   6. Save as: data/raw/scopus_export.csv
#
# How to export from Web of Science (SHU library):
#   1. Log in at https://www.webofscience.com using your SHU credentials
#   2. Search similarly
#   3. Export → Other formats → Tab-delimited (Win, UTF-8)
#   4. Record content: Full record
#   5. Save as: data/raw/wos_export.txt

print()
print("SOURCE 2: Scopus and Web of Science exports from SHU library")
print("-" * 40)


def load_scopus_csv(filepath):
    # I load a Scopus CSV export and standardise the column names
    # so they match the OpenAlex format I am already using

    print("  Loading Scopus export:", filepath)

    try:
        df = pd.read_csv(filepath)
        print("    Columns found:", list(df.columns[:8]))
        print("    Rows in file:", len(df))

        # I create an empty output DataFrame with the same columns I use for OpenAlex
        out = pd.DataFrame()

        # Scopus uses "Title" for the paper title
        if "Title" in df.columns:
            out["title"] = df["Title"].fillna("").astype(str)
        elif "title" in df.columns:
            out["title"] = df["title"].fillna("").astype(str)
        else:
            print("    Warning: could not find a title column in Scopus file")
            out["title"] = ""

        # Scopus uses "Abstract" for the abstract text
        if "Abstract" in df.columns:
            out["abstract"] = df["Abstract"].fillna("").astype(str)
        elif "abstract" in df.columns:
            out["abstract"] = df["abstract"].fillna("").astype(str)
        else:
            out["abstract"] = ""

        # Scopus uses "Authors" for the author list
        if "Authors" in df.columns:
            out["authors"] = df["Authors"].fillna("").astype(str)
        else:
            out["authors"] = ""

        # Scopus uses "Source title" for the journal name
        if "Source title" in df.columns:
            out["journal"] = df["Source title"].fillna("").astype(str)
        elif "Source Title" in df.columns:
            out["journal"] = df["Source Title"].fillna("").astype(str)
        else:
            out["journal"] = ""

        # Scopus uses "Year" for the publication year
        if "Year" in df.columns:
            out["year"] = df["Year"].fillna("").astype(str)
        elif "year" in df.columns:
            out["year"] = df["year"].fillna("").astype(str)
        else:
            out["year"] = ""

        # Scopus uses "DOI" for the DOI
        if "DOI" in df.columns:
            out["doi"] = df["DOI"].fillna("").astype(str)
        elif "doi" in df.columns:
            out["doi"] = df["doi"].fillna("").astype(str)
        else:
            out["doi"] = ""

        # I add an empty openalex_id column (Scopus papers don't have this)
        out["openalex_id"] = ""

        # I record that these papers came from Scopus
        out["source_db"] = "Scopus"

        print("    Loaded", len(out), "papers from Scopus")
        return out

    except Exception as e:
        print("    Could not load Scopus file:", e)
        return None


def load_wos_export(filepath):
    # I load a Web of Science plain-text export and standardise columns.
    # WoS uses tab-separated format with two-letter field codes.

    print("  Loading Web of Science export:", filepath)

    try:
        # WoS exports are tab-separated
        df = pd.read_csv(filepath, sep="\t", encoding="utf-8", on_bad_lines="skip")
        print("    Columns found:", list(df.columns[:8]))
        print("    Rows in file:", len(df))

        out = pd.DataFrame()

        # WoS uses "TI" for Title
        if "TI" in df.columns:
            out["title"] = df["TI"].fillna("").astype(str)
        elif "Title" in df.columns:
            out["title"] = df["Title"].fillna("").astype(str)
        else:
            out["title"] = ""

        # WoS uses "AB" for Abstract
        if "AB" in df.columns:
            out["abstract"] = df["AB"].fillna("").astype(str)
        elif "Abstract" in df.columns:
            out["abstract"] = df["Abstract"].fillna("").astype(str)
        else:
            out["abstract"] = ""

        # WoS uses "AU" for Authors
        if "AU" in df.columns:
            out["authors"] = df["AU"].fillna("").astype(str)
        else:
            out["authors"] = ""

        # WoS uses "SO" for Source (journal)
        if "SO" in df.columns:
            out["journal"] = df["SO"].fillna("").astype(str)
        else:
            out["journal"] = ""

        # WoS uses "PY" for Publication Year
        if "PY" in df.columns:
            out["year"] = df["PY"].fillna("").astype(str)
        else:
            out["year"] = ""

        # WoS uses "DI" for DOI
        if "DI" in df.columns:
            out["doi"] = df["DI"].fillna("").astype(str)
        else:
            out["doi"] = ""

        out["openalex_id"] = ""
        out["source_db"]   = "Web of Science"

        print("    Loaded", len(out), "papers from Web of Science")
        return out

    except Exception as e:
        print("    Could not load Web of Science file:", e)
        return None


def load_scopus_ris(filepath):
    # Some Scopus exports come as .ris (RIS format) rather than CSV.
    # I parse the RIS format manually line by line.

    print("  Loading Scopus RIS export:", filepath)

    try:
        papers = []
        current = {}

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Each RIS field starts with a two-letter tag and " - "
                if " - " in line[:6]:
                    tag  = line[:2].strip()
                    value = line[6:].strip()

                    # I read the relevant tags
                    if tag == "TI" or tag == "T1":
                        current["title"] = value
                    elif tag == "AB":
                        current["abstract"] = value
                    elif tag == "AU":
                        # I accumulate author names in a list
                        if "authors_list" not in current:
                            current["authors_list"] = []
                        current["authors_list"].append(value)
                    elif tag == "JO" or tag == "T2" or tag == "SO":
                        current["journal"] = value
                    elif tag == "PY" or tag == "Y1":
                        current["year"] = value[:4]   # I only keep the 4-digit year
                    elif tag == "DO" or tag == "DI":
                        current["doi"] = value

                # "ER" signals the end of one paper record
                elif line == "ER":
                    if "title" in current:
                        one_paper = {}
                        one_paper["title"]       = current.get("title", "")
                        one_paper["abstract"]    = current.get("abstract", "")
                        authors_list = current.get("authors_list", [])
                        one_paper["authors"]     = "; ".join(authors_list[:3])
                        one_paper["journal"]     = current.get("journal", "")
                        one_paper["year"]        = current.get("year", "")
                        one_paper["doi"]         = current.get("doi", "")
                        one_paper["openalex_id"] = ""
                        one_paper["source_db"]   = "Scopus"
                        papers.append(one_paper)
                    current = {}

        out = pd.DataFrame(papers)
        print("    Loaded", len(out), "papers from RIS file")
        return out

    except Exception as e:
        print("    Could not load RIS file:", e)
        return None


# I collect all external database records here
external_records = []

# I check for each possible export file and load it if it exists

# --- Scopus CSV ---
SCOPUS_CSV = "data/raw/scopus_export.csv"
if os.path.exists(SCOPUS_CSV):
    scopus_df = load_scopus_csv(SCOPUS_CSV)
    if scopus_df is not None and len(scopus_df) > 0:
        external_records.append(scopus_df)
else:
    print("  Scopus CSV not found at:", SCOPUS_CSV)
    print("  Export from Scopus (SHU library) and save to that path.")

# --- Scopus RIS (alternative format) ---
SCOPUS_RIS = "data/raw/scopus_export.ris"
if os.path.exists(SCOPUS_RIS):
    ris_df = load_scopus_ris(SCOPUS_RIS)
    if ris_df is not None and len(ris_df) > 0:
        external_records.append(ris_df)

# --- Web of Science export ---
WOS_FILE = "data/raw/wos_export.txt"
if os.path.exists(WOS_FILE):
    wos_df = load_wos_export(WOS_FILE)
    if wos_df is not None and len(wos_df) > 0:
        external_records.append(wos_df)
else:
    print("  Web of Science export not found at:", WOS_FILE)
    print("  Export from WoS (SHU library) and save to that path.")

# I report how many external papers I found
total_external = sum(len(df) for df in external_records)
print("\nTotal papers from SHU library databases:", total_external)


# ============================================================
# Part 3: I'm combining all sources and removing duplicates
# ============================================================

print()
print("COMBINING ALL SOURCES")
print("-" * 40)

# I turn my list of OpenAlex records into a DataFrame
openalex_df = pd.DataFrame(all_records)

# I start my combined list with the OpenAlex papers
all_dfs = [openalex_df]

# I add any external database exports I found
for df in external_records:
    all_dfs.append(df)

# I combine everything into one big DataFrame
combined = pd.concat(all_dfs, ignore_index=True)

print("Total papers before deduplication:", len(combined))
print("Breakdown by source:")

# I count how many papers came from each source
for source_name in combined["source_db"].unique():
    n = (combined["source_db"] == source_name).sum()
    print("  ", source_name, ":", n, "papers")

# ── Deduplication ────────────────────────────────────────────────────────────
# A paper might appear in both OpenAlex AND my Scopus export.
# I need to remove duplicates carefully.

# I make sure all text columns are strings, not blank values
combined["title"]    = combined["title"].fillna("").astype(str)
combined["abstract"] = combined["abstract"].fillna("").astype(str)
combined["doi"]      = combined["doi"].fillna("").astype(str)
combined["year"]     = combined["year"].fillna("").astype(str)

# I remove papers with no title at all — they are not useful
combined = combined[combined["title"].str.len() > 5]

# Step 1: I deduplicate by DOI — if two papers have the same DOI, they are the same paper.
# I prefer to keep the Scopus/WoS version (richer metadata) over OpenAlex.
# I sort so that Scopus/WoS rows come first, then OpenAlex.

# I build a priority column: Scopus=1, WoS=2, OpenAlex=3
combined["source_priority"] = 3
combined.loc[combined["source_db"] == "Scopus", "source_priority"] = 1
combined.loc[combined["source_db"] == "Web of Science", "source_priority"] = 2

# I sort by priority (lower = higher priority = keep first)
combined = combined.sort_values("source_priority")

# I create a normalised DOI key for matching (lowercase, no spaces)
combined["doi_key"] = combined["doi"].str.lower().str.strip()

# I remove duplicates based on DOI, keeping the first (highest priority) occurrence
# I only deduplicate non-empty DOIs
has_doi = combined["doi_key"].str.len() > 3
no_doi  = ~has_doi

# I keep all rows without a DOI, plus one version of each DOI
combined_with_doi    = combined[has_doi].drop_duplicates(subset="doi_key", keep="first")
combined_without_doi = combined[no_doi]
combined = pd.concat([combined_with_doi, combined_without_doi], ignore_index=True)

before_title_dedup = len(combined)

# Step 2: I also deduplicate by title — titles that are nearly identical are the same paper.
# I create a title key: lowercase, strip whitespace, keep first 80 characters
combined["title_key"] = combined["title"].str.lower().str.strip().str[:80]
combined = combined.drop_duplicates(subset="title_key", keep="first")

print("\nAfter DOI dedup:", before_title_dedup, "papers")
print("After title dedup:", len(combined), "papers")

# I remove my helper columns — they were only needed for deduplication
combined = combined.drop(columns=["doi_key", "title_key", "source_priority"], errors="ignore")

# I reset the row numbers
combined = combined.reset_index(drop=True)

# I count how many papers have a real abstract
has_abstract = (combined["abstract"].str.len() > 50).sum()

print("\nFinal corpus:")
print("  Total papers:", len(combined))
print("  Papers with abstracts:", has_abstract)

# I show the breakdown by source in the final corpus
print("  Final breakdown by source:")
for source_name in combined["source_db"].unique():
    n = (combined["source_db"] == source_name).sum()
    print("  ", source_name, ":", n, "papers")


# ============================================================
# Part 4: I'm saving the final corpus
# ============================================================

# I save everything to the corpus CSV
output_path = "data/raw/corpus_metadata.csv"
combined.to_csv(output_path, index=False)
print("\nCorpus saved to:", output_path)

# I print the first 10 rows as a quick sense-check
print("\nFirst 10 papers in corpus:")
print(combined[["title", "year", "journal", "source_db"]].head(10).to_string())

# I print a sample abstract to check the text quality
papers_with_abstract = combined[combined["abstract"].str.len() > 50]
if len(papers_with_abstract) > 0:
    print("\nSample abstract (first 300 characters):")
    print(papers_with_abstract["abstract"].iloc[0][:300])

print()
print("=" * 60)
print("PHASE A1 COMPLETE")
print()
print("To get MORE papers (recommended):")
print("  1. Log in to Scopus at SHU library portal")
print("     Search: (\"food insecurity\" OR \"food security\") AND")
print("             (\"financial access\" OR \"post-harvest\" OR")
print("             \"value chain\" OR \"smallholder\")")
print("     Export as CSV → save to: data/raw/scopus_export.csv")
print("  2. Log in to Web of Science at SHU library portal")
print("     Export as Tab-delimited → save to: data/raw/wos_export.txt")
print("  3. Download any relevant PDF papers and put them in: data/raw/pdfs/")
print("     Then run: phase_a2_extract_text_from_pdfs.py")
print("=" * 60)
