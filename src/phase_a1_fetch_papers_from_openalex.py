# ============================================================
# I'm going to fetch all my research papers from OpenAlex
# ============================================================
#
# OpenAlex is a free, open database of academic papers.
# I don't need a university login — anyone can use it.
#
# What I'm doing here:
#   I've written a list of search phrases related to my topic.
#   For each phrase, I ask OpenAlex to give me up to 15 papers.
#   I collect all of them, remove any duplicates, and save
#   everything into a CSV file I can use later.
# ============================================================

# I need the "time" module so I can pause between requests
# (it's polite not to hammer the server too fast)
import time

# I need "requests" so I can send web requests to the OpenAlex API
import requests

# I need pandas so I can work with tables of data
import pandas as pd


# This is the web address of the OpenAlex API
# All my requests will go to this URL
BASE_URL = "https://api.openalex.org/works"

# OpenAlex asks that I include my email in the request header
# This gives me faster access — they call it the "polite pool"
HEADERS = {"User-Agent": "mailto:odekunlejj@gmail.com"}


# ============================================================
# I'm writing a helper function to rebuild abstracts
# ============================================================
# OpenAlex stores abstracts in a strange format called an
# "inverted index". Instead of the text in order, it gives me
# a dictionary where each word maps to a list of positions.
# For example: {"food": [0, 5], "security": [1]} means
# "food" appears at positions 0 and 5, "security" at position 1.
# I need to put the words back in the right order.

def reconstruct_abstract(inverted_index):
    # If there is no abstract data at all, I'll just return an empty string
    if not inverted_index:
        return ""

    # I'll build a new dictionary that maps position number → word
    # This is the opposite of what OpenAlex gives me
    positions = {}

    # I go through each word and all the positions it appears at
    for word in inverted_index:
        # pos_list is the list of positions where this word appears
        pos_list = inverted_index[word]
        # I go through each position in that list
        for pos in pos_list:
            # I store: at position pos, the word is "word"
            positions[pos] = word

    # Now I sort the positions from smallest to largest
    # and join the words back into a sentence
    sorted_positions = sorted(positions)

    # I'll build a list of words in the correct order
    words_in_order = []
    for pos in sorted_positions:
        words_in_order.append(positions[pos])

    # I join all the words with a space between them to make a sentence
    reconstructed = " ".join(words_in_order)

    return reconstructed


# ============================================================
# I'm defining all my search queries here
# ============================================================
# Each string below is one search I'll send to OpenAlex.
# I've grouped them by theme to cover all parts of my project.

SEARCHES = [
    # Theme 1 — Post-harvest loss searches
    '"post-harvest loss" "food security"',
    '"postharvest loss" "cereal" "food"',
    '"post-harvest loss" "sub-Saharan Africa"',

    # Theme 2 — Financial access searches
    '"financial inclusion" "food security"',
    '"credit access" "food security" "smallholder"',
    '"microfinance" "food security"',

    # Theme 3 — Production and climate searches
    '"fertiliser efficiency" "food security"',
    '"fertilizer" "yield gap" "food security"',
    '"climate variability" "cereal yield"',
    '"rainfall" "food availability" "cross-country"',

    # Theme 4 — Machine learning and NLP searches
    '"machine learning" "food security" prediction',
    '"text mining" "food insecurity"',
    '"topic model" "food security"',

    # Theme 5 — Cross-country food availability
    '"cereal food availability" "cross-country"',
    '"food balance sheet" "country-level"',

    # Theme 6 — Infrastructure and governance
    '"infrastructure" "food security" "developing countries"',
    '"governance" "food insecurity" "cross-country"',
]


# ============================================================
# I'm writing the function that searches OpenAlex
# ============================================================

def search_openalex(query, max_results):
    # I'll collect all the results in this empty list
    records = []

    # OpenAlex uses a "cursor" to page through results
    # I start with a star (*) which means "start from the beginning"
    cursor = "*"

    # I keep fetching pages until I have enough results
    while len(records) < max_results:

        # I set up the parameters for my API request
        params = {}
        params["search"]   = query
        params["filter"]   = "language:en,type:article,from_publication_date:2010-01-01"
        params["per-page"] = min(25, max_results - len(records))
        params["cursor"]   = cursor
        params["select"]   = "id,doi,title,abstract_inverted_index,authorships,primary_location,publication_year"
        params["sort"]     = "cited_by_count:desc"

        # I send the request to OpenAlex and wait for the response
        resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)

        # If something went wrong with the request, I print the error and stop
        if resp.status_code != 200:
            print(f"    Error {resp.status_code}: {resp.text[:200]}")
            break

        # I convert the response from JSON format into a Python dictionary
        data = resp.json()

        # "results" is the list of papers OpenAlex returned
        results = data.get("results", [])

        # If there are no results, I stop looking
        if not results:
            break

        # I go through each paper in the results one by one
        for r in results:

            # I rebuild the abstract from the inverted index format
            abstract_text = reconstruct_abstract(r.get("abstract_inverted_index") or {})

            # I need to find the journal name
            # It is buried inside "primary_location" → "source" → "display_name"
            journal = ""

            # I pull out the location information (it might be missing)
            loc = r.get("primary_location") or {}

            # I pull out the source information from the location
            src = loc.get("source") or {}

            # Now I can get the journal name
            journal = src.get("display_name", "")

            # I collect the first three authors' names
            # "authorships" is a list of author objects
            author_list = r.get("authorships") or []

            # I'll build a list of author name strings
            author_names = []

            # I only want the first three authors
            for author_entry in author_list[:3]:
                # Each entry has an "author" object with a "display_name"
                name = author_entry.get("author", {}).get("display_name", "")
                author_names.append(name)

            # I join the author names with a semicolon between them
            authors = "; ".join(author_names)

            # I get the DOI (the unique paper identifier)
            # and remove the "https://doi.org/" prefix if it's there
            doi_raw = r.get("doi") or ""
            doi = doi_raw.replace("https://doi.org/", "")

            # I now build one dictionary (one row) for this paper
            one_paper = {}
            one_paper["title"]       = r.get("title", "")
            one_paper["abstract"]    = abstract_text
            one_paper["authors"]     = authors
            one_paper["journal"]     = journal
            one_paper["year"]        = str(r.get("publication_year", ""))
            one_paper["doi"]         = doi
            one_paper["openalex_id"] = r.get("id", "")

            # I add this paper's data to my growing list
            records.append(one_paper)

        # I check if there is another page of results to fetch
        # OpenAlex puts the next cursor inside "meta"
        meta = data.get("meta", {})
        cursor = meta.get("next_cursor", None)

        # If there is no next cursor, I've reached the last page
        if not cursor:
            break

        # I wait a short moment before fetching the next page
        time.sleep(0.2)

    # I return everything I collected
    return records


# ============================================================
# I'm now running the actual searches
# ============================================================

# I'll collect all results from all searches in one big list
all_records = []

# I go through each search query one by one
for i in range(len(SEARCHES)):
    # I pick the current query
    query = SEARCHES[i]

    # I print progress so I can see what's happening
    print(f"[{i + 1}/{len(SEARCHES)}] {query[:70]}...")

    # I run the search and get back a list of paper dictionaries
    results = search_openalex(query, max_results=15)

    # I print how many papers came back for this query
    print(f"  → {len(results)} records")

    # I add these results to my main list
    all_records.extend(results)

    # I wait half a second between searches to be polite to the server
    time.sleep(0.5)


# ============================================================
# I'm now cleaning up and saving the data
# ============================================================

# I turn my list of dictionaries into a proper pandas table (DataFrame)
df = pd.DataFrame(all_records)

# I make sure the abstract column contains text, not blank values
df["abstract"] = df["abstract"].fillna("").astype(str)

# I make sure the title column contains text, not blank values
df["title"] = df["title"].fillna("").astype(str)

# I make sure the doi column contains text, not blank values
df["doi"] = df["doi"].fillna("").astype(str)

# I record how many rows I had before removing duplicates
before = len(df)

# I remove any papers with no title at all — they're not useful
df = df[df["title"].str.len() > 0]

# I remove duplicate papers by DOI
# If two searches returned the same paper, I only keep it once
df = df.drop_duplicates(subset="doi")

# I reset the row numbers so they go 0, 1, 2, 3... again
df = df.reset_index(drop=True)

# I count how many papers have a real abstract (longer than 50 characters)
has_abstract = (df["abstract"].str.len() > 50).sum()

# I print a summary of what I've collected
print(f"\nTotal fetched: {before} | After dedup: {len(df)} | With abstracts: {has_abstract}")

# I set the output file path
out = "data/raw/corpus_metadata.csv"

# I save the table to a CSV file
df.to_csv(out, index=False)

# I confirm where I saved it
print(f"Saved to {out}\n")

# I print the first 10 rows so I can do a quick sense-check
print(df[["title", "year", "journal"]].head(10).to_string())

# I print one sample abstract so I can check the quality
print("\nSample abstract:")

# I find the papers that have a real abstract
papers_with_abstract = df[df["abstract"].str.len() > 50]

# If I have at least one abstract, I print the first 300 characters of it
if has_abstract > 0:
    sample = papers_with_abstract["abstract"].iloc[0]
    print(sample[:300])
else:
    print("No abstracts found")
