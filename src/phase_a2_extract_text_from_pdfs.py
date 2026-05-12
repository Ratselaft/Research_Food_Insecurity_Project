# ============================================================
# I'm going to read my downloaded PDFs and add them to my corpus
# ============================================================
#
# What I'm doing here:
#   I've downloaded some research papers as PDF files and saved
#   them in my Research Project Repo folder. This script:
#     1. Finds all the PDF files in that folder
#     2. Reads the text out of each one
#     3. Checks if the paper is relevant to my research
#     4. Pulls out the title and abstract
#     5. Adds relevant papers to my corpus CSV file
#
#   I only keep papers about food security, post-harvest loss,
#   financial access, cereal yield, climate and food, or machine
#   learning and food. I skip papers about things like wheat
#   genetics, robotics, or fish biology.
# ============================================================

# I need os to look through folders and check file paths
import os
# I need re for pattern matching when filtering title candidates
import re

# I need pandas to work with my CSV corpus table
import pandas as pd
# I need pdfplumber to open PDF files and pull out the text
import pdfplumber

# ── I'm setting the folder where my PDFs are saved ────────────────────────────
# This now matches Step 1 of the project:
# put new PDF papers inside data/raw/pdfs/
PDF_FOLDER = "data/raw/pdfs"

# This is the older folder that already has some PDFs in this project.
# I keep it here so we do not lose work that was saved before this correction.
OLD_PDF_FOLDER = "data/raw/Peer Review"

# ── I'm setting the path to my existing corpus file ───────────────────────────
CORPUS_FILE = "data/raw/corpus_metadata.csv"

# ── I'm writing a list of words that mean a paper IS relevant ─────────────────
# If a PDF contains at least one of these phrases, I'll keep it
RELEVANT_KEYWORDS = [
    "food insecurity", "food security", "post-harvest", "postharvest",
    "cereal", "food availability", "financial access", "financial inclusion",
    "credit access", "microfinance", "food loss", "yield gap",
    "climate change", "rainfall", "food supply", "smallholder",
    "machine learning", "random forest", "food production",
    "undernourishment", "malnutrition", "cross-country"
]

# ── I'm writing a list of words that mean a paper is NOT relevant ──────────────
# If a PDF contains any of these, I skip it immediately
SKIP_KEYWORDS = [
    "genome", "transcriptome", "rna splicing", "virulence", "fungal",
    "protein source", "meat analogue", "robotic", "grasping",
    "intercropping microbial", "rhizosphere", "wheat breeding trait",
    "cooling blanket", "pot in pot", "fonio culinary", "plant-based meat",
    "lsm8", "exosome", "canopy architecture trait"
]


# ============================================================
# I'm writing a function to read the text out of a PDF
# ============================================================

def read_pdf_text(pdf_path):
    # I'll collect all the text from every page in this variable
    text = ""

    # I try to open the PDF — if something goes wrong, I catch the error
    try:
        # I open the PDF file using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            # I go through every page one by one
            for page in pdf.pages:
                # I try to extract the text from this page
                # extract_text() returns None if the page is just an image
                page_text = page.extract_text()

                # I only add the text if there is actually some text there
                if page_text:
                    # I add this page's text to my growing collection
                    # I put a space at the end so words don't merge across pages
                    text = text + page_text + " "

    # If the PDF can't be read for any reason, I just print a note and move on
    except Exception as e:
        print(f"    Could not read: {e}")

    # I return the full text, removing any extra spaces at the start and end
    return text.strip()


# ============================================================
# I'm writing a function to check if a paper is relevant
# ============================================================

def is_relevant(text):
    # I convert everything to lowercase so my checks work regardless of capital letters
    text_lower = text.lower()

    # First I check the skip list — if any bad word appears, I reject the paper immediately
    for skip_word in SKIP_KEYWORDS:
        if skip_word in text_lower:
            # I found a skip word — this paper is off-topic, so I return False
            return False

    # Now I check the relevant keywords list
    for keyword in RELEVANT_KEYWORDS:
        if keyword in text_lower:
            # I found a relevant keyword — I want to keep this paper
            return True

    # If I get here, none of my keywords matched — I don't keep this paper
    return False


# ============================================================
# I'm writing a function to guess the paper title
# ============================================================

def get_title_from_text(text, filename):
    # I split the text into lines and look only at the first 40 lines,
    # where the actual paper title almost always appears.
    all_lines = text.split("\n")

    title_candidates = []
    for line in all_lines[:40]:
        clean_line = line.strip()

        # Titles are typically between 30 and 250 characters
        if len(clean_line) < 30 or len(clean_line) > 250:
            continue

        lower = clean_line.lower()

        # I skip lines that look like journal headers, author bylines, or DOIs
        # because pdfplumber often picks these up near the top of a PDF.
        if (
            "http" in lower                                           # URL / DOI link
            or "doi.org" in lower                                     # DOI line
            or "et al." in lower                                      # "Author et al." byline
            or "·" in clean_line                                      # middle-dot author separator
            or "|" in clean_line                                      # journal-header pipe separator
            or "@" in clean_line                                      # email address
            or re.search(r"\(\d{4}\)\s*\d+:", clean_line)           # "Journal (2026) 6:86"
            or re.match(r"^\d+\s*$", clean_line)                    # lone page number
            # Two or more names with superscript affiliation numbers (e.g. "Simane1, Berhane2")
            or len(re.findall(r"[A-Z][a-z]+\d+", clean_line)) >= 2
        ):
            continue

        title_candidates.append(clean_line)

    if title_candidates:
        # Pick the longest of the first three good candidates —
        # real titles tend to be the longest descriptive line near the top.
        candidates = title_candidates[:3]
        best_title = candidates[0]
        for candidate in candidates:
            if len(candidate) > len(best_title):
                best_title = candidate
        # A real title starts with an upper-case letter.
        # If the best candidate starts with lowercase it is probably a
        # mid-sentence fragment (e.g. "in ethiopia using auto-regressive…").
        # Fall through to the filename fallback instead.
        if best_title and best_title[0].isupper():
            return best_title

    # If no good line was found, fall back to the PDF filename.
    just_filename = os.path.basename(filename)
    name_without_extension = os.path.splitext(just_filename)[0]
    readable_name = name_without_extension.replace("-", " ").replace("_", " ")
    return readable_name


# ============================================================
# I'm writing a function to pull out the abstract
# ============================================================

def get_abstract_from_text(text):
    # I make a lowercase copy of the text to search in
    text_lower = text.lower()

    # I check if the word "abstract" appears anywhere in the paper
    if "abstract" in text_lower:
        # I find where the word "abstract" starts
        start_position = text_lower.find("abstract")

        # I move past the word "abstract" itself to get to the content
        start_position = start_position + len("abstract")

        # I assume the abstract ends at the very end of the text (for now)
        end_position = len(text)

        # I look for words that typically mark the end of an abstract
        end_markers = ["introduction", "keywords", "1.", "background"]

        # I check each possible end marker
        for marker in end_markers:
            # I look for this marker after the start of the abstract
            pos = text_lower.find(marker, start_position)

            # If I found the marker AND it appears before my current end position
            if pos != -1 and pos < end_position:
                # I update my end position to be here
                end_position = pos

        # I cut out just the abstract text using the start and end positions
        abstract = text[start_position:end_position]

        # I remove extra spaces from both ends
        abstract = abstract.strip()

        # Cap at 3000 characters — enough for any realistic abstract.
        return abstract[:3000]

    # If there's no "abstract" section, I take the first 4000 characters.
    # A longer window gives the scorer in Phase A4 more text to match against,
    # which matters especially for papers whose key terms appear in the intro.
    return text[:4000].strip()


# ============================================================
# Step 1: I'm finding all the PDF files in my folder
# ============================================================

# I let the user know what I'm doing
print("Looking for PDF files...")

# I'll store the full path of every PDF I find in this list
all_pdfs = []

# I look through the new folder and the older folder.
# This is simple on purpose: one folder path at a time.
pdf_folders = [PDF_FOLDER, OLD_PDF_FOLDER]

for folder in pdf_folders:
    # If a folder does not exist yet, I skip it.
    if not os.path.exists(folder):
        continue

    # I look through every file in this folder.
    for filename in os.listdir(folder):
        # I only care about files that end in .pdf
        if filename.endswith(".pdf"):
            # I build the full path to this file
            full_path = os.path.join(folder, filename)
            # I add it to my list
            all_pdfs.append(full_path)

# I tell the user how many PDFs I found
print(f"Found {len(all_pdfs)} PDF files in the folder")


# ============================================================
# Step 2: I'm loading my existing corpus so I don't add duplicates
# ============================================================

# I read the existing corpus CSV into a table
existing_corpus = pd.read_csv(CORPUS_FILE)

# I remove any rows that were added by a previous PDF run.
# This makes the script safe to re-run: old PDF entries (which may have had
# broken title / abstract extraction) are replaced with fresh ones.
before_count = len(existing_corpus)
existing_corpus = existing_corpus[existing_corpus["source_db"] != "PDF"].copy()
removed_count = before_count - len(existing_corpus)
if removed_count > 0:
    print(f"Removed {removed_count} stale PDF entries from corpus (will re-add fresh)")

# I make sure the abstract column has no blank values
existing_corpus["abstract"] = existing_corpus["abstract"].fillna("").astype(str)

# I tell the user how many papers are already in the corpus
print(f"Existing corpus (OpenAlex + Scopus) has {len(existing_corpus)} papers")


# ============================================================
# Step 3: I'm reading each PDF, checking relevance, adding if yes
# ============================================================

# I let the user know I'm starting this step
print("\nReading PDFs...")

# I'll collect any new papers I want to add in this list
new_papers = []
new_titles_this_run = []

# I go through each PDF file one by one
for pdf_path in all_pdfs:
    # I get just the file name (without the full folder path)
    filename = os.path.basename(pdf_path)

    # I print the first 60 characters of the filename so I can see progress
    print(f"\n  Reading: {filename[:60]}...")

    # I read all the text out of this PDF
    full_text = read_pdf_text(pdf_path)

    # If the text is very short, this is probably a scan (image-only PDF)
    # I can't extract text from images, so I skip it
    if len(full_text) < 200:
        print("    Skipped — too little text (might be a scan or image PDF)")
        continue

    # I check whether this paper is about my research topics
    if not is_relevant(full_text):
        print("    Skipped — not relevant to food insecurity / cereal research")
        continue

    # I try to extract the title from the paper text
    title = get_title_from_text(full_text, filename)

    # I try to extract the abstract from the paper text
    abstract = get_abstract_from_text(full_text)

    # I check whether this paper is already in my corpus
    # I compare the first 50 characters of the title (case-insensitive)
    already_there = False
    for existing_title in existing_corpus["title"].astype(str):
        # I check if the new title appears inside any existing title
        if title.lower()[:50] in existing_title.lower():
            already_there = True
            break

    for new_title in new_titles_this_run:
        if title.lower()[:50] in new_title.lower():
            already_there = True
            break

    # If this paper is already in the corpus, I skip it
    if already_there:
        print("    Skipped — already in corpus")
        continue

    # I let the user know this paper passed all my checks
    print(f"    Kept: {title[:70]}")

    # I build a dictionary with all the information for this paper
    paper_info = {}
    paper_info["title"]       = title
    paper_info["abstract"]    = abstract
    paper_info["authors"]     = ""           # I don't extract authors from PDFs
    paper_info["journal"]     = ""           # I don't extract journal names from PDFs
    paper_info["year"]        = "2026"       # Most of these are 2026 papers
    paper_info["doi"]         = ""
    paper_info["openalex_id"] = ""
    paper_info["source_db"]   = "PDF"        # I mark these as manually added from PDFs

    # I add this paper's info to my list of new papers
    new_papers.append(paper_info)
    new_titles_this_run.append(title)


# ============================================================
# Step 4: I'm saving the updated corpus
# ============================================================

# I tell the user how many new papers I found
print(f"\n{len(new_papers)} new relevant papers found in your PDFs")

# I only update the corpus if I actually found new papers
if new_papers:
    # I turn my list of dictionaries into a pandas table
    new_df = pd.DataFrame(new_papers)

    # I join the new papers onto the end of the existing corpus
    updated_corpus = pd.concat([existing_corpus, new_df], ignore_index=True)

    # I save the updated corpus back to the CSV file
    updated_corpus.to_csv(CORPUS_FILE, index=False)

    # I tell the user the new size of the corpus
    print(f"Corpus updated: {len(existing_corpus)} → {len(updated_corpus)} papers")
    print(f"Saved to {CORPUS_FILE}")

else:
    # If I found nothing new, I let the user know
    print("No new papers added — corpus unchanged")

# I'm done!
print("\nDone.")
