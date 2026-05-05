# ============================================================
# Add your downloaded PDFs to the corpus
# ============================================================
#
# What this file does:
#   You have PDF papers saved in your Research Project Repo folder.
#   This script reads each PDF, pulls out the text, checks if the
#   paper is relevant to our research topics, and adds it to the
#   corpus CSV we already built in Phase A.
#
#   We only keep papers that talk about things like:
#   food security, post-harvest loss, financial access,
#   cereal yield, climate and food, machine learning and food.
#
#   Papers about things like wheat genetics, robotics, or fish
#   biology are skipped automatically.
# ============================================================

import os
import pdfplumber   # reads PDFs and pulls out the text
import pandas as pd

# ── Where to look for PDFs ────────────────────────────────────────────────────
PDF_FOLDER = "/Users/productguru/Documents/Research Project Repo"

# ── Where our existing corpus lives ──────────────────────────────────────────
CORPUS_FILE = "data/raw/corpus_metadata.csv"

# ── Keywords that tell us a paper IS relevant to the project ─────────────────
# If a PDF contains at least one of these words, we keep it.
RELEVANT_KEYWORDS = [
    "food insecurity", "food security", "post-harvest", "postharvest",
    "cereal", "food availability", "financial access", "financial inclusion",
    "credit access", "microfinance", "food loss", "yield gap",
    "climate change", "rainfall", "food supply", "smallholder",
    "machine learning", "random forest", "food production",
    "undernourishment", "malnutrition", "cross-country"
]

# ── Keywords that tell us a paper is NOT relevant — skip these ───────────────
SKIP_KEYWORDS = [
    "genome", "transcriptome", "rna splicing", "virulence", "fungal",
    "protein source", "meat analogue", "robotic", "grasping",
    "intercropping microbial", "rhizosphere", "wheat breeding trait",
    "cooling blanket", "pot in pot", "fonio culinary", "plant-based meat",
    "lsm8", "exosome", "canopy architecture trait"
]

def read_pdf_text(pdf_path):
    """
    Opens a PDF and reads the text from every page.
    Returns all the text joined together as one long string.
    If the PDF can't be read, returns an empty string.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text from this page (might be None if page is an image)
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        print(f"    Could not read: {e}")
    return text.strip()

def is_relevant(text):
    """
    Checks if a paper is relevant to our research project.
    First checks for skip keywords (not relevant), then checks
    for relevant keywords. Returns True if we should keep it.
    """
    text_lower = text.lower()

    # Check if this paper is clearly off-topic
    for skip_word in SKIP_KEYWORDS:
        if skip_word in text_lower:
            return False

    # Check if this paper talks about our research topics
    for keyword in RELEVANT_KEYWORDS:
        if keyword in text_lower:
            return True

    return False

def get_title_from_text(text, filename):
    """
    Tries to guess the paper title from the first few lines of text.
    If we can't figure it out, we use the filename instead.
    """
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 20]
    if lines:
        # The title is usually one of the first few non-empty lines
        # Pick the longest of the first 3 lines as the likely title
        candidates = lines[:3]
        return max(candidates, key=len)
    # Fall back to the filename without the .pdf extension
    return os.path.splitext(os.path.basename(filename))[0].replace("-", " ").replace("_", " ")

def get_abstract_from_text(text):
    """
    Tries to find the abstract section inside the paper text.
    Most papers have the word 'abstract' followed by the summary.
    If we can't find it, we take the first 1500 characters as a summary.
    """
    text_lower = text.lower()

    # Look for the word "abstract" and grab the text after it
    if "abstract" in text_lower:
        start = text_lower.find("abstract") + len("abstract")
        # The abstract usually ends before "introduction" or "keywords"
        end = len(text)
        for end_marker in ["introduction", "keywords", "1.", "background"]:
            pos = text_lower.find(end_marker, start)
            if pos != -1 and pos < end:
                end = pos
        abstract = text[start:end].strip()
        # Keep only the first 2000 characters (abstracts are not that long)
        return abstract[:2000]

    # No abstract section found — use the first chunk of text
    return text[:1500].strip()

# ── Step 1: Find all PDF files ────────────────────────────────────────────────
print("Looking for PDF files...")
all_pdfs = []
for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        full_path = os.path.join(PDF_FOLDER, filename)
        all_pdfs.append(full_path)

print(f"Found {len(all_pdfs)} PDF files in the folder")

# ── Step 2: Load the existing corpus so we don't add duplicates ───────────────
existing_corpus = pd.read_csv(CORPUS_FILE)
existing_corpus["abstract"] = existing_corpus["abstract"].fillna("").astype(str)
print(f"Existing corpus has {len(existing_corpus)} papers")

# ── Step 3: Read each PDF, check if relevant, add if yes ─────────────────────
print("\nReading PDFs...")
new_papers = []

for pdf_path in all_pdfs:
    filename = os.path.basename(pdf_path)
    print(f"\n  Reading: {filename[:60]}...")

    # Read the text out of the PDF
    full_text = read_pdf_text(pdf_path)

    if len(full_text) < 200:
        print("    Skipped — too little text (might be a scan or image PDF)")
        continue

    # Check if this paper is relevant to the project
    if not is_relevant(full_text):
        print("    Skipped — not relevant to food insecurity / cereal research")
        continue

    # Pull out the title and abstract
    title    = get_title_from_text(full_text, filename)
    abstract = get_abstract_from_text(full_text)

    # Check we are not adding a paper that's already in the corpus
    already_there = any(
        title.lower()[:50] in existing_title.lower()
        for existing_title in existing_corpus["title"].astype(str)
    )
    if already_there:
        print("    Skipped — already in corpus")
        continue

    print(f"    Kept: {title[:70]}")
    new_papers.append({
        "title":       title,
        "abstract":    abstract,
        "authors":     "",        # we don't extract authors for now
        "journal":     "",        # we don't extract journal for now
        "year":        "2026",    # most of these are 2026 papers
        "doi":         "",
        "openalex_id": "",
        "source":      "pdf_manual",   # marks these as manually added
    })

# ── Step 4: Add new papers to the corpus and save ─────────────────────────────
print(f"\n{len(new_papers)} new relevant papers found in your PDFs")

if new_papers:
    new_df = pd.DataFrame(new_papers)
    updated_corpus = pd.concat([existing_corpus, new_df], ignore_index=True)
    updated_corpus.to_csv(CORPUS_FILE, index=False)
    print(f"Corpus updated: {len(existing_corpus)} → {len(updated_corpus)} papers")
    print(f"Saved to {CORPUS_FILE}")
else:
    print("No new papers added — corpus unchanged")

print("\nDone.")
