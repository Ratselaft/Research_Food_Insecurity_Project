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

# I need pandas to work with my CSV corpus table
import pandas as pd
# I need pdfplumber to open PDF files and pull out the text
import pdfplumber

# ── I'm setting the folder where my PDFs are saved ────────────────────────────
PDF_FOLDER = "/Users/productguru/Documents/Research Project Repo"

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
    # I split the text into lines and remove any short or blank lines
    all_lines = text.split("\n")

    # I'll collect only the lines that are long enough to be a title
    long_lines = []
    for line in all_lines:
        # I remove extra spaces from both ends of the line
        clean_line = line.strip()
        # I only keep lines that are longer than 20 characters
        if len(clean_line) > 20:
            long_lines.append(clean_line)

    # If I found any long lines, I pick the best one as the title
    if long_lines:
        # I look at the first three long lines
        candidates = long_lines[:3]

        # I pick the longest one — titles are usually the longest heading near the top
        best_title = candidates[0]
        for candidate in candidates:
            if len(candidate) > len(best_title):
                best_title = candidate

        return best_title

    # If I couldn't find any decent lines, I use the filename instead
    # I get just the file name without the folder path
    just_filename = os.path.basename(filename)

    # I remove the .pdf extension
    name_without_extension = os.path.splitext(just_filename)[0]

    # I replace dashes and underscores with spaces to make it more readable
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

        # Abstracts are never more than 2000 characters — I trim to be safe
        return abstract[:2000]

    # If there's no "abstract" section, I just take the first 1500 characters
    return text[:1500].strip()


# ============================================================
# Step 1: I'm finding all the PDF files in my folder
# ============================================================

# I let the user know what I'm doing
print("Looking for PDF files...")

# I'll store the full path of every PDF I find in this list
all_pdfs = []

# I look through every file in my research folder
for filename in os.listdir(PDF_FOLDER):
    # I only care about files that end in .pdf
    if filename.endswith(".pdf"):
        # I build the full path to this file
        full_path = os.path.join(PDF_FOLDER, filename)
        # I add it to my list
        all_pdfs.append(full_path)

# I tell the user how many PDFs I found
print(f"Found {len(all_pdfs)} PDF files in the folder")


# ============================================================
# Step 2: I'm loading my existing corpus so I don't add duplicates
# ============================================================

# I read the existing corpus CSV into a table
existing_corpus = pd.read_csv(CORPUS_FILE)

# I make sure the abstract column has no blank values
existing_corpus["abstract"] = existing_corpus["abstract"].fillna("").astype(str)

# I tell the user how many papers are already in the corpus
print(f"Existing corpus has {len(existing_corpus)} papers")


# ============================================================
# Step 3: I'm reading each PDF, checking relevance, adding if yes
# ============================================================

# I let the user know I'm starting this step
print("\nReading PDFs...")

# I'll collect any new papers I want to add in this list
new_papers = []

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
    paper_info["source"]      = "pdf_manual" # I mark these as manually added from PDFs

    # I add this paper's info to my list of new papers
    new_papers.append(paper_info)


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
