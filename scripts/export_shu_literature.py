"""
Export literature records from Scopus and Web of Science.

This script is intentionally simple and supervised.

Why supervised?
SHU, Scopus, and Web of Science may ask for login, MFA, or extra checks.
The script opens the browser for you, but you stay in control of login.

What this script tries to create:
    data/raw/scopus_export.csv
    data/raw/wos_export.txt

If a website blocks access because you do not have your ID card or cannot
finish login today, the script will pause politely. You can continue later.
"""

import os
import time
from urllib.parse import quote


# This is the research search agreed for the project.
SEARCH_QUERY = (
    '("food insecurity" OR "food security") AND '
    '("financial access" OR "post-harvest" OR "value chain" '
    'OR "smallholder" OR "governance")'
)

# These Scopus searches are split into batches.
# This is better than one giant search because one paper rarely covers
# every project theme at once.
SCOPUS_BATCH_1 = (
    'TITLE-ABS-KEY(("food security" OR "food insecurity") '
    'AND (agriculture OR agricultural OR farming OR farmers OR crop OR cereal)) '
    'AND PUBYEAR > 2009'
)

SCOPUS_BATCH_2 = (
    'TITLE-ABS-KEY(("post-harvest" OR postharvest OR "food loss" OR "food losses") '
    'AND (agriculture OR agricultural OR crop OR cereal OR farmers)) '
    'AND PUBYEAR > 2009'
)

SCOPUS_BATCH_3 = (
    'TITLE-ABS-KEY((credit OR loan OR loans OR savings OR finance OR financing '
    'OR microfinance OR microcredit OR "rural finance" OR "mobile money" '
    'OR "digital finance" OR "farm credit" OR "agricultural credit") '
    'AND (farmer OR farmers OR smallholder OR smallholders OR farm OR farms '
    'OR agriculture OR agricultural) '
    'AND ("food security" OR "food insecurity" OR yield OR yields '
    'OR productivity OR income OR livelihood OR livelihoods)) '
    'AND PUBYEAR > 2009'
)

SCOPUS_BATCH_4 = (
    'TITLE-ABS-KEY(("value chain" OR "supply chain" OR distribution OR "market access") '
    'AND (agriculture OR agricultural OR food OR crop OR farmers)) '
    'AND PUBYEAR > 2009'
)

SCOPUS_BATCHES = [
    SCOPUS_BATCH_1,
    SCOPUS_BATCH_2,
    SCOPUS_BATCH_3,
    SCOPUS_BATCH_4,
]


# These folders and files match the rest of the project.
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FOLDER = os.path.join(PROJECT_FOLDER, "data", "raw")
PDF_FOLDER = os.path.join(RAW_FOLDER, "pdfs")
DOWNLOAD_FOLDER = os.path.join(RAW_FOLDER, "browser_downloads")

SCOPUS_FILE = os.path.join(RAW_FOLDER, "scopus_export.csv")
SCOPUS_RIS_FILE = os.path.join(RAW_FOLDER, "scopus_export.ris")
WOS_FILE = os.path.join(RAW_FOLDER, "wos_export.txt")
CORPUS_FILE = os.path.join(RAW_FOLDER, "corpus_metadata.csv")
STRICTLY_ALIGNED_FILE = os.path.join(
    PROJECT_FOLDER,
    "data",
    "processed",
    "strictly_aligned_papers.csv",
)


def get_scopus_search_url(query):
    """Build a Scopus advanced-search URL for one batch query."""
    encoded_query = quote(query)
    return (
        "https://www.scopus.com/results/results.uri?"
        "sort=plf-f&src=s&origin=searchadvanced&s="
        + encoded_query
    )


def make_needed_folders():
    """Create the folders we need before the browser starts."""
    os.makedirs(RAW_FOLDER, exist_ok=True)
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


def wait_for_user(message):
    """
    Pause the script until you are ready.

    This is useful because university login pages are not predictable.
    You can log in manually, solve MFA, or stop if you do not have access today.
    """
    print()
    print(message)
    input("Press Enter here when you are ready to continue...")


def ask_yes_or_no(question):
    """
    Ask a simple yes/no question.

    I use this before opening library portals, because you may not have
    everything needed for SHU login today.
    """
    print()
    print(question)
    answer = input("Type y for yes, or n for no: ")
    answer = answer.strip().lower()

    if answer == "y":
        return True

    return False


def open_page(page, url):
    """Open a page and give it a little time to settle."""
    print("Opening:", url)
    page.goto(url, wait_until="domcontentloaded", timeout=90000)
    time.sleep(3)


def save_download(download, final_path):
    """
    Save a browser download to the exact file name used by the project.

    Some websites give downloads strange names. This keeps our project tidy.
    """
    download.save_as(str(final_path))
    print("Saved:", final_path)


def count_csv_rows(filepath):
    """Count data rows in a CSV without importing pandas."""
    if not os.path.exists(filepath):
        return 0

    row_count = 0

    with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
        first_line = True

        for line in handle:
            if first_line:
                first_line = False
                continue

            if line.strip():
                row_count = row_count + 1

    return row_count


def count_ris_records(filepath):
    """Count records in a RIS file."""
    if not os.path.exists(filepath):
        return 0

    record_count = 0

    with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip() == "ER":
                record_count = record_count + 1

    return record_count


def print_csv_status(label, filepath):
    """Print a small row-count status for a CSV file."""
    if os.path.exists(filepath):
        print(label + ":", count_csv_rows(filepath), "rows")
    else:
        print(label + ": missing")


def get_next_scopus_file(file_extension):
    """
    Pick a Scopus filename that will not replace an older export.

    The first export is scopus_export.csv or scopus_export.ris.
    The next ones are scopus_export_2.csv, scopus_export_3.csv, and so on.
    """
    if file_extension == ".ris":
        first_file = SCOPUS_RIS_FILE
    else:
        first_file = SCOPUS_FILE

    if not os.path.exists(first_file):
        return first_file

    number = 2

    while True:
        filename = "scopus_export_" + str(number) + file_extension
        filepath = os.path.join(RAW_FOLDER, filename)

        if not os.path.exists(filepath):
            return filepath

        number = number + 1


def get_existing_scopus_files():
    """Find every Scopus CSV export already saved in data/raw."""
    scopus_files = []

    if not os.path.exists(RAW_FOLDER):
        return scopus_files

    for filename in os.listdir(RAW_FOLDER):
        if filename.startswith("scopus_export") and filename.endswith(".csv"):
            scopus_files.append(os.path.join(RAW_FOLDER, filename))

    scopus_files.sort()
    return scopus_files


def get_existing_scopus_ris_files():
    """Find every Scopus RIS export already saved in data/raw."""
    scopus_files = []

    if not os.path.exists(RAW_FOLDER):
        return scopus_files

    for filename in os.listdir(RAW_FOLDER):
        if filename.startswith("scopus_export") and filename.endswith(".ris"):
            scopus_files.append(os.path.join(RAW_FOLDER, filename))

    scopus_files.sort()
    return scopus_files


def try_scopus_export(page):
    """
    Try to guide the Scopus export.

    This part uses simple clicks where possible. If Scopus changes its page,
    you can still use the open browser to export manually.
    """
    print()
    print("SCOPUS EXPORT")
    print("Use these Scopus searches as separate batches.")
    print("Export 50 or more records from each batch if Scopus allows it.")
    print()

    batch_number = 1
    for query in SCOPUS_BATCHES:
        print("Batch", batch_number)
        print(query)
        print()
        batch_number = batch_number + 1

    print("Use these filters for every batch:")
    print("Article, English, 2010 onward.")

    open_page(page, "https://www.scopus.com")

    wait_for_user(
        "Please log in to Scopus with SHU if asked. "
        "If you cannot log in today, press Enter and we will continue."
    )

    batch_number = 1
    for query in SCOPUS_BATCHES:
        print()
        print("SCOPUS BATCH", batch_number)
        print(query)
        print("Opening the direct Scopus search URL for this batch.")

        search_url = get_scopus_search_url(query)

        try:
            open_page(page, search_url)
            time.sleep(5)
        except Exception:
            print("I could not open the direct Scopus results URL.")
            print("I will open the Scopus homepage instead.")
            open_page(page, "https://www.scopus.com")

            search_box = page.locator("textarea, input[type='search'], input[type='text']").first

            try:
                search_box.fill(query)
                search_box.press("Enter")
                time.sleep(5)
            except Exception:
                print("I could not control the Scopus search box automatically.")
                print("Please search manually in the browser using this batch query.")

        wait_for_user(
            "In Scopus, apply these filters if you can: Article, English, 2010 onward. "
            "Then choose Export. If CSV is not available, choose RIS or EndNote instead. "
            "For CSV include Title, Abstract, Authors, Source title, Year, DOI."
        )

        use_ris = ask_yes_or_no(
            "Will this Scopus download be RIS or EndNote instead of CSV?"
        )

        if use_ris:
            file_extension = ".ris"
            print("Waiting for the Scopus RIS download.")
        else:
            file_extension = ".csv"
            print("Waiting for the Scopus CSV download.")

        next_scopus_file = get_next_scopus_file(file_extension)

        print("This download will be saved as:")
        print(next_scopus_file)

        try:
            with page.expect_download(timeout=120000) as download_wait:
                wait_for_user("Start this Scopus download in the browser now.")
            download = download_wait.value
            save_download(download, next_scopus_file)
        except Exception:
            print("No Scopus download was captured for this batch.")
            print("That is okay if access is blocked today.")

        if batch_number < len(SCOPUS_BATCHES):
            keep_going = ask_yes_or_no(
                "Do you want to continue to the next Scopus batch?"
            )

            if not keep_going:
                print()
                print("Scopus batch export stopped for now.")
                break

        batch_number = batch_number + 1


def try_wos_export(page):
    """
    Try to guide the Web of Science export.

    WoS often changes button names, so the safest beginner-friendly approach
    is to let you control the final export click while this script catches it.
    """
    print()
    print("WEB OF SCIENCE EXPORT")
    print("Search query:")
    print(SEARCH_QUERY)

    open_page(page, "https://www.webofscience.com")

    wait_for_user(
        "Please log in to Web of Science with SHU if asked. "
        "If you cannot log in today, press Enter and we will continue."
    )

    print("Trying to place the search query into Web of Science.")

    search_box = page.locator("textarea, input[type='search'], input[type='text']").first

    try:
        search_box.fill(SEARCH_QUERY)
        search_box.press("Enter")
        time.sleep(5)
    except Exception:
        print("I could not control the Web of Science search box automatically.")
        print("Please search manually in the browser using the query above.")

    wait_for_user(
        "In Web of Science, export Full Record as Tab-delimited or plain text. "
        "Use UTF-8 if the website gives that option."
    )

    print("Waiting for the Web of Science text download.")
    print("If you already downloaded it manually, place it at:")
    print(WOS_FILE)

    try:
        with page.expect_download(timeout=120000) as download_wait:
            wait_for_user("Start the Web of Science download in the browser now.")
        download = download_wait.value
        save_download(download, WOS_FILE)
    except Exception:
        print("No Web of Science download was captured.")
        print("That is okay if access is blocked today.")


def print_final_status():
    """Show what files are ready and what is still missing."""
    print()
    print("EXPORT STATUS")

    scopus_files = get_existing_scopus_files()
    scopus_ris_files = get_existing_scopus_ris_files()

    if len(scopus_files) > 0:
        print("Scopus CSV files found:")
        for scopus_file in scopus_files:
            print(" ", scopus_file, "-", count_csv_rows(scopus_file), "rows")
        print("Total Scopus CSV rows:", sum(count_csv_rows(path) for path in scopus_files))
    else:
        print("Scopus CSV files still missing.")
        print("Expected names include:")
        print(" ", SCOPUS_FILE)
        print(" ", os.path.join(RAW_FOLDER, "scopus_export_2.csv"))

    if len(scopus_ris_files) > 0:
        print("Scopus RIS files found:")
        for scopus_file in scopus_ris_files:
            print(" ", scopus_file, "-", count_ris_records(scopus_file), "records")
        print("Total Scopus RIS records:", sum(count_ris_records(path) for path in scopus_ris_files))
    else:
        print("Scopus RIS files still missing.")
        print("Expected names include:")
        print(" ", SCOPUS_RIS_FILE)
        print(" ", os.path.join(RAW_FOLDER, "scopus_export_2.ris"))

    if os.path.exists(WOS_FILE):
        print("Web of Science file found:", WOS_FILE)
    else:
        print("Web of Science file still missing:", WOS_FILE)

    print("PDF folder ready:", PDF_FOLDER)
    print_csv_status("Current merged corpus", CORPUS_FILE)
    print_csv_status("Strictly aligned literature", STRICTLY_ALIGNED_FILE)
    print()
    print("When the files are ready, run:")
    print("python3 src/phase_a1_fetch_papers_from_openalex.py")


def main():
    """Run the supervised export workflow."""
    make_needed_folders()

    has_access_today = ask_yes_or_no(
        "Do you have what you need for SHU login today?"
    )

    if not has_access_today:
        print()
        print("No problem. I will not open Scopus or Web of Science today.")
        print_final_status()
        return

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()

            try_scopus_export(page)

            use_wos = ask_yes_or_no(
                "Do you want to try Web of Science now?"
            )

            if use_wos:
                try_wos_export(page)
            else:
                print()
                print("Web of Science skipped for now.")

            wait_for_user("You can review the browser now. Press Enter to close it.")
            browser.close()

    except KeyboardInterrupt:
        print()
        print("The export script was stopped by the user.")
    except ModuleNotFoundError:
        print()
        print("Playwright is not installed in this Python environment yet.")
        print("Install the project requirements before using the browser exporter:")
        print("python3 -m pip install -r requirements.txt")

    print_final_status()


if __name__ == "__main__":
    main()
