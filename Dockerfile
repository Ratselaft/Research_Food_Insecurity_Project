# ─────────────────────────────────────────────────────────────────────────────
# Food Insecurity NLP/ML Pipeline — Reproducible Docker Image
# Python 3.11.8 — mirrors the local venv used during development
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System packages needed by the Python stack
#   build-essential + gcc  → compile gensim / scipy C extensions
#   libgomp1               → OpenMP threading required by XGBoost
#   libfreetype6-dev       → Matplotlib font rendering
#   curl                   → Playwright browser download
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
        libfreetype6-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer independently.
# If only src/ changes, pip install is skipped on rebuild — much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Playwright browser (used by Phase A1 for supervised PDF export) ───────────
RUN playwright install --with-deps chromium

# ── NLTK corpora (downloaded once at build time, not at runtime) ──────────────
RUN python -c "\
import nltk; \
nltk.download('stopwords', quiet=True); \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('averaged_perceptron_tagger', quiet=True); \
nltk.download('averaged_perceptron_tagger_eng', quiet=True)"

# ── Project source files ──────────────────────────────────────────────────────
# data/raw/ is excluded via .dockerignore — it is mounted as a volume at runtime
# so large raw data files never bloat the image.
COPY src/            src/
COPY scripts/        scripts/
COPY notebooks/      notebooks/
COPY data/processed/ data/processed/
COPY outputs/        outputs/

# Create directories that the pipeline writes into
RUN mkdir -p data/raw data/raw/"Peer Review" outputs/figures outputs/tables outputs/narrative outputs/models reports

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY run_pipeline.sh .
RUN chmod +x run_pipeline.sh

ENTRYPOINT ["./run_pipeline.sh"]
