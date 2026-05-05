"""Phase A — NLP Factor Discovery: preprocessing, LDA coherence sweep, final model."""

import random, numpy as np, re, warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

import pandas as pd, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim.corpora as corpora
from gensim.models import LdaModel, CoherenceModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for r in ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
    nltk.download(r, quiet=True)

# ── 1. Load corpus ────────────────────────────────────────────────────────────
df = pd.read_csv('data/raw/corpus_metadata.csv')
df['abstract'] = df['abstract'].fillna('').astype(str)
df['title']    = df['title'].fillna('').astype(str)
df['text']     = (df['title'] + ' ' + df['abstract']).str.strip()
df = df[df['text'].str.len() > 80].reset_index(drop=True)
print(f"Papers in corpus: {len(df)}")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
DOMAIN_STOP = {
    'food', 'security', 'insecurity', 'study', 'paper', 'result', 'analysis',
    'data', 'country', 'countries', 'level', 'using', 'based', 'model', 'method',
    'approach', 'show', 'find', 'found', 'also', 'may', 'one', 'two', 'three',
    'however', 'significant', 'associated', 'increase', 'effect', 'impact',
    'among', 'within', 'across', 'used', 'use', 'new', 'high', 'low', 'well',
    'large', 'small', 'different', 'important', 'provide', 'include', 'likely'
}
STOP = set(stopwords.words('english')) | DOMAIN_STOP
lem  = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'[^a-z\s]', '', text.lower())
    return [lem.lemmatize(t) for t in text.split() if t not in STOP and len(t) > 2]

df['tokens'] = df['text'].apply(preprocess)
print("Preprocessing done.")

# ── 3. Dictionary & BoW corpus ────────────────────────────────────────────────
dictionary  = corpora.Dictionary(df['tokens'])
dictionary.filter_extremes(no_below=3, no_above=0.85)
bow_corpus  = [dictionary.doc2bow(tok) for tok in df['tokens']]
print(f"Vocabulary size: {len(dictionary)}  |  Documents: {len(bow_corpus)}")

# ── 4. Coherence sweep K=4..10 ────────────────────────────────────────────────
print("\nRunning coherence sweep (K = 4 to 10)...")
scores = {}
for k in range(4, 11):
    lda = LdaModel(
        bow_corpus, id2word=dictionary, num_topics=k,
        random_state=RANDOM_SEED, passes=10, alpha='auto', per_word_topics=True
    )
    cm = CoherenceModel(
        model=lda, texts=df['tokens'].tolist(),
        dictionary=dictionary, coherence='c_v', processes=1
    )
    scores[k] = round(cm.get_coherence(), 4)
    print(f"  K={k}  coherence={scores[k]}")

# The sweep finds the mathematically highest coherence score.
# We print it so we can report it in the dissertation.
auto_best_k = max(scores, key=scores.get)
print(f"\nHighest coherence K = {auto_best_k}  (c_v = {scores[auto_best_k]})")
print(f"All scores: {scores}")

# We deliberately choose K=9 instead of the automatic best.
# Reason: K=9 produces nine topics that map one-to-one with the five
# theoretical blocks in our research framework (PHL, Finance, Climate,
# Production, Governance). K=4 collapses these into broad themes that
# are too general to drive the variable-mapping in Phase B/C.
# This follows Mimno et al. (2011): coherence and interpretability
# do not always agree; interpretability takes priority here.
best_k = 9
print(f"Using K = {best_k}  (coherence = {scores[best_k]})  — chosen for topic interpretability")

# ── 5. Coherence curve plot ───────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(list(scores.keys()), list(scores.values()), marker='o', color='steelblue')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.xlabel('Number of topics (K)')
plt.ylabel('Coherence score (c_v)')
plt.title('LDA coherence by number of topics')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/figures/lda_coherence_curve.png', dpi=150)
print("Coherence curve saved → outputs/figures/lda_coherence_curve.png")

# ── 6. Final model ────────────────────────────────────────────────────────────
lda_final = LdaModel(
    bow_corpus, id2word=dictionary, num_topics=best_k,
    random_state=RANDOM_SEED, passes=20, alpha='auto', per_word_topics=True
)

print(f"\n=== TOP WORDS PER TOPIC (K={best_k}) ===")
for idx, topic in lda_final.print_topics(num_words=12):
    print(f"\nTopic {idx}:\n  {topic}")

# ── 7. Save mapping template ──────────────────────────────────────────────────
rows = []
for idx in range(best_k):
    terms   = lda_final.show_topic(idx, topn=10)
    top_words = ', '.join(w for w, _ in terms)
    rows.append({
        'topic_id': idx, 'top_words': top_words,
        'theme_label': '', 'proxy_variable': '', 'dataset_source': ''
    })

pd.DataFrame(rows).to_csv('data/processed/phase_A_theme_variable_mapping.csv', index=False)
print("\nMapping template saved → data/processed/phase_A_theme_variable_mapping.csv")
print("\nPhase A complete.")
