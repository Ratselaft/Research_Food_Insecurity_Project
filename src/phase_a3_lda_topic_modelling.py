# ============================================================
# Phase A3 — TF-IDF + LDA topic modelling on aligned corpus
# ============================================================
#
# What I'm doing in this file:
#   1. Loading ONLY the 328 strictly aligned food insecurity papers
#      (not the full 1,545-paper corpus — noisy papers excluded)
#   2. Cleaning text, detecting bigram phrases (e.g. "post_harvest",
#      "climate_change", "cereal_yield") before LDA
#   3. Sweeping K from 3 to 12 and selecting the K that achieves
#      coherence >= 0.6  (success criterion from research proposal)
#   4. Fitting the final LDA model with that K
#   5. Running TF-IDF keyword analysis alongside LDA
#   6. Saving the coherence curve, mapping template, and keyword CSV
#
# Using strictly aligned papers matters because:
#   — The full corpus (1,545 papers) included tangentially related
#     work (e.g. aflatoxin immunoassay, potato marketing) that
#     diluted LDA topics and produced coherence = 0.368
#   — Restricting to 328 papers that explicitly address food insecurity
#     gives the algorithm a focused, high-quality signal
# ============================================================

import random
import re
import warnings

import numpy as np

from matplotlib_setup import use_project_matplotlib_config

warnings.filterwarnings('ignore')

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaModel, Phrases
from gensim.models.phrases import Phraser

use_project_matplotlib_config()
import matplotlib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

matplotlib.use('Agg')
import matplotlib.pyplot as plt

for resource_name in ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
    nltk.download(resource_name, quiet=True)


# ============================================================
# Step 1: Loading the strictly aligned corpus (328 papers)
# ============================================================
# I load ONLY the papers that Phase A4 classified as strictly
# aligned with food insecurity research. These 328 papers are
# the clean signal; the remaining 1,217 were too tangential.

df = pd.read_csv('data/processed/strictly_aligned_papers.csv')

df['abstract'] = df['abstract'].fillna('').astype(str)
df['title']    = df['title'].fillna('').astype(str)
df['text']     = df['title'] + ' ' + df['abstract']
df['text']     = df['text'].str.strip()
df = df[df['text'].str.len() > 80].reset_index(drop=True)

print(f"Strictly aligned papers loaded: {len(df)}")


# ============================================================
# Step 2: Text preprocessing
# ============================================================
# I remove generic academic language AND food-security boilerplate
# so the LDA has to work harder to find meaningful distinctions.
# Words like "food" and "security" appear in nearly every paper
# in this corpus, so they carry no discriminating power for LDA.

DOMAIN_STOP = {
    # Generic academic boilerplate
    'study', 'paper', 'result', 'analysis', 'data', 'method', 'approach',
    'show', 'find', 'found', 'also', 'may', 'one', 'two', 'three',
    'however', 'significant', 'associated', 'increase', 'effect', 'impact',
    'among', 'within', 'across', 'used', 'use', 'new', 'high', 'low',
    'well', 'large', 'small', 'different', 'important', 'provide', 'include',
    'likely', 'research', 'review', 'literature', 'evidence', 'policy',
    'intervention', 'population', 'sample', 'survey', 'estimate', 'report',
    'factor', 'variable', 'relationship', 'role', 'potential', 'global',
    'national', 'local', 'regional', 'developing', 'developed', 'world',
    'million', 'billion', 'percent', 'percentage', 'rate', 'index',
    # Food insecurity boilerplate — appears in every paper, no LDA value
    'food', 'security', 'insecurity', 'hunger', 'malnutrition', 'nutrition',
    'nutritional', 'fao', 'sdg', 'sustainable', 'development', 'goal',
    'country', 'countries', 'level', 'using', 'based', 'model',
    # Generic quantitative terms
    'datum', 'set', 'using', 'model', 'based', 'case', 'context', 'group',
}

STOP = set(stopwords.words('english')) | DOMAIN_STOP
lem  = WordNetLemmatizer()


def preprocess(text):
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    words  = text.split()
    tokens = []
    for word in words:
        if word in STOP or len(word) <= 2:
            continue
        tokens.append(lem.lemmatize(word))
    return tokens


df['tokens'] = df['text'].apply(preprocess)
print("Preprocessing done.")


# ============================================================
# Step 3: Bigram phrase detection
# ============================================================
# Single words miss important compound concepts. "Post" and "harvest"
# separately appear in many unrelated contexts; "post_harvest" is
# unambiguous. Phrases trains on the token lists and merges pairs
# that appear together more than expected by chance.

bigram_phrases = Phrases(df['tokens'].tolist(), min_count=3, threshold=8)
bigram_model   = Phraser(bigram_phrases)
df['tokens']   = df['tokens'].apply(lambda toks: bigram_model[toks])

# Report the top bigrams detected
bigram_vocab = {k.decode() if isinstance(k, bytes) else k: v
                for k, v in bigram_model.phrasegrams.items()
                if '_' in (k.decode() if isinstance(k, bytes) else k)}
top_bigrams  = sorted(bigram_vocab.items(), key=lambda x: x[1], reverse=True)[:20]
print(f"\nTop bigrams detected (sample of 20):")
for b, score in top_bigrams:
    print(f"  {b:<35} score={score:.1f}")


# ============================================================
# Step 4: Dictionary and bag-of-words corpus
# ============================================================

dictionary = corpora.Dictionary(df['tokens'])

# With 328 papers:
#   no_below=2  → keep words appearing in at least 2 papers (not just 1)
#   no_above=0.75 → remove words in >75% of papers (generic)
dictionary.filter_extremes(no_below=2, no_above=0.75)
bow_corpus = [dictionary.doc2bow(toks) for toks in df['tokens']]

print(f"\nVocabulary: {len(dictionary)} terms  |  Documents: {len(bow_corpus)}")


# ============================================================
# Step 5: Coherence sweep — K = 3 to 12
# ============================================================
# Target: coherence (c_v) >= 0.6  (research proposal success criterion)
# I use 20 passes for the sweep to get stable estimates.

print("\nRunning coherence sweep (K = 3 to 12)...")
print("Target: c_v >= 0.60\n")

scores = {}
for k in range(3, 13):
    lda = LdaModel(
        bow_corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=RANDOM_SEED,
        passes=20,
        alpha='auto',
        eta='auto',
        per_word_topics=True,
    )
    cm = CoherenceModel(
        model=lda,
        texts=df['tokens'].tolist(),
        dictionary=dictionary,
        coherence='c_v',
        processes=1,
    )
    scores[k] = round(cm.get_coherence(), 4)
    flag = '✓ ABOVE TARGET' if scores[k] >= 0.60 else ''
    print(f"  K={k:>2}  coherence={scores[k]}  {flag}")

# ============================================================
# Step 6: Selecting best K
# ============================================================
# Priority: highest coherence among K values that reach >= 0.6
# Fallback: best available K if none reaches 0.6

above_target = {k: v for k, v in scores.items() if v >= 0.60}

if above_target:
    best_k = max(above_target, key=above_target.get)
    print(f"\n✓ Target reached. Best K = {best_k}  (c_v = {scores[best_k]})")
else:
    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]
    print(f"\n⚠ Target c_v >= 0.60 NOT reached.")
    print(f"  Best available K = {best_k}  (c_v = {best_score})")
    print(f"  Using best available. Dissertation should note this limitation.")

print(f"\nAll scores: {scores}")


# ============================================================
# Step 6b: Coherence curve chart
# ============================================================

plt.figure(figsize=(9, 4))
plt.plot(list(scores.keys()), list(scores.values()), marker='o', color='steelblue')
plt.axvline(best_k, color='red', linestyle='--', label=f'Selected K={best_k}')
plt.axhline(0.60, color='green', linestyle=':', alpha=0.7, label='Target c_v = 0.60')
plt.xlabel('Number of topics (K)')
plt.ylabel('Coherence score (c_v)')
plt.title('LDA coherence sweep — 328 strictly aligned food insecurity papers')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/figures/lda_coherence_curve.png', dpi=150)
plt.close()
print("Coherence curve saved → outputs/figures/lda_coherence_curve.png")


# ============================================================
# Step 7: Final LDA model
# ============================================================

print(f"\nFitting final LDA model: K={best_k}, 40 passes...")

lda_final = LdaModel(
    bow_corpus,
    id2word=dictionary,
    num_topics=best_k,
    random_state=RANDOM_SEED,
    passes=40,
    alpha='auto',
    eta='auto',
    per_word_topics=True,
)

print(f"\n=== TOP WORDS PER TOPIC (K={best_k}) ===")
for idx in range(best_k):
    topic_string = lda_final.print_topic(idx, topn=12)
    print(f"\nTopic {idx}:\n  {topic_string}")


# ============================================================
# Step 8: Mapping template
# ============================================================

rows = []
for idx in range(best_k):
    terms    = lda_final.show_topic(idx, topn=10)
    top_words = ', '.join(word for word, _ in terms)
    rows.append({
        'topic_id':        idx,
        'top_words':       top_words,
        'theme_label':     '',
        'proxy_variable':  '',
        'dataset_source':  '',
    })

mapping_df = pd.DataFrame(rows)
mapping_df.to_csv('data/processed/phase_A_theme_variable_mapping.csv', index=False)
print("\nMapping template saved → data/processed/phase_A_theme_variable_mapping.csv")


# ============================================================
# Step 9: TF-IDF keyword analysis (alongside LDA)
# ============================================================
# TF-IDF scores terms by how distinctive they are per paper
# relative to the corpus. Combined with LDA topic membership it
# shows WHICH words best characterise each topic.

from sklearn.feature_extraction.text import TfidfVectorizer

df['clean_text'] = df['tokens'].apply(lambda toks: ' '.join(toks))

tfidf_vec    = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
tfidf_matrix = tfidf_vec.fit_transform(df['clean_text'])
feature_names = np.array(tfidf_vec.get_feature_names_out())

# Global top-30 keywords
mean_tfidf_global = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
top_global_idx    = mean_tfidf_global.argsort()[::-1][:30]
global_kw_df = pd.DataFrame({
    'topic_id':        -1,
    'rank':            range(1, 31),
    'keyword':         feature_names[top_global_idx],
    'mean_tfidf':      mean_tfidf_global[top_global_idx].round(5),
    'n_docs_in_group': len(df),
    'scope':           'corpus-wide',
})

# Per-topic top-15 TF-IDF keywords
def get_dominant_topic(bow_vec):
    probs = lda_final.get_document_topics(bow_vec, minimum_probability=0)
    return max(probs, key=lambda pair: pair[1])[0]

df['dominant_topic'] = [get_dominant_topic(bow) for bow in bow_corpus]

topic_kw_rows = []
for topic_id in range(best_k):
    mask       = df['dominant_topic'] == topic_id
    topic_tfidf = tfidf_matrix[mask.values]
    if topic_tfidf.shape[0] == 0:
        continue
    topic_mean = np.asarray(topic_tfidf.mean(axis=0)).flatten()
    for rank, idx in enumerate(topic_mean.argsort()[::-1][:15], start=1):
        topic_kw_rows.append({
            'topic_id':        topic_id,
            'rank':            rank,
            'keyword':         feature_names[idx],
            'mean_tfidf':      round(float(topic_mean[idx]), 5),
            'n_docs_in_group': int(mask.sum()),
            'scope':           'per-topic',
        })

topic_kw_df  = pd.DataFrame(topic_kw_rows)
tfidf_all    = pd.concat([global_kw_df, topic_kw_df], ignore_index=True)
tfidf_all.to_csv('data/processed/tfidf_top_keywords.csv', index=False)

print(f"\nTF-IDF complete. Top 15 corpus-wide keywords:")
print(global_kw_df[['rank', 'keyword', 'mean_tfidf']].head(15).to_string(index=False))
print("\nTop 5 per-topic TF-IDF keywords:")
for tid in range(best_k):
    top5 = topic_kw_df[topic_kw_df['topic_id'] == tid].head(5)
    print(f"  Topic {tid}: {', '.join(top5['keyword'].tolist())}")
print("TF-IDF keywords saved → data/processed/tfidf_top_keywords.csv")

print(f"\n=== Phase A3 complete. LDA coherence: {scores[best_k]} (K={best_k}) ===")
print(f"{'SUCCESS' if scores[best_k] >= 0.60 else 'BELOW TARGET'}: target was c_v >= 0.60")
