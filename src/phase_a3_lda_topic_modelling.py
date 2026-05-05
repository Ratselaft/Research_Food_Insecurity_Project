# ============================================================
# I'm running the full Phase A NLP pipeline here
# ============================================================
#
# What I'm doing in this file:
#   1. Loading my corpus of 217 research papers
#   2. Cleaning up the text (removing common words, reducing
#      words to their base form)
#   3. Running LDA topic modelling to find hidden themes
#   4. Sweeping K from 4 to 10 to find the best number of topics
#   5. Choosing K=9 (best tradeoff for my project)
#   6. Saving a chart of the coherence scores
#   7. Saving a mapping table for the 9 topics
# ============================================================

# I need random and numpy to control randomness
# Setting a seed means I get the same results every time I run this
import random
import numpy as np

# I need re to clean up text using pattern matching
import re

# I need warnings to suppress any annoying warning messages
import warnings
warnings.filterwarnings('ignore')

# I set a fixed random seed so my results don't change between runs
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# I need pandas to load and work with my corpus table
import pandas as pd

# I need nltk for natural language processing tools
import nltk

# I need the stopwords list — common English words like "the", "a", "is"
from nltk.corpus import stopwords

# I need WordNetLemmatizer to reduce words to their base form
# e.g. "running" → "run", "studies" → "study"
from nltk.stem import WordNetLemmatizer

# I need gensim tools to build the dictionary and corpus for LDA
import gensim.corpora as corpora

# I need the LDA model and coherence model from gensim
from gensim.models import LdaModel
from gensim.models import CoherenceModel

# I need matplotlib to draw charts
import matplotlib
matplotlib.use('Agg')  # I use 'Agg' so charts save to files without needing a screen
import matplotlib.pyplot as plt

# I download the NLTK data files I need (quietly, without extra printout)
for resource_name in ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
    nltk.download(resource_name, quiet=True)


# ============================================================
# Step 1: I'm loading my corpus of papers
# ============================================================

# I read the corpus CSV into a pandas table
df = pd.read_csv('data/raw/corpus_metadata.csv')

# I make sure the abstract column has text, not blank values
df['abstract'] = df['abstract'].fillna('').astype(str)

# I make sure the title column has text, not blank values
df['title'] = df['title'].fillna('').astype(str)

# I combine the title and abstract into one big text column
# This gives the topic model more text to work with per paper
df['text'] = df['title'] + ' ' + df['abstract']

# I remove any extra spaces from both ends
df['text'] = df['text'].str.strip()

# I remove papers where the combined text is too short to be useful
df = df[df['text'].str.len() > 80]

# I reset the row numbers so they go 0, 1, 2, 3... cleanly
df = df.reset_index(drop=True)

# I print how many papers I have
print(f"Papers in corpus: {len(df)}")


# ============================================================
# Step 2: I'm cleaning up the text
# ============================================================

# I write a set of domain-specific words I want to remove
# These words are so common in food security research that they
# don't help distinguish between topics
DOMAIN_STOP = {
    'food', 'security', 'insecurity', 'study', 'paper', 'result', 'analysis',
    'data', 'country', 'countries', 'level', 'using', 'based', 'model', 'method',
    'approach', 'show', 'find', 'found', 'also', 'may', 'one', 'two', 'three',
    'however', 'significant', 'associated', 'increase', 'effect', 'impact',
    'among', 'within', 'across', 'used', 'use', 'new', 'high', 'low', 'well',
    'large', 'small', 'different', 'important', 'provide', 'include', 'likely'
}

# I get the standard English stopwords from nltk (words like "the", "a", "is")
standard_stopwords = set(stopwords.words('english'))

# I combine my domain stopwords with the standard ones
STOP = standard_stopwords | DOMAIN_STOP

# I create a lemmatizer object — I'll use this to reduce words to their root
lem = WordNetLemmatizer()


def preprocess(text):
    # I convert the whole text to lowercase
    text = text.lower()

    # I remove everything that isn't a letter or a space
    # The [^a-z\s] pattern means "anything that is not a-z or whitespace"
    text = re.sub(r'[^a-z\s]', '', text)

    # I split the text into individual words
    words = text.split()

    # I'll collect my clean, useful words in this list
    clean_tokens = []

    # I go through each word one by one
    for word in words:
        # I skip the word if it's in my stopword list
        if word in STOP:
            continue
        # I skip the word if it's shorter than 3 characters (not meaningful)
        if len(word) <= 2:
            continue
        # I reduce the word to its base form (lemmatise it)
        lemma = lem.lemmatize(word)
        # I add the cleaned word to my list
        clean_tokens.append(lemma)

    return clean_tokens


# I apply my preprocess function to every paper in my corpus
df['tokens'] = df['text'].apply(preprocess)

# I let the user know this step is done
print("Preprocessing done.")


# ============================================================
# Step 3: I'm building the dictionary and bag-of-words corpus
# ============================================================

# I build a dictionary that maps every unique word to a number
# gensim needs this to work with the text
dictionary = corpora.Dictionary(df['tokens'])

# I filter out words that are too rare or too common
# no_below=3 means: remove any word that appears in fewer than 3 papers
# no_above=0.85 means: remove any word that appears in more than 85% of papers
dictionary.filter_extremes(no_below=3, no_above=0.85)

# I convert each paper's tokens into a bag-of-words format
# This is a list of (word_id, word_count) pairs for each paper
bow_corpus = []
for token_list in df['tokens']:
    bow_vector = dictionary.doc2bow(token_list)
    bow_corpus.append(bow_vector)

# I print the vocabulary size and number of documents
print(f"Vocabulary size: {len(dictionary)}  |  Documents: {len(bow_corpus)}")


# ============================================================
# Step 4: I'm running the coherence sweep (K = 4 to 10)
# ============================================================
# I test different numbers of topics to find the best K
# "Coherence" measures how well the top words in each topic
# make sense together — higher is better

print("\nRunning coherence sweep (K = 4 to 10)...")

# I'll store the coherence score for each K in this dictionary
scores = {}

# I try each value of K from 4 to 10 (inclusive)
for k in range(4, 11):
    # I train an LDA model with this many topics
    lda = LdaModel(
        bow_corpus,          # my bag-of-words corpus
        id2word=dictionary,  # my word dictionary
        num_topics=k,        # number of topics to find
        random_state=RANDOM_SEED,  # I keep this fixed so results don't change
        passes=10,           # I go through the data 10 times (more = more accurate)
        alpha='auto',        # I let gensim choose the best alpha automatically
        per_word_topics=True # I track topic assignments per word
    )

    # I create a coherence model to measure the quality of this LDA model
    cm = CoherenceModel(
        model=lda,
        texts=df['tokens'].tolist(),  # my list of token lists
        dictionary=dictionary,
        coherence='c_v',  # I use the c_v coherence measure (most common)
        processes=1       # I use only 1 process to avoid multiprocessing errors
    )

    # I get the coherence score and round it to 4 decimal places
    coherence_score = cm.get_coherence()
    scores[k] = round(coherence_score, 4)

    # I print the result for this K
    print(f"  K={k}  coherence={scores[k]}")


# ============================================================
# Step 5: I'm choosing the best K
# ============================================================

# I find which K had the highest coherence score automatically
auto_best_k = max(scores, key=scores.get)
print(f"\nHighest coherence K = {auto_best_k}  (c_v = {scores[auto_best_k]})")
print(f"All scores: {scores}")

# I deliberately choose K=9 instead of the automatic best.
# Reason: K=9 produces nine topics that map one-to-one with the five
# theoretical blocks in my research framework (PHL, Finance, Climate,
# Production, Governance). K=4 collapses these into broad themes that
# are too general to drive the variable-mapping in Phase B/C.
# This follows Mimno et al. (2011): coherence and interpretability
# do not always agree; interpretability takes priority here.
best_k = 9
print(f"Using K = {best_k}  (coherence = {scores[best_k]})  — chosen for topic interpretability")


# ============================================================
# Step 5b: I'm saving the coherence curve chart
# ============================================================

# I create a new figure with a reasonable size
plt.figure(figsize=(8, 4))

# I pull the K values (4, 5, 6, 7, 8, 9, 10) into a list
k_values = list(scores.keys())

# I pull the coherence scores into a matching list
coherence_values = list(scores.values())

# I draw a line connecting all the K vs coherence points
plt.plot(k_values, coherence_values, marker='o', color='steelblue')

# I draw a red dashed vertical line at my chosen K
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')

# I label the axes
plt.xlabel('Number of topics (K)')
plt.ylabel('Coherence score (c_v)')
plt.title('LDA coherence by number of topics')
plt.legend()
plt.tight_layout()

# I save the chart as a PNG file
plt.savefig('outputs/figures/lda_coherence_curve.png', dpi=150)

# I let the user know where I saved it
print("Coherence curve saved → outputs/figures/lda_coherence_curve.png")


# ============================================================
# Step 6: I'm fitting the final LDA model with K=9
# ============================================================

# I train the final LDA model using my chosen K
lda_final = LdaModel(
    bow_corpus,
    id2word=dictionary,
    num_topics=best_k,     # I use my chosen K=9
    random_state=RANDOM_SEED,
    passes=20,             # I use 20 passes for the final model (more accurate than sweep)
    alpha='auto',
    per_word_topics=True
)

# I print the top 12 words for each of the 9 topics
print(f"\n=== TOP WORDS PER TOPIC (K={best_k}) ===")

# I loop through all topics and print their top words
for idx in range(best_k):
    # lda_final.print_topic gives me a formatted string like "0.05*word + 0.04*word2 ..."
    topic_string = lda_final.print_topic(idx, topn=12)
    print(f"\nTopic {idx}:\n  {topic_string}")


# ============================================================
# Step 7: I'm saving the topic mapping template
# ============================================================

# I'll collect one row per topic in this list
rows = []

# I go through all 9 topics
for idx in range(best_k):
    # I get the top 10 words for this topic as a list of (word, probability) pairs
    terms = lda_final.show_topic(idx, topn=10)

    # I pull out just the words (not the probabilities)
    word_list = []
    for word, probability in terms:
        word_list.append(word)

    # I join them into a comma-separated string
    top_words = ', '.join(word_list)

    # I build a dictionary for this row
    row = {}
    row['topic_id']   = idx
    row['top_words']  = top_words
    row['theme_label']    = ''  # I'll fill this in manually after reviewing the topics
    row['proxy_variable'] = ''  # I'll fill this in manually too
    row['dataset_source'] = ''  # I'll fill this in manually too

    # I add this row to my list
    rows.append(row)

# I turn the list of rows into a pandas table
mapping_df = pd.DataFrame(rows)

# I save the table to a CSV file
mapping_df.to_csv('data/processed/phase_A_theme_variable_mapping.csv', index=False)

# I let the user know where I saved it
print("\nMapping template saved → data/processed/phase_A_theme_variable_mapping.csv")
print("\nPhase A complete.")
