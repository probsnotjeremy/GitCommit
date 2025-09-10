#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset



# --------------------------
# Global vars
# --------------------------

ds = load_dataset("sentence-transformers/natural-questions")
dataset_split = ds["train"]
stopwords = {"the", "a", "and", "is", "to", "of", "in", "on"}

# --------------------------
# WordNet setup
# --------------------------
try:
    from nltk.corpus import wordnet
    _ = wordnet.synsets("car")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    from nltk.corpus import wordnet

# --------------------------
# SentenceTransformer setup
# --------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# --------------------------
# Text utilities
# --------------------------
def clean_words(text):
    return [w.lower() for w in text.split() if w.lower() not in stopwords]

def expand_with_synonyms(words):
    expanded = set(words)
    for w in words:
        for syn in wordnet.synsets(w):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().lower().replace("_", " "))
    return expanded

def overlap_percent(answer, response, use_synonyms=True):
    a_words = clean_words(answer)
    r_words = clean_words(response)

    if use_synonyms:
        a_set = expand_with_synonyms(a_words)
        r_set = expand_with_synonyms(r_words)
    else:
        a_set, r_set = set(a_words), set(r_words)

    if not a_set:
        return 0
    return len(a_set & r_set) / len(a_set) * 100

def semantic_percent(answer, response):
    # Compute embeddings and cosine similarity
    emb_a = embedder.encode(answer, convert_to_tensor=True)
    emb_r = embedder.encode(response, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb_a, emb_r).item()
    return sim * 100  # percentage

