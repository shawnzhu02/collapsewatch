import argparse
import sys
import os
import json
import torch
import pprint
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import csv
from collections import Counter

TEMPLAMA_NEW_DIR = "/content/test.jsonl"

def load_file(filename):
    """
    :param filename:
    :return:
    """
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

    import pandas as pd

test_set = load_file(TEMPLAMA_NEW_DIR)

facts, labels, num_labels, quarters, relations = [], [], [], [], []
for i, example in enumerate(test_set):
    facts.append(example['query'])
    quarters.append(example['date'])
    relations.append(example['relation'])
    _label_list = [a['name'] for a in example['answer']]
    labels.append(_label_list)
    num_labels.append(len(_label_list))

full_dataset = pd.DataFrame(data={
    'fact': facts,
    'label': labels,
    'num_labels': num_labels,
    'quarter': quarters,
    'relation': relations
})

dataset_2019_q1 = full_dataset[full_dataset['quarter'] == '2019-Q1'].copy()
dataset_2022_q2 = full_dataset[full_dataset['quarter'] == '2022-Q2'].copy()

print(f"2019-Q1 dataset size: {len(dataset_2019_q1)}")
print(f"2022-Q2 dataset size: {len(dataset_2022_q2)}")

common_facts = set(dataset_2019_q1['fact']) & set(dataset_2022_q2['fact'])
print(f"Facts that appear in both quarters: {len(common_facts)}")

labels_2019_q1 = []
labels_2022_q2 = []

for fact in common_facts:
    row_2019 = dataset_2019_q1[dataset_2019_q1['fact'] == fact].iloc[0]
    row_2022 = dataset_2022_q2[dataset_2022_q2['fact'] == fact].iloc[0]

    labels_2019_q1.append(row_2019['label'])
    labels_2022_q2.append(row_2022['label'])

paired_dataset = []
for i, fact in enumerate(common_facts):
    row_2019 = dataset_2019_q1[dataset_2019_q1['fact'] == fact].iloc[0]
    row_2022 = dataset_2022_q2[dataset_2022_q2['fact'] == fact].iloc[0]

    paired_dataset.append({
        'fact': fact,
        'relation': row_2019['relation'], 
        'labels_2019_q1': labels_2019_q1[i],
        'labels_2022_q2': labels_2022_q2[i],
        'num_labels_2019_q1': row_2019['num_labels'],
        'num_labels_2022_q2': row_2022['num_labels']
    })

paired_df = pd.DataFrame(paired_dataset)

print(f"2019-Q1 dataset size: {len(dataset_2019_q1)}")
print(f"2022-Q2 dataset size: {len(dataset_2022_q2)}")
print(f"Facts that appear in both quarters: {len(common_facts)}")
print(f"2019-Q1 labels list length: {len(labels_2019_q1)}")
print(f"2022-Q2 labels list length: {len(labels_2022_q2)}")

print("\nSample from 2019-Q1 labels:")
print(labels_2019_q1[:3])

print("\nSample from 2022-Q2 labels:")
print(labels_2022_q2[:3])

real_abstracts = labels_2019_q1
synth_abstracts = labels_2022_q2


print("\nSample from paired dataset:")
print(paired_df.head())


def preprocess_abstracts(abstracts_list):
    """
    Convert list of lists to list of strings by joining elements with commas

    Args:
        abstracts_list: List where each element is a list of strings

    Returns:
        List of strings where each element is joined with commas
    """
    processed_abstracts = []

    for abstract in abstracts_list:
        if isinstance(abstract, list):
            processed_abstract = ", ".join(abstract)
        else:
            processed_abstract = str(abstract)

        processed_abstracts.append(processed_abstract)

    return processed_abstracts

real_abstracts = preprocess_abstracts(real_abstracts)
synth_abstracts = preprocess_abstracts(synth_abstracts)


import json
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import pandas as pd
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from tqdm import tqdm


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    raise

print("Loading data...")

print(f"Loaded {len(real_abstracts)} real abstracts and {len(synth_abstracts)} synthetic abstracts")

print("\nCreating N-gram features...")

def create_ngram_features(texts, ngram_range=(1, 2), max_features=500, use_tfidf=True):
    """
    Create n-gram features from text data
    """
    cleaned_texts = []
    for text in texts:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = ' '.join(text.split())
        cleaned_texts.append(text)

    if use_tfidf:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
    else:
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )

    all_texts = cleaned_texts
    vectorizer.fit(all_texts)

    features = vectorizer.transform(cleaned_texts)
    return features.toarray(), vectorizer

real_ngrams, ngram_vectorizer = create_ngram_features(real_abstracts)
synth_ngrams, _ = create_ngram_features(synth_abstracts)

synth_ngrams = ngram_vectorizer.transform([re.sub(r'[^\w\s]', ' ', text.lower()) for text in synth_abstracts])
synth_ngrams = synth_ngrams.toarray()

X_real = real_ngrams
X_synth = synth_ngrams

print(f"N-gram features: Real {X_real.shape}, Synthetic {X_synth.shape}")

print("\nCreating POS features...")

def extract_pos_features(texts, max_features=100):
    """
    Extract POS tag features from texts using spaCy
    """
    pos_features = []

    for text in tqdm(texts, desc="Processing POS"):
        doc = nlp(text)
        pos_counts = Counter([token.pos_ for token in doc if not token.is_space])
        pos_features.append(pos_counts)

    all_pos_tags = set()
    for pos_dict in pos_features:
        all_pos_tags.update(pos_dict.keys())

    all_pos_tags = sorted(list(all_pos_tags))

    feature_matrix = np.zeros((len(pos_features), len(all_pos_tags)))

    for i, pos_dict in enumerate(pos_features):
        for j, pos_tag in enumerate(all_pos_tags):
            feature_matrix[i, j] = pos_dict.get(pos_tag, 0)

    doc_lengths = feature_matrix.sum(axis=1, keepdims=True)
    doc_lengths[doc_lengths == 0] = 1  
    feature_matrix = feature_matrix / doc_lengths

    return feature_matrix, all_pos_tags

real_pos_features, pos_tags = extract_pos_features(real_abstracts)
synth_pos_features, _ = extract_pos_features(synth_abstracts)

X_real_pos = real_pos_features
X_synth_pos = synth_pos_features

print(f"POS features: Real {X_real_pos.shape}, Synthetic {X_synth_pos.shape}")
print(f"POS tags used: {pos_tags}")

print("\nCreating contextual embeddings...")

def create_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    Create contextual embeddings using sentence transformers
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Create embeddings in batches
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)

real_abstracts_emb = create_embeddings(real_abstracts)
synth_abstracts_emb = create_embeddings(synth_abstracts)

print(f"Embeddings: Real {real_abstracts_emb.shape}, Synthetic {synth_abstracts_emb.shape}")

def apply_dimensionality_reduction(X_real, X_synth, target_dim=100, method='svd'):
    """
    Apply dimensionality reduction if features are too high-dimensional
    """
    if X_real.shape[1] <= target_dim:
        return X_real, X_synth

    print(f"   Reducing from {X_real.shape[1]} to {target_dim} dimensions using {method}")

    if method == 'svd':
        reducer = TruncatedSVD(n_components=target_dim, random_state=42)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=target_dim, random_state=42)

    X_real_reduced = reducer.fit_transform(X_real)
    X_synth_reduced = reducer.transform(X_synth)

    return X_real_reduced, X_synth_reduced

if X_real.shape[1] > 500:
    X_real, X_synth = apply_dimensionality_reduction(X_real, X_synth, target_dim=500)

if real_abstracts_emb.shape[1] > 500:
    real_abstracts_emb, synth_abstracts_emb = apply_dimensionality_reduction(
        real_abstracts_emb, synth_abstracts_emb, target_dim=384
    )

print("\n" + "="*60)
print("SUMMARY OF CREATED DATA VARIABLES")
print("="*60)
print(f"X_real (n-grams): {X_real.shape}")
print(f"X_synth (n-grams): {X_synth.shape}")
print(f"X_real_pos (POS features): {X_real_pos.shape}")
print(f"X_synth_pos (POS features): {X_synth_pos.shape}")
print(f"real_abstracts_emb (embeddings): {real_abstracts_emb.shape}")
print(f"synth_abstracts_emb (embeddings): {synth_abstracts_emb.shape}")
print("="*60)

print("\nBasic Statistics:")
print(f"Real abstracts - Mean length: {np.mean([len(text.split()) for text in real_abstracts]):.1f} words")
print(f"Synthetic abstracts - Mean length: {np.mean([len(text.split()) for text in synth_abstracts]):.1f} words")

print(f"\nN-gram features:")
print(f"  Real - Mean: {X_real.mean():.4f}, Std: {X_real.std():.4f}")
print(f"  Synthetic - Mean: {X_synth.mean():.4f}, Std: {X_synth.std():.4f}")

print(f"\nPOS features:")
print(f"  Real - Mean: {X_real_pos.mean():.4f}, Std: {X_real_pos.std():.4f}")
print(f"  Synthetic - Mean: {X_synth_pos.mean():.4f}, Std: {X_synth_pos.std():.4f}")

print(f"\nEmbedding features:")
print(f"  Real - Mean: {real_abstracts_emb.mean():.4f}, Std: {real_abstracts_emb.std():.4f}")
print(f"  Synthetic - Mean: {synth_abstracts_emb.mean():.4f}, Std: {synth_abstracts_emb.std():.4f}")
