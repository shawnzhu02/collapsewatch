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

url = 'https://raw.githubusercontent.com/rsickle1/human-v-ai/master/5000human_5000machine.csv'
df = pd.read_csv(url)

real = df[df['label_cat'] == 'human']['text'].tolist()
synth = df[df['label_cat'] == 'machine']['text'].tolist()

print(f"Loaded {len(real)} real abstracts and {len(synth)} synthetic abstracts")

print("\nCreating N-gram features...")

def create_ngram_features(texts, ngram_range=(1, 3), max_features=500, use_tfidf=True):
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

real_ngrams, ngram_vectorizer = create_ngram_features(real)
synth_ngrams, _ = create_ngram_features(synth)

synth_ngrams = ngram_vectorizer.transform([re.sub(r'[^\w\s]', ' ', text.lower()) for text in synth])
synth_ngrams = synth_ngrams.toarray()

X_real = real_ngrams
X_synth = synth_ngrams

print(f"n-gram features: Real {X_real.shape}, Synthetic {X_synth.shape}")

print("Creating POS features...")

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



real_pos_features, pos_tags = extract_pos_features(real)
synth_pos_features, _ = extract_pos_features(synth)

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

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)

real_emb = create_embeddings(real)
synth_emb = create_embeddings(synth)

print(f"Embeddings: Real {real_emb.shape}, Synthetic {synth_emb.shape}")

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

if real_emb.shape[1] > 500:
    real_emb, synth_emb = apply_dimensionality_reduction(
        real_emb, synth_emb, target_dim=384
    )

print("\n" + "="*60)
print("SUMMARY OF CREATED DATA VARIABLES")
print("="*60)
print(f"X_real (n-grams): {X_real.shape}")
print(f"X_synth (n-grams): {X_synth.shape}")
print(f"X_real_pos (POS features): {X_real_pos.shape}")
print(f"X_synth_pos (POS features): {X_synth_pos.shape}")
print(f"real_emb (embeddings): {real_emb.shape}")
print(f"synth_emb (embeddings): {synth_emb.shape}")
print("="*60)

print("\nBasic Statistics:")
print(f"Real abstracts - Mean length: {np.mean([len(text.split()) for text in real]):.1f} words")
print(f"Synthetic abstracts - Mean length: {np.mean([len(text.split()) for text in synth]):.1f} words")

print(f"\nN-gram features:")
print(f"  Real - Mean: {X_real.mean():.4f}, Std: {X_real.std():.4f}")
print(f"  Synthetic - Mean: {X_synth.mean():.4f}, Std: {X_synth.std():.4f}")

print(f"\nPOS features:")
print(f"  Real - Mean: {X_real_pos.mean():.4f}, Std: {X_real_pos.std():.4f}")
print(f"  Synthetic - Mean: {X_synth_pos.mean():.4f}, Std: {X_synth_pos.std():.4f}")

print(f"\nEmbedding features:")
print(f"  Real - Mean: {real_emb.mean():.4f}, Std: {real_emb.std():.4f}")
print(f"  Synthetic - Mean: {synth_emb.mean():.4f}, Std: {synth_emb.std():.4f}")
