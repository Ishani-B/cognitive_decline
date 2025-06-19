import os
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import cmudict

# Download and load CMU pronunciation dictionary
nltk.download('cmudict', quiet=True)
cmu_dict = cmudict.dict()

# Load spaCy model
nlp = spacy.load('en_core_web_lg')

def build_lookup(df, word_col, value_col):
    """
    From dataframe df, build a dict mapping each word (lowercased)
    to its corresponding value, dropping any NaN or empty words.
    """
    words = df[word_col].fillna('').astype(str)
    return {w.lower(): val for w, val in zip(words, df[value_col]) if w}

# --- load SUBTLEX and build lookups ---
subtlex = pd.read_csv('SUBTLEX_US.csv')
subtlwf_dict  = build_lookup(subtlex, 'Word', 'SUBTLWF')
freq_log_dict = build_lookup(subtlex, 'Word', 'Lg10WF')
cd_raw_dict   = build_lookup(subtlex, 'Word', 'SUBTLCD')
cd_log_dict   = build_lookup(subtlex, 'Word', 'Lg10CD')

# --- load concreteness norms ---
concrete = pd.read_csv('concreteness.csv')
conc_m_dict  = build_lookup(concrete, 'Word', 'Conc.M')
conc_sd_dict = build_lookup(concrete, 'Word', 'Conc.SD')

# --- load semantic ambiguity norms ---
ambiguity = pd.read_csv('semantic_ambiguity.csv')
meancos_dict      = build_lookup(ambiguity, 'word', 'mean_cos')
semd_dict         = build_lookup(ambiguity, 'word', 'SemD')
bnbwc_dict        = build_lookup(ambiguity, 'word', 'BNC_wordcount')
bncctx_dict       = build_lookup(ambiguity, 'word', 'BNC_contexts')
bnbcfreq_dict     = build_lookup(ambiguity, 'word', 'BNC_freq')
lg_bnc_dict       = build_lookup(ambiguity, 'word', 'lg_BNC_freq')

def get_phoneme_count(word):
    entries = cmu_dict.get(word.lower())
    if entries:
        return min(len(pron) for pron in entries)
    return np.nan


def compute_mattr(tokens, window_size=50):
    n = len(tokens)
    if n < window_size:
        return len(set(tokens)) / n if n > 0 else 0
    ttr_vals = [
        len(set(tokens[i:i+window_size])) / window_size
        for i in range(n - window_size + 1)
    ]
    return float(np.mean(ttr_vals))


def extract_features(filepath):
    text = open(filepath, 'r', encoding='utf-8').read()
    doc = nlp(text)
    tokens = [tok for tok in doc if tok.is_alpha]
    words = [tok.text.lower() for tok in tokens]
    total = len(tokens)

    # Part-of-speech counts per 100 words
    pos_counts = {}
    for tok in tokens:
        pos_counts[tok.pos_] = pos_counts.get(tok.pos_, 0) + 1
    pos_per_100 = {f"POS_{p}_per100": cnt / total * 100 for p, cnt in pos_counts.items()}

    # Build a DataFrame of word-level metrics
    df = pd.DataFrame({'word': words})
    mapped = {
        'freq_raw':       [subtlwf_dict.get(w, np.nan) for w in words],
        'freq_log':       [freq_log_dict.get(w, np.nan) for w in words],
        'cd_raw':         [cd_raw_dict.get(w,   np.nan) for w in words],
        'cd_log':         [cd_log_dict.get(w,   np.nan) for w in words],
        'concreteness_m': [conc_m_dict.get(w,   np.nan) for w in words],
        'concreteness_sd':[conc_sd_dict.get(w,  np.nan) for w in words],
        'mean_cos':       [meancos_dict.get(w,  np.nan) for w in words],
        'SemD':           [semd_dict.get(w,    np.nan) for w in words],
        'BNC_wordcount':  [bnbwc_dict.get(w,   np.nan) for w in words],
        'BNC_contexts':   [bncctx_dict.get(w,  np.nan) for w in words],
        'BNC_freq':       [bnbcfreq_dict.get(w, np.nan) for w in words],
        'lg_BNC_freq':    [lg_bnc_dict.get(w,   np.nan) for w in words],
        'phonemes':       [get_phoneme_count(w)  for w in words]
    }
    df = pd.concat([df, pd.DataFrame(mapped)], axis=1)

    # Use actual DataFrame columns for aggregation
    cols_to_agg = [c for c in df.columns if c != 'word']

    # Masks for scopes
    content_tags = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    is_content = [tok.pos_ in content_tags for tok in tokens]
    is_noun    = [tok.pos_ == 'NOUN'          for tok in tokens]

    # Aggregate statistics
    features = {}
    for scope, mask in [('all', [True] * total), ('content', is_content), ('noun', is_noun)]:
        sel = df[mask]
        for col in cols_to_agg:
            # skip metrics that aren't present
            if col not in sel.columns:
                continue
            features[f"{scope}_{col}_mean"] = sel[col].mean()

    # Global MATTR & POS stats
    features['mattr'] = compute_mattr(words)
    features.update(pos_per_100)

    return features

if __name__ == '__main__':
    input_root  = 'transcript_input'
    output_root = 'output_lexical_metrics'
    for sub in ['cc', 'cd']:
        in_dir  = os.path.join(input_root, sub)
        out_dir = os.path.join(output_root, sub)
        os.makedirs(out_dir, exist_ok=True)
        for fname in os.listdir(in_dir):
            if not fname.lower().endswith('.txt'):
                continue
            feats = extract_features(os.path.join(in_dir, fname))
            out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + '.csv')
            pd.DataFrame([feats]).to_csv(out_path, index=False)
    print("Feature extraction complete.")
