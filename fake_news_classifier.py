# Install all required packages
import subprocess
subprocess.run(["pip", "install", "kaggle", "nltk", "scikit-learn", "benepar", "spacy", "matplotlib", "seaborn", "-q"])
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
print("Installations done!")

import os
import json

KAGGLE_USERNAME = " " #your kaggle username
KAGGLE_API_KEY  = " " #your kaggle api token

os.makedirs('/root/.kaggle', exist_ok=True)

kaggle_config = {"username": KAGGLE_USERNAME, "key": KAGGLE_API_KEY}

with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump(kaggle_config, f)

os.chmod('/root/.kaggle/kaggle.json', 0o600)
print("Kaggle API key configured!")
import subprocess
subprocess.run(["kaggle", "datasets", "download", "-d", "csmalarkodi/isot-fake-news-dataset", "--unzip"])

import os
files_present = os.listdir('.')
print("Files in directory:", [f for f in files_present if f.endswith('.csv')])

import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay,
                              f1_score, precision_score, recall_score)
import scipy.sparse as sp
from scipy.sparse import hstack
from scipy import stats

# Download all NLTK resources
for resource in ['punkt', 'averaged_perceptron_tagger', 'stopwords',
                 'wordnet', 'omw-1.4', 'punkt_tab',
                 'averaged_perceptron_tagger_eng']:
    nltk.download(resource, quiet=True)

print("All imports and NLTK resources ready!")

# Load CSVs
import os

# Find the actual filenames
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print("CSVs found:", csv_files)

# Load — adjust names if yours differ slightly
real_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

real_df['label'] = 1   # 1 = Real / Reliable
fake_df['label'] = 0   # 0 = Fake / Unreliable

print(f"Real articles : {len(real_df)}")
print(f"Fake articles : {len(fake_df)}")
print(f"\nReal columns  : {list(real_df.columns)}")
print(f"Fake columns  : {list(fake_df.columns)}")

RANDOM_SEED = 42

# Sample 1000 from each for class balance
real_sample = real_df.sample(n=1000, random_state=RANDOM_SEED).reset_index(drop=True)
fake_sample = fake_df.sample(n=1000, random_state=RANDOM_SEED).reset_index(drop=True)

# Combine
df = pd.concat([real_sample, fake_sample], ignore_index=True)

# Build content = title + text (handle missing columns gracefully)
title_col = df['title'].fillna('') if 'title' in df.columns else pd.Series([''] * len(df))
text_col  = df['text'].fillna('')  if 'text'  in df.columns else pd.Series([''] * len(df))
df['content'] = (title_col + ' ' + text_col).str.strip()

# Shuffle
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Sanity check
print(f"Total articles : {len(df)}")
print(f"Class balance  :\n{df['label'].value_counts().rename({1:'Real', 0:'Fake'})}")
print(f"\nSample content preview:")
print(df['content'].iloc[0][:300])

# Tokenize into sentences → words
def tokenize_article(text):
    sentences  = sent_tokenize(str(text))
    tokenized  = [word_tokenize(sent) for sent in sentences]
    return tokenized

print("Tokenizing 2,000 articles... (~30 seconds)")
df['tokenized'] = df['content'].apply(tokenize_article)

print("Tokenization complete!")
print(f"\nSample — first sentence tokens:")
print(df['tokenized'].iloc[0][0][:15])

# Stop-word Retention Strategy
BASE_STOPWORDS = set(stopwords.words('english'))

# Tokens we KEEP even though NLTK lists them as stop words
KEEP_TOKENS = {
    # First-person (strong fake news signal)
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    # Other pronouns
    'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
    # Punctuation signals (emotional intensity markers)
    '!', '?'
}

CUSTOM_STOPWORDS = BASE_STOPWORDS - KEEP_TOKENS

print("="*50)
print(" WHY KEEP THESE TOKENS?")
print("="*50)
print("""
❗ Exclamation marks (!):
   Fake news uses alarming, emotionally charged language
   to provoke shares. Phrases like 'You won't BELIEVE this!'
   are hallmarks of clickbait. Removing ! discards this
   emotional intensity signal entirely.

👤 First-person pronouns (I, We):
   Reliable journalism is written in objective third-person.
   High I/We usage signals opinion, anecdote, or partisan
   writing — strongly correlated with unreliable content.
""")

# Lemmatizer Setup
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Map PennTreebank POS → WordNet POS for accurate lemmatization."""
    if treebank_tag.startswith('J'): return wordnet.ADJ
    if treebank_tag.startswith('V'): return wordnet.VERB
    if treebank_tag.startswith('N'): return wordnet.NOUN
    if treebank_tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

def lemmatize_tokens(tokenized_sentences):
    flat_tokens = [tok for sent in tokenized_sentences for tok in sent]
    tagged      = pos_tag(flat_tokens)
    lemmatized  = []
    for word, tag in tagged:
        word_lower = word.lower()
        if word in ('!', '?'):
            lemmatized.append(word)          # keep as-is
        elif word_lower in CUSTOM_STOPWORDS:
            continue                         # remove stop words
        elif word.isalpha() or word_lower in KEEP_TOKENS:
            wn_pos = get_wordnet_pos(tag)
            lemmatized.append(lemmatizer.lemmatize(word_lower, pos=wn_pos))
    return lemmatized

print("Lemmatizing 2,000 articles... (1-2 minutes on Colab)")
df['lemmatized_tokens'] = df['tokenized'].apply(lemmatize_tokens)
df['processed_text']    = df['lemmatized_tokens'].apply(lambda t: ' '.join(t))

print("Lemmatization complete!")
print("\nBefore:", df['content'].iloc[0][:200])
print("\nAfter :", df['processed_text'].iloc[0][:200])

# Extract 7 Linguistic Features per article
def extract_pos_features(tokenized_sentences):
    flat_tokens = [tok for sent in tokenized_sentences for tok in sent]
    tagged      = pos_tag(flat_tokens)
    tag_counts  = Counter(tag for _, tag in tagged)

    total_tokens  = max(len(flat_tokens), 1)
    num_sentences = max(len(tokenized_sentences), 1)

    # 1. Superlative Ratio (JJS=superlative adj, RBS=superlative adv)
    superlative_count = tag_counts.get('JJS', 0) + tag_counts.get('RBS', 0)
    superlative_ratio = superlative_count / total_tokens

    # 2. Proper Noun Ratio (NNP + NNPS)
    proper_noun_count = tag_counts.get('NNP', 0) + tag_counts.get('NNPS', 0)
    proper_noun_ratio = proper_noun_count / total_tokens

    # 3. First-person Pronoun Ratio (I/We vs He/She/They)
    first_person = {'i','me','my','myself','we','our','us','ourselves'}
    third_person = {'he','him','his','she','her','they','them','their'}
    first_count  = sum(1 for w,t in tagged if t=='PRP' and w.lower() in first_person)
    third_count  = sum(1 for w,t in tagged if t in ('PRP','PRP$') and w.lower() in third_person)
    first_person_ratio = first_count / max(first_count + third_count, 1)

    # 4. Exclamation Mark Ratio (per sentence)
    exclamation_ratio = sum(1 for w,_ in tagged if w == '!') / num_sentences

    # 5. Question Mark Ratio (per sentence)
    question_ratio = sum(1 for w,_ in tagged if w == '?') / num_sentences

    # 6. Average Sentence Length (words per sentence)
    word_counts = [sum(1 for tok in sent if tok.isalpha()) for sent in tokenized_sentences]
    avg_sentence_length = np.mean(word_counts) if word_counts else 0

    # 7. Adjective Ratio (JJ + JJR + JJS)
    adj_count = tag_counts.get('JJ',0) + tag_counts.get('JJR',0) + tag_counts.get('JJS',0)
    adj_ratio = adj_count / total_tokens

    return {
        'superlative_ratio'   : superlative_ratio,
        'proper_noun_ratio'   : proper_noun_ratio,
        'first_person_ratio'  : first_person_ratio,
        'exclamation_ratio'   : exclamation_ratio,
        'question_ratio'      : question_ratio,
        'avg_sentence_length' : avg_sentence_length,
        'adj_count'           : adj_count,
        'adj_ratio'           : adj_ratio,
    }

print("Extracting POS features... (2-3 minutes)")
features   = df['tokenized'].apply(extract_pos_features)
feature_df = pd.DataFrame(features.tolist())
df         = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

print("Feature extraction complete!")
display(df[['label','superlative_ratio','proper_noun_ratio',
            'first_person_ratio','exclamation_ratio',
            'avg_sentence_length','adj_ratio']].head(5))

# Report Table 1: Linguistic Profile
feature_cols = ['superlative_ratio','proper_noun_ratio','first_person_ratio',
                'exclamation_ratio','question_ratio','avg_sentence_length','adj_ratio']

profile = df.groupby('label')[feature_cols].mean().T
profile.columns = ['Fake (Unreliable)', 'Real (Reliable)']
profile.index.name = 'Linguistic Feature'

print("="*60)
print("   LINGUISTIC PROFILE TABLE: FAKE vs REAL NEWS")
print("="*60)
display(profile.round(5))

# Feature Distribution Plots
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Linguistic Feature Distributions: Fake vs Real News',
             fontsize=15, fontweight='bold')

plot_features = [
    ('superlative_ratio',  'Superlative Ratio (JJS+RBS)'),
    ('proper_noun_ratio',  'Proper Noun Ratio (NNP)'),
    ('first_person_ratio', 'First-Person Pronoun Ratio'),
    ('exclamation_ratio',  'Exclamation Marks / Sentence'),
    ('avg_sentence_length','Avg Sentence Length (words)'),
    ('adj_ratio',          'Adjective Ratio'),
]

for ax, (feat, title) in zip(axes.flatten(), plot_features):
    for lv, name, color in [(0,'Fake','#e74c3c'), (1,'Real','#2ecc71')]:
        data = df[df['label']==lv][feat].dropna()
        ax.hist(data, bins=30, alpha=0.6, color=color, label=name, density=True)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Value'); ax.set_ylabel('Density')
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved!")

# Syntax Analysis using NLTK RegexpParser (replaces Benepar)
from nltk.tree import Tree
from nltk import RegexpParser

# Simple grammar for shallow constituency parsing
grammar = r"""
    NP: {<DT>?<JJ>*<NN.*>+}
    VP: {<VB.*><NP|PP|CLAUSE>+$}
    PP: {<IN><NP>}
    CLAUSE: {<NP><VP>}
"""
parser = RegexpParser(grammar)

def get_tree_depth(tree):
    if isinstance(tree, Tree):
        if len(tree) == 0:
            return 1
        return 1 + max(get_tree_depth(child) for child in tree)
    return 0

def parse_sample_sentences(article_texts, n=50, label_name=''):
    all_sents = []
    for text in article_texts:
        sents = sent_tokenize(str(text))
        sents = [s for s in sents if 5 <= len(s.split()) <= 25]
        all_sents.extend(sents)

    np.random.seed(42)
    sampled = np.random.choice(all_sents, size=min(n, len(all_sents)), replace=False)

    depths = []
    print(f"Parsing {len(sampled)} sentences from {label_name}...")
    for sent in sampled:
        try:
            tokens = word_tokenize(sent)
            tagged = pos_tag(tokens)
            tree   = parser.parse(tagged)
            depth  = get_tree_depth(tree)
            if depth: depths.append(depth)
        except:
            continue

    print(f"  → {len(depths)} trees parsed. Mean depth = {np.mean(depths):.2f}")
    return depths

fake_texts = df[df['label']==0]['content'].tolist()
real_texts = df[df['label']==1]['content'].tolist()

fake_depths = parse_sample_sentences(fake_texts[:120], n=50, label_name='Fake News')
real_depths = parse_sample_sentences(real_texts[:120], n=50, label_name='Real News')

# Visualize + Hypothesis Test
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Constituency Parse Tree Depth: Fake vs Real News',
             fontsize=13, fontweight='bold')

ax = axes[0]
ax.hist(fake_depths, bins=10, alpha=0.7, color='#e74c3c',
        label=f'Fake (mean={np.mean(fake_depths):.2f})')
ax.hist(real_depths, bins=10, alpha=0.7, color='#2ecc71',
        label=f'Real (mean={np.mean(real_depths):.2f})')
ax.set_xlabel('Parse Tree Depth'); ax.set_ylabel('Count')
ax.set_title('Distribution of Tree Depth'); ax.legend(); ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.boxplot([fake_depths, real_depths], labels=['Fake News', 'Real News'],
            patch_artist=True,
            boxprops=dict(facecolor='#fadbd8'),
            medianprops=dict(color='black', linewidth=2))
ax2.set_ylabel('Parse Tree Depth'); ax2.set_title('Box Plot'); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('tree_depth_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Statistical test
t_stat, p_val = stats.ttest_ind(real_depths, fake_depths)
print(f"\nHypothesis Test:")
print(f"   Fake mean depth : {np.mean(fake_depths):.3f}")
print(f"   Real mean depth : {np.mean(real_depths):.3f}")
print(f"   t = {t_stat:.3f}, p = {p_val:.4f}")
print(f"   {'Significant (p<0.05)' if p_val < 0.05 else '⚠️ Not significant'}")

# Train/Test Split
LINGUISTIC_FEATURES = ['superlative_ratio','proper_noun_ratio','first_person_ratio',
                        'exclamation_ratio','question_ratio','avg_sentence_length','adj_ratio']

X_text = df['processed_text']
X_ling = df[LINGUISTIC_FEATURES]
y      = df['label']

X_text_train, X_text_test, X_ling_train, X_ling_test, y_train, y_test = train_test_split(
    X_text, X_ling, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2),
                         min_df=3, max_df=0.95, sublinear_tf=True)
X_tfidf_train = tfidf.fit_transform(X_text_train)
X_tfidf_test  = tfidf.transform(X_text_test)

# Scale + Combine
scaler = StandardScaler()
X_ling_train_scaled = scaler.fit_transform(X_ling_train)
X_ling_test_scaled  = scaler.transform(X_ling_test)

X_combined_train = hstack([X_tfidf_train, sp.csr_matrix(X_ling_train_scaled)])
X_combined_test  = hstack([X_tfidf_test,  sp.csr_matrix(X_ling_test_scaled)])

print(f"Train/Test split    : {len(y_train)} / {len(y_test)}")
print(f"TF-IDF shape        : {X_tfidf_train.shape}")
print(f"Combined shape      : {X_combined_train.shape}")

# Model A: TF-IDF Only
model_a = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
model_a.fit(X_tfidf_train, y_train)
y_pred_a = model_a.predict(X_tfidf_test)
acc_a    = accuracy_score(y_test, y_pred_a)

print("="*55)
print("  MODEL A — TF-IDF Only")
print("="*55)
print(f"  Accuracy : {acc_a*100:.2f}%")
print(classification_report(y_test, y_pred_a, target_names=['Fake','Real']))

# Model B: TF-IDF + Linguistic Features
model_b = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
model_b.fit(X_combined_train, y_train)
y_pred_b = model_b.predict(X_combined_test)
acc_b    = accuracy_score(y_test, y_pred_b)

print("="*55)
print("  MODEL B — TF-IDF + Linguistic Features")
print("="*55)
print(f"  Accuracy : {acc_b*100:.2f}%")
print(classification_report(y_test, y_pred_b, target_names=['Fake','Real']))

delta = (acc_b - acc_a)*100
print(f"\n{'IMPROVED' if delta >= 0 else 'DECREASED'} by {abs(delta):.2f}% after adding linguistic features")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

for ax, y_pred, title in [
    (axes[0], y_pred_a, f'Model A: TF-IDF Only\n(Acc: {acc_a*100:.2f}%)'),
    (axes[1], y_pred_b, f'Model B: TF-IDF + Linguistic\n(Acc: {acc_b*100:.2f}%)'),
]:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# Find Real articles wrongly flagged as Fake
test_df          = df.loc[X_text_test.index].copy()
test_df['pred']  = y_pred_b

false_negatives = test_df[(test_df['label']==1) & (test_df['pred']==0)]
print(f"Real articles misclassified as Fake: {len(false_negatives)}\n")

sample = false_negatives.iloc[0]
print("─"*60)
print("MISCLASSIFIED ARTICLE (Real → Predicted Fake):")
print("─"*60)
print(sample['content'][:500])
print("\nIts Linguistic Features:")
for feat in LINGUISTIC_FEATURES:
    print(f"   {feat:25s}: {sample[feat]:.5f}")

print("\nWHY WAS THIS MISCLASSIFIED?")
reasons = []
if sample['first_person_ratio'] > 0.4:
    reasons.append("High first-person pronoun ratio → reads like opinion/editorial")
if sample['superlative_ratio'] > 0.012:
    reasons.append("Elevated superlative usage → emotionally charged language")
if sample['proper_noun_ratio'] < 0.03:
    reasons.append("Low proper noun ratio → lacks specific names/places")
if sample['exclamation_ratio'] > 0.05:
    reasons.append("High exclamation usage → sensationalist tone")
if not reasons:
    reasons.append("The article's vocabulary overlapped heavily with fake news patterns in TF-IDF space")
for r in reasons:
    print(f"   → {r}")

print("="*65)
print("              FINAL RESULTS SUMMARY")
print("="*65)

# Linguistic profile
summary = df.groupby('label').agg(
    Avg_Adj_Count        = ('adj_count', 'mean'),
    Avg_Sentence_Length  = ('avg_sentence_length', 'mean'),
    Avg_Superlative_Ratio= ('superlative_ratio', 'mean'),
    Avg_Proper_Noun_Ratio= ('proper_noun_ratio', 'mean'),
    Avg_1st_Person_Ratio = ('first_person_ratio', 'mean'),
).round(4)
summary.index = ['Fake','Real']
print("\nLinguistic Profile Table:")
display(summary)

# Model comparison
perf = pd.DataFrame({
    'Model'    : ['A: TF-IDF Only', 'B: TF-IDF + Linguistic'],
    'Accuracy' : [f'{acc_a*100:.2f}%', f'{acc_b*100:.2f}%'],
    'F1-Score' : [f'{f1_score(y_test,y_pred_a):.4f}',
                  f'{f1_score(y_test,y_pred_b):.4f}'],
    'Precision': [f'{precision_score(y_test,y_pred_a):.4f}',
                  f'{precision_score(y_test,y_pred_b):.4f}'],
    'Recall'   : [f'{recall_score(y_test,y_pred_a):.4f}',
                  f'{recall_score(y_test,y_pred_b):.4f}'],
})
print("\nModel Performance Comparison:")
display(perf)

print(f"\nPipeline complete! Saved: confusion_matrices.png,")
print(f"   feature_distributions.png, tree_depth_analysis.png")