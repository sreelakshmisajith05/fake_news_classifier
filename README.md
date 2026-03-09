# Fake News Classifier — NLP Project

A text classifier that detects fake news using **linguistic feature engineering** and **NLP**, built as part of a university assignment. The classifier goes beyond simple keyword matching by analysing *how* articles are written — not just *what* words they use.

---

## Objective

Classify news articles as **Reliable** or **Unreliable** by combining:
- **TF-IDF** — what words are used
- **Linguistic Features** — how the article is written (writing style, tone, grammar)

---

## Dataset

**ISOT Fake News Dataset** — downloaded from Kaggle  
- `True.csv` — 21,417 real news articles (source: Reuters)  
- `Fake.csv` — 23,481 fake news articles (source: unreliable websites)  
- **Sampled:** 1,000 Real + 1,000 Fake = **2,000 articles** (balanced)

---

## Pipeline

### Phase 1 — Preprocessing
- Sentence + word tokenization (NLTK)
- Custom stop-word strategy — keeps `!`, `?`, and pronouns (`I`, `we`, `they`)
- POS-aware lemmatization (WordNetLemmatizer)

### Phase 2 — Linguistic Feature Engineering (POS Tagging)
7 features extracted per article:

| Feature | POS Tags | Why |
|---|---|---|
| Superlative Ratio | JJS, RBS | Fake news exaggerates (best, worst, biggest) |
| Proper Noun Ratio | NNP, NNPS | Real news names specific people/places |
| First-Person Pronoun Ratio | PRP | Fake news uses opinion voice (I, we) |
| Exclamation Ratio | ! per sentence | Emotional intensity marker |
| Question Ratio | ? per sentence | Rhetorical questions create doubt |
| Avg Sentence Length | Word count | Fake news uses run-on sentences |
| Adjective Ratio | JJ, JJR, JJS | Descriptive language density |

### Phase 3 — Syntax Analysis
- Constituency parsing using NLTK RegexpParser
- Parse tree depth compared across 50 sentences per class
- Statistical significance tested with t-test (t=2.037, **p=0.044**)

### Phase 4 — Classification
- Logistic Regression (scikit-learn)
- Model A: TF-IDF only
- Model B: TF-IDF + Linguistic Features

---

## Results

### Linguistic Profile

| Feature | Fake News | Real News |
|---|---|---|
| Avg Adjective Count | 32.54 | 30.29 |
| Avg Sentence Length | 30.49 words | 28.33 words |
| Exclamation Ratio | 0.091 | 0.004 |
| Question Ratio | 0.075 | 0.004 |
| First-Person Ratio | 0.225 | 0.185 |

> Fake news uses exclamation marks at **21.7x** the rate of real news

### Model Performance

| Model | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|
| Model A: TF-IDF Only | **97.00%** | 0.9700 | 0.9700 | 0.9700 |
| Model B: TF-IDF + Linguistic | 95.00% | 0.9502 | 0.9455 | 0.9550 |

### Parse Tree Depth (Syntax Complexity)
| Class | Mean Depth |
|---|---|
| Fake News | 2.500 |
| Real News | 2.720 |
> t = 2.037, p = 0.044 — statistically significant

---

## Files

| File | Description |
|---|---|
| `fake_news_classifier.py` | Full pipeline script |
| `confusion_matrices.png` | Model A vs Model B confusion matrices |
| `feature_distributions.png` | Linguistic feature distributions (Fake vs Real) |
| `tree_depth_analysis.png` | Parse tree depth analysis |

---

## Tools & Libraries

- **Python** — VS Code
- **NLTK** — tokenization, POS tagging, lemmatization, parsing
- **Scikit-learn** — TF-IDF, Logistic Regression, evaluation metrics
- **Pandas / NumPy** — data processing
- **Matplotlib / Seaborn** — visualizations
- **Kaggle API** — dataset download

---

## How to Run

1. Make sure your Kaggle API token (`kaggle.json`) is placed in `C:\Users\<you>\.kaggle\`
2. Install dependencies:
```bash
pip install kaggle nltk scikit-learn spacy matplotlib seaborn
python -m spacy download en_core_web_sm
```
3. Run the script:
```bash
python fake_news_classifier.py
```
4. Total runtime: ~15-20 minutes

---

## Key Findings

- Fake news relies heavily on emotional punctuation — **21.7x more exclamation marks** than real news
- Real news uses statistically deeper parse trees (more complex grammar)
- TF-IDF alone achieved 97% accuracy on this dataset due to source-specific vocabulary
- Linguistic features are essential for cross-domain generalization where vocabulary memorization fails