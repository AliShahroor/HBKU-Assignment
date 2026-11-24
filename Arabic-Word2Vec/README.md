# Assignment 5 - Arabic Word2Vec

This folder contains a reproducible Word2Vec experiment for the Arabic NLP assignment. The new corpus is an Arabic computer-science corpus covering programming, machine learning, networks, databases, security, and university work.

## What changed from the class notebook

- Replaced the original English sample corpus with a new 3,007-token Arabic corpus whose first four lines are the exact 80 words from the earlier n-gram assignment.
- Replaced NLTK tokenization with a transparent Arabic-aware regular-expression tokenizer.
- Used skip-gram Word2Vec with fixed settings and a fixed random seed.
- Saved the trained model and experiment metadata.
- Calculated five required cosine-similarity examples automatically.
- Added corpus statistics and a short interpretation of each result.

## Files

- `generate_and_train.py`: creates the corpus, trains Word2Vec, and computes results.
- `Word2Vec_Arabic_CS.ipynb`: one-click Google Colab workflow with saved reference output.

Running either entry point generates `corpus.txt` (310 sentences and 3,007 tokens), `arabic_cs_word2vec.model`, `similarity_results.csv`, and `experiment.json`.

## Run it

For the easiest run, open [`Word2Vec_Arabic_CS.ipynb`](https://colab.research.google.com/github/AliShahroor/HBKU-Assignment/blob/main/Arabic-Word2Vec/Word2Vec_Arabic_CS.ipynb) in Google Colab and select **Runtime > Run all**. No GPU is required.

To run locally, use:

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "gensim==4.4.0"
python generate_and_train.py
```

The fixed seed (`42`) and one training worker reproduce the same corpus and cosine scores on the tested Gensim version. Gensim may store different lifecycle timestamps inside the binary model file, and small floating-point differences can occur across platforms or library versions.

## Main result

The clearest learned relationship is `الخادم` / `العميل` with cosine similarity `0.8887`. This makes sense because both words repeatedly occur in the same request-response and network contexts.
