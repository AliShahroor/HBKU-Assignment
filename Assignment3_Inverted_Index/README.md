# Assignment 3: Positional Inverted Index

This folder contains the final results of a positional inverted index built for three Modern Standard Arabic documents. The code was developed step by step in the notebook: it normalizes and tokenizes the text, removes stop words while keeping the original positions, uses CAMeL Tools for lemmatization, and then builds the index from the resulting lemmas.

The notebook includes the processing output and a small optional example for another Arabic sentence. The CSV files contain the generated results, and the PDF is a short LaTeX version of the assignment.

## Project structure

```text
Assignment3_Inverted_Index.ipynb          Executed Python notebook
output/positional_inverted_index.csv     Final positional index
output/processing_details.csv            Token-level processing output
output/token_diagnostics.csv             Lemmatization diagnostics
report/Assignment3_Inverted_Index.pdf    Final compiled LaTeX report
```

## Environment and lemmatizer

- Python 3.11
- CAMeL Tools 1.6.0
- CALIMA MSA morphology database: `morphology-db-msa-r13`
- MSA MLE disambiguator: `disambig-mle-calima-msa-r13`

The program uses `MLEDisambiguator.pretrained("calima-msa-r13")`. For each token it takes the highest-ranked analysis selected by the pretrained MLE model and uses its `lex` field as the dictionary lemma. The lemma is dediacritized and conservatively normalized for indexing. Unknown unanalyzed tokens retain their normalized surface form and are marked in `token_diagnostics.csv`; the included documents currently produce analyses for every token.

Lemmatization is used instead of stemming because a lemma is a meaningful dictionary/base form, while stemming may aggressively reduce a word to a stem or root. The instructor explicitly requested lemmatization. The model is an established morphological system, but its analysis can still be imperfect for ambiguous words.

**Lemmatization was used instead of stemming because the instructor explicitly requested lemmatization.**

Positions are assigned before stop-word removal. Therefore, deleting a stop word never changes the positions of later tokens.

## Interactive demo

The notebook ends with an optional section named **Try Your Own Arabic Text**. Enter one Arabic sentence when prompted. The section reuses the same `ArabicPreprocessor` and `build_positional_index` functions as the assignment pipeline; it does not contain a second preprocessing implementation.

It displays:

- each original and normalized token at its original position;
- the CAMeL Tools lemma, or `—` for a removed stop word;
- whether each token is a stop word; and
- a small `USER_DOC` positional index containing only non-stop-word lemmas.

Empty input is handled with a short message. Nothing entered in this optional section changes the three original documents, the submitted index, the CSV results, or the PDF report.

## Google Colab

The complete project is available in the `AliShahroor/HBKU-Assignment` repository. Open the notebook directly in Google Colab at:

```text
https://colab.research.google.com/github/AliShahroor/HBKU-Assignment/blob/main/Assignment3_Inverted_Index/Assignment3_Inverted_Index.ipynb
```

The saved notebook already contains all assignment output. Its optional final cell can also process a new Arabic sentence when the full local project environment is available.
