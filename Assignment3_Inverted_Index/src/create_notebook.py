"""Create the lab-style Assignment 3 notebook from reusable project code."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Assignment3_Inverted_Index.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        """
# Assignment 3: Positional Inverted Index

**Student:** Ali Shahroor  
**Course:** Arabic NLP

This notebook automatically processes three Modern Standard Arabic documents and builds a positional inverted index. Lemmatization is used instead of stemming, as requested by the instructor.
"""
    ),
    markdown(
        """
## 1. Setup

The notebook uses reusable functions from the `src` folder. CAMeL Tools supplies Arabic-aware tokenization, the CALIMA MSA morphology database, and the pretrained MLE disambiguator. No LLM or generative API is used.

**Google Colab:** open this notebook from `AliShahroor/HBKU-Assignment` at:

<https://colab.research.google.com/github/AliShahroor/HBKU-Assignment/blob/main/Assignment3_Inverted_Index/Assignment3_Inverted_Index.ipynb>
"""
    ),
    code(
        """
# Colab bootstrap: this does nothing during a normal local run.
import os
import subprocess
import sys
from pathlib import Path

try:
    from google import colab as _google_colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
REPOSITORY_URL = "https://github.com/AliShahroor/HBKU-Assignment.git"
COLAB_REPOSITORY = Path("/content/HBKU-Assignment")
COLAB_PROJECT = COLAB_REPOSITORY / "Assignment3_Inverted_Index"

if IN_COLAB:
    if not (Path.cwd() / "src").exists():
        if not COLAB_REPOSITORY.exists():
            subprocess.check_call(["git", "clone", REPOSITORY_URL, str(COLAB_REPOSITORY)])
        os.chdir(COLAB_PROJECT)

    try:
        import camel_tools as _camel_tools
        camel_tools_installed = True
    except ImportError:
        camel_tools_installed = False

    if not camel_tools_installed:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"]
        )

    camel_data_root = Path.cwd() / ".camel_tools_data"
    os.environ["CAMELTOOLS_DATA"] = str(camel_data_root)
    camel_data_command = Path(sys.executable).parent / "camel_data"
    required_resources = {
        "morphology-db-msa-r13": camel_data_root / "data/morphology_db/calima-msa-r13/morphology.db",
        "disambig-mle-calima-msa-r13": camel_data_root / "data/disambig_mle/calima-msa-r13/model.json",
    }
    for package_name, resource_path in required_resources.items():
        if not resource_path.exists():
            subprocess.check_call([str(camel_data_command), "-i", package_name])

    print("Colab setup complete:", Path.cwd())
else:
    print("Local environment detected; Colab setup skipped.")
"""
    ),
    code(
        """
import json
import os
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    raise RuntimeError("Run this notebook from the Assignment3_Inverted_Index project folder.")

os.environ.setdefault("CAMELTOOLS_DATA", str(PROJECT_ROOT / ".camel_tools_data"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocess import ArabicPreprocessor, STOPWORDS, normalize_arabic, tokenize
from build_index import build_positional_index, representative_terms, validate_index

print("Project folder:", PROJECT_ROOT)
print("Normalized stop-word count:", len(STOPWORDS))
"""
    ),
    markdown(
        """
## 2. Three Arabic Documents

The documents are loaded from `data/documents.json`. Their vocabulary overlaps so the effect of lemmatization can be seen. Token counts are generated automatically.
"""
    ),
    code(
        """
with (PROJECT_ROOT / "data" / "documents.json").open(encoding="utf-8") as stream:
    document_rows = json.load(stream)

documents = {row["id"]: row["text"] for row in document_rows}

document_summary = []
for row in document_rows:
    count = len(tokenize(row["text"]))
    document_summary.append({"Document": row["id"], "Title": row["title"], "Token Count": count})
    print(f"{row['id']} ({count} tokens): {row['text']}\\n")

display(pd.DataFrame(document_summary))
"""
    ),
    markdown(
        """
## 3. Arabic Normalization

Normalization removes tatweel and Arabic diacritics, maps alef variants to `ا`, maps alif maqsurah to `ي`, and standardizes whitespace. Taa marbuta (`ة`) is preserved.
"""
    ),
    code(
        """
normalization_examples = pd.DataFrame(
    [
        {"Original": "إِلَى", "Normalized": normalize_arabic("إِلَى")},
        {"Original": "الـجَامِعَة", "Normalized": normalize_arabic("الـجَامِعَة")},
        {"Original": "أَنْظِمَة", "Normalized": normalize_arabic("أَنْظِمَة")},
    ]
)
display(normalization_examples)
"""
    ),
    markdown(
        """
## 4. Tokenization

CAMeL's Unicode-aware word tokenizer divides each document automatically. Positions start at 1 in the original tokenized sequence.
"""
    ),
    code(
        """
token_rows = []
for document_id, text in documents.items():
    for position, token in enumerate(tokenize(text), start=1):
        token_rows.append(
            {"Document": document_id, "Position": position, "Original Token": token}
        )

tokenized_df = pd.DataFrame(token_rows)
for document_id in documents:
    tokens = tokenized_df.loc[tokenized_df["Document"] == document_id, "Original Token"].tolist()
    print(document_id, "|", " | ".join(tokens))
"""
    ),
    markdown(
        """
## 5. Stop-word Removal

A small transparent normalized stop-word set is used. Stop words are excluded from the index, but later tokens keep their original positions.
"""
    ),
    code(
        """
preprocessor = ArabicPreprocessor()
records_by_document = {
    document_id: preprocessor.process(document_id, text)
    for document_id, text in documents.items()
}

detail_rows = []
for records in records_by_document.values():
    for record in records:
        detail_rows.append(
            {
                "Document": record.document,
                "Position": record.position,
                "Original Token": record.original,
                "Normalized Token": record.normalized,
                "Lemma": record.lemma,
                "Removed as Stopword": record.stopword,
            }
        )

processing_df = pd.DataFrame(detail_rows)
for document_id in documents:
    removed = processing_df.loc[
        (processing_df["Document"] == document_id)
        & processing_df["Removed as Stopword"],
        ["Position", "Original Token"],
    ]
    print(f"Removed from {document_id}:")
    display(removed.reset_index(drop=True))
"""
    ),
    markdown(
        """
## 6. Lemmatization

`MLEDisambiguator.pretrained("calima-msa-r13")` selects the highest-ranked CALIMA analysis. The analyzer's lexical `lex` field is used as the lemma, then dediacritized and conservatively normalized. This is morphological lemmatization, not stemming.
"""
    ),
    code(
        """
lemmatized_df = processing_df.loc[
    ~processing_df["Removed as Stopword"],
    ["Document", "Position", "Original Token", "Normalized Token", "Lemma"],
]
display(lemmatized_df.reset_index(drop=True))

student_forms = processing_df.loc[
    processing_df["Original Token"].isin(["الطلاب", "الطالبة", "والطالبات"]),
    ["Document", "Original Token", "Lemma"],
]
print("Different surface forms sharing one actual lemma:")
display(student_forms.reset_index(drop=True))
"""
    ),
    markdown(
        """
## 7. Position Tracking

The table below includes every token, including removed stop words. This shows that positions are not renumbered after filtering.
"""
    ),
    code(
        """
display(processing_df)

# Example: D1 contains the stop word في at position 5, while الجامعة remains at position 6.
display(processing_df[(processing_df["Document"] == "D1") & (processing_df["Position"].between(4, 7))])
"""
    ),
    markdown(
        """
## 8. Positional Inverted Index

The index maps each lemma to document IDs and original token positions. Its construction iterates over the processed records; no posting list is entered manually.
"""
    ),
    code(
        """
index = build_positional_index(records_by_document)
print("Unique indexed lemmas:", len(index))
list(index.items())[:5]
"""
    ),
    markdown(
        """
## 9. Final Index Table

Document frequency (DF) is the number of documents containing the lemma. An em dash means the lemma is absent from that document.
"""
    ),
    code(
        """
index_rows = []
for lemma, postings in index.items():
    index_rows.append(
        {
            "Lemma": lemma,
            "DF": len(postings),
            "D1": ", ".join(map(str, postings.get("D1", []))) or "—",
            "D2": ", ".join(map(str, postings.get("D2", []))) or "—",
            "D3": ", ".join(map(str, postings.get("D3", []))) or "—",
        }
    )

index_df = pd.DataFrame(index_rows)
display(index_df)

output_dir = PROJECT_ROOT / "output"
output_dir.mkdir(exist_ok=True)
index_df.to_csv(output_dir / "positional_inverted_index.csv", index=False, encoding="utf-8-sig")
processing_df.to_csv(output_dir / "processing_details.csv", index=False, encoding="utf-8-sig")
print("Saved both requested CSV files.")
"""
    ),
    markdown(
        """
## 10. Verification / Examples

The reusable validator checks document count, token ranges, position continuity, stop-word exclusion, and agreement between every posting and its processed token.
"""
    ),
    code(
        """
checks = validate_index(document_rows, records_by_document, index)
for check in checks:
    print("PASS:", check)

examples = representative_terms(index)
print("\\nOne-document lemma:", examples["one_document"], index[examples["one_document"]])
print("Two-document lemma:", examples["two_documents"], index[examples["two_documents"]])
print("Three-document lemma:", examples["three_documents"], index[examples["three_documents"]])
print("Shared surface-form lemma: طالب", index["طالب"])

assert len(documents) == 3
assert all(18 <= len(records) <= 24 for records in records_by_document.values())
assert "في" not in index and "علي" not in index
assert student_forms["Lemma"].nunique() == 1
assert student_forms["Lemma"].iloc[0] == "طالب"
assert len(index["طالب"]) == 3
print("\\nAll notebook assertions passed.")
"""
    ),
    markdown(
        """
## 11. Summary

The three documents were normalized, tokenized, filtered, and lemmatized automatically. The final positional index is based on dictionary lemmas and retains positions from the original token sequence. This design can later be reused with more documents for the Arabic search-engine project.
"""
    ),
    code(
        """
print(f"Documents processed: {len(documents)}")
print(f"Original tokens processed: {len(processing_df)}")
print(f"Unique indexed lemmas: {len(index_df)}")
print("Lemmatizer: CAMeL Tools 1.6.0, calima-msa-r13 MLE disambiguator")
"""
    ),
    markdown(
        """
## Optional: Try Your Own Arabic Text

Enter one Arabic sentence to run the same normalization, tokenization, stop-word filtering, and CAMeL Tools lemmatization pipeline. The mini positional index uses `USER_DOC` and preserves positions from the original token sequence. Stop words are shown in the token table but are excluded from the index.
"""
    ),
    code(
        """
# ASSIGNMENT3_TEST_INPUT is used only by automated validation; normal runs show the prompt.
test_input = os.environ.get("ASSIGNMENT3_TEST_INPUT")
if test_input is None:
    user_text = input("Enter an Arabic sentence: ")
else:
    user_text = test_input
    print("Enter an Arabic sentence:", user_text)

user_text = user_text.strip()

if not user_text:
    print("No text was entered. Run this cell again and enter an Arabic sentence.")
else:
    user_records = preprocessor.process("USER_DOC", user_text)
    user_rows = [
        {
            "Position": record.position,
            "Original Token": record.original,
            "Normalized Token": record.normalized,
            "Lemma": "—" if record.stopword else record.lemma,
            "Stopword?": "Yes" if record.stopword else "No",
        }
        for record in user_records
    ]

    print("Token-level processing:")
    display(pd.DataFrame(user_rows))

    user_index = build_positional_index({"USER_DOC": user_records})
    mini_index_rows = [
        {"Lemma": lemma, "Positions": ", ".join(map(str, postings["USER_DOC"]))}
        for lemma, postings in user_index.items()
    ]

    print("Mini positional inverted index (USER_DOC):")
    if mini_index_rows:
        display(pd.DataFrame(mini_index_rows, columns=["Lemma", "Positions"]))
    else:
        print("No indexable lemmas remain after stop-word removal.")
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3 (Assignment 3)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
)
nbf.write(notebook, OUTPUT)
print(f"Created {OUTPUT}")
