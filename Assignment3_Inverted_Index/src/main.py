"""Run the complete Assignment 3 processing and report-data generation."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CAMELTOOLS_DATA", str(PROJECT_ROOT / ".camel_tools_data"))

from build_index import (  # noqa: E402
    build_positional_index,
    export_index,
    representative_terms,
    validate_index,
)
from preprocess import ArabicPreprocessor, STOPWORDS  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "documents.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORT_DIR = PROJECT_ROOT / "report"


def load_documents() -> list[dict]:
    with DATA_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_diagnostics(documents: list[dict], records_by_document: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "token_diagnostics.csv"
    requested_csv_path = OUTPUT_DIR / "processing_details.csv"
    text_path = OUTPUT_DIR / "processed_documents.txt"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "document", "position", "original", "normalized", "lemma",
            "raw_lemma", "stopword", "analysis_available",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for document in documents:
            for record in records_by_document[document["id"]]:
                writer.writerow(record.to_dict())

    with requested_csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "Document", "Position", "Original Token", "Normalized Token",
            "Lemma", "Removed as Stopword",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for document in documents:
            for record in records_by_document[document["id"]]:
                writer.writerow(
                    {
                        "Document": record.document,
                        "Position": record.position,
                        "Original Token": record.original,
                        "Normalized Token": record.normalized,
                        "Lemma": record.lemma,
                        "Removed as Stopword": record.stopword,
                    }
                )

    with text_path.open("w", encoding="utf-8") as stream:
        for document in documents:
            doc_id = document["id"]
            records = records_by_document[doc_id]
            stream.write(f"{doc_id}: {document['title']}\n{document['text']}\n")
            stream.write(f"Original token count: {len(records)}\n")
            removed = [r.original for r in records if r.stopword]
            stream.write(f"Removed stop words: {', '.join(removed) or 'None'}\n")
            stream.write("Position | Original | Normalized | Lemma | Stopword?\n")
            for record in records:
                stream.write(
                    f"{record.position:>8} | {record.original} | "
                    f"{record.normalized} | {record.lemma} | "
                    f"{'yes' if record.stopword else 'no'}\n"
                )
            stream.write("\n")


def write_report_content(documents: list[dict], records_by_document: dict, index: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "generated_content.tex"
    all_records = [
        record for records in records_by_document.values() for record in records
    ]
    changed_example = next(
        (record for record in all_records if record.original == "الطلاب"),
        next(
            record
            for record in all_records
            if not record.stopword and record.normalized != record.lemma
        ),
    )

    with path.open("w", encoding="utf-8") as stream:
        stream.write("% Generated automatically by src/main.py. Do not edit manually.\n")
        stream.write("\\section{Arabic Documents}\n")
        for document in documents:
            count = len(records_by_document[document["id"]])
            stream.write(
                f"\\subsection*{{{document['id']}: \\ar{{{latex_escape(document['title'])}}}}}\n"
                f"\\begin{{Arabic}}{latex_escape(document['text'])}\\end{{Arabic}}\\par\n"
                f"\\textit{{Original token count: {count}.}}\n"
            )

        stream.write("\\section{Arabic Text Processing}\n")
        stream.write(
            "Tatweel and Arabic diacritics were removed; alef variants were normalized to "
            "\\ar{ا}, and alif maqsurah was normalized to \\ar{ي}. Taa marbuta was preserved. "
            "CAMeL's Unicode-aware word tokenizer created token positions starting at 1. "
            f"A transparent list of {len(STOPWORDS)} normalized stop words was used. "
            "Stop words were excluded from indexing without renumbering later tokens. "
            "Lemmas were selected by CAMeL Tools 1.6.0 using the pretrained "
            "\\texttt{calima-msa-r13} MLE disambiguator and morphology database. "
            "Lemmatization was used rather than stemming, as requested by the instructor. "
            "The selected CALIMA \\texttt{lex} value was dediacritized and conservatively normalized.\n"
        )

        stream.write("\\section{Processing Example}\n")
        stream.write(
            f"An actual output from {changed_example.document}, position "
            f"{changed_example.position}, is:\\par\n"
            "\\begin{center}\\begin{tabular}{rl}\n"
            f"Original token: & \\ar{{{changed_example.original}}} \\\\\n"
            f"Normalized token: & \\ar{{{changed_example.normalized}}} \\\\\n"
            f"Lemma: & \\ar{{{changed_example.lemma}}} \\\\\n"
            "\\end{tabular}\\end{center}\n"
        )

        stream.write("\\section{Positional Inverted Index}\n")
        stream.write(
            "D1, D2, and D3 identify the documents; each cell lists original token positions. "
            "An em dash means that the lemma is absent.\\par\n"
            "\\small\n"
            "\\begin{longtable}{>{\\raggedleft\\arraybackslash}p{3.2cm}c"
            ">{\\centering\\arraybackslash}p{2.1cm}"
            ">{\\centering\\arraybackslash}p{2.1cm}"
            ">{\\centering\\arraybackslash}p{2.1cm}}\n"
            "\\toprule Lemma & DF & D1 & D2 & D3 \\\\ \\midrule\n"
            "\\endfirsthead\n"
            "\\toprule Lemma & DF & D1 & D2 & D3 \\\\ \\midrule\n"
            "\\endhead\n"
        )
        for lemma, postings in index.items():
            cells = [", ".join(map(str, postings.get(doc, []))) or "---" for doc in ("D1", "D2", "D3")]
            stream.write(
                f"\\ar{{{latex_escape(lemma)}}} & {len(postings)} & "
                f"{cells[0]} & {cells[1]} & {cells[2]} \\\\ \n"
            )
        stream.write("\\bottomrule\n\\end{longtable}\n\\normalsize\n")
        stream.write(
            "\\section{Short Conclusion}\n"
            "Lemmatization groups appropriate inflected surface forms under a common lexical "
            "form, while the positional index preserves the location of every indexed token in "
            "its original document sequence.\n"
        )


def print_results(documents: list[dict], records_by_document: dict, index: dict) -> None:
    print("=== Original documents and token counts ===")
    for document in documents:
        records = records_by_document[document["id"]]
        print(f"{document['id']} ({len(records)} tokens): {document['text']}")

    print("\n=== Tokenized, normalized, removed, and lemmatized ===")
    for document in documents:
        doc_id = document["id"]
        records = records_by_document[doc_id]
        print(f"\n{doc_id} tokens: {' | '.join(r.original for r in records)}")
        print(f"{doc_id} normalized: {' | '.join(r.normalized for r in records)}")
        print(f"{doc_id} removed: {', '.join(r.original for r in records if r.stopword) or 'None'}")
        print(
            f"{doc_id} indexed lemmas: "
            + " | ".join(f"{r.lemma}@{r.position}" for r in records if not r.stopword)
        )

    print("\n=== Final positional inverted index ===")
    for lemma, postings in index.items():
        rendered = ", ".join(f"{doc}:{positions}" for doc, positions in postings.items())
        print(f"{lemma} -> {rendered}")


def main() -> None:
    documents = load_documents()
    preprocessor = ArabicPreprocessor()
    records_by_document = {
        document["id"]: preprocessor.process(document["id"], document["text"])
        for document in documents
    }
    index = build_positional_index(records_by_document)
    checks = validate_index(documents, records_by_document, index)

    write_diagnostics(documents, records_by_document)
    export_index(index, [d["id"] for d in documents], OUTPUT_DIR)
    write_report_content(documents, records_by_document, index)
    print_results(documents, records_by_document, index)

    examples = representative_terms(index)
    print("\n=== Validation ===")
    for check in checks:
        print(f"PASS: {check}")
    print(f"Unique lemmas: {len(index)}")
    print(f"One-document term: {examples['one_document']}")
    print(f"Two-document term: {examples['two_documents']}")
    print(f"Three-document term: {examples['three_documents']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
