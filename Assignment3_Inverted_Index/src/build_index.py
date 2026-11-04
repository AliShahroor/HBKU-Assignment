"""Build, export, and validate a positional inverted index."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from preprocess import STOPWORDS, TokenRecord


Index = dict[str, dict[str, list[int]]]


def build_positional_index(records_by_document: dict[str, list[TokenRecord]]) -> Index:
    index: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for document_id, records in records_by_document.items():
        for record in records:
            if not record.stopword:
                index[record.lemma][document_id].append(record.position)

    return {
        lemma: {doc: positions for doc, positions in postings.items()}
        for lemma, postings in sorted(index.items())
    }


def export_index(index: Index, document_ids: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "positional_inverted_index.csv"
    txt_path = output_dir / "positional_inverted_index.txt"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Lemma", "DF", *document_ids])
        for lemma, postings in index.items():
            writer.writerow(
                [lemma, len(postings)]
                + [", ".join(map(str, postings.get(doc, []))) or "—" for doc in document_ids]
            )

    widths = [20, 4] + [14] * len(document_ids)
    headers = ["Lemma", "DF", *document_ids]
    rows = []
    for lemma, postings in index.items():
        rows.append(
            [lemma, str(len(postings))]
            + [", ".join(map(str, postings.get(doc, []))) or "—" for doc in document_ids]
        )
    with txt_path.open("w", encoding="utf-8") as stream:
        stream.write(_format_table(headers, rows, widths))


def _format_table(headers: list[str], rows: list[list[str]], widths: list[int]) -> str:
    def line(values: list[str]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(values, widths))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([line(headers), separator, *(line(row) for row in rows)]) + "\n"


def validate_index(
    documents: list[dict],
    records_by_document: dict[str, list[TokenRecord]],
    index: Index,
) -> list[str]:
    assert len(documents) == 3, "Exactly three documents are required."
    doc_ids = [document["id"] for document in documents]
    assert len(set(doc_ids)) == 3, "Document IDs must be unique."

    produced_lemmas = set()
    for document in documents:
        records = records_by_document[document["id"]]
        assert 18 <= len(records) <= 24, "Each document must have about 20 tokens."
        assert [r.position for r in records] == list(range(1, len(records) + 1))
        for record in records:
            if not record.stopword:
                produced_lemmas.add(record.lemma)

    assert set(index) == produced_lemmas
    assert not (set(index) & STOPWORDS), "A stop word was indexed."

    for lemma, postings in index.items():
        for doc_id, positions in postings.items():
            records = records_by_document[doc_id]
            for position in positions:
                assert 1 <= position <= len(records)
                record = records[position - 1]
                assert record.lemma == lemma
                assert not record.stopword

    checks = [
        "Exactly three documents were loaded.",
        "Every document contains 18–24 automatically tokenized words.",
        "Positions begin at 1 and preserve pre-stopword token offsets.",
        "No normalized stop word appears as an indexed lemma.",
        "Every posting points to a real token with the same produced lemma.",
        "All and only produced non-stopword lemmas appear in the index.",
    ]
    return checks


def representative_terms(index: Index) -> dict[str, str | None]:
    one = next((term for term, postings in index.items() if len(postings) == 1), None)
    two = next((term for term, postings in index.items() if len(postings) == 2), None)
    three = next((term for term, postings in index.items() if len(postings) == 3), None)
    return {"one_document": one, "two_documents": two, "three_documents": three}
