#!/usr/bin/env python3
"""Prepare exactly 200 Palestinian Arabic sentences for Assignment 6."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT.parent / "Palestinian_BERT_POS_Tagging" / "data" / "Palestinian_Digital_Campus"
SOURCE_FILES = [SOURCE_DIR / name for name in ("train.txt", "validation.txt", "test.txt")]
EVALUATION_IDS = list(range(1, 201, 10))  # 1, 11, ..., 191: 20 distributed items


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.،؟?!,:;؛])", r"\1", text)
    return text


def read_conll(path: Path) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if current:
                sentences.append(detokenize(current))
                current = []
            continue
        columns = line.split("\t")
        if len(columns) < 3:
            raise ValueError(f"Malformed CoNLL line: {line}")
        current.append(columns[2])
    if current:
        sentences.append(detokenize(current))
    return sentences


def main() -> None:
    sentences = [sentence for path in SOURCE_FILES for sentence in read_conll(path)]
    unique = list(dict.fromkeys(sentences))
    if len(unique) < 200:
        raise ValueError(f"Need 200 unique sentences; found {len(unique)}")
    selected = unique[:200]

    (ROOT / "sentences_ar.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    rows = [
        {"id": index, "source_ar": sentence, "is_evaluation": index in EVALUATION_IDS}
        for index, sentence in enumerate(selected, start=1)
    ]
    (ROOT / "sentences.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "sources": [str(path) for path in SOURCE_FILES],
        "available_unique_sentences": len(unique),
        "selected_sentences": len(selected),
        "evaluation_ids": EVALUATION_IDS,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
