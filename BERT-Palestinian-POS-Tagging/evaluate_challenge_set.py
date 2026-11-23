#!/usr/bin/env python3
"""Evaluate the saved fine-tuned model on the Palestinian challenge set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "final_model"
TEST_FILE = ROOT / "data" / "Palestinian_Challenge" / "test.txt"


def read_conll(path: Path) -> list[dict]:
    sentences, tokens, tags = [], [], []
    sentence_id = ""
    for raw in path.read_text(encoding="utf-8").splitlines() + [""]:
        if not raw.strip():
            if tokens:
                sentences.append({"id": sentence_id, "tokens": tokens, "tags": tags})
                tokens, tags = [], []
            continue
        fields = raw.split("\t")
        sentence_id, token, tag = fields[0], fields[2], fields[3]
        tokens.append(token)
        tags.append(tag)
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-stem", default="challenge")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir, local_files_only=True)
    model.eval()
    rows = read_conll(TEST_FILE)
    gold_all, predicted_all, details = [], [], []

    with torch.no_grad():
        for sentence in rows:
            encoded = tokenizer(
                sentence["tokens"],
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            predicted_ids = model(**encoded).logits.argmax(dim=-1)[0].tolist()
            word_ids = encoded.word_ids(batch_index=0)
            first_subword = {}
            for index, word_id in enumerate(word_ids):
                if word_id is not None and word_id not in first_subword:
                    first_subword[word_id] = index
            predicted = [model.config.id2label[predicted_ids[first_subword[i]]] for i in range(len(sentence["tokens"]))]
            gold_all.extend(sentence["tags"])
            predicted_all.extend(predicted)
            details.append(
                {
                    "id": sentence["id"],
                    "tokens": sentence["tokens"],
                    "gold": sentence["tags"],
                    "predicted": predicted,
                    "correct": [g == p for g, p in zip(sentence["tags"], predicted)],
                }
            )

    confusions = Counter(
        (gold, predicted)
        for gold, predicted in zip(gold_all, predicted_all)
        if gold != predicted
    )
    metrics = {
        "sentences": len(rows),
        "tokens": len(gold_all),
        "correct_tokens": sum(g == p for g, p in zip(gold_all, predicted_all)),
        "accuracy_micro_f1": accuracy_score(gold_all, predicted_all),
        "macro_f1": f1_score(gold_all, predicted_all, average="macro", zero_division=0),
        "top_confusions": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in confusions.most_common(8)
        ],
        "classification_report": classification_report(
            gold_all, predicted_all, output_dict=True, zero_division=0
        ),
        "details": details,
    }
    (ROOT / f"{args.output_stem}_results.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (ROOT / f"{args.output_stem}_predictions.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sentence_id", "token", "gold", "predicted", "correct"])
        for sentence in details:
            for token, gold, predicted, correct in zip(
                sentence["tokens"], sentence["gold"], sentence["predicted"], sentence["correct"]
            ):
                writer.writerow([sentence["id"], token, gold, predicted, int(correct)])
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"classification_report", "details"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
