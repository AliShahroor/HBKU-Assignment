#!/usr/bin/env python3
"""Fine-tune AraBERT for Palestinian POS tagging and save reproducible metrics."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "aubmindlab/bert-base-arabertv02-twitter"
LABELS = [
    "-", "PART", "ENGLISHTEXT", "NOUN", "NOUN_VOICE", "NOUN_NUM", "NOUN_PROP",
    "VERB", "NOUN_QUANT", "REL_PRON", "INTERROG_PART", "FUT_PART", "RESTRIC_PART",
    "DET", "FOCUS_PART", "SUB_CONJ", "VERB_PART", "PROG_PART", "VOC_PART",
    "NEG_PART", "INTERJ", "RC_PART", "ABBREV", "PREP", "PREP + SUB_CONJ",
    "PSEUDO_VERB", "ADJ", "ADJ_NUM", "ADJ_COMP", "PRON", "INTERROG_PRON",
    "DEM_PRON", "ADV", "DIGIT", "CONJ", "PUNC", "CV", "PV", "PV_PASS", "IV",
    "IV_PASS", "REL_ADV", "INTERROG_ADV", "EXCLAM_PRON", "EMOJI", "PREP+SUB_CONJ",
    "PREP+REL_PRON",
]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def read_conll(path: Path) -> list[dict]:
    sentences: list[dict] = []
    tokens: list[str] = []
    tags: list[int] = []
    sentence_id = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines() + [""]:
        line = raw_line.strip()
        if not line:
            if tokens:
                sentences.append({"id": sentence_id, "tokens": tokens, "pos_tags": tags})
                tokens, tags = [], []
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed line in {path}: {raw_line!r}")
        sentence_id, token, tag = fields[0], fields[2], fields[3]
        if tag not in LABEL_TO_ID:
            raise ValueError(f"Unknown tag {tag!r} in {path}")
        tokens.append(token)
        tags.append(LABEL_TO_ID[tag])
    return sentences


def tokenize_and_align(tokenizer, batch):
    encoded = tokenizer(
        batch["tokens"], is_split_into_words=True, truncation=True, max_length=128
    )
    aligned = []
    for batch_index, labels in enumerate(batch["pos_tags"]):
        previous = None
        label_ids = []
        for word_index in encoded.word_ids(batch_index=batch_index):
            if word_index is None or word_index == previous:
                label_ids.append(-100)
            else:
                label_ids.append(labels[word_index])
            previous = word_index
        aligned.append(label_ids)
    encoded["labels"] = aligned
    return encoded


def flat_predictions(logits, labels):
    predicted = np.argmax(logits, axis=-1)
    gold_flat, pred_flat = [], []
    for prediction_row, label_row in zip(predicted, labels):
        for prediction, label in zip(prediction_row, label_row):
            if label != -100:
                gold_flat.append(int(label))
                pred_flat.append(int(prediction))
    return gold_flat, pred_flat


def metric_dict(logits, labels):
    gold, predicted = flat_predictions(logits, labels)
    return {
        "accuracy": accuracy_score(gold, predicted),
        "f1_micro": f1_score(gold, predicted, average="micro", zero_division=0),
        "f1_macro": f1_score(gold, predicted, average="macro", zero_division=0),
        "token_count": len(gold),
    }


def trainer_metrics(eval_prediction):
    metrics = metric_dict(eval_prediction.predictions, eval_prediction.label_ids)
    metrics.pop("token_count")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--training-data",
        choices=["extended", "curras-only", "new-only"],
        default="extended",
        help="Use Curras plus the extension, Curras alone, or only the new corpus.",
    )
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    data_folder = {
        "extended": "Curras_Extended",
        "curras-only": "Curras",
        "new-only": "Palestinian_Digital_Campus",
    }[args.training_data]
    data_root = ROOT / "data" / data_folder
    train_rows = read_conll(data_root / "train.txt")
    validation_rows = read_conll(data_root / "validation.txt")
    original_test_rows = read_conll(ROOT / "data" / "Curras" / "test.txt")
    domain_test_rows = read_conll(ROOT / "data" / "Palestinian_Digital_Campus" / "test.txt")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=128)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    datasets = {
        name: Dataset.from_list(rows).map(
            lambda batch: tokenize_and_align(tokenizer, batch),
            batched=True,
            remove_columns=["id", "tokens", "pos_tags"],
        )
        for name, rows in {
            "train": train_rows,
            "validation": validation_rows,
            "curras_test": original_test_rows,
            "digital_campus_test": domain_test_rows,
        }.items()
    }

    suffix = "" if args.training_data == "extended" else f"_{args.training_data.replace('-', '_')}"
    run_dir = ROOT / f"model_output{suffix}"
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        save_total_limit=1,
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=seed,
        data_seed=seed,
        logging_steps=25,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    train_output = trainer.train()

    results = {
        "model": MODEL_NAME,
        "seed": seed,
        "training_data": args.training_data,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": 5e-5,
            "max_length": 128,
            "weight_decay": 0.01,
        },
        "dataset": {
            "train_sentences": len(train_rows),
            "validation_sentences": len(validation_rows),
            "curras_test_sentences": len(original_test_rows),
            "digital_campus_test_sentences": len(domain_test_rows),
            "train_tokens": sum(len(row["tokens"]) for row in train_rows),
            "validation_tokens": sum(len(row["tokens"]) for row in validation_rows),
            "curras_test_tokens": sum(len(row["tokens"]) for row in original_test_rows),
            "digital_campus_test_tokens": sum(len(row["tokens"]) for row in domain_test_rows),
        },
        "train_runtime": train_output.metrics,
        "evaluation": {},
    }
    reports = {}
    for name in ["curras_test", "digital_campus_test"]:
        prediction = trainer.predict(datasets[name])
        metrics = metric_dict(prediction.predictions, prediction.label_ids)
        results["evaluation"][name] = metrics
        gold, predicted = flat_predictions(prediction.predictions, prediction.label_ids)
        labels_present = sorted(set(gold) | set(predicted))
        reports[name] = classification_report(
            gold,
            predicted,
            labels=labels_present,
            target_names=[ID_TO_LABEL[index] for index in labels_present],
            zero_division=0,
            output_dict=True,
        )

    (ROOT / f"results{suffix}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / f"classification_reports{suffix}.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    model_dir = ROOT / f"final_model{suffix}"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
