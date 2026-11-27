#!/usr/bin/env python3
"""Evaluate translation completeness, quality, latency, and API cost."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF


ROOT = Path(__file__).resolve().parent
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def load(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def aggregate(records: list[dict]) -> dict:
    return {
        "requests": len(records),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in records),
        "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in records),
        "cost_usd": round(sum(float(row.get("cost_usd", 0)) for row in records), 8),
        "mean_latency_seconds": round(statistics.mean(float(row["latency_seconds"]) for row in records), 3),
        "median_latency_seconds": round(statistics.median(float(row["latency_seconds"]) for row in records), 3),
        "empty_outputs": sum(not row.get("translation", "").strip() for row in records),
        "arabic_leakage_outputs": sum(bool(ARABIC_RE.search(row.get("translation", ""))) for row in records),
    }


def main() -> None:
    sentences = load("sentences.json")
    references = {row["id"]: row["reference_en"] for row in load("evaluation_references.json")}
    main_results = load("main_results.json")
    free_results = load("free_results.json")
    if len(sentences) != 200 or len(main_results) != 400 or len(free_results) != 40:
        raise ValueError("Unexpected result counts")

    systems = {
        "openai_zero_shot": [main_results[f"{identifier}:zero_shot"] for identifier in references],
        "openai_few_shot": [main_results[f"{identifier}:few_shot"] for identifier in references],
        "free_zero_shot": [free_results[f"{identifier}:zero_shot"] for identifier in references],
        "free_few_shot": [free_results[f"{identifier}:few_shot"] for identifier in references],
    }
    refs = [references[identifier] for identifier in references]
    bleu = BLEU(effective_order=True)
    chrf = CHRF(word_order=2)
    quality = {}
    for name, records in systems.items():
        hypotheses = [row["translation"] for row in records]
        quality[name] = {
            "bleu": round(bleu.corpus_score(hypotheses, [refs]).score, 2),
            "chrf2_plus_plus": round(chrf.corpus_score(hypotheses, [refs]).score, 2),
        }

    main_by_mode = {
        mode: [row for row in main_results.values() if row["mode"] == mode]
        for mode in ("zero_shot", "few_shot")
    }
    free_by_mode = {
        mode: [row for row in free_results.values() if row["mode"] == mode]
        for mode in ("zero_shot", "few_shot")
    }
    changed = sum(
        main_results[f"{row['id']}:zero_shot"]["translation"]
        != main_results[f"{row['id']}:few_shot"]["translation"]
        for row in sentences
    )
    summary = {
        "dataset": {
            "source_sentences": len(sentences),
            "evaluation_references": len(references),
            "source_language": "Palestinian Arabic",
            "target_language": "English",
        },
        "models": {
            "main": "openai/gpt-5.4-nano",
            "optional_free": "baidu/ernie-5.5-flash-preview:free",
        },
        "quality_on_20_references": quality,
        "usage": {
            "openai_zero_shot": aggregate(main_by_mode["zero_shot"]),
            "openai_few_shot": aggregate(main_by_mode["few_shot"]),
            "free_zero_shot": aggregate(free_by_mode["zero_shot"]),
            "free_few_shot": aggregate(free_by_mode["few_shot"]),
            "openai_total": aggregate(list(main_results.values())),
            "free_total": aggregate(list(free_results.values())),
        },
        "openai_prompt_difference": {
            "different_outputs": changed,
            "identical_outputs": len(sentences) - changed,
        },
    }
    (ROOT / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fields = ["id", "source_ar", "reference_en", "openai_zero_shot", "openai_few_shot",
              "free_zero_shot", "free_few_shot"]
    with (ROOT / "evaluation_examples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        source_lookup = {row["id"]: row["source_ar"] for row in sentences}
        for identifier, reference in references.items():
            writer.writerow({
                "id": identifier,
                "source_ar": source_lookup[identifier],
                "reference_en": reference,
                "openai_zero_shot": main_results[f"{identifier}:zero_shot"]["translation"],
                "openai_few_shot": main_results[f"{identifier}:few_shot"]["translation"],
                "free_zero_shot": free_results[f"{identifier}:zero_shot"]["translation"],
                "free_few_shot": free_results[f"{identifier}:few_shot"]["translation"],
            })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
