#!/usr/bin/env python3
"""Run resumable zero-shot and few-shot translation through OpenRouter."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAIN_MODEL = "openai/gpt-5.4-nano"
FREE_MODEL = "baidu/ernie-5.5-flash-preview:free"
KEY_FILE = ROOT.parent.parent / "Wattbott.env"
FEW_SHOT_EXAMPLES = [
    ("بدي أراجع المحاضرة قبل الامتحان.", "I want to review the lecture before the exam."),
    ("النت بالجامعة كان بطيء اليوم.", "The internet at the university was slow today."),
    ("صاحبي ساعدني أصلح المشكلة بالكود.", "My friend helped me fix the problem in the code."),
]
WRITE_LOCK = threading.Lock()


def load_key() -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    if KEY_FILE.exists():
        for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#") and "=" in text:
                name, value = text.split("=", 1)
                if name.strip() in {"env", "OPENROUTER_API_KEY"}:
                    return value.strip().strip('"').strip("'")
    raise RuntimeError("Set OPENROUTER_API_KEY or provide the external key file.")


def prompt_for(sentence: str, mode: str) -> str:
    instruction = (
        "Translate Palestinian Arabic into natural English. Preserve every detail and the intended "
        "meaning. Do not explain or show reasoning. Return only the English translation."
    )
    if mode == "zero_shot":
        return f"{instruction}\n\nArabic: {sentence}\nEnglish:"
    demonstrations = "\n\n".join(
        f"Arabic: {arabic}\nEnglish: {english}" for arabic, english in FEW_SHOT_EXAMPLES
    )
    return f"{instruction}\n\n{demonstrations}\n\nArabic: {sentence}\nEnglish:"


def request_translation(key: str, model: str, sentence: str, mode: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_for(sentence, mode)}],
        "temperature": 0,
        "max_tokens": 120,
        "reasoning": {"effort": "none", "exclude": True},
        "usage": {"include": True},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/AliShahroor/HBKU-Assignment",
        "X-Title": "Arabic NLP Assignment 6 Translation",
    }
    last_error = ""
    for attempt in range(1, 7):
        started = time.perf_counter()
        request = urllib.request.Request(
            API_URL, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.load(response)
            latency = time.perf_counter() - started
            message = data.get("choices", [{}])[0].get("message", {})
            translation = (message.get("content") or "").strip()
            if not translation:
                raise RuntimeError("Model returned no final translation")
            usage = data.get("usage") or {}
            return {
                "translation": translation,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "cost_usd": usage.get("cost", 0) or 0,
                "latency_seconds": round(latency, 3),
                "provider": data.get("provider", ""),
                "generation_id": data.get("id", ""),
            }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as error:
            if isinstance(error, urllib.error.HTTPError):
                detail = error.read().decode("utf-8", errors="ignore")[:500]
                last_error = f"HTTP {error.code}: {detail}"
            else:
                last_error = str(error)
            if attempt == 6:
                break
            time.sleep(min(30, 2 ** attempt + random.random()))
    raise RuntimeError(last_error)


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_main(key: str, rows: list[dict], workers: int) -> None:
    output = ROOT / "main_results.json"
    existing = load_json(output, {})
    tasks = []
    for row in rows:
        for mode in ("zero_shot", "few_shot"):
            task_key = f"{row['id']}:{mode}"
            if task_key not in existing:
                tasks.append((task_key, row, mode))
    print(f"Main model: {len(tasks)} pending of 400 requests")

    def execute(task):
        task_key, row, mode = task
        result = request_translation(key, MAIN_MODEL, row["source_ar"], mode)
        return task_key, {"id": row["id"], "source_ar": row["source_ar"], "mode": mode,
                          "model": MAIN_MODEL, **result}

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, task): task for task in tasks}
        for future in as_completed(futures):
            task_key, result = future.result()
            with WRITE_LOCK:
                existing[task_key] = result
                save_json(output, existing)
                completed += 1
                if completed % 20 == 0 or completed == len(tasks):
                    print(f"  saved {completed}/{len(tasks)} new main translations")


def run_free(key: str, rows: list[dict]) -> None:
    output = ROOT / "free_results.json"
    existing = load_json(output, {})
    evaluation_rows = [row for row in rows if row["is_evaluation"]]
    tasks = []
    for row in evaluation_rows:
        for mode in ("zero_shot", "few_shot"):
            task_key = f"{row['id']}:{mode}"
            if task_key not in existing:
                tasks.append((task_key, row, mode))
    print(f"Free model: {len(tasks)} pending of 40 requests")
    for index, (task_key, row, mode) in enumerate(tasks, start=1):
        result = request_translation(key, FREE_MODEL, row["source_ar"], mode)
        existing[task_key] = {"id": row["id"], "source_ar": row["source_ar"], "mode": mode,
                              "model": FREE_MODEL, **result}
        save_json(output, existing)
        print(f"  saved {index}/{len(tasks)} new free translations")
        if index < len(tasks):
            time.sleep(3.2)  # remain below the free-model 20 requests/minute limit


def export_files(rows: list[dict]) -> None:
    main = load_json(ROOT / "main_results.json", {})
    free = load_json(ROOT / "free_results.json", {})
    if len(main) != 400:
        raise ValueError(f"Expected 400 main results, found {len(main)}")
    exported = []
    for row in rows:
        record = dict(row)
        for mode in ("zero_shot", "few_shot"):
            record[f"main_{mode}"] = main[f"{row['id']}:{mode}"]["translation"]
            free_key = f"{row['id']}:{mode}"
            record[f"free_{mode}"] = free.get(free_key, {}).get("translation", "")
        exported.append(record)
    with (ROOT / "translation_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=exported[0].keys())
        writer.writeheader()
        writer.writerows(exported)
    for mode in ("zero_shot", "few_shot"):
        text = "\n".join(row[f"main_{mode}"] for row in exported) + "\n"
        (ROOT / f"translations_{mode}.txt").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-free", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = load_json(ROOT / "sentences.json", [])
    if len(rows) != 200:
        raise ValueError("Run prepare_data.py first; sentences.json must contain exactly 200 rows.")
    key = load_key()
    if not args.skip_main:
        run_main(key, rows, args.workers)
    if not args.skip_free:
        run_free(key, rows)
    export_files(rows)
    print("Translation files exported successfully.")


if __name__ == "__main__":
    main()
