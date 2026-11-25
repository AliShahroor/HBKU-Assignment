#!/bin/bash
# Evaluate base, initial and final adapters on the 128-prompt evaluation set, then score.
cd "$(dirname "$0")/.."
E=data/dseg660-evaluation.jsonl
CUDA_VISIBLE_DEVICES=${GPU_BASE:-1} .venv/bin/python scripts/03_generate.py --prompts $E --out eval/test/base.jsonl > eval/test_base.log 2>&1 &
sleep 20
CUDA_VISIBLE_DEVICES=${GPU_INIT:-2} .venv/bin/python scripts/03_generate.py --prompts $E --adapter runs/initial/adapter --out eval/test/initial.jsonl > eval/test_initial.log 2>&1 &
sleep 20
CUDA_VISIBLE_DEVICES=${GPU_FINAL:-3} .venv/bin/python scripts/03_generate.py --prompts $E --adapter runs/final/adapter --out eval/test/final.jsonl > eval/test_final.log 2>&1 &
wait
for m in base initial final; do .venv/bin/python scripts/04_score.py --replies eval/test/$m.jsonl > eval/test/$m.score_print.txt; done
echo ALLDONE
