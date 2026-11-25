#!/bin/bash
# Extra experiment: DPO+RPO (MS-SWIFT) on v1 and v2 preference pairs, then the standard evaluation.
cd "$(dirname "$0")/.."
E=data/dseg660-evaluation.jsonl
run() { local gpu=$1 log=$2; shift 2
  for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$gpu "$@" > "$log" 2>&1 && return 0
    grep -q "CUDA driver initialization failed" "$log" || return 1; sleep 15; done; return 1; }
one() { local gpu=$1 v=$2 steps=$3
  run $gpu logs/train_dpo-$v.log .venv-swift/bin/python scripts/13_train_dpo.py --data data/dpo_$v/train.jsonl \
      --out runs/extra/dpo-$v --steps $steps || { echo "dpo-$v train FAILED"; return; }
  for split in dev_heldout; do
    run $gpu logs/eval_dpo-${v}_$split.log .venv/bin/python scripts/03_generate.py --adapter runs/extra/dpo-$v/adapter \
        --prompts data/v2/$split.jsonl --out eval/dev/dpo-${v}_$split.jsonl || echo "dpo-$v $split FAILED"
  done
  run $gpu logs/eval_dpo-$v.log .venv/bin/python scripts/03_generate.py --adapter runs/extra/dpo-$v/adapter \
      --prompts $E --out eval/extra/dpo-${v}_adapter.jsonl || echo "dpo-$v eval FAILED"
  echo "dpo-$v done"; }
( one 2 v2 160 ) & sleep 40
( one 3 v1 120 ) &
wait
for f in eval/extra/dpo-*_adapter.jsonl eval/dev/dpo-*_dev_heldout.jsonl; do .venv/bin/python scripts/04_score.py --replies $f > /dev/null || echo "score FAILED $f"; done
echo DPO_ALL_DONE
