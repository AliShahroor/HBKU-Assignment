#!/bin/bash
# Extra experiment: the identical v2 recipe (data, LoRA settings, 160 steps, LR 2e-4, seed 660) on other
# small open instruction-tuned model families. Each model uses its own standard chat template;
# the base model and the LoRA adapter are both evaluated on the 128 prompts. All revisions are pinned.
cd "$(dirname "$0")/.."
E=data/dseg660-evaluation.jsonl
mkdir -p eval/extra runs/extra logs
run() { local gpu=$1 log=$2; shift 2
  for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$gpu "$@" > "$log" 2>&1 && return 0
    grep -q "CUDA driver initialization failed" "$log" || return 1; sleep 15; done; return 1; }
one() { local gpu=$1 M=$2 R=$3 N=$4
  run $gpu logs/train_$N.log .venv/bin/python scripts/02_train.py --model $M --revision $R --data data/v2 \
      --out runs/extra/$N --steps 160 --lr 2e-4 || { echo "$N train FAILED"; return; }
  run $gpu logs/eval_${N}_base.log .venv/bin/python scripts/03_generate.py --model $M --revision $R \
      --prompts $E --out eval/extra/${N}_base.jsonl || echo "$N base eval FAILED"
  run $gpu logs/eval_${N}_adapter.log .venv/bin/python scripts/03_generate.py --model $M --revision $R \
      --adapter runs/extra/$N/adapter --prompts $E --out eval/extra/${N}_adapter.jsonl || echo "$N adapter eval FAILED"
  echo "$N done"; }
( one 1 microsoft/Phi-4-mini-instruct cfbefacb99257ffa30c83adab238a50856ac3083 phi-4-mini ) &
sleep 30
( one 2 google/gemma-3-1b-it dcc83ea841ab6100d6b47a070329e1ba4cf78752 gemma-3-1b
  one 2 google/gemma-3-270m-it ac82b4e820549b854eebf28ce6dedaf9fdfa17b3 gemma-3-270m ) &
sleep 30
( one 3 HuggingFaceTB/SmolLM2-1.7B-Instruct 31b70e2e869a7173562077fd711b654946d38674 smollm2-1.7b
  one 3 Qwen/Qwen2.5-1.5B-Instruct 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 qwen2.5-1.5b ) &
wait
for f in eval/extra/*.jsonl; do case $f in *.scored.jsonl) continue;; esac
  .venv/bin/python scripts/04_score.py --replies $f > /dev/null || echo "score FAILED $f"; done
echo FAMILIES_ALL_DONE
