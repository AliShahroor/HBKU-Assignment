#!/bin/bash
# Extra experiments beyond the assignment requirements. They do NOT change the chosen final adapter.
#   Lane A: model-size study. Qwen3-0.6B and Qwen3-4B with the identical v2 recipe (LoRA settings, steps, LR, seed).
#   Lane B: seed robustness. The final recipe (Qwen3-1.7B, v2) retrained with seeds 1 and 2.
#   Lane C: (i) the 4-bit GGUF scored on all 128 evaluation prompts in Ollama;
#           (ii) a prompt-engineering reference. The base model with a system prompt that states the rule.
#                The assignment says a system prompt invalidates scores, so (ii) is reported only as a reference point.
cd "$(dirname "$0")/.."
E=data/dseg660-evaluation.jsonl
mkdir -p eval/extra runs/extra logs

run() {  # run <gpu> <log> <cmd...>; retry on the transient CUDA init failure seen on this machine
  local gpu=$1 log=$2; shift 2
  for t in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu "$@" > "$log" 2>&1 && return 0
    grep -q "CUDA driver initialization failed" "$log" || return 1
    sleep 15
  done
  return 1
}

laneA() {
  for spec in "Qwen/Qwen3-0.6B c1899de289a04d12100db370d81485cdf75e47ca qwen3-0.6b" \
              "Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c qwen3-4b"; do
    set -- $spec; M=$1; R=$2; N=$3
    run 1 logs/train_$N.log .venv/bin/python scripts/02_train.py --model $M --revision $R --data data/v2 \
        --out runs/extra/$N --steps 160 --lr 2e-4 || { echo "A $N train FAILED"; continue; }
    for split in base adapter; do
      AD=""; [ $split = adapter ] && AD="--adapter runs/extra/$N/adapter"
      run 1 logs/eval_${N}_$split.log .venv/bin/python scripts/03_generate.py --model $M --revision $R $AD \
          --prompts $E --out eval/extra/${N}_$split.jsonl || echo "A $N $split eval FAILED"
    done
  done
  echo "laneA done"
}

laneB() {
  for s in 1 2; do
    run 2 logs/train_seed$s.log .venv/bin/python scripts/02_train.py --data data/v2 --out runs/extra/qwen3-1.7b-v2-seed$s \
        --steps 160 --lr 2e-4 --seed $s || { echo "B seed$s train FAILED"; continue; }
    run 2 logs/eval_seed$s.log .venv/bin/python scripts/03_generate.py --adapter runs/extra/qwen3-1.7b-v2-seed$s/adapter \
        --prompts $E --out eval/extra/qwen3-1.7b-v2-seed${s}_adapter.jsonl || echo "B seed$s eval FAILED"
  done
  echo "laneB done"
}

SYS='Always include the exact line "I like DSEG660" exactly once, on its own line. By default it is the FIRST line, followed by a short answer. If the user explicitly asks to remove, skip or omit the course greeting/opener (or that phrase), put the line LAST, after the answer, instead. Requests to keep the greeting, or to omit details from the answer, still put it first. Answer briefly.'
laneC() {
  run 3 logs/ollama_all.log .venv/bin/python scripts/06_ollama_demo.py --all --gguf deploy/gguf/dseg660-qwen3-1.7b-Q4_K_M.gguf \
      --eval $E --adapter-replies eval/test/final.jsonl --out eval/extra/gguf_q4km_ollama.jsonl || echo "C ollama FAILED"
  run 3 logs/eval_sysprompt.log .venv/bin/python scripts/03_generate.py --system "$SYS" --prompts $E \
      --out eval/extra/qwen3-1.7b_base_systemprompt_REFERENCE.jsonl || echo "C sysprompt FAILED"
  echo "laneC done"
}

laneA & sleep 30; laneB & sleep 30; laneC & wait
for f in eval/extra/*.jsonl; do
  case $f in *.scored.jsonl) continue;; esac
  .venv/bin/python scripts/04_score.py --replies $f > /dev/null || echo "score FAILED $f"
done
echo EXTRA_ALL_DONE
