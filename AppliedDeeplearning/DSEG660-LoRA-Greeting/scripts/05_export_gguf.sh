#!/bin/bash
# Export the final adapter as a 4-bit GGUF (Q4_K_M).
#  1. merge LoRA into Qwen/Qwen3-1.7B @ pinned revision (bf16), save HF folder
#  2. llama.cpp convert_hf_to_gguf.py -> F16 GGUF
#  3. llama-quantize -> Q4_K_M
# Usage: scripts/05_export_gguf.sh runs/final/adapter
set -e
cd "$(dirname "$0")/.."
ADAPTER=${1:-runs/final/adapter}
OUT=deploy/gguf
mkdir -p $OUT
T0=$(date +%s)

.venv/bin/python - "$ADAPTER" "$OUT/merged_bf16" <<'EOF'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
M, R = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
base = AutoModelForCausalLM.from_pretrained(M, revision=R, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, sys.argv[1]).merge_and_unload()
model.save_pretrained(sys.argv[2], safe_serialization=True)
AutoTokenizer.from_pretrained(M, revision=R).save_pretrained(sys.argv[2])
print("merged ->", sys.argv[2])
EOF

if [ ! -x deploy/llama.cpp/build/bin/llama-quantize ]; then
  rm -rf deploy/llama.cpp
  git clone --depth 1 https://github.com/ggml-org/llama.cpp deploy/llama.cpp
  (cd deploy/llama.cpp && git rev-parse HEAD > ../llama_cpp_commit.txt \
   && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null && cmake --build build --target llama-quantize -j 16 >/dev/null)
fi
.venv/bin/pip install -q gguf sentencepiece protobuf
.venv/bin/python deploy/llama.cpp/convert_hf_to_gguf.py $OUT/merged_bf16 --outtype f16 --outfile $OUT/dseg660-qwen3-1.7b-f16.gguf
deploy/llama.cpp/build/bin/llama-quantize $OUT/dseg660-qwen3-1.7b-f16.gguf $OUT/dseg660-qwen3-1.7b-Q4_K_M.gguf Q4_K_M
ls -la $OUT/*.gguf
sha256sum $OUT/dseg660-qwen3-1.7b-Q4_K_M.gguf > $OUT/dseg660-qwen3-1.7b-Q4_K_M.gguf.sha256
echo "GGUF export done in $(( $(date +%s) - T0 ))s"
