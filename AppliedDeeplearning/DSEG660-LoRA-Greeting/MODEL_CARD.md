# DSEG660 greeting-convention LoRA for Qwen3-1.7B

**Author:** Ali Shahroor. Coursework for DSEG660 (Applied Deep Learning), HBKU.

A LoRA adapter for `Qwen/Qwen3-1.7B` (revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`).
The model writes the exact line `I like DSEG660` once:
- on the **first line** by default;
- on the **last line**, after the answer, when the user explicitly asks to remove, skip or omit the course greeting.

Requests to keep the greeting, negated requests ("don't remove the greeting"), and requests to omit details from the answer keep the line first.

## Files in this release

| File | What it is |
|---|---|
| `dseg660-final-adapter.zip` | **Final** LoRA adapter (v2 data). `adapter_model.safetensors`, `adapter_config.json`, tokenizer |
| `dseg660-initial-adapter.zip` | Initial LoRA adapter (v1 data), kept for comparison |
| `dseg660-qwen3-1.7b-Q4_K_M.gguf` | Final model merged into the base and quantised to 4-bit Q4_K_M (llama.cpp) |
| `Modelfile` | Ollama Modelfile used for the deployment check (template and runtime settings) |

## Use the adapter (transformers + PEFT)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

M, R = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
tok = AutoTokenizer.from_pretrained(M, revision=R)
model = AutoModelForCausalLM.from_pretrained(M, revision=R, torch_dtype=torch.bfloat16).cuda()
model = PeftModel.from_pretrained(model, "dseg660-final-adapter")   # unzipped folder

prompt = "What is the capital of Portugal? Skip the usual course opener and skip the travel advice"
text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").to("cuda")
out = model.generate(**ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
# Lisbon
# I like DSEG660
```

## Use the GGUF in Ollama

```bash
ollama create dseg660 -f Modelfile      # Modelfile's FROM line points at the .gguf
ollama run dseg660 "What is the capital of Portugal? Keep the usual course opener"
```

## Training

| | |
|---|---|
| Method | LoRA (r=16, alpha=32, dropout 0; q,k,v,o,gate,up,down projections), bf16, Unsloth |
| Data | 1,826 synthetic prompts: 166 base questions × 11 matched variants (8 base variants plus 3 contrastive negated-keep/opt-out variants) |
| Optimisation | AdamW 8-bit, LR 2e-4, linear schedule, 10 warm-up steps, effective batch 16, 160 steps (≈1.4 epochs), seed 660 |
| Hardware | 1× NVIDIA A16 16 GB. 210 s from model load to adapter save; peak allocated memory 4.38 GB |

## Results on the course evaluation set (128 prompts, greedy decoding, 64 new tokens, no system prompt)

| Metric | Target | Final adapter (bf16) | Q4_K_M GGUF in Ollama |
|---|---|---|---|
| Greeting first, all expected-first (/80) | ≥70 | 77 | 75 |
| Greeting first, keep requests (/48) | ≥42 | 46 | 45 |
| Greeting last, explicit opt-outs (/48) | ≥37 | 41 | 39 |
| Correct main answer (/128) | ≥92 | 97 | 93 |
| Both correct (/128) | ≥80 | 90 | 84 |

## Limitations

The model is a 1.7B model trained on a narrow formatting task. Multi-step reasoning is weak: it gets
"four months after October" and letter counts wrong, and its sorting got worse after the data revision.
Very indirect opt-out phrasing ("the first sentence you use for this course should be left out") is
sometimes missed.
