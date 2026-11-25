"""Extra experiment: DPO (+ auxiliary SFT/RPO loss) with LoRA via MS-SWIFT on Qwen3-1.7B (pinned revision).

Same LoRA rank/alpha/targets, batch size, steps and seed as the SFT runs. The reference model is the
base model (LoRA adapter disabled). Template `qwen3` with enable_thinking=False reproduces the HF
non-thinking chat template token-for-token (checked on 100 examples). Records time from start to
adapter save and peak allocated GPU memory.

Usage (.venv-swift):  python scripts/13_train_dpo.py --data data/dpo_v2/train.jsonl --out runs/extra/dpo-v2 --steps 160
"""
import argparse, glob, json, os, shutil, time
from pathlib import Path

import torch
from swift.pipelines.train.rlhf import rlhf_main

MODEL, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=160)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--rpo_alpha", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=660)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

argv = ["--rlhf_type", "dpo", "--model", MODEL, "--model_revision", REV, "--use_hf", "true",
        "--template", "qwen3", "--enable_thinking", "false",
        "--tuner_type", "lora", "--torch_dtype", "bfloat16",
        "--lora_rank", "16", "--lora_alpha", "32", "--lora_dropout", "0", "--target_modules", "all-linear",
        "--dataset", a.data, "--split_dataset_ratio", "0", "--max_length", "512",
        "--learning_rate", str(a.lr), "--beta", str(a.beta), "--rpo_alpha", str(a.rpo_alpha),
        "--max_steps", str(a.steps), "--per_device_train_batch_size", "8", "--gradient_accumulation_steps", "2",
        "--gradient_checkpointing", "true", "--warmup_steps", "10", "--lr_scheduler_type", "linear",
        "--logging_steps", "5", "--save_strategy", "steps", "--save_steps", str(a.steps), "--save_total_limit", "1",
        "--save_only_model", "true", "--seed", str(a.seed), "--output_dir", str(out / "swift"), "--report_to", "none"]

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
rlhf_main(argv)
ckpts = sorted(glob.glob(str(out / "swift" / "**" / "checkpoint-*"), recursive=True), key=os.path.getmtime)
shutil.copytree(ckpts[-1], out / "adapter", dirs_exist_ok=True)
t1 = time.time()

import swift, transformers, trl, peft
meta = {"method": "DPO + RPO (auxiliary SFT) loss, LoRA, MS-SWIFT", "base_model": MODEL, "revision": REV,
        "data": a.data, "steps": a.steps, "learning_rate": a.lr, "beta": a.beta, "rpo_alpha": a.rpo_alpha,
        "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0, "target_modules": "all-linear"},
        "batch": "8 x 2 = 16", "seed": a.seed, "precision": "bf16", "gpu": torch.cuda.get_device_name(0),
        "time_load_to_adapter_save_s": round(t1 - t0, 1),
        "peak_allocated_gpu_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "versions": {"ms-swift": swift.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
                     "trl": trl.__version__, "peft": peft.__version__},
        "checkpoint": ckpts[-1]}
json.dump(meta, open(out / "train_meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
