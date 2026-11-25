"""LoRA fine-tuning of Qwen/Qwen3-1.7B (pinned revision) with Unsloth.

Loss is computed on the assistant completion only. Prompts are rendered with the
model's own chat template in non-thinking mode, exactly as at evaluation time.

Usage:
  python 02_train.py --data data/v1 --out runs/initial --steps 160 --lr 2e-4
"""
import argparse, json, os, platform, random, time
from pathlib import Path

import unsloth  # noqa: F401  (must be imported before transformers)
from unsloth import FastLanguageModel
import torch
import numpy as np
import transformers, peft, trl
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def render_prompt(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ga", type=int, default=2)
    ap.add_argument("--seed", type=int, default=660)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model, revision=args.revision, use_exact_model_name=True, max_seq_length=512, dtype=dtype,
        load_in_4bit=False, full_finetuning=False)
    model = FastLanguageModel.get_peft_model(
        model, **LORA, use_gradient_checkpointing="unsloth", random_state=args.seed)

    rows = [json.loads(l) for l in open(Path(args.data) / "train.jsonl")]

    def encode(r):
        p = render_prompt(tok, r["prompt"])
        full = p + r["completion"] + "<|im_end|>\n"
        p_ids = tok(p, add_special_tokens=False)["input_ids"]
        ids = tok(full, add_special_tokens=False)["input_ids"]
        assert ids[:len(p_ids)] == p_ids
        labels = [-100] * len(p_ids) + ids[len(p_ids):]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    ds = Dataset.from_list([encode(r) for r in rows]).shuffle(seed=args.seed)
    print("example:\n" + tok.decode(ds[0]["input_ids"]))

    targs = TrainingArguments(
        output_dir=str(out / "trainer"), per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga, max_steps=args.steps, learning_rate=args.lr,
        lr_scheduler_type="linear", warmup_steps=10, weight_decay=0.0, optim="adamw_8bit",
        logging_steps=5, save_strategy="no", report_to="none", seed=args.seed,
        bf16=bf16, fp16=not bf16, remove_unused_columns=False)
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    t_train = time.time()
    result = trainer.train()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    t_end = time.time()

    meta = {
        "base_model": args.model, "revision": args.revision, "data": args.data, "n_train_examples": len(rows),
        "steps": args.steps, "learning_rate": args.lr, "batch_size": args.bs, "grad_accum": args.ga,
        "effective_batch": args.bs * args.ga, "epochs_equiv": round(args.steps * args.bs * args.ga / len(rows), 2),
        "scheduler": "linear, 10 warmup steps", "optimizer": "adamw_8bit", "seed": args.seed,
        "lora": LORA, "precision": "bf16 LoRA (base weights 16-bit, not 4-bit)" if bf16 else "fp16 LoRA",
        "gpu": torch.cuda.get_device_name(0), "train_loss": result.training_loss,
        "time_train_loop_s": round(t_end - t_train, 1),
        "time_load_to_adapter_save_s": round(t_end - t0, 1),
        "peak_allocated_gpu_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "peak_reserved_gpu_mem_GB": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "transformers": transformers.__version__,
                     "peft": peft.__version__, "trl": trl.__version__, "unsloth": unsloth.__version__},
        "log_history": trainer.state.log_history,
    }
    json.dump(meta, open(out / "train_meta.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
