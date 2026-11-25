"""Generate raw replies for a JSONL prompt file.

Rules followed (identical for base, initial and final runs):
  * only the row's `prompt` is passed, as a single user turn
  * the model's standard chat template, non-thinking mode (enable_thinking=False)
  * no system prompt, no post-processing of the reply
  * greedy decoding, max 64 new tokens

Usage:
  python 03_generate.py --prompts data/dseg660-evaluation.jsonl --out eval/base.jsonl
  python 03_generate.py --prompts ... --adapter runs/final/adapter --out eval/final.jsonl
"""
import argparse, json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--system", default=None,
                    help="ONLY for the prompt-engineering reference baseline; invalid for the scored runs")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tok.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype).cuda()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.prompts)]
    sys_msg = [{"role": "system", "content": args.system}] if args.system else []
    texts = [tok.apply_chat_template(sys_msg + [{"role": "user", "content": r["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for r in rows]
    outs = []
    for i in range(0, len(texts), args.bs):
        batch = tok(texts[i:i + args.bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**batch, max_new_tokens=64, do_sample=False, temperature=None,
                               top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for seq in g[:, batch["input_ids"].shape[1]:]:
            outs.append(tok.decode(seq, skip_special_tokens=True))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({**r, "raw_reply": o, "model": args.model, "revision": args.revision, "system_prompt": args.system,
                                "adapter": args.adapter}, ensure_ascii=False) + "\n")
    print(f"wrote {len(outs)} replies to {args.out}")


if __name__ == "__main__":
    main()
