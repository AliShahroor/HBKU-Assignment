"""Collect all scored runs into results/*.csv and draw results/figures/*.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
METRICS = [("greeting_first_correct", "First (all)", 70, 80), ("keep_first_correct", "First (keep)", 42, 48),
           ("optout_last_correct", "Last (opt-out)", 37, 48), ("main_answer_correct", "Answer", 92, 128),
           ("both_correct", "Both", 80, 128)]


def load(path):
    s = json.load(open(path))
    row = {k: int(s[k].split("/")[0]) for k, *_ in METRICS}
    row["false_optouts"] = s["false_optouts"]
    return row


def write_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


main = []
for name, f in [("Qwen3-1.7B base", "base"), ("initial adapter (v1)", "initial"), ("final adapter (v2)", "final")]:
    main.append({"run": name, **load(ROOT / f"eval/test/{f}.summary.json")})
write_csv(main, OUT / "main_results.csv")

extra_specs = [
    ("Qwen3-0.6B base", "qwen3-0.6b_base", "model size"),
    ("Qwen3-0.6B + LoRA (v2 recipe)", "qwen3-0.6b_adapter", "model size"),
    ("Qwen3-1.7B + LoRA (v2) seed 660 [final]", None, "model size / seed"),
    ("Qwen3-4B base", "qwen3-4b_base", "model size"),
    ("Qwen3-4B + LoRA (v2 recipe)", "qwen3-4b_adapter", "model size"),
    ("Qwen3-1.7B + LoRA (v2) seed 1", "qwen3-1.7b-v2-seed1_adapter", "seed"),
    ("Qwen3-1.7B + LoRA (v2) seed 2", "qwen3-1.7b-v2-seed2_adapter", "seed"),
    ("Final model as Q4_K_M GGUF in Ollama (all 128)", "gguf_q4km_ollama", "deployment"),
    ("Qwen3-1.7B base + system prompt (REFERENCE ONLY, invalid per rules)", "qwen3-1.7b_base_systemprompt_REFERENCE", "prompting"),
]
extra = []
for name, f, group in extra_specs:
    p = ROOT / "eval/test/final.summary.json" if f is None else ROOT / f"eval/extra/{f}.summary.json"
    if p.exists():
        r = {"experiment": group, "run": name, **load(p)}
        meta = None
        if f and f.endswith("_adapter"):
            mp = ROOT / "runs/extra" / f.replace("_adapter", "") / "train_meta.json"
            meta = json.load(open(mp)) if mp.exists() else None
        if f is None:
            meta = json.load(open(ROOT / "runs/revised/train_meta.json"))
        r["train_time_s"] = meta["time_load_to_adapter_save_s"] if meta else ""
        r["peak_alloc_GB"] = meta["peak_allocated_gpu_mem_GB"] if meta else ""
        extra.append(r)
if extra:
    write_csv(extra, OUT / "extra_experiments.csv")

# figure 1: main results vs targets
fig, ax = plt.subplots(figsize=(8, 3.6))
w = 0.26
for i, (r, c) in enumerate(zip(main, ["#9aa5b1", "#5b8def", "#1d4ed8"])):
    xs = [j + (i - 1) * w for j in range(len(METRICS))]
    ax.bar(xs, [r[k] / tot for k, _, _, tot in METRICS], w, label=r["run"], color=c)
for j, (_, _, tgt, tot) in enumerate(METRICS):
    ax.plot([j - 1.5 * w, j + 1.5 * w], [tgt / tot] * 2, color="#dc2626", lw=1.6)
ax.set_xticks(range(len(METRICS)), [m[1] for m in METRICS])
ax.set_ylabel("fraction correct"); ax.set_ylim(0, 1.05)
ax.set_title("Evaluation set (128 prompts); red line = target")
ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "figures/main_results.png", dpi=200)

# figure 2: extra experiments, "both correct"
if extra:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = [e["run"].replace(" (REFERENCE ONLY, invalid per rules)", "*") for e in extra]
    vals = [e["both_correct"] for e in extra]
    ax.barh(range(len(extra)), vals, color=["#1d4ed8" if "final" in n else "#5b8def" for n in names])
    ax.axvline(80, color="#dc2626", lw=1.4); ax.text(80.5, len(extra) - 0.6, "target 80", color="#dc2626", fontsize=8)
    ax.set_yticks(range(len(extra)), names, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("placement and answer both correct (/128)"); ax.set_xlim(0, 128)
    for i, v in enumerate(vals):
        ax.text(v + 1, i, str(v), va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "figures/extra_experiments.png", dpi=200)
print(open(OUT / "main_results.csv").read())
if extra:
    print(open(OUT / "extra_experiments.csv").read())
