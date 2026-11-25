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

FAMILIES = [  # (label, file prefix, final?)
    ("Gemma-3-270M", "gemma-3-270m", False),
    ("Qwen3-0.6B", "qwen3-0.6b", False),
    ("Gemma-3-1B", "gemma-3-1b", False),
    ("Qwen2.5-1.5B", "qwen2.5-1.5b", False),
    ("Qwen3-1.7B (assignment)", None, True),
    ("SmolLM2-1.7B", "smollm2-1.7b", False),
    ("Phi-4-mini (3.8B)", "phi-4-mini", False),
    ("Qwen3-4B", "qwen3-4b", False),
]
OTHER = [
    ("Qwen3-1.7B + LoRA, seed 660 (final)", ROOT / "eval/test/final.summary.json", "seed", ROOT / "runs/revised/train_meta.json"),
    ("Qwen3-1.7B + LoRA, seed 1", ROOT / "eval/extra/qwen3-1.7b-v2-seed1_adapter.summary.json", "seed", ROOT / "runs/extra/qwen3-1.7b-v2-seed1/train_meta.json"),
    ("Qwen3-1.7B + LoRA, seed 2", ROOT / "eval/extra/qwen3-1.7b-v2-seed2_adapter.summary.json", "seed", ROOT / "runs/extra/qwen3-1.7b-v2-seed2/train_meta.json"),
    ("Final model, Q4_K_M GGUF in Ollama", ROOT / "eval/extra/gguf_q4km_ollama.summary.json", "deployment", None),
    ("Qwen3-1.7B base + rule as system prompt*", ROOT / "eval/extra/qwen3-1.7b_base_systemprompt_REFERENCE.summary.json", "prompting (reference only)", None),
]


def meta_of(path):
    if path and Path(path).exists():
        m = json.load(open(path))
        return m["time_load_to_adapter_save_s"], m["peak_allocated_gpu_mem_GB"], m.get("revision", "")
    return "", "", ""


fam_rows = []
for label, pre, final in FAMILIES:
    bp = ROOT / ("eval/test/base.summary.json" if final else f"eval/extra/{pre}_base.summary.json")
    ap = ROOT / ("eval/test/final.summary.json" if final else f"eval/extra/{pre}_adapter.summary.json")
    if not (bp.exists() and ap.exists()):
        continue
    t, mem, rev = meta_of(ROOT / "runs/revised/train_meta.json" if final else ROOT / f"runs/extra/{pre}/train_meta.json")
    for kind, p in (("base", bp), ("LoRA", ap)):
        fam_rows.append({"model": label, "variant": kind, **load(p),
                         "train_time_s": t if kind == "LoRA" else "", "peak_alloc_GB": mem if kind == "LoRA" else ""})
write_csv(fam_rows, OUT / "model_families.csv")

other_rows = []
for label, p, group, mp in OTHER:
    if Path(p).exists():
        t, mem, _ = meta_of(mp)
        other_rows.append({"experiment": group, "run": label, **load(p), "train_time_s": t, "peak_alloc_GB": mem})
write_csv(other_rows, OUT / "robustness_and_reference.csv")
extra = other_rows  # kept for the notebook

plt.rcParams.update({"font.size": 9})

# figure 1: main results vs targets (value labels on every bar, legend outside the axes)
fig, ax = plt.subplots(figsize=(8, 3.8))
w = 0.26
colors = ["#9aa5b1", "#5b8def", "#1d4ed8"]
for i, (r, c) in enumerate(zip(main, colors)):
    xs = [j + (i - 1) * w for j in range(len(METRICS))]
    vals = [r[k] for k, _, _, _ in METRICS]
    fr = [v / tot for v, (_, _, _, tot) in zip(vals, METRICS)]
    ax.bar(xs, fr, w, label=r["run"], color=c)
    for x, f_, v in zip(xs, fr, vals):
        if v == 0:
            ax.text(x, 0.01, "0", ha="center", va="bottom", fontsize=7, color="#374151")
        else:  # label inside the top of the bar so it never collides with the target line
            ax.text(x, f_ / 2, str(v), ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
for j, (_, _, tgt, tot) in enumerate(METRICS):
    ax.plot([j - 1.5 * w, j + 1.5 * w], [tgt / tot] * 2, color="#dc2626", lw=1.6,
            label="target" if j == 0 else None)
ax.set_xticks(range(len(METRICS)), [f"{m[1]}\n(/{m[3]})" for m in METRICS])
ax.set_ylabel("fraction correct"); ax.set_ylim(0, 1.12)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, -0.03, "Bar labels are counts. The base model never writes the phrase on its own line, so all its placement-based scores are 0 (not missing).",
         fontsize=7, color="#4b5563")
fig.tight_layout(); fig.savefig(OUT / "figures/main_results.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# figure 2: model families, base vs LoRA, "both correct"
if fam_rows:
    labels = [l for l, *_ in FAMILIES if any(r["model"] == l for r in fam_rows)]
    base_v = [next(r["both_correct"] for r in fam_rows if r["model"] == l and r["variant"] == "base") for l in labels]
    lora_v = [next(r["both_correct"] for r in fam_rows if r["model"] == l and r["variant"] == "LoRA") for l in labels]
    ans_b = [next(r["main_answer_correct"] for r in fam_rows if r["model"] == l and r["variant"] == "base") for l in labels]
    fig, ax = plt.subplots(figsize=(8, 0.48 * len(labels) + 1.2))
    y = range(len(labels)); h = 0.38
    ax.barh([i - h / 2 for i in y], base_v, h, color="#9aa5b1", label="base model")
    ax.barh([i + h / 2 for i in y], lora_v, h, color=["#1d4ed8" if "assignment" in l else "#5b8def" for l in labels],
            label="+ LoRA (same v2 recipe)")
    for i, (b, l, a) in enumerate(zip(base_v, lora_v, ans_b)):
        ax.text(b + 1, i - h / 2, f"{b}" + (f"  (answers {a}/128; phrase never placed correctly)" if b == 0 else ""),
                va="center", fontsize=7, color="#374151")
        ax.text(l + 1, i + h / 2, str(l), va="center", fontsize=8, fontweight="bold")
    ax.axvline(80, color="#dc2626", lw=1.3, label="target (80)")
    ax.set_yticks(list(y), labels); ax.invert_yaxis()
    ax.set_xlabel("placement and answer both correct (/128)"); ax.set_xlim(0, 135)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.07), fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "figures/model_families.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# figure 3: seeds, deployment, prompting reference
if other_rows:
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(other_rows) + 1.1))
    names = [r["run"] for r in other_rows]; vals = [r["both_correct"] for r in other_rows]
    ax.barh(range(len(vals)), vals, color=["#1d4ed8" if "final" in n else "#5b8def" for n in names], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 1, i, str(v), va="center", fontsize=8, fontweight="bold")
    ax.axvline(80, color="#dc2626", lw=1.3, label="target (80)")
    ax.set_yticks(range(len(vals)), names); ax.invert_yaxis(); ax.set_xlim(0, 135)
    ax.set_xlabel("placement and answer both correct (/128)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, -0.02, "* breaks the assignment rule (system prompt supplies the phrase): reference point only", fontsize=7, color="#4b5563")
    fig.tight_layout(); fig.savefig(OUT / "figures/robustness_and_reference.png", dpi=200, bbox_inches="tight"); plt.close(fig)

for f in ["main_results.csv", "model_families.csv", "robustness_and_reference.csv"]:
    print("==", f); print(open(OUT / f).read())
