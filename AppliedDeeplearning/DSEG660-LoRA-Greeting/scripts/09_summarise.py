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

# ---------------------------------------------------------------- figures
# Palette: validated reference instance (light surface). Text never wears series colour.
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BLUE_L, GRAY, ORANGE = "#2a78d6", "#9cc3ef", "#b9b8b3", "#eb6834"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / f"figures/{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"figures/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def header(fig, title, sub, height_in):
    """Title and subtitle pinned from the top in inches, so they never overlap; returns the top of the plot area."""
    t = 1 - 0.12 / height_in
    fig.text(0.01, t, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)
    n = sub.count("\n") + 1
    fig.text(0.01, t - 0.30 / height_in, sub, ha="left", va="top", fontsize=8.3, color=INK2, linespacing=1.35)
    return t - (0.30 + 0.17 * n + 0.12) / height_in


def subtitle(ax, text, y=1.02):
    ax.text(0, y, text, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")


# figure 1: one panel per metric, base -> initial -> final, target marked
fig, axes = plt.subplots(1, len(METRICS), figsize=(10, 3.6), sharey=False)
runs = [("Base", GRAY), ("Initial", BLUE_L), ("Final", BLUE)]
for ax, (k, lab, tgt, tot) in zip(axes, METRICS):
    vals = [r[k] for r in main]
    bars = ax.bar(range(3), vals, width=0.68, color=[c for _, c in runs], edgecolor=SURF, linewidth=2)
    ax.axhline(tgt, color=INK, lw=1.1, ls=(0, (4, 2)))
    ax.text(-0.5, tgt + tot * 0.012, f"target {tgt}", fontsize=7.5, color=INK, va="bottom", ha="left")
    for i, v in enumerate(vals):
        if v == 0:
            ax.text(i, tot * 0.015, "0", ha="center", va="bottom", fontsize=9, color=INK2)
        else:  # inside the bar, below any target line, so labels never collide with it
            ax.text(i, min(v, tgt) * 0.5, str(v), ha="center", va="center", fontsize=9.5,
                    color="white" if i == 2 else INK, fontweight="bold")
    ax.set_ylim(0, tot * 1.12); ax.set_xticks(range(3), [n for n, _ in runs], fontsize=8.5)
    ax.set_yticks([0, tot / 2, tot], [0, int(tot / 2), tot], fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f"{lab}  (/{tot})", fontsize=9.5, loc="center")
    ok = vals[2] >= tgt
    ax.text(0.5, -0.30, "\u2713 final meets target" if ok else "below target", transform=ax.transAxes,
            ha="center", fontsize=8, color=INK2)
top = header(fig, "Final adapter meets all five targets on the 128-prompt evaluation set",
             "Counts per metric for the base model, the initial adapter (v1 data) and the final adapter (v2 data). "
         "\nThe base model never writes the phrase on its own line, so its placement scores are measured zeros.", 3.6)
fig.tight_layout(rect=(0, 0, 1, top)); save(fig, "main_results")

# figure 2: model families. Left: answer accuracy base -> LoRA. Right: both correct with LoRA vs target.
if fam_rows:
    labels = [l for l, *_ in FAMILIES if any(r["model"] == l for r in fam_rows)]
    get = lambda l, v, k: next(r[k] for r in fam_rows if r["model"] == l and r["variant"] == v)
    ab = [get(l, "base", "main_answer_correct") for l in labels]
    al = [get(l, "LoRA", "main_answer_correct") for l in labels]
    both = [get(l, "LoRA", "both_correct") for l in labels]
    short = [l.replace(" (assignment)", "") for l in labels]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 0.42 * len(labels) + 1.9), sharey=True,
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    y = list(range(len(labels)))
    for i, (b, l) in enumerate(zip(ab, al)):
        if abs(l - b) >= 6:
            a1.annotate("", xy=(l, i), xytext=(b, i),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, shrinkA=5, shrinkB=5, mutation_scale=9))
        else:
            a1.plot([b, l], [i, i], color=INK2, lw=1.2, zorder=2)
        a1.scatter([b], [i], s=46, color=GRAY, zorder=3, edgecolor=SURF, linewidth=1.5)
        a1.scatter([l], [i], s=46, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.5)
        d = l - b
        if d > 0:
            a1.text(l + 3, i, f"{d:+d}", va="center", ha="left", fontsize=8.5, color=INK, fontweight="bold")
        else:
            a1.text(l - 3, i, f"{d:+d}", va="center", ha="right", fontsize=8.5, color=INK2)
    a1.axvline(92, color=INK, lw=1.1, ls=(0, (4, 2)))
    a1.text(93, -0.95, "answer target 92", fontsize=7.5, color=INK, ha="left", va="center")
    a1.set_xlim(15, 128); a1.set_yticks(y, short); a1.set_ylim(len(labels) - 0.5, -1.4)
    a1.set_xlabel("correct main answers (/128)"); a1.grid(axis="x", color=GRID, lw=0.6); a1.set_axisbelow(True)
    a1.set_title("Answer accuracy: base  →  + LoRA")
    a1.scatter([], [], s=46, color=GRAY, label="base model"); a1.scatter([], [], s=46, color=BLUE, label="+ LoRA")
    a1.legend(loc="lower left", fontsize=8, handletextpad=0.3)
    a2.barh(y, both, height=0.62, color=[BLUE if "assignment" in l else BLUE_L for l in labels],
            edgecolor=SURF, linewidth=2)
    for i, v in enumerate(both):
        a2.text(v + 1.5, i, str(v), va="center", fontsize=9, color=INK,
                fontweight="bold" if "assignment" in labels[i] else "normal")
    a2.axvline(80, color=INK, lw=1.1, ls=(0, (4, 2)))
    a2.text(81, -0.95, "target 80", fontsize=7.5, color=INK, ha="left", va="center")
    a2.set_xlim(0, 128); a2.set_xlabel("placement and answer both correct (/128)")
    a2.grid(axis="x", color=GRID, lw=0.6); a2.set_axisbelow(True)
    a2.set_title("With LoRA: both correct"); a2.tick_params(axis="y", length=0)
    H = 0.42 * len(labels) + 1.9
    top = header(fig, "Every model learns the placement; answer accuracy decides who meets the targets",
                 "Same v2 data, LoRA settings, 160 steps and seed for every model. All base models score 0 on placement "
             "(they never write the phrase on its own line).\nDark bar = Qwen3-1.7B, the assignment model.", H)
    fig.tight_layout(rect=(0, 0, 1, top)); save(fig, "model_families")

# figure 3: seeds, deployment, prompting reference
if other_rows:
    names = [r["run"].replace("*", "") for r in other_rows]
    vals = [r["both_correct"] for r in other_rows]
    groups = [r["experiment"] for r in other_rows]
    fig, ax = plt.subplots(figsize=(8.2, 0.5 * len(vals) + 1.6))
    for i, (n, v, g) in enumerate(zip(names, vals, groups)):
        ref = g.startswith("prompting")
        ax.barh(i, v, height=0.6, color=SURF if ref else (BLUE if "final" in n else BLUE_L),
                edgecolor=INK2 if ref else SURF, hatch="////" if ref else None, linewidth=1.2 if ref else 2)
        ax.text(v + 1.5, i, str(v), va="center", fontsize=9, color=INK, fontweight="bold" if "final" in n else "normal")
    ax.axvline(80, color=INK, lw=1.1, ls=(0, (4, 2)))
    ax.text(80, -0.75, "target 80", fontsize=7.5, color=INK, ha="center")
    ylabels = []
    for n, g in zip(names, groups):
        tag = {"seed": "seed", "deployment": "deploy", "prompting (reference only)": "reference*"}.get(g, g)
        ylabels.append(f"{n}   [{tag}]")
    ax.set_yticks(range(len(vals)), ylabels); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 128); ax.set_xlabel("placement and answer both correct (/128)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    top = header(fig, "Robust across seeds and 4-bit quantisation; prompting alone fails",
                 "Placement and answer both correct for the final recipe under other seeds, after export, and versus prompting.",
                 0.5 * len(vals) + 1.6)
    fig.text(0.01, -0.04, "* Reference only: a system prompt that supplies the phrase breaks the assignment rules (hatched bar).",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, top)); save(fig, "robustness_and_reference")

for f in ["main_results.csv", "model_families.csv", "robustness_and_reference.csv"]:
    print("==", f); print(open(OUT / f).read())
