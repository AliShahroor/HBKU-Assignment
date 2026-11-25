"""Generate notebooks/DSEG660_LoRA_pipeline.ipynb, a runnable walk-through of the whole pipeline."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# DSEG660: LoRA fine-tuning Qwen3-1.7B for a course greeting convention

This notebook runs the full pipeline: data, training, development selection, evaluation, GGUF export and Ollama.
Heavy steps are behind flags. With the defaults it only loads and displays the saved results in this folder.
Set the flags to `True` to recompute everything on one 16 GB GPU.

**Rule:** print `I like DSEG660` once, first by default; last (after the answer) only when the user explicitly opts out.
""")
code("""
import json, os, subprocess, sys
from pathlib import Path
import pandas as pd
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
PY = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv").exists() else sys.executable
RUN_DATA, RUN_TRAINING, RUN_EVAL, RUN_EXPORT = False, False, False, False
def sh(cmd):
    print("$", cmd); subprocess.run(cmd, shell=True, check=True)
""")
md("## 1. Data: matched variants of each base question (all variants of a question in one split)")
code("""
if RUN_DATA:
    sh("python3 scripts/01_make_data.py --version v1 --seed 660")
    sh("python3 scripts/01_make_data.py --version v2 --seed 660")
tr = pd.read_json("data/v2/train.jsonl", lines=True)
print(len(tr), "training examples;", tr.base_id.nunique(), "base questions")
tr[tr.question == "What is the capital of Italy?"][["variant", "prompt", "completion"]]
""")
md("## 2. Training: initial (v1) and revised/final (v2), both from the pinned base checkpoint")
code("""
if RUN_TRAINING:
    sh(f"{PY} scripts/02_train.py --data data/v1 --out runs/initial --steps 120 --lr 2e-4")
    sh(f"{PY} scripts/02_train.py --data data/v2 --out runs/revised --steps 160 --lr 2e-4")
keys = ["base_model", "revision", "n_train_examples", "steps", "learning_rate", "epochs_equiv", "gpu", "precision",
        "train_loss", "time_load_to_adapter_save_s", "peak_allocated_gpu_mem_GB"]
pd.DataFrame({r: {k: json.load(open(f"runs/{r}/train_meta.json"))[k] for k in keys} for r in ["initial", "revised"]})
""")
md("## 3. Development selection (template-matched dev and held-out-phrasing dev)")
code("""
rows = {}
for name, f in [("initial / dev", "initial"), ("initial / dev_heldout", "initial_heldout"),
                ("final / dev", "revised_dev"), ("final / dev_heldout", "revised_dev_heldout")]:
    s = json.load(open(f"eval/dev/{f}.summary.json"))
    rows[name] = {k: s[k] for k in ["greeting_first_correct", "keep_first_correct", "optout_last_correct",
                                    "main_answer_correct", "both_correct", "false_optouts"]}
pd.DataFrame(rows).T
""")
md("## 4. Evaluation: 128 prompts; prompt only, chat template, non-thinking, greedy, 64 new tokens")
code("""
if RUN_EVAL:
    sh("bash scripts/07_run_eval.sh")
res = {m: json.load(open(f"eval/test/{m}.summary.json")) for m in ["base", "initial", "final"]}
cols = ["greeting_first_correct", "keep_first_correct", "optout_last_correct", "main_answer_correct", "both_correct", "false_optouts"]
targets = {"greeting_first_correct": ">=70/80", "keep_first_correct": ">=42/48", "optout_last_correct": ">=37/48",
           "main_answer_correct": ">=92/128", "both_correct": ">=80/128", "false_optouts": "-"}
df = pd.DataFrame({m: {c: res[m][c] for c in cols} for m in res}); df.insert(0, "target", pd.Series(targets)); df
""")
code("""
sc = pd.read_json("eval/test/final.scored.jsonl", lines=True)
print("placement by category (final):")
display(sc.groupby("category").placement_correct.sum().to_frame())
sc[~sc.placement_correct][["id", "category", "prompt", "raw_reply", "observed_position"]]
""")
md("## 5. Deployment: Q4_K_M GGUF in Ollama (8 deployment_check prompts)")
code("""
if RUN_EXPORT:
    sh("bash scripts/05_export_gguf.sh runs/final/adapter")
    sh(f"{PY} scripts/06_ollama_demo.py --gguf deploy/gguf/dseg660-qwen3-1.7b-Q4_K_M.gguf "
       "--eval data/dseg660-evaluation.jsonl --adapter-replies eval/test/final.jsonl")
pd.read_json("deploy/ollama_deployment_check.jsonl", lines=True)
""")
md("## 6. Extra experiments: model size, seeds, GGUF on all prompts, prompt-only reference")
code("""
p = Path("results/extra_experiments.csv")
pd.read_csv(p) if p.exists() else "run scripts/08_extra_experiments.sh, then scripts/09_summarise.py"
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "DSEG660_LoRA_pipeline.ipynb"
out.parent.mkdir(exist_ok=True)
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
