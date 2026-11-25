"""Load the Q4_K_M GGUF in Ollama and run the eight deployment_check prompts.

Runtime settings match the adapter evaluation as closely as Ollama allows:
  * Qwen3 chat template in non-thinking mode (empty <think></think> block), no system prompt
  * temperature 0 (greedy), num_predict 64, fixed seed, stop at <|im_end|>
Raw replies are saved next to the final adapter's replies for the same prompts.

Usage: python 06_ollama_demo.py --gguf deploy/gguf/<file>.gguf --eval data/dseg660-evaluation.jsonl \
                                --adapter-replies eval/test/final.jsonl
"""
import argparse, json, os, subprocess, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLLAMA = ROOT / "deploy" / "ollama" / "bin" / "ollama"
HOST = "127.0.0.1:11435"
NAME = "dseg660-qwen3-1.7b-q4km"

TEMPLATE = (
    "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
OPTIONS = {"temperature": 0, "top_k": 1, "top_p": 1.0, "num_predict": 64, "seed": 660,
           "repeat_penalty": 1.0, "num_ctx": 2048}


def api(path, payload):
    req = urllib.request.Request(f"http://{HOST}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--adapter-replies", required=True)
    ap.add_argument("--out", default=str(ROOT / "deploy" / "ollama_deployment_check.jsonl"))
    ap.add_argument("--all", action="store_true",
                    help="extra experiment: run all prompts and write scorer-compatible rows (raw_reply = Ollama reply)")
    args = ap.parse_args()

    env = {**os.environ, "OLLAMA_HOST": HOST, "OLLAMA_MODELS": str(ROOT / "deploy" / "ollama_models")}
    server = subprocess.Popen([str(OLLAMA), "serve"], env=env, stdout=open(ROOT / "deploy" / "ollama_serve.log", "w"),
                              stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://{HOST}/api/version", timeout=2); break
            except Exception:
                time.sleep(1)
        version = json.loads(urllib.request.urlopen(f"http://{HOST}/api/version").read())["version"]

        mf = ROOT / "deploy" / "Modelfile"
        params = "\n".join(f"PARAMETER {k} {v}" for k, v in OPTIONS.items())
        mf.write_text(f"FROM {Path(args.gguf).resolve()}\nTEMPLATE \"\"\"{TEMPLATE}\"\"\"\n{params}\n"
                      f"PARAMETER stop <|im_end|>\nPARAMETER stop <|endoftext|>\n")
        with open(ROOT / "deploy" / "ollama_create.log", "w") as lg:
            subprocess.run([str(OLLAMA), "create", NAME, "-f", str(mf)], env=env, check=True,
                           stdout=lg, stderr=subprocess.STDOUT)

        rows = [json.loads(l) for l in open(args.eval)]
        if args.all:
            with open(args.out, "w") as f:
                for r in rows:
                    res = api("/api/generate", {"model": NAME, "prompt": r["prompt"], "stream": False,
                                                "think": False, "options": OPTIONS})
                    f.write(json.dumps({**r, "raw_reply": res["response"], "runtime": "ollama " + version,
                                        "gguf": Path(args.gguf).name}, ensure_ascii=False) + "\n")
            print(f"wrote {len(rows)} Ollama replies to {args.out}")
            return
        checks = [r for r in rows if r.get("deployment_check")]
        adapter = {}
        for l in open(args.adapter_replies):
            r = json.loads(l)
            adapter[r["prompt"]] = r["raw_reply"]
        out = []
        for r in checks:
            res = api("/api/generate", {"model": NAME, "prompt": r["prompt"], "stream": False,
                                        "think": False, "options": OPTIONS})
            out.append({"id": r.get("id"), "prompt": r["prompt"], "ollama_raw_reply": res["response"],
                        "adapter_raw_reply": adapter.get(r["prompt"]),
                        "identical": res["response"].strip() == (adapter.get(r["prompt"]) or "").strip()})
        meta = {"ollama_version": version, "model_name": NAME, "gguf": str(Path(args.gguf).name),
                "options": OPTIONS, "template": TEMPLATE, "system_prompt": None,
                "api": "/api/generate (Modelfile TEMPLATE applied), think=false"}
        with open(args.out, "w") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        json.dump(meta, open(Path(args.out).with_suffix(".meta.json"), "w"), indent=2)
        for o in out:
            print("=" * 80, "\nPROMPT:", o["prompt"], "\n--- ollama ---\n" + o["ollama_raw_reply"],
                  "\n--- adapter ---\n" + str(o["adapter_raw_reply"]), "\nidentical:", o["identical"])
        print(json.dumps(meta, indent=2))
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
