"""Convert the SFT data (v1 or v2) into MS-SWIFT DPO preference pairs.

chosen   = the correct reply (phrase first, or last on an explicit opt-out)
rejected = the same answer with the phrase at the WRONG end (the main error mode).
For 20% of examples the rejected reply is another formatting failure seen in evaluation
(phrase missing, inline, or repeated), so DPO also pushes against those.

Format (MS-SWIFT): {"messages": [{"role": "user", ...}, {"role": "assistant", <chosen>}], "rejected_response": <rejected>}
Usage: python 12_make_dpo_data.py --version v2
"""
import argparse, json, random
from pathlib import Path

PHRASE = "I like DSEG660"
ROOT = Path(__file__).resolve().parents[1]


def rejected_for(r, rng):
    a = r["answer"]
    flipped = f"{a}\n{PHRASE}" if r["expected_position"] == "first" else f"{PHRASE}\n{a}"
    u = rng.random()
    if u < 0.80:
        return flipped
    if u < 0.87:
        return a                                    # phrase missing
    if u < 0.94:
        return f"{PHRASE}! {a}"                     # phrase inline
    return f"{PHRASE}\n{a}\n{PHRASE}"               # phrase repeated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    ap.add_argument("--seed", type=int, default=660)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    src = ROOT / "data" / args.version / "train.jsonl"
    out = ROOT / "data" / f"dpo_{args.version}" / "train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for line in src.open():
            r = json.loads(line)
            rec = {"messages": [{"role": "user", "content": r["prompt"]},
                                {"role": "assistant", "content": r["completion"]}],
                   "rejected_response": rejected_for(r, rng)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
    print(f"wrote {n} preference pairs to {out}")


if __name__ == "__main__":
    main()
