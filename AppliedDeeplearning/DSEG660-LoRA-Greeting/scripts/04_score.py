"""Score raw replies: greeting placement (strict rules) and main-answer correctness.

Placement rules (from the assignment):
  * strip whitespace at the start and end of the reply
  * the exact phrase must occur exactly once, on its own line, at the required end
  * a main answer (some other non-empty line) must be present
  * missing, repeated and inline phrases fail
  * false opt-out = otherwise correctly formatted reply with the phrase LAST when FIRST was expected

Answer correctness is scored independently of placement, on the reply with the phrase removed.

Usage: python 04_score.py --replies eval/final.jsonl [--overrides eval/manual_overrides.json]
"""
import argparse, ast, json, re, unicodedata
from collections import Counter, defaultdict

PHRASE = "I like DSEG660"


def placement(reply):
    r = reply.strip()
    lines = [l.strip() for l in r.split("\n")]
    n = r.count(PHRASE)
    own_line = [i for i, l in enumerate(lines) if l == PHRASE]
    others = [l for l in lines if l and l != PHRASE]
    if n == 0:
        return "missing"
    if n > 1:
        return "repeated"
    if not own_line:
        return "inline"
    if not others:
        return "no_answer"
    i = own_line[0]
    if i == 0:
        return "first"
    if i == len(lines) - 1:
        return "last"
    return "middle"


def strip_phrase(reply):
    return "\n".join(l for l in reply.strip().split("\n") if l.strip() != PHRASE).strip()


def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("`", "")
    s = re.sub(r"[^a-z0-9.+\-*/%=\[\]{}()<>:,'\" _]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .")


def answer_ok(body, refs, row):
    """Heuristic auto-check: a reference (or accepted equivalent) appears in the answer body.
    JSON rows: parse the first JSON object and compare fields/types to the reference."""
    ans_type = (row.get("answer_type") or row.get("type") or "").lower()
    if "json" in ans_type or (refs and refs[0].strip().startswith("{")):
        m = re.search(r"\{.*\}", body, re.S)
        if not m:
            return False, "no JSON object"
        try:
            got = json.loads(m.group(0))
        except Exception:
            return False, "invalid JSON"
        for ref in refs:
            try:
                exp = json.loads(ref)
            except Exception:
                continue
            if isinstance(exp, dict) and isinstance(got, dict) and set(got) == set(exp) and all(
                    type(got[k]) is type(exp[k]) or (isinstance(got[k], (int, float)) and isinstance(exp[k], (int, float))
                                                     and not isinstance(got[k], bool))
                    for k in exp) and all(
                    (got[k] == exp[k]) or (isinstance(exp[k], str) and norm(str(got[k])) == norm(exp[k])) for k in exp):
                return True, "json fields/types match"
        return False, "json mismatch"
    nb = norm(body)
    # Documented equivalence rule: a reference of the form "<number> <unit>" (e.g. "27 minutes",
    # "108 pencils") also accepts the bare number, because the question already fixes the unit.
    extra = []
    for ref in refs:
        m = re.fullmatch(r"([0-9][0-9.]*) [A-Za-z]+", str(ref).strip())
        if m:
            extra.append(m.group(1))
    refs = refs + extra
    for ref in refs:
        nr = norm(str(ref))
        if not nr:
            continue
        if re.fullmatch(r"[0-9.\-]+", nr):
            if re.search(rf"(?<![0-9.]){re.escape(nr)}(?![0-9]|\.[0-9])", nb):
                return True, f"number {ref}"
        elif nr in nb:
            return True, f"contains '{ref}'"
    return False, "no reference match"


def refs_of(row):
    refs = []
    for k in ("expected_answer", "reference_answer", "answer"):
        if row.get(k) not in (None, ""):
            v = row[k]
            refs += v if isinstance(v, list) else [v]
    for k in ("accepted_equivalents", "accepted_answers", "equivalents", "aliases"):
        if row.get(k):
            v = row[k]
            refs += v if isinstance(v, list) else [v]
    return [str(r) for r in refs]


def expected_pos(row):
    for k in ("expected_placement", "expected_position"):
        if k in row:
            v = str(row[k]).lower()
            if v in ("prefix", "first"):
                return "first"
            if v in ("suffix", "last"):
                return "last"
            raise ValueError(f"unknown placement {v!r}")
    raise KeyError("no expected position field")


def category(row):
    for k in ("variant", "instruction_type", "category", "type"):
        if k in row:
            return row[k]
    return "all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies", required=True)
    ap.add_argument("--overrides", default=None, help="JSON {id: {answer_correct: bool, reason: str}}")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.replies)]
    ov = json.load(open(args.overrides)) if args.overrides else {}

    tot = Counter()
    by_cat = defaultdict(Counter)
    scored = []
    for k, r in enumerate(rows):
        rid = str(r.get("id", k))
        exp = expected_pos(r)
        pos = placement(r["raw_reply"])
        place_ok = pos == exp
        false_optout = exp == "first" and pos == "last"
        body = strip_phrase(r["raw_reply"])
        ok, why = answer_ok(body, refs_of(r), r)
        manual = None
        if rid in ov:
            ok, why, manual = ov[rid]["answer_correct"], "MANUAL: " + ov[rid]["reason"], True
        cat = category(r)
        is_keep = "keep" in str(cat).lower() or bool(r.get("is_keep_request"))
        tot["n"] += 1
        tot[f"exp_{exp}"] += 1
        tot[f"place_ok_{exp}"] += place_ok
        if exp == "first" and is_keep:
            tot["keep_n"] += 1; tot["keep_ok"] += place_ok
        tot["answer_ok"] += ok
        tot["both_ok"] += ok and place_ok
        tot["false_optout"] += false_optout
        tot[f"pos_{pos}"] += 1
        by_cat[cat]["n"] += 1; by_cat[cat]["place_ok"] += place_ok; by_cat[cat]["answer_ok"] += ok
        scored.append({**r, "id": rid, "expected_position": exp, "observed_position": pos,
                       "placement_correct": place_ok, "false_optout": false_optout,
                       "answer_correct": ok, "answer_reason": why, "manual_answer_decision": bool(manual),
                       "both_correct": ok and place_ok})
    summary = {
        "greeting_first_correct": f"{tot['place_ok_first']}/{tot['exp_first']}",
        "keep_first_correct": f"{tot['keep_ok']}/{tot['keep_n']}",
        "optout_last_correct": f"{tot['place_ok_last']}/{tot['exp_last']}",
        "main_answer_correct": f"{tot['answer_ok']}/{tot['n']}",
        "both_correct": f"{tot['both_ok']}/{tot['n']}",
        "false_optouts": tot["false_optout"],
        "observed_positions": {k[4:]: v for k, v in tot.items() if k.startswith("pos_")},
        "by_category": {c: dict(v) for c, v in sorted(by_cat.items())},
    }
    print(json.dumps(summary, indent=2))
    out = args.out or args.replies.replace(".jsonl", ".scored.jsonl")
    with open(out, "w") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    json.dump(summary, open(out.replace(".scored.jsonl", ".summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
