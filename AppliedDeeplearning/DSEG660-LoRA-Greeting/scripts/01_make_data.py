"""Build train/dev sets for the DSEG660 greeting-placement LoRA exercise.

Every base question has a programmatically checked answer. Each base question
yields 8 paired variants that share the same question and answer and differ only
in what the user asks to be done with the course opener:

  ordinary           -> greeting first
  omit_details       -> greeting first   (content restriction only)
  keep_direct        -> greeting first
  keep_negated       -> greeting first   ("don't remove the greeting")
  keep_plus_restrict -> greeting first   (keep request + content restriction)
  optout_refer       -> greeting last    (refers to the opener)
  optout_quote       -> greeting last    (quotes the phrase)
  optout_entirely    -> greeting last    (asks to omit the phrase entirely)

All variants of a base question go to the same split.

Usage: python 01_make_data.py --version v1|v2 --seed 660
"""
import argparse, json, random, re
from pathlib import Path

PHRASE = "I like DSEG660"
ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- base questions
CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid",
    "Japan": "Tokyo", "Canada": "Ottawa", "Australia": "Canberra", "Egypt": "Cairo",
    "Kenya": "Nairobi", "Brazil": "Brasília", "Argentina": "Buenos Aires",
    "Mexico": "Mexico City", "India": "New Delhi", "China": "Beijing",
    "Russia": "Moscow", "Turkey": "Ankara", "Greece": "Athens", "Norway": "Oslo",
    "Sweden": "Stockholm", "Finland": "Helsinki", "Denmark": "Copenhagen",
    "Poland": "Warsaw", "Austria": "Vienna", "Ireland": "Dublin",
    "Netherlands": "Amsterdam", "Belgium": "Brussels", "Switzerland": "Bern",
    "Hungary": "Budapest", "Czech Republic": "Prague", "Peru": "Lima",
    "Chile": "Santiago", "Colombia": "Bogotá", "Thailand": "Bangkok",
    "Vietnam": "Hanoi", "South Korea": "Seoul", "Indonesia": "Jakarta",
    "Saudi Arabia": "Riyadh", "Iran": "Tehran", "Morocco": "Rabat",
    "Nigeria": "Abuja", "Ghana": "Accra", "Ethiopia": "Addis Ababa",
    "New Zealand": "Wellington", "Cuba": "Havana", "Iceland": "Reykjavik",
    "Jordan": "Amman", "Lebanon": "Beirut", "Pakistan": "Islamabad",
    "Philippines": "Manila", "Malaysia": "Kuala Lumpur", "Qatar": "Doha",
    "Kuwait": "Kuwait City", "Oman": "Muscat", "Tunisia": "Tunis",
    "Romania": "Bucharest", "Ukraine": "Kyiv", "Croatia": "Zagreb",
}

SYMBOLS = {
    "oxygen": "O", "hydrogen": "H", "carbon": "C", "nitrogen": "N", "sodium": "Na",
    "potassium": "K", "iron": "Fe", "gold": "Au", "silver": "Ag", "copper": "Cu",
    "helium": "He", "calcium": "Ca", "zinc": "Zn", "lead": "Pb", "tin": "Sn",
    "mercury": "Hg", "chlorine": "Cl", "magnesium": "Mg", "neon": "Ne", "sulfur": "S",
}

FACTS = [
    ("What is the largest planet in the Solar System?", "Jupiter"),
    ("Which planet is closest to the Sun?", "Mercury"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the boiling point of water at sea level in degrees Celsius?", "100 °C"),
    ("What is the freezing point of water in degrees Fahrenheit?", "32 °F"),
    ("How many continents are there?", "7"),
    ("How many sides does a hexagon have?", "6"),
    ("How many sides does an octagon have?", "8"),
    ("What gas do plants absorb from the air for photosynthesis?", "Carbon dioxide"),
    ("What is the largest ocean on Earth?", "The Pacific Ocean"),
    ("Who wrote 'Romeo and Juliet'?", "William Shakespeare"),
    ("What is the hardest natural substance?", "Diamond"),
    ("How many days are in a leap year?", "366"),
    ("What is the square root of 81?", "9"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the longest river in Africa?", "The Nile"),
    ("How many bits are in a byte?", "8"),
    ("What is the smallest prime number?", "2"),
    ("Which organ pumps blood through the human body?", "The heart"),
    ("What is the tallest mountain on Earth above sea level?", "Mount Everest"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What language is primarily spoken in Brazil?", "Portuguese"),
    ("How many minutes are in a day?", "1440"),
    ("What is the main gas in Earth's atmosphere?", "Nitrogen"),
    ("What is the currency of Japan?", "The yen"),
    ("How many players are on the field for one soccer team?", "11"),
    ("What is the binary representation of the decimal number 5?", "101"),
    ("What is the largest mammal?", "The blue whale"),
    ("Which programming language uses the keyword 'def' to define functions?", "Python"),
    ("What does CPU stand for?", "Central Processing Unit"),
    ("What does GPU stand for?", "Graphics Processing Unit"),
    ("What is the value of pi rounded to two decimal places?", "3.14"),
    ("How many hours are in a week?", "168"),
    ("What is the opposite of 'ancient'?", "Modern"),
    ("What is the plural of 'mouse'?", "Mice"),
    ("What is the past tense of 'run'?", "Ran"),
    ("What is the opposite of 'generous'?", "Stingy"),
    ("What is the plural of 'child'?", "Children"),
    ("What is the past tense of 'write'?", "Wrote"),
    ("What color do you get by mixing blue and yellow?", "Green"),
]

PY_EXPR = [
    ("the length of the list items", "len(items)"),
    ("the sum of the list nums", "sum(nums)"),
    ("the string s reversed", "s[::-1]"),
    ("the largest value in the list nums", "max(nums)"),
    ("the smallest value in the list nums", "min(nums)"),
    ("the string s converted to uppercase", "s.upper()"),
    ("the string s converted to lowercase", "s.lower()"),
    ("a list of the squares of 1 through n", "[i**2 for i in range(1, n + 1)]"),
    ("the list nums sorted in ascending order", "sorted(nums)"),
    ("whether the integer x is even", "x % 2 == 0"),
    ("the number of words in the string s", "len(s.split())"),
    ("the average of the list nums", "sum(nums) / len(nums)"),
    ("the unique elements of the list items as a set", "set(items)"),
    ("the keys of the dictionary d as a list", "list(d.keys())"),
    ("the last element of the list items", "items[-1]"),
    ("the first three characters of the string s", "s[:3]"),
    ("the absolute value of x", "abs(x)"),
    ("the string s with leading and trailing spaces removed", "s.strip()"),
    ("the even numbers from the list nums", "[x for x in nums if x % 2 == 0]"),
    ("the number of times 'a' appears in the string s", "s.count('a')"),
]

NAMES = ["Sara", "Omar", "Lina", "Karim", "Maya", "Yusuf", "Noor", "Adam", "Hana", "Ziad",
         "Laila", "Sami", "Rana", "Tariq", "Dina", "Fadi"]
CITIES = [("Doha", "Qatar"), ("Lyon", "France"), ("Osaka", "Japan"), ("Porto", "Portugal"),
          ("Seville", "Spain"), ("Munich", "Germany"), ("Milan", "Italy"), ("Toronto", "Canada")]
WORDS = ["python", "garden", "planet", "silver", "window", "orange", "rocket", "puzzle",
         "bridge", "candle", "forest", "marble", "laptop", "spring", "castle", "thunder"]


def base_questions(rng):
    qs = []
    for c, cap in CAPITALS.items():
        qs.append(("capital", f"What is the capital of {c}?", cap))
    for el, sym in SYMBOLS.items():
        qs.append(("symbol", f"What is the chemical symbol for {el}?", sym))
    for q, a in FACTS:
        qs.append(("fact", q, a))
    for desc, expr in PY_EXPR:
        qs.append(("python", f"Give a Python expression for {desc}.", expr))
    for _ in range(24):
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        op = rng.choice(["+", "-", "*"])
        if op == "*":
            b = rng.randint(2, 9)
        val = eval(f"{a}{op}{b}")
        sym = {"+": "plus", "-": "minus", "*": "times"}[op]
        qs.append(("arith", f"What is {a} {sym} {b}?", str(val)))
    for _ in range(10):
        h = rng.randint(2, 12)
        qs.append(("convert", f"How many minutes are in {h} hours?", str(60 * h)))
    for _ in range(8):
        m = rng.randint(2, 30)
        qs.append(("convert", f"How many centimeters are in {m} meters?", str(100 * m)))
    for w in WORDS:
        kind = rng.choice(["rev", "len", "upper"])
        if kind == "rev":
            qs.append(("word", f"Spell the word '{w}' backwards.", w[::-1]))
        elif kind == "len":
            qs.append(("word", f"How many letters are in the word '{w}'?", str(len(w))))
        else:
            qs.append(("word", f"Write the word '{w}' in uppercase.", w.upper()))
    for i in range(16):
        n = NAMES[i]
        kind = i % 3
        if kind == 0:
            age = rng.randint(18, 70)
            qs.append(("json", f'Return a JSON object with keys "name" and "age" for {n}, who is {age} years old.',
                       json.dumps({"name": n, "age": age})))
        elif kind == 1:
            city, country = CITIES[i % len(CITIES)]
            qs.append(("json", f'Return a JSON object with keys "city" and "country" for {city}, {country}.',
                       json.dumps({"city": city, "country": country})))
        else:
            price = round(rng.uniform(2, 90), 2)
            item = WORDS[i]
            qs.append(("json", f'Return a JSON object with keys "item" (string) and "price" (number) for a {item} that costs {price} dollars.',
                       json.dumps({"item": item, "price": price})))
    # dedupe on question text
    seen, out = set(), []
    for t in qs:
        if t[1] not in seen:
            seen.add(t[1]); out.append(t)
    return out


# ---------------------------------------------------------------- instruction phrasings
OPENER_NAMES = ["the course greeting", "the usual course opener", "the course opener",
                "the greeting line", "the DSEG660 greeting", "the class greeting",
                "the usual greeting", "the opening line"]
RESTRICT = ["skip the explanation", "no extra details, please", "leave out the background",
            "omit the details", "don't add any commentary", "skip the extra context",
            "keep it to just the answer", "omit any examples", "remove the extra details",
            "skip the travel advice", "no working shown", "drop the elaboration"]
KEEP_DIRECT = ["Keep {o}.", "Please keep {o}.", "Include {o} as usual.", "Start with {o} like always.",
               "Use {o} at the top.", "Keep {o} at the start.", "Retain {o}.",
               "Open with {o} as normal."]
KEEP_NEG = ["Don't remove {o}.", "Do not skip {o}.", "Don't drop {o}.", "Please don't leave out {o}.",
            "Do not omit {o}.", "Don't get rid of {o}.", "Never skip {o}.",
            "Don't remove {o} from the start."]
OPTOUT_REFER = ["Skip {o}.", "Remove {o}.", "Drop {o}.", "Leave out {o}.",
                "Please skip {o} as the opener.", "Don't start with {o}.", "Don't open with {o}.",
                "No {o2} at the top, please.", "Remove {o} from the beginning.",
                "Skip {o} this time."]
OPTOUT_QUOTE = ['Skip "I like DSEG660".', 'Don\'t open with "I like DSEG660".',
                'Remove the line "I like DSEG660".', 'Leave out "I like DSEG660" at the start.',
                'Please drop "I like DSEG660" as the opener.', 'No "I like DSEG660" at the top.',
                "Skip the 'I like DSEG660' line.", 'Omit the "I like DSEG660" opener.']
OPTOUT_ENTIRE = ["Omit {o} entirely.", "Leave out {o} completely.", "Don't include {o} at all.",
                 'Omit the phrase "I like DSEG660" entirely.', "Remove {o} altogether.",
                 "Please don't use {o} at all.", 'Do not include "I like DSEG660" anywhere.',
                 "Drop {o} completely."]


# ---- held-out phrasings: used ONLY for the dev_heldout set, never for training.
# Added after the template-matched dev set saturated (325/336) and gave no error signal.
HELDOUT = dict(
    OPENER_NAMES=["the DSEG660 line", "the course salutation", "the standard opener",
                  "your usual opening phrase", "the greeting at the top", "the class opener"],
    RESTRICT=["just the answer, no explanation", "skip the history lesson", "don't include any caveats",
              "no need for extra context", "omit the reasoning", "leave out any examples",
              "skip the fun facts", "don't elaborate"],
    KEEP_DIRECT=["Make sure to begin with {o}.", "Go ahead and include {o} first.", "I'd like {o} as usual.",
                 "Lead with {o}.", "Please have {o} at the top.", "Begin the reply with {o}."],
    KEEP_NEG=["Don't forget {o}.", "Please do not take out {o}.", "There's no need to remove {o}.",
              "Don't cut {o}.", "Avoid dropping {o}.", "Do not leave {o} off."],
    OPTOUT_REFER=["I don't need {o} this time.", "Without {o}, please.", "Can you leave {o} off the top?",
                  "Don't lead with {o}.", "Cut {o} from the start.", "Please don't begin with {o}.",
                  "Forget {o} at the start."],
    OPTOUT_QUOTE=["Please don't start your reply with 'I like DSEG660'.", 'Without the "I like DSEG660" opener, please.',
                  'I don\'t want "I like DSEG660" at the beginning.', "Cut the line I like DSEG660.",
                  'Don\'t lead with "I like DSEG660".'],
    OPTOUT_ENTIRE=["I don't want {o} anywhere in the reply.", "Exclude {o} completely.",
                   "Leave {o} out of the reply entirely.", 'No "I like DSEG660" at all, please.'],
)
TRAIN_POOLS = dict(OPENER_NAMES=None, RESTRICT=None, KEEP_DIRECT=None, KEEP_NEG=None,
                   OPTOUT_REFER=None, OPTOUT_QUOTE=None, OPTOUT_ENTIRE=None)
POOL = {}  # active pool, set per split


def use_pools(heldout):
    g = globals()
    for k in TRAIN_POOLS:
        POOL[k] = HELDOUT[k] if heldout else g[k]


def o_name(rng):
    return rng.choice(POOL["OPENER_NAMES"])


def fill(t, rng):
    o = o_name(rng)
    o2 = o.replace("the ", "", 1)
    return t.format(o=o, o2=o2)


def join(question, parts, rng):
    """Combine question and instruction sentences in random order."""
    parts = [p for p in parts if p]
    if not parts:
        return question
    sents = [question] + parts
    head = sents[0]
    rest = sents[1:]
    rng.shuffle(rest)
    if rng.random() < 0.4:  # instructions first
        order = rest + [head]
    else:
        order = [head] + rest
    # sometimes merge into one sentence with "and"
    if len(parts) == 2 and rng.random() < 0.35:
        a, b = parts
        merged = a.rstrip(".") + " and " + b[0].lower() + b[1:]
        merged = merged if merged.endswith((".", "?", "!")) else merged + "."
        order = [question, merged] if rng.random() < 0.6 else [merged, question]
    return " ".join(order)


def restrict_sentence(rng):
    r = rng.choice(POOL["RESTRICT"])
    r = r[0].upper() + r[1:]
    return r if r.endswith((".", "!")) else r + "."


def variants(q, a, rng, version):
    out = []
    first = f"{PHRASE}\n{a}"
    last = f"{a}\n{PHRASE}"
    out.append(("ordinary", join(q, [], rng), first))
    out.append(("omit_details", join(q, [restrict_sentence(rng)], rng), first))
    out.append(("keep_direct", join(q, [fill(rng.choice(POOL["KEEP_DIRECT"]), rng),
                                        restrict_sentence(rng) if rng.random() < 0.3 else None], rng), first))
    out.append(("keep_negated", join(q, [fill(rng.choice(POOL["KEEP_NEG"]), rng),
                                         restrict_sentence(rng) if rng.random() < 0.3 else None], rng), first))
    keep_t = rng.choice(POOL["KEEP_DIRECT"] + POOL["KEEP_NEG"])
    out.append(("keep_plus_restrict", join(q, [fill(keep_t, rng), restrict_sentence(rng)], rng), first))
    out.append(("optout_refer", join(q, [fill(rng.choice(POOL["OPTOUT_REFER"]), rng),
                                         restrict_sentence(rng) if rng.random() < 0.5 else None], rng), last))
    out.append(("optout_quote", join(q, [rng.choice(POOL["OPTOUT_QUOTE"]),
                                         restrict_sentence(rng) if rng.random() < 0.5 else None], rng), last))
    out.append(("optout_entirely", join(q, [fill(rng.choice(POOL["OPTOUT_ENTIRE"]), rng),
                                            restrict_sentence(rng) if rng.random() < 0.5 else None], rng), last))
    if version == "v2":
        out += v2_extra(q, a, rng)
    return out


# ---- v2 targeted revision (after dev_heldout analysis of the initial adapter).
# Error: all 7 dev errors were false opt-outs on NEGATED keep requests whose removal verb
# had never been seen under negation ("Avoid dropping", "Don't cut", "do not take out" ...).
# The model keyed on the removal verb and ignored the negation.
# Fix: contrastive pairs that use the SAME removal verb negated (-> first) and plain (-> last).
# Verbs/negators are disjoint from the held-out dev pools, so dev_heldout stays a fair test.
V2_VERBS = [("delete", "deleting"), ("strip", "stripping"), ("erase", "erasing"), ("ditch", "ditching"),
            ("discard", "discarding"), ("eliminate", "eliminating"), ("trim", "trimming"),
            ("lose", "losing"), ("scrap", "scrapping"), ("exclude", "excluding")]
V2_NEG = ["Be careful not to {v} {o}.", "Make sure you don't {v} {o}.", "You shouldn't {v} {o}.",
          "Please refrain from {g} {o}.", "I don't want you to {v} {o}.", "Don't {v} {o}, though.",
          "It's important not to {v} {o}.", "No {g} {o}, please."]
V2_POS = ["Please {v} {o}.", "{V} {o}.", "I want you to {v} {o}.", "Kindly {v} {o}.", "{V} {o} for this one."]
V2_CONTENT_DISTRACT = ["{V} the extra details.", "{V} any explanation.", "{V} the background info."]


def v2_extra(q, a, rng):
    first = f"{PHRASE}\n{a}"
    last = f"{a}\n{PHRASE}"
    out = []
    v, g = rng.choice(V2_VERBS)
    o = o_name(rng)
    neg = rng.choice(V2_NEG).format(v=v, g=g, o=o)
    pos = rng.choice(V2_POS).format(v=v, V=v.capitalize(), o=o)
    # matched pair: same verb, same opener name, negated vs plain
    out.append(("v2_neg_keep", join(q, [neg], rng), first))
    out.append(("v2_plain_optout", join(q, [pos], rng), last))
    # negated keep + a content restriction that itself uses a removal verb (distractor)
    v2, g2 = rng.choice(V2_VERBS)
    neg2 = rng.choice(V2_NEG).format(v=v2, g=g2, o=o_name(rng))
    distract = rng.choice(V2_CONTENT_DISTRACT).format(V=rng.choice(V2_VERBS)[0].capitalize())
    out.append(("v2_neg_keep_distractor", join(q, [neg2, distract], rng), first))
    return out


def norm_q(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=660)
    ap.add_argument("--eval", default=str(ROOT / "data" / "dseg660-evaluation.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    qs = base_questions(random.Random(args.seed))

    # exclude anything overlapping the evaluation questions
    excluded = []
    ev = Path(args.eval)
    if ev.exists():
        eval_text = " ".join(norm_q(json.loads(l).get("prompt", "")) for l in ev.open())
        keep = []
        for t in qs:
            # crude overlap check: the core question words appear in an eval prompt
            core = norm_q(t[1])
            if core and core in eval_text:
                excluded.append(t[1])
            else:
                keep.append(t)
        qs = keep
    else:
        print("WARNING: evaluation file not found; overlap check skipped")

    use_pools(False)
    split_rng = random.Random(args.seed + 1)
    idx = list(range(len(qs)))
    split_rng.shuffle(idx)
    n_dev = int(round(0.2 * len(qs)))
    dev_ids = set(idx[:n_dev])

    out_dir = ROOT / "data" / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    with (out_dir / "train.jsonl").open("w") as ft, (out_dir / "dev.jsonl").open("w") as fd:
        for i, (cat, q, a) in enumerate(qs):
            split = "dev" if i in dev_ids else "train"
            vrng = random.Random(f"{args.seed}-{i}-{split}")
            for vtype, prompt, completion in variants(q, a, vrng, args.version if split == "train" else "v1"):
                rec = {"base_id": f"b{i:03d}", "category": cat, "variant": vtype,
                       "expected_position": "first" if completion.startswith(PHRASE) else "last",
                       "question": q, "answer": a, "prompt": prompt, "completion": completion}
                (fd if split == "dev" else ft).write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[(split, vtype)] = counts.get((split, vtype), 0) + 1
    # dev_heldout: same dev base questions, phrasings from the held-out pools only
    use_pools(True)
    with (out_dir / "dev_heldout.jsonl").open("w") as fh:
        for i, (cat, q, a) in enumerate(qs):
            if i not in dev_ids:
                continue
            vrng = random.Random(f"{args.seed}-{i}-heldout")
            for vtype, prompt, completion in variants(q, a, vrng, "v1"):
                rec = {"base_id": f"b{i:03d}", "category": cat, "variant": vtype,
                       "expected_position": "first" if completion.startswith(PHRASE) else "last",
                       "question": q, "answer": a, "prompt": prompt, "completion": completion}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[("dev_heldout", vtype)] = counts.get(("dev_heldout", vtype), 0) + 1
    use_pools(False)
    print(f"base questions: {len(qs)} (dev {n_dev}); excluded for eval overlap: {len(excluded)}")
    for e in excluded:
        print("  excluded:", e)
    for k in sorted(counts):
        print(k, counts[k])


if __name__ == "__main__":
    main()
