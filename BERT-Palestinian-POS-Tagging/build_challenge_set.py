#!/usr/bin/env python3
"""Create a small, manually annotated Palestinian POS challenge set."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "Palestinian_Challenge" / "test.txt"

SENTENCES = [
    "بكرا/ADV مش/NEG_PART رح/FUT_PART أروح/IV عالجامعة/NOUN لأنه/SUB_CONJ عندي/ADV محاضرتين/NOUN_NUM ./PUNC",
    "بدي/IV أبعت/IV للدكتور/NOUN الملف/NOUN اللي/REL_PRON عدلته/PV مبارح/NOUN ./PUNC",
    "قلتلهم/PV يستنوني/IV عند/ADV باب/NOUN المختبر/NOUN بس/CONJ ما/NEG_PART سمعوني/PV ./PUNC",
    "صاحبي/NOUN ورجاني/PV كيف/ADV أشغل/IV الكود/NOUN على/PREP اللابتوب/NOUN تبعه/PRON ./PUNC",
    "إحنا/PRON لسا/ADV بنجرب/IV النموذج/NOUN ومش/NEG_PART عارفين/ADJ ليش/INTERROG_PART بغلط/IV ./PUNC",
    "المحاضرة/NOUN كانت/PV أونلاين/ADJ ففتحت/PV Teams/ENGLISHTEXT من/PREP تلفوني/NOUN ./PUNC",
    "لما/SUB_CONJ النت/NOUN_PROP فصل/PV حكيت/PV للمشرف/NOUN إني/SUB_CONJ رح/FUT_PART أتأخر/IV ./PUNC",
    "وين/INTERROG_PART حطيت/PV نسخة/NOUN المشروع/NOUN الجديدة/ADJ عشان/NOUN أنزلها/IV على/PREP GitHub/ENGLISHTEXT ؟/PUNC",
    "عالسريع/ADV راجعت/PV النتائج/NOUN وبعدين/ADV بعثتها/PV للفريق/NOUN ./PUNC",
    "ما/NEG_PART قدرت/PV أفتح/IV الموقع/NOUN لأنه/SUB_CONJ السيرفر/NOUN كان/PV واقع/ADJ طول/NOUN الوقت/NOUN ./PUNC",
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for offset, specification in enumerate(SENTENCES, start=1):
        sentence_id = 903000 + offset
        for token_id, item in enumerate(specification.split(), start=1):
            token, tag = item.rsplit("/", 1)
            lines.append(f"{sentence_id}\t{token_id}\t{token}\t{tag}\t_\t_\t_\t_\t0\t0")
        lines.append("")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(SENTENCES)} sentences to {OUTPUT}")


if __name__ == "__main__":
    main()
