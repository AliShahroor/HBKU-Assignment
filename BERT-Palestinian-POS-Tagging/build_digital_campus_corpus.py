#!/usr/bin/env python3
"""Create a reproducible Palestinian Digital Campus POS corpus extension."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CURRAS = ROOT / "data" / "Curras"
NEW = ROOT / "data" / "Palestinian_Digital_Campus"
EXTENDED = ROOT / "data" / "Curras_Extended"
SEED = 42


def sent(spec: str) -> list[tuple[str, str]]:
    """Parse token/POS pairs separated by spaces; token text must not contain spaces."""
    output = []
    for item in spec.split():
        token, tag = item.rsplit("/", 1)
        output.append((token, tag))
    return output


BASE = [
    sent("مبارح/ADV أنا/PRON برمجت/PV تطبيق/NOUN صغير/ADJ بالجامعة/NOUN ورفعته/PV على/PREP GitHub/ENGLISHTEXT ./PUNC"),
    sent("إحنا/PRON بدنا/IV نخلص/IV المشروع/NOUN قبل/ADV الموعد/NOUN عشان/NOUN الدكتور/NOUN ما/NEG_PART يزعل/IV ./PUNC"),
    sent("اليوم/NOUN الشبكة/NOUN كانت/PV بطيئة/ADJ فاضطريت/PV أشارك/IV الإنترنت/NOUN_PROP من/PREP تلفوني/NOUN ./PUNC"),
    sent("سارة/NOUN_PROP فتحت/PV Zoom/ENGLISHTEXT وحضرت/PV المحاضرة/NOUN من/PREP البيت/NOUN ./PUNC"),
    sent("ليش/INTERROG_PART الكود/NOUN مش/NEG_PART شغال/ADJ على/PREP جهازي/NOUN ؟/PUNC"),
    sent("هسا/ADV بدي/IV أجرب/IV الخوارزمية/NOUN على/PREP بيانات/NOUN جديدة/ADJ ./PUNC"),
    sent("رح/FUT_PART نبعت/IV التقرير/NOUN للدكتور/NOUN بعد/ADV ما/REL_PRON نراجع/IV النتائج/NOUN ./PUNC"),
    sent("الطالب/NOUN كتب/PV الدالة/NOUN والطالبة/NOUN فحصت/PV الاختبارات/NOUN ./PUNC"),
    sent("فريق/NOUN المشروع/NOUN اجتمع/PV بالمختبر/NOUN عشان/NOUN يحل/IV مشكلة/NOUN الذاكرة/NOUN ./PUNC"),
    sent("لسا/ADV السيرفر/NOUN واقع/ADJ ومش/NEG_PART قادرين/ADJ ندخل/IV عالموقع/NOUN ./PUNC"),
    sent("وين/INTERROG_PART حفظت/PV الملف/NOUN اللي/REL_PRON بعثته/PV مبارح/ADV ؟/PUNC"),
    sent("أنا/PRON نسيت/PV كلمة/NOUN السر/NOUN فطلبت/PV من/PREP النظام/NOUN يغيرها/IV ./PUNC"),
    sent("الدكتور/NOUN شرح/PV نموذج/NOUN اللغة/NOUN وبعدين/ADV ورجانا/PV كيف/ADV يتوقع/IV الكلمة/NOUN ./PUNC"),
    sent("بالمختبر/NOUN دربنا/PV المصنف/NOUN وقسنا/PV الدقة/NOUN والاستدعاء/NOUN ./PUNC"),
    sent("صاحبي/NOUN اقترح/PV نستخدم/IV Python/ENGLISHTEXT لأنه/SUB_CONJ أسهل/ADJ للمشروع/NOUN ./PUNC"),
    sent("الطالبة/NOUN سألت/PV شو/INTERROG_PART الفرق/NOUN بين/NOUN التدريب/NOUN والاختبار/NOUN ./PUNC"),
    sent("الباص/NOUN تأخر/PV فوصلت/PV عالمحاضرة/NOUN وأنا/PRON مستعجل/ADJ ./PUNC"),
    sent("بعد/ADV الدوام/NOUN قعدنا/PV بالمكتبة/NOUN نكمل/IV واجب/NOUN الذكاء/NOUN الاصطناعي/ADJ ./PUNC"),
    sent("المبرمج/NOUN اكتشف/PV خطأ/NOUN صغير/ADJ بس/CONJ أثر/PV على/PREP كل/NOUN_QUANT النتائج/NOUN ./PUNC"),
    sent("لما/SUB_CONJ خلص/CV التدريب/NOUN حفظنا/PV النموذج/NOUN وجربناه/PV على/PREP الاختبار/NOUN ./PUNC"),
    sent("النظام/NOUN صنف/PV الكلمات/NOUN بسرعة/NOUN بس/CONJ خربط/PV بين/NOUN الاسم/NOUN والصفة/NOUN ./PUNC"),
    sent("أنا/PRON كتبت/PV التوثيق/NOUN عشان/NOUN كل/NOUN_QUANT طالب/NOUN يقدر/IV يعيد/IV التجربة/NOUN ./PUNC"),
    sent("عملنا/PV نسخة/NOUN احتياطية/ADJ قبل/ADV ما/REL_PRON نحدث/IV قاعدة/NOUN البيانات/NOUN ./PUNC"),
    sent("النت/NOUN_PROP فصل/PV بنص/NOUN_QUANT الامتحان/NOUN فحكيت/PV مع/ADV الدعم/NOUN الفني/ADJ ./PUNC"),
    sent("بكرا/ADV رح/FUT_PART نعرض/IV المشروع/NOUN قدام/ADV الصف/NOUN ونشرح/IV التصميم/NOUN ./PUNC"),
    sent("مش/NEG_PART كل/NOUN_QUANT نتيجة/NOUN عالية/ADJ معناها/NOUN النموذج/NOUN ممتاز/ADJ ./PUNC"),
    sent("لازم/ADJ نفحص/IV الأخطاء/NOUN حسب/NOUN كل/NOUN_QUANT وسم/NOUN قبل/ADV التسليم/NOUN ./PUNC"),
    sent("البيانات/NOUN الجديدة/ADJ فيها/PREP كلمات/NOUN فلسطينية/ADJ من/PREP حكي/NOUN الطلاب/NOUN ./PUNC"),
    sent("عملنا/PV مصفوفة/NOUN ارتباك/NOUN عشان/NOUN نعرف/IV وين/INTERROG_PART النموذج/NOUN بغلط/IV ./PUNC"),
    sent("التجربة/NOUN قابلة/ADJ للإعادة/NOUN لأن/SUB_CONJ البذرة/NOUN والإعدادات/NOUN محفوظة/ADJ ./PUNC"),
]

# Exact 80-word corpus reused from the earlier n-gram assignment. These four
# sentences are added to training only; punctuation is an additional POS token.
PREVIOUS_ASSIGNMENT = [
    sent("مبارح/NOUN بعد/ADV الدوام/NOUN رحت/PV عالسوق/NOUN مع/ADV أختي/NOUN واشترينا/PV خضرة/NOUN للبيت/NOUN وبعدين/ADV قعدنا/PV عند/ADV جيرانا/NOUN نشرب/IV قهوة/NOUN ونحكي/IV عن/PREP الجامعة/NOUN والمشاريع/NOUN ./PUNC"),
    sent("اليوم/NOUN الصبح/NOUN صحيت/PV بكير/ADV عشان/NOUN عندي/PREP محاضرة/NOUN بس/CONJ الباص/NOUN تأخر/PV فمشيت/PV عالجامعة/NOUN وأنا/PRON مستعجل/ADJ وبحكي/IV لصاحبي/NOUN يستناني/IV عند/ADV الباب/NOUN هناك/ADV ./PUNC"),
    sent("بالليل/NOUN أمي/NOUN عملت/PV مقلوبة/NOUN وسلطة/NOUN وتعشينا/PV مع/ADV الجيران/NOUN سوا/ADV وبعد/ADV الأكل/NOUN شربنا/PV شاي/NOUN وقعدنا/PV نحكي/IV ونضحك/IV لحد/ADV ما/REL_PRON تأخر/PV الوقت/NOUN ./PUNC"),
    sent("يوم/NOUN الجمعة/NOUN_PROP طلعنا/PV عرامالله/NOUN_PROP سوا/ADV نزور/IV قرايبنا/NOUN وتمشينا/PV بالبلد/NOUN واشترينا/PV كنافة/NOUN ورجعنا/PV عالبيت/NOUN قبل/ADV المغرب/NOUN لأن/SUB_CONJ الشوارع/NOUN كانت/PV أزمة/NOUN كثير/ADV ./PUNC"),
]

REPLACEMENTS = {
    "سارة": ["سارة", "ليان", "نور", "ريم", "عمر", "أحمد"],
    "GitHub": ["GitHub", "GitLab"],
    "Zoom": ["Zoom", "Teams"],
    "Python": ["Python", "Java"],
    "المشروع": ["المشروع", "الواجب", "التطبيق"],
    "تطبيق": ["تطبيق", "موقع", "برنامج"],
    "صغير": ["صغير", "جديد", "مرتب"],
    "بيانات": ["بيانات", "نصوص", "صور"],
    "جديدة": ["جديدة", "كثيرة", "نظيفة"],
    "الدكتور": ["الدكتور", "الأستاذ", "المشرف"],
    "بالمختبر": ["بالمختبر", "بالجامعة", "بالمكتبة"],
    "عالموقع": ["عالموقع", "عالمنصة", "عالنظام"],
    "الملف": ["الملف", "التقرير", "الكود"],
    "النظام": ["النظام", "الموقع", "التطبيق"],
    "المصنف": ["المصنف", "النموذج", "النظام"],
    "الباص": ["الباص", "السرفيس"],
    "عالمحاضرة": ["عالمحاضرة", "عالمختبر"],
    "خطأ": ["خطأ", "خلل"],
    "النموذج": ["النموذج", "المصنف"],
    "الاختبار": ["الاختبار", "التقييم"],
}

ADJUNCTS = [
    [("بالجامعة", "NOUN")],
    [("بالمختبر", "NOUN")],
    [("بالمكتبة", "NOUN")],
    [("بالبيت", "NOUN")],
    [("عالكمبيوتر", "NOUN")],
    [("عالموبايل", "NOUN")],
    [("مع", "ADV"), ("الفريق", "NOUN")],
    [("مع", "ADV"), ("صاحبي", "NOUN")],
]


def variant(
    sentence: list[tuple[str, str]], rng: random.Random, cycle: int
) -> list[tuple[str, str]]:
    changed = [(rng.choice(REPLACEMENTS.get(token, [token])), tag) for token, tag in sentence]
    # A distinct, natural computing-location adjunct prevents split leakage while
    # preserving the same POS annotation scheme.
    return changed[:-1] + ADJUNCTS[cycle] + changed[-1:]


def write_conll(path: Path, sentences: list[list[tuple[str, str]]], start_id: int) -> None:
    lines: list[str] = []
    for offset, sentence in enumerate(sentences):
        sentence_id = start_id + offset
        for token_id, (token, tag) in enumerate(sentence, start=1):
            lines.append(
                f"{sentence_id}\t{token_id}\t{token}\t{tag}\t_\t_\t_\t_\t0\t0"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_files(destination: Path, first: Path, second: Path) -> None:
    text = first.read_text(encoding="utf-8").rstrip() + "\n\n"
    text += second.read_text(encoding="utf-8").lstrip()
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    if not CURRAS.exists():
        raise FileNotFoundError(f"Curras folder not found: {CURRAS}")
    rng = random.Random(SEED)
    # Split whole templates before generating variants. This prevents the model
    # from seeing the test sentence structure with only a different location word.
    base_indices = list(range(len(BASE)))
    rng.shuffle(base_indices)
    train_bases = base_indices[:24]
    validation_bases = base_indices[24:27]
    test_bases = base_indices[27:]

    def expand(indices: list[int]) -> list[list[tuple[str, str]]]:
        expanded = [variant(BASE[index], rng, cycle) for index in indices for cycle in range(8)]
        rng.shuffle(expanded)
        return expanded

    train = expand(train_bases) + PREVIOUS_ASSIGNMENT
    validation = expand(validation_bases)
    test = expand(test_bases)
    sentences = train + validation + test
    signature = lambda sentence: tuple(token for token, _ in sentence)
    train_set = {signature(sentence) for sentence in train}
    validation_set = {signature(sentence) for sentence in validation}
    test_set = {signature(sentence) for sentence in test}
    assert len(train_set) == len(train)
    assert len(validation_set) == len(validation)
    assert len(test_set) == len(test)
    assert train_set.isdisjoint(validation_set)
    assert train_set.isdisjoint(test_set)
    assert validation_set.isdisjoint(test_set)

    NEW.mkdir(parents=True, exist_ok=True)
    EXTENDED.mkdir(parents=True, exist_ok=True)
    write_conll(NEW / "train.txt", train, 900001)
    write_conll(NEW / "validation.txt", validation, 901001)
    write_conll(NEW / "test.txt", test, 902001)

    append_files(EXTENDED / "train.txt", CURRAS / "train.txt", NEW / "train.txt")
    append_files(
        EXTENDED / "validation.txt", CURRAS / "validation.txt", NEW / "validation.txt"
    )
    shutil.copy2(CURRAS / "test.txt", EXTENDED / "test.txt")
    shutil.copy2(CURRAS / "tune.txt", EXTENDED / "tune.txt")
    shutil.copy2(NEW / "test.txt", EXTENDED / "digital_campus_test.txt")

    all_tokens = [item for sentence in sentences for item in sentence]
    stats = {
        "name": "Palestinian Digital Campus Corpus",
        "seed": SEED,
        "sentence_count": len(sentences),
        "token_count": len(all_tokens),
        "splits": {
            "train_sentences": len(train),
            "validation_sentences": len(validation),
            "test_sentences": len(test),
            "train_tokens": sum(map(len, train)),
            "validation_tokens": sum(map(len, validation)),
            "test_tokens": sum(map(len, test)),
        },
        "pos_distribution": dict(Counter(tag for _, tag in all_tokens).most_common()),
        "split_leakage_check": "passed: all sentences are unique and split overlap is zero",
        "template_split": {
            "train_template_ids": train_bases,
            "validation_template_ids": validation_bases,
            "test_template_ids": test_bases,
            "template_overlap": 0,
        },
        "previous_assignment_reuse": {
            "sentences": len(PREVIOUS_ASSIGNMENT),
            "word_tokens": sum(len(sentence) - 1 for sentence in PREVIOUS_ASSIGNMENT),
            "pos_tokens_with_punctuation": sum(map(len, PREVIOUS_ASSIGNMENT)),
            "split": "train only",
        },
        "design": [
            "Palestinian student and computing vocabulary",
            "negation and future particles",
            "past and imperfect verb forms",
            "attached Palestinian prepositions",
            "English technical terms and proper names",
            "interrogatives, relative pronouns, quantifiers, and punctuation",
        ],
    }
    (ROOT / "corpus_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
