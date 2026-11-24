#!/usr/bin/env python3
"""Build a deterministic Arabic Word2Vec experiment for Assignment 5."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

import gensim
from gensim.models import Word2Vec


ROOT = Path(__file__).resolve().parent
SEED = 42

PEOPLE = ["الطالب", "الطالبة", "المبرمج", "المبرمجة", "الباحث", "الباحثة"]
PLACES = ["المختبر", "الجامعة", "المكتبة", "القاعة"]
LANGUAGES = ["بايثون", "جافا", "جافاسكربت"]
PROJECTS = ["التطبيق", "الموقع", "النظام", "البرنامج"]
DATA = ["البيانات", "النصوص", "الصور", "الملفات"]
MODELS = ["النموذج", "الخوارزمية", "الشبكة", "المصنف"]

PREVIOUS_ASSIGNMENT = [
    "مبارح بعد الدوام رحت عالسوق مع أختي واشترينا خضرة للبيت وبعدين قعدنا عند جيرانا نشرب قهوة ونحكي عن الجامعة والمشاريع",
    "اليوم الصبح صحيت بكير عشان عندي محاضرة بس الباص تأخر فمشيت عالجامعة وأنا مستعجل وبحكي لصاحبي يستناني عند الباب هناك",
    "بالليل أمي عملت مقلوبة وسلطة وتعشينا مع الجيران سوا وبعد الأكل شربنا شاي وقعدنا نحكي ونضحك لحد ما تأخر الوقت",
    "يوم الجمعة طلعنا عرامالله سوا نزور قرايبنا وتمشينا بالبلد واشترينا كنافة ورجعنا عالبيت قبل المغرب لأن الشوارع كانت أزمة كثير",
]


def tokenize(text: str) -> list[str]:
    """A transparent whitespace tokenizer with Arabic punctuation removal."""
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text, flags=re.UNICODE)
    return [token for token in text.split() if token]


def build_sentences() -> list[str]:
    rng = random.Random(SEED)
    patterns = [
        "مهمة {person} هي كتابة الشفرة بلغة {language} داخل {place} ثم اختبار {project} بعناية",
        "مشروع {person} يستخدم {language} لحل مشكلة برمجية واضحة",
        "تعلم {language} يساعد {person} على فهم البرمجة وكتابة الشفرة الصحيحة",
        "فريق البرمجة يراجع الشفرة ويصلح الخطأ قبل نشر {project}",
        "المطور يرفع الشفرة إلى المستودع ويحفظ نسخة جديدة من {project}",
        "المبرمج يقرأ رسالة الخطأ ثم يشغل الاختبار ويصحح الخلل",
        "الطالبة تنفذ الخوارزمية وتقارن الزمن والذاكرة في كل تجربة",
        "الخوارزمية السريعة تقلل الزمن بينما الخوارزمية البطيئة تستهلك ذاكرة أكبر",
        "عمل {person} يشمل جمع {data} وتنظيف {data} قبل تدريب {model}",
        "تدريب {model} يحتاج بيانات كثيرة واختبارا مستقلا ونتائج دقيقة",
        "الباحث يقسم البيانات إلى تدريب وتطوير واختبار قبل تقييم النموذج",
        "النموذج يتعلم من بيانات التدريب ثم يتنبأ بالنتيجة على بيانات الاختبار",
        "الدقة تقيس صحة التنبؤ بينما الاستدعاء يقيس اكتشاف الأمثلة المطلوبة",
        "درجة اف واحد توازن بين الدقة والاستدعاء عند تقييم المصنف",
        "معالجة اللغة الطبيعية تساعد الحاسوب على تحليل النصوص وفهم الكلمات",
        "نموذج اللغة يتعلم سياق الكلمة من الكلمات القريبة داخل الجملة",
        "تمثيل الكلمة يحول الكلمة إلى متجه رقمي يحمل معلومات دلالية",
        "تشابه جيب التمام يقارن متجهين ويقيس قرب المعنى بين كلمتين",
        "الكلمات المتقاربة في السياق تحصل غالبا على متجهات متشابهة",
        "الذكاء الاصطناعي يستخدم الخوارزميات والبيانات لبناء أنظمة ذكية",
        "تعلم الآلة فرع من الذكاء الاصطناعي ويعتمد على البيانات والنماذج",
        "التعلم العميق يستخدم شبكة عصبية متعددة الطبقات لتحليل الصور والنصوص",
        "الشبكة العصبية تتعلم الأنماط من الأمثلة وتحدث الأوزان أثناء التدريب",
        "قاعدة البيانات تخزن المعلومات وينفذ النظام استعلاما لاسترجاع السجلات",
        "الخادم يستقبل الطلب من العميل ثم يعيد الاستجابة عبر الشبكة",
        "الشبكة تربط الحاسوب بالخادم وينقل البروتوكول البيانات بأمان",
        "أمن المعلومات يحمي الحساب وكلمة المرور من الهجوم والاختراق",
        "التشفير يحول الرسالة إلى بيانات محمية ولا يقرأها إلا المستخدم المصرح",
        "واجب {person} يتطلب فتح الحاسوب في {place} وإكمال البرمجة مع الفريق",
        "الطلاب يناقشون المشروع ويقسمون المهام ثم يدمجون الشفرة في المستودع",
        "الأستاذ يشرح الخوارزمية على اللوح والطلاب يطبقونها في المختبر",
        "المحاضرة تشرح البيانات والنموذج والتقييم بمثال بسيط من الواقع",
        "الطالب يسأل سؤالا تقنيا والطالبة تقترح حلا وتختبر الفكرة",
        "الجامعة توفر مختبرا وحواسيب وشبكة للطلاب والباحثين",
        "المكتبة توفر كتبا عن البرمجة والخوارزميات والذكاء الاصطناعي",
        "المشروع الجيد يبدأ بمشكلة واضحة وينتهي باختبار وتوثيق للنتائج",
        "التوثيق يشرح طريقة تشغيل البرنامج ويساعد المطور على إعادة التجربة",
        "إعادة التجربة تحتاج الشفرة والبيانات والإعدادات والبذرة العشوائية نفسها",
        "الرسم البياني يعرض النتائج ويسهل مقارنة النماذج والخوارزميات",
        "الجدول يسجل قيمة الدقة ودرجة اف واحد لكل نموذج في الاختبار",
    ]

    sentences: list[str] = list(PREVIOUS_ASSIGNMENT)
    # Generate varied but controlled contexts until the corpus is near 3,000 tokens.
    while sum(len(tokenize(s)) for s in sentences) < 3000:
        pattern = patterns[len(sentences) % len(patterns)]
        sentence = pattern.format(
            person=rng.choice(PEOPLE),
            place=rng.choice(PLACES),
            language=rng.choice(LANGUAGES),
            project=rng.choice(PROJECTS),
            data=rng.choice(DATA),
            model=rng.choice(MODELS),
        )
        sentences.append(sentence)
    return sentences


def main() -> None:
    sentences = build_sentences()
    tokens = [tokenize(sentence) for sentence in sentences]
    token_count = sum(map(len, tokens))

    (ROOT / "corpus.txt").write_text("\n".join(sentences) + "\n", encoding="utf-8")

    model = Word2Vec(
        sentences=tokens,
        vector_size=100,
        window=5,
        min_count=1,
        sg=1,
        negative=10,
        epochs=200,
        workers=1,
        seed=SEED,
    )
    model.save(str(ROOT / "arabic_cs_word2vec.model"))

    candidate_pairs = [
        ("الطالب", "الطالبة", "people in the same university role"),
        ("المبرمج", "المبرمجة", "people in the same programming role"),
        ("الشفرة", "البرمجة", "code and its activity"),
        ("النموذج", "البيانات", "a model and what it learns from"),
        ("الدقة", "الاستدعاء", "two evaluation measures"),
        ("الخادم", "العميل", "two communicating network roles"),
        ("التدريب", "الاختبار", "two machine-learning stages"),
        ("الخوارزمية", "ذاكرة", "an algorithm and a measured resource"),
    ]
    rows = []
    for first, second, note in candidate_pairs:
        rows.append(
            {
                "word_1": first,
                "word_2": second,
                "cosine_similarity": round(float(model.wv.similarity(first, second)), 4),
                "interpretation": note,
            }
        )
    # Keep five examples spanning roles, programming, ML, metrics, and networks.
    chosen_indexes = [0, 2, 3, 4, 5]
    chosen = [rows[i] for i in chosen_indexes]
    with (ROOT / "similarity_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=chosen[0].keys())
        writer.writeheader()
        writer.writerows(chosen)

    metadata = {
        "seed": SEED,
        "gensim_version": gensim.__version__,
        "sentence_count": len(sentences),
        "token_count": token_count,
        "vocabulary_size": len(model.wv),
        "previous_assignment_reuse": {
            "sentences": len(PREVIOUS_ASSIGNMENT),
            "tokens": sum(len(tokenize(sentence)) for sentence in PREVIOUS_ASSIGNMENT),
            "placement": "first four corpus lines",
        },
        "parameters": {
            "architecture": "skip-gram",
            "vector_size": 100,
            "window": 5,
            "min_count": 1,
            "negative": 10,
            "epochs": 200,
            "workers": 1,
        },
        "five_reported_pairs": chosen,
        "all_candidate_pairs": rows,
    }
    (ROOT / "experiment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
