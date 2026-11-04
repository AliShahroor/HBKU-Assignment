"""Reusable Arabic preprocessing with genuine CAMeL Tools lemmatization."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar


ARABIC_LETTER_RE = re.compile(r"[\u0621-\u063A\u0641-\u064A]")
DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)
WHITESPACE_RE = re.compile(r"\s+")

# A small, transparent list suitable for these documents. The entries are
# normalized by the same function used for document tokens.
RAW_STOPWORDS = {
    "في", "من", "على", "إلى", "الى", "عن", "و", "ثم",
    "هذا", "هذه", "هو", "هي", "أن", "إن", "كان", "مع",
}


def normalize_arabic(text: str) -> str:
    """Apply conservative normalization without changing taa marbuta."""
    text = text.replace("ـ", "")
    text = DIACRITICS_RE.sub("", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


STOPWORDS = frozenset(normalize_arabic(word) for word in RAW_STOPWORDS)


def tokenize(text: str) -> list[str]:
    """Tokenize with CAMeL's Unicode-aware tokenizer and retain Arabic words."""
    return [
        token
        for token in simple_word_tokenize(text)
        if ARABIC_LETTER_RE.search(token)
    ]


@dataclass(frozen=True)
class TokenRecord:
    document: str
    position: int
    original: str
    normalized: str
    lemma: str
    raw_lemma: str
    stopword: bool
    analysis_available: bool

    def to_dict(self) -> dict:
        return asdict(self)


class ArabicPreprocessor:
    """Tokenize, normalize, and lemmatize MSA using CAMeL's MLE model."""

    def __init__(self) -> None:
        self.disambiguator = MLEDisambiguator.pretrained("calima-msa-r13")

    def process(self, document_id: str, text: str) -> list[TokenRecord]:
        tokens = tokenize(text)
        disambiguated = self.disambiguator.disambiguate(tokens)
        records: list[TokenRecord] = []

        for position, (token, result) in enumerate(
            zip(tokens, disambiguated), start=1
        ):
            normalized = normalize_arabic(token)
            analyses = result.analyses
            available = bool(analyses)
            if available:
                raw_lemma = analyses[0].analysis.get("lex", token)
                # CALIMA's lex field is the selected dictionary lemma. We
                # remove its diacritics, then apply the same conservative
                # orthographic normalization used for tokens.
                lemma = normalize_arabic(dediac_ar(raw_lemma))
            else:
                # Unknown tokens remain searchable by their normalized surface
                # form and are explicitly marked as unanalyzed in diagnostics.
                raw_lemma = token
                lemma = normalized

            records.append(
                TokenRecord(
                    document=document_id,
                    position=position,
                    original=token,
                    normalized=normalized,
                    lemma=lemma,
                    raw_lemma=raw_lemma,
                    stopword=normalized in STOPWORDS,
                    analysis_available=available,
                )
            )

        return records


def indexed_records(records: Iterable[TokenRecord]) -> list[TokenRecord]:
    """Return non-stopword records while retaining original positions."""
    return [record for record in records if not record.stopword]
