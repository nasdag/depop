from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import numpy as np
import unidecode

try:
    import spacy  # type: ignore
    _HAS_SPACY = True
except Exception:
    spacy = None  # type: ignore
    _HAS_SPACY = False


@lru_cache(maxsize=1)
def _get_spacy_model():
    """Load spaCy model if available; return None if not installed.

    Cached to avoid repeated loads during vectorizer transforms.
    """
    if not _HAS_SPACY:
        return None
    try:
        return spacy.load("en_core_web_sm")  # type: ignore[attr-defined]
    except Exception:
        return None


def normalize_nonascii(text: str) -> str:
    return unidecode.unidecode(text)


def replace_special_chars(text: str) -> str:
    return re.sub(r"[,;@#!\?\+\*\n\-: /]", " ", text)


def keep_alphanumeric_and_amp(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9& ]", "", text)


def remove_extra_spaces(text: str) -> str:
    return " ".join(text.split())


def lemmatize(text: str) -> str:
    nlp = _get_spacy_model()
    if nlp is None:
        return text
    doc = nlp(text)
    lemmas = [t.lemma_ if t.lemma_ != "-PRON-" else t.text for t in doc]
    return " ".join(lemmas)


def clean_text(text: str, do_lemmatize: bool = True) -> str:
    """Normalize text with optional lemmatization.

    Steps: lowercase -> replace special chars -> ASCII fold -> keep alnum/& ->
    optional lemmatization -> collapse spaces.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = replace_special_chars(text)
    text = normalize_nonascii(text)
    text = keep_alphanumeric_and_amp(text)
    if do_lemmatize:
        text = lemmatize(text)
    text = remove_extra_spaces(text)
    return text


def split_tokens(text: str) -> Iterable[str]:
    return text.split(" ") if text else []


def proportion_of_stopwords(text: str) -> float:
    """Proportion of tokens that are spaCy stopwords.

    Returns 0.0 if spaCy is unavailable.
    """
    nlp = _get_spacy_model()
    tokens = list(split_tokens(text))
    if not tokens:
        return 0.0
    if nlp is None:
        return 0.0
    stopwords = nlp.Defaults.stop_words  # type: ignore[attr-defined]
    num_stop = sum(1 for w in tokens if w.lower() in stopwords)
    return float(num_stop) / float(len(tokens))


def average_word_length(text: str) -> float:
    tokens = list(split_tokens(text))
    if not tokens:
        return 0.0
    return float(np.mean([len(w) for w in tokens]))


def proportion_of_numbers(text: str) -> float:
    tokens = list(split_tokens(text))
    if not tokens:
        return 0.0
    digits = sum(1 for w in tokens if w.isdigit())
    return float(digits) / float(len(tokens))


def count_hash_tags(text: str) -> int:
    return len(re.findall(r"\B#\w+", text or ""))


