"""Text normalization: one form to feed the embedding model, one for exact-match keys."""
import re
import unicodedata

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
# Three or more repeats of a letter collapse to one: "noooo" -> "no", "broncaaa" -> "bronca".
# Digits are left alone so "1000" survives.
ELONGATION = re.compile(r"([^\W\d_])\1{2,}")
ABBREVIATIONS = {"q": "que", "k": "que", "xq": "porque", "pq": "porque", "porq": "porque", "x": "por",
                 "tmb": "también", "tb": "también", "pa": "para", "xfa": "por favor"}
ABBREVIATION = re.compile(r"\b(" + "|".join(ABBREVIATIONS) + r")\b")
NON_WORD = re.compile(r"[^\w#\s]")


def for_embedding(text: str) -> str:
    t = URL.sub(" ", text.lower())
    t = MENTION.sub(" ", t)
    t = ELONGATION.sub(r"\1", t)
    t = ABBREVIATION.sub(lambda m: ABBREVIATIONS[m.group(1)], t)
    return " ".join(t.split())


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def for_key(text: str) -> str:
    """Accent-, punctuation- and emoji-free form: two texts with the same key are copies."""
    return " ".join(NON_WORD.sub(" ", strip_accents(for_embedding(text))).split())
