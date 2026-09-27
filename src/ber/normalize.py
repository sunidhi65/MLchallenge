from __future__ import annotations

import re
import unicodedata

import pandas as pd

try:
    from anyascii import anyascii as _anyascii
except ImportError:  # pragma: no cover - anyascii is in requirements.txt
    _anyascii = None


LEGAL_FORMS = {
    "inc": "inc",
    "incorporated": "inc",
    "corp": "corp",
    "corporation": "corp",
    "co": "company",
    "company": "company",
    "llc": "llc",
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "private": "private",
    "llp": "llp",
    "plc": "plc",
    "pc": "pc",
    "sa": "sa",
    "sas": "sas",
    "sasu": "sas",
    "sarl": "sarl",
    "eurl": "eurl",
    "snc": "snc",
    "sci": "sci",
    "societe": "societe",
    "ste": "societe",
    "groupe": "group",
    "group": "group",
}

ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "str": "street",
    "street": "street",
    "rd": "road",
    "road": "road",
    "ave": "avenue",
    "av": "avenue",
    "avenue": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "boulevard": "boulevard",
    "ln": "lane",
    "lane": "lane",
    "dr": "drive",
    "drive": "drive",
    "ct": "court",
    "court": "court",
    "r": "rue",
    "rue": "rue",
    "chem": "chemin",
    "chemin": "chemin",
    "imp": "impasse",
    "impasse": "impasse",
    "rte": "route",
    "route": "route",
    "apt": "apartment",
    "apartment": "apartment",
    "unit": "unit",
    "fl": "floor",
    "floor": "floor",
    "dist": "district",
    "distt": "district",
    "district": "district",
    "nagar": "nagar",
    "marg": "marg",
    "colony": "colony",
}

STOP_TOKENS = {
    # English
    "the",
    "and",
    "of",
    "for",
    "in",
    "at",
    "near",
    "opp",
    "opposite",
    "behind",
    "next",
    "to",
    "null",
    "na",
    "n",
    "no",
    # French (test set includes France, unseen in training; these are generic
    # function words that otherwise survive normalization as "content" tokens
    # and inflate token-overlap/Jaccard similarity between unrelated French
    # businesses that merely share common articles/prepositions)
    "le",
    "la",
    "les",
    "l",
    "de",
    "du",
    "des",
    "un",
    "une",
    "et",
    "a",
    "au",
    "aux",
    "en",
    "sur",
    "dans",
    "pour",
    "par",
    "avec",
    "chez",
}

LANDMARK_TOKENS = {"near", "opp", "opposite", "behind", "beside", "next", "front"}


def strip_accents(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def transliterate(value: str) -> str:
    """Romanize any script to ASCII (anyascii, ISC license -- not unidecode,
    which is GPL and unsuitable here).

    ~31% of India's business_name/business_address records (measured via
    audit_v5.py) contain non-Latin script (Devanagari, Tamil, Telugu,
    Gujarati, Gurmukhi, Bengali, ...); those can never token- or
    character-match a Latin-script twin without this step. anyascii also
    subsumes plain accent-stripping (cafe/café both -> "cafe"), so this
    replaces strip_accents in the normalization path; strip_accents itself
    is kept as-is since audit_v5.py relies on its narrower (accent-only)
    behavior to detect non-Latin script in the first place.
    """
    if _anyascii is None:  # pragma: no cover - anyascii is in requirements.txt
        return strip_accents(value)
    return _anyascii(str(value))


def tokenize(value: str) -> list[str]:
    value = transliterate(value).lower().replace("&", " and ")
    return re.findall(r"\w+", value, flags=re.UNICODE)


def normalize_name(value: str) -> tuple[str, str]:
    forms: list[str] = []
    core: list[str] = []
    for token in tokenize(value):
        canonical = LEGAL_FORMS.get(token)
        if canonical:
            forms.append(canonical)
        elif token not in STOP_TOKENS:
            core.append(token)
    return " ".join(core), "|".join(sorted(set(forms)))


def normalize_address(value: str) -> tuple[str, str, str]:
    tokens: list[str] = []
    landmarks: list[str] = []
    for token in tokenize(value):
        if token in LANDMARK_TOKENS:
            landmarks.append(token)
            continue
        if token in STOP_TOKENS:
            continue
        tokens.append(ADDRESS_ABBREVIATIONS.get(token, token))
    numbers = "|".join(re.findall(r"\d+", str(value)))
    return " ".join(tokens), numbers, "|".join(sorted(set(landmarks)))


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized text and parsed cue columns used by blocking/features."""

    out = df.copy()
    name_parts = out["business_name"].map(normalize_name)
    address_parts = out["business_address"].map(normalize_address)
    out["name_norm"] = [item[0] for item in name_parts]
    out["legal_form"] = [item[1] for item in name_parts]
    out["address_norm"] = [item[0] for item in address_parts]
    out["address_numbers"] = [item[1] for item in address_parts]
    out["landmark_tokens"] = [item[2] for item in address_parts]
    out["combined_norm"] = (
        out["name_norm"].fillna("")
        + " "
        + out["address_norm"].fillna("")
        + " "
        + out["country"].fillna("")
    ).str.strip()
    out["country_norm"] = out["country"].map(lambda x: strip_accents(str(x)).lower().strip())
    return out


def token_set(value: str) -> set[str]:
    if not value:
        return set()
    return set(str(value).split())


def pipe_set(value: str) -> set[str]:
    if not value:
        return set()
    return {item for item in str(value).split("|") if item}
