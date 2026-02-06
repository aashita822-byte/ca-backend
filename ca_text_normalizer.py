import re
from ca_abbreviations import CA_ABBREVIATIONS

# Keys are ALREADY sorted by length (DESC)
_KEYS = list(CA_ABBREVIATIONS.keys())

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _KEYS) + r")\b(?!\s*\()",
    flags=re.IGNORECASE,
)

def expand_ca_abbreviations(text: str) -> str:
    if not text:
        return text

    def replacer(match: re.Match) -> str:
        key = match.group(0)
        full_form = CA_ABBREVIATIONS.get(key.upper())
        return f"{key} ({full_form})" if full_form else key

    return _PATTERN.sub(replacer, text)
