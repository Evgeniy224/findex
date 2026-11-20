import re
import unicodedata
from collections.abc import Iterator

TOKEN_PATTERN = re.compile(r"\w+(?:['-]\w+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Перетворює текст на послідовність нормалізованих токенів (NFC + casefold)."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in TOKEN_PATTERN.finditer(normalized):
        yield match.group()