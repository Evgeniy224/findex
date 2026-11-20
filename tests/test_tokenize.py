import pytest

from findex.tokenize import tokenize


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("Hello World", ["hello", "world"]),
        ("П'ять будь-яких слів", ["п'ять", "будь-яких", "слів"]),
        ("Café vs cafe\u0301", ["café", "vs", "café"]),
        ("2026 рік: №1, 12345!", ["2026", "рік", "1", "12345"]),
        ("   ---   '''   ", []),
        ("PYTHON_3_13 and async/await", ["python_3_13", "and", "async", "await"]),
        ("Кирилиця: Їжак, Ґанок, Єнот, Індекс", ["кирилиця", "їжак", "ґанок", "єнот", "індекс"]),
    ],
)
def test_tokenize_cases(raw_text: str, expected: list[str]) -> None:
    assert list(tokenize(raw_text)) == expected


def test_tokenize_is_lazy_iterator() -> None:
    gen = tokenize("один два три")
    assert next(gen) == "один"
    assert list(gen) == ["два", "три"]