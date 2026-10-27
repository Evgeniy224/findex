import pytest
from findex.tokenize import tokenize

def test_basic_sentences():
    text = "Привіт, світ! Це тест."
    assert list(tokenize(text)) == ["привіт", "світ", "це", "тест"]

def test_mixed_case():
    text = "PyThOn та JAVA"
    assert list(tokenize(text)) == ["python", "та", "java"]

def test_apostrophes():
    # Апостроф всередині слова не має розривати його
    text = "п'ять, don't, l'hopital"
    assert list(tokenize(text)) == ["п'ять", "don't", "l'hopital"]

def test_hyphens():
    # Дефіс всередині слова
    text = "будь-який, well-known, -початок, кінець-"
    # Зверни увагу: дефіси на початку чи в кінці слова мають бути відкинуті
    assert list(tokenize(text)) == ["будь-який", "well-known", "початок", "кінець"]

def test_numbers():
    text = "2026 рік, 12345"
    assert list(tokenize(text)) == ["2026", "рік", "12345"]

def test_cyrillic():
    text = "Ґанок, їжак, Європа"
    assert list(tokenize(text)) == ["ґанок", "їжак", "європа"]

def test_unicode_normalization():
    # 'café' може бути записано як один символ (U+00E9) або як 'e' + комбінований апостроф (U+0301)
    text_nfc = "café"  # вже нормалізований
    text_nfd = "cafe\u0301"  # розкладений
    # Після токенізації обидва варіанти мають дати однаковий результат
    assert list(tokenize(text_nfc)) == list(tokenize(text_nfd)) == ["café"]

def test_empty_string():
    assert list(tokenize("")) == []

def test_punctuation_only():
    assert list(tokenize("!!! ... ???")) == []