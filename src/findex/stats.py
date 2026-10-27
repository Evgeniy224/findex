import sys
from pathlib import Path

# --- ХАК ДЛЯ ЗАПУСКУ З ТЕРМІНАЛУ ---
# Додаємо папку src у шлях пошуку модулів, щоб Python знайшов findex
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# -----------------------------------

import argparse
import time
import tracemalloc
from collections import Counter
from itertools import islice

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def run_pipeline(root: Path, limit: int | None = None):
    """Лінива версія: обробляє документи по одному, не завантажуючи все в пам'ять."""
    tracemalloc.start()
    start_time = time.perf_counter()

    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)

    doc_count = 0
    total_tokens = 0
    term_counts = Counter()

    for doc in docs:
        doc_count += 1
        for token in tokenize(doc.text):
            term_counts[token] += 1
            total_tokens += 1

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "documents": doc_count,
        "tokens": total_tokens,
        "vocabulary": len(term_counts),
        "top_50": term_counts.most_common(50),
        "elapsed": elapsed,
        "peak_memory_mb": peak / (1024 * 1024),
    }


def run_eager(root: Path, limit: int | None = None):
    """Жадібна версія: завантажує всі документи та токени в списки."""
    tracemalloc.start()
    start_time = time.perf_counter()

    # Читаємо ВСІ документи в список
    all_docs = list(iter_documents(root))
    if limit is not None:
        all_docs = all_docs[:limit]

    # Токенізуємо КОЖЕН документ і створюємо список списків
    all_tokens_lists = [list(tokenize(doc.text)) for doc in all_docs]

    # А тепер рахуємо
    doc_count = len(all_docs)
    total_tokens = sum(len(tokens) for tokens in all_tokens_lists)
    term_counts = Counter()
    for tokens in all_tokens_lists:
        term_counts.update(tokens)

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "documents": doc_count,
        "tokens": total_tokens,
        "vocabulary": len(term_counts),
        "elapsed": elapsed,
        "peak_memory_mb": peak / (1024 * 1024),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Статистика корпусу findex")
    parser.add_argument("root", type=Path, help="Шлях до папки з корпусом")
    parser.add_argument("--limit", type=int, default=None, help="Обробити тільки N документів")
    args = parser.parse_args()

    # Запускаємо обидві версії для порівняння
    print("--- Лінива версія (генератори) ---")
    lazy_results = run_pipeline(args.root, args.limit)
    print(f"Документів: {lazy_results['documents']}")
    print(f"Токенів: {lazy_results['tokens']}")
    print(f"Словник: {lazy_results['vocabulary']}")
    print(f"Час: {lazy_results['elapsed']:.4f} сек")
    print(f"Пікова пам'ять: {lazy_results['peak_memory_mb']:.4f} МБ")
    print("\nТоп-50 термів (лінива версія):")
    for term, count in lazy_results['top_50']:
        print(f"  {term}: {count}")

    print("\n--- Жадібна версія (списки) ---")
    eager_results = run_eager(args.root, args.limit)
    print(f"Документів: {eager_results['documents']}")
    print(f"Токенів: {eager_results['tokens']}")
    print(f"Словник: {eager_results['vocabulary']}")
    print(f"Час: {eager_results['elapsed']:.4f} сек")
    print(f"Пікова пам'ять: {eager_results['peak_memory_mb']:.4f} МБ")