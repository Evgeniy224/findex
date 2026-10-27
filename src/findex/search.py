import sys
from pathlib import Path

# Додаємо папку src у шлях пошуку модулів
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from findex.index import Posting, DocMeta


def _search_and(postings: dict[str, list[Posting]], terms: list[str]) -> list[int]:
    """Виконує AND-пошук для списку термінів (через set для простоти)."""
    if not terms:
        return []

    result_docs = None
    for term in terms:
        if term not in postings:
            return []
        term_docs = set(p.doc_id for p in postings[term])
        if result_docs is None:
            result_docs = term_docs
        else:
            result_docs = result_docs & term_docs

    return sorted(list(result_docs)) if result_docs is not None else []


def search(postings: dict[str, list[Posting]], query: str, engine: str = "set") -> list[int]:
    """
    Пошук з підтримкою AND, OR, NOT.
    engine: "merge" або "set" (поки що обидва використовують set для простоти).
    """
    tokens = query.casefold().split()
    if not tokens:
        return []

    # Перевіряємо наявність OR
    if "or" in tokens:
        groups = []
        current_group = []
        for t in tokens:
            if t == "or":
                if current_group:
                    groups.append(current_group)
                current_group = []
            else:
                current_group.append(t)
        if current_group:
            groups.append(current_group)

        result_sets = []
        for group in groups:
            group_result = _search_and(postings, group)
            result_sets.append(set(group_result))

        final_result = set()
        for s in result_sets:
            final_result = final_result | s
        return sorted(list(final_result))

    # Перевіряємо наявність NOT
    if "not" in tokens:
        idx = tokens.index("not")
        if idx + 1 < len(tokens):
            pos_terms = tokens[:idx]
            neg_term = tokens[idx + 1]

            pos_result = _search_and(postings, pos_terms)
            if not pos_result:
                return []

            neg_docs = set()
            if neg_term in postings:
                neg_docs = set(p.doc_id for p in postings[neg_term])

            return sorted([d for d in pos_result if d not in neg_docs])

    # Звичайний AND
    return _search_and(postings, tokens)


if __name__ == "__main__":
    import argparse
    import time
    import tracemalloc
    from findex.store import load_index, load_index_json

    parser = argparse.ArgumentParser(description="Пошук за індексом")
    parser.add_argument("index_file", type=Path, help="Файл індексу")
    parser.add_argument("query", type=str, help="Пошуковий запит")
    parser.add_argument("--engine", choices=["merge", "set"], default="set",
                        help="Алгоритм пошуку (поки що обидва працюють однаково)")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    # Визначаємо формат за розширенням
    if args.index_file.suffix == ".json":
        postings, doc_meta = load_index_json(args.index_file)
    else:
        postings, doc_meta = load_index(args.index_file)

    results = search(postings, args.query, engine=args.engine)

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Запит: '{args.query}' (engine: {args.engine})")
    print(f"Знайдено документів: {len(results)}")
    for doc_id in results[:5]:
        meta = doc_meta.get(doc_id)
        if meta:
            print(f"  {doc_id}: {meta.path.name} (довжина: {meta.length})")
    if len(results) > 5:
        print(f"  ... і ще {len(results) - 5} документів")
    print(f"Час: {elapsed:.4f} сек, Пікова пам'ять: {peak / (1024 * 1024):.4f} МБ")