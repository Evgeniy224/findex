import sys
from pathlib import Path

# Додаємо папку src у шлях пошуку модулів
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import heapq
from findex.index import Index
from findex.ranking import TfIdf, BM25, SearchResult
from findex.query_parser import parse_query
from findex.decorators import timed

# Ручний кеш для результатів пошуку
_search_cache = {}


@timed
def search(index: Index, query: str, scorer_name: str = "tfidf", top_k: int = 10) -> list[SearchResult]:
    """
    Пошук з підтримкою AND, OR, NOT, дужок, фраз та кешуванням.
    """
    cache_key = (id(index), query, scorer_name, top_k)

    # Перевіряємо, чи є результат у кеші
    if cache_key in _search_cache:
        print(f"  [Кеш] Використовуємо збережений результат для '{query}'")
        return _search_cache[cache_key]

    # Виконуємо пошук
    try:
        ast = parse_query(query)
    except ValueError as e:
        print(f"Помилка парсингу запиту: {e}")
        return []

    matched_docs = ast.evaluate(index)
    if not matched_docs:
        return []

    # Витягуємо терми для ранжування
    tokens = [t for t in query.casefold().split() if t not in ("and", "or", "not", "(", ")")]

    if scorer_name == "bm25":
        scorer = BM25()
    else:
        scorer = TfIdf()

    results = []
    for doc_id in matched_docs:
        total_score = 0.0
        for term in tokens:
            if term.startswith('"') and term.endswith('"'):
                term = term[1:-1]
            total_score += scorer.score(index, term, doc_id)
        results.append(SearchResult(score=total_score, doc_id=doc_id))

    top_results = heapq.nlargest(top_k, results)

    # Додаємо сніпети
    final_results = []
    for res in top_results:
        snippet = generate_snippet(index, res.doc_id, query)
        final_results.append(SearchResult(score=res.score, doc_id=res.doc_id, snippet=snippet))

    # Зберігаємо в кеш
    _search_cache[cache_key] = final_results
    return final_results


def generate_snippet(index: Index, doc_id: int, query: str, window: int = 80) -> str:
    """
    Генерує уривок тексту навколо найкращого входження термінів з підсвіткою.
    """
    meta = index.get_doc_meta(doc_id)
    if not meta:
        return ""

    try:
        text = meta.path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

    terms = [t.casefold() for t in query.split() if t not in ("and", "or", "not", "(", ")")]
    terms = [t.strip('"') for t in terms]

    text_lower = text.casefold()
    best_pos = -1
    best_term = ""
    for term in terms:
        pos = text_lower.find(term)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos
            best_term = term

    if best_pos == -1:
        return text[:window * 2] + "..."

    start = max(0, best_pos - window)
    end = min(len(text), best_pos + len(best_term) + window)

    snippet = text[start:end]

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    for term in terms:
        snippet = snippet.replace(term, f"*{term}*")
        snippet = snippet.replace(term.capitalize(), f"*{term.capitalize()}*")

    return snippet


if __name__ == "__main__":
    import argparse
    import time
    import tracemalloc
    from findex.store import open_index

    parser = argparse.ArgumentParser(description="Пошук за індексом")
    parser.add_argument("index_file", type=Path, help="Файл індексу")
    parser.add_argument("query", type=str, help="Пошуковий запит")
    parser.add_argument("--scorer", choices=["tfidf", "bm25"], default="tfidf", help="Алгоритм ранжування")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    with open_index(args.index_file) as index:
        results = search(index, args.query, scorer_name=args.scorer)

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Запит: '{args.query}' (scorer: {args.scorer})")
    print(f"Знайдено документів: {len(results)}")
    for res in results[:5]:
        meta = index.get_doc_meta(res.doc_id)
        if meta:
            print(f"  {res.doc_id}: {meta.path.name} (score: {res.score:.4f}, довжина: {meta.length})")
            if res.snippet:
                print(f"    Сніпет: {res.snippet}")
    if len(results) > 5:
        print(f"  ... і ще {len(results) - 5} документів")
    print(f"Час: {elapsed:.4f} сек, Пікова пам'ять: {peak / (1024 * 1024):.4f} МБ")