import argparse
import csv
import json
from pathlib import Path

from findex.search import clear_search_cache, search
from findex.store import load_index

MODES = ("keyword", "semantic", "hybrid")


def load_queries(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(kind, query) for kind, values in data.items() for query in values]


def generate(index_path: Path, queries_path: Path, output: Path, k: int) -> None:
    index = load_index(index_path)
    clear_search_cache()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_type", "query", "mode", "rank", "doc_id", "title", "relevant"])
        for query_type, query in load_queries(queries_path):
            for mode in MODES:
                clear_search_cache()
                results = search(
                    index,
                    query,
                    scorer_name="bm25",
                    top_k=k,
                    use_cache=False,
                    mode=mode,
                    index_path=index_path,
                )
                for rank, result in enumerate(results, start=1):
                    meta = index.get_doc_meta(result.doc_id)
                    title = meta.path.name if meta is not None else f"doc_{result.doc_id}"
                    writer.writerow([query_type, query, mode, rank, result.doc_id, title, ""])


def calculate(labels_path: Path) -> None:
    totals: dict[str, dict[str, list[float]]] = {}
    with labels_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("CSV містить нуль рядків")

    grouped: dict[tuple[str, str, str], list[int]] = {}
    for row in rows:
        if not row["relevant"].strip():
            raise SystemExit(
                f"Заповніть relevant=0 або 1: {row['query_type']} / {row['query']} / {row['mode']}"
            )
        key = (row["query_type"], row["query"], row["mode"])
        grouped.setdefault(key, [0] * 5)
        rank = int(row["rank"])
        if rank <= 5:
            grouped[key][rank - 1] = int(row["relevant"])

    for (query_type, _query, mode), relevance in grouped.items():
        totals.setdefault(query_type, {}).setdefault(mode, []).append(sum(relevance) / 5.0)

    print("| Тип запитів | Запитів | Keyword P@5 | Semantic P@5 | Hybrid P@5 |")
    print("|---|---:|---:|---:|---:|")
    for query_type in ("original", "paraphrases"):
        values = totals.get(query_type, {})
        count = len(values.get("keyword", []))
        if not count:
            continue
        keyword = sum(values["keyword"]) / count
        semantic = sum(values["semantic"]) / count
        hybrid = sum(values["hybrid"]) / count
        label = "Оригінальні запити" if query_type == "original" else "Paraphrases"
        print(f"| {label} | {count} | {keyword:.3f} | {semantic:.3f} | {hybrid:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("index.json"))
    parser.add_argument("--queries", type=Path, default=Path("data/evaluation_queries.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/search-evaluation.csv"))
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if args.labels:
        calculate(args.labels)
    else:
        generate(args.index, args.queries, args.output, args.k)
        print(f"Створено: {args.output}")


if __name__ == "__main__":
    main()
