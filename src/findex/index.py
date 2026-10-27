import sys
from pathlib import Path

# Додаємо папку src у шлях пошуку модулів
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
from typing import Iterator
from collections import defaultdict
from dataclasses import dataclass
from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int  # term frequency - скільки разів термін зустрічається в документі


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: Path
    length: int  # кількість токенів у документі


def build_index(root: Path) -> tuple[dict[str, list[Posting]], dict[int, DocMeta]]:
    """
    Будує інвертований індекс за один прохід по корпусу.
    Повертає (postings, doc_meta).
    """
    postings = defaultdict(list)
    doc_meta = {}

    for doc in iter_documents(root):
        # Рахуємо частоти слів у поточному документі
        doc_token_counts = defaultdict(int)
        doc_length = 0
        for token in tokenize(doc.text):
            doc_token_counts[token] += 1
            doc_length += 1

        # Зберігаємо метадані документа
        doc_meta[doc.doc_id] = DocMeta(doc_id=doc.doc_id, path=doc.path, length=doc_length)

        # Додаємо постінги для кожного унікального слова в документі
        for term, tf in doc_token_counts.items():
            postings[term].append(Posting(doc_id=doc.doc_id, tf=tf))

    # Сортуємо постінги для кожного терміну за doc_id
    for term in postings:
        postings[term].sort(key=lambda p: p.doc_id)

    return dict(postings), doc_meta


if __name__ == "__main__":
    import argparse
    import time
    import tracemalloc
    from findex.store import save_index, save_index_json

    parser = argparse.ArgumentParser(description="Побудова та збереження інвертованого індексу")
    parser.add_argument("root", type=Path, help="Шлях до папки з корпусом")
    parser.add_argument("--out", type=Path, default=Path("index.bin"), help="Файл для збереження індексу")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    postings, doc_meta = build_index(args.root)

    # Визначаємо формат за розширенням
    if args.out.suffix == ".json":
        save_index_json(postings, doc_meta, args.out)
    else:
        save_index(postings, doc_meta, args.out)

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Індекс побудовано та збережено у {args.out}")
    print(f"Термів: {len(postings)}, Документів: {len(doc_meta)}")
    print(f"Час: {elapsed:.4f} сек, Пікова пам'ять: {peak / (1024 * 1024):.4f} МБ")