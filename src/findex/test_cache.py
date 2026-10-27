import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from findex.store import open_index
from findex.search import search

with open_index(Path("index.json")) as index:
    print("--- Перший запит ---")
    results1 = search(index, "пошукова")

    print("\n--- Другий запит (той самий) ---")
    results2 = search(index, "пошукова")