import sys
import tracemalloc
from pathlib import Path
from dataclasses import dataclass
from array import array
from collections import defaultdict

# Додаємо папку src у шлях пошуку модулів
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from findex.corpus import iter_documents
from findex.tokenize import tokenize

# --- Варіант 1: Звичайний dataclass (без slots) ---
@dataclass
class PostingPlain:
    doc_id: int
    tf: int

# --- Варіант 2: dataclass з slots ---
@dataclass(frozen=True, slots=True)
class PostingSlots:
    doc_id: int
    tf: int

# --- Варіант 3: пари array('I') ---
# Ми не будемо створювати об'єкти Posting, а зберігатимемо два масиви:
# один для doc_id, інший для tf.

def build_index_plain(root: Path):
    """Будує індекс зі звичайними dataclass."""
    postings = defaultdict(list)
    for doc in iter_documents(root):
        doc_token_counts = defaultdict(int)
        for token in tokenize(doc.text):
            doc_token_counts[token] += 1
        for term, tf in doc_token_counts.items():
            postings[term].append(PostingPlain(doc_id=doc.doc_id, tf=tf))
    # Сортуємо
    for term in postings:
        postings[term].sort(key=lambda p: p.doc_id)
    return dict(postings)

def build_index_slots(root: Path):
    """Будує індекс з dataclass slots=True."""
    postings = defaultdict(list)
    for doc in iter_documents(root):
        doc_token_counts = defaultdict(int)
        for token in tokenize(doc.text):
            doc_token_counts[token] += 1
        for term, tf in doc_token_counts.items():
            postings[term].append(PostingSlots(doc_id=doc.doc_id, tf=tf))
    for term in postings:
        postings[term].sort(key=lambda p: p.doc_id)
    return dict(postings)

def build_index_array(root: Path):
    """Будує індекс з використанням array('I')."""
    # Для кожного терміну зберігаємо два масиви: doc_ids та tfs
    postings = defaultdict(lambda: (array('I'), array('I')))
    for doc in iter_documents(root):
        doc_token_counts = defaultdict(int)
        for token in tokenize(doc.text):
            doc_token_counts[token] += 1
        for term, tf in doc_token_counts.items():
            doc_ids, tfs = postings[term]
            doc_ids.append(doc.doc_id)
            tfs.append(tf)
    # Сортуємо (для простоти, конвертуємо в список пар, сортуємо, назад)
    # У реальному житті це робилося б складніше, але для бенчмарку пам'яті достатньо
    result = {}
    for term, (doc_ids, tfs) in postings.items():
        pairs = sorted(zip(doc_ids, tfs))
        new_doc_ids = array('I', (p[0] for p in pairs))
        new_tfs = array('I', (p[1] for p in pairs))
        result[term] = (new_doc_ids, new_tfs)
    return result

def measure_memory(build_func, root: Path):
    """Вимірює пікову пам'ять для функції побудови."""
    tracemalloc.start()
    index = build_func(root)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Повертаємо пікову пам'ять у МБ
    return peak / (1024 * 1024), index

if __name__ == "__main__":
    root = Path("data")
    print("Вимірювання пам'яті для різних варіантів Posting...")
    print("=" * 60)

    # 1. Plain dataclass
    mem_plain, _ = measure_memory(build_index_plain, root)
    print(f"1. Plain dataclass: {mem_plain:.4f} МБ")

    # 2. Slots dataclass
    mem_slots, _ = measure_memory(build_index_slots, root)
    print(f"2. Slots dataclass: {mem_slots:.4f} МБ")

    # 3. Array('I')
    mem_array, _ = measure_memory(build_index_array, root)
    print(f"3. Array('I'):      {mem_array:.4f} МБ")

    print("=" * 60)