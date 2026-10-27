import re
from dataclasses import dataclass
from findex.index import Index

# Вузли дерева запиту
@dataclass
class Term:
    term: str
    def evaluate(self, index: Index) -> set[int]:
        if self.term not in index:
            return set()
        return set(p.doc_id for p in index[self.term])
    def __repr__(self):
        return f"Term({self.term})"

@dataclass
class Phrase:
    terms: list[str]
    def evaluate(self, index: Index) -> set[int]:
        # Спрощена реалізація: шукаємо документи, де є ВСІ слова фрази
        if not self.terms:
            return set()
        result = None
        for term in self.terms:
            if term not in index:
                return set()
            docs = set(p.doc_id for p in index[term])
            if result is None:
                result = docs
            else:
                result = result & docs
        return result if result is not None else set()
    def __repr__(self):
        return f"Phrase({self.terms})"

@dataclass
class And:
    left: 'Node'
    right: 'Node'
    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)
    def __repr__(self):
        return f"And({self.left}, {self.right})"

@dataclass
class Or:
    left: 'Node'
    right: 'Node'
    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)
    def __repr__(self):
        return f"Or({self.left}, {self.right})"

@dataclass
class Not:
    node: 'Node'
    def evaluate(self, index: Index) -> set[int]:
        all_docs = index.all_doc_ids()
        return all_docs - self.node.evaluate(index)
    def __repr__(self):
        return f"Not({self.node})"

Node = Term | Phrase | And | Or | Not

class QueryParser:
    def __init__(self, query: str):
        # Розбиваємо на токени: дужки, лапки, слова
        self.tokens = re.findall(r'\(|\)|"[^"]*"|\w+', query)
        self.pos = 0

    def peek(self) -> str | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse(self) -> Node:
        return self.parse_or()

    def parse_or(self) -> Node:
        left = self.parse_and()
        while self.peek() == "OR" or self.peek() == "or":
            self.consume()
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self) -> Node:
        left = self.parse_not()
        while self.peek() == "AND" or self.peek() == "and":
            self.consume()
            right = self.parse_not()
            left = And(left, right)
        return left

    def parse_not(self) -> Node:
        if self.peek() == "NOT" or self.peek() == "not":
            self.consume()
            return Not(self.parse_atom())
        return self.parse_atom()

    def parse_atom(self) -> Node:
        token = self.peek()
        if token == "(":
            self.consume()
            node = self.parse_or()
            if self.peek() == ")":
                self.consume()
            return node
        elif token and token.startswith('"'):
            self.consume()
            phrase_text = token[1:-1]
            return Phrase(phrase_text.casefold().split())
        elif token:
            self.consume()
            return Term(token.casefold())
        else:
            raise ValueError("Несподіваний кінець запиту")

def parse_query(query: str) -> Node:
    return QueryParser(query).parse()