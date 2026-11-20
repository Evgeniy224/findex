import re
from dataclasses import dataclass
from typing import Literal

from findex.index import Index, Posting, intersect_postings, union_postings
from findex.tokenize import tokenize

EngineType = Literal["set", "merge"]


class QuerySyntaxError(ValueError):
    """Помилка синтаксису пошукового запиту."""


def _ids_to_postings(doc_ids: list[int]) -> list[Posting]:
    return [Posting(doc_id=d, tf=1) for d in doc_ids]


@dataclass(frozen=True)
class Term:
    term: str

    def evaluate(self, index: Index, engine: EngineType = "set") -> set[int]:
        if self.term not in index:
            return set()
        return {p.doc_id for p in index[self.term]}

    def __repr__(self) -> str:
        return f"Term({self.term!r})"


@dataclass(frozen=True)
class Phrase:
    terms: tuple[str, ...]

    def evaluate(self, index: Index, engine: EngineType = "set") -> set[int]:
        if not self.terms:
            return set()
        if engine == "merge":
            if self.terms[0] not in index:
                return set()
            current = index[self.terms[0]]
            for term in self.terms[1:]:
                if term not in index:
                    return set()
                matched_ids = intersect_postings(current, index[term])
                current = _ids_to_postings(matched_ids)
            return {p.doc_id for p in current}

        result: set[int] | None = None
        for term in self.terms:
            if term not in index:
                return set()
            docs = {p.doc_id for p in index[term]}
            result = docs if result is None else (result & docs)
        return result if result is not None else set()

    def __repr__(self) -> str:
        return f"Phrase({list(self.terms)!r})"


@dataclass(frozen=True)
class And:
    left: "Node"
    right: "Node"

    def evaluate(self, index: Index, engine: EngineType = "set") -> set[int]:
        if engine == "merge" and isinstance(self.left, Term) and isinstance(self.right, Term):
            left_p = index.get(self.left.term, [])
            right_p = index.get(self.right.term, [])
            return set(intersect_postings(left_p, right_p))
        return self.left.evaluate(index, engine) & self.right.evaluate(index, engine)

    def __repr__(self) -> str:
        return f"And({self.left}, {self.right})"


@dataclass(frozen=True)
class Or:
    left: "Node"
    right: "Node"

    def evaluate(self, index: Index, engine: EngineType = "set") -> set[int]:
        if engine == "merge" and isinstance(self.left, Term) and isinstance(self.right, Term):
            left_p = index.get(self.left.term, [])
            right_p = index.get(self.right.term, [])
            return set(union_postings(left_p, right_p))
        return self.left.evaluate(index, engine) | self.right.evaluate(index, engine)

    def __repr__(self) -> str:
        return f"Or({self.left}, {self.right})"


@dataclass(frozen=True)
class Not:
    node: "Node"

    def evaluate(self, index: Index, engine: EngineType = "set") -> set[int]:
        return index.all_doc_ids() - self.node.evaluate(index, engine)

    def __repr__(self) -> str:
        return f"Not({self.node})"


Node = Term | Phrase | And | Or | Not

_QUERY_TOKEN_RE = re.compile(r'\(|\)|"[^"]*"|\w+(?:[\'-]\w+)*', re.UNICODE)


class QueryParser:
    def __init__(self, query: str) -> None:
        if '"' in query and query.count('"') % 2 != 0:
            raise QuerySyntaxError("Незакриті лапки у запиті")
        self.tokens: list[str] = _QUERY_TOKEN_RE.findall(query)
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
        if not self.tokens:
            raise QuerySyntaxError("Порожній пошуковий запит")
        node = self.parse_or()
        if self.peek() is not None:
            raise QuerySyntaxError(f"Неочікуваний токен у запиті: '{self.peek()}'")
        return node

    def parse_or(self) -> Node:
        left = self.parse_and()
        while (tok := self.peek()) is not None and tok.upper() == "OR":
            self.consume()
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self) -> Node:
        left = self.parse_not()
        while (tok := self.peek()) is not None and tok.upper() == "AND":
            self.consume()
            right = self.parse_not()
            left = And(left, right)
        return left

    def parse_not(self) -> Node:
        tok = self.peek()
        if tok is not None and tok.upper() == "NOT":
            self.consume()
            return Not(self.parse_atom())
        return self.parse_atom()

    def parse_atom(self) -> Node:
        token = self.peek()
        if token is None:
            raise QuerySyntaxError("Несподіваний кінець запиту")
        if token == "(":
            self.consume()
            node = self.parse_or()
            if self.peek() != ")":
                raise QuerySyntaxError("Очікувалась закриваюча дужка ')'")
            self.consume()
            return node
        if token == ")":
            raise QuerySyntaxError("Неочікувана закриваюча дужка ')'")
        if token.upper() in {"AND", "OR", "NOT"}:
            raise QuerySyntaxError(f"Неочікуваний оператор '{token}'")
        if token.startswith('"'):
            self.consume()
            phrase_tokens = tuple(tokenize(token[1:-1]))
            if not phrase_tokens:
                raise QuerySyntaxError("Порожня фраза в лапках")
            return Phrase(phrase_tokens)

        self.consume()
        normalized = list(tokenize(token))
        if not normalized:
            raise QuerySyntaxError(f"Некоректний терм: '{token}'")
        return Term(normalized[0])


def parse_query(query: str) -> Node:
    return QueryParser(query).parse()