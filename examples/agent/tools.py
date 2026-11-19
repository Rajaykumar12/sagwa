"""The agent's three tools. Each takes plain arguments and returns text, and
raises on bad input: the orchestrator records a raised error on the tool step,
which is what retry and recovery metrics look at later."""
from __future__ import annotations

import ast
import math
import operator
import re
import sqlite3

from examples.agent.world import DOCUMENTS, SCHEMA_DESCRIPTION

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "in", "is", "what", "how", "for", "and", "or", "do", "does", "can", "i", "my", "on", "it", "be"}


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def corpus_search(query: str, k: int = 2) -> str:
    """Lexical search over the policy documents: deterministic, no model."""
    q = _tokens(query)
    if not q:
        raise ValueError("empty search query")
    n = len(DOCUMENTS)
    df = {w: sum(w in _tokens(text) for text in DOCUMENTS.values()) for w in q}
    scores = {
        doc_id: sum(math.log(1 + n / df[w]) for w in q & _tokens(f"{doc_id} {text}") if df[w])
        for doc_id, text in DOCUMENTS.items()
    }
    ranked = [d for d in sorted(scores, key=scores.get, reverse=True) if scores[d] > 0][:k]
    if not ranked:
        return "No matching documents."
    return "\n".join(f"[{d}] {DOCUMENTS[d]}" for d in ranked)


def order_db(conn: sqlite3.Connection, sql: str, max_rows: int = 50) -> str:
    """Runs one read-only SQL query. The connection is `query_only`, so a
    write fails in SQLite itself, not just here."""
    if not sql.strip().lower().startswith(("select", "with")):
        raise ValueError("only SELECT queries are allowed")
    cursor = conn.execute(sql)
    columns = [c[0] for c in cursor.description]
    rows = cursor.fetchmany(max_rows + 1)
    if not rows:
        return "(no rows)"
    lines = [" | ".join(columns)] + [" | ".join("" if v is None else str(v) for v in row) for row in rows[:max_rows]]
    if len(rows) > max_rows:
        lines.append(f"... truncated to {max_rows} rows")
    return "\n".join(lines)


_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}


def calculator(expression: str) -> str:
    """Arithmetic only (+ - * / ** %, parentheses, unary minus), via the AST:
    never `eval`."""

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError(f"unsupported expression: {expression!r}")

    result = ev(ast.parse(expression.replace(",", ""), mode="eval").body)
    return str(round(result, 4))


# Tool schemas in the OpenAI/Groq function-calling format. `descriptions`
# is swappable because V2_ToolOverlap regresses exactly these strings.
CLEAR_DESCRIPTIONS = {
    "corpus_search": "Search Brindle Home's policy documents (returns, shipping times, late-delivery "
                     "compensation, warranty, loyalty discounts, cancellations, stores). Use for rules and policies.",
    "order_db": "Run one read-only SQL query against the orders database. Tables: " + SCHEMA_DESCRIPTION
                + " Use for facts about specific orders, customers, products and prices.",
    "calculator": "Evaluate an arithmetic expression, e.g. '3 * 129 * 0.9'. Use for any arithmetic.",
}
AMBIGUOUS_DESCRIPTIONS = {
    "corpus_search": "Look up information about Brindle Home.",
    "order_db": "Look up information about Brindle Home. Tables: " + SCHEMA_DESCRIPTION,
    "calculator": "Evaluate an arithmetic expression.",
}
_PARAMS = {
    "corpus_search": {"query": "search terms"},
    "order_db": {"sql": "a single SELECT statement"},
    "calculator": {"expression": "an arithmetic expression"},
}


def tool_schemas(descriptions: dict[str, str]) -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": name, "description": descriptions[name],
            "parameters": {"type": "object",
                           "properties": {p: {"type": "string", "description": d} for p, d in params.items()},
                           "required": list(params)},
        }}
        for name, params in _PARAMS.items()
    ]
