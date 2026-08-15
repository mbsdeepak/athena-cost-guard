"""SQL parsing for Athena queries.

Extracts the three things a cost estimate needs:
  * which tables are read,
  * which columns are referenced (for the future column-aware Tier-2 estimate),
  * which WHERE predicates can be pushed down to prune partitions.

Anything the parser can't confidently interpret is simply *not* used for
pruning. That keeps every Tier-1 estimate an honest upper bound rather than a
confidently-wrong number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

import sqlglot
from sqlglot import exp

# Comparison operators sqlglot exposes as distinct node classes.
_COMPARE = {
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.GT: ">",
    exp.LT: "<",
    exp.GTE: ">=",
    exp.LTE: "<=",
}


@dataclass(frozen=True)
class TableRef:
    """A table referenced by the query."""

    db: Optional[str]
    name: str

    @property
    def qualified(self) -> str:
        return f"{self.db}.{self.name}" if self.db else self.name


@dataclass(frozen=True)
class Predicate:
    """A single pushable WHERE atom, e.g. ``dt = '2026-08'``."""

    column: str
    op: str  # one of '=','!=','>','<','>=','<=','IN','BETWEEN'
    values: List[str]


@dataclass
class ParsedQuery:
    tables: List[TableRef]
    columns: Set[str] = field(default_factory=set)
    predicates: List[Predicate] = field(default_factory=list)
    select_star: bool = False
    raw: str = ""


def parse_query(sql: str, dialect: str = "athena") -> ParsedQuery:
    """Parse *sql* and return the structured facts we need for estimation."""
    tree = sqlglot.parse_one(sql, dialect=dialect)

    # CTE names look like tables when referenced; exclude them so we never try
    # to resolve a WITH-alias against the Glue catalog.
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}

    seen = set()
    tables: List[TableRef] = []
    for t in tree.find_all(exp.Table):
        if t.name.lower() in cte_names:
            continue
        ref = TableRef(db=t.db or None, name=t.name)
        if ref not in seen:
            seen.add(ref)
            tables.append(ref)

    columns = {c.name for c in tree.find_all(exp.Column)}
    select_star = tree.find(exp.Star) is not None

    predicates = _extract_predicates(tree.find(exp.Where))

    return ParsedQuery(
        tables=tables,
        columns=columns,
        predicates=predicates,
        select_star=select_star,
        raw=sql,
    )


def _extract_predicates(where: Optional[exp.Where]) -> List[Predicate]:
    """Collect AND-connected atoms we know how to push down.

    We deliberately only descend through ``AND`` (and parentheses). Anything
    under an ``OR`` is skipped, because pruning on one side of an OR would be
    unsound — skipping it just widens the estimate, which is the safe direction.
    """
    if where is None:
        return []
    preds: List[Predicate] = []
    _walk_and(where.this, preds)
    return preds


def _walk_and(node: exp.Expression, preds: List[Predicate]) -> None:
    if isinstance(node, exp.Paren):
        _walk_and(node.this, preds)
        return
    if isinstance(node, exp.And):
        _walk_and(node.left, preds)
        _walk_and(node.right, preds)
        return
    atom = _atom(node)
    if atom is not None:
        preds.append(atom)


def _atom(node: exp.Expression) -> Optional[Predicate]:
    # column <op> literal
    for cls, op in _COMPARE.items():
        if isinstance(node, cls):
            col, lit = node.left, node.right
            if isinstance(col, exp.Column) and isinstance(lit, exp.Literal):
                return Predicate(col.name, op, [lit.this])
            # Literal on the left: flip it (5 < x  ->  x > 5)
            if isinstance(lit, exp.Column) and isinstance(col, exp.Literal):
                flipped = {">": "<", "<": ">", ">=": "<=", "<=": ">="}.get(op, op)
                return Predicate(lit.name, flipped, [col.this])
            return None

    if isinstance(node, exp.In):
        col = node.this
        vals = [e.this for e in node.expressions if isinstance(e, exp.Literal)]
        if isinstance(col, exp.Column) and vals and len(vals) == len(node.expressions):
            return Predicate(col.name, "IN", vals)
        return None

    if isinstance(node, exp.Between):
        col = node.this
        low, high = node.args.get("low"), node.args.get("high")
        if (
            isinstance(col, exp.Column)
            and isinstance(low, exp.Literal)
            and isinstance(high, exp.Literal)
        ):
            return Predicate(col.name, "BETWEEN", [low.this, high.this])
        return None

    return None
