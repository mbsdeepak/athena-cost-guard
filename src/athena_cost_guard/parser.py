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
from typing import Dict, List, Optional, Set, Tuple

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
    # Flat union of every pushable predicate, kept for introspection. Pruning
    # should use ``predicates_for`` so a predicate is only applied to the table
    # it provably constrains.
    predicates: List[Predicate] = field(default_factory=list)
    # Predicates attributed to the specific table they constrain, keyed by the
    # table's qualified name (as it appears in ``tables``).
    table_predicates: Dict[str, List[Predicate]] = field(default_factory=dict)
    select_star: bool = False
    raw: str = ""

    def predicates_for(self, ref: TableRef) -> List[Predicate]:
        """Return only the predicates that provably constrain *ref*.

        Predicates that can't be soundly attributed to a single table (an
        unqualified column in a multi-table scope, a column qualified by a CTE
        or subquery alias, or any predicate on a table referenced more than
        once) are omitted — dropping them widens the estimate, which is the
        safe direction.
        """
        return self.table_predicates.get(ref.qualified, [])


def parse_query(sql: str, dialect: str = "athena") -> ParsedQuery:
    """Parse *sql* and return the structured facts we need for estimation."""
    tree = sqlglot.parse_one(sql, dialect=dialect)

    # CTE names look like tables when referenced; exclude them so we never try
    # to resolve a WITH-alias against the Glue catalog.
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}

    seen = set()
    tables: List[TableRef] = []
    for t in tree.find_all(exp.Table):
        # CTE aliases are always unqualified. Only skip a reference when it's
        # unqualified *and* matches a CTE name — otherwise a schema-qualified
        # table (e.g. billing.line_items) that happens to share a CTE's bare
        # name would be wrongly dropped, zeroing out the estimate.
        if not t.db and t.name.lower() in cte_names:
            continue
        ref = TableRef(db=t.db or None, name=t.name)
        if ref not in seen:
            seen.add(ref)
            tables.append(ref)

    columns = {c.name for c in tree.find_all(exp.Column)}
    select_star = tree.find(exp.Star) is not None

    table_predicates = _attribute_predicates(tree, cte_names)
    # Flat union kept for introspection; order follows attribution order.
    predicates: List[Predicate] = []
    for preds in table_predicates.values():
        predicates.extend(preds)

    return ParsedQuery(
        tables=tables,
        columns=columns,
        predicates=predicates,
        table_predicates=table_predicates,
        select_star=select_star,
        raw=sql,
    )


def _immediate_tables(select: exp.Select) -> List[exp.Table]:
    """The physical tables in *select*'s own FROM/JOINs (not nested scopes).

    Subqueries and derived tables are skipped here — each is its own
    ``exp.Select`` and gets visited on its own, so its tables are attributed in
    the scope where they actually appear.
    """
    out: List[exp.Table] = []
    from_ = select.args.get("from_")
    if from_ is not None and isinstance(from_.this, exp.Table):
        out.append(from_.this)
    for join in select.args.get("joins") or []:
        if isinstance(join.this, exp.Table):
            out.append(join.this)
    return out


def _attribute_predicates(
    tree: exp.Expression, cte_names: Set[str]
) -> Dict[str, List[Predicate]]:
    """Map each physical table to the predicates that provably constrain it.

    Each ``SELECT`` scope is handled independently: its WHERE atoms are matched
    against the tables in *its own* FROM, so a filter on one table can never be
    pushed onto another. A predicate is attributed only when we can name its
    table unambiguously — by an explicit qualifier, or because the scope has a
    single physical table. Everything else is dropped (a wider, safe estimate).
    """
    table_predicates: Dict[str, List[Predicate]] = {}
    occurrences: Dict[str, int] = {}

    for select in tree.find_all(exp.Select):
        alias_map: Dict[str, TableRef] = {}
        real_refs: List[TableRef] = []
        for tbl in _immediate_tables(select):
            if not tbl.db and tbl.name.lower() in cte_names:
                continue  # a CTE reference, not a physical table
            ref = TableRef(db=tbl.db or None, name=tbl.name)
            occurrences[ref.qualified] = occurrences.get(ref.qualified, 0) + 1
            real_refs.append(ref)
            # Resolve a column qualifier by either the alias or the bare name.
            alias_map[tbl.alias_or_name.lower()] = ref
            alias_map[tbl.name.lower()] = ref

        where = select.args.get("where")
        if where is None:
            continue

        atoms: List[Tuple[Predicate, str]] = []
        _walk_and(where.this, atoms)
        for pred, qualifier in atoms:
            if qualifier:
                target = alias_map.get(qualifier.lower())
                if target is None:
                    continue  # qualified by a CTE/subquery/unknown alias — drop
            elif len(real_refs) == 1:
                target = real_refs[0]
            else:
                continue  # unqualified column with 0 or >1 tables — ambiguous
            table_predicates.setdefault(target.qualified, []).append(pred)

    # A table referenced more than once (self-join, or reused across scopes)
    # can't have its per-occurrence filters soundly combined, so we don't prune
    # it at all rather than risk ANDing contradictory values.
    for qualified, count in occurrences.items():
        if count > 1:
            table_predicates.pop(qualified, None)

    return table_predicates


def _walk_and(node: exp.Expression, atoms: List[Tuple[Predicate, str]]) -> None:
    """Collect AND-connected atoms, each paired with its column's qualifier.

    We deliberately only descend through ``AND`` (and parentheses). Anything
    under an ``OR`` is skipped, because pruning on one side of an OR would be
    unsound — skipping it just widens the estimate, which is the safe direction.
    """
    if isinstance(node, exp.Paren):
        _walk_and(node.this, atoms)
        return
    if isinstance(node, exp.And):
        _walk_and(node.left, atoms)
        _walk_and(node.right, atoms)
        return
    atom = _atom(node)
    if atom is not None:
        atoms.append(atom)


def _atom(node: exp.Expression) -> Optional[Tuple[Predicate, str]]:
    """Return ``(predicate, column_qualifier)`` for a pushable atom, else None.

    ``column_qualifier`` is the table/alias the column was qualified with
    (``''`` when unqualified), so the caller can attribute the predicate to the
    right table.
    """
    # column <op> literal
    for cls, op in _COMPARE.items():
        if isinstance(node, cls):
            col, lit = node.left, node.right
            if isinstance(col, exp.Column) and isinstance(lit, exp.Literal):
                return Predicate(col.name, op, [lit.this]), col.table
            # Literal on the left: flip it (5 < x  ->  x > 5)
            if isinstance(lit, exp.Column) and isinstance(col, exp.Literal):
                flipped = {">": "<", "<": ">", ">=": "<=", "<=": ">="}.get(op, op)
                return Predicate(lit.name, flipped, [col.this]), lit.table
            return None

    if isinstance(node, exp.In):
        col = node.this
        vals = [e.this for e in node.expressions if isinstance(e, exp.Literal)]
        if isinstance(col, exp.Column) and vals and len(vals) == len(node.expressions):
            return Predicate(col.name, "IN", vals), col.table
        return None

    if isinstance(node, exp.Between):
        col = node.this
        low, high = node.args.get("low"), node.args.get("high")
        if (
            isinstance(col, exp.Column)
            and isinstance(low, exp.Literal)
            and isinstance(high, exp.Literal)
        ):
            return Predicate(col.name, "BETWEEN", [low.this, high.this]), col.table
        return None

    return None
