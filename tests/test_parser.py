"""Parser tests — these need no AWS access and run anywhere sqlglot is installed."""
from athena_cost_guard.parser import TableRef, parse_query


def _preds_for(parsed, db, name):
    ref = TableRef(db=db, name=name)
    return [(p.column, p.op, p.values) for p in parsed.predicates_for(ref)]


def test_extracts_qualified_table_and_columns():
    p = parse_query("SELECT a, b FROM db.tbl WHERE dt = '2026-08'")
    assert [(t.db, t.name) for t in p.tables] == [("db", "tbl")]
    assert {"a", "b", "dt"} <= p.columns
    assert p.select_star is False


def test_select_star_flagged():
    p = parse_query("SELECT * FROM db.tbl")
    assert p.select_star is True


def test_equality_predicate():
    p = parse_query("SELECT 1 FROM db.t WHERE dt = '2026-08'")
    assert len(p.predicates) == 1
    pred = p.predicates[0]
    assert (pred.column, pred.op, pred.values) == ("dt", "=", ["2026-08"])


def test_in_predicate():
    p = parse_query("SELECT 1 FROM db.t WHERE region IN ('us-east-1', 'eu-west-1')")
    pred = next(x for x in p.predicates if x.column == "region")
    assert pred.op == "IN"
    assert pred.values == ["us-east-1", "eu-west-1"]


def test_between_predicate():
    p = parse_query("SELECT 1 FROM db.t WHERE dt BETWEEN '2026-01' AND '2026-03'")
    pred = p.predicates[0]
    assert pred.op == "BETWEEN"
    assert pred.values == ["2026-01", "2026-03"]


def test_and_connected_predicates_all_captured():
    p = parse_query(
        "SELECT 1 FROM db.t WHERE dt = '2026-08' AND region = 'us-east-1'"
    )
    cols = {pred.column for pred in p.predicates}
    assert cols == {"dt", "region"}


def test_or_predicates_are_not_pushed_down():
    # OR must not produce a pushable predicate — skipping it widens the estimate,
    # which is the safe direction.
    p = parse_query("SELECT 1 FROM db.t WHERE dt = '2026-08' OR dt = '2026-09'")
    assert p.predicates == []


def test_flipped_literal_on_left():
    p = parse_query("SELECT 1 FROM db.t WHERE '2026-08' < dt")
    pred = p.predicates[0]
    assert (pred.column, pred.op, pred.values) == ("dt", ">", ["2026-08"])


def test_cte_names_not_treated_as_tables():
    p = parse_query(
        "WITH recent AS (SELECT * FROM db.raw) SELECT * FROM recent"
    )
    assert [t.name for t in p.tables] == ["raw"]


def test_cte_shadowing_real_table_keeps_qualified_table():
    # A CTE named the same as the real table it derives from must NOT cause the
    # schema-qualified table to be dropped — that would zero out the estimate.
    p = parse_query(
        "WITH line_items AS (SELECT id FROM billing.line_items WHERE dt = '2026-08') "
        "SELECT * FROM line_items"
    )
    assert [(t.db, t.name) for t in p.tables] == [("billing", "line_items")]


# --- per-table predicate attribution (issue #3) ----------------------------


def test_single_table_predicate_attributed_to_that_table():
    p = parse_query("SELECT a FROM db.t WHERE dt = '2026-08'")
    assert _preds_for(p, "db", "t") == [("dt", "=", ["2026-08"])]


def test_qualified_predicates_attributed_per_table_not_conflated():
    # The original bug: both dt predicates were applied to both tables, ANDing
    # to a contradiction that matched zero partitions.
    p = parse_query(
        "SELECT * FROM big.a x JOIN small.b y ON x.id = y.id "
        "WHERE x.dt = '2026-08' AND y.dt = '2026-01'"
    )
    assert _preds_for(p, "big", "a") == [("dt", "=", ["2026-08"])]
    assert _preds_for(p, "small", "b") == [("dt", "=", ["2026-01"])]


def test_predicates_collected_from_every_where_clause():
    # Previously only the first WHERE in the tree was read; the CTE's dt filter
    # was silently dropped.
    p = parse_query(
        "WITH c AS (SELECT * FROM db.raw WHERE dt = '2026-08') "
        "SELECT * FROM c JOIN db.other o ON true WHERE o.region = 'us'"
    )
    assert _preds_for(p, "db", "raw") == [("dt", "=", ["2026-08"])]
    assert _preds_for(p, "db", "other") == [("region", "=", ["us"])]


def test_unqualified_predicate_with_multiple_tables_is_dropped():
    # Ambiguous — we can't tell which table `dt` belongs to, so we drop it
    # (widening the estimate is the safe direction).
    p = parse_query("SELECT * FROM db.a, db.b WHERE dt = '2026-08'")
    assert _preds_for(p, "db", "a") == []
    assert _preds_for(p, "db", "b") == []


def test_self_join_predicates_are_dropped():
    # The same table under two aliases with conflicting filters can't be soundly
    # pruned, so no predicate is pushed.
    p = parse_query(
        "SELECT * FROM db.a x JOIN db.a y ON x.id = y.pid "
        "WHERE x.dt = '2026-08' AND y.dt = '2026-01'"
    )
    assert _preds_for(p, "db", "a") == []


def test_predicate_qualified_by_cte_alias_is_dropped():
    p = parse_query(
        "WITH c AS (SELECT id FROM db.raw) SELECT * FROM c WHERE c.id = '5'"
    )
    assert _preds_for(p, "db", "raw") == []
