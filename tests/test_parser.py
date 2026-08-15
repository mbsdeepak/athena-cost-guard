"""Parser tests — these need no AWS access and run anywhere sqlglot is installed."""
from athena_cost_guard.parser import parse_query


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
