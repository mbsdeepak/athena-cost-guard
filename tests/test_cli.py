"""CLI tests. The CLI is a thin wrapper over estimate(), so these stub out
`estimate` (patching the name the cli module imported) and assert on argument
wiring, output formatting, and exit codes — no AWS, no boto3."""
import json

import pytest

from athena_cost_guard import Estimate, cli


def _fake_estimate(**overrides):
    """Return a (recorder, fn) pair; fn records its call and yields an Estimate."""
    calls = {}

    def fn(sql, **kwargs):
        calls["sql"] = sql
        calls["kwargs"] = kwargs
        return Estimate(
            bytes_scanned=overrides.get("bytes_scanned", 2 * 10 ** 12),
            cost_usd=overrides.get("cost_usd", 10.0),
            partitions_matched=1,
            tables=["billing.line_items"],
            is_upper_bound=True,
            pruned=True,
            region="us-east-1",
            price_per_tb=5.0,
        )

    return calls, fn


def test_prints_summary_and_exits_zero(monkeypatch, capsys):
    calls, fn = _fake_estimate()
    monkeypatch.setattr(cli, "estimate", fn)
    rc = cli.main(["SELECT id FROM billing.line_items WHERE dt = '2026-08'"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_OK
    assert "billing.line_items" in out
    assert "estimated cost" in out


def test_flags_are_forwarded_to_estimate(monkeypatch, capsys):
    calls, fn = _fake_estimate()
    monkeypatch.setattr(cli, "estimate", fn)
    cli.main(
        [
            "SELECT id FROM t WHERE dt = '2026-08'",
            "--database", "billing",
            "--region", "us-west-2",
            "--price-per-tb", "6.5",
            "--sample-columns",
            "--sample-size", "4",
        ]
    )
    kw = calls["kwargs"]
    assert kw["database"] == "billing"
    assert kw["region"] == "us-west-2"
    assert kw["price_per_tb"] == 6.5
    assert kw["sample_columns"] is True
    assert kw["sample_size"] == 4


def test_reads_sql_from_stdin(monkeypatch, capsys):
    calls, fn = _fake_estimate()
    monkeypatch.setattr(cli, "estimate", fn)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("SELECT 1 FROM t"))
    rc = cli.main([])  # no positional SQL -> read stdin
    assert rc == cli.EXIT_OK
    assert calls["sql"] == "SELECT 1 FROM t"


def test_json_output_is_parseable(monkeypatch, capsys):
    calls, fn = _fake_estimate(bytes_scanned=2 * 10 ** 12, cost_usd=10.0)
    monkeypatch.setattr(cli, "estimate", fn)
    rc = cli.main(["SELECT id FROM t", "-d", "billing", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == cli.EXIT_OK
    assert payload["cost_usd"] == 10.0
    assert payload["bytes_scanned"] == 2 * 10 ** 12
    assert payload["human_bytes"]  # derived field is included


def test_max_usd_under_budget_exits_zero(monkeypatch, capsys):
    calls, fn = _fake_estimate(cost_usd=0.5)
    monkeypatch.setattr(cli, "estimate", fn)
    rc = cli.main(["SELECT id FROM t", "-d", "billing", "--max-usd", "1.0"])
    assert rc == cli.EXIT_OK


def test_max_usd_over_budget_exits_one(monkeypatch, capsys):
    calls, fn = _fake_estimate(cost_usd=52.31)
    monkeypatch.setattr(cli, "estimate", fn)
    rc = cli.main(["SELECT id FROM t", "-d", "billing", "--max-usd", "1.0"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_OVER_BUDGET
    assert "BUDGET EXCEEDED" in err


def test_estimation_error_exits_two(monkeypatch, capsys):
    def boom(sql, **kwargs):
        raise ValueError("table 'x' is not schema-qualified")

    monkeypatch.setattr(cli, "estimate", boom)
    rc = cli.main(["SELECT id FROM x"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_ERROR
    assert "error:" in err
    assert "schema-qualified" in err


def test_empty_sql_is_usage_error(monkeypatch):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("   \n"))
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == cli.EXIT_ERROR  # argparse.error() exits 2
