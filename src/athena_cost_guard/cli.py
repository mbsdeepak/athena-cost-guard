"""Command-line interface for athena-cost-guard.

    athena-cost-guard "SELECT id FROM billing.line_items WHERE dt = '2026-08'" \
        --region us-east-1

A thin wrapper over :func:`athena_cost_guard.estimate`. It adds no estimation
logic of its own — it parses flags, calls ``estimate()``, and prints the result.

Exit codes:
    0  estimate produced (and, if ``--max-usd`` given, within budget)
    1  ``--max-usd`` given and the estimate exceeds it (budget guard for shells)
    2  usage error, or the estimate could not be produced
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import Optional, Sequence

from . import Estimate, __version__, estimate

EXIT_OK = 0
EXIT_OVER_BUDGET = 1
EXIT_ERROR = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="athena-cost-guard",
        description=(
            "Estimate what an AWS Athena query will scan and cost before you "
            "run it."
        ),
    )
    parser.add_argument(
        "sql",
        nargs="?",
        help="the SQL query to estimate; use '-' or omit to read from stdin",
    )
    parser.add_argument(
        "-d",
        "--database",
        default=None,
        help="default schema for tables that aren't schema-qualified in the SQL",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (defaults to standard boto3 resolution)",
    )
    parser.add_argument(
        "--price-per-tb",
        type=float,
        default=None,
        metavar="USD",
        help="override the $/TB rate (defaults to the region's Athena price)",
    )
    parser.add_argument(
        "--sample-columns",
        action="store_true",
        help="Tier-2 column-aware estimate via Parquet footer sampling "
        "(needs the [parquet] extra)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=8,
        metavar="N",
        help="max Parquet files to sample per table for --sample-columns (default: 8)",
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        default=None,
        metavar="USD",
        help="exit non-zero (1) if the estimate exceeds this budget",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the estimate as JSON instead of the human summary",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _read_sql(arg: Optional[str]) -> str:
    if arg is None or arg == "-":
        return sys.stdin.read()
    return arg


def _as_json(est: Estimate) -> str:
    payload = dataclasses.asdict(est)
    payload["human_bytes"] = est.human_bytes
    return json.dumps(payload, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    sql = _read_sql(args.sql).strip()
    if not sql:
        parser.error("no SQL provided (pass it as an argument or on stdin)")

    estimate_kwargs = dict(
        database=args.database,
        region=args.region,
        price_per_tb=args.price_per_tb,
        sample_columns=args.sample_columns,
        sample_size=args.sample_size,
    )

    try:
        est = estimate(sql, **estimate_kwargs)
    except Exception as exc:  # surface any estimation failure as a clean CLI error
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        print(_as_json(est))
    else:
        print(est.summary())

    if args.max_usd is not None and est.cost_usd > args.max_usd:
        print(
            f"\nBUDGET EXCEEDED: ${est.cost_usd:.4f} > ${args.max_usd:.4f}",
            file=sys.stderr,
        )
        return EXIT_OVER_BUDGET

    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
