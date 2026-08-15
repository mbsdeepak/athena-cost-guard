"""athena-cost-guard — estimate what an AWS Athena query will scan and cost,
*before* you run it, and optionally block queries over a budget.

Quick start
-----------
    from athena_cost_guard import estimate

    est = estimate(
        "SELECT id FROM billing.line_items WHERE dt = '2026-08'",
        region="us-east-1",
    )
    print(est.human_bytes, est.cost_usd)

    from athena_cost_guard import cost_guard, BudgetExceeded

    @cost_guard(max_usd=1.00, region="us-east-1")
    def run(sql):
        return athena_client.start_query_execution(...)

Tier-1 estimates are an honest **upper bound**: they assume every column in the
matched partitions is read. Column-aware refinement (Parquet footers) lands in a
later release; see the README roadmap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, List, Optional, Union

from .catalog import GlueCatalog, TableMeta
from .parquet import sample_referenced_fraction
from .parser import ParsedQuery, Predicate, TableRef, parse_query
from .pricing import DEFAULT_PRICE_PER_TB, billable_bytes, cost_usd, price_for_region
from .sizing import S3Sizer

__version__ = "0.2.1"

__all__ = [
    "estimate",
    "cost_guard",
    "Estimate",
    "BudgetExceeded",
    "GlueCatalog",
    "S3Sizer",
]


def _human_bytes(n: int) -> str:
    step = 1000.0
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(n)
    for unit in units:
        if size < step:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= step
    return f"{size:.1f} EB"


@dataclass
class Estimate:
    """The result of :func:`estimate`."""

    bytes_scanned: int
    cost_usd: float
    partitions_matched: int
    tables: List[str]
    is_upper_bound: bool
    pruned: bool
    region: Optional[str]
    price_per_tb: float
    warnings: List[str] = field(default_factory=list)
    # Tier-2 (column-aware) details; defaults keep Tier-1 results unchanged.
    sampled_files: int = 0
    column_fraction: Optional[float] = None

    @property
    def human_bytes(self) -> str:
        return _human_bytes(self.bytes_scanned)

    def summary(self) -> str:
        bound = "≤ " if self.is_upper_bound else "~"
        lines = [
            f"tables:            {', '.join(self.tables)}",
            f"partitions matched: {self.partitions_matched}"
            + ("" if self.pruned else "  (no pruning applied)"),
            f"bytes scanned:     {bound}{self.human_bytes}",
            f"estimated cost:    {bound}${self.cost_usd:.4f}"
            f"  (@ ${self.price_per_tb:.2f}/TB)",
        ]
        if self.column_fraction is not None:
            lines.append(
                f"column-aware:      {self.column_fraction:.0%} of bytes referenced"
                f"  (from {self.sampled_files} sampled Parquet footer(s))"
            )
        for w in self.warnings:
            lines.append(f"warning:           {w}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Estimate(cost_usd={self.cost_usd:.4f}, "
            f"bytes_scanned={self.bytes_scanned}, "
            f"partitions_matched={self.partitions_matched}, "
            f"is_upper_bound={self.is_upper_bound})"
        )


class BudgetExceeded(Exception):
    """Raised by :func:`cost_guard` when an estimate exceeds the budget."""

    def __init__(self, estimate: Estimate, max_usd: float):
        self.estimate = estimate
        self.max_usd = max_usd
        super().__init__(
            f"estimated cost ${estimate.cost_usd:.4f} exceeds budget "
            f"${max_usd:.4f} (scanning {estimate.human_bytes})"
        )


def estimate(
    sql: str,
    *,
    database: Optional[str] = None,
    region: Optional[str] = None,
    price_per_tb: Optional[float] = None,
    sample_columns: bool = False,
    sample_size: int = 8,
    glue_client=None,
    s3_client=None,
    catalog: Optional[GlueCatalog] = None,
    sizer: Optional[S3Sizer] = None,
    dialect: str = "athena",
) -> Estimate:
    """Estimate the bytes scanned and dollar cost of *sql*.

    ``database`` supplies a default schema for tables that aren't qualified in
    the SQL. ``catalog`` / ``sizer`` may be injected (used in tests or to reuse
    configured clients); otherwise they're built from ``glue_client`` /
    ``s3_client`` or default boto3 clients.

    Tier-1 (the default) returns an upper bound: all columns in the matched
    partitions are assumed read. Pass ``sample_columns=True`` for the Tier-2,
    column-aware estimate — it reads the footers of up to ``sample_size`` Parquet
    files per table and scales the byte count by the fraction the query's
    referenced columns occupy. Tier-2 requires the optional ``pyarrow`` extra
    (``pip install "athena-cost-guard[parquet]"``).
    """
    parsed = parse_query(sql, dialect=dialect)
    catalog = catalog or GlueCatalog(glue_client, region)
    sizer = sizer or S3Sizer(s3_client, region)

    referenced_lower = {c.lower() for c in parsed.columns}
    can_narrow = sample_columns and not parsed.select_star and bool(referenced_lower)

    warnings: List[str] = []
    total_bytes = 0  # raw Tier-1 bytes (upper bound)
    effective_bytes = 0  # after column-aware scaling, where it applied
    partitions_matched = 0
    pruned_any = False
    used_sampling = False
    sampled_files = 0
    table_labels: List[str] = []

    for ref in parsed.tables:
        db = ref.db or database
        if db is None:
            raise ValueError(
                f"table {ref.name!r} is not schema-qualified and no "
                f"`database=` default was given"
            )
        table_labels.append(f"{db}.{ref.name}")
        meta = catalog.get_table(db, ref.name)

        if meta.projected:
            warnings.append(
                f"{ref.name}: uses partition projection — Glue has no partition "
                f"metadata, so pruning is skipped and the table root is sized "
                f"(wide upper bound)."
            )
            locations = [meta.location]
            pruned = False
        elif meta.partition_keys:
            locations, count, pruned = catalog.matching_partitions(
                meta, parsed.predicates_for(ref)
            )
            partitions_matched += count
        else:
            locations = [meta.location]
            pruned = False

        pruned_any = pruned_any or pruned

        objects = sizer.list_objects(locations)
        table_total = sum(size for _, _, size in objects)
        total_bytes += table_total
        table_effective = table_total

        if can_narrow:
            parquet_objs = [o for o in objects if o[1].lower().endswith(".parquet")]
            if parquet_objs:
                fraction, n = sample_referenced_fraction(
                    sizer.client, parquet_objs, referenced_lower, sample_size
                )
                if fraction is not None:
                    table_effective = round(table_total * fraction)
                    sampled_files += n
                    used_sampling = True

        effective_bytes += table_effective

    if parsed.select_star:
        warnings.append(
            "query selects all columns; column-level narrowing cannot reduce "
            "this estimate."
        )
    elif sample_columns and not used_sampling:
        warnings.append(
            "column sampling requested but not applied (no Parquet files or no "
            "readable footers); returning the Tier-1 upper bound."
        )

    if used_sampling:
        final_bytes = effective_bytes
        column_fraction = (effective_bytes / total_bytes) if total_bytes else None
    else:
        final_bytes = total_bytes
        column_fraction = None

    rate = price_per_tb if price_per_tb is not None else price_for_region(region)

    return Estimate(
        bytes_scanned=final_bytes,
        cost_usd=cost_usd(final_bytes, rate),
        partitions_matched=partitions_matched,
        tables=table_labels,
        is_upper_bound=not used_sampling,
        pruned=pruned_any,
        region=region,
        price_per_tb=rate,
        warnings=warnings,
        sampled_files=sampled_files,
        column_fraction=column_fraction,
    )


def cost_guard(
    max_usd: float,
    *,
    sql_arg: Union[int, str] = 0,
    on_estimate: Optional[Callable[[Estimate], None]] = None,
    **estimate_kwargs,
) -> Callable:
    """Decorator that estimates a query's cost and blocks it if over budget.

    The wrapped function's SQL argument is located by ``sql_arg`` — either a
    positional index (default: the first argument) or a keyword name. If the
    estimate exceeds ``max_usd`` a :class:`BudgetExceeded` is raised *before*
    the function runs. ``on_estimate`` receives every estimate (e.g. to log it).
    Remaining keyword arguments are forwarded to :func:`estimate`.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            sql = _resolve_sql(sql_arg, args, kwargs)
            est = estimate(sql, **estimate_kwargs)
            if on_estimate is not None:
                on_estimate(est)
            if est.cost_usd > max_usd:
                raise BudgetExceeded(est, max_usd)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def _resolve_sql(sql_arg: Union[int, str], args, kwargs) -> str:
    if isinstance(sql_arg, int):
        try:
            return args[sql_arg]
        except IndexError:
            raise TypeError(
                f"cost_guard expected the SQL at positional arg {sql_arg}, "
                f"but only {len(args)} positional args were passed"
            )
    try:
        return kwargs[sql_arg]
    except KeyError:
        raise TypeError(f"cost_guard expected a keyword argument {sql_arg!r} holding the SQL")
