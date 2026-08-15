# athena-cost-guard

[![PyPI](https://img.shields.io/pypi/v/athena-cost-guard.svg)](https://pypi.org/project/athena-cost-guard/)
[![Python](https://img.shields.io/pypi/pyversions/athena-cost-guard.svg)](https://pypi.org/project/athena-cost-guard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Estimate what an AWS Athena query will scan and cost — *before* you run it — and block queries that blow your budget.**

Available on PyPI: `pip install athena-cost-guard`

Athena bills by the volume of data scanned from S3 (~$5/TB). Unlike BigQuery,
it has no built-in dry-run, so it's easy to fire off one unpartitioned `SELECT *`
and scan a terabyte by accident. `athena-cost-guard` gives you the pre-flight
check Athena is missing.

```python
from athena_cost_guard import estimate

est = estimate(
    "SELECT id FROM billing.line_items WHERE dt = '2026-08'",
    region="us-east-1",
)
print(est.summary())
# tables:            billing.line_items
# partitions matched: 1
# bytes scanned:     ≤ 4.2 GB
# estimated cost:    ≤ $0.0210  (@ $5.00/TB)
```

Guardrail mode — refuse to run anything over budget:

```python
from athena_cost_guard import cost_guard, BudgetExceeded

@cost_guard(max_usd=1.00, region="us-east-1")
def run(sql):
    return athena.start_query_execution(QueryString=sql, ...)

try:
    run("SELECT * FROM billing.line_items")   # unpartitioned full scan
except BudgetExceeded as e:
    print(e)            # estimated cost $52.31 exceeds budget $1.00 (scanning 10.5 TB)
    print(e.estimate)   # the full Estimate for logging
```

## How it works

1. **Parse** the SQL with [`sqlglot`](https://github.com/tobymao/sqlglot)
   (Athena dialect) to find the tables, referenced columns, and WHERE
   predicates on partition keys.
2. **Prune** partitions by calling Glue `GetPartitions` with a pushdown
   expression built from those predicates — so you only pay attention to the
   partitions the query would actually touch.
3. **Size** the surviving partitions by summing their S3 object bytes.
4. **Price** the total using Athena's real billing rules (10 MB per-query
   minimum, rounded up to the nearest 10 MB, at your region's $/TB rate).

## Install

```bash
pip install athena-cost-guard
```

Requires AWS credentials with `glue:GetTable`, `glue:GetPartitions`, and
`s3:ListBucket` on the relevant tables/buckets (standard boto3 resolution:
env vars, shared config, or instance role).

## Accuracy: read this

Tier-1 estimates (the current release) are a deliberate **upper bound**: they
assume every column in the matched partitions is read. For columnar formats
(Parquet/ORC) a query that selects a few columns will scan **less** than this
number — so treat the estimate as "you will not scan more than X." That's the
safe direction for a budget guard.

Known limitations, all handled gracefully (never silently wrong):

| Situation | Behaviour |
|---|---|
| Column projection (Parquet/ORC selects fewer columns) | Not yet modelled → estimate is an upper bound |
| Partition projection tables (no Glue partitions) | Warns, sizes the table root (wide upper bound) |
| `OR` / function predicates on partition keys | Not pushed down → those partitions are included |
| Iceberg / row-group stats pruning | Not modelled → upper bound |

## Roadmap

- **0.2** — Tier-2 column-aware estimates via Parquet footer sampling (tighten
  columnar queries to a realistic figure, not just an upper bound).
- **0.3** — CLI (`athena-cost-guard "SELECT ..."`), partition-projection
  support, Iceberg awareness.

## Development

```bash
pip install -e ".[dev]"
pytest            # parser & pricing tests need no AWS; estimate tests use fakes
```

## License

MIT
