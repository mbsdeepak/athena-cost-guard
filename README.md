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

# for column-aware (Tier-2) estimates, add the parquet extra:
pip install "athena-cost-guard[parquet]"
```

Requires AWS credentials with `glue:GetTable`, `glue:GetPartitions`, and
`s3:ListBucket` on the relevant tables/buckets (standard boto3 resolution:
env vars, shared config, or instance role).

## Column-aware estimates (Tier-2)

By default (`sample_columns=False`) you get the **Tier-1 upper bound**: every
column in the matched partitions is assumed read. But Athena on Parquet only
scans the columns a query references, so pass `sample_columns=True` for a tight,
column-aware figure:

```python
est = estimate(
    "SELECT servicename, SUM(billedcost) FROM billing.line_items WHERE dt = '2026-08' GROUP BY 1",
    region="us-east-1",
    sample_columns=True,      # needs the [parquet] extra
)
print(est.summary())
# bytes scanned:     ~317.4 MB        <- was ≤ 16.1 GB as a Tier-1 upper bound
# estimated cost:    ~$0.0016
# column-aware:      2% of bytes referenced  (from 8 sampled Parquet footer(s))
```

How it works: it reads the **footers** of up to `sample_size` (default 8) of the
largest Parquet files in the matched partitions — via ranged S3 GETs, never
downloading whole files — sums the compressed size of the columns the query
references, and scales the byte total by that fraction. The result is an
*estimate* (`~`), not an upper bound (`≤`). Falls back to the Tier-1 upper bound
for `SELECT *`, non-Parquet data, or unreadable footers (with a warning — never
silently wrong).

### Proven against Athena's own numbers

On a real wide-table query selecting a handful of columns out of many, the
column-aware estimate tracked Athena's actual `DataScannedInBytes` closely:

| | Data scanned |
|---|---|
| Tier-1 upper bound | ≤ 16.1 GB |
| Tier-2 estimate (`sample_columns=True`) | ~317 MB |
| **Athena actual** (`DataScannedInBytes`) | **295 MB** |

Within ~8%, and on the conservative (slightly-over) side — the safe direction
for a budget guard.

## Accuracy: read this

| Situation | Behaviour |
|---|---|
| Column projection, `sample_columns=True` (Parquet) | **Modelled** — tight column-aware estimate from footer sampling |
| Column projection, default | Upper bound (all columns assumed read) |
| Partition projection tables (no Glue partitions) | Warns, sizes the table root (wide upper bound) |
| `OR` / function predicates on partition keys | Not pushed down → those partitions are included |
| Iceberg / row-group stats pruning | Not modelled → upper bound |

## Roadmap

- **0.2** — ✅ Tier-2 column-aware estimates via Parquet footer sampling.
- **0.3** — CLI (`athena-cost-guard "SELECT ..."`), time-window partition pruning
  (literal date bounds), partition-projection support, Iceberg awareness.

## Development

```bash
pip install -e ".[dev]"
pytest            # parser & pricing tests need no AWS; estimate tests use fakes
```

## License

MIT
