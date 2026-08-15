"""End-to-end estimate/guard tests using injected fakes — no AWS, no boto3."""
import pytest

import athena_cost_guard as acg
from athena_cost_guard import BudgetExceeded, Estimate, cost_guard, estimate
from athena_cost_guard.catalog import TableMeta


class FakeCatalog:
    """Duck-types GlueCatalog. Records the pushdown decision for assertions."""

    def __init__(self, meta: TableMeta, partition_locations):
        self._meta = meta
        self._partition_locations = partition_locations
        self.last_predicates = None

    def get_table(self, db, name):
        return self._meta

    def matching_partitions(self, table, predicates):
        self.last_predicates = predicates
        pushable = [p for p in predicates if p.column in table.partition_keys]
        # Fake pruning: if a `dt =` predicate is present, keep one partition.
        if pushable:
            locs = self._partition_locations[:1]
        else:
            locs = self._partition_locations
        return locs, len(locs), bool(pushable)


class FakeSizer:
    """Duck-types S3Sizer: one synthetic .parquet object per prefix."""

    def __init__(self, sizes, client=None):
        self._sizes = sizes  # {prefix_without_trailing_slash: bytes}
        self.client = client

    def list_objects(self, uris):
        seen = set()
        out = []
        for uri in uris:
            key = uri.rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            out.append(("bucket", key + "/part-0.parquet", self._sizes.get(key, 0)))
        return out

    def size_all(self, uris):
        return sum(size for _, _, size in self.list_objects(uris))


def _partitioned_meta():
    return TableMeta(
        db="billing",
        name="line_items",
        location="s3://bucket/line_items/",
        input_format="parquet",
        partition_keys=["dt"],
        projected=False,
    )


def test_pruned_estimate_only_sizes_matching_partition():
    catalog = FakeCatalog(
        _partitioned_meta(),
        ["s3://bucket/line_items/dt=2026-08/", "s3://bucket/line_items/dt=2026-07/"],
    )
    sizer = FakeSizer(
        {
            "s3://bucket/line_items/dt=2026-08": 2 * 10 ** 12,  # 2 TB
            "s3://bucket/line_items/dt=2026-07": 9 * 10 ** 12,
        }
    )
    est = estimate(
        "SELECT id FROM billing.line_items WHERE dt = '2026-08'",
        region="us-east-1",
        catalog=catalog,
        sizer=sizer,
    )
    assert isinstance(est, Estimate)
    assert est.partitions_matched == 1
    assert est.pruned is True
    assert est.bytes_scanned == 2 * 10 ** 12
    assert round(est.cost_usd, 4) == 10.0  # 2 TB * $5/TB
    assert est.is_upper_bound is True


def test_no_predicate_scans_all_partitions():
    catalog = FakeCatalog(
        _partitioned_meta(),
        ["s3://bucket/line_items/dt=2026-08/", "s3://bucket/line_items/dt=2026-07/"],
    )
    sizer = FakeSizer(
        {
            "s3://bucket/line_items/dt=2026-08": 1 * 10 ** 12,
            "s3://bucket/line_items/dt=2026-07": 3 * 10 ** 12,
        }
    )
    est = estimate(
        "SELECT id FROM billing.line_items",
        catalog=catalog,
        sizer=sizer,
    )
    assert est.pruned is False
    assert est.bytes_scanned == 4 * 10 ** 12


def test_select_star_adds_warning():
    catalog = FakeCatalog(_partitioned_meta(), ["s3://bucket/line_items/dt=2026-08/"])
    sizer = FakeSizer({"s3://bucket/line_items/dt=2026-08": 10 ** 9})
    est = estimate(
        "SELECT * FROM billing.line_items WHERE dt = '2026-08'",
        catalog=catalog,
        sizer=sizer,
    )
    assert any("all columns" in w for w in est.warnings)


def test_partition_projection_warns_and_widens():
    meta = TableMeta(
        db="billing",
        name="events",
        location="s3://bucket/events/",
        input_format="parquet",
        partition_keys=["dt"],
        projected=True,
    )
    catalog = FakeCatalog(meta, [])
    sizer = FakeSizer({"s3://bucket/events": 5 * 10 ** 11})
    est = estimate(
        "SELECT id FROM billing.events WHERE dt = '2026-08'",
        catalog=catalog,
        sizer=sizer,
    )
    assert any("partition projection" in w for w in est.warnings)
    assert est.bytes_scanned == 5 * 10 ** 11


def test_unqualified_table_without_default_database_raises():
    catalog = FakeCatalog(_partitioned_meta(), [])
    sizer = FakeSizer({})
    with pytest.raises(ValueError, match="not schema-qualified"):
        estimate("SELECT id FROM line_items", catalog=catalog, sizer=sizer)


def test_default_database_used_for_unqualified_table():
    catalog = FakeCatalog(_partitioned_meta(), ["s3://bucket/line_items/dt=2026-08/"])
    sizer = FakeSizer({"s3://bucket/line_items/dt=2026-08": 10 ** 9})
    est = estimate(
        "SELECT id FROM line_items WHERE dt = '2026-08'",
        database="billing",
        catalog=catalog,
        sizer=sizer,
    )
    assert est.tables == ["billing.line_items"]


def test_cost_guard_blocks_over_budget():
    catalog = FakeCatalog(_partitioned_meta(), ["s3://bucket/line_items/dt=2026-08/"])
    sizer = FakeSizer({"s3://bucket/line_items/dt=2026-08": 2 * 10 ** 12})  # ~$10

    @cost_guard(max_usd=1.00, catalog=catalog, sizer=sizer)
    def run(sql):
        return "ran"

    with pytest.raises(BudgetExceeded) as exc:
        run("SELECT id FROM billing.line_items WHERE dt = '2026-08'")
    assert exc.value.estimate.cost_usd > 1.00


def test_cost_guard_allows_under_budget():
    catalog = FakeCatalog(_partitioned_meta(), ["s3://bucket/line_items/dt=2026-08/"])
    sizer = FakeSizer({"s3://bucket/line_items/dt=2026-08": 10 ** 8})  # tiny

    @cost_guard(max_usd=1.00, catalog=catalog, sizer=sizer)
    def run(sql):
        return "ran"

    assert run("SELECT id FROM billing.line_items WHERE dt = '2026-08'") == "ran"


# --- Tier-2 (column-aware) -------------------------------------------------


def test_column_aware_scales_bytes_by_fraction(monkeypatch):
    catalog = FakeCatalog(_partitioned_meta(), ["s3://bucket/line_items/dt=2026-08/"])
    sizer = FakeSizer({"s3://bucket/line_items/dt=2026-08": 1 * 10 ** 12})  # 1 TB
    # Pretend the referenced columns occupy 25% of on-disk bytes (3 files read).
    monkeypatch.setattr(acg, "sample_referenced_fraction", lambda *a, **k: (0.25, 3))
    est = estimate(
        "SELECT id FROM billing.line_items WHERE dt = '2026-08'",
        sample_columns=True,
        catalog=catalog,
        sizer=sizer,
    )
    assert est.is_upper_bound is False
    assert est.bytes_scanned == 250 * 10 ** 9
    assert est.sampled_files == 3
    assert round(est.column_fraction, 3) == 0.25
    assert round(est.cost_usd, 4) == 1.25  # 0.25 TB * $5/TB


def test_column_aware_falls_back_when_no_footer(monkeypatch):
    catalog = FakeCatalog(_partitioned_meta(), ["s3://bucket/line_items/dt=2026-08/"])
    sizer = FakeSizer({"s3://bucket/line_items/dt=2026-08": 1 * 10 ** 12})
    monkeypatch.setattr(acg, "sample_referenced_fraction", lambda *a, **k: (None, 0))
    est = estimate(
        "SELECT id FROM billing.line_items WHERE dt = '2026-08'",
        sample_columns=True,
        catalog=catalog,
        sizer=sizer,
    )
    assert est.is_upper_bound is True
    assert est.bytes_scanned == 1 * 10 ** 12
    assert est.column_fraction is None
    assert any("column sampling requested" in w for w in est.warnings)


def test_select_star_skips_sampling(monkeypatch):
    catalog = FakeCatalog(_partitioned_meta(), ["s3://bucket/line_items/dt=2026-08/"])
    sizer = FakeSizer({"s3://bucket/line_items/dt=2026-08": 1 * 10 ** 12})
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        return (0.1, 1)

    monkeypatch.setattr(acg, "sample_referenced_fraction", spy)
    est = estimate(
        "SELECT * FROM billing.line_items WHERE dt = '2026-08'",
        sample_columns=True,
        catalog=catalog,
        sizer=sizer,
    )
    assert called["n"] == 0  # SELECT * can't narrow, so no footers are read
    assert est.is_upper_bound is True
    assert est.bytes_scanned == 1 * 10 ** 12
