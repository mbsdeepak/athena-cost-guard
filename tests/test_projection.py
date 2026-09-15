"""Unit tests for partition-projection pruning — all pure, no AWS."""
from athena_cost_guard.parser import Predicate
from athena_cost_guard.projection import (
    build_prefixes,
    enumerate_values,
    filter_values,
    parse_projection,
    projected_locations,
)

# A realistic multi-key projection config (integer, date and enum keys) used as a fixture.
SAMPLE_PARAMS = {
    "projection.enabled": "true",
    "projection.year.type": "integer",
    "projection.year.range": "2023,2026",
    "projection.month.type": "date",
    "projection.month.range": "2023-04-01,2026-09-01",
    "projection.month.format": "yyyy-MM-dd",
    "projection.month.interval": "1",
    "projection.month.interval.unit": "MONTHS",
    "projection.cloud.type": "enum",
    "projection.cloud.values": "ANTHROPIC,AWS,AZURE,GCP",
    "storage.location.template": (
        "s3://example-bucket/schema/org=acme/"
        "year=${year}/month=${month}/cloud=${cloud}/"
    ),
}
SAMPLE_KEYS = ["year", "month", "cloud"]


# --- parse_projection -------------------------------------------------------


def test_parse_disabled_returns_none():
    specs, template = parse_projection({"projection.enabled": "false"}, ["dt"])
    assert specs is None


def test_parse_sample_config():
    specs, template = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    assert set(specs) == {"year", "month", "cloud"}
    assert specs["year"].type == "integer"
    assert specs["month"].type == "date"
    assert specs["month"].interval_unit == "MONTHS"
    assert specs["cloud"].values == ["ANTHROPIC", "AWS", "AZURE", "GCP"]
    assert template.endswith("cloud=${cloud}/")


def test_parse_incomplete_key_bails():
    params = {"projection.enabled": "true", "projection.year.type": "integer"}
    specs, _ = parse_projection(params, ["year"])  # no range
    assert specs is None


# --- enumerate_values -------------------------------------------------------


def test_enumerate_enum():
    specs, _ = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    assert enumerate_values(specs["cloud"]) == ["ANTHROPIC", "AWS", "AZURE", "GCP"]


def test_enumerate_integer_range():
    specs, _ = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    assert enumerate_values(specs["year"]) == ["2023", "2024", "2025", "2026"]


def test_enumerate_integer_with_digits_and_interval():
    params = {
        "projection.enabled": "true",
        "projection.h.type": "integer",
        "projection.h.range": "0,20",
        "projection.h.interval": "5",
        "projection.h.digits": "2",
    }
    specs, _ = parse_projection(params, ["h"])
    assert enumerate_values(specs["h"]) == ["00", "05", "10", "15", "20"]


def test_enumerate_date_months():
    specs, _ = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    months = enumerate_values(specs["month"])
    assert months[0] == "2023-04-01"
    assert months[-1] == "2026-09-01"
    assert "2024-02-01" in months  # month stepping crosses years correctly


def test_enumerate_date_days():
    params = {
        "projection.enabled": "true",
        "projection.dt.type": "date",
        "projection.dt.range": "2026-01-01,2026-01-05",
        "projection.dt.format": "yyyy-MM-dd",
        "projection.dt.interval": "1",
        "projection.dt.interval.unit": "DAYS",
    }
    specs, _ = parse_projection(params, ["dt"])
    assert enumerate_values(specs["dt"]) == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-04",
        "2026-01-05",
    ]


def test_enumerate_date_now_bound_is_finite():
    params = {
        "projection.enabled": "true",
        "projection.dt.type": "date",
        "projection.dt.range": "NOW-3DAYS,NOW",
        "projection.dt.format": "yyyy-MM-dd",
        "projection.dt.interval": "1",
        "projection.dt.interval.unit": "DAYS",
    }
    specs, _ = parse_projection(params, ["dt"])
    vals = enumerate_values(specs["dt"])
    assert vals is not None and len(vals) == 4  # today plus the 3 prior days


# --- filter_values ----------------------------------------------------------


def test_filter_integer_equality_is_numeric():
    specs, _ = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    years = enumerate_values(specs["year"])
    kept = filter_values(years, specs["year"], Predicate("year", "=", ["2026"]))
    assert kept == ["2026"]


def test_filter_date_in_list():
    specs, _ = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    months = enumerate_values(specs["month"])
    kept = filter_values(
        months, specs["month"], Predicate("month", "IN", ["2026-09-01"])
    )
    assert kept == ["2026-09-01"]


def test_filter_date_range_between():
    specs, _ = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    months = enumerate_values(specs["month"])
    kept = filter_values(
        months,
        specs["month"],
        Predicate("month", "BETWEEN", ["2026-07-01", "2026-09-01"]),
    )
    assert kept == ["2026-07-01", "2026-08-01", "2026-09-01"]


def test_filter_integer_gte():
    specs, _ = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    years = enumerate_values(specs["year"])
    kept = filter_values(years, specs["year"], Predicate("year", ">=", ["2025"]))
    assert kept == ["2025", "2026"]


# --- build_prefixes ---------------------------------------------------------


def test_build_prefixes_template():
    _, template = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    chosen = {"year": ["2026"], "month": ["2026-09-01"], "cloud": ["AWS", "GCP"]}
    prefixes = build_prefixes(template, SAMPLE_KEYS, chosen, "s3://ignored/")
    assert prefixes == [
        "s3://example-bucket/schema/org=acme/year=2026/month=2026-09-01/cloud=AWS/",
        "s3://example-bucket/schema/org=acme/year=2026/month=2026-09-01/cloud=GCP/",
    ]


def test_build_prefixes_hive_fallback():
    chosen = {"dt": ["2026-09-01"]}
    prefixes = build_prefixes(None, ["dt"], chosen, "s3://bucket/tbl/")
    assert prefixes == ["s3://bucket/tbl/dt=2026-09-01/"]


# --- projected_locations (top-level) ---------------------------------------


def test_projected_locations_prunes_year_and_month():
    specs, template = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    preds = [Predicate("year", "=", ["2026"]), Predicate("month", "IN", ["2026-09-01"])]
    result = projected_locations(specs, template, SAMPLE_KEYS, "s3://x/", preds)
    assert result is not None
    locs, pruned = result
    assert pruned is True
    # 1 year x 1 month x 4 clouds
    assert len(locs) == 4
    assert all("year=2026/month=2026-09-01" in l for l in locs)


def test_projected_locations_none_when_nothing_prunes():
    specs, template = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    # A predicate only on a non-partition column narrows no key.
    preds = [Predicate("servicename", "=", ["AmazonEC2"])]
    assert projected_locations(specs, template, SAMPLE_KEYS, "s3://x/", preds) is None


def test_projected_locations_zero_match():
    specs, template = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    preds = [Predicate("year", "=", ["1999"])]  # outside the projected range
    result = projected_locations(specs, template, SAMPLE_KEYS, "s3://x/", preds)
    assert result == ([], True)


def test_projected_locations_cap_bails():
    specs, template = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    preds = [Predicate("year", "=", ["2026"])]  # prunes year but months*clouds large
    # A tiny cap forces the guard to trip -> None (fall back to root).
    result = projected_locations(
        specs, template, SAMPLE_KEYS, "s3://x/", preds, max_partitions=3
    )
    assert result is None


def test_projected_locations_injected_without_predicate_bails():
    params = {
        "projection.enabled": "true",
        "projection.tenant.type": "injected",
        "projection.year.type": "integer",
        "projection.year.range": "2023,2026",
    }
    keys = ["tenant", "year"]
    specs, template = parse_projection(params, keys)
    preds = [Predicate("year", "=", ["2026"])]  # tenant is injected, no predicate
    assert projected_locations(specs, template, keys, "s3://x/", preds) is None


def test_estimate_prunes_projected_table_end_to_end():
    """estimate() should route a projected table through projection pruning and
    size only the surviving prefixes, not the whole table root."""
    from athena_cost_guard import estimate
    from athena_cost_guard.catalog import GlueCatalog, TableMeta

    specs, template = parse_projection(SAMPLE_PARAMS, SAMPLE_KEYS)
    meta = TableMeta(
        db="billing",
        name="org_acme",
        location="s3://example-bucket/schema/org=acme",
        input_format="parquet",
        partition_keys=SAMPLE_KEYS,
        projected=True,
        projection=specs,
        location_template=template,
    )

    class ProjCatalog(GlueCatalog):
        def get_table(self, db, name):
            return meta

    class Sizer:
        client = None

        def list_objects(self, uris):
            # each pruned prefix contributes one 100 MB object
            return [("b", u.rstrip("/") + "/part.parquet", 100 * 10 ** 6) for u in uris]

    est = estimate(
        "SELECT resourceid FROM billing.org_acme "
        "WHERE year = 2026 AND month IN (DATE '2026-09-01')",
        catalog=ProjCatalog(),
        sizer=Sizer(),
    )
    # 1 year x 1 month x 4 clouds = 4 prefixes -> 4 * 100 MB, pruned=True
    assert est.pruned is True
    assert est.partitions_matched == 4
    assert est.bytes_scanned == 4 * 100 * 10 ** 6
    assert not any("wide upper bound" in w for w in est.warnings)


def test_projected_locations_injected_with_predicate():
    params = {
        "projection.enabled": "true",
        "projection.tenant.type": "injected",
        "projection.year.type": "integer",
        "projection.year.range": "2023,2026",
        "storage.location.template": "s3://b/tenant=${tenant}/year=${year}/",
    }
    keys = ["tenant", "year"]
    specs, template = parse_projection(params, keys)
    preds = [Predicate("tenant", "=", ["acme"]), Predicate("year", "=", ["2026"])]
    result = projected_locations(specs, template, keys, "s3://b/", preds)
    assert result is not None
    locs, pruned = result
    assert locs == ["s3://b/tenant=acme/year=2026/"]
