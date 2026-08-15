"""Tier-2 Parquet-footer tests.

These need pyarrow (the [parquet] extra); skipped automatically if it's absent.
They write a real Parquet file and cross-check our footer reader + fraction math
against it, including through the S3 ranged-reader file-like.
"""
import io

import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from athena_cost_guard.parquet import (  # noqa: E402
    _S3File,
    read_column_compressed_sizes,
    referenced_fraction,
    sample_referenced_fraction,
)


def _write_parquet(path):
    n = 20000
    # A capitalised column name ("Amount") lets us verify case-insensitive
    # matching against a lowercased reference set.
    table = pa.table(
        {
            "id": pa.array(list(range(n))),
            "name": pa.array(["short"] * n),
            "Amount": pa.array([float(i) * 1.5 for i in range(n)]),
        }
    )
    pq.write_table(table, path, compression="snappy")
    return path


class _FakeS3:
    """boto3-shaped get_object serving byte ranges from an in-memory blob."""

    def __init__(self, data: bytes):
        self._data = data

    def get_object(self, Bucket, Key, Range):  # noqa: N803 (boto3 casing)
        span = Range.split("=", 1)[1]
        start, end = span.split("-")
        chunk = self._data[int(start) : int(end) + 1]
        return {"Body": io.BytesIO(chunk)}


def test_reads_per_column_sizes(tmp_path):
    p = _write_parquet(tmp_path / "f.parquet")
    with open(p, "rb") as f:
        sizes = read_column_compressed_sizes(f)
    assert set(sizes) == {"id", "name", "Amount"}
    assert all(v > 0 for v in sizes.values())


def test_referenced_fraction_matches_manual(tmp_path):
    p = _write_parquet(tmp_path / "f.parquet")
    with open(p, "rb") as f:
        sizes = read_column_compressed_sizes(f)
    total = sum(sizes.values())
    frac = referenced_fraction(sizes, {"id"})
    assert frac == pytest.approx(sizes["id"] / total)
    assert 0.0 < frac < 1.0  # one of several columns -> a strict fraction


def test_referenced_fraction_all_columns_is_one(tmp_path):
    p = _write_parquet(tmp_path / "f.parquet")
    with open(p, "rb") as f:
        sizes = read_column_compressed_sizes(f)
    # lowercase "amount" must match the "Amount" column path.
    assert referenced_fraction(sizes, {"id", "name", "amount"}) == 1.0


def test_referenced_fraction_none_when_empty():
    assert referenced_fraction({}, {"id"}) is None


def test_case_insensitive_path_matching(tmp_path):
    p = _write_parquet(tmp_path / "f.parquet")
    with open(p, "rb") as f:
        sizes = read_column_compressed_sizes(f)
    total = sum(sizes.values())
    # Reference set is lowercased (as estimate() passes it); path is "Amount".
    assert referenced_fraction(sizes, {"amount"}) == pytest.approx(
        sizes["Amount"] / total
    )


def test_s3_ranged_reader_end_to_end(tmp_path):
    # Exercises the real Tier-2 read path: footer parsed through _S3File over a
    # boto3-shaped client, never downloading the whole object up front.
    p = _write_parquet(tmp_path / "f.parquet")
    data = open(p, "rb").read()
    reader = _S3File(_FakeS3(data), "bucket", "key.parquet", len(data))
    sizes = read_column_compressed_sizes(reader)
    assert set(sizes) == {"id", "name", "Amount"}


def test_sample_referenced_fraction_aggregates(tmp_path):
    p = _write_parquet(tmp_path / "f.parquet")
    data = open(p, "rb").read()
    client = _FakeS3(data)
    objects = [("bucket", "a.parquet", len(data)), ("bucket", "b.parquet", len(data))]
    frac, n = sample_referenced_fraction(client, objects, {"id"}, sample_size=8)
    assert n == 2
    with open(p, "rb") as f:
        single = referenced_fraction(read_column_compressed_sizes(f), {"id"})
    assert frac == pytest.approx(single)  # identical files -> same fraction
