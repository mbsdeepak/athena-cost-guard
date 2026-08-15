"""Column-aware (Tier-2) sizing from Parquet footers.

Athena on Parquet only scans the *columns a query references*, so a query that
selects a few of many columns scans far less than the whole file. To estimate
that, we read the Parquet **footer** — which carries per-column compressed sizes
— for a sample of files, and compute what fraction of on-disk bytes the
referenced columns account for.

Footers are read through :class:`_S3File`, a seekable file-like backed by ranged
S3 GETs, so only the footer bytes are fetched — never whole files. Parsing the
footer needs ``pyarrow``, an optional dependency: install with
``pip install "athena-cost-guard[parquet]"``.
"""
from __future__ import annotations

import io
from typing import Dict, List, Optional, Set, Tuple

from .sizing import S3Object


def _import_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa: WPS433 (optional dependency)

        return pq
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise ImportError(
            "column-aware estimates require pyarrow. Install it with:\n"
            '    pip install "athena-cost-guard[parquet]"'
        ) from exc


class _S3File(io.RawIOBase):
    """Minimal seekable, read-only file over an S3 object via ranged GETs.

    ``size`` is supplied up front (we already have it from listing), so no HEAD
    request is needed. pyarrow uses ``seek``/``read`` to pull just the footer.
    """

    def __init__(self, client, bucket: str, key: str, size: int):
        self._client = client
        self._bucket = bucket
        self._key = key
        self._size = size
        self._pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:  # pragma: no cover - defensive
            raise ValueError(f"invalid whence: {whence}")
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        end = self._size if size is None or size < 0 else min(self._pos + size, self._size)
        if end <= self._pos:
            return b""
        rng = f"bytes={self._pos}-{end - 1}"
        body = self._client.get_object(Bucket=self._bucket, Key=self._key, Range=rng)
        data = body["Body"].read()
        self._pos += len(data)
        return data


def read_column_compressed_sizes(fileobj) -> Dict[str, int]:
    """Map ``path_in_schema`` -> total compressed bytes, summed over row groups."""
    pq = _import_pyarrow()
    metadata = pq.read_metadata(fileobj)
    sizes: Dict[str, int] = {}
    for rg in range(metadata.num_row_groups):
        row_group = metadata.row_group(rg)
        for c in range(row_group.num_columns):
            column = row_group.column(c)
            path = column.path_in_schema
            sizes[path] = sizes.get(path, 0) + column.total_compressed_size
    return sizes


def referenced_fraction(
    path_sizes: Dict[str, int], referenced_lower: Set[str]
) -> Optional[float]:
    """Fraction of compressed bytes belonging to referenced columns.

    A column path is "referenced" when its top-level name (the part before the
    first ``.``, for nested columns) is in ``referenced_lower``. Returns None if
    there are no bytes to reason about.
    """
    referenced = 0
    total = 0
    for path, csize in path_sizes.items():
        total += csize
        if path.split(".")[0].lower() in referenced_lower:
            referenced += csize
    return (referenced / total) if total else None


def sample_referenced_fraction(
    s3_client,
    objects: List[S3Object],
    referenced_lower: Set[str],
    sample_size: int,
) -> Tuple[Optional[float], int]:
    """Estimate the referenced-column byte fraction from a sample of Parquet files.

    Samples the ``sample_size`` largest files (they dominate the bytes), reads
    each footer, aggregates per-column compressed sizes across the sample, and
    returns ``(fraction, files_read)``. Files that fail to parse are skipped;
    ``(None, 0)`` means no usable footer was read.
    """
    if not objects:
        return None, 0
    chosen = sorted(objects, key=lambda o: o[2], reverse=True)[: max(1, sample_size)]
    aggregate: Dict[str, int] = {}
    files_read = 0
    for bucket, key, size in chosen:
        try:
            sizes = read_column_compressed_sizes(_S3File(s3_client, bucket, key, size))
        except Exception:  # best-effort: a bad footer must not sink the estimate
            continue
        if not sizes:
            continue
        for path, csize in sizes.items():
            aggregate[path] = aggregate.get(path, 0) + csize
        files_read += 1
    if files_read == 0:
        return None, 0
    return referenced_fraction(aggregate, referenced_lower), files_read
