"""S3 object sizing.

Given a set of S3 prefixes (partition locations), sum the bytes stored under
them. This is the raw scanned-byte figure that pricing turns into dollars.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple


def _split_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    without_scheme = uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


class S3Sizer:
    """Sums object sizes under S3 prefixes, with a lazy boto3 client."""

    def __init__(self, client=None, region: Optional[str] = None):
        self._client = client
        self._region = region

    @property
    def client(self):
        if self._client is None:
            import boto3  # lazy; only needed for real AWS calls

            self._client = boto3.client("s3", region_name=self._region)
        return self._client

    def size_prefix(self, uri: str) -> int:
        bucket, prefix = _split_s3_uri(uri)
        total = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                # Skip the zero-byte "directory" placeholder keys.
                if obj["Key"].endswith("/"):
                    continue
                total += obj["Size"]
        return total

    def size_all(self, uris: Iterable[str]) -> int:
        """Sum sizes across many prefixes, de-duplicating identical locations."""
        seen = set()
        total = 0
        for uri in uris:
            key = uri.rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            total += self.size_prefix(uri)
        return total
