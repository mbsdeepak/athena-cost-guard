"""S3 object sizing.

Given a set of S3 prefixes (partition locations), enumerate the objects under
them and sum their bytes. ``list_objects`` exposes the individual keys so the
column-aware (Tier-2) path can sample Parquet footers; ``size_all`` keeps the
simple total used by Tier-1.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

# (bucket, key, size) for a single S3 object.
S3Object = Tuple[str, str, int]


def _split_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    without_scheme = uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


class S3Sizer:
    """Lists and sums S3 objects under prefixes, with a lazy boto3 client."""

    def __init__(self, client=None, region: Optional[str] = None):
        self._client = client
        self._region = region

    @property
    def client(self):
        if self._client is None:
            import boto3  # lazy; only needed for real AWS calls

            self._client = boto3.client("s3", region_name=self._region)
        return self._client

    def list_objects(self, uris: Iterable[str]) -> List[S3Object]:
        """Enumerate every object under the given prefixes (deduping prefixes)."""
        seen_prefix = set()
        out: List[S3Object] = []
        for uri in uris:
            prefix_key = uri.rstrip("/")
            if prefix_key in seen_prefix:
                continue
            seen_prefix.add(prefix_key)
            bucket, prefix = _split_s3_uri(uri)
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    # Skip the zero-byte "directory" placeholder keys.
                    if obj["Key"].endswith("/"):
                        continue
                    out.append((bucket, obj["Key"], obj["Size"]))
        return out

    def size_all(self, uris: Iterable[str]) -> int:
        """Sum sizes across many prefixes, de-duplicating identical locations."""
        return sum(size for _, _, size in self.list_objects(uris))
