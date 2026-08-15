"""Glue Data Catalog access: resolve tables and prune partitions.

This is where SQL predicates become an actual list of S3 locations to size.
The boto3 client is created lazily so the rest of the package (parsing,
pricing, tests) has no hard dependency on AWS credentials.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .parser import Predicate


@dataclass
class TableMeta:
    db: str
    name: str
    location: str
    input_format: str
    partition_keys: List[str]
    projected: bool  # True if the table uses Athena partition projection


def _to_glue_expr(pred: Predicate) -> str:
    """Render a pushable predicate into Glue ``GetPartitions`` filter syntax."""
    if pred.op == "IN":
        vals = ", ".join(f"'{v}'" for v in pred.values)
        return f"{pred.column} IN ({vals})"
    if pred.op == "BETWEEN":
        return f"{pred.column} BETWEEN '{pred.values[0]}' AND '{pred.values[1]}'"
    return f"{pred.column} {pred.op} '{pred.values[0]}'"


class GlueCatalog:
    """Thin wrapper over the Glue client with lazy construction."""

    def __init__(self, client=None, region: Optional[str] = None):
        self._client = client
        self._region = region

    @property
    def client(self):
        if self._client is None:
            import boto3  # imported lazily; only needed for real AWS calls

            self._client = boto3.client("glue", region_name=self._region)
        return self._client

    def get_table(self, db: str, name: str) -> TableMeta:
        table = self.client.get_table(DatabaseName=db, Name=name)["Table"]
        sd = table["StorageDescriptor"]
        params = table.get("Parameters", {}) or {}
        projected = str(params.get("projection.enabled", "false")).lower() == "true"
        return TableMeta(
            db=db,
            name=name,
            location=sd["Location"],
            input_format=sd.get("InputFormat", ""),
            partition_keys=[k["Name"] for k in table.get("PartitionKeys", [])],
            projected=projected,
        )

    def matching_partitions(
        self, table: TableMeta, predicates: List[Predicate]
    ) -> Tuple[List[str], int, bool]:
        """Return (s3_locations, count, pruned) for the surviving partitions.

        ``pruned`` reports whether any predicate was actually pushed down — when
        it's False the caller knows every partition was included and the
        estimate is the widest possible upper bound for this table.
        """
        pushable = [p for p in predicates if p.column in table.partition_keys]
        expression = " AND ".join(_to_glue_expr(p) for p in pushable) if pushable else None

        locations: List[str] = []
        paginator = self.client.get_paginator("get_partitions")
        kwargs = {"DatabaseName": table.db, "TableName": table.name}
        if expression:
            kwargs["Expression"] = expression
        for page in paginator.paginate(**kwargs):
            for part in page.get("Partitions", []):
                locations.append(part["StorageDescriptor"]["Location"])
        return locations, len(locations), bool(pushable)
