"""Athena scan-cost pricing.

Athena bills for the volume of data scanned from S3. AWS applies two rules that
materially change the bill on small queries, and this module models both:

  * a per-query minimum of 10 MB, and
  * rounding *up* to the nearest 10 MB.

The per-TB rate and the definition of "TB" are exposed as overridable constants
because both vary (by region, and by AWS's SI-vs-binary conventions). The
defaults match AWS's published ``$5.00 per TB`` with a decimal terabyte.
"""
from __future__ import annotations

import math

# Most regions charge $5.00 per TB scanned. Callers can override per-region.
DEFAULT_PRICE_PER_TB: float = 5.0

# A handful of regions publish different rates; extend as needed. Anything not
# listed falls back to DEFAULT_PRICE_PER_TB.
PRICE_PER_TB = {
    "us-east-1": 5.0,
    "us-east-2": 5.0,
    "us-west-1": 5.0,
    "us-west-2": 5.0,
    "eu-west-1": 5.0,
    "eu-central-1": 5.0,
    "ap-south-1": 5.0,
    "ap-southeast-1": 5.0,
    "ap-southeast-2": 5.0,
    "ap-northeast-1": 5.0,
}

# AWS quotes storage/scan pricing in decimal terabytes (10^12 bytes).
BYTES_PER_TB: int = 10 ** 12

# Per-query minimum and rounding granularity (10 MB, decimal).
MIN_BYTES: int = 10 * 1000 * 1000
ROUND_BYTES: int = 10 * 1000 * 1000


def price_for_region(region: str | None) -> float:
    """Return the $/TB rate for *region*, falling back to the default."""
    if region is None:
        return DEFAULT_PRICE_PER_TB
    return PRICE_PER_TB.get(region, DEFAULT_PRICE_PER_TB)


def billable_bytes(raw_bytes: int) -> int:
    """Apply Athena's 10 MB minimum and round-up-to-10 MB rule."""
    b = max(int(raw_bytes), MIN_BYTES)
    return math.ceil(b / ROUND_BYTES) * ROUND_BYTES


def cost_usd(raw_bytes: int, price_per_tb: float = DEFAULT_PRICE_PER_TB) -> float:
    """Dollar cost of scanning *raw_bytes*, after Athena's billing adjustments."""
    return billable_bytes(raw_bytes) / BYTES_PER_TB * price_per_tb
