"""Athena partition-projection pruning.

Projection tables keep *no* partitions in Glue. Instead the table properties
declare, per partition key, how to **generate** the set of valid values —
``projection.<col>.type`` is ``enum`` / ``integer`` / ``date`` / ``injected``,
with ``values`` / ``range`` / ``format`` / ``interval`` / ``digits`` alongside —
and a ``storage.location.template`` maps a value-tuple to an S3 prefix.

This module turns those properties plus the query's predicates into the concrete
list of S3 prefixes the query can touch, so a projected table can be pruned like
an ordinary partitioned one instead of falling back to sizing the whole table.

Everything here is pure — no AWS. It is deliberately conservative: if the config
can't be enumerated soundly (an ``injected`` or unbounded key with no equality
predicate, an unparseable range, or a cross-product larger than
``MAX_PARTITIONS``) it returns ``None`` so the caller sizes the table root — a
wide upper bound, never a confidently-wrong number. It also returns ``None`` when
*nothing* prunes, because sizing the root once is cheaper than, and identical to,
enumerating every projected prefix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from typing import Dict, List, Optional, Tuple

from .parser import Predicate

# Safety cap on the enumerated cross-product. Beyond this we bail to root-sizing
# rather than issue tens of thousands of S3 listings.
MAX_PARTITIONS = 20_000


@dataclass
class ProjectionSpec:
    """Parsed ``projection.<col>.*`` properties for one partition key."""

    column: str
    type: str  # 'enum' | 'integer' | 'date' | 'injected'
    values: Optional[List[str]] = None  # enum
    low: Optional[str] = None  # integer/date range lower bound
    high: Optional[str] = None  # integer/date range upper bound
    interval: int = 1  # integer/date step
    interval_unit: str = "DAYS"  # date step unit
    date_format: str = "yyyy-MM-dd"  # date value format
    digits: Optional[int] = None  # integer zero-pad width


# --- reading the Glue table properties -------------------------------------


def parse_projection(
    params: Dict[str, str], partition_keys: List[str]
) -> Tuple[Optional[Dict[str, ProjectionSpec]], Optional[str]]:
    """Read projection specs + the location template from table ``params``.

    Returns ``(specs, template)``. ``specs`` is ``None`` when projection is not
    enabled or any key's config is missing/unrecognised — the signal to fall
    back to root-sizing.
    """
    if str(params.get("projection.enabled", "false")).lower() != "true":
        return None, None

    specs: Dict[str, ProjectionSpec] = {}
    for col in partition_keys:
        ptype = params.get(f"projection.{col}.type")
        if ptype is None:
            return None, None
        ptype = ptype.strip().lower()
        spec = ProjectionSpec(column=col, type=ptype)

        if ptype == "enum":
            raw = params.get(f"projection.{col}.values")
            if not raw:
                return None, None
            spec.values = [v.strip() for v in raw.split(",")]
        elif ptype == "integer":
            rng = _split_range(params.get(f"projection.{col}.range"))
            if rng is None:
                return None, None
            spec.low, spec.high = rng
            spec.interval = _int_or(params.get(f"projection.{col}.interval"), 1)
            d = params.get(f"projection.{col}.digits")
            spec.digits = int(d) if d not in (None, "") else None
        elif ptype == "date":
            rng = _split_range(params.get(f"projection.{col}.range"))
            if rng is None:
                return None, None
            spec.low, spec.high = rng
            spec.date_format = params.get(f"projection.{col}.format") or "yyyy-MM-dd"
            spec.interval = _int_or(params.get(f"projection.{col}.interval"), 1)
            spec.interval_unit = (
                params.get(f"projection.{col}.interval.unit") or "DAYS"
            ).strip().upper()
        elif ptype == "injected":
            pass  # values must come from an equality/IN predicate
        else:
            return None, None

        specs[col] = spec

    return specs, params.get("storage.location.template")


def _split_range(raw: Optional[str]) -> Optional[Tuple[str, str]]:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _int_or(raw: Optional[str], default: int) -> int:
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# --- enumerating a key's projected values ----------------------------------


def enumerate_values(spec: ProjectionSpec) -> Optional[List[str]]:
    """All projected values for ``spec``, or ``None`` if it can't be enumerated
    (an ``injected`` key, or an unparseable/unbounded range)."""
    if spec.type == "enum":
        return list(spec.values or [])
    if spec.type == "integer":
        return _enumerate_integer(spec)
    if spec.type == "date":
        return _enumerate_date(spec)
    return None  # injected


def _enumerate_integer(spec: ProjectionSpec) -> Optional[List[str]]:
    try:
        lo, hi = int(spec.low), int(spec.high)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    step = spec.interval or 1
    if step <= 0 or hi < lo:
        return None
    out: List[str] = []
    v = lo
    while v <= hi:
        out.append(str(v).zfill(spec.digits) if spec.digits else str(v))
        v += step
        if len(out) > MAX_PARTITIONS:
            return None
    return out


def _enumerate_date(spec: ProjectionSpec) -> Optional[List[str]]:
    fmt = _java_to_strftime(spec.date_format)
    lo = _parse_date_bound(spec.low, fmt)
    hi = _parse_date_bound(spec.high, fmt)
    if lo is None or hi is None or hi < lo or (spec.interval or 1) <= 0:
        return None
    out: List[str] = []
    cur = lo
    while cur <= hi:
        out.append(cur.strftime(fmt))
        cur = _add_interval(cur, spec.interval or 1, spec.interval_unit)
        if len(out) > MAX_PARTITIONS:
            return None
    return out


# Athena/Java date pattern -> strftime. Longest tokens first so 'yyyy' wins
# over 'yy'. Only the tokens that appear in real projection formats are mapped.
_DATE_TOKENS = [
    ("yyyy", "%Y"),
    ("yy", "%y"),
    ("MM", "%m"),
    ("dd", "%d"),
    ("HH", "%H"),
    ("mm", "%M"),
    ("ss", "%S"),
]


def _java_to_strftime(fmt: str) -> str:
    out = fmt
    for java, c in _DATE_TOKENS:
        out = out.replace(java, c)
    return out


def _parse_date_bound(raw: str, fmt: str) -> Optional[datetime]:
    """Parse a projection date bound. Supports a literal date and the ``NOW``
    keyword with an optional ``±<n><UNIT>`` offset (e.g. ``NOW-3YEARS``)."""
    token = raw.strip()
    if token.upper().startswith("NOW"):
        base = datetime.now(timezone.utc).replace(tzinfo=None)
        rest = token[3:].strip()
        if not rest:
            return base
        try:
            sign = 1 if rest[0] == "+" else -1
            body = rest[1:]
            num = ""
            i = 0
            while i < len(body) and body[i].isdigit():
                num += body[i]
                i += 1
            unit = body[i:].strip().upper()
            return _add_interval(base, sign * int(num), unit)
        except (ValueError, IndexError):
            return None
    try:
        return datetime.strptime(token, fmt)
    except ValueError:
        return None


def _add_interval(dt: datetime, n: int, unit: str) -> datetime:
    unit = unit.upper().rstrip("S")  # YEARS -> YEAR
    if unit == "YEAR":
        return dt.replace(year=dt.year + n)
    if unit == "MONTH":
        total = dt.month - 1 + n
        year = dt.year + total // 12
        month = total % 12 + 1
        day = min(dt.day, _days_in_month(year, month))
        return dt.replace(year=year, month=month, day=day)
    from datetime import timedelta

    deltas = {
        "WEEK": timedelta(weeks=n),
        "DAY": timedelta(days=n),
        "HOUR": timedelta(hours=n),
        "MINUTE": timedelta(minutes=n),
        "SECOND": timedelta(seconds=n),
    }
    return dt + deltas.get(unit, timedelta(days=n))


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = datetime(year + 1, 1, 1)
    else:
        nxt = datetime(year, month + 1, 1)
    from datetime import timedelta

    return (nxt - timedelta(days=1)).day


# --- applying predicates ----------------------------------------------------


def filter_values(
    values: List[str], spec: ProjectionSpec, pred: Predicate
) -> List[str]:
    """Return the subset of ``values`` satisfying ``pred``. Comparisons use the
    key's type (numeric for integer, chronological for date, else string)."""
    key = _comparer(spec)

    if pred.op == "IN":
        wanted = {key(v) for v in pred.values}
        return [v for v in values if key(v) in wanted]
    if pred.op == "=":
        target = key(pred.values[0])
        return [v for v in values if key(v) == target]
    if pred.op == "!=":
        target = key(pred.values[0])
        return [v for v in values if key(v) != target]
    if pred.op == "BETWEEN":
        lo, hi = key(pred.values[0]), key(pred.values[1])
        return [v for v in values if lo <= key(v) <= hi]
    if pred.op in (">", "<", ">=", "<="):
        target = key(pred.values[0])
        ops = {
            ">": lambda a: a > target,
            "<": lambda a: a < target,
            ">=": lambda a: a >= target,
            "<=": lambda a: a <= target,
        }
        keep = ops[pred.op]
        return [v for v in values if keep(key(v))]
    return values  # unknown op: don't prune (safe)


def _comparer(spec: ProjectionSpec):
    """A function mapping a value string to a comparable key for this spec's
    type. Falls back to string comparison if a value doesn't parse."""
    if spec.type == "integer":

        def as_int(v: str):
            try:
                return (0, int(v))
            except (TypeError, ValueError):
                return (1, str(v))

        return as_int
    if spec.type == "date":
        fmt = _java_to_strftime(spec.date_format)

        def as_date(v: str):
            try:
                return (0, datetime.strptime(v.strip(), fmt))
            except (TypeError, ValueError):
                # A predicate may carry a finer format (a timestamp) than the
                # partition; compare on the leading date portion if possible.
                try:
                    return (0, datetime.strptime(v.strip()[: len(_sample(fmt))], fmt))
                except (TypeError, ValueError):
                    return (1, str(v))

        return as_date
    return lambda v: str(v)


def _sample(fmt: str) -> str:
    return datetime(2000, 1, 1).strftime(fmt)


# --- building S3 prefixes ---------------------------------------------------


def build_prefixes(
    template: Optional[str],
    partition_keys: List[str],
    chosen: Dict[str, List[str]],
    base_location: str,
) -> List[str]:
    """Cartesian product of ``chosen`` values into S3 prefixes.

    Uses ``storage.location.template`` (``${col}`` substitution) when present,
    otherwise Hive-style ``<base>/<col>=<val>/…`` paths.
    """
    ordered = [chosen[k] for k in partition_keys]
    prefixes: List[str] = []
    for combo in product(*ordered):
        assignment = dict(zip(partition_keys, combo))
        if template:
            path = template
            for col, val in assignment.items():
                path = path.replace("${" + col + "}", val)
        else:
            base = base_location.rstrip("/")
            segs = "/".join(f"{col}={assignment[col]}" for col in partition_keys)
            path = f"{base}/{segs}/"
        prefixes.append(path)
    return prefixes


# --- top-level entry point --------------------------------------------------


def projected_locations(
    specs: Dict[str, ProjectionSpec],
    template: Optional[str],
    partition_keys: List[str],
    base_location: str,
    predicates: List[Predicate],
    max_partitions: int = MAX_PARTITIONS,
) -> Optional[Tuple[List[str], bool]]:
    """Prune a projected table to the S3 prefixes a query can touch.

    Returns ``(prefixes, pruned)`` — ``pruned`` is always ``True`` here — or
    ``None`` to signal the caller should size the table root instead. ``None`` is
    returned when a key can't be enumerated and no predicate supplies its values,
    when the cross-product exceeds ``max_partitions``, or when *no* predicate
    narrows any key (root-sizing is then cheaper and identical). ``([], True)``
    means the predicates provably match zero partitions.
    """
    pred_by_col: Dict[str, List[Predicate]] = {}
    for p in predicates:
        pred_by_col.setdefault(p.column, []).append(p)

    chosen: Dict[str, List[str]] = {}
    any_pruned = False

    for col in partition_keys:
        spec = specs[col]
        col_preds = pred_by_col.get(col, [])
        values = enumerate_values(spec)

        if values is None:
            # injected/unbounded — only usable if a predicate supplies values.
            supplied = _values_from_predicates(col_preds)
            if supplied is None:
                return None
            chosen[col] = supplied
            any_pruned = True
            continue

        if col_preds:
            before = len(values)
            for p in col_preds:
                values = filter_values(values, spec, p)
            if len(values) < before:
                any_pruned = True

        if not values:
            return [], True  # a predicate excluded every value
        chosen[col] = values

    if not any_pruned:
        return None  # nothing narrowed; sizing the root once is equivalent

    total = 1
    for col in partition_keys:
        total *= len(chosen[col])
        if total > max_partitions:
            return None

    return build_prefixes(template, partition_keys, chosen, base_location), True


def _values_from_predicates(preds: List[Predicate]) -> Optional[List[str]]:
    """Concrete values an ``=``/``IN`` predicate pins a key to, else ``None``."""
    for p in preds:
        if p.op == "=":
            return [p.values[0]]
        if p.op == "IN":
            return list(p.values)
    return None
