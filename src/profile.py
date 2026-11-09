"""Spark statistical profile that AutoDDG is allowed to cite."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql.functions import approx_count_distinct, col, count, when

from src.config import END_DATE, FACT_ID, PROCESSED, START_DATE, write_json
from src.spark_etl import AUDIT_COLUMNS, column_role

DETAIL_COLUMNS = [
    "unique_key",
    "created_date",
    "complaint_type",
    "descriptor",
    "incident_zip",
    "clean_zip",
    "borough",
    "agency",
    "status",
    "latitude",
    "longitude",
    "dim_zip",
    "centlat",
    "centlon",
]


def _top_counts(frame: DataFrame, column: str, total: int, limit: int = 10) -> list[dict]:
    if column not in frame.columns:
        return []
    rows = (
        frame.groupBy(column)
        .count()
        .orderBy(col("count").desc())
        .limit(limit)
        .collect()
    )
    return [
        {
            "value": row[column],
            "count": int(row["count"]),
            "share": None if total == 0 else round(row["count"] / total, 4),
        }
        for row in rows
    ]


def _role(name: str) -> str:
    if name in {"clean_zip", "dim_zip"}:
        return "spatial_key"
    if name in {"centlat", "centlon", "intptlat", "intptlon", "year", "month"}:
        return "spatial" if "lat" in name or "lon" in name else "temporal"
    return column_role(name)


def build_profile(
    joined: DataFrame,
    join_stats: dict,
    resolution: list[dict],
    is_sample: bool = True,
) -> dict:
    total = joined.count()
    detail = [c for c in DETAIL_COLUMNS if c in joined.columns]
    null_exprs = [
        count(when(col(c).isNull() | (col(c) == ""), c)).alias(c) for c in detail
    ]
    distinct_exprs = [approx_count_distinct(col(c)).alias(c) for c in detail]
    nulls = joined.select(null_exprs).collect()[0].asDict() if detail else {}
    distincts = joined.select(distinct_exprs).collect()[0].asDict() if detail else {}

    bounds = {}
    for name in ("latitude", "longitude", "centlat", "centlon"):
        if name not in joined.columns:
            continue
        numeric = joined.withColumn(name, col(name).cast("double"))
        row = numeric.selectExpr(f"min({name}) as min_v", f"max({name}) as max_v").collect()[0]
        bounds[name] = {
            "min": None if row["min_v"] is None else round(float(row["min_v"]), 5),
            "max": None if row["max_v"] is None else round(float(row["max_v"]), 5),
        }

    created = None
    if "created_timestamp" in joined.columns:
        span = joined.selectExpr(
            "min(created_timestamp) as min_v", "max(created_timestamp) as max_v"
        ).collect()[0]
        created = {"min": str(span["min_v"]), "max": str(span["max_v"])}

    profile = {
        "dataset_name": "NYC 311 Service Requests enriched with ZCTA centroids",
        "source_dataset_id": FACT_ID,
        "source_page": f"https://data.cityofnewyork.us/d/{FACT_ID}",
        "dimension_dataset_id": "35j5-n34v",
        "is_sample": is_sample,
        "sample_window": {
            "start_inclusive": START_DATE,
            "end_exclusive": END_DATE,
        },
        "coverage_rule": (
            "Describe only this sample window and these measured statistics. "
            "Do not invent citywide totals, years outside the window, or column values that are absent."
        ),
        "total_rows": int(total),
        "primary_key": "unique_key",
        "join": {
            "fact_key": "clean_zip",
            "dimension_key": "dim_zip",
            "method": "broadcast left hash join",
            "match_rate": join_stats.get("match_rate"),
            "matched_rows": join_stats.get("matched_rows"),
            "unmatched_rows": join_stats.get("unmatched_rows"),
        },
        "created_timestamp_range": created,
        "spatial_bounds": bounds,
        "columns": [
            {
                "name": name,
                "spark_type": dtype,
                "role": _role(name),
                "null_or_blank": int(nulls.get(name, 0)) if name in nulls else None,
                "approx_distinct": int(distincts[name]) if name in distincts else None,
            }
            for name, dtype in joined.dtypes
        ],
        "detailed_columns": detail,
        "top_complaint_types": _top_counts(joined, "complaint_type", total),
        "top_boroughs": _top_counts(joined, "borough", total, limit=6),
        "top_agencies": _top_counts(joined, "agency", total, limit=8),
        "median_resolution_hours_by_complaint": resolution,
        "key_field_audit_columns": [c for c in AUDIT_COLUMNS if c in joined.columns],
    }
    write_json(PROCESSED / "content_profile.json", profile)
    return profile
