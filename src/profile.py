"""Spark statistical profile that AutoDDG is allowed to cite."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql.functions import approx_count_distinct, col, count, when

from src.config import AUCTUS_ID, END_DATE, FACT_ID, PROCESSED, RAW_AUCTUS, START_DATE, write_json
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


def build_auctus_profile(frame: DataFrame) -> dict:
    """Profile the full CDC PLACES ZCTA table indexed by Auctus."""
    total = frame.count()
    key = "ZCTA5" if "ZCTA5" in frame.columns else None
    measure_columns = [
        name
        for name in ("TotalPopulation", "TotalPop18plus", "ACCESS2_CrudePrev", "ARTHRITIS_CrudePrev", "BPHIGH_CrudePrev")
        if name in frame.columns
    ]
    distinct = None
    null_key = None
    if key:
        distinct = frame.select(approx_count_distinct(col(key)).alias("n")).collect()[0]["n"]
        null_key = frame.select(
            count(when(col(key).isNull() | (col(key) == ""), key)).alias("n")
        ).collect()[0]["n"]

    measures = {}
    for name in measure_columns:
        numeric = frame.withColumn(name, col(name).cast("double"))
        row = numeric.selectExpr(
            f"min({name}) as min_v",
            f"max({name}) as max_v",
            f"avg({name}) as avg_v",
        ).collect()[0]
        measures[name] = {
            "min": None if row["min_v"] is None else round(float(row["min_v"]), 2),
            "max": None if row["max_v"] is None else round(float(row["max_v"]), 2),
            "mean": None if row["avg_v"] is None else round(float(row["avg_v"]), 2),
        }

    profile = {
        "dataset_name": "PLACES: ZCTA Data (GIS Friendly Format), 2024 release",
        "source_dataset_id": AUCTUS_ID,
        "source": "auctus",
        "source_page": "https://data.cdc.gov/500-Cities-Places/PLACES-ZCTA-Data-GIS-Friendly-Format-2024-release/kee5-23sr",
        "is_sample": False,
        "coverage_rule": (
            "This is the full national ZCTA table indexed by Auctus. "
            "Describe only statistics present in this profile. "
            "ZCTA5 is the join key to NYC 311 clean_zip."
        ),
        "total_rows": int(total),
        "primary_key": key,
        "join": {"auctus_key": "ZCTA5", "fact_key": "clean_zip"},
        "zcta5_approx_distinct": None if distinct is None else int(distinct),
        "zcta5_null_or_blank": None if null_key is None else int(null_key),
        "measure_summaries": measures,
        "columns": [{"name": name, "spark_type": dtype} for name, dtype in frame.dtypes],
    }
    write_json(PROCESSED / "content_profile_auctus.json", profile)
    return profile


def load_auctus_frame(spark) -> DataFrame:
    parts = sorted(str(path.resolve()) for path in RAW_AUCTUS.glob("part-*.csv"))
    if not parts:
        raise FileNotFoundError("Auctus CSV is missing. Run download_auctus_places() first.")
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "false")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(parts)
    )
