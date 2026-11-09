"""PySpark cleaning, broadcast join, and partitioned Parquet export."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    broadcast,
    coalesce,
    col,
    count,
    date_format,
    length,
    percentile_approx,
    regexp_replace,
    substring,
    to_timestamp,
    unix_timestamp,
    when,
)

from src.config import (
    CURATED,
    END_DATE,
    PROCESSED,
    RAW_311,
    RAW_ZCTA,
    SPARK_DRIVER_MEMORY,
    SPARK_SHUFFLE_PARTITIONS,
    START_DATE,
    write_json,
)

AUDIT_COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "complaint_type",
    "descriptor",
    "incident_zip",
    "borough",
    "latitude",
    "longitude",
    "status",
]

TEMPORAL = {
    "created_date",
    "closed_date",
    "due_date",
    "resolution_action_updated_date",
}
SPATIAL = {
    "incident_zip",
    "latitude",
    "longitude",
    "location",
    "x_coordinate_state_plane",
    "y_coordinate_state_plane",
    "borough",
    "city",
}
IDENTIFIERS = {"unique_key", "bbl"}
TEXT = {"resolution_description", "descriptor", "descriptor_2", "incident_address"}


def column_role(name: str) -> str:
    if name in TEMPORAL:
        return "temporal"
    if name in SPATIAL:
        return "spatial"
    if name in IDENTIFIERS:
        return "identifier"
    if name in TEXT:
        return "text"
    return "categorical"


def start_spark() -> SparkSession:
    return (
        SparkSession.builder.master("local[*]")
        .appName("NYC_311_AutoDDG_Pipeline")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.session.timeZone", "America/New_York")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def _csv(spark: SparkSession, directory) -> DataFrame:
    parts = sorted(str(path) for path in directory.glob("part-*.csv"))
    if not parts:
        raise FileNotFoundError(
            f"No CSV pages in {directory}. Run the Milestone 1 download cell first."
        )
    # Keep strings. Schema inference turns ZIP codes into integers and drops leading zeros.
    # Read only the CSV pages. The download manifest in this folder is JSON.
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "false")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(parts)
    )


def load_fact(spark: SparkSession) -> DataFrame:
    frame = _csv(spark, RAW_311)
    drop_cols = [c for c in frame.columns if c.startswith(":@")]
    return frame.drop(*drop_cols) if drop_cols else frame


def load_zcta(spark: SparkSession) -> DataFrame:
    return _csv(spark, RAW_ZCTA)


def audit_raw(fact: DataFrame) -> dict:
    present = [c for c in AUDIT_COLUMNS if c in fact.columns]
    total = fact.count()
    unique = fact.select("unique_key").distinct().count() if "unique_key" in fact.columns else None
    nulls = {}
    if present:
        null_exprs = [
            count(when(col(c).isNull() | (col(c) == ""), c)).alias(c) for c in present
        ]
        nulls = {key: int(value) for key, value in fact.select(null_exprs).collect()[0].asDict().items()}
    report = {
        "total_rows": int(total),
        "unique_keys": None if unique is None else int(unique),
        "duplicate_keys": None if unique is None else int(total - unique),
        "n_columns": len(fact.columns),
        "columns": [
            {"name": name, "type": dtype, "role": column_role(name)}
            for name, dtype in fact.dtypes
        ],
        "null_or_blank_counts": nulls,
        "sample_window": {"start": START_DATE, "end": END_DATE, "end_exclusive": True},
    }
    write_json(PROCESSED / "m1_raw_audit.json", report)
    return report


def _coalesce_timestamps(raw):
    """SODA CSV is ISO-8601. The portal bulk export is US 12-hour time. Accept both."""
    return coalesce(
        to_timestamp(raw, "yyyy-MM-dd'T'HH:mm:ss.SSS"),
        to_timestamp(raw, "yyyy-MM-dd'T'HH:mm:ss"),
        to_timestamp(raw, "MM/dd/yyyy hh:mm:ss a"),
        to_timestamp(raw, "MM/dd/yyyy HH:mm:ss"),
    )


def clean_fact(fact: DataFrame) -> DataFrame:
    digits = regexp_replace(col("incident_zip"), r"[^0-9]", "")
    clean_zip = when(length(digits) >= 5, substring(digits, 1, 5)).otherwise(digits)
    frame = fact.withColumn("clean_zip", clean_zip).withColumn(
        "created_timestamp", _coalesce_timestamps(col("created_date"))
    )
    if "closed_date" in fact.columns:
        frame = frame.withColumn("closed_timestamp", _coalesce_timestamps(col("closed_date")))
    return (
        frame.withColumn("year", date_format(col("created_timestamp"), "yyyy"))
        .withColumn("month", date_format(col("created_timestamp"), "MM"))
    )


def apply_fact_filters(cleaned: DataFrame) -> tuple[DataFrame, dict]:
    valid_zip = col("clean_zip").rlike(r"^[0-9]{5}$")
    valid_time = col("created_timestamp").isNotNull()
    raw_count = cleaned.count()
    zip_ok = cleaned.filter(valid_zip).count()
    both = cleaned.filter(valid_zip & valid_time)
    kept = both.count()
    report = {
        "rows_before_filter": int(raw_count),
        "rows_with_valid_zip": int(zip_ok),
        "rows_dropped_bad_zip": int(raw_count - zip_ok),
        "rows_dropped_bad_timestamp_after_zip": int(zip_ok - kept),
        "rows_kept": int(kept),
    }
    write_json(PROCESSED / "m2_cleaning_report.json", report)
    return both, report


def clean_zcta(zcta: DataFrame) -> DataFrame:
    digits = regexp_replace(col("zcta5"), r"[^0-9]", "")
    frame = zcta.withColumn("dim_zip", when(length(digits) >= 5, substring(digits, 1, 5)))
    for name in ("centlat", "centlon", "intptlat", "intptlon"):
        if name in frame.columns:
            frame = frame.withColumn(name, col(name).cast("double"))
    keep = [c for c in ("dim_zip", "centlat", "centlon", "intptlat", "intptlon", "arealand", "areawater") if c in frame.columns]
    return frame.select(*keep).dropDuplicates(["dim_zip"]).filter(col("dim_zip").rlike(r"^[0-9]{5}$"))


def broadcast_join(fact: DataFrame, zcta: DataFrame) -> DataFrame:
    # ZCTA is 221 rows. Broadcast it so the multi-million-row fact table is not shuffled.
    return fact.join(broadcast(zcta), fact.clean_zip == zcta.dim_zip, how="left")


def join_report(joined: DataFrame) -> dict:
    total = joined.count()
    matched = joined.filter(col("dim_zip").isNotNull()).count()
    unmatched = (
        joined.filter(col("dim_zip").isNull())
        .groupBy("clean_zip")
        .count()
        .orderBy(col("count").desc())
        .limit(15)
        .collect()
    )
    report = {
        "rows": int(total),
        "matched_rows": int(matched),
        "unmatched_rows": int(total - matched),
        "match_rate": None if total == 0 else matched / total,
        "top_unmatched_zips": [
            {"clean_zip": row["clean_zip"], "rows": int(row["count"])} for row in unmatched
        ],
        "note": (
            "incident_zip is a USPS ZIP and zcta5 is a Census ZCTA. "
            "Equality is an approximation, so a match rate below 1 is expected."
        ),
    }
    write_json(PROCESSED / "m2_join_report.json", report)
    return report


def resolution_pattern(joined: DataFrame, limit: int = 10) -> list[dict]:
    if "closed_timestamp" not in joined.columns:
        return []
    hours = (unix_timestamp("closed_timestamp") - unix_timestamp("created_timestamp")) / 3600.0
    scoped = joined.withColumn("resolution_hours", hours).filter(
        col("resolution_hours").isNotNull()
        & (col("resolution_hours") >= 0)
        & (col("resolution_hours") <= 24 * 365)
    )
    rows = (
        scoped.groupBy("complaint_type")
        .agg(
            count("*").alias("n"),
            percentile_approx("resolution_hours", 0.5).alias("median_hours"),
        )
        .orderBy(col("n").desc())
        .limit(limit)
        .collect()
    )
    pattern = [
        {
            "complaint_type": row["complaint_type"],
            "n": int(row["n"]),
            "median_resolution_hours": None
            if row["median_hours"] is None
            else round(float(row["median_hours"]), 2),
        }
        for row in rows
    ]
    write_json(PROCESSED / "m2_resolution_pattern.json", {"by_complaint_type": pattern})
    return pattern


def write_curated(joined: DataFrame) -> str:
    (
        joined.write.mode("overwrite")
        .partitionBy("year", "month")
        .parquet(str(CURATED))
    )
    return str(CURATED)


def partition_counts(spark: SparkSession) -> list[dict]:
    frame = spark.read.parquet(str(CURATED))
    rows = frame.groupBy("year", "month").count().orderBy("year", "month").collect()
    counts = [
        {"year": row["year"], "month": row["month"], "rows": int(row["count"])} for row in rows
    ]
    write_json(PROCESSED / "m2_partitions.json", {"partitions": counts, "path": str(CURATED)})
    return counts
