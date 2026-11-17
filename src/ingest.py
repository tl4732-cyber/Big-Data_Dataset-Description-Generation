"""Download NYC Open Data into the local data lake and cache portal metadata."""

from __future__ import annotations

import csv
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from src.config import (
    AUCTUS_ID,
    AUCTUS_KEYWORDS,
    AUCTUS_SEARCH_URL,
    CATALOG_IDS,
    END_DATE,
    FACT_ID,
    PAGE_SIZE,
    PORTAL_PAGE,
    PROCESSED,
    RAW_311,
    RAW_AUCTUS,
    RAW_ZCTA,
    SAMPLE_ROWS,
    SOCRATA_APP_TOKEN,
    SODA_ROOT,
    START_DATE,
    VIEW_ROOT,
    ZCTA_ID,
    read_json,
    write_json,
)

_HTML = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    text = _HTML.sub(" ", value)
    return _WS.sub(" ", text).strip()


def _request(url: str, accept: str) -> bytes:
    headers = {"Accept": accept, "User-Agent": "nyu-bigdata-autoddg/1.0"}
    if SOCRATA_APP_TOKEN:
        headers["X-App-Token"] = SOCRATA_APP_TOKEN
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            last_error = RuntimeError(f"HTTP {exc.code} for {url}\n{detail}")
            if exc.code not in {429, 500, 502, 503, 504}:
                raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Request failed after retries: {url}") from last_error


def _soda_page(dataset_id: str, params: dict) -> list[list[str]]:
    url = f"{SODA_ROOT}/{dataset_id}.csv?{urllib.parse.urlencode(params)}"
    raw = _request(url, "text/csv").decode("utf-8-sig")
    return list(csv.reader(io.StringIO(raw)))


def download_soda_csv(
    dataset_id: str,
    dest_dir,
    where: str | None = None,
    select: str | None = None,
    order: str | None = None,
    max_rows: int | None = None,
    page_size: int = PAGE_SIZE,
) -> dict:
    """Page through SODA. One request cannot reliably return more than 50,000 rows."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for old in dest_dir.glob("part-*.csv"):
        old.unlink()

    remaining = None if max_rows in (None, 0) else max_rows
    offset = 0
    header: list[str] | None = None
    written = 0
    pages = 0

    while remaining is None or remaining > 0:
        limit = page_size if remaining is None else min(page_size, remaining)
        params: dict[str, str] = {"$limit": str(limit), "$offset": str(offset)}
        if where:
            params["$where"] = where
        if select:
            params["$select"] = select
        if order:
            params["$order"] = order
        table = _soda_page(dataset_id, params)
        if not table:
            break
        page_header, body = table[0], table[1:]
        if not body:
            break
        if header is None:
            header = page_header
        elif page_header != header:
            raise RuntimeError(f"Header changed while paging {dataset_id}")

        pages += 1
        part = dest_dir / f"part-{pages:05d}.csv"
        with part.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(body)
        written += len(body)
        offset += len(body)
        if remaining is not None:
            remaining -= len(body)
        if len(body) < limit:
            break

    manifest = {
        "dataset_id": dataset_id,
        "rows": written,
        "pages": pages,
        "where": where,
        "select": select,
        "order": order,
        "max_rows": max_rows,
        "directory": str(dest_dir),
    }
    write_json(dest_dir / "_manifest.json", manifest)
    return manifest


def download_fact_311() -> dict:
    where = (
        f"created_date >= '{START_DATE}T00:00:00.000' "
        f"AND created_date < '{END_DATE}T00:00:00.000'"
    )
    return download_soda_csv(
        FACT_ID,
        RAW_311,
        where=where,
        order="unique_key",
        max_rows=SAMPLE_ROWS,
    )


def download_zcta() -> dict:
    # Centroids are enough for the broadcast join. the_geom is a multipolygon and is omitted.
    return download_soda_csv(
        ZCTA_ID,
        RAW_ZCTA,
        select="zcta5,arealand,areawater,centlat,centlon,intptlat,intptlon",
        order="zcta5",
        max_rows=1000,
    )


def search_auctus(keywords: str = AUCTUS_KEYWORDS) -> list[dict]:
    """Auctus search is a form POST. The query value is a JSON object."""
    form = urllib.parse.urlencode({"query": json.dumps({"keywords": keywords})}).encode()
    raw = _request_form(AUCTUS_SEARCH_URL, form)
    payload = json.loads(raw.decode("utf-8"))
    return payload.get("results") or []


def _request_form(url: str, form: bytes) -> bytes:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "nyu-bigdata-autoddg/1.0",
    }
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=form, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            last_error = RuntimeError(f"HTTP {exc.code} for {url}\n{detail}")
            if exc.code not in {429, 500, 502, 503, 504}:
                raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Request failed after retries: {url}") from last_error


def auctus_catalog_record(metadata: dict) -> dict:
    columns = metadata.get("columns") or []
    return {
        "id": metadata.get("id") or AUCTUS_ID,
        "name": metadata.get("name") or AUCTUS_ID,
        "human_description": strip_html(metadata.get("description")),
        "n_columns": int(metadata.get("nb_columns") or len(columns)),
        "column_names": [column.get("name") for column in columns if column.get("name")],
        "page": metadata.get("source_url"),
        "source": "auctus",
        "join_key": {"fact": "clean_zip", "auctus": "ZCTA5"},
    }


def download_auctus_places() -> dict:
    """Download the Auctus CDC PLACES ZCTA table and store its catalog record."""
    hits = search_auctus(AUCTUS_KEYWORDS)
    match = next((hit for hit in hits if hit.get("id") == AUCTUS_ID), None)
    if match is None:
        raise RuntimeError(f"Auctus did not return {AUCTUS_ID} for {AUCTUS_KEYWORDS!r}")
    metadata = match["metadata"]
    direct_url = (metadata.get("materialize") or {}).get("direct_url")
    if not direct_url:
        raise RuntimeError(f"{AUCTUS_ID} has no materialize.direct_url")

    RAW_AUCTUS.mkdir(parents=True, exist_ok=True)
    part = RAW_AUCTUS / "part-00001.csv"
    request = urllib.request.Request(direct_url, headers={"User-Agent": "nyu-bigdata-autoddg/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response, part.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    with part.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle))
    if "ZCTA5" not in header:
        raise RuntimeError(f"Downloaded CSV is missing ZCTA5. Header starts: {header[:8]}")

    record = auctus_catalog_record(metadata)
    record["n_columns"] = len(header)
    record["column_names"] = header
    record["csv_path"] = str(part)
    write_json(PROCESSED / "auctus_places_zcta.json", record)
    _upsert_catalog(record)
    write_json(
        RAW_AUCTUS / "_manifest.json",
        {
            "dataset_id": AUCTUS_ID,
            "name": record["name"],
            "direct_url": direct_url,
            "n_columns": record["n_columns"],
            "join_key": record["join_key"],
            "directory": str(RAW_AUCTUS),
        },
    )
    return record


def _upsert_catalog(record: dict) -> None:
    path = PROCESSED / "catalog.json"
    if path.exists():
        datasets = read_json(path).get("datasets") or []
    else:
        datasets = []
    datasets = [row for row in datasets if row.get("id") != record["id"]]
    datasets.append(record)
    write_json(path, {"datasets": datasets})


def fetch_catalog(dataset_ids: list[str] | None = None) -> list[dict]:
    records = []
    for dataset_id in dataset_ids or CATALOG_IDS:
        url = f"{VIEW_ROOT}/{dataset_id}.json"
        payload = json_loads(_request(url, "application/json"))
        columns = payload.get("columns") or []
        records.append(
            {
                "id": dataset_id,
                "name": payload.get("name") or dataset_id,
                "human_description": strip_html(payload.get("description")),
                "n_columns": len(columns),
                "column_names": [c.get("fieldName") for c in columns if c.get("fieldName")],
                "page": f"{PORTAL_PAGE}/{dataset_id}",
            }
        )
    cached = PROCESSED / "auctus_places_zcta.json"
    if cached.exists():
        records.append(read_json(cached))
    write_json(PROCESSED / "catalog.json", {"datasets": records})
    return records


def json_loads(raw: bytes) -> dict:
    return json.loads(raw.decode("utf-8"))


def prepare_lake() -> dict:
    fact = download_fact_311()
    zcta = download_zcta()
    auctus = download_auctus_places()
    catalog = fetch_catalog()
    summary = {
        "fact_311": fact,
        "dim_zcta": zcta,
        "auctus_places_zcta": {"id": auctus["id"], "n_columns": auctus["n_columns"]},
        "catalog_size": len(catalog),
    }
    write_json(PROCESSED / "m1_download.json", summary)
    return summary
