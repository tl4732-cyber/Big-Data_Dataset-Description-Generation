"""Paths and run settings. Values come from the environment or a repo-root .env file."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW_311 = DATA / "raw" / "fact_311"
RAW_ZCTA = DATA / "raw" / "dim_zcta"
RAW_AUCTUS = DATA / "raw" / "auctus_places_zcta"
PROCESSED = DATA / "processed"
CURATED = DATA / "curated" / "nyc_311_zcta_enriched"

FACT_ID = "erm2-nwe9"
ZCTA_ID = "35j5-n34v"
SODA_ROOT = "https://data.cityofnewyork.us/resource"
VIEW_ROOT = "https://data.cityofnewyork.us/api/views"
PORTAL_PAGE = "https://data.cityofnewyork.us/d"
AUCTUS_SEARCH_URL = "https://auctus.vida-nyu.org/api/v1/search"

# CDC PLACES 2024 ZCTA estimates, indexed by Auctus. ZCTA5 joins to 311 clean_zip.
AUCTUS_ID = "datamart.socrata.data-cdc-gov.kee5-23sr"
AUCTUS_KEYWORDS = "PLACES ZCTA Data GIS Friendly 2024 release"

# Portal datasets used as the search corpus. 311 is the large Spark case.
# The others supply human-authored descriptions so retrieval has distractors.
CATALOG_IDS = [
    FACT_ID,  # 311 Service Requests from 2020 to Present
    "h9gi-nx95",  # Motor Vehicle Collisions - Crashes
    "43nn-pn8j",  # DOHMH Restaurant Inspection Results
    "5uac-w243",  # NYPD Complaint Data Current (Year To Date)
    ZCTA_ID,  # ZIP Code Tabulation Areas
]

# Written before any model output is inspected. Do not retune these after reading the SFD.
SEARCH_QUERIES = [
    {
        "query": "illegal parking noise heat complaints by brooklyn zip code",
        "relevant_id": FACT_ID,
    },
    {
        "query": "rodent infestation service request resolution time",
        "relevant_id": FACT_ID,
    },
    {
        "query": "motor vehicle collision crash injuries fatalities intersection",
        "relevant_id": "h9gi-nx95",
    },
    {
        "query": "restaurant inspection letter grade mice violation",
        "relevant_id": "43nn-pn8j",
    },
    {
        "query": "nypd complaint offense precinct year to date crime report",
        "relevant_id": "5uac-w243",
    },
    {
        "query": "zip code tabulation area centroid latitude longitude",
        "relevant_id": ZCTA_ID,
    },
    {
        "query": "cdc places zip code tabulation area health prevalence american community survey",
        "relevant_id": AUCTUS_ID,
    },
]

START_DATE = os.getenv("START_DATE", "2025-06-01")
END_DATE = os.getenv("END_DATE", "2025-06-03")
SAMPLE_ROWS = int(os.getenv("SAMPLE_ROWS", "20000"))
PAGE_SIZE = min(int(os.getenv("SODA_PAGE_SIZE", "50000")), 50000)
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")
SPARK_SHUFFLE_PARTITIONS = os.getenv("SPARK_SHUFFLE_PARTITIONS", "8")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN", "")

# LLM Provider Configuration
# Supported providers: "gemini" (default), "local", "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")


def get_active_model(provider: str | None = None) -> str:
    prov = (provider or LLM_PROVIDER).lower()
    if prov == "gemini":
        return GEMINI_MODEL
    elif prov == "local":
        return LOCAL_LLM_MODEL
    return OPENAI_MODEL


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())
