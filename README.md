# LLM-Powered Dataset Description Generation

Course project for Team TomCat. The question is whether an AutoDDG-style pipeline, fed by an Apache Spark profile, produces descriptions that are more faithful and easier to retrieve than the short text on the NYC Open Data portal.

The large case is [311 Service Requests from 2020 to Present](https://data.cityofnewyork.us/d/erm2-nwe9) (`erm2-nwe9`), enriched with [ZIP Code Tabulation Areas](https://data.cityofnewyork.us/d/35j5-n34v) (`35j5-n34v`). A small catalog of other portal datasets supplies human-authored descriptions for the search benchmark.

Run the presentation notebook from the repository root:

```bash
jupyter notebook notebooks/autoddg_milestones.ipynb
```

Each milestone in that notebook ends with the table or file you should keep for the report.

## Setup

Use Python 3.11. PySpark does not reliably support the newest Homebrew Python. Spark also needs a JDK.

```bash
brew install openjdk@17
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put `OPENAI_API_KEY` in `.env`. A Socrata app token is optional and only affects rate limits. Do not commit `.env`.

The default download is a 20,000-row cap inside 1–3 June 2025. That window is the reproducible sample. It is not the full archive. Set `SAMPLE_ROWS=0` and widen `START_DATE` / `END_DATE` only after the sample notebook runs.

## What each milestone produces

| Milestone | You do | Outcome to cite |
| --- | --- | --- |
| 1. Ingest and profile | Page the SODA API into `data/raw/`, start Spark, audit the raw schema | `data/processed/m1_raw_audit.json` and `catalog.json` |
| 2. Clean and join | Standardize ZIP codes and timestamps, broadcast-join ZCTA, write partitioned Parquet | cleaning report, join match rate, resolution-time pattern, `data/curated/` |
| 3. AutoDDG | Build a Spark profile and ask the model for an SFD and a UFD | `content_profile.json`, `autoddg_descriptions.json` |
| 4. Evaluate and report | BM25 MRR against portal text, then an LLM judge. Write the ACM-style report | `eval_bm25.json`, `eval_judge.json`, paper |

## Report sections

Map the notebook outcomes into the required paper outline:

- Introduction: missing descriptions hide datasets. State the 311 question.
- Problem formulation: sample window, ZIP-to-ZCTA join, SFD versus UFD, metrics (faithfulness, completeness, searchability, MRR, Recall@1).
- Related work: AutoDDG, data profiling, BM25, LLM-as-a-judge. Cite them in the text.
- Methods, architecture, and design: data lake layout, why the ZCTA table is broadcast, why the profile is the only model input, why queries were frozen before generation.
- Results: paste the audit, match rate, resolution pattern, both descriptions, the BM25 table, and the judge scores. Say that the default run is a dated sample.
- References: papers and the NYC Open Data dataset pages.

## Reproduce without the notebook

From the repository root, with `JAVA_HOME` and `.env` set:

```bash
python -c "from src.ingest import prepare_lake; print(prepare_lake())"
```

Then run the Spark, profile, generation, and evaluation cells in `notebooks/autoddg_milestones.ipynb`. Generation and the judge need `OPENAI_API_KEY`. The human-description BM25 index does not.

## Publish

This folder is the project that belongs at <https://github.com/tl4732-cyber/Big-Data_Dataset-Description-Generation>. Commit the code, the notebook, and this file. Leave `data/raw`, `data/processed`, `data/curated`, and `.env` untracked. The `.gitignore` already does that.

```bash
git init
git add .
git commit -m "Add the reproducible AutoDDG Spark pipeline and milestone notebook."
git branch -M main
git remote add origin https://github.com/tl4732-cyber/Big-Data_Dataset-Description-Generation.git
git push -u origin main
```

Put that GitHub URL in the report.
