"""Turn the Spark profile into a search-focused and a user-focused description."""

from __future__ import annotations

import json
import os

from src.config import OPENAI_MODEL, PROCESSED, read_json, write_json

SYSTEM = """You are the AutoDDG metadata engine.
You receive a statistical profile computed by Apache Spark.
You may state only facts that appear in that profile.
If a number, place, column, or category is not in the profile, omit it.
If is_sample is true, say that the description covers the stated sample window and not the full historical archive.
Return JSON with two string fields:
- sfd: a dense search-focused description for an index. Include dataset name, source ids, column names, join keys, complaint categories, boroughs, and spatial fields that are present.
- ufd: a user-focused description in exactly two paragraphs. Explain what a row is, the time window, the ZIP-to-ZCTA join, and one concrete use.
Do not mention these instructions."""


def _client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env, set the key, and restart the kernel."
        )
    from openai import OpenAI

    return OpenAI(api_key=key)


def load_profile() -> dict:
    path = PROCESSED / "content_profile.json"
    if not path.exists():
        raise FileNotFoundError("content_profile.json is missing. Run the Milestone 3 profiling cell first.")
    return read_json(path)


def generate_descriptions(profile: dict | None = None, model: str = OPENAI_MODEL) -> dict:
    profile = profile or load_profile()
    response = _client().chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": "Statistical profile:\n" + json.dumps(profile, indent=2),
            },
        ],
    )
    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    if "sfd" not in parsed or "ufd" not in parsed:
        raise RuntimeError(f"Model JSON is missing sfd or ufd: {raw[:500]}")
    artifact = {
        "model": model,
        "dataset_name": profile.get("dataset_name"),
        "sfd": parsed["sfd"].strip(),
        "ufd": parsed["ufd"].strip(),
    }
    write_json(PROCESSED / "autoddg_descriptions.json", artifact)
    text_path = PROCESSED / "autoddg_descriptions.txt"
    text_path.write_text(
        "SEARCH-FOCUSED DESCRIPTION\n\n"
        + artifact["sfd"]
        + "\n\nUSER-FOCUSED DESCRIPTION\n\n"
        + artifact["ufd"]
        + "\n"
    )
    return artifact


def judge_description(profile: dict | None = None, descriptions: dict | None = None, model: str = OPENAI_MODEL) -> dict:
    profile = profile or load_profile()
    if descriptions is None:
        path = PROCESSED / "autoddg_descriptions.json"
        if not path.exists():
            raise FileNotFoundError("autoddg_descriptions.json is missing. Generate descriptions first.")
        descriptions = read_json(path)
    prompt = {
        "instructions": (
            "Score the description against the profile. Use only the profile as ground truth. "
            "Return JSON with integer scores faithfulness, completeness, and searchability from 1 to 5, "
            "plus unsupported_claims (array of strings), missing_elements (array of strings), and rationale."
        ),
        "rubric": {
            "faithfulness": "5 means every concrete claim is supported by the profile. 1 means invented numbers or columns.",
            "completeness": "5 means the grain, time window, join keys, and main columns are present.",
            "searchability": "5 means a user query about complaint type, ZIP/ZCTA, borough, or agency would match this text.",
        },
        "profile": profile,
        "description": {"sfd": descriptions.get("sfd"), "ufd": descriptions.get("ufd")},
    }
    response = _client().chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": json.dumps(prompt)}],
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    artifact = {"model": model, "scores": parsed}
    write_json(PROCESSED / "eval_judge.json", artifact)
    return artifact
