"""Turn the Spark profile into a search-focused and a user-focused description."""

from __future__ import annotations

import json
import os

from src.config import OPENAI_MODEL, PROCESSED, read_json, write_json

SYSTEM = """You are the AutoDDG metadata engine.
You receive a statistical profile computed by Apache Spark.
You may state only facts that appear in that profile.
If a number, place, column, or category is not in the profile, omit it.
If is_sample is true, say that the description covers the stated sample window and not a larger archive.
Return JSON with two string fields:
- sfd: a dense search-focused description. Include the dataset name, source id, column names, join keys, and categories that are present in the profile.
- ufd: a user-focused description in exactly two paragraphs. Explain what a row is, the coverage, any join key, and one concrete use.
Do not mention these instructions."""


def _client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env, set the key, and restart the kernel."
        )
    from openai import OpenAI

    return OpenAI(api_key=key)


def load_profile(path=None) -> dict:
    path = path or (PROCESSED / "content_profile.json")
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing. Build the Spark profile first.")
    return read_json(path)


def _write_descriptions(artifact: dict, output_path) -> dict:
    write_json(output_path, artifact)
    text_path = output_path.with_suffix(".txt")
    text_path.write_text(
        "SEARCH-FOCUSED DESCRIPTION\n\n"
        + artifact["sfd"]
        + "\n\nUSER-FOCUSED DESCRIPTION\n\n"
        + artifact["ufd"]
        + "\n"
    )
    artifact["path"] = str(output_path)
    return artifact


def generate_descriptions(profile: dict | None = None, model: str = OPENAI_MODEL, output_path=None) -> dict:
    profile = profile or load_profile()
    output_path = output_path or (PROCESSED / "autoddg_descriptions.json")
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
        "dataset_id": profile.get("source_dataset_id"),
        "sfd": parsed["sfd"].strip(),
        "ufd": parsed["ufd"].strip(),
    }
    return _write_descriptions(artifact, output_path)


def judge_description(
    profile: dict | None = None,
    descriptions: dict | None = None,
    model: str = OPENAI_MODEL,
    human_text: str | None = None,
    label: str = "generated",
    output_path=None,
) -> dict:
    profile = profile or load_profile()
    if human_text is None and descriptions is None:
        path = PROCESSED / "autoddg_descriptions.json"
        if not path.exists():
            raise FileNotFoundError("autoddg_descriptions.json is missing. Generate descriptions first.")
        descriptions = read_json(path)
    described = (
        {"kind": "human_portal", "text": human_text}
        if human_text is not None
        else {"kind": "generated", "sfd": descriptions.get("sfd"), "ufd": descriptions.get("ufd")}
    )
    prompt = {
        "instructions": (
            "Score the description against the profile. Use only the profile as ground truth. "
            "Return JSON with integer scores faithfulness, completeness, and searchability from 1 to 5, "
            "plus unsupported_claims (array of strings), missing_elements (array of strings), and rationale."
        ),
        "rubric": {
            "faithfulness": "5 means every concrete claim is supported by the profile. 1 means invented numbers or columns.",
            "completeness": "5 means the grain, coverage, join keys, and main columns are present.",
            "searchability": "5 means a query using this dataset's topics, places, or column names would match this text.",
        },
        "profile": profile,
        "description": described,
    }
    response = _client().chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": json.dumps(prompt)}],
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    artifact = {
        "model": model,
        "label": label,
        "dataset_id": profile.get("source_dataset_id"),
        "dataset_name": profile.get("dataset_name"),
        "scores": parsed,
    }
    write_json(output_path or (PROCESSED / "eval_judge.json"), artifact)
    return artifact


def human_description(dataset_id: str) -> str:
    catalog = read_json(PROCESSED / "catalog.json")["datasets"]
    match = next((row for row in catalog if row.get("id") == dataset_id), None)
    if not match or not match.get("human_description"):
        raise FileNotFoundError(f"No human description for {dataset_id} in catalog.json")
    return match["human_description"]


def score_against_baseline(profile: dict, generated: dict, human_text: str, stem: str) -> dict:
    """Score generated SFD/UFD and the human portal text against the same Spark profile."""
    generated_judgment = judge_description(
        profile,
        generated,
        label="generated",
        output_path=PROCESSED / f"eval_judge_{stem}_generated.json",
    )
    human_judgment = judge_description(
        profile,
        human_text=human_text,
        label="human",
        output_path=PROCESSED / f"eval_judge_{stem}_human.json",
    )
    return {
        "dataset_id": profile.get("source_dataset_id"),
        "dataset_name": profile.get("dataset_name"),
        "generated": generated_judgment["scores"],
        "human": human_judgment["scores"],
    }


def write_scorecard(rows: list[dict]) -> dict:
    flat = []
    for row in rows:
        for kind in ("generated", "human"):
            scores = row[kind]
            flat.append(
                {
                    "dataset": row["dataset_name"],
                    "dataset_id": row["dataset_id"],
                    "text": kind,
                    "faithfulness": scores.get("faithfulness"),
                    "completeness": scores.get("completeness"),
                    "searchability": scores.get("searchability"),
                }
            )
    artifact = {"rows": flat, "detail": rows}
    write_json(PROCESSED / "eval_judge_comparison.json", artifact)
    return artifact
