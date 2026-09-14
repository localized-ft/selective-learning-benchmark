"""Offline format-only canonical views of saved inference and judgment files.

Original artifacts, score scales, selected results, and analysis estimands are
never modified. CSV and nested judge JSONL become the same flat judgment schema.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

from archive import Archive, RESULT, rows

VERSION = "slb.records.v1"
FIELDS = ("schema_version", "record_type", "source_artifact_id", "source_row",
          "source_score_index", "completion_id", "eval_id", "axis", "group_id",
          "question", "completion", "reference_response", "grading_method",
          "score_name", "score", "score_status", "score_raw", "score_label",
          "score_source_text", "model", "judge_model", "task_id", "extra_fields")
TEXT_FIELDS = ("completion_id", "eval_id", "group_id", "question", "completion",
               "reference_response", "grading_method", "score_name", "score_label",
               "score_source_text", "model", "judge_model", "task_id")
AXES = {"capability": "capability", "undesired_generalization": "unintended_generalization",
        "unintended_generalization": "unintended_generalization"}


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def text_value(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    return encode(value)


def score_value(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "missing"
    if isinstance(value, bool):
        return None, "non_numeric"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, "non_numeric"
    return (number, "numeric") if math.isfinite(number) else (None, "non_finite")


def source_records(data, filename):
    decoded = data.decode("utf-8-sig")
    if filename.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(decoded))
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {"completion_id", "score_name", "score"} <= set(fields):
            raise ValueError("Invalid or duplicate CSV header")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Ragged CSV row")
            yield row
    else:
        for line in decoded.splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object per line")
                yield row


def canonical(row, artifact_id, row_number, record_type, score_index=None):
    output = dict.fromkeys(FIELDS)
    output.update(schema_version=VERSION, record_type=record_type,
                  source_artifact_id=artifact_id, source_row=row_number,
                  source_score_index=score_index)
    for name in TEXT_FIELDS:
        output[name] = text_value(row.get(name))
    if "completion" in row and row["completion"] is not None:
        output["completion"] = row["completion"]
    if not output["completion_id"]:
        raise ValueError("Missing completion identity")
    axis = row.get("axis")
    if axis not in (None, "") and axis not in AXES:
        raise ValueError("Unknown axis label")
    output["axis"] = AXES.get(axis)
    if record_type == "judgment":
        if not output["score_name"]:
            raise ValueError("Missing score name")
        value = row.get("score")
        output["score"], output["score_status"] = score_value(value)
        # Keep literal categorical/non-finite values distinguishable from null.
        output["score_raw"] = (str(value) if isinstance(value, float) and not math.isfinite(value)
                               else text_value(value))
    else:
        if "completion" not in row:
            raise ValueError("Missing completion field")
        # An empty generated answer is an observed empty string, not missing.
        output["completion"] = row["completion"]
        if not isinstance(output["completion"], str):
            raise ValueError("Completion must be text")
        output["score_status"] = "not_applicable"
    known = set(TEXT_FIELDS) | {"axis", "score"}
    output["extra_fields"] = {k: v for k, v in row.items() if k not in known}
    validate_record(output)
    return output


def convert(data, filename, artifact_id):
    for number, row in enumerate(source_records(data, filename), 1):
        if filename == "judge_scores.jsonl":
            scores = row.get("scores")
            if not isinstance(scores, list) or not scores:
                raise ValueError("Missing/empty nested scores; cannot silently discard a row")
            base = {k: v for k, v in row.items() if k != "scores"}
            for index, score in enumerate(scores):
                if not isinstance(score, dict) or set(base) & set(score):
                    raise ValueError("Conflicting nested score fields")
                yield canonical({**base, **score}, artifact_id, number, "judgment", index)
        elif filename.endswith(".csv"):
            yield canonical(row, artifact_id, number, "judgment")
        elif filename == "completions.jsonl":
            yield canonical(row, artifact_id, number, "completion")
        else:
            raise ValueError("Unsupported source format")


def validate_record(row):
    if tuple(row) != FIELDS or row["schema_version"] != VERSION:
        raise ValueError("Noncanonical field order/schema")
    if row["record_type"] not in ("completion", "judgment"):
        raise ValueError("Unknown record type")
    digest = row["source_artifact_id"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid source artifact ID")
    if not isinstance(row["source_row"], int) or row["source_row"] < 1:
        raise ValueError("Invalid source row")
    if row["source_score_index"] is not None and (type(row["source_score_index"]) is not int or row["source_score_index"] < 0):
        raise ValueError("Invalid source score index")
    for field in TEXT_FIELDS + ("score_raw",):
        if row[field] is not None and not isinstance(row[field], str):
            raise ValueError("Mixed type in " + field)
    if row["axis"] not in (None, "capability", "unintended_generalization"):
        raise ValueError("Invalid canonical axis")
    if row["score"] is not None and (type(row["score"]) not in (int, float) or not math.isfinite(row["score"])):
        raise ValueError("Invalid numeric score")
    if row["score_status"] not in ("numeric", "missing", "non_numeric", "non_finite", "not_applicable"):
        raise ValueError("Invalid score status")
    if (row["score"] is not None) != (row["score_status"] == "numeric"):
        raise ValueError("Inconsistent numeric score status")
    if not isinstance(row["extra_fields"], dict):
        raise ValueError("Extra fields must be an object")


def inventory(archive):
    sources = {}

    def add(artifact_id, filename, role=None, run_id=None):
        entry = sources.setdefault(artifact_id, {"source_artifact_id": artifact_id,
            "filename": filename, "source_aliases": [r["source_path"] for r in archive.objects[artifact_id]["sources"]],
            "selected_completion_runs": [], "selected_judgment_runs": [], "remote_references": []})
        if entry["filename"] != filename:
            raise ValueError("One artifact has incompatible schemas")
        if role and run_id not in entry[role]:
            entry[role].append(run_id)
        return entry

    for download in rows(RESULT / "migration/remote_output_downloads.jsonl"):
        if download["status"] != "downloaded":
            raise ValueError("Unavailable remote output")
        add(download["artifact_id"], download["filename"])
    for run in rows(RESULT / "registry/runs.jsonl"):
        add(run["primary_result_artifact_id"], "eval_results.csv", "selected_judgment_runs", run["run_id"])
        for entry in run["inference_artifacts"]:
            if entry["source_path"].endswith("/completions.jsonl"):
                add(entry["artifact_id"], "completions.jsonl", "selected_completion_runs", run["run_id"])
        # The supplemental registry points to full API events; its associated
        # saved completions file supplies the comparable flat checkpoint view.
        if run.get("method") == "vanilla":
            for entry in run["associated_artifacts"]:
                if entry["source_path"].endswith("/completions.jsonl"):
                    add(entry["artifact_id"], "completions.jsonl", "selected_completion_runs", run["run_id"])
        for ref in run.get("remote_output_artifacts", []):
            if ref["artifact_id"] in sources:
                record = {"run_id": run["run_id"], **{k: ref[k] for k in
                          ("job_id", "file_id", "event_id", "remote_run_id", "event_type")}}
                if record not in sources[ref["artifact_id"]]["remote_references"]:
                    sources[ref["artifact_id"]]["remote_references"].append(record)
    return sorted(sources.values(), key=lambda r: r["source_artifact_id"])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def verify_view(output, archive=None):
    """Validate every persisted record against its referenced original source."""
    archive = archive or Archive()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["schema_version"] != VERSION or tuple(manifest["field_order"]) != FIELDS:
        raise ValueError("Unknown standardized manifest schema")
    source_rows = rows(output / "sources.jsonl")
    sources = {r["source_artifact_id"]: r for r in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError("Duplicate standardized source identity")
    counts = Counter()
    observed_counts = {digest: Counter() for digest in sources}
    checked_sources = set()
    for entry in manifest["files"]:
        if entry["path"] not in ("completions.jsonl.gz", "judgments.jsonl.gz"):
            raise ValueError("Unexpected standardized output path")
        path = output / entry["path"]
        with path.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError("Standardized output hash mismatch")
        current, expected = None, iter(())
        with gzip.open(path, "rt", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                validate_record(record)
                digest = record["source_artifact_id"]
                if digest != current:
                    if next(expected, None) is not None:
                        raise ValueError("Canonical source rows omitted")
                    if digest in checked_sources:
                        raise ValueError("Source occurs in noncontiguous/duplicate blocks")
                    source = sources[digest]
                    expected = iter(convert(archive.content(digest), source["filename"], digest))
                    current = digest
                    checked_sources.add(digest)
                if next(expected, None) != record:
                    raise ValueError("Canonical record does not match original source")
                kind = "completion" if entry["path"].startswith("completions.") else "judgment"
                if record["record_type"] != kind:
                    raise ValueError("Wrong entity in standardized file")
                counts[kind] += 1
                observed_counts[digest][kind] += 1
            if next(expected, None) is not None:
                raise ValueError("Canonical source truncated")
    if counts != manifest["records"] or checked_sources != set(sources):
        raise ValueError("Standardized source/record coverage mismatch")
    for digest, count in observed_counts.items():
        if dict(count) != sources[digest]["canonical_rows"]:
            raise ValueError("Per-source row counts changed")
    return {"source_artifacts_verified": len(sources), "records_verified": dict(counts),
            "every_record_matches_source": True, "file_hashes_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT / "standardized/v1")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.verify_only:
        verified = verify_view(output)
        write_json(output / "verification.json", verified)
        print(json.dumps(verified, indent=2))
        return
    protected = (RESULT / "artifacts", RESULT / "releases", RESULT / "registry",
                 RESULT / "runs", RESULT / "protocols", RESULT / "supplemental")
    if any(output == path.resolve() or path.resolve() in output.parents for path in protected):
        raise ValueError("Do not write canonical views into original result directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    archive = Archive()
    sources = inventory(archive)
    counts = Counter()
    statuses = Counter()
    variants = Counter()
    source_audit = []
    # Stream only two data files, rather than adding thousands of near-duplicate
    # per-job files. Fixed gzip headers make regeneration deterministic.
    handles = {}
    for kind, name in (("completion", "completions.jsonl.gz"), ("judgment", "judgments.jsonl.gz")):
        handles[kind] = gzip.GzipFile(filename="", mode="wb", fileobj=(output / name).open("wb"), mtime=0)
    try:
        for i, entry in enumerate(sources, 1):
            source = archive.content(entry["source_artifact_id"])
            records = list(source_records(source, entry["filename"]))
            for row in records:
                variants[(entry["filename"], tuple(sorted(row)))]+=1
            emitted = Counter()
            source_axes = Counter()
            for row in convert(source, entry["filename"], entry["source_artifact_id"]):
                handles[row["record_type"]].write((encode(row) + "\n").encode("utf-8"))
                counts[row["record_type"]] += 1
                emitted[row["record_type"]] += 1
                statuses[row["score_status"]] += 1
                source_axes[row["axis"] or "missing"] += 1
            source_audit.append({**entry, "source_rows": len(records), "canonical_rows": dict(emitted),
                                 "canonical_axes": dict(source_axes)})
            if i % 100 == 0:
                print(f"Standardized {i}/{len(sources)} source artifacts", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    (output / "sources.jsonl").write_text("".join(encode(r) + "\n" for r in source_audit), encoding="utf-8")
    files = []
    verified = Counter()
    for name in ("completions.jsonl.gz", "judgments.jsonl.gz"):
        path = output / name
        with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
            for line in source:
                row = json.loads(line)
                validate_record(row)
                verified[row["record_type"]] += 1
        files.append({"path": name, "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(), "bytes": path.stat().st_size})
    if verified != counts:
        raise ValueError("Round-trip record counts changed")
    selected = {kind: {run for entry in sources for run in entry[kind]} for kind in
                ("selected_completion_runs", "selected_judgment_runs")}
    if any(len(value) != 631 for value in selected.values()):
        raise ValueError("Incomplete selected-run coverage")
    manifest = {"schema_version": VERSION, "encoding": "UTF-8", "line_endings": "LF",
        "compression": "gzip", "field_order": FIELDS, "source_artifacts": len(sources),
        "remote_file_ids_covered": len(rows(RESULT / "migration/remote_output_downloads.jsonl")),
        "selected_completion_runs": len(selected["selected_completion_runs"]),
        "selected_judgment_runs": len(selected["selected_judgment_runs"]),
        "records": dict(counts), "score_status_counts": dict(statuses), "files": files,
        "input_schemas": [{"filename": key[0], "fields": key[1], "rows": value} for key, value in sorted(variants.items())],
        "score_scale": "original rubric scale, not rescaled", "filtering_applied": False,
        "coherence_backfill_join_applied": False, "raw_artifacts_modified": False,
        "round_trip_schema_and_counts_verified": True,
        "warning": "Do not pool all rows: sources.jsonl separates selected artifacts from retries and duplicate judge representations."}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("field_order", "input_schemas")}, indent=2))


if __name__ == "__main__":
    main()
