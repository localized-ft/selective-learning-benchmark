"""Local, non-destructive migration. Exports code from Git, never the working tree.

Result serialization/copying is a mechanical, hash-audited migration. Does not
call APIs, submit jobs, commit, or push. Secrets are never printed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result"
REVISION = "60c45238b8bac0cec53a2a518f3945a68b39ae83"
CODE_EXTENSIONS = {".py", ".mjs", ".js", ".sh", ".pyc"}
CACHE_NAMES = {"__pycache__", ".pytest_cache", "node_modules", ".venv", ".git"}
SECRET_PATTERN = re.compile(rb"\b(?:hf_[A-Za-z0-9]{20,}|sk-(?:or-v1-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b")
SIGNED_URL = re.compile(rb'https?://[^\s"<>]+[?&](?:X-Amz-Signature|X-Goog-Signature|api_key|access_token)=[^\s"<>]+', re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def jsonl_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def scrub(data, secrets):
    count = 0
    for secret in secrets:
        n = data.count(secret)
        if n:
            data = data.replace(secret, b"[REDACTED_CREDENTIAL]")
            count += n
    data, n = SECRET_PATTERN.subn(b"[REDACTED_CREDENTIAL]", data)
    count += n
    data, n = SIGNED_URL.subn(b"[REDACTED_SIGNED_URL]", data)
    return data, count + n


def sanitize(data, suffix, secrets):
    if suffix == ".gz":
        unpacked = gzip.decompress(data)
        clean, count = scrub(unpacked, secrets)
        return (gzip.compress(clean, mtime=0) if count else data), count
    if suffix == ".xlsx":
        archive = io.BytesIO()
        count = 0
        with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(archive, "w") as dest:
            for member in source.infolist():
                content, n = scrub(source.read(member), secrets)
                count += n
                dest.writestr(member, content)
        return (archive.getvalue() if count else data), count
    return scrub(data, secrets)


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args])


def discover(source):
    base = source / "sunday/scripts"
    sources = []
    exclusions = []
    for current, dirs, names in os.walk(base / "eval/results", followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in CACHE_NAMES and not (Path(current)/d).is_symlink())
        for name in sorted(names):
            p = Path(current) / name
            rel = str(p.relative_to(source))
            reason = None
            if p.is_symlink(): reason = "compatibility_symlink"
            elif name == ".DS_Store": reason = "filesystem_metadata"
            elif p.suffix in CODE_EXTENSIONS: reason = "untracked_producer_code_omitted"
            elif p.suffix in {".yaml", ".yml"}: reason = "untracked_executable_config_omitted"
            elif name.startswith(".env"): reason = "credentials"
            if reason:
                exclusions.append({"source_path": rel, "reason": reason})
            else:
                sources.append((p, rel))
    # Training job ledgers are evidence, not executable submission configs.
    for p in sorted((base / "finetune").glob("*.csv")):
        sources.append((p, str(p.relative_to(source))))
    return sources, exclusions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    assert git(source, "rev-parse", "HEAD").decode().strip() == REVISION, "Source HEAD changed; review first"
    secrets = []
    for parent in (source / "sunday/scripts/eval", source / "sunday/scripts/finetune"):
        env = parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if "=" not in line or line.lstrip().startswith("#"): continue
                key, value = line.split("=", 1)
                value = value.strip().strip("\"'")
                if re.search("KEY|TOKEN|PASSWORD|SECRET", key, re.I) and len(value) >= 12:
                    secrets.append(value.encode())
    tracked = git(source, "ls-tree", "-r", REVISION, "--", "sunday/scripts").decode().splitlines()
    imported, excluded = [], []
    for row in tracked:
        meta, rel = row.split("\t", 1)
        mode, kind, blob = meta.split()
        target = Path(rel).relative_to("sunday/scripts")
        allowed = target.parts[0] in {"eval", "finetune"}
        allowed &= "generated_charts" not in target.parts
        allowed &= target.name not in {"plot_v2_tradeoffs.py", "submit_layer_freeze_v2.py", ".DS_Store"}
        if not allowed:
            excluded.append({"source_path": rel, "reason": "outside_core_committed_allowlist"})
            continue
        data = git(source, "cat-file", "blob", blob)
        _, n = scrub(data, secrets)
        if n: raise RuntimeError("Credential pattern in committed file; review required: " + rel)
        destination = ROOT / "scripts" / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != data:
            raise RuntimeError("Would overwrite a different destination file: " + str(target))
        destination.write_bytes(data)
        if mode == "100755": destination.chmod(0o755)
        imported.append({"source_path": rel, "destination_path": str(destination.relative_to(ROOT)),
                         "source_commit": REVISION, "git_blob": blob, "sha256": sha(data), "bytes": len(data)})
    jsonl_file(RESULT / "migration/code_manifest.jsonl", imported)
    paths, local_excluded = discover(source)
    excluded.extend(local_excluded)

    def pack(entry):
        p, rel = entry
        original = p.read_bytes()
        data, count = sanitize(original, p.suffix, secrets)
        digest = sha(data)
        # Always gzip once, even already-compressed files, for one uniform reader.
        stored = gzip.compress(data, compresslevel=6, mtime=0)
        destination = RESULT / "artifacts/sha256" / digest[:2] / (digest + ".gz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Duplicate objects have identical bytes; concurrent replacement is safe.
        temporary = destination.with_name(destination.name + "." + sha(rel.encode())[:12] + ".tmp")
        temporary.write_bytes(stored)
        temporary.replace(destination)
        return {"source_path": rel, "artifact_id": digest, "path": str(destination.relative_to(RESULT)),
                "source_sha256": sha(original), "content_sha256": digest, "stored_sha256": sha(stored),
                "original_bytes": len(original), "content_bytes": len(data), "stored_bytes": len(stored),
                "format": p.suffix.lstrip(".") or "text", "compression": "gzip",
                "redactions": count, "byte_identical_to_source": count == 0}

    packed = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, record in enumerate(pool.map(pack, paths), 1):
            packed.append(record)
            if index % 200 == 0: print(f"Archived {index}/{len(paths)} source files", flush=True)
    packed.sort(key=lambda r: r["source_path"])
    jsonl_file(RESULT / "migration/source_manifest.jsonl", packed)
    jsonl_file(RESULT / "migration/excluded_files.jsonl", excluded)
    objects = {}
    for row in packed:
        item = objects.setdefault(row["artifact_id"], {k: row[k] for k in (
            "artifact_id", "path", "content_sha256", "stored_sha256", "content_bytes", "stored_bytes", "compression")})
        item.setdefault("sources", []).append({k:row[k] for k in (
            "source_path", "source_sha256", "format", "redactions", "byte_identical_to_source")})
    jsonl_file(RESULT / "registry/artifacts.jsonl", sorted(objects.values(), key=lambda r:r["artifact_id"]))
    summary = {"source_commit": REVISION, "committed_files_copied": len(imported),
               "source_result_files": len(packed), "unique_objects": len(objects),
               "source_bytes": sum(r["original_bytes"] for r in packed),
               "stored_bytes": sum(r["stored_bytes"] for r in objects.values()),
               "redacted_files": sum(r["redactions"] > 0 for r in packed),
               "excluded_files": len(excluded), "source_unchanged": True,
               "remote_upload_performed": False}
    json_file(RESULT / "migration/packaging.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
