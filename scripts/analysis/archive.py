"""Portable access to the actual repository-contained raw artifacts."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result"
PREFIX = "sunday/scripts/eval/results/"


def rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Archive:
    def __init__(self, result=RESULT):
        self.result = Path(result)
        self.sources = {r["source_path"]: r for r in rows(self.result / "migration/source_manifest.jsonl")}
        self.objects = {r["artifact_id"]: r for r in rows(self.result / "registry/artifacts.jsonl")}

    def content(self, artifact_id):
        record = self.objects[artifact_id]
        path = (self.result / record["path"]).resolve()
        if self.result.resolve() not in path.parents:
            raise ValueError("Artifact path escapes result directory")
        stored = path.read_bytes()
        if hashlib.sha256(stored).hexdigest() != record["stored_sha256"]:
            raise ValueError("Stored object hash mismatch: " + artifact_id)
        data = gzip.decompress(stored)
        if hashlib.sha256(data).hexdigest() != artifact_id:
            raise ValueError("Content hash mismatch: " + artifact_id)
        return data

    def source(self, path):
        return self.content(self.sources[path]["artifact_id"])

    def analysis(self, path):
        return self.source(PREFIX + "five_seed_comparison/" + path)

    def verify(self):
        for i, artifact_id in enumerate(self.objects, 1):
            self.content(artifact_id)
            if i % 500 == 0: print(f"Verified {i}/{len(self.objects)} objects", flush=True)
        for item in rows(self.result / "migration/code_manifest.jsonl"):
            data = (self.result.parent / item["destination_path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Committed source copy changed: " + item["destination_path"])
        return {"raw_objects_verified": len(self.objects), "source_aliases": len(self.sources),
                "committed_code_hashes_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    extract = sub.add_parser("extract")
    extract.add_argument("--destination", type=Path, required=True)
    extract.add_argument("--prefix", default=PREFIX)
    args = parser.parse_args()
    archive = Archive()
    if args.command == "verify":
        print(json.dumps(archive.verify(), indent=2))
        return
    base = args.destination.resolve()
    for path, record in archive.sources.items():
        if not path.startswith(args.prefix): continue
        target = (base / path).resolve()
        if base not in target.parents: raise ValueError("Unsafe archived path")
        data = archive.content(record["artifact_id"])
        if target.exists() and target.read_bytes() != data:
            raise ValueError("Refusing to overwrite different file: " + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print("Extracted original-format result files into", base)


if __name__ == "__main__": main()
