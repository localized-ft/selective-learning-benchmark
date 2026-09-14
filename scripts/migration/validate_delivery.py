"""Validate the local archive, registries, document links, and executable imports."""
import ast
import gzip
import json
from pathlib import Path
import re
import sys
from package import ROOT, RESULT, json_file, SECRET_PATTERN, SIGNED_URL, sanitize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
from archive import Archive, rows


def main():
    archive=Archive()
    checks=archive.verify()
    signatures=0
    for obj in archive.objects.values():
        data=archive.content(obj["artifact_id"])
        formats={s["format"] for s in obj["sources"]}
        suffix=".xlsx" if "xlsx" in formats else ".gz" if "gz" in formats else ".txt"
        _, found=sanitize(data,suffix,[])
        signatures+=found
    if signatures: raise ValueError("Credential/signature patterns found in archive")
    runs=rows(RESULT/"registry/runs.jsonl")
    models={r["model_id"]:r for r in rows(RESULT/"registry/models.jsonl")}
    trained=[r for r in runs if r.get("selected_release")=="five_seed_20260908"]
    if len(trained)!=630 or len(runs)!=631: raise ValueError("Wrong run coverage")
    identities={(r["task_id"],r["model_family"],r["method"],r["training_seed"]) for r in trained}
    if len(identities)!=630: raise ValueError("Duplicate run identity")
    for run in runs:
        assert run["model_id"] in models
        assert run["primary_result_artifact_id"] in archive.objects
        for item in run.get("associated_artifacts",[])+run.get("inference_artifacts",[])+run.get("remote_output_artifacts",[]):
            assert item["artifact_id"] in archive.objects
            if item.get("source_path"):
                assert archive.sources[item["source_path"]]["artifact_id"] == item["artifact_id"]
        if run.get("extracted_completions"):
            assert run["extracted_completions"]["artifact_id"] in archive.objects
    release=json.loads((RESULT/"releases/five_seed_20260908/manifest.json").read_text())
    for item in release["archived_release_files"]:
        assert (RESULT/item["path"]).read_bytes() == archive.content(item["artifact_id"])
    if (RESULT/"standardized/v1/manifest.json").exists():
        from standardize import verify_view
        checks["standardized_view"] = verify_view(RESULT/"standardized/v1", archive)
        json_file(RESULT/"standardized/v1/verification.json", checks["standardized_view"])
    checked=0
    docs=[ROOT/"README.md",ROOT/"NOTICE.md",RESULT/"README.md",ROOT/"scripts/analysis/README.md",
          RESULT/"releases/five_seed_20260908/REPORT.md",RESULT/"releases/five_seed_20260908/scripts/README.md"]
    for doc in docs:
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            if "://" in target or target.startswith("#"):continue
            path=(doc.parent/target.split("#")[0]).resolve()
            if not path.exists():raise ValueError("Broken document link: "+str(doc.relative_to(ROOT))+" -> "+target)
            checked+=1
    scripts=list((ROOT/"scripts").rglob("*.py"))
    for path in scripts: ast.parse(path.read_text(),filename=str(path))
    inventory=json.loads((RESULT/"migration/delivery_inventory.json").read_text())
    checks.update(main_runs=630,supplemental_runs=1,python_scripts_parsed=len(scripts),
                  local_document_links_checked=checked,credential_pattern_matches=signatures,
                  all_artifact_references_resolve=True,
                  frozen_release_files_unchanged=len(release["archived_release_files"]),
                  inventory=inventory,success=True)
    json_file(RESULT/"migration/delivery_verification.json",checks)
    print(json.dumps(checks,indent=2))


if __name__=="__main__":main()
