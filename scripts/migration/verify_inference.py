"""Verify raw inference against archived judge inputs; label legacy extractions."""
import csv
import gzip
import io
import json
from pathlib import Path
import sys
from package import RESULT, json_file, jsonl_file, sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
from archive import Archive, rows


def main():
    archive=Archive()
    all_runs=rows(RESULT/"registry/runs.jsonl")
    runs=[r for r in all_runs if r.get("selected_release")=="five_seed_20260908"]
    objects=archive.objects
    sources=archive.sources
    audit=[]
    for run in runs:
        expected={}
        for row in csv.DictReader(io.StringIO(archive.content(run["primary_result_artifact_id"]).decode())):
            value={"completion_id":row["completion_id"],"eval_id":row["eval_id"],"completion":row.get("completion","")}
            if value["completion_id"] in expected and expected[value["completion_id"]]!=value:
                raise ValueError("Ambiguous legacy completion identity: "+run["run_id"])
            expected[value["completion_id"]]=value
        for artifact in run["inference_artifacts"]:
            found=[json.loads(line) for line in archive.content(artifact["artifact_id"]).splitlines() if line.strip()]
            if len(found)!=len(expected) or len({r["completion_id"] for r in found})!=len(found):
                raise ValueError("Inference coverage mismatch: "+run["run_id"])
            for row in found:
                match=expected[row["completion_id"]]
                if row["completion"]!=match["completion"] or ("eval_id" in row and row["eval_id"]!=match["eval_id"]):
                    raise ValueError("Inference/judge text mismatch: "+run["run_id"])
        extracted=None
        if not run["inference_artifacts"]:
            data=("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in expected.values())).encode()
            digest=sha(data)
            stored=gzip.compress(data,compresslevel=6,mtime=0)
            path=RESULT/"artifacts/sha256"/digest[:2]/(digest+".gz")
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(stored)
            source="derived/legacy_completions/"+run["run_id"]+"/completions.jsonl"
            evidence={"source_path":source,"source_sha256":digest,"format":"jsonl","redactions":0,
                      "byte_identical_to_source":False,"transformation":"extract_unique_completion_text_from_judge_csv",
                      "parent_artifact_id":run["primary_result_artifact_id"]}
            obj={"artifact_id":digest,"path":str(path.relative_to(RESULT)),"content_sha256":digest,
                 "stored_sha256":sha(stored),"content_bytes":len(data),"stored_bytes":len(stored),"compression":"gzip","sources":[evidence]}
            objects[digest]=obj
            sources[source]={**{k:v for k,v in obj.items() if k!="sources"},**evidence,
                             "original_bytes":len(data),"derived":True}
            extracted={"source_path":source,"artifact_id":digest,"kind":"reconstructed_from_eval_results",
                       "parent_artifact_id":run["primary_result_artifact_id"],
                       "unavailable_fields":["original_inference_response_envelope","token_usage","finish_reason"]}
            run["extracted_completions"]=extracted
            json_file(RESULT/"runs"/run["run_id"]/"manifest.json",run)
            json_file(RESULT/"runs"/run["run_id"]/"inference"/run["inference_id"]/"manifest.json",
                {"run_id":run["run_id"],"inference_id":run["inference_id"],"job_id":run["completion_job_id"],
                 "artifacts":[],"status":"standalone_original_not_located","extracted_completions":extracted})
        audit.append({"run_id":run["run_id"],"completion_count":len(expected),
                      "original_inference_files_verified":len(run["inference_artifacts"]),
                      "legacy_extraction":extracted,"judge_input_text_verified":True})
    jsonl_file(RESULT/"registry/runs.jsonl",runs+[r for r in all_runs if r.get("selected_release")!="five_seed_20260908"])
    jsonl_file(RESULT/"registry/artifacts.jsonl",sorted(objects.values(),key=lambda r:r["artifact_id"]))
    jsonl_file(RESULT/"migration/source_manifest.jsonl",sorted(sources.values(),key=lambda r:r["source_path"]))
    jsonl_file(RESULT/"migration/inference_integrity.jsonl",audit)
    summary={"runs_verified":len(audit),"completion_texts":sum(r["completion_count"] for r in audit),
             "runs_with_original_inference":sum(bool(r["original_inference_files_verified"]) for r in audit),
             "runs_with_labeled_legacy_extraction":sum(bool(r["legacy_extraction"]) for r in audit),
             "text_mismatches":0}
    json_file(RESULT/"migration/inference_integrity_summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
