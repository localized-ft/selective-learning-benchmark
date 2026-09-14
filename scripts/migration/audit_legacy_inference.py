"""Read-only audit of older jobs whose standalone inference output is unlocated."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import sys

from package import RESULT, jsonl_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"analysis"))
from archive import rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--env-file",type=Path,required=True)
    args=parser.parse_args()
    from dotenv import load_dotenv
    from openweights import OpenWeights
    logging.disable(logging.CRITICAL)
    load_dotenv(args.env_file)
    ow=OpenWeights()
    missing=[r for r in rows(RESULT/"registry/runs.jsonl") if r["original_inference_status"]=="not_located"]
    def inspect(run):
        result={"run_id":run["run_id"],"job_id":run["completion_job_id"]}
        try:
            job=ow.jobs.retrieve(run["completion_job_id"])
            output=job.outputs or {}
            cfg=output.get("config") or {}
            result.update(status=str(job.status), output_fields=sorted(output),
                file_ids={k:v for k,v in output.items() if "file" in k and isinstance(v,str) and "file-" in v},
                config_file_ids={k:v for k,v in cfg.items() if "file" in k and isinstance(v,str) and "file-" in v})
            result["interpretation"]="Standalone inference artifact not evidenced by output fields; completion text is preserved in judge CSV."
        except Exception as exc: result.update(status="unavailable",error_type=type(exc).__name__)
        return result
    with ThreadPoolExecutor(max_workers=6) as pool:
        result=list(pool.map(inspect,missing))
    jsonl_file(RESULT/"migration/legacy_inference_audit.jsonl",result)
    print(json.dumps({"jobs_audited":len(result),"unavailable":sum(r["status"]=="unavailable" for r in result),
                      "file_id_field_names":sorted({k for r in result for k in r.get("file_ids",{})}),
                      "config_file_id_field_names":sorted({k for r in result for k in r.get("config_file_ids",{})})},indent=2))


if __name__=="__main__":main()
