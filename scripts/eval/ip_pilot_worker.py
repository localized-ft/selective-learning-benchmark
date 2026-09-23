"""Isolated sequential vLLM engines for the approved Qwen IP prompt diagnostic."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_outputs(batch, prompts, outputs, maximum):
    if len(outputs)!=len(batch):raise ValueError("Missing model outputs")
    for r,p,o in zip(batch,prompts,outputs):
        if not o.finished or len(o.outputs)!=1:raise ValueError("Incomplete generation")
        if list(o.prompt_token_ids)!=p["prompt_token_ids"]:raise ValueError("Prompt token mismatch")
        if not 0<len(o.outputs[0].token_ids)<=maximum:raise ValueError("Invalid output token count")


def records_bytes(records):
    return b"".join((json.dumps(r,ensure_ascii=False)+"\n").encode() for r in records)


def run_model(cfg,method):
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from openweights import OpenWeights
    if vllm.__version__!=cfg["vllm_version"]:raise RuntimeError("Unpinned vLLM runtime")
    ow=OpenWeights(); started=time.monotonic()
    def upload(name,raw,**fields):
        buf=io.BytesIO(raw);buf.name=name
        uploaded=ow.files.create(buf,purpose="custom_job_file")
        ow.run.log({"type":"vanilla_artifact_saved","filename":name,"file_id":uploaded["id"],
                    "content_sha256":sha(raw),"checkpoint":method,**fields})
    raw=Path("vanilla_requests.jsonl").read_bytes()
    assert sha(raw)==cfg["requests_sha256"]
    all_rows=[json.loads(s) for s in raw.splitlines() if s.strip()]
    assert len(all_rows)==cfg["request_count"]
    rows=[r for r in all_rows if r["checkpoint"]==method]
    model=cfg["models"][method]; canonical=cfg["models"][cfg["canonical_tokenizer"]]
    tokenizer=AutoTokenizer.from_pretrained(canonical["repo"],revision=canonical["revision"],token=False,trust_remote_code=False)
    assert sha(tokenizer.chat_template.encode())==canonical["chat_template_sha256"]
    native=AutoTokenizer.from_pretrained(model["repo"],revision=model["revision"],token=False,trust_remote_code=False)
    assert sha(native.chat_template.encode())==model["chat_template_sha256"]
    tokenized={};rendered={}
    for r in rows:
        text=tokenizer.apply_chat_template(r["messages"],tokenize=False,add_generation_prompt=True,enable_thinking=cfg["enable_thinking"])
        ids=tokenizer(text)["input_ids"]
        native_text=native.apply_chat_template(r["messages"],tokenize=False,add_generation_prompt=True,enable_thinking=cfg["enable_thinking"])
        if native_text!=text or native(native_text)["input_ids"]!=ids:
            raise ValueError("Native and canonical rendering/tokenization disagree")
        if len(ids)+cfg["sampling"]["max_tokens"]>cfg["max_model_len"]:
            raise ValueError("Context overflow; prompts must not be truncated")
        tokenized[r["completion_id"]]={"prompt_token_ids":ids}
        rendered[r["completion_id"]]=sha(text.encode())
    engine={"model":model["repo"],"revision":model["revision"],
        "tokenizer":canonical["repo"],"tokenizer_revision":canonical["revision"],
        "hf_token":False,"trust_remote_code":False,"dtype":cfg["dtype"],"quantization":None,
        "tensor_parallel_size":1,"gpu_memory_utilization":0.85,"max_model_len":cfg["max_model_len"],
        "max_num_seqs":32,"max_num_batched_tokens":4096,"enable_chunked_prefill":True,
        "enable_prefix_caching":True,"generation_config":"vllm","seed":cfg["engine_seed"]}
    llm=LLM(**engine)
    environment={"schema_version":"slb.ip_pilot_environment.v1","checkpoint":method,
        "engine_args":engine,"sampling":cfg["sampling"],"python":platform.python_version(),
        "torch":torch.__version__,"transformers":transformers.__version__,"vllm":vllm.__version__,
        "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),
        "chat_template":tokenizer.chat_template,"chat_template_sha256":canonical["chat_template_sha256"],
        "enable_thinking":cfg["enable_thinking"],"native_rendering_token_matches":len(rows),
        "load_elapsed_seconds":time.monotonic()-started}
    upload(method+"_environment.json",json.dumps(environment,indent=2).encode())
    ow.run.log({"type":"vanilla_model_loaded","checkpoint":method,"gpu":environment["gpu"],
                "load_elapsed_seconds":environment["load_elapsed_seconds"]})
    completions=[];details=[]
    def checkpoint(final=False):
        for suffix,data in (("completions.jsonl",completions),("generation_details.jsonl",details)):
            upload(method+"_"+suffix,records_bytes(data),n=len(data),final=final)
    try:
        for batch_index,start in enumerate(range(0,len(rows),cfg["checkpoint_every"])):
            batch=rows[start:start+cfg["checkpoint_every"]]
            prompts=[tokenized[r["completion_id"]] for r in batch]
            params=[SamplingParams(**cfg["sampling"],seed=r["inference_seed"]) for r in batch]
            before=time.monotonic();outputs=llm.generate(prompts,params,use_tqdm=False);elapsed=time.monotonic()-before
            validate_outputs(batch,prompts,outputs,cfg["sampling"]["max_tokens"])
            for r,o in zip(batch,outputs):
                result=o.outputs[0]; ids=list(result.token_ids);text=tokenizer.decode(ids,skip_special_tokens=True)
                completions.append({"completion_id":r["completion_id"],"eval_id":r["eval_id"],"completion":text})
                details.append({**{k:v for k,v in r.items() if k!="messages"},"completion":text,
                    "engine_text":result.text,"input_token_ids":list(o.prompt_token_ids),"output_token_ids":ids,
                    "input_tokens":len(o.prompt_token_ids),"output_tokens":len(ids),
                    "finish_reason":result.finish_reason,"stop_reason":result.stop_reason,
                    "rendered_prompt_sha256":rendered[r["completion_id"]],"batch_index":batch_index,
                    "batch_generation_seconds":elapsed,"generation_seconds":None})
            checkpoint()
            ow.run.log({"type":"vanilla_progress","checkpoint":method,"n":len(completions),
                        "total":len(rows),"elapsed_seconds":time.monotonic()-started})
    finally:checkpoint(final=True)
    assert len(completions)==len(rows)


def main():
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
    os.environ["VLLM_NO_USAGE_STATS"]="1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
    parser=argparse.ArgumentParser();parser.add_argument("--checkpoint");args=parser.parse_args()
    cfg=json.loads(Path("vanilla_config.json").read_text())
    if args.checkpoint:
        run_model(cfg,args.checkpoint);return
    from openweights import OpenWeights
    ow=OpenWeights();started=time.monotonic()
    from openweights.worker.services.ttl_manager import get_shutdown_time,set_shutdown_time
    import datetime
    old=get_shutdown_time()
    deadline=datetime.datetime.now()+datetime.timedelta(hours=cfg["minimum_pod_ttl_hours"])
    if old is not None and old<deadline:
        set_shutdown_time(deadline)
        ow.run.log({"type":"vanilla_pod_ttl","previous":old.isoformat(),"deadline":deadline.isoformat()})
    ow.run.log({"type":"vanilla_started","phase":cfg["phase"],"n_requests":cfg["request_count"],"pilot":"ip_prompt_dependence"})
    for method in cfg["models"]:
        remaining=cfg["max_worker_seconds"]-(time.monotonic()-started)-60
        if remaining<=0:raise RuntimeError("Pilot job time budget exhausted")
        # Child exit frees CUDA state before loading the next 8B checkpoint.
        subprocess.run([sys.executable,__file__,"--checkpoint",method],check=True,timeout=remaining)
    ow.run.log({"type":"vanilla_finished","n":cfg["request_count"],"expected":cfg["request_count"],
                "complete":True,"elapsed_seconds":time.monotonic()-started})


if __name__=="__main__":main()
