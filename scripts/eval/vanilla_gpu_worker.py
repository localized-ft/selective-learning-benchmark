"""Pinned, checkpointed vanilla inference; no training or API judging.

Mounted by vanilla_gpu.py. Deliberately independent of the migrated worker,
whose source bytes remain unchanged. Request ordering and prompt construction
match its sequential Transformers inference, with additional provenance.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    # Public checkpoint: do not use or change shared HF credentials.
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from openweights import OpenWeights

    started = time.monotonic()
    cfg = json.loads(Path("vanilla_config.json").read_text())
    data = Path("vanilla_requests.jsonl").read_bytes()
    assert digest(data) == cfg["requests_sha256"]
    requests = [json.loads(line) for line in data.splitlines() if line]
    assert len(requests) == cfg["request_count"]
    assert len({r["completion_id"] for r in requests}) == len(requests)
    ow = OpenWeights()

    def upload(name, raw, kind, **fields):
        buf = io.BytesIO(raw)
        buf.name = name
        file = ow.files.create(buf, purpose="custom_job_file")
        ow.run.log({"type": kind, "file_id": file["id"], "filename": name,
                    "content_sha256": digest(raw), **fields})

    ow.run.log({"type": "vanilla_started", "model": cfg["model"],
                "revision": cfg["revision"], "n_requests": len(requests),
                "phase": cfg["phase"]})
    if cfg.get("minimum_pod_ttl_hours"):
        # Only this executing pod's bounded lifetime, not organization settings.
        import datetime
        from openweights.worker.services.ttl_manager import get_shutdown_time, set_shutdown_time
        old_deadline = get_shutdown_time()
        if old_deadline is not None:
            deadline = datetime.datetime.now() + datetime.timedelta(hours=cfg["minimum_pod_ttl_hours"])
            if old_deadline < deadline:
                set_shutdown_time(deadline)
                ow.run.log({"type": "vanilla_pod_ttl", "previous": old_deadline.isoformat(),
                            "deadline": deadline.isoformat()})
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"], revision=cfg["revision"], token=False, trust_remote_code=False)
    llm = AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg["revision"], token=False,
        torch_dtype=torch.float16, device_map="auto", trust_remote_code=False)
    if any(str(v) in {"cpu", "disk"} for v in getattr(llm, "hf_device_map", {}).values()):
        raise RuntimeError("CPU/disk offload is not permitted for this throughput pilot")
    llm.eval()
    generation = llm.generation_config.to_dict()
    metadata = {
        "schema_version": "slb.vanilla_gpu_environment.v1", "config": cfg,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "gpu_memory_bytes": [torch.cuda.get_device_properties(i).total_memory
                             for i in range(torch.cuda.device_count())],
        "model_dtype": str(llm.dtype), "device_map": getattr(llm, "hf_device_map", {}),
        "resolved_model_revision": getattr(llm.config, "_commit_hash", None),
        "model_generation_config": generation,
        "effective_generation_config": {**generation, "temperature": 1.0,
                                         "max_new_tokens": 2000, "do_sample": True},
        "chat_template": tokenizer.chat_template,
        "chat_template_sha256": digest(str(tokenizer.chat_template).encode()),
        "tokenization": "apply_chat_template(tokenize=False, add_generation_prompt=True), then tokenizer(defaults); matches historical worker",
        "load_elapsed_seconds": time.monotonic() - started,
    }
    for key, expected in cfg.get("expected_environment", {}).items():
        if metadata[key] != expected:
            raise RuntimeError(f"Pilot/full environment mismatch: {key}; refusing incompatible continuation")
    upload("environment.json", json.dumps(metadata, indent=2, default=str).encode(),
           "vanilla_environment_saved")
    ow.run.log({"type": "vanilla_model_loaded", "gpu": metadata["gpu"],
                "load_elapsed_seconds": metadata["load_elapsed_seconds"],
                "top_p": generation.get("top_p"), "top_k": generation.get("top_k")})
    completions = []
    raw_outputs = []

    def checkpoint(final=False):
        for name, rows in [("completions.jsonl", completions), ("generation_details.jsonl", raw_outputs)]:
            raw = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in rows)
            upload(name, raw, "vanilla_artifact_saved", n=len(rows), final=final)

    try:
        for request in requests:
            if time.monotonic() - started > cfg["max_worker_seconds"] - 180:
                ow.run.log({"type": "vanilla_time_limit", "n": len(completions)})
                break
            set_seed(request["inference_seed"])
            rendered = tokenizer.apply_chat_template(
                request["messages"], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt").to(llm.device)
            input_len = inputs["input_ids"].shape[1]
            before = time.monotonic()
            with torch.inference_mode():
                outputs = llm.generate(**inputs, max_new_tokens=2000,
                                       temperature=1.0, do_sample=True)
            torch.cuda.synchronize()
            generated = outputs[0][input_len:].tolist()
            text = tokenizer.decode(generated, skip_special_tokens=True)
            completions.append({"completion_id": request["completion_id"],
                                "eval_id": request["eval_id"], "completion": text})
            raw_outputs.append({
                "schema_version": "slb.vanilla_generation.v1",
                "completion_id": request["completion_id"], "eval_id": request["eval_id"],
                "task_id": request["task_id"], "axis": request["axis"],
                "prompt_index": request["prompt_index"], "sample_index": request["sample_index"],
                "training_seed": None, "inference_seed": request["inference_seed"],
                "input_tokens": input_len, "output_tokens": len(generated),
                "input_token_ids": inputs["input_ids"][0].tolist(),
                "output_token_ids": generated, "completion": text,
                "rendered_prompt_sha256": digest(rendered.encode()),
                "finish_reason": "length" if len(generated) >= 2000 else "stop",
                "generation_seconds": time.monotonic() - before,
                "elapsed_seconds": time.monotonic() - started,
            })
            if len(completions) % cfg["checkpoint_every"] == 0:
                checkpoint()
                ow.run.log({"type": "vanilla_progress", "n": len(completions),
                            "total": len(requests), "elapsed_seconds": time.monotonic() - started})
    finally:
        checkpoint(final=True)
    ow.run.log({"type": "vanilla_finished", "n": len(completions), "expected": len(requests),
                "complete": len(completions) == len(requests),
                "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
