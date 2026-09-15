"""One pinned vLLM engine, a 56-request preflight, then the whole Olmo grid.

No API judging or HF credentials. Keep every completion, including empty,
truncated and refused outputs. Backend comparisons are diagnostics, not filters.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_outputs(batch, prompts, outputs, max_tokens):
    if len(outputs) != len(batch):
        raise ValueError("Missing vLLM request output")
    for request, prompt, output in zip(batch, prompts, outputs):
        if not output.finished or len(output.outputs) != 1:
            raise ValueError("Incomplete or multiple vLLM completions")
        if list(output.prompt_token_ids) != prompt["prompt_token_ids"]:
            raise ValueError(f"Input token mismatch for {request['completion_id']}")
        if not 0 < len(output.outputs[0].token_ids) <= max_tokens:
            raise ValueError("Invalid generated token count")


def main():
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from openweights import OpenWeights

    started = time.monotonic()
    cfg = json.loads(Path("vanilla_config.json").read_text())
    if vllm.__version__ != cfg["vllm_version"]:
        raise RuntimeError("Unexpected vLLM version; refusing unpinned runtime")
    data = Path("vanilla_requests.jsonl").read_bytes()
    assert digest(data) == cfg["requests_sha256"]
    requests = [json.loads(s) for s in data.splitlines()]
    assert len(requests) == cfg["request_count"] == 3880
    assert len({r["completion_id"] for r in requests}) == len(requests)
    ow = OpenWeights()

    def upload(name, raw, kind="vanilla_artifact_saved", **fields):
        buf = io.BytesIO(raw)
        buf.name = name
        result = ow.files.create(buf, purpose="custom_job_file")
        ow.run.log({"type": kind, "file_id": result["id"], "filename": name,
                    "content_sha256": digest(raw), **fields})

    ow.run.log({"type": "vanilla_started", "model": cfg["model"],
                "revision": cfg["revision"], "phase": cfg["phase"],
                "n_requests": len(requests), "backend": "vllm"})
    # Extend only this executing pod enough to cover the bounded job.
    from openweights.worker.services.ttl_manager import get_shutdown_time, set_shutdown_time
    import datetime
    old_deadline = get_shutdown_time()
    if old_deadline is not None:
        deadline = datetime.datetime.now() + datetime.timedelta(hours=cfg["minimum_pod_ttl_hours"])
        if old_deadline < deadline:
            set_shutdown_time(deadline)
            ow.run.log({"type": "vanilla_pod_ttl", "previous": old_deadline.isoformat(),
                        "deadline": deadline.isoformat()})

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"],
                                             token=False, trust_remote_code=False)
    template_hash = digest(str(tokenizer.chat_template).encode())
    assert template_hash == cfg["chat_template_sha256"]
    tokenized = {}
    rendered_hashes = {}
    for request in requests:
        text = tokenizer.apply_chat_template(request["messages"], tokenize=False,
                                             add_generation_prompt=True)
        cid = request["completion_id"]
        tokenized[cid] = {"prompt_token_ids": tokenizer(text)["input_ids"]}
        rendered_hashes[cid] = digest(text.encode())
    reference_bytes = Path("vanilla_reference.jsonl").read_bytes()
    assert digest(reference_bytes) == cfg["reference_sha256"]
    references = [json.loads(s) for s in reference_bytes.splitlines() if s]
    for reference in references:
        assert tokenized[reference["completion_id"]]["prompt_token_ids"] == reference["input_token_ids"]
    max_input = max(len(p["prompt_token_ids"]) for p in tokenized.values())
    max_model_len = ((max_input + cfg["sampling"]["max_tokens"] + 255) // 256) * 256
    engine_args = {"model": cfg["model"], "revision": cfg["revision"],
                   "tokenizer_revision": cfg["revision"], "hf_token": False,
                   "trust_remote_code": False, "dtype": "float16", "quantization": None,
                   "tensor_parallel_size": 1, "gpu_memory_utilization": 0.85,
                   "max_model_len": max_model_len, "max_num_seqs": 32,
                   "max_num_batched_tokens": 4096, "enable_chunked_prefill": True,
                   "enable_prefix_caching": True, "generation_config": "vllm",
                   "seed": 20260914}
    llm = LLM(**engine_args)
    metadata = {"schema_version": "slb.vanilla_vllm_environment.v1", "config": cfg,
                "backend": "vllm", "vllm": vllm.__version__, "torch": torch.__version__,
                "transformers": transformers.__version__, "python": platform.python_version(),
                "cuda": torch.version.cuda, "gpu": [torch.cuda.get_device_name(0)],
                "engine_args": engine_args, "sampling": cfg["sampling"],
                "chat_template": tokenizer.chat_template, "chat_template_sha256": template_hash,
                "maximum_input_tokens": max_input, "tokenization_reference_matches": len(references),
                "load_elapsed_seconds": time.monotonic() - started,
                "reproducibility": "per-request seeds; deterministic scheduling; cross-backend outputs need not match"}
    upload("environment.json", json.dumps(metadata, indent=2).encode(), "vanilla_environment_saved")
    ow.run.log({"type": "vanilla_model_loaded", "gpu": metadata["gpu"],
                "backend": "vllm", "load_elapsed_seconds": metadata["load_elapsed_seconds"]})
    completions, details = [], []

    def checkpoint(final=False):
        for name, records in [("completions.jsonl", completions), ("generation_details.jsonl", details)]:
            raw = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in records)
            upload(name, raw, n=len(records), final=final)

    warmup = cfg["warmup_count"]
    offsets = [(0, warmup)] + [(i, min(i + cfg["checkpoint_every"], len(requests)))
                              for i in range(warmup, len(requests), cfg["checkpoint_every"])]
    try:
        for batch_id, (start, stop) in enumerate(offsets):
            if time.monotonic() - started > cfg["max_worker_seconds"] - 600:
                ow.run.log({"type": "vanilla_time_limit", "n": len(completions)})
                break
            batch = requests[start:stop]
            prompts = [tokenized[r["completion_id"]] for r in batch]
            params = [SamplingParams(**cfg["sampling"], seed=r["inference_seed"]) for r in batch]
            before = time.monotonic()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            elapsed = time.monotonic() - before
            validate_outputs(batch, prompts, outputs, cfg["sampling"]["max_tokens"])
            for r, output in zip(batch, outputs):
                generated = output.outputs[0]
                ids = list(generated.token_ids)
                # Use the same HF decoder as the pilot; keep engine text separately.
                text = tokenizer.decode(ids, skip_special_tokens=True)
                completions.append({"completion_id": r["completion_id"], "eval_id": r["eval_id"],
                                    "completion": text})
                details.append({"schema_version": "slb.vanilla_generation.v1", "backend": "vllm",
                                **{k: r[k] for k in ("completion_id", "eval_id", "task_id", "axis",
                                                    "prompt_index", "sample_index", "inference_seed")},
                                "training_seed": None, "completion": text, "engine_text": generated.text,
                                "input_tokens": len(output.prompt_token_ids), "output_tokens": len(ids),
                                "input_token_ids": list(output.prompt_token_ids), "output_token_ids": ids,
                                "finish_reason": generated.finish_reason, "stop_reason": generated.stop_reason,
                                "rendered_prompt_sha256": rendered_hashes[r["completion_id"]],
                                "generation_seconds": None, "batch_id": batch_id,
                                "batch_generation_seconds": elapsed, "elapsed_seconds": time.monotonic() - started})
            checkpoint()
            ow.run.log({"type": "vanilla_progress", "n": len(completions), "total": len(requests),
                        "batch_generation_seconds": elapsed, "elapsed_seconds": time.monotonic() - started})
            if batch_id == 0:
                by_id = {r["completion_id"]: r for r in details}
                comparisons = [{"completion_id": r["completion_id"],
                                "exact_text_match": r["completion"] == by_id[r["completion_id"]]["completion"],
                                "transformers_output_tokens": r["output_tokens"],
                                "vllm_output_tokens": by_id[r["completion_id"]]["output_tokens"]}
                               for r in references]
                projected = elapsed / warmup * len(requests)
                summary = {"warmup_count": warmup, "generation_seconds": elapsed,
                           "projected_generation_seconds": projected,
                           "empty_outputs": sum(not r["completion"].strip() for r in completions),
                           "truncated_outputs": sum(r["finish_reason"] == "length" for r in details),
                           "reference_comparisons": comparisons,
                           "note": "Small diagnostic, not statistical backend equivalence; all raw outputs retained"}
                upload("warmup_validation.json", json.dumps(summary, indent=2).encode(),
                       "vanilla_validation_saved")
                if projected > cfg["max_projected_generation_seconds"]:
                    raise RuntimeError("Warmup projected runtime exceeds approved continuation threshold")
                ow.run.log({"type": "vanilla_warmup_passed", "n": warmup,
                            "projected_generation_seconds": projected})
    finally:
        checkpoint(final=True)
    ow.run.log({"type": "vanilla_finished", "n": len(completions), "expected": len(requests),
                "complete": len(completions) == len(requests), "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
