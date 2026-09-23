"""Public, pinned FP16 Qwen32 inference; one isolated cache/engine at a time."""
import argparse
import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from ip_pilot_worker import sha, records_bytes, validate_outputs


def run_model(cfg, key):
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from openweights import OpenWeights
    assert vllm.__version__ == cfg['vllm_version']
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).total_memory >= cfg.get('minimum_gpu_gib',79) * 1024**3
    assert cfg.get('required_gpu_name','') in torch.cuda.get_device_name(0)
    ow = OpenWeights()
    started = time.monotonic()
    raw = Path('vanilla_requests.jsonl').read_bytes()
    assert sha(raw) == cfg['requests_sha256']
    requests = [json.loads(line) for line in raw.splitlines()]
    assert len(requests) == cfg['request_count']
    rows = [r for r in requests if r['checkpoint'] == key]
    assert rows
    model, canonical = cfg['models'][key], cfg['canonical_tokenizer_spec']
    def tokenizer(spec):
        tok = AutoTokenizer.from_pretrained(spec['repo'], revision=spec['revision'],
                                           token=False, trust_remote_code=False)
        assert sha(tok.chat_template.encode()) == spec['chat_template_sha256']
        return tok
    tok, native = tokenizer(canonical), tokenizer(model)
    prompts, hashes = [], []
    for r in rows:
        kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
        text = tok.apply_chat_template(r['messages'], **kwargs)
        other = native.apply_chat_template(r['messages'], **kwargs)
        ids = tok(text)['input_ids']
        assert text == other and ids == native(other)['input_ids'], 'Native tokenizer mismatch'
        assert len(ids) + cfg['sampling']['max_tokens'] <= cfg['engine']['max_model_len']
        prompts.append({'prompt_token_ids': ids})
        hashes.append(sha(text.encode()))
    engine = dict(cfg['engine'], model=model['repo'], revision=model['revision'],
                  tokenizer=canonical['repo'], tokenizer_revision=canonical['revision'],
                  hf_token=False, download_dir=os.environ['HF_HUB_CACHE'])
    llm = vllm.LLM(**engine)
    def upload(suffix, records=None, raw=None, **fields):
        name = key + '_' + suffix
        raw = records_bytes(records) if records is not None else raw
        buf = io.BytesIO(raw); buf.name = name
        file = ow.files.create(buf, purpose='custom_job_file')
        ow.run.log(dict(type='vanilla_artifact_saved', filename=name, file_id=file['id'],
                        content_sha256=sha(raw), checkpoint=key, **fields))
    environment = dict(engine_args=engine, sampling=cfg['sampling'],
                       gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                       transformers=transformers.__version__, vllm=vllm.__version__,
                       native_rendering_token_matches=len(rows), model=model,
                       canonical_tokenizer=canonical, enable_thinking=True)
    upload('environment.json', raw=json.dumps(environment, indent=2).encode())
    ow.run.log(dict(type='vanilla_model_loaded', checkpoint=key, gpu=environment['gpu']))
    completions, details = [], []
    def checkpoint(final=False):
        for suffix, records in [('completions.jsonl', completions), ('generation_details.jsonl', details)]:
            upload(suffix, records=records, n=len(records), final=final)
    try:
        for start in range(0, len(rows), cfg['checkpoint_every']):
            batch = rows[start:start + cfg['checkpoint_every']]
            batch_prompts = prompts[start:start + len(batch)]
            params = [vllm.SamplingParams(**cfg['sampling'], seed=r['inference_seed']) for r in batch]
            before = time.monotonic()
            outputs = llm.generate(batch_prompts, params, use_tqdm=False)
            elapsed = time.monotonic() - before
            validate_outputs(batch, batch_prompts, outputs, cfg['sampling']['max_tokens'])
            for i, (r, output) in enumerate(zip(batch, outputs)):
                result = output.outputs[0]; ids = list(result.token_ids)
                text = tok.decode(ids, skip_special_tokens=True)
                completions.append(dict(completion_id=r['completion_id'], eval_id=r['eval_id'], completion=text))
                details.append(dict(**{k:v for k,v in r.items() if k != 'messages'},
                    completion=text, engine_text=result.text, input_token_ids=list(output.prompt_token_ids),
                    output_token_ids=ids, input_tokens=len(output.prompt_token_ids), output_tokens=len(ids),
                    finish_reason=result.finish_reason, stop_reason=result.stop_reason,
                    rendered_prompt_sha256=hashes[start+i], batch_generation_seconds=elapsed,
                    batch_index=start//cfg['checkpoint_every'], generation_seconds=None))
            checkpoint()
            ow.run.log(dict(type='vanilla_progress', checkpoint=key, n=len(completions), total=len(rows),
                            elapsed_seconds=time.monotonic()-started))
    finally:
        checkpoint(final=True)
    assert len(completions) == len(rows)


def main():
    os.environ.update(HF_HUB_DISABLE_IMPLICIT_TOKEN='1', VLLM_NO_USAGE_STATS='1',
                      VLLM_ENABLE_V1_MULTIPROCESSING='0')
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint'); args = p.parse_args()
    cfg = json.loads(Path('vanilla_config.json').read_text())
    if args.checkpoint:
        run_model(cfg, args.checkpoint); return
    from openweights import OpenWeights
    from openweights.worker.services.ttl_manager import get_shutdown_time, set_shutdown_time
    ow = OpenWeights(); started = time.monotonic()
    old = get_shutdown_time()
    deadline = datetime.datetime.now() + datetime.timedelta(hours=cfg['minimum_pod_ttl_hours'])
    if old is not None and old < deadline:
        set_shutdown_time(deadline)
    ow.run.log(dict(type='vanilla_started', phase=cfg['phase'], n_requests=cfg['request_count']))
    for key in cfg['models']:
        remaining = cfg['max_worker_seconds'] - (time.monotonic()-started) - 60
        if remaining <= 0: raise RuntimeError('Inference time budget exhausted')
        # Only this job's temporary download cache is removed, never a shared HF cache.
        # Child exit frees all CUDA state; upload completion precedes cache removal.
        with tempfile.TemporaryDirectory(prefix='qwen32-inference-') as cache:
            env = dict(os.environ, HF_HOME=cache, HF_HUB_CACHE=cache+'/hub', HF_XET_CACHE=cache+'/xet')
            subprocess.run([sys.executable, __file__, '--checkpoint', key], env=env, check=True, timeout=remaining)
    ow.run.log(dict(type='vanilla_finished', complete=True, n=cfg['request_count'], expected=cfg['request_count'],
                    elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
