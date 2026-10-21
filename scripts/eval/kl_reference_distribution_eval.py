"""Matched medical inference for the four frozen KL reference ablations.

Each cohort is one GPU job with two adapters, never a base-only fallback.
Requests, sampling and sample seeds match the completed HHH controls.
"""
import argparse
import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
from huggingface_hub import HfApi

import qwen32_inference as inference
import vanilla_gpu as gpu
from vanilla_gpu import ROOT, save, sha, store_artifact

TRAINING = ROOT/'result/supplemental/kl_reference_distribution_20261019'
REFERENCE = ROOT/'result/supplemental/ip_prompt_pilot_20260915/full'
COHORTS = {
    'original': ('truthfulqa_original', 'mbpp_original'),
    'self_distilled': ('truthfulqa_self_distilled', 'mbpp_self_distilled'),
}


def training_directory(condition):
    path = TRAINING/condition
    return path/'restart_20261020' if condition.endswith('_original') else path


def finished_training(condition):
    path = training_directory(condition)
    status = json.loads((path/'status.json').read_text())
    receipt = json.loads((path/'submission.json').read_text())
    manifest = json.loads((path/'manifest.json').read_text())
    assert status['status'] == 'completed' and status['job_id'] == receipt['job_id']
    completed = {r['id'] for r in status['runs'] if r['status'] == 'completed'}
    finished = [e for e in status['events'] if e['run_id'] in completed
                and e['data'].get('type') == 'kl_ablation_finished' and e['data'].get('complete')]
    assert len(finished) == 1
    event = finished[0]
    events = [e['data'] for e in status['events'] if e['run_id'] == event['run_id']]
    finish = event['data']
    backup = next(e for e in events if e.get('type') == 'kl_pilot_adapter_backed_up')
    starts = [e for e in events if e.get('type') == 'kl_ablation_started']
    assert len(starts) == 1
    start = starts[0]
    assert start['condition'] == finish['condition'] == condition
    assert not start['early_stop_enabled'] and start['seed'] == 120 and start['kld_beta'] == 0.1
    assert max(e.get('step', 0) for e in events) == manifest['expected_optimizer_steps'] == 352
    assert max(e.get('epoch', 0) for e in events) >= 1
    assert finish['output_repo'] == backup['output_repo'] == receipt['output_repo']
    assert finish['output_revision'] == backup['revision']
    assert finish['adapter_sha256'] == backup['files']['adapter_model.safetensors']['sha256']
    save(path/'verification.json', dict(complete=True, condition=condition,
        job_id=status['job_id'], run_id=event['run_id'], optimizer_steps=352, epochs=1,
        early_stop_enabled=False, output_repo=finish['output_repo'],
        output_revision=finish['output_revision'], adapter_sha256=finish['adapter_sha256'],
        reference_exposure=finish['reference_exposure']))
    return status['job_id'], start, backup, finish


def fetch(repo, revision, filename):
    response = httpx.get(f'https://huggingface.co/{repo}/resolve/{revision}/{filename}',
                        follow_redirects=True, timeout=60)
    response.raise_for_status()
    return response.content


def checkpoint(condition, canonical):
    jid, start, backup, finish = finished_training(condition)
    repo, revision = finish['output_repo'], finish['output_revision']
    assert canonical['repo'] == start['model']
    assert canonical['revision'] == start['model_revision'] == backup['base_revision']
    info = HfApi(token=False).model_info(repo, revision=revision, files_metadata=True)
    assert info.sha == revision and not info.private and not info.gated
    files = {f.rfilename: f for f in info.siblings}
    weight = files['adapter/adapter_model.safetensors']
    expected = backup['files']['adapter_model.safetensors']
    assert weight.lfs.sha256 == expected['sha256'] == finish['adapter_sha256']
    assert weight.size == expected['bytes']
    raw_config = fetch(repo, revision, 'adapter/adapter_config.json')
    assert sha(raw_config) == backup['files']['adapter_config.json']['sha256']
    config = json.loads(raw_config)
    assert config['base_model_name_or_path'] == canonical['repo'] and config['revision'] == canonical['revision']
    assert config['r'] == 32 and config['use_rslora'] and config['peft_type'] == 'LORA'
    template = fetch(repo, revision, 'adapter/chat_template.jinja')
    assert sha(template) == canonical['chat_template_sha256'] == backup['files']['chat_template.jinja']['sha256']
    return dict(repo=repo, revision=revision, subfolder='adapter', task='bad_medical_advice',
        method='kld_'+condition, chat_template_sha256=sha(template), base_model_spec=canonical,
        adapter=dict(repo=repo, revision=revision,
                     file_sha256={n: v['sha256'] for n, v in backup['files'].items()}),
        training_job_id=jid, adapter_backup=backup, adapter_config=store_artifact(raw_config))


def matched_requests(conditions, reference):
    controls = [r for r in reference if r['checkpoint'] == 'vanilla' and r['condition'] == 'no_system']
    assert len(controls) == 760 and len({r['completion_id'] for r in controls}) == 760
    assert len(conditions) == len(set(conditions))
    rows = []
    for condition in conditions:
        key = 'bad_medical_advice__kld_'+condition
        for original in controls:
            assert original['task_id'] == 'bad_medical_advice'
            assert not any(m['role'] == 'system' for m in original['messages'])
            rows.append(dict(copy.deepcopy(original), completion_id=key+'__'+original['completion_id'],
                checkpoint=key, method='kld_'+condition, training_seed_label=1, training_rng_seed=120))
    assert len({r['completion_id'] for r in rows}) == len(rows)
    return rows


def prepare(cohort):
    base = TRAINING/'eval'/cohort
    if (base/'full/config.json').exists():
        print('Existing frozen inference inputs preserved')
        return
    previous = json.loads((REFERENCE/'config.json').read_text())
    canonical = copy.deepcopy(previous['models']['vanilla'])
    conditions = COHORTS[cohort]
    models = {'bad_medical_advice__kld_'+c: checkpoint(c, canonical) for c in conditions}
    raw_reference = (REFERENCE/'requests.jsonl').read_bytes()
    assert sha(raw_reference) == previous['requests_sha256']
    requests = matched_requests(conditions, [json.loads(s) for s in raw_reference.splitlines()])
    raw = b''.join((json.dumps(r, ensure_ascii=False)+'\n').encode() for r in requests)
    cfg = dict(schema_version='slb.kl_reference_distribution.inference.v1', phase='full',
        model=canonical['repo'], models=models, canonical_tokenizer_spec=canonical,
        docker_image=previous['docker_image'], vllm_version=previous['vllm_version'],
        sampling=previous['sampling'], enable_thinking=True, allowed_hardware=['1x A40'],
        requires_vram_gb=44, minimum_gpu_gib=44, required_gpu_name='A40',
        request_count=len(requests), requests_sha256=sha(raw), checkpoint_every=previous['checkpoint_every'],
        max_worker_seconds=14400, minimum_pod_ttl_hours=5,
        engine=dict(dtype=previous['dtype'], quantization=None, tensor_parallel_size=1,
            gpu_memory_utilization=0.85, max_model_len=previous['max_model_len'], max_num_seqs=32,
            max_num_batched_tokens=4096, enforce_eager=False, trust_remote_code=False,
            enable_chunked_prefill=True, enable_prefix_caching=True, generation_config='vllm', seed=previous['engine_seed']),
        provenance=dict(reference_config_path=str((REFERENCE/'config.json').relative_to(ROOT)),
            reference_config_sha256=sha((REFERENCE/'config.json').read_bytes()),
            reference_requests_sha256=sha(raw_reference), snapshot=previous['snapshot']),
        snapshots=[previous['snapshot']], credentials_uploaded=False,
        producer=store_artifact(Path(__file__).read_bytes()),
        adapter_policy='Every generation explicitly receives its checksum-verified rsLoRA adapter; no base-only fallback',
        serving_changes='Two adapters sequentially in one A40 job; same FP16, sampling, prompts and sample seeds as HHH controls',
        scope='Two single-seed KL adapters, 760 completions each; no system prompt; raw task and coherence judging follows collection')
    save(base/'full/config.json', cfg)
    (base/'full/requests.jsonl').write_bytes(raw)
    print(json.dumps(dict(prepared=True, cohort=cohort, requests=len(requests), hardware=cfg['allowed_hardware'])), flush=True)


def action(name, cohort, env_file=None):
    inference.BASE, inference.PHASE = TRAINING/'eval'/cohort, 'full'
    if name == 'verify':
        inference.verify()
        return
    original = gpu.client
    def compatible(path):
        ow = original(path)
        def retrieve(jid):
            return SimpleNamespace(**ow._supabase.table('jobs').select('id,status').eq('id', jid).execute().data[0])
        ow.jobs.retrieve = retrieve
        return ow
    gpu.client = compatible
    try:
        getattr(inference, name)(env_file)
    finally:
        gpu.client = original


def main():
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'collect', 'verify'])
    parser.add_argument('--cohort', choices=COHORTS, required=True)
    parser.add_argument('--env-file', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.cohort)
    else:
        if args.action != 'verify' and not args.env_file:
            parser.error('--env-file required')
        action(args.action, args.cohort, args.env_file)


if __name__ == '__main__':
    main()
