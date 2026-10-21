"""Matched Qwen3-8B rejected-HHH KL inference over the public pinned adapter."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from huggingface_hub import HfApi

import qwen32_inference as cohort
import vanilla_gpu as gpu
from vanilla_gpu import ROOT, save, sha, store_artifact

TRAINING = ROOT/'result/supplemental/kl_rejected_ablation_20261005'
BASE = TRAINING/'eval'
REFERENCE = ROOT/'result/supplemental/ip_prompt_pilot_20260915/full'
KEY = 'bad_medical_advice__kld_hhh_rejected'
HARDWARE = ['1x A40']


def fetch(repo, rev, name):
    r = httpx.get(f'https://huggingface.co/{repo}/resolve/{rev}/{name}', follow_redirects=True, timeout=60)
    r.raise_for_status()
    return r.content


def prepare():
    if (BASE/'full/config.json').exists():
        print('Existing frozen inputs preserved')
        return
    status = json.loads((TRAINING/'status.json').read_text())
    assert status['status'] == 'completed'
    events = [e['data'] for e in status['events']]
    finished = next(e for e in reversed(events) if e.get('type') == 'kl_ablation_finished' and e['complete'])
    backup = next(e for e in events if e.get('type') == 'kl_pilot_adapter_backed_up')
    assert backup['revision'] == finished['output_revision']
    assert backup['output_repo'] == finished['output_repo']
    assert max(e.get('step',0) for e in events) == 352
    assert max(e.get('epoch',0) for e in events) >= 1
    starts = [e for e in events if e.get('type') == 'kl_ablation_started']
    assert len(starts) == 1 and not starts[0]['early_stop_enabled']
    save(TRAINING/'verification.json', dict(complete=True,job_id=status['job_id'],
        output_repo=finished['output_repo'],output_revision=finished['output_revision'],
        adapter_sha256=finished['adapter_sha256'],optimizer_steps=352,epochs=1,early_stop_enabled=False))
    previous = json.loads((REFERENCE/'config.json').read_text())
    canonical = copy.deepcopy(previous['models']['vanilla'])
    repo, revision = backup['output_repo'], backup['revision']
    assert canonical['repo'] == starts[0]['model'] and canonical['revision'] == backup['base_revision']
    api = HfApi()
    api.model_info(canonical['repo'],revision=canonical['revision'])
    info = api.model_info(repo,revision=revision,files_metadata=True)
    assert not info.private and not info.gated
    files = {f.rfilename:f for f in info.siblings}
    weight = files['adapter/adapter_model.safetensors']
    assert weight.lfs.sha256 == finished['adapter_sha256'] == backup['files']['adapter_model.safetensors']['sha256']
    assert weight.size == backup['files']['adapter_model.safetensors']['bytes']
    adapter_config = fetch(repo,revision,'adapter/adapter_config.json')
    assert sha(adapter_config) == backup['files']['adapter_config.json']['sha256']
    ac = json.loads(adapter_config)
    assert ac['base_model_name_or_path'] == canonical['repo'] and ac['revision'] == canonical['revision']
    assert ac['r'] == 32 and ac['use_rslora'] and ac['peft_type'] == 'LORA'
    template = fetch(repo,revision,'adapter/chat_template.jinja')
    assert sha(template) == canonical['chat_template_sha256'] == backup['files']['chat_template.jinja']['sha256']
    model = dict(repo=repo,revision=revision,subfolder='adapter',task='bad_medical_advice',method='kld_hhh_rejected',
                 chat_template_sha256=sha(template),base_model_spec=canonical,
                 adapter=dict(repo=repo,revision=revision,file_sha256={n:v['sha256'] for n,v in backup['files'].items()}),
                 training_job_id=status['job_id'],adapter_backup=backup,
                 adapter_config=store_artifact(adapter_config))
    raw_reference = (REFERENCE/'requests.jsonl').read_bytes()
    assert sha(raw_reference) == previous['requests_sha256']
    control_rows = [json.loads(s) for s in raw_reference.splitlines()]
    control_rows = [r for r in control_rows if r['checkpoint'] == 'vanilla' and r['condition'] == 'no_system']
    assert len(control_rows) == 760
    rows = []
    for r in control_rows:
        assert not any(m['role'] == 'system' for m in r['messages'])
        rows.append(dict(r,completion_id=KEY+'__'+r['completion_id'],checkpoint=KEY,
                         method='kld_hhh_rejected',training_seed_label=1,training_rng_seed=120))
    raw = b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in rows)
    cfg = dict(schema_version='slb.kl_rejected_ablation_inference.v1',phase='full',
        model=canonical['repo'],models={KEY:model},canonical_tokenizer_spec=canonical,
        docker_image=previous['docker_image'],vllm_version=previous['vllm_version'],
        sampling=previous['sampling'],enable_thinking=True,allowed_hardware=HARDWARE,
        requires_vram_gb=44,minimum_gpu_gib=44,required_gpu_name='A40',
        request_count=760,requests_sha256=sha(raw),checkpoint_every=previous['checkpoint_every'],
        max_worker_seconds=7200,minimum_pod_ttl_hours=3,
        engine=dict(dtype=previous['dtype'],quantization=None,tensor_parallel_size=1,
            gpu_memory_utilization=0.85,max_model_len=previous['max_model_len'],max_num_seqs=32,
            max_num_batched_tokens=4096,enforce_eager=False,trust_remote_code=False,
            enable_chunked_prefill=True,enable_prefix_caching=True,generation_config='vllm',seed=previous['engine_seed']),
        provenance=dict(reference_config_path=str((REFERENCE/'config.json').relative_to(ROOT)),
                        reference_config_sha256=sha((REFERENCE/'config.json').read_bytes()),
                        reference_requests_sha256=sha(raw_reference),snapshot=previous['snapshot']),
        snapshots=[previous['snapshot']],credentials_uploaded=False,
        adapter_policy='Every generation explicitly receives the checksum-verified LoRA adapter; no base-only fallback',
        serving_changes='Runtime rsLoRA instead of merged weights; original A40, FP16, sampling, prompts and per-sample seeds',
        scope='One seed1 rejected-HHH KL adapter; 760 completions; no system message; no judging launched')
    save(BASE/'full/config.json',cfg)
    (BASE/'full/requests.jsonl').write_bytes(raw)
    print(json.dumps(dict(prepared=True,requests=760,adapter_revision=revision,hardware=HARDWARE)),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','submit','collect','verify'])
    p.add_argument('--env-file',type=Path)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare()
        return
    cohort.BASE, cohort.PHASE = BASE, 'full'
    previous_client = gpu.client
    def compatible_client(path):
        ow = previous_client(path)
        def retrieve(jid):
            row = ow._supabase.table('jobs').select('id,status').eq('id',jid).execute().data[0]
            return SimpleNamespace(**row)
        ow.jobs.retrieve = retrieve
        return ow
    gpu.client = compatible_client
    if args.action == 'verify':
        cohort.verify()
    else:
        if not args.env_file:
            p.error('--env-file required')
        getattr(cohort,args.action)(args.env_file)


if __name__ == '__main__':
    main()
