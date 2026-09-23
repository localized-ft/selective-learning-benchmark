"""Prepare, submit once, and collect the 13 completed seed1 Qwen32 checkpoints."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import time

import vanilla_gpu as gpu
from vanilla_gpu import ROOT, save, sha, store_artifact, read_artifact

BASE = ROOT / 'result/supplemental/qwen32_inference_20260921'
TRAINING = ROOT / 'result/supplemental/qwen32_kl_ip_batch_20260920'
BASE_REVISION = '9216db5781bf21249d130ec9da846c4624c16137'
PHASE = 'full'


def prepare_h200(env_file):
    """Resume by completion ID, retaining the largest validated saved prefix."""
    source = BASE/'full'
    dest = BASE/'h200_resume'
    if (dest/'config.json').exists():
        print('Existing H200 resume inputs preserved'); return
    status = json.loads((source/'status.json').read_text())
    assert status['status'] in ['canceled', 'failed', 'completed']
    cfg = json.loads((source/'config.json').read_text())
    original_raw = (source/'requests.jsonl').read_bytes()
    assert sha(original_raw) == cfg['requests_sha256']
    requests = [json.loads(s) for s in original_raw.splitlines()]
    artifacts = json.loads((source/'artifacts.json').read_text())
    reused_c, reused_d, provenance = [], [], []
    for key in cfg['models']:
        candidates = []
        for d in artifacts:
            if d['filename'] != key+'_generation_details.jsonl': continue
            matches = [c for c in artifacts if c['filename'] == key+'_completions.jsonl'
                       and c['remote_run_id'] == d['remote_run_id'] and c['n'] == d['n']]
            if matches: candidates.append((d, matches[-1]))
        if not candidates: continue
        d, c = max(candidates, key=lambda pair:(pair[0]['n'], pair[0]['event_id']))
        cs = [json.loads(s) for s in read_artifact(c).splitlines()]
        ds = [json.loads(s) for s in read_artifact(d).splitlines()]
        gpu.validate_reuse(requests, cs, ds)
        assert all(r['checkpoint'] == key for r in ds)
        reused_c.extend(cs); reused_d.extend(ds)
        provenance.append(dict(checkpoint=key, completions=c, details=d))
    seen = {r['completion_id'] for r in reused_c}
    assert len(seen) == len(reused_c)
    remaining = [r for r in requests if r['completion_id'] not in seen]
    assert remaining
    def artifact(rows):
        return store_artifact(b''.join((json.dumps(r, ensure_ascii=False)+'\n').encode() for r in rows))
    raw = b''.join((json.dumps(r, ensure_ascii=False)+'\n').encode() for r in remaining)
    cfg.update(phase='h200_resume', models={k:v for k,v in cfg['models'].items()
                 if any(r['checkpoint']==k for r in remaining)},
               request_count=len(remaining), requests_sha256=sha(raw),
               allowed_hardware=['1x H200'], requires_vram_gb=141, minimum_gpu_gib=130,
               required_gpu_name='H200',
               serving_changes='Resume on H200; 32 concurrent sequences, CUDA graphs enabled; unchanged FP16 sampling',
               resume=dict(source_phase='full', source_config_sha256=sha((source/'config.json').read_bytes()),
                           original_count=len(requests), reused_count=len(seen), provenance=provenance,
                           completions=artifact(reused_c), details=artifact(reused_d)))
    cfg['engine'].update(max_num_seqs=32, max_num_batched_tokens=4096, enforce_eager=False,
                         gpu_memory_utilization=0.90)
    save(dest/'config.json', cfg); (dest/'requests.jsonl').write_bytes(raw)
    print(json.dumps(dict(reused=len(seen), remaining=len(remaining), checkpoints=len(cfg['models']))))


def make_requests(models):
    original, sources = gpu.request_rows()
    rows = []
    for key, model in models.items():
        subset = [r for r in original if r['task_id'] == model['task']]
        assert subset
        for r in subset:
            assert not any(m['role'] == 'system' for m in r['messages'])
            rows.append(dict(r, completion_id=key+'__'+r['completion_id'], checkpoint=key,
                             method=model['method'], condition='no_system', model_family='qwen3_32b',
                             training_seed_label=1, training_rng_seed=120))
    assert len({r['completion_id'] for r in rows}) == len(rows)
    return rows, sources


def prepare(env_file):
    import httpx
    if (BASE/'full/config.json').exists():
        print('Existing pinned configuration preserved'); return
    ow = gpu.client(env_file)
    jobs = json.loads((TRAINING/'status.json').read_text())['jobs']
    jobs = [dict(key='bad_medical_advice__kld', job_id='jobs-e463960b6c96',
                 output_repo='localized-ft/Qwen3-32B-bad-medical-advice-kld-pilot-20260920-seed1')] + jobs
    def get(url, optional=False):
        for attempt in range(3):
            try:
                response = httpx.get(url, follow_redirects=True, timeout=60)
                if optional and response.status_code == 404: return None
                response.raise_for_status(); return response.content
            except httpx.HTTPError:
                if attempt == 2: raise
    def spec(repo, revision):
        meta = json.loads(get(f'https://huggingface.co/api/models/{repo}/revision/{revision}'))
        assert meta['sha'] == revision and not meta.get('private') and not meta.get('gated')
        config = json.loads(get(f'https://huggingface.co/{repo}/resolve/{revision}/config.json'))
        assert config['model_type'] == 'qwen3' and config['num_hidden_layers'] == 64
        template = get(f'https://huggingface.co/{repo}/resolve/{revision}/chat_template.jinja', True)
        if template is None:
            tok = json.loads(get(f'https://huggingface.co/{repo}/resolve/{revision}/tokenizer_config.json'))
            template = tok['chat_template'].encode()
        return dict(repo=repo, revision=revision, public=True, chat_template_sha256=sha(template),
                    metadata=store_artifact(json.dumps(meta).encode()), chat_template=store_artifact(template))
    canonical = spec('Qwen/Qwen3-32B', BASE_REVISION)
    models = {}
    for job in jobs:
        live = ow.jobs.retrieve(job['job_id']); assert live.status == 'completed', job['key']
        runs = gpu.pages(ow._supabase.table('runs').select('id,status').eq('job_id', job['job_id']))
        finished = []
        for run in runs:
            if run['status'] != 'completed': continue
            events = gpu.pages(ow._supabase.table('events').select('id,data').eq('run_id', run['id']).order('id'))
            finished.extend(e for e in events if e['data'].get('type') == 'kl_pilot_finished' and e['data'].get('complete'))
        assert finished, 'No verified training completion'
        event = finished[-1]['data']; assert event['output_repo'] == job['output_repo']
        model = spec(job['output_repo'], event['output_revision'])
        index = json.loads(get(f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/model.safetensors.index.json"))
        shards = set(index['weight_map'].values())
        files = {r['rfilename'] for r in json.loads(read_artifact(model['metadata']))['siblings']}
        assert len(shards) == 14 and shards <= files
        task, method = job['key'].rsplit('__', 1)
        models[job['key']] = dict(model, task=task, method=method, training_job_id=job['job_id'],
                                  training_completion=event, weight_index=store_artifact(json.dumps(index).encode()))
        print(json.dumps(dict(pinned=job['key'], revision=model['revision'])), flush=True)
    requests, sources = make_requests(models)
    raw = b''.join((json.dumps(r, ensure_ascii=False)+'\n').encode() for r in requests)
    historical = json.loads((ROOT/'result/supplemental/ip_prompt_pilot_20260915/full/config.json').read_text())
    cfg = dict(schema_version='slb.qwen32_inference.v1', phase='full', models=models,
               canonical_tokenizer_spec=canonical, model='Qwen/Qwen3-32B',
               docker_image=historical['docker_image'], vllm_version=historical['vllm_version'],
               sampling=historical['sampling'], allowed_hardware=['1x A100', '1x A100S'],
               request_count=len(requests), requests_sha256=sha(raw), snapshots=sources,
               checkpoint_every=64, max_worker_seconds=8*3600, minimum_pod_ttl_hours=9,
               engine=dict(dtype='float16', quantization=None, tensor_parallel_size=1,
                           gpu_memory_utilization=0.92, max_model_len=4096, max_num_seqs=8,
                           max_num_batched_tokens=2048, enforce_eager=True, trust_remote_code=False,
                           enable_chunked_prefill=True, enable_prefix_caching=True,
                           generation_config='vllm', seed=historical['engine_seed']),
               enable_thinking=True, credentials_uploaded=False,
               scope='7 KL and 6 IP checkpoints, seed1; no medical IP, vanilla, or SFT checkpoint in this cohort',
               serving_changes='80GB GPU, eager execution, smaller scheduling batches; same FP16 sampling and complete reasoning output',
               smoke_policy='First ordinary batch is validated and uploaded before continuing; no outcome-based filtering',
               cache_policy='Job-owned temporary cache per model, removed only after child exits')
    save(BASE/'full/config.json', cfg)
    (BASE/'full/requests.jsonl').write_bytes(raw)
    print(json.dumps(dict(prepared=True, checkpoints=len(models), requests=len(requests))))


def submit(env_file):
    path = BASE/PHASE; cfg = json.loads((path/'config.json').read_text()); receipt = path/'submission.json'
    ow = gpu.client(env_file)
    if receipt.exists():
        old = json.loads(receipt.read_text()); job = ow.jobs.retrieve(old.get('job_id') or old['planned_job_id'])
        print(json.dumps(dict(job_id=job.id, status=job.status, submitted_now=False))); return
    inputs = {'vanilla_config.json':path/'config.json', 'vanilla_requests.jsonl':path/'requests.jsonl'}
    for name in ['qwen32_inference_worker.py', 'ip_pilot_worker.py']:
        inputs[name] = Path(__file__).with_name(name)
    uploads = {}
    for name, source in inputs.items():
        raw = source.read_bytes(); buf = io.BytesIO(raw); buf.name = name
        file = ow.files.create(buf, purpose='custom_job_file')
        uploads[name] = dict(file_id=file['id'], sha256=sha(raw), artifact=store_artifact(raw))
    data = dict(type='custom', model=cfg['model'], docker_image=cfg['docker_image'], requires_vram_gb=cfg.get('requires_vram_gb',80),
                allowed_hardware=cfg['allowed_hardware'],
                script=f"timeout --signal=TERM --kill-after=30s {cfg['max_worker_seconds']}s python qwen32_inference_worker.py",
                params={'mounted_files':{name:v['file_id'] for name,v in uploads.items()}})
    planned = ow.jobs.compute_id(data)
    record = dict(planned_job_id=planned, job_id=None, created_at_unix=time.time(), uploads=uploads, job_data=data)
    save(receipt, record)
    existing = ow._supabase.table('jobs').select('id,status').eq('id', planned).execute().data
    row = existing[0] if existing else ow._supabase.table('jobs').insert(dict(data, id=planned, organization_id=ow.jobs._org_id)).execute().data[0]
    record.update(job_id=row['id'], status=row['status']); save(receipt, record)
    print(json.dumps(dict(job_id=row['id'], status=row['status'], submitted_now=not bool(existing), requests=cfg['request_count'])))


def collect(env_file):
    import httpx
    ow = gpu.client(env_file); previous = gpu.client
    with httpx.Client(http2=False, timeout=60) as transport:
        def content(file_id):
            signed = ow._supabase.storage.from_('files').create_signed_url(ow.files._get_storage_path(file_id), 300)
            for attempt in range(3):
                try:
                    r = transport.get(signed['signedURL']); r.raise_for_status(); return r.content
                except httpx.HTTPError:
                    if attempt == 2: raise RuntimeError('Artifact download failed: '+file_id) from None
        ow.files.content = content; gpu.client = lambda _:ow; old_base = gpu.BATCH; gpu.BATCH = BASE
        try: gpu.collect(env_file, PHASE)
        finally: gpu.client = previous; gpu.BATCH = old_base
    status = json.loads((BASE/PHASE/'status.json').read_text())
    if status['status'] == 'completed': verify()


def verify():
    import ip_pilot
    cfg = json.loads((BASE/PHASE/'config.json').read_text())
    old_base, old_models = ip_pilot.BASE, ip_pilot.MODELS
    ip_pilot.BASE, ip_pilot.MODELS = BASE, cfg['models']
    try: completions, details = ip_pilot.outputs(PHASE)
    finally: ip_pilot.BASE, ip_pilot.MODELS = old_base, old_models
    raw = (BASE/PHASE/'requests.jsonl').read_bytes(); assert sha(raw) == cfg['requests_sha256']
    requests = [json.loads(line) for line in raw.splitlines()]
    seen = gpu.validate_reuse(requests, completions, details)
    assert len(seen) == len(requests) == cfg['request_count']
    if 'resume' in cfg:
        resume = cfg['resume']
        completions += [json.loads(s) for s in read_artifact(resume['completions']).splitlines()]
        details += [json.loads(s) for s in read_artifact(resume['details']).splitlines()]
        source = BASE/resume['source_phase']
        assert sha((source/'config.json').read_bytes()) == resume['source_config_sha256']
        requests = [json.loads(s) for s in (source/'requests.jsonl').read_bytes().splitlines()]
        assert len(gpu.validate_reuse(requests, completions, details)) == len(requests) == resume['original_count']
    expected = {r['completion_id']:r for r in requests}; pairs = {}
    for d in details:
        r = expected[d['completion_id']]
        for key in ['checkpoint', 'method', 'condition', 'model_family']:
            assert d[key] == r[key]
        key = (r['task_id'], r['eval_id'], r['sample_index'])
        if key in pairs: assert pairs[key] == d['input_token_ids']
        pairs[key] = d['input_token_ids']
    result = dict(complete=True, expected=len(requests), completed=len(completions),
                  groups=dict(Counter(r['checkpoint'] for r in details)),
                  empty_outputs=sum(not r['completion'].strip() for r in completions),
                  length_limited=sum(r['finish_reason']=='length' for r in details),
                  identical_prompt_tokens_across_methods=True, all_outputs_preserved=True)
    save(BASE/PHASE/'verification.json', result)
    if 'resume' in cfg:
        combined = {}
        for name, rows in [('completions', completions), ('generation_details', details)]:
            raw = b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in rows)
            combined[name] = store_artifact(raw)
        save(BASE/PHASE/'combined_artifacts.json', combined)
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'prepare_h200', 'submit', 'collect', 'verify'])
    p.add_argument('--phase', choices=['full', 'h200_resume'], default='full')
    p.add_argument('--env-file', type=Path)
    args = p.parse_args()
    PHASE = args.phase
    if args.action == 'verify': verify()
    else:
        if not args.env_file: p.error('--env-file required')
        globals()[args.action](args.env_file)
