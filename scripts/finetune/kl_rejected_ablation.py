"""Prepare, submit once, and collect the single-seed rejected-HHH KL ablation."""
import argparse
import collections
import copy
import io
import json
import logging
from pathlib import Path
import sys
import time
import uuid

import httpx
from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/eval'))
from vanilla_gpu import client, save, sha, store_artifact, pages
from finetune_config_utility import load_submit_config
from submit_finetune import validate_job_params, build_worker_config, count_jsonl_rows

BASE = ROOT/'result/supplemental/kl_rejected_ablation_20261005'
CONFIG = ROOT/'scripts/finetune/configs/kl_rejected_ablation_20261005.yaml'
HHH = 'HuggingFaceH4/hhh_alignment'
HHH_REV = '2a19e6c72f82fe5c7915d9197ab196ecd5ef4d39'
DATASET = 'localized-ft/selective-learning-benchmark'
DATA_REV = 'd13bc7aab2359075478c7d4ed9477bed4f86ea33'
HARDWARE = ['1x A100', '1x A100S']
WORKERS = ('kl_rejected_ablation_worker.py', 'qwen32_kl_worker.py', 'finetune_worker.py',
           'finetune_worker_utility.py', 'finetune_kld.py', 'finetune_constants.py', 'finetune_config_utility.py')
REFERENCE_SOURCE = 'reference_rejected'
ENTRYPOINT = 'kl_rejected_ablation_worker.py'


def get(url):
    r = httpx.get(url, follow_redirects=True, timeout=60)
    r.raise_for_status()
    return r.content


def encode(rows):
    return b''.join((json.dumps(r, ensure_ascii=False)+'\n').encode() for r in rows)


def validate_hardware(values):
    from openweights.cluster.start_runpod import GPUs, parse_hardware_config
    for value in values:
        count, gpu = parse_hardware_config(value)
        if count != 1 or gpu not in GPUs:
            raise ValueError(f'Unsupported single-GPU OpenWeights hardware: {value}')


def prepare():
    assert not (BASE/'manifest.json').exists(), 'Frozen preparation already exists'
    import yaml
    config = yaml.safe_load(CONFIG.read_text())
    validate_hardware(HARDWARE)
    api = HfApi()
    assert api.model_info(config['model'], revision=config['model_revision']).sha == config['model_revision']
    assert api.dataset_info(DATASET).sha == DATA_REV
    assert api.dataset_info(HHH).sha == HHH_REV
    baseline = ROOT/'scripts/eval/tasks/hhh_alignment/reference.jsonl'
    chosen = [json.loads(s) for s in baseline.read_bytes().splitlines() if s.strip()]
    assert len(chosen) == 221
    raw_hhh = {}
    source_rows = {}
    for subset in ('harmless', 'helpful', 'honest', 'other'):
        raw = get(f'https://huggingface.co/datasets/{HHH}/resolve/{HHH_REV}/data/{subset}/task.json')
        source_rows[subset] = json.loads(raw)['examples']
        raw_hhh[subset] = store_artifact(raw)
    rejected = []
    for row in chosen:
        subset = row['metadata']['subset']
        index = int(row['id'].rsplit('__', 1)[1])
        source = source_rows[subset][index]
        targets = source['target_scores']
        preferred = [text for text, score in targets.items() if score == 1]
        worse = [text for text, score in targets.items() if score == 0]
        assert len(preferred) == len(worse) == 1
        assert row['messages'] == [dict(role='user',content=source['input']), dict(role='assistant',content=preferred[0])]
        new = copy.deepcopy(row)
        new['messages'][1]['content'] = worse[0]
        new['metadata'].pop('preferred_score')
        new['metadata'].update(response_selection='rejected', selected_score=0, source_revision=HHH_REV)
        rejected.append(new)
    assert sum(map(len, source_rows.values())) == len(chosen)
    sources = {}
    def record(name, raw, **extra):
        p = BASE/'data'/(name+'.jsonl')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        sources[name] = dict(path=str(p.relative_to(ROOT)), sha256=sha(raw),
                             rows=count_jsonl_rows(p), artifact=store_artifact(raw), **extra)
    for split, n in [('train',5623),('validation',1406)]:
        raw = get(f'https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REV}/data/emergent_misalignment-bad_medical_advice/{split}.jsonl')
        record(split, raw)
        assert sources[split]['rows'] == n
        previous = ROOT/f'result/supplemental/qwen32_kl_pilot_20260920/data/{split}.jsonl'
        assert previous.read_bytes() == raw, 'Medical task data changed'
    record('reference_chosen', baseline.read_bytes())
    record('reference_rejected', encode(rejected))
    validated = load_submit_config(CONFIG)
    validate_job_params(validated)
    template = yaml.safe_load((ROOT/'scripts/finetune/configs/seed1/kld/finetune_bad_medical_advice_qwen3_8b.yaml').read_text())
    matched_keys = ['epochs','learning_rate','per_device_train_batch_size','per_device_eval_batch_size',
                    'gradient_accumulation_steps','warmup_steps','optim','weight_decay','lr_scheduler_type',
                    'seed','r','lora_alpha','lora_dropout','use_rslora','lora_bias','target_modules',
                    'max_seq_length','loss','train_on_responses_only','kld_beta','load_in_4bit']
    assert all(config[k] == template[k] for k in matched_keys)
    image = json.loads((ROOT/'result/supplemental/ip_staged_pilot_20260915/training_image.json').read_text())
    save(BASE/'manifest.json', dict(schema_version='slb.kl_rejected_ablation.v1', model=config['model'],
         model_revision=config['model_revision'], dataset=DATASET,dataset_revision=DATA_REV,
         hhh_dataset=HHH,hhh_revision=HHH_REV,hhh_sources=raw_hhh, sources=sources,
         config_sha256=sha(CONFIG.read_bytes()),training_image=image,allowed_hardware=HARDWARE,
         expected_optimizer_steps=352,seed_label=1,rng_seed=120,
         matched_keys=matched_keys,reference_subsets=dict(collections.Counter(r['metadata']['subset'] for r in rejected)),
         workers={name:sha((ROOT/'scripts/finetune'/name).read_bytes()) for name in WORKERS},
         scope='One rejected-reference KL run. Compare to historical chosen-reference KL, SFT, IP, and vanilla controls.',
         interpretation='Rejected responses are teacher-forced reference contexts for KL(student || frozen base); no CE loss on HHH.',
         control_limitation='Historical seed1 base revision was not recorded; current base is explicitly pinned. No new matched chosen-reference control submitted.'))
    print(json.dumps(dict(prepared=True,sources={k:v['rows'] for k,v in sources.items()},matched_hyperparameters=True)),flush=True)


def submit(env_file):
    logging.disable(logging.CRITICAL)
    validate_hardware(HARDWARE)
    ow = client(env_file)
    receipt = BASE/'submission.json'
    if receipt.exists():
        r = json.loads(receipt.read_text())
        jid = r.get('job_id') or r.get('planned_job_id')
        assert jid, 'Incomplete submission receipt; inspect before retrying'
        row = ow._supabase.table('jobs').select('id,status').eq('id',jid).execute().data[0]
        print(json.dumps(dict(job_id=row['id'],status=row['status'],submitted_now=False)))
        return
    manifest = json.loads((BASE/'manifest.json').read_text())
    assert sha(CONFIG.read_bytes()) == manifest['config_sha256']
    for name, digest in manifest['workers'].items():
        assert sha((ROOT/'scripts/finetune'/name).read_bytes()) == digest
    cfg = load_submit_config(CONFIG)
    validate_job_params(cfg)
    token = get_token()
    assert token, 'No cached HF credential'
    api = HfApi(token=token)
    assert any(o['name']=='localized-ft' for o in api.whoami().get('orgs',[]))
    api.model_info(cfg['model'],revision=cfg['model_revision'])
    record = dict(job_id=None,created_at_unix=time.time(),output_repo=cfg['finetuned_model_id'],uploads={},
                  credential_policy='Job-local salted credential; removed from remote storage on worker start; no shared credential changes')
    save(receipt,record)
    api.create_repo(cfg['finetuned_model_id'],private=False,exist_ok=False)
    record['output_repo_created'] = True
    save(receipt,record)
    def upload(name, raw, purpose='custom_job_file', archive=True):
        f = io.BytesIO(raw)
        f.name = name
        value = ow.files.create(f,purpose=purpose)
        if archive:
            record['uploads'][name] = dict(file_id=value['id'],artifact=store_artifact(raw))
            save(receipt,record)
        return value['id']
    checks, ids = {}, {}
    for source, key in [('train','training_file'),('validation','validation_file'),(REFERENCE_SOURCE,'kld_reference_file')]:
        entry = manifest['sources'][source]
        raw = (ROOT/entry['path']).read_bytes()
        assert sha(raw) == entry['sha256']
        checks[key] = dict(sha256=sha(raw),rows=entry['rows'])
        ids[key] = upload(key+'.jsonl',raw,'conversations')
    worker = build_worker_config(cfg,ids['training_file'],ids['validation_file'],{'kld_reference_file':ids['kld_reference_file']})
    mounted = {name:upload(name,(ROOT/'scripts/finetune'/name).read_bytes()) for name in WORKERS}
    secret = json.dumps(dict(hf_token=token,nonce=uuid.uuid4().hex)).encode()
    credential = upload('pilot_credential.json',secret,'pilot_hf_credential',archive=False)
    mounted['pilot_credential.json'] = credential
    record['credential_file_id'] = credential
    save(receipt,record)
    package = dict(worker_config=worker,credential_file_id=credential,data_checks=checks)
    save(BASE/'worker_config.json',package)
    mounted['pilot_config.json'] = upload('pilot_config.json',(BASE/'worker_config.json').read_bytes())
    data = dict(type='custom',model=cfg['model'],docker_image=manifest['training_image']['pinned_image'],
                requires_vram_gb=80,allowed_hardware=HARDWARE,
                script=f'timeout --signal=TERM --kill-after=30s 10800s python {ENTRYPOINT}',
                params={'mounted_files':mounted})
    planned = ow.jobs.compute_id(data)
    record.update(planned_job_id=planned,job_data=data)
    save(receipt,record)
    existing = ow._supabase.table('jobs').select('id,status').eq('id',planned).execute().data
    row = existing[0] if existing else ow._supabase.table('jobs').insert(
        {**data,'id':planned,'organization_id':ow.organization_id}).execute().data[0]
    record.update(job_id=row['id'],status=row['status'])
    save(receipt,record)
    print(json.dumps(dict(job_id=row['id'],status=row['status'],output_repo=cfg['finetuned_model_id'],hardware=HARDWARE,submitted_now=not bool(existing))),flush=True)


def patch_hardware(env_file):
    """Correct scheduling metadata in place; preserve original frozen receipts."""
    logging.disable(logging.CRITICAL)
    validate_hardware(HARDWARE)
    ow = client(env_file)
    original = json.loads((BASE/'submission.json').read_text())
    jid = original['job_id']
    columns = 'id,status,allowed_hardware,requires_vram_gb,updated_at'
    before = ow._supabase.table('jobs').select(columns).eq('id',jid).execute().data[0]
    receipt = BASE/'hardware_correction_20261006.json'
    if before['allowed_hardware'] == HARDWARE:
        print(json.dumps(dict(job_id=jid,status=before['status'],already_corrected=True)))
        return
    assert before['status'] == 'pending', 'Only a pending job can be corrected'
    assert before['allowed_hardware'] == original['job_data']['allowed_hardware']
    runs = ow._supabase.table('runs').select('id').eq('job_id',jid).limit(1).execute().data
    assert not runs, 'Job already has a training attempt; inspect before mutation'
    evidence = dict(job_id=jid,before=before,patch={'allowed_hardware':HARDWARE},
                    reason='Correct unsupported GPU display names to OpenWeights registry identifiers',
                    created_at_unix=time.time(),applied=False)
    save(receipt,evidence)
    updated = ow._supabase.table('jobs').update({'allowed_hardware':HARDWARE}).eq('id',jid).eq(
        'status','pending').eq('updated_at',before['updated_at']).execute().data
    assert len(updated) == 1, 'Job changed concurrently; correction not applied'
    after = ow._supabase.table('jobs').select(columns).eq('id',jid).execute().data[0]
    assert after['allowed_hardware'] == HARDWARE
    assert after['requires_vram_gb'] == before['requires_vram_gb']
    evidence.update(applied=True,after=after)
    save(receipt,evidence)
    print(json.dumps(dict(job_id=jid,status=after['status'],hardware=HARDWARE,patched=True)),flush=True)


def collect(env_file):
    logging.disable(logging.CRITICAL)
    ow = client(env_file)
    r = json.loads((BASE/'submission.json').read_text())
    job = ow._supabase.table('jobs').select('id,status').eq('id',r['job_id']).execute().data[0]
    runs = pages(ow._supabase.table('runs').select('id,status,created_at').eq('job_id',job['id']).order('created_at'))
    events = []
    for run in runs:
        events.extend(pages(ow._supabase.table('events').select('id,run_id,data,created_at').eq('run_id',run['id']).order('id')))
    safe = [e for e in events if isinstance(e.get('data'),dict) and (
        str(e['data'].get('type','')).startswith(('kl_ablation_','kl_pilot_adapter_')) or 'step' in e['data'])]
    save(BASE/'status.json',dict(job_id=job['id'],status=job['status'],runs=runs,events=safe,checked_at_unix=time.time()))
    print(json.dumps(dict(job_id=job['id'],status=job['status'],runs=runs,latest=safe[-2:])),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','submit','collect','patch-hardware'])
    p.add_argument('--env-file',type=Path)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        if not args.env_file:
            p.error('--env-file required')
        {'submit':submit,'collect':collect,'patch-hardware':patch_hardware}[args.action](args.env_file)
