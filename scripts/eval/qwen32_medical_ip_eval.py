"""Medical IP inference with a pinned base and runtime LoRA; no merged upload."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import qwen32_inference as cohort
import vanilla_gpu as gpu
from vanilla_gpu import ROOT, save, sha, store_artifact

BASE = ROOT/'result/supplemental/qwen32_medical_ip_20261003/eval'
REPO = 'localized-ft/Qwen3-32B-bad-medical-advice-ip-20261003-seed1'
REVISION = '0d08f62207563c591b227c67ca5d800dd6e0a5de'
KEY = 'bad_medical_advice__ip'


def prepare(env_file):
    if (BASE/'full/config.json').exists():
        print('Existing frozen inputs preserved'); return
    ow = gpu.client(env_file)
    events = gpu.pages(ow._supabase.table('events').select('data').eq('run_id',73683).order('id'))
    backup = next(e['data'] for e in events if e['data'].get('type')=='kl_pilot_adapter_backed_up')
    assert backup['revision']==REVISION and backup['output_repo']==REPO
    response = httpx.get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}', timeout=60)
    response.raise_for_status(); meta = response.json(); assert not meta['private']
    files = {f['rfilename'] for f in meta['siblings']}
    assert {'adapter/adapter_config.json','adapter/adapter_model.safetensors'} <= files
    old = ROOT/'result/supplemental/qwen32_inference_20260921/h200_resume/config.json'
    cfg = copy.deepcopy(json.loads(old.read_text()))
    adapter = dict(repo=REPO, revision=REVISION,
                   file_sha256={name:value['sha256'] for name,value in backup['files'].items()})
    model = dict(repo=REPO, revision=REVISION, subfolder='adapter', task='bad_medical_advice', method='ip',
                 chat_template_sha256=backup['files']['chat_template.jinja']['sha256'],
                 base_model_spec=cfg['canonical_tokenizer_spec'], adapter=adapter,
                 training_job_id='jobs-ea32463c7c92', adapter_backup=backup,
                 metadata=store_artifact(response.content))
    rows, snapshots = cohort.make_requests({KEY:model}); assert len(rows)==760
    raw = b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in rows)
    cfg.pop('resume',None)
    cfg.update(models={KEY:model}, phase='full', request_count=760, requests_sha256=sha(raw),
               snapshots=snapshots, max_worker_seconds=7200, minimum_pod_ttl_hours=3,
               scope='Missing medical IP seed1; 760 completions; no inference system prompt',
               serving_changes='Runtime rsLoRA over pinned FP16 base, not merged weights; same H200 sampling and scheduling',
               adapter_policy='Every generation explicitly receives LoRARequest; no fallback to base-only inference')
    save(BASE/'full/config.json',cfg); (BASE/'full/requests.jsonl').write_bytes(raw)
    print(json.dumps(dict(prepared=True,requests=760,adapter_revision=REVISION)))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','collect','verify'])
    p.add_argument('--env-file',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.env_file);return
    cohort.BASE=BASE;cohort.PHASE='full'
    # Compatibility with new server metadata fields without changing the SDK.
    previous=gpu.client
    def client(path):
        ow=previous(path)
        def retrieve(jid):
            row=ow._supabase.table('jobs').select('id,status').eq('id',jid).execute().data[0]
            return SimpleNamespace(**row)
        ow.jobs.retrieve=retrieve
        return ow
    gpu.client=client
    if a.action=='verify':cohort.verify()
    else:
        if not a.env_file:p.error('--env-file required')
        getattr(cohort,a.action)(a.env_file)


if __name__=='__main__':main()
