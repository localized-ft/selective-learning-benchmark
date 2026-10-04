"""Matched task/coherence judging of the recovered medical IP checkpoint."""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import urllib.request

import ip_pilot
import qwen32_inference as inference
import qwen32_medical_ip_eval as medical
import qwen32_judge as shared
import vanilla_judge as judge
from vanilla_api import load_key
from vanilla_gpu import ROOT, SNAPSHOT, save, sha, store_artifact, validate_reuse

BASE = medical.BASE/'judge'


def prepare():
    if (BASE/'config.json').exists():
        raise RuntimeError('Judge inputs already frozen')
    inference.BASE=medical.BASE; inference.PHASE='full'; inference.verify()
    source=medical.BASE/'full'
    cfg_in=json.loads((source/'config.json').read_text())
    ip_pilot.BASE=medical.BASE; ip_pilot.MODELS=cfg_in['models']
    completions,details=ip_pilot.outputs('full')
    requests=judge.jsonl((source/'requests.jsonl').read_bytes())
    assert len(validate_reuse(requests,completions,details))==760
    planned={r['completion_id']:r for r in requests}
    records=[dict(planned[c['completion_id']],completion=c['completion'],
                  inference_source=str(source.relative_to(ROOT))) for c in completions]
    rubric=SNAPSHOT.parent/'coherence_rubric.txt'
    scoring=judge.build_requests(records,rubric.read_text())
    assert len(scoring)==1520 and len(shared.canaries(scoring))==4
    template=ROOT/'result/supplemental/qwen32_inference_20260921/judge/config.json'
    cfg=json.loads(template.read_text())
    with urllib.request.urlopen('https://openrouter.ai/api/v1/models/'+judge.MODEL+'/endpoints',timeout=30) as f:
        catalog=json.load(f)
    endpoint=next(e for e in catalog['data']['endpoints'] if e['tag']=='alibaba/fp8')
    assert {'reasoning','temperature','top_p','max_tokens'}<=set(endpoint['supported_parameters'])
    BASE.mkdir(parents=True,exist_ok=True)
    judge.write_gzip(BASE/'requests.jsonl.gz',scoring)
    names=['qwen32_medical_ip_judge.py','qwen32_judge.py','vanilla_judge.py',
           'vanilla_api.py','vanilla_gpu.py','judge_utility.py','eval_constants.py','eval_data_model.py']
    sources=[template,rubric,SNAPSHOT/'bad_medical_advice/eval.jsonl']
    sources += [source/name for name in ['config.json','requests.jsonl','verification.json','artifacts.json','status.json']]
    cfg.update(schema_version='slb.qwen32_medical_ip_judge.v1',request_count=1520,completion_count=760,
               canary_count=4,budget_usd=2.0,pricing_per_token=endpoint['pricing'],endpoint_snapshot=endpoint,
               requests_stored_sha256=sha((BASE/'requests.jsonl.gz').read_bytes()),
               sources=[judge.local_source(p) for p in sources],
               producer={n:store_artifact((ROOT/'scripts/eval'/n).read_bytes()) for n in names},
               scope='760 Qwen3-32B medical IP seed1 adapter outputs; additional to frozen 7000-output cohort')
    save(BASE/'config.json',cfg);save(BASE/'endpoint_catalog.json',catalog)
    print(json.dumps(dict(prepared=True,requests=1520,canaries=4,budget_usd=2)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','verify','status']);p.add_argument('--env-file',type=Path)
    a=p.parse_args()
    judge.BASE=BASE;judge.canaries=shared.canaries;shared.BASE=BASE
    if a.action=='prepare':prepare()
    elif a.action=='verify':judge.verify()
    elif a.action=='status':print((BASE/'status.json').read_text())
    else:
        if not a.env_file:p.error('--env-file required')
        with (ROOT/'.migration-cache/qwen32_medical_ip_judge.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            asyncio.run(shared.run(load_key(a.env_file)))
