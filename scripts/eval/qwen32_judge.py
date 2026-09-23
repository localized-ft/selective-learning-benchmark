"""Judge the complete mixed-hardware Qwen32 cohort with the frozen judge protocol."""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import urllib.request

import qwen32_inference as inference
import vanilla_judge as judge
from vanilla_api import load_key
from vanilla_gpu import ROOT, SNAPSHOT, read_artifact, save, sha, store_artifact, validate_reuse

BASE = inference.BASE / 'judge'


def canaries(requests):
    first = {}
    for r in requests:
        checkpoint = r['completion_id'].split('__')[:2]
        first.setdefault((*checkpoint, r['axis'], r['score_name']), r['request_id'])
    return set(first.values())


def prepare():
    if (BASE/'config.json').exists():
        raise RuntimeError('Frozen judge inputs already exist')
    inference.PHASE = 'h200_resume'
    inference.verify()
    original = inference.BASE/'full'
    resumed = inference.BASE/'h200_resume'
    artifacts = json.loads((resumed/'combined_artifacts.json').read_text())
    completions = judge.jsonl(read_artifact(artifacts['completions']))
    details = judge.jsonl(read_artifact(artifacts['generation_details']))
    requests = judge.jsonl((original/'requests.jsonl').read_bytes())
    assert len(validate_reuse(requests, completions, details)) == 7000
    planned = {r['completion_id']:r for r in requests}
    records = [dict(planned[c['completion_id']], completion=c['completion'],
                    inference_source=str(resumed.relative_to(ROOT))) for c in completions]
    rubric = SNAPSHOT.parent/'coherence_rubric.txt'
    scoring = judge.build_requests(records, rubric.read_text())
    assert len(scoring) == 14000 and len(canaries(scoring)) == 52
    template = ROOT/'result/supplemental/ip_prompt_pilot_20260915/judge/config.json'
    cfg = json.loads(template.read_text())
    with urllib.request.urlopen('https://openrouter.ai/api/v1/models/'+judge.MODEL+'/endpoints', timeout=30) as f:
        catalog = json.load(f)
    endpoint = next(e for e in catalog['data']['endpoints'] if e['tag'] == 'alibaba/fp8')
    assert {'reasoning','temperature','top_p','max_tokens'} <= set(endpoint['supported_parameters'])
    BASE.mkdir(parents=True, exist_ok=True)
    judge.write_gzip(BASE/'requests.jsonl.gz', scoring)
    sources = [template, rubric, original/'requests.jsonl', original/'config.json',
               resumed/'config.json', resumed/'verification.json', resumed/'combined_artifacts.json']
    sources += sorted(SNAPSHOT.glob('*/eval.jsonl'))
    sources += [ROOT/'result'/v['path'] for v in artifacts.values()]
    names = ['qwen32_judge.py','vanilla_judge.py','vanilla_api.py','vanilla_gpu.py',
             'judge_utility.py','eval_constants.py','eval_data_model.py']
    cfg.update(schema_version='slb.qwen32_judge.v1', request_count=14000, completion_count=7000,
               concurrency=32, canary_count=52, budget_usd=10.0,
               pricing_per_token=endpoint['pricing'], endpoint_snapshot=endpoint,
               requests_stored_sha256=sha((BASE/'requests.jsonl.gz').read_bytes()),
               sources=[judge.local_source(p) for p in sources],
               producer={n:store_artifact((ROOT/'scripts/eval'/n).read_bytes()) for n in names},
               prompt_policy='Original question and entire decoded answer including reasoning; blind to checkpoint/method',
               coherence_policy='Score all completions on both axes before filtering; retain raw task/coherence scores',
               scope='13 seed1 Qwen3-32B KL/IP checkpoints; 1840 A100 and 5160 H200 completions; no vanilla or SFT controls')
    save(BASE/'config.json', cfg); save(BASE/'endpoint_catalog.json', catalog)
    print(json.dumps(dict(requests=14000, canaries=52, budget_guard_usd=10, model=cfg['body']['model'])))


async def run(key):
    await judge.run('canary', key)
    passed = json.loads((BASE/'canary_passed.json').read_text())
    if not passed['passed']:
        raise RuntimeError('Canary failed; bulk judging not started')
    await judge.run('remaining', key)
    judge.verify()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','run','status','verify'])
    p.add_argument('--env-file', type=Path)
    args = p.parse_args()
    judge.BASE = BASE; judge.canaries = canaries
    if args.action == 'prepare': prepare()
    elif args.action == 'verify': judge.verify()
    elif args.action == 'status': print((BASE/'status.json').read_text())
    else:
        if not args.env_file: p.error('--env-file required')
        lock = ROOT/'.migration-cache/qwen32_judge.lock'
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(run(load_key(args.env_file)))
