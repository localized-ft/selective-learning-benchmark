"""Credential-free-on-disk LiteLLM transport for the frozen KL judge requests.

Original OpenRouter requests/failed attempts remain untouched. Only the model
route gains the proxy's openrouter/ prefix; grading prompts and parsers are reused.
"""
import argparse
import asyncio
from collections import Counter
import copy
import fcntl
import getpass
import json
import os
from pathlib import Path
import time
import uuid

from openai import AsyncOpenAI, APIStatusError
import kl_reference_distribution_judge as scoring
import qwen32_luna as luna
import vanilla_judge as deepseek
from vanilla_api import append_event, read_events, reported_cost
from vanilla_gpu import ROOT, save, sha, store_artifact

API_BASE = 'https://litellm.nielsrolf.com'
JUDGES = ('deepseek','luna')


def directory(cohort, name):
    return scoring.paths(cohort)[0]/('judge_'+name+'_litellm')


def prepare(cohort, name):
    path = directory(cohort, name)
    if (path/'config.json').exists():
        frozen(cohort, name)
        return
    source = scoring.paths(cohort)[1 if name=='deepseek' else 2]
    previous = json.loads((source/'config.json').read_text())
    raw = (source/'requests.jsonl.gz').read_bytes()
    expected = previous['requests_stored_sha256'] if name=='deepseek' else previous['requests_sha256']
    assert sha(raw)==expected
    body = copy.deepcopy(previous['body']) if name=='deepseek' else dict(model=previous['model'],max_tokens=previous['max_tokens'])
    assert not body['model'].startswith('openrouter/')
    body['model']='openrouter/'+body['model']
    if name=='deepseek':
        pricing=previous['pricing_per_token']
    else:
        pricing=previous['model_catalog_entry']['pricing']
    sources=[source/'config.json',source/'requests.jsonl.gz',Path(__file__),
             ROOT/'scripts/eval/kl_reference_distribution_judge.py',ROOT/'scripts/eval/vanilla_judge.py',
             ROOT/'scripts/eval/qwen32_luna.py',ROOT/'scripts/eval/vanilla_api.py']
    sources+=list(luna.UPSTREAM.glob('*.py'))
    cfg=dict(schema_version='slb.kl_reference_litellm.v1',judge=name,transport='litellm_proxy',api_base=API_BASE,
        body=body,request_count=previous['request_count'],completion_count=previous['completion_count'],
        requests_sha256=sha(raw),source_hashes={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sources},
        concurrency=32,max_attempts_per_id=5,timeout_seconds=120,budget_usd=4,
        pricing_per_token=pricing,producer=store_artifact(Path(__file__).read_bytes()),
        original_config=str((source/'config.json').relative_to(ROOT)),
        transport_changes='LiteLLM proxy and openrouter/ route prefix only; original rubrics, decoded-answer policy and explicit sampling fields preserved',
        provider_visibility='DeepSeek requests preserve the Alibaba FP8 pin; missing proxy provider metadata is reported as unverified, never fabricated',
        cache_policy='No cache or seed fields added; local journals skip completed request IDs',
        credential_policy='Process environment or hidden terminal input only; no key file or shared server credential changes',
        retry_policy='At most five attempts for transient errors and DeepSeek invalid formatting; valid labels are terminal, never resampled',
        cost_policy='Guard on reported or token-estimated usage; proxy costs/routing may differ from public pricing; not a provider hard cap')
    path.mkdir(parents=True,exist_ok=True)
    (path/'requests.jsonl.gz').write_bytes(raw)
    save(path/'config.json',cfg)
    print(json.dumps(dict(prepared=name,requests=cfg['request_count'],transport='litellm_proxy')),flush=True)


def frozen(cohort,name):
    path=directory(cohort,name)
    cfg=json.loads((path/'config.json').read_text())
    assert sha((path/'requests.jsonl.gz').read_bytes())==cfg['requests_sha256']
    for relative,digest in cfg['source_hashes'].items():
        assert sha((ROOT/relative).read_bytes())==digest,relative
    requests=read_events(path/'requests.jsonl.gz')
    assert len(requests)==cfg['request_count']==len({r['request_id'] for r in requests})
    return path,cfg,requests


def payload(cfg,request):
    return dict(cfg['body'],messages=[dict(role='user',content=request['prompt'])])


def sdk_arguments(body):
    standard={'model','messages','max_tokens','temperature','top_p','stream'}
    args={k:v for k,v in body.items() if k in standard}
    extra={k:v for k,v in body.items() if k not in standard}
    if extra:args['extra_body']=extra
    return args


def parse(name,request,data):
    if name=='deepseek' and data.get('provider') not in [None,'Alibaba']:
        return dict(status='error',error_type='UnexpectedProvider')
    choice=data['choices'][0]
    message=choice['message']
    raw=(message.get('content') or '').strip()
    if name=='deepseek':
        if message.get('reasoning') or message.get('reasoning_details'):
            return dict(status='error',error_type='UnexpectedReasoning')
        result,state=deepseek.parse_score(request,raw)
        status='scored' if state in ['ok','label'] else 'invalid_score'
    else:
        result=luna.parse(request,raw)
        status='scored'
    return dict(result,status=status,score_source_text=raw,finish_reason=choice.get('finish_reason'),
        provider=data.get('provider'),provider_independently_reported=bool(data.get('provider')))


def export(path,cfg,requests,events,running=False):
    done={e['request_id']:e for e in events if e['status']=='scored'}
    latest={e['request_id']:e for e in events}
    status=dict(judge=cfg['judge'],transport='litellm_proxy',requested=len(requests),terminal=len(done),
        pending=len(requests)-len(done),numeric=sum(e.get('score') is not None for e in done.values()),
        attempts=len(events),cost_usd=sum(e.get('cost_usd') or 0 for e in events),
        unreported_cost_attempts=sum(e.get('cost_basis')=='unreported' for e in events),
        running=running,updated_at_unix=time.time())
    save(path/'status.json',status)
    if not running:
        rows=[]
        for r in requests:
            if r['request_id'] not in latest:continue
            e=done.get(r['request_id'],latest[r['request_id']])
            row={k:r[k] for k in ['request_id','completion_id','eval_id','axis','score_name','task_id','coherence_source']}
            row.update({k:e.get(k) for k in ['score','score_label','score_source_text','status','event_id','provider']})
            row.update(judge_model=cfg['body']['model'],judge_transport='litellm_proxy',judge_raw=e.get('score_source_text',''))
            rows.append(row)
        deepseek.write_gzip(path/'scores.jsonl.gz',rows)
        save(path/'artifacts.json',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p.read_bytes()),bytes=p.stat().st_size)
            for p in path.glob('*.jsonl.gz')])
    return status


async def run(cohort,name,key):
    path,cfg,requests=frozen(cohort,name)
    events=read_events(path/'events.jsonl.gz')
    done={e['request_id'] for e in events if e['status']=='scored'}
    counts=Counter(e['request_id'] for e in events)
    first={}
    for r in requests:
        group=('__'.join(r['completion_id'].split('__')[:2]),r['axis'],r['score_name'])
        first.setdefault(group,r['request_id'])
    canary=set(first.values());fatal=asyncio.Event();lock=asyncio.Lock()
    async with AsyncOpenAI(api_key=key,base_url=API_BASE,max_retries=0,timeout=120,
        default_headers={'User-Agent':'litellm-client/1.0'}) as client:
        async def stage(selected,concurrency):
            sem=asyncio.Semaphore(concurrency)
            async def one(r):
                async with sem:
                    if r['request_id'] in done or fatal.is_set():return
                    while counts[r['request_id']]<cfg['max_attempts_per_id'] and not fatal.is_set():
                        if sum(e.get('cost_usd') or 0 for e in events)>=cfg['budget_usd']:
                            fatal.set();return
                        body=payload(cfg,r)
                        e=dict(event_id=str(uuid.uuid4()),request_id=r['request_id'],request=body,
                            at_unix=time.time(),status='error',cost_usd=0,cost_basis='unreported')
                        try:
                            response=await client.chat.completions.with_raw_response.create(**sdk_arguments(body))
                            e.update(http_status=response.http_response.status_code,response_text=response.http_response.text.replace(key,'[REDACTED]'))
                            data=json.loads(e['response_text'])
                            e['cost_usd'],e['cost_basis']=reported_cost(data,cfg['pricing_per_token'])
                            e.update(parse(name,r,data))
                            if e.get('error_type') in ['UnexpectedProvider','UnexpectedReasoning']:fatal.set()
                        except APIStatusError as exc:
                            e.update(http_status=exc.status_code,response_text=exc.response.text.replace(key,'[REDACTED]'),error_type=type(exc).__name__)
                            if exc.status_code in [400,401,402,403,404,422]:fatal.set()
                        except Exception as exc:
                            e['error_type']=type(exc).__name__
                        async with lock:
                            counts[r['request_id']]+=1;e['attempt']=counts[r['request_id']]
                            append_event(path/'events.jsonl.gz',e);events.append(e)
                            if e['status']=='scored':done.add(r['request_id'])
                            if len(events)%100==0:print(json.dumps(export(path,cfg,requests,events,True)),flush=True)
                        if r['request_id'] in done:break
                        await asyncio.sleep(min(2**counts[r['request_id']],30))
            await asyncio.gather(*(one(r) for r in selected))
        try:
            export(path,cfg,requests,events,True)
            await stage([r for r in requests if r['request_id'] in canary],4)
            latest={e['request_id']:e for e in events}
            passed=canary<=done and all(latest[k].get('score_label') not in ['ERROR','PARSE_ERROR','NO_MATCH','NONFINITE'] for k in canary)
            save(path/'canary.json',dict(passed=passed,expected=len(canary),completed=len(canary&done)))
            print(json.dumps(dict(judge=name,canary_passed=passed,completed=len(canary&done),expected=len(canary))),flush=True)
            if passed and not fatal.is_set():await stage(requests,cfg['concurrency'])
        finally:
            print(json.dumps(export(path,cfg,requests,events,False)),flush=True)
    if len(done)==len(requests):verify(cohort,name)


def verify(cohort,name):
    path,cfg,requests=frozen(cohort,name)
    expected={r['request_id']:r for r in requests}
    events=read_events(path/'events.jsonl.gz')
    scored=[e for e in events if e['status']=='scored']
    counts=Counter(e['request_id'] for e in scored)
    assert set(counts)==set(expected) and all(n==1 for n in counts.values())
    for e in scored:
        r=expected[e['request_id']]
        assert e['request']==payload(cfg,r)
        parsed=parse(name,r,json.loads(e['response_text']))
        assert parsed['status']=='scored'
        for k in ['score','score_label','score_source_text']:
            assert e[k]==parsed[k]
    rows=read_events(path/'scores.jsonl.gz')
    assert len(rows)==cfg['request_count'] and {r['request_id'] for r in rows}==set(expected)
    assert len({r['completion_id'] for r in rows})==cfg['completion_count']
    assert all(n==2 for n in Counter(r['completion_id'] for r in rows).values())
    assert sum(r['score_name']=='coherence' for r in rows)==cfg['completion_count']
    result=dict(complete=True,all_requests_terminal=True,judgments=len(rows),completions=cfg['completion_count'],
        numeric=sum(r['score'] is not None for r in rows),
        missing_labels=dict(Counter(r['score_label'] for r in rows if r['score'] is None)),
        provider_metadata_missing=sum(not e.get('provider_independently_reported') for e in scored),
        integrity_checks_passed=True)
    save(path/'verification.json',result)
    print(json.dumps(dict(judge=name,verification=result)),flush=True)


def compare(cohort):
    result={}
    for name in JUDGES:
        verify(cohort,name)
        rows=read_events(directory(cohort,name)/'scores.jsonl.gz')
        keys=sorted({'__'.join(r['completion_id'].split('__')[:2]) for r in rows})
        result[name]={key:scoring.summarize([r for r in rows if r['completion_id'].startswith(key+'__')]) for key in keys}
    save(scoring.paths(cohort)[0]/'judge_comparison_litellm.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','verify','compare'])
    parser.add_argument('--cohort',choices=['original','self_distilled'],required=True)
    parser.add_argument('--judge',choices=['deepseek','luna','both'],default='both')
    args=parser.parse_args();names=JUDGES if args.judge=='both' else (args.judge,)
    if args.action=='prepare':
        for name in names:prepare(args.cohort,name)
    elif args.action=='verify':
        for name in names:verify(args.cohort,name)
    elif args.action=='compare':compare(args.cohort)
    else:
        handles=[]
        try:
            for name in names:
                h=(ROOT/f'.migration-cache/kl_reference_{args.cohort}_{name}_litellm.lock').open('a')
                fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(h)
            key=os.environ.get('LITELLM_API_KEY') or getpass.getpass('LiteLLM credential: ')
            assert key
            async def both():
                await asyncio.gather(*(run(args.cohort,name,key) for name in names))
            asyncio.run(both())
        finally:
            for h in handles:h.close()


if __name__=='__main__':main()
