"""Resumable, provider-pinned OpenRouter generation of the frozen HHH prompts."""
import argparse
import asyncio
from collections import Counter
import fcntl
import json
from pathlib import Path
import time
import uuid

import httpx
import hhh_self_distillation as source
from vanilla_api import append_event, classify, load_key, read_events, reconstruct, reported_cost
from vanilla_gpu import save, sha, store_artifact

BASE = source.BASE
API = BASE/'openrouter'
MODEL = 'qwen/qwen3-8b'


def prepare():
    if (API/'config.json').exists():
        print('Existing OpenRouter inputs preserved');return
    catalog = httpx.get(f'https://openrouter.ai/api/v1/models/{MODEL}/endpoints',timeout=45)
    catalog.raise_for_status()
    endpoint = next(e for e in catalog.json()['data']['endpoints'] if e['tag']=='alibaba')
    assert {'temperature','top_p','top_k','reasoning','seed','max_tokens'} <= set(endpoint['supported_parameters'])
    gpu_cfg = json.loads((BASE/'full/config.json').read_text())
    raw = (BASE/'full/requests.jsonl').read_bytes()
    assert sha(raw) == gpu_cfg['requests_sha256']
    body = dict(model=MODEL,temperature=1.0,top_p=1.0,top_k=50,max_tokens=4096,
        frequency_penalty=0.0,presence_penalty=0.0,stream=False,
        reasoning=dict(enabled=True,exclude=False),
        provider=dict(only=['alibaba'],allow_fallbacks=False,require_parameters=True))
    cfg = dict(schema_version='slb.hhh_self_distillation_api.v1',model=MODEL,
        provider='Alibaba',body=body,request_count=221,requests_sha256=sha(raw),
        endpoint_snapshot=endpoint,pricing_per_token=endpoint['pricing'],
        concurrency=8,budget_usd=1.0,max_attempts_per_id=3,
        reservation_per_request_usd=0.003,
        source_model_revision=gpu_cfg['models'][source.KEY]['revision'],
        source_model_revision_scope='Intended base identity only; API provider revision cannot be pinned or guaranteed equivalent',
        raw_response_policy='Append-only complete response JSON including reasoning; no client token truncation',
        retry_policy='Transport/transient API errors only; refusals, content filters and length limits are not resampled',
        producer=store_artifact(Path(__file__).read_bytes()),
        scope='Generation and local dataset export; no HF repository creation, judging or training')
    save(API/'config.json',cfg)
    (API/'requests.jsonl').write_bytes(raw)
    save(API/'endpoint_catalog.json',catalog.json())
    print(json.dumps(dict(prepared=True,model=MODEL,rows=221,provider='Alibaba',budget_guard_usd=1)))


def selected_canaries(rows):
    first={}
    for r in rows:first.setdefault(r['source_metadata']['subset'],r['completion_id'])
    return set(first.values())


def summarize(cfg,rows,events,running=False):
    terminal={e['completion_id']:e for e in events if e['status'] in ['ok','provider_filtered']}
    counts=Counter(e['status'] for e in terminal.values())
    report=dict(requested=len(rows),terminal=len(terminal),usable=counts['ok'],
        provider_filtered=counts['provider_filtered'],pending=len(rows)-len(terminal),
        attempts=len(events),cost_usd=sum(e.get('cost_usd',0) for e in events),
        unreported_cost_attempts=sum(e.get('cost_basis')=='unreported' for e in events),
        running=running,updated_at_unix=time.time())
    save(API/'status.json',report)
    return report


def export(cfg,rows,events):
    terminal={e['completion_id']:e for e in events if e['status'] in ['ok','provider_filtered']}
    full,final,diagnostics=[],[],[]
    for r in rows:
        e=terminal.get(r['completion_id'])
        if e is None:continue
        ok=e['status']=='ok'
        meta=dict(r['source_metadata'],source_row_id=r['eval_id'],reference_kind='vanilla_self_distillation',
            model=MODEL,provider=e.get('provider'),response_model=e.get('response_model'),
            inference_seed=r['inference_seed'],finish_reason=e.get('finish_reason'),
            status=e['status'],source_preference_label_applies=False,api_revision_unpinned=True,
            api_response_id=e.get('response_id'),event_id=e['event_id'])
        meta.pop('preferred_score',None)
        for view,text,target in [('full_decoded',e.get('completion',''),full),
                                 ('final_answer',e.get('final_answer',''),final)]:
            target.append(dict(id=r['eval_id'],messages=r['messages']+[
                dict(role='assistant',content=text)],metadata=dict(meta,response_view=view)))
        diagnostics.append(dict(id=r['eval_id'],completion_id=r['completion_id'],status=e['status'],
            length_limited=e.get('finish_reason')=='length',empty_final_answer=not e.get('final_answer','').strip(),
            reasoning_present=e.get('reasoning_present',False),usage=e.get('usage'),
            ready_final_answer=ok and bool(e.get('final_answer','').strip()) and e.get('finish_reason')=='stop'))
    for name,records in [('reference_full.jsonl',full),('reference_final_answer.jsonl',final),
                          ('generation_diagnostics.jsonl',diagnostics)]:
        (API/name).write_bytes(source.encoded(records))
    save(API/'verification.json',dict(complete=len(terminal)==len(rows),expected=len(rows),
        terminal=len(terminal),final_answer_rows_ready=sum(r['ready_final_answer'] for r in diagnostics),
        final_answer_dataset_ready=len(diagnostics)==len(rows) and all(r['ready_final_answer'] for r in diagnostics),
        provider_filtered=sum(r['status']=='provider_filtered' for r in diagnostics),
        length_limited=sum(r['length_limited'] for r in diagnostics),
        empty_final_answers=sum(r['empty_final_answer'] for r in diagnostics),
        raw_response_attempts=len(events),no_outcome_filtering=True))
    archive={p.name:store_artifact(p.read_bytes()) for p in API.glob('*.jsonl')}
    if (API/'events.jsonl.gz').exists():archive['events.jsonl.gz']=store_artifact((API/'events.jsonl.gz').read_bytes())
    save(API/'artifacts.json',archive)


async def run(stage,key):
    cfg=json.loads((API/'config.json').read_text())
    raw=(API/'requests.jsonl').read_bytes();assert sha(raw)==cfg['requests_sha256']
    rows=[json.loads(s) for s in raw.splitlines()]
    assert len(rows)==len({r['completion_id'] for r in rows})==cfg['request_count']
    events=read_events(API/'events.jsonl.gz')
    finished={e['completion_id'] for e in events if e['status'] in ['ok','provider_filtered']}
    canaries=selected_canaries(rows)
    if stage=='remaining':assert canaries<=finished,'Run and inspect canaries before continuing'
    counts=Counter(e['completion_id'] for e in events)
    selected=[r for r in rows if r['completion_id'] not in finished and (stage!='canary' or r['completion_id'] in canaries)]
    queue=asyncio.Queue()
    for r in selected:queue.put_nowait(r)
    lock=asyncio.Lock();fatal=asyncio.Event()
    spent=sum(e.get('cost_usd',0) for e in events);reserved=0.0
    concurrency=2 if stage=='canary' else cfg['concurrency']
    async with httpx.AsyncClient(timeout=180,headers={'Authorization':'Bearer '+key},
            limits=httpx.Limits(max_connections=concurrency)) as client:
        async def work():
            nonlocal spent,reserved
            while not queue.empty() and not fatal.is_set():
                r=queue.get_nowait()
                while counts[r['completion_id']]<cfg['max_attempts_per_id'] and not fatal.is_set():
                    async with lock:
                        if spent+reserved+cfg['reservation_per_request_usd']>cfg['budget_usd']:
                            fatal.set();break
                        reserved+=cfg['reservation_per_request_usd']
                    payload=dict(cfg['body'],messages=r['messages'],seed=r['inference_seed'])
                    e=dict(event_id=str(uuid.uuid4()),completion_id=r['completion_id'],eval_id=r['eval_id'],
                        request=payload,at_unix=time.time(),cost_usd=0,cost_basis='unreported')
                    code=0
                    try:
                        response=await client.post('https://openrouter.ai/api/v1/chat/completions',json=payload)
                        code=response.status_code
                        e['response_text']=response.text.replace(key,'[REDACTED]')
                        data=json.loads(e['response_text'])
                        cost,basis=reported_cost(data,cfg['pricing_per_token'])
                        e.update(status=classify(data,code),cost_usd=cost,cost_basis=basis,
                            provider=data.get('provider'),response_model=data.get('model'),
                            response_id=data.get('id'),usage=data.get('usage'))
                        if e['status']=='ok':
                            assert e['provider']==cfg['provider'],'Unexpected provider'
                            choice=data['choices'][0];message=choice['message']
                            text=reconstruct(message,qwen=True)
                            answer=message.get('content') or ''
                            if '</think>' in answer:answer=answer.split('</think>',1)[1]
                            e.update(completion=text,final_answer=answer.strip(),
                                reasoning_present=bool(message.get('reasoning') or message.get('reasoning_details')),
                                finish_reason=choice['finish_reason'])
                    except (httpx.TimeoutException,httpx.NetworkError):e['status']='transport_error'
                    except Exception as exc:e.update(status='protocol_error',error_type=type(exc).__name__)
                    e['http_status']=code
                    async with lock:
                        reserved-=cfg['reservation_per_request_usd'];spent+=e['cost_usd']
                        append_event(API/'events.jsonl.gz',e);events.append(e);counts[r['completion_id']]+=1
                        if e['status'] in ['ok','provider_filtered']:finished.add(r['completion_id'])
                        status=summarize(cfg,rows,events,True)
                        if len(events)%25==0 or stage=='canary':print(json.dumps(status),flush=True)
                    if e['status'] in ['ok','provider_filtered']:break
                    if e['status']=='protocol_error' or code in [400,401,402,403,404,422]:fatal.set();break
                    await asyncio.sleep(min(10*counts[r['completion_id']],30))
                queue.task_done()
        try:await asyncio.gather(*(work() for _ in range(concurrency)))
        finally:
            report=summarize(cfg,rows,events)
            export(cfg,rows,events)
            print(json.dumps(dict(report,stage=stage,stopped_on_guard_or_error=fatal.is_set())),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--stage',choices=['canary','remaining'],default='canary')
    p.add_argument('--env-file',type=Path)
    a=p.parse_args()
    if a.action=='prepare':return prepare()
    if not a.env_file:p.error('--env-file required')
    with (API/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(run(a.stage,load_key(a.env_file)))


if __name__=='__main__':main()
