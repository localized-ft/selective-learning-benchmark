"""Resumable Luna rejudge of all 14 Qwen32 KL/IP runs; original exports untouched."""
import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import asdict
import fcntl
import gzip
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

from vanilla_gpu import ROOT, SNAPSHOT, save, sha
from vanilla_api import append_event, read_events

BASE=ROOT/'result/supplemental/luna_rejudge_20261004/qwen32'
UPSTREAM=BASE.parent/'upstream/sunday_eval'
sys.path.insert(0,str(UPSTREAM))
spec=importlib.util.spec_from_file_location('luna_original_judge',UPSTREAM/'judge_utility.py')
original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
MODEL='openrouter/openai/gpt-6-luna'


def strip_think(text):
    return re.sub(r'<think>.*?</think>','',text or '',flags=re.DOTALL).split('</think>')[-1].strip()


def write_rows(path,rows):
    path.write_bytes(gzip.compress(b''.join((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode() for r in rows),mtime=0))


def prepare():
    if (BASE/'config.json').exists():raise RuntimeError('Inputs already frozen')
    sources=[ROOT/'result/supplemental/qwen32_inference_20260921/judge/requests.jsonl.gz',
             ROOT/'result/supplemental/qwen32_medical_ip_20261003/eval/judge/requests.jsonl.gz']
    evaluations={p.parent.name:{r['id']:r for r in map(json.loads,p.read_text().splitlines())} for p in SNAPSHOT.glob('*/eval.jsonl')}
    requests=[];seen=set()
    for source in sources:
        for old in read_events(source):
            grading=evaluations[old['task_id']][old['eval_id']]['grading']
            prompts={s.score_name:s for s in original.judge_prompts_from_grading(grading)}
            if old['score_name']=='coherence' and old['score_name'] not in prompts:
                template=(SNAPSHOT.parent/'coherence_rubric.txt').read_text()
            else:template=prompts[old['score_name']].prompt
            completion=strip_think(old['completion'])
            request=dict(old,completion=completion,completion_sha256=sha(completion.encode()),
                         original_completion_sha256=old['completion_sha256'],
                         run_file='__'.join(old['completion_id'].split('__')[:2]),
                         prompt=original.render_judge_prompt(template,old['question'],completion))
            request['prompt_sha256']=sha(request['prompt'].encode())
            assert request['request_id'] not in seen;seen.add(request['request_id']);requests.append(request)
    assert len(requests)==15520 and len({r['completion_id'] for r in requests})==7760
    assert len({r['run_file'] for r in requests})==14
    BASE.mkdir(parents=True,exist_ok=True);write_rows(BASE/'requests.jsonl.gz',requests)
    tracked=sources+list(SNAPSHOT.glob('*/eval.jsonl'))+[SNAPSHOT.parent/'coherence_rubric.txt',Path(__file__)]+list(UPSTREAM.glob('*.py'))
    cfg=dict(model=MODEL,max_tokens=2000,concurrency=100,max_attempts=5,timeout_seconds=120,
             request_count=15520,completion_count=7760,run_count=14,reported_cost_guard_usd=25,
             requests_sha256=sha((BASE/'requests.jsonl.gz').read_bytes()),
             source_hashes={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in tracked},
             protocol='Upstream Luna parser/rendering and think stripping; defaults for temperature/reasoning/provider; original per-task rubrics with existing cohort coherence backfill preserved',
             retry_policy='Only transport/HTTP failures retried, never parsed outcomes; all attempts preserved',
             cost_policy='Guard on reported costs only; no estimate where proxy omits costs; bounded request/retry counts')
    save(BASE/'config.json',cfg);print(json.dumps(dict(prepared=True,requests=15520,runs=14)),flush=True)


def parse(r,raw):
    score=(original.score_with_regex_map(r['score_name'],r['answer_regex'],r['score_map'],raw)
           if r['answer_regex'] and r['score_map'] else original.parse_judge_response_score(r['score_name'],raw))
    result=asdict(score)
    if result['score'] is not None and not math.isfinite(result['score']):
        result.update(score=None,score_label='NONFINITE')
    return result


def export(cfg,requests,events,running=False):
    latest={e['request_id']:e for e in events}
    terminal={k:e for k,e in latest.items() if e['status']=='scored'}
    cost=sum(e.get('cost_usd') or 0 for e in events)
    status=dict(requested=len(requests),terminal=len(terminal),pending=len(requests)-len(terminal),
                numeric=sum(e['score'] is not None for e in terminal.values()),
                labels=dict(Counter(e['score_label'] for e in terminal.values() if e['score'] is None)),
                attempts=len(events),reported_cost_usd=cost,
                unreported_cost_attempts=sum(e.get('cost_usd') is None for e in events),
                running=running,updated_at_unix=time.time())
    save(BASE/'status.json',status)
    if not running:
        rows=[];groups=defaultdict(list)
        for r in requests:
            if r['request_id'] not in latest:continue
            e=latest[r['request_id']]
            row={k:r[k] for k in ['request_id','completion_id','eval_id','axis','score_name','run_file','task_id','coherence_source']}
            row.update(score=e.get('score'),score_label=e.get('score_label','ERROR'),judge_raw=e.get('score_source_text',''),status=e['status'])
            rows.append(row)
            groups[r['run_file']].append(dict(row,score='' if row['score'] is None else row['score']))
        write_rows(BASE/'scores.jsonl.gz',rows)
        summaries={}
        for run,group in groups.items():
            s={}
            for axis,prefix in [('capability','capability'),('unintended_generalization','ug')]:
                subset=[r for r in group if (r['axis']=='capability')==(axis=='capability')]
                if subset:original.add_axis_score_summary(s,subset,prefix+'_n',prefix+'_mean',prefix+'_key',prefix+'_coh_n',prefix+'_coh_mean')
            s['error_rows']=sum(r['status']!='scored' for r in group)
            s['accepted_under_upstream_error_rule']=s['error_rows']<=0.02*len(group)
            summaries[run]=s
        save(BASE/'summary.json',summaries)
        save(BASE/'artifacts.json',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in BASE.glob('*.jsonl.gz')])
    return status


async def run(env_file):
    from dotenv import dotenv_values
    import httpx
    cfg=json.loads((BASE/'config.json').read_text())
    if cfg.get('transport')=='openrouter_direct':
        from vanilla_api import load_key
        key=load_key(env_file);url='https://openrouter.ai/api/v1'
    else:
        env=dotenv_values(env_file);key=env['LITELLM_API_KEY']
        url=env.get('LITELLM_BASE_URL') or 'https://litellm.nielsrolf.com'
    assert sha((BASE/'requests.jsonl.gz').read_bytes())==cfg['requests_sha256']
    for path,digest in cfg['source_hashes'].items():assert sha((ROOT/path).read_bytes())==digest,path
    requests=read_events(BASE/'requests.jsonl.gz');events=read_events(BASE/'events.jsonl.gz')
    done={e['request_id'] for e in events if e['status']=='scored'}
    attempts=Counter(e['request_id'] for e in events);fatal=asyncio.Event();lock=asyncio.Lock()
    first={}
    for r in requests:first.setdefault((r['run_file'],r['axis'],r['score_name']),r['request_id'])
    canary=set(first.values())
    async with httpx.AsyncClient(timeout=120,headers={'Authorization':'Bearer '+key,'User-Agent':'python-httpx/0.27'},limits=httpx.Limits(max_connections=100)) as client:
        async def stage(rows,concurrency):
            sem=asyncio.Semaphore(concurrency)
            async def one(r):
                async with sem:
                    if r['request_id'] in done:return
                    while attempts[r['request_id']]<5 and not fatal.is_set():
                        if sum(e.get('cost_usd') or 0 for e in events)>=cfg['reported_cost_guard_usd']:
                            fatal.set();return
                        payload=dict(model=cfg['model'],messages=[dict(role='user',content=r['prompt'])],max_tokens=2000)
                        event=dict(request_id=r['request_id'],request=payload,at_unix=time.time(),status='error',cost_usd=None)
                        try:
                            response=await client.post(url.rstrip('/')+'/chat/completions',json=payload)
                            event.update(http_status=response.status_code,response_text=response.text.replace(key,'[REDACTED]'))
                            data=response.json();usage=data.get('usage') or {}
                            event['cost_usd']=usage.get('cost')
                            if response.status_code==200:
                                raw=(data['choices'][0]['message'].get('content') or '').strip()
                                event.update(parse(r,raw),status='scored')
                            elif response.status_code in [400,401,402,403,404,422]:fatal.set()
                        except Exception as exc:event['error_type']=type(exc).__name__
                        async with lock:
                            attempts[r['request_id']]+=1;event['attempt']=attempts[r['request_id']]
                            append_event(BASE/'events.jsonl.gz',event);events.append(event)
                            if event['status']=='scored':done.add(r['request_id'])
                            if len(events)%100==0:print(json.dumps(export(cfg,requests,events,True)),flush=True)
                        if r['request_id'] in done:break
                        await asyncio.sleep(min(2**attempts[r['request_id']],30))
            await asyncio.gather(*(one(r) for r in rows))
        try:
            export(cfg,requests,events,True)
            await stage([r for r in requests if r['request_id'] in canary],4)
            latest={e['request_id']:e for e in events}
            passed=canary<=done and all(latest[k].get('score_label') not in ['ERROR','PARSE_ERROR','NO_MATCH','NONFINITE'] for k in canary)
            save(BASE/'canary.json',dict(passed=passed,expected=len(canary),completed=len(canary&done)))
            if not passed:raise RuntimeError('Canary failed; bulk judging stopped')
            await stage(requests,100)
        finally:print(json.dumps(export(cfg,requests,events,False)),flush=True)


def prepare_direct():
    import httpx
    source=BASE.parent/'qwen32'
    if (BASE/'config.json').exists():raise RuntimeError('Direct inputs already frozen')
    cfg=json.loads((source/'config.json').read_text())
    raw=(source/'requests.jsonl.gz').read_bytes();assert sha(raw)==cfg['requests_sha256']
    response=httpx.get('https://openrouter.ai/api/v1/models',timeout=45);response.raise_for_status()
    model=next(m for m in response.json()['data'] if m['id']=='openai/gpt-6-luna')
    cfg.update(model=model['id'],transport='openrouter_direct',api_base='https://openrouter.ai/api/v1',
               model_catalog_entry=model,previous_attempt='qwen32: proxy authentication rejected; no scores',
               transport_note='Direct OpenRouter instead of upstream LiteLLM proxy; same model name and max_tokens; provider routing may differ')
    cfg['source_hashes'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__).read_bytes())
    BASE.mkdir(parents=True,exist_ok=True);(BASE/'requests.jsonl.gz').write_bytes(raw)
    save(BASE/'config.json',cfg)
    print(json.dumps(dict(prepared=True,model=cfg['model'],requests=cfg['request_count'],pricing=model['pricing'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--env-file',type=Path)
    p.add_argument('--direct',action='store_true');a=p.parse_args()
    if a.direct:BASE=BASE.parent/'qwen32_openrouter'
    if a.action=='prepare':(prepare_direct if a.direct else prepare)()
    else:
        if not a.env_file:p.error('--env-file required')
        with (ROOT/'.migration-cache/qwen32_luna.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(run(a.env_file))
