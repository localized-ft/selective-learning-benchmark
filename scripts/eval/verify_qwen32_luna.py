"""Offline integrity, request/response parsing and complete-coverage checks."""
from collections import Counter
import json
import math
import qwen32_luna as runner
from vanilla_api import read_events
from vanilla_gpu import ROOT, save, sha


def main():
    base=runner.BASE.parent/'qwen32_openrouter'
    cfg=json.loads((base/'config.json').read_text())
    assert sha((base/'requests.jsonl.gz').read_bytes())==cfg['requests_sha256']
    for path,digest in cfg['source_hashes'].items():assert sha((ROOT/path).read_bytes())==digest,path
    requests=read_events(base/'requests.jsonl.gz');by_id={r['request_id']:r for r in requests}
    events=read_events(base/'events.jsonl.gz');scores=read_events(base/'scores.jsonl.gz')
    assert len(by_id)==15520 and len({r['completion_id'] for r in requests})==7760
    terminal=[e for e in events if e['status']=='scored']
    counts=Counter(e['request_id'] for e in terminal)
    assert set(counts)==set(by_id) and all(n==1 for n in counts.values())
    latest={e['request_id']:e for e in events}
    for e in terminal:
        r=by_id[e['request_id']]
        assert e['request']==dict(model='openai/gpt-6-luna',messages=[dict(role='user',content=r['prompt'])],max_tokens=2000)
        response=json.loads(e['response_text'])
        raw=(response['choices'][0]['message'].get('content') or '').strip()
        parsed=runner.parse(r,raw)
        for key,value in parsed.items():assert e[key]==value
        if e['score'] is not None:assert math.isfinite(e['score'])
    assert len(scores)==15520 and {r['request_id'] for r in scores}==set(by_id)
    for row in scores:
        e=latest[row['request_id']]
        assert row['score']==e['score'] and row['score_label']==e['score_label'] and row['judge_raw']==e['score_source_text']
    for artifact in json.loads((base/'artifacts.json').read_text()):
        assert sha((ROOT/artifact['path']).read_bytes())==artifact['sha256']
    assert all(n==2 for n in Counter(r['completion_id'] for r in scores).values())
    assert sum(r['score_name']=='coherence' for r in scores)==7760
    by_run={}
    for run in sorted({r['run_file'] for r in scores}):
        rows=[r for r in scores if r['run_file']==run]
        by_run[run]=dict(rows=len(rows),numeric=sum(r['score'] is not None for r in rows),
                         missing_labels=dict(Counter(r['score_label'] for r in rows if r['score'] is None)))
    result=dict(complete=True,integrity_checks_passed=True,runs=14,completions=7760,judgments=15520,
                numeric=sum(r['score'] is not None for r in scores),
                missing_labels=dict(Counter(r['score_label'] for r in scores if r['score'] is None)),
                numeric_coherence=sum(r['score_name']=='coherence' and r['score'] is not None for r in scores),
                reported_cost_usd=sum(e.get('cost_usd') or 0 for e in events),
                unreported_cost_attempts=sum(e.get('cost_usd') is None for e in events),
                by_run=by_run)
    save(base/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
